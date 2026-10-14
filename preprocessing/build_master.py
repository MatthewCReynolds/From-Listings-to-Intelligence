"""Build the listing-episode master table.

One row per market/unit (listing episode). Every column is prefixed with its
source as <schema>_<table>__<column>, so e.g. all ground-truth columns can be
dropped with `df.filter(regex="^(?!truth_)")`. Child tables (price windows,
concession events) are aggregated to the listing; parent tables (building,
floor plan) are joined 1:many.
"""

import csv
import re
import sys
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "accolade.duckdb"
OUT = REPO / "data" / "master_listing.parquet"
CATALOG = REPO / "data" / "master_columns.csv"
AS_OF = "2026-10-09"

KEYS = {"unit_id", "property_id", "floor_plan_id", "market"}


def norm(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    return s.strip("_").lower()


class Builder:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.selects: list[str] = []
        self.catalog: list[dict] = []

    def add(self, expr: str, name: str, source: str, source_col: str, kind: str):
        self.selects.append(f"{expr} AS {name}")
        self.catalog.append(
            {"column": name, "source_table": source, "source_column": source_col, "kind": kind}
        )

    def add_table(self, schema: str, table: str, alias: str, exclude: set[str] = frozenset()):
        cols = self.con.execute(f"DESCRIBE {schema}.{table}").fetchall()
        for col, *_ in cols:
            if col in KEYS or col in exclude:
                continue
            self.add(f'{alias}."{col}"', f"{schema}_{table}__{norm(col)}", f"{schema}.{table}", col, "raw")

    def add_multihot(self, schema: str, table: str, alias: str, list_col: str, label: str):
        vocab = [
            r[0]
            for r in self.con.execute(
                f"SELECT DISTINCT a FROM (SELECT unnest({list_col}) a FROM {schema}.{table}) ORDER BY 1"
            ).fetchall()
        ]
        for item in vocab:
            lit = item.replace("'", "''")
            expr = f'CASE WHEN len({alias}."{list_col}") = 0 THEN NULL ELSE list_contains({alias}."{list_col}", \'{lit}\') END'
            self.add(expr, f"{schema}_{table}__{label}_{norm(item)}", f"{schema}.{table}", f"{list_col}[{item}]", "multihot")


PRICE_AGG = f"""
price_agg AS (
    SELECT
        unit_id,
        count(*)                                                        AS n_windows,
        count(*) FILTER (WHERE asking_rent <> prev_asking)              AS n_price_changes,
        arg_min(asking_rent, from_date)                                 AS first_asking_rent,
        arg_max(asking_rent, from_date)                                 AS last_asking_rent,
        min(asking_rent)                                                AS min_asking_rent,
        max(asking_rent)                                                AS max_asking_rent,
        sum(asking_rent * days) / sum(days)                             AS day_weighted_asking_rent,
        arg_min(effective_rent, from_date)                              AS first_effective_rent,
        arg_max(effective_rent, from_date)                              AS last_effective_rent,
        sum(effective_rent * days) / sum(days)                          AS day_weighted_effective_rent,
        arg_min(asking_rent_psf, from_date)                             AS first_asking_rent_psf,
        arg_max(asking_rent_psf, from_date)                             AS last_asking_rent_psf,
        max(concession)                                                 AS max_concession,
        sum(concession * days) / sum(days)                              AS day_weighted_concession,
        sum(days) FILTER (WHERE concession > 0) * 1.0 / sum(days)       AS share_days_with_concession,
        arg_max(lease_start_date, from_date)                            AS inferred_lease_start_date,
        arg_max(lease_end_date, from_date)                              AS inferred_lease_end_date,
        arg_max(lease_term, from_date)                                  AS inferred_lease_term,
        arg_max(leased_asking_rent, from_date)                          AS leased_asking_rent,
        arg_max(leased_effective_rent, from_date)                       AS leased_effective_rent,
        arg_max(leased_asking_rent_psf, from_date)                      AS leased_asking_rent_psf,
        arg_max(leased_effective_rent_psf, from_date)                   AS leased_effective_rent_psf,
        arg_max(leased_concession, from_date)                           AS leased_concession
    FROM (
        SELECT *,
               lag(asking_rent) OVER (PARTITION BY unit_id ORDER BY from_date) AS prev_asking,
               date_diff('day', from_date, coalesce(to_date, DATE '{AS_OF}')) + 1 AS days
        FROM market.price_observation
    )
    GROUP BY unit_id
)"""

CONCESSION_AGG = """
concession_agg AS (
    SELECT
        unit_id,
        count(*)                                                AS n_events,
        sum(date_diff('day', from_date, to_date) + 1)           AS n_days_advertised,
        count(*) FILTER (WHERE type = 'upfront_free_months')    AS n_upfront_free_months,
        count(*) FILTER (WHERE type = 'recurring_discount')     AS n_recurring_discount,
        count(*) FILTER (WHERE type = 'one_time_credit')        AS n_one_time_credit,
        count(*) FILTER (WHERE type = 'look_and_lease')         AS n_look_and_lease,
        count(*) FILTER (WHERE type = 'unknown')                AS n_unknown,
        arg_min(type, from_date)                                AS first_type,
        arg_min(specials, from_date)                            AS first_specials_text
    FROM market.concession_event
    WHERE unit_id IS NOT NULL
    GROUP BY unit_id
)"""

PRICE_AGG_COLS = [
    "n_windows", "n_price_changes", "first_asking_rent", "last_asking_rent", "min_asking_rent",
    "max_asking_rent", "day_weighted_asking_rent", "first_effective_rent", "last_effective_rent",
    "day_weighted_effective_rent", "first_asking_rent_psf", "last_asking_rent_psf", "max_concession",
    "day_weighted_concession", "share_days_with_concession", "inferred_lease_start_date",
    "inferred_lease_end_date", "inferred_lease_term", "leased_asking_rent", "leased_effective_rent",
    "leased_asking_rent_psf", "leased_effective_rent_psf", "leased_concession",
]
CONCESSION_AGG_COLS = [
    "n_events", "n_days_advertised", "n_upfront_free_months", "n_recurring_discount",
    "n_one_time_credit", "n_look_and_lease", "n_unknown", "first_type", "first_specials_text",
]


def main() -> int:
    if not DB.exists():
        sys.exit(f"missing {DB}\nrun: python preprocessing/build_duckdb.py")

    con = duckdb.connect(str(DB), read_only=True)
    b = Builder(con)

    for k in ("unit_id", "property_id", "floor_plan_id", "market"):
        b.add(f"u.{k}", k, "market.unit", k, "key")

    b.add_table("market", "unit", "u")
    b.add_table("truth", "lease_event", "le")
    b.add_table("market", "floor_plan", "fp", exclude={"floorplan_name", "bed", "bath", "partial_bath"})
    b.add_table("market", "building", "bld", exclude={"geography", "building_amenities", "unit_amenities"})
    b.add("len(bld.building_amenities)", "market_building__n_building_amenities", "market.building", "building_amenities", "aggregate")
    b.add("len(bld.unit_amenities)", "market_building__n_unit_amenities", "market.building", "unit_amenities", "aggregate")
    b.add_multihot("market", "building", "bld", "building_amenities", "amenity")
    b.add_multihot("market", "building", "bld", "unit_amenities", "unit_amenity")
    b.add_table("truth", "building_params", "bp", exclude={"submarket"})

    for c in PRICE_AGG_COLS:
        b.add(f"pa.{c}", f"market_price_observation__{c}", "market.price_observation", c, "aggregate")
    for c in CONCESSION_AGG_COLS:
        b.add(f"ca.{c}", f"market_concession_event__{c}", "market.concession_event", c, "aggregate")

    b.add("la.leased_percentage", "market_leasing_activity_daily__leased_pct_at_enter", "market.leasing_activity_daily", "leased_percentage", "aggregate")
    b.add("la.exposure_percentage", "market_leasing_activity_daily__exposure_pct_at_enter", "market.leasing_activity_daily", "exposure_percentage", "aggregate")

    b.add("pp.sid IS NOT NULL", "pms_property__is_pms", "pms.property", "sid", "aggregate")
    b.add_table(
        "pms", "property", "pp",
        exclude={"name", "street", "city", "state", "zipcode", "country", "latitude", "longitude", "url", "timeCreated", "timeUpdated"},
    )

    names = [r["column"] for r in b.catalog]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate output columns: {sorted(dupes)}"

    sql = f"""
    WITH {PRICE_AGG.strip()},
    {CONCESSION_AGG.strip()}
    SELECT
        {",\n        ".join(b.selects)}
    FROM market.unit u
    LEFT JOIN truth.lease_event le             ON le.unit_id = u.unit_id
    LEFT JOIN market.floor_plan fp             ON fp.floor_plan_id = u.floor_plan_id
    LEFT JOIN market.building bld              ON bld.property_id = u.property_id
    LEFT JOIN truth.building_params bp         ON bp.property_id = u.property_id
    LEFT JOIN price_agg pa                     ON pa.unit_id = u.unit_id
    LEFT JOIN concession_agg ca                ON ca.unit_id = u.unit_id
    LEFT JOIN market.leasing_activity_daily la ON la.property_id = u.property_id AND la."As of Day" = u.enter_market
    LEFT JOIN pms.property pp                  ON pp.property_id = u.property_id
    """

    print(f"building {len(b.selects)} columns ...")
    OUT.unlink(missing_ok=True)
    con.execute(f"COPY ({sql}) TO '{OUT.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    expected = con.execute("SELECT count(*) FROM market.unit").fetchone()[0]
    n, n_distinct = con.execute(
        f"SELECT count(*), count(DISTINCT unit_id) FROM read_parquet('{OUT.as_posix()}')"
    ).fetchone()
    assert n == expected, f"row fan-out: {n} rows vs {expected} listings"
    assert n_distinct == n, f"duplicate unit_id: {n - n_distinct}"

    with CATALOG.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["column", "source_table", "source_column", "kind"])
        w.writeheader()
        w.writerows(b.catalog)

    print(f"wrote {OUT} ({n:,} rows x {len(b.selects)} cols, {OUT.stat().st_size / 1e6:.0f} MB)")
    print(f"wrote {CATALOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
