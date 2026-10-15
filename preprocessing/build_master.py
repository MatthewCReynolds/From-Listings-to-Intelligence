"""Build the listing-episode master table.

One row per market/unit (listing episode). Columns are named
<schema>_<table>__<column>, and master_columns.csv tags each one with a role:

    key       join keys
    feature   usable as X
    label     prediction targets (ground truth)
    leaky     market-layer columns that ARE this listing's own price
    latent    simulator internals, not observable in production
    inferred  platform inferences available at prediction time

Select features by role, not by prefix: the 14 `leaky` columns live in the
market_ namespace, so a `truth_`-prefix filter does not exclude them.
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

# Rent levels of the listing itself: with these in X, a rent model scores
# ~1.0 and learns nothing. Measured |r| vs true_rent is 0.989-0.9998.
LEAKY = {
    "market_price_observation__" + c for c in (
        "first_asking_rent", "last_asking_rent", "min_asking_rent", "max_asking_rent",
        "day_weighted_asking_rent", "first_effective_rent", "last_effective_rent",
        "day_weighted_effective_rent", "first_asking_rent_psf", "last_asking_rent_psf",
        "leased_asking_rent", "leased_effective_rent", "leased_asking_rent_psf",
        "leased_effective_rent_psf",
    )
}


def norm(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    return s.strip("_").lower()


def role_of(name: str, kind: str) -> str:
    if kind == "key":
        return "key"
    if name in LEAKY:
        return "leaky"
    if name.startswith("truth_building_params__"):
        return "latent"
    if name.startswith("truth_lease_event__"):
        return "inferred" if "inferred" in name else "label"
    if name.startswith(("truth_floorplan_market_rent__", "truth_concession__")):
        return "label"
    return "feature"


class Builder:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.selects: list[str] = []
        self.catalog: list[dict] = []

    def add(self, expr: str, name: str, source: str, source_col: str, kind: str):
        self.selects.append(f"{expr} AS {name}")
        self.catalog.append({
            "column": name, "source_table": source, "source_column": source_col,
            "kind": kind, "role": role_of(name, kind),
        })

    def add_table(self, schema: str, table: str, alias: str, exclude: set[str] = frozenset()):
        for col, *_ in self.con.execute(f"DESCRIBE {schema}.{table}").fetchall():
            if col in KEYS or col in exclude:
                continue
            self.add(f'{alias}."{col}"', f"{schema}_{table}__{norm(col)}",
                     f"{schema}.{table}", col, "raw")

    def add_multihot(self, schema: str, table: str, alias: str, list_col: str, seg: str):
        """seg gets its own __<seg>__ path segment so a regex on the flags cannot
        also match a same-stemmed scalar (market/building.amenity_fee)."""
        vocab = [r[0] for r in self.con.execute(
            f"SELECT DISTINCT a FROM (SELECT unnest({list_col}) a FROM {schema}.{table}) ORDER BY 1"
        ).fetchall()]
        for item in vocab:
            lit = item.replace("'", "''")
            # An empty list means the source masked the attribute, so the flag is
            # unknown (NULL), not false. 513 of 3,000 buildings, matching the
            # generator's realized 17.1% amenity mask rate.
            expr = (f'CASE WHEN len({alias}."{list_col}") = 0 THEN NULL '
                    f'ELSE list_contains({alias}."{list_col}", \'{lit}\') END')
            self.add(expr, f"{schema}_{table}__{seg}__{norm(item)}",
                     f"{schema}.{table}", f"{list_col}[{item}]", "multihot")


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
        coalesce(sum(days) FILTER (WHERE concession > 0), 0) * 1.0
            / sum(days)                                                 AS share_days_with_concession,
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
        ce.unit_id,
        count(*)                                                AS n_events,
        sum(date_diff('day', ce.from_date, ce.to_date) + 1)     AS n_days_advertised,
        count(*) FILTER (WHERE ce.type = 'upfront_free_months') AS n_upfront_free_months,
        count(*) FILTER (WHERE ce.type = 'recurring_discount')  AS n_recurring_discount,
        count(*) FILTER (WHERE ce.type = 'one_time_credit')     AS n_one_time_credit,
        count(*) FILTER (WHERE ce.type = 'look_and_lease')      AS n_look_and_lease,
        count(*) FILTER (WHERE ce.type = 'unknown')             AS n_unknown,
        arg_min(ce.type, ce.from_date)                          AS first_type,
        arg_min(ce.specials, ce.from_date)                      AS first_specials_text,
        max(tc.total_value)                                     AS truth_max_total_value,
        max(tc.amortized_monthly_value)                         AS truth_max_amortized_monthly,
        arg_min(tc.type, ce.from_date)                          AS truth_first_type,
        count(*) FILTER (WHERE tc.is_generic)                   AS truth_n_generic,
        count(*) FILTER (WHERE tc.advertises_net_effective)     AS truth_n_net_effective
    FROM market.concession_event ce
    LEFT JOIN truth.concession tc ON tc.concession_event_id = ce.concession_event_id
    WHERE ce.unit_id IS NOT NULL
    GROUP BY ce.unit_id
)"""

PRICE_COLS = [
    "n_windows", "n_price_changes", "first_asking_rent", "last_asking_rent", "min_asking_rent",
    "max_asking_rent", "day_weighted_asking_rent", "first_effective_rent", "last_effective_rent",
    "day_weighted_effective_rent", "first_asking_rent_psf", "last_asking_rent_psf",
    "max_concession", "day_weighted_concession", "share_days_with_concession",
    "leased_asking_rent", "leased_effective_rent", "leased_asking_rent_psf",
    "leased_effective_rent_psf", "leased_concession",
]
CONCESSION_COLS = [
    "n_events", "n_days_advertised", "n_upfront_free_months", "n_recurring_discount",
    "n_one_time_credit", "n_look_and_lease", "n_unknown", "first_type", "first_specials_text",
]
CONCESSION_TRUTH_COLS = [
    "truth_max_total_value", "truth_max_amortized_monthly", "truth_first_type",
    "truth_n_generic", "truth_n_net_effective",
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
    # floor_plan.sqft is identical to unit.sqft in 100% of rows; unit_count is the
    # only column it adds.
    b.add_table("market", "floor_plan", "fp",
                exclude={"floorplan_name", "bed", "bath", "partial_bath", "sqft"})
    b.add_table("market", "building", "bld",
                exclude={"geography", "building_amenities", "unit_amenities"})
    # NULL rather than 0 where the list was masked, so the count agrees with the flags.
    b.add("CASE WHEN len(bld.building_amenities) = 0 THEN NULL ELSE len(bld.building_amenities) END",
          "market_building__n_building_amenities", "market.building", "building_amenities", "aggregate")
    b.add("CASE WHEN len(bld.unit_amenities) = 0 THEN NULL ELSE len(bld.unit_amenities) END",
          "market_building__n_unit_amenities", "market.building", "unit_amenities", "aggregate")
    b.add_multihot("market", "building", "bld", "building_amenities", "amen")
    b.add_multihot("market", "building", "bld", "unit_amenities", "unitamen")
    b.add_table("truth", "building_params", "bp", exclude={"submarket"})

    for c in PRICE_COLS:
        b.add(f"pa.{c}", f"market_price_observation__{c}", "market.price_observation", c, "aggregate")
    for c in CONCESSION_COLS:
        b.add(f"ca.{c}", f"market_concession_event__{c}", "market.concession_event", c, "aggregate")
    for c in CONCESSION_TRUTH_COLS:
        b.add(f"ca.{c}", f"truth_concession__{c[6:]}", "truth.concession", c[6:], "aggregate")

    b.add("la.leased_percentage", "market_leasing_activity_daily__leased_pct_at_enter",
          "market.leasing_activity_daily", "leased_percentage", "aggregate")
    b.add("la.exposure_percentage", "market_leasing_activity_daily__exposure_pct_at_enter",
          "market.leasing_activity_daily", "exposure_percentage", "aggregate")

    # True market rent for this floor plan in the month the listing went live:
    # the market level, not this listing's own price, so it is a clean label.
    b.add("fmr.true_market_rent", "truth_floorplan_market_rent__true_market_rent_at_enter",
          "truth.floorplan_market_rent_monthly", "true_market_rent", "aggregate")
    b.add("fmr.true_market_rent_psf", "truth_floorplan_market_rent__true_market_rent_psf_at_enter",
          "truth.floorplan_market_rent_monthly", "true_market_rent_psf", "aggregate")

    b.add("pp.sid IS NOT NULL", "pms_property__is_pms", "pms.property", "sid", "aggregate")
    b.add_table("pms", "property", "pp", exclude={
        "name", "street", "city", "state", "zipcode", "country", "latitude", "longitude",
        "url", "timeCreated", "timeUpdated"})

    names = [r["column"] for r in b.catalog]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate output columns: {sorted(dupes)}"

    # Joined outside the f-string: a backslash in an f-string expression is a
    # SyntaxError before Python 3.12.
    selects = ",\n        ".join(b.selects)
    sql = f"""
    WITH {PRICE_AGG.strip()},
    {CONCESSION_AGG.strip()}
    SELECT
        {selects}
    FROM market.unit u
    LEFT JOIN truth.lease_event le             ON le.unit_id = u.unit_id
    LEFT JOIN market.floor_plan fp             ON fp.floor_plan_id = u.floor_plan_id
    LEFT JOIN market.building bld              ON bld.property_id = u.property_id
    LEFT JOIN truth.building_params bp         ON bp.property_id = u.property_id
    LEFT JOIN price_agg pa                     ON pa.unit_id = u.unit_id
    LEFT JOIN concession_agg ca                ON ca.unit_id = u.unit_id
    LEFT JOIN market.leasing_activity_daily la ON la.property_id = u.property_id
                                              AND la."As of Day" = u.enter_market
    LEFT JOIN truth.floorplan_market_rent_monthly fmr
                                               ON fmr.floor_plan_id = u.floor_plan_id
                                              AND fmr.month = date_trunc('month', u.enter_market)
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
        w = csv.DictWriter(fh, fieldnames=["column", "source_table", "source_column", "kind", "role"])
        w.writeheader()
        w.writerows(b.catalog)

    roles: dict[str, int] = {}
    for r in b.catalog:
        roles[r["role"]] = roles.get(r["role"], 0) + 1
    print(f"wrote {OUT} ({n:,} rows x {len(b.selects)} cols, {OUT.stat().st_size / 1e6:.0f} MB)")
    print(f"wrote {CATALOG}")
    print("roles: " + ", ".join(f"{k}={v}" for k, v in sorted(roles.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
