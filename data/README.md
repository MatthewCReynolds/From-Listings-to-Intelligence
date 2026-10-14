# Data

The dataset is **not** in git (1.5 GB unpacked). Fetch it locally:

```bash
python preprocessing/download_data.py   # downloads + checksums + unpacks
python preprocessing/build_duckdb.py    # builds data/accolade.duckdb
```

## Layout

`data/realistic-v2/` — hive-partitioned parquet (`market=chicago|dfw|dc`), 57 tables / ~46.1M rows:

| group | contents |
| --- | --- |
| `market/` | external/scraped view — buildings, units, price observations, leasing activity |
| `pms/` | property-management system records — leases, pricing, guestcards, recommendations |
| `truth/` | generator ground truth — demand params, market rents, lease events |
| `meta/` | `data_dictionary.md`, `manifest.json`, generation + validation reports |

Largest tables: `market/price_observation` (8.9M), `pms/unit_history` (8.1M), `market/leasing_activity_daily` (6.5M).

## Querying

Use `data/accolade.duckdb`, which exposes every table as a view under the `market`, `pms`, and `truth` schemas:

```python
import duckdb
con = duckdb.connect("data/accolade.duckdb", read_only=True)
con.execute("select market, count(*) from market.building group by 1").fetchall()
```

Note the shipped `realistic-v2/accolade_synthetic.duckdb` is **unusable** — its views hardcode the
generator machine's absolute paths. `build_duckdb.py` regenerates them against local paths.

## Master table

```bash
python preprocessing/build_master.py     # ~15 s -> data/master_listing.parquet (161 MB)
```

`data/master_listing.parquet` — **one row per listing episode** (`market/unit`), 967,661 rows x 378 columns.
`data/master_columns.csv` (committed) maps every column to its source table, kind, and **role**.

## Select features by role, not by prefix

The prefix says where a column came from; the `role` column says what it is. These differ, and
the difference matters: the 14 `leaky` columns live in the `market_` namespace, so filtering on
the `truth_` prefix does **not** exclude them.

| role | n | meaning |
| --- | --- | --- |
| `key` | 4 | join keys |
| `feature` | 317 | usable as X |
| `label` | 17 | prediction targets (ground truth) |
| `leaky` | 14 | market-layer columns that *are* this listing's own advertised price (\|r\| 0.989–0.9998 vs `true_rent`) |
| `latent` | 23 | simulator internals (`truth/building_params`) — not observable in production |
| `inferred` | 3 | platform inferences available at prediction time |

```python
import pandas as pd
cat = pd.read_csv("data/master_columns.csv")
df  = pd.read_parquet("data/master_listing.parquet")
X = df[cat.loc[cat.role == "feature", "column"]]
y = df["truth_lease_event__true_rent"]
```

With that feature set the strongest single correlation with rent is **0.53** — a real but
non-trivial signal. Keep any `leaky` column and it jumps to 0.9998.

## Columns by source

Every column is named `<schema>_<table>__<column>`:

| prefix | rows populated | what |
| --- | --- | --- |
| `unit_id`, `property_id`, `floor_plan_id`, `market` | all | join keys, unprefixed |
| `market_unit__*` | all | the listing itself (bed/bath/sqft/floor, enter/exit dates, days_on_market) |
| `market_building__*` | all | 72 building attrs + 201 amenity multi-hot flags (`__amen__*`, `__unitamen__*`) |
| `market_floor_plan__*` | all | floor plan unit count |
| `market_price_observation__*` | 99.96% | per-listing aggregates of the price windows: first/last/min/max asking, n_price_changes, day-weighted rent & concession, `leased_*` final-window values |
| `market_concession_event__*` | ~8% | per-listing special counts by type + first specials text |
| `market_leasing_activity_daily__*` | 90% | building leased % and exposure % on the day the listing went live |
| `pms_property__*` | 5% | `is_pms` flag + PMS property settings (150 buildings) |
| `truth_lease_event__*` | all | **labels** — `true_rent`, `true_effective_rent`, `true_lease_start`, `outcome` (96% leased, 4% still active with null rent) |
| `truth_floorplan_market_rent__*` | 98.6% | **label** — true market rent for the floor plan in the month the listing went live. Market-level, so a cleaner target than `true_rent` |
| `truth_concession__*` | 6.4% | **label** — structured truth behind the free-text specials (total value, amortized monthly, type) |
| `truth_building_params__*` | all | **latent** — true class, pricing policy, true unit count |

Gotchas:
- `market_building__*_quality_score` (13 cols) are all null — the generator's quality-score toggle was off.
- Amenity multi-hot flags are NULL (not False) for the 513 buildings whose amenity list was masked, and
  `n_building_amenities` / `n_unit_amenities` are NULL for those same rows rather than 0.
- `market_building__amenity_fee` is a dollar amount, not a flag. The flags live under `__amen__` /
  `__unitamen__` precisely so a regex on them cannot catch it.
- `share_days_with_concession` is 0.0 when a listing never carried a special, and NULL only for the
  384 listings with no price windows at all.
- `market_leasing_activity_daily__*` is null for ~9% of rows: listings can enter before that table's
  history starts (2020-01-01, vs listings from 2019-06-01).
- Cross-validate with `GroupKFold(groups=property_id)`; 3,000 buildings repeat ~320x each.

## Environment

Requires `pyarrow >= 20`; the files are written by Arrow 24, and pyarrow 19 fails to read them with
`Repetition level histogram size mismatch`. The repo `.venv` is set up correctly:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```
