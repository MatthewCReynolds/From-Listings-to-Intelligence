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

`data/master_listing.parquet` — **one row per listing episode** (`market/unit`), 967,661 rows x 375 columns.
`data/master_columns.csv` (committed) maps every column to its source table and kind.

Every column is prefixed `<schema>_<table>__<column>`:

| prefix | rows populated | what |
| --- | --- | --- |
| `unit_id`, `property_id`, `floor_plan_id`, `market` | all | join keys, unprefixed |
| `market_unit__*` | all | the listing itself (bed/bath/sqft/floor, enter/exit dates, days_on_market) |
| `market_building__*` | all | 72 building attrs + 201 amenity multi-hot flags (`amenity_*`, `unit_amenity_*`) |
| `market_floor_plan__*` | all | floor plan sqft and unit count |
| `market_price_observation__*` | 99.96% | per-listing aggregates of the price windows: first/last/min/max asking, n_price_changes, day-weighted rent & concession, `leased_*` final-window values |
| `market_concession_event__*` | ~8% | per-listing special counts by type + first specials text |
| `market_leasing_activity_daily__*` | 90% | building leased % and exposure % on the day the listing went live |
| `pms_property__*` | 5% | `is_pms` flag + PMS property settings (150 buildings) |
| `truth_lease_event__*` | all | **labels** — `true_rent`, `true_effective_rent`, `true_lease_start`, `outcome` (96% leased, 4% still active with null rent) |
| `truth_building_params__*` | all | **latent truth** — true class, pricing policy, true unit count. Leakage unless intentional |

Drop all ground truth from X with `df.filter(regex="^(?!truth_)")`.

Gotchas:
- `market_building__*_quality_score` (13 cols) are all null — the generator's quality-score toggle was off.
- Amenity multi-hot flags are NULL (not False) for the 513 buildings whose amenity list was masked.
- `market_price_observation__leased_asking_rent` correlates 0.9998 with `true_rent` — it is the last advertised
  price before the lease. Decide whether the model is allowed to see it before using it as a feature.
- Cross-validate with `GroupKFold(groups=property_id)`; 3,000 buildings repeat ~320x each.

## Environment

Requires `pyarrow >= 20`; the files are written by Arrow 24, and pyarrow 19 fails to read them with
`Repetition level histogram size mismatch`. The repo `.venv` is set up correctly:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```
