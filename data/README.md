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

## Environment

Requires `pyarrow >= 20`; the files are written by Arrow 24, and pyarrow 19 fails to read them with
`Repetition level histogram size mismatch`. The repo `.venv` is set up correctly:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```
