# From Listings to Intelligence

Columbia DS Capstone, Fall 2026, with Accolade. We estimate the market rent of rental units from
listing data (Phase 1) and then recommend prices from demand (Phase 2). The work runs on two datasets:

- **Synthetic data**: Accolade's generated portfolio for Chicago, Dallas–Fort Worth and Washington,
  DC. It has a truth layer, so we can measure error against the true market rent.
- **Real data**: Dwellsy listing exports for Chicago and Washington, DC. It has no truth layer, so we
  score models against asking rent.

## Layout

```
libs/                     shared code; notebooks and scripts import from here
  paths.py                every data and output location
  console.py              step / done / note progress messages
  notebook.py             Run: listing filters, output names, and loading the master table
  real_data.py            Dwellsy loaders (raw_file, load_real, count_csv_records)
  dtypes.py               type standardization
  stats.py                gini, eta_squared, cramers_v
  categorical.py          categorical and amenity analysis (real data)
  geo.py                  OpenStreetMap point-of-interest features
  plots.py                shared figure style
pipeline/synthetic/       scripts that build the synthetic master table and run batch analyses
pipeline/real/            scripts for the real data
notebooks/synthetic/      the synthetic-data analysis, in run order
notebooks/real/           the real-data analysis, in run order
outputs/synthetic/        tables and figures from synthetic data
outputs/real/             tables and figures from real data
reports/midterm/          the midterm report source (the team edits the text in Overleaf)
docs/                     project phases and the to-do list
data/                     local data only, never committed (see data/README.md)
```

Notebook run order and parameters: [notebooks/README.md](notebooks/README.md).
What is left to do: [docs/TODO.md](docs/TODO.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline/synthetic/download_data.py    # synthetic data -> data/realistic-v2/
python pipeline/synthetic/build_duckdb.py     # -> data/accolade.duckdb
python pipeline/synthetic/build_master.py     # -> data/master_listing.parquet
```

For the real data, put the Dwellsy files in `data/real/chicago/` and `data/real/dc/` (the extracted CSV
or the original zip). Details in [data/README.md](data/README.md).

## Working on the repo

- `master` is the only long-lived branch. Synthetic and real work live in their own folders.
- For a change, branch off `master` (`git switch -c <name>/<topic>`), push, open a pull request, and
  delete the branch once it is merged.
- Pull before you start: `git switch master && git pull`.
- Put reusable functions in `libs/`, not in notebooks. A notebook keeps only its parameters, its
  analysis, and its plots.
- Old branches were deleted on 2026-11-11 after being merged. Each one is kept as a tag,
  `archive/<branch>`; restore one with `git switch -c <branch> archive/<branch>`.
