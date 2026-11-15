# From Listings to Intelligence

Columbia DS Capstone, Fall 2026, with Accolade. We estimate market rent from rental listings (Phase 1) and recommend prices from demand (Phase 2). The project began with two datasets:

- **Synthetic data**: Accolade's generated portfolio for Chicago, Dallas–Fort Worth and Washington, DC, with a truth layer for measuring error. The `synthetic/` folder is kept as the record of the first half.
- **Real data**: Dwellsy listing exports for Chicago and Washington, DC. The team now works on the real data, which has no truth layer, so models are scored against asking rent.

## Layout

```
real/                 the current work: Dwellsy data for Chicago and DC
  notebooks/  pipeline/  libs/  outputs/  docs/  data/ (local only)
synthetic/            the first half: Accolade's synthetic portfolio, kept as a record
  notebooks/  pipeline/  libs/  outputs/  reports/midterm/  data/ (local only)
```

Each folder is self-contained: its notebooks and scripts import that folder's `libs/`, read its
`data/` and write to its `outputs/`. Run order: [real/README.md](real/README.md) and
[synthetic/README.md](synthetic/README.md). What is left to do: [real/docs/TODO.md](real/docs/TODO.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Where the data goes

For synthetic data, from the repository root:

```bash
cd synthetic && python pipeline/download_data.py && python pipeline/build_duckdb.py && python pipeline/build_master.py
```

For real data, put the Dwellsy exports in `real/data/raw/chicago/` and `real/data/raw/dc/`.

### Moving from the old layout

After pulling, move your local `data/` contents into `synthetic/data/` (`realistic-v2/`, the zip, `master_listing.parquet`, and `accolade.duckdb`) and `real/data/` (Dwellsy files into `raw/`, plus `processed/` and `osm/`). Then rebuild the DuckDB
views, which store absolute paths: `cd synthetic && python pipeline/build_duckdb.py` (a few seconds).

## Working on the repo

- `master` is the only long-lived branch. Synthetic and real work live in their own folders.
- For a change, branch off `master` (`git switch -c <name>/<topic>`), push, open a pull request, and delete the branch once it is merged.
- Pull before you start: `git switch master && git pull`.
- Put reusable functions in the track's `libs/`, not in notebooks. A notebook keeps only its parameters, its analysis, and its plots.
- Old branches were deleted on 2026-11-11 after being merged. Each one is kept as a tag, `archive/<branch>`; restore one with `git switch -c <branch> archive/<branch>`.
