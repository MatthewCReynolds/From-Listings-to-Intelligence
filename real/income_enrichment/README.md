# Income enrichment (median household income by ZIP)

Tests whether adding Census ACS median household income by ZIP improves the preliminary rent models in `phase_01_market_rent/05_baseline_models.ipynb`.

## Setup

1. Download ACS data once (free key from https://api.census.gov/data/key_signup.html, kept in an environment variable, never committed):
   ```
   export CENSUS_API_KEY=...
   python -c "import income; income.fetch_acs_json()"   # run from real/income_enrichment/
   ```
   or place the JSON at `real/data/external/acs_2024_median_income_zcta.json`. `real/data/*` is gitignored.
2. Make sure `real/data/processed/chicago_prepared_full.csv` exists (output of `02_data_preparation.ipynb`).

## Run

1. `01_income_features.ipynb` joins income onto listings, reports ZIP coverage, saves `data/processed/chicago_income_enriched.csv`.
2. `02_income_model_comparison.ipynb` fits the four models with and without income on the notebook-05 split and on a held-out-ZIP split; writes `outputs/income/`.

Code in this folder: `income.py` (ACS loading and join), `rent_models.py` (split, models, metrics). It imports `libs/paths.py` from `real/`, so keep this folder inside `real/`.

## Caveats

One 2020-2024 income snapshot is applied to 2020-2026 listings; ZIP and ZCTA do not match perfectly (unmatched ZIPs get missing income plus an `income_missing` flag); incomes are top-coded at 250,001.
