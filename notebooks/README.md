# Notebooks

Run each track in order. Every notebook starts with the same three cells: a description (data,
inputs, outputs, what to run first), a parameters cell, and a setup cell that imports `libs/`.
Tables and figures go to `outputs/synthetic/` or `outputs/real/`, named
`<notebook number>_phase_<phase>_<name>`.

## Synthetic data (`synthetic/`)

Needs `data/master_listing.parquet` and `data/master_columns.csv` (see `data/README.md`).

Phase 1, market rent (`synthetic/phase_01_market_rent/`):

1. `01_master_table_check` checks row counts, keys, and missing values in the master table.
2. `02_feature_funnel` sorts every column by type and role (predictor, leakage, identifier).
3. `03_numerical_selection` scores the numerical predictors with Gini, gradient boosting, and Lasso.
   Needs 02.
4. `04_numerical_correlations` finds redundant numerical pairs and writes the correlation workbook.
   Needs 03.
5. `05_numerical_eda` plots distributions, missingness, correlations, and monthly rent of the
   candidates. Needs 03 and 04.
6. `06_numerical_final_selection` applies the team's decisions in `numerical_decisions.csv` and checks
   stability by city. Needs 03 and 04.
7. `07_staged_selection_funnel` runs the univariate, bivariate, and multivariate selection stages.
8. `08_categorical_selection` scores the categorical predictors (eta², Cramér's V, held-out importance)
   with `pipeline/synthetic/analyze_categorical.py`.
9. `09_model_training` trains and compares linear regression, a decision tree, and gradient boosting.

Phase 2, price recommendation (`synthetic/phase_02_price_recommendation/`):

1. `01_concession_extraction` extracts concessions from the specials text with
   `pipeline/synthetic/extract_concessions.py` and scores them against the truth layer.

## Real data (`real/`)

Needs the Dwellsy exports in `data/real/chicago/` and `data/real/dc/` (see `data/README.md`).
Intermediate data files go to `data/processed/`.

Phase 1, market rent (`real/phase_01_market_rent/`):

1. `01_data_exploration` counts the listings in both markets and saves a Chicago sample.
2. `02_data_preparation` cleans the Chicago listings and writes `data/processed/chicago_prepared_full.csv`.
3. `03_categorical_analysis` measures how categorical features and amenities relate to rent (eta²).
4. `04_geospatial_features` builds point-of-interest counts and distances from OpenStreetMap. The first
   run downloads the state extracts from Geofabrik to `data/osm/`.
5. `05_baseline_models` trains and compares the first regression models on Chicago. Needs 02.
6. `06_geospatial_model` fits a Ridge regression with spline terms on latitude and longitude. Needs 02.

## Parameters and filters

Change values only in the parameters cell. The synthetic notebooks accept `CITY`, `NEIGHBORHOOD`,
`START_DATE`, `END_DATE`, and `DATE_COLUMN`. City and neighborhood take a string or a list; dates keep
listings whose `DATE_COLUMN` falls between the start and the end, inclusive. Active filters are added
to the output names, for example `03_phase_01_feature_selection_numerical__chicago__2023-01-01_to_end.csv`,
so filtered runs never overwrite the full run.
