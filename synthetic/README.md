# Synthetic data

This track records the first half of the project: analysis using Accolade's generated portfolio for Chicago, Dallas–Fort Worth and Washington, DC.
It includes the truth layer, pipeline, notebooks, outputs and midterm report.

```
notebooks/phase_01_market_rent/         market rent analysis
notebooks/phase_02_price_recommendation/ concession extraction
pipeline/                                data builds and batch analyses
libs/                                    reusable analysis helpers
outputs/                                 generated tables and figures
reports/midterm/                         report source and artifacts
```

Run each phase's notebooks in order. See [data/README.md](data/README.md) for setup.

## Phase 1: market rent

1. `01_master_table_check` checks master-table row counts, keys and missing values.
2. `02_feature_funnel` sorts columns by type and role.
3. `03_numerical_selection` scores numerical predictors. Needs 02.
4. `04_numerical_correlations` finds redundant numerical pairs. Needs 03.
5. `05_numerical_eda` plots candidate distributions and relationships. Needs 03 and 04.
6. `06_numerical_final_selection` applies `numerical_decisions.csv`. Needs 03 and 04.
7. `07_staged_selection_funnel` runs univariate, bivariate and multivariate selection.
8. `08_categorical_selection` scores categorical predictors with `pipeline/analyze_categorical.py`.
9. `09_model_training` compares linear regression, a decision tree and gradient boosting.

## Phase 2: price recommendation

`notebooks/phase_02_price_recommendation/01_concession_extraction` extracts concessions using `pipeline/extract_concessions.py` and scores them against the truth layer.

## Parameters and filters

Change values only in each notebook's parameters cell. Synthetic notebooks accept `CITY`, `NEIGHBORHOOD`, `START_DATE`, `END_DATE`, and `DATE_COLUMN`. City and neighborhood accept a string or list; dates include listings whose date falls between the start and end, inclusive. Filtered output names include the active filters so they do not overwrite full runs.
