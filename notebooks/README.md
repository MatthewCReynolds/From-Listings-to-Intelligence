# Notebooks

## Layout and run order

Phase 1 market rent analysis lives in `phase_01_market_rent/` and runs in order:

1. `01_master_table_check.ipynb`
2. `02_feature_funnel.ipynb`
3. `03_feature_selection_numerical.ipynb`
4. `04_numerical_correlations.ipynb`
5. `05_numerical_final_selection.ipynb`

Phase 2 planning notes live in `phase_02_price_recommendation/`. No Phase 2 code exists yet.

## Parameters and filters

Each notebook has a parameter cell for `INPUT`, `OUTPUT_DIR`, and optional `CITY`, `NEIGHBORHOOD`, `START_DATE`, `END_DATE`, and `DATE_COLUMN`. City and neighborhood accept a string or list. Dates keep listings whose selected date is on or after the start and on or before the end. Filtered output names use `__city__neighborhood__start_to_end`; unset parts are omitted (for example, `__chicago__2023-01-01_to_end`).

Notebook specific scoring and reporting thresholds are in the same parameter cell. All CSV, XLSX, and PNG outputs go to `outputs/`.
