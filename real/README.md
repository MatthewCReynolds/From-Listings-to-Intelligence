# Real data

The active track estimates market rent from Dwellsy exports for Chicago and Washington, DC.
Put CSV or ZIP exports in `data/raw/chicago/` and `data/raw/dc/`; intermediate files go in `data/processed/` and OSM extracts in `data/osm/`.

```
notebooks/phase_01_market_rent/  analysis in run order
pipeline/                        batch EDA plots
libs/                            data loading, types, categories and geography
outputs/eda/                     committed EDA figures and tables
docs/                            project phases, to-do list, comp selection ideas
data/                            local only: raw/ (Dwellsy exports), processed/, osm/
```

## Phase 1: market rent

Run notebooks in order from `notebooks/phase_01_market_rent/`:

1. `01_data_exploration` counts listings and saves a Chicago sample.
2. `02_data_preparation` cleans Chicago listings and writes `data/processed/chicago_prepared_full.csv`.
3. `03_categorical_analysis` measures how categorical features and amenities relate to rent.
4. `04_geospatial_features` builds points of interest and distances from OpenStreetMap. The first run downloads state extracts to `data/osm/`.
5. `05_baseline_models` trains and compares baseline regression models on Chicago. Needs 02.
6. `06_geospatial_model` fits a Ridge model with spline terms on latitude and longitude. Needs 02.
