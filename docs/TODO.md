# What we still need to do

Updated 2026-11-11. The semester ends in early December. Dates follow the Future Work section of the
midterm report. Put your name next to an item when you take it, and tick it when it is merged.

## Real data (`notebooks/real/`)

- [ ] Report the median absolute percentage error (MdAPE) on the Chicago models, so real and synthetic
      results can be compared directly. (November)
- [ ] Add amenities and the categorical features from `03_categorical_analysis` to the models. (November)
- [ ] Add the geospatial features from `04_geospatial_features` to the models. (November)
- [ ] Run the pipeline on Washington DC. Only Chicago has been modeled so far. (November)
- [ ] Test on the most recent period, since listings fell from about 50,000 to 15,000 per quarter after
      2021. (late November)

## Synthetic data (`notebooks/synthetic/`)

- [ ] Merge the numerical and categorical selections under the same exclusion rules, starting with
      property class, and train with cross-validation grouped by property. (mid-November)
- [ ] Build the comparable-property baseline from Accolade's plan (location, quality, size, vintage,
      amenities) and compare our models against it. (mid-November)

## Both

- [ ] Tune gradient boosting and Random Forest with Optuna; test XGBoost and LightGBM. (late November)
- [ ] Add prediction intervals (quantile or conformal), report their coverage, and explain predictions
      with SHAP. (late November)
- [ ] Start Phase 2: concession extraction on real data and the demand-based price recommendation.
      (early December)
- [ ] Evaluate the final model once on the held-out test set. (early December)

## Midterm report (Overleaf)

- [ ] Reconcile the number of numerical candidates: 44 in Feature selection vs 108 in Contributions.
- [ ] Reconcile the boolean predictor count: 215 vs 219.
- [ ] Fix the Stang et al. reference: the journal is now the *German Journal of Real Estate Research*,
      vol. 9, issue 2, pp. 81–108.
- [ ] Each member confirms their Contributions paragraph.
- [ ] Remove the `\revised{}` wrappers once the team accepts the edits.
