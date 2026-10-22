"""Time-to-lease survival models: RandomSurvivalForest vs XGBoost (Cox).

Target: days a listing sits on market. The 35,923 listings still active at the
as-of date are right-censored, which is why this is a survival problem and not
a regression.

The `role` column in master_columns.csv marks leakage for the RENT target. It
does not apply here: for time-to-lease, anything accumulated over the listing's
life leaks the answer (n_price_changes grows with duration, leased_* only exist
once leased, building avg_time_on_market_days is computed from these listings).
So features are restricted to what is known on enter_market.

STATUS: not yet validated. The data setup below is checked (3.6% censoring,
grouped split with zero building overlap), but no C-index has been produced:
RandomSurvivalForest did not finish on this data. It ran >10 min on 40k rows
and >16 min on 12k before being killed, because scikit-survival evaluates a
log-rank split statistic at every candidate split and scales far worse than
gradient boosting. XGBoost sits behind it in main() and so never ran.

To get a number quickly, run the XGBoost section alone, or drop rsf_n to ~2000.
"""

import time as _t

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import xgboost as xgb

RNG = 42
PQ = "data/master_listing.parquet"

# Known at enter_market.
NUMERIC = [
    "market_unit__bed", "market_unit__bath", "market_unit__partial_bath",
    "market_unit__floor", "market_unit__sqft",
    "market_floor_plan__unit_count",
    "market_price_observation__first_asking_rent",
    "market_price_observation__first_asking_rent_psf",
    "market_price_observation__first_effective_rent",
    "market_building__number_units", "market_building__year_built",
    "market_building__number_stories",
    "market_building__n_building_amenities", "market_building__n_unit_amenities",
    "market_building__lat", "market_building__lon",
    "market_building__count_reviews", "market_building__avg_review_score",
    "market_building__admin_fee", "market_building__amenity_fee",
    "market_building__application_fee",
    "market_building__cats_monthly_rent", "market_building__dogs_monthly_rent",
    "market_building__parking_garage",
    "market_leasing_activity_daily__leased_pct_at_enter",
    "market_leasing_activity_daily__exposure_pct_at_enter",
]
BOOL = [
    "market_building__is_apartment", "market_building__is_condo",
    "market_building__is_student", "market_building__is_senior",
    "market_building__is_affordable", "market_building__is_lease_up",
    "market_building__is_build_to_rent",
    "market_building__is_using_rev_management",
    "pms_property__is_pms",
]
CATEGORICAL = ["market", "market_building__property_class", "market_building__submarket"]
DATES = ["market_unit__enter_market", "market_unit__availability"]
TARGET = ["market_unit__days_on_market", "market_unit__status"]
GROUP = "property_id"


def load():
    df = pd.read_parquet(PQ, columns=NUMERIC + BOOL + CATEGORICAL + DATES + TARGET + [GROUP])

    # Event = the listing closed (a lease was signed). Still-active listings are
    # censored: we know they lasted at least this long, not how long in total.
    event = (df["market_unit__status"] == "closed").to_numpy(dtype=bool)
    time = df["market_unit__days_on_market"].astype("float64").to_numpy()

    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c.split("__", 1)[1]] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in BOOL:
        X[c.split("__", 1)[1]] = df[c].astype("float64")
    # Seasonality: leasing speed swings with the month a unit hits the market.
    enter = pd.to_datetime(df["market_unit__enter_market"])
    X["enter_month"] = enter.dt.month.astype("float64")
    X["enter_year"] = enter.dt.year.astype("float64")
    # Advertised lead time: days between listing and the move-in date offered.
    X["lead_days"] = (pd.to_datetime(df["market_unit__availability"]) - enter).dt.days.astype("float64")
    for c in CATEGORICAL:
        X[c.split("__", 1)[-1]] = df[c].astype("category").cat.codes.astype("float64")

    keep = np.isfinite(time) & (time > 0)
    return X[keep].reset_index(drop=True), event[keep], time[keep], df.loc[keep, GROUP].to_numpy()


def main():
    X, event, time, groups = load()
    print(f"rows {len(X):,} | features {X.shape[1]} | buildings {len(np.unique(groups)):,}")
    print(f"events (leased) {event.sum():,} | censored (active) {(~event).sum():,} "
          f"({100 * (~event).mean():.1f}%)")
    print(f"days on market: median {np.median(time):.0f}  p90 {np.percentile(time, 90):.0f}  max {time.max():.0f}")

    # Split by building so no building appears in both train and test.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RNG).split(X, groups=groups))
    print(f"\ntrain {len(tr):,} / test {len(te):,} rows; "
          f"{len(np.unique(groups[tr])):,} / {len(np.unique(groups[te])):,} buildings "
          f"(overlap {len(set(groups[tr]) & set(groups[te]))})")

    # RandomSurvivalForest cannot take NaN and scales poorly, so both models get
    # the same median-imputed subsample for a fair comparison.
    med = X.iloc[tr].median()
    Xtr_full, Xte = X.iloc[tr].fillna(med), X.iloc[te].fillna(med)

    rsf_n = min(12_000, len(tr))
    sub = np.random.default_rng(RNG).choice(len(tr), rsf_n, replace=False)
    Xtr, ytr_e, ytr_t = Xtr_full.iloc[sub], event[tr][sub], time[tr][sub]
    print(f"model training sample: {rsf_n:,} rows\n")

    ytr = Surv.from_arrays(event=ytr_e, time=ytr_t)
    cidx = lambda p: concordance_index_censored(event[te], time[te], p)[0]

    print("fitting RandomSurvivalForest ...", flush=True)
    _0 = _t.perf_counter()
    rsf = RandomSurvivalForest(n_estimators=60, min_samples_leaf=100, max_features="sqrt",
                               n_jobs=-1, random_state=RNG).fit(Xtr, ytr)
    c_rsf = cidx(rsf.predict(Xte))
    t_rsf = _t.perf_counter() - _0
    print(f"  RSF      C-index {c_rsf:.4f}   ({t_rsf:.0f}s)", flush=True)

    print("fitting XGBoost (survival:cox) ...", flush=True)
    _0 = _t.perf_counter()
    # XGBoost's Cox convention: censored rows carry a negative time.
    lab = np.where(ytr_e, ytr_t, -ytr_t)
    xgb_m = xgb.XGBRegressor(objective="survival:cox", n_estimators=400, learning_rate=0.05,
                             max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=20, random_state=RNG, n_jobs=-1)
    xgb_m.fit(Xtr, lab)
    c_xgb = cidx(xgb_m.predict(Xte))
    t_xgb = _t.perf_counter() - _0
    print(f"  XGBoost  C-index {c_xgb:.4f}   ({t_xgb:.0f}s)", flush=True)

    print("fitting XGBoost on the full training set (NaNs kept) ...", flush=True)
    _0 = _t.perf_counter()
    xgb_full = xgb.XGBRegressor(objective="survival:cox", n_estimators=600, learning_rate=0.05,
                                max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                min_child_weight=20, random_state=RNG, n_jobs=-1)
    xgb_full.fit(X.iloc[tr], np.where(event[tr], time[tr], -time[tr]))
    c_full = cidx(xgb_full.predict(X.iloc[te]))
    t_fl = _t.perf_counter() - _0
    print(f"  XGBoost (full {len(tr):,} rows) C-index {c_full:.4f}   ({t_fl:.0f}s)", flush=True)

    print("\n--- top 15 features (XGBoost full, gain) ---")
    imp = (pd.Series(xgb_full.get_booster().get_score(importance_type="gain"))
           .sort_values(ascending=False).head(15))
    for k, v in imp.items():
        print(f"  {v:10.1f}  {k}")

    print("\n--- C-index: 0.5 = coin flip, 1.0 = perfect ranking of who leases first ---")
    for name, c, n, el in [("RandomSurvivalForest", c_rsf, rsf_n, t_rsf),
                           ("XGBoost Cox", c_xgb, rsf_n, t_xgb),
                           ("XGBoost Cox", c_full, len(tr), t_fl)]:
        print(f"  {name:22} {c:.4f}   trained on {n:>7,} rows in {el:>5.0f}s")


if __name__ == "__main__":
    main()
