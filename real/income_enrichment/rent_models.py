"""Rent models shared by the income-enrichment notebooks.

Reproduces the split, preprocessing and hyperparameters of
notebooks/phase_01_market_rent/05_baseline_models.ipynb, but takes the feature lists as
arguments so the same code can run with and without extra features.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

BASE_NUMERIC = ["bedrooms", "full_baths", "square_feet", "year_built", "listing_year", "listing_month"]
BASE_CATEGORICAL = ["address_zip", "uru_type"]
TARGET = "rent_amount"
GROUP = "uru_id"


class FrequentCategoryGrouper(BaseEstimator, TransformerMixin):
    """Keep the most frequent categories (fit on training data) and map the rest to OTHER."""

    def __init__(self, max_categories=200):
        self.max_categories = max_categories

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.frequent_categories_ = [
            set(X[c].value_counts().head(self.max_categories - 1).index) for c in X.columns
        ]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for i, c in enumerate(X.columns):
            X[c] = X[c].where(X[c].isin(self.frequent_categories_[i]), "OTHER")
        return X


def make_split(df, seed=710, group=GROUP):
    """Same 64/16/20 group split as notebook 05. Returns positional index arrays."""
    groups = df[group]
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
                     .split(df, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
                  .split(df.iloc[dev], groups=groups.iloc[dev]))
    return {"train": dev[tr], "val": dev[va], "test": test}


def make_zip_holdout_split(df, seed=710, zip_col="address_zip", test_zip_frac=0.20, group=GROUP):
    """Harder split: test ZIPs never appear in train/val, so ZIP identity cannot be memorised."""
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=test_zip_frac, random_state=seed)
                     .split(df, groups=df[zip_col]))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
                  .split(df.iloc[dev], groups=df[group].iloc[dev]))
    return {"train": dev[tr], "val": dev[va], "test": test}


def build_models(numeric, categorical, seed=710):
    """The four notebook-05 models, parameterised by feature lists."""
    linear_pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    gb_pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("grouper", FrequentCategoryGrouper(max_categories=200)),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
    ])
    cat_idx = list(range(len(numeric), len(numeric) + len(categorical)))
    return {
        "Linear Regression": Pipeline([("preprocessor", linear_pre), ("regressor", LinearRegression())]),
        "Decision Tree": Pipeline([("preprocessor", linear_pre),
                                   ("regressor", DecisionTreeRegressor(max_depth=10, random_state=seed))]),
        "Random Forest": Pipeline([("preprocessor", linear_pre),
                                   ("regressor", RandomForestRegressor(
                                       n_estimators=100, max_depth=15, min_samples_leaf=5,
                                       random_state=seed, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("preprocessor", gb_pre),
                                       ("regressor", HistGradientBoostingRegressor(
                                           max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                           categorical_features=cat_idx, random_state=seed))]),
    }


def score(y_true, y_pred):
    """MAE, RMSE, R2 and MdAPE (percent), as in notebook 05."""
    y_true = np.asarray(y_true, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": r2_score(y_true, y_pred),
        "MdAPE": float(np.median(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def run_models(df, split, numeric, categorical, feature_set, seed=710, eval_on=("val", "test")):
    """Fit every model on the train rows and score it on each split named in eval_on."""
    features = numeric + categorical
    X, y = df[features], df[TARGET]
    X_tr, y_tr = X.iloc[split["train"]], y.iloc[split["train"]]
    rows = []
    for name, model in build_models(numeric, categorical, seed).items():
        model.fit(X_tr, y_tr)
        for part in eval_on:
            idx = split[part]
            rows.append({"Model": name, "Features": feature_set, "Eval": part,
                         **score(y.iloc[idx], model.predict(X.iloc[idx]))})
    return pd.DataFrame(rows)
