"""Rank numeric features for market rent prediction. 

--Keeps only master table columns with role == "feature" and type_group == "numeric", and scores each against rent with:
    --pearson / spearman: linear / monotonic association
    --mi_coef: nonlinear correlation
    --perm_dMAE: importance in a gradient-boosted tree model based on permutation of each feature
    --within_pearson: correlation within a property
"""

import argparse
import warnings
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "data" / "master_listing.parquet"
CATALOG = REPO / "data" / "master_columns.csv"
OUT_CSV = REPO / "data" / "numeric_feature_ranking.csv"
OUT_PNG = REPO / "data" / "numeric_feature_ranking.png"

TARGETS = {
    "market_rent": "truth_floorplan_market_rent__true_market_rent_at_enter",  # market-level, cleaner
    "true_rent": "truth_lease_event__true_rent",  # listing-level
}
# Known only after the listing plays out; role tags do not mark these as leaky.
POST_TABLES = {"market.price_observation", "market.concession_event"}
POST_COLS = {"market_unit__days_on_market"}

SEED = 42


def load(target: str, include_post: bool, sample_n: int):
    cat = pd.read_csv(CATALOG)
    feat = cat[(cat.role == "feature") & (cat.type_group == "numeric")]
    if not include_post:
        feat = feat[~(feat.source_table.isin(POST_TABLES) | feat.column.isin(POST_COLS))]
    cols = feat.column.tolist()

    sel = ", ".join(f'"{c}"' for c in ["property_id", target] + cols)
    df = duckdb.connect().execute(f"""
        SELECT {sel} FROM read_parquet('{PARQUET.as_posix()}')
        WHERE "{target}" IS NOT NULL
        USING SAMPLE reservoir({sample_n} ROWS) REPEATABLE ({SEED})
    """).df()

    y = df[target].astype(float)
    groups = df["property_id"]
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    const = [c for c in X if X[c].nunique(dropna=True) <= 1]
    if const:
        print(f"dropping {len(const)} constant columns: {const[:5]}{' ...' if len(const) > 5 else ''}")
    return X.drop(columns=const), y, groups


def univariate(X, y, mi_n):
    res = pd.DataFrame(index=X.columns)
    res["pearson"] = X.corrwith(y)
    res["spearman"] = X.rank().corrwith(y.rank())

    idx = np.random.RandomState(SEED).choice(len(X), min(mi_n, len(X)), replace=False)
    Xm = X.iloc[idx].fillna(X.median())  # MI estimator cannot take NaN
    mi = mutual_info_regression(Xm.values, y.iloc[idx].values, n_neighbors=3, random_state=SEED)
    res["mi_nats"] = mi
    res["mi_coef"] = np.sqrt(1 - np.exp(-2 * mi))
    res["nonlinear_gap"] = res.mi_coef - res.pearson.abs()
    return res


def model_importance(X, y, groups, reps=3):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups))
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=SEED)
    gb.fit(X.iloc[tr], y.iloc[tr])
    pred = gb.predict(X.iloc[te])
    print(f"GBM holdout (unseen properties): R2={r2_score(y.iloc[te], pred):.3f}  "
          f"MAE=${mean_absolute_error(y.iloc[te], pred):,.0f}")

    rng = np.random.RandomState(SEED)
    sub = rng.choice(te, min(20_000, len(te)), replace=False)
    Xs, ys = X.iloc[sub].reset_index(drop=True), y.iloc[sub].values
    base = mean_absolute_error(ys, gb.predict(Xs))
    out = {}
    for c in X.columns:
        deltas = []
        for _ in range(reps):
            Xp = Xs.copy()
            Xp[c] = rng.permutation(Xs[c].values)
            deltas.append(mean_absolute_error(ys, gb.predict(Xp)) - base)
        out[c] = np.mean(deltas)
    return pd.Series(out, name="perm_dMAE")


def within_property(X, y, groups):
    g = groups.values
    Xd = X - X.groupby(g).transform("mean")
    yd = y - y.groupby(g).transform("mean")
    varies = Xd.abs().sum() > 0
    print(f"{varies.sum()} of {X.shape[1]} features vary within a property; "
          f"the rest are building-level constants")
    return Xd.loc[:, varies].corrwith(yd).rename("within_pearson")


def redundant_pairs(X, ranked, k=30, thresh=0.9):
    top = ranked.index[:k]
    cm = X[top].rank().corr()
    pairs = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack().rename("rho").reset_index()
    pairs = pairs[pairs.rho.abs() > thresh].sort_values("rho", key=abs, ascending=False)
    return pairs.rename(columns={"level_0": "a", "level_1": "b"})


def plot(res, path, n=25):
    top = res.sort_values("mi_coef", ascending=False).head(n)[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top.index.str.replace("market_", "m_", regex=False).str[:55], top.mi_coef,
            alpha=.8, label="MI coefficient")
    ax.scatter(top.pearson.abs(), range(len(top)), color="k", zorder=3, label="|Pearson|")
    ax.set_xlabel("dependence with rent")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS, default="market_rent")
    ap.add_argument("--include-post-listing", action="store_true",
                    help="keep days_on_market and price-window / concession aggregates")
    ap.add_argument("--sample-n", type=int, default=200_000)
    ap.add_argument("--mi-n", type=int, default=50_000)
    a = ap.parse_args()

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)
    X, y, groups = load(TARGETS[a.target], a.include_post_listing, a.sample_n)
    print(f"target={TARGETS[a.target]}  X={X.shape}")

    res = univariate(X, y, a.mi_n)
    res["perm_dMAE"] = model_importance(X, y, groups)
    res["within_pearson"] = within_property(X, y, groups)

    res["avg_rank"] = pd.concat([
        res.spearman.abs().rank(ascending=False),
        res.mi_coef.rank(ascending=False),
        res.perm_dMAE.rank(ascending=False),
    ], axis=1).mean(axis=1)
    res = res.sort_values("avg_rank")

    cols = ["pearson", "spearman", "mi_coef", "nonlinear_gap",
            "perm_dMAE", "within_pearson", "avg_rank"]
    print("\n Top 30 by average rank (Spearman, MI, permutation)")
    print(res[cols].head(30).round(3).to_string())
    print("\nBiggest nonlinear gaps (MI >> |Pearson|)")
    print(res.sort_values("nonlinear_gap", ascending=False)[cols].head(10).round(3).to_string())
    print("\n Redundant pairs among the top 30 (|rho| > 0.9)")
    print(redundant_pairs(X, res).round(3).to_string(index=False))
    plot(res, OUT_PNG)


if __name__ == "__main__":
    main()