"""
--1 rent vs sqft (hexbin + binned median)
--2 geography: median rent over lat/lon, per market
--3 building-level check
--4 within vs between: sqft vs rent raw and after demeaning by property
--5 shape of the nonlinear effects
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
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "data" / "master_listing.parquet"
OUT = Path(__file__).resolve().parent / "plots"
SEED = 42

TARGET = "truth_floorplan_market_rent__true_market_rent_at_enter"
B, U = "market_building__", "market_unit__"
COLS = {
    "property_id": "property_id", "floor_plan_id": "floor_plan_id", "market": "market",
    "rent": TARGET,
    "sqft": U + "sqft", "bed": U + "bed", "bath": U + "bath", "enter": U + "enter_market",
    "lat": B + "lat", "lon": B + "lon", "year_built": B + "year_built",
    "n_amen": B + "n_building_amenities", "review": B + "avg_review_score",
    "n_reviews": B + "count_reviews", "price_chg": B + "avg_price_change",
    "units": B + "number_units", "cls": B + "property_class",
}


def load(n):
    sel = ", ".join(f'"{v}" AS {k}' for k, v in COLS.items())
    df = duckdb.connect().execute(f"""
        SELECT {sel} FROM read_parquet('{PARQUET.as_posix()}')
        WHERE "{TARGET}" IS NOT NULL
        USING SAMPLE reservoir({n} ROWS) REPEATABLE ({SEED})
    """).df()
    df["enter"] = pd.to_datetime(df["enter"])
    df["month"] = df.enter.dt.to_period("M").dt.to_timestamp()
    df["psf"] = df.rent / df.sqft
    return df


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("saved", name)



def p01_sqft(df):
    d = df.dropna(subset=["sqft", "rent"])
    d = d[d.sqft.between(*d.sqft.quantile([.005, .995]))]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    hb = ax.hexbin(d.sqft, d.rent, gridsize=60, bins="log", mincnt=1, cmap="viridis")
    q = d.groupby(pd.cut(d.sqft, 25)).agg(x=("sqft", "median"), y=("rent", "median"))
    ax.plot(q.x, q.y, "r-", lw=2, label="binned median")
    ax.set(xlabel="Sqft", ylabel="Rent", title="Rent vs Sqft"); ax.legend()
    fig.colorbar(hb, ax=ax, label="log10 count")
    save(fig, "01_rent_vs_sqft.png")

def p02_geo(df):
    d = df.dropna(subset=["lat", "lon"])
    mk = sorted(d.market.unique())
    fig, ax = plt.subplots(1, len(mk), figsize=(5.5 * len(mk), 5))
    ax = np.atleast_1d(ax)
    vmax = d.rent.quantile(.95)
    for a, m in zip(ax, mk):
        g = d[d.market == m]
        hb = a.hexbin(g.lon, g.lat, C=g.rent, reduce_C_function=np.median, gridsize=35,
                      mincnt=5, cmap="magma", vmin=d.rent.quantile(.05), vmax=vmax)
        market_name = {"chicago": "Chicago" , "dc": "DC", "dfw": "DFW"}.get(m.lower(), m.title())
        a.set(title=f"{market_name}: Median Rent", xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(hb, ax=ax, label="Median Rent ($)")
    fig.savefig(OUT / "02_geography.png", dpi=150); plt.close(fig); print("saved 02_geography.png")



def p03_building_level(df):
    feats = [("review", "avg_review_score"), ("price_chg", "avg_price_change"),
             ("year_built", "year_built"), ("n_amen", "n_building_amenities")]
    # one point per building: property mean rent vs its (constant) building feature
    pm = df.groupby("property_id").agg(rent=("rent", "mean"), n=("rent", "size"), **{k: (k, "first") for k, _ in feats})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for a, (k, label) in zip(ax.ravel(), feats):
        d = pm.dropna(subset=[k])
        rho = d[k].corr(d.rent, method="spearman")
        a.scatter(d[k], d.rent, s=8, alpha=.4)
        a.set(xlabel=label, ylabel="Property Mean Rent", title=f"{label}  (building-level rho={rho:.2f}, n={len(d):,})")
    fig.suptitle("One point per building: building features can only carry ~#buildings worth of information")
    save(fig, "03_building_level.png")


def p04_within_between(df):
    d = df.dropna(subset=["sqft", "rent"]).copy()
    d["sqft_w"] = d.sqft - d.groupby("property_id").sqft.transform("mean")
    d["rent_w"] = d.rent - d.groupby("property_id").rent.transform("mean")
    lim = d.sqft.quantile([.005, .995]).values
    d = d[d.sqft.between(*lim)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hexbin(d.sqft, d.rent, gridsize=50, bins="log", cmap="viridis", mincnt=1)
    ax[0].set(title=f"Raw: Pearson={d.sqft.corr(d.rent):.2f}", xlabel="Sqft", ylabel="Rent")
    ax[1].hexbin(d.sqft_w, d.rent_w, gridsize=50, bins="log", cmap="viridis", mincnt=1)
    ax[1].set(title=f"Within property: Pearson={d.sqft_w.corr(d.rent_w):.2f}",
              xlabel="Sqft - Building Mean", ylabel="Rent - Building Mean")
    save(fig, "04_within_vs_between_sqft.png")


def p05_pdp(df):
    feats = ["sqft", "bed", "bath", "lat", "lon", "n_amen", "year_built", "review", "units"]
    d = df.dropna(subset=["rent"]).reset_index(drop=True)
    X = d[feats].astype(float)
    tr, te = next(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X, d.rent, d.property_id))
    gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=SEED).fit(X.iloc[tr], d.rent.iloc[tr])
    show = ["sqft", "lat", "lon", "n_amen"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    PartialDependenceDisplay.from_estimator(
    gb, X.iloc[te].sample(min(5000, len(te)), random_state=SEED), show,
    grid_resolution=25, ax=ax, line_kw={"color": "C3"}
    )
    for a in ax.flat:
        a.set_ylabel("Partial Dependence")
    fig.suptitle("Partial Dependence of Predicted Rent (Compact GBM, Holdout = Unseen Properties)")
    save(fig, "05_partial_dependence.png")
    print(f"  compact GBM holdout R2={gb.score(X.iloc[te], d.rent.iloc[te]):.3f}")


def variance_decomposition(df):
    def r2(keys):
        g = df.groupby(keys).rent.transform("mean")
        return 1 - ((df.rent - g) ** 2).sum() / ((df.rent - df.rent.mean()) ** 2).sum()
    out = {"market": r2("market"), "month": r2("month"), "market x month": r2(["market", "month"]),
           "property_id": r2("property_id"), "floor_plan_id": r2("floor_plan_id"),
           "property_id x month": r2(["property_id", "month"])}
    print("\nShare of rent variance explained by group means (in-sample; large groups inflate it):")
    for k, v in out.items():
        print(f"  {k:<22} {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=200_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load(a.sample_n)
    print("sample:", df.shape)
    for f in (p01_sqft, p02_geo, p03_building_level, p04_within_between, p05_pdp):
        f(df)
    variance_decomposition(df)
    print(f"\nall plots in {OUT}")


if __name__ == "__main__":
    main()