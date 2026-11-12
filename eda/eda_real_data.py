"""
For each market it prints data-quality checks and writes five plots:
  01 rent over time (quarterly median + IQR, with listing counts)
  02 geography: median rent over lat/lon
  03 rent vs sqft: raw vs within-community
  04 same-unit rent change between consecutive listings
  05 partial dependence from a compact GBM (holdout = unseen communities)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
try:
    from load_real_data import load_real  # loader sitting next to this script
except ImportError:
    from preprocessing.load_real_data import load_real  # noqa: E402

OUT = REPO / "data" / "eda_real"
RNG = 0
NICE = {"square_feet": "Square Feet", "listing_t": "Listing Date (Year)",
        "n_amenities": "Number of Amenities", "latitude": "Latitude"}
START, END = "2021-01-01", "2026-10-01"  # skip the thin 2020 start and the partial current quarter


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print("  saved", out.name + "/" + name if out != OUT else name)


def checks(df, mkt):
    print(f"\n=== {mkt} ===")
    print(f"rows {len(df):,} | units {df.uru_id.nunique():,} | communities {df.community_id.nunique():,}")
    lpu = df.groupby("uru_id").size()
    print("listings per unit:", lpu.describe(percentiles=[.5, .9, .99]).round(1).to_dict())
    print("units with >20 listings:", int((lpu > 20).sum()), "| >50:", int((lpu > 50).sum()))
    print("rent < 500:", int((df.rent_amount < 500).sum()), "| rent > 10,000:", int((df.rent_amount > 10000).sum()))
    t = df.groupby(df.creation_time.dt.year).agg(
        listings=("listing_id", "size"), communities=("community_id", "nunique"),
        units=("uru_id", "nunique"), median_rent=("rent_amount", "median"))
    t["listings_per_community"] = (t.listings / t.communities).round(1)
    print("By listing year:")
    print(t.to_string())


def quarterly(d):
    d = d[(d.creation_time >= START) & (d.creation_time < END)]
    return d.groupby(d.creation_time.dt.to_period("Q")).rent_amount.agg(
        med="median", lo=lambda s: s.quantile(.25), hi=lambda s: s.quantile(.75), n="size")


def p01_time(df, out, mkt):
    q = quarterly(df)
    x = q.index.to_timestamp()
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    a.fill_between(x, q.lo, q.hi, alpha=.25, label="25-75th Percentile")
    a.plot(x, q.med, "o-", lw=2, label="Median")
    a.set(ylabel="Asking Rent ($)", title=f"Asking Rent Over Time, By Listing Quarter ({mkt})"); a.legend()
    b.bar(x, q.n, width=70); b.set(ylabel="Listings")
    save(fig, out, "01_rent_over_time.png")


def p02_geo(df, out, mkt):
    d = df[df.rent_amount.between(*df.rent_amount.quantile([.01, .99]))]
    lo, hi = d.rent_amount.quantile([.05, .95])
    fig, ax = plt.subplots(figsize=(7.5, 7))
    hb = ax.hexbin(d.longitude, d.latitude, C=d.rent_amount, reduce_C_function=np.median,
                   gridsize=70, mincnt=20, cmap="magma", vmin=lo, vmax=hi)
    ax.set_aspect(1 / np.cos(np.radians(d.latitude.median())))
    ax.set(xlabel="Longitude", ylabel="Latitude", title=f"Median Asking Rent, Hexes With 20+ Listings ({mkt})")
    fig.colorbar(hb, ax=ax, label="Median Rent ($)")
    save(fig, out, "02_geography.png")


def p03_sqft(df, out, mkt):
    d = df.dropna(subset=["square_feet"])
    d = d[d.square_feet.between(250, 3500) & d.rent_amount.between(*df.rent_amount.quantile([.005, .995]))].copy()
    d["rent_dm"] = d.rent_amount - d.groupby("community_id").rent_amount.transform("mean")
    d["sqft_dm"] = d.square_feet - d.groupby("community_id").square_feet.transform("mean")
    r_raw, r_in = d.square_feet.corr(d.rent_amount), d.sqft_dm.corr(d.rent_dm)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].hexbin(d.square_feet, d.rent_amount, gridsize=60, bins="log", mincnt=1, cmap="viridis")
    ax[0].set(xlabel="Sqft", ylabel="Rent ($)", title=f"Raw: Pearson={r_raw:.2f} ({mkt})")
    ax[1].hexbin(d.sqft_dm, d.rent_dm, gridsize=60, bins="log", mincnt=1, cmap="viridis")
    ax[1].set(xlabel="Sqft - Community Mean", ylabel="Rent - Community Mean ($)",
              title=f"Within Community: Pearson={r_in:.2f} ({mkt})")
    print(f"  sqft vs rent: raw r={r_raw:.2f}, within-community r={r_in:.2f}")
    save(fig, out, "03_within_vs_between_sqft.png")
    return r_raw, r_in


def p04_same_unit(df, out, mkt):
    d = df.sort_values(["uru_id", "creation_time"])[["uru_id", "creation_time", "rent_amount"]].copy()
    g = d.groupby("uru_id")
    d["gap_yrs"] = (d.creation_time - g.creation_time.shift()).dt.days / 365.25
    d["dlog"] = np.log(d.rent_amount / g.rent_amount.shift())
    d = d.dropna(subset=["gap_yrs", "dlog"])
    d = d[(d.gap_yrs >= 30 / 365.25) & (d.gap_yrs <= 4)]
    d["pct"] = (np.exp(d.dlog) - 1) * 100
    clip = d[d.pct.between(-40, 60)]
    bins = pd.cut(clip.gap_yrs, np.arange(0, 4.01, .25))
    m = clip.groupby(bins, observed=True).agg(x=("gap_yrs", "median"), y=("pct", "median"))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    hb = ax.hexbin(clip.gap_yrs, clip.pct, gridsize=55, bins="log", mincnt=1, cmap="viridis")
    ax.plot(m.x, m.y, "r-", lw=2, label="Binned Median"); ax.axhline(0, color="k", lw=.6)
    ax.set(xlabel="Years Between Consecutive Listings of the Same Unit", ylabel="Rent Change (%)",
           title=f"Same-Unit Rent Change Between Relistings ({mkt})"); ax.legend()
    fig.colorbar(hb, ax=ax, label="Log10 Count")
    print(f"  same-unit pairs: {len(d):,} | median change {d.pct.median():.1f}% | median gap {d.gap_yrs.median():.2f} yrs")
    save(fig, out, "04_same_unit_change.png")


def p05_pdp(df, out, mkt):
    d = df.copy()
    d["listing_t"] = d.creation_time.dt.year + (d.creation_time.dt.dayofyear - 1) / 365.25
    d = d[d.rent_amount.between(*df.rent_amount.quantile([.001, .999]))]
    d = d.sample(min(200_000, len(d)), random_state=RNG)
    feats = ["square_feet", "bedrooms", "full_baths", "latitude", "longitude", "year_built",
             "n_amenities", "top_down_community_count", "listing_t"]
    X, y, grp = d[feats].astype("float64"), d.rent_amount, d.community_id
    tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=RNG).split(X, y, grp))
    gbm = HistGradientBoostingRegressor(max_iter=200, learning_rate=.1, random_state=RNG).fit(X.iloc[tr], y.iloc[tr])
    r2 = r2_score(y.iloc[te], gbm.predict(X.iloc[te]))
    mae = mean_absolute_error(y.iloc[te], gbm.predict(X.iloc[te]))
    print(f"  compact GBM holdout (unseen communities): R2={r2:.2f}, MAE=${mae:,.0f}")
    Xs = X.iloc[te].sample(min(4000, len(te)), random_state=RNG)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for a, f in zip(ax.ravel(), ["square_feet", "listing_t", "n_amenities", "latitude"]):
        r = partial_dependence(gbm, Xs, [f], grid_resolution=30, kind="average")
        gv = r["grid_values"][0] if "grid_values" in r else r["values"][0]
        a.plot(gv, r["average"][0] - r["average"][0].mean(), "r-")
        a.plot(Xs[f].quantile(np.linspace(.1, .9, 9)), [a.get_ylim()[0]] * 9, "k|", ms=10)
        a.set(xlabel=NICE[f], ylabel="Change in Predicted Rent ($)")
    fig.suptitle(f"Partial Dependence, Compact GBM, Holdout = Unseen Communities ({mkt})")
    save(fig, out, "05_partial_dependence.png")
    return r2, mae


def comparison(df, markets):
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    for m in markets:
        q = quarterly(df[df.market == m])
        a.plot(q.index.to_timestamp(), q.med, "o-", label=m)
        b.plot(q.index.to_timestamp(), q.n, "o-", label=m)
    a.set(ylabel="Median Asking Rent ($)", title="Median Asking Rent By Market and Listing Quarter"); a.legend()
    b.set(ylabel="Listings")
    save(fig, OUT, "00_market_comparison.png")


def main():
    df = load_real()
    markets = [m for m in df.market.cat.categories if (df.market == m).any()]
    rows = []
    for m in markets:
        d = df[df.market == m]
        checks(d, m)
        out = OUT / m.replace(" ", "_")
        p01_time(d, out, m)
        p02_geo(d, out, m)
        r_raw, r_in = p03_sqft(d, out, m)
        p04_same_unit(d, out, m)
        r2, mae = p05_pdp(d, out, m)
        rows.append({"market": m, "listings": len(d), "units": d.uru_id.nunique(),
                     "communities": d.community_id.nunique(), "median_rent": d.rent_amount.median(),
                     "median_sqft": d.square_feet.median(), "r_raw": round(r_raw, 2),
                     "r_within": round(r_in, 2), "gbm_r2": round(r2, 2), "gbm_mae": round(mae)})
    print("\n=== Market Comparison ===")
    print(pd.DataFrame(rows).to_string(index=False))
    if len(markets) > 1:
        comparison(df, markets)
    print("\nall plots in", OUT)


if __name__ == "__main__":
    main()