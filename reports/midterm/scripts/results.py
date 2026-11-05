"""Compute every number the report quotes, from the master table.

The only script that reads data/master_listing.parquet. It writes small files
that the figures and the document consume, so a teammate without the 197 MB
table can still rebuild both:

    generated/numbers.tex              one LaTeX macro per number quoted in prose
    generated/tables/*.tex             booktabs tables
    generated/data/*.csv               the inputs to every figure

Models reproduce notebooks/phase_01_market_rent/06_model_training.ipynb exactly:
same 13 features, same building-level split (seed 710), same hyperparameters.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

REPORT = Path(__file__).resolve().parent.parent
REPO = REPORT.parent.parent
MASTER = REPO / "data" / "master_listing.parquet"
CATALOG = REPO / "data" / "master_columns.csv"
GEN = REPORT / "generated"

SYNTH_DB = REPO / "data" / "accolade.duckdb"
MANIFEST = REPO / "data" / "realistic-v2" / "meta" / "manifest.json"
DWELLSY = REPO / "data" / "dwellsy" / "listings_all.parquet"

TARGET = "market_price_observation__first_asking_rent"
TRUTH = "truth_floorplan_market_rent__true_market_rent_at_enter"
SEED = 710
FEATURES = [
    "market_unit__sqft", "market_unit__bed", "market_unit__bath", "market_unit__floor",
    "market_building__year_built", "market_building__number_units",
    "market_building__n_building_amenities", "market_building__n_unit_amenities",
    "market_building__avg_review_score", "market_building__application_fee",
    "market_building__lat", "market_building__lon", "market_floor_plan__unit_count",
]

numbers: dict[str, str] = {}
cache: dict = {}
POST_TABLES = {"market.price_observation", "market.concession_event"}
POST_COLS = {"market_unit__days_on_market", "market_unit__status", "market_unit__exit_market"}
MARKETS = {"chicago": "Chicago", "dfw": "Dallas\u2013Fort Worth", "dc": "Washington, DC"}


def count(n):
    return f"{int(n):,}".replace(",", "{,}")


def pct(x, digits=1):
    return f"{x:.{digits}f}\\%"


def mdape(pred, actual):
    return 100 * float(np.median(np.abs(pred - actual) / actual))


def dataset_facts():
    """The synthetic dataset's shape and the realism the generator built into it."""
    m = json.loads(MANIFEST.read_text())
    con = duckdb.connect()
    con.execute(f"ATTACH '{SYNTH_DB.as_posix()}' AS s (READ_ONLY)")
    q = lambda sql: con.execute(sql).fetchone()[0]
    b = m["building_counts"]
    numbers.update({
        "nSourceTables": count(len(m["tables"])),
        "nSourceRowsM": f"{sum(t['rows'] for t in m['tables'].values()) / 1e6:.1f}",
        "nBuildings": count(sum(b.values())),
        "nChicago": count(b["chicago"]), "nDfw": count(b["dfw"]), "nDc": count(b["dc"]),
        "nPms": count(sum(m["pms_counts"].values())),
        "asOf": m["as_of"], "historyStart": m["history_start"],
        "nLeaseUp": count(q("SELECT count(*) FROM s.truth.building_params WHERE is_lease_up_building")),
        "nRentControlled": count(q("SELECT count(*) FROM s.truth.building_params WHERE rent_controlled")),
        "rmShare": pct(100 * q("SELECT avg((pricing_policy = 'rm')::int) FROM s.truth.building_params"), 0),
        "nVacancySpikes": count(q("SELECT count(*) FROM s.truth.scenario_log WHERE toggle = 'vacancy_spikes'")),
        "nConcessionEvents": count(q("SELECT count(*) FROM s.market.concession_event")),
        "yearBuiltMasked": pct(100 * q("SELECT avg((year_built IS NULL)::int) FROM s.market.building"), 0),
        "storiesMasked": pct(100 * q("SELECT avg((number_stories IS NULL)::int) FROM s.market.building"), 0),
    })
    cat = pd.read_csv(CATALOG)
    rows = q(f"SELECT count(*) FROM read_parquet('{MASTER.as_posix()}')")
    numbers.update({
        "nMasterRows": count(rows),
        "listingsPerBuilding": count(round(rows / sum(b.values()))),
        "nMasterCols": count(len(cat)),
        "nMasterTables": count(cat.source_table.nunique()),
    })
    print(f"  dataset: {numbers['nSourceTables']} tables, {numbers['nSourceRowsM']}M rows, "
          f"{numbers['nBuildings']} buildings")


def dwellsy_facts():
    """One footnote's worth of the real extract. Falls back to the values recorded on
    2026-10-28 for teammates who do not have the extract locally."""
    if not DWELLSY.exists():
        print("  dwellsy: extract not found locally, using recorded values")
        numbers.update({"dwListings": count(975917), "dwUnits": count(263688),
                        "dwStart": "September 2020", "dwEnd": "October 2026"})
        return
    con = duckdb.connect()
    n, u, lo, hi = con.execute(f"""
        SELECT count(*), count(DISTINCT uru_id), min(creation_time::TIMESTAMP), max(creation_time::TIMESTAMP)
        FROM read_parquet('{DWELLSY.as_posix()}')""").fetchone()
    numbers.update({"dwListings": count(n), "dwUnits": count(u),
                    "dwStart": lo.strftime("%B %Y"), "dwEnd": hi.strftime("%B %Y")})
    print(f"  dwellsy: {n:,} listings, {u:,} units")


def leakage_correlations():
    """Correlation of every numeric column with asking rent, labelled by role."""
    cat = pd.read_csv(CATALOG)
    num = cat[(cat.type_group == "numeric") & (cat.role != "key") & (cat.column != TARGET)]
    con = duckdb.connect()
    sel = ", ".join(f'corr("{c}", "{TARGET}")' for c in num.column)
    r = con.execute(f"SELECT {sel} FROM read_parquet('{MASTER.as_posix()}')").fetchone()
    out = num[["column", "source_table", "role"]].assign(r=r)
    dropped = int(out.r.isna().sum())
    out = out.dropna(subset=["r"]).assign(abs_r=lambda d: d.r.abs())
    out.sort_values("abs_r", ascending=False).to_csv(GEN / "data" / "leakage_corr.csv", index=False)

    feat, leaky = out[out.role == "feature"], out[out.role == "leaky"]
    numbers.update({
        "nNumericCols": count(len(out)),
        "nNumericConstant": count(dropped),
        "nLeakyCols": count(len(leaky)),
        "minLeakyR": f"{leaky.abs_r.min():.3f}",
        "maxLeakyR": f"{leaky.abs_r.max():.4f}",
        "maxFeatureR": f"{feat.abs_r.max():.2f}",
        "maxFeatureName": feat.sort_values("abs_r").iloc[-1]["column"].split("__")[-1].replace("_", r"\_"),
    })
    # Per-square-foot prices look harmless by correlation (|r| ~0.56, beside the best
    # legitimate feature) yet times sq ft they reproduce asking rent exactly.
    psf = "market_price_observation__first_asking_rent_psf"
    n, exact, r_prod = con.execute(f"""
        SELECT count(*),
               count(*) FILTER (WHERE abs("{psf}" * market_unit__sqft - "{TARGET}") <= 0.01 * "{TARGET}"),
               corr("{psf}" * market_unit__sqft, "{TARGET}")
        FROM read_parquet('{MASTER.as_posix()}')
        WHERE "{psf}" IS NOT NULL AND market_unit__sqft IS NOT NULL""").fetchone()
    is_psf = leaky.column.str.endswith("_psf")
    numbers.update({
        "nLeakyLevel": count((~is_psf).sum()),
        "minLeakyLevelR": f"{leaky[~is_psf].abs_r.min():.3f}",
        "nLeakyPsf": count(is_psf.sum()),
        "psfAloneR": f"{leaky[leaky.column == psf].abs_r.iloc[0]:.2f}",
        "psfTimesSqftR": f"{r_prod:.6f}",
        "psfExactShare": pct(100 * exact / n, 0),
        "nPsfRows": count(n),
    })
    print(f"  leakage: {len(out)} numeric columns, {len(leaky)} leaky "
          f"(|r| {leaky.abs_r.min():.3f}-{leaky.abs_r.max():.4f}), "
          f"best feature |r| {feat.abs_r.max():.2f}")


def models():
    df = pd.read_parquet(MASTER, columns=FEATURES + [TARGET, TRUTH, "property_id", "market",
                                                     "market_building__property_class"])
    df = df[df[TARGET].notna()].reset_index(drop=True)
    X, y = df[FEATURES].astype(float), df[TARGET].astype(float)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
                  .split(X, y, groups=df.property_id))

    key = df.market.astype(str) + "|" + df.market_unit__bed.astype(str)
    baseline = key.iloc[te].map(y.iloc[tr].groupby(key.iloc[tr]).median()).fillna(y.iloc[tr].median())

    specs = {
        "Naive baseline (market \\(\\times\\) bedrooms median)": None,
        "Linear regression": Pipeline([("impute", SimpleImputer(strategy="median")),
                                        ("model", LinearRegression())]),
        "Decision tree": Pipeline([("impute", SimpleImputer(strategy="median")),
                                    ("model", DecisionTreeRegressor(max_depth=10, min_samples_leaf=20,
                                                                    random_state=SEED))]),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=31,
                                                           learning_rate=0.1, random_state=SEED),
    }
    ya = y.iloc[te].to_numpy()
    truth = df[TRUTH].iloc[te].to_numpy()
    has_truth = ~np.isnan(truth)

    rows, preds = [], {}
    for name, model in specs.items():
        p = baseline.to_numpy() if model is None else model.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        preds[name] = p
        rows.append(dict(model=name, mae=mean_absolute_error(ya, p),
                         rmse=float(np.sqrt(mean_squared_error(ya, p))), r2=r2_score(ya, p),
                         mdape_ask=mdape(p, ya), mdape_true=mdape(p[has_truth], truth[has_truth])))
        print(f"  {name[:22]:22} MAE ${rows[-1]['mae']:,.0f}  R2 {rows[-1]['r2']:.3f}  "
              f"MdAPE {rows[-1]['mdape_ask']:.1f}% / vs truth {rows[-1]['mdape_true']:.1f}%")
    m = pd.DataFrame(rows)
    m.to_csv(GEN / "data" / "model_metrics.csv", index=False)

    floor = mdape(ya[has_truth], truth[has_truth])
    pd.DataFrame({"actual": ya, "pred": preds["Gradient boosting"], "truth": truth,
                  "market": df.market.iloc[te].to_numpy(),
                  "bed": df.market_unit__bed.iloc[te].to_numpy()}
                 ).to_parquet(GEN / "data" / "test_predictions.parquet", index=False)

    cache.update(test=df.iloc[te].assign(pred=preds["Gradient boosting"]).reset_index(drop=True),
                 model=specs["Gradient boosting"], X_test=X.iloc[te], y_test=y.iloc[te])
    gbm, base = m.iloc[3], m.iloc[0]
    numbers.update({
        "nListings": count(len(df)),
        "nTrainListings": count(len(tr)), "nTestListings": count(len(te)),
        "nTestTruth": count(has_truth.sum()),
        "nTrainBuildings": count(df.property_id.iloc[tr].nunique()),
        "nTestBuildings": count(df.property_id.iloc[te].nunique()),
        "nFeatures": count(len(FEATURES)),
        "baseMdape": pct(base.mdape_ask), "baseMdapeTrue": pct(base.mdape_true),
        "linMdape": pct(m.iloc[1].mdape_ask), "treeMdape": pct(m.iloc[2].mdape_ask),
        "gbmMdape": pct(gbm.mdape_ask), "gbmMdapeTrue": pct(gbm.mdape_true),
        "gbmMae": f"\\${gbm.mae:,.0f}", "gbmRsq": f"{gbm.r2:.2f}",
        "baseMae": f"\\${base.mae:,.0f}",
        "linMae": f"\\${m.iloc[1].mae:,.0f}",
        "gbmVsLinear": pct(100 * (1 - gbm.mae / m.iloc[1].mae), 0),
        "errorReduction": pct(100 * (1 - gbm.mdape_ask / base.mdape_ask), 0),
        "noiseFloor": pct(floor),
    })
    write_model_table(m, floor)


def write_model_table(m, floor):
    best = m.mdape_ask.idxmin()
    lines = [
        r"\begin{tabular}{@{}l S[table-format=3.0] S[table-format=1.2] S[table-format=2.1] S[table-format=2.1]@{}}",
        r"\toprule",
        r" & {MAE} & {\(R^2\)} & \multicolumn{2}{c}{MdAPE (\%)} \\",
        r"\cmidrule(l){4-5}",
        r"Model & {(\$)} & & {vs.\ asking} & {vs.\ true} \\",
        r"\midrule",
    ]
    for i, r in m.iterrows():
        name = rf"\textbf{{{r.model}}}" if i == best else r.model
        lines.append(rf"{name} & {r.mae:.0f} & {r.r2:.2f} & {r.mdape_ask:.1f} & {r.mdape_true:.1f} \\")
    lines += [
        r"\midrule",
        rf"\textit{{The listing's own asking rent}} & {{---}} & {{---}} & {{---}} & {floor:.1f} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tables" / "model_comparison.tex").write_text("\n".join(lines) + "\n")
    print("  table   generated/tables/model_comparison.tex")


def eda():
    """Rent by market and bedrooms, and one point per building for the map."""
    df = pd.read_parquet(MASTER, columns=[TARGET, "market", "market_unit__bed", "property_id",
                                          "market_building__lat", "market_building__lon",
                                          "market_building__submarket"])
    df = df[df[TARGET].notna()].assign(beds=lambda d: d.market_unit__bed.clip(upper=4))
    g = df.groupby(["market", "beds"])[TARGET]
    q = g.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    q.columns = ["p10", "p25", "p50", "p75", "p90"]
    q.assign(n=g.size()).reset_index().to_csv(GEN / "data" / "rent_by_market_beds.csv", index=False)

    b = (df.groupby("property_id")
           .agg(market=("market", "first"), lat=("market_building__lat", "first"),
                lon=("market_building__lon", "first"), rent=(TARGET, "median"), n=(TARGET, "size"))
           .reset_index(drop=True).round(5))
    b.to_csv(GEN / "data" / "building_map.csv", index=False)
    (df.groupby(["market", "market_building__submarket"])
       .agg(lat=("market_building__lat", "mean"), lon=("market_building__lon", "mean"),
            rent=(TARGET, "median"), listings=(TARGET, "size"), buildings=("property_id", "nunique"))
       .reset_index().rename(columns={"market_building__submarket": "submarket"}).round(5)
       .to_csv(GEN / "data" / "submarket_map.csv", index=False))
    for k, med in df.groupby("market")[TARGET].median().items():
        numbers[f"medRent{k.capitalize()}"] = f"\\${med:,.0f}"
    print(f"  eda: {len(q)} market x bedroom cells, {len(b):,} buildings mapped")


def funnel():
    """Column counts from the master table to the model's inputs."""
    cat = pd.read_csv(CATALOG)
    feat = cat[cat.role == "feature"]
    known = feat[~(feat.source_table.isin(POST_TABLES) | feat.column.isin(POST_COLS))]
    num = pd.read_csv(REPO / "outputs" / "05_phase_01_numerical_final_features.csv")
    cg = pd.read_csv(REPO / "outputs" / "07_phase_01_categorical_final_features.csv", index_col=0)
    kept = int((num.decision == "keep").sum() + (cg.decision == "keep").sum())
    stages = pd.DataFrame([
        ("Columns in the master table", len(cat)),
        ("Tagged as features", len(feat)),
        ("Known when the unit is listed", len(known)),
        ("Scored so far", len(num) + len(cg)),
        ("Kept after scoring", kept),
        ("In the current model", len(FEATURES)),
    ], columns=["stage", "n"])
    stages.to_csv(GEN / "data" / "funnel.csv", index=False)
    numbers.update({"nNumCandidates": count(len(num)), "nCatCandidates": count(len(cg)),
                    "nFeatureCols": count(len(feat)), "nKnownCols": count(len(known)),
                    "nKnownBoolean": count((known.type_group == "boolean").sum()),
                    "nScored": count(len(num) + len(cg)), "nKept": count(kept),
                    "nKeptNumeric": count((num.decision == "keep").sum()),
                    "nKeptCategorical": count((cg.decision == "keep").sum())})
    print("  funnel: " + " -> ".join(str(n) for n in stages.n))


def errors():
    """Where the boosted model misses: by rent band, by segment, and the fit itself."""
    t = cache["test"].copy()
    t["ape"] = (t.pred - t[TARGET]).abs() / t[TARGET]
    t["resid"] = t[TARGET] - t.pred

    edges = [0, 1000, 1500, 2000, 2500, 3000, 4000, np.inf]
    labels = ["<1k", "1–1.5k", "1.5–2k", "2–2.5k", "2.5–3k", "3–4k", ">4k"]
    t["band"] = pd.cut(t[TARGET], edges, labels=labels)
    (t.groupby("band", observed=True)
      .agg(n=("resid", "size"), bias=("resid", "mean"), mdape=("ape", "median"))
      .reset_index().assign(mdape=lambda d: 100 * d.mdape)
      .to_csv(GEN / "data" / "bias_by_band.csv", index=False))

    con = duckdb.connect()
    con.execute(f"ATTACH '{SYNTH_DB.as_posix()}' AS s (READ_ONLY)")
    flags = con.execute("SELECT property_id, is_lease_up_building, rent_controlled "
                        "FROM s.truth.building_params").df()
    t = t.merge(flags, on="property_id", how="left")
    masked = t.market_building__year_built.isna() | t.market_building__number_units.isna()

    seg = []
    def add(group, label, mask):
        part = t[mask]
        if part.property_id.nunique() < 3:
            return
        seg.append(dict(group=group, label=label, n=len(part), buildings=part.property_id.nunique(),
                        mdape=100 * part.ape.median(), bias=part.resid.mean()))
    add("All", "All held-out listings", t.index == t.index)
    for k, v in MARKETS.items():
        add("Market", v, t.market == k)
    for b, v in [(0, "Studio"), (1, "1 bedroom"), (2, "2 bedrooms"), (3, "3 bedrooms")]:
        add("Bedrooms", v, t.market_unit__bed == b)
    add("Bedrooms", "4+ bedrooms", t.market_unit__bed >= 4)
    for c in ["A", "B", "C"]:
        add("Class", f"Class {c}", t.market_building__property_class == c)
    add("Building", "Stabilized", ~t.is_lease_up_building.astype(bool))
    add("Building", "Lease-up", t.is_lease_up_building.astype(bool))
    add("Building", "Year built or units masked", masked)
    add("Building", "Rent-controlled (DC)", t.rent_controlled.astype(bool))
    pd.DataFrame(seg).to_csv(GEN / "data" / "error_by_segment.csv", index=False)

    h, xe, ye = np.histogram2d(t[TARGET], t.pred, bins=64, range=[[400, 5200], [400, 5200]])
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    ii, jj = np.nonzero(h)
    pd.DataFrame({"actual": xc[ii], "pred": yc[jj], "count": h[ii, jj].astype(int)}).to_csv(
        GEN / "data" / "fit_hexgrid.csv", index=False)

    lo, hi = t[t.band == "<1k"].resid.mean(), t[t.band == ">4k"].resid.mean()
    sg = pd.DataFrame(seg).set_index("label")
    numbers.update({"biasLow": f"\\${abs(lo):,.0f}", "biasHigh": f"\\${hi:,.0f}",
                    "mdapeLeaseUp": pct(sg.loc["Lease-up", "mdape"]),
                    "mdapeStabilized": pct(sg.loc["Stabilized", "mdape"])})
    print(f"  errors: {len(seg)} segments, bias {lo:+.0f} (<1k) to {hi:+.0f} (>4k)")


def importance():
    """Held-out permutation importance: dollars of MAE lost when a feature is shuffled."""
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(cache["y_test"]), 20_000, replace=False)
    r = permutation_importance(cache["model"], cache["X_test"].iloc[idx], cache["y_test"].iloc[idx],
                               scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED)
    out = (pd.DataFrame({"feature": FEATURES, "dmae": r.importances_mean, "sd": r.importances_std})
             .sort_values("dmae", ascending=False))
    out.to_csv(GEN / "data" / "gbm_importance.csv", index=False)
    top = out.iloc[0]
    numbers.update({"topFeature": top.feature.split("__")[-1].replace("_", r"\_"),
                    "topFeatureDmae": f"\\${top.dmae:,.0f}"})
    print(f"  importance: top {top.feature} (+${top.dmae:,.0f} MAE)")


def borrowed():
    """Results from earlier notebooks the report cites, copied so figures stay self-contained."""
    cg = pd.read_csv(REPO / "outputs" / "07_phase_01_categorical_final_features.csv", index_col=0)
    cg.rename_axis("feature").reset_index().to_csv(GEN / "data" / "categorical_scores.csv", index=False)
    cs = pd.read_csv(REPO / "outputs" / "01_phase_02_concession_extraction_summary.csv")
    cs.to_csv(GEN / "data" / "concession_summary.csv", index=False)
    o = cs[cs.slice_by == "overall"].iloc[0]
    src = cs[cs.slice_by == "text_source"].set_index("slice")
    numbers.update({"concAmountAcc": pct(o.amount_acc, 0), "concWithinFive": pct(o.total_within_5pct, 0),
                    "concLlm": pct(src.loc["llm", "amount_acc"], 0),
                    "concGrammar": pct(src.loc["grammar", "amount_acc"], 0),
                    "concN": count(o.n),
                    "concMonthsFree": pct(cs[(cs.slice_by == "type") & (cs.slice == "upfront_free_months")]
                                          .amount_acc.iloc[0], 0)})
    print("  borrowed: categorical scores, concession summary")


def eda_stats():
    """Statistics the EDA prose quotes, recomputed from the master table."""
    con = duckdb.connect()
    m = f"read_parquet('{MASTER.as_posix()}')"
    q = lambda sql: con.execute(sql).fetchone()

    def spearman(col):
        return q(f"""WITH d AS (SELECT "{col}" x, "{TARGET}" y FROM {m}
                                WHERE "{col}" IS NOT NULL AND "{TARGET}" IS NOT NULL)
                     SELECT corr(rx, ry) FROM (SELECT rank() OVER (ORDER BY x) rx,
                                                      rank() OVER (ORDER BY y) ry FROM d)""")[0]

    overall = q(f'SELECT corr(market_unit__sqft, "{TARGET}") FROM {m}')[0]
    within = q(f"""WITH d AS (SELECT market_unit__sqft - avg(market_unit__sqft) OVER (PARTITION BY property_id) x,
                                     "{TARGET}" - avg("{TARGET}") OVER (PARTITION BY property_id) y
                              FROM {m} WHERE market_unit__sqft IS NOT NULL AND "{TARGET}" IS NOT NULL)
                   SELECT corr(x, y) FROM d""")[0]
    yearly = con.execute(f"""SELECT market, year(market_unit__enter_market) yr, median("{TARGET}") r
                             FROM {m} GROUP BY 1, 2 ORDER BY 1, 2""").df()
    piv = yearly.pivot(index="yr", columns="market", values="r")
    first, last = piv.index.min(), piv.index.max()
    growth = lambda k: 100 * (piv.loc[last, k] / piv.loc[first, k] - 1)
    dfw_peak = piv.dfw.idxmax()

    sparse = []
    for c in ["parking_surface_lot", "parking_covered", "parking_garage", "amenity_fee",
              "admin_fee", "application_fee", "storage_fee"]:
        col = f"market_building__{c}"
        miss = q(f'SELECT avg(("{col}" IS NULL)::int) FROM {m}')[0]
        rho = spearman(col)
        if miss >= 0.6 and rho >= 0.4:
            sparse.append((rho, miss))
    rhos, misses = [r for r, _ in sparse], [x for _, x in sparse]
    numbers.update({
        "sqftOverallR": f"{overall:.2f}", "sqftWithinR": f"{within:.2f}",
        "sqftSpearman": f"{spearman('market_unit__sqft'):.2f}",
        "firstYear": str(first), "lastYear": str(last),
        "growthChicago": pct(growth("chicago"), 0), "growthDc": pct(growth("dc"), 0),
        "dfwPeakYear": str(dfw_peak),
        "nSparseFees": count(len(sparse)),
        "sparseRhoLo": f"{min(rhos):.2f}", "sparseRhoHi": f"{max(rhos):.2f}",
        "sparseMissLo": pct(100 * min(misses), 0), "sparseMissHi": pct(100 * max(misses), 0),
    })
    print(f"  eda stats: sqft r {overall:.2f} overall, {within:.2f} within building; "
          f"{len(sparse)} sparse fee columns")


def write_numbers():
    body = ["% Generated by scripts/results.py. Do not edit: rerun `make results`.", ""]
    body += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(numbers.items())]
    (GEN / "numbers.tex").write_text("\n".join(body) + "\n")
    print(f"  numbers generated/numbers.tex ({len(numbers)} macros)")


if __name__ == "__main__":
    for d in ("data", "tables", "figures"):
        (GEN / d).mkdir(parents=True, exist_ok=True)
    print("results.py")
    dataset_facts()
    dwellsy_facts()
    leakage_correlations()
    models()
    eda()
    eda_stats()
    funnel()
    errors()
    importance()
    borrowed()
    write_numbers()
