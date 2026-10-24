"""Rank categorical (string) features for market rent prediction.

The string counterpart of analyze_numeric.py. Keeps master-table columns with
type_group == "string", separates identifiers from real categoricals, and scores
each survivor against asking rent with:
    --eta_sq: share of rent variance explained by the category (one-way ANOVA)
    --eta_sq_null / eta_sq_excess: the same statistic with rent shuffled, and the
      gap. eta_sq rises mechanically with the number of levels, so a 98-level zip
      code scores well on noise alone; the excess over its own shuffled baseline
      is what actually carries signal.
    --mi_coef: mutual information, as a nonlinear check
    --perm_dMAE: permutation importance in a gradient-boosted model, with levels
      target-encoded through sklearn's internal cross-fitting
    --within_eta_sq: variance explained within a single property, which is 0 for
      the building-level columns that are constant across a building's listings
    --top_level_share: concentration, to catch near-constant columns

Two rules this script exists to enforce, both learned the hard way:

1. Cardinality is judged per entity, not per row. market_building__street_address
   has 2,999 distinct values over 3,000 buildings: a building identifier, even
   though it is only 2.5% distinct across listings because each building repeats
   over ~320 of them. Judged per row it survives as a feature.
2. Decisions follow perm_dMAE, measured on held-out properties, not eta_sq,
   measured in sample. eta_sq ranks those building identifiers top (street_address
   0.73) while their held-out importance is nil (0.14, and building_name is
   negative): target-encoding a building label teaches you nothing about a
   building you have not seen. Keeping eta_sq as the decision metric selects
   identifiers and discards property_class, zip_code and submarket as redundant
   with them.

Redundancy is reported as nesting where one column is a strict refinement of
another (zip_code within submarket within city), since Cramer's V between a
2,999-level column and a 3-level one is ~0.99 by construction and says nothing.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "data" / "master_listing.parquet"
CATALOG = REPO / "data" / "master_columns.csv"

# Phase 1 predicts what comparable units are asking; the truth column is the
# answer key the team validates against (docs/project_phases.md section 2).
TARGET = "market_price_observation__first_asking_rent"
ANSWER_KEY = "truth_floorplan_market_rent__true_market_rent_at_enter"

# Known only after the listing plays out, so not Phase 1 inputs
# (docs/project_phases.md section 3 rule 1). Mirrors analyze_numeric.py.
POST_TABLES = {"market.price_observation", "market.concession_event"}
POST_COLS = {"market_unit__days_on_market", "market_unit__status"}

# A column whose distinct values outnumber this share of the entities it varies
# over is an identifier, not a category. The denominator is buildings for a
# column constant within a building, rows otherwise.
ID_CARDINALITY = 0.20
MAX_LEVELS = 5_000
SEED = 42


def entity_level(frame: pd.DataFrame, col: str) -> tuple[bool, int]:
    """Is col constant within a building, and how many entities does it span?

    Returns (building_level, n_entities). n_entities counts only entities with a
    non-null value, so a column present for 150 of 3,000 buildings is measured
    against those 150.
    """
    present = frame[frame[col].notna()]
    if present.empty:
        return True, 0
    per_prop = present.groupby("property_id")[col].nunique(dropna=True)
    building_level = bool(per_prop.max() <= 1)
    return building_level, int(present.property_id.nunique() if building_level else len(present))


def classify(cat: pd.DataFrame, frame: pd.DataFrame, n: int,
             include_truth: bool = False) -> pd.DataFrame:
    """Build the funnel: every string column, with why it is in or out."""
    rows = []
    for c in cat.column:
        s = frame[c]
        nn = int(s.notna().sum())
        levels = int(s.nunique(dropna=True))
        building_level, n_ent = entity_level(frame, c)
        ratio = levels / n_ent if n_ent else 0.0
        top = float(s.value_counts(normalize=True).iloc[0]) if nn else 0.0
        meta = cat.loc[cat.column == c].iloc[0]

        if meta.role == "key":
            status, why = "excluded", "join key"
        elif meta.source_table in POST_TABLES or c in POST_COLS:
            status, why = "excluded", "known only after the listing closes"
        elif levels <= 1:
            status, why = "excluded", "constant"
        elif ratio > ID_CARDINALITY or levels > MAX_LEVELS:
            unit = "buildings" if building_level else "rows"
            status, why = "excluded", f"identifier ({ratio:.0%} of {n_ent:,} {unit} distinct)"
        elif meta.role in {"latent", "label"} and not include_truth:
            status, why = "excluded", f"role={meta.role}, not known at prediction time"
        else:
            status, why = "candidate", ""
        rows.append(dict(
            variable=c, source_table=meta.source_table, role=meta.role,
            completeness_pct=round(100 * nn / n, 2), n_levels=levels,
            grain="building" if building_level else "listing", n_entities=n_ent,
            cardinality_ratio=round(ratio, 5), top_level_share=round(top, 4),
            x_status=status, reason=why,
        ))
    return pd.DataFrame(rows).sort_values(["x_status", "n_levels"], ascending=[True, False])


def eta_squared(codes: pd.Series, y: pd.Series) -> float:
    """Share of y's variance explained by group membership (one-way ANOVA)."""
    ok = codes.notna() & y.notna()
    if ok.sum() < 2:
        return np.nan
    c, v = codes[ok], y[ok]
    grand = v.mean()
    g = v.groupby(c)
    between = (g.count() * (g.mean() - grand) ** 2).sum()
    total = ((v - grand) ** 2).sum()
    return float(between / total) if total > 0 else np.nan


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categoricals, bias-corrected (Bergsma 2013)."""
    tab = pd.crosstab(a, b)
    if tab.size == 0 or min(tab.shape) < 2:
        return np.nan
    n = tab.to_numpy().sum()
    chi2 = ((tab - np.outer(tab.sum(1), tab.sum(0)) / n) ** 2
            / (np.outer(tab.sum(1), tab.sum(0)) / n)).to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def score(frame: pd.DataFrame, candidates: list[str], y: pd.Series,
          groups: pd.Series, shuffles: int, mi_n: int) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    codes = {c: frame[c].astype("category").cat.codes.replace(-1, np.nan) for c in candidates}

    res = pd.DataFrame(index=candidates)
    res["eta_sq"] = [eta_squared(codes[c], y) for c in candidates]

    # Null baseline: the same column against shuffled rent. A high-cardinality
    # column partitions the data finely enough to explain noise, so this is the
    # score it would get for free.
    null = []
    for c in candidates:
        vals = [eta_squared(codes[c], pd.Series(rng.permutation(y.values), index=y.index))
                for _ in range(shuffles)]
        null.append(float(np.mean(vals)))
    res["eta_sq_null"] = null
    res["eta_sq_excess"] = res.eta_sq - res.eta_sq_null

    idx = rng.choice(len(frame), min(mi_n, len(frame)), replace=False)
    Xm = pd.DataFrame({c: codes[c].iloc[idx].fillna(-1) for c in candidates})
    mi = mutual_info_regression(Xm.values, y.iloc[idx].values,
                                discrete_features=True, random_state=SEED)
    res["mi_coef"] = np.sqrt(1 - np.exp(-2 * mi))

    res["within_eta_sq"] = [
        eta_squared(codes[c], y - y.groupby(groups.values).transform("mean")) for c in candidates
    ]
    return res


def model_importance(frame, candidates, y, groups, reps=3):
    """Permutation importance with levels target-encoded (internally cross-fitted)."""
    X = frame[candidates].astype("string").fillna("<missing>")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                    random_state=SEED).split(X, y, groups))
    enc = TargetEncoder(target_type="continuous", random_state=SEED)
    Xtr = enc.fit_transform(X.iloc[tr], y.iloc[tr])
    Xte = enc.transform(X.iloc[te])

    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                       max_leaf_nodes=63, random_state=SEED)
    gb.fit(Xtr, y.iloc[tr])
    pred = gb.predict(Xte)
    print(f"GBM holdout (unseen properties): R2={r2_score(y.iloc[te], pred):.3f}  "
          f"MAE=${mean_absolute_error(y.iloc[te], pred):,.0f}")

    rng = np.random.RandomState(SEED)
    sub = rng.choice(len(Xte), min(20_000, len(Xte)), replace=False)
    Xs, ys = pd.DataFrame(Xte).iloc[sub].reset_index(drop=True), y.iloc[te].values[sub]
    base = mean_absolute_error(ys, gb.predict(Xs))
    out = {}
    for i, c in enumerate(candidates):
        d = []
        for _ in range(reps):
            Xp = Xs.copy()
            Xp.iloc[:, i] = rng.permutation(Xs.iloc[:, i].values)
            d.append(mean_absolute_error(ys, gb.predict(Xp)) - base)
        out[c] = float(np.mean(d))
    return pd.Series(out, name="perm_dMAE")


def nests_in(frame: pd.DataFrame, a: str, b: str) -> bool:
    """True when every level of a maps to exactly one level of b (a refines b)."""
    g = frame.groupby(a, observed=True)[b].nunique(dropna=True)
    return bool((g <= 1).all()) and g.size > 0


def redundancy(frame, candidates, thresh):
    """Pairwise association, labelled as nesting where one column refines another."""
    pairs = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            sa, sb = frame[a].astype("string"), frame[b].astype("string")
            v = cramers_v(sa, sb)
            if pd.isna(v) or v <= thresh:
                continue
            a_in_b, b_in_a = nests_in(frame, a, b), nests_in(frame, b, a)
            rel = ("equivalent" if a_in_b and b_in_a
                   else f"{a} refines {b}" if a_in_b
                   else f"{b} refines {a}" if b_in_a else "overlapping")
            pairs.append(dict(a=a, b=b, cramers_v=round(v, 4), relation=rel))
    cols = ["a", "b", "cramers_v", "relation"]
    return (pd.DataFrame(pairs).sort_values("cramers_v", ascending=False)
            if pairs else pd.DataFrame(columns=cols))


def decide(funnel, res, pairs, min_completeness, min_gain):
    """Keep a column only if it still pays on buildings the model never saw."""
    d = res.copy()
    f = funnel.set_index("variable")
    d["completeness_pct"] = f.completeness_pct.reindex(d.index)
    d["grain"] = f.grain.reindex(d.index)
    d["decision"], d["reason"], d["redundant_with"] = "keep", "", ""

    low = d.completeness_pct < min_completeness
    d.loc[low, ["decision", "reason"]] = ["drop", f"completeness < {min_completeness}%"]

    weak = (d.perm_dMAE <= min_gain) & (d.decision == "keep")
    d.loc[weak, ["decision", "reason"]] = [
        "drop", f"no held-out gain (perm_dMAE <= {min_gain})"]

    # Among survivors that nest or duplicate, keep the one that generalizes.
    for _, r in pairs.iterrows():
        if not {r.a, r.b} <= set(d.index):
            continue
        if not (d.loc[[r.a, r.b], "decision"] == "keep").all():
            continue
        loser = r.a if d.loc[r.a, "perm_dMAE"] < d.loc[r.b, "perm_dMAE"] else r.b
        keeper = r.b if loser == r.a else r.a
        d.loc[loser, ["decision", "reason", "redundant_with"]] = [
            "drop", f"{r.relation} (Cramer's V {r.cramers_v}); kept the stronger", keeper]
    return d.sort_values("perm_dMAE", ascending=False)


def load(sample_n: int, target: str):
    cat = pd.read_csv(CATALOG)
    strings = cat[cat.type_group == "string"]
    cols = strings.column.tolist()
    need = list(dict.fromkeys(cols + ["property_id", target]))
    frame = pd.read_parquet(PARQUET, columns=need)
    frame = frame[frame[target].notna()]
    if sample_n and sample_n < len(frame):
        frame = frame.sample(sample_n, random_state=SEED)
    return strings, frame.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET, help=f"default {TARGET}")
    ap.add_argument("--sample-n", type=int, default=200_000)
    ap.add_argument("--mi-n", type=int, default=50_000)
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--min-completeness", type=float, default=50.0)
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="minimum held-out perm_dMAE in dollars of MAE")
    ap.add_argument("--include-truth", action="store_true",
                    help="also score latent/label string columns, for validation")
    ap.add_argument("--redundant-v", type=float, default=0.9)
    ap.add_argument("--out-dir", default=str(REPO / "outputs"))
    ap.add_argument("--prefix", default="07_phase_01")
    a = ap.parse_args()

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60)
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    strings, frame = load(a.sample_n, a.target)
    n = len(frame)
    print(f"{n:,} rows with a target | {len(strings)} string columns\n")

    funnel = classify(strings, frame, n, a.include_truth)
    funnel.to_csv(outdir / f"{a.prefix}_feature_funnel_categorical.csv", index=False)
    print(funnel.to_string(index=False))

    candidates = funnel.loc[funnel.x_status == "candidate", "variable"].tolist()
    print(f"\n{len(candidates)} candidates: {candidates}\n")

    y = frame[a.target].astype(float)
    res = score(frame, candidates, y, frame.property_id, a.shuffles, a.mi_n)
    res["perm_dMAE"] = model_importance(frame, candidates, y, frame.property_id)
    res = res.sort_values("perm_dMAE", ascending=False)
    res.round(4).to_csv(outdir / f"{a.prefix}_feature_selection_categorical.csv")
    print("\nScores, sorted by held-out permutation importance")
    print(res.round(4).to_string())

    pairs = redundancy(frame, candidates, a.redundant_v)
    pairs.to_csv(outdir / f"{a.prefix}_cramers_v_pairs.csv", index=False)
    print(f"\nRedundant pairs (Cramer's V > {a.redundant_v}): {len(pairs)}")
    if len(pairs):
        print(pairs.to_string(index=False))

    final = decide(funnel, res, pairs, a.min_completeness, a.min_gain)
    final.round(4).to_csv(outdir / f"{a.prefix}_categorical_final_features.csv")
    print("\nDecisions")
    print(final[["grain", "completeness_pct", "eta_sq", "eta_sq_excess", "mi_coef",
                 "perm_dMAE", "within_eta_sq", "decision", "reason"]].round(4).to_string())
    print(f"\nkept {(final.decision == 'keep').sum()} of {len(final)} candidates")
    print(f"wrote 4 CSVs to {outdir}")


if __name__ == "__main__":
    main()
