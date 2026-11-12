"""Statistics shared by the feature-selection notebooks and scripts.

gini comes from the numerical selection notebooks; eta_squared and cramers_v from
pipeline/synthetic/analyze_categorical.py. All three are copied unchanged.
"""
import numpy as np
import pandas as pd


def gini(x, y):
    """Normalized Gini of ordering y by x.

    Sort by x from high to low, take the Gini of the Lorenz curve of cumulative y, and divide by
    the Gini of the perfect ordering (y by itself). Tied x values are taken as one block, so a
    feature with few distinct values (bed, bath) is not scored on an arbitrary tie order.
    """
    def lorenz_gini(key):
        g = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].agg(["size", "sum"]).iloc[::-1]
        px = np.r_[0, g["size"].cumsum() / len(y)]
        py = np.r_[0, g["sum"].cumsum() / y.sum()]
        return 2 * ((py[1:] + py[:-1]) / 2 * np.diff(px)).sum() - 1
    return lorenz_gini(x) / lorenz_gini(y)


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
