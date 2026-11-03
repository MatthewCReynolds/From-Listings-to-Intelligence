"""Render every report figure from the small files results.py writes.

Reads only generated/data/, never the master table, so anyone with the repo can
rebuild the figures. Add a figure by writing a function and listing it in FIGURES.

    python scripts/figures.py            # all figures
    python scripts/figures.py leakage    # one
"""

import sys

import numpy as np
import pandas as pd

import style as s


def leakage():
    """Every numeric column's correlation with asking rent, grouped by role."""
    d = pd.read_csv(s.DATA / "leakage_corr.csv")
    rows = [
        ("Features known at listing", d[d.role == "feature"], s.ACCENT),
        ("Simulator truth (answer keys)", d[d.role.isin(["label", "latent", "inferred"])], s.MUTED),
        ("The listing's own price", d[d.role == "leaky"], s.BRICK),
    ]
    fig, ax = s.figure(s.FULL, 2.05)
    rng = np.random.default_rng(7)

    for y, (_, part, color) in enumerate(rows):
        jitter = rng.uniform(-0.2, 0.2, len(part))
        psf = part.column.str.endswith("_psf").to_numpy()
        ax.scatter(part.abs_r[~psf], y + jitter[~psf], s=15, color=color,
                   edgecolor="white", linewidth=0.4, zorder=3)
        if psf.any():
            ax.scatter(part.abs_r[psf], y + jitter[psf], s=15, facecolor="white",
                       edgecolor=color, linewidth=1.0, zorder=3)

    feat, own = rows[0][1], rows[2][1]
    best = feat.loc[feat.abs_r.idxmax()]
    psf = own[own.column.str.endswith("_psf")]
    level = own[~own.column.str.endswith("_psf")]

    # The band where correlation cannot tell a usable feature from a leak.
    ax.axvspan(best.abs_r - 0.012, psf.abs_r.max() + 0.012, color=s.WASH, zorder=0)

    note = dict(fontsize=7, color=s.INK2, va="center")
    ax.annotate(rf"best feature: amenity count, $|r| = {best.abs_r:.2f}$",
                xy=(best.abs_r, 0), xytext=(best.abs_r - 0.035, -0.5), ha="right",
                arrowprops=dict(arrowstyle="-", color=s.RULE, lw=0.6), **note)
    ax.annotate(rf"{len(psf)} prices per sq ft, $|r| \approx {psf.abs_r.mean():.2f}$;"
                r" $\times$ sq ft they equal rent exactly",
                xy=(psf.abs_r.mean(), 2), xytext=(psf.abs_r.mean() - 0.035, 2.5), ha="right",
                arrowprops=dict(arrowstyle="-", color=s.RULE, lw=0.6), **note)
    ax.text(level.abs_r.min() - 0.02, 2.5, rf"{len(level)} rent columns, $|r| \geq {level.abs_r.min():.2f}$",
            ha="right", **note)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{name} ({len(part)})" for name, part, _ in rows])
    ax.set_ylim(-0.75, 2.75)
    ax.set_xlim(0, 1.01)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.xaxis.grid(True, zorder=0)
    ax.set_xlabel(r"Correlation with asking rent, $|r|$")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    s.save(fig, "leakage")


FIGURES = {"leakage": leakage}

# ---------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

MARKETS = [("chicago", "Chicago"), ("dfw", "Dallas–Fort Worth"), ("dc", "Washington, DC")]
BLUES = LinearSegmentedColormap.from_list("blues", ["#CFE0F3", "#86B6EF", "#2A78D6", "#184F95", "#0D366B"])
NAMES = {
    "market_unit__sqft": "Square feet", "market_building__lat": "Latitude",
    "market_building__lon": "Longitude", "market_building__n_building_amenities": "Building amenities",
    "market_building__year_built": "Year built", "market_unit__bed": "Bedrooms",
    "market_building__avg_review_score": "Review score", "market_building__n_unit_amenities": "Unit amenities",
    "market_building__number_units": "Units in building", "market_unit__floor": "Floor",
    "market_building__application_fee": "Application fee", "market_unit__bath": "Bathrooms",
    "market_floor_plan__unit_count": "Units on floor plan",
    "market_building__property_class": "Property class", "market_building__zip_code": "ZIP code",
    "market_building__submarket": "Submarket", "market_unit__floorplan_name": "Floor plan name",
    "market_unit__unit_name": "Unit name", "market_building__city": "City",
    "market_building__number_stories_prediction": "Stories (bucket)",
    "market_building__management_company": "Management company", "market_building__state": "State",
    "market_building__msa": "Metro area",
}
USD = FuncFormatter(s.dollars)
USD_K = FuncFormatter(lambda x, _p: rf"\textdollar{x / 1000:.0f}k" if x >= 1000 else rf"\textdollar{x:.0f}")


def _clean(ax, left=True):
    ax.spines["left"].set_visible(left)
    if not left:
        ax.tick_params(axis="y", length=0)


def rent_map():
    """Median asking rent by submarket: one bubble each, sized by listings."""
    d = pd.read_csv(s.DATA / "submarket_map.csv")
    lo, hi = d.rent.min(), d.rent.max()
    pad = 0.06
    ratios = []
    for k, _ in MARKETS:
        m = d[d.market == k]
        w = (m.lon.max() - m.lon.min()) * np.cos(np.radians(m.lat.mean()))
        ratios.append((w + 2 * pad) / (m.lat.max() - m.lat.min() + 2 * pad))
    fig, axes = s.figure(s.FULL, 2.2, ncols=3, gridspec_kw={"width_ratios": ratios})
    size = lambda n: 18 + 260 * n / d.listings.max()
    for ax, (k, name) in zip(axes, MARKETS):
        m = d[d.market == k].sort_values("listings", ascending=False)
        sc = ax.scatter(m.lon, m.lat, s=size(m.listings), c=m.rent, cmap=BLUES, vmin=lo, vmax=hi,
                        edgecolor="white", linewidth=0.7, zorder=2)
        ax.set_aspect(1 / np.cos(np.radians(m.lat.mean())))
        ax.set_xlim(m.lon.min() - pad, m.lon.max() + pad)
        ax.set_ylim(m.lat.min() - pad, m.lat.max() + pad)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(rf"\textbf{{{name}}}\enspace{{\color{{ink2}}{len(m)} submarkets}}", loc="left", fontsize=8)
    cb = fig.colorbar(sc, ax=axes, shrink=0.75, aspect=20, pad=0.015, format=USD)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7)
    cb.set_label("Median asking rent", fontsize=7.5)
    s.save(fig, "rent_map")


def rent_ladder():
    """Asking rent by bedrooms in each market: median, middle half, and 10th to 90th percentile."""
    d = pd.read_csv(s.DATA / "rent_by_market_beds.csv")
    labels = ["Studio", "1 bed", "2 beds", "3 beds", "4+ beds"]
    fig, axes = s.figure(s.FULL, 1.65, ncols=3, sharex=True, sharey=True)
    for ax, (k, name) in zip(axes, MARKETS):
        m = d[d.market == k].set_index("beds")
        for i, beds in enumerate(range(5)):
            if beds not in m.index:
                continue
            r = m.loc[beds]
            ax.plot([r.p10, r.p90], [i, i], color=s.MUTED, lw=1, solid_capstyle="round", zorder=1)
            ax.plot([r.p25, r.p75], [i, i], color=s.ACCENT, lw=4.5, solid_capstyle="round", zorder=2)
            ax.scatter([r.p50], [i], s=13, color="white", edgecolor=s.INK, linewidth=0.8, zorder=3)
        ax.set_title(rf"\textbf{{{name}}}", loc="left", fontsize=8)
        ax.xaxis.set_major_formatter(USD_K)
        ax.xaxis.grid(True, zorder=0)
        _clean(ax, left=False)
    axes[0].set_yticks(range(5)); axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(500, 5200)
    fig.supxlabel("Asking rent per month", fontsize=8, color=s.INK)
    s.save(fig, "rent_ladder")


def funnel():
    """From every column to the model's inputs."""
    d = pd.read_csv(s.DATA / "funnel.csv")
    fig, ax = s.figure(s.FULL, 1.55)
    y = np.arange(len(d))
    ax.barh(y, d.n, color=s.ACCENT, height=0.62, zorder=2)
    ax.barh(y[-1], d.n.iloc[-1], color=s.INK, height=0.62, zorder=2)
    for yi, n in zip(y, d.n):
        ax.text(n + 4, yi, f"{n}", va="center", fontsize=7.5, color=s.INK)
    ax.set_yticks(y); ax.set_yticklabels(d.stage)
    ax.invert_yaxis()
    ax.set_xlim(0, d.n.max() * 1.08)
    ax.set_xticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    s.save(fig, "funnel")


def fit():
    """Predicted against actual on held-out buildings, and the bias by rent level."""
    g = pd.read_csv(s.DATA / "fit_hexgrid.csv")
    b = pd.read_csv(s.DATA / "bias_by_band.csv")
    fig, (a1, a2) = s.figure(s.FULL, 2.45, ncols=2, gridspec_kw={"width_ratios": [1, 1.25]})
    step = g.actual.diff().abs().replace(0, np.nan).min()
    a1.scatter(g.actual, g.pred, c=g["count"], cmap=BLUES, norm=LogNorm(), s=4.2, marker="s",
               linewidths=0, rasterized=True)
    a1.plot([400, 5200], [400, 5200], color=s.BRICK, lw=0.8, zorder=3)
    a1.set_xlim(400, 5200); a1.set_ylim(400, 5200); a1.set_aspect("equal")
    for ax in (a1,):
        ax.xaxis.set_major_formatter(USD_K); ax.yaxis.set_major_formatter(USD_K)
    a1.set_xlabel("Actual asking rent"); a1.set_ylabel("Predicted")
    a1.set_title(r"\textbf{a}\enspace Fit on 600 unseen buildings", loc="left", fontsize=8)

    y = np.arange(len(b))
    colors = [s.ACCENT if v > 0 else s.BRICK for v in b.bias]
    a2.barh(y, b.bias, color=colors, height=0.62, zorder=2)
    a2.axvline(0, color=s.INK2, lw=0.7, zorder=3)
    for yi, v in zip(y, b.bias):
        a2.text(v + (12 if v > 0 else -12), yi, rf"{'+' if v > 0 else '−'}\textdollar{abs(v):,.0f}",
                va="center", ha="left" if v > 0 else "right", fontsize=7, color=s.INK)
    a2.set_yticks(y); a2.set_yticklabels([rf"\textdollar{l}" if not l.startswith(("<", ">")) else
                                          rf"{l[0]}\textdollar{l[1:]}" for l in b.band])
    a2.invert_yaxis()
    a2.set_xlim(-330, 1000)
    a2.set_xlabel(r"Mean actual minus predicted (\textdollar)")
    a2.xaxis.grid(True, zorder=0)
    _clean(a2, left=False)
    a2.set_title(r"\textbf{b}\enspace Underpredicts the expensive, overpredicts the cheap", loc="left", fontsize=8)
    s.save(fig, "fit")


def segments():
    """Typical error and bias by market, unit size, class and building situation."""
    d = pd.read_csv(s.DATA / "error_by_segment.csv")
    overall = d[d.group == "All"].iloc[0]
    d = d[d.group != "All"].reset_index(drop=True)
    gap = 0.7
    ys, y, prev = [], 0.0, None
    for g in d.group:
        if prev is not None and g != prev:
            y += gap
        ys.append(y); y += 1; prev = g
    ys = np.array(ys)
    fig, (a1, a2) = s.figure(s.FULL, 2.75, ncols=2, sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    a1.axvline(overall.mdape, color=s.RULE, lw=0.9, zorder=1)
    a1.hlines(ys, overall.mdape, d.mdape, color=s.RULE, lw=0.9, zorder=1)
    a1.scatter(d.mdape, ys, s=20, color=s.ACCENT, edgecolor="white", linewidth=0.5, zorder=3)
    a1.text(overall.mdape, ys.min() - 1.0, rf"all listings: {overall.mdape:.1f}\%", ha="center",
            fontsize=7, color=s.INK2)
    a1.set_xlabel(r"MdAPE (\%)")
    a1.xaxis.grid(True, zorder=0)
    a1.set_title(r"\textbf{a}\enspace Typical error", loc="left", fontsize=8)

    colors = [s.ACCENT if v > 0 else s.BRICK for v in d.bias]
    a2.barh(ys, d.bias, color=colors, height=0.62, zorder=2)
    a2.axvline(0, color=s.INK2, lw=0.7, zorder=3)
    a2.set_xlabel(r"Mean actual minus predicted (\textdollar)")
    a2.xaxis.grid(True, zorder=0)
    a2.set_title(r"\textbf{b}\enspace Bias: positive means underpredicted", loc="left", fontsize=8)
    _clean(a2, left=False)

    a1.set_yticks(ys); a1.set_yticklabels(d.label)
    a1.invert_yaxis()
    _clean(a1, left=False)
    for g in d.group.unique():
        top = ys[d.group == g].min()
        a1.text(-0.02, top - 0.62, rf"\textbf{{{g}}}", transform=a1.get_yaxis_transform(),
                ha="right", fontsize=7, color=s.INK2)
    s.save(fig, "segments")


def drivers():
    """Held-out importance of the model's inputs, and of the categorical candidates."""
    imp = pd.read_csv(s.DATA / "gbm_importance.csv")
    cg = pd.read_csv(s.DATA / "categorical_scores.csv").sort_values("perm_dMAE", ascending=False)
    fig, (a1, a2, a3) = s.figure(s.FULL, 2.4, ncols=3, gridspec_kw={"width_ratios": [1.15, 1, 0.75]})

    y = np.arange(len(imp))
    a1.barh(y, imp.dmae, xerr=imp.sd, color=s.ACCENT, height=0.62, zorder=2,
            error_kw=dict(lw=0.6, ecolor=s.INK2, capsize=0))
    a1.set_yticks(y); a1.set_yticklabels([NAMES[f] for f in imp.feature]); a1.invert_yaxis()
    a1.set_xlabel(r"\textdollar\ MAE lost when shuffled")
    a1.xaxis.grid(True, zorder=0); _clean(a1, left=False)
    a1.set_title(r"\textbf{a}\enspace Inputs to the current model", loc="left", fontsize=8)

    y2 = np.arange(len(cg))
    kept = cg.decision.eq("keep").to_numpy()
    col = np.where(kept, s.ACCENT, s.MUTED)
    a2.barh(y2, cg.perm_dMAE.clip(lower=0), color=col, height=0.62, zorder=2)
    a2.set_yticks(y2); a2.set_yticklabels([NAMES[f] for f in cg.feature]); a2.invert_yaxis()
    a2.set_xlabel(r"\textdollar\ MAE lost, unseen buildings")
    a2.xaxis.grid(True, zorder=0); _clean(a2, left=False)
    a2.set_title(r"\textbf{b}\enspace Categorical candidates", loc="left", fontsize=8)

    a3.barh(y2, cg.eta_sq_excess, color=col, height=0.62, zorder=2)
    a3.set_yticks(y2); a3.set_yticklabels([]); a3.invert_yaxis()
    a3.set_xlabel(r"In-sample fit, $\eta^2$")
    a3.xaxis.grid(True, zorder=0); _clean(a3, left=False)
    a3.set_title(r"\textbf{c}\enspace Same, in sample", loc="left", fontsize=8)
    s.save(fig, "drivers")


def concessions():
    """Rule-based concession extraction, by who wrote the text and by concession type."""
    d = pd.read_csv(s.DATA / "concession_summary.csv")
    src = d[d.slice_by == "text_source"].set_index("slice")
    typ = d[d.slice_by == "type"].set_index("slice").sort_values("amount_acc")
    fig, (a1, a2) = s.figure(s.FULL, 1.95, ncols=2, gridspec_kw={"width_ratios": [1.35, 1]})
    metrics = [("found_pct", "Amount\nfound"), ("basis_acc", "Basis\ncorrect"),
               ("amount_acc", "Amount\ncorrect"), ("total_within_5pct", "Value\nwithin 5\\%")]
    x = np.arange(len(metrics)); w = 0.36
    for i, (k, color, label) in enumerate([("llm", s.ACCENT, "LLM-written"), ("grammar", s.BRICK, "Template-written")]):
        v = [src.loc[k, m] for m, _ in metrics]
        a1.bar(x + (i - 0.5) * w, v, width=w, color=color, zorder=2,
               label=f"{label} (n = {int(src.loc[k, 'n']):,})")
        for xi, vi in zip(x + (i - 0.5) * w, v):
            a1.text(xi, vi + 2, f"{vi:.0f}", ha="center", fontsize=6.5, color=s.INK)
    a1.set_xticks(x); a1.set_xticklabels([l for _, l in metrics])
    a1.set_ylim(0, 138); a1.set_yticks([0, 50, 100]); a1.set_ylabel(r"\% of specials")
    a1.yaxis.grid(True, zorder=0)
    a1.legend(loc="upper left", ncols=2, handlelength=1, borderaxespad=0.1, columnspacing=1.2)
    a1.set_title(r"\textbf{a}\enspace By who wrote the text", loc="left", fontsize=8)

    names = {"upfront_free_months": "Months free", "recurring_discount": "Monthly discount",
             "one_time_credit": "One-time credit", "look_and_lease": "Look and lease"}
    y = np.arange(len(typ))
    a2.barh(y, typ.amount_acc, color=s.ACCENT, height=0.6, zorder=2)
    for yi, v in zip(y, typ.amount_acc):
        a2.text(v - 2, yi, f"{v:.0f}\\%", ha="right", va="center", fontsize=7, color="white")
    a2.set_yticks(y); a2.set_yticklabels([names[t] for t in typ.index])
    a2.set_xlim(0, 100); a2.set_xticks([0, 50, 100])
    a2.set_xlabel(r"Amount correct (\%)")
    _clean(a2, left=False)
    a2.set_title(r"\textbf{b}\enspace By concession type", loc="left", fontsize=8)
    s.save(fig, "concessions")


FIGURES.update({"rent_map": rent_map, "rent_ladder": rent_ladder, "funnel": funnel, "fit": fit,
                "segments": segments, "drivers": drivers, "concessions": concessions})


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    print("figures.py")
    for name in wanted:
        FIGURES[name]()
