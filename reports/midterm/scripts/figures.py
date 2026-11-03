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

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    print("figures.py")
    for name in wanted:
        FIGURES[name]()
