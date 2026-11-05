"""One visual language for every figure in the report.

Figures are typeset by LuaLaTeX through matplotlib's PGF backend, so their text
is set in the same Libertinus faces as the document. Each figure is drawn at the
width it occupies on the page and never rescaled, which keeps figure text at
exactly the size the reader sees beside it.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
FIGURES = REPORT / "generated" / "figures"
DATA = REPORT / "generated" / "data"
TABLES = REPORT / "generated" / "tables"

# \textwidth at letter paper with 1in margins; a half-width panel leaves 0.2in between two.
FULL = 6.5
HALF = 3.15

# Validated for colour-vision deficiency and contrast against white earlier in the
# project. ACCENT carries the series that matters; BRICK is its counterpart.
ACCENT = "#1F6FA8"
BRICK = "#B03E26"
INK = "#15202A"
INK2 = "#4E6273"
MUTED = "#9AA7B1"
RULE = "#C9D4DC"
GRID = "#E6ECF0"
WASH = "#F3F6F8"

plt.rcParams.update({
    "pgf.texsystem": "lualatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r"\usepackage{fontspec}",
        r"\setmainfont{Libertinus Serif}",
        r"\setsansfont{Libertinus Sans}",
        r"\usepackage{unicode-math}",
        r"\setmathfont{Libertinus Math}",
        r"\usepackage{xcolor}",
        r"\definecolor{ink2}{HTML}{4E6273}",
    ]),
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    # Plain-text tick labels: math-mode ticks would switch digits to the math font.
    "axes.formatter.use_mathtext": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "savefig.facecolor": "white",
})


def figure(width=FULL, height=2.2, **kw):
    """A figure at its printed size, laid out by matplotlib's constrained engine."""
    return plt.subplots(figsize=(width, height), layout="constrained", **kw)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure  {path.relative_to(REPORT)}")
    return path


def dollars(x, _pos=None):
    """Tick formatter: dollar amounts, with \\textdollar because matplotlib unescapes \\$."""
    return rf"\textdollar{x:,.0f}"
