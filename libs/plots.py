"""Plot formatting and labeling helpers shared by notebooks."""


def style_axes(axis):
    """Apply the shared chart spines and grid style."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color="#e5e4e0", linewidth=0.5)
    axis.set_axisbelow(True)


def variable_label(variable):
    """Remove the table prefix from a variable name for chart labels."""
    return variable.split("__", 1)[-1]
