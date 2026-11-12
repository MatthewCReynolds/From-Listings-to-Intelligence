"""Categorical and amenity analysis for the real Dwellsy listings.

Moved unchanged from
notebooks/real/phase_01_market_rent/03_categorical_analysis.ipynb (originally by Lady Infante Acosta).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_string_variables(df, columns, rare_threshold=0.01):
    """Profile string columns: missingness, cardinality, top category and rare categories."""
    results = []

    for col in columns:

        series = df[col].dropna()
        counts = series.value_counts(normalize=True)

        results.append({
            "column": col,
            "missing_pct": df[col].isna().mean() * 100,
            "n_unique": series.nunique(),
            "cardinality_ratio": (
                series.nunique() / len(series)
                if len(series) > 0 else np.nan
            ),
            "top_category": (
                counts.index[0] if len(counts) > 0 else None
            ),
            "top_category_pct": (
                counts.iloc[0] * 100 if len(counts) > 0 else np.nan
            ),
            "rare_categories": (
                counts < rare_threshold
            ).sum(),
            "rare_records_pct": (
                counts[counts < rare_threshold].sum() * 100
            ),
        })

    return pd.DataFrame(results)


def get_category_distribution(df, columns):
    """Count and percentage of every category (missing included) for each column."""
    distributions = []

    for col in columns:

        counts = df[col].value_counts(dropna=False)

        temp = pd.DataFrame({
            "column": col,
            "category": counts.index.astype(str),
            "count": counts.values,
            "percentage": counts.values / counts.sum() * 100
        })

        distributions.append(temp)

    return pd.concat(distributions, ignore_index=True)


def analyze_categorical_target(df, columns, target, min_count=20):
    """Eta-squared association of each categorical column with the target, plus per-category stats."""
    results = []
    category_details = {}

    for col in columns:

        temp = df[[col, target]].dropna().copy()

        if temp.empty:
            continue

        # Group statistics
        grouped = temp.groupby(col)[target].agg(
            count="count",
            mean="mean",
            median="median",
            std="std"
        )

        category_details[col] = grouped.sort_values(
            "mean", ascending=False
        )

        # Eta squared
        overall_mean = temp[target].mean()

        ## Sum of squares total
        ss_total = (
            (temp[target] - overall_mean) ** 2
        ).sum()

        ## Sum of squares between
        ss_between = (
            grouped["count"] *
            (grouped["mean"] - overall_mean) ** 2
        ).sum()

        ## Eta squared calculation
        eta_squared = (
            ss_between / ss_total
            if ss_total > 0 else np.nan
        )

        # Categories with insufficient observations
        small_categories = (
            grouped["count"] < min_count
        ).sum()

        results.append({
            "column": col,
            "eta_squared": eta_squared,
            "n_categories": len(grouped),
            "small_categories": small_categories,
            "min_category_count": grouped["count"].min(),
            "max_category_count": grouped["count"].max(),
            "target_mean": overall_mean
        })

    df_association = pd.DataFrame(results)

    return (
        df_association.sort_values(
            "eta_squared", ascending=False
        ),
        category_details
    )


def plot_category_target(
    df,
    column,
    target,
    min_count=20,
    top_n=15
):
    """Box plot of the target for the most frequent categories of one column."""
    temp = df[[column, target]].dropna().copy()

    # Group categories
    counts = temp[column].value_counts()

    valid_categories = counts[
        counts >= min_count
    ].index

    temp = temp[temp[column].isin(valid_categories)]

    # Select categories with highest frequency
    selected = temp[column].value_counts().head(top_n).index

    temp = temp[temp[column].isin(selected)]

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=temp,
        x=column,
        y=target
    )

    plt.xticks(rotation=45, ha="right")
    plt.title(f"{target} distribution by {column}")
    plt.tight_layout()
    plt.show()


def analyze_amenity_target(df, columns, target, min_count=20):
    """Compare the target with and without each binary amenity, with eta-squared."""
    results = []

    for col in columns:
        temp = df[[col, target]].dropna().copy()

        if temp.empty:
            continue

        # Ensure the amenity is treated as a binary indicator
        temp[col] = temp[col].astype(int)

        grouped = temp.groupby(col)[target].agg(
            count="count",
            mean="mean",
            median="median",
            std="std"
        )

        # Both groups must be present to compare rent with vs. without amenity
        if 0 not in grouped.index or 1 not in grouped.index:
            continue

        count_0 = grouped.loc[0, "count"]
        count_1 = grouped.loc[1, "count"]

        mean_0 = grouped.loc[0, "mean"]
        mean_1 = grouped.loc[1, "mean"]

        # Eta squared for a binary feature
        overall_mean = temp[target].mean()
        ss_total = ((temp[target] - overall_mean) ** 2).sum()

        ss_between = (
            grouped["count"] *
            (grouped["mean"] - overall_mean) ** 2
        ).sum()

        eta_squared = (
            ss_between / ss_total
            if ss_total > 0 else np.nan
        )

        mean_difference = mean_1 - mean_0
        pct_difference = (
            mean_difference / mean_0 * 100
            if mean_0 != 0 else np.nan
        )

        results.append({
            "amenity": col.replace("amenity_", "", 1),
            "column": col,
            "count_with": count_1,
            "count_without": count_0,
            "presence_pct": temp[col].mean() * 100,
            "mean_rent_with": mean_1,
            "mean_rent_without": mean_0,
            "median_rent_with": grouped.loc[1, "median"],
            "median_rent_without": grouped.loc[0, "median"],
            "mean_rent_difference": mean_difference,
            "mean_rent_difference_pct": pct_difference,
            "eta_squared": eta_squared,
            "meets_min_count": (
                count_1 >= min_count and count_0 >= min_count
            )
        })

    return pd.DataFrame(results)


def plot_top_amenity_rent_differences(
    df_ranking,
    top_n=15,
    ranking_metric="eta_squared"
):
    """Bar chart of the rent difference for the top amenities by a ranking metric."""
    if df_ranking.empty:
        print("No amenities meet the minimum observation threshold.")
        return

    plot_data = (
        df_ranking
        .nlargest(top_n, ranking_metric)
        .sort_values("mean_rent_difference")
        .copy()
    )

    plt.figure(figsize=(12, max(6, top_n * 0.4)))

    sns.barplot(
        data=plot_data,
        x="mean_rent_difference",
        y="amenity"
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Difference in mean rent (with amenity - without amenity)")
    plt.ylabel("Amenity")
    plt.title(f"Top {len(plot_data)} amenities by {ranking_metric}")
    plt.tight_layout()
    plt.show()
