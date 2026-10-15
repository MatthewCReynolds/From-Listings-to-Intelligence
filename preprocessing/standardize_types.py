
"""Standardize master listing dtypes and update column metadata."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

REPO = Path(__file__).resolve().parent.parent
MASTER_PARQUET = REPO / "data" / "master_listing.parquet"
MASTER_COLUMNS = REPO / "data" / "master_columns.csv"
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize date, boolean, and string columns."""
    df = df.copy()

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        # Convert regular boolean columns to nullable boolean.
        if is_bool_dtype(dtype):
            if dtype != pd.BooleanDtype():
                df[col] = series.astype("boolean")
            continue

        # Keep existing datetime and numeric columns unchanged.
        if is_datetime64_any_dtype(dtype) or is_numeric_dtype(dtype):
            continue

        # Only inspect text/object columns below.
        if not (dtype == object or is_string_dtype(dtype)):
            continue

        values = series.dropna()

        # Preserve empty columns as strings.
        if values.empty:
            df[col] = series.astype("string")
            continue

        strings = values.astype(str)

        # Detect date columns in YYYY-MM-DD format.
        if strings.str.fullmatch(DATE_PATTERN).all():
            parsed = pd.to_datetime(
                series,
                format="%Y-%m-%d",
                errors="coerce",
            )

            if parsed.notna().sum() == series.notna().sum():
                df[col] = parsed.astype("datetime64[ns]")
                continue

        # Detect actual Python/NumPy booleans, not strings or 0/1.
        if values.map(type).isin((bool, np.bool_)).all():
            df[col] = series.astype("boolean")
        else:
            # Standardize remaining text/object columns.
            df[col] = series.astype("string")

    return df


def get_type_group(dtype) -> str:
    """Map pandas dtypes to numeric, string, boolean, or datetime."""
    if is_bool_dtype(dtype):
        return "boolean"
    if is_datetime64_any_dtype(dtype):
        return "datetime"
    if is_numeric_dtype(dtype):
        return "numeric"
    return "string"


def main() -> None:
    if not MASTER_PARQUET.is_file():
        raise FileNotFoundError(
            f"Master parquet not found: {MASTER_PARQUET}"
        )
    if not MASTER_COLUMNS.is_file():
        raise FileNotFoundError(
            f"Column metadata not found: {MASTER_COLUMNS}"
        )

    df_master = pd.read_parquet(MASTER_PARQUET)
    df_vars_info = pd.read_csv(MASTER_COLUMNS)

    df_adjusted = standardize_types(df_master)

    # Generate updated column metadata.
    df_types = pd.DataFrame({
        "column": df_adjusted.columns,
        "dtype": df_adjusted.dtypes.astype(str).to_numpy(),
        "type_group": [
            get_type_group(dtype)
            for dtype in df_adjusted.dtypes
        ],
    })

    # Remove generated columns before recreating them.
    generated = ["non_null_count", "dtype", "type_group"]
    df_vars_info = df_vars_info.drop(
        columns=generated,
        errors="ignore",
    )

    df_vars_updated = df_vars_info.merge(
        df_types,
        on="column",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # Write parquet to a temporary file before replacing the original.
    temp_parquet = MASTER_PARQUET.with_name(
        "master_listing.tmp.parquet"
    )

    try:
        df_adjusted.to_parquet(temp_parquet, index=False)
        temp_parquet.replace(MASTER_PARQUET)
    finally:
        temp_parquet.unlink(missing_ok=True)

    df_vars_updated.to_csv(MASTER_COLUMNS, index=False)

    print(f"Master shape: {df_adjusted.shape}")
    print(f"Updated parquet: {MASTER_PARQUET}")
    print(f"Updated metadata: {MASTER_COLUMNS}")

    print("\nType groups:")
    print(df_types["type_group"].value_counts().to_string())

    print("\nDtypes:")
    print(df_types["dtype"].value_counts().to_string())


if __name__ == "__main__":
    raise SystemExit(main())
