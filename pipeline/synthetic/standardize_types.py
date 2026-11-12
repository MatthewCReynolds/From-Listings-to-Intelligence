"""Standardize master listing dtypes and update column metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so libs/ imports work

import pandas as pd
from libs.dtypes import get_type_group, standardize_types
from libs.paths import CATALOG as MASTER_COLUMNS, MASTER

MASTER_PARQUET = MASTER

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
