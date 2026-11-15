"""Shared pandas dtype normalization helpers."""
import re

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?")


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize date, boolean, and string columns."""
    df = df.copy()
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        if is_bool_dtype(dtype):
            if dtype != pd.BooleanDtype():
                df[col] = series.astype("boolean")
            continue
        if is_datetime64_any_dtype(dtype) or is_numeric_dtype(dtype):
            continue
        if not (dtype == object or is_string_dtype(dtype)):
            continue
        values = series.dropna()
        if values.empty:
            df[col] = series.astype("string")
            continue
        strings = values.astype(str)
        if strings.str.fullmatch(DATE_PATTERN).all():
            parsed = pd.to_datetime(series, format="mixed", errors="coerce")
            if parsed.notna().sum() == series.notna().sum():
                df[col] = parsed.astype("datetime64[ns]")
                continue
        if values.map(type).isin((bool, np.bool_)).all():
            df[col] = series.astype("boolean")
        else:
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
