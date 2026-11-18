"""Median household income by ZIP code (Census ACS 5-year, table B19013_001E).

Used by notebooks/phase_01_income_enrichment. The ACS file is a local download kept in
data/external/ (gitignored). The API key is read from the CENSUS_API_KEY environment
variable and is never stored in code or committed.
"""
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from libs import paths

EXTERNAL = paths.DATA / "external"
DEFAULT_ACS_FILE = EXTERNAL / "acs_2024_median_income_zcta.json"
INCOME_COLUMN = "median_hh_income"
MISSING_FLAG = "income_missing"
TOP_CODE = 250_001  # ACS reports "250,000+" as 250001


def fetch_acs_json(out_path=DEFAULT_ACS_FILE, vintage=2024):
    """Download ZCTA median household income from the Census API to out_path.

    Needs a free key in the CENSUS_API_KEY environment variable.
    """
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError("Set the CENSUS_API_KEY environment variable first.")
    query = urllib.parse.urlencode({
        "get": "NAME,B19013_001E",
        "for": "zip code tabulation area:*",
        "key": key,
    })
    url = f"https://api.census.gov/data/{vintage}/acs/acs5?{query}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read().decode("utf-8")
    json.loads(payload)  # fail early if the response is not JSON
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload)
    return out_path


def load_income_by_zip(path=DEFAULT_ACS_FILE):
    """Read the ACS JSON (list of lists, first row is the header) into a ZIP -> income table.

    Census suppression codes (large negative numbers) become NaN.
    """
    rows = json.loads(Path(path).read_text())
    raw = pd.DataFrame(rows[1:], columns=rows[0])
    zip_col = "zip code tabulation area"
    out = pd.DataFrame({
        "zip": raw[zip_col].astype(str).str.zfill(5),
        INCOME_COLUMN: pd.to_numeric(raw["B19013_001E"], errors="coerce"),
    })
    out.loc[out[INCOME_COLUMN] < 0, INCOME_COLUMN] = np.nan
    return out.drop_duplicates("zip").reset_index(drop=True)


def add_income(df, income, zip_col="address_zip"):
    """Left-join income onto listings by 5-digit ZIP and add a missing-income flag."""
    out = df.copy()
    key = out[zip_col].astype("string").str.strip().str[:5].str.zfill(5)
    mapped = key.map(income.set_index("zip")[INCOME_COLUMN])
    out[INCOME_COLUMN] = pd.to_numeric(mapped, errors="coerce")
    out[MISSING_FLAG] = out[INCOME_COLUMN].isna().astype(int)
    return out


def coverage_report(df, zip_col="address_zip"):
    """Share of listings and of distinct ZIPs that received an income value."""
    zips = df.groupby(zip_col, dropna=False)[INCOME_COLUMN].first()
    return pd.Series({
        "listings": len(df),
        "listings_with_income_pct": 100 * df[INCOME_COLUMN].notna().mean(),
        "distinct_zips": len(zips),
        "zips_with_income_pct": 100 * zips.notna().mean(),
        "listings_top_coded_pct": 100 * (df[INCOME_COLUMN] >= TOP_CODE).mean(),
    })
