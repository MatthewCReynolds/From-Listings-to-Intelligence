# Loading just the columns needed for EDA from the Dwellsy export CSV into a DataFrame.

from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "real" / "export_20261020160458.csv"

COLS = [
    "listing_id", "uru_id", "community_id", "msa_name", "uru_type",
    "latitude", "longitude", "bedrooms", "full_baths", "half_baths", "square_feet",
    "year_built", "rent_amount", "listing_deposit", "top_down_community_count",
    "creation_time", "deactivation_time", "property_listing_status",
]


def load_real(path: Path = CSV, nrows: int | None = None, chunksize: int = 250_000) -> pd.DataFrame:
    parts = []
    reader = pd.read_csv(
        path, usecols=COLS + ["amenities"], nrows=nrows, chunksize=chunksize,
        dtype={"year_built": "string"}, parse_dates=["creation_time", "deactivation_time"],
    )
    for ch in reader:
        # count of semicolon-separated amenity tags; NaN stays NaN
        ch["n_amenities"] = ch["amenities"].str.count(";") + 1
        ch = ch.drop(columns="amenities")
        ch["year_built"] = pd.to_numeric(ch["year_built"], errors="coerce")
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    for c in ("msa_name", "uru_type", "property_listing_status"):
        df[c] = df[c].astype("category")
    return df


if __name__ == "__main__":
    d = load_real()
    print(d.shape, f"{d.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(d.isna().mean().round(3).to_string())