"""Load just the columns needed for EDA from the Dwellsy export CSVs into one DataFrame.

No parquet, no files written. usecols skips photos/description entirely, and the
amenities text is turned into a count while reading, so the text is never kept.

    df = load_real()                       # every CSV in data/real/ (searched recursively)
    df = load_real([Path("a.csv")])        # specific files
    df = load_real(nrows=500_000)          # quick look (per file)

Adds a `market` column (short MSA name). Rows are de-duplicated on listing_id, so
loading the same data twice (e.g. an export plus the zip it came from) is harmless.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real"

COLS = [
    "listing_id", "uru_id", "community_id", "msa_name", "uru_type",
    "latitude", "longitude", "bedrooms", "full_baths", "half_baths", "square_feet",
    "year_built", "rent_amount", "listing_deposit", "top_down_community_count",
    "creation_time", "deactivation_time", "property_listing_status",
]


def _read_one(path: Path, nrows, chunksize) -> pd.DataFrame:
    parts = []
    reader = pd.read_csv(
        path, usecols=COLS + ["amenities"], nrows=nrows, chunksize=chunksize,
        dtype={"year_built": "string"}, parse_dates=["creation_time", "deactivation_time"],
    )
    for ch in reader:
        ch["n_amenities"] = ch["amenities"].str.count(";") + 1
        ch = ch.drop(columns="amenities")
        ch["year_built"] = pd.to_numeric(ch["year_built"], errors="coerce")
        parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def load_real(paths=None, nrows: int | None = None, chunksize: int = 250_000) -> pd.DataFrame:
    if paths is None:
        paths = sorted(DATA_DIR.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no CSV files found under {DATA_DIR}")
    frames = []
    for p in map(Path, paths):
        d = _read_one(p, nrows, chunksize)
        print(f"loaded {p.name}: {len(d):,} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates("listing_id", keep="first").reset_index(drop=True)
    if len(df) < n0:
        print(f"dropped {n0 - len(df):,} duplicate listing_id rows")
    df["market"] = df["msa_name"].str.split("-").str[0].str.split(",").str[0]
    for c in ("msa_name", "uru_type", "property_listing_status", "market"):
        df[c] = df[c].astype("category")
    return df


if __name__ == "__main__":
    d = load_real()
    print(d.shape, f"{d.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(d.market.value_counts().to_string())
    print(d.isna().mean().round(3).to_string())