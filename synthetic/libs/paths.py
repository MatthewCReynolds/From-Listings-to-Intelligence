"""Paths for the synthetic-data track: everything resolves inside synthetic/."""
from pathlib import Path

TRACK = "synthetic"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MASTER = DATA / "master_listing.parquet"
CATALOG = DATA / "master_columns.csv"
DUCKDB = DATA / "accolade.duckdb"
REALISTIC = DATA / "realistic-v2"
OUTPUTS = ROOT / "outputs"


def output_dir(track: str = TRACK) -> Path:
    """Return and create this track's output directory."""
    if track != TRACK:
        raise ValueError(f"this is the {TRACK!r} track, got {track!r}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS
