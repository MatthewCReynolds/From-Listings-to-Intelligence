"""Repository paths shared by notebooks and pipeline scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MASTER = DATA / "master_listing.parquet"
CATALOG = DATA / "master_columns.csv"
DUCKDB = DATA / "accolade.duckdb"
REALISTIC = DATA / "realistic-v2"
REAL_DATA = DATA / "real"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"


def output_dir(track: str) -> Path:
    """Return and create the output directory for a data track."""
    if track not in {"synthetic", "real"}:
        raise ValueError("track must be 'synthetic' or 'real'")
    path = OUTPUTS / track
    path.mkdir(parents=True, exist_ok=True)
    return path
