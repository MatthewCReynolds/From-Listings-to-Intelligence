"""Paths for the real-data track: everything resolves inside the folder that holds libs/."""
from pathlib import Path

TRACK = "real"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REAL_DATA = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"


def output_dir(track: str = TRACK) -> Path:
    """Return and create this track's output directory."""
    if track != TRACK:
        raise ValueError(f"this is the {TRACK!r} track, got {track!r}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS
