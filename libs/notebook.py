"""Notebook filters and standard output path helpers."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from libs.paths import MASTER, output_dir


def as_list(value):
    """Normalize one filter value or a list of values."""
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


def need(path, notebook):
    """Raise a clear instruction when a required upstream output is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Run {notebook} first; missing {path}")


@dataclass
class Run:
    """Filter configuration and output paths for a notebook run."""
    number: int
    phase: int = 1
    track: str = "synthetic"
    city: str | list | None = None
    neighborhood: str | list | None = None
    start_date: str | None = None
    end_date: str | None = None
    date_column: str = "market_unit__enter_market"
    input: Path = MASTER

    def suffix(self):
        """Build the shared output suffix from active filters."""
        parts = []
        if self.city is not None:
            parts.append("+".join(map(str, as_list(self.city))))
        if self.neighborhood is not None:
            parts.append("+".join(map(str, as_list(self.neighborhood))))
        if self.start_date is not None or self.end_date is not None:
            parts.append(f"{self.start_date or 'start'}_to_{self.end_date or 'end'}")
        return "" if not parts else "__" + "__".join(parts)

    def out(self, name, ext="csv"):
        """Build an output path with this run's number and active filters."""
        return output_dir(self.track) / f"{self.number:02d}_phase_{self.phase:02d}_{name}{self.suffix()}.{ext}"

    def upstream(self, number, name, ext="csv", phase=None):
        """Build an upstream notebook output path with active filters."""
        phase_number = self.phase if phase is None else phase
        return output_dir(self.track) / f"{number:02d}_phase_{phase_number:02d}_{name}{self.suffix()}.{ext}"

    def load(self, columns):
        """Read requested master columns and apply configured listing filters."""
        requested = list(dict.fromkeys(columns))
        filter_columns = ["market", "market_building__submarket", self.date_column]
        read_columns = list(dict.fromkeys(requested + filter_columns))
        frame = pd.read_parquet(self.input, columns=read_columns)
        valid_values = {
            column: sorted(frame[column].dropna().astype(str).unique())
            for column in ("market", "market_building__submarket")
        }
        from libs.console import note
        for column, value, label in (
            ("market", self.city, "city"),
            ("market_building__submarket", self.neighborhood, "neighborhood"),
        ):
            if value is not None:
                wanted = list(map(str, as_list(value)))
                valid = valid_values[column]
                unknown = sorted(set(wanted) - set(valid))
                if unknown:
                    raise ValueError(f"Unknown {label}(s) {unknown}. Valid values: {valid}")
                before = len(frame)
                frame = frame[frame[column].astype(str).isin(wanted)]
                note(f"{label}: {before:,} → {len(frame):,} rows")
        if self.start_date is not None:
            before = len(frame)
            frame = frame[frame[self.date_column] >= pd.Timestamp(self.start_date)]
            note(f"start date: {before:,} → {len(frame):,} rows")
        if self.end_date is not None:
            before = len(frame)
            frame = frame[frame[self.date_column] <= pd.Timestamp(self.end_date)]
            note(f"end date: {before:,} → {len(frame):,} rows")
        return frame[requested]
