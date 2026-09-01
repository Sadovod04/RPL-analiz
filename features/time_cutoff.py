"""The single choke point for time-cutoff enforcement (SPEC §7, §8).

Every season row MUST pass through :func:`before_cutoff` before it is aggregated
into a feature. Features may only use observations strictly before the player
reaches ``cutoff_age``. Keeping this in one tiny, tested function is the whole
defense against leakage.
"""

from __future__ import annotations

import pandas as pd

AGE_COL = "age_at_season"


def before_cutoff(df: pd.DataFrame, cutoff_age: float, *, age_col: str = AGE_COL) -> pd.DataFrame:
    """Return only the rows observed strictly before ``cutoff_age``.

    Rows with a missing age are dropped (cannot prove they are pre-cutoff).
    """
    if age_col not in df.columns:
        raise KeyError(f"{age_col!r} not in dataframe; cannot enforce time cutoff")
    mask = df[age_col].notna() & (df[age_col] < cutoff_age)
    return df.loc[mask].copy()


def assert_within_cutoff(df: pd.DataFrame, cutoff_age: float, *, age_col: str = AGE_COL) -> None:
    """Raise if any row is at/after the cutoff. Use in tests and pre-fit checks."""
    bad = df.loc[df[age_col].notna() & (df[age_col] >= cutoff_age)]
    if len(bad):
        raise AssertionError(f"{len(bad)} row(s) at/after cutoff age {cutoff_age}")
