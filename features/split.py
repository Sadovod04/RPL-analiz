"""Train/test splitting (SPEC §10).

Temporal split by birth-year cohort — imitates real use (predict future from
past). Censored rows (``target == -1``) are dropped for the binary model; keep
the full frame for the survival model (M5).
"""

from __future__ import annotations

import pandas as pd

from features.labels import CENSORED
from settings import load_settings


def drop_censored(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    return df[df[target_col] != CENSORED].copy()


def temporal_split(
    df: pd.DataFrame,
    *,
    test_cohort_from: int | None = None,
    cohort_col: str = "birth_year",
    drop_censored_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_cohort_from is None:
        test_cohort_from = load_settings()["split"]["test_cohort_from"]
    if drop_censored_rows:
        df = drop_censored(df)
    train = df[df[cohort_col] < test_cohort_from].copy()
    test = df[df[cohort_col] >= test_cohort_from].copy()
    return train, test
