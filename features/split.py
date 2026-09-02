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
    test_cohort_to: int | None = None,
    target_col: str = "target",
    cohort_col: str = "birth_year",
    drop_censored_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train = cohorts before ``test_cohort_from``; test = ``[from, to]``.

    ``test_cohort_to`` caps the test window at cohorts old enough that a *negative*
    outcome is actually possible (younger cohorts are "resolved" only if they
    already succeeded, which would inflate the test base rate). Defaults from
    ``config[split]``.
    """
    s = load_settings()["split"]
    if test_cohort_from is None:
        test_cohort_from = s["test_cohort_from"]
    if test_cohort_to is None:
        test_cohort_to = s.get("test_cohort_to", 9999)
    if drop_censored_rows:
        df = drop_censored(df, target_col)
    train = df[df[cohort_col] < test_cohort_from].copy()
    test = df[(df[cohort_col] >= test_cohort_from) & (df[cohort_col] <= test_cohort_to)].copy()
    return train, test
