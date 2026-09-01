"""Baseline models (M3).

- ``naive_scout``: rank purely by market value at cutoff age (the bar to beat).
- ``LogRegBaseline``: logistic regression on simple features (minutes, G+A/90,
  age, position), temporal split, calibrated probabilities.

Status: skeleton — implemented in M3.
"""

from __future__ import annotations

import pandas as pd


def naive_scout_scores(features: pd.DataFrame,
                       value_col: str = "market_value_at_cutoff_eur") -> pd.Series:
    raise NotImplementedError("M3")


class LogRegBaseline:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogRegBaseline":
        raise NotImplementedError("M3")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("M3")
