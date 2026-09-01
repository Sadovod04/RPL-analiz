"""Survival models (M5).

Cox Proportional Hazards (lifelines) and Random Survival Forest; optional
DeepSurv via the ``survival`` extra (pycox/torch). Consumes the
``(duration, event_observed)`` tuple from ``features.labels.survival_tuple`` and
yields P(breakthrough by age a).

Status: skeleton — implemented in M5.
"""

from __future__ import annotations

import pandas as pd


class CoxBreakthrough:
    def fit(
        self, df: pd.DataFrame, duration_col: str = "duration", event_col: str = "event_observed"
    ) -> CoxBreakthrough:
        raise NotImplementedError("M5")

    def probability_by_age(self, X: pd.DataFrame, age: float) -> pd.Series:
        raise NotImplementedError("M5")
