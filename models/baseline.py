"""Baseline models (M3).

- :func:`naive_scout_scores` — rank purely by market value at cutoff age. This is
  the bar the real model must clear (SPEC §9.1); "external scouting consensus".
- :class:`LogRegBaseline` — logistic regression on a handful of simple youth
  features + position, class-weighted for imbalance. LogReg is roughly calibrated
  out of the box; full calibration is assessed in eval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SIMPLE_NUMERIC = [
    "youth_minutes_total",
    "youth_goals_total",
    "youth_ga_per90",
    "youth_minutes_trend",
    "best_level_pre_cutoff",
    "minutes_U15",
    "minutes_U17",
    "minutes_U19",
    "height_cm",
]
SIMPLE_CATEGORICAL = ["position"]


def naive_scout_scores(
    features: pd.DataFrame, value_col: str = "market_value_at_cutoff_eur"
) -> pd.Series:
    """Higher = more promising. Missing market value -> lowest rank."""
    if value_col not in features:
        return pd.Series(0.0, index=features.index)
    v = pd.to_numeric(features[value_col], errors="coerce")
    floor = v.min(skipna=True) if v.notna().any() else 0.0
    return v.fillna(floor).astype(float)


class LogRegBaseline:
    def __init__(self, numeric=None, categorical=None):
        self.numeric = list(numeric or SIMPLE_NUMERIC)
        self.categorical = list(categorical or SIMPLE_CATEGORICAL)
        self.pipeline: Pipeline | None = None

    def _build(self, X: pd.DataFrame) -> Pipeline:
        num = [c for c in self.numeric if c in X.columns]
        cat = [c for c in self.categorical if c in X.columns]
        pre = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                    num,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("oh", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat,
                ),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    def fit(self, X: pd.DataFrame, y) -> LogRegBaseline:
        self.pipeline = self._build(X)
        self.pipeline.fit(X, np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("fit first")
        return self.pipeline.predict_proba(X)[:, 1]
