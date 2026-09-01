"""Survival models (M5).

The binary target throws away *when* a player broke through and forces a call on
players whose careers are still open. Survival analysis keeps both: event time =
age at RPL debut, right-censored for those who haven't debuted (SPEC §3, §9.3).

- :class:`CoxBreakthrough` — Cox Proportional Hazards (lifelines). Always available.
- :class:`RSFBreakthrough` — Random Survival Forest (scikit-survival, ``survival``
  extra). Non-linear, no proportional-hazards assumption. Optional.

Both expose ``probability_by_age(X, age)`` = P(debut by that age) = 1 - S(age | x).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DURATION_COL = "duration"
EVENT_COL = "event_observed"


def survival_frame(
    features: pd.DataFrame, feature_cols: list[str], *, max_corr: float = 0.98
) -> pd.DataFrame:
    """Numeric, NaN-free frame of [features + duration + event] for a linear model.

    Categoricals are one-hot encoded; near-duplicate columns dropped (Cox dislikes
    collinearity); remaining NaNs median-imputed.
    """
    X = features[feature_cols].copy()
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if cat:
        X = pd.get_dummies(X, columns=cat, dummy_na=False, dtype=int)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]  # drop constants

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > max_corr).any()]
    X = X.drop(columns=drop)

    X[DURATION_COL] = features[DURATION_COL].to_numpy()
    X[EVENT_COL] = features[EVENT_COL].astype(int).to_numpy()
    return X


class CoxBreakthrough:
    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.model = None
        self._features: list[str] = []

    def fit(self, frame: pd.DataFrame) -> CoxBreakthrough:
        from lifelines import CoxPHFitter

        self.model = CoxPHFitter(penalizer=self.penalizer)
        self.model.fit(frame, duration_col=DURATION_COL, event_col=EVENT_COL)
        self._features = [c for c in frame.columns if c not in (DURATION_COL, EVENT_COL)]
        return self

    def probability_by_age(self, X: pd.DataFrame, age: float) -> np.ndarray:
        sf = self.model.predict_survival_function(X[self._features], times=[age])
        return (1.0 - sf.iloc[0].to_numpy()).astype(float)

    @property
    def concordance_index_(self) -> float:
        return float(self.model.concordance_index_)


class RSFBreakthrough:
    """Random Survival Forest (needs the ``survival`` extra: ``uv sync --extra survival``)."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self._features: list[str] = []

    def fit(self, frame: pd.DataFrame) -> RSFBreakthrough:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv

        self._features = [c for c in frame.columns if c not in (DURATION_COL, EVENT_COL)]
        y = Surv.from_arrays(
            event=frame[EVENT_COL].astype(bool), time=frame[DURATION_COL].astype(float)
        )
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
        )
        self.model.fit(frame[self._features].to_numpy(float), y)
        return self

    def probability_by_age(self, X: pd.DataFrame, age: float) -> np.ndarray:
        fns = self.model.predict_survival_function(X[self._features].to_numpy(float))
        return np.array([1.0 - fn(age) for fn in fns], dtype=float)
