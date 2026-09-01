"""Gradient boosting model (M4).

CatBoost (native categoricals: club, position), Optuna tuning, GroupKFold by
``player_id``, class weights for imbalance. Logged to MLflow. SHAP handled in
``eval.shap_analysis``.

Status: skeleton — implemented in M4.
"""

from __future__ import annotations

import pandas as pd


class CatBoostBreakthrough:
    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        cat_features: list[str] | None = None,
    ) -> CatBoostBreakthrough:
        raise NotImplementedError("M4")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("M4")


def tune(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_trials: int = 50) -> dict:
    raise NotImplementedError("M4")
