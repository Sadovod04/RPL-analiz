"""Gradient boosting (M4).

CatBoost with native categorical handling (``academy_club``, ``position``),
class weights for imbalance, Optuna tuning under **GroupKFold by ``player_id``**
so a player never straddles train/val folds (SPEC §9.2, §10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from eval.metrics import pr_auc

CATEGORICAL = ["academy_club", "position", "position_detail"]
_CAT_NA = "__NA__"


def _prep(X: pd.DataFrame, cat_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X = X.copy()
    cats = [c for c in cat_features if c in X.columns]
    for c in cats:
        X[c] = X[c].astype("object").where(X[c].notna(), _CAT_NA).astype(str)
    for c in [col for col in X.columns if col not in cats]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cats


@dataclass
class CatBoostBreakthrough:
    params: dict = field(default_factory=dict)
    cat_features: list[str] = field(default_factory=lambda: list(CATEGORICAL))
    model: object = None
    _cats: list[str] = field(default_factory=list)

    def _default_params(self) -> dict:
        return {
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "iterations": 400,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,  # no catboost_info/ litter
            **self.params,
        }

    def fit(self, X: pd.DataFrame, y, groups=None) -> CatBoostBreakthrough:
        from catboost import CatBoostClassifier, Pool

        Xp, self._cats = _prep(X, self.cat_features)
        pool = Pool(Xp, np.asarray(y).astype(int), cat_features=self._cats)
        self.model = CatBoostClassifier(**self._default_params())
        self.model.fit(pool)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit first")
        Xp, _ = _prep(X, self.cat_features)
        return self.model.predict_proba(Xp)[:, 1]

    def feature_names(self) -> list[str]:
        return list(self.model.feature_names_) if self.model is not None else []


def group_kfold_scores(
    X: pd.DataFrame, y, groups, params: dict | None = None, n_splits: int = 5
) -> list[float]:
    """PR-AUC per fold, GroupKFold by ``groups`` (player_id)."""
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    scores = []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        m = CatBoostBreakthrough(params=params or {}).fit(X.iloc[tr], y[tr])
        scores.append(pr_auc(y[va], m.predict_proba(X.iloc[va])))
    return scores


def tune(X: pd.DataFrame, y, groups, n_trials: int = 40, n_splits: int = 5, seed: int = 42) -> dict:
    """Optuna search over CatBoost params; objective = mean fold PR-AUC (GroupKFold)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        }
        s = group_kfold_scores(X, y, groups, params=params, n_splits=n_splits)
        return float(np.mean(s)) if s else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
