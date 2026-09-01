"""SHAP analysis (M4).

Global importance + interaction values (esp. ``academy_club`` × everything — the
club is a confounder per SPEC §5.4) and partial dependence for key features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (shap_matrix, prepared_X) for a fitted :class:`CatBoostBreakthrough`."""
    from catboost import Pool

    from models.gbm import _prep

    Xp, cats = _prep(X, model.cat_features)
    pool = Pool(Xp, cat_features=cats)
    raw = model.model.get_feature_importance(pool, type="ShapValues")
    return raw[:, :-1], Xp  # drop the expected-value column


def mean_abs_importance(model, X: pd.DataFrame) -> pd.Series:
    sv, Xp = shap_values(model, X)
    return pd.Series(np.abs(sv).mean(axis=0), index=Xp.columns).sort_values(ascending=False)


def top_interactions(model, X: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    from catboost import Pool

    from models.gbm import _prep

    Xp, cats = _prep(X, model.cat_features)
    inter = model.model.get_feature_importance(Pool(Xp, cat_features=cats), type="Interaction")
    names = model.model.feature_names_
    rows = [
        {"feature_a": names[int(a)], "feature_b": names[int(b)], "strength": float(s)}
        for a, b, s in inter[:top]
    ]
    return pd.DataFrame(rows)


def explain(model, X: pd.DataFrame) -> dict:
    """Convenience bundle for the report notebook."""
    return {
        "importance": mean_abs_importance(model, X),
        "interactions": top_interactions(model, X),
    }
