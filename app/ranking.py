"""Prospect ranking (M6) — the logic behind the dashboard, kept pure and testable.

Train the CatBoost model on players whose outcome is resolved (``target`` in
{0, 1}); score the *open* cohort (``target == -1``, i.e. still too young to judge)
and return them ranked by P(breakthrough), with a per-player SHAP breakdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.build_features import feature_columns
from features.labels import CENSORED
from models.gbm import CatBoostBreakthrough


def split_resolved_open(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved = df[df["target"] != CENSORED].copy()
    open_cohort = df[df["target"] == CENSORED].copy()
    return resolved, open_cohort


def train_ranker(resolved: pd.DataFrame, params: dict | None = None) -> CatBoostBreakthrough:
    feats = feature_columns(resolved)
    y = resolved["target"].astype(int)
    return CatBoostBreakthrough(params=params or {"iterations": 300}).fit(resolved[feats], y)


def rank_prospects(
    df: pd.DataFrame, model: CatBoostBreakthrough | None = None, *, top: int | None = None
) -> pd.DataFrame:
    resolved, open_cohort = split_resolved_open(df)
    if model is None:
        if resolved["target"].nunique() < 2:
            raise ValueError("need both classes in the resolved set to train a ranker")
        model = train_ranker(resolved)

    feats = feature_columns(df)
    scored = open_cohort.copy()
    scored["breakthrough_score"] = (
        model.predict_proba(scored[feats]) if len(scored) else np.array([])
    )
    keep = [
        "player_id",
        "canonical_name",
        "birth_year",
        "academy_club",
        "position",
        "breakthrough_score",
        "youth_minutes_total",
        "youth_ga_per90",
        "best_level_pre_cutoff",
        "played_youth_league",
    ]
    out = scored[[c for c in keep if c in scored.columns]].sort_values(
        "breakthrough_score", ascending=False
    )
    return out.head(top) if top else out


def explain_player(model: CatBoostBreakthrough, df: pd.DataFrame, player_id: str) -> pd.Series:
    """Signed SHAP contributions for one player (positive => pushes score up)."""
    from eval.shap_analysis import shap_values

    feats = feature_columns(df)
    row = df[df["player_id"] == player_id]
    if row.empty:
        raise KeyError(player_id)
    sv, Xp = shap_values(model, row[feats])
    return pd.Series(sv[0], index=Xp.columns).sort_values(key=np.abs, ascending=False)
