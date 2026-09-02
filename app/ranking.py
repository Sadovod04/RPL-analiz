"""Prospect ranking + analogy (M6/M7) — logic behind the dashboard, kept pure.

Train on players whose outcome is resolved (``{target_col}`` in {0, 1}); score the
*open* cohort (still too young to judge) and return them ranked by P(breakthrough).
``similar_breakthrough_players`` gives the analogy the user wants: which players who
ALREADY made it have the closest year-by-year youth profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.build_features import feature_columns
from features.labels import CENSORED
from models.gbm import CatBoostBreakthrough

DEFAULT_TARGET = "pro_target"  # reached RPL / FNL / FNL-2

# numeric youth features used for the "similar player" nearest-neighbour search
PROFILE_FEATURES = [
    "youth_minutes_total",
    "youth_goals_total",
    "youth_ga_per90",
    "youth_minutes_trend",
    "best_level_pre_cutoff",
    "minutes_U15",
    "minutes_U17",
    "minutes_U19",
    "ga_per90_U17",
    "ga_per90_U19",
    "height_cm",
]

RAW_STAT_FIELDS = [
    "position",
    "position_detail",
    "academy_club",
    "height_cm",
    "youth_seasons",
    "youth_minutes_total",
    "youth_goals_total",
    "youth_ga_per90",
    "youth_minutes_trend",
    "best_level_pre_cutoff",
    "played_youth_league",
    "minutes_U15",
    "minutes_U17",
    "minutes_U19",
    "market_value_at_cutoff_eur",
]


def split_resolved_open(
    df: pd.DataFrame, target_col: str = DEFAULT_TARGET
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved = df[df[target_col] != CENSORED].copy()
    open_cohort = df[df[target_col] == CENSORED].copy()
    return resolved, open_cohort


def train_ranker(
    resolved: pd.DataFrame, target_col: str = DEFAULT_TARGET, params: dict | None = None
) -> CatBoostBreakthrough:
    feats = feature_columns(resolved)
    y = resolved[target_col].astype(int)
    return CatBoostBreakthrough(params=params or {"iterations": 300}).fit(resolved[feats], y)


def rank_prospects(
    df: pd.DataFrame,
    model: CatBoostBreakthrough | None = None,
    *,
    target_col: str = DEFAULT_TARGET,
    top: int | None = None,
) -> pd.DataFrame:
    resolved, open_cohort = split_resolved_open(df, target_col)
    if model is None:
        if resolved[target_col].nunique() < 2:
            raise ValueError("need both classes in the resolved set to train a ranker")
        model = train_ranker(resolved, target_col)

    scored = _score_frame(model, df, open_cohort)
    keep = [
        "player_id",
        "canonical_name",
        "birth_year",
        "academy_club",
        "position",
        "position_detail",
        "outcome_level",
        "breakthrough_score",
        "source",
        "proj_level",
        "youth_minutes_total",
        "youth_ga_per90",
        "best_level_pre_cutoff",
        "played_youth_league",
    ]
    out = scored[[c for c in keep if c in scored.columns]].sort_values(
        "breakthrough_score", ascending=False
    )
    return out.head(top) if top else out


def _score_frame(
    model: CatBoostBreakthrough, schema_df: pd.DataFrame, rows: pd.DataFrame
) -> pd.DataFrame:
    """Add ``breakthrough_score``: CatBoost for TM players, the 0–100 youth heuristic
    (``pers_score`` / 100) for ффспб rows the model has no real features for."""
    feats = feature_columns(schema_df)
    out = rows.copy()
    if not len(out):
        out["breakthrough_score"] = np.array([])
        return out
    out["breakthrough_score"] = model.predict_proba(out[feats])
    if "source" in out.columns and "pers_score" in out.columns:
        is_youth = out["source"].eq("ffspb") & out["pers_score"].notna()
        out.loc[is_youth, "breakthrough_score"] = (
            pd.to_numeric(out.loc[is_youth, "pers_score"], errors="coerce") / 100.0
        )
    return out


def rank_resolved(
    df: pd.DataFrame, model: CatBoostBreakthrough, *, target_col: str = DEFAULT_TARGET
) -> pd.DataFrame:
    """The other tab: players whose outcome is settled, with their fitted score."""
    resolved, _ = split_resolved_open(df, target_col)
    feats = feature_columns(df)
    out = resolved.copy()
    out["breakthrough_score"] = model.predict_proba(out[feats])
    out["outcome"] = out[target_col].astype(int)
    keep = [
        "player_id",
        "canonical_name",
        "birth_year",
        "academy_club",
        "position",
        "position_detail",
        "outcome_level",
        "outcome",
        "breakthrough_score",
        "youth_minutes_total",
        "youth_ga_per90",
        "best_level_pre_cutoff",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values(
        "breakthrough_score", ascending=False
    )


def explain_player(model: CatBoostBreakthrough, df: pd.DataFrame, player_id: str) -> pd.Series:
    """Signed SHAP contributions for one player (positive => pushes score up)."""
    from eval.shap_analysis import shap_values

    feats = feature_columns(df)
    row = df[df["player_id"] == player_id]
    if row.empty:
        raise KeyError(player_id)
    sv, Xp = shap_values(model, row[feats])
    return pd.Series(sv[0], index=Xp.columns).sort_values(key=np.abs, ascending=False)


def player_raw_stats(df: pd.DataFrame, player_id: str) -> dict:
    row = df[df["player_id"] == player_id]
    if row.empty:
        raise KeyError(player_id)
    r = row.iloc[0]
    return {f: r[f] for f in RAW_STAT_FIELDS if f in df.columns}


def _profile_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in PROFILE_FEATURES if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, cols


def similar_breakthrough_players(
    df: pd.DataFrame,
    player_id: str,
    *,
    k: int = 5,
    target_col: str = DEFAULT_TARGET,
    same_position: bool = True,
) -> pd.DataFrame:
    """Nearest breakthrough players (``target_col == 1``) to ``player_id`` in
    standardized youth-profile space. Returns name, birth_year, position, distance.
    """
    if player_id not in set(df["player_id"]):
        raise KeyError(player_id)
    Z, _ = _profile_matrix(df)
    idx = {pid: i for i, pid in enumerate(df["player_id"])}
    i = idx[player_id]

    made_it = df[target_col] == 1
    if same_position and "position" in df.columns:
        made_it &= df["position"] == df.iloc[i]["position"]
    cand = df.index[made_it & (df["player_id"] != player_id)]
    if len(cand) == 0:
        return pd.DataFrame(
            columns=["player_id", "canonical_name", "birth_year", "position", "distance"]
        )

    pos = [df.index.get_loc(j) for j in cand]
    d = np.linalg.norm(Z[pos] - Z[i], axis=1)
    order = np.argsort(d)[:k]
    res = df.loc[cand[order], ["player_id", "canonical_name", "birth_year", "position"]].copy()
    res["distance"] = np.round(d[order], 2)
    return res.reset_index(drop=True)
