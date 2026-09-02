"""M6/M7: prospect ranking, analogy, raw stats (dashboard-independent)."""

import numpy as np
import pandas as pd

from app.ranking import (
    explain_player,
    player_raw_stats,
    rank_prospects,
    rank_resolved,
    similar_breakthrough_players,
    split_resolved_open,
    train_ranker,
)
from features.labels import CENSORED

rng = np.random.default_rng(3)
TARGET = "pro_target"


def _matrix(n=260):
    mins = rng.uniform(0, 3000, n)
    ga = rng.uniform(0, 1, n)
    score = 0.0004 * mins + 1.5 * ga + rng.normal(0, 0.3, n)
    birth_year = rng.integers(1996, 2011, n)
    resolved = birth_year <= 2003
    made_it = (score > np.median(score)).astype(int)
    pos = rng.choice(["GK", "CB", "CM", "W", "ST"], n)
    detail = np.where(
        pos == "W", "Left Winger", np.where(pos == "CM", "Central Midfield", "Centre-Back")
    )
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(n)],
            "canonical_name": [f"Player {i}" for i in range(n)],
            "birth_year": birth_year,
            "academy_club": rng.choice(["Zenit", "Spartak", "CSKA", None], n),
            "position": pos,
            "position_detail": detail,
            "target": np.where(resolved, made_it, CENSORED),
            "pro_target": np.where(resolved, made_it, CENSORED),
            "youth_minutes_total": mins,
            "youth_goals_total": ga * mins / 90,
            "youth_ga_per90": ga,
            "youth_minutes_trend": rng.normal(0, 40, n),
            "youth_seasons": rng.integers(1, 6, n),
            "best_level_pre_cutoff": rng.integers(0, 4, n).astype(float),
            "minutes_U15": rng.uniform(0, 400, n),
            "minutes_U17": rng.uniform(0, 1400, n),
            "minutes_U19": rng.uniform(0, 2400, n),
            "ga_per90_U17": rng.uniform(0, 1, n),
            "ga_per90_U19": rng.uniform(0, 1, n),
            "played_youth_league": rng.random(n) > 0.5,
            "height_cm": rng.integers(165, 195, n).astype(float),
            "academy_conversion_rate": rng.random(n),
            "market_value_at_cutoff_eur": rng.uniform(0, 3e6, n),
            "is_foreigner": rng.choice([True, False, None], n),
        }
    )


def test_split_resolved_open():
    df = _matrix()
    resolved, open_c = split_resolved_open(df, TARGET)
    assert (resolved[TARGET] != CENSORED).all()
    assert (open_c[TARGET] == CENSORED).all()
    assert len(resolved) + len(open_c) == len(df)


def test_rank_prospects_orders_and_scopes():
    df = _matrix()
    ranked = rank_prospects(df, target_col=TARGET, top=15)
    assert len(ranked) == 15
    assert ranked["breakthrough_score"].is_monotonic_decreasing
    open_ids = set(df.loc[df[TARGET] == CENSORED, "player_id"])
    assert set(ranked["player_id"]) <= open_ids
    assert ranked["breakthrough_score"].between(0, 1).all()


def test_rank_resolved_has_outcome_and_score():
    df = _matrix()
    model = train_ranker(split_resolved_open(df, TARGET)[0], TARGET)
    res = rank_resolved(df, model, target_col=TARGET)
    assert set(res["outcome"].unique()) <= {0, 1}
    assert res["breakthrough_score"].between(0, 1).all()


def test_explain_player_signed_contributions():
    df = _matrix()
    model = train_ranker(split_resolved_open(df, TARGET)[0], TARGET)
    pid = rank_prospects(df, model=model, target_col=TARGET).iloc[0]["player_id"]
    contrib = explain_player(model, df, pid)
    assert len(contrib) >= 5
    assert (contrib.abs().values == contrib.abs().sort_values(ascending=False).values).all()


def test_player_raw_stats_keys():
    df = _matrix()
    raw = player_raw_stats(df, "p0")
    assert {"position", "youth_minutes_total", "minutes_U17"} <= raw.keys()


def test_similar_breakthrough_players_same_position():
    df = _matrix()
    prospect = df.loc[df[TARGET] == CENSORED].iloc[0]["player_id"]
    sim = similar_breakthrough_players(df, prospect, k=5, target_col=TARGET)
    if len(sim):
        assert (sim["distance"].values == np.sort(sim["distance"].values)).all()
        target_pos = df.loc[df["player_id"] == prospect, "position"].iloc[0]
        assert (sim["position"] == target_pos).all()
