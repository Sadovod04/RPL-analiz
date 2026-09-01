"""M6: prospect ranking logic (dashboard-independent)."""

import numpy as np
import pandas as pd

from app.ranking import explain_player, rank_prospects, split_resolved_open, train_ranker
from features.labels import CENSORED

rng = np.random.default_rng(3)


def _matrix(n=240):
    mins = rng.uniform(0, 3000, n)
    ga = rng.uniform(0, 1, n)
    score = 0.0004 * mins + 1.5 * ga + rng.normal(0, 0.3, n)
    birth_year = rng.integers(1996, 2010, n)
    resolved = birth_year <= 2004
    target = np.where(resolved, (score > np.median(score)).astype(int), CENSORED)
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(n)],
            "canonical_name": [f"Player {i}" for i in range(n)],
            "birth_year": birth_year,
            "academy_club": rng.choice(["Zenit", "Spartak", "CSKA", None], n),
            "position": rng.choice(["GK", "CB", "CM", "W", "ST"], n),
            "target": target,
            "youth_minutes_total": mins,
            "youth_goals_total": ga * mins / 90,
            "youth_ga_per90": ga,
            "youth_minutes_trend": rng.normal(0, 40, n),
            "best_level_pre_cutoff": rng.integers(0, 4, n).astype(float),
            "minutes_U15": rng.uniform(0, 400, n),
            "minutes_U17": rng.uniform(0, 1400, n),
            "minutes_U19": rng.uniform(0, 2400, n),
            "played_youth_league": rng.random(n) > 0.5,
            "height_cm": rng.integers(165, 195, n).astype(float),
            "academy_conversion_rate": rng.random(n),
            "market_value_at_cutoff_eur": rng.uniform(0, 3e6, n),
            "is_foreigner": rng.choice([True, False, None], n),
        }
    )


def test_split_resolved_open():
    df = _matrix()
    resolved, open_c = split_resolved_open(df)
    assert (resolved["target"] != CENSORED).all()
    assert (open_c["target"] == CENSORED).all()
    assert len(resolved) + len(open_c) == len(df)


def test_rank_prospects_orders_and_scopes():
    df = _matrix()
    ranked = rank_prospects(df, top=15)
    assert len(ranked) == 15
    assert ranked["breakthrough_score"].is_monotonic_decreasing
    # only open-cohort players are ranked
    open_ids = set(df.loc[df["target"] == CENSORED, "player_id"])
    assert set(ranked["player_id"]) <= open_ids
    assert ranked["breakthrough_score"].between(0, 1).all()


def test_explain_player_signed_contributions():
    df = _matrix()
    model = train_ranker(split_resolved_open(df)[0])
    pid = rank_prospects(df, model=model).iloc[0]["player_id"]
    contrib = explain_player(model, df, pid)
    assert len(contrib) >= 5
    assert (contrib.abs().sort_values(ascending=False).values == contrib.abs().values).all()
