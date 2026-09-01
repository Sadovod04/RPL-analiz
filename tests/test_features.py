"""M2: feature matrix + labels on synthetic multi-cohort data."""

import numpy as np
import pandas as pd
import pytest

from eval.leakage_check import LeakageError
from features.build_features import (
    age_bucket,
    assert_matrix_is_clean,
    build_feature_matrix,
    feature_columns,
)
from features.labels import CENSORED, LabelConfig, attach_labels

CFG = LabelConfig(rpl_minutes_threshold=200, settled_age=26)


def _players():
    return pd.DataFrame(
        [
            # broke through, old cohort
            {
                "player_id": "a",
                "canonical_name": "A",
                "birth_year": 1995,
                "position": "W",
                "height_cm": 180,
                "is_foreigner": None,
                "academy_club": "Zenit",
            },
            # did not, settled (old)
            {
                "player_id": "b",
                "canonical_name": "B",
                "birth_year": 1996,
                "position": "CB",
                "height_cm": 185,
                "is_foreigner": None,
                "academy_club": "Zenit",
            },
            # young, still open -> censored
            {
                "player_id": "c",
                "canonical_name": "C",
                "birth_year": 2006,
                "position": "ST",
                "height_cm": 178,
                "is_foreigner": None,
                "academy_club": "Zenit",
            },
        ]
    )


def _seasons():
    return pd.DataFrame(
        [
            # a: youth then RPL
            {
                "player_id": "a",
                "season": "12/13",
                "league": "Russian Youth League",
                "club": "964",
                "minutes": 900,
                "matches": 12,
                "goals": 4,
                "assists": 3,
                "is_rpl": False,
                "age_at_season": 17.0,
            },
            {
                "player_id": "a",
                "season": "14/15",
                "league": "Premier Liga",
                "club": "964",
                "minutes": 1500,
                "matches": 20,
                "goals": 6,
                "assists": 5,
                "is_rpl": True,
                "age_at_season": 19.5,
            },
            # b: only youth, never senior
            {
                "player_id": "b",
                "season": "13/14",
                "league": "Russian Youth League",
                "club": "964",
                "minutes": 300,
                "matches": 8,
                "goals": 0,
                "assists": 1,
                "is_rpl": False,
                "age_at_season": 16.0,
            },
            # c: promising youth, no senior yet
            {
                "player_id": "c",
                "season": "22/23",
                "league": "Russian Youth League",
                "club": "964",
                "minutes": 1200,
                "matches": 18,
                "goals": 10,
                "assists": 4,
                "is_rpl": False,
                "age_at_season": 16.5,
            },
        ]
    )


def test_age_bucket():
    assert age_bucket(12.0) == "U13"
    assert age_bucket(14.0) == "U15"
    assert age_bucket(16.9) == "U17"
    assert age_bucket(18.0) == "U19"
    assert age_bucket(20.0) == "U21"
    assert age_bucket(21.0) is None
    assert age_bucket(np.nan) is None


def test_attach_labels_binary_ordinal_survival():
    out = attach_labels(_players(), _seasons(), as_of_year=2024, cfg=CFG).set_index("player_id")
    assert out.loc["a", "target"] == 1
    assert out.loc["b", "target"] == 0
    assert out.loc["c", "target"] == CENSORED
    assert out.loc["a", "ordinal_target"] == "rpl"
    assert out.loc["b", "ordinal_target"] == "none"
    assert out.loc["a", "event_observed"] == 1
    assert out.loc["a", "duration"] == pytest.approx(19.5)
    assert out.loc["c", "event_observed"] == 0  # censored at current age


def test_build_matrix_time_cutoff_and_shape():
    m = build_feature_matrix(_players(), _seasons(), cutoff_age=19, as_of_year=2024, cfg=CFG)
    assert len(m) == 3
    mi = m.set_index("player_id")
    # a's RPL season (age 19.5) is AFTER cutoff -> excluded; the age-17 youth season stays
    assert mi.loc["a", "youth_minutes_total"] == 900
    assert mi.loc["a", "minutes_U19"] == 900  # age 17.0 -> U19 bucket [17,19)
    assert mi.loc["a", "minutes_U17"] == 0
    # c has strong youth output
    assert mi.loc["c", "youth_goals_total"] == 10
    assert mi.loc["c", "played_youth_league"]


def test_academy_conversion_rate_is_time_aware():
    m = build_feature_matrix(_players(), _seasons(), cutoff_age=19, as_of_year=2024, cfg=CFG)
    mi = m.set_index("player_id")
    # oldest cohort (a, 1995) has no prior history -> NaN
    assert pd.isna(mi.loc["a", "academy_conversion_rate"])
    # c (2006) sees a(1) and b(0) before it -> 0.5
    assert mi.loc["c", "academy_conversion_rate"] == pytest.approx(0.5)


def test_feature_columns_exclude_labels_and_pass_leakage():
    m = build_feature_matrix(_players(), _seasons(), cutoff_age=19, as_of_year=2024, cfg=CFG)
    fcols = feature_columns(m)
    for banned in ("target", "ordinal_target", "rpl_minutes_ever", "current_age", "birth_year"):
        assert banned not in fcols
    assert_matrix_is_clean(m)  # must not raise


def test_leakage_guard_trips_on_bad_column():
    m = build_feature_matrix(_players(), _seasons(), cutoff_age=19, as_of_year=2024, cfg=CFG)
    m["senior_national_team_caps"] = 5
    with pytest.raises(LeakageError):
        assert_matrix_is_clean(m)
