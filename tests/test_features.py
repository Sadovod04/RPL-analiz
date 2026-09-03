"""M2: feature matrix + labels on synthetic multi-cohort data."""

import numpy as np
import pandas as pd
import pytest

from eval.leakage_check import LeakageError
from features.build_features import (
    _attach_recognition,
    _norm_academy,
    _trajectory_features,
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


# --- Phase A: trajectory + cohort features -----------------------------------
def _peer_seasons():
    """One league, two seasons, each with a trusted 17-y.o. / 20-match peer norm."""
    rows = [
        {
            "player_id": f"peer{i}_{season}",
            "season": season,
            "league": "Russian Youth League",
            "club": "x",
            "minutes": 1000,
            "matches": 20,
            "goals": 0,
            "assists": 0,
            "is_rpl": False,
            "age_at_season": 17.0,
        }
        for season in ("20/21", "21/22")
        for i in range(9)
    ]
    # prospect: two seasons in that league, young for it, then a minutes collapse
    rows += [
        {
            "player_id": "p",
            "season": "20/21",
            "league": "Russian Youth League",
            "club": "x",
            "minutes": 1200,
            "matches": 18,
            "goals": 5,
            "assists": 2,
            "is_rpl": False,
            "age_at_season": 15.0,
        },
        {
            "player_id": "p",
            "season": "21/22",
            "league": "Russian Youth League",
            "club": "x",
            "minutes": 150,
            "matches": 3,
            "goals": 0,
            "assists": 0,
            "is_rpl": False,
            "age_at_season": 16.0,
        },
    ]
    return pd.DataFrame(rows)


def test_trajectory_features_age_gap_matches_share_and_collapse():
    s = _peer_seasons()
    out = _trajectory_features(s, s).set_index("player_id").loc["p"]
    # 15 vs a 17.0 peer mean (prospect's own rows barely move it) -> ~ -2
    assert out["min_age_gap_vs_peers"] == pytest.approx(-2.0, abs=0.3)
    assert out["mean_age_gap_vs_peers"] < 0
    # season 2: 3 matches vs a 20-match full season -> low share
    assert out["matches_share_min"] == pytest.approx(0.15, abs=0.02)
    # 1200' -> 150' between consecutive pre-cutoff seasons
    assert out["minutes_dropoff_max"] == pytest.approx(0.875, abs=0.01)
    assert bool(out["had_minutes_collapse"]) is True


def test_trajectory_features_no_history_is_neutral():
    s = _peer_seasons()
    out = _trajectory_features(s, s).set_index("player_id")
    # a peer with a single steady season: no drop, no collapse
    assert out.loc["peer0_20/21", "minutes_dropoff_max"] == 0.0
    assert bool(out.loc["peer0_20/21", "had_minutes_collapse"]) is False


def test_trajectory_first_senior_age_and_role():
    rows = _peer_seasons().to_dict("records")
    # prospect "p" already has two youth seasons; add a senior FNL-2 season at 16
    # (a sub role: 400' over 10 games) and a starter youth season
    rows += [
        {"player_id": "p", "season": "20/21", "league": "Vtoraya Liga", "club": "x",
         "minutes": 400, "matches": 10, "goals": 0, "assists": 0, "is_rpl": False,
         "age_at_season": 16.0},
    ]
    df = pd.DataFrame(rows)
    out = _trajectory_features(df, df).set_index("player_id").loc["p"]
    assert out["played_senior_pre_cutoff"] == 1
    assert out["first_senior_age"] == 16.0
    # p total across kept rows: youth 1200'/18 + 150'/3 + senior 400'/10 -> ~50'/app
    assert 40 < out["min_per_appearance"] < 60
    # peers never touch a senior league
    assert _trajectory_features(df, df).set_index("player_id").loc["peer0_20/21",
                                                                   "played_senior_pre_cutoff"] == 0


def test_build_matrix_phase_a_columns_present_and_clean():
    players = _players().assign(birth_month=[3, 11, 7])  # Q1, Q4, Q3
    m = build_feature_matrix(players, _seasons(), cutoff_age=19, as_of_year=2024, cfg=CFG)
    mi = m.set_index("player_id")
    assert mi.loc["a", "cohort_year"] == 1995
    assert mi.loc["a", "birth_quarter"] == 1
    assert mi.loc["b", "birth_quarter"] == 4
    # born in March => relatively old within the calendar-year cohort => high frac
    assert mi.loc["a", "rel_age_frac"] > mi.loc["b", "rel_age_frac"]
    for col in ("cohort_year", "birth_quarter", "rel_age_frac", "min_age_gap_vs_peers",
                "minutes_dropoff_max", "had_minutes_collapse"):
        assert col in feature_columns(m)
    assert_matrix_is_clean(m)  # new columns must not trip the leakage guard


# --- Phase B: ru.wikipedia recognition -------------------------------------
def _recognition_matrix(wiki):
    base = pd.DataFrame({"player_id": ["a", "b", "c"]})
    return _attach_recognition(base, wiki, cutoff_age=19)


def test_recognition_none_is_all_zero():
    m = _recognition_matrix(None)
    for c in ("wiki_article_pre_cutoff", "wiki_youth_honours",
              "recognition_count", "pre_cutoff_recognition_score"):
        assert (m[c] == 0).all()
    assert (~m["any_recognition"]).all()


def test_recognition_pre_vs_post_cutoff():
    wiki = pd.DataFrame([
        # a: article at 17 (pre-cutoff) + two youth honours -> 3 + 2
        {"player_id": "a", "article_created_age": 17.0, "youth_honours_count": 2},
        # b: article at 20 (post-cutoff), no youth honours -> nothing counts
        {"player_id": "b", "article_created_age": 20.4, "youth_honours_count": 0},
        # c: no article row at all -> absent from wiki frame
    ])
    m = _recognition_matrix(wiki).set_index("player_id")
    assert m.loc["a", "wiki_article_pre_cutoff"] == 1
    assert m.loc["a", "wiki_youth_honours"] == 2
    assert m.loc["a", "pre_cutoff_recognition_score"] == 5
    assert bool(m.loc["a", "any_recognition"]) is True
    assert m.loc["b", "wiki_article_pre_cutoff"] == 0
    assert m.loc["b", "pre_cutoff_recognition_score"] == 0
    assert m.loc["c", "recognition_count"] == 0
    assert bool(m.loc["c", "any_recognition"]) is False


def test_norm_academy_groups_trivial_spellings():
    assert _norm_academy("Akademia Fakel Voronezh.") == "Akademia Fakel Voronezh"
    assert _norm_academy("Akademia Fakel Voronezh") == "Akademia Fakel Voronezh"
    assert _norm_academy("Dinamo-SUOR Stavropol\r\nAF Krasnodar") == "Dinamo-SUOR Stavropol"
    assert _norm_academy('  "Nevskiy front" St. Petersburg  ') == 'Nevskiy front" St. Petersburg'
    assert _norm_academy(None) is None
    assert _norm_academy(float("nan")) is None


def test_recognition_columns_pass_leakage_and_are_features():
    wiki = pd.DataFrame([{"player_id": "a", "article_created_age": 16.0,
                          "youth_honours_count": 2}])
    m = build_feature_matrix(_players(), _seasons(), wiki=wiki, cutoff_age=19,
                             as_of_year=2024, cfg=CFG)
    fcols = feature_columns(m)
    for c in ("pre_cutoff_recognition_score", "recognition_count", "any_recognition",
              "wiki_article_pre_cutoff", "wiki_youth_honours"):
        assert c in fcols
    for raw in ("wiki_title", "article_created_age", "nt_youth_levels", "honours_years"):
        assert raw not in fcols
    assert_matrix_is_clean(m)
    assert m.set_index("player_id").loc["a", "pre_cutoff_recognition_score"] == 3 + 2
