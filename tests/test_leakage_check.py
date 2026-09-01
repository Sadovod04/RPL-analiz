import pytest

from eval.leakage_check import LeakageError, assert_no_leakage, find_leaks


def test_clean_matrix_passes():
    cols = ["age_at_cutoff", "minutes_u17", "goals_per90_u19", "position_ST", "club"]
    assert_no_leakage(cols)


def test_exact_forbidden_detected():
    assert "current_club" in find_leaks(["current_club", "minutes_u17"])


def test_substring_forbidden_detected():
    leaks = find_leaks(["minutes_next_season", "senior_caps_future", "target_binary"])
    assert set(leaks) == {"minutes_next_season", "senior_caps_future", "target_binary"}


def test_assert_raises():
    with pytest.raises(LeakageError):
        assert_no_leakage(["minutes_u17", "outcome_rpl"])


def test_extra_forbidden():
    with pytest.raises(LeakageError):
        assert_no_leakage(["scout_note"], extra_forbidden=["scout_note"])
