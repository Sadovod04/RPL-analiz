import pandas as pd
import pytest

from features.time_cutoff import assert_within_cutoff, before_cutoff


def _df():
    return pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "age_at_season": [12.0, 15.5, 19.0, None],
            "minutes": [100, 900, 1500, 200],
        }
    )


def test_keeps_only_pre_cutoff_rows():
    out = before_cutoff(_df(), cutoff_age=19)
    assert list(out["age_at_season"]) == [12.0, 15.5]


def test_drops_missing_age():
    out = before_cutoff(_df(), cutoff_age=99)
    assert out["age_at_season"].isna().sum() == 0
    assert len(out) == 3


def test_missing_age_column_raises():
    with pytest.raises(KeyError):
        before_cutoff(pd.DataFrame({"minutes": [1]}), cutoff_age=19)


def test_assert_within_cutoff():
    ok = before_cutoff(_df(), cutoff_age=19)
    assert_within_cutoff(ok, cutoff_age=19)
    with pytest.raises(AssertionError):
        assert_within_cutoff(_df(), cutoff_age=19)
