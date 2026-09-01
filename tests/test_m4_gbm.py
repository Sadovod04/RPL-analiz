"""M4: CatBoost fit/predict, GroupKFold isolation, tune, SHAP."""

import numpy as np
import pandas as pd
import pytest

from models.gbm import CatBoostBreakthrough, group_kfold_scores, tune

rng = np.random.default_rng(1)


def _data(n=300, n_players=150):
    mins = rng.uniform(0, 3000, n)
    ga = rng.uniform(0, 1, n)
    logit = -2.5 + 0.0016 * mins + 1.0 * ga + rng.normal(0, 0.6, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "player_id": rng.integers(0, n_players, n).astype(str),
            "youth_minutes_total": mins,
            "youth_ga_per90": ga,
            "minutes_U17": rng.uniform(0, 1500, n),
            "minutes_U19": rng.uniform(0, 2500, n),
            "best_level_pre_cutoff": rng.integers(0, 4, n).astype(float),
            "height_cm": rng.integers(165, 195, n).astype(float),
            "position": rng.choice(["GK", "CB", "CM", "W", "ST"], n),
            "academy_club": rng.choice(["Zenit", "Spartak", "CSKA", None], n),
        }
    ), y


def test_fit_predict_range():
    X, y = _data()
    m = CatBoostBreakthrough(params={"iterations": 60}).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),)
    assert p.min() >= 0 and p.max() <= 1


def test_handles_nan_categoricals():
    X, y = _data()
    assert X["academy_club"].isna().any()
    CatBoostBreakthrough(params={"iterations": 40}).fit(X, y).predict_proba(X)


def test_group_kfold_scores_returns_per_fold():
    X, y = _data()
    scores = group_kfold_scores(X, y, X["player_id"], params={"iterations": 50}, n_splits=4)
    assert 1 <= len(scores) <= 4
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_tune_returns_param_dict():
    X, y = _data(200, 100)
    best = tune(X, y, X["player_id"], n_trials=3, n_splits=3)
    assert {"depth", "learning_rate", "l2_leaf_reg", "iterations"} <= best.keys()


def test_shap_importance_shape():
    from eval.shap_analysis import mean_abs_importance

    X, y = _data(200, 120)
    m = CatBoostBreakthrough(params={"iterations": 60}).fit(X, y)
    imp = mean_abs_importance(m, X)
    assert set(imp.index) == set(X.columns)
    assert (imp >= 0).all()


@pytest.mark.parametrize("bad_col", ["target", "rpl_minutes_ever", "reached_pro_level"])
def test_leakage_guard_covers_gbm_inputs(bad_col):
    from eval.leakage_check import LeakageError, assert_no_leakage

    X, _ = _data()
    with pytest.raises(LeakageError):
        assert_no_leakage([*X.columns, bad_col])
