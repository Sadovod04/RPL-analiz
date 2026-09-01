"""M5: Cox survival model on synthetic right-censored data."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from models.survival import CoxBreakthrough, survival_frame

rng = np.random.default_rng(2)
_HAS_SKSURV = importlib.util.find_spec("sksurv") is not None


def _features(n=250):
    talent = rng.normal(0, 1, n)
    # more talent -> earlier debut; censor those who "haven't debuted" by their current age
    debut_age = 24 - 2.5 * talent + rng.normal(0, 1.5, n)
    current_age = rng.uniform(17, 30, n)
    event = (debut_age <= current_age).astype(int)
    duration = np.where(event == 1, np.clip(debut_age, 15, None), current_age)
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(n)],
            "birth_year": rng.integers(1994, 2005, n),
            "talent_proxy": talent,
            "youth_minutes_total": np.clip(1500 + 400 * talent + rng.normal(0, 300, n), 0, None),
            "position": rng.choice(["GK", "CM", "ST"], n),
            "duration": duration,
            "event_observed": event,
        }
    )


FEATS = ["talent_proxy", "youth_minutes_total", "position"]


def test_survival_frame_is_model_ready():
    fr = survival_frame(_features(), FEATS)
    assert {"duration", "event_observed"} <= set(fr.columns)
    assert fr.isna().sum().sum() == 0
    assert fr.select_dtypes(exclude="number").empty  # all numeric


def test_cox_fits_and_ranks():
    df = _features(300)
    fr = survival_frame(df, FEATS)
    cox = CoxBreakthrough().fit(fr)
    assert cox.concordance_index_ > 0.55  # better than random on this signal


def test_probability_by_age_is_monotone():
    df = _features(200)
    fr = survival_frame(df, FEATS)
    cox = CoxBreakthrough().fit(fr)
    p20 = cox.probability_by_age(fr, 20)
    p25 = cox.probability_by_age(fr, 25)
    assert np.all(p25 >= p20 - 1e-9)  # P(debut by 25) >= P(debut by 20)
    assert p20.min() >= 0 and p25.max() <= 1


def test_more_talent_earlier_debut_probability():
    df = _features(300)
    fr = survival_frame(df, FEATS)
    cox = CoxBreakthrough().fit(fr)
    p22 = cox.probability_by_age(fr, 22)
    hi = df["talent_proxy"] > df["talent_proxy"].quantile(0.75)
    lo = df["talent_proxy"] < df["talent_proxy"].quantile(0.25)
    assert p22[hi.to_numpy()].mean() > p22[lo.to_numpy()].mean()


@pytest.mark.skipif(not _HAS_SKSURV, reason="scikit-survival not installed (survival extra)")
def test_rsf_optional():
    from models.survival import RSFBreakthrough

    fr = survival_frame(_features(200), FEATS)
    rsf = RSFBreakthrough(n_estimators=50).fit(fr)
    p = rsf.probability_by_age(fr, 23)
    assert p.min() >= 0 and p.max() <= 1
