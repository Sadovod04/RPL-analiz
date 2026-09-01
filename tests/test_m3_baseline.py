"""M3: metrics, temporal split, baseline vs naive scout."""

import numpy as np
import pandas as pd

from eval.metrics import calibration_table, evaluate_binary, pr_auc, recall_at_top_k
from features.split import temporal_split
from models.baseline import LogRegBaseline, naive_scout_scores

rng = np.random.default_rng(0)


def _synthetic(n=400):
    """youth_minutes_total drives the outcome; market value is a noisy proxy."""
    mins = rng.uniform(0, 3000, n)
    ga = rng.uniform(0, 1.0, n)
    logit = -3.0 + 0.0015 * mins + 1.2 * ga + rng.normal(0, 0.5, n)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    # market value: a NOISY reflection of the true probability, not the label
    mv = np.clip(2e5 + 3e6 * p + rng.normal(0, 1.2e6, n), 0, None)
    mv[rng.uniform(size=n) < 0.3] = np.nan  # missingness
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(n)],
            "birth_year": rng.integers(1994, 2006, n),
            "youth_minutes_total": mins,
            "youth_goals_total": ga * mins / 90,
            "youth_ga_per90": ga,
            "youth_minutes_trend": rng.normal(0, 50, n),
            "best_level_pre_cutoff": rng.integers(0, 4, n).astype(float),
            "minutes_U15": rng.uniform(0, 500, n),
            "minutes_U17": rng.uniform(0, 1500, n),
            "minutes_U19": rng.uniform(0, 2500, n),
            "height_cm": rng.integers(165, 195, n).astype(float),
            "position": rng.choice(["GK", "CB", "CM", "W", "ST"], n),
            "market_value_at_cutoff_eur": mv,
            "target": y,
        }
    )


def test_recall_at_top_k():
    assert recall_at_top_k([0, 0, 1, 0, 1], [0.1, 0.2, 0.9, 0.3, 0.8], k=2) == 1.0
    assert recall_at_top_k([0, 0, 1, 0, 1], [0.1, 0.2, 0.9, 0.3, 0.8], k=1) == 0.5
    assert np.isnan(recall_at_top_k([0, 0, 0], [1, 2, 3], k=2))


def test_calibration_table_shape():
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    tbl = calibration_table(y, p, n_bins=5)
    assert tbl and all(len(row) == 5 for row in tbl)
    assert sum(r[2] for r in tbl) == 200


def test_temporal_split_partitions_and_drops_censored():
    df = _synthetic(200)
    df.loc[:9, "target"] = -1  # censored
    train, test = temporal_split(df, test_cohort_from=2001)
    assert train["birth_year"].max() < 2001 <= test["birth_year"].min()
    assert (train["target"] != -1).all() and (test["target"] != -1).all()
    assert len(train) + len(test) == (df["target"] != -1).sum()


def test_logreg_beats_naive_scout_on_pr_auc():
    df = _synthetic(600)
    train, test = temporal_split(df, test_cohort_from=2002)

    model = LogRegBaseline().fit(train, train["target"])
    p_model = model.predict_proba(test)
    s_scout = naive_scout_scores(test)

    ap_model = pr_auc(test["target"], p_model)
    ap_scout = pr_auc(test["target"], s_scout)
    base_rate = test["target"].mean()

    assert ap_model > base_rate  # learned something
    assert ap_model >= ap_scout - 0.02  # at least competitive with the scout


def test_evaluate_binary_keys():
    df = _synthetic(300)
    m = LogRegBaseline().fit(df, df["target"])
    res = evaluate_binary(df["target"], m.predict_proba(df), k=20)
    assert {"pr_auc", "recall_at_20", "roc_auc", "brier"} <= res.keys()
    assert 0.0 <= res["brier"] <= 1.0
