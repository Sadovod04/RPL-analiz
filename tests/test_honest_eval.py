"""Phase 3: honest-eval helpers (pure)."""

import numpy as np

from scripts.run_honest_eval import _bootstrap_ci, _ece


def test_ece_zero_for_perfectly_calibrated():
    # prob p, outcome Bernoulli(p) on a big sample -> ECE ~ 0
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert _ece(y, p) < 0.03


def test_ece_large_when_overconfident():
    y = np.zeros(1000, dtype=int)
    p = np.full(1000, 0.9)  # says 90%, truth 0%
    assert _ece(y, p) > 0.8


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = y + rng.normal(0, 0.5, 400)  # scores correlate with y
    from eval.metrics import roc_auc

    point = roc_auc(y, s)
    mean, lo, hi = _bootstrap_ci(y, s, roc_auc, n=300)
    assert lo < point < hi
    assert lo < mean < hi
