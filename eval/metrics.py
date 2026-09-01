"""Evaluation metrics (SPEC §10).

Primary: PR-AUC (imbalance). Also Brier + calibration curve (the product outputs
a *probability*, not just a ranking) and Recall@Top-K — of the real future RPL
players in a candidate pool, how many land in the top K by model score.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def recall_at_top_k(y_true, scores, k: int = 20) -> float:
    """Share of positives captured in the k highest-scored candidates."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    top = np.argsort(-scores)[:k]
    return float(y_true[top].sum()) / n_pos


def pr_auc(y_true, scores) -> float:
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))


def brier(y_true, probs) -> float:
    return float(brier_score_loss(y_true, probs))


def calibration_table(y_true, probs, n_bins: int = 10):
    """-> list of (bin_lo, bin_hi, n, mean_pred, frac_pos). Non-empty bins only."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if m.any():
            out.append(
                (
                    float(lo),
                    float(hi),
                    int(m.sum()),
                    float(probs[m].mean()),
                    float(y_true[m].mean()),
                )
            )
    return out


def evaluate_binary(y_true, scores, probs=None, *, k: int = 20) -> dict:
    """Rank metrics from ``scores``; calibration from ``probs`` (defaults to scores)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    probs = scores if probs is None else np.asarray(probs, dtype=float)
    res = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "pr_auc": pr_auc(y_true, scores),
        f"recall_at_{k}": recall_at_top_k(y_true, scores, k),
    }
    if len(np.unique(y_true)) == 2:
        res["roc_auc"] = roc_auc(y_true, scores)
    if probs.min() >= 0.0 and probs.max() <= 1.0:
        res["brier"] = brier(y_true, probs)
    return res
