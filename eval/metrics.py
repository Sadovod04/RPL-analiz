"""Evaluation metrics (SPEC §10).

PR-AUC (primary), Brier score, calibration curve, and Recall@Top-K — the
scouting-relevant metric: of the real future RPL players in a candidate pool,
how many land in the top K ranked by the model.
"""

from __future__ import annotations

import numpy as np


def recall_at_top_k(y_true: np.ndarray, scores: np.ndarray, k: int = 20) -> float:
    """Share of positives captured in the k highest-scored candidates."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    top = np.argsort(-scores)[:k]
    return float(y_true[top].sum()) / n_pos


# pr_auc, brier, calibration_curve are thin wrappers over sklearn — added in M3
def pr_auc(y_true, scores) -> float:  # noqa: ANN001
    raise NotImplementedError("M3")


def brier(y_true, probs) -> float:  # noqa: ANN001
    raise NotImplementedError("M3")
