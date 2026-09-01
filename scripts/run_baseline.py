"""M3: baseline vs naive scout on the feature parquet.

    uv run python scripts/run_baseline.py [path.parquet]

Temporal split by birth-year cohort; reports PR-AUC / ROC-AUC / Recall@Top-K /
Brier for the logistic-regression baseline and the market-value naive scout.
With the convenience demo sample (few negatives) the numbers are degenerate —
this is a wiring check; real evaluation needs the full run_ingest dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from eval.metrics import calibration_table, evaluate_binary
from features.split import temporal_split
from models.baseline import LogRegBaseline, naive_scout_scores
from settings import load_settings


def main(path: str | None = None) -> None:
    proc = Path(load_settings()["paths"]["data_processed"])
    src = (
        Path(path)
        if path
        else (
            proc / "features.parquet"
            if (proc / "features.parquet").exists()
            else proc / "features_demo.parquet"
        )
    )
    df = pd.read_parquet(src)
    print(f"loaded {src.name}  {df.shape}")

    test_from = load_settings()["split"]["test_cohort_from"]
    train, test = temporal_split(df, test_cohort_from=test_from)
    print(
        f"temporal split @ birth_year {test_from}:  train={len(train)}  test={len(test)}  "
        f"test base rate={test['target'].mean():.1%}"
    )
    if test["target"].nunique() < 2 or train["target"].nunique() < 2:
        print("!! not enough class variety for a meaningful eval (expected with the demo sample)")

    model = LogRegBaseline().fit(train, train["target"])
    p_model = model.predict_proba(test)
    s_scout = naive_scout_scores(test)

    k = min(20, len(test))
    print("\n            model (logreg)   naive scout (market value)")
    m_res = evaluate_binary(test["target"], p_model, k=k)
    s_res = evaluate_binary(test["target"], s_scout, k=k)
    for key in ("pr_auc", "roc_auc", f"recall_at_{k}", "brier"):
        mv = m_res.get(key, float("nan"))
        sv = s_res.get(key, float("nan"))
        print(f"  {key:14} {mv:>8.3f}        {sv:>8.3f}")

    print("\ncalibration (model):  bin  n  mean_pred  frac_pos")
    for lo, hi, n, mp, fp in calibration_table(test["target"], p_model, n_bins=5):
        print(f"  [{lo:.1f},{hi:.1f})  n={n:<4} pred={mp:.2f}  obs={fp:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
