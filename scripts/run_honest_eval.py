"""Phase 3 — honest evaluation of the breakthrough model (SPEC §10).

    uv run python scripts/run_honest_eval.py [path.parquet] [--target target] [--boot 500]

Not about squeezing the score — about how much to trust it:

1. Recall@Top-K **per test cohort year** — of the players born in year Y who
   later reached the target, how many are in that year's top-K by model score.
   This is the real use ("give me this year's top 20 prospects"), and the pooled
   number hides that some years are easy and some impossible.
2. Calibration — reliability table + ECE. The product emits a probability.
3. Bootstrap 90% CIs on PR-AUC / ROC-AUC / Recall@20 — the positives are few
   (~130 in the RPL-target test set); a point estimate alone is misleading.
4. `cohort_year` probe — refit without it and compare, plus its permutation
   importance. It dominates SHAP on the RPL target; is that "less time elapsed"
   (legitimate) or reading the temporal split?

CatBoost runs on default params here (tuning is `run_gbm.py`'s job).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from eval.metrics import calibration_table, pr_auc, recall_at_top_k, roc_auc
from features.build_features import feature_columns
from features.split import temporal_split
from models.baseline import naive_scout_scores
from models.gbm import CatBoostBreakthrough
from settings import load_settings

RNG = np.random.default_rng(42)


def _load(path: str | None) -> tuple[pd.DataFrame, str]:
    proc = Path(load_settings()["paths"]["data_processed"])
    src = Path(path) if path else proc / "features.parquet"
    if not src.exists():
        src = proc / "features_demo.parquet"
    return pd.read_parquet(src), src.name


def _ece(y_true, probs, n_bins: int = 10) -> float:
    tbl = calibration_table(y_true, probs, n_bins=n_bins)
    n = len(y_true)
    return float(sum(cnt / n * abs(mp - fp) for _, _, cnt, mp, fp in tbl))


def _bootstrap_ci(y_true, scores, fn, n: int, lo=5, hi=95) -> tuple[float, float, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            continue
        vals.append(fn(y_true[b], scores[b]))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def _fit_predict(train, test, feats, tgt):
    m = CatBoostBreakthrough().fit(train[feats], train[tgt])
    return m, m.predict_proba(test[feats])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--target", default="target")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args(argv)

    df, name = _load(args.path)
    tgt = args.target if args.target in df.columns else "target"
    feats = feature_columns(df)
    train, test = temporal_split(df, target_col=tgt)
    yte = test[tgt].to_numpy(int)
    print(f"\n{name}: target={tgt}  train={len(train)}  test={len(test)}  "
          f"feats={len(feats)}  test positives={yte.sum()}  base rate={yte.mean():.1%}\n")

    model, p = _fit_predict(train, test, feats, tgt)
    scout = naive_scout_scores(test).to_numpy()

    # 1. Recall@Top-K per test cohort year -----------------------------------
    print(f"--- Recall@Top-{args.k} per cohort year (candidate pool = that year) ---")
    print(f"{'year':>6} {'n':>5} {'pos':>4} {'model':>7} {'scout':>7}")
    trows = []
    for yr, g in test.assign(_p=p, _s=scout).groupby("cohort_year"):
        yv = g[tgt].to_numpy(int)
        if yv.sum() == 0:
            continue
        rm = recall_at_top_k(yv, g["_p"].to_numpy(), args.k)
        rs = recall_at_top_k(yv, g["_s"].to_numpy(), args.k)
        trows.append((int(yr), len(g), int(yv.sum()), rm, rs))
        print(f"{int(yr):>6} {len(g):>5} {int(yv.sum()):>4} {rm:>7.2f} {rs:>7.2f}")
    if trows:
        wm = np.average([r[3] for r in trows], weights=[r[2] for r in trows])
        ws = np.average([r[4] for r in trows], weights=[r[2] for r in trows])
        print(f"{'pooled':>6} {'':>5} {sum(r[2] for r in trows):>4} {wm:>7.2f} {ws:>7.2f}"
              f"   (positive-weighted mean of per-year recall)")

    # 2. Calibration --------------------------------------------------------
    print("\n--- Calibration (reliability) ---")
    print(f"{'bin':>12} {'n':>5} {'pred':>7} {'obs':>7}")
    for lo, hi, cnt, mp, fp in calibration_table(yte, p, n_bins=10):
        print(f"[{lo:.1f},{hi:.1f}){'':>3} {cnt:>5} {mp:>7.2f} {fp:>7.2f}")
    print(f"ECE = {_ece(yte, p):.3f}   Brier = {np.mean((p - yte) ** 2):.3f}")

    # 3. Bootstrap CIs ----------------------------------------------------
    print(f"\n--- Bootstrap 90% CI ({args.boot} resamples) ---")
    for label, fn, sc in [
        ("PR-AUC   model", pr_auc, p),
        ("PR-AUC   scout", pr_auc, scout),
        ("ROC-AUC  model", roc_auc, p),
        (f"Recall@{args.k} model", lambda a, b: recall_at_top_k(a, b, args.k), p),
    ]:
        mean, lo, hi = _bootstrap_ci(yte, sc, fn, args.boot)
        print(f"{label:>16}: {mean:.3f}  [{lo:.3f}, {hi:.3f}]")

    # 4. cohort_year probe ---------------------------------------------------
    if "cohort_year" in feats:
        print("\n--- cohort_year probe ---")
        feats_no = [c for c in feats if c != "cohort_year"]
        _, p_no = _fit_predict(train, test, feats_no, tgt)
        print(f"{'':>18}{'with':>8}{'without':>9}")
        print(f"{'PR-AUC':>18}{pr_auc(yte, p):>8.3f}{pr_auc(yte, p_no):>9.3f}")
        print(f"{'ROC-AUC':>18}{roc_auc(yte, p):>8.3f}{roc_auc(yte, p_no):>9.3f}")
        print(f"{'Recall@'+str(args.k):>18}{recall_at_top_k(yte, p, args.k):>8.3f}"
              f"{recall_at_top_k(yte, p_no, args.k):>9.3f}")
        # permutation importance of cohort_year (drop in PR-AUC when shuffled)
        base = pr_auc(yte, p)
        drops = []
        Xp = test[feats].copy()
        for _ in range(20):
            Xp["cohort_year"] = RNG.permutation(test["cohort_year"].to_numpy())
            drops.append(base - pr_auc(yte, model.predict_proba(Xp)))
        print(f"permutation importance (mean PR-AUC drop when cohort_year shuffled): "
              f"{np.mean(drops):+.3f}")

    print()


if __name__ == "__main__":
    main()
