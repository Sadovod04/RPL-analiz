"""M4: CatBoost end-to-end with MLflow tracking.

    uv run python scripts/run_gbm.py [path.parquet] [--trials N]

Temporal split -> leakage check -> Optuna tune (GroupKFold by player_id) ->
fit final -> eval vs logreg baseline & naive scout on the held-out cohort ->
SHAP importance. Everything logged to the ``rpl-breakthrough`` MLflow experiment
(file backend under ./mlruns).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from eval.metrics import evaluate_binary
from eval.shap_analysis import mean_abs_importance
from features.build_features import assert_matrix_is_clean, feature_columns
from features.split import temporal_split
from models.baseline import LogRegBaseline, naive_scout_scores
from models.gbm import CatBoostBreakthrough, group_kfold_scores, tune
from settings import load_settings

EXPERIMENT = "rpl-breakthrough"


def _load(path: str | None) -> tuple[pd.DataFrame, str]:
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
    return pd.read_parquet(src), src.name


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--trials", type=int, default=25)
    args = ap.parse_args(argv)

    df, name = _load(args.path)
    feats = feature_columns(df)
    assert_matrix_is_clean(df)

    test_from = load_settings()["split"]["test_cohort_from"]
    train, test = temporal_split(df, test_cohort_from=test_from)
    print(
        f"{name}: train={len(train)} test={len(test)} feats={len(feats)} "
        f"test base rate={test['target'].mean():.1%}"
    )

    degenerate = train["target"].nunique() < 2 or test["target"].nunique() < 2
    if degenerate:
        print("!! degenerate split (demo sample) — running for wiring only")

    Xtr, ytr, gtr = train[feats], train["target"], train["player_id"]

    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{Path('mlruns/mlflow.db').resolve()}")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="catboost"):
        mlflow.log_params(
            {
                "dataset": name,
                "n_train": len(train),
                "n_test": len(test),
                "test_cohort_from": test_from,
                "n_features": len(feats),
            }
        )

        best = {}
        if not degenerate:
            best = tune(Xtr, ytr, gtr, n_trials=args.trials)
            cv = group_kfold_scores(Xtr, ytr, gtr, params=best)
            mlflow.log_params({f"best_{k}": v for k, v in best.items()})
            if cv:
                mlflow.log_metric("cv_pr_auc_mean", float(pd.Series(cv).mean()))
                print(f"CV PR-AUC (GroupKFold): {pd.Series(cv).round(3).tolist()}")

        gbm = CatBoostBreakthrough(params=best).fit(Xtr, ytr)
        p_gbm = gbm.predict_proba(test[feats])

        base = LogRegBaseline().fit(train, ytr)
        results = {
            "catboost": evaluate_binary(test["target"], p_gbm, k=min(20, len(test))),
            "logreg": evaluate_binary(
                test["target"], base.predict_proba(test), k=min(20, len(test))
            ),
            "naive_scout": evaluate_binary(
                test["target"], naive_scout_scores(test), k=min(20, len(test))
            ),
        }
        for model_name, res in results.items():
            for metric, val in res.items():
                mlflow.log_metric(f"{model_name}.{metric}", float(val))
        print(pd.DataFrame(results).T)

        imp = mean_abs_importance(gbm, train[feats])
        print("\ntop SHAP features:\n", imp.head(12))
        imp.to_csv("mlruns/_shap_importance.csv")
        mlflow.log_artifact("mlruns/_shap_importance.csv")

    print("\nlogged to MLflow experiment", EXPERIMENT)


if __name__ == "__main__":
    main()
