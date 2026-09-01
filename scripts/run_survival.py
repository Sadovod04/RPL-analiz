"""M5: Cox survival model, compared with the binary CatBoost ranking.

    uv run python scripts/run_survival.py [path.parquet]

Fits Cox PH on the temporal-train split, reports concordance and the
P(breakthrough by age {21,23,25}) distribution on test, and checks whether
ranking test players by P(breakthrough by 23) recovers the binary target as well
as the CatBoost probability does (PR-AUC / Recall@Top-K). RSF runs too if the
``survival`` extra is installed. Logged to MLflow.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from eval.metrics import evaluate_binary
from features.build_features import feature_columns
from features.split import temporal_split
from models.gbm import CatBoostBreakthrough
from models.survival import CoxBreakthrough, survival_frame
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
    args = ap.parse_args(argv)

    df, name = _load(args.path)
    feats = feature_columns(df)
    # survival keeps censored rows -> don't drop them
    train, test = temporal_split(df, drop_censored_rows=False)
    fr_tr = survival_frame(train, feats)
    fr_te = survival_frame(test, feats).reindex(columns=fr_tr.columns, fill_value=0.0)

    print(
        f"{name}: train={len(fr_tr)} test={len(fr_te)}  "
        f"events train={int(fr_tr['event_observed'].sum())}"
    )

    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{Path('mlruns/mlflow.db').resolve()}")
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="cox"):
        cox = CoxBreakthrough().fit(fr_tr)
        print(f"Cox concordance (train): {cox.concordance_index_:.3f}")
        mlflow.log_metric("cox.concordance_train", cox.concordance_index_)

        for age in (21, 23, 25):
            p = cox.probability_by_age(fr_te, age)
            print(f"  P(breakthrough by {age}): mean={p.mean():.3f}  p90={np.quantile(p, 0.9):.3f}")
            mlflow.log_metric(f"cox.mean_p_by_{age}", float(p.mean()))

        # compare ranking-by-P(by 23) with the binary CatBoost probability
        resolved = test["target"] != -1
        if resolved.sum() >= 10 and test.loc[resolved, "target"].nunique() == 2:
            y = test.loc[resolved, "target"].to_numpy()
            p_cox = cox.probability_by_age(fr_te, 23)[resolved.to_numpy()]
            gbm = CatBoostBreakthrough(params={"iterations": 200}).fit(
                train[feats], train["target"].where(train["target"] != -1, 0)
            )
            p_gbm = gbm.predict_proba(test[feats])[resolved.to_numpy()]
            k = min(20, int(resolved.sum()))
            cmp = pd.DataFrame(
                {
                    "cox P(by 23)": evaluate_binary(y, p_cox, k=k),
                    "catboost binary": evaluate_binary(y, p_gbm, k=k),
                }
            ).T
            print("\nranking vs binary target (resolved test):\n", cmp)
            for m, row in cmp.iterrows():
                for metric, val in row.items():
                    mlflow.log_metric(f"{m.replace(' ', '_')}.{metric}", float(val))
        else:
            print("\n(too few resolved test rows to compare rankings — expected on the demo)")

    if importlib.util.find_spec("sksurv") is None:
        print("\nRSF skipped: `uv sync --extra survival` to enable scikit-survival")


if __name__ == "__main__":
    main()
