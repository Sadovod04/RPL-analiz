"""Feature engineering pipeline (M2).

Reads resolved raw data from Postgres, applies the time cutoff, aggregates
per-player features (per age bucket), attaches labels, writes Parquet to
``data/processed/``.

Every season row goes through ``features.time_cutoff.before_cutoff`` first.
Output columns are checked by ``eval.leakage_check.assert_no_leakage``.

Status: skeleton — implemented in M2.
"""

from __future__ import annotations

import pandas as pd


def build_feature_matrix(cutoff_age: float) -> pd.DataFrame:
    raise NotImplementedError("M2")
