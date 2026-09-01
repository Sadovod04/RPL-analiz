"""SHAP analysis (M4).

Global importance, interaction values (esp. club x everything — club is a
confounder per SPEC §5.4), and partial dependence plots for key features.

Status: skeleton — implemented in M4.
"""

from __future__ import annotations

import pandas as pd


def explain(model, X: pd.DataFrame):  # noqa: ANN001
    raise NotImplementedError("M4")
