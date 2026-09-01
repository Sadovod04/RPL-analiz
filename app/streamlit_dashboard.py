"""Streamlit dashboard (M6).

Interactive ranking of current prospects (players whose outcome is still open):
filter by academy / position / youth output, inspect a player card with the SHAP
breakdown of why the model ranks them where it does.

    uv run --extra app streamlit run app/streamlit_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.ranking import (  # noqa: E402
    explain_player,
    rank_prospects,
    split_resolved_open,
    train_ranker,
)
from settings import load_settings  # noqa: E402

st.set_page_config(page_title="RPL-analiz — prospects", layout="wide")


@st.cache_data
def _load() -> tuple[pd.DataFrame, str]:
    proc = ROOT / load_settings()["paths"]["data_processed"]
    path = proc / "features.parquet"
    if not path.exists():
        path = proc / "features_demo.parquet"
    return pd.read_parquet(path), path.name


@st.cache_resource
def _model(_resolved: pd.DataFrame):
    return train_ranker(_resolved)


df, src = _load()
st.title("RPL-analiz — breakthrough prospects")
st.caption(
    f"data: `{src}` · ranking model: CatBoost trained on resolved cohorts · "
    "scores are calibrated probabilities of ever reaching ≥200 RPL minutes. "
    "Small, noisy problem — read as a shortlist, not a verdict (SPEC §14)."
)

resolved, open_cohort = split_resolved_open(df)
if resolved["target"].nunique() < 2:
    st.error(
        "The loaded dataset has only one outcome class among resolved players "
        "(expected with the demo sample). Run a full `run_ingest` to populate negatives."
    )
    st.stop()

model = _model(resolved)
ranked = rank_prospects(df, model=model)

with st.sidebar:
    st.header("filters")
    positions = sorted(x for x in ranked["position"].dropna().unique())
    pos_sel = st.multiselect("position", positions, default=positions)
    min_minutes = st.slider("min youth minutes", 0, 5000, 0, step=100)
    academies = sorted(x for x in ranked["academy_club"].dropna().unique())
    ac_sel = st.multiselect("academy", academies, default=academies[:50])
    top_n = st.slider("show top N", 5, 100, 25)

view = ranked[
    ranked["position"].isin(pos_sel)
    & (ranked["youth_minutes_total"] >= min_minutes)
    & (ranked["academy_club"].isin(ac_sel) | ranked["academy_club"].isna())
].head(top_n)

left, right = st.columns([3, 2])
with left:
    st.subheader(f"top {len(view)} prospects")
    st.dataframe(
        view.assign(breakthrough_score=lambda d: (d["breakthrough_score"] * 100).round(1)),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("why this player?")
    if len(view):
        pick = st.selectbox(
            "player",
            view["player_id"],
            format_func=lambda pid: view.loc[view["player_id"] == pid, "canonical_name"].iloc[0],
        )
        contrib = explain_player(model, df, pick).head(12)
        st.bar_chart(contrib)
        st.caption("signed SHAP contributions · positive pushes the breakthrough score up")
    else:
        st.info("no players match the filters")
