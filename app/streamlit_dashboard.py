"""Streamlit dashboard (M7).

Bilingual (RU/EN). Two tabs — "already pro / RPL" and "prospects". Per-player card
shows the raw youth stats, an explained SHAP breakdown, and the closest players
who already broke through (the analogy).

    uv run --extra app streamlit run app/streamlit_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.i18n import feat_label, position_label, t  # noqa: E402
from app.ranking import (  # noqa: E402
    explain_player,
    player_raw_stats,
    rank_prospects,
    rank_resolved,
    similar_breakthrough_players,
    split_resolved_open,
    train_ranker,
)
from settings import load_settings  # noqa: E402

st.set_page_config(page_title="RPL-analiz", layout="wide")


@st.cache_data
def _load() -> tuple[pd.DataFrame, str]:
    proc = ROOT / load_settings()["paths"]["data_processed"]
    path = proc / "features.parquet"
    if not path.exists():
        path = proc / "features_demo.parquet"
    return pd.read_parquet(path), path.name


@st.cache_resource
def _model(_resolved: pd.DataFrame, target_col: str):
    return train_ranker(_resolved, target_col)


# -- language ---------------------------------------------------------
lang = st.sidebar.radio("Язык / Language", ["ru", "en"], format_func=str.upper, horizontal=True)

df, src = _load()
is_demo = src.endswith("_demo.parquet")

_TARGET_LABELS = {
    "ru": {"pro_target": "Проф. уровень (РПЛ / ФНЛ / ФНЛ-2)", "target": "Только РПЛ (≥200 мин)"},
    "en": {"pro_target": "Any pro (RPL / FNL / FNL-2)", "target": "RPL only (≥200 min)"},
}
_target_opts = [c for c in ("pro_target", "target") if c in df.columns] or ["target"]
target_col = st.sidebar.radio(
    "Цель / Target",
    _target_opts,
    format_func=lambda c: _TARGET_LABELS[lang].get(c, c),
)

st.title(t("title", lang))
st.caption(t("subtitle", lang))
if is_demo:
    st.warning(t("demo_warning", lang))

resolved, open_cohort = split_resolved_open(df, target_col)
if resolved[target_col].nunique() < 2:
    st.error(t("single_class", lang))
    st.stop()

model = _model(resolved, target_col)


def _pos(row) -> str:
    return position_label(row.get("position"), row.get("position_detail"), lang)


def _table(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d[t("col_pos", lang)] = d.apply(_pos, axis=1)
    d[t("score", lang)] = (d["breakthrough_score"] * 100).round(1)
    cols = {
        "canonical_name": t("col_name", lang),
        "birth_year": t("col_birth", lang),
        "academy_club": t("col_academy", lang),
        "youth_minutes_total": t("col_minutes", lang),
        "youth_ga_per90": t("col_ga90", lang),
        "best_level_pre_cutoff": t("col_level", lang),
    }
    keep = [
        t("col_name", lang),
        t("col_birth", lang),
        t("col_pos", lang),
        t("score", lang),
        t("col_academy", lang),
        t("col_minutes", lang),
        t("col_ga90", lang),
        t("col_level", lang),
    ]
    d = d.rename(columns=cols)
    d[t("col_ga90", lang)] = d[t("col_ga90", lang)].round(2)
    return d[[c for c in keep if c in d.columns]]


def _sidebar_filters(frame: pd.DataFrame, key: str):
    with st.sidebar:
        st.header(t("filters", lang))
        frame = frame.assign(_pos=frame.apply(_pos, axis=1))
        poss = sorted(frame["_pos"].dropna().unique())
        pos_sel = st.multiselect(t("position", lang), poss, default=poss, key=f"{key}_pos")
        mm = st.slider(t("min_minutes", lang), 0, 5000, 0, 100, key=f"{key}_min")
        acs = sorted(x for x in frame["academy_club"].dropna().unique())
        ac_sel = st.multiselect(t("academy", lang), acs, default=acs, key=f"{key}_ac")
        top_n = st.slider(t("top_n", lang), 5, 200, 30, key=f"{key}_top")
    m = (
        frame["_pos"].isin(pos_sel)
        & (frame["youth_minutes_total"] >= mm)
        & (frame["academy_club"].isin(ac_sel) | frame["academy_club"].isna() | (len(acs) == 0))
    )
    return frame[m].head(top_n)


def _player_card(view: pd.DataFrame):
    st.subheader(t("why", lang))
    if not len(view):
        st.info(t("no_match", lang))
        return
    pick = st.selectbox(
        t("pick_player", lang),
        view["player_id"],
        format_func=lambda pid: view.loc[view["player_id"] == pid, "canonical_name"].iloc[0],
    )

    raw = player_raw_stats(df, pick)
    nice = {feat_label(k, lang): _fmt(k, v) for k, v in raw.items()}
    st.markdown(f"**{t('raw_stats', lang)}**")
    st.table(pd.Series(nice, name=""))

    st.markdown(f"**{t('shap_hdr', lang)}**")
    st.caption(t("shap_help", lang))
    contrib = explain_player(model, df, pick).head(12)
    contrib.index = [feat_label(c, lang) for c in contrib.index]
    st.bar_chart(contrib)

    st.markdown(f"**{t('similar_hdr', lang)}**")
    st.caption(t("similar_help", lang))
    sim = similar_breakthrough_players(df, pick, k=6, target_col=target_col)
    if len(sim):
        sim = sim.assign(**{t("col_pos", lang): sim.apply(_pos, axis=1)})
        sim = sim.rename(
            columns={
                "canonical_name": t("col_name", lang),
                "birth_year": t("col_birth", lang),
                "distance": t("similar_dist", lang),
            }
        )
        st.table(
            sim[
                [
                    t("col_name", lang),
                    t("col_birth", lang),
                    t("col_pos", lang),
                    t("similar_dist", lang),
                ]
            ]
        )
    else:
        st.info(t("similar_none", lang))


def _fmt(key: str, v) -> str:
    if isinstance(v, bool):
        return "да / yes" if v else "нет / no"
    if isinstance(v, float):
        return "—" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.2f}")
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


tab_pro, tab_prospects = st.tabs([t("tab_pro", lang), t("tab_prospects", lang)])

with tab_prospects:
    st.subheader(t("prospects_hdr", lang))
    ranked = rank_prospects(df, model=model, target_col=target_col)
    view = _sidebar_filters(ranked, "pr")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(_table(view), use_container_width=True, hide_index=True)
    with right:
        _player_card(view)

with tab_pro:
    st.subheader(t("pro_hdr", lang))
    res = rank_resolved(df, model, target_col=target_col)
    res["_outcome"] = res["outcome"].map({1: t("outcome_yes", lang), 0: t("outcome_no", lang)})
    show = _table(res)
    show.insert(3, "→", res["_outcome"].to_numpy())
    st.dataframe(show, use_container_width=True, hide_index=True)
