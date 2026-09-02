"""Streamlit dashboard (M7).

Bilingual (RU/EN). Tabs: prospects · already pro/RPL · compare. Sidebar filters
(position, level reached, birth year, minutes, academy) apply to both list tabs.
Per-player card: raw youth stats, explained SHAP breakdown, closest breakthrough
players (the analogy).

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

LEVELS = ("РПЛ", "ФНЛ", "ФНЛ-2", "не дошёл")
_INT_KEYS = {
    "youth_minutes_total",
    "youth_goals_total",
    "youth_seasons",
    "height_cm",
    "minutes_U13",
    "minutes_U15",
    "minutes_U17",
    "minutes_U19",
    "minutes_U21",
    "rpl_minutes_ever",
    "market_value_at_cutoff_eur",
}


@st.cache_data
def _load(mtime: float) -> tuple[pd.DataFrame, str]:
    proc = ROOT / load_settings()["paths"]["data_processed"]
    path = proc / "features.parquet"
    if not path.exists():
        path = proc / "features_demo.parquet"
    return pd.read_parquet(path), path.name


def _parquet_mtime() -> float:
    proc = ROOT / load_settings()["paths"]["data_processed"]
    for name in ("features.parquet", "features_demo.parquet"):
        p = proc / name
        if p.exists():
            return p.stat().st_mtime
    return 0.0


@st.cache_resource
def _model(_resolved: pd.DataFrame, target_col: str, mtime: float):
    return train_ranker(_resolved, target_col)


# -- sidebar: language + target -------------------------------------
lang = st.sidebar.radio("Язык / Language", ["ru", "en"], format_func=str.upper, horizontal=True)

df, src = _load(_parquet_mtime())
is_demo = src.endswith("_demo.parquet")
has_level = "outcome_level" in df.columns

_TARGET_LABELS = {
    "ru": {"pro_target": "Проф. уровень (РПЛ / ФНЛ / ФНЛ-2)", "target": "Только РПЛ (≥200 мин)"},
    "en": {"pro_target": "Any pro (RPL / FNL / FNL-2)", "target": "RPL only (≥200 min)"},
}
_target_opts = [c for c in ("pro_target", "target") if c in df.columns] or ["target"]
target_col = st.sidebar.radio(
    "Цель / Target", _target_opts, format_func=lambda c: _TARGET_LABELS[lang].get(c, c)
)

st.title(t("title", lang))
st.caption(t("subtitle", lang))
if is_demo:
    st.warning(t("demo_warning", lang))

resolved, open_cohort = split_resolved_open(df, target_col)
if resolved[target_col].nunique() < 2:
    st.error(t("single_class", lang))
    st.stop()

model = _model(resolved, target_col, _parquet_mtime())


# -- helpers -------------------------------------------------------
def _pos(row) -> str:
    return position_label(row.get("position"), row.get("position_detail"), lang)


def _fmt(key: str, v) -> str:
    if isinstance(v, bool):
        return ("да" if lang == "ru" else "yes") if v else ("нет" if lang == "ru" else "no")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, (int, float)):
        if key in _INT_KEYS or abs(v) >= 1000:
            return f"{v:,.0f}".replace(",", " ")
        return f"{v:.2f}"
    return str(v)


# -- sidebar: filters (shared by both list tabs) --------------------
def _filter_widgets(frame: pd.DataFrame) -> dict:
    by = frame["birth_year"].dropna().astype(int)
    lo, hi = (int(by.min()), int(by.max())) if len(by) else (1990, 2013)
    with st.sidebar:
        st.header(t("filters", lang))
        poss = sorted(
            {position_label(r.position, r.position_detail, lang) for r in frame.itertuples()}
        )
        pos_sel = st.multiselect(
            t("position", lang), poss, placeholder=t("all_ph", lang), key="f_pos"
        )
        lvl_sel = (
            st.multiselect(
                t("level_reached", lang), list(LEVELS), placeholder=t("all_ph", lang), key="f_lvl"
            )
            if has_level
            else []
        )
        yr = (
            st.slider(t("birth_range", lang), lo, hi, (lo, hi), key="f_yr") if lo < hi else (lo, hi)
        )
        mm = st.slider(t("min_minutes", lang), 0, 5000, 0, 100, key="f_min")
        acs = sorted({str(x) for x in frame["academy_club"].dropna()})
        ac_sel = st.multiselect(t("academy", lang), acs, placeholder=t("all_ph", lang), key="f_ac")
        top_n = st.slider(t("top_n", lang), 5, 300, 50, key="f_top")
    return {"pos": pos_sel, "lvl": lvl_sel, "yr": yr, "mm": mm, "ac": ac_sel, "top": top_n}


def _apply(frame: pd.DataFrame, f: dict) -> pd.DataFrame:
    d = frame.assign(
        _pos=[position_label(r.position, r.position_detail, lang) for r in frame.itertuples()]
    )
    m = (d["youth_minutes_total"] >= f["mm"]) & (d["birth_year"].fillna(0).between(*f["yr"]))
    if f["pos"]:
        m &= d["_pos"].isin(f["pos"])
    if f["lvl"] and "outcome_level" in d:
        m &= d["outcome_level"].isin(f["lvl"])
    if f["ac"]:
        m &= d["academy_club"].astype(str).isin(f["ac"])
    return d[m].head(f["top"])


def _table(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d[t("col_pos", lang)] = [
        position_label(r.position, r.position_detail, lang) for r in d.itertuples()
    ]
    d[t("score", lang)] = (d["breakthrough_score"] * 100).round(1)
    d[t("col_ga90", lang)] = d["youth_ga_per90"].round(2)
    d[t("col_minutes", lang)] = d["youth_minutes_total"].round(0).astype("Int64")
    ren = {
        "canonical_name": t("col_name", lang),
        "birth_year": t("col_birth", lang),
        "academy_club": t("col_academy", lang),
        "outcome_level": t("col_level_reached", lang),
    }
    d = d.rename(columns=ren)
    keep = [
        t("col_name", lang),
        t("col_birth", lang),
        t("col_pos", lang),
        t("score", lang),
        t("col_level_reached", lang),
        t("col_academy", lang),
        t("col_minutes", lang),
        t("col_ga90", lang),
    ]
    return d[[c for c in keep if c in d.columns]]


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
    st.markdown(f"**{t('raw_stats', lang)}**")
    st.table(pd.Series({feat_label(k, lang): _fmt(k, v) for k, v in raw.items()}, name=""))

    st.markdown(f"**{t('shap_hdr', lang)}**")
    st.caption(t("shap_help", lang))
    contrib = explain_player(model, df, pick).head(12)
    contrib.index = [feat_label(c, lang) for c in contrib.index]
    st.bar_chart(contrib, horizontal=True)

    st.markdown(f"**{t('similar_hdr', lang)}**")
    st.caption(t("similar_help", lang))
    sim = similar_breakthrough_players(df, pick, k=6, target_col=target_col)
    if len(sim):
        sim = sim.assign(
            **{
                t("col_pos", lang): [
                    position_label(r.position, None, lang) for r in sim.itertuples()
                ]
            }
        )
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


# -- compare ----------------------------------------------------
_CMP_MIN = ["youth_minutes_total", "minutes_U15", "minutes_U17", "minutes_U19", "rpl_minutes_ever"]
_CMP_RATE = ["youth_ga_per90", "ga_per90_U17", "ga_per90_U19"]
_CMP_TABLE = [
    "outcome_level",
    "height_cm",
    "youth_seasons",
    "youth_minutes_total",
    "youth_goals_total",
    "youth_ga_per90",
    "minutes_U15",
    "minutes_U17",
    "minutes_U19",
    "ga_per90_U17",
    "ga_per90_U19",
    "youth_minutes_trend",
    "played_youth_league",
    "academy_conversion_rate",
    "market_value_at_cutoff_eur",
    "rpl_minutes_ever",
]


@st.cache_resource
def _scored_all(_df: pd.DataFrame, tc: str, mtime: float) -> pd.DataFrame:
    from features.build_features import feature_columns

    out = _df.copy()
    out["breakthrough_score"] = model.predict_proba(out[feature_columns(_df)])
    return out


def _compare(scored: pd.DataFrame):
    st.subheader(t("compare_hdr", lang))
    names = dict(zip(scored["player_id"], scored["canonical_name"], strict=False))
    picked = st.multiselect(
        t("compare_pick", lang),
        scored.sort_values("canonical_name")["player_id"].tolist(),
        format_func=lambda p: names.get(p, p),
        max_selections=6,
    )
    if len(picked) < 2:
        st.info(t("compare_need", lang))
        return

    sub = scored[scored["player_id"].isin(picked)].set_index("canonical_name")
    st.table(
        pd.DataFrame(
            {
                t("score", lang): (sub["breakthrough_score"] * 100).round(1),
                t("col_birth", lang): sub["birth_year"].astype("Int64"),
                t("col_pos", lang): [
                    position_label(r.position, r.position_detail, lang) for r in sub.itertuples()
                ],
                t("col_level_reached", lang): sub.get("outcome_level"),
                t("col_academy", lang): sub["academy_club"],
            }
        )
    )

    st.markdown(f"**{t('compare_stats', lang)}**")
    st.table(
        pd.DataFrame(
            {
                feat_label(c, lang): sub[c].map(lambda v, k=c: _fmt(k, v))
                for c in _CMP_TABLE
                if c in sub
            }
        ).T
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{t('compare_minutes', lang)}**")
        mcols = [c for c in _CMP_MIN if c in sub]
        st.bar_chart(sub[mcols].rename(columns=lambda c: feat_label(c, lang)))
    with c2:
        st.markdown(f"**{t('compare_rates', lang)}**")
        rcols = [c for c in _CMP_RATE if c in sub]
        st.bar_chart(sub[rcols].rename(columns=lambda c: feat_label(c, lang)))


# -- layout ---------------------------------------------------
flt = _filter_widgets(df)

tab_prospects, tab_pro, tab_compare = st.tabs(
    [t("tab_prospects", lang), t("tab_pro", lang), t("tab_compare", lang)]
)

with tab_prospects:
    st.subheader(t("prospects_hdr", lang))
    view = _apply(rank_prospects(df, model=model, target_col=target_col), flt)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(_table(view), use_container_width=True, hide_index=True)
    with right:
        _player_card(view)

with tab_pro:
    st.subheader(t("pro_hdr", lang))
    res = _apply(rank_resolved(df, model, target_col=target_col), flt)
    show = _table(res)
    if "outcome" in res:
        show.insert(
            4,
            "→",
            res["outcome"].map({1: t("outcome_yes", lang), 0: t("outcome_no", lang)}).to_numpy(),
        )
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab_compare:
    _compare(_scored_all(df, target_col, _parquet_mtime()))
