"""Streamlit dashboard (M7).

Bilingual (RU/EN). Tabs: prospects · already pro/RPL · compare · youth (ФФ СПб).
One combined dataset: Transfermarkt academy players (CatBoost score) + ФФ СПб kids
2012–2015 (transparent 0–100 heuristic). A "Источник / Source" filter and a
"Метод" column tell them apart. Sidebar filters (position, level reached, birth
year, academy, source) apply to both list tabs. Per-player card: raw youth stats,
explained SHAP breakdown, closest breakthrough players — or, for a kid, a plain
projected-level card (no career yet -> no SHAP).

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
    _score_frame,
    explain_player,
    player_raw_stats,
    rank_prospects,
    rank_resolved,
    similar_breakthrough_players,
    split_resolved_open,
    train_ranker,
)
from app.youth_features import combined_frame, youth_frame  # noqa: E402
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


def _proc() -> Path:
    return ROOT / load_settings()["paths"]["data_processed"]


def _ffspb_path() -> Path:
    return _proc() / "ffspb_players.parquet"


@st.cache_data
def _load(mtime: float) -> tuple[pd.DataFrame, str]:
    proc = _proc()
    path = proc / "features.parquet"
    if not path.exists():
        path = proc / "features_demo.parquet"
    ff = _ffspb_path()
    return combined_frame(path, ff if ff.exists() else None), path.name


def _parquet_mtime() -> float:
    proc = _proc()
    m = 0.0
    for name in ("features.parquet", "features_demo.parquet", "ffspb_players.parquet"):
        p = proc / name
        if p.exists():
            m = max(m, p.stat().st_mtime)
    return m


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
    "Цель / Target",
    _target_opts,
    format_func=lambda c: _TARGET_LABELS[lang].get(c, c),
    help=t("target_help", lang),
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
            t("position", lang),
            poss,
            placeholder=t("all_ph", lang),
            help=t("pos_help", lang),
            key="f_pos",
        )
        lvl_sel = (
            st.multiselect(
                t("level_reached", lang),
                list(LEVELS),
                placeholder=t("all_ph", lang),
                help=t("lvl_help", lang),
                key="f_lvl",
            )
            if has_level
            else []
        )
        yr = (
            st.slider(t("birth_range", lang), lo, hi, (lo, hi), help=t("yr_help", lang), key="f_yr")
            if lo < hi
            else (lo, hi)
        )
        acs = sorted({str(x) for x in frame["academy_club"].dropna()})
        ac_sel = st.multiselect(
            t("academy", lang),
            acs,
            placeholder=t("all_ph", lang),
            help=t("ac_help", lang),
            key="f_ac",
        )
        src_sel = []
        if "source" in frame.columns and frame["source"].nunique() > 1:
            src_map = {"tm": t("src_tm", lang), "ffspb": t("src_ffspb", lang)}
            src_sel = st.multiselect(
                t("source", lang),
                list(src_map),
                format_func=lambda s: src_map.get(s, s),
                placeholder=t("all_ph", lang),
                help=t("src_help", lang),
                key="f_src",
            )
    return {"pos": pos_sel, "lvl": lvl_sel, "yr": yr, "ac": ac_sel, "src": src_sel}


def _apply(frame: pd.DataFrame, f: dict) -> pd.DataFrame:
    d = frame.assign(
        _pos=[position_label(r.position, r.position_detail, lang) for r in frame.itertuples()]
    )
    m = d["birth_year"].fillna(0).between(*f["yr"])
    if f["pos"]:
        m &= d["_pos"].isin(f["pos"])
    if f["lvl"] and "outcome_level" in d:
        m &= d["outcome_level"].isin(f["lvl"])
    if f["ac"]:
        m &= d["academy_club"].astype(str).isin(f["ac"])
    if f.get("src") and "source" in d:
        m &= d["source"].isin(f["src"])
    return d[m]


def _table(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d[t("col_pos", lang)] = [
        position_label(r.position, r.position_detail, lang) for r in d.itertuples()
    ]
    d[t("score", lang)] = (d["breakthrough_score"] * 100).round(1)
    d[t("col_ga90", lang)] = pd.to_numeric(d["youth_ga_per90"], errors="coerce").round(2)
    d[t("col_minutes", lang)] = (
        pd.to_numeric(d["youth_minutes_total"], errors="coerce").round(0).astype("Int64")
    )
    if "source" in d:
        d[t("col_method", lang)] = d["source"].map(
            {"tm": t("method_model", lang), "ffspb": t("method_heur", lang)}
        )
    # level column: reached-level for TM players, ≈projection for youth
    lvl_txt = d.get("outcome_level")
    if "proj_level" in d and lvl_txt is not None:
        proj = d["proj_level"].map(
            lambda k: "≈ " + t(k, lang) if isinstance(k, str) and k.startswith("yl_") else None
        )
        lvl_txt = lvl_txt.where(lvl_txt.notna(), proj)
        d["outcome_level"] = lvl_txt
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
        t("col_method", lang),
        t("col_level_reached", lang),
        t("col_academy", lang),
        t("col_minutes", lang),
        t("col_ga90", lang),
    ]
    return d[[c for c in keep if c in d.columns]]


@st.cache_data
def _quantiles(mtime: float) -> dict:
    cols = [
        "youth_minutes_total",
        "youth_ga_per90",
        "youth_minutes_trend",
        "market_value_at_cutoff_eur",
        "academy_conversion_rate",
    ]
    return {c: df[c].quantile([0.25, 0.5, 0.75]).to_dict() for c in cols if c in df}


def _projected_level(score: float) -> str:
    if target_col == "target":  # RPL-only model
        return (
            "lvl_rpl_high"
            if score >= 0.5
            else "lvl_rpl_mid"
            if score >= 0.3
            else "lvl_fnl"
            if score >= 0.15
            else "lvl_fnl2"
            if score >= 0.05
            else "lvl_low"
        )
    return (  # any-pro model
        "lvl_rpl_mid"
        if score >= 0.88
        else "lvl_fnl"
        if score >= 0.62
        else "lvl_fnl2"
        if score >= 0.35
        else "lvl_low"
    )


def _player_report(row: pd.Series, score: float):
    q = _quantiles(_parquet_mtime())
    st.markdown(f"**{t('report_hdr', lang)}**")
    st.markdown(
        f"{t('report_level', lang)}: **{t(_projected_level(score), lang)}** ({score * 100:.0f}%)"
    )

    def hi(c):
        return c in q and pd.notna(row.get(c)) and row[c] >= q[c][0.75]

    def lo(c):
        return c in q and pd.notna(row.get(c)) and row[c] <= q[c][0.25]

    strengths, weaknesses = [], []
    if hi("youth_minutes_total"):
        strengths.append("s_minutes")
    elif lo("youth_minutes_total"):
        weaknesses.append("w_minutes")
    if hi("youth_ga_per90"):
        strengths.append("s_output")
    elif lo("youth_ga_per90"):
        weaknesses.append("w_output")
    trend = row.get("youth_minutes_trend")
    if pd.notna(trend) and trend > 5:
        strengths.append("s_trend")
    elif pd.notna(trend) and trend < -5:
        weaknesses.append("w_trend")
    lvl = row.get("best_level_pre_cutoff", 0) or 0
    if lvl >= 3:
        strengths.append("s_level")
    elif lvl == 0:
        weaknesses.append("w_level")
    if bool(row.get("played_youth_league")):
        strengths.append("s_youthleague")
    if hi("market_value_at_cutoff_eur"):
        strengths.append("s_value")
    if hi("academy_conversion_rate"):
        strengths.append("s_academy")

    if strengths:
        st.markdown(f"{t('report_strengths', lang)}: " + ", ".join(t(s, lang) for s in strengths))
    if weaknesses:
        st.markdown(f"{t('report_weaknesses', lang)}: " + ", ".join(t(w, lang) for w in weaknesses))
    if not strengths and not weaknesses:
        st.caption(t("w_nodata", lang))
    st.divider()


def _youth_player_card(row: pd.Series, score: float):
    lvl_key = row.get("proj_level") if isinstance(row.get("proj_level"), str) else "yl_low"
    st.markdown(f"**{t('report_hdr', lang)}**")
    st.markdown(f"{t('report_level', lang)}: **≈ {t(lvl_key, lang)}** ({score * 100:.0f}/100)")
    st.caption(t("youth_card_note", lang))
    g = pd.to_numeric(pd.Series([row.get("youth_goals_total")]), errors="coerce").iloc[0]
    mins = pd.to_numeric(pd.Series([row.get("youth_minutes_total")]), errors="coerce").iloc[0]
    trn = pd.to_numeric(pd.Series([row.get("youth_seasons")]), errors="coerce").iloc[0]
    by = row.get("birth_year")
    st.table(
        pd.Series(
            {
                t("y_teams", lang): str(row.get("academy_club") or "—"),
                t("col_birth", lang): "—" if pd.isna(by) else str(int(by)),
                t("y_goals", lang): "—" if pd.isna(g) else f"{g:.0f}",
                t("col_minutes", lang): "—" if pd.isna(mins) else f"{mins:,.0f}".replace(",", " "),
                t("y_gpg", lang): str(_fmt("youth_ga_per90", row.get("youth_ga_per90"))),
                t("y_trn", lang): "—" if pd.isna(trn) else str(int(trn)),
            },
            name="",
        )
    )
    st.caption(t("y_score_help", lang))
    st.divider()


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
    prow = df[df["player_id"] == pick].iloc[0]
    pscore = float(view.loc[view["player_id"] == pick, "breakthrough_score"].iloc[0])

    if str(prow.get("source")) == "ffspb":
        _youth_player_card(prow, pscore)
        return

    _player_report(prow, pscore)

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
    return _score_frame(model, _df, _df)


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

    # side-by-side spec sheet: rows = metrics, columns = players, best value bold
    st.markdown(f"**{t('compare_stats', lang)}**")
    present = [c for c in _CMP_TABLE if c in sub]
    disp = pd.DataFrame(
        {name: [_fmt(c, sub.loc[name, c]) for c in present] for name in sub.index},
        index=[feat_label(c, lang) for c in present],
    )
    raw_num = pd.DataFrame(
        {
            name: [pd.to_numeric(sub.loc[name, c], errors="coerce") for c in present]
            for name in sub.index
        },
        index=disp.index,
    )
    higher_better = {feat_label(c, lang) for c in present if c != "youth_minutes_trend"}

    def _bold_best(row):
        vals = raw_num.loc[row.name]
        if vals.notna().sum() < 2:
            return [""] * len(row)
        best = vals.max() if row.name in higher_better else vals.min()
        return ["font-weight:700" if v == best else "" for v in vals]

    st.dataframe(disp.style.apply(_bold_best, axis=1), use_container_width=True)

    # one readable view: each key metric as % of the best player in the group.
    # A dataframe with per-player progress bars — always grouped, never stacked.
    st.markdown(f"**{t('compare_norm', lang)}**")
    key = [
        c
        for c in (
            "youth_minutes_total",
            "minutes_U17",
            "minutes_U19",
            "youth_goals_total",
            "youth_ga_per90",
            "rpl_minutes_ever",
        )
        if c in sub
    ]
    norm = sub[key].apply(pd.to_numeric, errors="coerce").fillna(0)
    norm = (norm / norm.max().replace(0, 1) * 100).round(0)
    norm.columns = [feat_label(c, lang) for c in key]
    norm_t = norm.T
    norm_t.index.name = t("compare_stats", lang)
    st.dataframe(
        norm_t,
        use_container_width=True,
        column_config={
            c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%d%%")
            for c in norm_t.columns
        },
    )


# -- youth (ФФ СПб) ------------------------------------------------
@st.cache_data
def _load_youth(mtime: float) -> pd.DataFrame | None:
    p = _ffspb_path()
    if not p.exists():
        return None
    return youth_frame(p)


def _youth_tab():
    d = _load_youth(_parquet_mtime())
    st.subheader(t("youth_hdr", lang))
    if d is None:
        st.info(t("youth_none", lang))
        return
    st.caption(t("youth_note", lang))
    st.caption("ℹ️ " + t("y_score_help", lang))
    with st.expander(t("filters", lang), expanded=True):
        c1, c2, c3 = st.columns(3)
        years = sorted(int(y) for y in d["birth_year"].dropna().unique())
        yr = c1.multiselect(t("birth_range", lang), years, placeholder=t("all_ph", lang))
        min_g = c2.number_input(t("y_min_games", lang), 0, 200, 10)
        q = c3.text_input(t("y_search", lang), "")
    v = d[d["games"] >= min_g]
    if yr:
        v = v[v["birth_year"].isin(yr)]
    if q:
        v = v[v["full_name"].str.contains(q, case=False, na=False)]
    v = v.sort_values("pers_score", ascending=False)

    show = v.assign(proj=v["proj_level"].map(lambda k: t(k, lang))).rename(
        columns={
            "full_name": t("y_name", lang),
            "patronymic": t("y_patr", lang),
            "birth_year": t("col_birth", lang),
            "teams": t("y_teams", lang),
            "games": t("y_games", lang),
            "goals": t("y_goals", lang),
            "gpg": t("y_gpg", lang),
            "pers_score": t("y_score", lang),
            "proj": t("y_level", lang),
        }
    )
    st.caption(f"{len(v)} / {len(d)}")
    st.dataframe(
        show[
            [
                t("y_name", lang),
                t("y_patr", lang),
                t("col_birth", lang),
                t("y_score", lang),
                t("y_level", lang),
                t("y_teams", lang),
                t("y_games", lang),
                t("y_goals", lang),
                t("y_gpg", lang),
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    # compare a few kids
    st.markdown(f"**{t('y_compare', lang)}**")
    names = dict(zip(v["full_name"] + " · " + v["birth_year"].astype(str), v.index, strict=False))
    picks = st.multiselect(t("y_compare_pick", lang), list(names), max_selections=6)
    if len(picks) >= 2:
        sub = v.loc[[names[p] for p in picks]].set_index("full_name")
        cmp = pd.DataFrame(
            {
                t("y_score", lang): sub["pers_score"].astype(int),
                t("y_level", lang): sub["proj_level"].map(lambda k: t(k, lang)),
                t("col_birth", lang): sub["birth_year"].astype("Int64"),
                t("y_games", lang): sub["games"].astype(int),
                t("y_goals", lang): sub["goals"].astype(int),
                t("y_gpg", lang): sub["gpg"],
                t("y_teams", lang): sub["teams"],
            }
        )
        st.table(cmp.T)


# -- layout ---------------------------------------------------
flt = _filter_widgets(df)

tab_prospects, tab_pro, tab_compare, tab_youth = st.tabs(
    [t("tab_prospects", lang), t("tab_pro", lang), t("tab_compare", lang), t("tab_youth", lang)]
)

with tab_prospects:
    st.subheader(t("prospects_hdr", lang))
    view = _apply(rank_prospects(df, model=model, target_col=target_col), flt)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(_table(view), use_container_width=True, hide_index=True, height=640)
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
    st.dataframe(show, use_container_width=True, hide_index=True, height=640)

with tab_compare:
    _compare(_scored_all(df, target_col, _parquet_mtime()))

with tab_youth:
    _youth_tab()
