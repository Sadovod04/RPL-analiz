"""Feature engineering pipeline (M2).

One row per player. Every season row is passed through
:func:`features.time_cutoff.before_cutoff` first, so features use *only* what was
observable before the player reached ``modeling_cutoff_age`` (SPEC §7, §8). Label
columns (``target`` etc.) and post-cutoff aggregates (``current_age``,
``rpl_minutes_ever`` …) travel in the same frame but are **not** features — use
:func:`feature_columns` / :func:`assert_matrix_is_clean` before fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eval.leakage_check import assert_no_leakage
from features.labels import LabelConfig, attach_labels
from features.time_cutoff import before_cutoff
from settings import load_settings

# age (at season) -> youth bucket
_BUCKET_EDGES = [(13, "U13"), (15, "U15"), (17, "U17"), (19, "U19"), (21, "U21")]
# competition -> ordinal "level reached" (higher = closer to the first team)
_LEVEL_RANK = {
    "Russian Youth League": 1,
    "Vtoraya Liga": 2,
    "Pervaya Liga": 3,
    "Premier Liga (relegation)": 3,
    "Premier Liga": 4,
}

# columns that exist in the matrix but must never be fed to a model
NON_FEATURE_COLS = {
    "player_id",
    "canonical_name",
    "birth_year",
    "birth_month",  # raw -> birth_quarter / rel_age_frac are the features
    "academy_club",
    "target",
    "pro_target",
    "ordinal_target",
    "outcome_level",
    "duration",
    "event_observed",
    "rpl_minutes_ever",
    "rpl_debut_age",
    "reached_pro_level",
    "current_age",
    "source",  # "tm" | "ffspb" — provenance, not a feature
    "pers_score",  # ffspb youth heuristic (0-100), not a model input
    "proj_level",
}


def age_bucket(age: float) -> str | None:
    if pd.isna(age):
        return None
    for edge, name in _BUCKET_EDGES:
        if age < edge:
            return name
    return None  # >= 21: senior, not a youth bucket


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _youth_features(seasons: pd.DataFrame) -> pd.DataFrame:
    """seasons already restricted to the pre-cutoff window; -> per-player row."""
    s = seasons.copy()
    for col in ("minutes", "matches", "goals", "assists"):
        s[col] = pd.to_numeric(s.get(col), errors="coerce").fillna(0)
    s["bucket"] = s["age_at_season"].map(age_bucket)
    s["level"] = s["league"].map(_LEVEL_RANK).fillna(0)

    rows = []
    for pid, g in s.groupby("player_id"):
        row: dict[str, object] = {"player_id": pid}
        tot_min = g["minutes"].sum()
        row["youth_seasons"] = int(g["season"].nunique())
        row["youth_minutes_total"] = float(tot_min)
        row["youth_goals_total"] = float(g["goals"].sum())
        row["youth_ga_per90"] = (
            90 * (g["goals"].sum() + g["assists"].sum()) / tot_min if tot_min else 0.0
        )
        row["youth_minutes_trend"] = _slope(
            g["age_at_season"].to_numpy(float), g["minutes"].to_numpy(float)
        )
        row["played_youth_league"] = bool((g["league"] == "Russian Youth League").any())
        row["best_level_pre_cutoff"] = float(g["level"].max())
        for _, bname in _BUCKET_EDGES:
            gb = g[g["bucket"] == bname]
            bmin = gb["minutes"].sum()
            row[f"minutes_{bname}"] = float(bmin)
            row[f"ga_per90_{bname}"] = (
                90 * (gb["goals"].sum() + gb["assists"].sum()) / bmin if bmin else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


# a (league, season) needs at least this many player-rows before we trust its
# average age / match-load as a "peer norm" (small leagues give noisy baselines)
_MIN_PEER_GROUP = 8


def _league_season_norms(seasons: pd.DataFrame) -> pd.DataFrame:
    """Per (league, season): peer count, mean age, and max matches (a full-season
    proxy). Groups smaller than ``_MIN_PEER_GROUP`` get NaN norms."""
    s = seasons[["league", "season", "age_at_season", "matches"]].copy()
    s["age_at_season"] = pd.to_numeric(s["age_at_season"], errors="coerce")
    s["matches"] = pd.to_numeric(s["matches"], errors="coerce")
    norms = (
        s.groupby(["league", "season"])
        .agg(peers=("age_at_season", "size"), mean_age=("age_at_season", "mean"),
             max_matches=("matches", "max"))
        .reset_index()
    )
    norms.loc[norms["peers"] < _MIN_PEER_GROUP, ["mean_age", "max_matches"]] = np.nan
    return norms


def _trajectory_features(youth: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Signals from the *shape* of the pre-cutoff career, one row per player:

    - ``*_age_gap_vs_peers`` — season age minus the mean age of that league/season.
      Negative = playing above their age group (a strong prospect signal).
    - ``matches_share_*`` — matches played / the fullest season in that league
      (a weak availability / injury proxy).
    - ``minutes_dropoff_max`` — largest season-over-season relative fall in minutes
      from a meaningful base (>=500'); ``had_minutes_collapse`` flags a >=900' ->
      <300' fall in consecutive pre-cutoff seasons.

    ``youth`` must already be time-cutoff filtered; ``seasons`` is the full frame
    (peer norms are contemporaneous, not outcome-derived).
    """
    norms = _league_season_norms(seasons)
    y = youth.merge(norms, on=["league", "season"], how="left")
    for col in ("minutes", "matches"):
        y[col] = pd.to_numeric(y.get(col), errors="coerce").fillna(0.0)
    y["age_at_season"] = pd.to_numeric(y["age_at_season"], errors="coerce")
    y["age_gap_vs_peers"] = y["age_at_season"] - y["mean_age"]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = y["matches"] / y["max_matches"]
    y["matches_share"] = share.where(y["max_matches"] > 0).clip(upper=1.0)

    rows = []
    for pid, g in y.sort_values("age_at_season").groupby("player_id"):
        row: dict[str, object] = {"player_id": pid}
        gap = g["age_gap_vs_peers"].dropna()
        row["min_age_gap_vs_peers"] = float(gap.min()) if len(gap) else 0.0
        row["mean_age_gap_vs_peers"] = float(gap.mean()) if len(gap) else 0.0
        sh = g["matches_share"].dropna()
        row["matches_share_min"] = float(sh.min()) if len(sh) else np.nan
        row["matches_share_mean"] = float(sh.mean()) if len(sh) else np.nan
        mins = g["minutes"].to_numpy(float)
        drop, collapse = 0.0, False
        for prev, cur in zip(mins[:-1], mins[1:], strict=True):
            if prev >= 500:
                drop = max(drop, (prev - cur) / prev)
            if prev >= 900 and cur < 300:
                collapse = True
        row["minutes_dropoff_max"] = float(max(drop, 0.0))
        row["had_minutes_collapse"] = bool(collapse)
        rows.append(row)
    return pd.DataFrame(rows)


def _academy_conversion_rate(labeled: pd.DataFrame) -> pd.Series:
    """Time-aware: for each player, P(target=1) among SAME academy, EARLIER cohorts
    only, excluding censored. NaN when there is no prior history (SPEC §7 leakage rule).
    """
    df = labeled[["player_id", "academy_club", "birth_year", "target"]].copy()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for i, r in df.iterrows():
        if pd.isna(r["academy_club"]) or pd.isna(r["birth_year"]):
            continue
        prior = df[
            (df["academy_club"] == r["academy_club"])
            & (df["birth_year"] < r["birth_year"])
            & (df["target"] != -1)
        ]
        if len(prior):
            out.at[i] = float((prior["target"] == 1).mean())
    return out


def _market_value_at_cutoff(
    market_values: pd.DataFrame, players: pd.DataFrame, cutoff_age: float
) -> pd.Series:
    if market_values is None or market_values.empty:
        return pd.Series(np.nan, index=players["player_id"])
    mv = market_values.merge(players[["player_id", "birth_year"]], on="player_id", how="left")
    mv["date"] = pd.to_datetime(mv["date"], errors="coerce")
    mv["age_at_point"] = mv["date"].dt.year - mv["birth_year"]
    mv = mv[mv["age_at_point"] < cutoff_age].sort_values("date")
    return mv.groupby("player_id")["value_eur"].last()


def build_feature_matrix(
    players: pd.DataFrame,
    seasons: pd.DataFrame,
    market_values: pd.DataFrame | None = None,
    *,
    cutoff_age: float | None = None,
    as_of_year: int | None = None,
    cfg: LabelConfig | None = None,
) -> pd.DataFrame:
    settings = load_settings()
    cutoff_age = cutoff_age or settings["features"]["modeling_cutoff_age"]
    as_of_year = as_of_year or pd.Timestamp.now().year
    cfg = cfg or LabelConfig.from_settings()

    labeled = attach_labels(players, seasons, as_of_year=as_of_year, cfg=cfg)

    youth = before_cutoff(seasons, cutoff_age, age_col="age_at_season")
    feats = _youth_features(youth)

    m = labeled.merge(feats, on="player_id", how="left")
    # players with zero pre-cutoff seasons -> explicit zeros, keep the row
    num_fill = [c for c in feats.columns if c not in ("player_id", "played_youth_league")]
    m[num_fill] = m[num_fill].fillna(0.0)
    m["played_youth_league"] = m["played_youth_league"].astype("boolean").fillna(False).astype(bool)

    traj = _trajectory_features(youth, seasons)
    m = m.merge(traj, on="player_id", how="left")
    for col in ("min_age_gap_vs_peers", "mean_age_gap_vs_peers", "minutes_dropoff_max"):
        m[col] = m[col].fillna(0.0)
    m["had_minutes_collapse"] = (
        m["had_minutes_collapse"].astype("boolean").fillna(False).astype(bool)
    )
    # matches_share_* stay NaN when unknown (CatBoost handles NaN; logreg imputes)

    # era control: rule / legionnaire-limit changes shift the base rate by cohort.
    # In the temporal split test cohorts sit outside the train range, so the model
    # can only read a trend here, not memorise a cohort -> outcome mapping.
    m["cohort_year"] = pd.to_numeric(m["birth_year"], errors="coerce")
    if "birth_month" in players.columns:
        bm = pd.to_numeric(
            m["player_id"].map(players.set_index("player_id")["birth_month"]), errors="coerce"
        )
        # RFU age groups cut on 1 Jan: born early in the year => oldest in the cohort
        m["birth_quarter"] = np.ceil(bm / 3).clip(1, 4)
        m["rel_age_frac"] = (12 - bm) / 11.0
    else:
        m["birth_quarter"] = np.nan
        m["rel_age_frac"] = np.nan

    m["academy_conversion_rate"] = _academy_conversion_rate(m).to_numpy()
    m["market_value_at_cutoff_eur"] = m["player_id"].map(
        _market_value_at_cutoff(market_values, players, cutoff_age)
    )

    # static player attributes (categoricals kept raw for CatBoost)
    for col in ("position", "position_detail", "height_cm", "is_foreigner"):
        if col in players.columns:
            m[col] = m["player_id"].map(players.set_index("player_id")[col])

    assert_no_leakage(feature_columns(m))
    return m


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def assert_matrix_is_clean(df: pd.DataFrame) -> None:
    assert_no_leakage(feature_columns(df))


# raw tmapi competition codes that older crawls stored unmapped -> readable name
_LEAGUE_REMAP = {
    "2DVB": "Vtoraya Liga",
    "R3D1": "Vtoraya Liga",
    "R3D2": "Vtoraya Liga",
    "RJL2": "Russian Youth League",
}


# --- I/O -------------------------------------------------------------
def from_db(engine):
    players = pd.read_sql(
        "select p.player_id, p.canonical_name, p.position, p.position_detail, p.height_cm, "
        "p.is_foreigner, p.academy_club, extract(year from p.birth_date)::int as birth_year, "
        "extract(month from p.birth_date)::int as birth_month "
        "from player p",
        engine,
    )
    seasons = pd.read_sql(
        "select player_id, season, league, club, age_at_season, minutes, matches, "
        "goals, assists, is_rpl from season_stats",
        engine,
    )
    seasons["league"] = seasons["league"].replace(_LEAGUE_REMAP)
    market_values = pd.read_sql("select player_id, date, value_eur from market_value", engine)
    return players, seasons, market_values


def write_parquet(df: pd.DataFrame, path: str | None = None) -> str:
    path = path or (load_settings()["paths"]["data_processed"] + "/features.parquet")
    df.to_parquet(path, index=False)
    return path
