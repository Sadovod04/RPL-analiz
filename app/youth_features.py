"""Bring the regional youth pools (``<source>_players.parquet``: ФФ СПб, Москва …)
into the main feature schema so the kids show up in the same tables / filters /
compare as the Transfermarkt players.

They have no career yet, so their outcome labels are all ``CENSORED`` (they can
never enter the training set) and most model features are missing. Their score in
the dashboard is the transparent 0–100 heuristic (:func:`youth_frame` -> ``pers_score``),
not the CatBoost model.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features.build_features import feature_columns
from features.labels import CENSORED

# regional youth parquets live next to features.parquet as "<source>_players.parquet"
YOUTH_SOURCES = ("ffspb", "mosff")

_STRONG = ("зенит", "спартак", "цска", "краснодар", "локомотив", "динамо", "чертаново", "рубин")


def _club_tier(teams: str) -> float:
    low = str(teams).lower()
    if any(k in low for k in _STRONG):
        return 1.0
    if "сшор" in low or "сш " in low or "спортивн" in low:
        return 0.6
    return 0.3


def _primary_club(teams: str) -> str | None:
    parts = [p for p in str(teams).split(";") if p]
    if not parts:
        return None
    strong = [p for p in parts if any(k in p.lower() for k in _STRONG)]
    return (strong or sorted(parts, key=len, reverse=True))[0]


def _pid(name: str, dob) -> str:
    return "youth_" + hashlib.sha1(f"{name}|{dob}".encode()).hexdigest()[:12]


def youth_paths(processed_dir: Path) -> list[Path]:
    return [
        p for s in YOUTH_SOURCES if (p := Path(processed_dir) / f"{s}_players.parquet").exists()
    ]


def _read_youth_parquet(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    src = path.stem.replace("_players", "")
    if "source" not in d.columns:
        d["source"] = src
    # birth_year: explicit column wins (mosff has no DOB), else derive from birth_date
    if "birth_year" not in d.columns or d["birth_year"].isna().all():
        d["birth_year"] = pd.to_datetime(d.get("birth_date"), errors="coerce").dt.year
    for c in ("patronymic", "birth_date", "minutes", "n_tournaments"):
        if c not in d.columns:
            d[c] = pd.NA
    return d


def youth_frame(paths: Path | list[Path]) -> pd.DataFrame:
    """One deduped row per kid across every regional source, with games/goals/gpg
    + ``pers_score`` / ``proj_level``."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    d = pd.concat([_read_youth_parquet(Path(p)) for p in paths], ignore_index=True)
    d["birth_year"] = pd.to_numeric(d["birth_year"], errors="coerce").astype("Int64")
    agg = (
        d.groupby(["full_name", "patronymic", "birth_year"], dropna=False)
        .agg(
            birth_date=("birth_date", "first"),
            source=("source", lambda s: ";".join(sorted({str(x) for x in s if pd.notna(x)}))),
            teams=(
                "teams",
                lambda s: ";".join(sorted({x for v in s for x in str(v).split(";") if x})),
            ),
            n_tournaments=("n_tournaments", "max"),
            games=("games", "max"),
            goals=("goals", "max"),
            minutes=("minutes", "max"),
        )
        .reset_index()
    )
    agg["gpg"] = (agg["goals"] / agg["games"].replace(0, 1)).round(2)

    # percentiles WITHIN each source — ффспб rows carry career totals, mosff rows a
    # single season, so a global rank would just reward "more games logged".
    grp = agg.groupby(agg["source"].str.split(";").str[0])
    gpg_pct = grp["gpg"].rank(pct=True)
    games_pct = grp["games"].rank(pct=True)
    tier = agg["teams"].map(_club_tier)
    agg["pers_score"] = (100 * (0.45 * gpg_pct + 0.30 * games_pct + 0.25 * tier)).round(0)
    agg["proj_level"] = pd.cut(
        agg["pers_score"], [-1, 40, 60, 80, 101], labels=["yl_low", "yl_fnl2", "yl_fnl", "yl_rpl"]
    ).astype(str)
    return agg


def youth_feature_rows(paths: Path | list[Path], template: pd.DataFrame) -> pd.DataFrame:
    """Map :func:`youth_frame` onto ``template``'s columns (the main feature schema).

    Returns rows with ``source`` in {ffspb, mosff, …}, all labels ``CENSORED``,
    model features mostly NaN, ``pers_score`` / ``proj_level`` carried through.
    """
    y = youth_frame(paths)
    n = len(y)
    # start every model feature as float NaN so an all-missing column stays numeric
    out = pd.DataFrame(
        {c: np.full(n, np.nan, dtype="float64") for c in feature_columns(template)},
        index=range(n),
    )
    for c in template.columns:
        if c not in out:
            out[c] = pd.NA

    out["player_id"] = [
        _pid(nm, by) for nm, by in zip(y["full_name"], y["birth_year"], strict=True)
    ]
    out["canonical_name"] = y["full_name"].to_numpy()
    out["birth_year"] = y["birth_year"].astype("Int64").to_numpy()
    out["academy_club"] = y["teams"].map(_primary_club).to_numpy()
    out["position"] = None
    out["position_detail"] = None
    out["is_foreigner"] = 0.0

    # youth stats we do have -> the general (non age-bucketed) youth features
    out["youth_seasons"] = y["n_tournaments"].fillna(0).astype("float64").to_numpy()
    out["youth_minutes_total"] = y["minutes"].fillna(0).astype("float64").to_numpy()
    out["youth_goals_total"] = y["goals"].fillna(0).astype("float64").to_numpy()
    mins = y["minutes"].to_numpy(dtype="float64")
    goals = y["goals"].to_numpy(dtype="float64")
    ga90 = np.divide(goals, mins / 90.0, out=np.full(n, np.nan), where=mins > 0)
    ga90 = np.where(mins > 0, ga90, y["gpg"].to_numpy(dtype="float64"))
    out["youth_ga_per90"] = np.round(ga90, 3)
    out["played_youth_league"] = 0.0
    out["best_level_pre_cutoff"] = 0.0

    # labels: not resolved -> censored (never trains, always in the "prospects" pool)
    for c in ("target", "pro_target", "ordinal_target"):
        if c in out:
            out[c] = CENSORED
    out["outcome_level"] = None
    out["reached_pro_level"] = False
    out["rpl_minutes_ever"] = 0
    out["event_observed"] = False
    out["current_age"] = (2026 - y["birth_year"].astype("float")).to_numpy()

    # "ffspb", "mosff", or "ffspb;mosff" if a kid shows up in both regions
    out["source"] = y["source"].str.split(";").str[0].to_numpy()
    out["pers_score"] = y["pers_score"].to_numpy()
    out["proj_level"] = y["proj_level"].to_numpy()

    extras = [c for c in ("source", "pers_score", "proj_level") if c not in template.columns]
    return out[list(template.columns) + extras]


def combined_frame(features_path: Path, youth: Path | list[Path] | None) -> pd.DataFrame:
    """Transfermarkt feature matrix + every regional youth pool, one table.

    ``youth`` may be the processed-data directory (auto-discovers
    ``<source>_players.parquet``), an explicit list of parquet paths, or None.
    """
    base = pd.read_parquet(features_path)
    if "source" not in base.columns:
        base = base.assign(source="tm")
    if "pers_score" not in base.columns:
        base = base.assign(pers_score=np.nan, proj_level=pd.NA)

    if youth is None:
        return base
    if isinstance(youth, (str, Path)) and Path(youth).is_dir():
        paths = youth_paths(Path(youth))
    else:
        paths = [Path(p) for p in ([youth] if isinstance(youth, (str, Path)) else youth)]
        paths = [p for p in paths if p.exists()]
    if not paths:
        return base

    kids = youth_feature_rows(paths, base).reindex(columns=base.columns)
    with warnings.catch_warnings():
        # kids legitimately have all-NA columns (age-bucket minutes, market value,
        # career labels the kids can't have yet); the resulting dtypes are correct.
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat([base, kids], ignore_index=True)
