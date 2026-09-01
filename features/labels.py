"""Target construction (SPEC §3).

Primary binary target, secondary ordinal target, and the survival tuple.
Thresholds come from ``config/settings.toml`` -> ``[target]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from settings import load_settings

CENSORED = -1  # binary target value for "outcome still open"

# competitions that count as "reached a professional level" for the ordinal target
PRO_LEAGUE_NAMES = {"Premier Liga", "Pervaya Liga", "Vtoraya Liga", "Premier Liga (relegation)"}


@dataclass(frozen=True)
class LabelConfig:
    rpl_minutes_threshold: int = 200
    settled_age: int = 26

    @classmethod
    def from_settings(cls) -> LabelConfig:
        t = load_settings()["target"]
        return cls(rpl_minutes_threshold=t["rpl_minutes_threshold"], settled_age=t["settled_age"])


def binary_target(
    rpl_minutes_ever: float, current_age: float, cfg: LabelConfig | None = None
) -> int:
    """1 = broke through; 0 = settled non-breakthrough; CENSORED = still open.

    - 1  if career RPL minutes >= threshold
    - 0  if below threshold AND current age >= settled_age
    - CENSORED otherwise (feeds the survival model only)
    """
    cfg = cfg or LabelConfig.from_settings()
    if rpl_minutes_ever >= cfg.rpl_minutes_threshold:
        return 1
    if current_age >= cfg.settled_age:
        return 0
    return CENSORED


# Secondary ordinal target levels (SPEC §3)
ORDINAL_LEVELS = ("none", "lower_leagues", "rpl")


def ordinal_target(
    rpl_minutes_ever: float, reached_pro_level: bool, cfg: LabelConfig | None = None
) -> str:
    cfg = cfg or LabelConfig.from_settings()
    if rpl_minutes_ever >= cfg.rpl_minutes_threshold:
        return "rpl"
    if reached_pro_level:
        return "lower_leagues"
    return "none"


def survival_tuple(rpl_debut_age: float | None, current_age: float) -> tuple[float, int]:
    """(duration, event_observed) for lifelines/pycox.

    duration = age at RPL debut if it happened, else current age (right-censored).
    """
    if rpl_debut_age is not None:
        return float(rpl_debut_age), 1
    return float(current_age), 0


# --- DataFrame-level ----------------------------------------------------
def _current_age(birth_year: float, as_of_year: int) -> float:
    return as_of_year - birth_year if pd.notna(birth_year) else float("nan")


def attach_labels(
    player_df: pd.DataFrame,
    seasons_df: pd.DataFrame,
    *,
    as_of_year: int,
    cfg: LabelConfig | None = None,
) -> pd.DataFrame:
    """Add target / ordinal_target / duration / event_observed / rpl_minutes_ever.

    ``player_df``  : player_id, birth_year
    ``seasons_df`` : player_id, season, league, minutes, is_rpl, age_at_season
    """
    cfg = cfg or LabelConfig.from_settings()
    s = seasons_df.copy()
    s["minutes"] = s["minutes"].fillna(0)

    rpl = s[s["is_rpl"]]
    rpl_minutes = rpl.groupby("player_id")["minutes"].sum()
    debut_age = rpl[rpl["minutes"] > 0].groupby("player_id")["age_at_season"].min()
    pro_ids = set(s.loc[s["league"].isin(PRO_LEAGUE_NAMES) & (s["minutes"] > 0), "player_id"])

    out = player_df.copy()
    out["rpl_minutes_ever"] = out["player_id"].map(rpl_minutes).fillna(0.0)
    out["rpl_debut_age"] = out["player_id"].map(debut_age)
    out["reached_pro_level"] = out["player_id"].isin(pro_ids)
    out["current_age"] = out["birth_year"].map(lambda by: _current_age(by, as_of_year))

    out["target"] = [
        binary_target(m, a, cfg)
        for m, a in zip(out["rpl_minutes_ever"], out["current_age"], strict=True)
    ]
    out["ordinal_target"] = [
        ordinal_target(m, bool(p), cfg)
        for m, p in zip(out["rpl_minutes_ever"], out["reached_pro_level"], strict=True)
    ]
    dur_evt = [
        survival_tuple(None if pd.isna(d) else d, a)
        for d, a in zip(out["rpl_debut_age"], out["current_age"], strict=True)
    ]
    out["duration"] = [d for d, _ in dur_evt]
    out["event_observed"] = [e for _, e in dur_evt]
    return out
