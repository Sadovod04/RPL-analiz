"""Target construction (SPEC §3).

Primary binary target, secondary ordinal target, and the survival tuple.
Thresholds come from ``config/settings.toml`` -> ``[target]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from settings import load_settings

CENSORED = -1  # binary target value for "outcome still open"


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
