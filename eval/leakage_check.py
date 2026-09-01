"""Leakage guardrail (SPEC §7, §8).

Fails loudly if a forbidden (post-hoc / outcome-derived) column reaches the model
matrix. Call ``assert_no_leakage(X.columns)`` right before every ``fit``.
"""

from __future__ import annotations

from collections.abc import Iterable

# Exact column names that must never be used as features.
FORBIDDEN_EXACT: frozenset[str] = frozenset(
    {
        "current_club",
        "club_at_collection",
        "senior_national_team_caps",
        "senior_national_team",
        "market_value_current_eur",
        "market_value_now_eur",
        "last_known_club",  # fine as an outcome marker, never as a feature
    }
)

# Any column containing one of these substrings is suspect.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "_post_cutoff",
    "_after_cutoff",
    "_future",
    "_next_season",
    "outcome_",
    "target",
    "label",
)


class LeakageError(AssertionError):
    pass


def find_leaks(columns: Iterable[str], *, extra_forbidden: Iterable[str] = ()) -> list[str]:
    extra = set(extra_forbidden)
    hits = []
    for c in columns:
        lc = c.lower()
        if c in FORBIDDEN_EXACT or c in extra or any(s in lc for s in FORBIDDEN_SUBSTRINGS):
            hits.append(c)
    return sorted(set(hits))


def assert_no_leakage(columns: Iterable[str], *, extra_forbidden: Iterable[str] = ()) -> None:
    leaks = find_leaks(columns, extra_forbidden=extra_forbidden)
    if leaks:
        raise LeakageError(f"Forbidden / leakage-prone feature(s) in matrix: {leaks}")
