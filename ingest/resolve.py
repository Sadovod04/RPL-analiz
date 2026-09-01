"""Entity resolution: merge records for the same physical player across sources.

Match key: normalized full name + birth date + club + age category. Produces a
stable ``player_id`` and a crosswalk table (source, source_id) -> player_id.

Status: skeleton — implemented in M1a.
"""

from __future__ import annotations

import pandas as pd


def resolve_players(raw_players: pd.DataFrame) -> pd.DataFrame:
    """Return ``raw_players`` with an added stable ``player_id`` column."""
    raise NotImplementedError("M1a")


def build_crosswalk(resolved: pd.DataFrame) -> pd.DataFrame:
    """(source, source_id) -> player_id lookup."""
    raise NotImplementedError("M1a")
