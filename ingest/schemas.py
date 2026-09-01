"""Canonical raw-data schemas shared by every source adapter.

Adapters parse their source into these pydantic models; ``resolve.py`` then
merges records that refer to the same physical player across sources.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Position(str, Enum):
    GK = "GK"
    CB = "CB"
    FB = "FB"
    CM = "CM"
    W = "W"
    ST = "ST"
    UNKNOWN = "UNKNOWN"


class Player(BaseModel):
    source: str
    source_id: str
    full_name: str
    birth_date: date | None = None
    position: Position = Position.UNKNOWN
    height_cm: int | None = None
    weight_kg: int | None = None
    nationality: str | None = None
    is_foreigner: bool | None = None
    academy_club: str | None = None


class SeasonStats(BaseModel):
    source: str
    source_player_id: str
    season: str = Field(description="e.g. '2019/2020'")
    age_at_season: float | None = None
    club: str | None = None
    league: str | None = None
    age_bucket: str | None = Field(default=None, description="U13/U15/U17/U19/U21 when applicable")
    minutes: int | None = None
    matches: int | None = None
    goals: int | None = None
    assists: int | None = None
    is_rpl: bool = False


class Transfer(BaseModel):
    source: str
    source_player_id: str
    date: date | None = None
    from_club: str | None = None
    to_club: str | None = None
    fee_eur: float | None = None
    market_value_eur: float | None = None
    market_value_as_of: date | None = None


class NationalTeamCap(BaseModel):
    source: str
    source_player_id: str
    team: str = Field(description="e.g. 'Russia U17'")
    level: str = Field(description="youth | u21 | senior")
    date: date | None = None
    caps: int | None = None
