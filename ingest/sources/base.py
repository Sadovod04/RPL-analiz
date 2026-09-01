"""Common interface every source adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ingest.rate_limiter import RateLimiter


class Source(ABC):
    name: str = "base"

    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        self.rate_limiter = rate_limiter or RateLimiter()

    @abstractmethod
    def iter_academy_players(self, academy_ref: str) -> Iterator[str]:
        """Yield source-local player ids/refs for one academy."""

    @abstractmethod
    def fetch_player(self, player_ref: str) -> dict:
        """Return raw structured records for one player.

        Expected keys: ``player`` (Player), ``seasons`` (list[SeasonStats]),
        ``transfers`` (list[Transfer]), ``caps`` (list[NationalTeamCap]).
        """
