"""youfl.ru adapter — DEFERRED to M1b.

youfl.ru (official РФС Юношеская футбольная лига site) rejects the TLS handshake
from non-RU egress (geo / anti-bot) and is unreachable from CI. Youth-league
context in M1a comes from :mod:`ingest.sources.wikipedia` instead.

When run from a Russian network this adapter can be implemented against
youfl.ru's season / tour / squad pages (httpx + BeautifulSoup) — the interface
below matches the other sources so it drops into ``run_ingest`` unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator

from ingest.sources.base import Source

_UNREACHABLE = (
    "youfl.ru is unreachable from this environment (TLS rejected — geo/anti-bot). "
    "Deferred to M1b; use the 'wikipedia' source for youth-league context. "
    "See SPEC.md §5 and ingest/sources/youfl.py."
)


class Youfl(Source):
    name = "youfl"

    def iter_academy_players(self, academy_ref: str) -> Iterator[str]:
        raise NotImplementedError(_UNREACHABLE)

    def fetch_player(self, player_ref: str) -> dict:
        raise NotImplementedError(_UNREACHABLE)
