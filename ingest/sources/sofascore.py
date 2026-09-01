"""Sofascore adapter (M1b) — ENRICHMENT.

Adds: per-match player ratings (a scouting-consensus signal). Sofascore has an
undocumented JSON API (``api.sofascore.com/api/v1/...``) but fronts it with bot
protection: plain ``httpx`` gets HTTP 403 from CI. Needs TLS impersonation
(``curl_cffi``) or a browser context, plus careful rate limiting and ToS respect
— do not redistribute raw payloads (SPEC §5).

Approach: resolve player -> Sofascore id via search endpoint; pull
``/player/{id}/statistics/seasons`` and per-tournament rating aggregates.

Low priority — youth-match rating coverage for RU academies is thin.
"""

from __future__ import annotations

SOFASCORE_API = "https://api.sofascore.com/api/v1"


class SofascoreRatings:
    name = "sofascore"

    def player_season_ratings(self, sofascore_id: int) -> list[dict]:
        raise NotImplementedError(
            "M1b: needs curl_cffi/browser (httpx -> 403); "
            "GET /player/{id}/statistics/seasons then per-tournament rating"
        )
