"""ФФ СПб statistics adapter — stat.ffspb.org (Наградион / Nagradion platform).

This is a SEPARATE source for the youngest players (~12–15 y.o.), where
Transfermarkt has nothing. It is collected into its own tables/parquet first and
merged into the main dataset later via :mod:`ingest.resolve`.

Mechanism (no browser needed — plain rate-limited httpx):
  1. GET a page (``/tournament{id}``, ``/tournament{id}/match/{mid}``, …) — it is a
     React shell whose components mount with ``data-block-id`` attributes.
  2. POST ``/_anon/{component}/load_props`` with multipart ``{block_id, ...params}``
     -> JSON props for that component.

Known components / calls (confirmed against tournament 40530 = "Первенство СПб,
мальчики до 14 лет, 2025"):
  * ``match_feed``          {block_id, on_screen, tournaments[]}  -> matches[] (with match urls)
  * ``top_players_block``   {block_id, tournaments[]}             -> top scorers
  * ``tournament_table``    {block_id, tournaments[]}             -> standings

TODO (next chunk, needed for player-level data + merge):
  * match lineups   -> per-team rosters (player id, name, shirt no)
  * player card     -> birth date, position, per-season minutes/goals
  * tournament discovery -> list youth-tournament ids per season / age group
"""

from __future__ import annotations

import re

from ingest.fetcher import HttpFetcher

FFSPB_BASE = "https://stat.ffspb.org"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def _snake(component: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", component).lower()


class Nagradion:
    """Thin client over the ``/_anon/{component}/load_props`` RPC."""

    def __init__(self, fetcher: HttpFetcher | None = None, base: str = FFSPB_BASE):
        self.base = base
        self._f = fetcher or HttpFetcher(user_agent=_UA)

    def page_html(self, path: str) -> str:
        return self._f.get(f"{self.base}/{path.lstrip('/')}").text

    def block_ids(self, html: str) -> dict[str, list[str]]:
        """component name -> [block_id] (order of appearance in the page)."""
        out: dict[str, list[str]] = {}
        for m in re.finditer(
            r'data-block-id="(\d+)"[^>]*?data-component-name="([A-Za-z]+)"'
            r'|data-component-name="([A-Za-z]+)"[^>]*?data-block-id="(\d+)"',
            html,
        ):
            bid = m.group(1) or m.group(4)
            comp = m.group(2) or m.group(3)
            out.setdefault(comp, []).append(bid)
        return out

    def load_props(self, component: str, block_id: str, **params) -> dict:
        self._f.rate_limiter.wait()
        data = {"block_id": str(block_id), **{k: str(v) for k, v in params.items()}}
        r = self._f._client.post(f"{self.base}/_anon/{_snake(component)}/load_props", data=data)
        r.raise_for_status()
        return r.json()


# --- parsers ---------------------------------------------------------
def parse_matches(payload: dict) -> list[dict]:
    """match_feed payload -> [{id, number, home, guest, goals, url, date, finished}]."""
    out = []
    for m in payload.get("matches", []):
        out.append(
            {
                "match_id": m.get("id"),
                "number": m.get("number"),
                "home": m.get("home_team_name"),
                "guest": m.get("guest_team_name"),
                "goals": m.get("goals"),
                "url": m.get("url"),
                "date": m.get("publicDate"),
                "finished": bool(m.get("finished")),
                "tournament_id": m.get("tournament_id"),
            }
        )
    return out


class FfspbSource:
    name = "ffspb"

    def __init__(self, client: Nagradion | None = None):
        self.api = client or Nagradion()

    def tournament_matches(self, tournament_id: int | str) -> list[dict]:
        html = self.api.page_html(f"tournament{tournament_id}")
        blocks = self.api.block_ids(html)
        bids = blocks.get("MatchFeed") or ["16215"]
        payload = self.api.load_props(
            "match_feed", bids[0], on_screen=500, **{"tournaments[]": tournament_id}
        )
        return parse_matches(payload)

    def match_lineups(self, match_id: int | str) -> dict:
        raise NotImplementedError(
            "next chunk: parse the lineup component on /tournament{t}/match/{match_id}"
        )

    def player_card(self, player_id: int | str) -> dict:
        raise NotImplementedError("next chunk: birth date + per-season stats from the player page")

    def discover_youth_tournaments(self, season_year: int) -> list[int]:
        raise NotImplementedError("next chunk: enumerate U11–U15 tournament ids per season")
