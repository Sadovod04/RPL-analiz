"""ФФ СПб statistics adapter — stat.ffspb.org (Наградион / Nagradion platform).

Separate source for the youngest players (~10–16 y.o.) that Transfermarkt lacks.
Collected into its own parquet first, merged into the main dataset later via
:mod:`ingest.resolve` (full name + birth date + club).

All data is plain httpx (no browser, no WAF):
  * ``/calendar`` -> every tournament id + name (filter by age).
  * ``POST /_anon/match_feed/load_props`` {block_id, on_screen, tournaments[]}
    -> matches[] (each with a match-page url).
  * ``/tournament{tid}/match/{mid}`` -> inline ``renderComponent(.., 'GameProtocolBlock', {...})``
    -> host/guestProtocol.protocol.start_players + substitute_players -> per player:
    full_name, number, url = /tournament{tid}/player/{tsid}.
  * ``/tournament{tid}/player/{tsid}`` -> server-rendered ``.person-info__title`` + a
    ``<table class="table">`` with «Дата рождения» / «Возраст», plus inline
    ``PlayerStats`` props (tournament history).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from ingest.fetcher import HttpFetcher

FFSPB_BASE = "https://stat.ffspb.org"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
_MATCH_FEED_BLOCK = "16215"

# tournament-name patterns that mark a youth competition
_AGE_RE = re.compile(r"до\s*(\d{1,2})\s*лет|мальчики\s*(20\d\d)", re.IGNORECASE)


def _snake(component: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", component).lower()


def render_props(html: str, component: str) -> dict | None:
    """Pull the inline ``renderComponent("uuid", 'Component', {JSON})`` props object."""
    m = re.search(
        r'renderComponent\(\s*"[^"]+"\s*,\s*[\'"]' + re.escape(component) + r'[\'"]\s*,\s*',
        html,
    )
    if not m:
        return None
    try:  # raw_decode reads exactly one JSON value (string-aware) from the offset
        obj, _ = json.JSONDecoder().raw_decode(html, m.end())
        return obj
    except json.JSONDecodeError:
        return None


class Nagradion:
    def __init__(self, fetcher: HttpFetcher | None = None, base: str = FFSPB_BASE):
        self.base = base
        self._f = fetcher or HttpFetcher(user_agent=_UA)

    def page_html(self, path: str) -> str:
        return self._f.get(f"{self.base}/{path.lstrip('/')}").text

    def load_props(self, component: str, block_id: str, **params) -> dict:
        self._f.rate_limiter.wait()
        data = {"block_id": str(block_id), **{k: str(v) for k, v in params.items()}}
        r = self._f._client.post(f"{self.base}/_anon/{_snake(component)}/load_props", data=data)
        r.raise_for_status()
        return r.json()


# --- parsers ---------------------------------------------------------
def parse_calendar(html: str, max_age: int = 16) -> list[dict]:
    """-> [{tournament_id, name, age}] for youth competitions only."""
    soup = BeautifulSoup(html, "lxml")
    out, seen = [], set()
    for a in soup.select('a[href*="/tournament"]'):
        m = re.search(r"/tournament(\d{4,7})", a.get("href", ""))
        if not m:
            continue
        tid = m.group(1)
        name = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
        if not name or tid in seen:
            continue
        am = _AGE_RE.search(name)
        if not am:
            continue
        age = int(am.group(1)) if am.group(1) else datetime.now().year - int(am.group(2))
        if age <= max_age:
            seen.add(tid)
            out.append({"tournament_id": tid, "name": name, "age": age})
    return out


def parse_matches(payload: dict) -> list[dict]:
    out = []
    for m in payload.get("matches", []):
        out.append(
            {
                "match_id": m.get("id"),
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


def _lineup_side(protocol: dict, side: str) -> list[dict]:
    rows = []
    for group in ("start_players", "substitute_players"):
        for p in protocol.get(group) or []:
            m = re.search(r"/tournament(\d+)/player/(\d+)", p.get("url", ""))
            if not m:
                continue
            rows.append(
                {
                    "tournament_id": m.group(1),
                    "ffspb_id": m.group(2),  # tournament-scoped player id
                    "full_name": p.get("full_name"),
                    "number": p.get("number"),
                    "side": side,
                    "started": group == "start_players",
                }
            )
    return rows


def parse_lineup(game_protocol: dict) -> list[dict]:
    """GameProtocolBlock props -> flat list of players (both teams)."""
    out = []
    for key, side in (("hostProtocol", "home"), ("guestProtocol", "away")):
        proto = (game_protocol.get(key) or {}).get("protocol") or {}
        out.extend(_lineup_side(proto, side))
    return out


def _parse_ru_date(text: str | None) -> date | None:
    if not text:
        return None
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", text)
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def parse_player_profile(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = soup.select_one(".person-info__title")
    name = title.get_text(" ", strip=True) if title else None

    info: dict[str, str] = {}
    for tr in soup.select(".person-info table tr, .person-info__table tr, table.table tr"):
        th, td = tr.find("th"), tr.find("td")
        if th and td:
            info[th.get_text(strip=True).lower()] = td.get_text(" ", strip=True)

    stats = render_props(html, "PlayerStats") or {}
    tournaments = []
    for sport in stats.get("sports", []):
        for row in sport.get("rows", []):
            trn = row.get("tournament") or {}
            team = row.get("team") or {}
            tournaments.append(
                {
                    "season": row.get("seasonName"),
                    "sport": sport.get("name"),
                    "tournament_id": trn.get("id"),
                    "tournament_name": trn.get("name"),
                    "team": team.get("name"),
                    "details_url": row.get("detailsUrl"),
                }
            )

    return {
        "full_name": name,
        "birth_date": _parse_ru_date(info.get("дата рождения")),
        "age_text": info.get("возраст"),
        "tournaments": tournaments,
    }


# --- source ----------------------------------------------------------
class FfspbSource:
    name = "ffspb"

    def __init__(self, client: Nagradion | None = None):
        self.api = client or Nagradion()

    def discover_youth_tournaments(self, max_age: int = 16) -> list[dict]:
        return parse_calendar(self.api.page_html("calendar"), max_age=max_age)

    def tournament_matches(self, tournament_id: int | str) -> list[dict]:
        payload = self.api.load_props(
            "match_feed", _MATCH_FEED_BLOCK, on_screen=1000, **{"tournaments[]": tournament_id}
        )
        return parse_matches(payload)

    def match_lineup(self, tournament_id: int | str, match_id: int | str) -> list[dict]:
        html = self.api.page_html(f"tournament{tournament_id}/match/{match_id}")
        gp = render_props(html, "GameProtocolBlock")
        return parse_lineup(gp) if gp else []

    def player_profile(self, tournament_id: int | str, ffspb_id: int | str) -> dict:
        html = self.api.page_html(f"tournament{tournament_id}/player/{ffspb_id}")
        prof = parse_player_profile(html)
        prof["ffspb_id"] = str(ffspb_id)
        return prof
