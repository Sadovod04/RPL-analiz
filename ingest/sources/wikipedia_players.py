"""ru.wikipedia player-bio adapter (Phase B / "recognition").

For each resolved player, look for a ru.wikipedia footballer article and pull
three *recognition* signals, all of which are meaningful only **before** the
modelling cutoff age (see ``features.build_features._recognition_features``):

* ``article_created_age`` — the player's age when the article was first created.
  An encyclopedia article about a 17-year-old is an independent "notable young"
  signal. The article existing *now* is post-hoc and is **not** used as a
  feature; only "created before the cutoff age" is.
* ``youth_honours_count`` — honours in the "Достижения" section whose year is
  ``<= birth_year + 18``.
* ``nt_youth_levels`` — U-levels of the player's Russia youth-NT categories
  (молодёжная = 21). The feature layer keeps only levels <= 19 as pre-cutoff.

Name matching: our ``name_home_country`` carries a patronymic (sometimes wrong);
ru.wikipedia titles often don't. Search / match on surname + given name only,
then accept a candidate only if it (a) is in a "Футболисты…" category and (b)
lines up on birth year (intro text or "Родившиеся в YYYY году" category).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from rapidfuzz import fuzz

from ingest.rate_limiter import RateLimiter
from ingest.resolve import normalize_name

WIKI_API = "https://ru.wikipedia.org/w/api.php"
WIKI_UA = "RPL-analiz/0.1 (research, non-commercial; MediaWiki API read-only)"

# token_set_ratio so a patronymic present on one side but not the other
# ("Пиняев Сергей Андреевич" vs title "Пиняев, Сергей") doesn't tank the score;
# the birth-year-in-intro and footballer-category gates handle namesakes.
_TITLE_MATCH_MIN = 90.0
_FOOTBALLER_CAT = "Футболист"
# categories like "Игроки сборной России по футболу (до 19 лет)" /
# "Игроки молодёжной сборной России по футболу" -> capture the U-level (молодёжная = 21).
# The feature layer decides which levels count as pre-cutoff (see build_features).
_NT_RU_RE = re.compile(r"сборной России по футболу", re.I)
_NT_LEVEL_RE = re.compile(r"до (\d+)\s*(?:лет|года|годов)", re.I)
_NT_MOLODEZH_RE = re.compile(r"молодёжн\w*\s+сборной", re.I)
_HONOURS_HEAD_RE = re.compile(
    r"==+\s*(?:Достижения|Награды|Титулы)\s*==+(.+?)(?:\n==[^=]|\Z)", re.S
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
# patronymic endings (Ivan-o-VICH, Ivan-OVNA, Ali kyzy / ogly) — our TM
# name_home_country carries a patronymic (sometimes wrong); wiki titles often
# don't. Drop it and match/search on surname + given name only.
_PATRONYMIC_RE = re.compile(r"(вич|вна|ична|оглы|оглу|кызы|гызы|уулу)$", re.I)


def _search_name(name_home: str | None, canonical: str) -> str:
    """Surname + given name, patronymic removed. Cyrillic if we have it."""
    raw = (name_home or "").strip()
    if raw:
        toks = raw.split()
        if len(toks) >= 3 and _PATRONYMIC_RE.search(toks[-1]):
            toks = toks[:-1]
        return " ".join(toks[:2])
    # romanised fallback ("Last, First" or "First Last")
    c = canonical.strip()
    if "," in c:
        last, _, first = c.partition(",")
        return f"{first.strip()} {last.strip()}"
    return " ".join(c.split()[:2])


@dataclass(frozen=True)
class WikiBio:
    player_id: str
    wiki_title: str | None  # None -> searched, no confident match
    match_score: float
    article_created: date | None
    article_created_age: float | None
    youth_honours_count: int
    nt_youth_levels: list[int]  # U-levels of Russia youth-NT categories (молодёжная = 21)
    honours_years: list[int]

    @classmethod
    def not_found(cls, player_id: str) -> WikiBio:
        return cls(player_id, None, 0.0, None, None, 0, [], [])


def _nt_youth_levels(categories: list[str]) -> list[int]:
    levels: set[int] = set()
    for c in categories:
        if not _NT_RU_RE.search(c):
            continue
        for lvl in _NT_LEVEL_RE.findall(c):
            levels.add(int(lvl))
        if _NT_MOLODEZH_RE.search(c):
            levels.add(21)
    return sorted(levels)


def _parse_ts(ts: str | None) -> date | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _honour_years(wikitext: str) -> list[int]:
    m = _HONOURS_HEAD_RE.search(wikitext or "")
    if not m:
        return []
    seg = "\n".join(
        ln for ln in m.group(1).splitlines() if ln.strip().startswith(("*", "#"))
    )
    return sorted({int(y) for y in _YEAR_RE.findall(seg)})


class WikiPlayerBios:
    """Look up ru.wikipedia footballer articles. One instance per crawl."""

    name = "wikipedia_players"

    def __init__(self, fetcher=None, rate_limiter: RateLimiter | None = None):
        if fetcher is None:
            from ingest.fetcher import HttpFetcher

            fetcher = HttpFetcher(
                rate_limiter=rate_limiter or RateLimiter(min_interval=1.3, jitter=0.6),
                user_agent=WIKI_UA,
            )
        self._f = fetcher

    # -- raw API calls ---------------------------------------------------
    def _api(self, **params) -> dict:
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        params.setdefault("maxlag", 5)  # MediaWiki etiquette: back off when servers are busy
        return self._f.get(WIKI_API, params=params).json()

    def _search(self, query: str, limit: int = 4) -> list[str]:
        j = self._api(action="query", list="search", srsearch=query, srlimit=limit, srprop="")
        return [h["title"] for h in j.get("query", {}).get("search", [])]

    def _page(self, title: str) -> dict:
        """First-revision date, categories, intro plaintext — one call."""
        j = self._api(
            action="query",
            titles=title,
            redirects=1,
            prop="revisions|categories|extracts",
            rvlimit=1,
            rvdir="newer",
            rvprop="timestamp",
            cllimit="max",
            clshow="!hidden",
            explaintext=1,
            exintro=1,
            exchars=800,
        )
        pages = j.get("query", {}).get("pages", [])
        return pages[0] if pages else {}

    def _wikitext(self, title: str) -> str:
        j = self._api(action="parse", page=title, prop="wikitext", redirects=1)
        wt = j.get("parse", {}).get("wikitext", "")
        return wt if isinstance(wt, str) else wt.get("*", "")

    # -- main ----------------------------------------------------------
    def lookup(
        self,
        player_id: str,
        name_home: str | None,
        canonical: str,
        birth_date: date | None,
    ) -> WikiBio:
        birth_year = birth_date.year if birth_date else None
        sname = _search_name(name_home, canonical)
        if not sname:
            return WikiBio.not_found(player_id)
        want = normalize_name(sname)
        year_cat = f"Родившиеся в {birth_year} году" if birth_year else None

        # search ranks by relevance, so take the first candidate that clears the
        # fuzzy threshold AND the footballer + birth-year gates (no need to page
        # every hit).
        best: tuple[float, dict, str] | None = None
        for title in self._search(f"{sname} футболист"):
            score = fuzz.token_set_ratio(want, normalize_name(title))
            if score < _TITLE_MATCH_MIN:
                continue
            pg = self._page(title)
            cats = [c.get("title", "") for c in pg.get("categories", [])]
            intro = pg.get("extract", "") or ""
            is_footballer = any(_FOOTBALLER_CAT in c for c in cats)
            # birth year must line up (guards namesakes): in the intro text OR the
            # "Родившиеся в YYYY году" category. Unknown birth year -> skip the gate.
            year_ok = birth_year is None or (
                str(birth_year) in intro or any(year_cat == c for c in cats)
            )
            if is_footballer and year_ok:
                best = (score, pg, title)
                break

        if best is None:
            return WikiBio.not_found(player_id)

        score, pg, title = best
        created = _parse_ts((pg.get("revisions") or [{}])[0].get("timestamp"))
        created_age = (
            round((created - birth_date).days / 365.25, 2)
            if created and birth_date
            else None
        )
        cats = [c.get("title", "") for c in pg.get("categories", [])]
        years = _honour_years(self._wikitext(title))
        youth_years = [y for y in years if birth_year and y <= birth_year + 18]
        return WikiBio(
            player_id=player_id,
            wiki_title=title,
            match_score=float(score),
            article_created=created,
            article_created_age=created_age,
            youth_honours_count=len(youth_years),
            nt_youth_levels=_nt_youth_levels(cats),
            honours_years=years,
        )
