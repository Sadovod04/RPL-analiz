"""FBref adapter (M1b) — ENRICHMENT.

Adds: advanced metrics for senior leagues — xG/xA per 90, progressive
passes/carries, duel win % (SPEC §7.2). Enriches the *adult-career* side of a
record, not the youth-predictor side (FBref youth coverage for RU academies is
near-zero).

FBref (Cloudflare) returns HTTP 403 to plain ``httpx`` from CI. Options:
  * ``curl_cffi`` with a Chrome TLS fingerprint, or a browser context;
  * the ``soccerdata`` package (wraps FBref with caching + throttling).
FBref asks for <= ~1 req / 3 s — keep the rate limiter conservative.

Approach: map player -> FBref id; parse the ``scout`` / ``stats`` tables
(``pandas.read_html`` on the commented-out table blocks FBref ships).
"""

from __future__ import annotations

FBREF_BASE = "https://fbref.com"


class FbrefAdvancedStats:
    name = "fbref"

    def player_stats(self, fbref_id: str) -> list[dict]:
        raise NotImplementedError(
            "M1b: needs curl_cffi/browser/soccerdata (httpx -> 403); "
            "parse standard + advanced stat tables from /en/players/{id}/"
        )
