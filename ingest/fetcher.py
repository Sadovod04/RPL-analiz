"""Browser-based fetch layer (M1a).

Transfermarkt sits behind an AWS WAF JS challenge and renders its data grids in
web-component shadow DOM, so plain ``httpx`` cannot read it. This module drives a
real headless Chromium via Playwright:

* solves the WAF challenge (real JS engine),
* dismisses the consent modal,
* waits for hydration and returns the rendered HTML,
* captures JSON responses from Transfermarkt's ``tmapi`` XHRs (the grids are
  populated from these — parsing the JSON is far more robust than the DOM),
* throttles every navigation through :class:`ingest.rate_limiter.RateLimiter`.

``httpx``-only sources (Wikipedia) use :class:`HttpFetcher` instead.

Chromium is not vendored: run ``uv run playwright install chromium`` once.
"""

from __future__ import annotations

import json
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ingest.rate_limiter import RateLimiter
from settings import load_settings

if TYPE_CHECKING:
    from collections.abc import Iterator

# tmapi hosts / paths whose JSON responses we want to keep
_TMAPI_MARKERS = ("tmapi", "/ceapi/", "transfermarkt.technology/api")


class WafChallengeError(RuntimeError):
    """Page still shows the AWS WAF challenge after load."""


class HttpFetcher:
    """Thin rate-limited ``httpx`` client for sources without bot protection."""

    def __init__(self, rate_limiter: RateLimiter | None = None, user_agent: str | None = None):
        cfg = load_settings()
        self.rate_limiter = rate_limiter or RateLimiter.from_config(cfg)
        ua = user_agent or cfg["scrape"]["user_agent"]
        self._client = httpx.Client(
            headers={"User-Agent": ua, "Accept-Language": "ru,en;q=0.8"},
            timeout=30.0,
            follow_redirects=True,
        )

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, max=30),
        reraise=True,
    )
    def get(self, url: str, **kwargs) -> httpx.Response:
        self.rate_limiter.wait()
        resp = self._client.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    def get_json(self, url: str, **kwargs) -> dict:
        return self.get(url, **kwargs).json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpFetcher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TmApiClient(HttpFetcher):
    """Transfermarkt's ``tmapi`` JSON API.

    Discovered from the site's own XHRs; served from
    ``tmapi.transfermarkt.technology`` **without** the AWS WAF that guards
    ``www.transfermarkt.com``, so plain rate-limited HTTP is enough here. This is
    the primary data path — the browser is only needed for historical squad
    (``kader``) pages.
    """

    BASE = "https://tmapi.transfermarkt.technology"

    def players(self, *ids: str | int) -> dict:
        q = "&".join(f"ids[]={i}" for i in ids)
        return self.get_json(f"{self.BASE}/players?{q}")

    def clubs(self, *ids: str | int) -> dict:
        q = "&".join(f"ids[]={i}" for i in ids)
        return self.get_json(f"{self.BASE}/clubs?{q}")

    def performance_game(self, player_id: str | int) -> dict:
        return self.get_json(f"{self.BASE}/player/{player_id}/performance-game")

    def club_squad(self, club_id: str | int) -> dict:
        return self.get_json(f"{self.BASE}/club/{club_id}/squad")

    def competition_table(self, competition_id: str, season_id: int) -> dict:
        return self.get_json(f"{self.BASE}/competition/{competition_id}/table?seasonId={season_id}")


@dataclass
class BrowserResult:
    url: str
    html: str
    captured_json: dict[str, object] = field(default_factory=dict)  # request URL -> parsed JSON


class BrowserFetcher:
    """Headless-Chromium fetcher for Transfermarkt.

    Usage::

        with BrowserFetcher() as bf:
            res = bf.get(url, wait_for="tm-player-performance-table-new")
            res.html            # hydrated DOM
            res.captured_json   # {xhr_url: {...}} from tmapi
    """

    def __init__(self, rate_limiter: RateLimiter | None = None, headless: bool = True):
        cfg = load_settings()
        self.rate_limiter = rate_limiter or RateLimiter.from_config(cfg)
        self._ua = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        self._headless = headless
        self._pw = None
        self._browser = None
        self._ctx = None

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> BrowserFetcher:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._ctx = self._browser.new_context(
            user_agent=self._ua,
            locale="ru-RU",
            viewport={"width": 1440, "height": 900},
        )
        # skip Transfermarkt consent wall
        self._ctx.add_cookies(
            [{"name": "eupubconsent-v2", "value": "1", "domain": ".transfermarkt.com", "path": "/"}]
        )
        return self

    def __exit__(self, *exc) -> None:
        for closer in (self._ctx, self._browser):
            with suppress(Exception):  # best-effort teardown
                closer and closer.close()
        if self._pw:
            self._pw.stop()

    # -- fetch -------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(WafChallengeError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, max=30),
        reraise=True,
    )
    def get(self, url: str, wait_for: str | None = None, settle_ms: int = 2500) -> BrowserResult:
        if self._ctx is None:
            raise RuntimeError("use BrowserFetcher as a context manager")
        self.rate_limiter.wait()

        captured: dict[str, object] = {}
        page = self._ctx.new_page()

        def _on_response(resp) -> None:
            if any(m in resp.url for m in _TMAPI_MARKERS):
                with suppress(Exception):  # non-JSON / streamed
                    captured[resp.url] = resp.json()

        page.on("response", _on_response)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            self._dismiss_consent(page)
            if wait_for:
                # Playwright selectors pierce shadow DOM by default
                with suppress(Exception):  # fall through to settle
                    page.wait_for_selector(wait_for, timeout=15_000, state="attached")
            page.wait_for_timeout(settle_ms)
            html = page.content()
            if "awswaf" in html and "challenge-container" in html:
                raise WafChallengeError(url)
            return BrowserResult(url=url, html=html, captured_json=captured)
        finally:
            page.close()

    @staticmethod
    def _dismiss_consent(page) -> None:
        for sel in (
            "button[aria-label='Accept & continue']",
            "text=Accept & continue",
            ".sp_choice_type_11",
            "#onetrust-accept-btn-handler",
        ):
            try:
                loc = page.locator(sel)
                if loc.count():
                    loc.first.click(timeout=2000)
                    return
            except Exception:  # noqa: BLE001
                continue


# ---------------------------------------------------------------------------
@contextmanager
def capture_fixtures(out_dir: str | Path) -> Iterator[callable]:
    """Helper for building test fixtures.

    ::

        with capture_fixtures("tests/fixtures/transfermarkt") as grab:
            grab("profile_x.html", url, wait_for=None)
            grab("perf_x.json", url, wait_for="tm-player-performance-table-new", json_only=True)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with BrowserFetcher() as bf:

        def grab(
            name: str, url: str, *, wait_for: str | None = None, json_only: bool = False
        ) -> None:
            res = bf.get(url, wait_for=wait_for)
            if json_only:
                payload = next(iter(res.captured_json.values()), {})
                (out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                (out / name).write_text(res.html)

        yield grab
