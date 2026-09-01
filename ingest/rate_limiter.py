"""Politeness throttle for scrapers (SPEC §5, legal note).

``RateLimiter`` enforces a minimum spacing (plus random jitter) between calls.
Thread-safe so a threaded crawler still respects one global cadence per host.
"""

from __future__ import annotations

import random
import threading
import time


class RateLimiter:
    def __init__(self, min_interval: float = 2.0, jitter: float = 1.0) -> None:
        if min_interval < 0 or jitter < 0:
            raise ValueError("min_interval and jitter must be non-negative")
        self.min_interval = min_interval
        self.jitter = jitter
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        """Block until it is polite to make the next request."""
        with self._lock:
            now = time.monotonic()
            target = self._last + self.min_interval + random.uniform(0, self.jitter)
            sleep_for = target - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()

    @classmethod
    def from_config(cls, cfg: dict) -> "RateLimiter":
        s = cfg["scrape"]
        return cls(min_interval=s["min_interval_seconds"], jitter=s["jitter_seconds"])
