import time

from ingest.rate_limiter import RateLimiter


def test_first_call_does_not_block():
    rl = RateLimiter(min_interval=10, jitter=0)
    t0 = time.monotonic()
    rl.wait()
    assert time.monotonic() - t0 < 0.5


def test_second_call_respects_min_interval():
    rl = RateLimiter(min_interval=0.2, jitter=0)
    rl.wait()
    t0 = time.monotonic()
    rl.wait()
    assert time.monotonic() - t0 >= 0.19


def test_rejects_negative():
    import pytest

    with pytest.raises(ValueError):
        RateLimiter(min_interval=-1)
