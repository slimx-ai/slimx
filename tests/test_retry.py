from __future__ import annotations

import asyncio

import pytest

from slimx.errors import ProviderAuthError, ProviderRateLimitError
from slimx.utils.retry import async_retry, retry


def test_retry_recovers_from_transient_error():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderRateLimitError("429")
        return "ok"

    assert retry(flaky, retries=3, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_non_transient_error():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ProviderAuthError("401 invalid key")

    with pytest.raises(ProviderAuthError):
        retry(boom, retries=2, base_delay=0)
    assert calls["n"] == 1  # auth errors fail fast


def test_retry_gives_up_after_retries():
    calls = {"n": 0}

    def always_429():
        calls["n"] += 1
        raise ProviderRateLimitError("429")

    with pytest.raises(ProviderRateLimitError):
        retry(always_429, retries=2, base_delay=0)
    assert calls["n"] == 3  # initial + 2 retries


def test_async_retry_matches_sync_policy():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ProviderRateLimitError("429")
        return "ok"

    assert asyncio.run(async_retry(flaky, retries=2, base_delay=0)) == "ok"
    assert calls["n"] == 2

    auth_calls = {"n": 0}

    async def boom():
        auth_calls["n"] += 1
        raise ProviderAuthError("401")

    with pytest.raises(ProviderAuthError):
        asyncio.run(async_retry(boom, retries=2, base_delay=0))
    assert auth_calls["n"] == 1
