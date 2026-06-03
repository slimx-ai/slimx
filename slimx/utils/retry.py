"""Retry helpers with a shared, transient-only policy.

Only *transient* failures are retried (timeouts, rate limits, transport/network
errors). Deterministic failures — bad API keys (``ProviderAuthError``), schema
errors, tool-execution errors — are raised immediately so callers fail fast
instead of waiting through pointless backoff.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Tuple, Type, TypeVar

import httpx

from ..errors import ProviderRateLimitError, ProviderTimeoutError

T = TypeVar("T")

# Exception types considered worth retrying.
TRANSIENT_ERRORS: Tuple[Type[BaseException], ...] = (
    ProviderRateLimitError,
    ProviderTimeoutError,
    httpx.TimeoutException,
    httpx.TransportError,
)


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, TRANSIENT_ERRORS)


def retry(fn: Callable[[], T], retries: int = 2, base_delay: float = 0.5) -> T:
    last: BaseException | None = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if not _is_transient(e) or i >= retries:
                raise
            time.sleep(base_delay * (2 ** i))
    # Unreachable: the loop always returns or raises. Kept for the type-checker.
    assert last is not None
    raise last


async def async_retry(fn: Callable[[], Awaitable[T]], retries: int = 2, base_delay: float = 0.5) -> T:
    last: BaseException | None = None
    for i in range(retries + 1):
        try:
            return await fn()
        except Exception as e:
            last = e
            if not _is_transient(e) or i >= retries:
                raise
            await asyncio.sleep(base_delay * (2 ** i))
    assert last is not None
    raise last
