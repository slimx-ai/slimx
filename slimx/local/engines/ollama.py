"""Ollama inference engine.

Reuses :class:`slimx.providers.ollama.OllamaProvider` for the base URL (``OLLAMA_BASE_URL``)
and ``/api/tags`` listing, and adds the bits SlimX needs for local-GPU UX:

- ``detect`` / ``health`` — is the daemon reachable, is the CLI installed?
- ``runtime_status`` — reads ``/api/ps`` and reports, per running model, whether it is
  fully on GPU, split CPU/GPU, or CPU-only (from Ollama's ``size`` vs ``size_vram``).
- ``pull_or_prepare_model`` — streams ``/api/pull`` progress.

``httpx`` is already a SlimX dependency; it is imported lazily inside methods so importing
this module stays cheap.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import httpx

from .base import (
    EngineHealth,
    EngineStatus,
    InferenceEngine,
    LocalModel,
    PullEvent,
    RunningModel,
    RuntimeStatus,
)

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaEngine(InferenceEngine):
    name = "ollama"
    kind = "ollama"
    supports_gpu = True
    supports_model_listing = True
    supports_runtime_status = True
    supports_launch = False

    def __init__(self, base_url: str | None = None, *, timeout: float = 5.0) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout

    # -- discovery / health ------------------------------------------------

    def detect(self) -> EngineStatus:
        installed = shutil.which("ollama") is not None
        health = self.health()
        detail = health.detail or ("reachable" if health.reachable else "")
        return EngineStatus(
            name=self.name,
            kind=self.kind,
            installed=installed,
            reachable=health.reachable,
            base_url=self.base_url,
            detail=detail,
        )

    def health(self) -> EngineHealth:
        import httpx

        try:
            resp = httpx.get(f"{self.base_url}/api/version", timeout=self.timeout)
        except httpx.HTTPError as exc:
            return EngineHealth(reachable=False, detail=f"{type(exc).__name__}: {exc}")
        if resp.status_code >= 400:
            return EngineHealth(reachable=False, detail=f"HTTP {resp.status_code}")
        version = ""
        try:
            version = str(resp.json().get("version", ""))
        except ValueError:
            pass
        return EngineHealth(reachable=True, detail=f"ollama {version}".strip())

    # -- models ------------------------------------------------------------

    def list_models(self) -> list[LocalModel]:
        import httpx

        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return []
        models: list[LocalModel] = []
        for entry in data.get("models", []) or []:
            name = entry.get("name")
            if not name:
                continue
            size = entry.get("size")
            models.append(
                LocalModel(
                    id=str(name),
                    engine=self.name,
                    size_gb=round(size / 1024**3, 2) if isinstance(size, (int, float)) else None,
                    raw=entry,
                )
            )
        return models

    def runtime_status(self) -> RuntimeStatus:
        import httpx

        try:
            resp = httpx.get(f"{self.base_url}/api/ps", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return RuntimeStatus(engine=self.name, running=[])
        running: list[RunningModel] = []
        for entry in data.get("models", []) or []:
            running.append(
                RunningModel(
                    name=str(entry.get("name", "")),
                    size_bytes=_as_int(entry.get("size")),
                    size_vram_bytes=_as_int(entry.get("size_vram")),
                )
            )
        return RuntimeStatus(engine=self.name, running=running)

    # -- lifecycle ---------------------------------------------------------

    def pull_or_prepare_model(self, model_id: str) -> Iterator[PullEvent]:
        import httpx

        from ...utils.ndjson import iter_ndjson

        url = f"{self.base_url}/api/pull"
        # Pulls can take minutes; only the connect phase is time-bounded.
        with httpx.Client(timeout=httpx.Timeout(None, connect=10.0)) as client:
            # Ask for diagnostics we can read without decompression. A peer may ignore this;
            # _bounded_body still declines actually encoded refusals before consuming them.
            with client.stream(
                "POST",
                url,
                json={"model": model_id, "stream": True},
                headers={"Accept-Encoding": "identity"},
            ) as resp:
                if not resp.is_success:
                    # Any non-2xx, exactly as ``raise_for_status()`` judged it before: a redirect
                    # this client does not follow is a failure too, not an empty pull. Ollama
                    # refuses a pull it cannot start (e.g. an invalid model name) with a JSON
                    # body; keep the exception type, but carry the engine's reason, since the
                    # stock message names only the status and a generic help link.
                    try:
                        reason = _error_reason(_bounded_body(resp))
                    except httpx.HTTPError:
                        # The status is already known. An optional diagnostic read must not
                        # replace it with a transport/decoding error or lose request/response.
                        # Process-control exceptions are deliberately not caught here.
                        reason = None
                    if reason:
                        raise httpx.HTTPStatusError(
                            f"Ollama refused the pull (HTTP {resp.status_code}): {reason}",
                            request=resp.request,
                            response=resp,
                        )
                    resp.raise_for_status()
                for obj in iter_ndjson(resp.iter_bytes()):
                    yield PullEvent(
                        status=str(obj.get("status", "")),
                        completed=_as_int(obj.get("completed")),
                        total=_as_int(obj.get("total")),
                        # A failure after the stream starts arrives as an ``{"error": ...}`` line
                        # on HTTP 200, with no ``status``.
                        error=_reason_text(obj.get("error")),
                    )


# Failure diagnostics retain at most 64 KiB of an unencoded body. Encoded failures are declined
# before consumption: httpx iter_bytes() can allocate an arbitrarily large decoded chunk before
# yielding it. The transport can hand us one crossing raw chunk; it is never appended or decoded.
# Reading it is bounded in SIZE, not in time: this client bounds only the connect phase, because a
# pull's own stream takes minutes, so a peer that stalls mid-body stalls the call. That is the same
# exposure the success path has always had, now also reachable on a failure.
_MAX_ERROR_BODY_BYTES = 65_536
_MAX_ERROR_REASON_CHARS = 2_000


def _bounded_body(resp: httpx.Response) -> bytes | None:
    """Retain at most ``_MAX_ERROR_BODY_BYTES`` of identity-encoded diagnostic bytes.

    Encoded bodies are not consumed. Raw transport consumption may include one crossing chunk;
    that chunk is not retained. Content-Length is not trusted. Size, not time, is bounded.
    """
    if resp.headers.get("content-encoding", "").strip().lower() not in ("", "identity"):
        return None
    body = bytearray()
    # Cached responses (including fixture transports) may already have been consumed. A real
    # streamed response uses iter_raw(), which never invokes a content decoder.
    chunks = (resp.content,) if resp.is_stream_consumed else resp.iter_raw()
    for chunk in chunks:
        if len(chunk) > _MAX_ERROR_BODY_BYTES - len(body):
            return None
        body.extend(chunk)
    return bytes(body)


def _error_reason(body: bytes | None) -> str | None:
    """The ``error`` of an Ollama JSON error body, if it has a usable one."""
    import json

    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (ValueError, RecursionError):
        # Malformed JSON, bytes that are not text, or nesting deep enough to exhaust the parser's
        # stack. A body this engine cannot read is not a reason; the stock error stands.
        return None
    return _reason_text(parsed.get("error")) if isinstance(parsed, dict) else None


def _reason_text(error: object) -> str | None:
    """An engine-reported reason as bounded text; None when the engine reported none.

    ``None`` and ``""`` are "no reason". Anything else is one, including values that are falsy in
    Python (``0``, ``False``, ``[]``): another engine may say that much and it is not this
    function's place to drop it.
    """
    import json

    if error is None:
        return None
    text = error if isinstance(error, str) else json.dumps(error, ensure_ascii=False, default=str)
    # Engine text reaches a caller's UI verbatim. A lone surrogate (a ``\udXXX`` escape in the
    # engine's JSON) survives json.loads but cannot be encoded as UTF-8 later, so replace it here.
    text = text.encode("utf-8", "replace").decode("utf-8").strip()
    if len(text) > _MAX_ERROR_REASON_CHARS:
        text = text[:_MAX_ERROR_REASON_CHARS].rstrip() + "…"
    return text or None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
