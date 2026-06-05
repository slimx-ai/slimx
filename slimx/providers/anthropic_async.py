# slimx/providers/anthropic_async.py
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

import httpx

from ..errors import ProviderAuthError
from ..tooling import ToolSpec
from ..types import InspectedRequest, Result, StreamEvent, redact_headers
from ..utils.sse_async import aiter_sse_data
from .anthropic import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_ANTHROPIC_VERSION,
    _StreamDecoder,
    _build_payload,
    _parse_response,
    _raise_for_status,
)
from .base import Provider, ProviderCapabilities


class AnthropicAsyncProvider(Provider):
    name = "anthropic"
    capabilities = ProviderCapabilities(
        tools=True,
        structured_output=False,
        async_chat=True,
        async_streaming=True,
        vision=True,
        documents=True,
    )

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
        version: str = DEFAULT_ANTHROPIC_VERSION,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.version = version

    @classmethod
    def from_env(cls, **overrides):
        """Build from env (`ANTHROPIC_API_KEY`/`_BASE_URL`/`_VERSION`); kwargs win."""
        api_key = overrides.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not set")
        return cls(
            api_key=api_key,
            base_url=overrides.get("base_url")
            or os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
            version=overrides.get("version")
            or os.environ.get("ANTHROPIC_VERSION", DEFAULT_ANTHROPIC_VERSION),
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json",
        }

    def build_request(self, req, *, tools: Sequence[ToolSpec] = (), stream: bool = False):
        return InspectedRequest(
            provider=self.name,
            method="POST",
            url=f"{self.base_url}/v1/messages",
            headers=redact_headers(self._headers()),
            payload=_build_payload(req, tools, stream=stream),
        )

    def chat(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        raise NotImplementedError

    def stream(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        raise NotImplementedError

    async def achat(
        self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None
    ) -> Result:
        payload = _build_payload(req, tools)
        url = f"{self.base_url}/v1/messages"
        async with httpx.AsyncClient(timeout=timeout or 30.0) as c:
            r = await c.post(url, headers=self._headers(), json=payload)
        _raise_for_status(r.status_code, r.text)
        return _parse_response(r.json())

    async def astream(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        payload = _build_payload(req, tools, stream=True)
        url = f"{self.base_url}/v1/messages"
        decoder = _StreamDecoder()
        async with httpx.AsyncClient(timeout=timeout or 30.0) as c:
            async with c.stream("POST", url, headers=self._headers(), json=payload) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", errors="replace")
                    _raise_for_status(r.status_code, body)
                async for chunk in aiter_sse_data(r.aiter_bytes()):
                    try:
                        obj = json.loads(chunk)
                    except Exception:
                        continue
                    for event in decoder.feed(obj):
                        yield event
                    if _StreamDecoder.is_done(obj):
                        break
        yield StreamEvent.done()
