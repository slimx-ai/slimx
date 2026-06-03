# slimx/providers/google_async.py
from __future__ import annotations

import json
from typing import Optional, Sequence

import httpx

from ..errors import ProviderAuthError
from ..low.types import ChatRequest
from ..tooling import ToolSpec
from ..types import Result, StreamEvent
from ..utils.sse_async import aiter_sse_data
from .base import Provider, ProviderCapabilities
from .google import (
    DEFAULT_GOOGLE_BASE_URL,
    _extract_text_parts,
    _extract_tool_calls,
    _model_path,
    _parse_response,
    _payload,
    _raise_for_status,
)


class GoogleAsyncProvider(Provider):
    name = "google"
    capabilities = ProviderCapabilities(
        tools=True,
        structured_output=True,
        streaming=True,
        async_chat=True,
        async_streaming=True,
    )

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_GOOGLE_BASE_URL,
    ):
        if not api_key:
            raise ProviderAuthError("GOOGLE_API_KEY or GEMINI_API_KEY is not set")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def chat(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec] = (),
        timeout: Optional[float] = None,
    ) -> Result:
        raise NotImplementedError

    def stream(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec] = (),
        timeout: Optional[float] = None,
    ):
        raise NotImplementedError

    async def achat(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec] = (),
        timeout: Optional[float] = None,
    ) -> Result:
        payload = _payload(req, tools=tools)
        url = f"{self.base_url}/{_model_path(req.model)}:generateContent"

        async with httpx.AsyncClient(timeout=timeout or 30.0) as client:
            response = await client.post(url, headers=self._headers(), json=payload)

        _raise_for_status(response.status_code, response.text)
        return _parse_response(response.json())

    async def astream(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec] = (),
        timeout: Optional[float] = None,
    ):
        payload = _payload(req, tools=tools)
        url = f"{self.base_url}/{_model_path(req.model)}:streamGenerateContent?alt=sse"

        async with httpx.AsyncClient(timeout=timeout or 30.0) as client:
            async with client.stream("POST", url, headers=self._headers(), json=payload) as response:
                body = ""
                if response.status_code >= 400:
                    raw = await response.aread()
                    body = raw.decode("utf-8", errors="replace")
                _raise_for_status(response.status_code, body)

                async for chunk in aiter_sse_data(response.aiter_bytes()):
                    if not chunk or chunk == "[DONE]":
                        continue

                    try:
                        data = json.loads(chunk)
                    except Exception:
                        continue

                    for text in _extract_text_parts(data):
                        yield StreamEvent.text_delta(text, raw=data)

                    for tool_call in _extract_tool_calls(data):
                        yield StreamEvent.tool(tool_call, raw=data)

        yield StreamEvent.done()