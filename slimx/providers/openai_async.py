import json
import os
from typing import Dict, Optional, Sequence

import httpx

from ..errors import ProviderAuthError
from ..low.types import ImageRequest
from ..tooling import ToolSpec
from ..types import InspectedRequest, StreamEvent, redact_headers
from ..utils.sse_async import aiter_sse_data
from ._openai_shape import (
    StreamToolAccumulator,
    build_payload,
    parse_chat_response,
    parse_image_response,
    raise_for_status,
    text_delta_from_chunk,
)
from .base import Provider, ProviderCapabilities


class OpenAIAsyncProvider(Provider):
    name = "openai"
    capabilities = ProviderCapabilities(
        tools=True,
        structured_output=True,
        streaming=True,
        async_chat=True,
        async_streaming=True,
        vision=True,
        documents=True,
        audio_in=True,
        image_out=True,
    )

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls, **overrides):
        """Build from env vars (`OPENAI_API_KEY`, `OPENAI_BASE_URL`); kwargs win."""
        api_key = overrides.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set")
        base_url = overrides.get("base_url") or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        return cls(api_key=api_key, base_url=base_url)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def build_request(self, req, *, tools: Sequence[ToolSpec] = (), stream: bool = False):
        return InspectedRequest(
            provider=self.name,
            method="POST",
            url=f"{self.base_url}/chat/completions",
            headers=redact_headers(self._headers()),
            payload=build_payload(req, tools, stream=stream, caps=self.capabilities, provider=self.name),
        )

    def build_image_request(self, req: ImageRequest) -> InspectedRequest:
        return InspectedRequest(
            provider=self.name,
            method="POST",
            url=f"{self.base_url}/images/generations",
            headers=redact_headers(self._headers()),
            payload=req.to_dict(),
        )

    async def agenerate_image(self, req: ImageRequest, *, timeout: Optional[float] = None):
        url = f"{self.base_url}/images/generations"
        async with httpx.AsyncClient(timeout=timeout or 60.0) as c:
            r = await c.post(url, headers=self._headers(), json=req.to_dict())
        raise_for_status(r.status_code, r.text)
        return parse_image_response(r.json())

    def chat(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        raise NotImplementedError

    def stream(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        raise NotImplementedError

    async def achat(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        payload = build_payload(req, tools, caps=self.capabilities, provider=self.name)
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=timeout or 30.0) as c:
            r = await c.post(url, headers=self._headers(), json=payload)
        raise_for_status(r.status_code, r.text)
        return parse_chat_response(r.json())

    async def astream(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        payload = build_payload(req, tools, stream=True, caps=self.capabilities, provider=self.name)
        url = f"{self.base_url}/chat/completions"
        acc = StreamToolAccumulator()
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", url, headers=self._headers(), json=payload) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", errors="replace")
                    raise_for_status(r.status_code, body)
                async for chunk in aiter_sse_data(r.aiter_bytes()):
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except Exception:
                        continue
                    event = text_delta_from_chunk(obj, acc)
                    if event is not None:
                        yield event
        for event in acc.events():
            yield event
        yield StreamEvent.done()
