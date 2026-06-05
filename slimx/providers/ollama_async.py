# slimx/providers/ollama_async.py

import os

import httpx
from typing import Any, Dict, List, Sequence

from ..errors import ProviderError
from ..tooling import ToolSpec
from ..types import InspectedRequest, Result, StreamEvent, Usage
from ..utils.ndjson import aiter_ndjson
from .base import Provider, ProviderCapabilities
from .ollama import _parse_tool_calls, _payload, _timeout, _timeout_message


class OllamaAsyncProvider(Provider):
    name = "ollama"
    capabilities = ProviderCapabilities(
        tools=True,
        structured_output=True,
        streaming=True,
        async_chat=True,
        async_streaming=True,
        vision=True,
    )

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls, **overrides):
        """Build from env (`OLLAMA_BASE_URL`); kwargs win."""
        base_url = overrides.get("base_url") or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        return cls(base_url=base_url)

    def chat(self, req, *, tools: Sequence[ToolSpec] = (), timeout=None):
        raise NotImplementedError

    def stream(self, req, *, tools: Sequence[ToolSpec] = (), timeout=None):
        raise NotImplementedError

    def build_request(self, req, *, tools: Sequence[ToolSpec] = (), stream: bool = False):
        return InspectedRequest(
            provider=self.name,
            method="POST",
            url=f"{self.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            payload=_payload(req, stream=stream, tools=tools),
        )

    async def achat(self, req, *, tools: Sequence[ToolSpec] = (), timeout=None):
        payload = _payload(req, stream=True, tools=tools)
        url = f"{self.base_url}/api/chat"
        text_parts: List[str] = []
        raw_tool_calls: List[Dict[str, Any]] = []
        data: Dict[str, Any] = {}

        try:
            async with httpx.AsyncClient(timeout=_timeout(timeout)) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = await _aread_response_text(response)
                        raise ProviderError(f"Ollama error {response.status_code}: {body}")

                    async for obj in aiter_ndjson(response.aiter_bytes()):
                        data = obj
                        message = obj.get("message") or {}
                        chunk = message.get("content") or ""
                        if chunk:
                            text_parts.append(chunk)
                        raw_tool_calls.extend(message.get("tool_calls") or [])

                        if obj.get("done") is True:
                            break

        except httpx.TimeoutException as e:
            raise ProviderError(_timeout_message(req.model, url, streaming=False)) from e

        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
        return Result(
            text="".join(text_parts),
            raw=data,
            usage=usage,
            tool_calls=_parse_tool_calls(raw_tool_calls),
        )

    async def astream(self, req, *, tools: Sequence[ToolSpec] = (), timeout=None):
        payload = _payload(req, stream=True, tools=tools)
        url = f"{self.base_url}/api/chat"

        try:
            async with httpx.AsyncClient(timeout=_timeout(timeout)) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = await _aread_response_text(response)
                        raise ProviderError(f"Ollama error {response.status_code}: {body}")

                    async for obj in aiter_ndjson(response.aiter_bytes()):
                        message = obj.get("message") or {}
                        chunk = message.get("content") or ""
                        if chunk:
                            yield StreamEvent(type="text_delta", text=chunk, raw=obj)
                        for call in _parse_tool_calls(message.get("tool_calls") or []):
                            yield StreamEvent.tool(call, raw=obj)

                        if obj.get("done") is True:
                            break

        except httpx.TimeoutException as e:
            raise ProviderError(_timeout_message(req.model, url, streaming=True)) from e

        yield StreamEvent(type="done")


async def _aread_response_text(response: httpx.Response) -> str:
    try:
        return (await response.aread()).decode("utf-8", errors="replace")
    except Exception:
        return ""
