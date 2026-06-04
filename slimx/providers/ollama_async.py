# slimx/providers/ollama_async.py

import os

import httpx
from typing import Any, Dict, Optional, Sequence

from ..errors import ProviderError
from ..tooling import ToolSpec
from ..types import InspectedRequest, Result, StreamEvent, Usage
from ..utils.ndjson import aiter_ndjson
from .base import Provider, ProviderCapabilities


class OllamaAsyncProvider(Provider):
    name = "ollama"
    capabilities = ProviderCapabilities(
        streaming=True,
        async_chat=True,
        async_streaming=True,
    )

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

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
            payload=_payload(req, stream=stream),
        )

    async def achat(self, req, *, tools: Sequence[ToolSpec] = (), timeout=None):
        payload = _payload(req, stream=True)
        url = f"{self.base_url}/api/chat"
        text_parts: list[str] = []
        data: Dict[str, Any] = {}

        try:
            async with httpx.AsyncClient(timeout=_timeout(timeout)) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = await _aread_response_text(response)
                        raise ProviderError(f"Ollama error {response.status_code}: {body}")

                    async for obj in aiter_ndjson(response.aiter_bytes()):
                        data = obj
                        chunk = (obj.get("message") or {}).get("content") or ""

                        if chunk:
                            text_parts.append(chunk)

                        if obj.get("done") is True:
                            break

        except httpx.TimeoutException as e:
            raise ProviderError(
                f"Ollama request timed out for model {req.model!r} at {url}. "
                "Try warming the model with `ollama run`, using a smaller model, "
                "reducing max_tokens/context, or increasing the SlimX timeout."
            ) from e

        text = "".join(text_parts)
        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

        return Result(text=text, raw=data, usage=usage)

    async def astream(self, req, *, tools: Sequence[ToolSpec] = (), timeout=None):
        payload = _payload(req, stream=True)
        url = f"{self.base_url}/api/chat"

        try:
            async with httpx.AsyncClient(timeout=_timeout(timeout)) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = await _aread_response_text(response)
                        raise ProviderError(f"Ollama error {response.status_code}: {body}")

                    async for obj in aiter_ndjson(response.aiter_bytes()):
                        if obj.get("done") is True:
                            break

                        chunk = (obj.get("message") or {}).get("content") or ""
                        if chunk:
                            yield StreamEvent(type="text_delta", text=chunk, raw=obj)

        except httpx.TimeoutException as e:
            raise ProviderError(
                f"Ollama stream timed out for model {req.model!r} at {url}. "
                "Try warming the model with `ollama run`, using a smaller model, "
                "reducing max_tokens/context, or increasing the SlimX timeout."
            ) from e

        yield StreamEvent(type="done")


async def _aread_response_text(response: httpx.Response) -> str:
    try:
        return (await response.aread()).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _timeout(timeout: Optional[float]) -> httpx.Timeout:
    if timeout is None:
        return httpx.Timeout(None, connect=10.0)

    return httpx.Timeout(timeout, connect=min(float(timeout), 10.0))


def _payload(req, *, stream: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": [
            message.to_dict()
            for message in req.messages
            if message.role in ("user", "assistant", "system")
        ],
        "stream": stream,
    }

    options: Dict[str, Any] = {}

    if req.temperature is not None:
        options["temperature"] = req.temperature

    if req.max_tokens is not None:
        options["num_predict"] = req.max_tokens

    if req.extra:
        extra_options = req.extra.get("options")
        if isinstance(extra_options, dict):
            options.update(extra_options)

        for key in ("format", "keep_alive"):
            if key in req.extra:
                payload[key] = req.extra[key]

    if options:
        payload["options"] = options

    return payload