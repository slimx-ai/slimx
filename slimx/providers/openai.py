import json
import os
from typing import Dict, Iterable, Optional, Sequence

import httpx

from ..errors import ProviderAuthError
from ..low.types import ImageEditRequest, ImageRequest
from ..tooling import ToolSpec
from ..types import InspectedRequest, StreamEvent, redact_headers
from ..utils.sse import iter_sse_data
from ._openai_responses import (
    ResponsesStreamTranslator,
    build_edit_payload,
    build_responses_payload,
    operation_for_options,
    parse_responses_response,
)
from ._openai_shape import (
    StreamToolAccumulator,
    build_payload,
    parse_chat_response,
    parse_image_response,
    raise_for_status,
    text_delta_from_chunk,
)
from .base import Provider, ProviderCapabilities

# A hosted-image-tool request goes to the Responses API, which can run an image
# generation for tens of seconds; give it a roomier default than chat.
RESPONSES_DEFAULT_TIMEOUT = 120.0


class OpenAIProvider(Provider):
    name = "openai"
    capabilities = ProviderCapabilities(
        tools=True,
        structured_output=True,
        streaming=True,
        vision=True,
        documents=True,
        audio_in=True,
        image_out=True,
        image_edit=True,
        hosted_image_tool=True,
        image_partial_streaming=True,
    )

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls, **overrides):
        """Build from env vars (`OPENAI_API_KEY`, `OPENAI_BASE_URL`); kwargs win.

        This is the single source of truth for OpenAI env configuration — the
        provider factory in `providers/_defaults.py` delegates here.
        """
        api_key = overrides.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set")
        base_url = overrides.get("base_url") or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        return cls(api_key=api_key, base_url=base_url)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def list_models(self, *, timeout: Optional[float] = None) -> list:
        url = f"{self.base_url}/models"
        with httpx.Client(timeout=timeout or 10.0) as c:
            r = c.get(url, headers=self._headers())
        raise_for_status(r.status_code, r.text)
        data = r.json()
        return [m.get("id") for m in (data.get("data") or []) if m.get("id")]

    def build_request(self, req, *, tools: Sequence[ToolSpec] = (), stream: bool = False):
        if getattr(req, "image_generation", None) is not None:
            return InspectedRequest(
                provider=self.name,
                method="POST",
                url=f"{self.base_url}/responses",
                headers=redact_headers(self._headers()),
                payload=build_responses_payload(
                    req, tools, stream=stream, caps=self.capabilities, provider=self.name
                ),
            )
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

    def generate_image(self, req: ImageRequest, *, timeout: Optional[float] = None):
        url = f"{self.base_url}/images/generations"
        with httpx.Client(timeout=timeout or 60.0) as c:
            r = c.post(url, headers=self._headers(), json=req.to_dict())
        raise_for_status(r.status_code, r.text)
        return parse_image_response(r.json())

    def edit_image(self, req: ImageEditRequest, *, timeout: Optional[float] = None):
        """Edit/refine source image(s) via the Responses API (forced image tool)."""
        payload = build_edit_payload(req)
        url = f"{self.base_url}/responses"
        with httpx.Client(timeout=timeout or RESPONSES_DEFAULT_TIMEOUT) as c:
            r = c.post(url, headers=self._headers(), json=payload)
        raise_for_status(r.status_code, r.text)
        return parse_responses_response(
            r.json(), provider=self.name, model=req.model, operation="edit"
        )

    def chat(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        # Hosted image tool requested → Responses API; otherwise Chat Completions.
        if getattr(req, "image_generation", None) is not None:
            payload = build_responses_payload(req, tools, caps=self.capabilities, provider=self.name)
            url = f"{self.base_url}/responses"
            with httpx.Client(timeout=timeout or RESPONSES_DEFAULT_TIMEOUT) as c:
                r = c.post(url, headers=self._headers(), json=payload)
            raise_for_status(r.status_code, r.text)
            return parse_responses_response(
                r.json(),
                provider=self.name,
                model=req.model,
                operation=operation_for_options(req.image_generation),
            )
        payload = build_payload(req, tools, caps=self.capabilities, provider=self.name)
        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=timeout or 30.0) as c:
            r = c.post(url, headers=self._headers(), json=payload)
        raise_for_status(r.status_code, r.text)
        return parse_chat_response(r.json())

    def stream(
        self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None
    ) -> Iterable[StreamEvent]:
        # Return the right generator; the method itself is a plain dispatcher so a
        # `return` here is a value, not a swallowed StopIteration.
        if getattr(req, "image_generation", None) is not None:
            return self._responses_stream(req, tools, timeout)
        return self._chat_stream(req, tools, timeout)

    def _chat_stream(self, req, tools, timeout) -> Iterable[StreamEvent]:
        payload = build_payload(req, tools, stream=True, caps=self.capabilities, provider=self.name)
        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=timeout) as c:
            with c.stream("POST", url, headers=self._headers(), json=payload) as r:
                if r.status_code >= 400:
                    # Body must be read before access on a streamed response.
                    body = r.read().decode("utf-8", errors="replace")
                    raise_for_status(r.status_code, body)
                acc = StreamToolAccumulator()
                for chunk in iter_sse_data(r.iter_bytes()):
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

    def _responses_stream(self, req, tools, timeout) -> Iterable[StreamEvent]:
        payload = build_responses_payload(
            req, tools, stream=True, caps=self.capabilities, provider=self.name
        )
        url = f"{self.base_url}/responses"
        translator = ResponsesStreamTranslator(
            provider=self.name, model=req.model, operation=operation_for_options(req.image_generation)
        )
        with httpx.Client(timeout=timeout) as c:
            with c.stream("POST", url, headers=self._headers(), json=payload) as r:
                if r.status_code >= 400:
                    body = r.read().decode("utf-8", errors="replace")
                    raise_for_status(r.status_code, body)
                for chunk in iter_sse_data(r.iter_bytes()):
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except Exception:
                        continue
                    for event in translator.feed(obj):
                        yield event
        for event in translator.finish():
            yield event
