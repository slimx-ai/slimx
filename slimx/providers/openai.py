import json
import os
from typing import Dict, Iterable, Optional, Sequence

import httpx

from ..errors import ProviderAuthError
from ..tooling import ToolSpec
from ..types import StreamEvent
from ..utils.sse import iter_sse_data
from ._openai_shape import (
    StreamToolAccumulator,
    build_payload,
    parse_chat_response,
    raise_for_status,
    text_delta_from_chunk,
)
from .base import Provider, ProviderCapabilities


class OpenAIProvider(Provider):
    name = "openai"
    capabilities = ProviderCapabilities(tools=True, structured_output=True, streaming=True)

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(api_key, base_url)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None):
        payload = build_payload(req, tools)
        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=timeout or 30.0) as c:
            r = c.post(url, headers=self._headers(), json=payload)
        raise_for_status(r.status_code, r.text)
        return parse_chat_response(r.json())

    def stream(
        self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None
    ) -> Iterable[StreamEvent]:
        payload = build_payload(req, tools, stream=True)
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
