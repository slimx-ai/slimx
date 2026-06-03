# slimx/providers/anthropic.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from ..errors import ProviderAuthError, ProviderError, ProviderRateLimitError
from ..messages import Message
from ..tooling import ToolSpec
from ..types import Result, StreamEvent, ToolCall, Usage
from .base import Provider, ProviderCapabilities

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"
    capabilities = ProviderCapabilities(tools=True, structured_output=False, streaming=False)

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
    def from_env(cls):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not set")
        return cls(
            api_key,
            os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
            os.environ.get("ANTHROPIC_VERSION", DEFAULT_ANTHROPIC_VERSION),
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json",
        }

    def chat(self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None) -> Result:
        payload = _build_payload(req, tools)
        url = f"{self.base_url}/v1/messages"
        with httpx.Client(timeout=timeout or 30.0) as c:
            r = c.post(url, headers=self._headers(), json=payload)
        _raise_for_status(r.status_code, r.text)
        return _parse_response(r.json())

    def stream(
        self, req, *, tools: Sequence[ToolSpec] = (), timeout: Optional[float] = None
    ) -> Iterable[StreamEvent]:
        # Native token streaming isn't implemented; emit the full result as one delta.
        res = self.chat(req, tools=tools, timeout=timeout)
        yield StreamEvent(type="text_delta", text=res.text, raw=res.raw)
        for tc in res.tool_calls:
            yield StreamEvent.tool(tc, raw=res.raw)
        yield StreamEvent(type="done")


# --------------------------------------------------------------------------
# Shared request/response mapping (also used by the async provider).
# --------------------------------------------------------------------------

def _tools_payload(tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def _build_payload(req, tools: Sequence[ToolSpec]) -> Dict[str, Any]:
    system, messages = _messages_to_anthropic(req.messages)
    payload: Dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens or 1024,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if tools:
        payload["tools"] = _tools_payload(tools)
    return payload


def _messages_to_anthropic(
    messages: Sequence[Message],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert SlimX messages to Anthropic's Messages API shape.

    Handles the SlimX auto-tool-loop format: assistant messages carry
    OpenAI-style ``tool_calls`` dicts, and tool results arrive as separate
    ``tool``-role messages. Consecutive tool results are merged into a single
    user turn so the conversation keeps strict user/assistant alternation.
    """
    out: List[Dict[str, Any]] = []
    system_parts: List[str] = []
    pending_tool_results: List[Dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue

        if m.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or m.tool_name or "",
                    "content": m.content,
                }
            )
            continue

        flush_tool_results()

        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name") or ""
                raw_args = fn.get("arguments")
                if raw_args is None:
                    raw_args = tc.get("arguments") or {}
                args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(args, dict):
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or name,
                        "name": name,
                        "input": args,
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        else:
            # Unknown role: keep content as user text rather than dropping it.
            if m.content:
                out.append({"role": "user", "content": m.content})

    flush_tool_results()
    system = "\n".join(system_parts) if system_parts else None
    return system, out


def _parse_response(data: Dict[str, Any]) -> Result:
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    tool_calls: List[ToolCall] = []
    for b in blocks:
        if b.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=b.get("id", ""),
                    name=b.get("name", ""),
                    arguments=b.get("input") or {},
                )
            )
    usage_data = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=usage_data.get("input_tokens"),
        completion_tokens=usage_data.get("output_tokens"),
    )
    return Result(text=text, raw=data, usage=usage, tool_calls=tool_calls)


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _raise_for_status(status_code: int, body: str) -> None:
    if status_code in (401, 403):
        raise ProviderAuthError(body)
    if status_code == 429:
        raise ProviderRateLimitError(body)
    if status_code >= 400:
        raise ProviderError(f"Anthropic error {status_code}: {body}")
