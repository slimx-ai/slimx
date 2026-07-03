# slimx/providers/anthropic.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from ..content import DocumentPart, ImagePart, TextPart, guard_modalities, to_base64
from ..errors import ProviderAuthError, ProviderError, ProviderRateLimitError
from ..messages import Message
from ..tooling import ToolSpec
from ..types import InspectedRequest, Result, StreamEvent, ToolCall, Usage, redact_headers
from ..utils.sse import iter_sse_data
from .base import Provider, ProviderCapabilities

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"
    capabilities = ProviderCapabilities(
        tools=True,
        structured_output=False,
        streaming=True,
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
        """Build from env (`ANTHROPIC_API_KEY`/`_BASE_URL`/`_VERSION`); kwargs win.

        Single source of truth for Anthropic env configuration; the factory in
        `providers/_defaults.py` delegates here.
        """
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

    def list_models(self, *, timeout: Optional[float] = None) -> list:
        url = f"{self.base_url}/v1/models"
        with httpx.Client(timeout=timeout or 10.0) as c:
            r = c.get(url, headers=self._headers())
        _raise_for_status(r.status_code, r.text)
        data = r.json()
        return [m.get("id") for m in (data.get("data") or []) if m.get("id")]

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
        payload = _build_payload(req, tools, stream=True)
        url = f"{self.base_url}/v1/messages"
        decoder = _StreamDecoder()
        with httpx.Client(timeout=timeout or 30.0) as c:
            with c.stream("POST", url, headers=self._headers(), json=payload) as r:
                if r.status_code >= 400:
                    body = r.read().decode("utf-8", errors="replace")
                    _raise_for_status(r.status_code, body)
                for chunk in iter_sse_data(r.iter_bytes()):
                    try:
                        obj = json.loads(chunk)
                    except Exception:
                        continue
                    for event in decoder.feed(obj):
                        yield event
                    if _StreamDecoder.is_done(obj):
                        break
        yield StreamEvent.done()


# --------------------------------------------------------------------------
# Shared request/response mapping (also used by the async provider).
# --------------------------------------------------------------------------

def _tools_payload(tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


# Recent Claude families reject non-default sampling params outright (HTTP 400,
# "`temperature` is deprecated for this model"); Anthropic's guidance is to omit them.
# Conservative model-prefix rule; sampling keys are dropped (never defaulted/nulled).
ANTHROPIC_UNSUPPORTED_SAMPLING_PARAMS = frozenset({"temperature", "top_p", "top_k"})
_NO_SAMPLING_MODEL_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def _sampling_params_supported(model: str) -> bool:
    model_name = model.strip().lower()
    return not model_name.startswith(_NO_SAMPLING_MODEL_PREFIXES)


def _filtered_extra(model: str, extra):
    """req.extra with the rejected sampling keys removed for no-sampling models."""
    if not extra:
        return None
    filtered = dict(extra)
    if not _sampling_params_supported(model):
        for key in ANTHROPIC_UNSUPPORTED_SAMPLING_PARAMS:
            filtered.pop(key, None)
    return filtered or None


def _build_payload(req, tools: Sequence[ToolSpec], *, stream: bool = False) -> Dict[str, Any]:
    guard_modalities(req.messages, AnthropicProvider.capabilities, AnthropicProvider.name)
    system, messages = _messages_to_anthropic(req.messages)
    payload: Dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens or 1024,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if req.temperature is not None and _sampling_params_supported(req.model):
        payload["temperature"] = req.temperature
    if tools:
        payload["tools"] = _tools_payload(tools)
    # Provider-specific escape hatch: top_p, stop_sequences, tool_choice, metadata,
    # prompt caching, beta fields, etc. flow straight through `req.extra` — minus the
    # sampling keys the selected model rejects.
    extra = _filtered_extra(req.model, req.extra)
    if extra:
        for key, value in extra.items():
            payload[key] = value
    if stream:
        payload["stream"] = True
    return payload


class _StreamDecoder:
    """Turns Anthropic's Messages SSE events into normalized StreamEvents.

    Text arrives as ``content_block_delta`` / ``text_delta``; tool calls arrive as a
    ``tool_use`` block whose arguments stream in as ``input_json_delta`` fragments and
    are emitted as one ToolCall when the block stops.
    """

    def __init__(self) -> None:
        self._tool_blocks: Dict[Any, Dict[str, Any]] = {}

    def feed(self, obj: Dict[str, Any]) -> List[StreamEvent]:
        kind = obj.get("type")
        if kind == "content_block_start":
            block = obj.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._tool_blocks[obj.get("index")] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "json": "",
                }
            return []
        if kind == "content_block_delta":
            delta = obj.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text")
                return [StreamEvent.text_delta(text, raw=obj)] if text else []
            if dtype == "input_json_delta":
                block = self._tool_blocks.get(obj.get("index"))
                if block is not None:
                    block["json"] += delta.get("partial_json", "")
            return []
        if kind == "content_block_stop":
            block = self._tool_blocks.pop(obj.get("index"), None)
            if block is None:
                return []
            try:
                args = json.loads(block["json"] or "{}")
            except Exception:
                args = {}
            call = ToolCall(id=block["id"], name=block["name"], arguments=args)
            return [StreamEvent.tool(call, raw=obj)]
        if kind == "error":
            err = obj.get("error") or {}
            return [StreamEvent.err(str(err.get("message") or err), raw=obj)]
        return []

    @staticmethod
    def is_done(obj: Dict[str, Any]) -> bool:
        return obj.get("type") == "message_stop"


def _media_source(part) -> Dict[str, Any]:
    """Anthropic image/document `source` object (base64 or url)."""
    if part.url and part.data is None:
        return {"type": "url", "url": part.url}
    return {
        "type": "base64",
        "media_type": part.mime_type or "application/octet-stream",
        "data": to_base64(part.data or b""),
    }


def _anthropic_blocks(message: Message) -> List[Dict[str, Any]]:
    """Convert a multimodal SlimX message into Anthropic content blocks."""
    blocks: List[Dict[str, Any]] = []
    for p in message.content_parts():
        if isinstance(p, TextPart):
            blocks.append({"type": "text", "text": p.text})
        elif isinstance(p, ImagePart):
            blocks.append({"type": "image", "source": _media_source(p)})
        elif isinstance(p, DocumentPart):
            blocks.append({"type": "document", "source": _media_source(p)})
    return blocks


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
            if m.is_multimodal():
                out.append({"role": "user", "content": _anthropic_blocks(m)})
            else:
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
