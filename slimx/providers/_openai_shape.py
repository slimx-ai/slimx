"""Shared helpers for providers that speak the OpenAI Chat Completions shape.

Both the sync and async OpenAI providers (and, by inheritance, the ``oai``
OpenAI-compatible providers) build on these helpers so request building,
response parsing, streaming tool-call accumulation, and error mapping live in
exactly one place. Keeping this logic shared is what prevents the sync and
async paths from silently drifting apart.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..errors import ProviderAuthError, ProviderError, ProviderRateLimitError
from ..tooling import ToolSpec
from ..types import Result, StreamEvent, ToolCall, Usage


def tools_payload(tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def build_payload(req, tools: Sequence[ToolSpec], *, stream: bool = False) -> Dict[str, Any]:
    payload = req.to_dict()
    if tools:
        payload["tools"] = tools_payload(tools)
    if payload.get("response_format") == "json_object":
        payload["response_format"] = {"type": "json_object"}
    if stream:
        payload["stream"] = True
    return payload


def raise_for_status(status_code: int, body: str, *, provider: str = "OpenAI") -> None:
    if status_code == 401:
        raise ProviderAuthError(body)
    if status_code == 429:
        raise ProviderRateLimitError(body)
    if status_code >= 400:
        raise ProviderError(f"{provider} error {status_code}: {body}")


def parse_chat_response(data: Dict[str, Any]) -> Result:
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    tool_calls: List[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        try:
            args_obj = json.loads(args) if isinstance(args, str) else args
        except Exception:
            args_obj = {}
        tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args_obj))
    usage = Usage.from_openai(data.get("usage") or {})
    return Result(text=text, raw=data, usage=usage, tool_calls=tool_calls)


class StreamToolAccumulator:
    """Reassembles streamed OpenAI tool-call deltas.

    OpenAI streams the ``id`` and ``name`` of a tool call only in its first
    delta; every delta carries ``index``. Accumulating by ``index`` (not by
    ``id``) is what keeps a single call's argument fragments together.
    """

    def __init__(self) -> None:
        self._slots: Dict[Any, Dict[str, Any]] = {}

    def add(self, delta_tool_calls: Optional[Sequence[Dict[str, Any]]]) -> None:
        for tc in delta_tool_calls or []:
            index = tc.get("index", 0)
            fn = tc.get("function") or {}
            slot = self._slots.setdefault(index, {"id": None, "name": None, "args": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]

    def events(self) -> List[StreamEvent]:
        out: List[StreamEvent] = []
        for index, slot in self._slots.items():
            try:
                args_obj = json.loads(slot["args"] or "{}")
            except Exception:
                args_obj = {}
            out.append(
                StreamEvent.tool(
                    ToolCall(
                        id=slot["id"] or str(index),
                        name=slot["name"] or "",
                        arguments=args_obj,
                    ),
                    raw=slot,
                )
            )
        return out


def text_delta_from_chunk(obj: Dict[str, Any], acc: StreamToolAccumulator) -> Optional[StreamEvent]:
    """Process one decoded SSE chunk.

    Feeds any tool-call deltas into ``acc`` and returns a text-delta event when
    the chunk carried content, otherwise ``None``.
    """
    delta = (obj.get("choices") or [{}])[0].get("delta", {}) or {}
    acc.add(delta.get("tool_calls"))
    content = delta.get("content")
    if content:
        return StreamEvent.text_delta(content, raw=obj)
    return None
