# slimx/types.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# -------------------------
# Usage
# -------------------------

@dataclass(frozen=True)
class Usage:
    """
    Token usage (best-effort).

    Backwards compatible with v0.4.1 fields:
      - prompt_tokens
      - completion_tokens
      - total_tokens

    Provider-neutral aliases:
      - input_tokens
      - output_tokens
    """
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @property
    def input_tokens(self) -> Optional[int]:
        return self.prompt_tokens

    @property
    def output_tokens(self) -> Optional[int]:
        return self.completion_tokens

    @staticmethod
    def from_openai(d: Dict[str, Any]) -> "Usage":
        return Usage(
            d.get("prompt_tokens"),
            d.get("completion_tokens"),
            d.get("total_tokens"),
        )


# -------------------------
# Tool calls
# -------------------------

@dataclass(frozen=True)
class ToolCall:
    """
    A tool call requested by the model.

    Compatibility & normalization:
    - Accepts `arguments` as dict OR JSON string.
    - Stores:
        - arguments: Dict[str, Any]  (backwards compatible)
        - arguments_json: str        (canonical form for providers/streaming)
    """
    id: str
    name: str

    # Back-compat: many parts of v0.4.1 expect a dict here.
    arguments: Dict[str, Any] = field(default_factory=dict)

    # Canonical representation (useful for streaming and cross-provider consistency)
    arguments_json: str = "{}"

    def __post_init__(self) -> None:
        # Allow callers to pass a JSON string by stuffing it into arguments
        # (some providers stream/return args as a string).
        # We normalize both representations.
        raw_args = getattr(self, "arguments")

        # If caller passed a string, keep it as arguments_json and parse best-effort into dict.
        if isinstance(raw_args, str):
            object.__setattr__(self, "arguments_json", raw_args)
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
                object.__setattr__(self, "arguments", parsed if isinstance(parsed, dict) else {})
            except Exception:
                object.__setattr__(self, "arguments", {})
            return

        # If caller passed dict-like, keep dict and generate canonical JSON.
        if isinstance(raw_args, dict):
            try:
                object.__setattr__(
                    self,
                    "arguments_json",
                    json.dumps(raw_args, ensure_ascii=False, separators=(",", ":")),
                )
            except Exception:
                object.__setattr__(self, "arguments_json", "{}")
            return

        # Any other type: degrade gracefully
        object.__setattr__(self, "arguments", {})
        object.__setattr__(self, "arguments_json", "{}")

    def arguments_dict(self) -> Dict[str, Any]:
        return self.arguments


# -------------------------
# Streaming
# -------------------------

StreamEventType = Literal["text_delta", "tool_call", "done", "error"]


@dataclass(frozen=True)
class StreamEvent:
    """
    Normalized streaming event across providers.
    """
    type: StreamEventType
    text: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    error: Optional[str] = None
    raw: Any = None

    @staticmethod
    def text_delta(delta: str, *, raw: Any = None) -> "StreamEvent":
        return StreamEvent(type="text_delta", text=delta, raw=raw)

    @staticmethod
    def tool(call: ToolCall, *, raw: Any = None) -> "StreamEvent":
        return StreamEvent(type="tool_call", tool_call=call, raw=raw)

    @staticmethod
    def done(*, raw: Any = None) -> "StreamEvent":
        return StreamEvent(type="done", raw=raw)

    @staticmethod
    def err(message: str, *, raw: Any = None) -> "StreamEvent":
        return StreamEvent(type="error", error=message, raw=raw)


# -------------------------
# Result
# -------------------------

@dataclass
class Result:
    """
    Normalized completion result.

    Keeps v0.4.1 fields:
    - text
    - raw
    - usage
    - tool_calls
    - data (optional)

    Adds `parsed` alias for clarity (future-friendly).
    """
    text: str
    raw: Any = None
    usage: Usage = field(default_factory=Usage)
    tool_calls: List[ToolCall] = field(default_factory=list)

    # Back-compat; we may later rename this to `parsed` officially.
    data: Any = None

    @property
    def parsed(self) -> Any:
        return self.data
