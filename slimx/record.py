"""Reproducible call records.

A `CallRecord` is a serializable snapshot of one completed call — the request
that went out, the response that came back, usage, and trace — plus the SlimX
version. It is built from a `Result` (the Client attaches a request snapshot to
every Result so records are self-contained), and can be saved to / loaded from
JSON for debugging, audits, evals, or regression fixtures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict

from . import __version__

if TYPE_CHECKING:
    from .types import Result


@dataclass
class CallRecord:
    slimx_version: str
    provider: str
    model: str
    request: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    trace: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: "Result") -> "CallRecord":
        trace = dict(result.trace or {})
        request = dict(result.request or {})
        response = {
            "text": result.text,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "arguments_json": tc.arguments_json,
                    "extra": tc.extra,
                }
                for tc in result.tool_calls
            ],
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            "data": result.data,
        }
        return cls(
            slimx_version=__version__,
            provider=trace.get("provider") or request.get("provider") or "",
            model=trace.get("model") or request.get("model") or "",
            request=request,
            response=response,
            raw=result.raw,
            trace=trace,
        )

    def to_dict(self) -> Dict[str, Any]:
        from .content import elide_media

        # Elide large base64 media so records stay small and diffable. This only
        # affects the serialized view; the live CallRecord keeps the real bytes.
        d = asdict(self)
        d["request"] = elide_media(d.get("request"))
        d["response"] = elide_media(d.get("response"))
        d["raw"] = elide_media(d.get("raw"))
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "CallRecord":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
