from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @staticmethod
    def from_openai(d: Dict[str, Any]) -> "Usage":
        return Usage(d.get("prompt_tokens"), d.get("completion_tokens"), d.get("total_tokens"))

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass(frozen=True)
class StreamEvent:
    type: str
    text: str = ""
    tool_call: Optional[ToolCall] = None
    raw: Any = None

@dataclass
class Result:
    text: str
    raw: Any = None
    usage: Usage = field(default_factory=Usage)
    tool_calls: List[ToolCall] = field(default_factory=list)
    data: Any = None
