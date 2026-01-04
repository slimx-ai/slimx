from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ..messages import Message

@dataclass
class ChatRequest:
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"model": self.model, "messages": [m.to_dict() for m in self.messages]}
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        if self.response_format:
            d["response_format"] = self.response_format
        if self.extra:
            d.update(self.extra)
        return d
