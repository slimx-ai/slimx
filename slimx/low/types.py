from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ..messages import Message


@dataclass
class ImageRequest:
    """A request to generate image(s) from a text prompt.

    Image generation is a distinct endpoint from chat (a prompt, not a message
    list), so it gets its own request type rather than overloading ChatRequest.
    Provider-specific knobs (quality, style, response_format, …) flow through
    ``extra``.
    """
    model: str
    prompt: str
    n: int = 1
    size: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"model": self.model, "prompt": self.prompt, "n": self.n}
        if self.size is not None:
            d["size"] = self.size
        if self.extra:
            d.update(self.extra)
        return d


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
