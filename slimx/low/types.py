from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ..messages import Message
from ..types import ImageGenerationOptions, ImageInput


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
class ImageEditRequest:
    """A request to edit/refine source image(s) with a text instruction.

    Editing is a distinct intent from generation: it always carries one or more
    source ``images`` plus an ``instruction``. ``options`` configures the hosted
    image tool; ``previous_response_id`` is an optional provider-side optimization
    and must never be the only way to reach a source image — ``images`` (inline
    bytes) is the durable path that survives a reload or provider state expiry.
    """
    model: str
    instruction: str
    images: List[ImageInput] = field(default_factory=list)
    n: int = 1
    size: Optional[str] = None
    options: Optional[ImageGenerationOptions] = None
    previous_response_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class ChatRequest:
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    # Hosted image-generation tool config. When set, OpenAI-shaped providers route
    # the call to the Responses API (/responses) instead of /chat/completions and
    # expose the model the `image_generation` tool. None keeps the classic path.
    image_generation: Optional[ImageGenerationOptions] = None
    # Conversational image revision: continue from an earlier provider response.
    previous_response_id: Optional[str] = None
    # Provider tool_choice passthrough (e.g. force the hosted image tool).
    tool_choice: Optional[Any] = None

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
