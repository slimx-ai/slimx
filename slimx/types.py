# slimx/types.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from .record import CallRecord


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
        - extra: Dict[str, Any]      (provider-specific opaque data that must be
          round-tripped back to the provider, e.g. Gemini `thoughtSignature`)
    """
    id: str
    name: str

    # Back-compat: many parts of v0.4.1 expect a dict here.
    arguments: Dict[str, Any] = field(default_factory=dict)

    # Canonical representation (useful for streaming and cross-provider consistency)
    arguments_json: str = "{}"

    # Provider-specific opaque metadata that must survive the tool loop and be
    # replayed to the same provider (e.g. Gemini's required `thoughtSignature`).
    extra: Dict[str, Any] = field(default_factory=dict)

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

# Additive: the image_* events only appear from providers/models that support
# hosted image generation (OpenAI Responses). Existing consumers that switch on
# the original four types keep working — they simply never see the new ones.
StreamEventType = Literal[
    "text_delta",
    "tool_call",
    "done",
    "error",
    "image_started",
    "image_partial",
    "image_completed",
]


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

    # Image streaming (image_started / image_partial / image_completed). `image`
    # carries the final normalized result on completion; `image_partial_b64` is a
    # transient preview frame (base64, intentionally not a `GeneratedImage` so it
    # is never mistaken for a final asset). `image_index` identifies the output.
    image: Optional["GeneratedImage"] = None
    image_partial_b64: Optional[str] = None
    image_index: Optional[int] = None

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

    @staticmethod
    def image_started(*, index: Optional[int] = None, raw: Any = None) -> "StreamEvent":
        return StreamEvent(type="image_started", image_index=index, raw=raw)

    @staticmethod
    def image_partial(
        b64: str, *, index: Optional[int] = None, raw: Any = None
    ) -> "StreamEvent":
        return StreamEvent(
            type="image_partial", image_partial_b64=b64, image_index=index, raw=raw
        )

    @staticmethod
    def image_completed(
        image: "GeneratedImage", *, index: Optional[int] = None, raw: Any = None
    ) -> "StreamEvent":
        return StreamEvent(type="image_completed", image=image, image_index=index, raw=raw)


# -------------------------
# Result
# -------------------------

@dataclass(frozen=True)
class GeneratedImage:
    """An image returned by an image-generation model.

    Carried on `Result.images`. Inline bytes live in `data`; some providers
    instead return a hosted `url`. The remaining fields are best-effort
    provenance/normalization metadata: providers fill what they actually return,
    everything else stays None so callers can persist a self-describing asset
    (operation, lineage, provider ids, revised prompt, dimensions) without
    overloading the image bytes.
    """
    mime_type: Optional[str] = None
    data: Optional[bytes] = None
    url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    # "generate" | "edit" | "auto" — how this image was produced.
    operation: Optional[str] = None
    # Provider-side optimization/state ids (never the only copy of the image).
    provider_response_id: Optional[str] = None
    provider_call_id: Optional[str] = None
    # The model's optimized/revised prompt, when the provider returns one.
    revised_prompt: Optional[str] = None
    # Provider ids of source images this output was edited/derived from.
    source_ids: tuple = ()
    output_index: Optional[int] = None
    # Safe-to-persist provider extras (no bytes/secrets).
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def suggested_extension(self) -> str:
        return _MIME_EXTENSION.get((self.mime_type or "").lower(), "png")


_MIME_EXTENSION = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


@dataclass(frozen=True)
class ImageGenerationOptions:
    """Configuration for the hosted image-generation tool.

    Provider-neutral knobs that map onto the OpenAI Responses ``image_generation``
    tool object. ``action`` selects automatic/forced-generate/forced-edit at the
    tool level; ``force`` additionally makes the model *call* the tool (provider
    ``tool_choice``) rather than deciding on its own. Unset fields are omitted so
    the provider applies its own defaults.
    """
    size: Optional[str] = None
    quality: Optional[str] = None
    output_format: Optional[str] = None  # png | jpeg | webp
    background: Optional[str] = None  # opaque | transparent | auto
    output_compression: Optional[int] = None
    partial_images: Optional[int] = None
    # auto | generate | edit
    action: str = "auto"
    # When True, request that the model invoke the image tool (tool_choice).
    force: bool = False
    # Provider-specific tool keys passed through verbatim (e.g. input_image_mask).
    extra: Optional[Dict[str, Any]] = None

    def to_tool_dict(self) -> Dict[str, Any]:
        """The ``{"type": "image_generation", ...}`` tool object (None elided)."""
        tool: Dict[str, Any] = {"type": "image_generation"}
        for key in ("size", "quality", "output_format", "background"):
            value = getattr(self, key)
            if value is not None:
                tool[key] = value
        if self.output_compression is not None:
            tool["output_compression"] = self.output_compression
        if self.partial_images is not None:
            tool["partial_images"] = self.partial_images
        if self.extra:
            tool.update(self.extra)
        return tool


@dataclass(frozen=True)
class ImageInput:
    """A source image for editing or visual reference.

    Carries inline ``data`` bytes (the durable path — survives provider state
    expiry) or a provider ``file_id`` / hosted ``url``.
    """
    data: Optional[bytes] = None
    mime_type: Optional[str] = None
    file_id: Optional[str] = None
    url: Optional[str] = None


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
    trace: Dict[str, Any] = field(default_factory=dict)

    # Generated media (image-out). Empty for text/tool responses.
    images: List[GeneratedImage] = field(default_factory=list)

    # A compact snapshot of the originating request, attached by the Client so a
    # Result is self-describing (used by `to_record()`). None for raw provider calls.
    request: Optional[Dict[str, Any]] = None

    @property
    def parsed(self) -> Any:
        return self.data

    def to_record(self) -> "CallRecord":
        """Build a reproducible, serializable record of this call."""
        from .record import CallRecord  # local import to avoid a cycle

        return CallRecord.from_result(self)


# -------------------------
# Request inspection (dry-run)
# -------------------------

_SECRET_HEADER_KEYS = {"authorization", "x-api-key", "x-goog-api-key", "api-key"}


def redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of headers with secret values masked (for inspection/logging)."""
    out: Dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _SECRET_HEADER_KEYS:
            out[k] = "Bearer ***" if v.lower().startswith("bearer ") else "***"
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class InspectedRequest:
    """The exact HTTP request SlimX would send for a call, without sending it.

    Secret header values are redacted. `payload` is the JSON body.
    """

    provider: str
    method: str
    url: str
    headers: Dict[str, str]
    payload: Dict[str, Any]

    def pretty(self) -> str:
        import json

        from .content import elide_media

        lines = [f"{self.method} {self.url}", f"# provider: {self.provider}", "# headers:"]
        lines += [f"  {k}: {v}" for k, v in self.headers.items()]
        lines.append("# payload:")
        lines.append(json.dumps(elide_media(self.payload), indent=2, ensure_ascii=False, default=str))
        return "\n".join(lines)
