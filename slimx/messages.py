# slimx/messages.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .content import (
    AudioPart,
    DocumentPart,
    ImagePart,
    Part,
    TextPart,
    audio_format,
    to_base64,
    to_data_uri,
)


@dataclass(frozen=True)
class Message:
    """
    SlimX canonical message.

    - `role`: system | user | assistant | tool
    - `content`: message text (always a str; the empty string when a message is
      media-only)
    - `parts`: optional non-text content (images, documents, audio). When present
      the message is multimodal; `content` is still the text portion.
    - `name`: optional participant name
    - `tool_call_id`: provider tool call identifier (OpenAI/Anthropic)
    - `tool_name`: required by some providers for tool result messages (e.g., Ollama)
    - `metadata`: extension point for provider-specific or app-specific fields
    """
    role: str
    content: str
    name: Optional[str] = None

    # Tool message fields
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    # Multimodal content (additive; text-only messages leave this empty).
    parts: Tuple[Part, ...] = field(default_factory=tuple)

    @staticmethod
    def system(content: str, *, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "Message":
        return Message("system", content, name=name, metadata=metadata or {})

    @staticmethod
    def user(
        content: str = "",
        *,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        images: Optional[Sequence[ImagePart]] = None,
        documents: Optional[Sequence[DocumentPart]] = None,
        audio: Optional[Sequence[AudioPart]] = None,
        parts: Optional[Sequence[Part]] = None,
    ) -> "Message":
        collected: List[Part] = list(parts or [])
        collected += list(images or [])
        collected += list(documents or [])
        collected += list(audio or [])
        return Message(
            "user",
            content,
            name=name,
            metadata=metadata or {},
            parts=tuple(collected),
        )

    @staticmethod
    def assistant(
        content: str,
        *,
        name: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Message":
        return Message("assistant", content, name=name, tool_calls=tool_calls or [], metadata=metadata or {})

    @staticmethod
    def tool(
        content: str,
        *,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Message":
        # tool_call_id: OpenAI/Anthropic
        # tool_name: Ollama and some adapters
        return Message(
            "tool",
            content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            metadata=metadata or {},
        )

    def is_multimodal(self) -> bool:
        """True when the message carries non-text parts (images/documents/audio)."""
        return bool(self.parts)

    def content_parts(self) -> List[Part]:
        """Normalized content as a list of parts.

        Synthesizes a leading ``TextPart`` from ``content`` when the message has
        text but no explicit text part, so providers can iterate one uniform list
        regardless of how the message was constructed.
        """
        parts: List[Part] = list(self.parts)
        if self.content and not any(isinstance(p, TextPart) for p in parts):
            parts.insert(0, TextPart(self.content))
        return parts

    def _openai_content(self) -> Any:
        """OpenAI Chat Completions content: a str when text-only, else typed parts."""
        if not self.parts:
            return self.content
        out: List[Dict[str, Any]] = []
        for p in self.content_parts():
            if isinstance(p, TextPart):
                out.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                img: Dict[str, Any] = {"url": p.url or to_data_uri(p.mime_type, p.data or b"")}
                if p.detail:
                    img["detail"] = p.detail
                out.append({"type": "image_url", "image_url": img})
            elif isinstance(p, DocumentPart):
                file_obj: Dict[str, Any] = {}
                if p.filename:
                    file_obj["filename"] = p.filename
                if p.url and p.data is None:
                    file_obj["file_data"] = p.url
                else:
                    file_obj["file_data"] = to_data_uri(p.mime_type, p.data or b"")
                out.append({"type": "file", "file": file_obj})
            elif isinstance(p, AudioPart):
                out.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": to_base64(p.data or b""),
                        "format": audio_format(p.mime_type),
                    },
                })
        return out

    def to_dict(self) -> Dict[str, Any]:
        """
        Best-effort provider-agnostic serialization (OpenAI Chat Completions shape).
        Providers may ignore unknown keys; adapters/providers can override if needed.
        """
        d: Dict[str, Any] = {"role": self.role, "content": self._openai_content()}

        if self.name:
            d["name"] = self.name

        # Tool-related fields (only set if present)
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls

        # Optional extra fields
        if self.metadata:
            d["metadata"] = self.metadata

        return d
