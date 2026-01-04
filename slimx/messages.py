# slimx/messages.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Message:
    """
    SlimX canonical message.

    - `role`: system | user | assistant | tool
    - `content`: message content (text)
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

    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def system(content: str, *, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "Message":
        return Message("system", content, name=name, metadata=metadata or {})

    @staticmethod
    def user(content: str, *, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "Message":
        return Message("user", content, name=name, metadata=metadata or {})

    @staticmethod
    def assistant(content: str, *, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "Message":
        return Message("assistant", content, name=name, metadata=metadata or {})

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Best-effort provider-agnostic serialization.
        Providers may ignore unknown keys; adapters/providers can override if needed.
        """
        d: Dict[str, Any] = {"role": self.role, "content": self.content}

        if self.name:
            d["name"] = self.name

        # Tool-related fields (only set if present)
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            d["tool_name"] = self.tool_name

        # Optional extra fields
        if self.metadata:
            d["metadata"] = self.metadata

        return d
