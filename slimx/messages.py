from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class Message:
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    @staticmethod
    def system(content: str) -> "Message":
        return Message("system", content)

    @staticmethod
    def user(content: str, name: Optional[str] = None) -> "Message":
        return Message("user", content, name=name)

    @staticmethod
    def assistant(content: str, name: Optional[str] = None) -> "Message":
        return Message("assistant", content, name=name)

    @staticmethod
    def tool(content: str, tool_call_id: str) -> "Message":
        return Message("tool", content, tool_call_id=tool_call_id)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d
