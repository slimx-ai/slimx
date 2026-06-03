from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Iterable, Optional, Sequence
from ..tooling import ToolSpec
from ..low.types import ChatRequest
from ..types import Result, StreamEvent

@dataclass(frozen=True)
class ProviderCapabilities:
    tools: bool = False
    structured_output: bool = False
    streaming: bool = False
    async_chat: bool = False
    async_streaming: bool = False


class Provider(ABC):
    name: str
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abstractmethod
    def chat(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        timeout: Optional[float]=None,
    ) -> Result: ...

    @abstractmethod
    def stream(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        timeout: Optional[float]=None,
    ) -> Iterable[StreamEvent]: ...

    # NOTE: These are intentionally *not* declared as `async def` so static
    # type checkers treat them as returning an awaitable/async-iterator.
    # Implementations may use `async def` / async generators.
    def achat(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        timeout: Optional[float]=None,
    ) -> Awaitable[Result]:
        raise NotImplementedError("Async not implemented for this provider")

    def astream(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        timeout: Optional[float]=None,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("Async streaming not implemented for this provider")
