from abc import ABC, abstractmethod
from typing import AsyncIterator, Awaitable, Iterable, Sequence
from ..tooling import ToolSpec
from ..low.types import ChatRequest
from ..types import Result, StreamEvent

class Provider(ABC):
    name: str
    @abstractmethod
    def chat(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()) -> Result: ...
    @abstractmethod
    def stream(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()) -> Iterable[StreamEvent]: ...
    # NOTE: These are intentionally *not* declared as `async def` so static
    # type checkers treat them as returning an awaitable/async-iterator.
    # Implementations may use `async def` / async generators.
    def achat(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()) -> Awaitable[Result]:
        raise NotImplementedError("Async not implemented for this provider")
    def astream(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("Async streaming not implemented for this provider")
