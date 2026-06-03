from __future__ import annotations

from typing import Iterable, Sequence

from slimx.errors import ProviderTimeoutError
from slimx.low.types import ChatRequest
from slimx.providers.base import Provider, ProviderCapabilities
from slimx.tooling import ToolSpec
from slimx.types import Result, StreamEvent, ToolCall


class FakeProvider(Provider):
    name = "fake"
    capabilities = ProviderCapabilities(tools=True, structured_output=True, streaming=True)

    def __init__(self, *, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls: list[ChatRequest] = []
        self.timeouts: list[float | None] = []

    def chat(self, req: ChatRequest, *, tools: Sequence[ToolSpec] = (), timeout: float | None = None) -> Result:
        self.calls.append(req)
        self.timeouts.append(timeout)
        if self.fail_times:
            self.fail_times -= 1
            # Simulate a transient failure (the kind retry() is meant to recover from).
            raise ProviderTimeoutError("transient")

        if tools and len(self.calls) == 1:
            return Result(
                text="",
                tool_calls=[ToolCall(id="call_add", name=tools[0].name, arguments={"a": 2, "b": 3})],
            )

        last = next((m.content for m in reversed(req.messages) if m.role in {"user", "tool"}), "")
        if "json" in last.lower():
            return Result(text='{"name":"Paris","country":"France"}')
        return Result(text=f"fake:{last}")

    def stream(self, req: ChatRequest, *, tools: Sequence[ToolSpec] = (), timeout: float | None = None) -> Iterable[StreamEvent]:
        self.timeouts.append(timeout)
        yield StreamEvent.text_delta("fake")
        yield StreamEvent.done()
