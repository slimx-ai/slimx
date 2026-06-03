from __future__ import annotations

from dataclasses import dataclass

from fakes import FakeProvider
from slimx import Message, tool
from slimx.low import ChatRequest, Client
from slimx.providers.base import Provider, ProviderCapabilities
from slimx.schema import parse_json
from slimx.types import Result, StreamEvent, ToolCall


@tool
def add(a: int, b: int) -> int:
    return a + b


@dataclass
class City:
    name: str
    country: str


def test_client_attaches_trace_and_timeout():
    provider = FakeProvider()
    client = Client(provider, timeout=7.5, retries=1)
    res = client.chat(ChatRequest(model="demo", messages=[Message.user("hello")]))

    assert res.text == "fake:hello"
    assert res.trace["provider"] == "fake"
    assert res.trace["model"] == "demo"
    assert provider.timeouts == [7.5]


def test_client_stream_uses_text_delta_contract():
    provider = FakeProvider()
    events = list(Client(provider).stream(ChatRequest(model="demo", messages=[Message.user("hello")])))

    assert [event.type for event in events] == ["text_delta", "done"]


def test_auto_tool_loop_preserves_assistant_tool_call_message():
    provider = FakeProvider()
    client = Client(provider)
    res = client.chat(
        ChatRequest(model="demo", messages=[Message.user("What is 2+3?")]),
        tools=[add],
        tool_runtime="auto",
    )

    assert res.text == "fake:5"
    assert len(provider.calls) == 2
    assert provider.calls[1].messages[-2].role == "assistant"
    assert provider.calls[1].messages[-2].tool_calls[0]["function"]["name"] == "add"
    assert provider.calls[1].messages[-1].role == "tool"
    assert res.trace["tool_steps"] == 1


def test_tool_loop_preserves_toolcall_extra():
    # Provider-specific opaque data (e.g. Gemini thoughtSignature) must survive
    # the auto tool loop so it can be replayed to the same provider.
    class ExtraProvider(Provider):
        name = "extra"
        capabilities = ProviderCapabilities(tools=True)

        def __init__(self):
            self.calls: list = []

        def chat(self, req, *, tools=(), timeout=None):
            self.calls.append(req)
            if len(self.calls) == 1:
                return Result(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="add",
                            arguments={"a": 2, "b": 3},
                            extra={"thoughtSignature": "SIG"},
                        )
                    ],
                )
            return Result(text="done")

        def stream(self, req, *, tools=(), timeout=None):
            yield StreamEvent.done()

    provider = ExtraProvider()
    res = Client(provider).chat(
        ChatRequest(model="m", messages=[Message.user("2+3?")]),
        tools=[add],
        tool_runtime="auto",
    )

    assert res.text == "done"
    assistant_msg = provider.calls[1].messages[-2]
    assert assistant_msg.tool_calls[0]["extra"] == {"thoughtSignature": "SIG"}


def test_retry_succeeds_and_trace_is_backward_compatible():
    provider = FakeProvider(fail_times=1)
    res = Client(provider, retries=1).chat(ChatRequest(model="demo", messages=[Message.user("ok")]))

    assert res.text == "fake:ok"
    assert res.data is None
    assert res.parsed is None
    assert isinstance(res.trace, dict)


def test_provider_capabilities_shape():
    caps = FakeProvider.capabilities
    assert isinstance(caps, ProviderCapabilities)
    assert caps.tools is True
    assert caps.structured_output is True


def test_fake_json_output_is_parseable():
    res = Client(FakeProvider()).chat(ChatRequest(model="demo", messages=[Message.user("json please")]))
    assert parse_json(res.text)["name"] == "Paris"
