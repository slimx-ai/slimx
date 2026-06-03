from __future__ import annotations

import json

import pytest

from slimx import Message, tool
from slimx.errors import ProviderAuthError, ProviderError, ProviderRateLimitError
from slimx.low import ChatRequest
from slimx.providers.openai import OpenAIProvider


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class FakeStreamResponse:
    def __init__(self, status_code=200, chunks=None, body=b""):
        self.status_code = status_code
        self._chunks = chunks or []
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_bytes(self):
        yield from self._chunks

    def read(self):
        return self._body


def make_client(response):
    class FakeClient:
        def __init__(self, *, timeout=None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream(self, method, url, *, headers, json):
            return response

    return FakeClient


def _sse(*objs):
    out = b""
    for obj in objs:
        out += b"data: " + json.dumps(obj).encode() + b"\n\n"
    out += b"data: [DONE]\n\n"
    return [out]


def test_streaming_reassembles_split_tool_call(monkeypatch):
    # OpenAI sends id+name in the first delta and only index afterwards.
    chunks = _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_abc", "function": {"name": "add", "arguments": '{"a":'}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '2,"b":3}'}}
        ]}}]},
    )
    monkeypatch.setattr(
        "slimx.providers.openai.httpx.Client",
        make_client(FakeStreamResponse(chunks=chunks)),
    )

    provider = OpenAIProvider(api_key="x")
    events = list(provider.stream(ChatRequest(model="m", messages=[Message.user("hi")]), tools=[add]))

    tool_events = [e for e in events if e.type == "tool_call"]
    assert len(tool_events) == 1
    call = tool_events[0].tool_call
    assert call is not None
    assert call.id == "call_abc"
    assert call.name == "add"
    assert call.arguments == {"a": 2, "b": 3}
    assert events[-1].type == "done"


def test_streaming_text_deltas(monkeypatch):
    chunks = _sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
    )
    monkeypatch.setattr(
        "slimx.providers.openai.httpx.Client",
        make_client(FakeStreamResponse(chunks=chunks)),
    )

    provider = OpenAIProvider(api_key="x")
    events = list(provider.stream(ChatRequest(model="m", messages=[Message.user("hi")])))

    assert "".join(e.text or "" for e in events) == "Hello"
    assert [e.type for e in events] == ["text_delta", "text_delta", "done"]


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, ProviderAuthError), (429, ProviderRateLimitError), (500, ProviderError)],
)
def test_streaming_error_reads_body_before_raising(monkeypatch, status_code, error_type):
    # On error the body must be read first; accessing it lazily would raise
    # httpx.ResponseNotRead and mask the real provider error.
    response = FakeStreamResponse(status_code=status_code, body=b"boom")
    monkeypatch.setattr(
        "slimx.providers.openai.httpx.Client",
        make_client(response),
    )

    provider = OpenAIProvider(api_key="x")
    with pytest.raises(error_type):
        list(provider.stream(ChatRequest(model="m", messages=[Message.user("hi")])))
