"""The SlimX Provider Contract — reusable conformance checks.

These helpers take a *provider instance* and assert it satisfies the contract
described in DEVELOPMENT.md (Part III). They are deliberately transport-agnostic:
built-in providers are exercised offline by installing an ``httpx.MockTransport``
that returns provider-shaped wire responses, and third-party plugins can reuse
the exact same checks to claim conformance.

Nothing here touches the network.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import httpx
import pytest

from slimx.errors import ProviderError
from slimx.low.types import ChatRequest
from slimx.messages import Message
from slimx.providers.base import Provider, ProviderCapabilities
from slimx.types import Result, StreamEvent, ToolCall, Usage

# ---------------------------------------------------------------------------
# A reference implementation that fully satisfies the contract (no httpx).
# ---------------------------------------------------------------------------


class FakeConformantProvider(Provider):
    name = "fake"
    capabilities = ProviderCapabilities(
        tools=True,
        structured_output=True,
        streaming=True,
        async_chat=True,
        async_streaming=True,
    )

    def chat(self, req, *, tools=(), timeout=None) -> Result:
        return Result(
            text="hello",
            raw={"ok": True},
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
        )

    def stream(self, req, *, tools=(), timeout=None):
        yield StreamEvent.text_delta("hel")
        yield StreamEvent.text_delta("lo")
        yield StreamEvent.done()

    async def achat(self, req, *, tools=(), timeout=None) -> Result:
        return self.chat(req)

    async def astream(self, req, *, tools=(), timeout=None):
        for event in self.stream(req):
            yield event


# ---------------------------------------------------------------------------
# Offline transport: provider-shaped canned wire responses.
# ---------------------------------------------------------------------------


def _success_response(name: str, request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if name in ("openai", "oai"):
        body = json.loads(request.content or b"{}")
        if body.get("stream"):
            sse = (
                b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    if name == "anthropic":
        body = json.loads(request.content or b"{}")
        if body.get("stream"):
            sse = (
                b'event: content_block_delta\n'
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"text_delta","text":"hello"}}\n\n'
                b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            )
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    if name == "google":
        if "streamGenerateContent" in path:
            sse = (
                b'data: {"candidates":[{"content":{"parts":[{"text":"hel"}]}}]}\n\n'
                b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}\n\n'
            )
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            },
        )
    if name == "ollama":
        ndjson = (
            b'{"message":{"content":"hel"}}\n'
            b'{"message":{"content":"lo"}}\n'
            b'{"done":true,"prompt_eval_count":1,"eval_count":1}\n'
        )
        return httpx.Response(200, content=ndjson, headers={"content-type": "application/x-ndjson"})
    return httpx.Response(200, json={})


def make_transport(name: str, *, error_status: int | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if error_status is not None:
            return httpx.Response(error_status, json={"error": "boom"})
        return _success_response(name, request)

    return httpx.MockTransport(handler)


@contextmanager
def transport_installed(transport: httpx.MockTransport):
    """Route every httpx client through ``transport`` for the duration of the block."""
    real_sync, real_async = httpx.Client, httpx.AsyncClient

    def sync_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_sync(*args, **kwargs)

    def async_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async(*args, **kwargs)

    httpx.Client = sync_factory  # type: ignore[assignment, misc]
    httpx.AsyncClient = async_factory  # type: ignore[assignment, misc]
    try:
        yield
    finally:
        httpx.Client = real_sync  # type: ignore[misc]
        httpx.AsyncClient = real_async  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Contract checks. Each takes a provider instance (transport already installed).
# ---------------------------------------------------------------------------

_REQ = ChatRequest(model="m", messages=[Message.user("hi")])


def check_identity(provider: Provider) -> None:
    assert isinstance(provider.name, str) and provider.name
    assert isinstance(provider.capabilities, ProviderCapabilities)


def check_chat(provider: Provider) -> Result:
    res = provider.chat(_REQ)
    _assert_result(res)
    return res


def check_stream(provider: Provider) -> list[StreamEvent]:
    events = list(provider.stream(_REQ))
    _assert_stream(events)
    return events


def check_error(provider: Provider) -> None:
    with pytest.raises(ProviderError):
        provider.chat(_REQ)


async def check_achat(provider: Provider) -> Result:
    res = await provider.achat(_REQ)
    _assert_result(res)
    return res


async def check_astream(provider: Provider) -> list[StreamEvent]:
    events = [event async for event in provider.astream(_REQ)]
    _assert_stream(events)
    return events


async def check_aerror(provider: Provider) -> None:
    with pytest.raises(ProviderError):
        await provider.achat(_REQ)


def _assert_result(res: Result) -> None:
    assert isinstance(res, Result), "chat() must return a Result"
    assert isinstance(res.text, str), "Result.text must be a str (never None)"
    assert res.raw is not None, "Result.raw must be preserved"
    assert isinstance(res.usage, Usage), "Result.usage must be a Usage"
    assert isinstance(res.tool_calls, list)
    assert all(isinstance(tc, ToolCall) for tc in res.tool_calls), "tool_calls must be ToolCall"


def _assert_stream(events: list[StreamEvent]) -> None:
    assert events, "stream() emitted no events"
    assert all(isinstance(e, StreamEvent) for e in events)
    assert events[-1].type == "done", "stream() must terminate with a `done` event"
    assert sum(1 for e in events if e.type == "done") == 1, "exactly one `done` event"
    for e in events:
        if e.type == "text_delta":
            assert isinstance(e.text, str), "text_delta events must carry str text"
