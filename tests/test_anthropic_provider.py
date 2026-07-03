from __future__ import annotations

import asyncio

import pytest

from slimx import Message, tool
from slimx.errors import ProviderAuthError, ProviderError, ProviderRateLimitError
from slimx.low import ChatRequest
from slimx.providers.anthropic import AnthropicProvider
from slimx.providers.anthropic_async import AnthropicAsyncProvider

captured = {}


@pytest.fixture(autouse=True)
def clear_captured():
    captured.clear()


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data


def make_client(response, *, is_async=False):
    class _Client:
        def __init__(self, *, timeout=None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return response

        def get(self, url, *, headers):
            captured["url"] = url
            captured["headers"] = headers
            return response

        async def apost(self, url, *, headers, json):
            return self.post(url, headers=headers, json=json)

    if is_async:
        class _AsyncClient(_Client):
            async def post(self, url, *, headers, json):  # type: ignore[override]
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return response

        return _AsyncClient
    return _Client


def test_anthropic_tools_map_to_input_schema(monkeypatch):
    response = FakeResponse(data={"content": [{"type": "text", "text": "ok"}]})
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    provider.chat(
        ChatRequest(model="claude-x", messages=[Message.user("What is 2+3?")]),
        tools=[add],
    )

    tool_def = captured["json"]["tools"][0]
    assert tool_def["name"] == "add"
    assert tool_def["description"] == "Add two integers."
    assert tool_def["input_schema"]["type"] == "object"
    assert "a" in tool_def["input_schema"]["properties"]


def test_anthropic_parses_tool_use(monkeypatch):
    response = FakeResponse(
        data={
            "content": [
                {"type": "text", "text": "Let me add"},
                {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2, "b": 3}},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
    )
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    result = provider.chat(
        ChatRequest(model="claude-x", messages=[Message.user("What is 2+3?")]),
        tools=[add],
    )

    assert result.text == "Let me add"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "toolu_1"
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].arguments == {"a": 2, "b": 3}
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 7


def test_anthropic_maps_tool_loop_messages(monkeypatch):
    response = FakeResponse(data={"content": [{"type": "text", "text": "5"}]})
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    provider.chat(
        ChatRequest(
            model="claude-x",
            messages=[
                Message.system("Be terse."),
                Message.user("What is 2+3?"),
                Message.assistant(
                    "",
                    tool_calls=[
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
                        }
                    ],
                ),
                Message.tool("5", tool_call_id="toolu_1"),
            ],
        ),
        tools=[add],
    )

    body = captured["json"]
    assert body["system"] == "Be terse."
    # user, assistant(tool_use), user(tool_result)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assistant_block = body["messages"][1]["content"][0]
    assert assistant_block["type"] == "tool_use"
    assert assistant_block["name"] == "add"
    assert assistant_block["input"] == {"a": 2, "b": 3}
    tool_result = body["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["content"] == "5"


def test_anthropic_merges_consecutive_tool_results(monkeypatch):
    response = FakeResponse(data={"content": [{"type": "text", "text": "done"}]})
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    provider.chat(
        ChatRequest(
            model="claude-x",
            messages=[
                Message.user("Do two things"),
                Message.assistant(
                    "",
                    tool_calls=[
                        {"id": "t1", "type": "function", "function": {"name": "add", "arguments": "{}"}},
                        {"id": "t2", "type": "function", "function": {"name": "add", "arguments": "{}"}},
                    ],
                ),
                Message.tool("r1", tool_call_id="t1"),
                Message.tool("r2", tool_call_id="t2"),
            ],
        ),
        tools=[add],
    )

    messages = captured["json"]["messages"]
    # The two tool results must collapse into ONE user turn (strict alternation).
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    tool_results = messages[2]["content"]
    assert [b["tool_use_id"] for b in tool_results] == ["t1", "t2"]


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, ProviderAuthError), (403, ProviderAuthError), (429, ProviderRateLimitError), (500, ProviderError)],
)
def test_anthropic_error_mapping(monkeypatch, status_code, error_type):
    response = FakeResponse(status_code=status_code, text="boom")
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    with pytest.raises(error_type):
        provider.chat(ChatRequest(model="claude-x", messages=[Message.user("Hi")]))


def test_anthropic_async_chat_parses_tool_use(monkeypatch):
    response = FakeResponse(
        data={
            "content": [
                {"type": "tool_use", "id": "toolu_9", "name": "add", "input": {"a": 1, "b": 1}},
            ]
        }
    )
    monkeypatch.setattr(
        "slimx.providers.anthropic_async.httpx.AsyncClient",
        make_client(response, is_async=True),
    )

    provider = AnthropicAsyncProvider(api_key="k")
    result = asyncio.run(
        provider.achat(
            ChatRequest(model="claude-x", messages=[Message.user("1+1?")]),
            tools=[add],
        )
    )

    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].arguments == {"a": 1, "b": 1}


# --------------------------------------------------------------------------
# Native streaming, extra passthrough, model discovery
# --------------------------------------------------------------------------

class FakeStreamResponse:
    def __init__(self, status_code=200, chunks=None, body=b""):
        self.status_code = status_code
        self._chunks = chunks or []
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    def iter_bytes(self):
        yield from self._chunks

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c

    def read(self):
        return self._body

    async def aread(self):
        return self._body


def make_stream_client(response):
    class _Client:
        def __init__(self, *, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        def stream(self, method, url, *, headers, json):
            captured["json"] = json
            captured["url"] = url
            return response

    return _Client


def _sse(*events):
    out = b""
    for obj in events:
        import json as _json
        out += b"event: " + obj["type"].encode() + b"\ndata: " + _json.dumps(obj).encode() + b"\n\n"
    return [out]


_STREAM = _sse(
    {"type": "message_start", "message": {}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1,
     "content_block": {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '2,"b":3}'}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_stop"},
)


def test_anthropic_streams_text_and_tool_use(monkeypatch):
    monkeypatch.setattr(
        "slimx.providers.anthropic.httpx.Client", make_stream_client(FakeStreamResponse(chunks=_STREAM))
    )
    provider = AnthropicProvider(api_key="k")
    events = list(provider.stream(ChatRequest(model="claude-x", messages=[Message.user("hi")]), tools=[add]))

    assert "".join(e.text or "" for e in events if e.type == "text_delta") == "Hello"
    tool_events = [e for e in events if e.type == "tool_call"]
    assert len(tool_events) == 1
    call = tool_events[0].tool_call
    assert call is not None
    assert call.id == "toolu_1" and call.name == "add" and call.arguments == {"a": 2, "b": 3}
    assert events[-1].type == "done"
    assert captured["json"]["stream"] is True


def test_anthropic_async_streams(monkeypatch):
    monkeypatch.setattr(
        "slimx.providers.anthropic_async.httpx.AsyncClient",
        make_stream_client(FakeStreamResponse(chunks=_STREAM)),
    )
    provider = AnthropicAsyncProvider(api_key="k")

    async def collect():
        return [e async for e in provider.astream(ChatRequest(model="claude-x", messages=[Message.user("hi")]))]

    events = asyncio.run(collect())
    assert "".join(e.text or "" for e in events if e.type == "text_delta") == "Hello"
    assert events[-1].type == "done"


def test_anthropic_extra_passes_through_to_payload(monkeypatch):
    response = FakeResponse(data={"content": [{"type": "text", "text": "ok"}]})
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    provider.chat(
        ChatRequest(
            model="claude-x",
            messages=[Message.user("hi")],
            extra={"top_p": 0.9, "stop_sequences": ["STOP"], "tool_choice": {"type": "auto"}},
        )
    )
    body = captured["json"]
    assert body["top_p"] == 0.9
    assert body["stop_sequences"] == ["STOP"]
    assert body["tool_choice"] == {"type": "auto"}


def test_no_sampling_models_omit_temperature_and_sampling_extra(monkeypatch):
    # Recent Claude families (Opus 4.7/4.8, Sonnet 5, Fable/Mythos 5) reject sampling params
    # with a 400 — the payload must OMIT the keys entirely (never default or null them).
    response = FakeResponse(data={"content": [{"type": "text", "text": "ok"}]})
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    provider.chat(
        ChatRequest(
            model="claude-opus-4-8",
            messages=[Message.user("hi")],
            temperature=0.2,
            extra={"top_p": 0.9, "top_k": 40, "stop_sequences": ["STOP"]},
        )
    )
    body = captured["json"]
    assert "temperature" not in body
    assert "top_p" not in body
    assert "top_k" not in body
    # Non-sampling extra keys still pass through untouched.
    assert body["stop_sequences"] == ["STOP"]


def test_older_claude_models_keep_temperature_and_sampling_extra(monkeypatch):
    response = FakeResponse(data={"content": [{"type": "text", "text": "ok"}]})
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    provider.chat(
        ChatRequest(
            model="claude-sonnet-4-6",
            messages=[Message.user("hi")],
            temperature=0.2,
            extra={"top_p": 0.9},
        )
    )
    body = captured["json"]
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


def test_anthropic_list_models(monkeypatch):
    response = FakeResponse(
        data={"data": [{"id": "claude-haiku-4-5"}, {"id": "claude-sonnet-4-6"}, {}]}
    )
    monkeypatch.setattr("slimx.providers.anthropic.httpx.Client", make_client(response))

    provider = AnthropicProvider(api_key="k")
    assert provider.list_models() == ["claude-haiku-4-5", "claude-sonnet-4-6"]
    assert captured["url"].endswith("/v1/models")


def test_anthropic_capabilities_now_advertise_streaming():
    from slimx.providers import describe_provider

    assert describe_provider("anthropic")["streaming"] is True
    assert describe_provider("anthropic", async_mode=True)["async_streaming"] is True
