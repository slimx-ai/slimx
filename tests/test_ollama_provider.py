from __future__ import annotations

import asyncio

from slimx import Message, tool
from slimx.low import ChatRequest
from slimx.providers.ollama import OllamaProvider
from slimx.providers.ollama_async import OllamaAsyncProvider


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _make_client(chunks, captured):
    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def iter_bytes(self):
            yield from chunks

        async def aiter_bytes(self):
            for c in chunks:
                yield c

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def stream(self, method, url, *, json):
            captured["json"] = json
            captured["url"] = url
            return FakeResponse()

    return FakeClient


def test_ollama_chat_streams_and_maps_max_tokens(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_bytes(self):
            yield b'{"message":{"content":"Hello "}}\n'
            yield b'{"message":{"content":"there"}}\n'
            yield b'{"done":true,"prompt_eval_count":3,"eval_count":2}\n'

    class FakeClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream(self, method, url, *, json):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("slimx.providers.ollama.httpx.Client", FakeClient)

    provider = OllamaProvider("http://ollama.local")
    result = provider.chat(
        ChatRequest(
            model="llama3.2:3b",
            messages=[Message.user("Hi")],
            temperature=0.2,
            max_tokens=42,
        ),
        timeout=123,
    )

    assert result.text == "Hello there"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 2
    assert captured["method"] == "POST"
    assert captured["url"] == "http://ollama.local/api/chat"
    assert captured["json"]["stream"] is True
    assert captured["json"]["options"]["temperature"] == 0.2
    assert captured["json"]["options"]["num_predict"] == 42


def test_ollama_sends_tools_and_parses_tool_calls(monkeypatch):
    captured = {}
    chunks = [
        b'{"message":{"role":"assistant","content":"",'
        b'"tool_calls":[{"function":{"name":"add","arguments":{"a":2,"b":3}}}]}}\n',
        b'{"done":true,"prompt_eval_count":5,"eval_count":3}\n',
    ]
    monkeypatch.setattr("slimx.providers.ollama.httpx.Client", _make_client(chunks, captured))

    provider = OllamaProvider("http://ollama.local")
    res = provider.chat(
        ChatRequest(model="llama3.2:3b", messages=[Message.user("What is 2+3?")]), tools=[add]
    )

    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert call.name == "add"
    assert call.arguments == {"a": 2, "b": 3}

    tool_def = captured["json"]["tools"][0]
    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == "add"
    assert "a" in tool_def["function"]["parameters"]["properties"]


def test_ollama_maps_tool_loop_messages(monkeypatch):
    captured = {}
    chunks = [b'{"message":{"content":"5"},"done":true}\n']
    monkeypatch.setattr("slimx.providers.ollama.httpx.Client", _make_client(chunks, captured))

    OllamaProvider("http://x").chat(
        ChatRequest(
            model="m",
            messages=[
                Message.user("What is 2+3?"),
                Message.assistant(
                    "",
                    tool_calls=[
                        {"id": "add", "type": "function",
                         "function": {"name": "add", "arguments": '{"a":2,"b":3}'}}
                    ],
                ),
                Message.tool("5", tool_call_id="add", tool_name="add"),
            ],
        ),
        tools=[add],
    )

    messages = captured["json"]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "add"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"a": 2, "b": 3}
    assert messages[2]["tool_name"] == "add"
    assert messages[2]["content"] == "5"


def test_ollama_json_mode_maps_to_format(monkeypatch):
    captured = {}
    chunks = [b'{"message":{"content":"{}"},"done":true}\n']
    monkeypatch.setattr("slimx.providers.ollama.httpx.Client", _make_client(chunks, captured))

    OllamaProvider("http://x").chat(
        ChatRequest(model="m", messages=[Message.user("json")], response_format="json_object")
    )
    assert captured["json"]["format"] == "json"


def test_ollama_async_parses_tool_calls(monkeypatch):
    captured = {}
    chunks = [
        b'{"message":{"tool_calls":[{"function":{"name":"add","arguments":{"a":1,"b":1}}}]}}\n',
        b'{"done":true}\n',
    ]
    monkeypatch.setattr("slimx.providers.ollama_async.httpx.AsyncClient", _make_client(chunks, captured))

    provider = OllamaAsyncProvider("http://x")
    res = asyncio.run(
        provider.achat(ChatRequest(model="m", messages=[Message.user("1+1?")]), tools=[add])
    )
    assert res.tool_calls[0].name == "add"
    assert res.tool_calls[0].arguments == {"a": 1, "b": 1}


def test_ollama_capabilities_now_advertise_tools_and_json():
    from slimx.providers import describe_provider

    info = describe_provider("ollama")
    assert info["tools"] is True
    assert info["structured_output"] is True
