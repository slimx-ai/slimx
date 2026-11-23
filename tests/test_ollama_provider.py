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


_DONE_LINE = (
    b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",'
    b'"prompt_eval_count":3,"eval_count":2}\n'
)
_STREAM_ENDINGS = {
    # The runtime's final line arrives: the `done` event carries it as the completion record.
    "completed": (
        [b'{"message":{"content":"Hi"}}\n', _DONE_LINE],
        ["text_delta", "done"],
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 3,
            "eval_count": 2,
        },
    ),
    # Cut short: the stream still ends in one `done`, with no completion record.
    "cut short": ([b'{"message":{"content":"Hi"}}\n'], ["text_delta", "done"], None),
    # A failure after the response started is a line of its own, never silently dropped.
    "runtime error": (
        [b'{"message":{"content":"Hi"}}\n', b'{"error":"CUDA error: out of memory"}\n'],
        ["text_delta", "error", "done"],
        None,
    ),
}


def _stream_sync(monkeypatch, chunks):
    monkeypatch.setattr("slimx.providers.ollama.httpx.Client", _make_client(chunks, {}))
    req = ChatRequest(model="m", messages=[Message.user("hi")])
    return list(OllamaProvider("http://x").stream(req))


def _stream_async(monkeypatch, chunks):
    monkeypatch.setattr("slimx.providers.ollama_async.httpx.AsyncClient", _make_client(chunks, {}))
    req = ChatRequest(model="m", messages=[Message.user("hi")])

    async def collect():
        return [event async for event in OllamaAsyncProvider("http://x").astream(req)]

    return asyncio.run(collect())


def test_ollama_stream_ends_with_the_runtime_completion_record_or_none(monkeypatch):
    for collect in (_stream_sync, _stream_async):
        for ending, (chunks, types, completion) in _STREAM_ENDINGS.items():
            events = collect(monkeypatch, chunks)
            assert [event.type for event in events] == types, (collect.__name__, ending)
            assert events[-1].raw == completion, (collect.__name__, ending)
            assert events[0].text == "Hi"


def test_ollama_stream_reports_a_runtime_error_line_as_an_error_event(monkeypatch):
    chunks = _STREAM_ENDINGS["runtime error"][0]
    for collect in (_stream_sync, _stream_async):
        error = [event for event in collect(monkeypatch, chunks) if event.type == "error"]
        assert len(error) == 1, collect.__name__
        assert error[0].error == "CUDA error: out of memory"
        assert error[0].raw == {"error": "CUDA error: out of memory"}


def test_ollama_stream_keeps_tool_calls_and_stops_at_the_final_line(monkeypatch):
    chunks = [
        b'{"message":{"tool_calls":[{"function":{"name":"add","arguments":{"a":1,"b":1}}}]}}\n',
        _DONE_LINE,
        b'{"message":{"content":"after the end"}}\n',
    ]
    for collect in (_stream_sync, _stream_async):
        events = collect(monkeypatch, chunks)
        assert [event.type for event in events] == ["tool_call", "done"], collect.__name__
        call = events[0].tool_call
        assert call is not None and call.arguments == {"a": 1, "b": 1}
        assert events[-1].raw["done_reason"] == "stop"


def test_ollama_capabilities_now_advertise_tools_and_json():
    from slimx.providers import describe_provider

    info = describe_provider("ollama")
    assert info["tools"] is True
    assert info["structured_output"] is True


def test_model_request_context_reaches_every_ollama_http_body():
    import json
    import httpx
    from slimx import llm, allm
    from tests.conformance.contract import transport_installed

    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        text = '{}' if len(body["messages"]) > 2 else 'first proposal'
        frame = {"message": {"content": text}, "done": True, "done_reason": "stop"}
        return httpx.Response(200, content=json.dumps(frame).encode() + b'\n')

    options = {"num_ctx": 16384, "truncate": False, "shift": False}
    kwargs = {"completion_options": options, "max_tokens": 4096, "retries": 0}

    async def exercise_async():
        model = allm("ollama:test", **kwargs)
        await model("chat")
        _ = [event async for event in model.astream("stream")]
        await model.json("plan", schema={"type": "object"}, repair=1)

    with transport_installed(httpx.MockTransport(respond)):
        model = llm("ollama:test", **kwargs)
        model("chat")
        list(model.stream("stream"))
        model.json("plan", schema={"type": "object"}, repair=1)
        asyncio.run(exercise_async())
    assert len(bodies) == 8
    for body in bodies:
        assert body["options"] == {"num_ctx": 16384, "num_predict": 4096}
        assert body["truncate"] is False and body["shift"] is False
