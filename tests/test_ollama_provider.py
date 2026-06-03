from __future__ import annotations

from slimx import Message
from slimx.low import ChatRequest
from slimx.providers.ollama import OllamaProvider


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
