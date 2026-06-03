from __future__ import annotations

import asyncio
import pytest

from slimx import Message, tool
from slimx.errors import ProviderAuthError, ProviderError, ProviderRateLimitError
from slimx.low import ChatRequest
from slimx.providers.google import GoogleProvider
from slimx.providers.google_async import GoogleAsyncProvider


class FakeResponse:
    def __init__(self, status_code=200, data=None, text="", chunks=None):
        self.status_code = status_code
        self._data = data or {}
        self.text = text
        self._chunks = chunks or []

    def json(self):
        return self._data

    def iter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class FakeClient:
    def __init__(self, *, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, *, headers, json):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse(
            data={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Hello from Gemini"},
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 7,
                },
            }
        )

    def stream(self, method, url, *, headers, json):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse(
            chunks=[
                b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n\n',
                b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}\n\n',
            ]
        )


class AsyncFakeResponse:
    def __init__(self, status_code=200, data=None, text="", chunks=None):
        self.status_code = status_code
        self._data = data or {}
        self.text = text
        self._chunks = chunks or []

    def json(self):
        return self._data

    async def aread(self):
        return self.text.encode("utf-8")

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class AsyncFakeClient:
    def __init__(self, *, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, *, headers, json):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return AsyncFakeResponse(
            data={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Async Gemini"},
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 5,
                },
            }
        )

    def stream(self, method, url, *, headers, json):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return AsyncFakeResponse(
            chunks=[
                b'data: {"candidates":[{"content":{"parts":[{"text":"As"}]}}]}\n\n',
                b'data: {"candidates":[{"content":{"parts":[{"text":"ync"}]}}]}\n\n',
            ]
        )


captured = {}


@pytest.fixture(autouse=True)
def clear_captured():
    captured.clear()


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_google_chat_builds_generate_content_payload(monkeypatch):
    monkeypatch.setattr("slimx.providers.google.httpx.Client", FakeClient)

    provider = GoogleProvider(api_key="test-key", base_url="https://google.local/v1beta")
    result = provider.chat(
        ChatRequest(
            model="gemini-3.5-flash",
            messages=[
                Message.system("You are concise."),
                Message.user("Hello"),
            ],
            temperature=0.2,
            max_tokens=64,
        )
    )

    assert result.text == "Hello from Gemini"
    assert captured["url"] == "https://google.local/v1beta/models/gemini-3.5-flash:generateContent"
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["json"]["systemInstruction"]["parts"][0]["text"] == "You are concise."
    assert captured["json"]["contents"][0]["role"] == "user"
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "Hello"
    assert captured["json"]["generationConfig"]["temperature"] == 0.2
    assert captured["json"]["generationConfig"]["maxOutputTokens"] == 64


def test_google_chat_parses_text_and_usage(monkeypatch):
    monkeypatch.setattr("slimx.providers.google.httpx.Client", FakeClient)

    provider = GoogleProvider(api_key="test-key")
    result = provider.chat(
        ChatRequest(
            model="gemini-3.5-flash",
            messages=[Message.user("Hi")],
        )
    )

    assert result.text == "Hello from Gemini"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 4
    assert result.usage.total_tokens == 7


def test_google_json_mode_maps_response_mime_type(monkeypatch):
    monkeypatch.setattr("slimx.providers.google.httpx.Client", FakeClient)

    provider = GoogleProvider(api_key="test-key")
    provider.chat(
        ChatRequest(
            model="gemini-3.5-flash",
            messages=[Message.user("Return JSON")],
            response_format="json_object",
        )
    )

    assert captured["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_google_tools_map_to_function_declarations(monkeypatch):
    monkeypatch.setattr("slimx.providers.google.httpx.Client", FakeClient)

    provider = GoogleProvider(api_key="test-key")
    provider.chat(
        ChatRequest(
            model="gemini-3.5-flash",
            messages=[Message.user("What is 2+3?")],
        ),
        tools=[add],
    )

    declaration = captured["json"]["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "add"
    assert declaration["description"] == "Add two integers."
    assert declaration["parameters"]["type"] == "object"
    assert "a" in declaration["parameters"]["properties"]
    assert "b" in declaration["parameters"]["properties"]


def test_google_parses_function_call(monkeypatch):
    class FunctionCallClient(FakeClient):
        def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse(
                data={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "call_1",
                                            "name": "add",
                                            "args": {"a": 2, "b": 3},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr("slimx.providers.google.httpx.Client", FunctionCallClient)

    provider = GoogleProvider(api_key="test-key")
    result = provider.chat(
        ChatRequest(
            model="gemini-3.5-flash",
            messages=[Message.user("What is 2+3?")],
        ),
        tools=[add],
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].arguments == {"a": 2, "b": 3}


def test_google_maps_slimx_tool_loop_messages_to_function_response(monkeypatch):
    monkeypatch.setattr("slimx.providers.google.httpx.Client", FakeClient)

    provider = GoogleProvider(api_key="test-key")
    provider.chat(
        ChatRequest(
            model="gemini-3.5-flash",
            messages=[
                Message.user("What is 2+3?"),
                Message.assistant(
                    "",
                    tool_calls=[
                        {
                            "id": "call_add",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a":2,"b":3}',
                            },
                        }
                    ],
                ),
                Message.tool("5", tool_call_id="call_add"),
            ],
        ),
        tools=[add],
    )

    model_turn = captured["json"]["contents"][1]
    tool_turn = captured["json"]["contents"][2]

    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["functionCall"]["name"] == "add"
    assert model_turn["parts"][0]["functionCall"]["args"] == {"a": 2, "b": 3}

    assert tool_turn["role"] == "user"
    assert tool_turn["parts"][0]["functionResponse"]["name"] == "add"
    assert tool_turn["parts"][0]["functionResponse"]["id"] == "call_add"
    assert tool_turn["parts"][0]["functionResponse"]["response"]["result"] == 5


def test_google_stream_emits_text_delta_and_done(monkeypatch):
    monkeypatch.setattr("slimx.providers.google.httpx.Client", FakeClient)

    provider = GoogleProvider(api_key="test-key")
    events = list(
        provider.stream(
            ChatRequest(
                model="gemini-3.5-flash",
                messages=[Message.user("Hello")],
            )
        )
    )

    assert [event.type for event in events] == ["text_delta", "text_delta", "done"]
    assert "".join(event.text or "" for event in events) == "Hello"
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.5-flash:streamGenerateContent?alt=sse"
    )


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, ProviderAuthError),
        (403, ProviderAuthError),
        (429, ProviderRateLimitError),
        (500, ProviderError),
    ],
)
def test_google_error_mapping(monkeypatch, status_code, error_type):
    class ErrorClient(FakeClient):
        def post(self, url, *, headers, json):
            return FakeResponse(status_code=status_code, text="provider error")

    monkeypatch.setattr("slimx.providers.google.httpx.Client", ErrorClient)

    provider = GoogleProvider(api_key="test-key")

    with pytest.raises(error_type):
        provider.chat(
            ChatRequest(
                model="gemini-3.5-flash",
                messages=[Message.user("Hi")],
            )
        )


def test_google_async_chat(monkeypatch):
    monkeypatch.setattr("slimx.providers.google_async.httpx.AsyncClient", AsyncFakeClient)

    provider = GoogleAsyncProvider(api_key="test-key")
    result = asyncio.run(
        provider.achat(
            ChatRequest(
                model="gemini-3.5-flash",
                messages=[Message.user("Hi")],
            )
        )
    )

    assert result.text == "Async Gemini"
    assert result.usage.prompt_tokens == 2
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 5


def test_google_async_stream(monkeypatch):
    monkeypatch.setattr("slimx.providers.google_async.httpx.AsyncClient", AsyncFakeClient)

    async def collect_events():
        provider = GoogleAsyncProvider(api_key="test-key")
        events = []

        async for event in provider.astream(
            ChatRequest(
                model="gemini-3.5-flash",
                messages=[Message.user("Hi")],
            )
        ):
            events.append(event)

        return events

    events = asyncio.run(collect_events())

    assert [event.type for event in events] == ["text_delta", "text_delta", "done"]
    assert "".join(event.text or "" for event in events) == "Async"