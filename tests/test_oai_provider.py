from __future__ import annotations

import pytest

from slimx import Message
from slimx.errors import ProviderAuthError
from slimx.low import ChatRequest
from slimx.providers import get_provider, list_providers
from slimx.providers.oai import OAIProvider
from slimx.providers.oai_async import OAIAsyncProvider


def test_oai_provider_is_registered(monkeypatch):
    monkeypatch.setenv("SLIMX_OAI_BASE_URL", "http://localhost:8000/v1")

    assert "oai" in list_providers()

    provider = get_provider("oai")

    assert isinstance(provider, OAIProvider)
    assert provider.name == "oai"


def test_oai_async_provider_is_registered(monkeypatch):
    monkeypatch.setenv("SLIMX_OAI_BASE_URL", "http://localhost:8000/v1")

    provider = get_provider("oai", async_mode=True)

    assert isinstance(provider, OAIAsyncProvider)
    assert provider.name == "oai"


def test_oai_requires_base_url(monkeypatch):
    monkeypatch.delenv("SLIMX_OAI_BASE_URL", raising=False)
    monkeypatch.delenv("OAI_BASE_URL", raising=False)

    with pytest.raises(ProviderAuthError):
        get_provider("oai")


def test_oai_uses_slimx_env_vars(monkeypatch):
    monkeypatch.setenv("SLIMX_OAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SLIMX_OAI_API_KEY", "test-key")

    provider = get_provider("oai")

    assert isinstance(provider, OAIProvider)
    assert provider.base_url == "http://localhost:8000/v1"
    assert provider.api_key == "test-key"


def test_oai_defaults_api_key_to_empty_for_local_servers(monkeypatch):
    monkeypatch.setenv("SLIMX_OAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("SLIMX_OAI_API_KEY", raising=False)
    monkeypatch.delenv("OAI_API_KEY", raising=False)

    provider = get_provider("oai")

    assert isinstance(provider, OAIProvider)
    assert provider.api_key == "EMPTY"


def test_oai_provider_kwargs_override_env(monkeypatch):
    monkeypatch.setenv("SLIMX_OAI_BASE_URL", "http://env.local/v1")
    monkeypatch.setenv("SLIMX_OAI_API_KEY", "env-key")

    provider = get_provider(
        "oai",
        base_url="http://kwargs.local/v1",
        api_key="kwargs-key",
    )

    assert isinstance(provider, OAIProvider)
    assert provider.base_url == "http://kwargs.local/v1"
    assert provider.api_key == "kwargs-key"

    
def test_oai_chat_uses_openai_compatible_chat_completions_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Hello from an OpenAI-compatible server",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            }

    class FakeClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("slimx.providers.openai.httpx.Client", FakeClient)

    provider = OAIProvider(api_key="test-key", base_url="http://localhost:8000/v1")
    result = provider.chat(
        ChatRequest(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[Message.user("Hello")],
            temperature=0.2,
            max_tokens=64,
        )
    )

    assert result.text == "Hello from an OpenAI-compatible server"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert captured["json"]["messages"][0]["role"] == "user"
    assert captured["json"]["messages"][0]["content"] == "Hello"
    assert captured["json"]["temperature"] == 0.2
    assert captured["json"]["max_tokens"] == 64


@pytest.mark.parametrize("contents,final", [
    ([{"type": "text", "text": "  let x"}, {"type": "reasoning", "text": "private"},
      {"type": "output_text", "text": " = 1;\n"}], "  let x = 1;\n"),
    ("", ""), ([{"type": "unrecognized", "text": "never code"}], ""),
    # Do not present only a prefix of a file as though the whole response were understood.
    ([{"type": "text", "text": "partial"}, {"type": "unknown"}], ""),
])
def test_file_generation_options_and_final_content(monkeypatch, contents, final):
    import httpx
    from slimx import llm

    bodies = []
    def send(client, url, *, headers, json):
        bodies.append(json)
        return httpx.Response(200, json={"choices": [{"finish_reason": "length",
            "message": {"content": contents, "reasoning_content": "private"}}]})
    monkeypatch.setattr(httpx.Client, "post", send)
    options = {"thinking": {"type": "disabled"}, "response_format": {"type": "text"}}
    model = llm("oai:fixture", max_tokens=64, retries=0,
                provider_kwargs={"base_url": "https://fixture.example", "api_key": "fixture"},
                completion_options=options)
    reply = model("Generate a file")
    assert reply.text == final
    assert len(bodies) == 1
    assert bodies[0] == model.inspect("Generate a file").payload
    assert all(bodies[0][key] == value for key, value in options.items())
    assert bodies[0]["max_tokens"] == 64


def test_structured_completion_keeps_json_and_validates_options(monkeypatch):
    import httpx
    from slimx import llm

    def send(client, url, *, headers, json):
        assert json["response_format"] == {"type": "json_object"}
        assert json["reasoning_effort"] == "low"
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr(httpx.Client, "post", send)
    model = llm("oai:fixture", provider_kwargs={"base_url": "https://fixture.example"},
                completion_options={"response_format": {"type": "text"}, "reasoning_effort": "low"})
    assert model.json("JSON", schema={"type": "object"}).data == {}
    for provider, options in [("ollama", {"thinking": {"type": "disabled"}}),
                              ("oai", {"thinking": {"type": "off"}}),
                              ("openai", {"thinking": {"type": "disabled"}}),
                              ("oai", {"stop": [""]}), ("oai", {"stop": ["x" * 513]}),
                              ("oai", {"reasoning_effort": "unknown"}),
                              ("oai", {"response_format": {"type": "other"}}),
                              ("oai", {"messages": []})]:
        with pytest.raises(ValueError):
            llm(f"{provider}:fixture", completion_options=options)


def test_async_structured_serialization_and_finish_reason(monkeypatch):
    import asyncio
    import httpx
    from slimx.high.api import allm

    options = {"thinking": {"type": "disabled"}, "response_format": {"type": "text"}}

    async def send(client, url, *, headers, json):
        assert json["thinking"] == {"type": "disabled"}
        assert json["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": "{}", "reasoning_content": "private"}
        }]})

    monkeypatch.setattr(httpx.AsyncClient, "post", send)
    model = allm("oai:fixture", retries=0, provider_kwargs={"base_url": "https://fixture.example"},
                 completion_options=options)
    options["thinking"]["type"] = "mutated"
    result = asyncio.run(model.json("JSON", schema={"type": "object"}))
    assert result.data == {} and result.text == "{}" and result.finish_reason == "stop"
