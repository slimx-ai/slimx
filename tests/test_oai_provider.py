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