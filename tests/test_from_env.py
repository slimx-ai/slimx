"""`from_env` is the single source of truth for provider env configuration, and
the `_defaults` factories delegate to it (review item R5)."""

from __future__ import annotations

import pytest

from slimx.errors import ProviderAuthError
from slimx.providers.anthropic import AnthropicProvider
from slimx.providers.google import GoogleProvider
from slimx.providers.oai import OAIProvider
from slimx.providers.ollama import OllamaProvider
from slimx.providers.openai import OpenAIProvider
from slimx.providers.openai_async import OpenAIAsyncProvider
from slimx.providers.registry import get_provider

_ENV_VARS = [
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_VERSION",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_BASE_URL",
    "OLLAMA_BASE_URL",
    "SLIMX_OAI_API_KEY", "OAI_API_KEY", "SLIMX_OAI_BASE_URL", "OAI_BASE_URL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# Overrides win over the environment
# --------------------------------------------------------------------------


def test_openai_override_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env")
    p = OpenAIProvider.from_env(api_key="explicit", base_url="http://h/v1")
    assert p.api_key == "explicit"
    assert p.base_url == "http://h/v1"


def test_anthropic_from_env_all_three(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://anthro")
    monkeypatch.setenv("ANTHROPIC_VERSION", "2099-01-01")
    p = AnthropicProvider.from_env()
    assert (p.api_key, p.base_url, p.version) == ("ak", "http://anthro", "2099-01-01")


def test_ollama_default_base_url():
    assert OllamaProvider.from_env().base_url == "http://localhost:11434"


# --------------------------------------------------------------------------
# Factories delegate to from_env (single source of truth)
# --------------------------------------------------------------------------


def test_factory_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env/v1")
    p = get_provider("openai")
    assert isinstance(p, OpenAIProvider)
    assert p.api_key == "env-key"
    assert p.base_url == "http://env/v1"


def test_missing_key_raises(monkeypatch):
    with pytest.raises(ProviderAuthError):
        get_provider("openai")


def test_google_gemini_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem")
    assert GoogleProvider.from_env().api_key == "gem"


def test_google_prefers_google_over_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "primary")
    monkeypatch.setenv("GEMINI_API_KEY", "secondary")
    assert GoogleProvider.from_env().api_key == "primary"


def test_oai_requires_base_url():
    with pytest.raises(ProviderAuthError):
        OAIProvider.from_env()  # no SLIMX_OAI_BASE_URL / OAI_BASE_URL


def test_oai_defaults_api_key_to_empty(monkeypatch):
    monkeypatch.setenv("OAI_BASE_URL", "http://localhost:8000/v1")
    p = OAIProvider.from_env()
    assert p.base_url == "http://localhost:8000/v1"
    assert p.api_key == "EMPTY"


def test_async_factory_delegates(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ak")
    p = get_provider("openai", async_mode=True)
    assert isinstance(p, OpenAIAsyncProvider)
    assert p.api_key == "ak"
    assert p.capabilities.async_chat is True
