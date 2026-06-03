from __future__ import annotations

from slimx.providers import describe_provider, list_providers
from slimx.providers.base import ProviderCapabilities


def test_describe_provider_needs_no_credentials():
    # Introspection must work without API keys or a running server.
    info = describe_provider("google")
    assert info["name"] == "google"
    assert info["native"] is True
    assert info["tools"] is True
    assert info["structured_output"] is True
    assert info["streaming"] is True


def test_describe_provider_marks_oai_as_non_native():
    info = describe_provider("oai")
    assert info["native"] is False


def test_describe_provider_async_mode_reports_async_capabilities():
    sync_info = describe_provider("openai", async_mode=False)
    async_info = describe_provider("openai", async_mode=True)
    assert sync_info["async_chat"] is False
    assert async_info["async_chat"] is True
    assert async_info["async_streaming"] is True


def test_every_default_provider_is_describable():
    for name in list_providers():
        info = describe_provider(name)
        assert set(info) == {
            "name",
            "native",
            "tools",
            "structured_output",
            "streaming",
            "async_chat",
            "async_streaming",
        }


def test_model_exposes_capabilities(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from slimx import llm

    m = llm("openai:gpt-4.1-nano")
    assert isinstance(m.capabilities, ProviderCapabilities)
    assert m.capabilities.tools is True
