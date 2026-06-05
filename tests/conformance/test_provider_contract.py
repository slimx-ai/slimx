"""Run the SlimX Provider Contract against the reference fake and every
built-in provider, fully offline (httpx.MockTransport).

If you add a provider, add it to `BUILTINS` below — that is the whole cost of
opting into conformance.
"""

from __future__ import annotations

import asyncio

import pytest
from contract import (
    FakeConformantProvider,
    check_achat,
    check_aerror,
    check_astream,
    check_chat,
    check_error,
    check_identity,
    check_modalities,
    check_stream,
    make_transport,
    transport_installed,
)

from slimx.providers.anthropic import AnthropicProvider
from slimx.providers.anthropic_async import AnthropicAsyncProvider
from slimx.providers.google import GoogleProvider
from slimx.providers.google_async import GoogleAsyncProvider
from slimx.providers.oai import OAIProvider
from slimx.providers.oai_async import OAIAsyncProvider
from slimx.providers.ollama import OllamaProvider
from slimx.providers.ollama_async import OllamaAsyncProvider
from slimx.providers.openai import OpenAIProvider
from slimx.providers.openai_async import OpenAIAsyncProvider

_KW = {"api_key": "x", "base_url": "http://api.test/v1"}
_ANTHRO = {"api_key": "x", "base_url": "http://api.test"}
_GOOGLE = {"api_key": "x", "base_url": "http://api.test/v1beta"}

# name -> (build_sync, build_async)
BUILTINS = {
    "openai": (lambda: OpenAIProvider(**_KW), lambda: OpenAIAsyncProvider(**_KW)),
    "oai": (lambda: OAIProvider(**_KW), lambda: OAIAsyncProvider(**_KW)),
    "anthropic": (lambda: AnthropicProvider(**_ANTHRO), lambda: AnthropicAsyncProvider(**_ANTHRO)),
    "google": (lambda: GoogleProvider(**_GOOGLE), lambda: GoogleAsyncProvider(**_GOOGLE)),
    "ollama": (
        lambda: OllamaProvider("http://api.test"),
        lambda: OllamaAsyncProvider("http://api.test"),
    ),
}


def test_reference_fake_conforms():
    provider = FakeConformantProvider()
    check_identity(provider)
    check_chat(provider)
    check_stream(provider)
    asyncio.run(check_achat(provider))
    asyncio.run(check_astream(provider))


@pytest.mark.parametrize("name", list(BUILTINS))
def test_builtin_sync_contract(name):
    build_sync, _ = BUILTINS[name]
    provider = build_sync()
    check_identity(provider)
    check_modalities(provider)

    with transport_installed(make_transport(name)):
        check_chat(provider)
        if provider.capabilities.streaming:
            check_stream(provider)

    with transport_installed(make_transport(name, error_status=500)):
        check_error(provider)


@pytest.mark.parametrize("name", list(BUILTINS))
def test_builtin_async_contract(name):
    _, build_async = BUILTINS[name]
    provider = build_async()
    caps = provider.capabilities
    check_identity(provider)
    check_modalities(provider)

    async def run():
        with transport_installed(make_transport(name)):
            if caps.async_chat:
                await check_achat(provider)
            if caps.async_streaming:
                await check_astream(provider)
        if caps.async_chat:
            with transport_installed(make_transport(name, error_status=500)):
                await check_aerror(provider)

    asyncio.run(run())


def test_every_default_provider_is_covered():
    # Guard against adding a provider to the registry but forgetting conformance.
    from slimx.providers._defaults import DEFAULT_FACTORIES

    assert set(DEFAULT_FACTORIES) == set(BUILTINS)
