"""Image-generation (image-out) tests.

Offline only: an `httpx.MockTransport` returns canned wire responses so the
OpenAI Images endpoint and the Gemini image path are exercised without network.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager

import httpx
import pytest

from slimx.errors import UnsupportedModalityError
from slimx.high.api import AsyncModel, Model
from slimx.low.types import ImageRequest
from slimx.providers.anthropic import AnthropicProvider
from slimx.providers.google import GoogleProvider
from slimx.providers.oai import OAIProvider
from slimx.providers.openai import OpenAIProvider
from slimx.providers.openai_async import OpenAIAsyncProvider

BLOB = b"PNG_IMAGE_BYTES"
B64 = base64.b64encode(BLOB).decode()


@contextmanager
def transport_installed(transport: httpx.MockTransport):
    real_sync, real_async = httpx.Client, httpx.AsyncClient

    def sync_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_sync(*args, **kwargs)

    def async_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async(*args, **kwargs)

    httpx.Client = sync_factory  # type: ignore[assignment, misc]
    httpx.AsyncClient = async_factory  # type: ignore[assignment, misc]
    try:
        yield
    finally:
        httpx.Client = real_sync  # type: ignore[misc]
        httpx.AsyncClient = real_async  # type: ignore[misc]


def _openai_images_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/images/generations")
        return httpx.Response(200, json={"data": [{"b64_json": B64}]})

    return httpx.MockTransport(handler)


def _google_image_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": B64}}]}}
                ]
            },
        )

    return httpx.MockTransport(handler)


# --------------------------------------------------------------------------
# OpenAI Images endpoint
# --------------------------------------------------------------------------


def test_openai_generate_image_sync():
    provider = OpenAIProvider("k")
    req = ImageRequest(model="gpt-image-1", prompt="a red bike")
    with transport_installed(_openai_images_transport()):
        res = provider.generate_image(req)
    assert res.text == ""
    assert len(res.images) == 1
    assert res.images[0].data == BLOB
    assert res.images[0].mime_type == "image/png"


def test_openai_generate_image_async():
    provider = OpenAIAsyncProvider("k")
    req = ImageRequest(model="gpt-image-1", prompt="a red bike")

    async def run():
        with transport_installed(_openai_images_transport()):
            return await provider.agenerate_image(req)

    res = asyncio.run(run())
    assert res.images[0].data == BLOB


def test_openai_inspect_image_shape():
    insp = OpenAIProvider("k").build_image_request(
        ImageRequest(model="gpt-image-1", prompt="cat", size="1024x1024", extra={"quality": "high"})
    )
    assert insp.url.endswith("/images/generations")
    assert insp.payload == {
        "model": "gpt-image-1",
        "prompt": "cat",
        "n": 1,
        "size": "1024x1024",
        "quality": "high",
    }


def test_high_level_generate_image():
    m = Model("openai:gpt-image-1", provider_kwargs={"api_key": "k"})
    with transport_installed(_openai_images_transport()):
        res = m.generate_image("a red bike")
    assert res.images[0].data == BLOB
    assert res.trace["provider"] == "openai"  # Client attached trace


# --------------------------------------------------------------------------
# Gemini image path (routed through generateContent)
# --------------------------------------------------------------------------


def test_google_generate_image_sync():
    provider = GoogleProvider("k")
    with transport_installed(_google_image_transport()):
        res = provider.generate_image(ImageRequest(model="gemini-2.5-flash-image", prompt="a red bike"))
    assert len(res.images) == 1
    assert res.images[0].data == BLOB


def test_high_level_async_generate_image_google():
    m = AsyncModel("google:gemini-2.5-flash-image", provider_kwargs={"api_key": "k"})

    async def run():
        with transport_installed(_google_image_transport()):
            return await m.generate_image("a red bike")

    res = asyncio.run(run())
    assert res.images[0].data == BLOB


# --------------------------------------------------------------------------
# Capability gating — providers without image_out refuse
# --------------------------------------------------------------------------


def test_anthropic_provider_has_no_image_out():
    assert AnthropicProvider("k").capabilities.image_out is False
    with pytest.raises(NotImplementedError):
        AnthropicProvider("k").generate_image(ImageRequest(model="m", prompt="x"))


def test_oai_does_not_promise_image_out():
    assert OAIProvider(api_key="k", base_url="http://x/v1").capabilities.image_out is False


def test_high_level_generate_image_gated():
    m = Model("anthropic:claude-sonnet-4-6", provider_kwargs={"api_key": "k"})
    with pytest.raises(UnsupportedModalityError):
        m.generate_image("x")
