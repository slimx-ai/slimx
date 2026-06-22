"""OpenAI **Responses API** image-tool tests.

Offline only: an ``httpx.MockTransport`` returns canned ``/responses`` bodies and
SSE streams so the hosted ``image_generation`` tool, image editing, multi-output
parsing, and partial/final streaming events are exercised without any network.
"""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager

import httpx
import pytest

from slimx import ImageGenerationOptions, ImageInput, Message
from slimx.errors import UnsupportedModalityError
from slimx.high.api import Model
from slimx.low.types import ChatRequest, ImageEditRequest
from slimx.providers._openai_responses import (
    ResponsesStreamTranslator,
    build_responses_payload,
    parse_responses_response,
)
from slimx.providers.anthropic import AnthropicProvider
from slimx.providers.oai import OAIProvider
from slimx.providers.openai import OpenAIProvider

# A real 1x1 PNG (valid signature + IHDR), so MIME/dimension sniffing is exercised.
PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000000020001e221bc330000000049454e44ae426082"
)
B64 = base64.b64encode(PNG_1x1).decode()


@contextmanager
def transport_installed(handler):
    transport = httpx.MockTransport(handler)
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


def _capture(response_json, *, captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=response_json)

    return handler


def _msg_item(text):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def _image_item(call_id="ig_1", b64=B64, revised="a gray tabby cat"):
    return {
        "type": "image_generation_call",
        "id": call_id,
        "status": "completed",
        "revised_prompt": revised,
        "result": b64,
    }


def _responses_body(*output, response_id="resp_abc"):
    return {
        "id": response_id,
        "output": list(output),
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


# --------------------------------------------------------------------------
# Payload building / routing
# --------------------------------------------------------------------------


def test_chat_with_image_tool_routes_to_responses():
    captured: dict = {}
    body = _responses_body(_msg_item("Here it is."), _image_item())
    req = ChatRequest(
        model="gpt-5.5",
        messages=[Message.user("draw a cat")],
        image_generation=ImageGenerationOptions(size="1024x1024"),
    )
    with transport_installed(_capture(body, captured=captured)):
        res = OpenAIProvider("k").chat(req)
    assert captured["path"].endswith("/responses")
    # The hosted tool object is present and configured.
    assert {"type": "image_generation", "size": "1024x1024"} in captured["payload"]["tools"]
    # input is the Responses shape (input_text), not chat/completions messages.
    assert captured["payload"]["input"][0]["content"][0]["type"] == "input_text"
    assert res.images[0].data == PNG_1x1


def test_force_generation_sets_tool_choice():
    captured: dict = {}
    req = ChatRequest(
        model="gpt-5.5",
        messages=[Message.user("make an image")],
        image_generation=ImageGenerationOptions(action="generate", force=True),
    )
    with transport_installed(_capture(_responses_body(_image_item()), captured=captured)):
        OpenAIProvider("k").chat(req)
    assert captured["payload"]["tool_choice"] == {"type": "image_generation"}


def test_previous_response_id_passed_through():
    captured: dict = {}
    req = ChatRequest(
        model="gpt-5.5",
        messages=[Message.user("make it snowy")],
        image_generation=ImageGenerationOptions(),
        previous_response_id="resp_prev",
    )
    with transport_installed(_capture(_responses_body(_image_item()), captured=captured)):
        OpenAIProvider("k").chat(req)
    assert captured["payload"]["previous_response_id"] == "resp_prev"


def test_text_only_chat_still_uses_chat_completions():
    captured: dict = {}
    chat_body = {"choices": [{"message": {"content": "hello"}}], "usage": {}}
    req = ChatRequest(model="gpt-5.5", messages=[Message.user("hi")])
    with transport_installed(_capture(chat_body, captured=captured)):
        res = OpenAIProvider("k").chat(req)
    assert captured["path"].endswith("/chat/completions")
    assert res.text == "hello"


def test_build_request_inspect_switches_to_responses():
    req = ChatRequest(
        model="gpt-5.5",
        messages=[Message.user("cat")],
        image_generation=ImageGenerationOptions(),
    )
    insp = OpenAIProvider("k").build_request(req)
    assert insp.url.endswith("/responses")
    assert any(t["type"] == "image_generation" for t in insp.payload["tools"])


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def test_parse_text_and_image():
    res = parse_responses_response(
        _responses_body(_msg_item("Here you go."), _image_item()), model="gpt-5.5"
    )
    assert res.text == "Here you go."
    assert len(res.images) == 1
    img = res.images[0]
    assert img.data == PNG_1x1
    assert img.mime_type == "image/png"  # sniffed from bytes, not declared
    assert (img.width, img.height) == (1, 1)
    assert img.provider_response_id == "resp_abc"
    assert img.provider_call_id == "ig_1"
    assert img.revised_prompt == "a gray tabby cat"
    assert img.operation == "generate"
    assert res.usage.total_tokens == 15


def test_parse_multiple_image_calls():
    res = parse_responses_response(
        _responses_body(_image_item("ig_a"), _image_item("ig_b")), model="gpt-5.5"
    )
    assert [i.provider_call_id for i in res.images] == ["ig_a", "ig_b"]
    assert [i.output_index for i in res.images] == [0, 1]


def test_parse_refusal_no_image():
    res = parse_responses_response(_responses_body(_msg_item("I can't create that.")))
    assert res.text == "I can't create that."
    assert res.images == []


def test_parse_malformed_base64_skips_image():
    bad = _responses_body(_image_item(b64="not%%%base64"))
    res = parse_responses_response(bad)
    assert res.images == []  # undecodable result is dropped, not crashed


# --------------------------------------------------------------------------
# Image editing
# --------------------------------------------------------------------------


def test_edit_image_sends_input_image_and_forces_tool():
    captured: dict = {}
    body = _responses_body(_image_item("ig_edit"))
    req = ImageEditRequest(
        model="gpt-5.5",
        instruction="make the scarf blue",
        images=[ImageInput(data=PNG_1x1, mime_type="image/png")],
    )
    with transport_installed(_capture(body, captured=captured)):
        res = OpenAIProvider("k").edit_image(req)
    content = captured["payload"]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "make the scarf blue"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert captured["payload"]["tool_choice"] == {"type": "image_generation"}
    assert res.images[0].operation == "edit"


def test_edit_image_file_id_and_multiple_sources():
    captured: dict = {}
    req = ImageEditRequest(
        model="gpt-5.5",
        instruction="combine these",
        images=[ImageInput(file_id="file_123"), ImageInput(data=PNG_1x1)],
    )
    with transport_installed(_capture(_responses_body(_image_item()), captured=captured)):
        OpenAIProvider("k").edit_image(req)
    content = captured["payload"]["input"][0]["content"]
    images = [c for c in content if c["type"] == "input_image"]
    assert images[0] == {"type": "input_image", "file_id": "file_123"}
    assert images[1]["image_url"].startswith("data:")


def test_high_level_edit_image():
    m = Model("openai:gpt-5.5", provider_kwargs={"api_key": "k"})
    with transport_installed(_capture(_responses_body(_image_item()), captured={})):
        res = m.edit_image(PNG_1x1, "add snow", quality="high")
    assert res.images[0].data == PNG_1x1
    assert res.trace["provider"] == "openai"


def test_high_level_generate_via_tool():
    captured: dict = {}
    m = Model("openai:gpt-5.5", provider_kwargs={"api_key": "k"})
    body = _responses_body(_msg_item(""), _image_item())
    with transport_installed(_capture(body, captured=captured)):
        res = m("a gray tabby cat hugging an otter", image_generation=ImageGenerationOptions())
    assert captured["path"].endswith("/responses")
    assert res.images[0].data == PNG_1x1


# --------------------------------------------------------------------------
# Streaming translation
# --------------------------------------------------------------------------


def test_stream_translator_text_partial_and_final():
    translator = ResponsesStreamTranslator(provider="openai", model="gpt-5.5")
    events = []
    events += translator.feed({"type": "response.output_text.delta", "delta": "Here "})
    events += translator.feed({"type": "response.output_text.delta", "delta": "you go"})
    events += translator.feed(
        {"type": "response.image_generation_call.in_progress", "output_index": 0}
    )
    events += translator.feed(
        {
            "type": "response.image_generation_call.partial_image",
            "partial_image_b64": B64,
            "partial_image_index": 0,
        }
    )
    events += translator.feed(
        {"type": "response.completed", "response": _responses_body(_image_item())}
    )
    events += translator.finish()
    kinds = [e.type for e in events]
    assert kinds == [
        "text_delta",
        "text_delta",
        "image_started",
        "image_partial",
        "image_completed",
        "done",
    ]
    partial = next(e for e in events if e.type == "image_partial")
    assert partial.image_partial_b64 == B64 and partial.image is None
    completed = next(e for e in events if e.type == "image_completed")
    assert completed.image.data == PNG_1x1


def test_stream_translator_error_event():
    translator = ResponsesStreamTranslator()
    events = translator.feed({"type": "response.failed", "response": {"error": {"message": "nope"}}})
    assert events[0].type == "error" and events[0].error == "nope"


def test_provider_stream_full_path():
    sse = (
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        b"event: response.completed\n"
        b"data: " + json.dumps(
            {"type": "response.completed", "response": _responses_body(_image_item())}
        ).encode() + b"\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        return httpx.Response(200, content=sse)

    req = ChatRequest(
        model="gpt-5.5",
        messages=[Message.user("cat")],
        image_generation=ImageGenerationOptions(),
    )
    with transport_installed(handler):
        events = list(OpenAIProvider("k").stream(req))
    assert "".join(e.text or "" for e in events if e.type == "text_delta") == "hi"
    completed = [e for e in events if e.type == "image_completed"]
    assert completed and completed[0].image.data == PNG_1x1
    assert events[-1].type == "done"


# --------------------------------------------------------------------------
# Capabilities + gating
# --------------------------------------------------------------------------


def test_openai_capabilities_advertise_image_tooling():
    caps = OpenAIProvider("k").capabilities
    assert caps.image_out and caps.image_edit
    assert caps.hosted_image_tool and caps.image_partial_streaming
    assert caps.image_in is caps.vision is True


def test_oai_does_not_advertise_hosted_image_tool():
    caps = OAIProvider(api_key="k", base_url="http://x/v1").capabilities
    assert caps.hosted_image_tool is False
    assert caps.image_edit is False


def test_edit_image_gated_on_unsupported_provider():
    m = Model("anthropic:claude-sonnet-4-6", provider_kwargs={"api_key": "k"})
    with pytest.raises(UnsupportedModalityError):
        m.edit_image(PNG_1x1, "x")


def test_anthropic_provider_refuses_edit_image():
    assert AnthropicProvider("k").capabilities.image_edit is False
    with pytest.raises(NotImplementedError):
        AnthropicProvider("k").edit_image(ImageEditRequest(model="m", instruction="x"))
