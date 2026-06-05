"""Per-provider multimodal serialization tests.

All assertions go through `build_request` (dry-run) so nothing here touches the
network. Each provider must emit its native shape for the modalities it declares,
and raise `UnsupportedModalityError` for those it doesn't.
"""

from __future__ import annotations

import base64

import pytest

from slimx import audio, document, image
from slimx.errors import UnsupportedModalityError
from slimx.low.types import ChatRequest
from slimx.messages import Message
from slimx.providers.anthropic import AnthropicProvider
from slimx.providers.google import GoogleProvider, _parse_response
from slimx.providers.ollama import OllamaProvider
from slimx.providers.openai import OpenAIProvider

PNG = b"\x89PNG\r\n\x1a\n" + b"\x01" * 40
PDF = b"%PDF-1.7\n" + b"d" * 40
WAV = b"RIFF\x00\x00\x00\x00WAVE" + b"a" * 20


def _img_req():
    return ChatRequest(model="m", messages=[Message.user("describe", images=[image(PNG, mime_type="image/png")])])


# --------------------------------------------------------------------------
# Image input — all four providers, native shapes
# --------------------------------------------------------------------------


def test_openai_image_url_shape():
    content = OpenAIProvider("k").build_request(_img_req()).payload["messages"][-1]["content"]
    img = next(p for p in content if p["type"] == "image_url")
    assert img["image_url"]["url"].startswith("data:image/png;base64,")


def test_anthropic_image_block_shape():
    content = AnthropicProvider("k").build_request(_img_req()).payload["messages"][-1]["content"]
    img = next(p for p in content if p["type"] == "image")
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"
    assert img["source"]["data"]


def test_google_inline_data_shape():
    parts = GoogleProvider("k").build_request(_img_req()).payload["contents"][-1]["parts"]
    inline = next(p for p in parts if "inlineData" in p)
    assert inline["inlineData"]["mimeType"] == "image/png"
    assert inline["inlineData"]["data"]


def test_ollama_images_array_shape():
    msg = OllamaProvider("http://x").build_request(_img_req()).payload["messages"][-1]
    assert msg["content"] == "describe"
    assert isinstance(msg["images"], list) and len(msg["images"]) == 1
    # Ollama wants bare base64, not a data: URI.
    assert not msg["images"][0].startswith("data:")
    base64.b64decode(msg["images"][0])  # valid base64


def test_image_url_passthrough_not_fetched():
    req = ChatRequest(model="m", messages=[Message.user("x", images=[image("https://ex.com/a.png")])])
    content = OpenAIProvider("k").build_request(req).payload["messages"][-1]["content"]
    img = next(p for p in content if p["type"] == "image_url")
    assert img["image_url"]["url"] == "https://ex.com/a.png"


def test_anthropic_image_url_source():
    req = ChatRequest(model="m", messages=[Message.user("x", images=[image("https://ex.com/a.png")])])
    content = AnthropicProvider("k").build_request(req).payload["messages"][-1]["content"]
    img = next(p for p in content if p["type"] == "image")
    assert img["source"] == {"type": "url", "url": "https://ex.com/a.png"}


# --------------------------------------------------------------------------
# Document input
# --------------------------------------------------------------------------


def _doc_req():
    return ChatRequest(model="m", messages=[Message.user("read", documents=[document(PDF, filename="r.pdf")])])


def test_openai_file_shape():
    content = OpenAIProvider("k").build_request(_doc_req()).payload["messages"][-1]["content"]
    f = next(p for p in content if p["type"] == "file")
    assert f["file"]["filename"] == "r.pdf"
    assert f["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_anthropic_document_shape():
    content = AnthropicProvider("k").build_request(_doc_req()).payload["messages"][-1]["content"]
    doc = next(p for p in content if p["type"] == "document")
    assert doc["source"]["media_type"] == "application/pdf"


def test_ollama_rejects_documents():
    with pytest.raises(UnsupportedModalityError):
        OllamaProvider("http://x").build_request(_doc_req())


# --------------------------------------------------------------------------
# Audio input
# --------------------------------------------------------------------------


def _audio_req():
    return ChatRequest(model="m", messages=[Message.user("hear", audio=[audio(WAV, mime_type="audio/wav")])])


def test_openai_input_audio_shape():
    content = OpenAIProvider("k").build_request(_audio_req()).payload["messages"][-1]["content"]
    a = next(p for p in content if p["type"] == "input_audio")
    assert a["input_audio"]["format"] == "wav"
    assert a["input_audio"]["data"]


def test_anthropic_rejects_audio():
    with pytest.raises(UnsupportedModalityError):
        AnthropicProvider("k").build_request(_audio_req())


def test_ollama_rejects_audio():
    with pytest.raises(UnsupportedModalityError):
        OllamaProvider("http://x").build_request(_audio_req())


# --------------------------------------------------------------------------
# Image-generation output (Gemini inline image parts -> Result.images)
# --------------------------------------------------------------------------


def test_google_parses_generated_image():
    blob = b"GENERATED_IMAGE_BYTES"
    resp = {
        "candidates": [
            {"content": {"parts": [{"text": "here you go"},
                                   {"inlineData": {"mimeType": "image/png",
                                                   "data": base64.b64encode(blob).decode()}}]}}
        ]
    }
    res = _parse_response(resp)
    assert res.text == "here you go"
    assert len(res.images) == 1
    assert res.images[0].mime_type == "image/png"
    assert res.images[0].data == blob


# --------------------------------------------------------------------------
# Inspect elision keeps the real bytes on the wire
# --------------------------------------------------------------------------


def test_inspect_pretty_elides_but_payload_keeps_bytes():
    insp = OpenAIProvider("k").build_request(_img_req())
    # Real payload still carries the base64 image (what actually gets sent).
    raw = insp.payload["messages"][-1]["content"][1]["image_url"]["url"]
    assert raw.startswith("data:image/png;base64,") and len(raw) > 64
    # The human-facing view elides it.
    assert "elided" in insp.pretty()
