"""Unit tests for the multimodal content primitive: helpers, MIME inference,
Message normalization/serialization, capability gating, and media elision."""

from __future__ import annotations

import base64
import io

import pytest

from slimx import audio, document, image
from slimx.content import (
    AudioPart,
    DocumentPart,
    ImagePart,
    TextPart,
    elide_media,
    guard_modalities,
    to_base64,
    to_data_uri,
)
from slimx.errors import SlimXError, UnsupportedModalityError
from slimx.messages import Message
from slimx.providers.base import ProviderCapabilities

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"x" * 32
WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"x" * 16


# --------------------------------------------------------------------------
# Source loading + MIME inference
# --------------------------------------------------------------------------


def test_image_from_bytes_sniffs_png():
    part = image(PNG)
    assert isinstance(part, ImagePart)
    assert part.mime_type == "image/png"
    assert part.data == PNG
    assert part.url is None


def test_image_from_bytes_sniffs_jpeg():
    assert image(JPEG).mime_type == "image/jpeg"


def test_image_from_path(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(PNG)
    part = image(str(p))
    assert part.mime_type == "image/png"
    assert part.data == PNG


def test_image_from_file_like():
    part = image(io.BytesIO(JPEG), mime_type="image/jpeg")
    assert part.data == JPEG


def test_image_from_http_url_is_passthrough_no_fetch():
    part = image("https://example.com/cat.png")
    assert part.url == "https://example.com/cat.png"
    assert part.data is None  # never fetched during construction
    assert part.mime_type == "image/png"  # inferred from extension


def test_image_from_data_uri():
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    part = image(uri)
    assert part.mime_type == "image/png"
    assert part.data == PNG


def test_unknown_bytes_without_mime_raises():
    with pytest.raises(SlimXError):
        image(b"not a known magic header at all")


def test_explicit_mime_wins():
    assert image(b"whatever", mime_type="image/webp").mime_type == "image/webp"


def test_document_infers_pdf_and_filename(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(PDF)
    part = document(str(p))
    assert isinstance(part, DocumentPart)
    assert part.mime_type == "application/pdf"
    assert part.filename == "report.pdf"


def test_audio_sniffs_wav():
    part = audio(WAV)
    assert isinstance(part, AudioPart)
    assert part.mime_type == "audio/wav"


def test_detail_passthrough():
    assert image(PNG, detail="high").detail == "high"


# --------------------------------------------------------------------------
# Message normalization + serialization
# --------------------------------------------------------------------------


def test_text_only_message_is_unchanged():
    m = Message.user("hello")
    assert not m.is_multimodal()
    assert m.to_dict() == {"role": "user", "content": "hello"}


def test_content_parts_synthesizes_leading_text():
    m = Message.user("look", images=[image(PNG)])
    parts = m.content_parts()
    assert isinstance(parts[0], TextPart) and parts[0].text == "look"
    assert isinstance(parts[1], ImagePart)


def test_no_duplicate_text_when_textpart_present():
    m = Message.user("", parts=[TextPart("only"), image(PNG)])
    texts = [p for p in m.content_parts() if isinstance(p, TextPart)]
    assert len(texts) == 1 and texts[0].text == "only"


def test_openai_content_shape():
    m = Message.user("hi", images=[image(PNG)])
    d = m.to_dict()
    assert isinstance(d["content"], list)
    assert d["content"][0] == {"type": "text", "text": "hi"}
    assert d["content"][1]["type"] == "image_url"
    assert d["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_media_only_message_has_empty_text():
    m = Message.user(images=[image(PNG)])
    assert m.content == ""
    assert all(p["type"] != "text" for p in m.to_dict()["content"])


# --------------------------------------------------------------------------
# Serialization helpers
# --------------------------------------------------------------------------


def test_to_data_uri_roundtrip():
    uri = to_data_uri("image/png", PNG)
    assert uri == "data:image/png;base64," + to_base64(PNG)


# --------------------------------------------------------------------------
# Capability gating
# --------------------------------------------------------------------------


def test_guard_blocks_image_when_no_vision():
    caps = ProviderCapabilities()  # all multimodal flags default False
    msgs = [Message.user("x", images=[image(PNG)])]
    with pytest.raises(UnsupportedModalityError):
        guard_modalities(msgs, caps, "fake")


def test_guard_allows_image_when_vision():
    caps = ProviderCapabilities(vision=True)
    guard_modalities([Message.user("x", images=[image(PNG)])], caps, "fake")  # no raise


def test_guard_blocks_audio_when_no_audio_in():
    caps = ProviderCapabilities(vision=True)  # vision yes, audio no
    with pytest.raises(UnsupportedModalityError):
        guard_modalities([Message.user("x", audio=[audio(WAV)])], caps, "fake")


def test_guard_ignores_text_only():
    guard_modalities([Message.user("plain text")], ProviderCapabilities(), "fake")  # no raise


# --------------------------------------------------------------------------
# Media elision (display/record only)
# --------------------------------------------------------------------------


def test_elide_data_uri():
    big = "data:image/png;base64," + ("A" * 4000)
    out = elide_media({"url": big})
    assert "elided" in out["url"]
    assert "AAAA" not in out["url"]


def test_elide_bare_base64_blob():
    out = elide_media({"data": "Q" * 1000})
    assert "elided" in out["data"]


def test_elide_leaves_short_strings_alone():
    assert elide_media({"text": "hello"}) == {"text": "hello"}


def test_elide_is_recursive_over_lists():
    out = elide_media({"messages": [{"content": [{"image_url": {"url": "Z" * 1000}}]}]})
    assert "elided" in out["messages"][0]["content"][0]["image_url"]["url"]
