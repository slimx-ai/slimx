# slimx/content.py
"""Multimodal content parts and the helpers that build them.

A SlimX message is text-first (`Message.content: str`) but can also carry
non-text `parts` — images, documents, audio. This module defines those part
types, the `image()` / `document()` / `audio()` constructors that load them from
a path / bytes / file-like / URL, and the small utilities providers share to
serialize, gate, and elide media.

Nothing here imports `messages`, `types`, or any provider, so it sits at the
bottom of the import graph and can't create cycles.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

from .errors import SlimXError, UnsupportedModalityError

# ---------------------------------------------------------------------------
# Part types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    """An image, sourced either as inline `data` bytes or a remote `url`."""

    data: Optional[bytes] = None
    url: Optional[str] = None
    mime_type: Optional[str] = None
    detail: Optional[str] = None  # OpenAI "low" | "high" | "auto"; ignored elsewhere


@dataclass(frozen=True)
class DocumentPart:
    """A document (e.g. a PDF), inline `data` bytes or a remote `url`."""

    data: Optional[bytes] = None
    url: Optional[str] = None
    mime_type: Optional[str] = None
    filename: Optional[str] = None


@dataclass(frozen=True)
class AudioPart:
    """An audio clip, inline `data` bytes or a remote `url`."""

    data: Optional[bytes] = None
    url: Optional[str] = None
    mime_type: Optional[str] = None


Part = Union[TextPart, ImagePart, DocumentPart, AudioPart]
MediaPart = Union[ImagePart, DocumentPart, AudioPart]


# ---------------------------------------------------------------------------
# Source loading + MIME inference
# ---------------------------------------------------------------------------


def _sniff_mime(data: bytes) -> Optional[str]:
    """Best-effort MIME detection from magic bytes (no external deps)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"%PDF":
        return "application/pdf"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return "audio/mpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    return None


def _read_source(src: Any, *, fetch: bool) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Resolve a source into ``(data, url, mime_hint)``.

    Accepts raw bytes, a file-like object, a filesystem path, a ``data:`` URI, or
    an ``http(s)://`` URL. Remote URLs are passed through untouched (``url`` set,
    ``data`` None) unless ``fetch=True``, which downloads and inlines the bytes —
    so request building never makes a surprise network call by default.
    """
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src), None, None

    if hasattr(src, "read"):  # file-like
        data = src.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        name = getattr(src, "name", None)
        hint = mimetypes.guess_type(name)[0] if isinstance(name, str) else None
        return data, None, hint

    s = os.fspath(src) if isinstance(src, os.PathLike) else str(src)

    if s.startswith("data:"):
        # data:<mime>;base64,<payload>
        try:
            header, payload = s.split(",", 1)
            mime = header[len("data:"):].split(";", 1)[0] or None
            data = base64.b64decode(payload)
            return data, None, mime
        except Exception as exc:  # pragma: no cover - malformed input
            raise SlimXError(f"Invalid data URI: {exc}") from exc

    if s.startswith(("http://", "https://")):
        if fetch:
            return _fetch_url(s), None, mimetypes.guess_type(s)[0]
        return None, s, mimetypes.guess_type(s)[0]

    with open(s, "rb") as f:
        data = f.read()
    return data, None, mimetypes.guess_type(s)[0]


def _fetch_url(url: str) -> bytes:
    import httpx

    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _resolve(
    src: Any, *, mime_type: Optional[str], fetch: bool, kind: str
) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    data, url, hint = _read_source(src, fetch=fetch)
    mime = mime_type or hint or (_sniff_mime(data) if data else None)
    if data is not None and not mime:
        raise SlimXError(
            f"Could not infer {kind} MIME type; pass mime_type=… explicitly."
        )
    return data, url, mime


def image(src: Any, *, mime_type: Optional[str] = None, detail: Optional[str] = None,
          fetch: bool = False) -> ImagePart:
    """Build an :class:`ImagePart` from a path, bytes, file-like, data URI, or URL."""
    data, url, mime = _resolve(src, mime_type=mime_type, fetch=fetch, kind="image")
    return ImagePart(data=data, url=url, mime_type=mime, detail=detail)


def document(src: Any, *, mime_type: Optional[str] = None, filename: Optional[str] = None,
             fetch: bool = False) -> DocumentPart:
    """Build a :class:`DocumentPart` (e.g. a PDF) from a path, bytes, file-like, or URL."""
    data, url, mime = _resolve(src, mime_type=mime_type, fetch=fetch, kind="document")
    if filename is None and not isinstance(src, (bytes, bytearray, memoryview)):
        try:
            filename = os.path.basename(os.fspath(src) if isinstance(src, os.PathLike) else str(src)) or None
        except Exception:
            filename = None
    return DocumentPart(data=data, url=url, mime_type=mime, filename=filename)


def audio(src: Any, *, mime_type: Optional[str] = None, fetch: bool = False) -> AudioPart:
    """Build an :class:`AudioPart` from a path, bytes, file-like, data URI, or URL."""
    data, url, mime = _resolve(src, mime_type=mime_type, fetch=fetch, kind="audio")
    return AudioPart(data=data, url=url, mime_type=mime)


# ---------------------------------------------------------------------------
# Serialization helpers (used by providers)
# ---------------------------------------------------------------------------


def to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def to_data_uri(mime_type: Optional[str], data: bytes) -> str:
    return f"data:{mime_type or 'application/octet-stream'};base64,{to_base64(data)}"


_AUDIO_FORMAT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
}


def audio_format(mime_type: Optional[str]) -> str:
    if mime_type and mime_type in _AUDIO_FORMAT:
        return _AUDIO_FORMAT[mime_type]
    if mime_type and "/" in mime_type:
        return mime_type.split("/", 1)[1]
    return "wav"


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------

# (part type, capability attribute, human label)
_MODALITY_RULES = (
    (ImagePart, "vision", "image input"),
    (DocumentPart, "documents", "document input"),
    (AudioPart, "audio_in", "audio input"),
)


def guard_modalities(messages: Sequence[Any], caps: Any, provider: str) -> None:
    """Raise :class:`UnsupportedModalityError` if a message carries media the
    provider hasn't truthfully declared support for."""
    for m in messages:
        for p in getattr(m, "parts", ()) or ():
            for part_type, attr, label in _MODALITY_RULES:
                if isinstance(p, part_type) and not getattr(caps, attr, False):
                    raise UnsupportedModalityError(
                        f"provider '{provider}' does not support {label}"
                    )


# ---------------------------------------------------------------------------
# Media elision (keeps inspect()/CallRecord readable; never alters the wire)
# ---------------------------------------------------------------------------

_B64_RE = re.compile(r"^[A-Za-z0-9+/=\r\n]+$")
_ELIDE_THRESHOLD = 256


def _elide_str(s: str) -> str:
    if s.startswith("data:") and ";base64," in s:
        head, b64 = s.split(";base64,", 1)
        return f"{head};base64,<{len(b64)} base64 chars elided>"
    if len(s) >= _ELIDE_THRESHOLD and _B64_RE.match(s):
        return f"<{len(s)} base64 chars elided>"
    return s


def elide_media(obj: Any) -> Any:
    """Return a copy of ``obj`` with large base64 media replaced by placeholders.

    For display and serialization only — the request SlimX actually sends keeps
    the real bytes.
    """
    if isinstance(obj, dict):
        return {k: elide_media(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [elide_media(v) for v in obj]
    if isinstance(obj, str):
        return _elide_str(obj)
    return obj
