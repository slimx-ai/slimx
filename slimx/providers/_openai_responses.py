"""OpenAI **Responses API** support (``POST /v1/responses``).

SlimX speaks Chat Completions by default (see ``_openai_shape``). The Responses
API is a *different* request/response shape, and it is the only OpenAI surface
that exposes the hosted ``image_generation`` tool — a text model (e.g. GPT-5.5)
that can produce real image bytes inline, optionally editing a supplied image.

OpenAI-shaped providers route here automatically whenever a call carries an
``image_generation`` config (``ChatRequest.image_generation`` or an
``ImageEditRequest``); plain text/function-tool chat stays on Chat Completions.

What this module owns, in one place so sync and async never drift:
- building the ``/responses`` request body from SlimX messages/options,
- parsing ``image_generation_call`` + ``output_text`` outputs into a ``Result``
  (base64 decoded exactly once into ``GeneratedImage`` bytes), and
- translating the Responses SSE event stream into normalized ``StreamEvent``s
  (text deltas, partial-image previews, final images, done/error).

Reference request:
    {"model": "gpt-5.5",
     "input": [{"role": "user", "content": [{"type": "input_text", ...},
                                            {"type": "input_image", ...}]}],
     "tools": [{"type": "image_generation", "size": "1024x1024", ...}],
     "tool_choice": {"type": "image_generation"},      # when forced
     "previous_response_id": "resp_..."}                # conversational revision

Reference output item:
    {"type": "image_generation_call", "id": "ig_...", "status": "completed",
     "revised_prompt": "...", "result": "<base64>"}
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Sequence

from ..content import ImagePart, TextPart, image_dimensions, to_data_uri
from ..content import _sniff_mime  # internal: MIME from magic bytes (never trust declared)
from ..low.types import ChatRequest, ImageEditRequest
from ..tooling import ToolSpec
from ..types import GeneratedImage, ImageGenerationOptions, Result, StreamEvent, Usage


def operation_for_options(options: Any) -> str:
    """Map an ImageGenerationOptions action onto a GeneratedImage operation."""
    return "edit" if options is not None and getattr(options, "action", None) == "edit" else "generate"


def responses_input_from_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Convert SlimX messages into the Responses ``input`` array.

    Text becomes ``input_text`` (``output_text`` for assistant turns) and images
    become ``input_image`` with a ``data:`` URI (or passthrough URL). Other part
    types are not part of the image path and are skipped.
    """
    items: List[Dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "role", "user")
        text_type = "output_text" if role == "assistant" else "input_text"
        content: List[Dict[str, Any]] = []
        for part in m.content_parts():
            if isinstance(part, TextPart):
                if part.text:
                    content.append({"type": text_type, "text": part.text})
            elif isinstance(part, ImagePart):
                url = part.url or to_data_uri(part.mime_type, part.data or b"")
                img: Dict[str, Any] = {"type": "input_image", "image_url": url}
                if part.detail:
                    img["detail"] = part.detail
                content.append(img)
        items.append({"role": role, "content": content})
    return items


def _function_tools(tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
    # Responses uses a flat function-tool shape (no nested "function" wrapper).
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]


def build_responses_payload(
    req: ChatRequest,
    tools: Sequence[ToolSpec] = (),
    *,
    stream: bool = False,
    caps: Any = None,
    provider: str = "openai",
) -> Dict[str, Any]:
    """Build the ``/responses`` body for a hosted-image-tool chat request."""
    if caps is not None:
        from ..content import guard_modalities

        guard_modalities(req.messages, caps, provider)

    payload: Dict[str, Any] = {
        "model": req.model,
        "input": responses_input_from_messages(req.messages),
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_output_tokens"] = req.max_tokens

    tool_objs: List[Dict[str, Any]] = _function_tools(tools)
    options = req.image_generation
    if options is not None:
        tool_objs.append(options.to_tool_dict())
    if tool_objs:
        payload["tools"] = tool_objs

    if options is not None and options.force:
        payload["tool_choice"] = {"type": "image_generation"}
    elif req.tool_choice is not None:
        payload["tool_choice"] = req.tool_choice

    if req.previous_response_id:
        payload["previous_response_id"] = req.previous_response_id
    if stream:
        payload["stream"] = True
    if req.extra:
        payload.update(req.extra)
    return payload


def build_edit_payload(req: ImageEditRequest, *, stream: bool = False) -> Dict[str, Any]:
    """Build the ``/responses`` body for an image-edit request.

    The instruction plus every source image ride in one user message; the hosted
    image tool is always forced (``tool_choice``) so the model edits rather than
    just describing. Inline bytes are the durable path; ``file_id``/``url`` are
    used when supplied.
    """
    content: List[Dict[str, Any]] = []
    if req.instruction:
        content.append({"type": "input_text", "text": req.instruction})
    for img in req.images:
        if img.url:
            content.append({"type": "input_image", "image_url": img.url})
        elif img.file_id:
            content.append({"type": "input_image", "file_id": img.file_id})
        elif img.data is not None:
            content.append(
                {"type": "input_image", "image_url": to_data_uri(img.mime_type, img.data)}
            )

    options = req.options or ImageGenerationOptions(action="edit")
    tool = options.to_tool_dict()
    if req.size and "size" not in tool:
        tool["size"] = req.size

    payload: Dict[str, Any] = {
        "model": req.model,
        "input": [{"role": "user", "content": content}],
        "tools": [tool],
        "tool_choice": {"type": "image_generation"},
    }
    if req.previous_response_id:
        payload["previous_response_id"] = req.previous_response_id
    if stream:
        payload["stream"] = True
    if req.extra:
        payload.update(req.extra)
    return payload


def _decode_b64(value: Any) -> Optional[bytes]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return base64.b64decode(value)
    except Exception:
        return None


def _mime_for(blob: bytes, item: Dict[str, Any]) -> str:
    """MIME from the actual bytes first, then the declared output_format."""
    sniffed = _sniff_mime(blob)
    if sniffed:
        return sniffed
    fmt = item.get("output_format")
    if isinstance(fmt, str) and fmt:
        return f"image/{fmt.lower()}"
    return "image/png"


def _image_from_call(
    item: Dict[str, Any],
    *,
    provider: str,
    model: Optional[str],
    response_id: Optional[str],
    output_index: int,
    operation: str,
) -> Optional[GeneratedImage]:
    """Normalize one ``image_generation_call`` item; ``None`` when it has no image
    (a refusal or a still-pending call)."""
    blob = _decode_b64(item.get("result"))
    if blob is None:
        return None
    mime = _mime_for(blob, item)
    width, height = image_dimensions(blob)
    metadata: Dict[str, Any] = {}
    if item.get("status"):
        metadata["status"] = item["status"]
    return GeneratedImage(
        mime_type=mime,
        data=blob,
        width=width,
        height=height,
        provider=provider,
        model=model,
        operation=operation,
        provider_response_id=response_id,
        provider_call_id=item.get("id"),
        revised_prompt=item.get("revised_prompt"),
        output_index=output_index,
        metadata=metadata,
    )


def _responses_usage(usage: Dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def parse_responses_response(
    data: Dict[str, Any],
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    operation: str = "generate",
) -> Result:
    """Parse a non-streaming ``/responses`` body into a ``Result``.

    Collects ``output_text`` from message items as ``Result.text`` and every
    ``image_generation_call`` with a base64 ``result`` as a ``GeneratedImage``.
    A response with text but no image (refusal) yields empty ``images``.
    """
    response_id = data.get("id")
    text_chunks: List[str] = []
    images: List[GeneratedImage] = []
    output_index = 0
    for item in data.get("output") or []:
        itype = item.get("type")
        if itype == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    text_chunks.append(c["text"])
                elif c.get("type") == "refusal" and isinstance(c.get("refusal"), str):
                    text_chunks.append(c["refusal"])
        elif itype == "image_generation_call":
            img = _image_from_call(
                item,
                provider=provider,
                model=model,
                response_id=response_id,
                output_index=output_index,
                operation=operation,
            )
            if img is not None:
                images.append(img)
                output_index += 1
    text = "".join(text_chunks)
    if not text and isinstance(data.get("output_text"), str):
        text = data["output_text"]
    return Result(
        text=text, raw=data, usage=_responses_usage(data.get("usage") or {}), images=images
    )


class ResponsesStreamTranslator:
    """Translate decoded Responses SSE events into normalized ``StreamEvent``s.

    Text deltas pass through verbatim. Partial-image events become transient
    ``image_partial`` previews (base64, never persisted as a final asset). Final
    images are read from the terminal ``response.completed`` payload (always
    complete and authoritative), emitted as ``image_completed`` then ``done``.
    """

    def __init__(self, *, provider: str = "openai", model: Optional[str] = None,
                 operation: str = "generate") -> None:
        self.provider = provider
        self.model = model
        self.operation = operation
        self._completed = False

    def feed(self, obj: Dict[str, Any]) -> List[StreamEvent]:
        etype = obj.get("type") or ""
        if etype == "response.output_text.delta":
            delta = obj.get("delta")
            if isinstance(delta, str) and delta:
                return [StreamEvent.text_delta(delta, raw=obj)]
            return []
        if etype == "response.image_generation_call.in_progress":
            return [StreamEvent.image_started(index=obj.get("output_index"), raw=obj)]
        if etype == "response.image_generation_call.partial_image":
            b64 = obj.get("partial_image_b64")
            if isinstance(b64, str) and b64:
                return [
                    StreamEvent.image_partial(
                        b64, index=obj.get("partial_image_index"), raw=obj
                    )
                ]
            return []
        if etype in ("response.completed", "response.incomplete"):
            return self._finalize(obj.get("response") or {}, raw=obj)
        if etype in ("response.failed", "error"):
            return [StreamEvent.err(_stream_error_message(obj), raw=obj)]
        return []

    def _finalize(self, response: Dict[str, Any], *, raw: Any) -> List[StreamEvent]:
        result = parse_responses_response(
            response, provider=self.provider, model=self.model, operation=self.operation
        )
        events: List[StreamEvent] = []
        for i, img in enumerate(result.images):
            idx = img.output_index if img.output_index is not None else i
            events.append(StreamEvent.image_completed(img, index=idx, raw=None))
        events.append(StreamEvent.done(raw=raw))
        self._completed = True
        return events

    def finish(self) -> List[StreamEvent]:
        """Terminal ``done`` if the provider stream ended without one."""
        return [] if self._completed else [StreamEvent.done()]


def _stream_error_message(obj: Dict[str, Any]) -> str:
    err = obj.get("error")
    if isinstance(err, dict):
        return err.get("message") or "Responses stream error"
    response = obj.get("response")
    if isinstance(response, dict) and isinstance(response.get("error"), dict):
        return response["error"].get("message") or "Responses stream error"
    return obj.get("message") or "Responses stream error"
