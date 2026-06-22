from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

from ..messages import Message
from ..types import ImageGenerationOptions, ImageInput, Result, StreamEvent
from ..errors import SchemaError, UnsupportedModalityError
from ..schema import parse_json, schema_for, coerce_dataclass
from ..tooling import ToolSpec
from ..providers import get_provider
from ..low import Client, ChatRequest, ImageEditRequest, ImageRequest


def _parse_model(model: str):
    if ":" in model:
        p, m = model.split(":", 1)
        return p.strip(), m.strip()
    return "openai", model.strip()


# ---- structured output helpers (shared by Model.json / AsyncModel.json) ----

def _json_schema_parts(schema: Any):
    if isinstance(schema, dict):
        return schema, None
    return schema_for(schema), schema


def _json_system_prompt(schema_dict: Any) -> str:
    return "Return ONLY valid JSON (no markdown). Match this JSON Schema exactly: " + str(schema_dict)


def _parse_into_schema(text: str, schema_type: Any) -> Any:
    obj = parse_json(text)
    return coerce_dataclass(schema_type, obj) if schema_type else obj


_MEDIA_KEYS = ("images", "documents", "audio", "parts")


def _user_message(prompt: str, overrides: Dict[str, Any]) -> Message:
    """Build a user message, pulling any multimodal media out of `overrides`.

    Lets `m(prompt, images=[image(...)])` work across __call__/stream/json/inspect
    without each method growing positional media params.
    """
    media = {k: overrides.pop(k) for k in _MEDIA_KEYS if k in overrides}
    return Message.user(prompt, **media)


# A high-level call accepts either a single prompt string or a full message list.
PromptInput = Union[str, Sequence[Message]]


def _messages_from(prompt: PromptInput, overrides: Dict[str, Any]) -> list:
    """Normalize a prompt into a message list.

    A `str` becomes a single user message (with any media kwargs attached); a
    `Sequence[Message]` is used as-is for multi-turn conversations, so callers no
    longer have to drop to the low-level `ChatRequest` API for history.
    """
    if isinstance(prompt, str):
        return [_user_message(prompt, overrides)]
    # Explicit message list: media kwargs don't apply — discard so they don't
    # leak into `overrides` and get mistaken for temperature/max_tokens.
    for k in _MEDIA_KEYS:
        overrides.pop(k, None)
    return list(prompt)


def _image_request(model: str, prompt: str, overrides: Dict[str, Any]) -> ImageRequest:
    """Build an ImageRequest, mapping `n`/`size` and routing the rest to `extra`."""
    n = overrides.pop("n", 1)
    size = overrides.pop("size", None)
    return ImageRequest(model=model, prompt=prompt, n=n, size=size, extra=overrides or None)


def _chat_request(
    model: str,
    prompt: "PromptInput",
    overrides: Dict[str, Any],
    *,
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> ChatRequest:
    """Build a ChatRequest, threading the hosted-image-tool fields from overrides.

    ``image_generation`` (an ``ImageGenerationOptions``) routes OpenAI-shaped
    providers to the Responses API and exposes the hosted image tool;
    ``previous_response_id`` continues an image conversation; ``tool_choice``
    forces a tool. Pulling them here lets ``__call__`` / ``stream`` / ``inspect``
    all gain image generation without each growing positional parameters.
    """
    return ChatRequest(
        model=model,
        messages=_messages_from(prompt, overrides),
        temperature=overrides.get("temperature", temperature),
        max_tokens=overrides.get("max_tokens", max_tokens),
        image_generation=overrides.get("image_generation"),
        previous_response_id=overrides.get("previous_response_id"),
        tool_choice=overrides.get("tool_choice"),
    )


def _normalize_image_inputs(images: Any) -> list:
    """Coerce edit_image() source(s) into ImageInput list (bytes / ImagePart / dict)."""
    from ..content import ImagePart

    if images is None:
        return []
    items = images if isinstance(images, (list, tuple)) else [images]
    out: list = []
    for item in items:
        if isinstance(item, ImageInput):
            out.append(item)
        elif isinstance(item, ImagePart):
            out.append(ImageInput(data=item.data, mime_type=item.mime_type, url=item.url))
        elif isinstance(item, (bytes, bytearray)):
            out.append(ImageInput(data=bytes(item)))
        elif isinstance(item, dict):
            out.append(ImageInput(**item))
        else:
            raise TypeError(f"Unsupported image source for edit_image(): {type(item)!r}")
    return out


def _image_edit_request(model: str, instruction: str, overrides: Dict[str, Any]) -> ImageEditRequest:
    """Build an ImageEditRequest from edit_image(image=..., instruction, **opts)."""
    images = _normalize_image_inputs(overrides.pop("image", None) or overrides.pop("images", None))
    n = overrides.pop("n", 1)
    size = overrides.pop("size", None)
    options = overrides.pop("options", None) or overrides.pop("image_generation", None)
    if options is None and overrides:
        # Treat leftover tool knobs (quality/output_format/background/...) as options.
        known = {"quality", "output_format", "background", "output_compression", "partial_images"}
        opt_kwargs = {k: overrides.pop(k) for k in list(overrides) if k in known}
        if size is not None:
            opt_kwargs.setdefault("size", size)
        options = ImageGenerationOptions(action="edit", **opt_kwargs) if opt_kwargs else None
    previous_response_id = overrides.pop("previous_response_id", None)
    return ImageEditRequest(
        model=model,
        instruction=instruction,
        images=images,
        n=n,
        size=size,
        options=options,
        previous_response_id=previous_response_id,
        extra=overrides or None,
    )


def _repair_turn(bad_text: str, error: Exception):
    return [
        Message.assistant(bad_text),
        Message.user(
            f"That response was not valid for the schema ({error}). "
            "Return ONLY corrected JSON that matches the schema exactly, with no prose."
        ),
    ]


class Model:
    def __init__(
        self,
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[Sequence[ToolSpec]] = None,
        tool_runtime: str = "none",
        timeout: Optional[float] = None,
        retries: int = 2,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        hooks: Optional[Mapping[str, Any]] = None,
    ):
        provider_name, model_name = _parse_model(model)
        provider = get_provider(provider_name, async_mode=False, **(provider_kwargs or {}))
        self._client = Client(provider, timeout=timeout, retries=retries, hooks=hooks)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tools = list(tools or [])
        self._tool_runtime = tool_runtime

    @property
    def capabilities(self):
        """The selected provider's declared capabilities (`ProviderCapabilities`)."""
        return self._client.provider.capabilities

    def inspect(self, prompt: PromptInput, *, stream: bool = False, **overrides: Any):
        """Dry-run: return the exact request SlimX would send, without sending it."""
        req = _chat_request(
            self._model, prompt, overrides,
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        return self._client.inspect(req, tools=self._tools, stream=stream)

    def __call__(self, prompt: PromptInput, **overrides: Any) -> Result:
        req = _chat_request(
            self._model, prompt, overrides,
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        return self._client.chat(req, tools=self._tools, tool_runtime=self._tool_runtime)

    def stream(self, prompt: PromptInput, **overrides: Any) -> Iterable[StreamEvent]:
        req = _chat_request(
            self._model, prompt, overrides,
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        return self._client.stream(req, tools=self._tools)

    def json(self, prompt: PromptInput, *, schema: Any, repair: int = 0, **overrides: Any) -> Result:
        schema_dict, schema_type = _json_schema_parts(schema)
        messages = [Message.system(_json_system_prompt(schema_dict))] + _messages_from(prompt, overrides)
        for attempt in range(repair + 1):
            req = ChatRequest(
                model=self._model,
                messages=list(messages),
                temperature=overrides.get("temperature", self._temperature),
                max_tokens=overrides.get("max_tokens", self._max_tokens),
                response_format="json_object",
            )
            res = self._client.chat(req, tools=self._tools, tool_runtime=self._tool_runtime)
            try:
                res.data = _parse_into_schema(res.text, schema_type)
                return res
            except Exception as e:
                if attempt >= repair:
                    raise
                messages = messages + _repair_turn(res.text, e)
        raise SchemaError("unreachable")  # pragma: no cover

    def generate_image(self, prompt: str, **overrides: Any) -> Result:
        """Generate image(s) from a text prompt. Images land on `Result.images`."""
        if not self.capabilities.image_out:
            raise UnsupportedModalityError(
                f"provider '{self._client.provider_name}' does not support image generation"
            )
        return self._client.generate_image(_image_request(self._model, prompt, overrides))

    def inspect_image(self, prompt: str, **overrides: Any):
        """Dry-run: the exact image-generation request, without sending it."""
        return self._client.inspect_image(_image_request(self._model, prompt, overrides))

    def edit_image(self, image: Any, instruction: str, **overrides: Any) -> Result:
        """Edit/refine ``image`` with a text ``instruction``; edited image(s) land
        on ``Result.images``.

        ``image`` accepts raw ``bytes``, an ``ImagePart``, an ``ImageInput``, a
        dict, or a list of those (multiple source images). Editing operates on the
        supplied bytes, so it stays durable — it does not require provider-side
        conversation state.
        """
        if not self.capabilities.image_edit:
            raise UnsupportedModalityError(
                f"provider '{self._client.provider_name}' does not support image editing"
            )
        overrides["image"] = image
        return self._client.edit_image(_image_edit_request(self._model, instruction, overrides))


class AsyncModel:
    def __init__(
        self,
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[Sequence[ToolSpec]] = None,
        tool_runtime: str = "none",
        timeout: Optional[float] = None,
        retries: int = 2,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        hooks: Optional[Mapping[str, Any]] = None,
    ):
        provider_name, model_name = _parse_model(model)
        provider = get_provider(provider_name, async_mode=True, **(provider_kwargs or {}))
        self._client = Client(provider, timeout=timeout, retries=retries, hooks=hooks)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tools = list(tools or [])
        self._tool_runtime = tool_runtime

    @property
    def capabilities(self):
        """The selected provider's declared capabilities (`ProviderCapabilities`)."""
        return self._client.provider.capabilities

    def inspect(self, prompt: PromptInput, *, stream: bool = False, **overrides: Any):
        """Dry-run: return the exact request SlimX would send, without sending it."""
        req = _chat_request(
            self._model, prompt, overrides,
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        return self._client.inspect(req, tools=self._tools, stream=stream)

    async def __call__(self, prompt: PromptInput, **overrides: Any) -> Result:
        req = _chat_request(
            self._model, prompt, overrides,
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        return await self._client.achat(req, tools=self._tools, tool_runtime=self._tool_runtime)

    async def astream(self, prompt: PromptInput, **overrides: Any):
        req = _chat_request(
            self._model, prompt, overrides,
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        async for ev in self._client.astream(req, tools=self._tools):
            yield ev

    async def json(self, prompt: PromptInput, *, schema: Any, repair: int = 0, **overrides: Any) -> Result:
        schema_dict, schema_type = _json_schema_parts(schema)
        messages = [Message.system(_json_system_prompt(schema_dict))] + _messages_from(prompt, overrides)
        for attempt in range(repair + 1):
            req = ChatRequest(
                model=self._model,
                messages=list(messages),
                temperature=overrides.get("temperature", self._temperature),
                max_tokens=overrides.get("max_tokens", self._max_tokens),
                response_format="json_object",
            )
            res = await self._client.achat(req, tools=self._tools, tool_runtime=self._tool_runtime)
            try:
                res.data = _parse_into_schema(res.text, schema_type)
                return res
            except Exception as e:
                if attempt >= repair:
                    raise
                messages = messages + _repair_turn(res.text, e)
        raise SchemaError("unreachable")  # pragma: no cover

    async def generate_image(self, prompt: str, **overrides: Any) -> Result:
        """Generate image(s) from a text prompt. Images land on `Result.images`."""
        if not self.capabilities.image_out:
            raise UnsupportedModalityError(
                f"provider '{self._client.provider_name}' does not support image generation"
            )
        return await self._client.agenerate_image(_image_request(self._model, prompt, overrides))

    def inspect_image(self, prompt: str, **overrides: Any):
        """Dry-run: the exact image-generation request, without sending it."""
        return self._client.inspect_image(_image_request(self._model, prompt, overrides))

    async def edit_image(self, image: Any, instruction: str, **overrides: Any) -> Result:
        """Async sibling of :meth:`Model.edit_image`; edited image(s) on ``Result.images``."""
        if not self.capabilities.image_edit:
            raise UnsupportedModalityError(
                f"provider '{self._client.provider_name}' does not support image editing"
            )
        overrides["image"] = image
        return await self._client.aedit_image(_image_edit_request(self._model, instruction, overrides))


def llm(model: str, **kwargs: Any) -> Model:
    return Model(model, **kwargs)


def allm(model: str, **kwargs: Any) -> AsyncModel:
    return AsyncModel(model, **kwargs)
