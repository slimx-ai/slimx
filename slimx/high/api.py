from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from ..messages import Message
from ..types import Result, StreamEvent
from ..errors import SchemaError, UnsupportedModalityError
from ..schema import parse_json, schema_for, coerce_dataclass
from ..tooling import ToolSpec
from ..providers import get_provider
from ..low import Client, ChatRequest, ImageRequest


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


def _image_request(model: str, prompt: str, overrides: Dict[str, Any]) -> ImageRequest:
    """Build an ImageRequest, mapping `n`/`size` and routing the rest to `extra`."""
    n = overrides.pop("n", 1)
    size = overrides.pop("size", None)
    return ImageRequest(model=model, prompt=prompt, n=n, size=size, extra=overrides or None)


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

    def inspect(self, prompt: str, *, stream: bool = False, **overrides: Any):
        """Dry-run: return the exact request SlimX would send, without sending it."""
        req = ChatRequest(
            model=self._model,
            messages=[_user_message(prompt, overrides)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        return self._client.inspect(req, tools=self._tools, stream=stream)

    def __call__(self, prompt: str, **overrides: Any) -> Result:
        req = ChatRequest(
            model=self._model,
            messages=[_user_message(prompt, overrides)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        return self._client.chat(req, tools=self._tools, tool_runtime=self._tool_runtime)

    def stream(self, prompt: str, **overrides: Any) -> Iterable[StreamEvent]:
        req = ChatRequest(
            model=self._model,
            messages=[_user_message(prompt, overrides)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        return self._client.stream(req, tools=self._tools)

    def json(self, prompt: str, *, schema: Any, repair: int = 0, **overrides: Any) -> Result:
        schema_dict, schema_type = _json_schema_parts(schema)
        messages = [Message.system(_json_system_prompt(schema_dict)), _user_message(prompt, overrides)]
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

    def inspect(self, prompt: str, *, stream: bool = False, **overrides: Any):
        """Dry-run: return the exact request SlimX would send, without sending it."""
        req = ChatRequest(
            model=self._model,
            messages=[_user_message(prompt, overrides)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        return self._client.inspect(req, tools=self._tools, stream=stream)

    async def __call__(self, prompt: str, **overrides: Any) -> Result:
        req = ChatRequest(
            model=self._model,
            messages=[_user_message(prompt, overrides)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        return await self._client.achat(req, tools=self._tools, tool_runtime=self._tool_runtime)

    async def astream(self, prompt: str, **overrides: Any):
        req = ChatRequest(
            model=self._model,
            messages=[_user_message(prompt, overrides)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        async for ev in self._client.astream(req, tools=self._tools):
            yield ev

    async def json(self, prompt: str, *, schema: Any, repair: int = 0, **overrides: Any) -> Result:
        schema_dict, schema_type = _json_schema_parts(schema)
        messages = [Message.system(_json_system_prompt(schema_dict)), _user_message(prompt, overrides)]
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


def llm(model: str, **kwargs: Any) -> Model:
    return Model(model, **kwargs)


def allm(model: str, **kwargs: Any) -> AsyncModel:
    return AsyncModel(model, **kwargs)
