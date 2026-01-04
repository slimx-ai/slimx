from typing import Any, Dict, Iterable, Optional, Sequence

from ..messages import Message
from ..types import Result, StreamEvent
from ..schema import parse_json, schema_for, coerce_dataclass
from ..tooling import ToolSpec
from ..providers import get_provider
from ..low import Client, ChatRequest


def _parse_model(model: str):
    if ":" in model:
        p, m = model.split(":", 1)
        return p.strip(), m.strip()
    return "openai", model.strip()


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
    ):
        provider_name, model_name = _parse_model(model)
        provider = get_provider(provider_name, async_mode=False, **(provider_kwargs or {}))
        self._client = Client(provider, timeout=timeout, retries=retries)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tools = list(tools or [])
        self._tool_runtime = tool_runtime

    def __call__(self, prompt: str, **overrides: Any) -> Result:
        req = ChatRequest(
            model=self._model,
            messages=[Message.user(prompt)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        return self._client.chat(req, tools=self._tools, tool_runtime=self._tool_runtime)

    def stream(self, prompt: str, **overrides: Any) -> Iterable[StreamEvent]:
        req = ChatRequest(
            model=self._model,
            messages=[Message.user(prompt)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        return self._client.stream(req, tools=self._tools)

    def json(self, prompt: str, *, schema: Any, **overrides: Any) -> Result:
        if isinstance(schema, dict):
            schema_dict = schema
            schema_type = None
        else:
            schema_type = schema
            schema_dict = schema_for(schema)

        sys = "Return ONLY valid JSON (no markdown). Match this JSON Schema exactly: " + str(schema_dict)
        req = ChatRequest(
            model=self._model,
            messages=[Message.system(sys), Message.user(prompt)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
            response_format="json_object",
        )
        res = self._client.chat(req, tools=self._tools, tool_runtime=self._tool_runtime)
        obj = parse_json(res.text)
        res.data = coerce_dataclass(schema_type, obj) if schema_type else obj
        return res


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
    ):
        provider_name, model_name = _parse_model(model)
        provider = get_provider(provider_name, async_mode=True, **(provider_kwargs or {}))
        self._client = Client(provider, timeout=timeout, retries=retries)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tools = list(tools or [])
        self._tool_runtime = tool_runtime

    async def __call__(self, prompt: str, **overrides: Any) -> Result:
        req = ChatRequest(
            model=self._model,
            messages=[Message.user(prompt)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        return await self._client.achat(req, tools=self._tools, tool_runtime=self._tool_runtime)

    async def astream(self, prompt: str, **overrides: Any):
        req = ChatRequest(
            model=self._model,
            messages=[Message.user(prompt)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        async for ev in self._client.astream(req, tools=self._tools):
            yield ev

    async def json(self, prompt: str, *, schema: Any, **overrides: Any) -> Result:
        if isinstance(schema, dict):
            schema_dict = schema
            schema_type = None
        else:
            schema_type = schema
            schema_dict = schema_for(schema)

        sys = "Return ONLY valid JSON (no markdown). Match this JSON Schema exactly: " + str(schema_dict)
        req = ChatRequest(
            model=self._model,
            messages=[Message.system(sys), Message.user(prompt)],
            temperature=overrides.get("temperature", self._temperature),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
            response_format="json_object",
        )
        res = await self._client.achat(req, tools=self._tools, tool_runtime=self._tool_runtime)
        obj = parse_json(res.text)
        res.data = coerce_dataclass(schema_type, obj) if schema_type else obj
        return res


def llm(model: str, **kwargs: Any) -> Model:
    return Model(model, **kwargs)


def allm(model: str, **kwargs: Any) -> AsyncModel:
    return AsyncModel(model, **kwargs)
