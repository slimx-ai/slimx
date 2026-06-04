# slimx/providers/ollama.py

import json
import os

import httpx
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..errors import ProviderError
from ..messages import Message
from ..tooling import ToolSpec
from ..types import InspectedRequest, Result, StreamEvent, ToolCall, Usage
from ..utils.ndjson import iter_ndjson
from .base import Provider, ProviderCapabilities


class OllamaProvider(Provider):
    name = "ollama"
    capabilities = ProviderCapabilities(tools=True, structured_output=True, streaming=True)

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    def list_models(self, *, timeout: Optional[float] = None) -> list:
        url = f"{self.base_url}/api/tags"
        with httpx.Client(timeout=_timeout(timeout)) as client:
            response = client.get(url)
        if response.status_code >= 400:
            raise ProviderError(f"Ollama error {response.status_code}: {_read_response_text(response)}")
        data = response.json()
        return [m.get("name") for m in (data.get("models") or []) if m.get("name")]

    def build_request(self, req, *, tools: Sequence[ToolSpec] = (), stream: bool = False):
        return InspectedRequest(
            provider=self.name,
            method="POST",
            url=f"{self.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            payload=_payload(req, stream=stream, tools=tools),
        )

    def chat(self, req, *, tools: Sequence[ToolSpec] = (), timeout=None):
        payload = _payload(req, stream=True, tools=tools)
        url = f"{self.base_url}/api/chat"
        text_parts: List[str] = []
        raw_tool_calls: List[Dict[str, Any]] = []
        data: Dict[str, Any] = {}

        try:
            with httpx.Client(timeout=_timeout(timeout)) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = _read_response_text(response)
                        raise ProviderError(f"Ollama error {response.status_code}: {body}")

                    for obj in iter_ndjson(response.iter_bytes()):
                        data = obj
                        message = obj.get("message") or {}
                        chunk = message.get("content") or ""
                        if chunk:
                            text_parts.append(chunk)
                        raw_tool_calls.extend(message.get("tool_calls") or [])

                        if obj.get("done") is True:
                            break

        except httpx.TimeoutException as e:
            raise ProviderError(_timeout_message(req.model, url, streaming=False)) from e

        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
        return Result(
            text="".join(text_parts),
            raw=data,
            usage=usage,
            tool_calls=_parse_tool_calls(raw_tool_calls),
        )

    def stream(
        self,
        req,
        *,
        tools: Sequence[ToolSpec] = (),
        timeout=None,
    ) -> Iterable[StreamEvent]:
        payload = _payload(req, stream=True, tools=tools)
        url = f"{self.base_url}/api/chat"

        try:
            with httpx.Client(timeout=_timeout(timeout)) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = _read_response_text(response)
                        raise ProviderError(f"Ollama error {response.status_code}: {body}")

                    for obj in iter_ndjson(response.iter_bytes()):
                        message = obj.get("message") or {}
                        chunk = message.get("content") or ""
                        if chunk:
                            yield StreamEvent(type="text_delta", text=chunk, raw=obj)
                        for call in _parse_tool_calls(message.get("tool_calls") or []):
                            yield StreamEvent.tool(call, raw=obj)

                        if obj.get("done") is True:
                            break

        except httpx.TimeoutException as e:
            raise ProviderError(_timeout_message(req.model, url, streaming=True)) from e

        yield StreamEvent(type="done")


# --------------------------------------------------------------------------
# Shared request/response mapping (also used by the async provider).
# --------------------------------------------------------------------------

def _read_response_text(response: httpx.Response) -> str:
    try:
        return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _timeout(timeout: Optional[float]) -> httpx.Timeout:
    if timeout is None:
        return httpx.Timeout(None, connect=10.0)
    return httpx.Timeout(timeout, connect=min(float(timeout), 10.0))


def _timeout_message(model: str, url: str, *, streaming: bool) -> str:
    kind = "stream" if streaming else "request"
    return (
        f"Ollama {kind} timed out for model {model!r} at {url}. "
        "Try warming the model with `ollama run`, using a smaller model, "
        "reducing max_tokens/context, or increasing the SlimX timeout."
    )


def _tools_payload(tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _messages_to_ollama(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    """Convert SlimX messages to Ollama's /api/chat shape.

    Ollama tool calls use ``{"function": {"name", "arguments": <object>}}`` and tool
    results are ``{"role": "tool", "content": ..., "tool_name": ...}`` — different from
    the OpenAI-style dicts the SlimX auto-tool-loop stores, so they are translated here.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        if m.role in ("system", "user"):
            out.append({"role": m.role, "content": m.content})
        elif m.role == "assistant":
            msg: Dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            tool_calls = []
            for tc in m.tool_calls or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name") or ""
                raw_args = fn.get("arguments")
                if raw_args is None:
                    raw_args = tc.get("arguments") or {}
                args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append({"function": {"name": name, "arguments": args}})
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif m.role == "tool":
            out.append(
                {"role": "tool", "content": m.content, "tool_name": m.tool_name or m.tool_call_id or ""}
            )
        elif m.content:
            out.append({"role": "user", "content": m.content})
    return out


def _parse_tool_calls(raw: Sequence[Dict[str, Any]]) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for tc in raw or []:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _safe_json_loads(args)
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(id=str(tc.get("id") or name), name=str(name), arguments=args))
    return calls


def _payload(req, *, stream: bool, tools: Sequence[ToolSpec] = ()) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": _messages_to_ollama(req.messages),
        "stream": stream,
    }

    if tools:
        payload["tools"] = _tools_payload(tools)

    # Map SlimX JSON mode to Ollama's native structured-output `format`.
    if req.response_format == "json_object":
        payload["format"] = "json"

    options: Dict[str, Any] = {}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.max_tokens is not None:
        options["num_predict"] = req.max_tokens

    if req.extra:
        extra_options = req.extra.get("options")
        if isinstance(extra_options, dict):
            options.update(extra_options)
        # Explicit `format`/`keep_alive` in extra win over the JSON-mode default.
        for key in ("format", "keep_alive"):
            if key in req.extra:
                payload[key] = req.extra[key]

    if options:
        payload["options"] = options

    return payload
