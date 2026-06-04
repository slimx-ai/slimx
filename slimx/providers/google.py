# slimx/providers/google.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Sequence

import httpx

from ..errors import ProviderAuthError, ProviderError, ProviderRateLimitError
from ..low.types import ChatRequest
from ..messages import Message
from ..tooling import ToolSpec
from ..types import InspectedRequest, Result, StreamEvent, ToolCall, Usage, redact_headers
from ..utils.sse import iter_sse_data
from .base import Provider, ProviderCapabilities


DEFAULT_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleProvider(Provider):
    name = "google"
    capabilities = ProviderCapabilities(
        tools=True,
        structured_output=True,
        streaming=True,
    )

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_GOOGLE_BASE_URL,
    ):
        if not api_key:
            raise ProviderAuthError("GOOGLE_API_KEY or GEMINI_API_KEY is not set")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def build_request(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec] = (),
        stream: bool = False,
    ) -> InspectedRequest:
        verb = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return InspectedRequest(
            provider=self.name,
            method="POST",
            url=f"{self.base_url}/{_model_path(req.model)}:{verb}",
            headers=redact_headers(self._headers()),
            payload=_payload(req, tools=tools),
        )

    def chat(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec] = (),
        timeout: Optional[float] = None,
    ) -> Result:
        payload = _payload(req, tools=tools)
        url = f"{self.base_url}/{_model_path(req.model)}:generateContent"

        with httpx.Client(timeout=timeout or 30.0) as client:
            response = client.post(url, headers=self._headers(), json=payload)

        _raise_for_status(response.status_code, response.text)
        return _parse_response(response.json())

    def stream(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec] = (),
        timeout: Optional[float] = None,
    ) -> Iterable[StreamEvent]:
        payload = _payload(req, tools=tools)
        url = f"{self.base_url}/{_model_path(req.model)}:streamGenerateContent?alt=sse"

        with httpx.Client(timeout=timeout or 30.0) as client:
            with client.stream("POST", url, headers=self._headers(), json=payload) as response:
                if response.status_code >= 400:
                    # Body must be read before access on a streamed response,
                    # otherwise httpx raises ResponseNotRead.
                    body = response.read().decode("utf-8", errors="replace")
                    _raise_for_status(response.status_code, body)

                for chunk in iter_sse_data(response.iter_bytes()):
                    if not chunk or chunk == "[DONE]":
                        continue

                    try:
                        data = json.loads(chunk)
                    except Exception:
                        continue

                    for text in _extract_text_parts(data):
                        yield StreamEvent.text_delta(text, raw=data)

                    for tool_call in _extract_tool_calls(data):
                        yield StreamEvent.tool(tool_call, raw=data)

        yield StreamEvent.done()


def _model_path(model: str) -> str:
    model = model.strip().lstrip("/")
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _payload(req: ChatRequest, *, tools: Sequence[ToolSpec] = ()) -> Dict[str, Any]:
    contents, system_instruction = _contents_from_messages(req.messages)

    payload: Dict[str, Any] = {
        "contents": contents,
    }

    if system_instruction:
        payload["systemInstruction"] = system_instruction

    # Preserve provider-specific escape hatches.
    extra = dict(req.extra or {})
    extra_generation_config = extra.pop("generationConfig", None)

    for key, value in extra.items():
        payload[key] = value

    generation_config = _generation_config(req, extra_generation_config)
    if generation_config:
        payload["generationConfig"] = generation_config

    if tools:
        payload["tools"] = _tools_payload(tools)

    return payload


def _contents_from_messages(messages: Sequence[Message]) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []

    # Used to map SlimX tool-result messages back to Gemini functionResponse parts.
    # SlimX Client stores assistant tool calls in OpenAI-style `Message.tool_calls`.
    tool_call_names_by_id: dict[str, str] = {}
    last_tool_call_name: Optional[str] = None

    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append({"text": message.content})
            continue

        if message.role == "user":
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": message.content}],
                }
            )
            continue

        if message.role == "assistant":
            parts: list[dict[str, Any]] = []

            if message.content:
                parts.append({"text": message.content})

            for tool_call in message.tool_calls or []:
                function_call = _function_call_part_from_slimx_tool_call(tool_call)
                if function_call:
                    fc = function_call["functionCall"]
                    call_id = str(fc.get("id") or fc.get("name") or "")
                    call_name = str(fc.get("name") or "")
                    if call_id and call_name:
                        tool_call_names_by_id[call_id] = call_name
                    if call_name:
                        last_tool_call_name = call_name
                    parts.append(function_call)

            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if message.role == "tool":
            response_obj = _safe_json_loads(message.content)
            call_id = message.tool_call_id or message.tool_name or ""
            name = (
                message.tool_name
                or tool_call_names_by_id.get(call_id)
                or last_tool_call_name
                or call_id
                or "tool"
            )

            function_response: dict[str, Any] = {
                "name": name,
                "response": {"result": response_obj},
            }

            if call_id:
                function_response["id"] = call_id

            contents.append(
                {
                    "role": "user",
                    "parts": [{"functionResponse": function_response}],
                }
            )
            continue

        # Fallback: keep unknown roles as user text instead of dropping content.
        if message.content:
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": message.content}],
                }
            )

    system_instruction = {"parts": system_parts} if system_parts else None
    return contents, system_instruction


def _function_call_part_from_slimx_tool_call(tool_call: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Converts SlimX/OpenAI-style assistant tool call dictionaries to Gemini functionCall parts.

    SlimX Client currently stores tool calls like:
        {
            "id": "...",
            "type": "function",
            "function": {
                "name": "...",
                "arguments": "{...}"
            }
        }
    """
    fn = tool_call.get("function") or {}
    name = fn.get("name") or tool_call.get("name")
    if not name:
        return None

    raw_args = fn.get("arguments") or tool_call.get("arguments") or {}
    args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else raw_args
    if not isinstance(args, dict):
        args = {}

    function_call: dict[str, Any] = {
        "name": name,
        "args": args,
    }

    call_id = tool_call.get("id")
    if call_id:
        function_call["id"] = call_id

    part: dict[str, Any] = {"functionCall": function_call}

    # Replay Gemini's required thought signature (captured into ToolCall.extra
    # at parse time and carried through the tool loop as `extra`).
    extra = tool_call.get("extra") or {}
    signature = extra.get("thoughtSignature")
    if signature:
        part["thoughtSignature"] = signature

    return part


def _generation_config(
    req: ChatRequest,
    extra_generation_config: Any = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    if isinstance(extra_generation_config, dict):
        config.update(extra_generation_config)

    if req.temperature is not None:
        config["temperature"] = req.temperature

    if req.max_tokens is not None:
        config["maxOutputTokens"] = req.max_tokens

    if req.response_format == "json_object":
        config["responseMimeType"] = "application/json"

    return config


def _tools_payload(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    declarations = []

    for tool in tools:
        declarations.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": _google_schema(tool.parameters),
            }
        )

    return [{"functionDeclarations": declarations}]


def _google_schema(schema: Any) -> Any:
    """
    Gemini function declarations accept an OpenAPI/JSON-schema-like subset.
    SlimX schemas are already close. This helper removes fields that commonly
    cause provider-side schema rejections and normalizes optional unions.
    """
    if isinstance(schema, list):
        return [_google_schema(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    # SlimX Optional[T] currently becomes {"anyOf": [T, {"type": "null"}]}.
    # For Gemini function declarations, prefer the non-null branch.
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        non_null = [
            item
            for item in schema["anyOf"]
            if not (isinstance(item, dict) and item.get("type") == "null")
        ]
        if non_null:
            return _google_schema(non_null[0])

    out: dict[str, Any] = {}

    for key, value in schema.items():
        # Gemini function declaration schemas often reject additionalProperties=False.
        if key == "additionalProperties" and value is False:
            continue

        out[key] = _google_schema(value)

    return out


def _parse_response(data: Dict[str, Any]) -> Result:
    text = "".join(_extract_text_parts(data))
    tool_calls = _extract_tool_calls(data)
    usage = _parse_usage(data)

    return Result(
        text=text,
        raw=data,
        usage=usage,
        tool_calls=tool_calls,
    )


def _parse_usage(data: Dict[str, Any]) -> Usage:
    usage = data.get("usageMetadata") or {}

    return Usage(
        prompt_tokens=usage.get("promptTokenCount"),
        completion_tokens=usage.get("candidatesTokenCount"),
        total_tokens=usage.get("totalTokenCount"),
    )


def _extract_text_parts(data: Dict[str, Any]) -> list[str]:
    out: list[str] = []

    for part in _candidate_parts(data):
        text = part.get("text")
        if isinstance(text, str):
            out.append(text)

    return out


def _extract_tool_calls(data: Dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []

    for part in _candidate_parts(data):
        function_call = part.get("functionCall")
        if not isinstance(function_call, dict):
            continue

        name = function_call.get("name") or ""
        call_id = function_call.get("id") or name
        args = function_call.get("args") or {}

        if isinstance(args, str):
            args = _safe_json_loads(args)

        if not isinstance(args, dict):
            args = {}

        # Gemini 3+ attaches a `thoughtSignature` to function-call parts that
        # MUST be echoed back on the next turn, or the follow-up request fails
        # with "Function call is missing a thought_signature". Carry it through
        # the tool loop via ToolCall.extra. (Sibling of functionCall on the part;
        # some payloads nest it inside functionCall instead.)
        signature = part.get("thoughtSignature") or function_call.get("thoughtSignature")
        extra = {"thoughtSignature": signature} if signature else {}

        calls.append(
            ToolCall(
                id=str(call_id or ""),
                name=str(name or ""),
                arguments=args,
                extra=extra,
            )
        )

    return calls


def _candidate_parts(data: Dict[str, Any]) -> list[dict[str, Any]]:
    candidates = data.get("candidates") or []
    if not candidates:
        return []

    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []

    return [part for part in parts if isinstance(part, dict)]


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except Exception:
        return value


def _raise_for_status(status_code: int, text: str) -> None:
    safe_text = _redact_error_text(text)

    if status_code in (401, 403):
        raise ProviderAuthError(f"Google error {status_code}: {safe_text}")

    if status_code == 429:
        raise ProviderRateLimitError(f"Google error {status_code}: {safe_text}")

    if status_code >= 400:
        raise ProviderError(f"Google error {status_code}: {safe_text}")


def _redact_error_text(text: str) -> str:
    if not text:
        return ""

    # Keep provider errors useful but avoid dumping huge bodies.
    return text[:2000]