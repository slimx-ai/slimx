import json
import time
from typing import Callable, Iterable, Mapping, Optional, Sequence
from ..messages import Message
from ..types import Result, StreamEvent
from ..tooling import ToolSpec, execute_tool
from ..utils.retry import retry, async_retry
from ..providers.base import Provider
from .types import ChatRequest

Hooks = Mapping[str, Callable[[dict], None]]


class Client:
    def __init__(
        self,
        provider: Provider,
        *,
        timeout: Optional[float] = None,
        retries: int = 2,
        hooks: Optional[Hooks] = None,
    ):
        self.provider = provider
        self.timeout = timeout
        self.retries = retries
        self.hooks = hooks or {}
        self.provider_name = getattr(provider, "name", "provider")

    def chat(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=(), tool_runtime: str="none", max_steps: int=6) -> Result:
        tool_map = {t.name: t for t in tools}
        started = time.perf_counter()
        snapshot = self._request_snapshot(req)
        self._fire("before_call", {"phase": "before_call", "provider": self.provider_name, "model": req.model})

        try:
            res = retry(lambda: self.provider.chat(req, tools=tools, timeout=self.timeout), retries=self.retries)

            if tool_runtime != "auto" or not res.tool_calls or not tool_map:
                return self._finish(res, req=req, started=started, steps=0, snapshot=snapshot)

            # Auto tool loop (best-effort cross-provider)
            messages = list(req.messages)
            steps = 0
            while res.tool_calls and steps < max_steps:
                steps += 1
                messages.append(Message.assistant("", tool_calls=[_tool_call_to_provider_dict(tc) for tc in res.tool_calls]))
                for tc in res.tool_calls:
                    spec = tool_map.get(tc.name)
                    if not spec:
                        continue
                    out = execute_tool(spec, tc.arguments)
                    messages.append(Message.tool(content=json.dumps(out), tool_call_id=tc.id or tc.name))

                req = ChatRequest(
                    model=req.model,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    response_format=req.response_format,
                    extra=req.extra,
                )
                res = retry(lambda: self.provider.chat(req, tools=tools, timeout=self.timeout), retries=self.retries)
                if not res.tool_calls:
                    break
            return self._finish(res, req=req, started=started, steps=steps, snapshot=snapshot)
        except Exception as e:
            self._fire_error(req, started, e)
            raise

    def stream(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()) -> Iterable[StreamEvent]:
        return self.provider.stream(req, tools=tools, timeout=self.timeout)

    def inspect(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=(), stream: bool=False):
        """Dry-run: return the exact HTTP request the provider would send."""
        return self.provider.build_request(req, tools=tools, stream=stream)

    async def achat(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=(), tool_runtime: str="none", max_steps: int=6) -> Result:
        started = time.perf_counter()
        snapshot = self._request_snapshot(req)
        self._fire("before_call", {"phase": "before_call", "provider": self.provider_name, "model": req.model})

        try:
            res = await async_retry(
                lambda: self.provider.achat(req, tools=tools, timeout=self.timeout),
                retries=self.retries,
            )

            tool_map = {t.name: t for t in tools}
            if tool_runtime != "auto" or not res.tool_calls or not tool_map:
                return self._finish(res, req=req, started=started, steps=0, snapshot=snapshot)

            messages = list(req.messages)
            steps = 0
            while res.tool_calls and steps < max_steps:
                steps += 1
                messages.append(Message.assistant("", tool_calls=[_tool_call_to_provider_dict(tc) for tc in res.tool_calls]))
                for tc in res.tool_calls:
                    spec = tool_map.get(tc.name)
                    if not spec:
                        continue
                    out = execute_tool(spec, tc.arguments)
                    messages.append(Message.tool(content=json.dumps(out), tool_call_id=tc.id or tc.name))

                req = ChatRequest(
                    model=req.model,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    response_format=req.response_format,
                    extra=req.extra,
                )

                res = await async_retry(
                    lambda: self.provider.achat(req, tools=tools, timeout=self.timeout),
                    retries=self.retries,
                )

                if not res.tool_calls:
                    break
            return self._finish(res, req=req, started=started, steps=steps, snapshot=snapshot)
        except Exception as e:
            self._fire_error(req, started, e)
            raise

    async def astream(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()):
        async for ev in self.provider.astream(req, tools=tools, timeout=self.timeout):
            yield ev

    # ---- internals -------------------------------------------------------

    def _finish(self, res: Result, *, req: ChatRequest, started: float, steps: int, snapshot: dict) -> Result:
        self._attach_trace(res, req=req, started=started, steps=steps)
        res.request = snapshot
        self._fire("after_call", {**res.trace, "ok": True})
        return res

    def _attach_trace(self, res: Result, *, req: ChatRequest, started: float, steps: int) -> None:
        res.trace.update({
            "provider": self.provider_name,
            "model": req.model,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "retries": self.retries,
            "tool_steps": steps,
            "tool_call_count": len(res.tool_calls or []),
            "timeout": self.timeout,
        })

    def _request_snapshot(self, req: ChatRequest) -> dict:
        return {
            "provider": self.provider_name,
            "model": req.model,
            "messages": [m.to_dict() for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "response_format": req.response_format,
            "extra": req.extra,
        }

    def _fire(self, name: str, event: dict) -> None:
        fn = self.hooks.get(name) if self.hooks else None
        if fn is None:
            return
        try:
            fn(event)
        except Exception:
            # A misbehaving hook must never break the underlying call.
            pass

    def _fire_error(self, req: ChatRequest, started: float, exc: BaseException) -> None:
        self._fire("after_call", {
            "provider": self.provider_name,
            "model": req.model,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        })


def _tool_call_to_provider_dict(tc) -> dict:
    d = {
        "id": tc.id or tc.name,
        "type": "function",
        "function": {"name": tc.name, "arguments": tc.arguments_json},
    }
    # Preserve provider-specific opaque data (e.g. Gemini thoughtSignature) so it
    # can be replayed to the same provider on the next tool-loop turn. Only the
    # originating provider reads it; OpenAI-shaped providers never see it because
    # their own tool calls carry no `extra`.
    extra = getattr(tc, "extra", None)
    if extra:
        d["extra"] = extra
    return d
