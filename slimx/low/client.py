import json
import time
from typing import Iterable, Optional, Sequence
from ..messages import Message
from ..types import Result, StreamEvent
from ..tooling import ToolSpec, execute_tool
from ..utils.retry import retry, async_retry
from ..providers.base import Provider
from .types import ChatRequest

class Client:
    def __init__(self, provider: Provider, *, timeout: Optional[float]=None, retries: int=2):
        self.provider = provider
        self.timeout = timeout
        self.retries = retries
        self.provider_name = getattr(provider, "name", "provider")

    def chat(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=(), tool_runtime: str="none", max_steps: int=6) -> Result:
        tool_map = {t.name: t for t in tools}
        started = time.perf_counter()

        res = retry(lambda: self.provider.chat(req, tools=tools, timeout=self.timeout), retries=self.retries)

        if tool_runtime != "auto" or not res.tool_calls or not tool_map:
            self._attach_trace(res, req=req, started=started, steps=0)
            return res

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
        self._attach_trace(res, req=req, started=started, steps=steps)
        return res

    def stream(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()) -> Iterable[StreamEvent]:
        return self.provider.stream(req, tools=tools, timeout=self.timeout)

    async def achat(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=(), tool_runtime: str="none", max_steps: int=6) -> Result:
        started = time.perf_counter()
        res = await async_retry(
            lambda: self.provider.achat(req, tools=tools, timeout=self.timeout),
            retries=self.retries,
        )

        tool_map = {t.name: t for t in tools}
        if tool_runtime != "auto" or not res.tool_calls or not tool_map:
            self._attach_trace(res, req=req, started=started, steps=0)
            return res

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
        self._attach_trace(res, req=req, started=started, steps=steps)
        return res

    async def astream(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()):
        async for ev in self.provider.astream(req, tools=tools, timeout=self.timeout):
            yield ev

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


def _tool_call_to_provider_dict(tc) -> dict:
    return {
        "id": tc.id or tc.name,
        "type": "function",
        "function": {"name": tc.name, "arguments": tc.arguments_json},
    }
