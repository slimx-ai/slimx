import json
from typing import Iterable, Optional, Sequence
from ..messages import Message
from ..types import Result, StreamEvent
from ..tooling import ToolSpec, execute_tool
from ..utils.retry import retry
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

        res = retry(lambda: self.provider.chat(req, tools=tools), retries=self.retries)

        if tool_runtime != "auto" or not res.tool_calls or not tool_map:
            return res

        # Auto tool loop (best-effort cross-provider)
        messages = list(req.messages)
        steps = 0
        while res.tool_calls and steps < max_steps:
            steps += 1
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
            res = retry(lambda: self.provider.chat(req, tools=tools), retries=self.retries)
            if not res.tool_calls:
                break
        return res

    def stream(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()) -> Iterable[StreamEvent]:
        return self.provider.stream(req, tools=tools)

    async def achat(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=(), tool_runtime: str="none", max_steps: int=6) -> Result:
        # async retry
        last = None
        for i in range(self.retries + 1):
            try:
                res = await self.provider.achat(req, tools=tools)
                break
            except Exception as e:
                last = e
                if i >= self.retries:
                    raise
        else:
            raise last  # type: ignore[misc]

        tool_map = {t.name: t for t in tools}
        if tool_runtime != "auto" or not res.tool_calls or not tool_map:
            return res

        messages = list(req.messages)
        steps = 0
        while res.tool_calls and steps < max_steps:
            steps += 1
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

            last = None
            for i in range(self.retries + 1):
                try:
                    res = await self.provider.achat(req, tools=tools)
                    break
                except Exception as e:
                    last = e
                    if i >= self.retries:
                        raise
            else:
                raise last  # type: ignore[misc]

            if not res.tool_calls:
                break
        return res

    async def astream(self, req: ChatRequest, *, tools: Sequence[ToolSpec]=()):
        async for ev in self.provider.astream(req, tools=tools):
            yield ev
