import os

import httpx
from typing import Any, Dict, Sequence
from ..types import Result, StreamEvent, Usage
from ..tooling import ToolSpec
from ..errors import ProviderError, ProviderAuthError, ProviderRateLimitError
from .base import Provider

class AnthropicAsyncProvider(Provider):
    name="anthropic"
    def __init__(self, api_key: str, base_url: str="https://api.anthropic.com", version: str="2023-06-01"):
        self.api_key=api_key
        self.base_url=base_url.rstrip("/")
        self.version=version
    @classmethod
    def from_env(cls):
        api_key=os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not set")
        return cls(api_key, os.environ.get("ANTHROPIC_BASE_URL","https://api.anthropic.com"), os.environ.get("ANTHROPIC_VERSION","2023-06-01"))
    def _headers(self)->Dict[str,str]:
        return {"x-api-key":self.api_key,"anthropic-version":self.version,"content-type":"application/json"}
    def chat(self, req, *, tools: Sequence[ToolSpec]=()):
        raise NotImplementedError
    def stream(self, req, *, tools: Sequence[ToolSpec]=()):
        raise NotImplementedError
    async def achat(self, req, *, tools: Sequence[ToolSpec]=()):
        sys=[]
        msgs=[]
        for m in req.messages:
            if m.role=="system":
                sys.append(m.content)
            elif m.role in ("user","assistant"):
                msgs.append({"role":m.role,"content":m.content})
        payload: Dict[str, Any]={"model":req.model,"max_tokens":req.max_tokens or 1024,"messages":msgs}
        if sys:
            payload["system"]="\n".join(sys)
        if req.temperature is not None:
            payload["temperature"]=req.temperature
        url=f"{self.base_url}/v1/messages"
        async with httpx.AsyncClient(timeout=30.0) as c:
            r=await c.post(url, headers=self._headers(), json=payload)
        if r.status_code in (401,403):
            raise ProviderAuthError(r.text)
        if r.status_code==429:
            raise ProviderRateLimitError(r.text)
        if r.status_code>=400:
            raise ProviderError(f"Anthropic error {r.status_code}: {r.text}")
        data=r.json()
        text="".join([b.get("text","") for b in (data.get("content") or []) if b.get("type")=="text"])
        usage=Usage(prompt_tokens=(data.get("usage") or {}).get("input_tokens"), completion_tokens=(data.get("usage") or {}).get("output_tokens"))
        return Result(text=text, raw=data, usage=usage)
    async def astream(self, req, *, tools: Sequence[ToolSpec]=()):
        res=await self.achat(req, tools=tools)
        yield StreamEvent(type="message", text=res.text, raw=res.raw)
        yield StreamEvent(type="done")
