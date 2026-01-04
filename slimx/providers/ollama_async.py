import os

import httpx
from typing import Any, Dict, Sequence
from ..types import Result, StreamEvent, Usage
from ..tooling import ToolSpec
from ..errors import ProviderError
from ..utils.ndjson import aiter_ndjson
from .base import Provider

class OllamaAsyncProvider(Provider):
    name="ollama"
    def __init__(self, base_url: str="http://localhost:11434"):
        self.base_url=base_url.rstrip("/")
    @classmethod
    def from_env(cls):
        return cls(os.environ.get("OLLAMA_BASE_URL","http://localhost:11434"))
    def chat(self, req, *, tools: Sequence[ToolSpec]=()):
        raise NotImplementedError
    def stream(self, req, *, tools: Sequence[ToolSpec]=()):
        raise NotImplementedError
    async def achat(self, req, *, tools: Sequence[ToolSpec]=()):
        payload: Dict[str, Any]={"model":req.model,"messages":[m.to_dict() for m in req.messages if m.role in ("user","assistant","system")],"stream":False}
        if req.temperature is not None:
            payload.setdefault("options",{})["temperature"]=req.temperature
        url=f"{self.base_url}/api/chat"
        async with httpx.AsyncClient(timeout=60.0) as c:
            r=await c.post(url, json=payload)
        if r.status_code>=400:
            raise ProviderError(f"Ollama error {r.status_code}: {r.text}")
        data=r.json()
        text=(data.get("message") or {}).get("content") or ""
        usage=Usage(prompt_tokens=data.get("prompt_eval_count"), completion_tokens=data.get("eval_count"))
        return Result(text=text, raw=data, usage=usage)
    async def astream(self, req, *, tools: Sequence[ToolSpec]=()):
        payload: Dict[str, Any]={"model":req.model,"messages":[m.to_dict() for m in req.messages if m.role in ("user","assistant","system")],"stream":True}
        if req.temperature is not None:
            payload.setdefault("options",{})["temperature"]=req.temperature
        url=f"{self.base_url}/api/chat"
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", url, json=payload) as r:
                if r.status_code>=400:
                    raise ProviderError(f"Ollama error {r.status_code}: {await r.aread()}")
                async for obj in aiter_ndjson(r.aiter_bytes()):
                    if obj.get("done") is True:
                        break
                    chunk=(obj.get("message") or {}).get("content") or ""
                    if chunk:
                        yield StreamEvent(type="token", text=chunk, raw=obj)
        yield StreamEvent(type="done")
