import json
import os

import httpx
from typing import Dict, Iterable, Sequence
from ..types import Result, StreamEvent, ToolCall, Usage
from ..tooling import ToolSpec
from ..errors import ProviderError, ProviderAuthError, ProviderRateLimitError
from ..utils.sse import iter_sse_data
from .base import Provider

def _tools_payload(tools: Sequence[ToolSpec]):
    return [{"type":"function","function":{"name":t.name,"description":t.description,"parameters":t.parameters}} for t in tools]

class OpenAIProvider(Provider):
    name="openai"
    def __init__(self, api_key: str, base_url: str="https://api.openai.com/v1"):
        self.api_key=api_key
        self.base_url=base_url.rstrip("/")
    @classmethod
    def from_env(cls):
        api_key=os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set")
        base_url=os.environ.get("OPENAI_BASE_URL","https://api.openai.com/v1")
        return cls(api_key, base_url)
    def _headers(self)->Dict[str,str]:
        return {"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"}
    def chat(self, req, *, tools: Sequence[ToolSpec]=()):
        payload=req.to_dict()
        if tools:
            payload["tools"]=_tools_payload(tools)
        if payload.get("response_format")=="json_object":
            payload["response_format"]={"type":"json_object"}
        url=f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=30.0) as c:
            r=c.post(url, headers=self._headers(), json=payload)
        if r.status_code==401:
            raise ProviderAuthError(r.text)
        if r.status_code==429:
            raise ProviderRateLimitError(r.text)
        if r.status_code>=400:
            raise ProviderError(f"OpenAI error {r.status_code}: {r.text}")
        data=r.json()
        msg=data["choices"][0]["message"]
        text=msg.get("content") or ""
        tool_calls=[]
        for tc in msg.get("tool_calls") or []:
            fn=tc.get("function") or {}
            args=fn.get("arguments") or "{}"
            try:
                args_obj=json.loads(args) if isinstance(args,str) else args
            except Exception:
                args_obj={}
            tool_calls.append(ToolCall(id=tc.get("id",""), name=fn.get("name",""), arguments=args_obj))
        usage=Usage.from_openai(data.get("usage") or {})
        return Result(text=text, raw=data, usage=usage, tool_calls=tool_calls)
    def stream(self, req, *, tools: Sequence[ToolSpec]=()) -> Iterable[StreamEvent]:
        payload=req.to_dict()
        if tools:
            payload["tools"]=_tools_payload(tools)
        if payload.get("response_format")=="json_object":
            payload["response_format"]={"type":"json_object"}
        payload["stream"]=True
        url=f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=None) as c:
            with c.stream("POST", url, headers=self._headers(), json=payload) as r:
                if r.status_code==401:
                    raise ProviderAuthError(r.text)
                if r.status_code==429:
                    raise ProviderRateLimitError(r.text)
                if r.status_code>=400:
                    raise ProviderError(f"OpenAI error {r.status_code}: {r.text}")
                tool_acc={}
                for chunk in iter_sse_data(r.iter_bytes()):
                    if chunk=="[DONE]":
                        break
                    try:
                        obj=json.loads(chunk)
                    except Exception:
                        continue
                    delta=obj["choices"][0].get("delta",{})
                    if delta.get("content"):
                        yield StreamEvent(type="text_delta", text=delta["content"], raw=obj)
                    for tc in delta.get("tool_calls") or []:
                        tc_id=str(tc.get("id") or tc.get("index"))
                        fn=tc.get("function") or {}
                        tool_acc.setdefault(tc_id, {"name":fn.get("name"), "args":""})
                        if fn.get("name"):
                            tool_acc[tc_id]["name"]=fn.get("name")
                        if fn.get("arguments"):
                            tool_acc[tc_id]["args"]+=fn.get("arguments")
                for tc_id, acc in tool_acc.items():
                    try:
                        args_obj=json.loads(acc["args"] or "{}")
                    except Exception:
                        args_obj={}
                    yield StreamEvent(type="tool_call", tool_call=ToolCall(id=tc_id, name=acc.get("name") or "", arguments=args_obj), raw=acc)
        yield StreamEvent(type="done")
