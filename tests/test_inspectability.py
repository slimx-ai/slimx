from __future__ import annotations

import json
import os

import pytest
from fakes import FakeProvider

from slimx import CallRecord, InspectedRequest, Message
from slimx.errors import ProviderAuthError
from slimx.low import ChatRequest, Client
from slimx.providers.anthropic import AnthropicProvider
from slimx.providers.base import Provider, ProviderCapabilities
from slimx.providers.google import GoogleProvider
from slimx.providers.openai import OpenAIProvider
from slimx.types import StreamEvent


# --------------------------------------------------------------------------
# Inspect / dry-run mode
# --------------------------------------------------------------------------

def test_openai_build_request_redacts_and_builds_payload():
    p = OpenAIProvider(api_key="sk-secret")
    ir = p.build_request(ChatRequest(model="gpt-4.1-nano", messages=[Message.user("hi")]))
    assert isinstance(ir, InspectedRequest)
    assert ir.method == "POST"
    assert ir.url == "https://api.openai.com/v1/chat/completions"
    assert ir.headers["Authorization"] == "Bearer ***"  # secret never leaked
    assert ir.payload["model"] == "gpt-4.1-nano"
    assert ir.payload["messages"][0]["content"] == "hi"


def test_google_build_request_picks_stream_endpoint_and_redacts():
    p = GoogleProvider(api_key="g-secret")
    chat_url = p.build_request(ChatRequest(model="gemini-3.5-flash", messages=[Message.user("hi")])).url
    stream_url = p.build_request(
        ChatRequest(model="gemini-3.5-flash", messages=[Message.user("hi")]), stream=True
    ).url
    assert chat_url.endswith("models/gemini-3.5-flash:generateContent")
    assert stream_url.endswith("models/gemini-3.5-flash:streamGenerateContent?alt=sse")
    ir = p.build_request(ChatRequest(model="gemini-3.5-flash", messages=[Message.user("hi")]))
    assert ir.headers["x-goog-api-key"] == "***"


def test_anthropic_build_request_redacts():
    p = AnthropicProvider(api_key="a-secret")
    ir = p.build_request(ChatRequest(model="claude-x", messages=[Message.user("hi")]))
    assert ir.url.endswith("/v1/messages")
    assert ir.headers["x-api-key"] == "***"


def test_model_inspect_does_not_call_network(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    from slimx import llm

    ir = llm("openai:gpt-4.1-nano", temperature=0.1).inspect("Hello")
    assert ir.payload["temperature"] == 0.1
    assert "Bearer ***" == ir.headers["Authorization"]
    # pretty() is a readable, secret-free dump
    assert "sk-secret" not in ir.pretty()


# --------------------------------------------------------------------------
# Reproducible call records
# --------------------------------------------------------------------------

def test_result_carries_request_snapshot():
    res = Client(FakeProvider()).chat(ChatRequest(model="demo", messages=[Message.user("hello")]))
    assert res.request is not None
    assert res.request["model"] == "demo"
    assert res.request["messages"][0]["content"] == "hello"
    assert res.request["provider"] == "fake"


def test_to_record_and_save_load_roundtrip(tmp_path):
    res = Client(FakeProvider()).chat(ChatRequest(model="demo", messages=[Message.user("hello")]))
    rec = res.to_record()
    assert isinstance(rec, CallRecord)
    assert rec.provider == "fake"
    assert rec.model == "demo"
    assert rec.response["text"] == "fake:hello"
    assert rec.slimx_version

    path = os.path.join(tmp_path, "run.json")
    rec.save(path)
    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk["response"]["text"] == "fake:hello"

    loaded = CallRecord.load(path)
    assert loaded.provider == "fake"
    assert loaded.response["text"] == "fake:hello"


# --------------------------------------------------------------------------
# Trace hooks
# --------------------------------------------------------------------------

class _BoomProvider(Provider):
    name = "boom"
    capabilities = ProviderCapabilities()

    def chat(self, req, *, tools=(), timeout=None):
        raise ProviderAuthError("bad key")

    def stream(self, req, *, tools=(), timeout=None):
        yield StreamEvent.done()


def test_hooks_fire_on_success():
    events = []
    hooks = {
        "before_call": lambda e: events.append(("before", e["model"])),
        "after_call": lambda e: events.append(("after", e.get("ok"), e.get("provider"))),
    }
    Client(FakeProvider(), hooks=hooks).chat(ChatRequest(model="demo", messages=[Message.user("hi")]))
    assert ("before", "demo") in events
    assert ("after", True, "fake") in events


def test_hooks_fire_on_error_and_reraise():
    events = []
    hooks = {"after_call": lambda e: events.append((e.get("ok"), e.get("error")))}
    with pytest.raises(ProviderAuthError):
        Client(_BoomProvider(), retries=0, hooks=hooks).chat(
            ChatRequest(model="m", messages=[Message.user("hi")])
        )
    assert len(events) == 1
    ok, error = events[0]
    assert ok is False
    assert "ProviderAuthError" in error


def test_misbehaving_hook_never_breaks_the_call():
    def boom(_event):
        raise RuntimeError("hook exploded")

    res = Client(FakeProvider(), hooks={"after_call": boom}).chat(
        ChatRequest(model="demo", messages=[Message.user("hi")])
    )
    assert res.text == "fake:hi"  # call still succeeds despite the hook raising
