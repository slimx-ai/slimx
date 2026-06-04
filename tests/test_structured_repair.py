from __future__ import annotations

from dataclasses import dataclass

import pytest

from slimx import llm
from slimx.errors import SchemaError
from slimx.providers import register
from slimx.providers.base import Provider, ProviderCapabilities
from slimx.types import Result, StreamEvent


@dataclass
class City:
    name: str
    country: str


class _BadThenGood(Provider):
    name = "repairok"
    capabilities = ProviderCapabilities(structured_output=True)

    def __init__(self):
        self.calls = 0

    def chat(self, req, *, tools=(), timeout=None):
        self.calls += 1
        if self.calls == 1:
            return Result(text="sorry, here is the city: not valid json")
        return Result(text='{"name":"Paris","country":"France"}')

    def stream(self, req, *, tools=(), timeout=None):
        yield StreamEvent.done()


class _AlwaysBad(Provider):
    name = "repairbad"
    capabilities = ProviderCapabilities(structured_output=True)

    def chat(self, req, *, tools=(), timeout=None):
        return Result(text="definitely not json")

    def stream(self, req, *, tools=(), timeout=None):
        yield StreamEvent.done()


def test_json_repair_recovers_from_bad_output():
    register("repairok", lambda **kw: _BadThenGood())
    res = llm("repairok:x").json("Give me a city.", schema=City, repair=2)
    assert isinstance(res.data, City)
    assert res.data.name == "Paris"
    assert res.data.country == "France"


def test_json_without_repair_raises_on_bad_output():
    register("repairbad", lambda **kw: _AlwaysBad())
    with pytest.raises(SchemaError):
        llm("repairbad:x").json("Give me a city.", schema=City)  # repair defaults to 0


def test_json_repair_gives_up_after_attempts():
    register("repairbad2", lambda **kw: _AlwaysBad())
    with pytest.raises(SchemaError):
        llm("repairbad2:x").json("Give me a city.", schema=City, repair=2)
