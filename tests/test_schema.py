from dataclasses import dataclass
from slimx.schema import schema_for, coerce_dataclass

@dataclass
class City:
    name: str
    country: str

def test_schema_for_dataclass():
    s = schema_for(City)
    assert s["type"] == "object"
    assert "name" in s["properties"]

def test_coerce_dataclass():
    obj = {"name": "Paris", "country": "France"}
    c = coerce_dataclass(City, obj)
    assert c.name == "Paris"


def test_parse_json_plain():
    from slimx.schema import parse_json

    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_tolerates_markdown_fences():
    from slimx.schema import parse_json

    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('```\n{"a": 1}\n```') == {"a": 1}
    # Surrounding prose around the fenced block (common Anthropic shape).
    assert parse_json('Here is the JSON:\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}


def test_parse_json_still_raises_on_garbage():
    import pytest

    from slimx.errors import SchemaError
    from slimx.schema import parse_json

    with pytest.raises(SchemaError):
        parse_json("not json at all")
    with pytest.raises(SchemaError):
        parse_json("```\nstill not json\n```")
