"""Schema generation must work when the dataclass module uses PEP 563
(`from __future__ import annotations`), which turns annotations into strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from slimx.schema import _schema_for_type, schema_for


@dataclass
class Item:
    name: str
    qty: int
    price: float
    in_stock: bool


@dataclass
class Order:
    item: Item
    tags: List[str]
    note: Optional[str] = None


def test_schema_for_resolves_string_annotations():
    s = schema_for(Item)
    props = s["properties"]
    assert props["name"]["type"] == "string"
    assert props["qty"]["type"] == "integer"
    assert props["price"]["type"] == "number"
    assert props["in_stock"]["type"] == "boolean"
    assert set(s["required"]) == {"name", "qty", "price", "in_stock"}


def test_schema_for_handles_nested_and_optional():
    s = schema_for(Order)
    props = s["properties"]
    assert props["item"]["type"] == "object"
    assert props["item"]["properties"]["qty"]["type"] == "integer"
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    # Optional[str] -> nullable string; not required (has a default).
    assert props["note"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert "note" not in s["required"]


def test_pep604_optional_is_nullable():
    assert _schema_for_type(int | None) == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
