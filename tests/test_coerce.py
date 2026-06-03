from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from slimx.schema import coerce_dataclass


@dataclass
class Address:
    city: str
    zip: int


@dataclass
class Person:
    name: str
    age: int
    height: float
    active: bool
    address: Address
    nicknames: List[str]
    note: Optional[str] = None


def test_coerce_nested_and_scalars():
    p = coerce_dataclass(
        Person,
        {
            "name": "Ada",
            "age": "36",          # string -> int
            "height": 1,          # int -> float
            "active": "true",     # string -> bool
            "address": {"city": "London", "zip": "55"},
            "nicknames": ["countess"],
        },
    )
    assert p.age == 36 and isinstance(p.age, int)
    assert p.height == 1.0 and isinstance(p.height, float)
    assert p.active is True
    assert isinstance(p.address, Address)
    assert p.address.zip == 55
    assert p.nicknames == ["countess"]
    assert p.note is None


def test_coerce_passes_through_uncoercible():
    # Non-numeric string for an int field is left untouched, not raised.
    p = coerce_dataclass(
        Person,
        {
            "name": "X",
            "age": "not-a-number",
            "height": 2.5,
            "active": False,
            "address": {"city": "Y", "zip": 1},
            "nicknames": [],
        },
    )
    assert p.age == "not-a-number"
    assert p.height == 2.5
