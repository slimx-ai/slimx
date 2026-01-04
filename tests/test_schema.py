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
