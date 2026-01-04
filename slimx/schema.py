import dataclasses
import json

from typing import Any, Dict, List, Tuple, Union, get_args, get_origin
from .errors import SchemaError

def _is_optional(tp: Any) -> Tuple[bool, Any]:
    origin = get_origin(tp)
    if origin is Union:
        args = list(get_args(tp))
        if type(None) in args and len(args) == 2:
            other = args[0] if args[1] is type(None) else args[1]
            return True, other
    return False, tp

def _schema_for_type(tp: Any) -> Dict[str, Any]:
    opt, inner = _is_optional(tp)
    tp = inner
    origin = get_origin(tp)
    if tp is str:
        s = {"type": "string"}
    elif tp is int:
        s = {"type": "integer"}
    elif tp is float:
        s = {"type": "number"}
    elif tp is bool:
        s = {"type": "boolean"}
    elif tp is Any:
        s = {}
    elif origin in (list, List):
        args = get_args(tp) or (Any,)
        s = {"type": "array", "items": _schema_for_type(args[0])}
    elif origin in (dict, Dict):
        args = get_args(tp) or (Any, Any)
        s = {"type": "object", "additionalProperties": _schema_for_type(args[1] if len(args)>1 else Any)}
    elif dataclasses.is_dataclass(tp):
        s = schema_for(tp)
    else:
        s = {"type": "string"}
    if opt:
        s = {"anyOf": [s, {"type":"null"}]}
    return s

def schema_for(cls: Any) -> Dict[str, Any]:
    """Return a JSON Schema (draft-ish) for a dataclass.

    Accepts either a dataclass *type* or a dataclass *instance*.
    """
    if dataclasses.is_dataclass(cls) and not isinstance(cls, type):
        cls = type(cls)
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise SchemaError("schema_for expects a dataclass type")
    props: Dict[str, Any] = {}
    required: List[str] = []
    for f in dataclasses.fields(cls):
        props[f.name] = _schema_for_type(f.type)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)
    return {"type":"object","properties":props,"required":required,"additionalProperties":False}

def parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception as e:
        raise SchemaError(f"Failed to parse JSON: {e}") from e

def coerce_dataclass(cls: Any, obj: Any) -> Any:
    if dataclasses.is_dataclass(cls) and not isinstance(cls, type):
        cls = type(cls)
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise SchemaError("coerce_dataclass expects dataclass type")
    if not isinstance(obj, dict):
        raise SchemaError(f"Expected object for {cls.__name__}")
    kwargs = {f.name: obj.get(f.name) for f in dataclasses.fields(cls) if f.name in obj}
    return cls(**kwargs)
