import dataclasses
import json
import types

from typing import Any, Dict, List, Tuple, Union, get_args, get_origin, get_type_hints
from .errors import SchemaError

# Union origins to recognize: typing.Union[...] and PEP 604 (X | Y).
_UNION_ORIGINS = (Union, getattr(types, "UnionType", Union))


def _is_optional(tp: Any) -> Tuple[bool, Any]:
    origin = get_origin(tp)
    if origin in _UNION_ORIGINS:
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
    # Resolve string annotations (PEP 563 / `from __future__ import annotations`).
    # Falling back to the raw ``f.type`` keeps this working even if a forward
    # reference can't be resolved.
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    props: Dict[str, Any] = {}
    required: List[str] = []
    for f in dataclasses.fields(cls):
        props[f.name] = _schema_for_type(hints.get(f.name, f.type))
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)
    return {"type":"object","properties":props,"required":required,"additionalProperties":False}

def parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception as e:
        raise SchemaError(f"Failed to parse JSON: {e}") from e

def coerce_dataclass(cls: Any, obj: Any) -> Any:
    """Build a dataclass instance from a plain ``dict`` (best-effort).

    Recurses into nested dataclasses and ``List``/``Dict`` fields, and applies
    light scalar coercion (e.g. ``"3"`` -> ``3`` for an ``int`` field). Values
    that can't be coerced are passed through unchanged rather than raising, so
    a slightly-off model response still yields a usable object.
    """
    if dataclasses.is_dataclass(cls) and not isinstance(cls, type):
        cls = type(cls)
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise SchemaError("coerce_dataclass expects dataclass type")
    if not isinstance(obj, dict):
        raise SchemaError(f"Expected object for {cls.__name__}")
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in obj:
            continue
        tp = hints.get(f.name, f.type)
        kwargs[f.name] = _coerce_value(tp, obj[f.name])
    return cls(**kwargs)


def _coerce_value(tp: Any, value: Any) -> Any:
    # Unwrap Optional[T] / T | None.
    opt, inner = _is_optional(tp)
    if opt:
        if value is None:
            return None
        tp = inner

    if value is None:
        return None

    if dataclasses.is_dataclass(tp) and isinstance(value, dict):
        return coerce_dataclass(tp, value)

    origin = get_origin(tp)
    if origin in (list, List) and isinstance(value, list):
        args = get_args(tp) or (Any,)
        return [_coerce_value(args[0], item) for item in value]
    if origin in (dict, Dict) and isinstance(value, dict):
        args = get_args(tp)
        val_tp = args[1] if len(args) > 1 else Any
        return {k: _coerce_value(val_tp, v) for k, v in value.items()}

    return _coerce_scalar(tp, value)


def _coerce_scalar(tp: Any, value: Any) -> Any:
    # bool must be checked before int (bool is a subclass of int).
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return value
    if tp is int and not isinstance(value, bool):
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return value
        return value
    if tp is float:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value
        return value
    return value
