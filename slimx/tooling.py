import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, get_type_hints
from .errors import ToolExecutionError
from .schema import _schema_for_type

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]

def tool(fn: Callable[..., Any]) -> ToolSpec:
    name = fn.__name__
    desc = (inspect.getdoc(fn) or "").strip() or f"Tool: {name}"
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    props: Dict[str, Any] = {}
    required = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError("*args/**kwargs not supported for @tool")
        props[p.name] = _schema_for_type(hints.get(p.name, Any))
        if p.default is inspect._empty:
            required.append(p.name)
    schema = {"type":"object","properties":props,"required":required,"additionalProperties":False}
    return ToolSpec(name=name, description=desc, parameters=schema, fn=fn)

def execute_tool(spec: ToolSpec, arguments: Dict[str, Any]) -> Any:
    try:
        return spec.fn(**arguments)
    except Exception as e:
        raise ToolExecutionError(f"Tool '{spec.name}' failed: {e}") from e
