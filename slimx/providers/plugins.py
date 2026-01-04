from importlib import metadata
from typing import Callable, Dict
from .base import Provider

def load_entrypoint_providers() -> Dict[str, Callable[..., Provider]]:
    out: Dict[str, Callable[..., Provider]] = {}
    try:
        eps = metadata.entry_points()
        group = eps.select(group="slimx.providers") if hasattr(eps, "select") else eps.get("slimx.providers", [])
    except Exception:
        group = []
    for ep in group:
        try:
            out[ep.name] = ep.load()
        except Exception:
            continue
    return out
