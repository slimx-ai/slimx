from typing import Any, Callable, Dict

from .base import Provider
from ._defaults import DEFAULT_FACTORIES
from .plugins import load_entrypoint_providers

ProviderFactory = Callable[..., Provider]
_REGISTRY: Dict[str, ProviderFactory] = {}


def _ensure_defaults() -> None:
    # Register built-in providers lazily so importing `slimx` has no side effects.
    for name, factory in DEFAULT_FACTORIES.items():
        _REGISTRY.setdefault(name, factory)


def register(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name] = factory


def load_plugins() -> None:
    _ensure_defaults()
    for name, factory in load_entrypoint_providers().items():
        _REGISTRY.setdefault(name, factory)


def list_providers() -> list[str]:
    load_plugins()
    return sorted(_REGISTRY.keys())


def get_provider(name: str, **kwargs: Any) -> Provider:
    load_plugins()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown provider '{name}'. Available: {list_providers()}")
    return _REGISTRY[name](**kwargs)
