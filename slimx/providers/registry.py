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


def describe_provider(name: str, *, async_mode: bool = False) -> Dict[str, Any]:
    """Return a provider's declared capabilities without making any network call.

    Constructs a throwaway provider instance with placeholder credentials purely
    to read its class-level ``capabilities``, so introspection never requires API
    keys or a running server.

    Example:
        >>> describe_provider("google")
        {'name': 'google', 'native': True, 'tools': True, 'structured_output': True,
         'streaming': True, 'async_chat': False, 'async_streaming': False,
         'vision': True, 'documents': True, 'audio_in': True, 'image_out': True,
         'image_in': True, 'image_edit': False, 'hosted_image_tool': False,
         'image_partial_streaming': False}
    """
    provider = get_provider(
        name,
        async_mode=async_mode,
        api_key="__introspect__",
        base_url="http://localhost",
    )
    caps = provider.capabilities
    return {
        "name": provider.name,
        "native": name != "oai",
        "tools": caps.tools,
        "structured_output": caps.structured_output,
        "streaming": caps.streaming,
        "async_chat": caps.async_chat,
        "async_streaming": caps.async_streaming,
        "vision": caps.vision,
        "documents": caps.documents,
        "audio_in": caps.audio_in,
        "image_out": caps.image_out,
        # Image input alias + the image-tool modalities (callers gate the
        # generate/edit UI on these).
        "image_in": caps.image_in,
        "image_edit": caps.image_edit,
        "hosted_image_tool": caps.hosted_image_tool,
        "image_partial_streaming": caps.image_partial_streaming,
    }
