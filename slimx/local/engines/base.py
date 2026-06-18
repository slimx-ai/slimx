"""Inference-engine abstraction shared by Ollama, vLLM, and llama.cpp.

An :class:`InferenceEngine` is a thin, explicit wrapper around a local model runtime. It
reports whether the engine is installed/reachable (``detect``), lists local models, exposes
runtime placement (``runtime_status`` — is a running model on GPU, CPU, or split?), and can
prepare/pull a model. Serving still goes through SlimX's existing providers (``ollama:`` /
``oai:``); engines only add discovery, health, and lifecycle around that.

The dataclasses here are deliberately small and serializable so they map cleanly onto an
API/JSON layer in downstream apps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

# Runtime placement of a loaded model.
GPU = "gpu"
PARTIAL = "partial"
CPU = "cpu"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class LocalModel:
    id: str
    engine: str
    size_gb: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "engine": self.engine, "size_gb": self.size_gb}


@dataclass(frozen=True)
class EngineStatus:
    name: str
    kind: str
    installed: bool
    reachable: bool
    base_url: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineHealth:
    reachable: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunningModel:
    name: str
    size_bytes: int | None = None
    size_vram_bytes: int | None = None

    @property
    def placement(self) -> str:
        """How the model is loaded: fully on GPU, split, CPU-only, or unknown."""
        total = self.size_bytes
        vram = self.size_vram_bytes
        if total is None or vram is None:
            return UNKNOWN
        if total <= 0:
            return UNKNOWN
        if vram <= 0:
            return CPU
        if vram >= total:
            return GPU
        return PARTIAL

    @property
    def gpu_fraction(self) -> float | None:
        """Fraction of the model resident in VRAM (0.0–1.0), or None if unknown."""
        total = self.size_bytes
        vram = self.size_vram_bytes
        if not total or vram is None or total <= 0:
            return None
        return max(0.0, min(1.0, vram / total))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "size_vram_bytes": self.size_vram_bytes,
            "placement": self.placement,
            "gpu_fraction": self.gpu_fraction,
        }


@dataclass(frozen=True)
class RuntimeStatus:
    engine: str
    running: list[RunningModel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "running": [m.to_dict() for m in self.running]}


@dataclass(frozen=True)
class PullEvent:
    status: str
    completed: int | None = None
    total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InferenceEngine(ABC):
    name: str = "engine"
    kind: str = "openai_compatible"
    supports_gpu: bool = True
    supports_model_listing: bool = True
    supports_runtime_status: bool = False
    supports_launch: bool = False

    @abstractmethod
    def detect(self) -> EngineStatus: ...

    @abstractmethod
    def list_models(self) -> list[LocalModel]: ...

    @abstractmethod
    def health(self) -> EngineHealth: ...

    def runtime_status(self) -> RuntimeStatus:
        """Engines without a runtime-introspection API report an empty status."""
        return RuntimeStatus(engine=self.name, running=[])

    def pull_or_prepare_model(self, model_id: str) -> Iterator[PullEvent]:
        """Default: nothing to do. Override for engines that can fetch models."""
        raise NotImplementedError(f"{self.name} does not support pulling models")
