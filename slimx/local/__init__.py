"""Local hardware + inference-engine awareness for SlimX.

An **opt-in** subpackage: importing ``slimx`` does not import ``slimx.local``, and this
package adds no new required dependencies (stdlib + the already-present ``httpx``). It gives
SlimX — and anything that depends on it (SlimX-RAG, ControlRoom) — a single, inspectable
place to:

- detect local hardware (``hardware.detect``)
- describe local inference engines and runtime placement (``engines``)
- recommend local models that fit the hardware for a task (``recommend``)

Nothing here calls a cloud provider or leaks data off the machine.
"""

from __future__ import annotations

from .catalog import CatalogModel, load_catalog, models_for_task
from .engines import (
    EngineHealth,
    EngineStatus,
    InferenceEngine,
    LocalModel,
    OllamaEngine,
    RunningModel,
    RuntimeStatus,
)
from .hardware import GpuInfo, HardwareProfile, detect
from .recommend import Recommendation, Recommendations, recommend

__all__ = [
    # hardware
    "detect",
    "HardwareProfile",
    "GpuInfo",
    # catalog
    "load_catalog",
    "models_for_task",
    "CatalogModel",
    # recommend
    "recommend",
    "Recommendations",
    "Recommendation",
    # engines
    "InferenceEngine",
    "OllamaEngine",
    "EngineStatus",
    "EngineHealth",
    "RuntimeStatus",
    "RunningModel",
    "LocalModel",
]
