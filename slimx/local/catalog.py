"""Bundled catalog of local models with hardware-relevant metadata.

The catalog is a static JSON file (``data/local_models.json``) read lazily. It carries
just enough per-model data to reason about fit — parameter count, quantization, VRAM
floor/ideal, disk size, context window, suitable tasks — without pulling in any provider
code or making network calls. ``recommend`` consumes this; nothing here is dynamic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "local_models.json"


@dataclass(frozen=True)
class CatalogModel:
    id: str
    engine: str
    family: str
    params_b: float
    quantization: str
    min_vram_gb: float
    ideal_vram_gb: float
    disk_size_gb: float
    context: int
    tasks: tuple[str, ...]
    privacy: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def load_catalog() -> tuple[CatalogModel, ...]:
    """Load and cache the bundled model catalog."""
    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    models: list[CatalogModel] = []
    for entry in raw.get("models", []):
        models.append(
            CatalogModel(
                id=str(entry["id"]),
                engine=str(entry.get("engine", "ollama")),
                family=str(entry.get("family", "")),
                params_b=float(entry.get("params_b", 0)),
                quantization=str(entry.get("quantization", "default")),
                min_vram_gb=float(entry.get("min_vram_gb", 0)),
                ideal_vram_gb=float(entry.get("ideal_vram_gb", 0)),
                disk_size_gb=float(entry.get("disk_size_gb", 0)),
                context=int(entry.get("context", 0)),
                tasks=tuple(entry.get("tasks", ())),
                privacy=str(entry.get("privacy", "local")),
            )
        )
    return tuple(models)


def models_for_task(task: str | None = None, *, engine: str | None = None) -> list[CatalogModel]:
    """Return catalog models matching a task (and optionally an engine).

    ``task=None`` returns everything. Task matching is membership in the model's
    declared ``tasks`` tuple.
    """
    out: list[CatalogModel] = []
    for model in load_catalog():
        if engine is not None and model.engine != engine:
            continue
        if task is not None and task not in model.tasks:
            continue
        out.append(model)
    return out
