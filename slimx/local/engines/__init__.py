"""Local inference engines (Ollama today; vLLM / llama.cpp via ``oai:`` later)."""

from __future__ import annotations

from .base import (
    EngineHealth,
    EngineStatus,
    InferenceEngine,
    LocalModel,
    PullEvent,
    RunningModel,
    RuntimeStatus,
)
from .ollama import OllamaEngine

__all__ = [
    "InferenceEngine",
    "EngineStatus",
    "EngineHealth",
    "RuntimeStatus",
    "RunningModel",
    "LocalModel",
    "PullEvent",
    "OllamaEngine",
]
