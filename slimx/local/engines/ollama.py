"""Ollama inference engine.

Reuses :class:`slimx.providers.ollama.OllamaProvider` for the base URL (``OLLAMA_BASE_URL``)
and ``/api/tags`` listing, and adds the bits SlimX needs for local-GPU UX:

- ``detect`` / ``health`` — is the daemon reachable, is the CLI installed?
- ``runtime_status`` — reads ``/api/ps`` and reports, per running model, whether it is
  fully on GPU, split CPU/GPU, or CPU-only (from Ollama's ``size`` vs ``size_vram``).
- ``pull_or_prepare_model`` — streams ``/api/pull`` progress.

``httpx`` is already a SlimX dependency; it is imported lazily inside methods so importing
this module stays cheap.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterator

from .base import (
    EngineHealth,
    EngineStatus,
    InferenceEngine,
    LocalModel,
    PullEvent,
    RunningModel,
    RuntimeStatus,
)

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaEngine(InferenceEngine):
    name = "ollama"
    kind = "ollama"
    supports_gpu = True
    supports_model_listing = True
    supports_runtime_status = True
    supports_launch = False

    def __init__(self, base_url: str | None = None, *, timeout: float = 5.0) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout

    # -- discovery / health ------------------------------------------------

    def detect(self) -> EngineStatus:
        installed = shutil.which("ollama") is not None
        health = self.health()
        detail = health.detail or ("reachable" if health.reachable else "")
        return EngineStatus(
            name=self.name,
            kind=self.kind,
            installed=installed,
            reachable=health.reachable,
            base_url=self.base_url,
            detail=detail,
        )

    def health(self) -> EngineHealth:
        import httpx

        try:
            resp = httpx.get(f"{self.base_url}/api/version", timeout=self.timeout)
        except httpx.HTTPError as exc:
            return EngineHealth(reachable=False, detail=f"{type(exc).__name__}: {exc}")
        if resp.status_code >= 400:
            return EngineHealth(reachable=False, detail=f"HTTP {resp.status_code}")
        version = ""
        try:
            version = str(resp.json().get("version", ""))
        except ValueError:
            pass
        return EngineHealth(reachable=True, detail=f"ollama {version}".strip())

    # -- models ------------------------------------------------------------

    def list_models(self) -> list[LocalModel]:
        import httpx

        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return []
        models: list[LocalModel] = []
        for entry in data.get("models", []) or []:
            name = entry.get("name")
            if not name:
                continue
            size = entry.get("size")
            models.append(
                LocalModel(
                    id=str(name),
                    engine=self.name,
                    size_gb=round(size / 1024**3, 2) if isinstance(size, (int, float)) else None,
                    raw=entry,
                )
            )
        return models

    def runtime_status(self) -> RuntimeStatus:
        import httpx

        try:
            resp = httpx.get(f"{self.base_url}/api/ps", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return RuntimeStatus(engine=self.name, running=[])
        running: list[RunningModel] = []
        for entry in data.get("models", []) or []:
            running.append(
                RunningModel(
                    name=str(entry.get("name", "")),
                    size_bytes=_as_int(entry.get("size")),
                    size_vram_bytes=_as_int(entry.get("size_vram")),
                )
            )
        return RuntimeStatus(engine=self.name, running=running)

    # -- lifecycle ---------------------------------------------------------

    def pull_or_prepare_model(self, model_id: str) -> Iterator[PullEvent]:
        import httpx

        from ...utils.ndjson import iter_ndjson

        url = f"{self.base_url}/api/pull"
        # Pulls can take minutes; only the connect phase is time-bounded.
        with httpx.Client(timeout=httpx.Timeout(None, connect=10.0)) as client:
            with client.stream("POST", url, json={"model": model_id, "stream": True}) as resp:
                resp.raise_for_status()
                for obj in iter_ndjson(resp.iter_bytes()):
                    yield PullEvent(
                        status=str(obj.get("status", "")),
                        completed=_as_int(obj.get("completed")),
                        total=_as_int(obj.get("total")),
                    )


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
