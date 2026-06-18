"""Local hardware discovery — a normalized, inspectable view of CPU/GPU resources.

`detect()` returns a :class:`HardwareProfile` describing the host's OS, CPU RAM, and
any GPUs SlimX can reason about, plus whether Docker can pass a GPU through and which
local runtime is the sensible default. It shells out to vendor tools (``nvidia-smi``,
``rocm-smi``, ``sysctl``) but **never raises**: a missing tool, a non-zero exit, or
unparseable output simply yields an empty/None result for that probe. Heavy imports are
avoided entirely (stdlib + ``subprocess`` only), keeping ``import slimx.local`` cheap and
side-effect-free.

This is consumed by ``slimx.local.recommend`` and surfaced through ``slimx doctor
--hardware``; downstream apps (SlimX-RAG, ControlRoom) call it instead of re-detecting.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

# Backends we *claim* a GPU can drive once the corresponding engine is installed. These
# are advisory tags for the recommender/UI, not a guarantee the engine is present.
_NVIDIA_BACKENDS = ["ollama-cuda", "llama.cpp-cuda", "vllm-cuda"]
_AMD_BACKENDS = ["ollama-rocm", "llama.cpp-hip", "vllm-rocm"]
_APPLE_BACKENDS = ["ollama-metal", "llama.cpp-metal"]

_PROBE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class GpuInfo:
    vendor: str  # nvidia | amd | apple | intel | unknown
    name: str
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    driver: str | None = None
    backends: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HardwareProfile:
    os: str
    arch: str
    cpu_ram_gb: float | None
    gpus: list[GpuInfo]
    docker_gpu_available: bool
    recommended_runtime: str  # "ollama" | "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Low-level probe helper
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str | None:
    """Run a command, returning stdout on success or None on any failure.

    Swallows missing binaries (FileNotFoundError), non-zero exits, and timeouts so
    callers can treat "no signal" uniformly.
    """
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _to_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# GPU vendor probes
# ---------------------------------------------------------------------------

def detect_nvidia() -> list[GpuInfo]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus: list[GpuInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        name, total_mib, free_mib, driver = parts[0], parts[1], parts[2], parts[3]
        total = _to_float(total_mib)
        free = _to_float(free_mib)
        gpus.append(
            GpuInfo(
                vendor="nvidia",
                name=name,
                vram_total_gb=round(total / 1024, 1) if total is not None else None,
                vram_free_gb=round(free / 1024, 1) if free is not None else None,
                driver=driver or None,
                backends=list(_NVIDIA_BACKENDS),
            )
        )
    return gpus


def detect_amd() -> list[GpuInfo]:
    # rocm-smi output varies a lot by version; we parse defensively and bail to [] on
    # anything unexpected rather than guessing wrong numbers.
    import json

    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    gpus: list[GpuInfo] = []
    for key, card in data.items() if isinstance(data, dict) else []:
        if not str(key).lower().startswith("card") or not isinstance(card, dict):
            continue
        name = str(
            card.get("Card series") or card.get("Card model") or card.get("Card SKU") or "AMD GPU"
        )
        total_b = _to_float(str(card.get("VRAM Total Memory (B)", "")))
        used_b = _to_float(str(card.get("VRAM Total Used Memory (B)", "")))
        total_gb = round(total_b / 1024**3, 1) if total_b else None
        free_gb = (
            round((total_b - used_b) / 1024**3, 1)
            if (total_b is not None and used_b is not None)
            else None
        )
        gpus.append(
            GpuInfo(
                vendor="amd",
                name=name,
                vram_total_gb=total_gb,
                vram_free_gb=free_gb,
                driver=None,
                backends=list(_AMD_BACKENDS),
            )
        )
    return gpus


def detect_apple() -> list[GpuInfo]:
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        return []
    # Apple Silicon uses unified memory: the GPU shares system RAM, so total RAM is the
    # practical VRAM ceiling. We report it as both total and free (free is unknowable here).
    ram_gb = detect_cpu_ram_gb()
    name = "Apple Silicon GPU"
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if out and out.strip():
        name = f"{out.strip()} (Metal)"
    return [
        GpuInfo(
            vendor="apple",
            name=name,
            vram_total_gb=ram_gb,
            vram_free_gb=None,
            driver=None,
            backends=list(_APPLE_BACKENDS),
        )
    ]


# ---------------------------------------------------------------------------
# CPU / Docker probes
# ---------------------------------------------------------------------------

def detect_cpu_ram_gb() -> float | None:
    # Linux: /proc/meminfo (kB).
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = _to_float(line.split()[1])
                    if kb is not None:
                        return round(kb / 1024**2, 1)
    except OSError:
        pass
    # macOS / BSD: sysctl hw.memsize (bytes).
    out = _run(["sysctl", "-n", "hw.memsize"])
    if out:
        b = _to_float(out)
        if b is not None:
            return round(b / 1024**3, 1)
    # Portable fallback via sysconf where available.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def detect_docker_gpu() -> bool:
    if shutil.which("nvidia-ctk") or shutil.which("nvidia-container-runtime"):
        return True
    return os.path.exists("/proc/driver/nvidia")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def detect() -> HardwareProfile:
    """Return a normalized snapshot of local hardware. Best-effort, never raises."""
    gpus: list[GpuInfo] = []
    for probe in (detect_nvidia, detect_amd, detect_apple):
        try:
            gpus.extend(probe())
        except Exception:
            # A misbehaving vendor tool must never take down discovery.
            continue

    recommended_runtime = "ollama" if gpus else "cpu"

    return HardwareProfile(
        os=platform.system().lower() or "unknown",
        arch=platform.machine() or "unknown",
        cpu_ram_gb=detect_cpu_ram_gb(),
        gpus=gpus,
        docker_gpu_available=detect_docker_gpu(),
        recommended_runtime=recommended_runtime,
    )
