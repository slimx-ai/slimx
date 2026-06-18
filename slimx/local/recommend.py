"""Hardware-aware local-model recommendations.

``recommend(profile, task=...)`` scores each catalog model against the host's *free*
VRAM (falling back to total, then CPU RAM) and sorts it into three buckets —
``recommended`` / ``possible`` / ``not_recommended`` — each entry carrying a plain-English
``why``, an ``estimated_speed``, and a ``risk``. The scoring is intentionally simple and
explicit (no ML, no network): it uses the catalog's VRAM floor/ideal plus a rough KV-cache
estimate so the output is inspectable and reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .catalog import CatalogModel, models_for_task
from .hardware import HardwareProfile

# Fit buckets.
SAFE = "safe"
TIGHT = "tight"
OFFLOAD = "offload"
TOO_BIG = "too_big"

# Above this size, CPU offload is so slow it is not worth recommending even when the
# machine technically has the RAM — such models fall straight to "not_recommended".
_OFFLOAD_MAX_PARAMS_B = 13.0


@dataclass(frozen=True)
class Recommendation:
    model: str  # provider-prefixed ref, e.g. "ollama:llama3.2:3b"
    engine: str
    fit: str
    why: str
    estimated_speed: str  # fast | moderate | slow
    risk: str  # low | medium | high
    installed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendations:
    recommended: list[Recommendation] = field(default_factory=list)
    possible: list[Recommendation] = field(default_factory=list)
    not_recommended: list[Recommendation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended": [r.to_dict() for r in self.recommended],
            "possible": [r.to_dict() for r in self.possible],
            "not_recommended": [r.to_dict() for r in self.not_recommended],
        }


def _best_free_vram_gb(profile: HardwareProfile) -> float:
    """Largest usable VRAM across GPUs, preferring free over total. 0 if GPU-less."""
    best = 0.0
    for gpu in profile.gpus:
        vram = gpu.vram_free_gb if gpu.vram_free_gb is not None else gpu.vram_total_gb
        if vram is not None and vram > best:
            best = vram
    return best


def _kv_cache_gb(context: int, params_b: float) -> float:
    """Rough KV-cache memory estimate added on top of model weights.

    Deliberately coarse: scales with context length and model size relative to a
    7B / 4k baseline (~0.5 GB). Good enough to push large-context big models out of the
    "safe" bucket without pretending to be exact.
    """
    if context <= 0 or params_b <= 0:
        return 0.0
    return (context / 4096.0) * (params_b / 7.0) * 0.5


def _classify(model: CatalogModel, free_vram_gb: float, cpu_ram_gb: float | None) -> str:
    need = model.ideal_vram_gb + _kv_cache_gb(model.context, model.params_b)
    if free_vram_gb >= need:
        return SAFE
    if free_vram_gb >= model.min_vram_gb:
        return TIGHT
    if (
        cpu_ram_gb is not None
        and model.params_b <= _OFFLOAD_MAX_PARAMS_B
        and cpu_ram_gb >= (model.disk_size_gb + 2.0)
    ):
        return OFFLOAD
    return TOO_BIG


def _entry(model: CatalogModel, fit: str, *, installed: bool) -> Recommendation:
    ref = f"{model.engine}:{model.id}"
    tasks = ", ".join(model.tasks)
    if fit == SAFE:
        speed = "moderate" if model.params_b >= 14 else "fast"
        why = f"Fits comfortably in available VRAM; good for {tasks}."
        return Recommendation(ref, model.engine, "safe", why, speed, "low", installed)
    if fit == TIGHT:
        why = (
            f"Fits but leaves little VRAM headroom (needs ~{model.ideal_vram_gb:g} GB ideal); "
            "expect reduced context or slower first token."
        )
        return Recommendation(ref, model.engine, "possible", why, "moderate", "medium", installed)
    if fit == OFFLOAD:
        why = (
            "Will not fit in VRAM and will offload to CPU RAM; runs but is noticeably slow."
        )
        return Recommendation(ref, model.engine, "possible", why, "slow", "high", installed)
    why = (
        f"Needs ~{model.min_vram_gb:g} GB VRAM minimum; not enough VRAM or RAM on this machine."
    )
    return Recommendation(ref, model.engine, "not_recommended", why, "slow", "high", installed)


def recommend(
    profile: HardwareProfile,
    *,
    task: str | None = None,
    installed: Iterable[str] = (),
    local_only: bool = True,
    engine: str | None = None,
) -> Recommendations:
    """Bucket catalog models by how well they fit ``profile`` for ``task``.

    ``installed`` are model ids already present locally (sorted first within a bucket and
    flagged). ``local_only`` drops any non-local catalog entries (all current entries are
    local). ``engine`` optionally restricts to one engine (e.g. ``"ollama"``).
    """
    installed_ids = {i.split(":", 1)[1] if i.startswith(("ollama:",)) else i for i in installed}
    free_vram = _best_free_vram_gb(profile)
    cpu_ram = profile.cpu_ram_gb

    recommended: list[Recommendation] = []
    possible: list[Recommendation] = []
    not_recommended: list[Recommendation] = []

    for model in models_for_task(task, engine=engine):
        if local_only and model.privacy != "local":
            continue
        is_installed = model.id in installed_ids
        fit = _classify(model, free_vram, cpu_ram)
        entry = _entry(model, fit, installed=is_installed)
        if entry.fit == "safe":
            recommended.append(entry)
        elif entry.fit == "possible":
            possible.append(entry)
        else:
            not_recommended.append(entry)

    # Within each bucket: installed first, then smallest (fastest) model first so the UI
    # can present fast -> balanced -> higher-quality top to bottom.
    def _sort_key(r: Recommendation) -> tuple[int, float]:
        return (0 if r.installed else 1, _params_for_ref(r.model))

    recommended.sort(key=_sort_key)
    possible.sort(key=_sort_key)
    not_recommended.sort(key=_sort_key)

    return Recommendations(recommended, possible, not_recommended)


def _params_for_ref(ref: str) -> float:
    model_id = ref.split(":", 1)[1] if ":" in ref else ref
    for model in models_for_task(None):
        if model.id == model_id:
            return model.params_b
    return 0.0
