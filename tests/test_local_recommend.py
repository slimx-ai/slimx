from __future__ import annotations

from slimx.local.hardware import GpuInfo, HardwareProfile
from slimx.local.recommend import recommend


def _profile(vram_free_gb: float | None, cpu_ram_gb: float = 64.0) -> HardwareProfile:
    gpus = []
    if vram_free_gb:
        gpus = [
            GpuInfo(
                vendor="nvidia",
                name="test-gpu",
                vram_total_gb=vram_free_gb,
                vram_free_gb=vram_free_gb,
                backends=["ollama-cuda"],
            )
        ]
    return HardwareProfile(
        os="linux",
        arch="x86_64",
        cpu_ram_gb=cpu_ram_gb,
        gpus=gpus,
        docker_gpu_available=bool(gpus),
        recommended_runtime="ollama" if gpus else "cpu",
    )


def _ids(entries) -> set[str]:
    return {e.model for e in entries}


def test_recommendations_have_three_buckets():
    recs = recommend(_profile(12.0), task="chat")
    d = recs.to_dict()
    assert set(d) == {"recommended", "possible", "not_recommended"}
    # Output entries carry the explainability fields.
    sample = (recs.recommended or recs.possible)[0]
    assert sample.fit in {"safe", "possible"}
    assert sample.why and sample.estimated_speed and sample.risk


def test_4gb_small_safe_huge_not_recommended():
    recs = recommend(_profile(4.0), task="chat")
    assert "ollama:llama3.2:1b" in _ids(recs.recommended)
    # A 70B model cannot run usefully on a 4 GB GPU.
    assert "ollama:llama3.1:70b" in _ids(recs.not_recommended)
    assert "ollama:llama3.1:70b" not in _ids(recs.recommended)


def test_8gb_recommends_3b():
    recs = recommend(_profile(8.0), task="chat")
    assert "ollama:llama3.2:3b" in _ids(recs.recommended)


def test_24gb_recommends_14b():
    recs = recommend(_profile(24.0), task="chat")
    assert "ollama:qwen2.5:14b" in _ids(recs.recommended)


def test_48gb_recommends_27b():
    recs = recommend(_profile(48.0), task="chat")
    assert "ollama:gemma2:27b" in _ids(recs.recommended)


def test_cpu_only_keeps_big_models_out_of_recommended():
    recs = recommend(_profile(None), task="chat")
    assert "ollama:gemma2:27b" not in _ids(recs.recommended)
    assert "ollama:llama3.1:70b" not in _ids(recs.recommended)
    # Tiny models can still be suggested as "possible" via CPU offload.
    possible_and_rec = _ids(recs.recommended) | _ids(recs.possible)
    assert "ollama:llama3.2:1b" in possible_and_rec


def test_safe_set_grows_monotonically_with_vram():
    small = _ids(recommend(_profile(6.0), task="chat").recommended)
    large = _ids(recommend(_profile(48.0), task="chat").recommended)
    assert small.issubset(large)


def test_task_filter_limits_catalog():
    coding = recommend(_profile(12.0), task="coding")
    all_models = _ids(coding.recommended) | _ids(coding.possible) | _ids(coding.not_recommended)
    # A summarization-only model should not appear under task=coding.
    assert "ollama:llama3.2:1b" not in all_models
    assert "ollama:qwen2.5-coder:7b" in all_models


def test_installed_models_flagged_and_sorted_first():
    recs = recommend(_profile(24.0), task="chat", installed={"qwen2.5:14b"})
    flagged = [e for e in recs.recommended if e.installed]
    assert any(e.model == "ollama:qwen2.5:14b" for e in flagged)
    # Installed entry sorts ahead of non-installed ones in its bucket.
    assert recs.recommended[0].installed is True


def test_recommended_sorted_fast_first():
    recs = recommend(_profile(48.0), task="chat")
    params = [_params(e.model) for e in recs.recommended]
    assert params == sorted(params)


def _params(ref: str) -> float:
    from slimx.local.catalog import load_catalog

    model_id = ref.split(":", 1)[1]
    return next(m.params_b for m in load_catalog() if m.id == model_id)
