from __future__ import annotations

import slimx.local.hardware as hw

_NVIDIA_CSV = "NVIDIA GeForce RTX 3060, 12288, 10752, 570.86.10\n"


def test_detect_nvidia_parses_csv(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: _NVIDIA_CSV if cmd[0] == "nvidia-smi" else None)
    gpus = hw.detect_nvidia()
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.vendor == "nvidia"
    assert gpu.name == "NVIDIA GeForce RTX 3060"
    assert gpu.vram_total_gb == 12.0
    assert gpu.vram_free_gb == 10.5
    assert gpu.driver == "570.86.10"
    assert "ollama-cuda" in gpu.backends


def test_detect_nvidia_absent_returns_empty(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: None)
    assert hw.detect_nvidia() == []


def test_detect_nvidia_ignores_garbage_lines(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: "garbage-without-commas\n")
    assert hw.detect_nvidia() == []


def test_detect_no_gpu_recommends_cpu(monkeypatch):
    monkeypatch.setattr(hw, "detect_nvidia", lambda: [])
    monkeypatch.setattr(hw, "detect_amd", lambda: [])
    monkeypatch.setattr(hw, "detect_apple", lambda: [])
    monkeypatch.setattr(hw, "detect_docker_gpu", lambda: False)
    monkeypatch.setattr(hw, "detect_cpu_ram_gb", lambda: 64.0)
    profile = hw.detect()
    assert profile.gpus == []
    assert profile.recommended_runtime == "cpu"
    assert profile.cpu_ram_gb == 64.0
    assert profile.docker_gpu_available is False


def test_detect_with_gpu_recommends_ollama(monkeypatch):
    gpu = hw.GpuInfo(vendor="nvidia", name="RTX 3060", vram_total_gb=12.0, vram_free_gb=10.5)
    monkeypatch.setattr(hw, "detect_nvidia", lambda: [gpu])
    monkeypatch.setattr(hw, "detect_amd", lambda: [])
    monkeypatch.setattr(hw, "detect_apple", lambda: [])
    monkeypatch.setattr(hw, "detect_docker_gpu", lambda: True)
    monkeypatch.setattr(hw, "detect_cpu_ram_gb", lambda: 32.0)
    profile = hw.detect()
    assert profile.recommended_runtime == "ollama"
    assert profile.docker_gpu_available is True
    assert profile.to_dict()["gpus"][0]["name"] == "RTX 3060"


def test_detect_never_raises_when_probe_explodes(monkeypatch):
    def boom():
        raise RuntimeError("vendor tool crashed")

    monkeypatch.setattr(hw, "detect_nvidia", boom)
    monkeypatch.setattr(hw, "detect_amd", lambda: [])
    monkeypatch.setattr(hw, "detect_apple", lambda: [])
    # Should swallow the exception and still produce a profile.
    profile = hw.detect()
    assert profile.gpus == []
