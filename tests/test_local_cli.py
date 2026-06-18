from __future__ import annotations

import json

import slimx.cli as cli
import slimx.local as local
from slimx.local.engines import LocalModel
from slimx.local.hardware import GpuInfo, HardwareProfile


def _gpu_profile() -> HardwareProfile:
    return HardwareProfile(
        os="linux",
        arch="x86_64",
        cpu_ram_gb=32.0,
        gpus=[GpuInfo(vendor="nvidia", name="RTX 3060", vram_total_gb=12.0, vram_free_gb=10.5)],
        docker_gpu_available=True,
        recommended_runtime="ollama",
    )


def test_doctor_hardware_prints_snapshot(monkeypatch, capsys):
    monkeypatch.setattr(local, "detect", _gpu_profile)
    # Avoid real network from the provider probe.
    monkeypatch.setattr(cli, "list_models", lambda name, **k: ["m1"])
    assert cli.main(["doctor", "ollama", "--hardware"]) == 0
    out = capsys.readouterr().out
    assert "Hardware" in out
    assert "RTX 3060" in out
    assert "recommended: ollama" in out


def test_doctor_json_emits_profile(monkeypatch, capsys):
    monkeypatch.setattr(local, "detect", _gpu_profile)
    assert cli.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_runtime"] == "ollama"
    assert payload["gpus"][0]["name"] == "RTX 3060"


def test_models_recommend_json(monkeypatch, capsys):
    monkeypatch.setattr(local, "detect", _gpu_profile)
    monkeypatch.setattr(cli, "_installed_local_ids", lambda: set())
    assert cli.main(["models", "recommend", "--task", "chat", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"recommended", "possible", "not_recommended"}
    assert any(e["model"] == "ollama:llama3.2:3b" for e in payload["recommended"])


def test_models_recommend_human_readable(monkeypatch, capsys):
    monkeypatch.setattr(local, "detect", _gpu_profile)
    monkeypatch.setattr(cli, "_installed_local_ids", lambda: set())
    assert cli.main(["models", "recommend"]) == 0
    out = capsys.readouterr().out
    assert "Recommended" in out and "Not recommended" in out


def test_models_local_json(monkeypatch, capsys):
    class _FakeEngine:
        def list_models(self):
            return [LocalModel(id="llama3.2:3b", engine="ollama", size_gb=2.0)]

    monkeypatch.setattr(local, "OllamaEngine", _FakeEngine)
    assert cli.main(["models", "local", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["id"] == "llama3.2:3b"


def test_models_provider_path_still_works(monkeypatch, capsys):
    # The existing `models <provider>` behaviour must be unchanged.
    monkeypatch.setattr(cli, "list_models", lambda name, **k: ["a", "b"])
    assert cli.main(["models", "ollama"]) == 0
    assert capsys.readouterr().out.splitlines() == ["a", "b"]
