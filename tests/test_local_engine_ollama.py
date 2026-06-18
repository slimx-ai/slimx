from __future__ import annotations

from slimx.local.engines import OllamaEngine, RunningModel


class _Resp:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


def _fake_get(routes):
    def get(url, *a, **k):
        for suffix, data in routes.items():
            if url.endswith(suffix):
                return _Resp(data)
        return _Resp({}, status_code=404)

    return get


# --------------------------------------------------------------------------
# Runtime placement (the GPU vs CPU signal)
# --------------------------------------------------------------------------

def test_running_model_placement_gpu():
    assert RunningModel("m", size_bytes=100, size_vram_bytes=100).placement == "gpu"


def test_running_model_placement_partial():
    rm = RunningModel("m", size_bytes=100, size_vram_bytes=40)
    assert rm.placement == "partial"
    assert rm.gpu_fraction == 0.4


def test_running_model_placement_cpu():
    assert RunningModel("m", size_bytes=100, size_vram_bytes=0).placement == "cpu"


def test_running_model_placement_unknown():
    assert RunningModel("m", size_bytes=None, size_vram_bytes=None).placement == "unknown"
    assert RunningModel("m").gpu_fraction is None


# --------------------------------------------------------------------------
# Engine HTTP surfaces (mocked httpx)
# --------------------------------------------------------------------------

def test_list_models_parses_tags(monkeypatch):
    routes = {"/api/tags": {"models": [{"name": "llama3.2:3b", "size": 2 * 1024**3}]}}
    monkeypatch.setattr("httpx.get", _fake_get(routes))
    models = OllamaEngine("http://ollama.test").list_models()
    assert len(models) == 1
    assert models[0].id == "llama3.2:3b"
    assert models[0].engine == "ollama"
    assert models[0].size_gb == 2.0


def test_runtime_status_reports_placement(monkeypatch):
    routes = {
        "/api/ps": {
            "models": [
                {"name": "on-gpu", "size": 100, "size_vram": 100},
                {"name": "split", "size": 100, "size_vram": 40},
                {"name": "on-cpu", "size": 100, "size_vram": 0},
            ]
        }
    }
    monkeypatch.setattr("httpx.get", _fake_get(routes))
    status = OllamaEngine("http://ollama.test").runtime_status()
    placements = {m.name: m.placement for m in status.running}
    assert placements == {"on-gpu": "gpu", "split": "partial", "on-cpu": "cpu"}


def test_detect_reports_reachable(monkeypatch):
    routes = {"/api/version": {"version": "0.5.0"}}
    monkeypatch.setattr("httpx.get", _fake_get(routes))
    status = OllamaEngine("http://ollama.test").detect()
    assert status.reachable is True
    assert status.kind == "ollama"
    assert status.base_url == "http://ollama.test"


def test_health_handles_unreachable(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("httpx.get", boom)
    health = OllamaEngine("http://ollama.test").health()
    assert health.reachable is False
    assert "ConnectError" in health.detail


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    assert OllamaEngine().base_url == "http://host.docker.internal:11434"
