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


# --------------------------------------------------------------------------
# Model pull (/api/pull). Frames below are the shapes a real Ollama 0.30.6 sent.
# --------------------------------------------------------------------------

def _pull_transport(monkeypatch, handler):
    """Route the engine's internally-built ``httpx.Client`` through a mock transport."""
    import httpx

    real_client = httpx.Client
    monkeypatch.setattr(
        "httpx.Client",
        lambda *a, **k: real_client(*a, transport=httpx.MockTransport(handler), **k),
    )


def _ndjson(*frames):
    import json

    return "".join(json.dumps(f) + "\n" for f in frames).encode()


def test_pull_streams_progress_to_success(monkeypatch):
    import httpx

    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson(
                {"status": "pulling manifest"},
                {"status": "pulling 58c187648007", "total": 986833312, "completed": 902702688},
                {"status": "verifying sha256 digest"},
                {"status": "writing manifest"},
                {"status": "success"},
            ),
        )

    _pull_transport(monkeypatch, handler)
    events = list(OllamaEngine("http://ollama.test").pull_or_prepare_model("gemma4:e2b-it-qat"))
    assert seen["body"] == {"model": "gemma4:e2b-it-qat", "stream": True}
    assert [e.status for e in events][-1] == "success"
    assert (events[1].completed, events[1].total) == (902702688, 986833312)
    # Ordinary frames never carry an error.
    assert all(e.error is None for e in events)
    assert events[0].to_dict() == {
        "status": "pulling manifest",
        "completed": None,
        "total": None,
        "error": None,
    }


def test_pull_in_stream_error_keeps_the_engine_reason(monkeypatch):
    """An unknown tag fails on HTTP 200 with an ``{"error": ...}`` line. The reason must reach
    the caller; it used to collapse into an empty-status frame with nothing attached."""
    import httpx

    def handler(request):
        return httpx.Response(
            200,
            content=_ndjson(
                {"status": "pulling manifest"},
                {"error": "pull model manifest: file does not exist"},
            ),
        )

    _pull_transport(monkeypatch, handler)
    events = list(OllamaEngine("http://ollama.test").pull_or_prepare_model("gemma4:4b"))
    assert events[-1].error == "pull model manifest: file does not exist"
    assert events[-1].status == ""
    assert events[-1].to_dict()["error"] == "pull model manifest: file does not exist"


def test_pull_refused_with_non_2xx_raises_with_the_engine_reason(monkeypatch):
    """An invalid model name is refused with HTTP 400 and a JSON reason. The exception type is
    unchanged, but its message now says why instead of only naming the status code."""
    import httpx
    import pytest

    def handler(request):
        return httpx.Response(400, json={"error": "invalid model name"})

    _pull_transport(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        list(OllamaEngine("http://ollama.test").pull_or_prepare_model("Gemma 4"))
    assert "invalid model name" in str(excinfo.value)
    assert "HTTP 400" in str(excinfo.value)


def test_pull_non_2xx_without_a_reason_still_raises(monkeypatch):
    import httpx
    import pytest

    def handler(request):
        return httpx.Response(502, content=b"<html>bad gateway</html>")

    _pull_transport(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        list(OllamaEngine("http://ollama.test").pull_or_prepare_model("gemma4"))
    assert "502" in str(excinfo.value)
