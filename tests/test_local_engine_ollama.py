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

    # A test may route several responses in turn, so always wrap the real class, never a wrapper.
    real_client = getattr(httpx.Client, "_real_client", httpx.Client)

    def client(*a, **k):
        return real_client(*a, transport=httpx.MockTransport(handler), **k)

    client._real_client = real_client  # type: ignore[attr-defined]
    monkeypatch.setattr("httpx.Client", client)


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
    assert events[0].to_dict() == {"status": "pulling manifest", "completed": None, "total": None}


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


# -- review of the pull-failure change: compatibility, bad bodies, bounds, resource closure -------


class _CountingStream:
    """A response body that records how much of it was read and whether it was closed."""

    def __init__(self, chunks, *, fail_after=None):
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self.bytes_read = 0
        self.closed = False

    def __iter__(self):
        import httpx

        for index, chunk in enumerate(self._chunks):
            if self._fail_after is not None and index >= self._fail_after:
                raise httpx.ReadError("connection reset by peer")
            self.bytes_read += len(chunk)
            yield chunk

    def close(self):
        self.closed = True


def _stream_response(monkeypatch, status, stream):
    import httpx

    class _Body(httpx.SyncByteStream):
        def __iter__(self):
            return iter(stream)

        def close(self):
            stream.close()

    _pull_transport(monkeypatch, lambda request: httpx.Response(status, stream=_Body()))


def test_an_ordinary_progress_frame_serializes_exactly_as_before():
    """``to_dict`` is what callers forward to their own clients (ControlRoom streams it verbatim),
    so an ordinary frame must not grow an ``error`` key; only a failure frame carries one."""
    from slimx.local.engines import PullEvent

    assert PullEvent("downloading", 5, 10).to_dict() == {
        "status": "downloading",
        "completed": 5,
        "total": 10,
    }
    assert PullEvent(status="success").to_dict() == {
        "status": "success",
        "completed": None,
        "total": None,
    }
    assert PullEvent("downloading", 5, 10) == PullEvent(status="downloading", completed=5, total=10)
    assert PullEvent(status="", error="boom").to_dict() == {
        "status": "",
        "completed": None,
        "total": None,
        "error": "boom",
    }


def test_progress_frames_before_an_in_stream_error_are_unchanged(monkeypatch):
    import httpx

    def handler(request):
        return httpx.Response(
            200,
            content=_ndjson(
                {"status": "pulling manifest"},
                {"status": "pulling abc", "total": 10, "completed": 5},
                {"error": "unexpected EOF"},
            ),
        )

    _pull_transport(monkeypatch, handler)
    events = list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
    assert [e.to_dict() for e in events[:2]] == [
        {"status": "pulling manifest", "completed": None, "total": None},
        {"status": "pulling abc", "completed": 5, "total": 10},
    ]
    assert events[2].error == "unexpected EOF" and all(e.error is None for e in events[:2])


def test_a_non_2xx_body_that_carries_no_usable_reason_falls_back_to_the_stock_error(monkeypatch):
    import httpx
    import pytest

    bodies = [
        b"",  # empty
        b'{"error": "trunc',  # malformed JSON
        b"\xff\xfe\x00not text",  # not decodable text
        b'["not", "an", "object"]',  # JSON, but not an object
        b'{"message": "no error key"}',
        b'{"error": ""}',
        b'{"error": null}',
    ]
    for body in bodies:
        _pull_transport(monkeypatch, lambda request, body=body: httpx.Response(500, content=body))
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
        message = str(excinfo.value)
        assert "500" in message and "refused the pull" not in message, (body, message)
        assert excinfo.value.response.status_code == 500


def test_a_structured_reason_is_carried_as_compact_json(monkeypatch):
    import httpx
    import pytest

    _pull_transport(
        monkeypatch,
        lambda request: httpx.Response(409, json={"error": {"code": "busy", "detail": "try later"}}),
    )
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
    assert '"code": "busy"' in str(excinfo.value)


def test_an_oversized_failure_body_is_not_read_into_memory(monkeypatch):
    """A proxy can answer a pull with a body of any size. Only a bounded prefix is read; a body
    that does not fit is not a reason, so the stock error is raised."""
    import httpx
    import pytest

    from slimx.local.engines import ollama as engine_module

    chunk = b"x" * 8192
    stream = _CountingStream([b'{"error": "'] + [chunk] * 640 + [b'"}'])  # about 5 MB
    _stream_response(monkeypatch, 502, stream)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
    assert "refused the pull" not in str(excinfo.value)
    assert stream.bytes_read <= engine_module._MAX_ERROR_BODY_BYTES + len(chunk)
    assert stream.closed


def test_a_long_reason_is_bounded_on_both_failure_paths(monkeypatch):
    import httpx
    import pytest

    from slimx.local.engines import ollama as engine_module

    long_reason = "manifest error " * 1000  # 15,000 characters, well inside the body bound
    limit = engine_module._MAX_ERROR_REASON_CHARS

    _pull_transport(monkeypatch, lambda request: httpx.Response(400, json={"error": long_reason}))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
    assert long_reason[:200] in str(excinfo.value)
    assert len(str(excinfo.value)) < limit + 200

    _pull_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=_ndjson({"error": long_reason})),
    )
    (event,) = list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
    assert event.error is not None and len(event.error) <= limit + 1
    assert event.error.startswith(long_reason[:200]) and event.error.endswith("…")


def test_the_response_is_closed_on_success_failure_and_interruption(monkeypatch):
    from typing import Generator, cast

    import httpx
    import pytest

    from slimx.local.engines import PullEvent

    engine = OllamaEngine("http://ollama.test")
    frames = [_ndjson({"status": "pulling manifest"}), _ndjson({"status": "success"})]

    finished = _CountingStream(frames)
    _stream_response(monkeypatch, 200, finished)
    assert [e.status for e in engine.pull_or_prepare_model("m")] == ["pulling manifest", "success"]
    assert finished.closed

    refused = _CountingStream([b'{"error": "invalid model name"}'])
    _stream_response(monkeypatch, 400, refused)
    with pytest.raises(httpx.HTTPStatusError, match="invalid model name"):
        list(engine.pull_or_prepare_model("Bad Name"))
    assert refused.closed

    failed_mid_stream = _CountingStream([_ndjson({"error": "unexpected EOF"})])
    _stream_response(monkeypatch, 200, failed_mid_stream)
    assert [e.error for e in engine.pull_or_prepare_model("m")] == ["unexpected EOF"]
    assert failed_mid_stream.closed

    dropped = _CountingStream(frames, fail_after=1)  # the transport dies after the first frame
    _stream_response(monkeypatch, 200, dropped)
    with pytest.raises(httpx.ReadError):
        list(engine.pull_or_prepare_model("m"))
    assert dropped.closed

    abandoned = _CountingStream(frames)  # the caller stops listening (a browser went away)
    _stream_response(monkeypatch, 200, abandoned)
    # The engine method is a generator; its declared type is the narrower Iterator.
    pull = cast("Generator[PullEvent, None, None]", engine.pull_or_prepare_model("m"))
    assert next(pull).status == "pulling manifest"
    assert not abandoned.closed
    pull.close()
    assert abandoned.closed


def test_any_non_2xx_is_a_failure_not_an_empty_pull(monkeypatch):
    """Peer review of this PR: narrowing ``raise_for_status()`` to ``>= 400`` made a redirect or an
    informational response yield nothing and raise nothing — a pull that looks like it completed.
    This client does not follow redirects, so an ingress that 308s http→https is exactly that."""
    import httpx
    import pytest

    for status in (100, 199, 301, 302, 304, 307, 308, 400, 404, 500, 502):
        _pull_transport(monkeypatch, lambda request, s=status: httpx.Response(s, json={"x": 1}))
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
        assert excinfo.value.response.status_code == status

    for status in (200, 201, 204):
        _pull_transport(monkeypatch, lambda request, s=status: httpx.Response(s, content=b""))
        assert list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m")) == []


def test_a_failure_body_the_parser_cannot_read_still_raises_the_stock_error(monkeypatch):
    """Peer review of this PR: ``json.loads`` raises RecursionError, not ValueError, on deeply
    nested input, so a 400 whose body is 60,000 open brackets escaped as RecursionError."""
    import httpx
    import pytest

    _pull_transport(monkeypatch, lambda request: httpx.Response(400, content=b"[" * 60_000))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
    assert "400" in str(excinfo.value) and "refused the pull" not in str(excinfo.value)


def test_a_reason_is_reported_even_when_python_would_call_it_falsy(monkeypatch):
    """``None`` and ``""`` mean "no reason"; ``0`` or ``[]`` are something the engine said."""
    import httpx

    from slimx.local.engines import ollama as engine_module

    for reported, expected in [
        (None, None),
        ("", None),
        ("   ", None),
        (0, "0"),
        (False, "false"),
        ([], "[]"),
        ("  padded  ", "padded"),
    ]:
        assert engine_module._reason_text(reported) == expected

    # A lone surrogate survives json.loads but cannot be encoded as UTF-8 by a caller's framework.
    _pull_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=b'{"error": "bad \\ud800 tag"}\n'),
    )
    (event,) = list(OllamaEngine("http://ollama.test").pull_or_prepare_model("m"))
    assert event.error is not None
    assert event.error.encode("utf-8")  # would raise UnicodeEncodeError on a lone surrogate
    assert "\ud800" not in event.error
