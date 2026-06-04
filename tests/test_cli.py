from __future__ import annotations

import slimx.cli as cli
from slimx.providers.ollama import OllamaProvider
from slimx.providers.openai import OpenAIProvider


class _Resp:
    def __init__(self, data, status_code=200, text=""):
        self._data = data
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._data


def _client_returning(resp):
    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def get(self, url, *a, **k):
            return resp

    return _Client


# --------------------------------------------------------------------------
# Provider model discovery
# --------------------------------------------------------------------------

def test_openai_list_models_parses_data_ids(monkeypatch):
    resp = _Resp({"data": [{"id": "gpt-4.1-nano"}, {"id": "gpt-4.1-mini"}, {}]})
    monkeypatch.setattr("slimx.providers.openai.httpx.Client", _client_returning(resp))
    models = OpenAIProvider(api_key="x").list_models()
    assert models == ["gpt-4.1-nano", "gpt-4.1-mini"]


def test_ollama_list_models_parses_tag_names(monkeypatch):
    resp = _Resp({"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}]})
    monkeypatch.setattr("slimx.providers.ollama.httpx.Client", _client_returning(resp))
    models = OllamaProvider("http://ollama.local").list_models()
    assert models == ["llama3.2:3b", "qwen2.5:7b"]


def test_discovery_list_models_dispatches(monkeypatch):
    resp = _Resp({"models": [{"name": "llama3.2:3b"}]})
    monkeypatch.setattr("slimx.providers.ollama.httpx.Client", _client_returning(resp))
    from slimx import list_models

    assert list_models("ollama") == ["llama3.2:3b"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_version(capsys):
    assert cli.main(["version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out  # the version string


def test_cli_providers_lists_capabilities(capsys):
    assert cli.main(["providers"]) == 0
    out = capsys.readouterr().out
    for name in ("openai", "google", "anthropic", "ollama", "oai"):
        assert name in out
    assert "openai-compatible" in out  # oai is flagged non-native


def test_cli_doctor_runs_without_network(monkeypatch, capsys):
    # Avoid real network: stub discovery used by doctor.
    monkeypatch.setattr(cli, "list_models", lambda name, **k: ["m1", "m2", "m3"])
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SLIMX_OAI_BASE_URL", "http://localhost:8000/v1")

    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "SlimX" in out
    assert "openai" in out and "key OPENAI_API_KEY: found" in out
    assert "anthropic" in out and "key missing" in out
    assert "reachable · 3 model(s)" in out  # ollama + oai probed via stubbed list_models


def test_cli_models_command(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_models", lambda name, **k: ["a", "b"])
    assert cli.main(["models", "ollama"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["a", "b"]


def test_cli_models_command_handles_errors(monkeypatch, capsys):
    def boom(name, **k):
        raise RuntimeError("no server")

    monkeypatch.setattr(cli, "list_models", boom)
    assert cli.main(["models", "ollama"]) == 1
