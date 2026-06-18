"""SlimX command-line tools.

    slimx doctor [provider] [--probe] [--hardware]
                                        diagnose keys, base URLs, reachable servers
    slimx models <provider>             list models a provider exposes
    slimx models recommend [--task T]   recommend local models that fit this hardware
    slimx models local                  list locally installed models (per engine)
    slimx providers                     list registered providers + capabilities
    slimx version                       print the SlimX version

`doctor` is safe to run with nothing configured: it reports what's missing and,
for local servers (ollama, oai), probes reachability and lists models. It only
makes network calls to cloud providers when you pass --probe. `--hardware` adds a
local CPU/GPU snapshot (no network) and a recommended runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import List, Optional

from . import __version__
from .discovery import list_models
from .providers import describe_provider, list_providers

# Env vars that carry credentials for each built-in provider.
PROVIDER_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "ollama": [],
    "oai": ["SLIMX_OAI_API_KEY", "OAI_API_KEY"],
}
LOCAL_PROVIDERS = {"ollama", "oai"}


def _first_present(keys: List[str]) -> Optional[str]:
    return next((k for k in keys if os.environ.get(k)), None)


def _oai_base_url() -> Optional[str]:
    return os.environ.get("SLIMX_OAI_BASE_URL") or os.environ.get("OAI_BASE_URL")


def _report_local(name: str) -> None:
    if name == "oai" and not _oai_base_url():
        print(f"  {name:<10} not configured — set SLIMX_OAI_BASE_URL or OAI_BASE_URL")
        return
    try:
        models = list_models(name)
    except Exception as e:  # unreachable server, unset base url, etc.
        print(f"  {name:<10} unreachable ({type(e).__name__})")
        return
    preview = ", ".join(m for m in models[:4] if m)
    if len(models) > 4:
        preview += ", …"
    suffix = f" — {preview}" if preview else ""
    print(f"  {name:<10} reachable · {len(models)} model(s){suffix}")


def _report_cloud(name: str, probe: bool) -> None:
    keys = PROVIDER_KEYS.get(name, [])
    found = _first_present(keys)
    if found:
        line = f"  {name:<10} key {found}: found"
        if probe:
            try:
                models = list_models(name)
                line += f" · {len(models)} model(s)"
            except Exception as e:
                line += f" · probe failed ({type(e).__name__})"
    else:
        line = f"  {name:<10} key missing — set {' or '.join(keys)}"
    print(line)


def doctor(
    provider: Optional[str] = None,
    *,
    probe: bool = False,
    hardware: bool = False,
    as_json: bool = False,
) -> int:
    if as_json:
        # Machine-readable hardware snapshot only (scripting/integration path).
        from .local import detect as detect_hardware

        print(json.dumps(detect_hardware().to_dict(), indent=2))
        return 0

    print(f"SlimX {__version__}  ·  Python {platform.python_version()}")
    print()
    targets = [provider] if provider else list_providers()
    for name in targets:
        if name not in PROVIDER_KEYS:
            print(f"  {name:<10} (third-party provider)")
        elif name in LOCAL_PROVIDERS:
            _report_local(name)
        else:
            _report_cloud(name, probe)
    if hardware:
        print()
        _print_hardware()
    return 0


def _print_hardware() -> None:
    from .local import detect as detect_hardware

    profile = detect_hardware()
    ram = f"{profile.cpu_ram_gb} GB" if profile.cpu_ram_gb is not None else "unknown"
    print("Hardware")
    print(f"  os         {profile.os} ({profile.arch}) · cpu_ram {ram}")
    if not profile.gpus:
        print("  gpu        none detected (CPU-only)")
    for gpu in profile.gpus:
        total = f"{gpu.vram_total_gb} GB" if gpu.vram_total_gb is not None else "?"
        free = f"{gpu.vram_free_gb} GB free" if gpu.vram_free_gb is not None else "free ?"
        driver = f" · driver {gpu.driver}" if gpu.driver else ""
        print(f"  gpu        {gpu.vendor} {gpu.name} · {total} total / {free}{driver}")
    print(f"  docker gpu {'available' if profile.docker_gpu_available else 'no'}")
    print(f"  runtime    recommended: {profile.recommended_runtime}")


def models_cmd(provider: str) -> int:
    try:
        models = list_models(provider)
    except Exception as e:
        print(f"error: could not list models for {provider!r}: {e}", file=sys.stderr)
        return 1
    if not models:
        print(f"(no models reported by {provider})")
        return 0
    for m in models:
        print(m)
    return 0


def _installed_local_ids() -> set[str]:
    """Best-effort set of locally installed model ids (Ollama today)."""
    try:
        from .local import OllamaEngine

        return {m.id for m in OllamaEngine().list_models()}
    except Exception:
        return set()


def recommend_cmd(task: Optional[str], as_json: bool) -> int:
    from .local import detect as detect_hardware, recommend

    recs = recommend(detect_hardware(), task=task, installed=_installed_local_ids())
    if as_json:
        print(json.dumps(recs.to_dict(), indent=2))
        return 0

    def _bucket(title: str, items: list) -> None:
        print(title)
        if not items:
            print("  (none)")
        for r in items:
            mark = " *installed" if r.installed else ""
            print(f"  {r.fit:<16} {r.model}  [{r.estimated_speed}, risk {r.risk}]{mark}")
            print(f"  {'':<16} {r.why}")

    label = f" for task '{task}'" if task else ""
    print(f"Local model recommendations{label}:\n")
    _bucket("Recommended", recs.recommended)
    print()
    _bucket("Possible", recs.possible)
    print()
    _bucket("Not recommended", recs.not_recommended)
    return 0


def local_models_cmd(as_json: bool) -> int:
    from .local import OllamaEngine

    models = []
    try:
        models = OllamaEngine().list_models()
    except Exception as e:
        print(f"error: could not list local models: {e}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps([m.to_dict() for m in models], indent=2))
        return 0
    if not models:
        print("(no local models installed)")
        return 0
    for m in models:
        size = f"{m.size_gb} GB" if m.size_gb is not None else "?"
        print(f"  {m.engine}:{m.id}  ({size})")
    return 0


def providers_cmd() -> int:
    for name in list_providers():
        try:
            caps = describe_provider(name)
            flags = [k for k in ("tools", "structured_output", "streaming") if caps.get(k)]
            kind = "native" if caps.get("native") else "openai-compatible"
            print(f"  {name:<10} {kind:<18} {', '.join(flags) or 'chat only'}")
        except Exception:
            print(f"  {name:<10} (capabilities unavailable)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="slimx", description="SlimX command-line tools")
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("doctor", help="diagnose providers, keys, and reachable servers")
    d.add_argument("provider", nargs="?", help="limit to a single provider")
    d.add_argument("--probe", action="store_true", help="also probe cloud providers (network calls)")
    d.add_argument("--hardware", action="store_true", help="add a local CPU/GPU snapshot (no network)")
    d.add_argument("--json", action="store_true", help="emit the hardware snapshot as JSON")

    m = sub.add_parser(
        "models",
        help="list models a provider exposes; 'recommend'/'local' for local-model helpers",
    )
    m.add_argument(
        "provider",
        help="provider name (e.g. ollama, oai), or 'recommend' / 'local'",
    )
    m.add_argument("--task", help="task filter for 'recommend' (chat, rag, coding, summarization)")
    m.add_argument("--json", action="store_true", help="emit JSON output")

    sub.add_parser("providers", help="list registered providers and capabilities")
    sub.add_parser("version", help="print the SlimX version")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return doctor(args.provider, probe=args.probe, hardware=args.hardware, as_json=args.json)
    if args.command == "models":
        if args.provider == "recommend":
            return recommend_cmd(args.task, args.json)
        if args.provider == "local":
            return local_models_cmd(args.json)
        return models_cmd(args.provider)
    if args.command == "providers":
        return providers_cmd()
    if args.command == "version":
        print(__version__)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
