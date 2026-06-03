"""Parallel (ensemble) execution: fan one SlimX call out to multiple models.

This sits ABOVE the Model/Client layer and composes them — it never reaches into
provider internals. v1 supports two modes:

- ``all``  — run every model concurrently, return every result (and every error).
- ``race`` — return the first successful result; the rest are abandoned.

Inspectability is the contract: failures are surfaced in ``errors`` (never
swallowed), every attempt keeps its full ``Result`` (including ``raw``), and
``trace`` records timings and counts. Tools, streaming, and the judge/consensus
modes are intentionally out of scope for v1.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .high.api import Model, llm
from .types import Result

_MODES = ("all", "race")


@dataclass
class ParallelItem:
    """One model's attempt within a parallel call."""

    provider: str
    model: str
    result: Optional[Result] = None
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.result is not None


@dataclass
class ParallelResult:
    """The combined outcome of a parallel call.

    - ``results``: every attempt, in input order (successes and failures).
    - ``errors``: the failed subset of ``results`` (convenience view).
    - ``winner``: the chosen attempt for single-answer modes (``race``).
    - ``text``: the winner's text when a mode yields one answer, else ``None``.
    - ``trace``: mode, model list, per-call elapsed time, and ok/error counts.
    """

    text: Optional[str]
    results: List[ParallelItem] = field(default_factory=list)
    errors: List[ParallelItem] = field(default_factory=list)
    winner: Optional[ParallelItem] = None
    trace: Dict[str, Any] = field(default_factory=dict)


class Parallel:
    def __init__(
        self,
        models: Sequence[str],
        *,
        mode: str = "all",
        max_workers: Optional[int] = None,
        **model_kwargs: Any,
    ):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self._model_strings: List[str] = list(models)
        if not self._model_strings:
            raise ValueError("parallel() requires at least one model")
        # Resolve providers once, up front, so missing keys / unknown providers
        # fail fast rather than inside a worker thread.
        self._models: List[tuple[str, Model]] = [
            (m, llm(m, **model_kwargs)) for m in self._model_strings
        ]
        self._max_workers = max_workers or len(self._models)

    def __call__(self, prompt: str, **overrides: Any) -> ParallelResult:
        if self.mode == "race":
            return self._race(prompt, overrides)
        return self._all(prompt, overrides)

    def _invoke(
        self, model_string: str, model: Model, prompt: str, overrides: Dict[str, Any]
    ) -> ParallelItem:
        provider = model_string.split(":", 1)[0] if ":" in model_string else "openai"
        start = time.perf_counter()
        try:
            res = model(prompt, **overrides)
            return ParallelItem(
                provider=provider,
                model=model_string,
                result=res,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
        except Exception as e:
            return ParallelItem(
                provider=provider,
                model=model_string,
                error=f"{type(e).__name__}: {e}",
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )

    def _all(self, prompt: str, overrides: Dict[str, Any]) -> ParallelResult:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = [
                ex.submit(self._invoke, ms, m, prompt, overrides) for ms, m in self._models
            ]
            items = [f.result() for f in futures]  # preserve input order
        errors = [it for it in items if not it.ok]
        return ParallelResult(
            text=None,
            results=items,
            errors=errors,
            winner=None,
            trace=self._trace("all", items, started),
        )

    def _race(self, prompt: str, overrides: Dict[str, Any]) -> ParallelResult:
        started = time.perf_counter()
        items: List[ParallelItem] = []
        winner: Optional[ParallelItem] = None
        # Manage the executor manually so a win returns immediately: queued calls
        # are cancelled and already-running calls are abandoned (not awaited).
        ex = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            futures = {
                ex.submit(self._invoke, ms, m, prompt, overrides): ms for ms, m in self._models
            }
            for fut in as_completed(futures):
                item = fut.result()
                items.append(item)
                if item.ok:
                    winner = item
                    break
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        errors = [it for it in items if not it.ok]
        text = winner.result.text if winner is not None and winner.result is not None else None
        return ParallelResult(
            text=text,
            results=items,
            errors=errors,
            winner=winner,
            trace=self._trace("race", items, started),
        )

    def _trace(self, mode: str, items: List[ParallelItem], started: float) -> Dict[str, Any]:
        return {
            "mode": mode,
            "models": self._model_strings,
            "model_count": len(self._model_strings),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "ok_count": sum(1 for it in items if it.ok),
            "error_count": sum(1 for it in items if not it.ok),
        }


def parallel(
    models: Sequence[str],
    *,
    mode: str = "all",
    max_workers: Optional[int] = None,
    **model_kwargs: Any,
) -> Parallel:
    """Fan a single prompt out to multiple models concurrently.

    Example:
        >>> m = parallel(["google:gemini-3.5-flash", "openai:gpt-4.1-nano"])
        >>> res = m("Explain SlimX in one paragraph.")
        >>> for item in res.results:
        ...     print(item.model, item.result.text if item.ok else item.error)

    Modes:
        - ``"all"``  : return every model's result; ``text`` is ``None``.
        - ``"race"`` : return the first success in ``winner``/``text``.

    Extra keyword arguments (e.g. ``temperature``, ``timeout``, ``retries``) are
    forwarded to each underlying model.
    """
    return Parallel(models, mode=mode, max_workers=max_workers, **model_kwargs)
