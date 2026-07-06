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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .high.api import Model, _parse_model, llm
from .types import Result

_MODES = ("all", "race", "compare", "judge")


@dataclass
class ParallelItem:
    """One model's attempt within a parallel call."""

    provider: str
    model: str
    result: Optional[Result] = None
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None
    # True when the call never started because a ``cancel_event`` was set (additive;
    # ``ok`` stays False and ``error`` carries the human-readable reason).
    cancelled: bool = False

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

    @property
    def candidates(self) -> List[ParallelItem]:
        """Alias for ``results`` — reads naturally in judge/compare modes."""
        return self.results


class Parallel:
    def __init__(
        self,
        models: Sequence[str],
        *,
        mode: str = "all",
        judge: Optional[str] = None,
        max_workers: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
        **model_kwargs: Any,
    ):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if mode == "judge" and not judge:
            raise ValueError("mode='judge' requires a judge model, e.g. judge='openai:gpt-4.1-mini'")
        self.mode = mode
        self._model_strings: List[str] = list(models)
        if not self._model_strings:
            raise ValueError("parallel() requires at least one model")
        # Resolve providers once, up front, so missing keys / unknown providers
        # fail fast rather than inside a worker thread.
        self._models: List[tuple[str, Model]] = [
            (m, llm(m, **model_kwargs)) for m in self._model_strings
        ]
        self._judge_string = judge
        self._judge_model: Optional[Model] = llm(judge, **model_kwargs) if judge else None
        self._max_workers = max_workers or len(self._models)
        # Cooperative cancellation: checked before each model call starts (and before the
        # judge synthesis). An in-flight provider request is NOT aborted mid-HTTP — the
        # guarantee is "no NEW work begins after the event is set".
        self._cancel_event = cancel_event

    def __call__(self, prompt: str, **overrides: Any) -> ParallelResult:
        # Per-call cancel_event overrides the constructor's; popped so it never reaches the
        # underlying model call as a generation parameter.
        cancel_event = overrides.pop("cancel_event", None) or self._cancel_event
        if self.mode == "race":
            return self._race(prompt, overrides, cancel_event)
        if self.mode == "compare":
            return self._compare(prompt, overrides, cancel_event)
        if self.mode == "judge":
            return self._judge(prompt, overrides, cancel_event)
        return self._all(prompt, overrides, cancel_event)

    def _invoke(
        self,
        model_string: str,
        model: Model,
        prompt: str,
        overrides: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> ParallelItem:
        provider = _parse_model(model_string)[0]
        if cancel_event is not None and cancel_event.is_set():
            return ParallelItem(
                provider=provider,
                model=model_string,
                error="Cancelled before this model call started (cancel_event set)",
                cancelled=True,
            )
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

    def _gather(
        self,
        prompt: str,
        overrides: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> List[ParallelItem]:
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = [
                ex.submit(self._invoke, ms, m, prompt, overrides, cancel_event)
                for ms, m in self._models
            ]
            return [f.result() for f in futures]  # preserve input order

    def _all(
        self,
        prompt: str,
        overrides: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> ParallelResult:
        started = time.perf_counter()
        items = self._gather(prompt, overrides, cancel_event)
        return ParallelResult(
            text=None,
            results=items,
            errors=[it for it in items if not it.ok],
            winner=None,
            trace=self._trace("all", items, started),
        )

    def _compare(
        self,
        prompt: str,
        overrides: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> ParallelResult:
        started = time.perf_counter()
        items = self._gather(prompt, overrides, cancel_event)
        blocks = []
        for it in items:
            body = it.result.text if it.ok and it.result else f"[error] {it.error}"
            blocks.append(f"### {it.model}\n{body}")
        return ParallelResult(
            text="\n\n".join(blocks),  # a readable side-by-side comparison
            results=items,
            errors=[it for it in items if not it.ok],
            winner=None,
            trace=self._trace("compare", items, started),
        )

    def _judge(
        self,
        prompt: str,
        overrides: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> ParallelResult:
        started = time.perf_counter()
        candidates = self._gather(prompt, overrides, cancel_event)
        ok = [it for it in candidates if it.ok and it.result]
        trace = self._trace("judge", candidates, started)
        trace["judge"] = self._judge_string

        # Cancellation between the fan-out and the synthesis: return the candidates that
        # finished, but never start the judge call after the event is set.
        if cancel_event is not None and cancel_event.is_set():
            trace["judge_cancelled"] = True
            return ParallelResult(
                text=None,
                results=candidates,
                errors=[it for it in candidates if not it.ok],
                winner=None,
                trace=trace,
            )

        if not ok or self._judge_model is None:
            return ParallelResult(
                text=None,
                results=candidates,
                errors=[it for it in candidates if not it.ok],
                winner=None,
                trace=trace,
            )

        listing = "\n\n".join(
            f"[{i + 1}] (from {it.model})\n{it.result.text}" for i, it in enumerate(ok)  # type: ignore[union-attr]
        )
        judge_prompt = (
            f"You are judging candidate answers to the request below.\n\n"
            f"REQUEST:\n{prompt}\n\nCANDIDATES:\n{listing}\n\n"
            "Choose the single best answer, or synthesize a better one by combining their "
            "strengths. Reply with ONLY the final answer — no commentary, no numbering."
        )
        jstart = time.perf_counter()
        judge_str = self._judge_string or ""
        jprovider = _parse_model(judge_str)[0]
        try:
            jres = self._judge_model(judge_prompt)
            winner = ParallelItem(
                provider=jprovider,
                model=judge_str,
                result=jres,
                elapsed_ms=int((time.perf_counter() - jstart) * 1000),
            )
            text = jres.text
        except Exception as e:
            winner = ParallelItem(
                provider=jprovider,
                model=judge_str,
                error=f"{type(e).__name__}: {e}",
                elapsed_ms=int((time.perf_counter() - jstart) * 1000),
            )
            text = None

        return ParallelResult(
            text=text,
            results=candidates,
            errors=[it for it in candidates if not it.ok],
            winner=winner,
            trace=trace,
        )

    def _race(
        self,
        prompt: str,
        overrides: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> ParallelResult:
        started = time.perf_counter()
        items: List[ParallelItem] = []
        winner: Optional[ParallelItem] = None
        # Manage the executor manually so a win returns immediately: queued calls
        # are cancelled and already-running calls are abandoned (not awaited).
        ex = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            futures = {
                ex.submit(self._invoke, ms, m, prompt, overrides, cancel_event): ms
                for ms, m in self._models
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
    judge: Optional[str] = None,
    max_workers: Optional[int] = None,
    cancel_event: Optional[threading.Event] = None,
    **model_kwargs: Any,
) -> Parallel:
    """Fan a single prompt out to multiple models concurrently.

    Modes:
        - ``"all"``     : return every model's result; ``text`` is ``None``.
        - ``"race"``    : return the first success in ``winner``/``text``.
        - ``"compare"`` : run all; ``text`` is a readable side-by-side of every answer.
        - ``"judge"``   : run all candidates, then a ``judge`` model picks or synthesizes
          the best answer (``text``/``winner``); candidates stay in ``results``.

    Example:
        >>> m = parallel(["google:gemini-3.5-flash", "openai:gpt-4.1-nano"])
        >>> res = m("Explain SlimX in one paragraph.")
        >>> for item in res.results:
        ...     print(item.model, item.result.text if item.ok else item.error)

    Extra keyword arguments (e.g. ``temperature``, ``timeout``, ``retries``) are
    forwarded to each underlying model.

    Cooperative cancellation: pass ``cancel_event`` (a ``threading.Event``) here or per
    call (``p(prompt, cancel_event=evt)``). Once set, no NEW model call starts — pending
    items return ``cancelled=True`` with an explanatory ``error`` — and judge synthesis is
    skipped. An already-in-flight provider request is not aborted mid-HTTP.
    """
    return Parallel(
        models,
        mode=mode,
        judge=judge,
        max_workers=max_workers,
        cancel_event=cancel_event,
        **model_kwargs,
    )
