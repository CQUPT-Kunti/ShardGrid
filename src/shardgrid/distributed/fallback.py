"""NCCL -> Gloo fallback orchestration (T050).

Fallback runs only after NCCL fails and its failure evidence is preserved.
The result is labelled ``NCCL FAILED`` and ``GLOO FALLBACK`` (or
``NCCL FAILED, GLOO FALLBACK FAILED``); a Gloo success is never mislabelled
as an NCCL success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from shardgrid.distributed.collectives import (
    RankCollectiveResult,
    collectives_outcome,
)

LABEL_NCCL_PASS = "NCCL PASS"
LABEL_NCCL_FAILED = "NCCL FAILED"
LABEL_GLOO_FALLBACK = "GLOO FALLBACK"
LABEL_GLOO_FALLBACK_FAILED = "NCCL FAILED, GLOO FALLBACK FAILED"


@dataclass(frozen=True)
class BackendAttempt:
    backend: str
    outcome: str
    result: dict[str, Any] | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "outcome": self.outcome,
            "result": self.result,
            "error": self.error,
        }


@dataclass(frozen=True)
class FallbackOutcome:
    nccl: BackendAttempt
    gloo: BackendAttempt | None
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "nccl": self.nccl.to_dict(),
            "gloo": None if self.gloo is None else self.gloo.to_dict(),
        }


def _attempt_from_ranks(
    backend: str, rank0: RankCollectiveResult, rank1: RankCollectiveResult
) -> BackendAttempt:
    outcome = collectives_outcome(rank0, rank1)
    error = None
    if outcome != "PASS":
        errors = []
        for result in (rank0.result, rank1.result):
            if result is not None and result.get("error"):
                errors.append(str(result["error"]))
        if not errors:
            errors.append(f"rank exits: {rank0.exit_code}/{rank1.exit_code}, "
                          f"timed_out: {rank0.timed_out}/{rank1.timed_out}")
        error = "; ".join(errors)
    return BackendAttempt(
        backend=backend,
        outcome="PASS" if outcome == "PASS" else "FAILED",
        result=rank0.result or rank1.result,
        error=error,
    )


def run_with_fallback(
    run_nccl: Callable[[], tuple[RankCollectiveResult, RankCollectiveResult]],
    run_gloo: Callable[[], tuple[RankCollectiveResult, RankCollectiveResult]],
) -> FallbackOutcome:
    """Run NCCL first; fall back to Gloo only after NCCL fails."""
    nccl_rank0, nccl_rank1 = run_nccl()
    nccl_attempt = _attempt_from_ranks("nccl", nccl_rank0, nccl_rank1)
    if nccl_attempt.outcome == "PASS":
        return FallbackOutcome(nccl=nccl_attempt, gloo=None, label=LABEL_NCCL_PASS)

    gloo_rank0, gloo_rank1 = run_gloo()
    gloo_attempt = _attempt_from_ranks("gloo", gloo_rank0, gloo_rank1)
    if gloo_attempt.outcome == "PASS":
        label = LABEL_GLOO_FALLBACK
    else:
        label = LABEL_GLOO_FALLBACK_FAILED
    return FallbackOutcome(nccl=nccl_attempt, gloo=gloo_attempt, label=label)