"""NCCL -> Gloo fallback orchestration (T050).

NCCL is always the preferred backend.  Gloo is only ever used as a labelled
fallback after NCCL has genuinely failed AND the baseline conditions required
for any distributed backend (raw TCP reachability, rendezvous, runtime) are
present.  A broken baseline (WSL TCP down, rendezvous port blocked, Worker
unreachable, SSH/runtime failure, base network misconfiguration) must NOT be
wrapped as ``NCCL FAILED -> GLOO FALLBACK``; it stays ``FALLBACK NOT ALLOWED``.

Every decision exposes an explicit state label so a Gloo success is never
mislabelled as an NCCL success.  The original NCCL failure evidence (per-rank
exit codes, errors, and log tails) is preserved in the returned decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from shardgrid.distributed.collectives import (
    RankCollectiveResult,
    collectives_outcome,
)

STATE_NCCL_SUCCESS = "NCCL SUCCESS"
STATE_NCCL_FAILED = "NCCL FAILED"
STATE_GLOO_FALLBACK = "GLOO FALLBACK"
STATE_FALLBACK_NOT_ALLOWED = "FALLBACK NOT ALLOWED"
STATE_FALLBACK_FAILED = "FALLBACK FAILED"

VALID_REQUESTED_BACKENDS = ("nccl", "auto")

# Legacy aliases kept for callers that imported the previous T048-era labels.
LABEL_NCCL_PASS = STATE_NCCL_SUCCESS
LABEL_NCCL_FAILED = STATE_NCCL_FAILED
LABEL_GLOO_FALLBACK = STATE_GLOO_FALLBACK
LABEL_GLOO_FALLBACK_FAILED = STATE_FALLBACK_FAILED


class FallbackError(ValueError):
    """Raised when a fallback decision cannot be formed (invalid state)."""


@dataclass(frozen=True)
class FallbackEligibility:
    """Baseline conditions that must hold before Gloo fallback may be attempted.

    These describe the network/rendezvous/runtime baseline, NOT the NCCL result
    itself.  When any of them is False, a failed NCCL attempt is reported as
    ``FALLBACK NOT ALLOWED`` instead of being retried on Gloo.
    """

    network_ok: bool
    rendezvous_ok: bool
    runtime_ok: bool

    @property
    def allowed(self) -> bool:
        return self.network_ok and self.rendezvous_ok and self.runtime_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "network_ok": self.network_ok,
            "rendezvous_ok": self.rendezvous_ok,
            "runtime_ok": self.runtime_ok,
            "allowed": self.allowed,
        }


@dataclass(frozen=True)
class BackendAttempt:
    backend: str
    outcome: str
    result: dict[str, Any] | None
    error: str | None
    run_id: str | None = None
    ranks: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "outcome": self.outcome,
            "result": self.result,
            "error": self.error,
            "run_id": self.run_id,
            "ranks": self.ranks,
        }


@dataclass(frozen=True)
class FallbackDecision:
    state: str
    nccl: BackendAttempt
    gloo: BackendAttempt | None
    eligibility: FallbackEligibility
    requested_backend: str

    @property
    def label(self) -> str:
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "label": self.label,
            "requested_backend": self.requested_backend,
            "eligibility": self.eligibility.to_dict(),
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
            errors.append(
                f"rank exits: {rank0.exit_code}/{rank1.exit_code}, "
                f"timed_out: {rank0.timed_out}/{rank1.timed_out}"
            )
        error = "; ".join(errors)
    ranks = {
        str(rank0.rank): {
            "worker_id": rank0.worker_id,
            "exit_code": rank0.exit_code,
            "timed_out": rank0.timed_out,
            "stdout_tail": rank0.stdout[-2000:],
            "stderr_tail": rank0.stderr[-2000:],
        },
        str(rank1.rank): {
            "worker_id": rank1.worker_id,
            "exit_code": rank1.exit_code,
            "timed_out": rank1.timed_out,
            "stdout_tail": rank1.stdout[-2000:],
            "stderr_tail": rank1.stderr[-2000:],
        },
    }
    return BackendAttempt(
        backend=backend,
        outcome="PASS" if outcome == "PASS" else "FAILED",
        result=rank0.result or rank1.result,
        error=error,
        run_id=(rank0.result or {}).get("run_id") or (rank1.result or {}).get("run_id"),
        ranks=ranks,
    )


def decide_fallback(
    *,
    requested_backend: str,
    nccl: BackendAttempt,
    gloo: BackendAttempt | None,
    eligibility: FallbackEligibility,
) -> FallbackDecision:
    """Pure decision: map NCCL/Gloo attempt outcomes + eligibility to a state.

    - NCCL PASS                    -> ``NCCL SUCCESS`` (Gloo never attempted)
    - requested ``nccl``, NCCL fail -> ``NCCL FAILED`` (no fallback allowed)
    - requested ``auto``, NCCL fail, baseline blocked
                                 -> ``FALLBACK NOT ALLOWED``
    - requested ``auto``, NCCL fail, baseline OK, Gloo PASS
                                 -> ``GLOO FALLBACK``
    - requested ``auto``, NCCL fail, baseline OK, Gloo FAIL
                                 -> ``FALLBACK FAILED``
    """
    normalized = requested_backend.lower()
    if normalized not in VALID_REQUESTED_BACKENDS:
        raise FallbackError(
            f"invalid fallback state: requested backend {requested_backend!r} is "
            f"not one of {VALID_REQUESTED_BACKENDS}"
        )
    if nccl.outcome == "PASS":
        return FallbackDecision(
            state=STATE_NCCL_SUCCESS,
            nccl=nccl,
            gloo=None,
            eligibility=eligibility,
            requested_backend=requested_backend,
        )
    if normalized == "nccl":
        return FallbackDecision(
            state=STATE_NCCL_FAILED,
            nccl=nccl,
            gloo=None,
            eligibility=eligibility,
            requested_backend=requested_backend,
        )
    if not eligibility.allowed:
        return FallbackDecision(
            state=STATE_FALLBACK_NOT_ALLOWED,
            nccl=nccl,
            gloo=None,
            eligibility=eligibility,
            requested_backend=requested_backend,
        )
    if gloo is None:
        raise FallbackError(
            "invalid fallback state: auto fallback requested but no Gloo attempt "
            "was provided"
        )
    state = STATE_GLOO_FALLBACK if gloo.outcome == "PASS" else STATE_FALLBACK_FAILED
    return FallbackDecision(
        state=state,
        nccl=nccl,
        gloo=gloo,
        eligibility=eligibility,
        requested_backend=requested_backend,
    )


def run_with_fallback(
    run_nccl: Callable[[], tuple[RankCollectiveResult, RankCollectiveResult]],
    run_gloo: Callable[[], tuple[RankCollectiveResult, RankCollectiveResult]],
    *,
    requested_backend: str = "auto",
    eligibility: FallbackEligibility | None = None,
) -> FallbackDecision:
    """Run NCCL first; fall back to Gloo only under a valid, eligible decision.

    The NCCL run always happens first.  Gloo is only invoked when the requested
    backend is ``auto``, NCCL genuinely failed, and the baseline conditions in
    ``eligibility`` permit a fallback.
    """
    if requested_backend.lower() not in VALID_REQUESTED_BACKENDS:
        raise FallbackError(
            f"invalid fallback state: requested backend {requested_backend!r} is "
            f"not one of {VALID_REQUESTED_BACKENDS}"
        )
    effective_eligibility = eligibility or FallbackEligibility(
        network_ok=True, rendezvous_ok=True, runtime_ok=True
    )
    nccl_rank0, nccl_rank1 = run_nccl()
    nccl_attempt = _attempt_from_ranks("nccl", nccl_rank0, nccl_rank1)
    gloo_attempt: BackendAttempt | None = None
    if (
        nccl_attempt.outcome == "FAILED"
        and requested_backend.lower() == "auto"
        and effective_eligibility.allowed
    ):
        gloo_rank0, gloo_rank1 = run_gloo()
        gloo_attempt = _attempt_from_ranks("gloo", gloo_rank0, gloo_rank1)
    return decide_fallback(
        requested_backend=requested_backend,
        nccl=nccl_attempt,
        gloo=gloo_attempt,
        eligibility=effective_eligibility,
    )
