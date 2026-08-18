"""Formal Gate 2 distributed acceptance (T053).

Gate 2 proves that two different physical GPU Workers (RTX 4060 rank 0,
GTX 1650 rank 1, world_size 2) complete real PyTorch Distributed communication
from their own WSL2 selected Conda training runtimes.

The gate consumes the ``shardgrid dist-test`` report produced by T051, so no
launcher or distributed runtime is reimplemented here.  The backend state must
be real and explicit: either ``NCCL SUCCESS`` or ``GLOO FALLBACK`` (with the
original NCCL failure evidence preserved).  A Gloo success is never treated as
an NCCL success.

The gate is all-or-nothing and additionally requires Gate 1 (T052) to have
passed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

GATE2_PASS = "PASS"
GATE2_FAIL = "FAIL"
GATE2_BLOCKED = "BLOCKED"
GATE2_PENDING = "PENDING"

VALID_BACKEND_STATES = ("NCCL SUCCESS", "GLOO FALLBACK")

EXPECTED_BROADCAST = [11.0, 22.0, 33.0, 44.0]
EXPECTED_SEND_RECV = [5.0, 6.0, 7.0, 8.0]
EXPECTED_ALL_REDUCE = [3.0, 3.0, 3.0, 3.0]

REQUIRED_RUNTIME_KEYS = (
    "conda_environment",
    "conda_prefix",
    "python_executable",
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "gpu_name",
)


@dataclass(frozen=True)
class Gate2Result:
    status: str
    gate1_status: str | None
    backend_state: str | None
    backend_actual: str | None
    problems: tuple[str, ...]
    report: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": "gate2-distributed",
            "run_id": None if self.report is None else self.report.get("run_id"),
            "status": self.status,
            "gate1_status": self.gate1_status,
            "backend_state": self.backend_state,
            "backend_actual": self.backend_actual,
            "problems": list(self.problems),
            "report": self.report,
        }


def _collective_ok(
    summary: dict[str, Any], name: str, expected: list[float]
) -> bool:
    section = summary.get(name) or {}
    ok = section.get("ok") or {}
    tensor = section.get("tensor") or {}
    return (
        ok.get("rank0") is True
        and ok.get("rank1") is True
        and tensor.get("rank0") == expected
        and tensor.get("rank1") == expected
    )


def _evidence_complete(report: dict[str, Any]) -> bool:
    actual_backend = report.get("backend_actual")
    per_rank = _runtime_per_rank(report, actual_backend)
    for rank in ("0", "1"):
        entry = per_rank.get(rank) or {}
        for key in REQUIRED_RUNTIME_KEYS:
            if not entry.get(key):
                return False
    network = report.get("network") or {}
    if not network.get("master_addr") or not network.get("master_port"):
        return False
    interfaces = network.get("interfaces") or {}
    if not interfaces.get("0") or not interfaces.get("1"):
        return False
    return True


def _runtime_per_rank(
    report: dict[str, Any], actual_backend: str | None
) -> dict[str, dict[str, Any]]:
    runtime = report.get("runtime_evidence") or {}
    per_rank = runtime.get("per_rank") or {}
    if all((per_rank.get(rank) or {}).get("python_executable") for rank in ("0", "1")):
        return per_rank

    collectives = (report.get("collectives") or {}).get(actual_backend or "") or {}
    ranks = collectives.get("ranks") or []
    fallback: dict[str, dict[str, Any]] = {}
    for entry in ranks:
        rank = entry.get("rank")
        if rank in (0, 1, "0", "1"):
            fallback[str(rank)] = entry
    return fallback or per_rank


def _diagnostics_present(report: dict[str, Any], actual_backend: str | None) -> bool:
    if report.get("diagnostics_path"):
        return True

    collectives = (report.get("collectives") or {}).get(actual_backend or "") or {}
    for entry in collectives.get("ranks") or []:
        if entry.get("stdout_tail") or entry.get("stderr_tail"):
            return True

    nccl_failure = report.get("nccl_failure_evidence") or {}
    if actual_backend == "gloo" and nccl_failure:
        return True
    return False


def _collect_run_ids(report: dict[str, Any]) -> set[str]:
    run_ids: set[str] = set()
    root_run_id = report.get("run_id")
    if root_run_id:
        run_ids.add(str(root_run_id))

    runtime = report.get("runtime_evidence") or {}
    runtime_run_id = runtime.get("run_id")
    if runtime_run_id:
        run_ids.add(str(runtime_run_id))
    for entry in (runtime.get("per_rank") or {}).values():
        run_id = (entry or {}).get("run_id")
        if run_id:
            run_ids.add(str(run_id))

    for summary in (report.get("collectives") or {}).values():
        summary_run_id = summary.get("run_id")
        if summary_run_id:
            run_ids.add(str(summary_run_id))
        for entry in summary.get("ranks") or []:
            run_id = (entry or {}).get("run_id")
            if run_id:
                run_ids.add(str(run_id))

    for key in ("nccl_failure_evidence", "gloo_fallback"):
        attempt = report.get(key) or {}
        run_id = attempt.get("run_id")
        if run_id:
            run_ids.add(str(run_id))
    return run_ids


def evaluate_gate2(
    report: dict[str, Any] | None,
    *,
    gate1_status: str,
    expected_workers: Sequence[str] | None = None,
) -> Gate2Result:
    """Decide Gate 2 from a real dist-test report plus the Gate 1 status."""
    if not report:
        return Gate2Result(
            status=GATE2_PENDING,
            gate1_status=gate1_status,
            backend_state=None,
            backend_actual=None,
            problems=("no dist-test report provided; nothing executed",),
            report=None,
        )

    if gate1_status != GATE2_PASS:
        return Gate2Result(
            status=GATE2_BLOCKED,
            gate1_status=gate1_status,
            backend_state=report.get("backend_state"),
            backend_actual=report.get("backend_actual"),
            problems=(f"Gate 1 is {gate1_status}, required PASS",),
            report=report,
        )

    fatal: list[str] = []
    blocked: list[str] = []

    backend_state = report.get("backend_state")
    backend_actual = report.get("backend_actual")
    workers = report.get("workers") or []

    if len(workers) != 2:
        blocked.append(f"report has {len(workers)} workers, expected 2")
    else:
        worker_ids = [str(worker.get("worker_id")) for worker in workers]
        ips = [worker.get("ip") for worker in workers]
        if (
            len(set(worker_ids)) != 2
            or len(set(ips)) != 2
            or any(ip is None for ip in ips)
        ):
            blocked.append("workers are not two distinct physical hosts")
        elif expected_workers is not None and set(worker_ids) != set(expected_workers):
            blocked.append(
                f"workers {sorted(worker_ids)} != expected {sorted(expected_workers)}"
            )

    if report.get("world_size") != 2 or report.get("local_world_size") != 1:
        blocked.append("world_size/local_world_size is not 2/1")

    if backend_state == "FALLBACK NOT ALLOWED":
        blocked.append("backend blocked: FALLBACK NOT ALLOWED")
    elif backend_state not in VALID_BACKEND_STATES:
        fatal.append(
            f"backend_state {backend_state!r} is not a valid gate result "
            "(NCCL SUCCESS or GLOO FALLBACK)"
        )
    elif backend_state == "GLOO FALLBACK" and not report.get("nccl_failure_evidence"):
        fatal.append("GLOO FALLBACK reported without preserved NCCL failure evidence")

    coll = (report.get("collectives") or {}).get(backend_actual or "")
    if not coll:
        blocked.append(
            f"missing collectives evidence for actual backend {backend_actual!r}"
        )
    else:
        if not coll.get("process_group", {}).get("ok"):
            fatal.append("process group did not initialize on both ranks")
        if not _collective_ok(coll, "broadcast", EXPECTED_BROADCAST):
            fatal.append("broadcast failed or tensor result invalid")
        if not _collective_ok(coll, "send_recv", EXPECTED_SEND_RECV):
            fatal.append("send/recv failed or tensor result invalid")
        if not _collective_ok(coll, "all_reduce", EXPECTED_ALL_REDUCE):
            fatal.append("all_reduce failed or tensor result invalid")

    if not _evidence_complete(report):
        blocked.append("runtime/network/backend evidence incomplete")
    if not _diagnostics_present(report, backend_actual):
        blocked.append("diagnostics evidence missing")
    run_ids = _collect_run_ids(report)
    if not report.get("run_id"):
        blocked.append("run_id missing from distributed evidence")
    elif len(run_ids) > 1:
        blocked.append(f"mixed run_id evidence detected: {sorted(run_ids)}")

    problems = tuple(fatal + blocked)
    if fatal:
        status = GATE2_FAIL
    elif blocked:
        status = GATE2_BLOCKED
    else:
        status = GATE2_PASS

    return Gate2Result(
        status=status,
        gate1_status=gate1_status,
        backend_state=backend_state,
        backend_actual=backend_actual,
        problems=problems,
        report=report,
    )


def save_gate2_evidence(
    result: Gate2Result,
    output_dir: str | Path,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = result.to_dict()
    payload["timestamp"] = timestamp
    path = directory / f"gate2-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (directory / "gate2-latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return path
