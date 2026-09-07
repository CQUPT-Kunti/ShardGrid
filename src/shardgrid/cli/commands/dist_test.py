"""`shardgrid dist-test` CLI command (T051).

Runs the real multi-host PyTorch Distributed communication test (T049
collectives harness) between two GPU Workers from Machine A with an explicit
backend policy:

- ``--backend nccl``: NCCL only; a failure is reported as ``NCCL FAILED`` and
  never silently retried on Gloo.
- ``--backend gloo``: explicit Gloo; output always labels ``backend=gloo``.
- ``--backend auto``: NCCL first; Gloo is only attempted when the T050
  fallback decision allows it (real NCCL failure + healthy baseline).

The command only wires configuration, Workers, baseline checks, the T049
collective execution, the T050 fallback decision, and report output.  It does
not reimplement distributed launch, SSH, WSL, or the collectives themselves.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shardgrid.cli.context import (
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE,
)
from shardgrid.common.config import WorkerConfig
from shardgrid.common.models import as_hostname
from shardgrid.distributed.collectives import (
    RankCollectiveResult,
    run_pair_collectives,
)
from shardgrid.distributed.fallback import (
    STATE_FALLBACK_FAILED,
    STATE_FALLBACK_NOT_ALLOWED,
    STATE_GLOO_FALLBACK,
    STATE_NCCL_FAILED,
    STATE_NCCL_SUCCESS,
    FallbackEligibility,
    _attempt_from_ranks,
    run_with_fallback,
)
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport

STATE_GLOO_PASS = "GLOO PASS"
STATE_GLOO_FAILED = "GLOO FAILED"

VALID_BACKENDS = ("auto", "nccl", "gloo")

DEFAULT_REPORT_DIR = Path("/var/tmp/shardgrid/distributed/reports")

_PASS_STATES = (STATE_NCCL_SUCCESS, STATE_GLOO_FALLBACK, STATE_GLOO_PASS)
_FAIL_STATES = (STATE_NCCL_FAILED, STATE_GLOO_FAILED, STATE_FALLBACK_FAILED)


def register_dist_test_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser(
        "dist-test",
        help=(
            "Run a multi-host PyTorch Distributed communication test between two "
            "GPU Workers (broadcast, send/recv, all_reduce)"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=VALID_BACKENDS,
        default="auto",
        help=(
            "Backend policy: auto (NCCL first, Gloo fallback), nccl (NCCL only), "
            "or gloo (explicit Gloo)"
        ),
    )
    parser.add_argument(
        "--workers",
        help="Comma-separated worker IDs, e.g. gpu4060,gpu1060 (rank order)",
    )
    parser.add_argument(
        "--save-report",
        nargs="?",
        const=None,
        default=None,
        metavar="PATH",
        help="Write the JSON report to PATH (default: a reports directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_dist_test_command, command_name="dist-test")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_workers(workers_arg: str | None) -> list[str]:
    if not workers_arg:
        raise ValueError("--workers is required: e.g. --workers gpu4060,gpu1060")
    ids = [
        item.strip()
        for item in workers_arg.replace(" ", ",").split(",")
        if item.strip()
    ]
    if len(ids) != 2:
        raise ValueError(f"--workers must name exactly 2 workers, got {len(ids)}")
    if ids[0] == ids[1]:
        raise ValueError("--workers must name two distinct workers")
    return ids


def _resolve_workers(
    config: Any, worker_ids: list[str]
) -> list[WorkerConfig]:
    by_id = {str(worker.worker_id): worker for worker in config.workers}
    missing = [worker_id for worker_id in worker_ids if worker_id not in by_id]
    if missing:
        raise ValueError(
            f"unknown worker id(s): {', '.join(missing)}; "
            f"known: {', '.join(sorted(by_id))}"
        )
    return [by_id[worker_id] for worker_id in worker_ids]


def _build_wrappers(
    config: Any, workers: list[WorkerConfig]
) -> list[dict[str, Any]]:
    """Resolve each Worker to its WSL2 runtime wrapper and host metadata."""
    address_book = json.loads(Path("tests/address.json").read_text(encoding="utf-8"))
    infos: list[dict[str, Any]] = []
    for rank, worker in enumerate(workers):
        gpu_label = worker.labels.get("gpu", "").upper().replace(" ", "")
        matches = [
            item
            for item in address_book
            if gpu_label
            and gpu_label
            in str(item.get("gpu_model") or "").replace(" ", "").upper()
        ]
        if not matches:
            raise ValueError(
                f"no address entry for worker {worker.worker_id} "
                f"(gpu label {gpu_label!r}) in tests/address.json"
            )
        entry = matches[0]
        ip = str(entry["ip"])
        resolved = replace(worker, host=as_hostname(ip), ssh_user=str(entry["username"]))
        transport = SSHTransport(
            SSHOptions.from_ssh_config(
                config.ssh,
                host=ip,
                user=resolved.ssh_user,
                port=resolved.ssh_port,
            )
        )
        wrapper = WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(resolved, config.runtime),
            transport,
        )
        infos.append(
            {
                "wrapper": wrapper,
                "worker_id": str(resolved.worker_id),
                "rank": rank,
                "ip": ip,
                "hostname": entry.get("hostname"),
                "gpu_model": entry.get("gpu_model"),
            }
        )
    return infos


def _interface_from_route(text: str) -> str | None:
    first_line = text.splitlines()[0] if text else ""
    tokens = first_line.split()
    if "dev" in tokens:
        return tokens[tokens.index("dev") + 1]
    return None


def _check_baseline(
    infos: list[dict[str, Any]],
) -> tuple[FallbackEligibility, dict[str, Any]]:
    """Verify each Worker is reachable over SSH/WSL and has a route to its peer."""
    workers_diag: dict[str, Any] = {}
    interfaces: dict[str, str] = {}
    runtime_ok = True
    network_ok = True
    route_probe_only = True
    for info in infos:
        wrapper = info["wrapper"]
        rank = info["rank"]
        peer_ip = infos[1 - rank]["ip"]
        diag: dict[str, Any] = {
            "distro": wrapper.config.distro,
            "conda_prefix": wrapper.config.conda_prefix,
            "conda_environment": wrapper.config.conda_environment,
            "peer_ip": peer_ip,
            "network_probe": "route_only",
        }
        if not wrapper.config.distro or not wrapper.config.conda_prefix:
            runtime_ok = False
            diag["runtime_configured"] = False
            diag["interface"] = None
            workers_diag[str(rank)] = diag
            continue
        diag["runtime_configured"] = True
        try:
            result = wrapper.run(f"ip route get {peer_ip}", timeout=10.0)
            text = (result.stdout or result.stderr).strip()
            diag["ssh_reachable"] = result.exit_code == 0
            diag["route_output"] = text[:500]
            diag["interface"] = _interface_from_route(text)
        except Exception as error:  # noqa: BLE001 - surfaced into diagnostics
            diag["ssh_reachable"] = False
            diag["interface"] = None
            diag["error"] = str(error)
        if not diag.get("ssh_reachable") or not diag.get("interface"):
            network_ok = False
        workers_diag[str(rank)] = diag
        interfaces[str(rank)] = diag.get("interface") or ""
    eligibility = FallbackEligibility(
        network_ok=network_ok,
        rendezvous_ok=bool(
            network_ok
            and infos
            and infos[0]["ip"]
            and interfaces.get("0")
            and interfaces.get("1")
        ),
        runtime_ok=runtime_ok,
    )
    return eligibility, {
        "workers": workers_diag,
        "interfaces": interfaces,
        "network_probe": "route_only",
        "route_probe_only": route_probe_only,
    }


def _run_collectives(
    infos: list[dict[str, Any]],
    *,
    run_id: str,
    backend: str,
    master_addr: str,
    master_port: int,
    interfaces: list[str],
    timeout: float = 180.0,
) -> tuple[RankCollectiveResult, RankCollectiveResult]:
    return run_pair_collectives(
        infos[0]["wrapper"],
        infos[1]["wrapper"],
        rank0_worker_id=infos[0]["worker_id"],
        rank1_worker_id=infos[1]["worker_id"],
        rank0_worker_ip=infos[0]["ip"],
        rank1_worker_ip=infos[1]["ip"],
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
        rank0_interface=interfaces[0],
        rank1_interface=interfaces[1],
        run_id=run_id,
        timeout=timeout,
    )


def _rank_result_dict(result: RankCollectiveResult) -> dict[str, Any]:
    base = result.result or {}
    return {
        "run_id": base.get("run_id"),
        "rank": result.rank,
        "worker_id": result.worker_id,
        "worker_host_ip": base.get("worker_host_ip"),
        "peer_ip": base.get("peer_ip"),
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "error": base.get("error"),
        "init_ok": base.get("init_ok"),
        "broadcast_ok": base.get("broadcast_ok"),
        "broadcast_tensor": base.get("broadcast_tensor"),
        "send_recv_ok": base.get("send_recv_ok"),
        "send_recv_tensor": base.get("send_recv_tensor"),
        "all_reduce_ok": base.get("all_reduce_ok"),
        "all_reduce_tensor": base.get("all_reduce_tensor"),
        "stages": base.get("stages"),
        "last_stage": base.get("last_stage"),
        "elapsed_s": base.get("elapsed_s"),
        "conda_environment": base.get("conda_environment"),
        "conda_prefix": base.get("conda_prefix"),
        "python_executable": base.get("python_executable"),
        "python_version": base.get("python_version"),
        "torch_version": base.get("torch_version"),
        "torch_cuda_version": base.get("torch_cuda_version"),
        "gpu_name": base.get("gpu_name"),
        "network_interface": base.get("network_interface"),
        "master_addr": base.get("master_addr"),
        "master_port": base.get("master_port"),
        "route_output": base.get("route_output"),
        "port_range": base.get("port_range"),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def _collectives_summary(
    rank0: RankCollectiveResult, rank1: RankCollectiveResult
) -> dict[str, Any]:
    r0 = rank0.result or {}
    r1 = rank1.result or {}

    def combined(field: str) -> dict[str, Any]:
        return {"rank0": r0.get(field), "rank1": r1.get(field)}

    return {
        "run_id": r0.get("run_id") or r1.get("run_id"),
        "process_group": {
            "ok": bool(r0.get("init_ok")) and bool(r1.get("init_ok")),
            "rank0_init": r0.get("init_ok"),
            "rank1_init": r1.get("init_ok"),
        },
        "broadcast": {
            "ok": combined("broadcast_ok"),
            "tensor": combined("broadcast_tensor"),
        },
        "send_recv": {
            "ok": combined("send_recv_ok"),
            "tensor": combined("send_recv_tensor"),
        },
        "all_reduce": {
            "ok": combined("all_reduce_ok"),
            "tensor": combined("all_reduce_tensor"),
        },
        "elapsed_s": combined("elapsed_s"),
        "ranks": [_rank_result_dict(rank0), _rank_result_dict(rank1)],
    }


def _runtime_evidence(
    rank0: RankCollectiveResult, rank1: RankCollectiveResult
) -> dict[str, Any]:
    base = (rank0.result or {}) or (rank1.result or {})
    return {
        "run_id": base.get("run_id"),
        "conda_environment": base.get("conda_environment"),
        "conda_prefix": base.get("conda_prefix"),
        "python_executable": base.get("python_executable"),
        "python_version": base.get("python_version"),
        "torch_version": base.get("torch_version"),
        "torch_cuda_version": base.get("torch_cuda_version"),
        "cuda_available": base.get("cuda_available"),
        "per_rank": {
            str(rank0.rank): _rank_result_dict(rank0),
            str(rank1.rank): _rank_result_dict(rank1),
        },
    }


def _backend_for_state(state: str) -> str:
    if state in (STATE_NCCL_SUCCESS, STATE_NCCL_FAILED):
        return "nccl"
    if state in (
        STATE_GLOO_PASS,
        STATE_GLOO_FAILED,
        STATE_GLOO_FALLBACK,
        STATE_FALLBACK_FAILED,
    ):
        return "gloo"
    return "none"


def _status_for_state(state: str) -> str:
    if state in _PASS_STATES:
        return "PASS"
    if state in _FAIL_STATES:
        return "FAIL"
    if state == STATE_FALLBACK_NOT_ALLOWED:
        return "BLOCKED"
    return "UNKNOWN"


def _build_report(
    *,
    run_id: str,
    timestamp: str,
    requested_backend: str,
    state: str,
    eligibility: FallbackEligibility,
    baseline: dict[str, Any],
    infos: list[dict[str, Any]],
    captured: dict[str, tuple[RankCollectiveResult, RankCollectiveResult]],
    master_addr: str,
    master_port: int,
    elapsed_s: float,
    diagnostics_path: str | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "command": "dist-test",
        "status": _status_for_state(state),
        "backend_requested": requested_backend,
        "backend_state": state,
        "backend_actual": _backend_for_state(state),
        "world_size": 2,
        "local_world_size": 1,
        "workers": [
            {
                "worker_id": info["worker_id"],
                "rank": info["rank"],
                "ip": info["ip"],
                "hostname": info.get("hostname"),
                "gpu_model": info.get("gpu_model"),
            }
            for info in infos
        ],
        "network": {
            "master_addr": master_addr,
            "master_port": master_port,
            "interfaces": baseline.get("interfaces", {}),
            "eligibility": eligibility.to_dict(),
            "baseline": baseline,
        },
        "collectives": {},
        "runtime_evidence": {},
        "elapsed_s": elapsed_s,
        "nccl_failure_evidence": None,
        "gloo_fallback": None,
        "diagnostics_path": diagnostics_path,
    }
    for backend_key in ("nccl", "gloo"):
        if backend_key not in captured:
            continue
        rank0, rank1 = captured[backend_key]
        attempt = _attempt_from_ranks(backend_key, rank0, rank1)
        report["collectives"][backend_key] = _collectives_summary(rank0, rank1)
        if backend_key == "nccl":
            if attempt.outcome == "FAILED":
                report["nccl_failure_evidence"] = attempt.to_dict()
        else:
            report["gloo_fallback"] = attempt.to_dict()

    # Runtime evidence must come from the backend actually used: a failed /
    # timed-out NCCL attempt can leave rank results incomplete, while the
    # succeeded Gloo attempt carries the full per-rank runtime evidence.
    actual_backend = _backend_for_state(state)
    if actual_backend in captured:
        rank0, rank1 = captured[actual_backend]
        report["runtime_evidence"] = _runtime_evidence(rank0, rank1)
    elif "nccl" in captured:
        rank0, rank1 = captured["nccl"]
        report["runtime_evidence"] = _runtime_evidence(rank0, rank1)
    return report


def _human_output(report: dict[str, Any]) -> str:
    lines = [
        "ShardGrid dist-test",
        f"backend requested: {report['backend_requested']}",
        f"backend used: {report['backend_state']} ({report['backend_actual']})",
        f"status: {report['status']}",
    ]
    for worker in report["workers"]:
        lines.append(
            f"  worker {worker['rank']}: {worker['worker_id']} "
            f"({worker['gpu_model'] or 'n/a'} @ {worker['ip']})"
        )
    network = report["network"]
    lines.append(
        f"master: {network['master_addr']}:{network['master_port']} "
        f"interfaces: {network['interfaces']}"
    )
    for backend_key, summary in report["collectives"].items():
        lines.append(f"[{backend_key}] process group: {summary['process_group']}")
        lines.append(f"[{backend_key}] broadcast: {summary['broadcast']}")
        lines.append(f"[{backend_key}] send/recv: {summary['send_recv']}")
        lines.append(f"[{backend_key}] all_reduce: {summary['all_reduce']}")
    runtime = report.get("runtime_evidence") or {}
    if runtime:
        lines.append(
            f"runtime: torch {runtime.get('torch_version')} / "
            f"CUDA {runtime.get('torch_cuda_version')} / "
            f"conda {runtime.get('conda_environment')}"
        )
    lines.append(f"elapsed: {report.get('elapsed_s')}s")
    if report["backend_state"] == STATE_GLOO_FALLBACK:
        lines.append("NCCL FAILED")
        lines.append("GLOO FALLBACK: PASS")
    if report.get("report_path"):
        lines.append(f"report saved: {report['report_path']}")
    return "\n".join(lines)


def _report_target(args: argparse.Namespace, timestamp: str) -> Path:
    target = getattr(args, "save_report", None)
    if target:
        path = Path(target)
        if path.is_dir() or str(target).endswith("/"):
            return path / f"dist-test-{timestamp}.json"
        return path
    return DEFAULT_REPORT_DIR / f"dist-test-{timestamp}.json"


def _save_report(report: dict[str, Any], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True))
    return target


def _cleanup_remote_python(infos: list[dict[str, Any]]) -> None:
    """Best-effort kill of stale WSL Conda python processes on the Workers.

    A timed-out SSH rank can leave a WSL python process alive that keeps the
    rendezvous port bound; killing it before the next backend attempt prevents
    EADDRINUSE from poisoning the Gloo fallback (documented T049 practice).
    """
    for info in infos:
        try:
            info["wrapper"].run(
                "pkill -9 -f miniconda3/envs/shardgrid/bin/python || true",
                timeout=15.0,
            )
        except Exception:
            pass


def run_dist_test_command(args: argparse.Namespace) -> int:
    context = getattr(args, "context", None)
    config = getattr(context, "config", None)
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(context, "json_output", False)
    )
    if config is None:
        print(
            "dist-test requires a cluster config: "
            "shardgrid --config examples/workers.yaml dist-test --backend auto "
            "--workers gpu4060,gpu1060"
        )
        return EXIT_CONFIG_ERROR

    backend = getattr(args, "backend", "auto")
    try:
        worker_ids = _parse_workers(getattr(args, "workers", None))
        workers = _resolve_workers(config, worker_ids)
    except ValueError as error:
        print(f"dist-test: {error}")
        return EXIT_USAGE

    try:
        infos = _build_wrappers(config, workers)
    except Exception as error:  # noqa: BLE001 - surfaced into CLI error
        print(f"dist-test: cannot resolve worker runtimes: {error}")
        return EXIT_RUNTIME_ERROR

    master_addr = infos[0]["ip"]
    master_port = config.network.rendezvous_port

    try:
        eligibility, baseline = _check_baseline(infos)
    except Exception as error:  # noqa: BLE001 - surfaced into CLI error
        print(f"dist-test: baseline check failed: {error}")
        return EXIT_RUNTIME_ERROR

    captured: dict[str, tuple[RankCollectiveResult, RankCollectiveResult]] = {}
    state = ""
    started = time.time()
    run_id = f"dist-test-{uuid.uuid4().hex}"

    if eligibility.allowed:
        interfaces = [
            baseline["interfaces"]["0"],
            baseline["interfaces"]["1"],
        ]
        _cleanup_remote_python(infos)

        def run_backend(
            run_backend_name: str,
        ) -> tuple[RankCollectiveResult, RankCollectiveResult]:
            result = _run_collectives(
                infos,
                run_id=run_id,
                backend=run_backend_name,
                master_addr=master_addr,
                master_port=master_port,
                interfaces=interfaces,
            )
            captured[run_backend_name] = result
            _cleanup_remote_python(infos)
            return result

        if backend == "gloo":
            rank0, rank1 = run_backend("gloo")
            attempt = _attempt_from_ranks("gloo", rank0, rank1)
            state = STATE_GLOO_PASS if attempt.outcome == "PASS" else STATE_GLOO_FAILED
        elif backend == "nccl":
            rank0, rank1 = run_backend("nccl")
            attempt = _attempt_from_ranks("nccl", rank0, rank1)
            state = (
                STATE_NCCL_SUCCESS
                if attempt.outcome == "PASS"
                else STATE_NCCL_FAILED
            )
        else:  # auto: NCCL first, Gloo only via the T050 fallback decision
            decision = run_with_fallback(
                lambda: run_backend("nccl"),
                lambda: run_backend("gloo"),
                requested_backend="auto",
                eligibility=eligibility,
            )
            state = decision.state
    else:
        state = STATE_FALLBACK_NOT_ALLOWED

    elapsed_s = round(time.time() - started, 3)

    timestamp = _now()
    report = _build_report(
        run_id=run_id,
        timestamp=timestamp,
        requested_backend=backend,
        state=state,
        eligibility=eligibility,
        baseline=baseline,
        infos=infos,
        captured=captured,
        master_addr=master_addr,
        master_port=master_port,
        elapsed_s=elapsed_s,
        diagnostics_path=None,
    )

    try:
        target = _report_target(args, timestamp)
        report["report_path"] = str(_save_report(report, target))
    except Exception as error:  # noqa: BLE001 - report write must not mask result
        print(f"dist-test: warning: could not save report: {error}")

    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_human_output(report))

    if state in _PASS_STATES:
        return EXIT_OK
    return EXIT_RUNTIME_ERROR
