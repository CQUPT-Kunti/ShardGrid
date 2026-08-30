"""`shardgrid stop` CLI command (T104)."""

from __future__ import annotations

import argparse
import builtins
import json
import sys
from pathlib import Path
from typing import Any

from shardgrid.cli.context import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUNTIME_ERROR, EXIT_USAGE
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.control.status_store import StatusStore
from shardgrid.jobs.models import JobSnapshot, JobStatus, TrainingJob
from shardgrid.launchers.base import LauncherContext, LauncherResult, LauncherResultStatus
from shardgrid.launchers.ssh import SSHLauncher
from shardgrid.planner.models import ExecutionPlan


def register_stop_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser("stop", help="Stop a persisted job")
    parser.add_argument("job_id", help="Job ID to stop")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm stop without interactive prompt",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_stop_command, command_name="stop")


def _build_launcher(cluster_config) -> SSHLauncher:
    return SSHLauncher(cluster_config)


def _jobs_root(args: argparse.Namespace) -> Path:
    cli_context = args.context
    if cli_context.jobs_root is not None:
        return Path(cli_context.jobs_root)
    if cli_context.config is not None:
        return Path(cli_context.config.jobs_root)
    raise RuntimeError(
        "stop requires --config or --jobs-root: "
        "shardgrid --config examples/workers.yaml stop JOB --yes"
    )


def _job_root(args: argparse.Namespace) -> Path:
    return _jobs_root(args) / args.job_id


def _load_execution_plan(job_root: Path) -> ExecutionPlan:
    path = job_root / "plan" / "execution-plan.json"
    if not path.is_file():
        raise RuntimeError(f"execution plan not found: {path}")
    return ExecutionPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_job_status(job_root: Path, job_id: str) -> JobStatus:
    path = job_root / "diagnostics" / "job-status.json"
    if not path.is_file():
        raise RuntimeError(f"job status not found: {path}")
    return StatusStore(job_root.parent).load_path(path)


def _snapshot_for_job(job_root: Path, job_id: str) -> JobSnapshot:
    return JobSnapshot(
        job_id=as_job_id(job_id),
        root_path=str(job_root),
        code_path=str(job_root / "code"),
        config_path=str(job_root / "config"),
        plan_path=str(job_root / "plan"),
        logs_path=str(job_root / "logs"),
        environment_path=str(job_root / "environment"),
        checkpoint_path=str(job_root / "checkpoint"),
        diagnostics_path=str(job_root / "diagnostics"),
    )


def _load_launcher_context(args: argparse.Namespace) -> tuple[LauncherContext, JobStatus]:
    cli_context = args.context
    if cli_context.config is None:
        raise RuntimeError("stop requires a cluster config")
    job_root = _job_root(args)
    job_status = _load_job_status(job_root, args.job_id)
    execution_plan = _load_execution_plan(job_root)
    runtime_ref = next(iter(job_status.runtime_environment_refs.values()), None)
    job = TrainingJob(
        job_id=as_job_id(args.job_id),
        config_path=str(job_root / "config"),
        model="snapshot",
        requested_world_size=execution_plan.world_size,
        backend_preference=as_backend_name(str(job_status.backend or execution_plan.backend)),
        runtime_environment_ref=runtime_ref,
        state=job_status.state,
    )
    return (
        LauncherContext(
            job=job,
            execution_plan=execution_plan,
            cluster_state=ResourceManager().build_cluster_state([]),
            snapshot=_snapshot_for_job(job_root, args.job_id),
            job_status=job_status,
            runtime_environment_refs=dict(job_status.runtime_environment_refs),
        ),
        job_status,
    )


def _confirm_stop(args: argparse.Namespace, status: JobStatus) -> bool:
    if args.yes:
        return True
    if not sys.stdin.isatty():
        raise ValueError("use --yes to confirm stop in non-interactive mode")
    worker_count = len({str(item.worker_id) for item in status.assignments})
    rank_count = len(status.assignments)
    answer = builtins.input(
        "Stop job "
        f"{status.job_id} "
        f"(state={status.state.value}, workers={worker_count}, ranks={rank_count})? [y/N] "
    )
    return answer.strip().lower() in {"y", "yes"}


def _execute_stop(context: LauncherContext, status: JobStatus) -> LauncherResult:
    del status
    cluster_config = getattr(context, "_cluster_config")
    return _build_launcher(cluster_config).stop(context)


def _result_payload(
    *,
    result: LauncherResult,
    previous_status: JobStatus,
) -> dict[str, object]:
    return {
        "job_id": result.job_id,
        "previous_state": previous_status.state.value,
        "previous_phase": previous_status.phase,
        "status": result.status.value,
        "final_state": (
            None if result.next_job_state is None else result.next_job_state.value
        ),
        "backend": result.backend,
        "preserved_artifacts": True,
        "worker_results": [item.to_dict() for item in result.worker_results],
        "failure": None if result.failure is None else result.failure.to_dict(),
        "blocker": result.blocker,
        "message": result.message,
    }


def _render_human(result: LauncherResult, previous_status: JobStatus) -> str:
    lines = [
        f"Job: {result.job_id}",
        f"Previous State: {previous_status.state.value.upper()}",
        f"Previous Phase: {previous_status.phase}",
        f"Overall Stop Result: {result.status.value.upper()}",
        (
            "Final State: "
            f"{result.next_job_state.value.upper()}"
            if result.next_job_state is not None
            else "Final State: pending"
        ),
        "Preserved Artifacts: yes",
        "Rank Results:",
    ]
    for worker_result in result.worker_results:
        for rank_result in worker_result.rank_results:
            line = (
                f"  worker={worker_result.worker_id} "
                f"rank={rank_result.rank} "
                f"stage={rank_result.stage or '-'} "
                f"pid={rank_result.pid if rank_result.pid is not None else 'unavailable'} "
                f"status={rank_result.status.value.upper()}"
            )
            if rank_result.message:
                line += f" message={rank_result.message}"
            lines.append(line)
        if worker_result.failure is not None:
            lines.append(
                f"  worker={worker_result.worker_id} "
                f"failure={worker_result.failure.message}"
            )
            lines.append(
                f"  recommended_action={worker_result.failure.recommended_action}"
            )
    return "\n".join(lines)


def run_stop_command(args: argparse.Namespace) -> int:
    cli_context = getattr(args, "context", None)
    config = getattr(cli_context, "config", None)
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(cli_context, "json_output", False)
    )
    if config is None:
        print(
            "stop requires --config or --jobs-root: "
            "shardgrid --config examples/workers.yaml stop JOB --yes"
        )
        return EXIT_CONFIG_ERROR
    try:
        context, previous_status = _load_launcher_context(args)
        object.__setattr__(context, "_cluster_config", config)
        if not _confirm_stop(args, previous_status):
            print("stop: cancelled; no action taken")
            return EXIT_OK
        result = _execute_stop(context, previous_status)
    except ValueError as error:
        print(f"stop: {error}")
        return EXIT_USAGE
    except Exception as error:  # noqa: BLE001
        print(f"stop: {error}")
        return EXIT_RUNTIME_ERROR
    payload = _result_payload(result=result, previous_status=previous_status)
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if json_output
        else _render_human(result, previous_status)
    )
    if result.status in {
        LauncherResultStatus.SUCCESS,
        LauncherResultStatus.PARTIAL,
        LauncherResultStatus.NOOP,
    }:
        return EXIT_OK
    if result.status is LauncherResultStatus.UNSUPPORTED:
        return EXIT_CONFIG_ERROR
    return EXIT_RUNTIME_ERROR
