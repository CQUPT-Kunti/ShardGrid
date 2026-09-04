"""`shardgrid logs` CLI command (T098)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shardgrid.cli.context import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUNTIME_ERROR
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.jobs.models import JobSnapshot, JobStatus, TrainingJob
from shardgrid.launchers.base import LauncherContext, LauncherResult, LauncherResultStatus
from shardgrid.launchers.ssh import SSHLauncher
from shardgrid.planner.models import ExecutionPlan


def register_logs_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser("logs", help="Show collected or live job logs")
    parser.add_argument("job_id", help="Job ID to inspect")
    parser.add_argument("--worker", help="Filter to a single worker_id")
    parser.add_argument("--rank", type=int, help="Filter to a single rank")
    parser.add_argument(
        "--tail",
        type=_positive_int,
        default=50,
        help="Show the last N lines of each matching log",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_logs_command, command_name="logs")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--tail must be > 0")
    return parsed


def _build_launcher(cluster_config) -> SSHLauncher:
    return SSHLauncher(cluster_config)


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


def _load_execution_plan(job_root: Path) -> ExecutionPlan:
    path = job_root / "plan" / "execution-plan.json"
    if not path.is_file():
        raise RuntimeError(f"execution plan not found: {path}")
    return ExecutionPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_job_status(job_root: Path) -> JobStatus:
    path = job_root / "job-status.json"
    if not path.is_file():
        raise RuntimeError(f"job status not found: {path}")
    return JobStatus.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_launcher_context(args: argparse.Namespace) -> LauncherContext:
    cli_context = args.context
    if cli_context.config is None:
        raise RuntimeError("logs requires a cluster config")
    jobs_root = cli_context.jobs_root or cli_context.config.jobs_root
    job_root = Path(jobs_root) / args.job_id
    if not job_root.is_dir():
        raise RuntimeError(f"job snapshot not found: {job_root}")
    execution_plan = _load_execution_plan(job_root)
    job_status = _load_job_status(job_root)
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
    return LauncherContext(
        job=job,
        execution_plan=execution_plan,
        cluster_state=ResourceManager().build_cluster_state([]),
        snapshot=_snapshot_for_job(job_root, args.job_id),
        job_status=job_status,
        runtime_environment_refs=dict(job_status.runtime_environment_refs),
        backend_config={
            "worker_id": args.worker or "",
            "rank": "" if args.rank is None else str(args.rank),
            "tail": str(args.tail),
        },
    )


def _render_human(result: LauncherResult) -> str:
    if result.blocker and not result.log_results:
        return result.blocker
    lines = [
        f"Job: {result.job_id}",
        f"Overall Status: {result.status.value.upper()}",
        (
            "Job State: "
            f"{result.next_job_state.value.upper()}"
            if result.next_job_state is not None
            else "Job State: PENDING"
        ),
        f"Entries: {len(result.log_results)}",
        f"Selector: {_selector_label(result)}",
    ]
    blocks: list[str] = []
    for item in result.log_results:
        block = [
            (
                f"[worker={item.worker_id} rank={item.rank if item.rank is not None else '-'} "
                f"stage={item.stage or '-'} stream={item.stream or '-'} "
                f"source={item.source or '-'}]"
            ),
            f"Job: {item.job_id or result.job_id}",
            f"Worker: {item.worker_id}",
            f"Rank: {item.rank if item.rank is not None else '-'}",
            f"Stage: {item.stage or '-'}",
            f"Stream: {item.stream or '-'}",
            f"Source: {item.source or '-'}",
            f"Source Location: {item.location or '-'}",
            f"Source Path: {item.source_path or item.location or '-'}",
            f"Status: {item.status.value.upper()}",
        ]
        if item.message:
            block.append(f"Message: {item.message}")
        if item.failure is not None:
            block.append(f"Failure: {item.failure.message}")
        block.append("Content:")
        block.append(item.content)
        blocks.append("\n".join(block).rstrip())
    return "\n".join(lines) + "\n\n" + "\n\n".join(blocks)


def _selector_payload(context: LauncherContext) -> dict[str, object]:
    backend_config = context.backend_config
    return {
        "worker": backend_config.get("worker_id") or None,
        "rank": (
            None
            if not backend_config.get("rank")
            else int(str(backend_config["rank"]))
        ),
        "tail": int(str(backend_config.get("tail", "50"))),
    }


def _selector_label(result: LauncherResult) -> str:
    if not result.log_results:
        return "no-match"
    workers = {item.worker_id for item in result.log_results if item.worker_id}
    ranks = {item.rank for item in result.log_results if item.rank is not None}
    if len(workers) == 1 and len(ranks) == 1:
        return f"worker={next(iter(workers))}, rank={next(iter(ranks))}"
    if len(workers) == 1:
        return f"worker={next(iter(workers))}"
    if len(ranks) == 1:
        return f"rank={next(iter(ranks))}"
    return "whole-job"


def _render_json(result: LauncherResult, context: LauncherContext) -> str:
    payload = {
        "job_id": result.job_id,
        "backend": result.backend,
        "status": result.status.value,
        "job_state": (
            None if result.next_job_state is None else result.next_job_state.value
        ),
        "selector": _selector_payload(context),
        "logs": [item.to_dict() for item in result.log_results],
        "message": result.message,
        "blocker": result.blocker,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def run_logs_command(args: argparse.Namespace) -> int:
    cli_context = getattr(args, "context", None)
    config = getattr(cli_context, "config", None)
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(cli_context, "json_output", False)
    )
    if config is None:
        print(
            "logs requires a cluster config: "
            "shardgrid --config examples/workers.yaml logs JOB --tail 50"
        )
        return EXIT_CONFIG_ERROR
    try:
        context = _load_launcher_context(args)
        launcher = _build_launcher(config)
        result = launcher.logs(context)
    except ValueError as error:
        print(f"logs: {error}")
        return 2
    except Exception as error:  # noqa: BLE001 - surfaced into CLI error
        print(f"logs: {error}")
        return EXIT_RUNTIME_ERROR
    print(_render_json(result, context) if json_output else _render_human(result))
    if result.status in {
        LauncherResultStatus.SUCCESS,
        LauncherResultStatus.PARTIAL,
    }:
        return EXIT_OK
    if result.status is LauncherResultStatus.UNSUPPORTED:
        return EXIT_CONFIG_ERROR
    return EXIT_RUNTIME_ERROR
