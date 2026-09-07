"""`shardgrid status` CLI command (T102)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from shardgrid.cli.context import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUNTIME_ERROR
from shardgrid.common.enums import JobState
from shardgrid.control.status_store import StatusStore
from shardgrid.jobs.models import FailureRecord, JobStatus

_TERMINAL_STATES = {JobState.COMPLETED, JobState.FAILED, JobState.STOPPED}
_WATCH_INTERVAL_SECONDS = 1.0


def register_status_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser("status", help="Show persisted job status")
    parser.add_argument("job_id", help="Job ID to inspect")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll the control-node status store until the job reaches a terminal state",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_status_command, command_name="status")


def _jobs_root(args: argparse.Namespace) -> Path:
    cli_context = args.context
    if cli_context.jobs_root is not None:
        return Path(cli_context.jobs_root)
    if cli_context.config is not None:
        return Path(cli_context.config.jobs_root)
    raise RuntimeError(
        "status requires --config or --jobs-root: "
        "shardgrid --config examples/workers.yaml status JOB"
    )


def _load_status(args: argparse.Namespace) -> JobStatus:
    store = StatusStore(_jobs_root(args))
    try:
        return store.load(args.job_id)
    except FileNotFoundError as error:
        raise RuntimeError(f"job status not found: {error.filename}") from error


def _format_runtime_ref(status: JobStatus, rank: int) -> str:
    return status.runtime_environment_refs.get(str(rank), "pending")


def _format_failure_runtime(failure: FailureRecord | None) -> str:
    if failure is None or not failure.runtime_environment:
        return "unavailable"
    return json.dumps(failure.runtime_environment, sort_keys=True)


def _render_human(status: JobStatus) -> str:
    lines = [
        f"Job: {status.job_id}",
        f"State: {status.state.value.upper()}",
        f"Phase: {status.phase}",
        f"Started: {status.started_at or 'pending'}",
        f"Finished: {status.finished_at or 'pending'}",
        f"Backend: {status.backend or 'pending'}",
        f"Fallback: {'true' if status.fallback_used else 'false'}",
        f"Latest Loss: {status.latest_loss if status.latest_loss is not None else 'pending'}",
        (
            "Loss History: "
            f"{len(status.loss_history)} points; "
            f"latest_index={len(status.loss_history) - 1}"
            if status.loss_history
            else "Loss History: pending"
        ),
        f"Checkpoint: {status.checkpoint_ref or 'pending'}",
        (
            "Final Metrics: " + json.dumps(status.final_metrics, sort_keys=True)
            if status.final_metrics
            else "Final Metrics: pending"
        ),
        "Assignments:",
    ]
    if status.assignments:
        for assignment in status.assignments:
            lines.append(
                "  "
                f"{assignment.worker_id} | rank={assignment.rank} | "
                f"stage={assignment.stage or '-'} | "
                f"runtime={_format_runtime_ref(status, assignment.rank)}"
            )
    else:
        lines.append("  pending")
    if status.failure is not None:
        lines.extend(
            [
                "Failure:",
                f"  Failure Stage: {status.failure.stage.value}",
                f"  Failure Worker: {status.failure.worker_id or 'unavailable'}",
                f"  Failure Host: {status.failure.host}",
                f"  Failure Command: {status.failure.command or 'unavailable'}",
                (
                    "  Failure Exit Code: "
                    f"{status.failure.exit_code}"
                    if status.failure.exit_code is not None
                    else "  Failure Exit Code: unavailable"
                ),
                f"  Failure Runtime: {_format_failure_runtime(status.failure)}",
                f"  Failure Message: {status.failure.message}",
                f"  Recommended Action: {status.failure.recommended_action}",
            ]
        )
    return "\n".join(lines)


def _render_json(status: JobStatus) -> str:
    return json.dumps(status.to_dict(), indent=2, sort_keys=True)


def _emit(status: JobStatus, *, json_output: bool) -> None:
    print(_render_json(status) if json_output else _render_human(status))


def run_status_command(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(args.context, "json_output", False)
    )
    try:
        while True:
            status = _load_status(args)
            _emit(status, json_output=json_output)
            if not args.watch or status.state in _TERMINAL_STATES:
                return EXIT_OK
            time.sleep(_WATCH_INTERVAL_SECONDS)
    except RuntimeError as error:
        print(f"status: {error}")
        if "requires --config or --jobs-root" in str(error):
            return EXIT_CONFIG_ERROR
        return EXIT_RUNTIME_ERROR
    except KeyboardInterrupt:
        return EXIT_OK
