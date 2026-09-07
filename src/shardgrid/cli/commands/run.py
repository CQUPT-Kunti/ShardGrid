"""`shardgrid run` CLI shell."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardgrid.cli.commands.train import _resolve_cluster_config
from shardgrid.cli.context import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUNTIME_ERROR
from shardgrid.common.config import ConfigValidationError
from shardgrid.common.enums import JobState
from shardgrid.control.job_manager import JobManager, JobRunResult


@dataclass(frozen=True)
class TrainingEntrypoint:
    entrypoint: Path
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    cluster_config_path: Path | None
    dry_run: bool
    json_output: bool


def register_run_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser("run", help="Run a Python training entrypoint")
    parser.add_argument(
        "--config",
        dest="config",
        default=argparse.SUPPRESS,
        help="Path to cluster configuration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Capture and plan without launching formal training",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.add_argument("entrypoint", help="Python training entrypoint")
    parser.add_argument("entrypoint_args", nargs=argparse.REMAINDER)
    parser.set_defaults(handler=run_run_command, command_name="run")


def _entrypoint_from_args(args: argparse.Namespace) -> TrainingEntrypoint:
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(args.context, "json_output", False)
    )
    return TrainingEntrypoint(
        entrypoint=Path(args.entrypoint),
        argv=tuple(str(item) for item in getattr(args, "entrypoint_args", ())),
        cwd=Path.cwd(),
        environment={},
        cluster_config_path=args.context.config_path,
        dry_run=bool(getattr(args, "dry_run", False)),
        json_output=json_output,
    )


def _payload(entrypoint: TrainingEntrypoint, result: JobRunResult) -> dict[str, object]:
    failure = None if result.status.failure is None else result.status.failure.to_dict()
    snapshot = result.snapshot
    return {
        "command": "run",
        "entrypoint": str(entrypoint.entrypoint),
        "entrypoint_args": list(entrypoint.argv),
        "cwd": str(entrypoint.cwd),
        "dry_run": entrypoint.dry_run,
        "job_id": str(result.job.job_id),
        "state": result.status.state.value,
        "phase": result.status.phase,
        "snapshot_path": None if snapshot is None else snapshot.root_path,
        "failure": failure,
    }


def _render_human(entrypoint: TrainingEntrypoint, result: JobRunResult) -> str:
    payload = _payload(entrypoint, result)
    lines = [
        f"Job: {payload['job_id']}",
        f"Entrypoint: {payload['entrypoint']}",
        "Entrypoint Args: " + " ".join(str(item) for item in payload["entrypoint_args"]),
        f"Dry Run: {'YES' if payload['dry_run'] else 'NO'}",
        f"State: {str(payload['state']).upper()}",
        f"Phase: {payload['phase']}",
    ]
    if payload["snapshot_path"] is not None:
        lines.append(f"Snapshot: {payload['snapshot_path']}")
    failure = result.status.failure
    if failure is not None:
        lines.append(f"Stage: {failure.stage.value}")
        lines.append(f"Failure: {failure.message}")
        if failure.recommended_action:
            lines.append(f"Recommended Action: {failure.recommended_action}")
        artifact_log = failure.runtime_environment.get("artifact_log")
        if artifact_log:
            lines.append(f"Artifact Log: {artifact_log}")
    return "\n".join(lines)


def _exit_code(result: JobRunResult) -> int:
    return EXIT_OK if result.status.state is JobState.COMPLETED else EXIT_RUNTIME_ERROR


def run_run_command(args: argparse.Namespace) -> int:
    entrypoint = _entrypoint_from_args(args)
    try:
        manager = JobManager(_resolve_cluster_config(args))
        result = manager.run_entrypoint(entrypoint)
    except (FileNotFoundError, ConfigValidationError, ValueError) as error:
        print(
            json.dumps({"error": type(error).__name__, "message": str(error)}, sort_keys=True)
            if entrypoint.json_output
            else f"run: {error}"
        )
        return EXIT_CONFIG_ERROR
    except Exception as error:
        print(
            json.dumps({"error": type(error).__name__, "message": str(error)}, sort_keys=True)
            if entrypoint.json_output
            else f"run: {error}"
        )
        return EXIT_RUNTIME_ERROR

    print(
        json.dumps(_payload(entrypoint, result), indent=2, sort_keys=True)
        if entrypoint.json_output
        else _render_human(entrypoint, result)
    )
    return EXIT_OK if entrypoint.dry_run and result.status.failure is None else _exit_code(result)
