"""`shardgrid train` CLI command (T105)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from shardgrid.cli.context import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUNTIME_ERROR
from shardgrid.common.config import ClusterConfig, ConfigValidationError, load_cluster_config
from shardgrid.common.enums import JobState
from shardgrid.control.job_manager import JobManager, JobRunResult

_DEFAULT_CLUSTER_CONFIGS = ("shardgrid.yaml", "examples/workers.yaml")


def register_train_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser("train", help="Run a training job")
    parser.add_argument("config_path", help="Training config path")
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_train_command, command_name="train")


def _resolve_cluster_config(args: argparse.Namespace) -> ClusterConfig:
    cli_context = args.context
    config = cli_context.config
    if config is None:
        for candidate in _DEFAULT_CLUSTER_CONFIGS:
            path = Path(candidate)
            if path.is_file():
                config = load_cluster_config(path)
                break
    if config is None:
        raise FileNotFoundError(
            "train requires a cluster config: "
            "use --config or provide examples/workers.yaml"
        )
    runtime = replace(
        config.runtime,
        conda_environment=cli_context.conda_env or config.runtime.conda_environment,
        conda_prefix=(
            str(cli_context.conda_prefix)
            if cli_context.conda_prefix
            else config.runtime.conda_prefix
        ),
    )
    return replace(
        config,
        jobs_root=cli_context.jobs_root or config.jobs_root,
        runtime=runtime,
    )


def _payload(result: JobRunResult) -> dict[str, object]:
    snapshot = result.snapshot
    return {
        "job_id": str(result.job.job_id),
        "backend": str(result.status.backend or result.job.backend_preference),
        "state": result.status.state.value,
        "phase": result.status.phase,
        "snapshot_path": None if snapshot is None else snapshot.root_path,
        "execution_plan_path": None
        if snapshot is None
        else str(Path(snapshot.plan_path) / "execution-plan.json"),
        "status_path": None
        if snapshot is None
        else str(Path(snapshot.diagnostics_path) / "job-status.json"),
        "failure": None if result.status.failure is None else result.status.failure.to_dict(),
    }


def _render_human(result: JobRunResult) -> str:
    payload = _payload(result)
    lines = [
        f"Job: {payload['job_id']}",
        f"Backend: {payload['backend']}",
        f"State: {str(payload['state']).upper()}",
        f"Phase: {payload['phase']}",
    ]
    if payload["snapshot_path"] is not None:
        lines.append(f"Snapshot: {payload['snapshot_path']}")
        lines.append(f"Execution Plan: {payload['execution_plan_path']}")
        lines.append(f"Status File: {payload['status_path']}")
    if result.status.failure is not None:
        lines.append(f"Failure Stage: {result.status.failure.stage.value}")
        lines.append(f"Failure: {result.status.failure.message}")
    return "\n".join(lines)


def _exit_code(result: JobRunResult) -> int:
    return EXIT_OK if result.status.state is JobState.COMPLETED else EXIT_RUNTIME_ERROR


def run_train_command(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(args.context, "json_output", False)
    )
    try:
        manager = JobManager(_resolve_cluster_config(args))
        result = manager.run(args.config_path)
    except (FileNotFoundError, ConfigValidationError, ValueError) as error:
        print(
            json.dumps({"error": type(error).__name__, "message": str(error)}, sort_keys=True)
            if json_output
            else f"train: {error}"
        )
        return EXIT_CONFIG_ERROR
    except Exception as error:
        print(
            json.dumps({"error": type(error).__name__, "message": str(error)}, sort_keys=True)
            if json_output
            else f"train: {error}"
        )
        return EXIT_RUNTIME_ERROR

    print(
        json.dumps(_payload(result), indent=2, sort_keys=True)
        if json_output
        else _render_human(result)
    )
    return _exit_code(result)
