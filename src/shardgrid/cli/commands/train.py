"""`shardgrid train` CLI command (T105)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from shardgrid.artifacts.metadata import load_snapshot_metadata
from shardgrid.cli.context import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUNTIME_ERROR
from shardgrid.common.config import ClusterConfig, ConfigValidationError, load_cluster_config
from shardgrid.common.enums import JobState
from shardgrid.control.job_manager import JobManager, JobRunResult
from shardgrid.planner.execution_plan import build_execution_plan_audit_payload

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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and snapshot the final execution plan without launching ranks",
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
    audit = _audit_payload(result)
    return {
        **audit,
        "state": result.status.state.value,
        "phase": result.status.phase,
        "snapshot_path": None if snapshot is None else snapshot.root_path,
        "original_parallel_plan_path": None
        if snapshot is None
        else str(Path(snapshot.plan_path) / "original-parallel-plan.json"),
        "original_parallel_plan_yaml_path": None
        if snapshot is None
        else str(Path(snapshot.plan_path) / "original-parallel-plan.yaml"),
        "execution_plan_path": None
        if snapshot is None
        else str(Path(snapshot.plan_path) / "execution-plan.json"),
        "execution_plan_yaml_path": None
        if snapshot is None
        else str(Path(snapshot.plan_path) / "execution-plan.yaml"),
        "snapshot_metadata_path": None
        if snapshot is None
        else str(Path(snapshot.diagnostics_path) / "snapshot-metadata.json"),
        "snapshot_metadata_yaml_path": None
        if snapshot is None
        else str(Path(snapshot.diagnostics_path) / "snapshot-metadata.yaml"),
        "status_path": None
        if snapshot is None
        else str(Path(snapshot.diagnostics_path) / "job-status.json"),
        "failure": None if result.status.failure is None else result.status.failure.to_dict(),
    }


def _audit_payload(result: JobRunResult) -> dict[str, object]:
    metadata_path = None
    if result.snapshot is not None:
        metadata_path = Path(result.snapshot.diagnostics_path) / "snapshot-metadata.json"
    if metadata_path is not None and metadata_path.is_file():
        metadata = load_snapshot_metadata(metadata_path)
        if metadata.execution_plan_audit is not None:
            return metadata.execution_plan_audit
    if result.execution_plan is None:
        return {
            "job_id": str(result.job.job_id),
            "dry_run": False,
            "engine": None,
            "backend": str(result.status.backend or result.job.backend_preference),
            "world_size": result.job.requested_world_size,
            "master": None,
            "placement_reason": None,
            "placement": [],
            "assignments": [],
            "launch_metadata": {},
            "original_plan": {},
            "fallback": {"status": "NONE", "label": "NONE", "reason": None},
            "planning": {},
            "labels": {},
        }
    return build_execution_plan_audit_payload(
        result.execution_plan,
        parallel_plan=result.parallel_plan,
    )


def _render_human(result: JobRunResult) -> str:
    payload = _payload(result)
    lines = [
        f"Job: {payload['job_id']}",
        f"Dry Run: {'YES' if payload['dry_run'] else 'NO'}",
        f"Plan Mode: {payload.get('plan_mode')}",
        f"Engine: {payload['engine']}",
        f"Backend: {payload['backend']}",
        f"State: {str(payload['state']).upper()}",
        f"Phase: {payload['phase']}",
        f"World Size: {payload['world_size']}",
    ]
    master = payload.get("master")
    if isinstance(master, dict):
        lines.append(f"Master: {master.get('address')}:{master.get('port')}")
    fallback = payload.get("fallback")
    if isinstance(fallback, dict):
        lines.append(
            "Fallback: "
            f"{fallback.get('label')} ({fallback.get('status')})"
        )
        if fallback.get("reason"):
            lines.append(f"Fallback Reason: {fallback.get('reason')}")
    original_plan = payload.get("original_plan")
    if isinstance(original_plan, dict):
        for label, key in (
            ("Parallel Plan Ref", "parallel_plan_ref"),
            ("Original Engine Plan", "original_engine_plan_ref"),
            ("Model Profile Ref", "model_profile_ref"),
            ("Candidate Evaluation Ref", "candidate_evaluation_ref"),
        ):
            if original_plan.get(key):
                lines.append(f"{label}: {original_plan.get(key)}")
    if payload.get("placement_reason"):
        lines.append(f"Placement Reason: {payload['placement_reason']}")
    planning = payload.get("planning")
    if isinstance(planning, dict):
        if planning.get("partition_source"):
            lines.append(f"Partition Source: {planning.get('partition_source')}")
        if planning.get("selected_candidate_id"):
            lines.append(f"Selected Candidate: {planning.get('selected_candidate_id')}")
        if planning.get("selected_worker_count"):
            lines.append(f"Selected Worker Count: {planning.get('selected_worker_count')}")
        attempted = planning.get("attempted_worker_counts")
        if isinstance(attempted, list) and attempted:
            lines.append(
                "Attempted Worker Counts: " + ", ".join(str(item) for item in attempted)
            )
        selected_workers = planning.get("selected_workers")
        if isinstance(selected_workers, list) and selected_workers:
            lines.append("Selected Workers: " + ", ".join(str(item) for item in selected_workers))
        if planning.get("total_cross_worker_communication_bytes"):
            lines.append(
                "Cross-Worker Communication Bytes: "
                f"{planning.get('total_cross_worker_communication_bytes')}"
            )
    if payload["snapshot_path"] is not None:
        lines.append(f"Snapshot: {payload['snapshot_path']}")
        lines.append(f"Original Parallel Plan: {payload['original_parallel_plan_path']}")
        lines.append(
            f"Original Parallel Plan YAML: {payload['original_parallel_plan_yaml_path']}"
        )
        lines.append(f"Execution Plan: {payload['execution_plan_path']}")
        lines.append(f"Execution Plan YAML: {payload['execution_plan_yaml_path']}")
        lines.append(f"Snapshot Metadata: {payload['snapshot_metadata_path']}")
        lines.append(f"Snapshot Metadata YAML: {payload['snapshot_metadata_yaml_path']}")
        lines.append(f"Status File: {payload['status_path']}")
    placement = payload.get("placement")
    if isinstance(placement, list) and placement:
        lines.append("Placement:")
        for item in placement:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  "
                f"Stage {item.get('stage')} -> Worker {item.get('worker_id')} "
                f"-> rank {item.get('rank')} -> GPU {item.get('gpu_index')} "
                f"({item.get('host')}, {item.get('runtime_os')})"
            )
    assignments = payload.get("assignments")
    if isinstance(assignments, list) and assignments:
        lines.append("Assignments:")
        for item in assignments:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  "
                f"{item.get('worker_id')} rank={item.get('rank')} "
                f"local_rank={item.get('local_rank')} stage={item.get('stage')} "
                f"device={item.get('device')} machine={item.get('machine_id')} "
                f"runtime={item.get('runtime')} "
                f"python={item.get('python_executable')}"
            )
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
    dry_run = bool(getattr(args, "dry_run", False))
    try:
        manager = JobManager(_resolve_cluster_config(args))
        result = manager.run(args.config_path, dry_run=dry_run)
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
    return EXIT_OK if dry_run and result.status.failure is None else _exit_code(result)
