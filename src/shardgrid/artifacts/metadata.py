"""Snapshot metadata persistence for replay and diagnosis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from shardgrid.common.logging import redact_mapping
from shardgrid.common.serialization import validate_execution_plan, validate_job_status
from shardgrid.engines.models import ParallelPlan
from shardgrid.jobs.models import JobSnapshot, JobStatus, TrainingJob
from shardgrid.planner.models import ExecutionPlan
from shardgrid.resources.models import NetworkState
from shardgrid.workers.environment_report import (
    EnvironmentReport,
    load_environment_report,
    validate_environment_report,
    write_environment_report,
)

_CONFIG_FILE = "training-config.json"
_ORIGINAL_PLAN_FILE = "original-parallel-plan.json"
_EXECUTION_PLAN_FILE = "execution-plan.json"
_NETWORK_STATE_FILE = "network-state.json"
_CHECKPOINT_METADATA_FILE = "checkpoint-metadata.json"
_JOB_STATUS_FILE = "job-status.json"
_FAILURE_FILE = "failure.json"
_MANIFEST_FILE = "snapshot-metadata.json"


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class SnapshotMetadata:
    job_id: str
    snapshot_root: str
    config_path: str
    original_parallel_plan_path: str
    execution_plan_path: str
    environment_report_path: str
    network_state_path: str
    checkpoint_metadata_path: str
    job_status_path: str
    failure_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotMetadata":
        return cls(
            job_id=str(data["job_id"]),
            snapshot_root=str(data["snapshot_root"]),
            config_path=str(data["config_path"]),
            original_parallel_plan_path=str(data["original_parallel_plan_path"]),
            execution_plan_path=str(data["execution_plan_path"]),
            environment_report_path=str(data["environment_report_path"]),
            network_state_path=str(data["network_state_path"]),
            checkpoint_metadata_path=str(data["checkpoint_metadata_path"]),
            job_status_path=str(data["job_status_path"]),
            failure_path=data.get("failure_path"),
        )


def write_snapshot_metadata(
    *,
    snapshot: JobSnapshot,
    job: TrainingJob,
    config: Mapping[str, Any],
    parallel_plan: ParallelPlan,
    execution_plan: ExecutionPlan,
    environment_report: EnvironmentReport,
    network_state: NetworkState,
    job_status: JobStatus,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    secrets: Sequence[str] = (),
) -> SnapshotMetadata:
    _validate_inputs(
        snapshot=snapshot,
        job=job,
        parallel_plan=parallel_plan,
        execution_plan=execution_plan,
        environment_report=environment_report,
        network_state=network_state,
        job_status=job_status,
        checkpoint_metadata=checkpoint_metadata,
    )
    snapshot_root = Path(snapshot.root_path).resolve()
    paths = {
        "config": _contained_path(snapshot.config_path, _CONFIG_FILE, snapshot_root),
        "parallel_plan": _contained_path(
            snapshot.plan_path, _ORIGINAL_PLAN_FILE, snapshot_root
        ),
        "execution_plan": _contained_path(
            snapshot.plan_path, _EXECUTION_PLAN_FILE, snapshot_root
        ),
        "environment_report": None,
        "network_state": _contained_path(
            snapshot.diagnostics_path, _NETWORK_STATE_FILE, snapshot_root
        ),
        "checkpoint": _contained_path(
            snapshot.checkpoint_path, _CHECKPOINT_METADATA_FILE, snapshot_root
        ),
        "job_status": _contained_path(
            snapshot.diagnostics_path, _JOB_STATUS_FILE, snapshot_root
        ),
        "failure": _contained_path(snapshot.diagnostics_path, _FAILURE_FILE, snapshot_root),
        "manifest": _contained_path(snapshot.diagnostics_path, _MANIFEST_FILE, snapshot_root),
    }

    _write_json(paths["config"], redact_mapping(_serialize(dict(config)), secrets))
    _write_json(paths["parallel_plan"], redact_mapping(parallel_plan.to_dict(), secrets))
    validate_execution_plan(execution_plan)
    _write_json(paths["execution_plan"], redact_mapping(execution_plan.to_dict(), secrets))

    write_environment_report(environment_report, snapshot.environment_path)
    environment_report_path = _find_environment_report(snapshot.environment_path)
    validate_environment_report(load_environment_report(environment_report_path))
    paths["environment_report"] = environment_report_path

    _write_json(paths["network_state"], redact_mapping(network_state.to_dict(), secrets))
    validate_job_status(job_status)
    _write_json(paths["job_status"], redact_mapping(job_status.to_dict(), secrets))

    checkpoint_payload = _checkpoint_payload(
        job_status=job_status,
        checkpoint_metadata=checkpoint_metadata,
        snapshot=snapshot,
    )
    _write_json(paths["checkpoint"], redact_mapping(checkpoint_payload, secrets))

    failure_path = None
    if job_status.failure is not None:
        failure_path = str(paths["failure"])
        _write_json(paths["failure"], redact_mapping(job_status.failure.to_dict(), secrets))

    metadata = SnapshotMetadata(
        job_id=str(job.job_id),
        snapshot_root=str(snapshot_root),
        config_path=str(paths["config"]),
        original_parallel_plan_path=str(paths["parallel_plan"]),
        execution_plan_path=str(paths["execution_plan"]),
        environment_report_path=str(paths["environment_report"]),
        network_state_path=str(paths["network_state"]),
        checkpoint_metadata_path=str(paths["checkpoint"]),
        job_status_path=str(paths["job_status"]),
        failure_path=failure_path,
    )
    _write_json(paths["manifest"], metadata.to_dict())
    validate_snapshot_metadata(metadata.to_dict())
    return metadata


def load_snapshot_metadata(path: str | Path) -> SnapshotMetadata:
    return SnapshotMetadata.from_dict(json.loads(Path(path).read_text()))


def validate_snapshot_metadata(payload: Mapping[str, Any]) -> None:
    required = {
        "job_id",
        "snapshot_root",
        "config_path",
        "original_parallel_plan_path",
        "execution_plan_path",
        "environment_report_path",
        "network_state_path",
        "checkpoint_metadata_path",
        "job_status_path",
    }
    missing = sorted(field for field in required if not payload.get(field))
    if missing:
        raise ValueError(f"snapshot metadata missing required fields: {', '.join(missing)}")

    snapshot_root = Path(str(payload["snapshot_root"])).resolve(strict=False)
    for key in (
        "config_path",
        "original_parallel_plan_path",
        "execution_plan_path",
        "environment_report_path",
        "network_state_path",
        "checkpoint_metadata_path",
        "job_status_path",
    ):
        _ensure_contained(Path(str(payload[key])), snapshot_root)
    if payload.get("failure_path"):
        _ensure_contained(Path(str(payload["failure_path"])), snapshot_root)


def _validate_inputs(
    *,
    snapshot: JobSnapshot,
    job: TrainingJob,
    parallel_plan: ParallelPlan,
    execution_plan: ExecutionPlan,
    environment_report: EnvironmentReport,
    network_state: NetworkState,
    job_status: JobStatus,
    checkpoint_metadata: Mapping[str, Any] | None,
) -> None:
    if snapshot.job_id != job.job_id:
        raise ValueError("snapshot job_id must match training job")
    if execution_plan.job_id != job.job_id:
        raise ValueError("execution plan job_id must match training job")
    if job_status.job_id != job.job_id:
        raise ValueError("job status job_id must match training job")
    if parallel_plan.world_size != job.requested_world_size:
        raise ValueError("parallel plan world_size must match training job")
    if execution_plan.world_size != job.requested_world_size:
        raise ValueError("execution plan world_size must match training job")
    if checkpoint_metadata is None and job_status.state.value == "completed":
        raise ValueError("completed job metadata requires checkpoint metadata")


def _contained_path(root: str, filename: str, snapshot_root: Path) -> Path:
    path = Path(root).resolve() / filename
    _ensure_contained(path, snapshot_root)
    return path


def _ensure_contained(path: Path, snapshot_root: Path) -> None:
    candidate = path.resolve(strict=False)
    if snapshot_root not in candidate.parents and candidate != snapshot_root:
        raise ValueError("metadata path escaped snapshot root")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True))


def _find_environment_report(root: str | Path) -> Path:
    reports = sorted(Path(root).glob("*-environment-report.json"))
    if len(reports) != 1:
        raise ValueError("expected exactly one environment report in snapshot")
    return reports[0].resolve()


def _checkpoint_payload(
    *,
    job_status: JobStatus,
    checkpoint_metadata: Mapping[str, Any] | None,
    snapshot: JobSnapshot,
) -> dict[str, Any]:
    payload = {
        "job_id": str(job_status.job_id),
        "status": job_status.state.value,
        "checkpoint_ref": job_status.checkpoint_ref,
        "checkpoint_dir": snapshot.checkpoint_path,
    }
    if checkpoint_metadata is not None:
        payload.update(_serialize(dict(checkpoint_metadata)))
    return payload
