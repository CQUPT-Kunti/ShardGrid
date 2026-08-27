"""Training job and lifecycle data models for ShardGrid."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast

from shardgrid.common.enums import FailureStage, JobState
from shardgrid.common.models import (
    BackendName,
    JobId,
    WorkerId,
    as_backend_name,
    as_job_id,
    as_worker_id,
)
from shardgrid.planner.models import WorkerAssignment

_ALLOWED_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.CREATED: {JobState.PROBING, JobState.STOPPING, JobState.FAILED},
    JobState.PROBING: {JobState.PLANNING, JobState.FAILED, JobState.STOPPING},
    JobState.PLANNING: {JobState.SNAPSHOTTING, JobState.FAILED, JobState.STOPPING},
    JobState.SNAPSHOTTING: {JobState.DISTRIBUTING, JobState.FAILED, JobState.STOPPING},
    JobState.DISTRIBUTING: {JobState.LAUNCHING, JobState.FAILED, JobState.STOPPING},
    JobState.LAUNCHING: {JobState.RENDEZVOUS, JobState.FAILED, JobState.STOPPING},
    JobState.RENDEZVOUS: {JobState.TRAINING, JobState.FAILED, JobState.STOPPING},
    JobState.TRAINING: {JobState.CHECKPOINTING, JobState.FAILED, JobState.STOPPING},
    JobState.CHECKPOINTING: {JobState.COMPLETED, JobState.FAILED, JobState.STOPPING},
    JobState.STOPPING: {JobState.STOPPED, JobState.FAILED},
    JobState.COMPLETED: set(),
    JobState.FAILED: set(),
    JobState.STOPPED: set(),
}


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    return None if value is None else float(value)


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return None if value is None else int(value)


def _serialize(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class FailureRecord:
    stage: FailureStage
    host: str
    worker_id: WorkerId | None = None
    command: str | None = None
    exit_code: int | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    runtime_environment: dict[str, str] = field(default_factory=dict)
    python_executable: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    message: str = ""
    recommended_action: str = ""
    retryable: bool = False
    manual_action_required: bool = False

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("failure message is required")
        if not self.recommended_action:
            raise ValueError("recommended_action is required")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureRecord":
        return cls(
            stage=FailureStage(data["stage"]),
            host=str(data["host"]),
            worker_id=(
                None
                if data.get("worker_id") is None
                else as_worker_id(str(data["worker_id"]))
            ),
            command=data.get("command"),
            exit_code=_optional_int(data, "exit_code"),
            stdout_path=data.get("stdout_path"),
            stderr_path=data.get("stderr_path"),
            runtime_environment={
                str(key): str(value)
                for key, value in data.get("runtime_environment", {}).items()
            },
            python_executable=data.get("python_executable"),
            conda_environment=data.get("conda_environment"),
            conda_prefix=data.get("conda_prefix"),
            message=str(data["message"]),
            recommended_action=str(data["recommended_action"]),
            retryable=bool(data.get("retryable", False)),
            manual_action_required=bool(data.get("manual_action_required", False)),
        )


@dataclass(frozen=True)
class TrainingResult:
    job_id: JobId
    forward_success: bool = False
    activation_transfer_success: bool = False
    loss_success: bool = False
    backward_success: bool = False
    gradient_transfer_success: bool = False
    optimizer_step_success: bool = False
    parameters_changed: bool = False
    initial_loss: float | None = None
    final_loss: float | None = None
    loss_decrease_percent: float | None = None
    checkpoint_path: str | None = None
    backend_label: str | None = None
    diagnostics_path: str | None = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingResult":
        return cls(
            job_id=as_job_id(str(data["job_id"])),
            forward_success=bool(data.get("forward_success", False)),
            activation_transfer_success=bool(data.get("activation_transfer_success", False)),
            loss_success=bool(data.get("loss_success", False)),
            backward_success=bool(data.get("backward_success", False)),
            gradient_transfer_success=bool(data.get("gradient_transfer_success", False)),
            optimizer_step_success=bool(data.get("optimizer_step_success", False)),
            parameters_changed=bool(data.get("parameters_changed", False)),
            initial_loss=_optional_float(data, "initial_loss"),
            final_loss=_optional_float(data, "final_loss"),
            loss_decrease_percent=_optional_float(data, "loss_decrease_percent"),
            checkpoint_path=data.get("checkpoint_path"),
            backend_label=data.get("backend_label"),
            diagnostics_path=data.get("diagnostics_path"),
            status=str(data.get("status", "pending")),
        )


@dataclass(frozen=True)
class TrainingJob:
    job_id: JobId
    config_path: str
    model: str
    requested_world_size: int
    backend_preference: BackendName
    runtime_environment_ref: str | None = None
    state: JobState = JobState.CREATED
    created_at: str | None = None
    updated_at: str | None = None
    snapshot_path: str | None = None
    execution_plan_path: str | None = None
    status_path: str | None = None

    def __post_init__(self) -> None:
        if self.requested_world_size <= 0:
            raise ValueError("requested_world_size must be > 0")
        if not self.config_path.strip():
            raise ValueError("config_path must be a non-empty string")
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            self.runtime_environment_ref is not None
            and not self.runtime_environment_ref.strip()
        ):
            raise ValueError("runtime_environment_ref must be a non-empty string")

    def transition_to(self, next_state: JobState) -> "TrainingJob":
        if next_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"invalid job state transition: {self.state.value} -> {next_state.value}"
            )
        return replace(
            self,
            state=next_state,
            updated_at=datetime.now(tz=UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJob":
        return cls(
            job_id=as_job_id(str(data["job_id"])),
            config_path=str(data["config_path"]),
            model=str(data["model"]),
            requested_world_size=int(data["requested_world_size"]),
            backend_preference=as_backend_name(str(data["backend_preference"])),
            runtime_environment_ref=data.get("runtime_environment_ref"),
            state=JobState(data.get("state", JobState.CREATED.value)),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            snapshot_path=data.get("snapshot_path"),
            execution_plan_path=data.get("execution_plan_path"),
            status_path=data.get("status_path"),
        )


@dataclass(frozen=True)
class JobSnapshot:
    job_id: JobId
    root_path: str
    code_path: str
    config_path: str
    plan_path: str
    logs_path: str
    environment_path: str
    checkpoint_path: str
    diagnostics_path: str
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobSnapshot":
        return cls(
            job_id=as_job_id(str(data["job_id"])),
            root_path=str(data["root_path"]),
            code_path=str(data["code_path"]),
            config_path=str(data["config_path"]),
            plan_path=str(data["plan_path"]),
            logs_path=str(data["logs_path"]),
            environment_path=str(data["environment_path"]),
            checkpoint_path=str(data["checkpoint_path"]),
            diagnostics_path=str(data["diagnostics_path"]),
            created_at=data.get("created_at"),
        )


@dataclass(frozen=True)
class EnvironmentSnapshot:
    snapshot_id: str
    scope: str
    environment_manager: str = "conda"
    conda_executable: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    python_executable: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_version: str | None = None
    components: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentSnapshot":
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            scope=str(data["scope"]),
            environment_manager=str(data.get("environment_manager", "conda")),
            conda_executable=data.get("conda_executable"),
            conda_environment=data.get("conda_environment"),
            conda_prefix=data.get("conda_prefix"),
            python_executable=data.get("python_executable"),
            python_version=data.get("python_version"),
            torch_version=data.get("torch_version"),
            torch_cuda_version=data.get("torch_cuda_version"),
            cuda_version=data.get("cuda_version"),
            components={str(key): str(value) for key, value in data.get("components", {}).items()},
        )


@dataclass(frozen=True)
class JobStatus:
    job_id: JobId
    state: JobState
    phase: str
    workers: list[WorkerId] = field(default_factory=list)
    assignments: list[WorkerAssignment] = field(default_factory=list)
    runtime_environment_refs: dict[str, str] = field(default_factory=dict)
    latest_loss: float | None = None
    loss_history: list[float] = field(default_factory=list)
    final_metrics: dict[str, float] = field(default_factory=dict)
    backend: BackendName | None = None
    fallback_used: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    failure: FailureRecord | None = None
    checkpoint_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.phase.strip():
            raise ValueError("phase must be a non-empty string")
        if any(not math.isfinite(value) for value in self.loss_history):
            raise ValueError("loss_history must contain only finite values")
        if self.latest_loss is not None and not math.isfinite(self.latest_loss):
            raise ValueError("latest_loss must be finite")
        if any(not math.isfinite(value) for value in self.final_metrics.values()):
            raise ValueError("final_metrics must contain only finite values")
        if any(
            not key.strip() or not value.strip()
            for key, value in self.runtime_environment_refs.items()
        ):
            raise ValueError("runtime_environment_refs must use non-empty keys and values")
        assignment_workers = [assignment.worker_id for assignment in self.assignments]
        if self.workers and self.assignments and self.workers != assignment_workers:
            raise ValueError(
                "workers must match assignment worker order when assignments are present"
            )
        if self.assignments:
            ranks = [assignment.rank for assignment in self.assignments]
            if len(ranks) != len(set(ranks)):
                raise ValueError("assignments must not contain duplicate ranks")
            expected_refs = {str(assignment.rank) for assignment in self.assignments}
            unexpected_refs = set(self.runtime_environment_refs) - expected_refs
            if unexpected_refs:
                raise ValueError("runtime_environment_refs keys must be assignment ranks")
        if self.state is JobState.FAILED and self.failure is None:
            raise ValueError("failed job status requires a failure record")
        if self.state is JobState.COMPLETED:
            if self.checkpoint_ref is None:
                raise ValueError("completed job status requires checkpoint_ref")
            if not self.final_metrics or "final_loss" not in self.final_metrics:
                raise ValueError("completed job status requires final_metrics with final_loss")

    def transition_to(
        self,
        next_state: JobState,
        *,
        phase: str | None = None,
        failure: FailureRecord | None = None,
        checkpoint_ref: str | None = None,
        final_metrics: dict[str, float] | None = None,
        finished_at: str | None = None,
    ) -> "JobStatus":
        if next_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"invalid job status transition: {self.state.value} -> {next_state.value}"
            )
        return replace(
            self,
            state=next_state,
            phase=self.phase if phase is None else phase,
            failure=failure if failure is not None else self.failure,
            checkpoint_ref=checkpoint_ref if checkpoint_ref is not None else self.checkpoint_ref,
            final_metrics=final_metrics if final_metrics is not None else self.final_metrics,
            finished_at=finished_at if finished_at is not None else self.finished_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobStatus":
        return cls(
            job_id=as_job_id(str(data["job_id"])),
            state=JobState(data["state"]),
            phase=str(data["phase"]),
            workers=[as_worker_id(str(item)) for item in data.get("workers", [])],
            assignments=[
                WorkerAssignment.from_dict(item) for item in data.get("assignments", [])
            ],
            runtime_environment_refs={
                str(key): str(value)
                for key, value in data.get("runtime_environment_refs", {}).items()
            },
            latest_loss=_optional_float(data, "latest_loss"),
            loss_history=[float(item) for item in data.get("loss_history", [])],
            final_metrics={
                str(key): float(value)
                for key, value in data.get("final_metrics", {}).items()
            },
            backend=(
                None
                if data.get("backend") is None
                else as_backend_name(str(data["backend"]))
            ),
            fallback_used=bool(data.get("fallback_used", False)),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            failure=(
                None
                if data.get("failure") is None
                else FailureRecord.from_dict(data["failure"])
            ),
            checkpoint_ref=data.get("checkpoint_ref"),
        )
