"""Planning and launch data models for ShardGrid."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, cast

from shardgrid.common.models import (
    BackendName,
    EngineName,
    JobId,
    WorkerId,
    as_backend_name,
    as_engine_name,
    as_job_id,
    as_worker_id,
)

AssignmentStatus = str



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
class WorkerAssignment:
    worker_id: WorkerId
    rank: int
    local_rank: int = 0
    stage: str | None = None
    stage_metadata_ref: str | None = None
    estimated_peak_training_memory: int | None = None
    communication_edges: list[str] = field(default_factory=list)
    gpu_index: int = 0
    host: str | None = None
    machine_id: str | None = None
    physical_os: str | None = None
    runtime_os: str | None = None
    runtime: str | None = None
    runtime_distro: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    python_executable: str | None = None
    launch_command: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    status: AssignmentStatus = "pending"
    pid: int | None = None
    log_path: str | None = None

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be >= 0")
        if self.local_rank < 0:
            raise ValueError("local_rank must be >= 0")
        if (
            self.estimated_peak_training_memory is not None
            and self.estimated_peak_training_memory < 0
        ):
            raise ValueError("estimated_peak_training_memory must be >= 0")
        if self.gpu_index < 0:
            raise ValueError("gpu_index must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerAssignment":
        return cls(
            worker_id=as_worker_id(str(data["worker_id"])),
            rank=int(data["rank"]),
            local_rank=int(data.get("local_rank", 0)),
            stage=data.get("stage"),
            stage_metadata_ref=data.get("stage_metadata_ref"),
            estimated_peak_training_memory=(
                None
                if data.get("estimated_peak_training_memory") is None
                else int(data["estimated_peak_training_memory"])
            ),
            communication_edges=[
                str(item) for item in data.get("communication_edges", [])
            ],
            gpu_index=int(data.get("gpu_index", 0)),
            host=data.get("host"),
            machine_id=data.get("machine_id"),
            physical_os=data.get("physical_os"),
            runtime_os=data.get("runtime_os"),
            runtime=data.get("runtime"),
            runtime_distro=data.get("runtime_distro"),
            conda_environment=data.get("conda_environment"),
            conda_prefix=data.get("conda_prefix"),
            python_executable=data.get("python_executable"),
            launch_command=data.get("launch_command"),
            environment={
                str(key): str(value)
                for key, value in data.get("environment", {}).items()
            },
            status=str(data.get("status", "pending")),
            pid=(None if data.get("pid") is None else int(data["pid"])),
            log_path=data.get("log_path"),
        )


@dataclass(frozen=True)
class MasterMetadata:
    address: str
    port: int

    def __post_init__(self) -> None:
        if self.port <= 0:
            raise ValueError("master.port must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MasterMetadata":
        return cls(address=str(data["address"]), port=int(data["port"]))


@dataclass(frozen=True)
class ExecutionPlan:
    job_id: JobId
    engine: EngineName
    backend: BackendName
    world_size: int
    master: MasterMetadata
    workers: list[WorkerAssignment]
    model_profile_ref: str | None = None
    candidate_evaluation_ref: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    python_executable: str | None = None
    placement_reason: str | None = None
    parallel_plan_ref: str | None = None
    original_engine_plan_ref: str | None = None
    distributed_checkpoint_ref: str | None = None
    consolidated_model_ref: str | None = None
    reload_validation_ref: str | None = None
    snapshot_ref: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.world_size != len(self.workers):
            raise ValueError("world_size must equal number of worker assignments")
        ranks = [worker.rank for worker in self.workers]
        if len(ranks) != len(set(ranks)):
            raise ValueError("worker assignments must not contain duplicate ranks")
        if any(worker.local_rank != 0 for worker in self.workers):
            raise ValueError("Stage A-C worker assignments must use local_rank = 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionPlan":
        return cls(
            job_id=as_job_id(str(data["job_id"])),
            engine=as_engine_name(str(data["engine"])),
            backend=as_backend_name(str(data["backend"])),
            world_size=int(data["world_size"]),
            master=MasterMetadata.from_dict(data["master"]),
            workers=[WorkerAssignment.from_dict(item) for item in data.get("workers", [])],
            model_profile_ref=data.get("model_profile_ref"),
            candidate_evaluation_ref=data.get("candidate_evaluation_ref"),
            conda_environment=data.get("conda_environment"),
            conda_prefix=data.get("conda_prefix"),
            python_executable=data.get("python_executable"),
            placement_reason=data.get("placement_reason"),
            parallel_plan_ref=data.get("parallel_plan_ref"),
            original_engine_plan_ref=data.get("original_engine_plan_ref"),
            distributed_checkpoint_ref=data.get("distributed_checkpoint_ref"),
            consolidated_model_ref=data.get("consolidated_model_ref"),
            reload_validation_ref=data.get("reload_validation_ref"),
            snapshot_ref=data.get("snapshot_ref"),
            environment={
                str(key): str(value)
                for key, value in data.get("environment", {}).items()
            },
            labels={
                str(key): str(value)
                for key, value in data.get("labels", {}).items()
            },
        )


@dataclass(frozen=True)
class PlatformAdapterState:
    adapter_id: str
    platform: str
    shell: str
    path_rules: str
    environment_manager: str = "conda"
    conda_executable: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    supports_bootstrap: bool = False
    supports_probe: bool = False
    manual_action_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlatformAdapterState":
        return cls(
            adapter_id=str(data["adapter_id"]),
            platform=str(data["platform"]),
            shell=str(data["shell"]),
            path_rules=str(data["path_rules"]),
            environment_manager=str(data.get("environment_manager", "conda")),
            conda_executable=data.get("conda_executable"),
            conda_environment=data.get("conda_environment"),
            conda_prefix=data.get("conda_prefix"),
            supports_bootstrap=bool(data.get("supports_bootstrap", False)),
            supports_probe=bool(data.get("supports_probe", False)),
            manual_action_rules=[str(item) for item in data.get("manual_action_rules", [])],
        )
