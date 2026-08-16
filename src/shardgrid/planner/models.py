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
    gpu_index: int = 0
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
            gpu_index=int(data.get("gpu_index", 0)),
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
    placement_reason: str | None = None
    parallel_plan_ref: str | None = None
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
            placement_reason=data.get("placement_reason"),
            parallel_plan_ref=data.get("parallel_plan_ref"),
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
            supports_bootstrap=bool(data.get("supports_bootstrap", False)),
            supports_probe=bool(data.get("supports_probe", False)),
            manual_action_rules=[str(item) for item in data.get("manual_action_rules", [])],
        )
