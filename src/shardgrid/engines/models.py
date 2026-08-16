"""Engine evaluation and compatibility data models for ShardGrid."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, cast

from shardgrid.common.enums import BackendStatus, FailureStage
from shardgrid.common.models import EngineName, JobId, as_engine_name, as_job_id


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
class ParallelEngineCandidate:
    engine_id: str
    name: EngineName
    version: str | None = None
    source: str | None = None
    status: BackendStatus = BackendStatus.NOT_CHECKED
    capabilities: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    compatibility_report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelEngineCandidate":
        return cls(
            engine_id=str(data["engine_id"]),
            name=as_engine_name(str(data["name"])),
            version=data.get("version"),
            source=data.get("source"),
            status=BackendStatus(data.get("status", BackendStatus.NOT_CHECKED.value)),
            capabilities=[str(item) for item in data.get("capabilities", [])],
            limitations=[str(item) for item in data.get("limitations", [])],
            compatibility_report_path=data.get("compatibility_report_path"),
        )


@dataclass(frozen=True)
class CompatibilitySpikeReport:
    report_id: str
    component: str
    stage: FailureStage
    machines_tested: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    logs_path: str | None = None
    status: BackendStatus = BackendStatus.NOT_CHECKED
    blockers: list[str] = field(default_factory=list)
    decision: str | None = None
    recommended_next_action: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if self.status in {BackendStatus.FAILED, BackendStatus.BLOCKED} and not self.decision:
            raise ValueError("failed or blocked spike report requires a decision")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompatibilitySpikeReport":
        return cls(
            report_id=str(data["report_id"]),
            component=str(data["component"]),
            stage=FailureStage(data["stage"]),
            machines_tested=[str(item) for item in data.get("machines_tested", [])],
            versions={str(key): str(value) for key, value in data.get("versions", {}).items()},
            commands=[str(item) for item in data.get("commands", [])],
            results=[str(item) for item in data.get("results", [])],
            logs_path=data.get("logs_path"),
            status=BackendStatus(data.get("status", BackendStatus.NOT_CHECKED.value)),
            blockers=[str(item) for item in data.get("blockers", [])],
            decision=data.get("decision"),
            recommended_next_action=data.get("recommended_next_action"),
            created_at=data.get("created_at"),
        )


@dataclass(frozen=True)
class ParallelPlan:
    parallel_plan_id: str
    engine: EngineName
    engine_plan_path: str | None = None
    model_name: str = ""
    world_size: int = 1
    stages: list[str] = field(default_factory=list)
    requirements: dict[str, str] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelPlan":
        return cls(
            parallel_plan_id=str(data["parallel_plan_id"]),
            engine=as_engine_name(str(data["engine"])),
            engine_plan_path=data.get("engine_plan_path"),
            model_name=str(data.get("model_name", "")),
            world_size=int(data.get("world_size", 1)),
            stages=[str(item) for item in data.get("stages", [])],
            requirements={
                str(key): str(value)
                for key, value in data.get("requirements", {}).items()
            },
            limitations=[str(item) for item in data.get("limitations", [])],
        )


@dataclass(frozen=True)
class GPUShare:
    worker_id: str
    gpu_index: int = 0
    total_memory_mb: int = 0
    allocated_memory_mb: int = 0
    free_memory_mb: int = 0
    memory_slices: list[int] = field(default_factory=list)
    compute_slices: list[int] = field(default_factory=list)
    owner_job_ids: list[JobId] = field(default_factory=list)
    isolation_status: str = "not_enabled"

    def __post_init__(self) -> None:
        if self.gpu_index < 0:
            raise ValueError("gpu_index must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPUShare":
        return cls(
            worker_id=str(data["worker_id"]),
            gpu_index=int(data.get("gpu_index", 0)),
            total_memory_mb=int(data.get("total_memory_mb", 0)),
            allocated_memory_mb=int(data.get("allocated_memory_mb", 0)),
            free_memory_mb=int(data.get("free_memory_mb", 0)),
            memory_slices=[int(item) for item in data.get("memory_slices", [])],
            compute_slices=[int(item) for item in data.get("compute_slices", [])],
            owner_job_ids=[as_job_id(str(item)) for item in data.get("owner_job_ids", [])],
            isolation_status=str(data.get("isolation_status", "not_enabled")),
        )
