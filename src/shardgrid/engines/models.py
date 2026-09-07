"""Engine evaluation and compatibility data models for ShardGrid."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, cast

from shardgrid.common.enums import BackendStatus, FailureStage, SerializableStrEnum
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


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return None if value is None else int(value)


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    return None if value is None else float(value)


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


class PartitionSupportStatus(SerializableStrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"


class EstimateKind(SerializableStrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class BoundaryTensorSpec:
    name: str
    shape: tuple[int | str, ...] = ()
    dtype: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("boundary tensor name must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundaryTensorSpec":
        return cls(
            name=str(data["name"]),
            shape=tuple(data.get("shape", ())),
            dtype=data.get("dtype"),
        )


@dataclass(frozen=True)
class PartitionBoundary:
    boundary_id: str
    source_module: str
    target_module: str
    parameter_owners: tuple[str, ...] = ()
    shared_parameter_names: tuple[str, ...] = ()
    boundary_tensors: tuple[BoundaryTensorSpec, ...] = ()
    forward_dependencies: tuple[str, ...] = ()
    backward_dependencies: tuple[str, ...] = ()
    status: PartitionSupportStatus = PartitionSupportStatus.SUPPORTED
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.boundary_id.strip():
            raise ValueError("boundary_id must be a non-empty string")
        if not self.source_module.strip():
            raise ValueError("source_module must be a non-empty string")
        if not self.target_module.strip():
            raise ValueError("target_module must be a non-empty string")
        if self.status is not PartitionSupportStatus.SUPPORTED and not self.reasons:
            raise ValueError("unsupported or infeasible boundaries require reasons")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartitionBoundary":
        return cls(
            boundary_id=str(data["boundary_id"]),
            source_module=str(data["source_module"]),
            target_module=str(data["target_module"]),
            parameter_owners=tuple(str(item) for item in data.get("parameter_owners", ())),
            shared_parameter_names=tuple(
                str(item) for item in data.get("shared_parameter_names", ())
            ),
            boundary_tensors=tuple(
                BoundaryTensorSpec.from_dict(item)
                for item in data.get("boundary_tensors", ())
            ),
            forward_dependencies=tuple(
                str(item) for item in data.get("forward_dependencies", ())
            ),
            backward_dependencies=tuple(
                str(item) for item in data.get("backward_dependencies", ())
            ),
            status=PartitionSupportStatus(
                data.get("status", PartitionSupportStatus.SUPPORTED.value)
            ),
            reasons=tuple(str(item) for item in data.get("reasons", ())),
        )


@dataclass(frozen=True)
class AutomaticPartitionSupport:
    engine_id: str
    status: PartitionSupportStatus = PartitionSupportStatus.SUPPORTED
    supported_runtime: str | None = None
    supported_backends: tuple[str, ...] = ()
    boundaries: tuple[PartitionBoundary, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.engine_id.strip():
            raise ValueError("engine_id must be a non-empty string")
        if self.status is not PartitionSupportStatus.SUPPORTED and not self.reasons:
            raise ValueError("unsupported or infeasible partition support requires reasons")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutomaticPartitionSupport":
        return cls(
            engine_id=str(data["engine_id"]),
            status=PartitionSupportStatus(
                data.get("status", PartitionSupportStatus.SUPPORTED.value)
            ),
            supported_runtime=data.get("supported_runtime"),
            supported_backends=tuple(
                str(item) for item in data.get("supported_backends", ())
            ),
            boundaries=tuple(
                PartitionBoundary.from_dict(item)
                for item in data.get("boundaries", ())
            ),
            reasons=tuple(str(item) for item in data.get("reasons", ())),
        )


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shape: tuple[int | str, ...] = ()
    dtype: str | None = None
    estimated_bytes: int | None = None
    estimate_kind: EstimateKind = EstimateKind.UNKNOWN
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tensor name must be a non-empty string")
        if self.estimated_bytes is not None and self.estimated_bytes < 0:
            raise ValueError("estimated_bytes must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorMetadata":
        return cls(
            name=str(data["name"]),
            shape=tuple(data.get("shape", ())),
            dtype=data.get("dtype"),
            estimated_bytes=(
                None
                if data.get("estimated_bytes") is None
                else int(data["estimated_bytes"])
            ),
            estimate_kind=EstimateKind(
                data.get("estimate_kind", EstimateKind.UNKNOWN.value)
            ),
            source=data.get("source"),
        )


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    parameter_bytes: int | None = None
    gradient_bytes: int | None = None
    optimizer_bytes: int | None = None
    activation_bytes: int | None = None
    temporary_bytes: int | None = None
    runtime_overhead_bytes: int | None = None
    communication_buffer_bytes: int | None = None
    estimated_peak_bytes: int | None = None
    safety_headroom_bytes: int = 0
    planner_required_bytes: int | None = None
    estimate_kind: EstimateKind = EstimateKind.UNKNOWN
    source: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "parameter_bytes",
            "gradient_bytes",
            "optimizer_bytes",
            "activation_bytes",
            "temporary_bytes",
            "runtime_overhead_bytes",
            "communication_buffer_bytes",
            "estimated_peak_bytes",
            "planner_required_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if self.safety_headroom_bytes < 0:
            raise ValueError("safety_headroom_bytes must be >= 0")
        if (
            self.estimated_peak_bytes is not None
            and self.planner_required_bytes is None
        ):
            object.__setattr__(
                self,
                "planner_required_bytes",
                self.estimated_peak_bytes + self.safety_headroom_bytes,
            )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingMemoryEstimate":
        return cls(
            parameter_bytes=_optional_int(data, "parameter_bytes"),
            gradient_bytes=_optional_int(data, "gradient_bytes"),
            optimizer_bytes=_optional_int(data, "optimizer_bytes"),
            activation_bytes=_optional_int(data, "activation_bytes"),
            temporary_bytes=_optional_int(data, "temporary_bytes"),
            runtime_overhead_bytes=_optional_int(data, "runtime_overhead_bytes"),
            communication_buffer_bytes=_optional_int(data, "communication_buffer_bytes"),
            estimated_peak_bytes=_optional_int(data, "estimated_peak_bytes"),
            safety_headroom_bytes=int(data.get("safety_headroom_bytes", 0)),
            planner_required_bytes=_optional_int(data, "planner_required_bytes"),
            estimate_kind=EstimateKind(
                data.get("estimate_kind", EstimateKind.UNKNOWN.value)
            ),
            source=data.get("source"),
            notes=tuple(str(item) for item in data.get("notes", ())),
        )


@dataclass(frozen=True)
class ModuleProfile:
    module_id: str
    module_path: str
    module_type: str
    parameter_names: tuple[str, ...] = ()
    parameter_count: int = 0
    parameter_bytes: int = 0
    trainable_parameter_count: int = 0
    trainable_parameter_bytes: int = 0
    input_tensors: tuple[TensorMetadata, ...] = ()
    output_tensors: tuple[TensorMetadata, ...] = ()
    memory: TrainingMemoryEstimate = TrainingMemoryEstimate()

    def __post_init__(self) -> None:
        if not self.module_id.strip():
            raise ValueError("module_id must be a non-empty string")
        if not self.module_path.strip():
            raise ValueError("module_path must be a non-empty string")
        if not self.module_type.strip():
            raise ValueError("module_type must be a non-empty string")
        for field_name in (
            "parameter_count",
            "parameter_bytes",
            "trainable_parameter_count",
            "trainable_parameter_bytes",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleProfile":
        return cls(
            module_id=str(data["module_id"]),
            module_path=str(data["module_path"]),
            module_type=str(data["module_type"]),
            parameter_names=tuple(str(item) for item in data.get("parameter_names", ())),
            parameter_count=int(data.get("parameter_count", 0)),
            parameter_bytes=int(data.get("parameter_bytes", 0)),
            trainable_parameter_count=int(data.get("trainable_parameter_count", 0)),
            trainable_parameter_bytes=int(data.get("trainable_parameter_bytes", 0)),
            input_tensors=tuple(
                TensorMetadata.from_dict(item)
                for item in data.get("input_tensors", ())
            ),
            output_tensors=tuple(
                TensorMetadata.from_dict(item)
                for item in data.get("output_tensors", ())
            ),
            memory=TrainingMemoryEstimate.from_dict(data.get("memory", {})),
        )


@dataclass(frozen=True)
class CommunicationEdge:
    source_module_id: str
    target_module_id: str
    activation: tuple[TensorMetadata, ...] = ()
    gradient: tuple[TensorMetadata, ...] = ()
    estimate_kind: EstimateKind = EstimateKind.UNKNOWN
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.source_module_id.strip():
            raise ValueError("source_module_id must be a non-empty string")
        if not self.target_module_id.strip():
            raise ValueError("target_module_id must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommunicationEdge":
        return cls(
            source_module_id=str(data["source_module_id"]),
            target_module_id=str(data["target_module_id"]),
            activation=tuple(
                TensorMetadata.from_dict(item)
                for item in data.get("activation", ())
            ),
            gradient=tuple(
                TensorMetadata.from_dict(item)
                for item in data.get("gradient", ())
            ),
            estimate_kind=EstimateKind(
                data.get("estimate_kind", EstimateKind.UNKNOWN.value)
            ),
            source=data.get("source"),
        )


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    engine_id: str
    model_name: str
    modules: tuple[ModuleProfile, ...]
    module_order: tuple[str, ...]
    partition_support: AutomaticPartitionSupport | None = None
    communication_edges: tuple[CommunicationEdge, ...] = ()
    shared_parameter_groups: tuple[tuple[str, ...], ...] = ()
    required_runtime: str | None = None
    required_backends: tuple[str, ...] = ()
    total_memory: TrainingMemoryEstimate = TrainingMemoryEstimate()
    evidence_paths: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if not self.engine_id.strip():
            raise ValueError("engine_id must be a non-empty string")
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not self.modules:
            raise ValueError("modules must not be empty")
        if len(self.modules) != len(self.module_order):
            raise ValueError("module_order must contain one entry per module")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        return cls(
            profile_id=str(data["profile_id"]),
            engine_id=str(data["engine_id"]),
            model_name=str(data["model_name"]),
            modules=tuple(
                ModuleProfile.from_dict(item) for item in data.get("modules", ())
            ),
            module_order=tuple(str(item) for item in data.get("module_order", ())),
            partition_support=(
                None
                if data.get("partition_support") is None
                else AutomaticPartitionSupport.from_dict(data["partition_support"])
            ),
            communication_edges=tuple(
                CommunicationEdge.from_dict(item)
                for item in data.get("communication_edges", ())
            ),
            shared_parameter_groups=tuple(
                tuple(str(name) for name in group)
                for group in data.get("shared_parameter_groups", ())
            ),
            required_runtime=data.get("required_runtime"),
            required_backends=tuple(
                str(item) for item in data.get("required_backends", ())
            ),
            total_memory=TrainingMemoryEstimate.from_dict(
                data.get("total_memory", {})
            ),
            evidence_paths=tuple(str(item) for item in data.get("evidence_paths", ())),
            diagnostics=tuple(str(item) for item in data.get("diagnostics", ())),
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
class ParallelPlanPlacement:
    worker_id: str
    rank: int
    machine_id: str | None = None
    gpu_index: int = 0
    usable_memory_before_bytes: int | None = None
    remaining_memory_bytes: int | None = None
    utilization_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self.worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if self.rank < 0:
            raise ValueError("rank must be >= 0")
        if self.gpu_index < 0:
            raise ValueError("gpu_index must be >= 0")
        for field_name in (
            "usable_memory_before_bytes",
            "remaining_memory_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if self.utilization_ratio is not None and self.utilization_ratio < 0:
            raise ValueError("utilization_ratio must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelPlanPlacement":
        return cls(
            worker_id=str(data["worker_id"]),
            rank=int(data["rank"]),
            machine_id=data.get("machine_id"),
            gpu_index=int(data.get("gpu_index", 0)),
            usable_memory_before_bytes=_optional_int(
                data, "usable_memory_before_bytes"
            ),
            remaining_memory_bytes=_optional_int(data, "remaining_memory_bytes"),
            utilization_ratio=_optional_float(data, "utilization_ratio"),
        )


@dataclass(frozen=True)
class ParallelPlanStage:
    stage_id: str
    rank: int
    module_ids: tuple[str, ...]
    module_paths: tuple[str, ...]
    start_index: int
    stop_index: int
    boundary_before_id: str | None = None
    boundary_after_id: str | None = None
    parameter_names_or_ranges: tuple[str, ...] = ()
    parameter_bytes: int = 0
    gradient_bytes: int | None = None
    activation_bytes: int | None = None
    estimated_compute_units: int = 0
    estimated_peak_training_memory: TrainingMemoryEstimate = TrainingMemoryEstimate()
    required_runtime: str | None = None
    required_backends: tuple[str, ...] = ()
    placement: ParallelPlanPlacement | None = None

    def __post_init__(self) -> None:
        if not self.stage_id.strip():
            raise ValueError("stage_id must be a non-empty string")
        if self.rank < 0:
            raise ValueError("rank must be >= 0")
        if not self.module_ids:
            raise ValueError("module_ids must not be empty")
        if not self.module_paths:
            raise ValueError("module_paths must not be empty")
        if len(self.module_ids) != len(self.module_paths):
            raise ValueError("module_ids and module_paths must align")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise ValueError("stage indices must define a non-empty range")
        for field_name in ("parameter_bytes", "estimated_compute_units"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")
        for field_name in ("gradient_bytes", "activation_bytes"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelPlanStage":
        return cls(
            stage_id=str(data["stage_id"]),
            rank=int(data["rank"]),
            module_ids=tuple(str(item) for item in data.get("module_ids", ())),
            module_paths=tuple(str(item) for item in data.get("module_paths", ())),
            start_index=int(data["start_index"]),
            stop_index=int(data["stop_index"]),
            boundary_before_id=data.get("boundary_before_id"),
            boundary_after_id=data.get("boundary_after_id"),
            parameter_names_or_ranges=tuple(
                str(item) for item in data.get("parameter_names_or_ranges", ())
            ),
            parameter_bytes=int(data.get("parameter_bytes", 0)),
            gradient_bytes=_optional_int(data, "gradient_bytes"),
            activation_bytes=_optional_int(data, "activation_bytes"),
            estimated_compute_units=int(data.get("estimated_compute_units", 0)),
            estimated_peak_training_memory=TrainingMemoryEstimate.from_dict(
                data.get("estimated_peak_training_memory", {})
            ),
            required_runtime=data.get("required_runtime"),
            required_backends=tuple(
                str(item) for item in data.get("required_backends", ())
            ),
            placement=(
                None
                if data.get("placement") is None
                else ParallelPlanPlacement.from_dict(data["placement"])
            ),
        )


@dataclass(frozen=True)
class ParallelPlanCommunicationEdge:
    source_stage_id: str
    target_stage_id: str
    source_module_id: str
    target_module_id: str
    activation: tuple[TensorMetadata, ...] = ()
    gradient: tuple[TensorMetadata, ...] = ()
    activation_bytes: int | None = None
    gradient_bytes: int | None = None
    estimated_bytes_per_step: int | None = None
    estimate_kind: EstimateKind = EstimateKind.UNKNOWN
    bandwidth_mbps: float | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "source_stage_id",
            "target_stage_id",
            "source_module_id",
            "target_module_id",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "activation_bytes",
            "gradient_bytes",
            "estimated_bytes_per_step",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        for field_name in ("bandwidth_mbps", "latency_ms"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if self.activation_bytes is None:
            object.__setattr__(
                self,
                "activation_bytes",
                sum(
                    tensor.estimated_bytes or 0 for tensor in self.activation
                ) or None,
            )
        if self.gradient_bytes is None:
            object.__setattr__(
                self,
                "gradient_bytes",
                sum(
                    tensor.estimated_bytes or 0 for tensor in self.gradient
                ) or None,
            )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelPlanCommunicationEdge":
        return cls(
            source_stage_id=str(data["source_stage_id"]),
            target_stage_id=str(data["target_stage_id"]),
            source_module_id=str(data["source_module_id"]),
            target_module_id=str(data["target_module_id"]),
            activation=tuple(
                TensorMetadata.from_dict(item) for item in data.get("activation", ())
            ),
            gradient=tuple(
                TensorMetadata.from_dict(item) for item in data.get("gradient", ())
            ),
            activation_bytes=_optional_int(data, "activation_bytes"),
            gradient_bytes=_optional_int(data, "gradient_bytes"),
            estimated_bytes_per_step=_optional_int(data, "estimated_bytes_per_step"),
            estimate_kind=EstimateKind(
                data.get("estimate_kind", EstimateKind.UNKNOWN.value)
            ),
            bandwidth_mbps=_optional_float(data, "bandwidth_mbps"),
            latency_ms=_optional_float(data, "latency_ms"),
        )


@dataclass(frozen=True)
class ParallelPlanAttempt:
    worker_count: int
    worker_ids: tuple[str, ...]
    candidate_id: str | None = None
    status: str = "unknown"
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("worker_count must be > 0")
        if not self.worker_ids:
            raise ValueError("worker_ids must not be empty")
        if not self.status.strip():
            raise ValueError("status must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelPlanAttempt":
        return cls(
            worker_count=int(data["worker_count"]),
            worker_ids=tuple(str(item) for item in data.get("worker_ids", ())),
            candidate_id=data.get("candidate_id"),
            status=str(data.get("status", "unknown")),
            reasons=tuple(str(item) for item in data.get("reasons", ())),
        )


@dataclass(frozen=True)
class ParallelPlanProvenance:
    partition_source: str = "automatic"
    model_profile_id: str | None = None
    selected_candidate_id: str | None = None
    selected_worker_count: int | None = None
    attempted_worker_counts: tuple[int, ...] = ()
    attempts: tuple[ParallelPlanAttempt, ...] = ()
    partition_algorithm: str | None = None
    total_cross_worker_communication_bytes: int | None = None
    selected_reason: str | None = None
    fallback_reason: str | None = None
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.partition_source.strip():
            raise ValueError("partition_source must be a non-empty string")
        if self.selected_worker_count is not None and self.selected_worker_count <= 0:
            raise ValueError("selected_worker_count must be > 0")
        if (
            self.total_cross_worker_communication_bytes is not None
            and self.total_cross_worker_communication_bytes < 0
        ):
            raise ValueError("total_cross_worker_communication_bytes must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelPlanProvenance":
        return cls(
            partition_source=str(data.get("partition_source", "automatic")),
            model_profile_id=data.get("model_profile_id"),
            selected_candidate_id=data.get("selected_candidate_id"),
            selected_worker_count=_optional_int(data, "selected_worker_count"),
            attempted_worker_counts=tuple(
                int(item) for item in data.get("attempted_worker_counts", ())
            ),
            attempts=tuple(
                ParallelPlanAttempt.from_dict(item) for item in data.get("attempts", ())
            ),
            partition_algorithm=data.get("partition_algorithm"),
            total_cross_worker_communication_bytes=_optional_int(
                data, "total_cross_worker_communication_bytes"
            ),
            selected_reason=data.get("selected_reason"),
            fallback_reason=data.get("fallback_reason"),
            rejection_reasons=tuple(str(item) for item in data.get("rejection_reasons", ())),
        )


@dataclass(frozen=True)
class ParallelPlan:
    parallel_plan_id: str
    engine: EngineName
    engine_plan_path: str | None = None
    model_name: str = ""
    world_size: int = 1
    stages: list[str] = field(default_factory=list)
    partition_source: str | None = None
    model_profile_id: str | None = None
    selected_candidate_id: str | None = None
    candidate_evaluation_ref: str | None = None
    stage_metadata: list[ParallelPlanStage] = field(default_factory=list)
    communication_edges: list[ParallelPlanCommunicationEdge] = field(default_factory=list)
    planning_provenance: ParallelPlanProvenance | None = None
    requirements: dict[str, str] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.stage_metadata and len(self.stage_metadata) != self.world_size:
            raise ValueError("stage_metadata count must match world_size")
        if self.stages and self.stage_metadata:
            metadata_ids = [stage.stage_id for stage in self.stage_metadata]
            if self.stages != metadata_ids:
                raise ValueError("stages must match stage_metadata stage ids")

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
            partition_source=data.get("partition_source"),
            model_profile_id=data.get("model_profile_id"),
            selected_candidate_id=data.get("selected_candidate_id"),
            candidate_evaluation_ref=data.get("candidate_evaluation_ref"),
            stage_metadata=[
                ParallelPlanStage.from_dict(item)
                for item in data.get("stage_metadata", [])
            ],
            communication_edges=[
                ParallelPlanCommunicationEdge.from_dict(item)
                for item in data.get("communication_edges", [])
            ],
            planning_provenance=(
                None
                if data.get("planning_provenance") is None
                else ParallelPlanProvenance.from_dict(data["planning_provenance"])
            ),
            requirements={
                str(key): str(value)
                for key, value in data.get("requirements", {}).items()
            },
            limitations=[str(item) for item in data.get("limitations", [])],
        )


@dataclass(frozen=True)
class ProfileResult:
    """Engine profiler output contract (T066 ``profile``)."""

    engine_id: str
    status: BackendStatus
    model_profile: ModelProfile | None = None
    evidence_paths: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileResult":
        return cls(
            engine_id=str(data["engine_id"]),
            status=BackendStatus(data.get("status", BackendStatus.NOT_CHECKED.value)),
            model_profile=(
                None
                if data.get("model_profile") is None
                else ModelProfile.from_dict(data["model_profile"])
            ),
            evidence_paths=[str(item) for item in data.get("evidence_paths", [])],
            diagnostics=[str(item) for item in data.get("diagnostics", [])],
            notes=[str(item) for item in data.get("notes", [])],
        )


@dataclass(frozen=True)
class EnginePreparation:
    """Engine runtime-preparation contract (T066 ``prepare``)."""

    engine_id: str
    status: BackendStatus
    snapshot_artifact_refs: list[str] = field(default_factory=list)
    manual_actions: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnginePreparation":
        return cls(
            engine_id=str(data["engine_id"]),
            status=BackendStatus(data.get("status", BackendStatus.NOT_CHECKED.value)),
            snapshot_artifact_refs=[str(item) for item in data.get("snapshot_artifact_refs", [])],
            manual_actions=[str(item) for item in data.get("manual_actions", [])],
            diagnostics=[str(item) for item in data.get("diagnostics", [])],
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
