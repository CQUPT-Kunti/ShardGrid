"""Planner constraints and feasibility checks for T108.

This module defines the stable constraint vocabulary that later planning tasks
use. It deliberately does not generate partitions or search placements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from shardgrid.common.enums import Health, RuntimeOS, SerializableStrEnum
from shardgrid.common.models import WorkerId
from shardgrid.engines.models import AutomaticPartitionSupport, PartitionBoundary, PartitionSupportStatus
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


class FeasibilityStatus(SerializableStrEnum):
    FEASIBLE = "feasible"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class ConstraintViolation:
    code: str
    reason: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("constraint violation code must be a non-empty string")
        if not self.reason.strip():
            raise ValueError("constraint violation reason must be a non-empty string")


@dataclass(frozen=True)
class WorkerEligibilityRequirements:
    required_runtime_os: RuntimeOS | None = None
    allowed_healths: tuple[Health, ...] = (Health.HEALTHY,)
    required_environment_manager: str = "conda"
    require_enabled: bool = True
    require_gpu: bool = True
    require_cuda_runtime: bool = True
    require_torch_runtime: bool = True
    required_backends: tuple[str, ...] = ()
    require_runtime_environment: bool = True


@dataclass(frozen=True)
class MemoryConstraint:
    reserved_memory_mb: int = 0
    usable_memory_mb: int | None = None
    estimated_stage_memory_mb: int | None = None

    def __post_init__(self) -> None:
        if self.reserved_memory_mb < 0:
            raise ValueError("reserved_memory_mb must be >= 0")
        if self.usable_memory_mb is not None and self.usable_memory_mb < 0:
            raise ValueError("usable_memory_mb must be >= 0")
        if (
            self.estimated_stage_memory_mb is not None
            and self.estimated_stage_memory_mb < 0
        ):
            raise ValueError("estimated_stage_memory_mb must be >= 0")

    @classmethod
    def from_worker(
        cls,
        worker: WorkerResource,
        *,
        reserved_memory_mb: int = 0,
        estimated_stage_memory_mb: int | None = None,
    ) -> "MemoryConstraint":
        usable = None
        if worker.gpu_total_memory is not None:
            usable = max(worker.gpu_total_memory - reserved_memory_mb, 0)
        elif worker.gpu_free_memory is not None:
            usable = max(worker.gpu_free_memory - reserved_memory_mb, 0)
        return cls(
            reserved_memory_mb=reserved_memory_mb,
            usable_memory_mb=usable,
            estimated_stage_memory_mb=estimated_stage_memory_mb,
        )


@dataclass(frozen=True)
class CommunicationRequirement:
    source_stage_id: str
    target_stage_id: str
    required_backend: str | None = None
    require_bidirectional: bool = True

    def __post_init__(self) -> None:
        if not self.source_stage_id.strip():
            raise ValueError("source_stage_id must be a non-empty string")
        if not self.target_stage_id.strip():
            raise ValueError("target_stage_id must be a non-empty string")


@dataclass(frozen=True)
class PlacementRequirements:
    stage_ids: tuple[str, ...]
    allowed_stage_to_worker_ratio: tuple[int, int] = (1, 1)
    require_unique_physical_hosts: bool = True
    communication: tuple[CommunicationRequirement, ...] = ()

    def __post_init__(self) -> None:
        if not self.stage_ids:
            raise ValueError("stage_ids must not be empty")
        if len(set(self.stage_ids)) != len(self.stage_ids):
            raise ValueError("stage_ids must be unique")
        minimum, maximum = self.allowed_stage_to_worker_ratio
        if minimum <= 0 or maximum <= 0 or minimum > maximum:
            raise ValueError("allowed_stage_to_worker_ratio must be a valid positive range")


@dataclass(frozen=True)
class ParallelPlanningRequirements:
    engine_id: str
    worker: WorkerEligibilityRequirements = WorkerEligibilityRequirements()
    partition_support: AutomaticPartitionSupport | None = None
    placement: PlacementRequirements | None = None
    stage_memory_by_stage: Mapping[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.engine_id.strip():
            raise ValueError("engine_id must be a non-empty string")


@dataclass(frozen=True)
class WorkerEligibilityResult:
    worker_id: WorkerId
    eligible: bool
    status: FeasibilityStatus
    usable_memory_mb: int | None
    violations: tuple[ConstraintViolation, ...] = ()


@dataclass(frozen=True)
class PlacementFeasibilityResult:
    feasible: bool
    status: FeasibilityStatus
    violations: tuple[ConstraintViolation, ...] = ()


def evaluate_worker_eligibility(
    worker: WorkerResource,
    requirements: WorkerEligibilityRequirements,
    *,
    estimated_stage_memory_mb: int | None = None,
    reserved_memory_mb: int = 0,
) -> WorkerEligibilityResult:
    violations: list[ConstraintViolation] = []
    if requirements.require_enabled and not worker.enabled:
        violations.append(
            ConstraintViolation("worker_disabled", "worker is disabled in configuration")
        )
    if worker.health not in requirements.allowed_healths:
        violations.append(
            ConstraintViolation(
                "worker_health",
                f"worker health {worker.health.value} is not eligible",
            )
        )
    if requirements.required_runtime_os is not None and worker.runtime_os is not requirements.required_runtime_os:
        violations.append(
            ConstraintViolation(
                "runtime_os",
                f"worker runtime {worker.runtime_os.value} does not match required runtime {requirements.required_runtime_os.value}",
            )
        )
    if worker.environment_manager != requirements.required_environment_manager:
        violations.append(
            ConstraintViolation(
                "environment_manager",
                f"worker environment manager {worker.environment_manager!r} does not match required {requirements.required_environment_manager!r}",
            )
        )
    if requirements.require_runtime_environment and not (worker.conda_environment or worker.conda_prefix):
        violations.append(
            ConstraintViolation(
                "runtime_environment",
                "worker runtime environment reference is missing",
            )
        )
    if requirements.require_gpu and not worker.gpu_name:
        violations.append(
            ConstraintViolation("gpu_missing", "worker GPU evidence is missing")
        )
    if requirements.require_cuda_runtime and not worker.cuda_version:
        violations.append(
            ConstraintViolation("cuda_missing", "worker CUDA/runtime evidence is missing")
        )
    if requirements.require_torch_runtime and not worker.torch_version:
        violations.append(
            ConstraintViolation("torch_missing", "worker torch runtime evidence is missing")
        )

    for backend in requirements.required_backends:
        if backend == "nccl" and not worker.nccl_available:
            violations.append(
                ConstraintViolation("backend_nccl", "worker does not expose NCCL")
            )
        elif backend == "gloo" and not worker.gloo_available:
            violations.append(
                ConstraintViolation("backend_gloo", "worker does not expose Gloo")
            )

    memory = MemoryConstraint.from_worker(
        worker,
        reserved_memory_mb=reserved_memory_mb,
        estimated_stage_memory_mb=estimated_stage_memory_mb,
    )
    if estimated_stage_memory_mb is not None:
        if memory.usable_memory_mb is None:
            violations.append(
                ConstraintViolation(
                    "memory_unknown",
                    "worker usable GPU memory is unknown for stage fit validation",
                )
            )
        elif estimated_stage_memory_mb > memory.usable_memory_mb:
            violations.append(
                ConstraintViolation(
                    "memory_fit",
                    f"estimated stage memory {estimated_stage_memory_mb} MB exceeds usable GPU memory {memory.usable_memory_mb} MB",
                )
            )

    status = FeasibilityStatus.FEASIBLE if not violations else FeasibilityStatus.INFEASIBLE
    return WorkerEligibilityResult(
        worker_id=worker.worker_id,
        eligible=not violations,
        status=status,
        usable_memory_mb=memory.usable_memory_mb,
        violations=tuple(violations),
    )


def validate_partition_boundary(
    boundary: PartitionBoundary,
    support: AutomaticPartitionSupport,
) -> PlacementFeasibilityResult:
    violations: list[ConstraintViolation] = []
    if support.status is not PartitionSupportStatus.SUPPORTED:
        violations.extend(
            ConstraintViolation("engine_support", reason) for reason in support.reasons
        )
        return PlacementFeasibilityResult(
            feasible=False,
            status=_support_status_to_feasibility(support.status),
            violations=tuple(violations),
        )
    if boundary.status is not PartitionSupportStatus.SUPPORTED:
        violations.extend(
            ConstraintViolation("boundary_status", reason) for reason in boundary.reasons
        )
    if boundary.shared_parameter_names and any(
        "shared" in reason.lower() or "tied" in reason.lower()
        for reason in boundary.reasons
    ):
        violations.append(
            ConstraintViolation(
                "shared_parameters",
                "shared or tied parameters cross the boundary without engine support",
            )
        )
    if not boundary.boundary_tensors:
        violations.append(
            ConstraintViolation(
                "boundary_tensors",
                "boundary tensor metadata is missing",
            )
        )
    if not boundary.forward_dependencies or not boundary.backward_dependencies:
        violations.append(
            ConstraintViolation(
                "boundary_dependencies",
                "forward or backward dependency metadata is missing",
            )
        )
    return PlacementFeasibilityResult(
        feasible=not violations,
        status=FeasibilityStatus.FEASIBLE if not violations else FeasibilityStatus.UNSUPPORTED,
        violations=tuple(violations),
    )


def validate_placement_feasibility(
    assignments: Mapping[str, str],
    *,
    workers: Sequence[WorkerResource],
    requirements: PlacementRequirements,
    worker_requirements: WorkerEligibilityRequirements,
    stage_memory_by_stage: Mapping[str, int | None] | None = None,
    network_state: NetworkState | None = None,
    reserved_memory_mb: int = 0,
) -> PlacementFeasibilityResult:
    violations: list[ConstraintViolation] = []
    stage_memory_by_stage = stage_memory_by_stage or {}
    if set(assignments) != set(requirements.stage_ids):
        missing = sorted(set(requirements.stage_ids) - set(assignments))
        extra = sorted(set(assignments) - set(requirements.stage_ids))
        if missing:
            violations.append(
                ConstraintViolation("stage_missing", f"stages missing assignments: {missing}")
            )
        if extra:
            violations.append(
                ConstraintViolation("stage_unknown", f"unknown stage assignments: {extra}")
            )

    workers_by_id = {str(worker.worker_id): worker for worker in workers}
    if any(worker_id not in workers_by_id for worker_id in assignments.values()):
        unknown = sorted({worker_id for worker_id in assignments.values() if worker_id not in workers_by_id})
        violations.append(
            ConstraintViolation("worker_unknown", f"unknown worker assignments: {unknown}")
        )

    assigned_worker_ids = tuple(assignments.get(stage_id) for stage_id in requirements.stage_ids)
    if len(assigned_worker_ids) < requirements.allowed_stage_to_worker_ratio[0]:
        violations.append(
            ConstraintViolation("worker_ratio", "assigned worker count is below the allowed minimum")
        )
    if len(set(assigned_worker_ids)) > requirements.allowed_stage_to_worker_ratio[1] * len(requirements.stage_ids):
        violations.append(
            ConstraintViolation("worker_ratio", "assigned worker count exceeds the allowed maximum")
        )

    if requirements.require_unique_physical_hosts:
        seen_hosts: dict[str, str] = {}
        for stage_id, worker_id in assignments.items():
            worker = workers_by_id.get(worker_id)
            if worker is None:
                continue
            host_key = str(worker.machine_id or worker.hostname)
            duplicate = seen_hosts.get(host_key)
            if duplicate is not None:
                violations.append(
                    ConstraintViolation(
                        "physical_host_conflict",
                        f"stages {duplicate!r} and {stage_id!r} share physical host {host_key!r}",
                    )
                )
            else:
                seen_hosts[host_key] = stage_id

    for stage_id, worker_id in assignments.items():
        worker = workers_by_id.get(worker_id)
        if worker is None:
            continue
        eligibility = evaluate_worker_eligibility(
            worker,
            worker_requirements,
            estimated_stage_memory_mb=stage_memory_by_stage.get(stage_id),
            reserved_memory_mb=reserved_memory_mb,
        )
        violations.extend(eligibility.violations)

    for edge in requirements.communication:
        source_worker = assignments.get(edge.source_stage_id)
        target_worker = assignments.get(edge.target_stage_id)
        if source_worker is None or target_worker is None:
            continue
        if source_worker == target_worker:
            continue
        violations.extend(
            _validate_network_edge(
                edge=edge,
                source_worker_id=source_worker,
                target_worker_id=target_worker,
                network_state=network_state,
            )
        )

    return PlacementFeasibilityResult(
        feasible=not violations,
        status=FeasibilityStatus.FEASIBLE if not violations else FeasibilityStatus.INFEASIBLE,
        violations=tuple(violations),
    )


def _validate_network_edge(
    *,
    edge: CommunicationRequirement,
    source_worker_id: str,
    target_worker_id: str,
    network_state: NetworkState | None,
) -> list[ConstraintViolation]:
    if network_state is None:
        return [
            ConstraintViolation(
                "network_missing",
                "network state is required for cross-worker communication validation",
            )
        ]
    forward = _find_link(network_state, source_worker_id, target_worker_id)
    reverse = _find_link(network_state, target_worker_id, source_worker_id)
    violations: list[ConstraintViolation] = []
    if forward is None:
        violations.append(
            ConstraintViolation(
                "network_missing",
                f"missing network link for {source_worker_id} -> {target_worker_id}",
            )
        )
    elif not forward.tcp_reachable or forward.failure_reason:
        violations.append(
            ConstraintViolation(
                "network_unreachable",
                f"network link {source_worker_id} -> {target_worker_id} is not reachable",
            )
        )
    if edge.require_bidirectional:
        if reverse is None:
            violations.append(
                ConstraintViolation(
                    "network_missing",
                    f"missing network link for {target_worker_id} -> {source_worker_id}",
                )
            )
        elif not reverse.tcp_reachable or reverse.failure_reason:
            violations.append(
                ConstraintViolation(
                    "network_unreachable",
                    f"network link {target_worker_id} -> {source_worker_id} is not reachable",
                )
            )
    return violations


def _find_link(
    network_state: NetworkState,
    source_worker_id: str,
    target_worker_id: str,
) -> NetworkLink | None:
    for link in network_state.links:
        if (
            str(link.source_worker_id) == source_worker_id
            and str(link.target_worker_id) == target_worker_id
        ):
            return link
    return None


def _support_status_to_feasibility(status: PartitionSupportStatus) -> FeasibilityStatus:
    if status is PartitionSupportStatus.UNSUPPORTED:
        return FeasibilityStatus.UNSUPPORTED
    if status is PartitionSupportStatus.INFEASIBLE:
        return FeasibilityStatus.INFEASIBLE
    return FeasibilityStatus.FEASIBLE
