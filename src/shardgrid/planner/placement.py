"""Joint partition and worker-placement search for T111."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil
from typing import Any, Mapping, Sequence

from shardgrid.common.enums import RuntimeOS
from shardgrid.control.resource_manager import ClusterState, WorkerEligibility
from shardgrid.engines.models import ModelProfile
from shardgrid.planner.memory import MemoryEstimationConfig
from shardgrid.planner.partitioning import PartitionCandidate, build_partition_profile
from shardgrid.planner.requirements import (
    CommunicationRequirement,
    FeasibilityStatus,
    PlacementRequirements,
    WorkerEligibilityRequirements,
    evaluate_worker_eligibility,
    validate_placement_feasibility,
)
from shardgrid.resources.models import WorkerResource

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class StagePlacement:
    stage_id: str
    worker_id: str
    rank: int
    machine_id: str | None
    usable_memory_before_bytes: int | None
    stage_required_bytes: int | None
    remaining_memory_bytes: int | None
    utilization_ratio: float | None


@dataclass(frozen=True)
class WorkerSubsetAttempt:
    worker_count: int
    worker_ids: tuple[str, ...]
    candidate_id: str | None
    status: FeasibilityStatus
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointPlacementPlan:
    status: FeasibilityStatus
    selected_worker_count: int | None
    selected_worker_ids: tuple[str, ...]
    partition_candidate: PartitionCandidate | None
    stage_placements: tuple[StagePlacement, ...]
    attempted_worker_counts: tuple[int, ...]
    attempts: tuple[WorkerSubsetAttempt, ...]
    reasons: tuple[str, ...] = ()
    selected_reason: str | None = None
    fallback_reason: str | None = None


def search_joint_partition_placement(
    model: Any,
    profile: ModelProfile,
    cluster_state: ClusterState,
    *,
    sample_args: Sequence[Any] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
    memory_config: MemoryEstimationConfig | None = None,
    worker_requirements: WorkerEligibilityRequirements | None = None,
    min_worker_count: int = 2,
    max_worker_count: int | None = None,
) -> JointPlacementPlan:
    config = memory_config or MemoryEstimationConfig()
    requirements = worker_requirements or _worker_requirements_for_profile(profile)
    eligible_workers, base_reasons = _eligible_workers(cluster_state, requirements)
    if len(eligible_workers) < min_worker_count:
        reasons = (
            "NO_ELIGIBLE_WORKERS",
            *base_reasons,
        )
        return JointPlacementPlan(
            status=FeasibilityStatus.INFEASIBLE,
            selected_worker_count=None,
            selected_worker_ids=(),
            partition_candidate=None,
            stage_placements=(),
            attempted_worker_counts=(),
            attempts=(),
            reasons=tuple(dict.fromkeys(reasons)),
            fallback_reason="no eligible workers remain after capability prefilter",
        )

    upper = min(max_worker_count or len(eligible_workers), len(eligible_workers))
    attempts: list[WorkerSubsetAttempt] = []
    attempted_counts: list[int] = []
    failure_reasons: list[str] = list(base_reasons)
    sample_kwargs = dict(sample_kwargs or {})

    for worker_count in range(min_worker_count, upper + 1):
        attempted_counts.append(worker_count)
        feasible_for_count = False
        for subset in _worker_subsets(eligible_workers, worker_count):
            partition = build_partition_profile(
                model,
                profile,
                sample_args=sample_args,
                sample_kwargs=sample_kwargs,
                memory_config=config,
                usable_memory_bytes=tuple(
                    _usable_memory_bytes(entry.resource) or 0 for entry in subset
                ),
                min_stage_count=worker_count,
                max_stage_count=worker_count,
            )
            candidate = next(
                (
                    item
                    for item in partition.candidates
                    if item.hard_constraint_status is FeasibilityStatus.FEASIBLE
                ),
                None,
            )
            if candidate is None:
                reasons = partition.reasons or _candidate_attempt_reasons(partition.candidates)
                attempts.append(
                    WorkerSubsetAttempt(
                        worker_count=worker_count,
                        worker_ids=tuple(str(entry.resource.worker_id) for entry in subset),
                        candidate_id=None,
                        status=partition.status,
                        reasons=tuple(reasons) or ("PARTITION_INFEASIBLE",),
                    )
                )
                failure_reasons.extend(reasons or ("PARTITION_INFEASIBLE",))
                continue

            stage_placements = _stage_placements(candidate, subset)
            assignments = {
                placement.stage_id: placement.worker_id for placement in stage_placements
            }
            structural = validate_placement_feasibility(
                assignments,
                workers=[entry.resource for entry in subset],
                requirements=PlacementRequirements(
                    stage_ids=tuple(stage.stage_id for stage in candidate.stages),
                    communication=tuple(
                        CommunicationRequirement(
                            edge.source_stage_id,
                            edge.target_stage_id,
                        )
                        for edge in candidate.communication_edges
                    ),
                ),
                worker_requirements=requirements,
                network_state=cluster_state.network_state,
            )
            fit_reasons = _stage_fit_reasons(stage_placements)
            reasons = tuple(
                dict.fromkeys(
                    [violation.reason for violation in structural.violations] + list(fit_reasons)
                )
            )
            if not reasons:
                feasible_for_count = True
                return JointPlacementPlan(
                    status=FeasibilityStatus.FEASIBLE,
                    selected_worker_count=worker_count,
                    selected_worker_ids=tuple(
                        placement.worker_id for placement in stage_placements
                    ),
                    partition_candidate=candidate,
                    stage_placements=stage_placements,
                    attempted_worker_counts=tuple(attempted_counts),
                    attempts=tuple(attempts),
                    reasons=tuple(dict.fromkeys(failure_reasons)),
                    selected_reason=(
                        f"first feasible {worker_count}-worker plan"
                    ),
                )

            attempts.append(
                WorkerSubsetAttempt(
                    worker_count=worker_count,
                    worker_ids=tuple(str(entry.resource.worker_id) for entry in subset),
                    candidate_id=candidate.candidate_id,
                    status=FeasibilityStatus.INFEASIBLE,
                    reasons=reasons,
                )
            )
            failure_reasons.extend(reasons)

        if not feasible_for_count:
            failure_reasons.append(f"NO_FEASIBLE_{worker_count}_WORKER_PLAN")

    return JointPlacementPlan(
        status=FeasibilityStatus.INFEASIBLE,
        selected_worker_count=None,
        selected_worker_ids=(),
        partition_candidate=None,
        stage_placements=(),
        attempted_worker_counts=tuple(attempted_counts),
        attempts=tuple(attempts),
        reasons=tuple(dict.fromkeys(failure_reasons)) or ("PLACEMENT_CONSTRAINT_VIOLATION",),
        fallback_reason="no feasible worker subset can host the partitioned model",
    )


def _worker_requirements_for_profile(profile: ModelProfile) -> WorkerEligibilityRequirements:
    runtime_os = None
    if profile.required_runtime:
        runtime_os = next(
            (
                runtime
                for runtime in RuntimeOS
                if runtime.value == profile.required_runtime
            ),
            None,
        )
    return WorkerEligibilityRequirements(
        required_runtime_os=runtime_os,
        required_backends=tuple(profile.required_backends),
    )


def _eligible_workers(
    cluster_state: ClusterState,
    requirements: WorkerEligibilityRequirements,
) -> tuple[tuple[WorkerEligibility, ...], tuple[str, ...]]:
    eligible: list[WorkerEligibility] = []
    reasons: list[str] = []
    for entry in cluster_state.workers:
        if not entry.eligible:
            reasons.extend(entry.exclusion_reasons)
            continue
        evaluation = evaluate_worker_eligibility(entry.resource, requirements)
        if evaluation.eligible and (_usable_memory_bytes(entry.resource) or 0) > 0:
            eligible.append(entry)
            continue
        if evaluation.eligible:
            reasons.append("worker has no usable GPU memory")
            continue
        reasons.extend(violation.reason for violation in evaluation.violations)
    ordered = tuple(
        sorted(
            eligible,
            key=lambda entry: (
                -(_usable_memory_bytes(entry.resource) or -1),
                str(entry.resource.worker_id),
            ),
        )
    )
    return ordered, tuple(dict.fromkeys(reasons))


def _worker_subsets(
    workers: Sequence[WorkerEligibility],
    worker_count: int,
) -> tuple[tuple[WorkerEligibility, ...], ...]:
    subsets = [
        tuple(sorted(subset, key=lambda entry: str(entry.resource.worker_id)))
        for subset in combinations(workers, worker_count)
    ]
    return tuple(
        sorted(
            subsets,
            key=lambda subset: (
                -sum(_usable_memory_bytes(entry.resource) or 0 for entry in subset),
                tuple(str(entry.resource.worker_id) for entry in subset),
            ),
        )
    )


def _stage_placements(
    candidate: PartitionCandidate,
    workers: Sequence[WorkerEligibility],
) -> tuple[StagePlacement, ...]:
    stages = sorted(
        candidate.stages,
        key=lambda stage: (
            -(stage.estimated_peak_training_memory.planner_required_bytes or 0),
            stage.stage_id,
        ),
    )
    ordered_workers = sorted(
        workers,
        key=lambda entry: (
            -(_usable_memory_bytes(entry.resource) or -1),
            str(entry.resource.worker_id),
        ),
    )
    placement_by_stage: dict[str, StagePlacement] = {}
    rank_by_stage = {
        stage.stage_id: index for index, stage in enumerate(candidate.stages)
    }
    for stage, worker in zip(stages, ordered_workers, strict=True):
        usable = _usable_memory_bytes(worker.resource)
        required = stage.estimated_peak_training_memory.planner_required_bytes
        remaining = None if usable is None or required is None else usable - required
        utilization = (
            None
            if usable in {None, 0} or required is None
            else required / usable
        )
        placement_by_stage[stage.stage_id] = StagePlacement(
            stage_id=stage.stage_id,
            worker_id=str(worker.resource.worker_id),
            rank=rank_by_stage[stage.stage_id],
            machine_id=(
                None
                if worker.resource.machine_id is None
                else str(worker.resource.machine_id)
            ),
            usable_memory_before_bytes=usable,
            stage_required_bytes=required,
            remaining_memory_bytes=remaining,
            utilization_ratio=utilization,
        )
    return tuple(placement_by_stage[stage.stage_id] for stage in candidate.stages)


def _stage_fit_reasons(stage_placements: Sequence[StagePlacement]) -> tuple[str, ...]:
    reasons: list[str] = []
    for placement in stage_placements:
        if placement.stage_required_bytes is None:
            reasons.append(
                f"stage {placement.stage_id} training peak memory is unavailable"
            )
            continue
        if placement.usable_memory_before_bytes is None:
            reasons.append(
                f"worker {placement.worker_id} usable GPU memory is unknown"
            )
            continue
        if placement.stage_required_bytes > placement.usable_memory_before_bytes:
            reasons.append(
                f"stage {placement.stage_id} peak {placement.stage_required_bytes} exceeds "
                f"worker {placement.worker_id} usable {placement.usable_memory_before_bytes}"
            )
    return tuple(reasons)


def _candidate_attempt_reasons(candidates: Sequence[PartitionCandidate]) -> tuple[str, ...]:
    reasons = [
        reason
        for candidate in candidates
        for reason in candidate.rejection_reasons
    ]
    return tuple(dict.fromkeys(reasons))


def _usable_memory_bytes(worker: WorkerResource) -> int | None:
    memory_mb = worker.gpu_free_memory
    if memory_mb is None:
        memory_mb = worker.gpu_total_memory
    if memory_mb is None:
        return None
    return int(memory_mb) * _BYTES_PER_MB


def bytes_to_mb(value: int | None) -> int | None:
    if value is None:
        return None
    return ceil(value / _BYTES_PER_MB)
