"""Replay validation for persisted automatic planner artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from shardgrid.control.resource_manager import ClusterState
from shardgrid.engines.models import ParallelPlan
from shardgrid.planner.models import ExecutionPlan
from shardgrid.planner.requirements import (
    CommunicationRequirement,
    FeasibilityStatus,
    PlacementRequirements,
    WorkerEligibilityRequirements,
    evaluate_worker_eligibility,
    validate_placement_feasibility,
)

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class ReplayValidationResult:
    safe: bool
    status: FeasibilityStatus
    reasons: tuple[str, ...] = ()


def load_replay_bundle(
    snapshot_metadata_path: str | Path,
) -> tuple[ExecutionPlan, ParallelPlan]:
    from shardgrid.artifacts.metadata import load_snapshot_metadata

    metadata = load_snapshot_metadata(snapshot_metadata_path)
    execution_plan = ExecutionPlan.from_dict(
        json.loads(Path(metadata.execution_plan_path).read_text(encoding="utf-8"))
    )
    parallel_plan = ParallelPlan.from_dict(
        json.loads(Path(metadata.original_parallel_plan_path).read_text(encoding="utf-8"))
    )
    return execution_plan, parallel_plan


def validate_snapshot_replay(
    snapshot_metadata_path: str | Path,
    *,
    cluster_state: ClusterState,
) -> ReplayValidationResult:
    execution_plan, parallel_plan = load_replay_bundle(snapshot_metadata_path)
    return validate_execution_plan_replay(
        execution_plan,
        parallel_plan=parallel_plan,
        cluster_state=cluster_state,
    )


def validate_execution_plan_replay(
    execution_plan: ExecutionPlan,
    *,
    parallel_plan: ParallelPlan | None,
    cluster_state: ClusterState,
) -> ReplayValidationResult:
    reasons: list[str] = []
    if parallel_plan is None:
        reasons.append("automatic replay validation requires the saved parallel plan")
        deduped = tuple(dict.fromkeys(reasons))
        return ReplayValidationResult(
            safe=False,
            status=FeasibilityStatus.INFEASIBLE,
            reasons=deduped,
        )
    if parallel_plan.partition_source != "automatic":
        reasons.append("replay validation only supports automatic planner artifacts")
    if execution_plan.world_size != parallel_plan.world_size:
        reasons.append("saved execution plan world_size does not match parallel plan")
    if not execution_plan.parallel_plan_ref:
        reasons.append("saved execution plan is missing parallel_plan_ref")

    worker_entries = {
        str(entry.resource.worker_id): entry
        for entry in cluster_state.workers
    }
    stage_metadata = {
        stage.stage_id: stage for stage in parallel_plan.stage_metadata
    }
    stage_assignments = {
        str(assignment.stage): assignment
        for assignment in execution_plan.workers
        if assignment.stage is not None
    }

    unique_workers = {str(assignment.worker_id) for assignment in execution_plan.workers}
    selected_worker_count = execution_plan.labels.get("selected_worker_count")
    if selected_worker_count and selected_worker_count != "NONE":
        if len(unique_workers) != int(selected_worker_count):
            reasons.append(
                "saved selected_worker_count does not match execution plan assignments"
            )

    for assignment in execution_plan.workers:
        worker_entry = worker_entries.get(str(assignment.worker_id))
        if worker_entry is None:
            reasons.append(f"assigned worker is unavailable for replay: {assignment.worker_id}")
            continue
        stage = stage_metadata.get(str(assignment.stage))
        required_backends = (str(execution_plan.backend),)
        required_runtime = None
        if stage is not None:
            required_backends = stage.required_backends or required_backends
            required_runtime = stage.required_runtime
        eligibility = evaluate_worker_eligibility(
            worker_entry.resource,
            WorkerEligibilityRequirements(
                required_backends=tuple(required_backends),
                required_runtime_os=_runtime_os(required_runtime),
            ),
        )
        if not eligibility.eligible:
            reasons.extend(
                f"worker {assignment.worker_id} replay rejected: {violation.reason}"
                for violation in eligibility.violations
            )
        usable_bytes = _usable_memory_bytes(worker_entry.resource)
        required_bytes = assignment.estimated_peak_training_memory
        if (
            required_bytes is not None
            and usable_bytes is not None
            and required_bytes > usable_bytes
        ):
            reasons.append(
                f"worker {assignment.worker_id} usable memory dropped below saved stage peak "
                f"({usable_bytes} < {required_bytes})"
            )
        if stage is not None and stage.placement is not None:
            if str(assignment.worker_id) != stage.placement.worker_id:
                reasons.append(
                    f"replay changed worker mapping for {stage.stage_id}"
                )
            if assignment.rank != stage.placement.rank:
                reasons.append(
                    f"replay changed rank mapping for {stage.stage_id}"
                )
            if assignment.gpu_index != stage.placement.gpu_index:
                reasons.append(
                    f"replay changed gpu mapping for {stage.stage_id}"
                )

    for stage in parallel_plan.stage_metadata:
        assignment = stage_assignments.get(stage.stage_id)
        if assignment is None:
            reasons.append(
                f"saved execution plan is missing stage assignment for {stage.stage_id}"
            )
            continue
        if assignment.stage_metadata_ref is None:
            reasons.append(f"stage {stage.stage_id} lost stage_metadata_ref")

    selected_resources = [
        worker_entries[str(assignment.worker_id)].resource
        for assignment in execution_plan.workers
        if str(assignment.worker_id) in worker_entries
    ]
    structural = validate_placement_feasibility(
        {
            str(assignment.stage): str(assignment.worker_id)
            for assignment in execution_plan.workers
            if assignment.stage is not None
        },
        workers=selected_resources,
        requirements=PlacementRequirements(
            stage_ids=tuple(stage.stage_id for stage in parallel_plan.stage_metadata),
            communication=tuple(
                CommunicationRequirement(edge.source_stage_id, edge.target_stage_id)
                for edge in parallel_plan.communication_edges
            ),
        ),
        worker_requirements=WorkerEligibilityRequirements(
            required_backends=tuple({str(execution_plan.backend)}),
        ),
        network_state=cluster_state.network_state,
    )
    reasons.extend(violation.reason for violation in structural.violations)

    deduped = tuple(dict.fromkeys(reasons))
    return ReplayValidationResult(
        safe=not deduped,
        status=FeasibilityStatus.FEASIBLE if not deduped else FeasibilityStatus.INFEASIBLE,
        reasons=deduped,
    )


def _usable_memory_bytes(resource: object) -> int | None:
    free_memory = getattr(resource, "gpu_free_memory", None)
    total_memory = getattr(resource, "gpu_total_memory", None)
    memory_mb = free_memory if free_memory is not None else total_memory
    if memory_mb is None:
        return None
    return int(memory_mb) * _BYTES_PER_MB


def _runtime_os(value: str | None):
    if not value:
        return None
    from shardgrid.common.enums import RuntimeOS

    try:
        return RuntimeOS(value)
    except ValueError:
        return None
