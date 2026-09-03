"""Automatic plan materialization and audit helpers for T114-T115."""

from __future__ import annotations

from typing import Any

from shardgrid.common.models import as_engine_name
from shardgrid.engines.models import (
    ModelProfile,
    ParallelPlan,
    ParallelPlanAttempt,
    ParallelPlanCommunicationEdge,
    ParallelPlanPlacement,
    ParallelPlanProvenance,
    ParallelPlanStage,
)
from shardgrid.planner.models import ExecutionPlan, WorkerAssignment
from shardgrid.planner.placement import JointPlacementPlan
from shardgrid.planner.requirements import FeasibilityStatus


def build_execution_plan_audit_payload(
    execution_plan: ExecutionPlan,
    *,
    parallel_plan: ParallelPlan | None = None,
    launch_metadata: dict[str, object] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    stage_details = {stage.stage_id: stage for stage in parallel_plan.stage_metadata} if parallel_plan else {}
    placement = [
        _placement_entry(assignment, stage_details.get(str(assignment.stage)))
        for assignment in execution_plan.workers
    ]
    assignments = [
        _assignment_entry(assignment, stage_details.get(str(assignment.stage)))
        for assignment in execution_plan.workers
    ]
    provenance = None if parallel_plan is None else parallel_plan.planning_provenance
    fallback_reason = execution_plan.labels.get("fallback_reason") or (
        None if provenance is None else provenance.fallback_reason
    )
    fallback_label = execution_plan.labels.get("fallback_label") or (
        "planner_fallback" if fallback_reason else "NONE"
    )
    fallback_status = execution_plan.labels.get("fallback_status") or (
        "USED" if fallback_reason else "NONE"
    )
    return {
        "job_id": str(execution_plan.job_id),
        "dry_run": dry_run,
        "engine": str(execution_plan.engine),
        "backend": str(execution_plan.backend),
        "world_size": execution_plan.world_size,
        "master": execution_plan.master.to_dict(),
        "placement_reason": execution_plan.placement_reason,
        "placement": placement,
        "assignments": assignments,
        "launch_metadata": dict(launch_metadata or {}),
        "original_plan": {
            "parallel_plan_id": None if parallel_plan is None else parallel_plan.parallel_plan_id,
            "parallel_plan_ref": execution_plan.parallel_plan_ref,
            "original_engine_plan_ref": execution_plan.original_engine_plan_ref,
            "model_profile_ref": execution_plan.model_profile_ref,
            "candidate_evaluation_ref": execution_plan.candidate_evaluation_ref,
            "selected_candidate_id": execution_plan.labels.get("selected_candidate_id"),
        },
        "fallback": {
            "status": fallback_status,
            "label": fallback_label,
            "reason": fallback_reason,
        },
        "planning": {
            "partition_source": execution_plan.labels.get("partition_source"),
            "selected_worker_count": execution_plan.labels.get("selected_worker_count"),
            "attempted_worker_counts": [] if provenance is None else list(provenance.attempted_worker_counts),
            "rejection_reasons": [] if provenance is None else list(provenance.rejection_reasons),
            "total_cross_worker_communication_bytes": execution_plan.labels.get(
                "total_cross_worker_communication_bytes"
            ),
        },
        "labels": dict(execution_plan.labels),
    }


def _placement_entry(
    assignment: WorkerAssignment,
    stage: ParallelPlanStage | None,
) -> dict[str, Any]:
    placement = None if stage is None else stage.placement
    return {
        "stage": assignment.stage,
        "stage_metadata_ref": assignment.stage_metadata_ref,
        "worker_id": str(assignment.worker_id),
        "rank": assignment.rank,
        "local_rank": assignment.local_rank,
        "gpu_index": assignment.gpu_index,
        "device": f"cuda:{assignment.gpu_index}",
        "host": assignment.host,
        "machine_id": assignment.machine_id,
        "physical_os": assignment.physical_os,
        "runtime_os": assignment.runtime_os,
        "runtime": assignment.runtime,
        "runtime_distro": assignment.runtime_distro,
        "usable_memory_before_bytes": None
        if placement is None
        else placement.usable_memory_before_bytes,
        "remaining_memory_bytes": None
        if placement is None
        else placement.remaining_memory_bytes,
        "utilization_ratio": None if placement is None else placement.utilization_ratio,
    }


def _assignment_entry(
    assignment: WorkerAssignment,
    stage: ParallelPlanStage | None,
) -> dict[str, Any]:
    payload = _placement_entry(assignment, stage)
    payload.update(
        {
            "estimated_peak_training_memory": assignment.estimated_peak_training_memory,
            "communication_edges": list(assignment.communication_edges),
            "conda_environment": assignment.conda_environment,
            "conda_prefix": assignment.conda_prefix,
            "python_executable": assignment.python_executable,
            "launch_command": assignment.launch_command,
            "environment": dict(assignment.environment),
            "status": assignment.status,
            "pid": assignment.pid,
            "log_path": assignment.log_path,
        }
    )
    return payload


def build_automatic_parallel_plan(
    profile: ModelProfile,
    selected_plan: JointPlacementPlan,
    *,
    parallel_plan_id: str | None = None,
    partition_algorithm: str = "weighted_graph_partition",
) -> ParallelPlan:
    if selected_plan.status is not FeasibilityStatus.FEASIBLE:
        raise ValueError("T114 requires a FEASIBLE selected plan")
    candidate = selected_plan.partition_candidate
    if candidate is None:
        raise ValueError("T114 requires partition candidate metadata")
    if candidate.model_profile_id != profile.profile_id:
        raise ValueError("selected candidate does not match model profile")
    if selected_plan.selected_worker_count != candidate.required_worker_count:
        raise ValueError("selected worker count must match candidate worker count")
    if len(selected_plan.stage_placements) != len(candidate.stages):
        raise ValueError("selected plan must place each stage exactly once")

    placements = {placement.stage_id: placement for placement in selected_plan.stage_placements}
    stage_metadata = [
        ParallelPlanStage(
            stage_id=stage.stage_id,
            rank=placements[stage.stage_id].rank,
            module_ids=stage.module_ids,
            module_paths=stage.module_paths,
            start_index=stage.start_index,
            stop_index=stage.stop_index,
            boundary_before_id=stage.boundary_before_id,
            boundary_after_id=stage.boundary_after_id,
            parameter_names_or_ranges=stage.parameter_names_or_ranges,
            parameter_bytes=stage.parameter_bytes,
            gradient_bytes=stage.gradient_bytes,
            activation_bytes=stage.activation_bytes,
            estimated_compute_units=stage.estimated_compute_units,
            estimated_peak_training_memory=stage.estimated_peak_training_memory,
            required_runtime=stage.required_runtime,
            required_backends=stage.required_backends,
            placement=ParallelPlanPlacement(
                worker_id=placements[stage.stage_id].worker_id,
                rank=placements[stage.stage_id].rank,
                machine_id=placements[stage.stage_id].machine_id,
                gpu_index=0,
                usable_memory_before_bytes=placements[stage.stage_id].usable_memory_before_bytes,
                remaining_memory_bytes=placements[stage.stage_id].remaining_memory_bytes,
                utilization_ratio=placements[stage.stage_id].utilization_ratio,
            ),
        )
        for stage in candidate.stages
    ]
    plan = ParallelPlan(
        parallel_plan_id=parallel_plan_id or f"{profile.profile_id}:{candidate.candidate_id}",
        engine=as_engine_name(profile.engine_id),
        engine_plan_path=candidate.original_engine_plan_ref,
        model_name=profile.model_name,
        world_size=len(stage_metadata),
        stages=[stage.stage_id for stage in stage_metadata],
        partition_source="automatic",
        model_profile_id=profile.profile_id,
        selected_candidate_id=candidate.candidate_id,
        stage_metadata=stage_metadata,
        communication_edges=[
            ParallelPlanCommunicationEdge(
                source_stage_id=edge.source_stage_id,
                target_stage_id=edge.target_stage_id,
                source_module_id=edge.source_module_id,
                target_module_id=edge.target_module_id,
                activation=edge.activation,
                gradient=edge.gradient,
                estimated_bytes_per_step=edge.estimated_bytes_per_step,
                estimate_kind=edge.estimate_kind,
            )
            for edge in candidate.communication_edges
        ],
        planning_provenance=ParallelPlanProvenance(
            partition_source="automatic",
            model_profile_id=profile.profile_id,
            selected_candidate_id=candidate.candidate_id,
            selected_worker_count=selected_plan.selected_worker_count,
            attempted_worker_counts=selected_plan.attempted_worker_counts,
            attempts=tuple(
                ParallelPlanAttempt(
                    worker_count=attempt.worker_count,
                    worker_ids=attempt.worker_ids,
                    candidate_id=attempt.candidate_id,
                    status=attempt.status.value,
                    reasons=attempt.reasons,
                )
                for attempt in selected_plan.attempts
            ),
            partition_algorithm=partition_algorithm,
            total_cross_worker_communication_bytes=candidate.estimated_bytes_per_step,
            selected_reason=selected_plan.selected_reason,
            fallback_reason=selected_plan.fallback_reason,
            rejection_reasons=selected_plan.reasons,
        ),
        requirements={
            "partition_source": "automatic",
            "required_runtime": profile.required_runtime or "",
            "required_backends": ",".join(profile.required_backends),
            "selected_worker_count": str(selected_plan.selected_worker_count),
        },
        limitations=[],
    )
    validate_automatic_parallel_plan_or_raise(
        plan,
        model_profile=profile,
        selected_plan=selected_plan,
    )
    return plan


def validate_automatic_parallel_plan(
    plan: ParallelPlan,
    *,
    model_profile: ModelProfile | None = None,
    selected_plan: JointPlacementPlan | None = None,
) -> list[str]:
    problems: list[str] = []
    if plan.partition_source != "automatic":
        problems.append("automatic parallel plan must set partition_source=automatic")
    if not plan.stage_metadata:
        problems.append("automatic parallel plan requires stage metadata")
        return problems
    if plan.world_size != len(plan.stage_metadata):
        problems.append("parallel plan world_size must match stage metadata count")
    if plan.stages != [stage.stage_id for stage in plan.stage_metadata]:
        problems.append("parallel plan stage ids must match stage metadata order")

    stage_ids: list[str] = []
    rank_ids: list[int] = []
    worker_refs: list[str] = []
    machine_refs: list[str] = []
    flattened_module_ids: list[str] = []
    ownership: set[str] = set()
    for stage in plan.stage_metadata:
        stage_ids.append(stage.stage_id)
        rank_ids.append(stage.rank)
        flattened_module_ids.extend(stage.module_ids)
        if stage.placement is None:
            problems.append(f"stage {stage.stage_id} is missing placement metadata")
            continue
        worker_refs.append(stage.placement.worker_id)
        machine_refs.append(stage.placement.machine_id or stage.placement.worker_id)
        required_bytes = (
            stage.estimated_peak_training_memory.planner_required_bytes
            or stage.estimated_peak_training_memory.estimated_peak_bytes
        )
        usable_bytes = stage.placement.usable_memory_before_bytes
        if (
            required_bytes is not None
            and usable_bytes is not None
            and required_bytes > usable_bytes
        ):
            problems.append(
                f"stage {stage.stage_id} requires {required_bytes} bytes but worker "
                f"{stage.placement.worker_id} only records {usable_bytes} usable bytes"
            )
        for owner in stage.parameter_names_or_ranges:
            if owner in ownership:
                problems.append(f"parameter ownership repeats across stages: {owner}")
            ownership.add(owner)

    if len(stage_ids) != len(set(stage_ids)):
        problems.append("parallel plan stage ids must be unique")
    if len(rank_ids) != len(set(rank_ids)):
        problems.append("parallel plan ranks must be unique")
    if model_profile is not None:
        if plan.model_profile_id != profile_id(model_profile):
            problems.append("parallel plan model_profile_id does not match profile")
        if tuple(flattened_module_ids) != tuple(model_profile.module_order):
            problems.append("parallel plan stages do not cover model modules exactly once")
        for stage in plan.stage_metadata:
            expected = model_profile.modules[stage.start_index:stage.stop_index]
            expected_ids = tuple(module.module_id for module in expected)
            expected_paths = tuple(module.module_path for module in expected)
            if stage.module_ids != expected_ids:
                problems.append(f"stage {stage.stage_id} module_ids do not match profile slice")
            if stage.module_paths != expected_paths:
                problems.append(
                    f"stage {stage.stage_id} module_paths do not match profile slice"
                )

    edge_stage_ids = set(stage_ids)
    for edge in plan.communication_edges:
        if edge.source_stage_id not in edge_stage_ids:
            problems.append(
                f"communication edge source stage is unknown: {edge.source_stage_id}"
            )
        if edge.target_stage_id not in edge_stage_ids:
            problems.append(
                f"communication edge target stage is unknown: {edge.target_stage_id}"
            )

    provenance = plan.planning_provenance
    if provenance is None:
        problems.append("automatic parallel plan requires planning provenance")
    else:
        if provenance.partition_source != "automatic":
            problems.append("planning provenance must record automatic partition source")
        if provenance.selected_worker_count != len(set(machine_refs)):
            problems.append(
                "selected_worker_count must match unique physical workers in stage placement"
            )

    if selected_plan is not None:
        candidate = selected_plan.partition_candidate
        if candidate is None:
            problems.append("selected plan is missing partition candidate metadata")
        else:
            if plan.selected_candidate_id != candidate.candidate_id:
                problems.append("parallel plan selected candidate id does not match T112")
            if plan.engine_plan_path != candidate.original_engine_plan_ref:
                problems.append("parallel plan did not preserve original engine plan ref")
            expected_edges = {
                _edge_key_from_candidate(edge)
                for edge in candidate.communication_edges
            }
            actual_edges = {_edge_key_from_plan(edge) for edge in plan.communication_edges}
            missing_edges = expected_edges - actual_edges
            if missing_edges:
                problems.append(
                    "parallel plan is missing selected candidate communication edges"
                )
            expected_placements = {
                placement.stage_id: placement for placement in selected_plan.stage_placements
            }
            if len(expected_placements) != len(plan.stage_metadata):
                problems.append("selected plan stage placement count does not match parallel plan")
            for stage in plan.stage_metadata:
                placement = expected_placements.get(stage.stage_id)
                if placement is None:
                    problems.append(
                        f"parallel plan stage {stage.stage_id} is absent from selected placement"
                    )
                    continue
                actual = stage.placement
                if actual is None:
                    continue
                if actual.worker_id != placement.worker_id or actual.rank != placement.rank:
                    problems.append(
                        f"parallel plan stage {stage.stage_id} placement changed from T112"
                    )
                if (
                    actual.machine_id is not None
                    and actual.machine_id != placement.machine_id
                ):
                    problems.append(
                        f"parallel plan stage {stage.stage_id} machine_id changed from T112"
                    )
    return problems


def validate_automatic_parallel_plan_or_raise(
    plan: ParallelPlan,
    *,
    model_profile: ModelProfile | None = None,
    selected_plan: JointPlacementPlan | None = None,
) -> None:
    problems = validate_automatic_parallel_plan(
        plan,
        model_profile=model_profile,
        selected_plan=selected_plan,
    )
    if problems:
        raise ValueError("; ".join(problems))


def profile_id(profile: ModelProfile) -> str:
    return profile.profile_id


def _edge_key_from_candidate(edge: object) -> tuple[str, str, str, str]:
    return (
        str(getattr(edge, "source_stage_id")),
        str(getattr(edge, "target_stage_id")),
        str(getattr(edge, "source_module_id")),
        str(getattr(edge, "target_module_id")),
    )


def _edge_key_from_plan(
    edge: ParallelPlanCommunicationEdge,
) -> tuple[str, str, str, str]:
    return (
        edge.source_stage_id,
        edge.target_stage_id,
        edge.source_module_id,
        edge.target_module_id,
    )
