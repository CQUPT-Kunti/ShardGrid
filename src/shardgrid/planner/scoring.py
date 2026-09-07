"""Final candidate selection helpers for T112."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

from shardgrid.planner.placement import JointPlacementPlan, StagePlacement
from shardgrid.planner.requirements import FeasibilityStatus

_BYTES_PER_MIB = 1024 * 1024

DEFAULT_COMMUNICATION_CLOSE_RATIO = 0.05
DEFAULT_COMMUNICATION_CLOSE_BYTES = _BYTES_PER_MIB
DEFAULT_MEANINGFUL_REMAINING_BYTES = 512 * _BYTES_PER_MIB


@dataclass(frozen=True)
class PlanMemorySummary:
    total_usable_memory_bytes: int
    total_required_memory_bytes: int
    total_remaining_memory_bytes: int
    utilization_ratio: float
    fragmented_worker_count: int
    max_remaining_memory_bytes: int
    remaining_memory_bytes: tuple[int, ...]


def select_best_joint_placement_plan(
    candidates: Sequence[JointPlacementPlan],
    *,
    communication_close_ratio: float = DEFAULT_COMMUNICATION_CLOSE_RATIO,
    communication_close_bytes: int = DEFAULT_COMMUNICATION_CLOSE_BYTES,
    meaningful_remaining_bytes: int = DEFAULT_MEANINGFUL_REMAINING_BYTES,
) -> JointPlacementPlan:
    plans = _validated_candidates(
        candidates,
        communication_close_ratio=communication_close_ratio,
        communication_close_bytes=communication_close_bytes,
        meaningful_remaining_bytes=meaningful_remaining_bytes,
    )
    if len(plans) == 1:
        return plans[0]

    minimum_communication = min(_communication_bytes(plan) for plan in plans)
    communication_frontier = minimum_communication + max(
        communication_close_bytes,
        ceil(minimum_communication * communication_close_ratio),
    )
    frontier = tuple(
        plan
        for plan in plans
        if _communication_bytes(plan) <= communication_frontier
    )
    return min(
        frontier,
        key=lambda plan: (
            _memory_sort_key(
                summarize_plan_memory_utilization(
                    plan,
                    meaningful_remaining_bytes=meaningful_remaining_bytes,
                )
            ),
            _tie_break_key(plan),
        ),
    )


def summarize_plan_memory_utilization(
    candidate: JointPlacementPlan,
    *,
    meaningful_remaining_bytes: int = DEFAULT_MEANINGFUL_REMAINING_BYTES,
) -> PlanMemorySummary:
    if meaningful_remaining_bytes < 0:
        raise ValueError("meaningful_remaining_bytes must be >= 0")
    placements = _validated_stage_placements(candidate)
    usable = tuple(
        _required_int(
            placement.usable_memory_before_bytes,
            f"stage {placement.stage_id} usable memory is required for T112 scoring",
        )
        for placement in placements
    )
    required = tuple(
        _required_int(
            placement.stage_required_bytes,
            f"stage {placement.stage_id} required memory is required for T112 scoring",
        )
        for placement in placements
    )
    remaining = tuple(
        _required_int(
            placement.remaining_memory_bytes,
            f"stage {placement.stage_id} remaining memory is required for T112 scoring",
        )
        for placement in placements
    )
    total_usable = sum(usable)
    total_required = sum(required)
    total_remaining = sum(remaining)
    fragmented = sum(
        1
        for value in remaining
        if 0 < value < meaningful_remaining_bytes
    )
    return PlanMemorySummary(
        total_usable_memory_bytes=total_usable,
        total_required_memory_bytes=total_required,
        total_remaining_memory_bytes=total_remaining,
        utilization_ratio=(
            0.0
            if total_usable == 0
            else total_required / total_usable
        ),
        fragmented_worker_count=fragmented,
        max_remaining_memory_bytes=max(remaining, default=0),
        remaining_memory_bytes=remaining,
    )


def _validated_candidates(
    candidates: Sequence[JointPlacementPlan],
    *,
    communication_close_ratio: float,
    communication_close_bytes: int,
    meaningful_remaining_bytes: int,
) -> tuple[JointPlacementPlan, ...]:
    if not candidates:
        raise ValueError("T112 requires at least one FEASIBLE joint placement candidate")
    if not 0 <= communication_close_ratio <= 1:
        raise ValueError("communication_close_ratio must be between 0 and 1")
    if communication_close_bytes < 0:
        raise ValueError("communication_close_bytes must be >= 0")
    if meaningful_remaining_bytes < 0:
        raise ValueError("meaningful_remaining_bytes must be >= 0")

    plans = tuple(candidates)
    worker_counts = {
        plan.selected_worker_count
        for plan in plans
    }
    if None in worker_counts:
        raise ValueError("T112 requires candidates with selected_worker_count")
    if len(worker_counts) != 1:
        raise ValueError(
            "T112 requires one shared selected_worker_count; mixed worker counts "
            "indicate a T111 flow error"
        )
    for plan in plans:
        if plan.status is not FeasibilityStatus.FEASIBLE:
            raise ValueError("T112 only ranks FEASIBLE candidates from T111")
        if plan.partition_candidate is None:
            raise ValueError("T112 requires partition_candidate metadata")
        _communication_bytes(plan)
        _validated_stage_placements(plan)
    return plans


def _validated_stage_placements(candidate: JointPlacementPlan) -> tuple[StagePlacement, ...]:
    if not candidate.stage_placements:
        raise ValueError("T112 requires non-empty stage placements")
    return tuple(candidate.stage_placements)


def _communication_bytes(candidate: JointPlacementPlan) -> int:
    partition_candidate = candidate.partition_candidate
    if partition_candidate is None:
        raise ValueError("T112 requires partition_candidate metadata")
    if partition_candidate.estimated_bytes_per_step is not None:
        return int(partition_candidate.estimated_bytes_per_step)

    total = 0
    known_edge = False
    for edge in partition_candidate.communication_edges:
        if edge.estimated_bytes_per_step is None:
            continue
        total += int(edge.estimated_bytes_per_step)
        known_edge = True
    if known_edge:
        return total
    raise ValueError(
        f"candidate {partition_candidate.candidate_id} is missing communication estimates"
    )


def _memory_sort_key(summary: PlanMemorySummary) -> tuple[int, int, int, tuple[int, ...]]:
    unused_ratio_ppm = 0
    if summary.total_usable_memory_bytes:
        unused_ratio_ppm = (
            summary.total_remaining_memory_bytes * 1_000_000
        ) // summary.total_usable_memory_bytes
    return (
        summary.fragmented_worker_count,
        unused_ratio_ppm,
        summary.total_remaining_memory_bytes,
        summary.max_remaining_memory_bytes,
        tuple(sorted(summary.remaining_memory_bytes, reverse=True)),
    )


def _tie_break_key(
    candidate: JointPlacementPlan,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...], tuple[str, ...], str]:
    partition_candidate = candidate.partition_candidate
    if partition_candidate is None:
        raise ValueError("T112 requires partition_candidate metadata")
    return (
        tuple(candidate.selected_worker_ids),
        tuple(
            (stage.start_index, stage.stop_index, stage.stage_id)
            for stage in partition_candidate.stages
        ),
        tuple(partition_candidate.selected_boundary_ids),
        partition_candidate.candidate_id,
    )


def _required_int(value: int | None, message: str) -> int:
    if value is None:
        raise ValueError(message)
    return int(value)
