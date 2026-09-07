from __future__ import annotations

import pytest

from shardgrid.engines.models import EstimateKind, TensorMetadata, TrainingMemoryEstimate
from shardgrid.planner.partitioning import (
    PartitionCandidate,
    StageCommunicationEdge,
    StagePartition,
)
from shardgrid.planner.placement import JointPlacementPlan, StagePlacement
from shardgrid.planner.requirements import FeasibilityStatus
from shardgrid.planner.scoring import (
    select_best_joint_placement_plan,
    summarize_plan_memory_utilization,
)

_MIB = 1024 * 1024
_GIB = 1024 * _MIB


def _plan(
    candidate_id: str,
    *,
    communication_mib: int,
    remaining_gib: tuple[float, ...],
    selected_worker_ids: tuple[str, ...] = ("worker-a", "worker-b"),
    selected_worker_count: int = 2,
    stage_ranges: tuple[tuple[int, int], ...] = ((0, 2), (2, 4)),
    usable_gib: tuple[int, ...] = (12, 12),
) -> JointPlacementPlan:
    stage_placements = []
    stages = []
    for index, (remaining_gib_value, usable_gib_value, stage_range) in enumerate(
        zip(remaining_gib, usable_gib, stage_ranges, strict=True)
    ):
        usable_bytes = usable_gib_value * _GIB
        remaining_bytes = int(remaining_gib_value * _GIB)
        required_bytes = usable_bytes - remaining_bytes
        stage_id = f"stage-{index}"
        stage_placements.append(
            StagePlacement(
                stage_id=stage_id,
                worker_id=selected_worker_ids[index],
                rank=index,
                machine_id=f"machine-{index}",
                usable_memory_before_bytes=usable_bytes,
                stage_required_bytes=required_bytes,
                remaining_memory_bytes=remaining_bytes,
                utilization_ratio=required_bytes / usable_bytes,
            )
        )
        stages.append(
            StagePartition(
                stage_id=stage_id,
                module_ids=(f"module-{index}",),
                module_paths=(f"layers.{index}",),
                start_index=stage_range[0],
                stop_index=stage_range[1],
                parameter_names_or_ranges=(f"layers.{index}",),
                parameter_bytes=required_bytes // 4,
                gradient_bytes=required_bytes // 8,
                activation_bytes=communication_mib * _MIB,
                estimated_compute_units=100 + index,
                estimated_peak_training_memory=TrainingMemoryEstimate(
                    estimated_peak_bytes=required_bytes,
                    planner_required_bytes=required_bytes,
                ),
                required_runtime="wsl2-linux",
                required_backends=("nccl",),
            )
        )

    communication_bytes = communication_mib * _MIB
    partition_candidate = PartitionCandidate(
        candidate_id=candidate_id,
        model_profile_id="profile-1",
        stage_count=len(stage_placements),
        stages=tuple(stages),
        communication_edges=(
            StageCommunicationEdge(
                source_stage_id="stage-0",
                target_stage_id="stage-1",
                source_module_id="module-0",
                target_module_id="module-1",
                activation=(
                    TensorMetadata(
                        name="activation",
                        estimated_bytes=communication_bytes // 2,
                        estimate_kind=EstimateKind.ESTIMATED,
                    ),
                ),
                gradient=(
                    TensorMetadata(
                        name="gradient",
                        estimated_bytes=communication_bytes // 3,
                        estimate_kind=EstimateKind.ESTIMATED,
                    ),
                ),
                estimated_bytes_per_step=communication_bytes,
                estimate_kind=EstimateKind.ESTIMATED,
            ),
        ),
        estimated_bytes_per_step=communication_bytes,
        required_worker_count=selected_worker_count,
        hard_constraint_status=FeasibilityStatus.FEASIBLE,
        engine_id="pytorch_pipeline",
        required_runtime="wsl2-linux",
        required_backends=("nccl",),
        selected_boundary_ids=("boundary-0",),
    )
    return JointPlacementPlan(
        status=FeasibilityStatus.FEASIBLE,
        selected_worker_count=selected_worker_count,
        selected_worker_ids=selected_worker_ids,
        partition_candidate=partition_candidate,
        stage_placements=tuple(stage_placements),
        attempted_worker_counts=(selected_worker_count,),
        attempts=(),
        selected_reason="ranked by T112",
    )


def test_lower_cross_worker_communication_wins() -> None:
    best = select_best_joint_placement_plan(
        [
            _plan("candidate-low-comm", communication_mib=96, remaining_gib=(1.0, 1.0)),
            _plan("candidate-high-comm", communication_mib=128, remaining_gib=(0.5, 0.5)),
        ]
    )

    assert best.partition_candidate is not None
    assert best.partition_candidate.candidate_id == "candidate-low-comm"


def test_close_communication_uses_memory_utilization_tie_break() -> None:
    best = select_best_joint_placement_plan(
        [
            _plan(
                "candidate-fragmented",
                communication_mib=100,
                remaining_gib=(0.25, 0.375),
            ),
            _plan(
                "candidate-cleaner-memory",
                communication_mib=103,
                remaining_gib=(1.0, 1.0),
            ),
        ]
    )

    assert best.partition_candidate is not None
    assert best.partition_candidate.candidate_id == "candidate-cleaner-memory"
    summary = summarize_plan_memory_utilization(best)
    assert summary.fragmented_worker_count == 0


def test_tie_break_is_deterministic() -> None:
    plan_a = _plan(
        "candidate-a",
        communication_mib=100,
        remaining_gib=(1.0, 1.0),
        selected_worker_ids=("worker-a", "worker-b"),
    )
    plan_b = _plan(
        "candidate-b",
        communication_mib=100,
        remaining_gib=(1.0, 1.0),
        selected_worker_ids=("worker-b", "worker-c"),
    )

    first = select_best_joint_placement_plan([plan_b, plan_a])
    second = select_best_joint_placement_plan([plan_a, plan_b])

    assert first.partition_candidate is not None
    assert second.partition_candidate is not None
    assert first.partition_candidate.candidate_id == "candidate-a"
    assert second.partition_candidate.candidate_id == "candidate-a"


def test_mixed_worker_counts_raise_t111_flow_error() -> None:
    with pytest.raises(ValueError, match="T111 flow error"):
        select_best_joint_placement_plan(
            [
                _plan("candidate-two", communication_mib=96, remaining_gib=(1.0, 1.0)),
                _plan(
                    "candidate-three",
                    communication_mib=96,
                    remaining_gib=(1.0, 1.0, 1.0),
                    selected_worker_ids=("worker-a", "worker-b", "worker-c"),
                    selected_worker_count=3,
                    stage_ranges=((0, 1), (1, 2), (2, 4)),
                    usable_gib=(12, 12, 12),
                ),
            ]
        )
