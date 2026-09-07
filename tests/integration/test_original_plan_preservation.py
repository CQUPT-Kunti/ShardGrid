from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import torch
from examples.models.minimal_transformer import (
    MinimalTransformerConfig,
    build_minimal_transformer,
)
from examples.models.partition_stress_model import build_partition_stress_model, make_training_batch

from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_machine_id, as_worker_id
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.planner.execution_plan import (
    build_automatic_parallel_plan,
    validate_automatic_parallel_plan,
)
from shardgrid.planner.memory import MemoryEstimationConfig, build_model_profile
from shardgrid.planner.partitioning import build_partition_profile
from shardgrid.planner.placement import JointPlacementPlan, StagePlacement
from shardgrid.planner.requirements import FeasibilityStatus
from shardgrid.planner.scoring import select_best_joint_placement_plan
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource

_BYTES_PER_MB = 1024 * 1024


def _memory_config() -> MemoryEstimationConfig:
    return MemoryEstimationConfig(
        optimizer_type="adamw",
        gradient_dtype="float32",
        optimizer_state_dtype="float32",
        runtime_overhead_bytes=1024,
        communication_buffer_bytes=2048,
        safety_headroom_bytes=4096,
        temporary_buffer_factor=0.25,
    )


def _timestamp() -> str:
    return datetime(2026, 9, 3, 0, 0, tzinfo=UTC).isoformat()


def _worker(
    worker_id: str,
    *,
    machine_id: str,
    free_memory_mb: int,
) -> WorkerResource:
    return WorkerResource(
        worker_id=as_worker_id(worker_id),
        hostname=as_hostname(f"{worker_id}.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        machine_id=as_machine_id(machine_id),
        enabled=True,
        conda_environment="shardgrid",
        python_executable="/opt/conda/bin/python",
        ip=f"10.0.0.{len(worker_id)}",
        gpu_name="GPU",
        gpu_total_memory=free_memory_mb,
        gpu_free_memory=free_memory_mb,
        cuda_version="12.4",
        torch_version="2.7.1",
        torch_cuda_version="12.4",
        nccl_available=True,
        gloo_available=True,
        health=Health.HEALTHY,
        last_probe_at=_timestamp(),
    )


def _network(workers: list[WorkerResource]) -> NetworkState:
    links = [
        NetworkLink(
            source_worker_id=source.worker_id,
            target_worker_id=target.worker_id,
            source_ip=source.ip or "10.0.0.1",
            target_ip=target.ip or "10.0.0.2",
            interface="eth0",
            tcp_reachable=True,
            bandwidth_mbps=900.0,
            latency_ms=1.5,
            measured_at=_timestamp(),
        )
        for source in workers
        for target in workers
        if source.worker_id != target.worker_id
    ]
    return NetworkState(
        network_id="net-test",
        workers=[worker.worker_id for worker in workers],
        links=links,
        created_at=_timestamp(),
    )


def _cluster_state(workers: list[WorkerResource]) -> object:
    return ResourceManager().build_cluster_state(
        workers,
        network_state=_network(workers),
        now=datetime(2026, 9, 3, 0, 0, tzinfo=UTC),
    )


def _selected_plan_from_candidate(
    candidate,
    *,
    worker_ids: tuple[str, ...],
    machine_ids: tuple[str, ...],
    free_memory_mb: tuple[int, ...],
) -> JointPlacementPlan:
    stage_placements = []
    for rank, (stage, worker_id, machine_id, memory_mb) in enumerate(
        zip(candidate.stages, worker_ids, machine_ids, free_memory_mb, strict=True)
    ):
        usable = memory_mb * _BYTES_PER_MB
        required = stage.estimated_peak_training_memory.planner_required_bytes
        remaining = None if required is None else usable - required
        utilization = None if required is None or usable == 0 else required / usable
        stage_placements.append(
            StagePlacement(
                stage_id=stage.stage_id,
                worker_id=worker_id,
                rank=rank,
                machine_id=machine_id,
                usable_memory_before_bytes=usable,
                stage_required_bytes=required,
                remaining_memory_bytes=remaining,
                utilization_ratio=utilization,
            )
        )
    return JointPlacementPlan(
        status=FeasibilityStatus.FEASIBLE,
        selected_worker_count=len(worker_ids),
        selected_worker_ids=worker_ids,
        partition_candidate=candidate,
        stage_placements=tuple(stage_placements),
        attempted_worker_counts=(len(worker_ids),),
        attempts=(),
        selected_reason="first feasible selected candidate",
    )


def test_automatic_parallel_plan_preserves_selected_metadata_and_serialization() -> None:
    model = build_minimal_transformer(MinimalTransformerConfig(), seed=42)
    sample = torch.ones((2, 16), dtype=torch.long)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="minimal-transformer",
        sample_args=(sample,),
        memory_config=_memory_config(),
        required_backends=("nccl",),
    )
    result = build_partition_profile(
        model,
        profile,
        sample_args=(sample,),
        memory_config=_memory_config(),
        usable_memory_bytes=(32 * _BYTES_PER_MB, 32 * _BYTES_PER_MB),
        min_stage_count=2,
        max_stage_count=2,
        original_engine_plan_ref="/tmp/original-engine-plan.json",
    )
    candidate = next(
        item
        for item in result.candidates
        if item.hard_constraint_status is FeasibilityStatus.FEASIBLE
    )
    selected = select_best_joint_placement_plan(
        [
            _selected_plan_from_candidate(
                candidate,
                worker_ids=("worker-a", "worker-b"),
                machine_ids=("machine-a", "machine-b"),
                free_memory_mb=(32, 32),
            )
        ]
    )

    plan = build_automatic_parallel_plan(profile, selected)
    problems = validate_automatic_parallel_plan(
        plan,
        model_profile=profile,
        selected_plan=selected,
    )

    assert problems == []
    assert plan.partition_source == "automatic"
    assert plan.model_profile_id == profile.profile_id
    assert plan.engine_plan_path == "/tmp/original-engine-plan.json"
    assert plan.selected_candidate_id == candidate.candidate_id
    assert [stage.stage_id for stage in plan.stage_metadata] == plan.stages
    assert [stage.placement.worker_id for stage in plan.stage_metadata] == [
        placement.worker_id for placement in selected.stage_placements
    ]
    assert all(
        stage.estimated_peak_training_memory.planner_required_bytes is not None
        for stage in plan.stage_metadata
    )
    assert all(
        stage.placement is not None
        and stage.placement.remaining_memory_bytes is not None
        and stage.placement.utilization_ratio is not None
        for stage in plan.stage_metadata
    )
    restored = plan.from_dict(plan.to_dict())
    assert restored == plan


def test_parallel_plan_preserves_residual_edges_and_supports_three_stages() -> None:
    model = build_partition_stress_model(seed=42)
    inputs, _targets = make_training_batch(seed=23, step=0)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="partition-stress-model",
        sample_args=(inputs,),
        memory_config=_memory_config(),
        required_backends=("nccl",),
    )
    workers = [
        _worker("worker-a", machine_id="machine-a", free_memory_mb=8),
        _worker("worker-b", machine_id="machine-b", free_memory_mb=8),
        _worker("worker-c", machine_id="machine-c", free_memory_mb=8),
    ]
    _cluster_state(workers)
    result = build_partition_profile(
        model,
        profile,
        sample_args=(inputs,),
        memory_config=_memory_config(),
        usable_memory_bytes=(120_000, 120_000, 120_000),
        min_stage_count=3,
        max_stage_count=3,
        original_engine_plan_ref="/tmp/original-stress-plan.json",
    )
    candidate = next(
        replace(item, original_engine_plan_ref="/tmp/original-stress-plan.json")
        for item in result.candidates
        if item.stage_count == 3
        and item.hard_constraint_status is FeasibilityStatus.FEASIBLE
    )
    selected = select_best_joint_placement_plan(
        [
            _selected_plan_from_candidate(
                candidate,
                worker_ids=("worker-a", "worker-b", "worker-c"),
                machine_ids=("machine-a", "machine-b", "machine-c"),
                free_memory_mb=(8, 8, 8),
            )
        ]
    )

    plan = build_automatic_parallel_plan(profile, selected)

    assert plan.world_size == 3
    assert len(plan.stage_metadata) == 3
    assert plan.stages == [stage.stage_id for stage in plan.stage_metadata]
    assert any(
        abs(int(edge.target_stage_id[-1]) - int(edge.source_stage_id[-1])) > 1
        for edge in plan.communication_edges
    )
    assert any(
        abs(int(edge.target_stage_id[-1]) - int(edge.source_stage_id[-1])) > 1
        for edge in candidate.communication_edges
    )


def test_parallel_plan_validation_fails_when_selected_worker_count_is_inconsistent() -> None:
    model = build_partition_stress_model(seed=42)
    inputs, _targets = make_training_batch(seed=23, step=0)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="partition-stress-model",
        sample_args=(inputs,),
        memory_config=_memory_config(),
        required_backends=("nccl",),
    )
    result = build_partition_profile(
        model,
        profile,
        sample_args=(inputs,),
        memory_config=_memory_config(),
        usable_memory_bytes=(120_000, 120_000, 120_000),
        min_stage_count=3,
        max_stage_count=3,
    )
    candidate = next(
        item
        for item in result.candidates
        if item.stage_count == 3
        and item.hard_constraint_status is FeasibilityStatus.FEASIBLE
    )
    selected = _selected_plan_from_candidate(
        candidate,
        worker_ids=("worker-a", "worker-b", "worker-c"),
        machine_ids=("machine-a", "machine-b", "machine-c"),
        free_memory_mb=(8, 8, 8),
    )
    plan = build_automatic_parallel_plan(profile, selected)
    broken = replace(
        plan,
        planning_provenance=replace(
            plan.planning_provenance,
            selected_worker_count=2,
        ),
    )

    problems = validate_automatic_parallel_plan(
        broken,
        model_profile=profile,
        selected_plan=selected,
    )

    assert any("selected_worker_count" in problem for problem in problems)
