from __future__ import annotations

from datetime import UTC, datetime

import torch
from examples.models.partition_stress_model import build_partition_stress_model, make_training_batch
from torch import nn

from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_machine_id, as_worker_id
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.planner.memory import MemoryEstimationConfig, build_model_profile
from shardgrid.planner.partitioning import build_partition_profile
from shardgrid.planner.placement import search_joint_partition_placement
from shardgrid.planner.requirements import FeasibilityStatus
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


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


class ChainModel(nn.Module):
    def __init__(self, *, depth: int, width: int = 512) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index != len(self.layers) - 1:
                x = self.act(x)
        return x


class ImbalancedTwoStageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.small = nn.Linear(4, 4)
        self.big = nn.Linear(4, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.big(self.small(x))


def _worker(
    worker_id: str,
    *,
    machine_id: str,
    free_memory_mb: int,
    enabled: bool = True,
    health: Health = Health.HEALTHY,
    nccl_available: bool = True,
) -> WorkerResource:
    return WorkerResource(
        worker_id=as_worker_id(worker_id),
        hostname=as_hostname(f"{worker_id}.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        machine_id=as_machine_id(machine_id),
        enabled=enabled,
        conda_environment="shardgrid",
        python_executable="/opt/conda/bin/python",
        ip=f"10.0.0.{len(worker_id)}",
        gpu_name="GPU",
        gpu_total_memory=free_memory_mb,
        gpu_free_memory=free_memory_mb,
        cuda_version="12.4",
        torch_version="2.7.1",
        torch_cuda_version="12.4",
        nccl_available=nccl_available,
        gloo_available=True,
        health=health,
        last_probe_at=_timestamp(),
    )


def _network(workers: list[WorkerResource], *, reachable: bool = True) -> NetworkState:
    links = [
        NetworkLink(
            source_worker_id=source.worker_id,
            target_worker_id=target.worker_id,
            source_ip=source.ip or "10.0.0.1",
            target_ip=target.ip or "10.0.0.2",
            interface="eth0",
            tcp_reachable=reachable,
            bandwidth_mbps=900.0 if reachable else None,
            latency_ms=1.5 if reachable else None,
            measured_at=_timestamp(),
            failure_reason=None if reachable else "timeout",
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


def _cluster_state(
    workers: list[WorkerResource],
    *,
    reachable: bool = True,
) -> object:
    return ResourceManager().build_cluster_state(
        workers,
        network_state=_network(workers, reachable=reachable),
        now=datetime(2026, 9, 3, 0, 0, tzinfo=UTC),
    )


def _chain_profile(depth: int) -> tuple[ChainModel, torch.Tensor, object]:
    model = ChainModel(depth=depth)
    sample = torch.ones((2, 512))
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name=f"chain-{depth}",
        sample_args=(sample,),
        memory_config=_memory_config(),
        required_backends=("nccl",),
    )
    return model, sample, profile


def test_two_worker_feasible_stops_without_trying_three() -> None:
    model, sample, profile = _chain_profile(5)
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=13),
            _worker("worker-b", machine_id="machine-b", free_memory_mb=13),
            _worker("worker-c", machine_id="machine-c", free_memory_mb=13),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    assert plan.status == FeasibilityStatus.FEASIBLE
    assert plan.selected_worker_count == 2
    assert plan.attempted_worker_counts == (2,)


def test_three_workers_are_tried_only_after_two_is_infeasible() -> None:
    model, sample, profile = _chain_profile(5)
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=12),
            _worker("worker-b", machine_id="machine-b", free_memory_mb=12),
            _worker("worker-c", machine_id="machine-c", free_memory_mb=12),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    assert plan.status == FeasibilityStatus.FEASIBLE
    assert plan.selected_worker_count == 3
    assert plan.attempted_worker_counts == (2, 3)
    assert any(
        attempt.worker_count == 2 and attempt.status == FeasibilityStatus.INFEASIBLE
        for attempt in plan.attempts
    )


def test_four_workers_are_tried_only_after_three_is_infeasible() -> None:
    model, sample, profile = _chain_profile(7)
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=10),
            _worker("worker-b", machine_id="machine-b", free_memory_mb=10),
            _worker("worker-c", machine_id="machine-c", free_memory_mb=10),
            _worker("worker-d", machine_id="machine-d", free_memory_mb=10),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    assert plan.status == FeasibilityStatus.FEASIBLE
    assert plan.selected_worker_count == 4
    assert plan.attempted_worker_counts == (2, 3, 4)


def test_parameter_bytes_do_not_override_training_peak_memory() -> None:
    model, sample, profile = _chain_profile(5)
    partition = build_partition_profile(
        model,
        profile,
        sample_args=(sample,),
        memory_config=_memory_config(),
        usable_memory_bytes=(12 * 1024 * 1024, 12 * 1024 * 1024),
        min_stage_count=2,
        max_stage_count=2,
    )
    candidate = partition.candidates[0]
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=12),
            _worker("worker-b", machine_id="machine-b", free_memory_mb=12),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    assert all(stage.parameter_bytes < 12 * 1024 * 1024 for stage in candidate.stages)
    assert any(
        (stage.estimated_peak_training_memory.planner_required_bytes or 0) > 12 * 1024 * 1024
        for stage in candidate.stages
    )
    assert plan.status == FeasibilityStatus.INFEASIBLE
    assert any("peak" in reason and "usable" in reason for reason in plan.reasons)


def test_capacity_aware_heterogeneous_workers_allow_uneven_partition() -> None:
    model, sample, profile = _chain_profile(5)
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=20),
            _worker("worker-b", machine_id="machine-b", free_memory_mb=8),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    assert plan.status == FeasibilityStatus.FEASIBLE
    assert plan.selected_worker_count == 2
    assert (
        plan.stage_placements[0].stage_required_bytes
        > plan.stage_placements[1].stage_required_bytes
    )
    assert plan.stage_placements[0].remaining_memory_bytes > 0
    assert plan.stage_placements[1].remaining_memory_bytes > 0


def test_extreme_imbalance_for_similar_gpus_is_rejected() -> None:
    model = ImbalancedTwoStageModel()
    sample = torch.ones((2, 4))
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="imbalanced-two-stage",
        sample_args=(sample,),
        memory_config=_memory_config(),
        required_backends=("nccl",),
    )
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=512),
            _worker("worker-b", machine_id="machine-b", free_memory_mb=512),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    assert plan.status == FeasibilityStatus.INFEASIBLE
    assert any("capacity-aware imbalance is too extreme" in reason for reason in plan.reasons)


def test_residual_skip_communication_is_preserved_in_selected_plan() -> None:
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
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=8),
            _worker("worker-b", machine_id="machine-b", free_memory_mb=8),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(inputs,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    assert plan.status == FeasibilityStatus.FEASIBLE
    assert plan.partition_candidate is not None
    assert plan.partition_candidate.estimated_bytes_per_step is not None
    assert plan.partition_candidate.estimated_bytes_per_step > 0
    assert any(
        abs(
            int(edge.target_module_id[1:]) - int(edge.source_module_id[1:])
        )
        > 1
        for edge in plan.partition_candidate.communication_edges
    )


def test_disabled_or_backend_ineligible_workers_are_not_selected() -> None:
    model, sample, profile = _chain_profile(5)
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=13),
            _worker(
                "worker-b",
                machine_id="machine-b",
                free_memory_mb=13,
                enabled=False,
            ),
            _worker(
                "worker-c",
                machine_id="machine-c",
                free_memory_mb=13,
                nccl_available=False,
            ),
            _worker("worker-d", machine_id="machine-d", free_memory_mb=13),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    assert plan.status == FeasibilityStatus.FEASIBLE
    assert plan.selected_worker_ids == ("worker-a", "worker-d")
    assert "worker is disabled in configuration" in " ".join(plan.reasons)


def test_duplicate_host_subset_is_rejected() -> None:
    model, sample, profile = _chain_profile(5)
    cluster = _cluster_state(
        [
            _worker("worker-a", machine_id="machine-a", free_memory_mb=13),
            _worker("worker-b", machine_id="machine-a", free_memory_mb=13),
        ]
    )

    plan = search_joint_partition_placement(
        model,
        profile,
        cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    assert plan.status == FeasibilityStatus.INFEASIBLE
    assert any("share physical host" in reason for reason in plan.reasons)
