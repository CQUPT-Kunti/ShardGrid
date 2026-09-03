from __future__ import annotations

from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_machine_id, as_worker_id
from shardgrid.engines.models import (
    AutomaticPartitionSupport,
    BoundaryTensorSpec,
    PartitionBoundary,
    PartitionSupportStatus,
)
from shardgrid.planner.requirements import (
    CommunicationRequirement,
    PlacementRequirements,
    WorkerEligibilityRequirements,
    evaluate_worker_eligibility,
    validate_partition_boundary,
    validate_placement_feasibility,
)
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


def _worker(
    worker_id: str,
    *,
    machine_id: str,
    enabled: bool = True,
    health: Health = Health.HEALTHY,
    runtime_os: RuntimeOS = RuntimeOS.WSL2_LINUX,
    gpu_total_memory: int | None = 8192,
    nccl_available: bool = True,
) -> WorkerResource:
    return WorkerResource(
        worker_id=as_worker_id(worker_id),
        machine_id=as_machine_id(machine_id),
        hostname=as_hostname(f"{worker_id}.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=runtime_os,
        enabled=enabled,
        conda_environment="shardgrid",
        python_executable="python",
        gpu_name="RTX",
        gpu_total_memory=gpu_total_memory,
        torch_version="2.7.1",
        cuda_version="12.4",
        nccl_available=nccl_available,
        gloo_available=True,
        health=health,
    )


def test_worker_eligibility_rejects_disabled_unhealthy_or_memory_fit_failures() -> None:
    result = evaluate_worker_eligibility(
        _worker("gpu0", machine_id="machine-a", enabled=False, health=Health.DEGRADED),
        WorkerEligibilityRequirements(required_backends=("nccl",)),
        estimated_stage_memory_mb=7000,
        reserved_memory_mb=2048,
    )

    assert result.eligible is False
    reasons = {item.code for item in result.violations}
    assert "worker_disabled" in reasons
    assert "worker_health" in reasons
    assert "memory_fit" in reasons


def test_worker_eligibility_accepts_generalized_worker_without_hardcoded_ids() -> None:
    result = evaluate_worker_eligibility(
        _worker("custom-worker", machine_id="machine-z"),
        WorkerEligibilityRequirements(required_backends=("nccl",)),
        estimated_stage_memory_mb=1024,
        reserved_memory_mb=512,
    )

    assert result.eligible is True
    assert result.usable_memory_mb == 7680


def test_partition_boundary_reports_unsupported_engine_or_illegal_boundary() -> None:
    boundary = PartitionBoundary(
        boundary_id="block0:block1",
        source_module="model.block0",
        target_module="model.block1",
        shared_parameter_names=("embed.weight",),
        boundary_tensors=(),
        forward_dependencies=("hidden",),
        backward_dependencies=(),
        status=PartitionSupportStatus.UNSUPPORTED,
        reasons=("shared parameters cross stage and engine does not support them",),
    )
    support = AutomaticPartitionSupport(
        engine_id="galvatron",
        supported_backends=("nccl",),
        boundaries=(boundary,),
    )

    result = validate_partition_boundary(boundary, support)

    assert result.feasible is False
    reasons = {item.code for item in result.violations}
    assert "boundary_status" in reasons
    assert "shared_parameters" in reasons
    assert "boundary_tensors" in reasons
    assert "boundary_dependencies" in reasons


def test_placement_feasibility_rejects_unknown_workers_duplicate_hosts_and_unreachable_edges() -> None:
    workers = (
        _worker("gpu-a", machine_id="machine-1"),
        _worker("gpu-b", machine_id="machine-1"),
    )
    network = NetworkState(
        network_id="net-1",
        workers=[as_worker_id("gpu-a"), as_worker_id("gpu-b")],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id("gpu-a"),
                target_worker_id=as_worker_id("gpu-b"),
                source_ip="10.0.0.1",
                target_ip="10.0.0.2",
                interface="eth0",
                tcp_reachable=False,
                failure_reason="route failed",
            )
        ],
    )

    result = validate_placement_feasibility(
        {"stage0": "gpu-a", "stage1": "gpu-b"},
        workers=workers,
        requirements=PlacementRequirements(
            stage_ids=("stage0", "stage1"),
            communication=(CommunicationRequirement("stage0", "stage1"),),
        ),
        worker_requirements=WorkerEligibilityRequirements(required_backends=("nccl",)),
        stage_memory_by_stage={"stage0": 1024, "stage1": 1024},
        network_state=network,
    )

    assert result.feasible is False
    reasons = {item.code for item in result.violations}
    assert "physical_host_conflict" in reasons
    assert "network_unreachable" in reasons or "network_missing" in reasons


def test_placement_feasibility_does_not_assume_stage_count_equals_worker_count() -> None:
    workers = (
        _worker("gpu-a", machine_id="machine-1"),
        _worker("gpu-b", machine_id="machine-2"),
        _worker("gpu-c", machine_id="machine-3"),
    )
    network = NetworkState(
        network_id="net-2",
        workers=[as_worker_id("gpu-a"), as_worker_id("gpu-b"), as_worker_id("gpu-c")],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id(source),
                target_worker_id=as_worker_id(target),
                source_ip="10.0.0.1",
                target_ip="10.0.0.2",
                interface="eth0",
                tcp_reachable=True,
            )
            for source in ("gpu-a", "gpu-b", "gpu-c")
            for target in ("gpu-a", "gpu-b", "gpu-c")
            if source != target
        ],
    )

    result = validate_placement_feasibility(
        {"stage0": "gpu-a", "stage1": "gpu-b"},
        workers=workers,
        requirements=PlacementRequirements(
            stage_ids=("stage0", "stage1"),
            allowed_stage_to_worker_ratio=(1, 2),
            communication=(CommunicationRequirement("stage0", "stage1"),),
        ),
        worker_requirements=WorkerEligibilityRequirements(required_backends=("nccl",)),
        stage_memory_by_stage={"stage0": 2048, "stage1": 1024},
        network_state=network,
    )

    assert result.feasible is True
