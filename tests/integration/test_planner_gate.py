from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml
from examples.models.partition_stress_model import (
    build_partition_stress_model,
    make_training_batch,
)
from torch import nn

from shardgrid.common.config import ClusterConfig, TrainingConfig
from shardgrid.common.enums import BackendStatus, Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_engine_name, as_hostname, as_job_id, as_machine_id
from shardgrid.control.job_manager import JobManager, create_training_job
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.engines.models import (
    ParallelEngineCandidate,
    ParallelPlan,
    ParallelPlanCommunicationEdge,
    ParallelPlanPlacement,
    ParallelPlanProvenance,
    ParallelPlanStage,
    TrainingMemoryEstimate,
)
from shardgrid.jobs.models import JobStatus
from shardgrid.planner.execution_plan import (
    build_automatic_parallel_plan,
    validate_automatic_parallel_plan,
)
from shardgrid.planner.memory import MemoryEstimationConfig, build_model_profile
from shardgrid.planner.partitioning import build_partition_profile
from shardgrid.planner.placement import (
    JointPlacementPlan,
    StagePlacement,
    search_joint_partition_placement,
)
from shardgrid.planner.replay import validate_snapshot_replay
from shardgrid.planner.requirements import FeasibilityStatus
from shardgrid.planner.scoring import select_best_joint_placement_plan
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource
from shardgrid.workers.models import WorkerRuntime
from shardgrid.workers.probe import WindowsHostInfo, WorkerProbeResult

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class PlannerArtifacts:
    cluster_config: ClusterConfig
    training_config: TrainingConfig
    manager: JobManager
    cluster_state: object
    parallel_plan: ParallelPlan
    execution_plan: object
    snapshot_root: Path
    snapshot_metadata_path: Path


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


def _worker_resource(
    worker_id: str,
    *,
    machine_id: str,
    host: str,
    free_memory_mb: int,
    runtime_os: RuntimeOS = RuntimeOS.WSL2_LINUX,
    physical_os: PhysicalOS = PhysicalOS.WINDOWS,
    health: Health = Health.HEALTHY,
    enabled: bool = True,
    nccl_available: bool = True,
) -> WorkerResource:
    return WorkerResource(
        worker_id=worker_id,
        hostname=as_hostname(host),
        physical_os=physical_os,
        runtime_os=runtime_os,
        machine_id=as_machine_id(machine_id),
        enabled=enabled,
        conda_environment="shardgrid",
        conda_prefix="/opt/conda/envs/shardgrid",
        python_executable="/opt/conda/envs/shardgrid/bin/python",
        ip=host,
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


def _network_state(
    workers: list[WorkerResource],
    *,
    reachable: bool = True,
) -> NetworkState:
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
        network_id="planner-gate-net",
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
        network_state=_network_state(workers, reachable=reachable),
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


def _cluster_config(tmp_path: Path, *, worker_ids: tuple[str, ...]) -> ClusterConfig:
    worker_payloads = []
    for index, worker_id in enumerate(worker_ids, start=10):
        worker_payloads.append(
            {
                "id": worker_id,
                "machine_id": f"machine-{worker_id[-1]}",
                "physical_os": "windows",
                "runtime_os": "wsl2_linux",
                "runtime": "wsl2",
                "host": f"10.0.0.{index}",
                "ssh_user": "shardgrid",
                "ssh_port": 22,
                "runtime_distro": "Ubuntu-22.04",
                "conda_environment": "shardgrid",
                "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                "local_world_size": 1,
                "enabled": True,
            }
        )
    return ClusterConfig.from_dict(
        {
            "control": {
                "machine_id": "machine-a",
                "hostname": "control-a.local",
            },
            "jobs_root": str((tmp_path / "jobs").resolve()),
            "ssh": {},
            "runtime": {
                "python_executable": "python3",
                "conda_environment": "shardgrid",
                "conda_prefix": "/opt/conda/envs/shardgrid",
            },
            "network": {"rendezvous_port": 29500},
            "backend_preference": {
                "launcher": "ssh",
                "communication_backend": "nccl",
                "parallel_engine": "galvatron",
            },
            "manual_override": {},
            "workers": worker_payloads,
        }
    )


def _training_config(*, world_size: int, worker_ids: tuple[str, ...]) -> TrainingConfig:
    return TrainingConfig.from_dict(
        {
            "job": {
                "name": "planner-gate",
                "backend": "ssh",
                "communication_backend": "nccl",
            },
            "model": {
                "name": "partition-stress-model",
                "type": "hf_style",
                "stage_count": world_size,
            },
            "resources": {
                "world_size": world_size,
                "preferred_workers": list(worker_ids),
            },
            "artifacts": {
                "snapshot_name": "planner-gate",
                "keep_failed_snapshots": True,
                "transport": "auto",
            },
        }
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
        selected_reason="deterministic selected candidate",
    )


def _materialize_automatic_artifacts(tmp_path: Path) -> PlannerArtifacts:
    worker_ids = ("worker-a", "worker-b", "worker-c")
    config = _cluster_config(tmp_path, worker_ids=worker_ids)
    training_config = _training_config(world_size=3, worker_ids=worker_ids)
    manager = JobManager(config, source_root=Path(__file__).resolve().parents[2])
    workers = [
        _worker_resource(
            worker_id="worker-a",
            machine_id="machine-a",
            host="10.0.0.10",
            free_memory_mb=8,
        ),
        _worker_resource(
            worker_id="worker-b",
            machine_id="machine-b",
            host="10.0.0.11",
            free_memory_mb=8,
        ),
        _worker_resource(
            worker_id="worker-c",
            machine_id="machine-c",
            host="10.0.0.12",
            free_memory_mb=8,
            runtime_os=RuntimeOS.LINUX,
            physical_os=PhysicalOS.LINUX,
        ),
    ]
    cluster_state = _cluster_state(workers)

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
    partition = build_partition_profile(
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
        item
        for item in partition.candidates
        if item.stage_count == 3
        and item.hard_constraint_status is FeasibilityStatus.FEASIBLE
    )
    selected = select_best_joint_placement_plan(
        [
            _selected_plan_from_candidate(
                candidate,
                worker_ids=worker_ids,
                machine_ids=("machine-a", "machine-b", "machine-c"),
                free_memory_mb=(8, 8, 8),
            )
        ]
    )
    parallel_plan = build_automatic_parallel_plan(profile, selected)
    job = create_training_job(
        config_path=str(tmp_path / "train-planner-gate.yaml"),
        model=training_config.model.name,
        requested_world_size=training_config.resources.world_size,
        backend_preference=training_config.job.communication_backend,
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-t116-gate"),
    )
    snapshot_root = manager._artifact_store.snapshot_paths(job.job_id).root
    job = replace(job, snapshot_path=str(snapshot_root))
    snapshot = manager._artifact_store.create_snapshot(job)
    selected_workers = [
        worker
        for worker in config.workers
        if str(worker.worker_id) in selected.selected_worker_ids
    ]
    execution_plan = manager._build_execution_plan(
        job=job,
        training_config=training_config,
        parallel_plan=parallel_plan,
        workers=selected_workers,
        snapshot=snapshot,
    )
    status = JobStatus(
        job_id=job.job_id,
        state=JobState.SNAPSHOTTING,
        phase="plan",
        workers=[assignment.worker_id for assignment in execution_plan.workers],
        assignments=list(execution_plan.workers),
        runtime_environment_refs=manager._runtime_refs(execution_plan),
        backend=execution_plan.backend,
    )
    manager._write_snapshot_metadata(
        snapshot=snapshot,
        job=job,
        training_config=training_config,
        parallel_plan=parallel_plan,
        execution_plan=execution_plan,
        network_state=cluster_state.network_state,
        job_status=status,
        launch_metadata={"engine": "pytorch_pipeline", "plan_mode": "automatic"},
        dry_run=True,
    )
    return PlannerArtifacts(
        cluster_config=config,
        training_config=training_config,
        manager=manager,
        cluster_state=cluster_state,
        parallel_plan=parallel_plan,
        execution_plan=execution_plan,
        snapshot_root=snapshot_root,
        snapshot_metadata_path=snapshot_root / "diagnostics" / "snapshot-metadata.json",
    )


class _DryRunEngine:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def prepare(self, job_snapshot: object, execution_plan: object) -> object:
        del job_snapshot, execution_plan
        self.events.append("engine_prepare")
        raise AssertionError("dry-run must not call engine.prepare")

    def launch_metadata(self, parallel_plan: ParallelPlan) -> dict[str, object]:
        self.events.append("engine_launch_metadata")
        return {
            "engine": "galvatron",
            "plan_mode": "automatic",
            "parallel_plan_id": parallel_plan.parallel_plan_id,
        }


class _SelectedEngine:
    def __init__(self, events: list[str]) -> None:
        self.engine = _DryRunEngine(events)
        self.candidate = ParallelEngineCandidate(
            engine_id="galvatron",
            name=as_engine_name("galvatron"),
            status=BackendStatus.EXPERIMENTAL,
            capabilities=["automatic_partition", "ssh", "nccl"],
            limitations=["planner gate dry-run fixture"],
        )
        self.parallel_plan = ParallelPlan(
            parallel_plan_id="planner-gate-dry-run",
            engine=as_engine_name("galvatron"),
            model_name="tiny-transformer",
            world_size=2,
            stages=["stage0", "stage1"],
            partition_source="automatic",
            selected_candidate_id="candidate-dry-run",
            stage_metadata=[
                ParallelPlanStage(
                    stage_id="stage0",
                    rank=0,
                    module_ids=("embed",),
                    module_paths=("model.embed",),
                    start_index=0,
                    stop_index=1,
                    parameter_names_or_ranges=("embed.*",),
                    estimated_peak_training_memory=TrainingMemoryEstimate(
                        estimated_peak_bytes=1024,
                        safety_headroom_bytes=256,
                    ),
                    placement=ParallelPlanPlacement(
                        worker_id="worker-a",
                        rank=0,
                        machine_id="machine-a",
                        gpu_index=0,
                        usable_memory_before_bytes=8 * _BYTES_PER_MB,
                        remaining_memory_bytes=(8 * _BYTES_PER_MB) - 1280,
                        utilization_ratio=1280 / (8 * _BYTES_PER_MB),
                    ),
                ),
                ParallelPlanStage(
                    stage_id="stage1",
                    rank=1,
                    module_ids=("head",),
                    module_paths=("model.head",),
                    start_index=1,
                    stop_index=2,
                    parameter_names_or_ranges=("head.*",),
                    estimated_peak_training_memory=TrainingMemoryEstimate(
                        estimated_peak_bytes=1536,
                        safety_headroom_bytes=256,
                    ),
                    placement=ParallelPlanPlacement(
                        worker_id="worker-b",
                        rank=1,
                        machine_id="machine-b",
                        gpu_index=0,
                        usable_memory_before_bytes=8 * _BYTES_PER_MB,
                        remaining_memory_bytes=(8 * _BYTES_PER_MB) - 1792,
                        utilization_ratio=1792 / (8 * _BYTES_PER_MB),
                    ),
                ),
            ],
            communication_edges=[
                ParallelPlanCommunicationEdge(
                    source_stage_id="stage0",
                    target_stage_id="stage1",
                    source_module_id="embed",
                    target_module_id="head",
                    estimated_bytes_per_step=4096,
                )
            ],
            planning_provenance=ParallelPlanProvenance(
                partition_source="automatic",
                selected_candidate_id="candidate-dry-run",
                selected_worker_count=2,
                attempted_worker_counts=(2,),
                total_cross_worker_communication_bytes=4096,
                selected_reason="first feasible 2-worker plan",
            ),
        )
        self.rejected_engine_ids = ()


def _probe_result(resource: WorkerResource) -> WorkerProbeResult:
    return WorkerProbeResult(
        worker_resource=resource,
        worker_runtime=WorkerRuntime(
            worker_id=resource.worker_id,
            runtime_os=resource.runtime_os,
            runtime_version="Ubuntu",
            conda_environment="shardgrid",
            conda_prefix="/opt/conda/envs/shardgrid",
            python_executable="/opt/conda/envs/shardgrid/bin/python",
            torch_version="2.7.1",
            torch_cuda_version="12.4",
            cuda_available=True,
            nccl_available=True,
            gloo_available=True,
            health=Health.HEALTHY,
        ),
        windows_host=WindowsHostInfo(
            os_version="Windows 11" if resource.physical_os is PhysicalOS.WINDOWS else "Ubuntu",
            openssh_available=True,
            wsl_available=resource.runtime_os is RuntimeOS.WSL2_LINUX,
            nvidia_driver_visible=True,
            driver_name=resource.gpu_name,
        ),
        failures=(),
        health=Health.HEALTHY,
        probe_status="live",
    )


def _build_dry_run_manager(tmp_path: Path, events: list[str]) -> tuple[JobManager, Path]:
    worker_ids = ("worker-a", "worker-b")
    config = _cluster_config(tmp_path, worker_ids=worker_ids)
    training_path = tmp_path / "train-dry-run.yaml"
    training_path.write_text(
        yaml.safe_dump(
            _training_config(world_size=2, worker_ids=worker_ids).to_dict(),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    resources = {
        "worker-a": _worker_resource(
            "worker-a",
            machine_id="machine-a",
            host="10.0.0.10",
            free_memory_mb=8,
        ),
        "worker-b": _worker_resource(
            "worker-b",
            machine_id="machine-b",
            host="10.0.0.11",
            free_memory_mb=8,
        ),
    }

    def probe_worker(worker):
        events.append(f"probe:{worker.worker_id}")
        return _probe_result(resources[str(worker.worker_id)])

    def probe_network(worker_resources):
        del worker_resources
        events.append("network")
        return _network_state(list(resources.values()))

    def select_engine(engine_id, job, resources, network, *, registry=None):
        del engine_id, job, resources, network, registry
        events.append("select_engine")
        return _SelectedEngine(events)

    def launcher_factory(backend: str):
        del backend
        events.append("launcher_factory")
        raise AssertionError("dry-run must not create a launcher")

    manager = JobManager(
        config,
        probe_worker=probe_worker,
        probe_network=probe_network,
        select_engine=select_engine,
        launcher_factory=launcher_factory,
        source_root=Path(__file__).resolve().parents[2],
    )
    return manager, training_path


def test_planner_gate_rejects_peak_memory_overflow_using_training_peak_memory() -> None:
    model, sample, profile = _chain_profile(5)
    partition = build_partition_profile(
        model,
        profile,
        sample_args=(sample,),
        memory_config=_memory_config(),
        usable_memory_bytes=(12 * _BYTES_PER_MB, 12 * _BYTES_PER_MB),
        min_stage_count=2,
        max_stage_count=2,
    )
    plan = search_joint_partition_placement(
        model,
        profile,
        _cluster_state(
            [
                _worker_resource(
                    "worker-a",
                    machine_id="machine-a",
                    host="10.0.0.10",
                    free_memory_mb=12,
                ),
                _worker_resource(
                    "worker-b",
                    machine_id="machine-b",
                    host="10.0.0.11",
                    free_memory_mb=12,
                ),
            ]
        ),
        sample_args=(sample,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    candidate = partition.candidates[0]
    assert all(stage.parameter_bytes < 12 * _BYTES_PER_MB for stage in candidate.stages)
    assert any(
        (stage.estimated_peak_training_memory.planner_required_bytes or 0)
        > 12 * _BYTES_PER_MB
        for stage in candidate.stages
    )
    assert plan.status is FeasibilityStatus.INFEASIBLE
    assert any("peak" in reason and "usable" in reason for reason in plan.reasons)


def test_planner_gate_stops_on_first_feasible_worker_count_and_is_deterministic() -> None:
    model, sample, profile = _chain_profile(5)
    two_worker_cluster = _cluster_state(
        [
            _worker_resource(
                "worker-a",
                machine_id="machine-a",
                host="10.0.0.10",
                free_memory_mb=13,
            ),
            _worker_resource(
                "worker-b",
                machine_id="machine-b",
                host="10.0.0.11",
                free_memory_mb=13,
            ),
            _worker_resource(
                "worker-c",
                machine_id="machine-c",
                host="10.0.0.12",
                free_memory_mb=13,
            ),
        ]
    )

    first = search_joint_partition_placement(
        model,
        profile,
        two_worker_cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )
    second = search_joint_partition_placement(
        model,
        profile,
        two_worker_cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    assert first.status is FeasibilityStatus.FEASIBLE
    assert first.selected_worker_count == 2
    assert first.attempted_worker_counts == (2,)
    assert first.selected_worker_ids == second.selected_worker_ids
    assert first.partition_candidate is not None
    assert second.partition_candidate is not None
    assert first.partition_candidate.candidate_id == second.partition_candidate.candidate_id
    assert first.stage_placements == second.stage_placements
    for placement in first.stage_placements:
        assert placement.stage_required_bytes is not None
        assert placement.usable_memory_before_bytes is not None
        assert placement.stage_required_bytes <= placement.usable_memory_before_bytes


def test_planner_gate_only_tries_three_after_two_fails_and_handles_worker_constraints() -> None:
    model, sample, profile = _chain_profile(5)
    three_worker_cluster = _cluster_state(
        [
            _worker_resource(
                "worker-a",
                machine_id="machine-a",
                host="10.0.0.10",
                free_memory_mb=12,
            ),
            _worker_resource(
                "worker-b",
                machine_id="machine-b",
                host="10.0.0.11",
                free_memory_mb=12,
            ),
            _worker_resource(
                "worker-c",
                machine_id="machine-c",
                host="10.0.0.12",
                free_memory_mb=12,
            ),
        ]
    )
    fallback_plan = search_joint_partition_placement(
        model,
        profile,
        three_worker_cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )
    unhealthy_cluster = _cluster_state(
        [
            _worker_resource(
                "worker-a",
                machine_id="machine-a",
                host="10.0.0.10",
                free_memory_mb=13,
            ),
            _worker_resource(
                "worker-b",
                machine_id="machine-b",
                host="10.0.0.11",
                free_memory_mb=13,
                health=Health.FAILED,
            ),
            _worker_resource(
                "worker-c",
                machine_id="machine-c",
                host="10.0.0.12",
                free_memory_mb=13,
            ),
        ]
    )
    healthy_plan = search_joint_partition_placement(
        model,
        profile,
        unhealthy_cluster,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )
    unreachable_plan = search_joint_partition_placement(
        model,
        profile,
        _cluster_state(
            [
                _worker_resource(
                    "worker-a",
                    machine_id="machine-a",
                    host="10.0.0.10",
                    free_memory_mb=13,
                ),
                _worker_resource(
                    "worker-b",
                    machine_id="machine-b",
                    host="10.0.0.11",
                    free_memory_mb=13,
                ),
            ],
            reachable=False,
        ),
        sample_args=(sample,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    assert fallback_plan.status is FeasibilityStatus.FEASIBLE
    assert fallback_plan.selected_worker_count == 3
    assert fallback_plan.attempted_worker_counts == (2, 3)
    assert any(
        attempt.worker_count == 2 and attempt.status is FeasibilityStatus.INFEASIBLE
        for attempt in fallback_plan.attempts
    )
    assert healthy_plan.status is FeasibilityStatus.FEASIBLE
    assert "worker-b" not in healthy_plan.selected_worker_ids
    assert any("worker is unhealthy" in reason for reason in healthy_plan.reasons)
    assert unreachable_plan.status is FeasibilityStatus.INFEASIBLE
    assert any("not reachable" in reason for reason in unreachable_plan.reasons)


def test_planner_gate_handles_heterogeneous_vram_and_rejects_extreme_equal_capacity_imbalance(
) -> None:
    model, sample, profile = _chain_profile(5)
    heterogeneous_plan = search_joint_partition_placement(
        model,
        profile,
        _cluster_state(
            [
                _worker_resource(
                    "worker-a",
                    machine_id="machine-a",
                    host="10.0.0.10",
                    free_memory_mb=20,
                ),
                _worker_resource(
                    "worker-b",
                    machine_id="machine-b",
                    host="10.0.0.11",
                    free_memory_mb=8,
                ),
            ]
        ),
        sample_args=(sample,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    imbalanced_model = ImbalancedTwoStageModel()
    imbalanced_sample = torch.ones((2, 4))
    imbalanced_profile = build_model_profile(
        imbalanced_model,
        engine_id="pytorch_pipeline",
        model_name="imbalanced-two-stage",
        sample_args=(imbalanced_sample,),
        memory_config=_memory_config(),
        required_backends=("nccl",),
    )
    imbalanced_plan = search_joint_partition_placement(
        imbalanced_model,
        imbalanced_profile,
        _cluster_state(
            [
                _worker_resource(
                    "worker-a",
                    machine_id="machine-a",
                    host="10.0.0.10",
                    free_memory_mb=512,
                ),
                _worker_resource(
                    "worker-b",
                    machine_id="machine-b",
                    host="10.0.0.11",
                    free_memory_mb=512,
                ),
            ]
        ),
        sample_args=(imbalanced_sample,),
        memory_config=_memory_config(),
        max_worker_count=2,
    )

    assert heterogeneous_plan.status is FeasibilityStatus.FEASIBLE
    assert heterogeneous_plan.stage_placements[0].stage_required_bytes is not None
    assert heterogeneous_plan.stage_placements[1].stage_required_bytes is not None
    assert (
        heterogeneous_plan.stage_placements[0].stage_required_bytes
        > heterogeneous_plan.stage_placements[1].stage_required_bytes
    )
    assert imbalanced_plan.status is FeasibilityStatus.INFEASIBLE
    assert any(
        "capacity-aware imbalance is too extreme" in reason
        for reason in imbalanced_plan.reasons
    )


def test_planner_gate_preserves_automatic_partition_and_blocks_unsafe_replay(
    tmp_path: Path,
) -> None:
    artifacts = _materialize_automatic_artifacts(tmp_path)

    parallel_problems = validate_automatic_parallel_plan(artifacts.parallel_plan)
    execution_json = json.loads(
        (artifacts.snapshot_root / "plan" / "execution-plan.json").read_text()
    )
    execution_yaml = yaml.safe_load(
        (artifacts.snapshot_root / "plan" / "execution-plan.yaml").read_text()
    )
    parallel_json = json.loads(
        (artifacts.snapshot_root / "plan" / "original-parallel-plan.json").read_text()
    )
    parallel_yaml = yaml.safe_load(
        (artifacts.snapshot_root / "plan" / "original-parallel-plan.yaml").read_text()
    )
    metadata_json = json.loads(artifacts.snapshot_metadata_path.read_text())
    metadata_yaml = yaml.safe_load(
        (artifacts.snapshot_root / "diagnostics" / "snapshot-metadata.yaml").read_text()
    )
    safe_replay = validate_snapshot_replay(
        artifacts.snapshot_metadata_path,
        cluster_state=artifacts.cluster_state,
    )
    unsafe_replay = validate_snapshot_replay(
        artifacts.snapshot_metadata_path,
        cluster_state=_cluster_state(
            [
                _worker_resource(
                    "worker-a",
                    machine_id="machine-a",
                    host="10.0.0.10",
                    free_memory_mb=8,
                ),
                _worker_resource(
                    "worker-b",
                    machine_id="machine-b",
                    host="10.0.0.11",
                    free_memory_mb=0,
                ),
                _worker_resource(
                    "worker-c",
                    machine_id="machine-c",
                    host="10.0.0.12",
                    free_memory_mb=8,
                    health=Health.FAILED,
                    runtime_os=RuntimeOS.LINUX,
                    physical_os=PhysicalOS.LINUX,
                ),
            ]
        ),
    )

    assert parallel_problems == []
    assert artifacts.parallel_plan.partition_source == "automatic"
    assert artifacts.parallel_plan.planning_provenance is not None
    assert artifacts.parallel_plan.planning_provenance.selected_worker_count == 3
    assert len(artifacts.parallel_plan.stage_metadata) == 3
    assert len({stage.stage_id for stage in artifacts.parallel_plan.stage_metadata}) == 3
    assert len({stage.placement.worker_id for stage in artifacts.parallel_plan.stage_metadata}) == 3
    assert any(
        abs(int(edge.target_stage_id[-1]) - int(edge.source_stage_id[-1])) > 1
        for edge in artifacts.parallel_plan.communication_edges
    )
    assert execution_json == execution_yaml
    assert parallel_json == parallel_yaml
    assert metadata_json == metadata_yaml
    assert execution_json["labels"]["selected_worker_count"] == "3"
    assert execution_json["labels"]["partition_source"] == "automatic"
    assert execution_json["world_size"] == 3
    assert metadata_json["execution_plan_audit"]["planning"]["selected_worker_count"] == "3"
    assert metadata_json["execution_plan_audit"]["engine"] == "pytorch_pipeline"
    assert metadata_json["execution_plan_audit"]["backend"] == "nccl"
    assert metadata_json["execution_plan_audit"]["world_size"] == 3
    assert metadata_json["execution_plan_audit"]["master"]["port"] == 29500
    assert len(metadata_json["execution_plan_audit"]["assignments"]) == 3
    placement = metadata_json["execution_plan_audit"]["placement"]
    assignments = metadata_json["execution_plan_audit"]["assignments"]
    assert len(placement) == len(assignments)
    assert [
        (
            item["stage"],
            item["worker_id"],
            item["rank"],
            item["gpu_index"],
        )
        for item in placement
    ] == [
        (
            item["stage"],
            item["worker_id"],
            item["rank"],
            item["gpu_index"],
        )
        for item in assignments
    ]
    assert metadata_json["execution_plan_audit"]["original_plan"]["parallel_plan_ref"]
    assert safe_replay.safe is True
    assert safe_replay.status is FeasibilityStatus.FEASIBLE
    assert unsafe_replay.safe is False
    assert unsafe_replay.status is FeasibilityStatus.INFEASIBLE
    assert any(
        "usable memory dropped below saved stage peak" in reason
        for reason in unsafe_replay.reasons
    )
    assert any(
        "worker worker-c replay rejected" in reason
        for reason in unsafe_replay.reasons
    )

    placement_by_stage = {
        stage.stage_id: stage.placement
        for stage in artifacts.parallel_plan.stage_metadata
    }
    for assignment in artifacts.execution_plan.workers:
        placement = placement_by_stage[assignment.stage]
        assert placement is not None
        assert assignment.worker_id == placement.worker_id
        assert assignment.rank == placement.rank
        assert assignment.gpu_index == placement.gpu_index
        assert assignment.stage_metadata_ref is not None


def test_planner_gate_dry_run_does_not_launch_remote_ranks(tmp_path: Path) -> None:
    events: list[str] = []
    manager, training_path = _build_dry_run_manager(tmp_path, events)

    result = manager.run(training_path, job_id=as_job_id("job-t116-dry-run"), dry_run=True)

    assert result.status.state is JobState.SNAPSHOTTING
    assert result.status.phase == "plan"
    assert events == [
        "probe:worker-a",
        "probe:worker-b",
        "network",
        "select_engine",
        "engine_launch_metadata",
    ]
