from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from examples.models.minimal_transformer import MinimalTransformer, MinimalTransformerConfig
from examples.models.stage0 import build_stage0
from examples.models.stage1 import build_stage1

from shardgrid.artifacts.collector import (
    ArtifactCollectionResult,
    ArtifactCollectionState,
    CollectedArtifact,
    CollectionStatus,
    WorkerArtifactCollection,
)
from shardgrid.artifacts.transport import (
    ArtifactTransferItemResult,
    ArtifactTransferResult,
    ArtifactTransferStatus,
    ArtifactTransportName,
)
from shardgrid.common.config import ClusterConfig, load_training_config
from shardgrid.common.enums import (
    BackendStatus,
    FailureStage,
    Health,
    JobState,
    PhysicalOS,
    RuntimeOS,
)
from shardgrid.common.errors import make_failure_record
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_worker_id,
)
from shardgrid.control.job_manager import JobManager, create_training_job
from shardgrid.control.status_store import StatusStore
from shardgrid.engines.models import ParallelPlan
from shardgrid.jobs.models import JobStatus
from shardgrid.launchers.base import (
    LauncherOperation,
    LauncherResult,
    LauncherResultStatus,
    WorkerResult,
)
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource
from shardgrid.workers.models import WorkerRuntime
from shardgrid.workers.probe import (
    ProbeFailure,
    WindowsHostInfo,
    WorkerProbeResult,
)


def _cluster_config(tmp_path: Path) -> ClusterConfig:
    return ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": str((tmp_path / "jobs").resolve()),
            "ssh": {},
            "runtime": {
                "python_executable": "python3",
                "conda_environment": "shardgrid",
                "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                "default_wsl_distro": "Ubuntu-22.04",
            },
            "network": {"rendezvous_port": 29500},
            "backend_preference": {
                "launcher": "ssh",
                "communication_backend": "nccl",
                "parallel_engine": "galvatron",
            },
            "manual_override": {},
            "workers": [
                {
                    "id": "gpu4060",
                    "machine_id": "machine-c",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.87.5.155",
                    "ssh_user": "shardgrid",
                    "runtime_distro": "Ubuntu-22.04",
                    "conda_environment": "shardgrid",
                    "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                },
                {
                    "id": "gpu1060",
                    "machine_id": "machine-d",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.87.5.15",
                    "ssh_user": "shardgrid",
                    "runtime_distro": "Ubuntu-22.04",
                    "conda_environment": "shardgrid",
                    "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                },
            ],
        }
    )


def _training_config(
    tmp_path: Path,
    *,
    consolidation_enabled: bool = False,
    consolidation_device: str = "auto",
    consolidation_required: bool = False,
) -> Path:
    path = tmp_path / "train-minimal.yaml"
    path.write_text(
        f"""
job:
  name: train-minimal
  backend: ssh
  communication_backend: nccl
model:
  name: tiny-sequential
  type: minimal_sequential
  stage_count: 2
resources:
  world_size: 2
  preferred_workers: [gpu4060, gpu1060]
artifacts:
  snapshot_name: train-minimal
  keep_failed_snapshots: true
  transport: auto
  checkpoint:
    consolidation:
      enabled: {"true" if consolidation_enabled else "false"}
      device: {consolidation_device}
      required: {"true" if consolidation_required else "false"}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _worker_resource(worker_id: str, host: str, gpu: str) -> WorkerResource:
    return WorkerResource(
        worker_id=worker_id,
        hostname=as_hostname(host),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        ip=host,
        gpu_name=gpu,
        gpu_total_memory=8192 if worker_id == "gpu4060" else 4096,
        gpu_free_memory=6144 if worker_id == "gpu4060" else 3072,
        compute_capability="8.9" if worker_id == "gpu4060" else "7.5",
        cuda_version="11.8",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        nccl_available=True,
        gloo_available=True,
        network_interface="eth3" if worker_id == "gpu4060" else "eth0",
        health=Health.HEALTHY,
        last_probe_at="2026-08-29T10:00:00+00:00",
    )


def _probe_result(resource: WorkerResource) -> WorkerProbeResult:
    return WorkerProbeResult(
        worker_resource=resource,
        worker_runtime=WorkerRuntime(
            worker_id=resource.worker_id,
            runtime_os=RuntimeOS.WSL2_LINUX,
            runtime_version="Ubuntu-22.04",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            torch_version="2.7.1+cu118",
            torch_cuda_version="11.8",
            cuda_available=True,
            nccl_available=True,
            gloo_available=True,
            health=Health.HEALTHY,
        ),
        windows_host=WindowsHostInfo(
            os_version="Windows 11",
            openssh_available=True,
            wsl_available=True,
            nvidia_driver_visible=True,
            driver_name=resource.gpu_name,
        ),
        failures=(),
        health=Health.HEALTHY,
        probe_status="live",
    )


def _network_state() -> NetworkState:
    return NetworkState(
        network_id="pair",
        workers=["gpu4060", "gpu1060"],
        links=[
            NetworkLink(
                source_worker_id="gpu4060",
                target_worker_id="gpu1060",
                source_ip="10.87.5.155",
                target_ip="10.87.5.15",
                interface="eth3",
                tcp_reachable=True,
                bandwidth_mbps=940.0,
                latency_ms=0.8,
                port=29500,
                measured_at="2026-08-29T10:00:00+00:00",
            ),
            NetworkLink(
                source_worker_id="gpu1060",
                target_worker_id="gpu4060",
                source_ip="10.87.5.15",
                target_ip="10.87.5.155",
                interface="eth0",
                tcp_reachable=True,
                bandwidth_mbps=940.0,
                latency_ms=0.9,
                port=29500,
                measured_at="2026-08-29T10:00:00+00:00",
            ),
        ],
        created_at="2026-08-29T10:00:00+00:00",
        selected_interfaces={"gpu4060": "eth3", "gpu1060": "eth0"},
    )


def _failure(stage: FailureStage, message: str) -> object:
    return make_failure_record(
        stage=stage,
        host="10.87.5.155",
        worker_id="gpu4060",
        message=message,
        recommended_action=f"fix {stage.value.lower()}",
    )


def _failure_stage_for_phase(phase: str) -> FailureStage:
    return {
        "rendezvous": FailureStage.RENDEZVOUS,
        "training": FailureStage.TRAIN,
        "checkpoint": FailureStage.CHECKPOINT,
    }.get(phase, FailureStage.LAUNCH)


def _checkpoint_model_config() -> dict[str, object]:
    config = MinimalTransformerConfig()
    return {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "max_seq_length": config.max_seq_length,
        "dropout": config.dropout,
    }


def _write_checkpoint_shard(path: Path, *, rank: int, stage: str) -> tuple[str, int]:
    model = build_stage0() if stage == "stage0" else build_stage1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    payload = {
        "checkpoint_version": 1,
        "task": "t074",
        "rank": rank,
        "world_size": 2,
        "stage_id": stage,
        "step": 20,
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": [1.0, 0.5, 0.25] if stage == "stage1" else [],
        "metadata": {
            "producer_id": "stage0",
            "consumer_id": "stage1",
            "producer_rank": 0,
            "consumer_rank": 1,
            "activation_shape": [2, 8, 128],
            "activation_dtype": "float32",
            "model_config": _checkpoint_model_config(),
            "learning_rate": 1e-3,
            "steps": 20,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


class FakeSelectedEngine:
    def __init__(self, events: list[str]) -> None:
        from shardgrid.engines.models import EnginePreparation

        self.events = events
        self.engine_id = "galvatron"
        self.candidate = type(
            "Candidate",
            (),
            {"engine_id": "galvatron", "status": BackendStatus.EXPERIMENTAL},
        )()
        self.parallel_plan = ParallelPlan(
            parallel_plan_id="static-parallel-plan-minimal",
            engine=as_engine_name("galvatron"),
            engine_plan_path="/var/tmp/shardgrid/original-external-plan.json",
            model_name="tiny-sequential",
            world_size=2,
            stages=["stage0", "stage1"],
            requirements={"plan_mode": "static"},
            limitations=["limited support"],
        )
        self.original_plan_path = self.parallel_plan.engine_plan_path
        self.rejected_engine_ids = ()
        self._preparation = EnginePreparation(
            engine_id="galvatron",
            status=BackendStatus.AVAILABLE,
            diagnostics=["ready"],
        )
        self.engine = self

    def prepare(self, job_snapshot: object, execution_plan: object):
        del job_snapshot, execution_plan
        self.events.append("engine_prepare")
        return self._preparation

    def launch_metadata(self, parallel_plan):
        del parallel_plan
        return {"engine": "galvatron"}


class FakeLauncher:
    def __init__(
        self,
        events: list[str],
        *,
        prepare_status: LauncherResultStatus = LauncherResultStatus.SUCCESS,
        distribute_status: LauncherResultStatus = LauncherResultStatus.SUCCESS,
        launch_status: LauncherResultStatus = LauncherResultStatus.SUCCESS,
        stop_status: LauncherResultStatus = LauncherResultStatus.NOOP,
        monitor_state: JobState = JobState.COMPLETED,
        monitor_phase: str = "checkpoint",
        monitor_sequence: list[tuple[JobState, str]] | None = None,
    ) -> None:
        self.events = events
        self.prepare_status = prepare_status
        self.distribute_status = distribute_status
        self.launch_status = launch_status
        self.stop_status = stop_status
        self.monitor_state = monitor_state
        self.monitor_phase = monitor_phase
        self.monitor_sequence = list(monitor_sequence or [])

    def prepare(self, context):
        self.events.append("launcher_prepare")
        return _launcher_result(
            LauncherOperation.PREPARE,
            self.prepare_status,
            context.job.job_id,
            next_state=JobState.DISTRIBUTING,
            failure_stage=FailureStage.DISTRIBUTE,
        )

    def distribute(self, context):
        self.events.append("launcher_distribute")
        return _launcher_result(
            LauncherOperation.DISTRIBUTE,
            self.distribute_status,
            context.job.job_id,
            next_state=JobState.DISTRIBUTING,
            failure_stage=FailureStage.DISTRIBUTE,
        )

    def launch(self, context):
        self.events.append("launcher_launch")
        return _launcher_result(
            LauncherOperation.LAUNCH,
            self.launch_status,
            context.job.job_id,
            next_state=JobState.LAUNCHING,
            failure_stage=FailureStage.LAUNCH,
        )

    def monitor(self, context):
        self.events.append("launcher_monitor")
        if self.monitor_sequence:
            state, phase = self.monitor_sequence.pop(0)
        else:
            state, phase = self.monitor_state, self.monitor_phase
        status = JobStatus(
            job_id=context.job.job_id,
            state=state,
            phase=phase,
            workers=[assignment.worker_id for assignment in context.execution_plan.workers],
            assignments=list(context.execution_plan.workers),
            runtime_environment_refs=dict(context.runtime_environment_refs),
            backend=as_backend_name("nccl"),
            started_at="2026-08-29T12:00:00+00:00",
            latest_loss=0.25 if state is JobState.COMPLETED else None,
            loss_history=[1.0, 0.5, 0.25] if state is JobState.COMPLETED else [],
            final_metrics=({"final_loss": 0.25} if state is JobState.COMPLETED else {}),
            checkpoint_ref=(
                "checkpoint/files/gpu1060/rank1-stage1/model.pt"
                if state is JobState.COMPLETED
                else None
            ),
            failure=(
                None
                if state is not JobState.FAILED
                else _failure(
                    _failure_stage_for_phase(phase),
                    f"{phase} failed",
                )
            ),
        )
        path = Path(context.snapshot.diagnostics_path) / "job-status.json"
        path.write_text(json.dumps(status.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        result_status = (
            LauncherResultStatus.SUCCESS
            if state in {
                JobState.COMPLETED,
                JobState.CHECKPOINTING,
                JobState.LAUNCHING,
                JobState.RENDEZVOUS,
                JobState.TRAINING,
            }
            else LauncherResultStatus.FAILED
        )
        return _launcher_result(
            LauncherOperation.MONITOR,
            result_status,
            context.job.job_id,
            next_state=state,
            failure_stage=FailureStage.RENDEZVOUS,
        )

    def stop(self, context):
        self.events.append("launcher_stop")
        return _launcher_result(
            LauncherOperation.STOP,
            self.stop_status,
            context.job.job_id,
            next_state=context.job_status.state,
            failure_stage=FailureStage.STOP,
        )


class FakeCollector:
    def __init__(self, events: list[str], status: CollectionStatus = CollectionStatus.SUCCESS):
        self.events = events
        self.status = status
        self.calls: list[object] = []

    def collect(self, snapshot, *, sources, secrets=(), artifact_paths=None):
        del secrets, artifact_paths
        self.events.append("collector_collect")
        self.calls.append((snapshot, tuple(sources)))
        workers = []
        for source in sources:
            artifacts = ()
            checkpoint_state = ArtifactCollectionState.PARTIAL
            if self.status is CollectionStatus.SUCCESS:
                local_path = (
                    Path(snapshot.checkpoint_path)
                    / "files"
                    / source.worker_id
                    / f"rank{source.rank}-{source.stage}"
                    / f"checkpoint_rank{source.rank}.pt"
                )
                checksum, size = _write_checkpoint_shard(
                    local_path,
                    rank=source.rank,
                    stage=source.stage or f"stage{source.rank}",
                )
                remote_relpath = (
                    source.checkpoint_paths[0]
                    if source.checkpoint_paths
                    else f"checkpoint/checkpoint_rank{source.rank}.pt"
                )
                artifacts = (
                    CollectedArtifact(
                        worker_id=source.worker_id,
                        rank=source.rank,
                        stage=source.stage,
                        artifact_type="checkpoint_file",
                        relative_path=remote_relpath,
                        remote_path=f"{source.remote_root}/{remote_relpath}",
                        local_path=str(local_path),
                        status=ArtifactCollectionState.COMPLETE,
                        size_bytes=size,
                        checksum=checksum,
                    ),
                )
                checkpoint_state = ArtifactCollectionState.COMPLETE
            workers.append(
                WorkerArtifactCollection(
                    worker_id=source.worker_id,
                    host=source.host,
                    rank=source.rank,
                    stage=source.stage,
                    status=self.status,
                    checkpoint_state=checkpoint_state,
                    artifacts=artifacts,
                )
            )
        return ArtifactCollectionResult(
            job_id=Path(snapshot.root_path).name,
            snapshot_root=snapshot.root_path,
            status=self.status,
            workers=tuple(workers),
        )


def _launcher_result(
    operation: LauncherOperation,
    status: LauncherResultStatus,
    job_id,
    *,
    next_state: JobState,
    failure_stage: FailureStage,
) -> LauncherResult:
    failure = None
    if status is not LauncherResultStatus.SUCCESS:
        failure = _failure(failure_stage, f"{operation.value} failed")
    return LauncherResult(
        operation=operation,
        status=status,
        backend="ssh",
        job_id=str(job_id),
        worker_results=(
            WorkerResult(worker_id="gpu4060", status=status, failure=failure),
            WorkerResult(worker_id="gpu1060", status=status, failure=failure),
        ),
        failure=failure,
        next_job_state=next_state,
    )


def _build_manager(
    tmp_path: Path,
    events: list[str],
    *,
    probe_fail: bool = False,
    network_fail: bool = False,
    plan_fail: bool = False,
    launcher_prepare_status: LauncherResultStatus = LauncherResultStatus.SUCCESS,
    launcher_distribute_status: LauncherResultStatus = LauncherResultStatus.SUCCESS,
    launcher_launch_status: LauncherResultStatus = LauncherResultStatus.SUCCESS,
    launcher_stop_status: LauncherResultStatus = LauncherResultStatus.NOOP,
    monitor_state: JobState = JobState.COMPLETED,
    monitor_phase: str = "checkpoint",
    monitor_sequence: list[tuple[JobState, str]] | None = None,
    collect_status: CollectionStatus = CollectionStatus.SUCCESS,
) -> tuple[JobManager, Path]:
    config = _cluster_config(tmp_path)
    config_path = _training_config(tmp_path)
    workers = {
        "gpu4060": _worker_resource("gpu4060", "10.87.5.155", "RTX 4060"),
        "gpu1060": _worker_resource("gpu1060", "10.87.5.15", "GTX 1650"),
    }

    def probe_worker(worker):
        events.append(f"probe:{worker.worker_id}")
        if probe_fail and str(worker.worker_id) == "gpu4060":
            return WorkerProbeResult(
                worker_resource=replace(workers["gpu4060"], health=Health.FAILED),
                worker_runtime=WorkerRuntime(
                    worker_id=workers["gpu4060"].worker_id,
                    runtime_os=RuntimeOS.WSL2_LINUX,
                    runtime_version="Ubuntu-22.04",
                    health=Health.FAILED,
                ),
                windows_host=WindowsHostInfo(
                    os_version="Windows 11",
                    openssh_available=True,
                    wsl_available=True,
                    nvidia_driver_visible=False,
                    driver_name=None,
                ),
                failures=(
                    ProbeFailure(
                        layer="wsl_runtime",
                        check="cuda",
                        message="CUDA missing",
                    ),
                ),
                health=Health.FAILED,
                probe_status="live",
            )
        return _probe_result(workers[str(worker.worker_id)])

    def probe_network(worker_resources):
        events.append("network")
        if network_fail:
            raise RuntimeError("network probe failed")
        assert len(worker_resources) == 2
        return _network_state()

    def select_engine(engine_id, job, resources, network, *, registry=None):
        del engine_id, job, resources, network, registry
        events.append("select_engine")
        if plan_fail:
            raise RuntimeError("engine plan rejected")
        return FakeSelectedEngine(events)

    manager = JobManager(
        config,
        probe_worker=probe_worker,
        probe_network=probe_network,
        select_engine=select_engine,
        launcher_factory=lambda backend: FakeLauncher(
            events,
            prepare_status=launcher_prepare_status,
            distribute_status=launcher_distribute_status,
            launch_status=launcher_launch_status,
            stop_status=launcher_stop_status,
            monitor_state=monitor_state,
            monitor_phase=monitor_phase,
            monitor_sequence=monitor_sequence,
        ),
        artifact_collector=FakeCollector(events, status=collect_status),
        source_root=Path(__file__).resolve().parents[2],
    )
    return manager, config_path


def test_orchestrates_full_training_lifecycle(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(tmp_path, events)

    result = manager.run(config_path, job_id=as_job_id("job-101"))
    metadata = json.loads(
        (Path(result.snapshot.diagnostics_path) / "snapshot-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    persisted = JobStatus.from_dict(
        json.loads(
            (Path(result.snapshot.diagnostics_path) / "job-status.json").read_text(
                encoding="utf-8"
            )
        )
    )
    canonical = StatusStore(tmp_path / "jobs").load(result.job.job_id)

    assert result.status.state is JobState.COMPLETED
    assert result.status.phase == "checkpoint"
    assert result.status.checkpoint_ref == "checkpoint/manifest.json"
    assert canonical.state is JobState.COMPLETED
    assert canonical.phase == persisted.phase
    assert canonical.assignments == persisted.assignments
    assert canonical.final_metrics == persisted.final_metrics
    assert canonical.checkpoint_ref == persisted.checkpoint_ref
    assert result.execution_plan.master.port == 29500
    assert persisted.final_metrics["final_loss"] == 0.25
    assert persisted.checkpoint_ref == "checkpoint/manifest.json"
    assert metadata["job_status_path"].endswith("job-status.json")
    manifest = json.loads((Path(result.snapshot.checkpoint_path) / "manifest.json").read_text())
    assert manifest["checkpoint_ref"] == "checkpoint/manifest.json"
    assert manifest["required_shard_count"] == 2
    assert manifest["optional_artifacts"]["consolidated_model"]["status"] == "not_requested"
    assert "consolidated_model_ref" not in manifest
    assert events == [
        "probe:gpu4060",
        "probe:gpu1060",
        "network",
        "select_engine",
        "engine_prepare",
        "launcher_prepare",
        "launcher_distribute",
        "launcher_launch",
        "launcher_monitor",
        "collector_collect",
    ]


def test_collection_failure_persists_diagnostics_and_first_artifact_failure(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(tmp_path, events)

    class FailingCollector:
        def collect(self, snapshot, *, sources, secrets=(), artifact_paths=None):
            del secrets, artifact_paths
            source = tuple(sources)[0]
            failure = make_failure_record(
                stage=FailureStage.CHECKPOINT,
                host=source.host,
                worker_id=source.worker_id,
                command="scp host:.shardgrid/artifacts/job/checkpoint.pt local",
                exit_code=1,
                message="ARTIFACT_SCP_FAILED: scp failed",
                recommended_action="inspect artifact pull diagnostics and retry",
            )
            artifact = CollectedArtifact(
                worker_id=source.worker_id,
                rank=source.rank,
                stage=source.stage,
                artifact_type="checkpoint_file",
                relative_path="checkpoint/checkpoint_rank0.pt",
                remote_path=f"{source.remote_root}/checkpoint/checkpoint_rank0.pt",
                local_path=f"{snapshot.checkpoint_path}/files/checkpoint_rank0.pt",
                status=ArtifactCollectionState.FAILED,
                recorded_command=failure.command,
                failure_class="ARTIFACT_SCP_FAILED",
                stderr_summary="scp failed",
                exit_code=1,
                failure=failure,
            )
            worker = WorkerArtifactCollection(
                worker_id=source.worker_id,
                host=source.host,
                rank=source.rank,
                stage=source.stage,
                status=CollectionStatus.PARTIAL,
                checkpoint_state=ArtifactCollectionState.PARTIAL,
                artifacts=(artifact,),
            )
            return ArtifactCollectionResult(
                job_id=str(snapshot.job_id),
                snapshot_root=snapshot.root_path,
                status=CollectionStatus.PARTIAL,
                workers=(worker,),
            )

    manager._artifact_collector = FailingCollector()

    result = manager.run(config_path, job_id=as_job_id("job-101-collection-fail"))
    diagnostics_path = Path(result.snapshot.diagnostics_path) / "artifact-collection.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    canonical = StatusStore(tmp_path / "jobs").load(result.job.job_id)
    mirror = StatusStore(tmp_path / "jobs").load_path(
        Path(result.snapshot.diagnostics_path) / "job-status.json"
    )

    assert result.status.state is JobState.FAILED
    assert canonical.state is JobState.FAILED
    assert canonical.phase == "checkpoint"
    assert mirror.state is JobState.FAILED
    assert mirror.phase == canonical.phase
    assert mirror.failure == canonical.failure
    assert result.status.failure is not None
    assert result.status.failure.message == "ARTIFACT_SCP_FAILED: scp failed"
    assert result.status.failure.exit_code == 1
    assert diagnostics["workers"][0]["artifacts"][0]["failure_class"] == "ARTIFACT_SCP_FAILED"
    assert diagnostics["workers"][0]["artifacts"][0]["stderr_summary"] == "scp failed"


def test_probe_failure_stops_before_network(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(tmp_path, events, probe_fail=True)

    result = manager.run(config_path, job_id=as_job_id("job-101-probe-fail"))

    assert result.status.state is JobState.FAILED
    assert result.status.phase == "probe"
    assert result.status.failure is not None
    assert result.status.failure.stage is FailureStage.PROBE
    assert events == ["probe:gpu4060", "probe:gpu1060"]


def test_network_failure_stops_before_plan(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(tmp_path, events, network_fail=True)

    result = manager.run(config_path, job_id=as_job_id("job-101-network-fail"))

    assert result.status.state is JobState.FAILED
    assert result.status.phase == "probe"
    assert result.status.failure is not None
    assert result.status.failure.stage is FailureStage.NETWORK
    assert events == ["probe:gpu4060", "probe:gpu1060", "network"]


def test_plan_failure_stops_before_distribution(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(tmp_path, events, plan_fail=True)

    result = manager.run(config_path, job_id=as_job_id("job-101-plan-fail"))

    assert result.status.state is JobState.FAILED
    assert result.status.phase == "plan"
    assert result.status.failure is not None
    assert result.status.failure.stage is FailureStage.PLAN
    assert events == ["probe:gpu4060", "probe:gpu1060", "network", "select_engine"]


def test_distribute_failure_stops_before_launch(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(
        tmp_path,
        events,
        launcher_distribute_status=LauncherResultStatus.FAILED,
    )

    result = manager.run(config_path, job_id=as_job_id("job-101-distribute-fail"))

    assert result.status.state is JobState.FAILED
    assert result.status.phase == "distribute"
    assert result.status.failure is not None
    assert result.status.failure.stage is FailureStage.DISTRIBUTE
    assert "launcher_launch" not in events


def test_launch_failure_stops_before_monitor(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(
        tmp_path,
        events,
        launcher_launch_status=LauncherResultStatus.FAILED,
    )

    result = manager.run(config_path, job_id=as_job_id("job-101-launch-fail"))

    assert result.status.state is JobState.FAILED
    assert result.status.phase == "launch"
    assert result.status.failure is not None
    assert result.status.failure.stage is FailureStage.LAUNCH
    assert "launcher_monitor" not in events


@pytest.mark.parametrize(
    ("monitor_phase", "expected_stage"),
    [
        ("rendezvous", FailureStage.RENDEZVOUS),
        ("training", FailureStage.TRAIN),
    ],
)
def test_monitor_failure_stops_before_collection(
    tmp_path: Path,
    monitor_phase: str,
    expected_stage: FailureStage,
) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(
        tmp_path,
        events,
        monitor_state=JobState.FAILED,
        monitor_phase=monitor_phase,
    )

    result = manager.run(
        config_path,
        job_id=as_job_id(f"job-101-{monitor_phase}-fail"),
    )

    assert result.status.state is JobState.FAILED
    assert result.status.phase == monitor_phase
    assert result.status.failure is not None
    assert result.status.failure.stage is expected_stage
    assert "launcher_stop" in events
    assert "collector_collect" not in events


def test_execution_plan_uses_resolved_rendezvous_port(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(tmp_path, events)
    manager.cluster_config = replace(
        manager.cluster_config,
        network=replace(manager.cluster_config.network, rendezvous_port=29617),
    )

    result = manager.run(config_path, job_id=as_job_id("job-101-rendezvous-port"))

    assert result.execution_plan.master.port == 29617


def test_checkpoint_failure_after_collection_marks_job_failed(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(
        tmp_path,
        events,
        collect_status=CollectionStatus.FAILED,
    )

    result = manager.run(config_path, job_id=as_job_id("job-101-checkpoint-fail"))

    assert result.status.state is JobState.FAILED
    assert result.status.phase == "checkpoint"
    assert result.status.failure is not None
    assert result.status.failure.stage is FailureStage.CHECKPOINT
    assert events[-1] == "collector_collect"


def test_monitor_waits_for_terminal_state_before_checkpoint_collection(tmp_path: Path) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(
        tmp_path,
        events,
        monitor_sequence=[
            (JobState.LAUNCHING, "launch"),
            (JobState.RENDEZVOUS, "rendezvous"),
            (JobState.COMPLETED, "checkpoint"),
        ],
    )

    result = manager.run(config_path, job_id=as_job_id("job-101-monitor-waits"))

    assert result.status.state is JobState.COMPLETED
    assert result.status.checkpoint_ref is not None
    assert result.status.final_metrics["final_loss"] == 0.25
    assert events.count("launcher_monitor") == 3
    assert events[-1] == "collector_collect"


def test_finalize_checkpoint_bundle_writes_manifest_and_consolidated_model(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    manager, _ = _build_manager(tmp_path, events)
    job = create_training_job(
        job_id=as_job_id("job-101-snapshot"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    snapshot = manager._artifact_store.create_snapshot(job)
    training_config = load_training_config(
        _training_config(tmp_path, consolidation_enabled=True, consolidation_device="cpu")
    )
    execution_plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("ssh"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, stage="stage0"),
            WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1, stage="stage1"),
        ],
    )
    status = JobStatus(
        job_id=job.job_id,
        state=JobState.CHECKPOINTING,
        phase="checkpoint",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        assignments=list(execution_plan.workers),
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        backend=as_backend_name("ssh"),
        final_metrics={"final_loss": 0.25},
    )

    collection_result = manager._collect_artifacts(
        snapshot,
        training_config,
        execution_plan,
        status,
    )
    payload = manager._finalize_checkpoint_bundle(
        snapshot=snapshot,
        training_config=training_config,
        execution_plan=execution_plan,
        current=status,
        collection_result=collection_result,
    )

    manifest = json.loads((Path(snapshot.checkpoint_path) / "manifest.json").read_text())
    consolidated = torch.load(
        Path(snapshot.checkpoint_path) / "consolidated_model.pt",
        weights_only=False,
    )
    model = MinimalTransformer(config=MinimalTransformerConfig())
    model.load_state_dict(consolidated["model_state_dict"], strict=True)

    assert payload["checkpoint_ref"] == "checkpoint/manifest.json"
    assert payload["consolidated_model_ref"] == "checkpoint/consolidated_model.pt"
    assert payload["optional_artifacts"]["consolidated_model"]["status"] == "complete"
    assert [shard["stage"] for shard in manifest["shards"]] == ["stage0", "stage1"]
    assert consolidated["checkpoint_ref"] == "checkpoint/manifest.json"


def test_finalize_checkpoint_bundle_skips_torch_load_when_consolidation_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    manager, _ = _build_manager(tmp_path, events)
    job = create_training_job(
        job_id=as_job_id("job-101-manifest-only"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    snapshot = manager._artifact_store.create_snapshot(job)
    training_config = load_training_config(_training_config(tmp_path))
    execution_plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("ssh"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, stage="stage0"),
            WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1, stage="stage1"),
        ],
    )
    status = JobStatus(
        job_id=job.job_id,
        state=JobState.CHECKPOINTING,
        phase="checkpoint",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        assignments=list(execution_plan.workers),
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        backend=as_backend_name("ssh"),
        final_metrics={"final_loss": 0.25},
    )
    collection_result = manager._collect_artifacts(
        snapshot,
        training_config,
        execution_plan,
        status,
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(torch, "load", fail_load)

    payload = manager._finalize_checkpoint_bundle(
        snapshot=snapshot,
        training_config=training_config,
        execution_plan=execution_plan,
        current=status,
        collection_result=collection_result,
    )

    manifest = json.loads((Path(snapshot.checkpoint_path) / "manifest.json").read_text())
    assert payload["checkpoint_ref"] == "checkpoint/manifest.json"
    assert payload["optional_artifacts"]["consolidated_model"]["status"] == "not_requested"
    assert manifest["optional_artifacts"]["consolidated_model"]["status"] == "not_requested"
    assert (Path(snapshot.checkpoint_path) / "consolidated_model.pt").exists() is False


@pytest.mark.parametrize(
    ("requested", "cuda_available", "expected"),
    [
        ("cpu", False, "cpu"),
        ("auto", False, "cpu"),
        ("auto", True, "cuda"),
        ("cuda", True, "cuda"),
    ],
)
def test_resolve_consolidation_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    cuda_available: bool,
    expected: str,
) -> None:
    manager, _ = _build_manager(tmp_path, [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert manager._resolve_consolidation_device(requested) == expected


def test_resolve_consolidation_device_rejects_unavailable_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _build_manager(tmp_path, [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        ValueError,
        match="consolidation device 'cuda' requires CUDA on the finalization host",
    ):
        manager._resolve_consolidation_device("cuda")


def test_consolidation_uses_explicit_cpu_map_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    manager, _ = _build_manager(tmp_path, events)
    job = create_training_job(
        job_id=as_job_id("job-101-consolidate-cpu"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    snapshot = manager._artifact_store.create_snapshot(job)
    training_config = load_training_config(
        _training_config(tmp_path, consolidation_enabled=True, consolidation_device="cpu")
    )
    execution_plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("ssh"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, stage="stage0"),
            WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1, stage="stage1"),
        ],
    )
    status = JobStatus(
        job_id=job.job_id,
        state=JobState.CHECKPOINTING,
        phase="checkpoint",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        assignments=list(execution_plan.workers),
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        backend=as_backend_name("ssh"),
        final_metrics={"final_loss": 0.25},
    )
    collection_result = manager._collect_artifacts(
        snapshot,
        training_config,
        execution_plan,
        status,
    )
    original_load = torch.load
    shard_paths = {
        str(Path(artifact.local_path))
        for worker in collection_result.workers
        for artifact in worker.artifacts
        if artifact.artifact_type == "checkpoint_file"
    }
    seen_map_locations: dict[str, object] = {}

    def recording_load(path, *args, **kwargs):
        resolved = str(Path(path))
        if resolved in shard_paths:
            seen_map_locations[resolved] = kwargs.get("map_location")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    payload = manager._finalize_checkpoint_bundle(
        snapshot=snapshot,
        training_config=training_config,
        execution_plan=execution_plan,
        current=status,
        collection_result=collection_result,
    )

    assert payload["optional_artifacts"]["consolidated_model"]["resolved_device"] == "cpu"
    assert seen_map_locations
    assert set(seen_map_locations.values()) == {"cpu"}


def test_optional_consolidation_failure_does_not_fail_completed_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    manager, _ = _build_manager(tmp_path, events)
    config_path = _training_config(tmp_path, consolidation_enabled=True, consolidation_device="cpu")

    def fail_consolidation(**kwargs):
        del kwargs
        raise RuntimeError("cpu export failed")

    monkeypatch.setattr(manager, "_write_consolidated_model", fail_consolidation)

    result = manager.run(config_path, job_id=as_job_id("job-101-optional-export-fail"))
    manifest = json.loads((Path(result.snapshot.checkpoint_path) / "manifest.json").read_text())
    checkpoint_metadata = json.loads(
        (Path(result.snapshot.checkpoint_path) / "checkpoint-metadata.json").read_text()
    )

    assert result.status.state is JobState.COMPLETED
    assert result.status.checkpoint_ref == "checkpoint/manifest.json"
    assert result.status.final_metrics["final_loss"] == 0.25
    assert manifest["optional_artifacts"]["consolidated_model"]["status"] == "failed"
    assert manifest["optional_artifacts"]["consolidated_model"]["message"] == "cpu export failed"
    assert checkpoint_metadata["optional_artifacts"]["consolidated_model"]["status"] == "failed"


def test_finalize_checkpoint_bundle_requires_all_assignments(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    manager, _ = _build_manager(tmp_path, events)
    job = create_training_job(
        job_id=as_job_id("job-101-missing-shard"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=3,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-triple/shardgrid",
    )
    snapshot = manager._artifact_store.create_snapshot(job)
    training_config = load_training_config(_training_config(tmp_path))
    execution_plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("ssh"),
        world_size=3,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, stage="stage0"),
            WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1, stage="stage1"),
            WorkerAssignment(worker_id=as_worker_id("gpu3090"), rank=2, stage="stage2"),
        ],
    )
    status = JobStatus(
        job_id=job.job_id,
        state=JobState.CHECKPOINTING,
        phase="checkpoint",
        workers=[
            as_worker_id("gpu4060"),
            as_worker_id("gpu1060"),
            as_worker_id("gpu3090"),
        ],
        assignments=list(execution_plan.workers),
        runtime_environment_refs={
            "0": "env:gpu4060/shardgrid",
            "1": "env:gpu1060/shardgrid",
            "2": "env:gpu3090/shardgrid",
        },
        backend=as_backend_name("ssh"),
        final_metrics={"final_loss": 0.25},
    )
    local_rank0 = (
        Path(snapshot.checkpoint_path)
        / "files"
        / "gpu4060"
        / "rank0-stage0"
        / "checkpoint_rank0.pt"
    )
    checksum0, size0 = _write_checkpoint_shard(local_rank0, rank=0, stage="stage0")
    local_rank1 = (
        Path(snapshot.checkpoint_path)
        / "files"
        / "gpu1060"
        / "rank1-stage1"
        / "checkpoint_rank1.pt"
    )
    checksum1, size1 = _write_checkpoint_shard(local_rank1, rank=1, stage="stage1")
    collection_result = ArtifactCollectionResult(
        job_id=str(job.job_id),
        snapshot_root=snapshot.root_path,
        status=CollectionStatus.PARTIAL,
        workers=(
            WorkerArtifactCollection(
                worker_id="gpu4060",
                host="10.87.5.155",
                rank=0,
                stage="stage0",
                status=CollectionStatus.SUCCESS,
                checkpoint_state=ArtifactCollectionState.COMPLETE,
                artifacts=(
                    CollectedArtifact(
                        worker_id="gpu4060",
                        rank=0,
                        stage="stage0",
                        artifact_type="checkpoint_file",
                        relative_path="checkpoint/checkpoint_rank0.pt",
                        remote_path="/remote/checkpoint/checkpoint_rank0.pt",
                        local_path=str(local_rank0),
                        status=ArtifactCollectionState.COMPLETE,
                        size_bytes=size0,
                        checksum=checksum0,
                    ),
                ),
            ),
            WorkerArtifactCollection(
                worker_id="gpu1060",
                host="10.87.5.15",
                rank=1,
                stage="stage1",
                status=CollectionStatus.SUCCESS,
                checkpoint_state=ArtifactCollectionState.COMPLETE,
                artifacts=(
                    CollectedArtifact(
                        worker_id="gpu1060",
                        rank=1,
                        stage="stage1",
                        artifact_type="checkpoint_file",
                        relative_path="checkpoint/checkpoint_rank1.pt",
                        remote_path="/remote/checkpoint/checkpoint_rank1.pt",
                        local_path=str(local_rank1),
                        status=ArtifactCollectionState.COMPLETE,
                        size_bytes=size1,
                        checksum=checksum1,
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="missing artifact collection for rank 2"):
        manager._finalize_checkpoint_bundle(
            snapshot=snapshot,
            training_config=training_config,
            execution_plan=execution_plan,
            current=status,
            collection_result=collection_result,
        )


def test_collect_artifacts_prefers_scp_for_windows_workers_when_transport_auto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    manager, config_path = _build_manager(tmp_path, events)
    manager._artifact_collector = None
    training_config = load_training_config(config_path)
    job = create_training_job(
        job_id=as_job_id("job-101-collect-transport"),
        config_path=str(config_path),
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    snapshot = manager._artifact_store.create_snapshot(job)
    execution_plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("ssh"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, stage="stage0"),
            WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1, stage="stage1"),
        ],
    )
    status = JobStatus(
        job_id=job.job_id,
        state=JobState.CHECKPOINTING,
        phase="checkpoint",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        assignments=list(execution_plan.workers),
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        backend=as_backend_name("ssh"),
        final_metrics={"final_loss": 0.25},
    )
    _write_checkpoint_shard(
        Path(snapshot.checkpoint_path) / "checkpoint_rank0.pt",
        rank=0,
        stage="stage0",
    )
    _write_checkpoint_shard(
        Path(snapshot.checkpoint_path) / "checkpoint_rank1.pt",
        rank=1,
        stage="stage1",
    )
    (Path(snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").write_text(
        json.dumps({"checkpoint_ref": "checkpoint/checkpoint_rank0.pt"}),
        encoding="utf-8",
    )
    (Path(snapshot.diagnostics_path) / "monitor-gpu1060-rank1.json").write_text(
        json.dumps({"checkpoint_ref": "checkpoint/checkpoint_rank1.pt"}),
        encoding="utf-8",
    )

    seen: dict[str, str] = {}

    class PullTransport:
        name = ArtifactTransportName.SCP

        def transfer(self, items, *, remote, secrets=()):
            del remote, secrets
            results = []
            for item in items:
                source = Path(item.source)
                destination = Path(item.destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                results.append(
                    ArtifactTransferItemResult(
                        label=item.label,
                        transport="scp",
                        status=ArtifactTransferStatus.SUCCESS,
                        source=item.source,
                        destination=item.destination,
                        recorded_command="scp",
                        exit_code=0,
                    )
                )
            return ArtifactTransferResult(
                transport="scp",
                status=ArtifactTransferStatus.SUCCESS,
                items=results,
            )

    def fake_select_transport(config, *, which=None, runner=None):
        del which, runner
        seen["preferred"] = config.preferred.value
        return PullTransport()

    monkeypatch.setattr(
        "shardgrid.control.job_manager.select_artifact_transport",
        fake_select_transport,
    )

    class RecordingCollector(FakeCollector):
        def __init__(self, *, transport, ssh_factory=None, runtime_factory=None):
            del ssh_factory, runtime_factory
            seen["collector_transport"] = transport.name.value
            super().__init__(events, status=CollectionStatus.SUCCESS)

    monkeypatch.setattr(
        "shardgrid.control.job_manager.ArtifactCollector",
        RecordingCollector,
    )

    result = manager._collect_artifacts(
        snapshot,
        training_config,
        execution_plan,
        status,
    )

    assert seen["preferred"] == "scp"
    assert result.status is CollectionStatus.SUCCESS
