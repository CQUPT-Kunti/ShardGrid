"""T062 prerequisite regression: ordinary entrypoint production path.

These tests lock the production wiring that was missing before T062:

* ``shardgrid.runtime.generic_bootstrap`` must exist and be importable
  (the SSHLauncher launch command points at it).
* The launcher's generic runtime argv must resolve to a real executable
  module, not a string snapshot.
* ``JobManager.run_entrypoint(...)`` with ``dry_run=False`` must proceed
  past planning into the formal execution path (memory probe selection,
  launcher prepare/distribute/launch, monitor, checkpoint finalization).
* Captured runtime artifacts (graph, backend graph, initial state, inputs)
  are persisted so the worker-side runtime never rebuilds the user model.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from shardgrid.common.enums import (
    BackendStatus,
    FailureStage,
    Health,
    JobState,
    PhysicalOS,
    RuntimeOS,
)
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.control.job_manager import JobManager
from shardgrid.launchers.ssh import _GENERIC_RUNTIME_BOOTSTRAP_MODULE
from shardgrid.resources.models import WorkerResource
from shardgrid.workers.models import WorkerRuntime
from shardgrid.workers.probe import (
    ProbeFailure,
    WindowsHostInfo,
    WorkerProbeResult,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO / "tests" / "fixtures" / "ordinary_training_scripts"


def _load_state_from_plan(plan_root: Path):
    manifest_path = plan_root / "state-manifest.json"
    shards_root = plan_root / "state-shards"
    if manifest_path.is_file() and shards_root.is_dir():
        merged: dict[str, torch.Tensor] = {}
        for shard_path in sorted(shards_root.glob("*.pt")):
            merged.update(torch.load(shard_path, map_location="cpu", weights_only=False))
        return merged
    state_path = plan_root / "initial-state.pt"
    if state_path.is_file():
        return torch.load(state_path, map_location="cpu", weights_only=False)
    return torch.load(
        plan_root / "backend-graph.pt", map_location="cpu", weights_only=False
    ).state_dict()


def test_generic_bootstrap_module_is_importable() -> None:
    module = importlib.import_module("shardgrid.runtime.generic_bootstrap")
    assert callable(getattr(module, "main", None))


def test_launcher_bootstrap_module_reference_is_real() -> None:
    assert _GENERIC_RUNTIME_BOOTSTRAP_MODULE == "shardgrid.runtime.generic_bootstrap"
    assert importlib.util.find_spec("shardgrid.runtime.generic_bootstrap") is not None


def _worker_resource(worker_id: str, host: str) -> WorkerResource:
    from datetime import UTC, datetime

    return WorkerResource(
        worker_id=worker_id,
        hostname=host,
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment="shardgrid",
        conda_prefix="/opt/conda/envs/shardgrid",
        python_executable="/opt/conda/envs/shardgrid/bin/python",
        ip=host,
        gpu_name="RTX 4060",
        gpu_total_memory=8192,
        gpu_free_memory=7168,
        compute_capability="8.9",
        driver_version="566.07",
        cuda_version="11.8",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        nccl_available=True,
        gloo_available=True,
        health=Health.HEALTHY,
        last_probe_at=datetime.now(tz=UTC).isoformat(),
    )


def _probe_result(resource: WorkerResource) -> WorkerProbeResult:
    return WorkerProbeResult(
        worker_resource=resource,
        worker_runtime=WorkerRuntime(
            worker_id=resource.worker_id,
            runtime_os=RuntimeOS.WSL2_LINUX,
            runtime_version="Ubuntu-22.04",
            conda_environment="shardgrid",
            conda_prefix="/opt/conda/envs/shardgrid",
            health=Health.HEALTHY,
        ),
        windows_host=WindowsHostInfo(
            os_version="Windows",
            openssh_available=True,
            wsl_available=True,
            nvidia_driver_visible=True,
            driver_name="566.07",
        ),
        failures=(),
        health=Health.HEALTHY,
        probe_status="live",
    )


def _network_state(worker_ids: tuple[str, ...]):
    from datetime import UTC, datetime

    from shardgrid.resources.models import NetworkLink, NetworkState

    now = datetime.now(tz=UTC).isoformat()
    links = []
    for source in worker_ids:
        for target in worker_ids:
            if source == target:
                continue
            links.append(
                NetworkLink(
                    source_worker_id=source,
                    target_worker_id=target,
                    source_ip="10.0.0.1",
                    target_ip="10.0.0.2",
                    interface="eth0",
                    tcp_reachable=True,
                    bandwidth_mbps=900.0,
                    latency_ms=1.5,
                    measured_at=now,
                )
            )
    return NetworkState(
        network_id="net-1",
        workers=list(worker_ids),
        links=links,
        created_at=now,
        selected_interfaces={worker_id: "eth0" for worker_id in worker_ids},
    )


class _FlexibleSelectedEngine:
    def __init__(self, plan) -> None:
        from shardgrid.engines.models import EnginePreparation

        self.engine_id = "pytorch_pipeline"
        self.candidate = type(
            "Candidate",
            (),
            {
                "engine_id": "pytorch_pipeline",
                "status": BackendStatus.AVAILABLE,
            },
        )()
        self.parallel_plan = plan
        self.original_plan_path = plan.engine_plan_path
        self.rejected_engine_ids = ()
        self._preparation = EnginePreparation(
            engine_id="pytorch_pipeline",
            status=BackendStatus.AVAILABLE,
        )

        preparation = self._preparation

        class _FakeEngine:
            def launch_metadata(self, plan):
                return {"engine": "pytorch_pipeline", "plan": plan.parallel_plan_id}

            def prepare(self, snapshot, execution_plan):
                del snapshot, execution_plan
                return preparation

        self.engine = _FakeEngine()


class _FlexibleLauncher:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def prepare(self, context):
        self.events.append("launcher_prepare")
        from shardgrid.launchers.base import (
            LauncherOperation,
            LauncherResult,
            LauncherResultStatus,
        )

        return LauncherResult(
            operation=LauncherOperation.PREPARE,
            status=LauncherResultStatus.SUCCESS,
            backend=as_backend_name("ssh"),
            job_id=str(context.job.job_id),
            next_job_state=JobState.DISTRIBUTING,
        )

    def distribute(self, context):
        self.events.append("launcher_distribute")
        from shardgrid.launchers.base import (
            LauncherOperation,
            LauncherResult,
            LauncherResultStatus,
        )

        return LauncherResult(
            operation=LauncherOperation.DISTRIBUTE,
            status=LauncherResultStatus.SUCCESS,
            backend=as_backend_name("ssh"),
            job_id=str(context.job.job_id),
            next_job_state=JobState.DISTRIBUTING,
        )

    def launch(self, context):
        self.events.append("launcher_launch")
        from shardgrid.launchers.base import (
            LauncherOperation,
            LauncherResult,
            LauncherResultStatus,
        )

        return LauncherResult(
            operation=LauncherOperation.LAUNCH,
            status=LauncherResultStatus.SUCCESS,
            backend=as_backend_name("ssh"),
            job_id=str(context.job.job_id),
            next_job_state=JobState.LAUNCHING,
        )

    def monitor(self, context):
        self.events.append("launcher_monitor")
        from shardgrid.jobs.models import JobStatus

        status = JobStatus(
            job_id=context.job.job_id,
            state=JobState.COMPLETED,
            phase="checkpoint",
            backend=as_backend_name("nccl"),
            final_metrics={"final_loss": 0.25},
            checkpoint_ref="checkpoint/model.pt",
        )
        path = Path(context.snapshot.root_path) / "job-status.json"
        path.write_text(
            json.dumps(status.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        from shardgrid.launchers.base import (
            LauncherOperation,
            LauncherResult,
            LauncherResultStatus,
        )

        return LauncherResult(
            operation=LauncherOperation.MONITOR,
            status=LauncherResultStatus.SUCCESS,
            backend=as_backend_name("ssh"),
            job_id=str(context.job.job_id),
            next_job_state=status.state,
        )

    def stop(self, context):
        self.events.append("launcher_stop")
        from shardgrid.launchers.base import (
            LauncherOperation,
            LauncherResult,
            LauncherResultStatus,
        )

        return LauncherResult(
            operation=LauncherOperation.STOP,
            status=LauncherResultStatus.NOOP,
            backend=as_backend_name("ssh"),
            job_id=str(context.job.job_id),
            next_job_state=context.job_status.state,
        )

    def cleanup(self, context):
        self.events.append("launcher_cleanup")
        return None


def JobStatus_completed(context):
    from shardgrid.jobs.models import JobStatus

    return JobStatus(
        job_id=context.job.job_id,
        state=JobState.COMPLETED,
        phase="checkpoint",
        backend=as_backend_name("nccl"),
        final_metrics={"final_loss": 0.25},
        checkpoint_ref="checkpoint/model.pt",
    )


class _FlexibleCollector:
    def collect(self, snapshot, *, sources, secrets=(), artifact_paths=None):
        from shardgrid.artifacts.collector import (
            ArtifactCollectionResult,
            ArtifactCollectionState,
            CollectionStatus,
            CollectedArtifact,
            WorkerArtifactCollection,
        )
        from shardgrid.runtime.checkpoint import CHECKPOINT_SCHEMA_VERSION
        from shardgrid.runtime.dag import RuntimePlan
        from shardgrid.planner.generic_graph import GenericGraphIR

        del secrets, artifact_paths
        plan_root = Path(snapshot.plan_path)
        graph = GenericGraphIR.from_dict(
            json.loads((plan_root / "captured-graph.json").read_text())
        )
        runtime_plan = RuntimePlan.from_dict(
            json.loads((plan_root / "runtime-plan.json").read_text())
        )
        workers = []
        for source in sources:
            shard_path = Path(snapshot.checkpoint_path) / f"model_rank{source.rank}.pt"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            state = _load_state_from_plan(plan_root)
            from shardgrid.runtime.checkpoint import save_worker_state_shard

            result = save_worker_state_shard(
                shard_path,
                graph=graph,
                runtime_plan=runtime_plan,
                worker_id=str(source.worker_id),
                gpu_index=0,
                state_dict=state,
                job_id=str(snapshot.job_id),
                plan_id="captured-plan",
                training_step=1,
                metadata={"rank": source.rank, "step": 1},
            )
            metadata_path = (
                Path(snapshot.checkpoint_path) / f"metadata_rank{source.rank}.json"
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "rank": source.rank,
                        "step": 1,
                        "checkpoint_version": 1,
                        "status": "complete",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            workers.append(
                WorkerArtifactCollection(
                    worker_id=source.worker_id,
                    host=source.host,
                    rank=source.rank,
                    stage=source.stage,
                    status=CollectionStatus.SUCCESS,
                    checkpoint_state=ArtifactCollectionState.COMPLETE,
                    artifacts=(
                        CollectedArtifact(
                            worker_id=source.worker_id,
                            rank=source.rank,
                            stage=source.stage,
                            artifact_type="checkpoint_metadata",
                            relative_path=f"checkpoint/metadata_rank{source.rank}.json",
                            remote_path=f"checkpoint/metadata_rank{source.rank}.json",
                            local_path=str(metadata_path),
                            status=ArtifactCollectionState.COMPLETE,
                            size_bytes=metadata_path.stat().st_size,
                            checksum="meta",
                        ),
                        CollectedArtifact(
                            worker_id=source.worker_id,
                            rank=source.rank,
                            stage=source.stage,
                            artifact_type="checkpoint_file",
                            relative_path="checkpoint/model.pt",
                            remote_path="checkpoint/model.pt",
                            local_path=str(shard_path),
                            status=ArtifactCollectionState.COMPLETE,
                            size_bytes=result.bytes,
                            checksum="shard",
                        ),
                    ),
                )
            )
        return ArtifactCollectionResult(
            job_id=Path(snapshot.root_path).name,
            snapshot_root=snapshot.root_path,
            status=CollectionStatus.SUCCESS,
            workers=tuple(workers),
        )


def _config(tmp_path: Path):
    from shardgrid.common.config import ClusterConfig

    return ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": str((tmp_path / "jobs").resolve()),
            "ssh": {},
            "runtime": {
                "conda_environment": "shardgrid",
                "conda_prefix": "/opt/conda/envs/shardgrid",
            },
            "network": {"rendezvous_port": 29500},
            "backend_preference": {},
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
                    "conda_prefix": "/opt/conda/envs/shardgrid",
                }
            ],
        }
    )


def test_run_entrypoint_non_dry_run_reaches_formal_execution(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path)
    resources = {
        "gpu4060": _worker_resource("gpu4060", "10.87.5.155"),
    }

    def probe_worker(worker):
        events.append(f"probe:{worker.worker_id}")
        return _probe_result(resources[str(worker.worker_id)])

    def probe_network(worker_resources):
        events.append("network")
        return _network_state(tuple(str(r.worker_id) for r in worker_resources))

    def select_engine(engine_id, job, resources_state, network, *, registry=None):
        del engine_id, job, resources_state, network, registry
        events.append("select_engine")
        return _FlexibleSelectedEngine(
            type(
                "Plan",
                (),
                {
                    "parallel_plan_id": "captured-plan",
                    "engine": as_backend_name("pytorch_pipeline"),
                    "engine_plan_path": "plan/original-parallel-plan.json",
                    "model_name": "captured",
                    "world_size": 1,
                    "stages": ["stage0"],
                    "partition_source": "automatic",
                    "selected_candidate_id": "cand0",
                    "stage_metadata": [],
                    "requirements": {"workload_source": "captured_context"},
                },
            )()
        )

    manager = JobManager(
        config,
        probe_worker=probe_worker,
        probe_network=probe_network,
        select_engine=select_engine,
        launcher_factory=lambda backend: _FlexibleLauncher(events),
        artifact_collector=_FlexibleCollector(),
        source_root=REPO,
    )
    # captured planning path must be used, not legacy model.type planning
    def fail_legacy(_training_config):
        raise AssertionError("run_entrypoint fell back to model.type workload")

    manager._planner_workload = fail_legacy

    def flexible_memory_probe(
        *,
        training_config,
        probe_job,
        probe_snapshot,
        probe_plan,
        probe_execution,
        cluster_state,
        current,
        attempt=1,
        free_memory_baseline=None,
    ):
        del (
            training_config,
            probe_job,
            probe_snapshot,
            probe_execution,
            cluster_state,
            current,
            attempt,
            free_memory_baseline,
        )
        return type(
            "ProbeResult",
            (),
            {
                "status": "PASS",
                "subtype": None,
                "message": "memory probe passed",
                "candidate_id": probe_plan.selected_candidate_id,
                "rendezvous_ready": True,
                "phase": "PROBE_COMPLETE",
                "attempt": 1,
                "master_addr": "10.0.0.1",
                "master_port": 29500,
                "oom_evidence": False,
                "actual_peak_allocated_bytes": 1024,
                "actual_peak_reserved_bytes": 2048,
            },
        )()

    manager._run_memory_probe = flexible_memory_probe

    def flexible_live_preflight(
        *,
        training_config,
        execution_plan,
        cluster_state,
        network_state,
    ):
        from shardgrid.planner.models import MasterMetadata

        del training_config
        updated = replace(
            execution_plan,
            master=MasterMetadata(address="10.0.0.1", port=29500),
        )
        return updated, cluster_state, network_state

    manager._prepare_live_execution_plan = flexible_live_preflight

    result = manager.run_entrypoint(
        SimpleNamespace(
            entrypoint=FIXTURE_ROOT / "positional_tuple_train.py",
            argv=("--epochs", "1", "--checkpoint", "out/tuple.pt"),
            cwd=FIXTURE_ROOT,
            environment={"SHARDGRID_TEST_CAPTURE": "1"},
            cluster_config_path=str(tmp_path / "workers.yaml"),
            dry_run=False,
        ),
        job_id=as_job_id("job-captured-formal"),
    )

    assert result.status.state is JobState.COMPLETED
    assert result.snapshot is not None
    assert "launcher_prepare" in events
    assert "launcher_distribute" in events
    assert "launcher_launch" in events
    assert "launcher_monitor" in events
    assert (Path(result.snapshot.plan_path) / "captured-graph.json").is_file()
    assert (Path(result.snapshot.plan_path) / "backend-graph.pt").is_file()
    assert (Path(result.snapshot.plan_path) / "backend-graph.pt").is_file()
    assert (Path(result.snapshot.plan_path) / "captured-context.json").is_file()


def test_captured_runtime_artifacts_are_persisted_for_bootstrap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    resources = {"gpu4060": _worker_resource("gpu4060", "10.87.5.155")}

    def probe_worker(worker):
        return _probe_result(resources[str(worker.worker_id)])

    def probe_network(worker_resources):
        return _network_state(tuple(str(r.worker_id) for r in worker_resources))

    def select_engine(engine_id, job, resources_state, network, *, registry=None):
        del engine_id, job, resources_state, network, registry
        return _FlexibleSelectedEngine(
            type(
                "Plan",
                (),
                {
                    "parallel_plan_id": "captured-plan",
                    "engine": as_backend_name("pytorch_pipeline"),
                    "engine_plan_path": "plan/original-parallel-plan.json",
                    "model_name": "captured",
                    "world_size": 1,
                    "stages": ["stage0"],
                    "partition_source": "automatic",
                    "selected_candidate_id": "cand0",
                    "stage_metadata": [],
                    "requirements": {"workload_source": "captured_context"},
                },
            )()
        )

    manager = JobManager(
        config,
        probe_worker=probe_worker,
        probe_network=probe_network,
        select_engine=select_engine,
        source_root=REPO,
    )
    capture = SimpleNamespace(
        model=None,
        sample_args=(),
        sample_kwargs={},
        context=SimpleNamespace(
            model_identity={"class_name": "FixtureModel"},
            tensor_metadata={},
        ),
    )
    from shardgrid.bootstrap.runner import capture_entrypoint_workload

    workload = capture_entrypoint_workload(
        FIXTURE_ROOT / "positional_tuple_train.py",
        argv=("--epochs", "1", "--checkpoint", "out/tuple.pt"),
        cwd=FIXTURE_ROOT,
        environment={"SHARDGRID_TEST_CAPTURE": "1"},
        dry_run=True,
    )
    capture = workload
    from shardgrid.jobs.models import JobSnapshot

    snapshot = JobSnapshot(
        job_id=as_job_id("job-artifacts"),
        root_path=str(tmp_path / "job-artifacts"),
        code_path=str(tmp_path / "job-artifacts" / "code"),
        config_path=str(tmp_path / "job-artifacts" / "config"),
        plan_path=str(tmp_path / "job-artifacts" / "plan"),
        logs_path=str(tmp_path / "job-artifacts" / "logs"),
        environment_path=str(tmp_path / "job-artifacts" / "environment"),
        checkpoint_path=str(tmp_path / "job-artifacts" / "checkpoint"),
        diagnostics_path=str(tmp_path / "job-artifacts" / "diagnostics"),
    )
    plan = type(
        "Plan",
        (),
        {
            "stage_metadata": (),
            "graph_fingerprint": None,
        },
    )()
    manager._persist_captured_runtime_artifacts(
        snapshot=snapshot,
        capture=capture,
        parallel_plan=plan,
    )
    plan_root = Path(snapshot.plan_path)
    assert (plan_root / "captured-context.json").is_file()
    assert (plan_root / "captured-graph.json").is_file()
    graph = json.loads((plan_root / "captured-graph.json").read_text())
    assert graph["nodes"]
    backend = torch.load(plan_root / "backend-graph.pt", weights_only=False)
    assert backend is not None
    assert backend.state_dict()


def _worker_state_shard_root(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot-root"
    plan_root = root / "plan"
    shards_root = plan_root / "state-shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"layers.0.weight": torch.zeros(4, 4), "layers.0.bias": torch.zeros(4)},
        shards_root / "stage0.pt",
    )
    torch.save(
        {"layers.1.weight": torch.zeros(4, 4), "layers.1.bias": torch.zeros(4)},
        shards_root / "stage1.pt",
    )
    (plan_root / "state-manifest.json").write_text(
        json.dumps(
            [
                {
                    "state_id": "layers.0.weight",
                    "kind": "parameter",
                    "owner_stage": "stage0",
                    "read_only_stages": [],
                    "shard_ref": "state-shards/stage0.pt",
                },
                {
                    "state_id": "layers.0.bias",
                    "kind": "parameter",
                    "owner_stage": "stage0",
                    "read_only_stages": [],
                    "shard_ref": "state-shards/stage0.pt",
                },
                {
                    "state_id": "layers.1.weight",
                    "kind": "parameter",
                    "owner_stage": "stage1",
                    "read_only_stages": [],
                    "shard_ref": "state-shards/stage1.pt",
                },
                {
                    "state_id": "layers.1.bias",
                    "kind": "parameter",
                    "owner_stage": "stage1",
                    "read_only_stages": [],
                    "shard_ref": "state-shards/stage1.pt",
                },
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T083 expected-red: _load_initial_state merges all worker shards into "
        "CPU before ownership selection; T084 restricts loading to owned and "
        "explicit read-only state ids only."
    ),
)
def test_worker_loads_only_owned_and_read_only_state_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shardgrid.runtime.generic_bootstrap as bootstrap

    root = _worker_state_shard_root(tmp_path)
    loaded_shards: list[str] = []
    real_load = torch.load

    def spy_load(path, *args, **kwargs):
        loaded_shards.append(str(Path(path).name))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(bootstrap.torch, "load", spy_load)
    monkeypatch.setenv("SHARDGRID_REMOTE_SNAPSHOT_ROOT", str(root))

    state = bootstrap._load_initial_state(root / "plan")

    assert set(state) == {"layers.0.weight", "layers.0.bias"}
    assert loaded_shards == ["stage0.pt"], (
        f"worker loaded non-owned shards: {loaded_shards}; "
        "ownership must gate state payload load"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T083 expected-red: ownership is resolved after full state payload load; "
        "T084 resolves ownership before touching any state shard."
    ),
)
def test_ownership_resolved_before_state_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shardgrid.runtime.generic_bootstrap as bootstrap

    root = _worker_state_shard_root(tmp_path)
    shards_root = root / "plan" / "state-shards"
    (root / "plan" / "runtime-plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "graph_fingerprint": "fp",
                "logical": {
                    "partitions": [
                        {
                            "partition_id": "stage0",
                            "node_ids": ["n0"],
                            "read_only_state_ids": [],
                        },
                        {
                            "partition_id": "stage1",
                            "node_ids": ["n1"],
                            "read_only_state_ids": [],
                        },
                    ]
                },
                "placement": {"workers": []},
                "ownership": {
                    "workers": [
                        {
                            "worker_id": "worker-a",
                            "owned_partitions": ["stage0"],
                            "read_only_state_ids": [],
                        }
                    ]
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    order: list[str] = []
    real_load = torch.load

    def spy_load(path, *args, **kwargs):
        order.append(str(Path(path).name))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(bootstrap.torch, "load", spy_load)

    bootstrap._load_initial_state(root / "plan")

    assert order == ["stage0.pt"], (
        f"state shards loaded before ownership was resolved: {order}"
    )