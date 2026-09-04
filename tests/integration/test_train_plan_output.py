from __future__ import annotations

import json
from pathlib import Path

import yaml

from shardgrid.cli.app import main
from shardgrid.cli.commands import train as train_command
from shardgrid.common.config import ClusterConfig
from shardgrid.common.enums import BackendStatus, Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_engine_name, as_hostname, as_job_id
from shardgrid.control import job_manager as job_manager_module
from shardgrid.control.job_manager import JobManager
from shardgrid.engines.models import (
    ParallelEngineCandidate,
    ParallelPlan,
    ParallelPlanCommunicationEdge,
    ParallelPlanPlacement,
    ParallelPlanProvenance,
    ParallelPlanStage,
    TrainingMemoryEstimate,
)
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource
from shardgrid.workers.models import WorkerRuntime
from shardgrid.workers.probe import WindowsHostInfo, WorkerProbeResult


def _cluster_config(tmp_path: Path) -> tuple[ClusterConfig, Path, Path]:
    cluster_path = tmp_path / "workers.yaml"
    training_path = tmp_path / "train-automatic.yaml"
    cluster_path.write_text(
        f"""
control:
  machine_id: machine-a
  hostname: control-a.local
jobs_root: {(tmp_path / "jobs").resolve()}
ssh: {{}}
runtime:
  python_executable: python3
  conda_environment: shardgrid
  conda_prefix: /opt/conda/envs/shardgrid
network:
  rendezvous_port: 29500
backend_preference:
  launcher: ssh
  communication_backend: nccl
  parallel_engine: galvatron
manual_override: {{}}
workers:
  - id: gpu4060
    machine_id: machine-c
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.0.0.10
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
  - id: gpu1060
    machine_id: machine-d
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.0.0.11
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
  - id: gpu3090
    machine_id: machine-e
    physical_os: linux
    runtime_os: linux
    runtime: linux
    host: 10.0.0.12
    ssh_user: shardgrid
    runtime_distro: Ubuntu-24.04
    conda_environment: shardgrid
    conda_prefix: /opt/conda/envs/shardgrid
""".strip()
        + "\n",
        encoding="utf-8",
    )
    training_path.write_text(
        """
job:
  name: train-automatic
  backend: ssh
  communication_backend: nccl
model:
  name: tiny-transformer
  type: hf_style
  stage_count: 3
resources:
  world_size: 3
  preferred_workers: [gpu4060, gpu1060, gpu3090]
artifacts:
  snapshot_name: train-automatic
  keep_failed_snapshots: true
  transport: auto
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return (
        ClusterConfig.from_dict(yaml.safe_load(cluster_path.read_text())),
        cluster_path,
        training_path,
    )


def _worker_resource(
    *,
    worker_id: str,
    host: str,
    physical_os: PhysicalOS,
    runtime_os: RuntimeOS,
    gpu_name: str,
    gpu_free_memory: int,
) -> WorkerResource:
    return WorkerResource(
        worker_id=worker_id,
        hostname=as_hostname(host),
        physical_os=physical_os,
        runtime_os=runtime_os,
        conda_environment="shardgrid",
        conda_prefix="/opt/conda/envs/shardgrid",
        python_executable="/opt/conda/envs/shardgrid/bin/python",
        ip=host,
        gpu_name=gpu_name,
        gpu_total_memory=24576,
        gpu_free_memory=gpu_free_memory,
        compute_capability="8.6",
        cuda_version="12.1",
        torch_version="2.7.1+cu121",
        torch_cuda_version="12.1",
        nccl_available=True,
        gloo_available=True,
        network_interface="eth0",
        health=Health.HEALTHY,
        last_probe_at="2026-09-03T10:00:00+00:00",
    )


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
            torch_version="2.7.1+cu121",
            torch_cuda_version="12.1",
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


def _network_state() -> NetworkState:
    workers = ["gpu4060", "gpu1060", "gpu3090"]
    links = []
    for source, source_ip in (
        ("gpu4060", "10.0.0.10"),
        ("gpu1060", "10.0.0.11"),
        ("gpu3090", "10.0.0.12"),
    ):
        for target, target_ip in (
            ("gpu4060", "10.0.0.10"),
            ("gpu1060", "10.0.0.11"),
            ("gpu3090", "10.0.0.12"),
        ):
            if source == target:
                continue
            links.append(
                NetworkLink(
                    source_worker_id=source,
                    target_worker_id=target,
                    source_ip=source_ip,
                    target_ip=target_ip,
                    interface="eth0",
                    tcp_reachable=True,
                    bandwidth_mbps=940.0,
                    latency_ms=0.8,
                    port=29500,
                    measured_at="2026-09-03T10:00:00+00:00",
                )
            )
    return NetworkState(
        network_id="lan-a",
        workers=workers,
        links=links,
        created_at="2026-09-03T10:00:00+00:00",
        selected_interfaces={worker: "eth0" for worker in workers},
    )


class _DryRunEngine:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def prepare(self, job_snapshot: object, execution_plan: object):
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
            limitations=["experimental automatic partition path"],
        )
        self.parallel_plan = ParallelPlan(
            parallel_plan_id="auto-plan-3w",
            engine=as_engine_name("galvatron"),
            engine_plan_path="/var/tmp/engine/original-plan.json",
            model_name="tiny-transformer",
            world_size=3,
            stages=["stage0", "stage1", "stage2"],
            partition_source="automatic",
            model_profile_id="profile-transformer-3stage",
            selected_candidate_id="candidate-3w-a",
            candidate_evaluation_ref="/var/tmp/planner/candidate-eval.json",
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
                        estimated_peak_bytes=1_200,
                        safety_headroom_bytes=300,
                    ),
                    placement=ParallelPlanPlacement(
                        worker_id="gpu4060",
                        rank=0,
                        machine_id="machine-c",
                        gpu_index=0,
                        usable_memory_before_bytes=8_000,
                        remaining_memory_bytes=6_500,
                        utilization_ratio=0.19,
                    ),
                ),
                ParallelPlanStage(
                    stage_id="stage1",
                    rank=1,
                    module_ids=("block0", "block1"),
                    module_paths=("model.block0", "model.block1"),
                    start_index=1,
                    stop_index=3,
                    parameter_names_or_ranges=("block0.*", "block1.*"),
                    estimated_peak_training_memory=TrainingMemoryEstimate(
                        estimated_peak_bytes=2_400,
                        safety_headroom_bytes=400,
                    ),
                    placement=ParallelPlanPlacement(
                        worker_id="gpu1060",
                        rank=1,
                        machine_id="machine-d",
                        gpu_index=0,
                        usable_memory_before_bytes=7_000,
                        remaining_memory_bytes=4_200,
                        utilization_ratio=0.40,
                    ),
                ),
                ParallelPlanStage(
                    stage_id="stage2",
                    rank=2,
                    module_ids=("head",),
                    module_paths=("model.head",),
                    start_index=3,
                    stop_index=4,
                    parameter_names_or_ranges=("head.*",),
                    estimated_peak_training_memory=TrainingMemoryEstimate(
                        estimated_peak_bytes=1_600,
                        safety_headroom_bytes=250,
                    ),
                    placement=ParallelPlanPlacement(
                        worker_id="gpu3090",
                        rank=2,
                        machine_id="machine-e",
                        gpu_index=0,
                        usable_memory_before_bytes=20_000,
                        remaining_memory_bytes=18_150,
                        utilization_ratio=0.09,
                    ),
                ),
            ],
            communication_edges=[
                ParallelPlanCommunicationEdge(
                    source_stage_id="stage0",
                    target_stage_id="stage1",
                    source_module_id="embed",
                    target_module_id="block0",
                    estimated_bytes_per_step=4096,
                ),
                ParallelPlanCommunicationEdge(
                    source_stage_id="stage1",
                    target_stage_id="stage2",
                    source_module_id="block1",
                    target_module_id="head",
                    estimated_bytes_per_step=2048,
                ),
            ],
            planning_provenance=ParallelPlanProvenance(
                partition_source="automatic",
                model_profile_id="profile-transformer-3stage",
                selected_candidate_id="candidate-3w-a",
                selected_worker_count=3,
                attempted_worker_counts=(2, 3, 4),
                total_cross_worker_communication_bytes=6144,
                selected_reason=(
                    "selected first feasible 3-worker candidate after 2-worker memory rejection"
                ),
                fallback_reason="2-worker candidate exceeded usable memory headroom on gpu1060",
                rejection_reasons=(
                    "worker_count=2 rejected: stage1 planner_required_bytes > usable_memory",
                ),
            ),
            requirements={"selected_worker_count": "3", "required_runtime": "wsl2_or_linux"},
        )
        self.rejected_engine_ids = ("pytorch_pipeline (plan rejected: unsupported model)",)


def _build_manager(tmp_path: Path, events: list[str]) -> tuple[JobManager, Path, Path]:
    config, cluster_path, training_path = _cluster_config(tmp_path)
    resources = {
        "gpu4060": _worker_resource(
            worker_id="gpu4060",
            host="10.0.0.10",
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            gpu_name="RTX 4060",
            gpu_free_memory=8_000,
        ),
        "gpu1060": _worker_resource(
            worker_id="gpu1060",
            host="10.0.0.11",
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            gpu_name="GTX 1060",
            gpu_free_memory=7_000,
        ),
        "gpu3090": _worker_resource(
            worker_id="gpu3090",
            host="10.0.0.12",
            physical_os=PhysicalOS.LINUX,
            runtime_os=RuntimeOS.LINUX,
            gpu_name="RTX 3090",
            gpu_free_memory=20_000,
        ),
    }

    def probe_worker(worker):
        events.append(f"probe:{worker.worker_id}")
        return _probe_result(resources[str(worker.worker_id)])

    def probe_network(worker_resources):
        events.append("network")
        links = []
        for source in worker_resources:
            for target in worker_resources:
                if source.worker_id == target.worker_id:
                    continue
                links.append(
                    NetworkLink(
                        source_worker_id=source.worker_id,
                        target_worker_id=target.worker_id,
                        source_ip=source.ip or "10.0.0.10",
                        target_ip=target.ip or "10.0.0.11",
                        interface="eth0",
                        tcp_reachable=True,
                        bandwidth_mbps=940.0,
                        latency_ms=0.8,
                        port=29500,
                        measured_at="2026-09-03T10:00:00+00:00",
                    )
                )
        return NetworkState(
            network_id="lan-auto",
            workers=[item.worker_id for item in worker_resources],
            links=links,
            created_at="2026-09-03T10:00:00+00:00",
            selected_interfaces={item.worker_id: "eth0" for item in worker_resources},
        )

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
    return manager, cluster_path, training_path


def test_dry_run_writes_complete_audit_output_and_artifacts(tmp_path: Path) -> None:
    events: list[str] = []
    manager, _, training_path = _build_manager(tmp_path, events)

    result = manager.run(training_path, job_id=as_job_id("job-t115"), dry_run=True)
    payload = train_command._payload(result)
    human = train_command._render_human(result)
    snapshot_root = Path(result.snapshot.root_path)
    execution_json = json.loads((snapshot_root / "plan" / "execution-plan.json").read_text())
    execution_yaml = yaml.safe_load((snapshot_root / "plan" / "execution-plan.yaml").read_text())
    metadata_json = json.loads(
        (snapshot_root / "diagnostics" / "snapshot-metadata.json").read_text()
    )
    metadata_yaml = yaml.safe_load(
        (snapshot_root / "diagnostics" / "snapshot-metadata.yaml").read_text()
    )
    original_plan_json = json.loads(
        (snapshot_root / "plan" / "original-parallel-plan.json").read_text()
    )
    original_plan_yaml = yaml.safe_load(
        (snapshot_root / "plan" / "original-parallel-plan.yaml").read_text()
    )

    assert result.status.state is JobState.SNAPSHOTTING
    assert result.status.phase == "plan"
    assert events == [
        "probe:gpu4060",
        "probe:gpu1060",
        "probe:gpu3090",
        "network",
        "select_engine",
        "engine_launch_metadata",
    ]

    assert payload["dry_run"] is True
    assert payload["plan_mode"] == "automatic"
    assert payload["engine"] == "galvatron"
    assert payload["backend"] == "nccl"
    assert payload["world_size"] == 3
    assert payload["master"] == {"address": "10.0.0.10", "port": 29500}
    assert payload["fallback"] == {
        "status": "USED",
        "label": "engine_and_planner_fallback",
        "reason": (
            "rejected engines: pytorch_pipeline (plan rejected: unsupported model) | "
            "2-worker candidate exceeded usable memory headroom on gpu1060"
        ),
    }
    assert (
        payload["original_plan"]["original_engine_plan_ref"]
        == "/var/tmp/engine/original-plan.json"
    )
    assert (
        payload["original_plan"]["candidate_evaluation_ref"]
        == "/var/tmp/planner/candidate-eval.json"
    )
    assert [item["stage"] for item in payload["placement"]] == ["stage0", "stage1", "stage2"]
    assert [item["worker_id"] for item in payload["assignments"]] == [
        "gpu4060",
        "gpu1060",
        "gpu3090",
    ]
    assert payload["assignments"][1]["rank"] == 1
    assert payload["assignments"][1]["local_rank"] == 0
    assert payload["assignments"][1]["stage"] == "stage1"
    assert payload["assignments"][1]["device"] == "cuda:0"
    assert payload["assignments"][1]["runtime_os"] == "wsl2_linux"
    assert payload["assignments"][1]["estimated_peak_training_memory"] == 2800
    assert payload["planning"]["selected_worker_count"] == "3"
    assert payload["planning"]["total_cross_worker_communication_bytes"] == "6144"
    assert payload["planning"]["rejection_reasons"] == [
        "worker_count=2 rejected: stage1 planner_required_bytes > usable_memory"
    ]

    assert "Plan Mode: automatic" in human
    assert "Engine: galvatron" in human
    assert "Backend: nccl" in human
    assert "Master: 10.0.0.10:29500" in human
    assert "World Size: 3" in human
    assert "Partition Source: automatic" in human
    assert "Selected Candidate: candidate-3w-a" in human
    assert "Stage stage0 -> Worker gpu4060 -> rank 0 -> GPU 0" in human
    assert "Stage stage2 -> Worker gpu3090 -> rank 2 -> GPU 0" in human
    assert "Parallel Plan Ref:" in human
    assert "Fallback: engine_and_planner_fallback (USED)" in human

    assert execution_json == execution_yaml
    assert original_plan_json == original_plan_yaml
    assert metadata_json == metadata_yaml
    assert execution_json["workers"][0]["host"] == "10.0.0.10"
    assert execution_json["workers"][1]["machine_id"] == "machine-d"
    assert execution_json["workers"][2]["runtime_os"] == "linux"
    assert execution_json["workers"][1]["stage"] == "stage1"
    assert execution_json["workers"][1]["communication_edges"] == [
        "stage0->stage1:4096",
        "stage1->stage2:2048",
    ]
    assert execution_json["parallel_plan_ref"].endswith("original-parallel-plan.json")
    assert execution_json["original_engine_plan_ref"] == "/var/tmp/engine/original-plan.json"
    assert metadata_json["execution_plan_audit"]["assignments"] == payload["assignments"]
    assert metadata_json["execution_plan_audit"]["placement"] == payload["placement"]
    assert metadata_json["execution_plan_audit"]["master"] == payload["master"]
    assert metadata_json["execution_plan_audit"]["engine"] == payload["engine"]
    assert metadata_json["execution_plan_audit"]["backend"] == payload["backend"]
    assert metadata_json["execution_plan_audit"]["world_size"] == payload["world_size"]


def test_train_cli_passes_dry_run_and_returns_zero(tmp_path: Path, monkeypatch, capsys) -> None:
    events: list[str] = []
    manager, cluster_path, training_path = _build_manager(tmp_path, events)
    result = manager.run(training_path, job_id=as_job_id("job-t115-cli"), dry_run=True)
    captured: dict[str, object] = {}

    class FakeManager:
        def __init__(self, config) -> None:
            captured["jobs_root"] = str(config.jobs_root)

        def run(self, config_path: str, *, dry_run: bool = False):
            captured["config_path"] = config_path
            captured["dry_run"] = dry_run
            return result

    monkeypatch.setattr(train_command, "JobManager", FakeManager)

    exit_code = main(
        ["--config", str(cluster_path), "train", str(training_path), "--dry-run", "--json"]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["config_path"] == str(training_path)
    assert captured["dry_run"] is True
    assert output["dry_run"] is True
    assert output["plan_mode"] == "automatic"
    assert output["engine"] == "galvatron"
    assert output["backend"] == "nccl"
    assert output["master"] == {"address": "10.0.0.10", "port": 29500}
    assert len(output["assignments"]) == 3


def test_dry_run_automatic_mode_invokes_real_planner_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cluster_config = ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
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
            "workers": [
                {
                    "id": "gpu4060",
                    "machine_id": "machine-c",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.10",
                    "ssh_user": "shardgrid",
                },
                {
                    "id": "gpu1060",
                    "machine_id": "machine-d",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.11",
                    "ssh_user": "shardgrid",
                },
                {
                    "id": "gpu4060-cqupt",
                    "machine_id": "machine-e",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.12",
                    "ssh_user": "shardgrid",
                },
            ],
        }
    )
    training_path = tmp_path / "train-automatic.yaml"
    training_path.write_text(
        """
job:
  name: train-automatic
  backend: ssh
  communication_backend: nccl
model:
  name: tiny-sequential
  type: minimal_sequential
  stage_count: 2
resources:
  world_size: 4
  preferred_workers: [gpu4060, gpu1060, gpu4060-cqupt]
planning:
  mode: automatic
artifacts:
  snapshot_name: train-automatic
  keep_failed_snapshots: true
  transport: auto
""".strip()
        + "\n",
        encoding="utf-8",
    )
    events: list[str] = []
    planner_events: list[str] = []
    resources = {
        "gpu4060": _worker_resource(
            worker_id="gpu4060",
            host="10.0.0.10",
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            gpu_name="RTX 4060",
            gpu_free_memory=8_000,
        ),
        "gpu1060": _worker_resource(
            worker_id="gpu1060",
            host="10.0.0.11",
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            gpu_name="GTX 1060",
            gpu_free_memory=7_000,
        ),
        "gpu4060-cqupt": _worker_resource(
            worker_id="gpu4060-cqupt",
            host="10.0.0.12",
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            gpu_name="RTX 4060",
            gpu_free_memory=8_000,
        ),
    }

    class StaticDryRunEngine:
        def prepare(self, job_snapshot: object, execution_plan: object):
            del job_snapshot, execution_plan
            events.append("engine_prepare")
            raise AssertionError("dry-run must not call engine.prepare")

        def launch_metadata(self, parallel_plan: ParallelPlan) -> dict[str, object]:
            events.append("engine_launch_metadata")
            return {
                "engine": "galvatron",
                "plan_mode": "static",
                "parallel_plan_id": parallel_plan.parallel_plan_id,
            }

    class StaticSelectedEngine:
        def __init__(self) -> None:
            self.engine = StaticDryRunEngine()
            self.candidate = ParallelEngineCandidate(
                engine_id="galvatron",
                name=as_engine_name("galvatron"),
                status=BackendStatus.EXPERIMENTAL,
                capabilities=["ssh", "nccl"],
                limitations=["static fixture"],
            )
            self.parallel_plan = ParallelPlan(
                parallel_plan_id="static-plan",
                engine=as_engine_name("galvatron"),
                model_name="tiny-sequential",
                world_size=2,
                stages=["stage0", "stage1"],
                requirements={"plan_mode": "static"},
            )
            self.original_plan_path = None
            self.rejected_engine_ids = ()

    def probe_worker(worker):
        events.append(f"probe:{worker.worker_id}")
        return _probe_result(resources[str(worker.worker_id)])

    def probe_network(worker_resources):
        events.append("network")
        links = []
        for source in worker_resources:
            for target in worker_resources:
                if source.worker_id == target.worker_id:
                    continue
                links.append(
                    NetworkLink(
                        source_worker_id=source.worker_id,
                        target_worker_id=target.worker_id,
                        source_ip=source.ip or "10.0.0.10",
                        target_ip=target.ip or "10.0.0.11",
                        interface="eth0",
                        tcp_reachable=True,
                        bandwidth_mbps=940.0,
                        latency_ms=0.8,
                        port=29500,
                        measured_at="2026-09-03T10:00:00+00:00",
                    )
                )
        return NetworkState(
            network_id="lan-auto",
            workers=[item.worker_id for item in worker_resources],
            links=links,
            created_at="2026-09-03T10:00:00+00:00",
            selected_interfaces={item.worker_id: "eth0" for item in worker_resources},
        )

    def select_engine(engine_id, job, resources, network, *, registry=None):
        del engine_id, job, resources, network, registry
        events.append("select_engine")
        return StaticSelectedEngine()

    def launcher_factory(backend: str):
        del backend
        events.append("launcher_factory")
        raise AssertionError("dry-run must not create a launcher")

    original_build_model_profile = job_manager_module.build_model_profile
    original_search_joint_partition_placement = job_manager_module.search_joint_partition_placement
    original_select_best_joint_placement_plan = job_manager_module.select_best_joint_placement_plan
    original_build_automatic_parallel_plan = job_manager_module.build_automatic_parallel_plan

    def wrapped_build_model_profile(*args, **kwargs):
        planner_events.append("t109")
        return original_build_model_profile(*args, **kwargs)

    def wrapped_search_joint_partition_placement(*args, **kwargs):
        planner_events.append("t111")
        return original_search_joint_partition_placement(*args, **kwargs)

    def wrapped_select_best_joint_placement_plan(*args, **kwargs):
        planner_events.append("t112")
        return original_select_best_joint_placement_plan(*args, **kwargs)

    def wrapped_build_automatic_parallel_plan(*args, **kwargs):
        planner_events.append("t114")
        return original_build_automatic_parallel_plan(*args, **kwargs)

    monkeypatch.setattr(job_manager_module, "build_model_profile", wrapped_build_model_profile)
    monkeypatch.setattr(
        job_manager_module,
        "search_joint_partition_placement",
        wrapped_search_joint_partition_placement,
    )
    monkeypatch.setattr(
        job_manager_module,
        "select_best_joint_placement_plan",
        wrapped_select_best_joint_placement_plan,
    )
    monkeypatch.setattr(
        job_manager_module,
        "build_automatic_parallel_plan",
        wrapped_build_automatic_parallel_plan,
    )

    manager = JobManager(
        cluster_config,
        probe_worker=probe_worker,
        probe_network=probe_network,
        select_engine=select_engine,
        launcher_factory=launcher_factory,
        source_root=Path(__file__).resolve().parents[2],
    )

    result = manager.run(training_path, job_id=as_job_id("job-auto-cli"), dry_run=True)
    payload = train_command._payload(result)
    human = train_command._render_human(result)
    metadata_json = json.loads(
        (Path(result.snapshot.diagnostics_path) / "snapshot-metadata.json").read_text()
    )

    assert planner_events == ["t109", "t111", "t112", "t114"]
    assert events == [
        "probe:gpu4060",
        "probe:gpu1060",
        "probe:gpu4060-cqupt",
        "network",
        "select_engine",
        "engine_launch_metadata",
    ]
    assert result.parallel_plan is not None
    assert result.parallel_plan.partition_source == "automatic"
    assert result.parallel_plan.requirements["plan_mode"] == "automatic"
    assert payload["plan_mode"] == "automatic"
    assert payload["planning"]["partition_source"] == "automatic"
    assert payload["planning"]["selected_worker_count"] == "2"
    assert payload["planning"]["attempted_worker_counts"] == [2]
    assert payload["planning"]["selected_workers"] == [
        item["worker_id"] for item in payload["placement"]
    ]
    assert set(payload["planning"]["selected_workers"]) == {"gpu4060", "gpu4060-cqupt"}
    assert payload["world_size"] == 2
    assert len(payload["placement"]) == 2
    assert len(payload["assignments"]) == 2
    assert result.execution_plan.workers[0].launch_command == (
        "python examples/models/train_automatic_plan.py --rank 0"
    )
    assert result.execution_plan.workers[1].launch_command == (
        "python examples/models/train_automatic_plan.py --rank 1"
    )
    assert result.execution_plan.workers[0].environment["SHARDGRID_PARTITION_SOURCE"] == "automatic"
    assert result.execution_plan.workers[0].environment["SHARDGRID_SELECTED_CANDIDATE_ID"]
    assert result.execution_plan.workers[0].environment["SHARDGRID_AUTOMATIC_MICROBATCHES"] == "2"
    assert result.execution_plan.workers[1].environment["SHARDGRID_AUTOMATIC_MICROBATCHES"] == "2"
    assert (
        result.execution_plan.workers[0].environment["SHARDGRID_AUTOMATIC_CHECKPOINT_DIR"]
        == "checkpoint"
    )
    assert metadata_json["execution_plan_audit"]["plan_mode"] == "automatic"
    assert metadata_json["execution_plan_audit"]["planning"] == payload["planning"]
    planning_evidence = payload["launch_metadata"]["planning_evidence"]
    assert set(planning_evidence) >= {
        "planner_worker_resources",
        "control_rss_before_planning",
        "control_rss_after_profile",
        "control_rss_after_plan_created",
        "control_rss_after_cleanup",
    }
    assert metadata_json["execution_plan_audit"]["launch_metadata"]["planning_evidence"] == (
        planning_evidence
    )
    assert "Plan Mode: automatic" in human
    assert "Partition Source: automatic" in human
    assert "Selected Worker Count: 2" in human
    assert "Attempted Worker Counts: 2" in human
    assert "Selected Workers: gpu4060, gpu4060-cqupt" in human


def test_static_dry_run_path_still_uses_engine_plan(tmp_path: Path) -> None:
    cluster_config = ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
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
            "workers": [
                {
                    "id": "gpu4060",
                    "machine_id": "machine-c",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.10",
                    "ssh_user": "shardgrid",
                },
                {
                    "id": "gpu1060",
                    "machine_id": "machine-d",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.11",
                    "ssh_user": "shardgrid",
                },
            ],
        }
    )
    training_path = tmp_path / "train-static.yaml"
    training_path.write_text(
        """
job:
  name: train-static
  backend: ssh
  communication_backend: nccl
model:
  name: tiny-sequential
  type: minimal_sequential
resources:
  world_size: 2
  preferred_workers: [gpu4060, gpu1060]
artifacts:
  snapshot_name: train-static
  keep_failed_snapshots: true
  transport: auto
""".strip()
        + "\n",
        encoding="utf-8",
    )
    events: list[str] = []
    resources = {
        "gpu4060": _worker_resource(
            worker_id="gpu4060",
            host="10.0.0.10",
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            gpu_name="RTX 4060",
            gpu_free_memory=8_000,
        ),
        "gpu1060": _worker_resource(
            worker_id="gpu1060",
            host="10.0.0.11",
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            gpu_name="GTX 1060",
            gpu_free_memory=7_000,
        ),
    }

    class StaticDryRunEngine:
        def prepare(self, job_snapshot: object, execution_plan: object):
            del job_snapshot, execution_plan
            events.append("engine_prepare")
            raise AssertionError("dry-run must not call engine.prepare")

        def launch_metadata(self, parallel_plan: ParallelPlan) -> dict[str, object]:
            events.append("engine_launch_metadata")
            return {
                "engine": "galvatron",
                "plan_mode": "static",
                "parallel_plan_id": parallel_plan.parallel_plan_id,
            }

    class StaticSelectedEngine:
        def __init__(self) -> None:
            self.engine = StaticDryRunEngine()
            self.candidate = ParallelEngineCandidate(
                engine_id="galvatron",
                name=as_engine_name("galvatron"),
                status=BackendStatus.EXPERIMENTAL,
                capabilities=["ssh", "nccl"],
                limitations=["static fixture"],
            )
            self.parallel_plan = ParallelPlan(
                parallel_plan_id="static-plan",
                engine=as_engine_name("galvatron"),
                model_name="tiny-sequential",
                world_size=2,
                stages=["stage0", "stage1"],
                requirements={"plan_mode": "static"},
            )
            self.original_plan_path = None
            self.rejected_engine_ids = ()

    def probe_worker(worker):
        events.append(f"probe:{worker.worker_id}")
        return _probe_result(resources[str(worker.worker_id)])

    def probe_network(worker_resources):
        del worker_resources
        events.append("network")
        return _network_state()

    def select_engine(engine_id, job, resources, network, *, registry=None):
        del engine_id, job, resources, network, registry
        events.append("select_engine")
        return StaticSelectedEngine()

    def launcher_factory(backend: str):
        del backend
        events.append("launcher_factory")
        raise AssertionError("dry-run must not create a launcher")

    manager = JobManager(
        cluster_config,
        probe_worker=probe_worker,
        probe_network=probe_network,
        select_engine=select_engine,
        launcher_factory=launcher_factory,
        source_root=Path(__file__).resolve().parents[2],
    )

    result = manager.run(training_path, job_id=as_job_id("job-static-cli"), dry_run=True)
    payload = train_command._payload(result)

    assert result.parallel_plan is not None
    assert result.parallel_plan.partition_source is None
    assert payload["plan_mode"] == "static"
    assert payload["planning"]["partition_source"] == "manual"
    assert payload["world_size"] == 2
    assert events == [
        "probe:gpu4060",
        "probe:gpu1060",
        "network",
        "select_engine",
        "engine_launch_metadata",
    ]
