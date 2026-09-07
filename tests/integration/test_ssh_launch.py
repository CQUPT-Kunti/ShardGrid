from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shardgrid.artifacts.ssh_transport import DistributionStatus, WorkerDistributionResult
from shardgrid.common.config import (
    BackendPreferenceConfig,
    ClusterConfig,
    ControlNodeConfig,
    ManualOverrideConfig,
    NetworkConfig,
    RuntimeConfig,
    SSHConfig,
    WorkerConfig,
)
from shardgrid.common.enums import FailureStage, Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_machine_id,
    as_worker_id,
)
from shardgrid.common.process import ProcessResult
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.jobs.models import JobSnapshot, JobStatus, TrainingJob
from shardgrid.launchers.base import LauncherContext, LauncherResultStatus
from shardgrid.launchers.ssh import SSHLauncher
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


class FakeRuntime:
    def __init__(self, *, script_results: list[ProcessResult] | None = None) -> None:
        self.script_results = list(script_results or [])
        self.script_calls: list[str] = []

    def run(self, command, **kwargs) -> ProcessResult:
        return ProcessResult(
            args=command,
            recorded_command="python --version",
            shell=False,
            cwd=None,
            exit_code=0,
            stdout="Python 3.12.13",
            stderr="",
            timed_out=False,
            runtime_environment={"via": "ssh-wsl"},
        )

    def run_script(self, script: str, **kwargs) -> ProcessResult:
        self.script_calls.append(script)
        if not self.script_results:
            raise AssertionError("unexpected runtime.run_script call")
        return self.script_results.pop(0)


def _result(
    command: str,
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> ProcessResult:
    return ProcessResult(
        args=command,
        recorded_command=command,
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        runtime_environment={"via": "ssh-wsl"},
    )


def _cluster_config() -> ClusterConfig:
    return ClusterConfig(
        control=ControlNodeConfig(
            machine_id=as_machine_id("machine-a"),
            hostname=as_hostname("control"),
        ),
        jobs_root=Path("/var/tmp/shardgrid/jobs"),
        ssh=SSHConfig(
            default_port=22,
            connect_timeout_seconds=15,
            strict_host_key_checking=True,
            private_key_path="/home/test/.ssh/id_ed25519",
        ),
        runtime=RuntimeConfig(
            python_executable="python3",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            default_wsl_distro="Ubuntu-22.04",
        ),
        network=NetworkConfig(),
        backend_preference=BackendPreferenceConfig(
            launcher=as_backend_name("ssh"),
            communication_backend=as_backend_name("nccl"),
            parallel_engine=as_engine_name("torchrun"),
        ),
        manual_override=ManualOverrideConfig(),
        workers=[
            WorkerConfig(
                worker_id=as_worker_id("gpu4060"),
                machine_id=as_machine_id("machine-c"),
                physical_os=PhysicalOS.WINDOWS,
                runtime_os=RuntimeOS.WSL2_LINUX,
                runtime="wsl2",
                host=as_hostname("10.87.5.155"),
                ssh_user="shardgrid",
                runtime_distro="Ubuntu-22.04",
                conda_environment="shardgrid",
                conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            ),
            WorkerConfig(
                worker_id=as_worker_id("gpu1060"),
                machine_id=as_machine_id("machine-d"),
                physical_os=PhysicalOS.WINDOWS,
                runtime_os=RuntimeOS.WSL2_LINUX,
                runtime="wsl2",
                host=as_hostname("10.87.5.15"),
                ssh_user="shardgrid",
                runtime_distro="Ubuntu-22.04",
                conda_environment="shardgrid",
                conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            ),
        ],
    )


def _context(tmp_path: Path) -> LauncherContext:
    snapshot_root = tmp_path / "job-0096"
    for rel in ("code", "config", "plan", "logs", "checkpoint", "environment", "diagnostics"):
        (snapshot_root / rel).mkdir(parents=True, exist_ok=True)
    target = snapshot_root / "code" / "examples" / "models"
    target.mkdir(parents=True, exist_ok=True)
    (target / "train_pipeline.py").write_text("print('ok')\n")
    workers = [
        WorkerResource(
            worker_id=as_worker_id("gpu4060"),
            hostname=as_hostname("ldj"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            ip="10.87.5.155",
            gpu_name="RTX 4060",
            torch_version="2.7.1+cu118",
            health=Health.HEALTHY,
            last_probe_at="2026-08-28T11:00:00+00:00",
        ),
        WorkerResource(
            worker_id=as_worker_id("gpu1060"),
            hostname=as_hostname("laptop"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            ip="10.87.5.15",
            gpu_name="GTX 1650",
            torch_version="2.7.1+cu118",
            health=Health.HEALTHY,
            last_probe_at="2026-08-28T11:00:00+00:00",
        ),
    ]
    network = NetworkState(
        network_id="lan",
        workers=[item.worker_id for item in workers],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id("gpu4060"),
                target_worker_id=as_worker_id("gpu1060"),
                source_ip="10.87.5.155",
                target_ip="10.87.5.15",
                interface="eth3",
                tcp_reachable=True,
                measured_at="2026-08-28T11:00:00+00:00",
            ),
            NetworkLink(
                source_worker_id=as_worker_id("gpu1060"),
                target_worker_id=as_worker_id("gpu4060"),
                source_ip="10.87.5.15",
                target_ip="10.87.5.155",
                interface="eth0",
                tcp_reachable=True,
                measured_at="2026-08-28T11:00:00+00:00",
            ),
        ],
        created_at="2026-08-28T11:00:00+00:00",
        selected_interfaces={"gpu4060": "eth3", "gpu1060": "eth0"},
    )
    cluster_state = ResourceManager().build_cluster_state(
        workers,
        network_state=network,
        require_network=True,
        now=datetime(2026, 8, 28, 11, 0, tzinfo=UTC),
    )
    job = TrainingJob(
        job_id=as_job_id("job-0096"),
        config_path="examples/train-minimal.yaml",
        model="tiny",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("nccl"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(
                worker_id=as_worker_id("gpu4060"),
                rank=0,
                local_rank=0,
                stage="stage0",
                gpu_index=0,
                launch_command="python examples/models/train_pipeline.py --rank 0",
                log_path="jobs/job-0096/logs/gpu4060-rank0.log",
                environment={"SHARDGRID_PIPELINE_TASK": "t071"},
            ),
            WorkerAssignment(
                worker_id=as_worker_id("gpu1060"),
                rank=1,
                local_rank=0,
                stage="stage1",
                gpu_index=0,
                launch_command="python examples/models/train_pipeline.py --rank 1",
                log_path="jobs/job-0096/logs/gpu1060-rank1.log",
                environment={"SHARDGRID_PIPELINE_TASK": "t071"},
            ),
        ],
        snapshot_ref=str(snapshot_root),
    )
    snapshot = JobSnapshot(
        job_id=job.job_id,
        root_path=str(snapshot_root),
        code_path=str(snapshot_root / "code"),
        config_path=str(snapshot_root / "config"),
        plan_path=str(snapshot_root / "plan"),
        logs_path=str(snapshot_root / "logs"),
        environment_path=str(snapshot_root / "environment"),
        checkpoint_path=str(snapshot_root / "checkpoint"),
        diagnostics_path=str(snapshot_root / "diagnostics"),
    )
    return LauncherContext(
        job=job,
        execution_plan=plan,
        cluster_state=cluster_state,
        snapshot=snapshot,
        job_status=JobStatus(job_id=job.job_id, state=JobState.CREATED, phase="created"),
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
    )


def _distribution(worker_id: str, remote_root: str) -> WorkerDistributionResult:
    return WorkerDistributionResult(
        worker_id=worker_id,
        host="host",
        transport="scp",
        status=DistributionStatus.PASS,
        remote_snapshot_root=remote_root,
        control_checksum="same",
        remote_checksum="same",
        remote_job_id="job-0096",
        metadata_ready=True,
    )


def _distribution_records(
    job_id: str,
    remote_root: str,
) -> dict[tuple[str, str], WorkerDistributionResult]:
    return {
        (job_id, "gpu4060"): _distribution("gpu4060", remote_root),
        (job_id, "gpu1060"): _distribution("gpu1060", remote_root),
    }


def _payload(script: str) -> dict[str, object]:
    prefix = "payload = json.loads("
    line = next(item for item in script.splitlines() if item.startswith(prefix))
    return json.loads(json.loads(line[len(prefix) : -1]))


@pytest.mark.integration
def test_launch_injects_remote_snapshot_env_and_independent_logs(tmp_path: Path) -> None:
    context = _context(tmp_path)
    runtimes = {
        "gpu4060": FakeRuntime(script_results=[_result("launch", stdout="4100")]),
        "gpu1060": FakeRuntime(script_results=[_result("launch", stdout="4200")]),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    launcher._distribution_records = _distribution_records(
        str(context.job.job_id),
        "/var/tmp/shardgrid/jobs/job-0096",
    )

    result = launcher.launch(context)

    assert result.status is LauncherResultStatus.SUCCESS
    rank0 = _payload(runtimes["gpu4060"].script_calls[0])
    rank1 = _payload(runtimes["gpu1060"].script_calls[0])
    assert rank0["cwd"] == "/var/tmp/shardgrid/jobs/job-0096/code"
    assert rank1["cwd"] == "/var/tmp/shardgrid/jobs/job-0096/code"
    assert rank0["env"]["RANK"] == "0"
    assert rank1["env"]["RANK"] == "1"
    assert rank0["env"]["WORLD_SIZE"] == "2"
    assert rank1["env"]["WORLD_SIZE"] == "2"
    assert rank0["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert rank0["env"]["MASTER_ADDR"] == "10.87.5.155"
    assert rank0["env"]["MASTER_PORT"] == "29500"
    assert rank0["env"]["SHARDGRID_BACKEND"] == "nccl"
    assert rank1["env"]["SHARDGRID_BACKEND"] == "nccl"
    assert rank0["env"]["NCCL_SOCKET_IFNAME"] == "eth3"
    assert rank1["env"]["NCCL_SOCKET_IFNAME"] == "eth0"
    assert rank0["env"]["GLOO_SOCKET_IFNAME"] == "eth3"
    assert rank1["env"]["GLOO_SOCKET_IFNAME"] == "eth0"
    assert rank0["env"]["NCCL_SOCKET_FAMILY"] == "AF_INET"
    assert rank1["env"]["NCCL_IB_DISABLE"] == "1"
    assert rank0["env"]["SHARDGRID_NETWORK_SOURCE_IP"] == "10.87.5.155"
    assert rank1["env"]["SHARDGRID_NETWORK_SOURCE_IP"] == "10.87.5.15"
    assert rank0["env"]["SHARDGRID_NETWORK_PEER_IP"] == "10.87.5.15"
    assert rank1["env"]["SHARDGRID_NETWORK_PEER_IP"] == "10.87.5.155"
    assert rank0["env"]["LOCAL_RANK"] == "0"
    assert rank1["env"]["LOCAL_RANK"] == "0"
    assert rank0["env"]["SHARDGRID_STAGE"] == "stage0"
    assert rank1["env"]["SHARDGRID_STAGE"] == "stage1"
    assert rank0["env"]["SHARDGRID_LAUNCHER_OWNS_LOG_SINK"] == "1"
    assert rank1["env"]["SHARDGRID_LAUNCHER_OWNS_LOG_SINK"] == "1"
    assert rank0["env"]["PYTHONPATH"] == (
        "/var/tmp/shardgrid/jobs/job-0096/code:/var/tmp/shardgrid/jobs/job-0096/code/src"
    )
    assert rank1["env"]["PYTHONPATH"] == (
        "/var/tmp/shardgrid/jobs/job-0096/code:/var/tmp/shardgrid/jobs/job-0096/code/src"
    )
    assert rank0["env"]["SHARDGRID_REMOTE_SNAPSHOT_ROOT"] == "/var/tmp/shardgrid/jobs/job-0096"
    assert rank1["env"]["SHARDGRID_REMOTE_SNAPSHOT_ROOT"] == "/var/tmp/shardgrid/jobs/job-0096"
    assert rank0["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert rank1["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert rank0["log_path"] == (
        "/var/tmp/shardgrid/jobs/job-0096/jobs/job-0096/logs/gpu4060-rank0.log"
    )
    assert rank1["log_path"] == (
        "/var/tmp/shardgrid/jobs/job-0096/jobs/job-0096/logs/gpu1060-rank1.log"
    )
    assert rank0["log_path"] != rank1["log_path"]
    assert rank0["argv"][0] == "/home/shardgrid/miniconda3/envs/shardgrid/bin/python"
    assert rank0["argv"][1] == (
        "/var/tmp/shardgrid/jobs/job-0096/code/examples/models/train_pipeline.py"
    )
    assert rank1["argv"][1] == (
        "/var/tmp/shardgrid/jobs/job-0096/code/examples/models/train_pipeline.py"
    )
    records = launcher.process_records()
    assert {(item.rank, item.pid, item.stage) for item in records} == {
        (0, 4100, "stage0"),
        (1, 4200, "stage1"),
    }


@pytest.mark.integration
def test_launch_resolves_auto_backend_before_remote_launch(tmp_path: Path) -> None:
    base = _context(tmp_path)
    context = replace(
        base,
        execution_plan=replace(base.execution_plan, backend=as_backend_name("auto")),
    )
    runtimes = {
        "gpu4060": FakeRuntime(script_results=[_result("launch", stdout="4100")]),
        "gpu1060": FakeRuntime(script_results=[_result("launch", stdout="4200")]),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    launcher._distribution_records = _distribution_records(
        str(context.job.job_id),
        "/var/tmp/shardgrid/jobs/job-0096",
    )

    result = launcher.launch(context)

    assert result.status is LauncherResultStatus.SUCCESS
    rank0 = _payload(runtimes["gpu4060"].script_calls[0])
    rank1 = _payload(runtimes["gpu1060"].script_calls[0])
    assert rank0["env"]["SHARDGRID_BACKEND"] == "nccl"
    assert rank1["env"]["SHARDGRID_BACKEND"] == "nccl"


@pytest.mark.integration
def test_launch_blocks_invalid_master_and_partial_failure_is_preserved(tmp_path: Path) -> None:
    context = _context(tmp_path)
    blocked = replace(
        context,
        execution_plan=replace(
            context.execution_plan,
            master=MasterMetadata(address="10.87.5.155", port=29500),
            workers=[
                context.execution_plan.workers[0],
                replace(context.execution_plan.workers[1], launch_command=None),
            ],
        ),
    )
    launcher = SSHLauncher(_cluster_config(), runtime_factory=lambda worker: FakeRuntime())
    launcher._distribution_records = _distribution_records(
        str(blocked.job.job_id),
        "/var/tmp/shardgrid/jobs/job-0096",
    )
    result = launcher.launch(blocked)
    assert result.status is LauncherResultStatus.BLOCKED

    context = _context(tmp_path / "partial")
    runtimes = {
        "gpu4060": FakeRuntime(script_results=[_result("launch", stdout="4100")]),
        "gpu1060": FakeRuntime(script_results=[_result("launch", stderr="boom", exit_code=1)]),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    launcher._distribution_records = _distribution_records(
        str(context.job.job_id),
        "/var/tmp/shardgrid/jobs/job-0096",
    )
    result = launcher.launch(context)

    assert result.status is LauncherResultStatus.PARTIAL
    workers = {item.worker_id: item for item in result.worker_results}
    assert workers["gpu4060"].rank_results[0].pid == 4100
    assert workers["gpu1060"].failure is not None
    assert workers["gpu1060"].failure.stage is FailureStage.LAUNCH


@pytest.mark.integration
def test_duplicate_launch_is_noop_per_job_worker_rank(tmp_path: Path) -> None:
    context = _context(tmp_path)
    runtimes = {
        "gpu4060": FakeRuntime(script_results=[_result("launch", stdout="4100")]),
        "gpu1060": FakeRuntime(script_results=[_result("launch", stdout="4200")]),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    launcher._distribution_records = _distribution_records(
        str(context.job.job_id),
        "/var/tmp/shardgrid/jobs/job-0096",
    )

    first = launcher.launch(context)
    second = launcher.launch(context)

    assert first.status is LauncherResultStatus.SUCCESS
    assert second.status is LauncherResultStatus.NOOP
    assert all(item.status is LauncherResultStatus.NOOP for item in second.worker_results)
