from __future__ import annotations

from pathlib import Path

from shardgrid.common.config import ClusterConfig
from shardgrid.common.enums import Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_backend_name, as_engine_name, as_hostname, as_job_id
from shardgrid.common.process import ProcessResult
from shardgrid.control.job_manager import JobManager
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.jobs.models import JobSnapshot, JobStatus, TrainingJob
from shardgrid.launchers.base import (
    LauncherContext,
    LauncherOperation,
    LauncherResult,
    LauncherResultStatus,
)
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkState, WorkerResource
from shardgrid.workers.gpu_probe import GPUProbeResult
from shardgrid.workers.models import WorkerRuntime


def _cluster_config(tmp_path: Path) -> ClusterConfig:
    return ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": str((tmp_path / "jobs").resolve()),
            "ssh": {},
            "runtime": {
                "conda_environment": "shardgrid",
                "conda_prefix": "/opt/conda/envs/shardgrid",
            },
            "network": {},
            "backend_preference": {},
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
                    "runtime_distro": "Ubuntu-22.04",
                    "conda_environment": "shardgrid",
                    "conda_prefix": "/opt/conda/envs/shardgrid",
                },
                {
                    "id": "gpu1060",
                    "machine_id": "machine-d",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.11",
                    "ssh_user": "shardgrid",
                    "runtime_distro": "Ubuntu-22.04",
                    "conda_environment": "shardgrid",
                    "conda_prefix": "/opt/conda/envs/shardgrid",
                }
            ],
        }
    )


def test_default_probe_worker_uses_structured_remote_probe_without_reprobing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    worker = manager.cluster_config.workers[0]
    gpu_probe_result = GPUProbeResult(
        worker_resource=WorkerResource(
            worker_id=worker.worker_id,
            hostname=worker.host,
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            conda_environment="shardgrid",
            conda_prefix="/opt/conda/envs/shardgrid",
            gpu_name="RTX 4060",
            health=Health.HEALTHY,
        ),
        worker_runtime=WorkerRuntime(
            worker_id=worker.worker_id,
            runtime_os=RuntimeOS.WSL2_LINUX,
            runtime_version="Ubuntu-22.04",
            conda_environment="shardgrid",
            conda_prefix="/opt/conda/envs/shardgrid",
            python_executable="/opt/conda/envs/shardgrid/bin/python",
            python_version="Python 3.12.0",
            health=Health.HEALTHY,
        ),
        failures=(),
        health=Health.HEALTHY,
        probe_status="live",
    )

    monkeypatch.setattr(
        "shardgrid.control.job_manager.run_remote_access_check",
        lambda transport, worker, **kwargs: type(
            "Access",
            (),
            {
                "status": "PASS",
                "runtime_identity": type(
                    "Identity",
                    (),
                    {
                        "wsl_distro": "Ubuntu-22.04",
                        "conda_executable": "/opt/conda/bin/conda",
                        "conda_environment": "shardgrid",
                        "conda_prefix": "/opt/conda/envs/shardgrid",
                        "python_executable": "/opt/conda/envs/shardgrid/bin/python",
                        "python_version": "Python 3.12.0",
                    },
                )(),
                "windows_identity": "worker-c",
                "gpu_probe_result": gpu_probe_result,
            },
        )(),
    )
    monkeypatch.setattr(
        "shardgrid.control.job_manager.probe_gpu",
        lambda wrapper, worker, **kwargs: (_ for _ in ()).throw(
            AssertionError("probe_gpu should not run after structured remote probe")
        ),
    )

    result = manager._default_probe_worker(worker)

    assert result.health is Health.HEALTHY
    assert result.worker_resource.ip == "10.0.0.10"
    assert result.worker_runtime.conda_executable == "/opt/conda/bin/conda"
    assert result.worker_runtime.python_executable == "/opt/conda/envs/shardgrid/bin/python"
    assert result.windows_host.os_version == "worker-c"


def test_default_probe_worker_preserves_wsl_reachability_for_late_probe_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    worker = manager.cluster_config.workers[0]

    monkeypatch.setattr(
        "shardgrid.control.job_manager.run_remote_access_check",
        lambda transport, worker, **kwargs: type(
            "Access",
            (),
            {
                "status": "FAIL",
                "runtime_identity": None,
                "windows_identity": "worker-c",
                "wsl_distro": "Ubuntu-22.04",
                "failure_category": "runtime_python_identity_timeout",
                "failure_reason": (
                    "WSL is reachable but the runtime Python identity command timed out"
                ),
                "exit_code": -1,
                "stderr": "timed out",
                "stdout": "",
            },
        )(),
    )

    result = manager._default_probe_worker(worker)

    assert result.health is Health.FAILED
    assert result.windows_host.os_version == "worker-c"
    assert result.windows_host.openssh_available is True
    assert result.windows_host.wsl_available is True
    assert result.failures[0].check == "runtime_python_identity_timeout"
    assert result.failures[0].message == (
        "WSL is reachable but the runtime Python identity command timed out"
    )


def test_default_probe_network_uses_route_specific_interface_and_source_ip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))

    class FakeRuntime:
        def __init__(self, route_output: str, link_output: str) -> None:
            self.route_output = route_output
            self.link_output = link_output

        def run(self, command, **kwargs) -> ProcessResult:
            del kwargs
            if command[:3] == ["ip", "route", "get"]:
                return ProcessResult(
                    args=command,
                    recorded_command=" ".join(command),
                    shell=False,
                    cwd=None,
                    exit_code=0,
                    stdout=self.route_output,
                    stderr="",
                    timed_out=False,
                    runtime_environment={},
                )
            if command[:4] == ["ip", "link", "show", "dev"]:
                return ProcessResult(
                    args=command,
                    recorded_command=" ".join(command),
                    shell=False,
                    cwd=None,
                    exit_code=0,
                    stdout=self.link_output,
                    stderr="",
                    timed_out=False,
                    runtime_environment={},
                )
            raise AssertionError(f"unexpected command: {command}")

    runtimes = {
        "gpu4060": FakeRuntime(
            "10.0.0.11 dev eth3 src 10.87.5.155 uid 1000\n",
            "5: eth3: <BROADCAST> mtu 1500 qdisc mq state UP mode DEFAULT\n",
        ),
        "gpu1060": FakeRuntime(
            "10.0.0.10 dev eth0 src 10.87.5.15 uid 1000\n",
            "3: eth0: <BROADCAST> mtu 1500 qdisc mq state UP mode DEFAULT\n",
        ),
    }
    monkeypatch.setattr(
        manager,
        "_runtime_wrapper",
        lambda worker: runtimes[str(worker.worker_id)],
    )

    state = manager._default_probe_network(
        [
            WorkerResource(
                worker_id=manager.cluster_config.workers[0].worker_id,
                hostname=manager.cluster_config.workers[0].host,
                physical_os=PhysicalOS.WINDOWS,
                runtime_os=RuntimeOS.WSL2_LINUX,
                ip="10.0.0.10",
                network_interface="eth0",
                health=Health.HEALTHY,
            ),
            WorkerResource(
                worker_id=manager.cluster_config.workers[1].worker_id,
                hostname=manager.cluster_config.workers[1].host,
                physical_os=PhysicalOS.WINDOWS,
                runtime_os=RuntimeOS.WSL2_LINUX,
                ip="10.0.0.11",
                network_interface="eth3",
                health=Health.HEALTHY,
            ),
        ]
    )

    assert isinstance(state, NetworkState)
    assert state.selected_interfaces == {"gpu4060": "eth3", "gpu1060": "eth0"}
    by_source = {str(link.source_worker_id): link for link in state.links}
    assert by_source["gpu4060"].source_ip == "10.87.5.155"
    assert by_source["gpu4060"].interface == "eth3"
    assert by_source["gpu1060"].source_ip == "10.87.5.15"
    assert by_source["gpu1060"].interface == "eth0"


def test_monitor_until_terminal_does_not_use_wall_clock_deadline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    snapshot_root = tmp_path / "snapshot"
    diagnostics_path = snapshot_root / "diagnostics"
    diagnostics_path.mkdir(parents=True, exist_ok=True)
    snapshot = JobSnapshot(
        job_id=as_job_id("job-monitor"),
        root_path=str(snapshot_root),
        code_path=str(snapshot_root / "code"),
        config_path=str(snapshot_root / "config"),
        plan_path=str(snapshot_root / "plan"),
        logs_path=str(snapshot_root / "logs"),
        environment_path=str(snapshot_root / "environment"),
        checkpoint_path=str(snapshot_root / "checkpoint"),
        diagnostics_path=str(diagnostics_path),
    )
    job = TrainingJob(
        job_id=as_job_id("job-monitor"),
        config_path="examples/train-minimal.yaml",
        model="tiny",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    assignments = [
        WorkerAssignment(worker_id="gpu4060", rank=0, stage="stage0"),
        WorkerAssignment(worker_id="gpu1060", rank=1, stage="stage1"),
    ]
    status = JobStatus(
        job_id=job.job_id,
        state=JobState.TRAINING,
        phase="training",
        workers=["gpu4060", "gpu1060"],
        assignments=assignments,
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        backend=as_backend_name("nccl"),
        started_at="2026-08-30T12:00:00+00:00",
    )
    plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("nccl"),
        world_size=2,
        master=MasterMetadata(address="10.0.0.10", port=29500),
        workers=assignments,
        snapshot_ref=str(snapshot_root),
    )
    context = LauncherContext(
        job=job,
        execution_plan=plan,
        cluster_state=ResourceManager().build_cluster_state(
            [
                WorkerResource(
                    worker_id="gpu4060",
                    hostname=as_hostname("10.0.0.10"),
                    physical_os=PhysicalOS.WINDOWS,
                    runtime_os=RuntimeOS.WSL2_LINUX,
                    health=Health.HEALTHY,
                ),
                WorkerResource(
                    worker_id="gpu1060",
                    hostname=as_hostname("10.0.0.11"),
                    physical_os=PhysicalOS.WINDOWS,
                    runtime_os=RuntimeOS.WSL2_LINUX,
                    health=Health.HEALTHY,
                ),
            ],
            network_state=None,
            require_network=False,
        ),
        snapshot=snapshot,
        job_status=status,
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
    )

    class FakeLauncher:
        def __init__(self) -> None:
            self.calls = 0

        def monitor(self, context) -> LauncherResult:
            self.calls += 1
            return LauncherResult(
                operation=LauncherOperation.MONITOR,
                status=LauncherResultStatus.SUCCESS,
                backend="ssh",
                job_id=str(context.job.job_id),
            )

    launcher = FakeLauncher()
    terminal = JobStatus(
        job_id=job.job_id,
        state=JobState.CHECKPOINTING,
        phase="checkpoint",
        workers=["gpu4060", "gpu1060"],
        assignments=assignments,
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        backend=as_backend_name("nccl"),
        started_at="2026-08-30T12:00:00+00:00",
    )
    seen = {"count": 0}

    def fake_load_status(snapshot, fallback):
        del snapshot, fallback
        seen["count"] += 1
        return terminal if seen["count"] >= 2 else status

    monkeypatch.setattr(manager, "_load_status", fake_load_status)
    monkeypatch.setattr(
        "shardgrid.control.job_manager.time.monotonic",
        lambda: (_ for _ in ()).throw(AssertionError("wall-clock deadline should be unused")),
    )
    monkeypatch.setattr("shardgrid.control.job_manager.time.sleep", lambda seconds: None)

    result, latest = manager._monitor_until_terminal(
        launcher=launcher,
        context=context,
        snapshot=snapshot,
        current=status,
        training_config=None,
    )

    assert launcher.calls == 2
    assert result.status is LauncherResultStatus.SUCCESS
    assert latest.state is JobState.CHECKPOINTING
