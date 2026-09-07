from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp

import pytest

from shardgrid.artifacts.transport import (
    ArtifactTransferItemResult,
    ArtifactTransferResult,
    ArtifactTransferSpec,
    ArtifactTransferStatus,
    RemoteArtifactLocation,
)
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
from shardgrid.common.enums import (
    FailureStage,
    Health,
    JobState,
    PhysicalOS,
    RuntimeOS,
)
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
from shardgrid.jobs.models import FailureRecord, JobSnapshot, JobStatus, TrainingJob
from shardgrid.launchers.base import LauncherContext, LauncherResultStatus
from shardgrid.launchers.ssh import SSHLauncher
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


class FakeArtifactTransport:
    name = type("Name", (), {"value": "scp"})()

    def __init__(self, result: ArtifactTransferResult) -> None:
        self.result = result
        self.calls: list[tuple[list[ArtifactTransferSpec], RemoteArtifactLocation]] = []

    def transfer(
        self,
        items,
        *,
        remote,
        secrets=(),
    ) -> ArtifactTransferResult:
        self.calls.append((list(items), remote))
        return self.result


class FakeRuntime:
    def __init__(
        self,
        *,
        run_results: list[ProcessResult] | None = None,
        script_results: list[ProcessResult] | None = None,
        run_error: Exception | None = None,
        script_error: Exception | None = None,
    ) -> None:
        self.run_results = list(run_results or [])
        self.script_results = list(script_results or [])
        self.run_error = run_error
        self.script_error = script_error
        self.run_calls: list[object] = []
        self.script_calls: list[str] = []

    def run(self, command, **kwargs) -> ProcessResult:
        self.run_calls.append(command)
        if self.run_error is not None:
            raise self.run_error
        if not self.run_results:
            raise AssertionError("unexpected runtime.run call")
        return self.run_results.pop(0)

    def run_script(self, script: str, **kwargs) -> ProcessResult:
        self.script_calls.append(script)
        if self.script_error is not None:
            raise self.script_error
        if not self.script_results:
            raise AssertionError("unexpected runtime.run_script call")
        return self.script_results.pop(0)


def _ok_result(
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


def _probe_payload(
    *,
    snapshot_present: bool = False,
    metadata_job_id: str | None = None,
    entrypoint_exists: bool = True,
    created_dirs: list[str] | None = None,
    output_dirs_ready: bool = True,
) -> str:
    import json

    return json.dumps(
        {
            "remote_root": "/var/tmp/shardgrid/jobs/job-0093",
            "snapshot_present": snapshot_present,
            "metadata_job_id": metadata_job_id,
            "entrypoint_exists": entrypoint_exists,
            "created_dirs": created_dirs or [],
            "output_dirs_ready": output_dirs_ready,
        }
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
            WorkerConfig(
                worker_id=as_worker_id("gpu3090"),
                machine_id=as_machine_id("machine-e"),
                physical_os=PhysicalOS.WINDOWS,
                runtime_os=RuntimeOS.WSL2_LINUX,
                runtime="wsl2",
                host=as_hostname("10.87.5.99"),
                ssh_user="shardgrid",
                runtime_distro="Ubuntu-22.04",
                conda_environment="shardgrid",
                conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            ),
        ],
    )


def _context() -> LauncherContext:
    snapshot_root = Path(mkdtemp(prefix="shardgrid-t093-"))
    for rel in (
        "code",
        "config",
        "plan",
        "logs",
        "checkpoint",
        "environment",
        "diagnostics",
    ):
        (snapshot_root / rel).mkdir(parents=True, exist_ok=True)
    (snapshot_root / "code" / "train.py").write_text("print('ok')\n")
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
            last_probe_at="2026-08-27T11:00:00+00:00",
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
            last_probe_at="2026-08-27T11:00:00+00:00",
        ),
        WorkerResource(
            worker_id=as_worker_id("gpu3090"),
            hostname=as_hostname("bonus"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            ip="10.87.5.99",
            gpu_name="RTX 3090",
            torch_version="2.7.1+cu118",
            health=Health.HEALTHY,
            last_probe_at="2026-08-27T11:00:00+00:00",
        ),
    ]
    network = NetworkState(
        network_id="lan",
        workers=[item.worker_id for item in workers],
        links=[
            NetworkLink(
                source_worker_id=source.worker_id,
                target_worker_id=target.worker_id,
                source_ip=source.ip or "",
                target_ip=target.ip or "",
                interface="eth0",
                tcp_reachable=True,
                measured_at="2026-08-27T11:00:00+00:00",
            )
            for source in workers
            for target in workers
            if source.worker_id != target.worker_id
        ],
        created_at="2026-08-27T11:00:00+00:00",
    )
    cluster_state = ResourceManager().build_cluster_state(
        workers,
        network_state=network,
        require_network=True,
        now=datetime(2026, 8, 27, 11, 0, tzinfo=UTC),
    )
    job = TrainingJob(
        job_id=as_job_id("job-0093"),
        config_path="examples/train-minimal.yaml",
        model="tiny",
        requested_world_size=3,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("ssh"),
        world_size=3,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(
                worker_id=as_worker_id("gpu4060"),
                rank=0,
                stage="0",
                launch_command="python train.py --rank 0",
                log_path="jobs/job-0093/logs/rank0.log",
            ),
            WorkerAssignment(
                worker_id=as_worker_id("gpu1060"),
                rank=1,
                stage="1",
                launch_command="python train.py --rank 1",
                log_path="jobs/job-0093/logs/rank1.log",
            ),
            WorkerAssignment(
                worker_id=as_worker_id("gpu3090"),
                rank=2,
                stage="2",
                launch_command="python train.py --rank 2",
                log_path="jobs/job-0093/logs/rank2.log",
            ),
        ],
        snapshot_ref="jobs/job-0093",
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
        runtime_environment_refs={
            "0": "env:gpu4060/shardgrid",
            "1": "env:gpu1060/shardgrid",
            "2": "env:gpu3090/shardgrid",
        },
    )


def _single_worker_context(worker_id: str) -> LauncherContext:
    context = _context()
    workers = [
        assignment
        for assignment in context.execution_plan.workers
        if str(assignment.worker_id) == worker_id
    ]
    resources = [
        entry.resource
        for entry in context.cluster_state.workers
        if entry.worker_id == worker_id
    ]
    network = NetworkState(
        network_id="single",
        workers=[as_worker_id(worker_id)],
        links=[],
        created_at="2026-08-27T11:00:00+00:00",
    )
    return replace(
        context,
        execution_plan=replace(
            context.execution_plan,
            world_size=1,
            workers=workers,
        ),
        cluster_state=ResourceManager().build_cluster_state(
            resources,
            network_state=network,
            now=datetime(2026, 8, 27, 11, 0, tzinfo=UTC),
        ),
        runtime_environment_refs={"0": context.runtime_environment_refs["0"]},
    )


def _success_transfer() -> ArtifactTransferResult:
    return ArtifactTransferResult(
        transport="scp",
        status=ArtifactTransferStatus.SUCCESS,
        items=[
            ArtifactTransferItemResult(
                label="snapshot",
                transport="scp",
                status=ArtifactTransferStatus.SUCCESS,
                source="jobs/job-0093",
                destination=".shardgrid/launchers/job-0093",
                recorded_command="scp jobs/job-0093",
                exit_code=0,
            )
        ],
    )


def _distribution_result(
    worker_id: str,
    *,
    status: str = "PASS",
    skipped: bool = False,
    retryable: bool = False,
    message: str = "distribution failed",
):
    from shardgrid.artifacts.ssh_transport import (
        DistributionStatus,
        WorkerDistributionResult,
    )

    failure = None
    if status != "PASS":
        failure = FailureRecord(
            stage=FailureStage.DISTRIBUTE,
            host="10.87.5.155",
            worker_id=as_worker_id(worker_id),
            message=message,
            recommended_action="retry distribute",
            retryable=retryable,
        )
    return WorkerDistributionResult(
        worker_id=worker_id,
        host="10.87.5.155",
        transport="scp",
        status=DistributionStatus(status),
        remote_snapshot_root=f"/var/tmp/shardgrid/jobs/{worker_id}",
        control_checksum="expected",
        remote_checksum="expected" if status == "PASS" else "wrong",
        remote_job_id="job-0093",
        metadata_ready=status == "PASS",
        skipped=skipped,
        failure=failure,
    )


def test_ssh_launcher_satisfies_base_contract_and_backend_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeArtifactTransport(_success_transfer())
    runtimes = {
        "gpu4060": FakeRuntime(
            run_results=[_ok_result("python --version", stdout="Python 3.12.13")],
            script_results=[
                _ok_result("probe", stdout=_probe_payload(created_dirs=["logs"])),
                _ok_result("launch", stdout="4100"),
            ],
        ),
        "gpu1060": FakeRuntime(
            run_results=[_ok_result("python --version", stdout="Python 3.12.13")],
            script_results=[
                _ok_result("probe", stdout=_probe_payload(created_dirs=["logs"])),
                _ok_result("launch", stdout="4200"),
            ],
        ),
        "gpu3090": FakeRuntime(
            run_results=[_ok_result("python --version", stdout="Python 3.12.13")],
            script_results=[
                _ok_result("probe", stdout=_probe_payload(created_dirs=["logs"])),
                _ok_result("launch", stdout="4300"),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        artifact_transport=transport,
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    context = _context()

    def fake_distribute(*args, worker, transport, **kwargs):
        transport.transfer(
            [],
            remote=RemoteArtifactLocation(
                host=str(worker.host),
                user=worker.ssh_user,
                port=worker.ssh_port,
                path=".",
            ),
        )
        return _distribution_result(str(worker.worker_id))

    monkeypatch.setattr(
        "shardgrid.launchers.ssh.distribute_job_snapshot_to_worker",
        fake_distribute,
    )

    prepare = launcher.prepare(context)
    distribute = launcher.distribute(context)
    launch = launcher.launch(context)

    assert launcher.capabilities.backend == "ssh"
    assert prepare.status is LauncherResultStatus.SUCCESS
    assert distribute.status is LauncherResultStatus.SUCCESS
    assert launch.status is LauncherResultStatus.SUCCESS
    assert len(launcher.process_records()) == 3


def test_remote_runtime_command_uses_wrapper_and_transport_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeArtifactTransport(_success_transfer())
    runtime = FakeRuntime(
        run_results=[_ok_result("python --version", stdout="Python 3.12.13")],
        script_results=[
            _ok_result("probe", stdout=_probe_payload(created_dirs=["logs"])),
            _ok_result("launch", stdout="4100"),
        ],
    )
    launcher = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=transport,
        runtime_factory=lambda worker: runtime,
    )
    context = _single_worker_context("gpu4060")

    def fake_distribute(*args, worker, transport, **kwargs):
        transport.transfer(
            [],
            remote=RemoteArtifactLocation(
                host=str(worker.host),
                user=worker.ssh_user,
                port=worker.ssh_port,
                path=".",
            ),
        )
        return _distribution_result(str(worker.worker_id))

    monkeypatch.setattr(
        "shardgrid.launchers.ssh.distribute_job_snapshot_to_worker",
        fake_distribute,
    )

    launcher.prepare(context)
    launcher.distribute(context)
    launcher.launch(context)

    assert runtime.run_calls
    assert runtime.script_calls
    assert transport.calls


def test_duplicate_launch_protection_and_process_identity_are_per_job_worker_rank() -> None:
    runtime = FakeRuntime(
        script_results=[
            _ok_result("launch", stdout="4100"),
            _ok_result("launch", stdout="4200"),
            _ok_result("launch", stdout="4300"),
        ]
    )
    launcher = SSHLauncher(
        _cluster_config(),
        artifact_transport=FakeArtifactTransport(_success_transfer()),
        runtime_factory=lambda worker: runtime,
    )
    context = _context()
    launcher._distribution_records = {
        (str(context.job.job_id), worker_id): _distribution_result(worker_id)
        for worker_id in ("gpu4060", "gpu1060", "gpu3090")
    }

    first = launcher.launch(context)
    second = launcher.launch(context)
    records = launcher.process_records()

    assert first.status is LauncherResultStatus.SUCCESS
    assert second.status is LauncherResultStatus.NOOP
    assert len(records) == 3
    assert {(item.worker_id, item.rank, item.pid) for item in records} == {
        ("gpu4060", 0, 4100),
        ("gpu1060", 1, 4200),
        ("gpu3090", 2, 4300),
    }


@pytest.mark.parametrize(
    ("stderr", "expected_status"),
    [
        ("ssh executable not found: ssh", LauncherResultStatus.FAILED),
        ("Permission denied (publickey)", LauncherResultStatus.BLOCKED),
        ("Connection timed out", LauncherResultStatus.FAILED),
    ],
)
def test_prepare_structures_ssh_failures(
    stderr: str,
    expected_status: LauncherResultStatus,
) -> None:
    runtime = FakeRuntime(
        run_results=[
            _ok_result("python --version", stderr=stderr, exit_code=255)
        ]
    )
    launcher = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=FakeArtifactTransport(_success_transfer()),
        runtime_factory=lambda worker: runtime,
        secrets=("publickey",),
    )
    context = _single_worker_context("gpu4060")

    result = launcher.prepare(context)

    assert result.status is expected_status
    assert result.failure is not None
    assert result.failure.stage is FailureStage.LAUNCH
    assert "publickey" not in (
        result.failure.message + (result.failure.command or "")
    )


def test_runtime_wrapper_failure_and_remote_command_failure_become_failure_records() -> None:
    wrapper_fail = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=FakeArtifactTransport(_success_transfer()),
        runtime_factory=lambda worker: FakeRuntime(script_error=ValueError("missing distro")),
    )
    remote_fail = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=FakeArtifactTransport(_success_transfer()),
        runtime_factory=lambda worker: FakeRuntime(
            script_results=[_ok_result("launch", stderr="boom", exit_code=1)]
        ),
    )
    context = _single_worker_context("gpu4060")
    pass_distribution = {
        (str(context.job.job_id), "gpu4060"): _distribution_result("gpu4060")
    }
    wrapper_fail._distribution_records = dict(pass_distribution)
    remote_fail._distribution_records = dict(pass_distribution)

    wrapper_result = wrapper_fail.launch(context)
    remote_result = remote_fail.launch(context)

    assert wrapper_result.failure is not None
    assert "runtime wrapper failure" in wrapper_result.failure.message
    assert remote_result.failure is not None
    assert remote_result.failure.exit_code == 1


def test_artifact_transport_failure_is_structured() -> None:
    transport = FakeArtifactTransport(
        ArtifactTransferResult(
            transport="scp",
            status=ArtifactTransferStatus.FAILED,
            items=[
                ArtifactTransferItemResult(
                    label="snapshot",
                    transport="scp",
                    status=ArtifactTransferStatus.FAILED,
                    source="jobs/job-0093",
                    destination=".shardgrid/launchers/job-0093",
                    recorded_command="scp secret-token jobs/job-0093",
                    exit_code=1,
                    stderr="Permission denied secret-token",
                    retryable=False,
                )
            ],
        )
    )
    launcher = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=transport,
        runtime_factory=lambda worker: FakeRuntime(),
        secrets=("secret-token",),
    )
    context = _single_worker_context("gpu4060")

    result = launcher.distribute(context)

    assert result.status is LauncherResultStatus.BLOCKED
    assert result.failure is not None
    assert "secret-token" not in (result.failure.command or "")
    assert "secret-token" not in result.failure.message


def test_malformed_pid_does_not_create_process_record() -> None:
    launcher = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=FakeArtifactTransport(_success_transfer()),
        runtime_factory=lambda worker: FakeRuntime(
            script_results=[_ok_result("launch", stdout="not-a-pid")]
        ),
    )
    context = _single_worker_context("gpu4060")
    launcher._distribution_records = {
        (str(context.job.job_id), "gpu4060"): _distribution_result("gpu4060")
    }

    result = launcher.launch(context)

    assert result.status is LauncherResultStatus.FAILED
    assert not launcher.process_records()
    assert result.failure is not None
    assert "valid PID" in result.failure.message


def test_logs_and_monitor_use_tracked_processes_without_rank_confusion() -> None:
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_ok_result("launch", stdout="4100")],
            run_results=[
                _ok_result("kill -0", stdout=""),
                _ok_result("tail", stdout="r0"),
                _ok_result("tail", stdout="r0"),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_ok_result("launch", stdout="4200")],
            run_results=[
                _ok_result("kill -0", stdout=""),
                _ok_result("tail", stdout="r1"),
                _ok_result("tail", stdout="r1"),
            ],
        ),
        "gpu3090": FakeRuntime(
            script_results=[_ok_result("launch", stdout="4300")],
            run_results=[
                _ok_result("kill -0", stdout=""),
                _ok_result("tail", stdout="r2"),
                _ok_result("tail", stdout="r2"),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        artifact_transport=FakeArtifactTransport(_success_transfer()),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    context = _context()
    launcher._distribution_records = {
        (str(context.job.job_id), worker_id): _distribution_result(worker_id)
        for worker_id in ("gpu4060", "gpu1060", "gpu3090")
    }
    launcher.launch(context)

    monitor = launcher.monitor(context)
    logs = launcher.logs(context)

    assert monitor.status is LauncherResultStatus.SUCCESS
    assert [item.rank_results[0].pid for item in monitor.worker_results] == [4100, 4200, 4300]
    assert [item.tail for item in logs.log_results] == ["r0", "r1", "r2"]


def test_launch_is_blocked_without_verified_distribution() -> None:
    launcher = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=FakeArtifactTransport(_success_transfer()),
        runtime_factory=lambda worker: FakeRuntime(
            script_results=[_ok_result("launch", stdout="4100")]
        ),
    )
    context = _single_worker_context("gpu4060")

    result = launcher.launch(context)

    assert result.status is LauncherResultStatus.BLOCKED
    assert result.failure is not None
    assert result.failure.stage is FailureStage.DISTRIBUTE
