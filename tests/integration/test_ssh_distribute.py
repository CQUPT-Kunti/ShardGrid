from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shardgrid.artifacts.ssh_transport import DistributionStatus, WorkerDistributionResult
from shardgrid.artifacts.transport import ArtifactTransportName
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
from shardgrid.jobs.models import FailureRecord, JobSnapshot, JobStatus, TrainingJob
from shardgrid.launchers.base import LauncherContext, LauncherResultStatus
from shardgrid.launchers.ssh import SSHLauncher
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


class FakeArtifactTransport:
    name = ArtifactTransportName.SCP

    def transfer(self, items, *, remote, secrets=()):
        raise AssertionError("launcher distribute test patches worker distribution directly")


class FakeRuntime:
    def __init__(self, *, script_results: list[ProcessResult] | None = None) -> None:
        self.script_results = list(script_results or [])

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
        if not self.script_results:
            raise AssertionError("unexpected runtime.run_script call")
        return self.script_results.pop(0)


def _ok_result(command: str, *, stdout: str = "") -> ProcessResult:
    return ProcessResult(
        args=command,
        recorded_command=command,
        shell=False,
        cwd=None,
        exit_code=0,
        stdout=stdout,
        stderr="",
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


def _context(
    tmp_path: Path,
    worker_ids: tuple[str, ...] = ("gpu4060", "gpu1060"),
) -> LauncherContext:
    snapshot_root = tmp_path / "job-0095"
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
    config = _cluster_config()
    selected_configs = [worker for worker in config.workers if str(worker.worker_id) in worker_ids]
    resources = [
        WorkerResource(
            worker_id=worker.worker_id,
            hostname=as_hostname(str(worker.host)),
            physical_os=worker.physical_os,
            runtime_os=worker.runtime_os,
            conda_environment=worker.conda_environment,
            conda_prefix=worker.conda_prefix,
            python_executable=f"{worker.conda_prefix}/bin/python",
            ip=str(worker.host),
            gpu_name="gpu",
            torch_version="2.7.1+cu118",
            health=Health.HEALTHY,
            last_probe_at="2026-08-27T11:00:00+00:00",
        )
        for worker in selected_configs
    ]
    network = NetworkState(
        network_id="lan",
        workers=[item.worker_id for item in resources],
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
            for source in resources
            for target in resources
            if source.worker_id != target.worker_id
        ],
        created_at="2026-08-27T11:00:00+00:00",
    )
    cluster_state = ResourceManager().build_cluster_state(
        resources,
        network_state=network,
        require_network=len(resources) > 1,
        now=datetime(2026, 8, 27, 11, 0, tzinfo=UTC),
    )
    job = TrainingJob(
        job_id=as_job_id("job-0095"),
        config_path="examples/train-minimal.yaml",
        model="tiny",
        requested_world_size=len(worker_ids),
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("ssh"),
        world_size=len(worker_ids),
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(
                worker_id=as_worker_id(worker_id),
                rank=index,
                stage=str(index),
                launch_command=f"python train.py --rank {index}",
                log_path=f"jobs/job-0095/logs/rank{index}.log",
            )
            for index, worker_id in enumerate(worker_ids)
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
        runtime_environment_refs={
            str(index): f"env:{worker_id}/shardgrid"
            for index, worker_id in enumerate(worker_ids)
        },
    )


def _distribution_result(
    worker_id: str,
    *,
    status: DistributionStatus = DistributionStatus.PASS,
    skipped: bool = False,
    retryable: bool = False,
    remote_checksum: str = "expected",
    remote_job_id: str = "job-0095",
    message: str = "distribution failed",
) -> WorkerDistributionResult:
    failure = None
    if status is not DistributionStatus.PASS:
        failure = FailureRecord(
            stage=FailureStage.DISTRIBUTE,
            host=f"{worker_id}.host",
            worker_id=as_worker_id(worker_id),
            message=message,
            recommended_action="retry distribute",
            retryable=retryable,
        )
    return WorkerDistributionResult(
        worker_id=worker_id,
        host=f"{worker_id}.host",
        transport="scp",
        status=status,
        remote_snapshot_root=f"/var/tmp/shardgrid/jobs/job-0095/{worker_id}",
        control_checksum="expected",
        remote_checksum=remote_checksum,
        remote_job_id=remote_job_id,
        metadata_ready=status is DistributionStatus.PASS,
        skipped=skipped,
        transfer_result={"transport": "scp"},
        failure=failure,
    )


def test_launch_gate_stays_blocked_without_every_worker_verified(tmp_path: Path) -> None:
    context = _context(tmp_path)
    launcher = SSHLauncher(
        _cluster_config(),
        artifact_transport=FakeArtifactTransport(),
        runtime_factory=lambda worker: FakeRuntime(),
    )
    launcher._distribution_records = {
        (str(context.job.job_id), "gpu4060"): _distribution_result("gpu4060"),
    }

    result = launcher.launch(context)

    assert result.status is LauncherResultStatus.BLOCKED
    assert result.failure is not None
    assert result.failure.stage is FailureStage.DISTRIBUTE


def test_distribute_passes_and_second_run_is_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    launcher = SSHLauncher(
        _cluster_config(),
        artifact_transport=FakeArtifactTransport(),
        runtime_factory=lambda worker: FakeRuntime(
            script_results=[_ok_result("launch", stdout="4100")]
        ),
    )
    calls: dict[str, int] = {}

    def fake_distribute(*args, worker, **kwargs):
        calls[str(worker.worker_id)] = calls.get(str(worker.worker_id), 0) + 1
        return _distribution_result(
            str(worker.worker_id),
            skipped=calls[str(worker.worker_id)] > 1,
        )

    monkeypatch.setattr(
        "shardgrid.launchers.ssh.distribute_job_snapshot_to_worker",
        fake_distribute,
    )

    first = launcher.distribute(context)
    second = launcher.distribute(context)
    launch = launcher.launch(context)

    assert first.status is LauncherResultStatus.SUCCESS
    assert second.status is LauncherResultStatus.NOOP
    assert launch.status is LauncherResultStatus.SUCCESS
    assert all(item.status is LauncherResultStatus.NOOP for item in second.worker_results)


def test_distribute_partial_failure_blocks_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    launcher = SSHLauncher(
        _cluster_config(),
        artifact_transport=FakeArtifactTransport(),
        runtime_factory=lambda worker: FakeRuntime(
            script_results=[_ok_result("launch", stdout="4100")]
        ),
    )

    def fake_distribute(*args, worker, **kwargs):
        if str(worker.worker_id) == "gpu4060":
            return _distribution_result("gpu4060")
        return _distribution_result(
            "gpu1060",
            status=DistributionStatus.BLOCKED,
            message="artifact transport failed before the remote snapshot was ready",
        )

    monkeypatch.setattr(
        "shardgrid.launchers.ssh.distribute_job_snapshot_to_worker",
        fake_distribute,
    )

    distribute = launcher.distribute(context)
    launch = launcher.launch(context)

    assert distribute.status is LauncherResultStatus.PARTIAL
    results = {item.worker_id: item for item in distribute.worker_results}
    assert results["gpu4060"].status is LauncherResultStatus.SUCCESS
    assert results["gpu1060"].status is LauncherResultStatus.BLOCKED
    assert launch.status is LauncherResultStatus.BLOCKED
    assert launch.failure is not None


@pytest.mark.parametrize(
    ("distribution", "expected"),
    [
        (
            _distribution_result(
                "gpu4060",
                status=DistributionStatus.FAIL,
                remote_checksum="wrong",
                message="remote snapshot verification failed after distribution",
            ),
            LauncherResultStatus.FAILED,
        ),
        (
            _distribution_result(
                "gpu4060",
                status=DistributionStatus.FAIL,
                remote_checksum=None,
                remote_job_id=None,
                message="remote snapshot verification failed after distribution",
            ),
            LauncherResultStatus.FAILED,
        ),
    ],
)
def test_distribute_reports_checksum_and_missing_snapshot_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distribution: WorkerDistributionResult,
    expected: LauncherResultStatus,
) -> None:
    context = _context(tmp_path, ("gpu4060",))
    launcher = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=FakeArtifactTransport(),
        runtime_factory=lambda worker: FakeRuntime(),
    )
    monkeypatch.setattr(
        "shardgrid.launchers.ssh.distribute_job_snapshot_to_worker",
        lambda *args, **kwargs: distribution,
    )

    result = launcher.distribute(context)

    assert result.status is expected
    assert result.failure is not None
    evidence_path = Path(context.snapshot.diagnostics_path) / "distribute-gpu4060.json"
    assert evidence_path.exists()
    payload = evidence_path.read_text(encoding="utf-8")
    assert "remote_checksum" in payload


def test_distribute_retries_retryable_failure_once_and_keeps_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, ("gpu4060",))
    launcher = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        artifact_transport=FakeArtifactTransport(),
        runtime_factory=lambda worker: FakeRuntime(
            script_results=[_ok_result("launch", stdout="4100")]
        ),
    )
    attempts = {"count": 0}

    def fake_distribute(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _distribution_result(
                "gpu4060",
                status=DistributionStatus.FAIL,
                retryable=True,
                message="artifact transport failed before the remote snapshot was ready",
            )
        return _distribution_result("gpu4060")

    monkeypatch.setattr(
        "shardgrid.launchers.ssh.distribute_job_snapshot_to_worker",
        fake_distribute,
    )

    result = launcher.distribute(context)
    launch = launcher.launch(context)

    assert attempts["count"] == 2
    assert result.status is LauncherResultStatus.SUCCESS
    assert "2 attempts" in result.worker_results[0].message
    assert launch.status is LauncherResultStatus.SUCCESS


def test_distribute_uses_worker_aware_transport_selection_for_windows_auto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, ("gpu4060",))
    launcher = SSHLauncher(
        replace(_cluster_config(), workers=_cluster_config().workers[:1]),
        runtime_factory=lambda worker: FakeRuntime(),
    )
    seen: dict[str, object] = {}

    def fake_distribute(*args, transport=None, preferred_transport, **kwargs):
        seen["transport"] = transport
        seen["preferred_transport"] = preferred_transport
        return _distribution_result("gpu4060")

    monkeypatch.setattr(
        "shardgrid.launchers.ssh.distribute_job_snapshot_to_worker",
        fake_distribute,
    )

    result = launcher.distribute(context)

    assert result.status is LauncherResultStatus.SUCCESS
    assert seen["transport"] is None
    assert seen["preferred_transport"] == "auto"
