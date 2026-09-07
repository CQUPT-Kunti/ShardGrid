from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shardgrid.artifacts.ssh_transport import (
    DistributionStatus,
    RemoteSnapshotProbe,
    WorkerDistributionResult,
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
from shardgrid.launchers.ssh import SSHLauncher, SSHProcessRecord
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


class FakeRuntime:
    def __init__(
        self,
        *,
        script_results: list[ProcessResult] | None = None,
        run_results: list[ProcessResult] | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.script_results = list(script_results or [])
        self.run_results = list(run_results or [])
        self.run_error = run_error
        self.script_calls: list[str] = []
        self.run_calls: list[object] = []

    def run(self, command, **kwargs) -> ProcessResult:
        self.run_calls.append(command)
        if self.run_error is not None:
            raise self.run_error
        if not self.run_results:
            raise AssertionError("unexpected runtime.run call")
        return self.run_results.pop(0)

    def run_script(self, script: str, **kwargs) -> ProcessResult:
        self.script_calls.append(script)
        if not self.script_results:
            raise AssertionError("unexpected runtime.run_script call")
        return self.script_results.pop(0)


class FakeSSH:
    def __init__(
        self,
        *,
        run_results: list[ProcessResult] | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.run_results = list(run_results or [])
        self.run_error = run_error
        self.run_calls: list[str] = []

    def run(self, command: str, **kwargs) -> ProcessResult:
        self.run_calls.append(command)
        if self.run_error is not None:
            raise self.run_error
        if not self.run_results:
            raise AssertionError("unexpected ssh.run call")
        return self.run_results.pop(0)


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


def _stop_stdout(pid: int, final: str, signals: list[str], elapsed_seconds: float) -> str:
    return json.dumps(
        {
            "elapsed_seconds": elapsed_seconds,
            "final": final,
            "initial": "running",
            "pid": pid,
            "signals": signals,
        },
        sort_keys=True,
    )


def _cleanup_stdout(
    path: str,
    *,
    final_state: str = "removed",
    removed: list[str] | None = None,
    skipped: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "path": path,
            "final_state": final_state,
            "removed_paths": removed or [path],
            "skipped_items": skipped or [],
        },
        sort_keys=True,
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


def _context(
    tmp_path: Path,
    *,
    phase: str = "training",
    state: JobState = JobState.TRAINING,
) -> LauncherContext:
    snapshot_root = tmp_path / "job-0100"
    for rel in ("code", "config", "plan", "logs", "checkpoint", "environment", "diagnostics"):
        (snapshot_root / rel).mkdir(parents=True, exist_ok=True)
    target = snapshot_root / "code" / "examples" / "models"
    target.mkdir(parents=True, exist_ok=True)
    (target / "train_pipeline.py").write_text("print('ok')\n", encoding="utf-8")
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
            last_probe_at="2026-08-29T11:00:00+00:00",
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
            last_probe_at="2026-08-29T11:00:00+00:00",
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
                measured_at="2026-08-29T11:00:00+00:00",
            )
            for source in workers
            for target in workers
            if source.worker_id != target.worker_id
        ],
        created_at="2026-08-29T11:00:00+00:00",
    )
    cluster_state = ResourceManager().build_cluster_state(
        workers,
        network_state=network,
        require_network=True,
        now=datetime(2026, 8, 29, 11, 0, tzinfo=UTC),
    )
    job = TrainingJob(
        job_id=as_job_id("job-0100"),
        config_path="examples/train-minimal.yaml",
        model="tiny",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    assignments = [
        WorkerAssignment(
            worker_id=as_worker_id("gpu4060"),
            rank=0,
            local_rank=0,
            stage="stage0",
            gpu_index=0,
            launch_command="python examples/models/train_pipeline.py --rank 0",
            log_path="/var/tmp/shardgrid/jobs/job-0100/logs/gpu4060/rank0.log",
        ),
        WorkerAssignment(
            worker_id=as_worker_id("gpu1060"),
            rank=1,
            local_rank=0,
            stage="stage1",
            gpu_index=0,
            launch_command="python examples/models/train_pipeline.py --rank 1",
            log_path="/var/tmp/shardgrid/jobs/job-0100/logs/gpu1060/rank1.log",
        ),
    ]
    plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("nccl"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=assignments,
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
    failure = _failure("gpu1060", "fixture failure") if state is JobState.FAILED else None
    checkpoint_ref = "checkpoint/model.pt" if state is JobState.COMPLETED else None
    final_metrics = {"final_loss": 0.25} if state is JobState.COMPLETED else {}
    status = JobStatus(
        job_id=job.job_id,
        state=state,
        phase=phase,
        workers=[assignment.worker_id for assignment in assignments],
        assignments=assignments,
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        backend=as_backend_name("nccl"),
        started_at="2026-08-29T12:00:00+00:00",
        loss_history=[1.0] if state is JobState.TRAINING else [],
        latest_loss=1.0 if state is JobState.TRAINING else None,
        final_metrics=final_metrics,
        failure=failure,
        checkpoint_ref=checkpoint_ref,
    )
    return LauncherContext(
        job=job,
        execution_plan=plan,
        cluster_state=cluster_state,
        snapshot=snapshot,
        job_status=status,
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
    )


def _distribution_records(job_id: str) -> dict[tuple[str, str], WorkerDistributionResult]:
    return {
        (job_id, worker_id): WorkerDistributionResult(
            worker_id=worker_id,
            host="10.87.5.155" if worker_id == "gpu4060" else "10.87.5.15",
            transport="scp",
            status=DistributionStatus.PASS,
            remote_snapshot_root=f"/var/tmp/shardgrid/jobs/{job_id}",
            control_checksum="same",
            remote_checksum="same",
            remote_job_id=job_id,
            metadata_ready=True,
        )
        for worker_id in ("gpu4060", "gpu1060")
    }


def _process_record(
    job_id: str,
    worker_id: str,
    rank: int,
    *,
    pid: int,
    stage: str,
    remote_root: str | None = None,
    status: str = "running",
) -> SSHProcessRecord:
    return SSHProcessRecord(
        job_id=job_id,
        worker_id=worker_id,
        rank=rank,
        local_rank=0,
        stage=stage,
        pid=pid,
        command_argv=("python", "train.py"),
        log_path=f"/var/tmp/shardgrid/jobs/{job_id}/logs/{worker_id}/rank{rank}.log",
        launched_at="2026-08-29T12:00:00+00:00",
        remote_host="10.87.5.155" if worker_id == "gpu4060" else "10.87.5.15",
        remote_root=remote_root or f"/var/tmp/shardgrid/jobs/{job_id}",
        status=status,
    )


def _failure(worker_id: str, message: str) -> FailureRecord:
    return FailureRecord(
        stage=FailureStage.TRAIN,
        host="10.87.5.15" if worker_id == "gpu1060" else "10.87.5.155",
        worker_id=as_worker_id(worker_id),
        command=f"monitor {worker_id}",
        exit_code=1,
        runtime_environment={"worker": worker_id},
        python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        message=message,
        recommended_action=f"inspect {worker_id} logs",
    )


def _persisted_status(context: LauncherContext) -> JobStatus:
    path = Path(context.snapshot.root_path) / "job-status.json"
    if not path.is_file():
        return context.job_status
    return JobStatus.from_dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.integration
def test_cleanup_after_completed_job_removes_temp_and_preserves_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, phase="checkpoint", state=JobState.COMPLETED)
    context = replace(
        context,
        job_status=replace(
            context.job_status,
            state=JobState.COMPLETED,
            phase="checkpoint",
            checkpoint_ref="checkpoint/model.pt",
            final_metrics={"final_loss": 0.25},
            latest_loss=0.25,
            loss_history=[1.0, 0.5, 0.25],
        ),
    )
    local_log = Path(context.snapshot.logs_path) / "gpu4060" / "rank0-stage0" / "stdout.log"
    local_log.parent.mkdir(parents=True, exist_ok=True)
    local_log.write_text("local log", encoding="utf-8")
    diagnostics = Path(context.snapshot.root_path) / "job-status.json"
    diagnostics.write_text(json.dumps(context.job_status.to_dict()), encoding="utf-8")
    checkpoint = (
        Path(context.snapshot.checkpoint_path) / "files" / "gpu4060" / "rank0-stage0" / "model.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout("/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"),
                )
            ]
        ),
        "gpu1060": FakeRuntime(
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout("/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"),
                )
            ]
        ),
    }
    ssh = {
        "gpu4060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
        "gpu1060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
        ssh_factory=lambda worker: ssh[str(worker.worker_id)],
    )
    launcher._distribution_records = _distribution_records(str(context.job.job_id))
    monkeypatch.setattr(
        "shardgrid.launchers.ssh._probe_remote_snapshot",
        lambda runtime, remote_root: RemoteSnapshotProbe(
            exists=True,
            empty=False,
            checksum="same",
            job_id="job-0100",
            top_level_entries=(
                "checkpoint",
                "code",
                "config",
                "diagnostics",
                "environment",
                "logs",
                "plan",
            ),
            metadata_ready=True,
        ),
    )

    result = launcher.cleanup(context)
    persisted = _persisted_status(context)
    evidence = json.loads(
        (Path(context.snapshot.diagnostics_path) / "cleanup-gpu4060.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert result.next_job_state is JobState.COMPLETED
    assert persisted.state is JobState.COMPLETED
    assert persisted.checkpoint_ref == "checkpoint/model.pt"
    assert local_log.is_file() is True
    assert checkpoint.is_file() is True
    assert Path(context.snapshot.root_path).is_dir() is True
    assert evidence["removed_temp_paths"] == [
        "/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"
    ]
    assert evidence["preserved_paths"][0].endswith("/var/tmp/shardgrid/jobs/job-0100")


@pytest.mark.integration
def test_cleanup_after_failed_job_preserves_failure_and_unrelated_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, phase="training", state=JobState.FAILED)
    context = replace(
        context,
        job_status=replace(
            context.job_status,
            state=JobState.FAILED,
            phase="training",
            failure=_failure("gpu1060", "rank 1 crashed"),
        ),
    )
    runtimes = {
        "gpu4060": FakeRuntime(
            run_results=[_result("kill -0 4100")],
            script_results=[
                _result("stop", stdout=_stop_stdout(4100, "stopped", ["SIGTERM"], 0.2)),
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout("/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"),
                ),
            ],
        ),
        "gpu1060": FakeRuntime(
            run_results=[_result("kill -0 4200", exit_code=1)],
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout("/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"),
                )
            ],
        ),
    }
    ssh = {
        "gpu4060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
        "gpu1060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
        ssh_factory=lambda worker: ssh[str(worker.worker_id)],
    )
    launcher._distribution_records = _distribution_records(str(context.job.job_id))
    launcher._process_records = {
        ("job-0100", "gpu4060", 0): _process_record(
            "job-0100", "gpu4060", 0, pid=4100, stage="stage0"
        ),
        ("job-0100", "gpu1060", 1): _process_record(
            "job-0100", "gpu1060", 1, pid=4200, stage="stage1", status="failed"
        ),
        ("other-job", "gpu4060", 9): _process_record(
            "other-job", "gpu4060", 9, pid=9999, stage="stage9"
        ),
    }
    monkeypatch.setattr(
        "shardgrid.launchers.ssh._probe_remote_snapshot",
        lambda runtime, remote_root: RemoteSnapshotProbe(
            exists=True,
            empty=False,
            checksum="same",
            job_id="job-0100",
            top_level_entries=(
                "checkpoint",
                "code",
                "config",
                "diagnostics",
                "environment",
                "logs",
                "plan",
            ),
            metadata_ready=True,
        ),
    )

    result = launcher.cleanup(context)
    persisted = _persisted_status(context)

    assert result.status is LauncherResultStatus.SUCCESS
    assert result.next_job_state is JobState.FAILED
    assert persisted.state is JobState.FAILED
    assert persisted.failure is not None
    assert persisted.failure.message == "rank 1 crashed"
    assert launcher._process_records[("job-0100", "gpu4060", 0)].status == "stopped"
    assert launcher._process_records[("other-job", "gpu4060", 9)].pid == 9999


@pytest.mark.integration
def test_cleanup_after_stopped_job_is_idempotent_and_preserves_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, phase="stopped", state=JobState.STOPPED)
    first_runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout("/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"),
                )
            ]
        ),
        "gpu1060": FakeRuntime(
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout("/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"),
                )
            ]
        ),
    }
    second_runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout(
                        "/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100",
                        final_state="missing",
                        removed=[],
                        skipped=["missing"],
                    ),
                )
            ]
        ),
        "gpu1060": FakeRuntime(
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout(
                        "/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100",
                        final_state="missing",
                        removed=[],
                        skipped=["missing"],
                    ),
                )
            ]
        ),
    }
    ssh = {
        "gpu4060": FakeSSH(
            run_results=[
                _result("userprofile", stdout="C:\\Users\\shardgrid"),
                _result("userprofile", stdout="C:\\Users\\shardgrid"),
            ]
        ),
        "gpu1060": FakeSSH(
            run_results=[
                _result("userprofile", stdout="C:\\Users\\shardgrid"),
                _result("userprofile", stdout="C:\\Users\\shardgrid"),
            ]
        ),
    }
    runtimes = first_runtimes
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
        ssh_factory=lambda worker: ssh[str(worker.worker_id)],
    )
    launcher._distribution_records = _distribution_records(str(context.job.job_id))
    monkeypatch.setattr(
        "shardgrid.launchers.ssh._probe_remote_snapshot",
        lambda runtime, remote_root: RemoteSnapshotProbe(
            exists=True,
            empty=False,
            checksum="same",
            job_id="job-0100",
            top_level_entries=(
                "checkpoint",
                "code",
                "config",
                "diagnostics",
                "environment",
                "logs",
                "plan",
            ),
            metadata_ready=True,
        ),
    )

    first = launcher.cleanup(context)
    runtimes = second_runtimes
    second = launcher.cleanup(context)
    persisted = _persisted_status(context)

    assert first.status is LauncherResultStatus.SUCCESS
    assert second.status is LauncherResultStatus.NOOP
    assert second.next_job_state is JobState.STOPPED
    assert persisted.state is JobState.STOPPED


@pytest.mark.integration
def test_cleanup_after_partial_prepare_removes_prepare_only_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, phase="probing", state=JobState.PROBING)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[
                _result("cleanup", stdout=_cleanup_stdout("/var/tmp/shardgrid/jobs/job-0100")),
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout(
                        "/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100",
                        final_state="missing",
                        removed=[],
                        skipped=["missing"],
                    ),
                ),
            ]
        ),
        "gpu1060": FakeRuntime(
            script_results=[
                _result("cleanup", stdout=_cleanup_stdout("/var/tmp/shardgrid/jobs/job-0100")),
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout(
                        "/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100",
                        final_state="missing",
                        removed=[],
                        skipped=["missing"],
                    ),
                ),
            ]
        ),
    }
    ssh = {
        "gpu4060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
        "gpu1060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
        ssh_factory=lambda worker: ssh[str(worker.worker_id)],
    )
    monkeypatch.setattr(
        "shardgrid.launchers.ssh._probe_remote_snapshot",
        lambda runtime, remote_root: RemoteSnapshotProbe(
            exists=True,
            empty=False,
            checksum=None,
            job_id=None,
            top_level_entries=("checkpoint", "diagnostics", "logs"),
            metadata_ready=False,
        ),
    )

    result = launcher.cleanup(context)
    evidence = json.loads(
        (Path(context.snapshot.diagnostics_path) / "cleanup-gpu4060.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert result.next_job_state is JobState.PROBING
    assert "/var/tmp/shardgrid/jobs/job-0100" in evidence["removed_temp_paths"]
    assert evidence["preserved_paths"] == []


@pytest.mark.integration
def test_cleanup_partial_failure_preserves_worker_success_and_reports_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, phase="stopped", state=JobState.STOPPED)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout("/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"),
                )
            ]
        ),
        "gpu1060": FakeRuntime(script_results=[]),
    }
    ssh = {
        "gpu4060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
        "gpu1060": FakeSSH(run_error=RuntimeError("permission denied")),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
        ssh_factory=lambda worker: ssh[str(worker.worker_id)],
    )
    launcher._distribution_records = _distribution_records(str(context.job.job_id))
    monkeypatch.setattr(
        "shardgrid.launchers.ssh._probe_remote_snapshot",
        lambda runtime, remote_root: RemoteSnapshotProbe(
            exists=True,
            empty=False,
            checksum="same",
            job_id="job-0100",
            top_level_entries=(
                "checkpoint",
                "code",
                "config",
                "diagnostics",
                "environment",
                "logs",
                "plan",
            ),
            metadata_ready=True,
        ),
    )

    result = launcher.cleanup(context)
    worker_ok = next(item for item in result.worker_results if item.worker_id == "gpu4060")
    worker_fail = next(item for item in result.worker_results if item.worker_id == "gpu1060")

    assert result.status is LauncherResultStatus.PARTIAL
    assert worker_ok.status is LauncherResultStatus.SUCCESS
    assert worker_fail.status is LauncherResultStatus.FAILED
    assert worker_fail.failure is not None
    assert worker_fail.failure.stage is FailureStage.CLEANUP


@pytest.mark.integration
def test_cleanup_rejects_escaped_temp_path_and_preserves_remote_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, phase="stopped", state=JobState.STOPPED)
    runtimes = {
        "gpu4060": FakeRuntime(script_results=[]),
        "gpu1060": FakeRuntime(
            script_results=[
                _result(
                    "cleanup",
                    stdout=_cleanup_stdout("/mnt/c/Users/shardgrid/.shardgrid/snapshots/job-0100"),
                )
            ]
        ),
    }
    ssh = {
        "gpu4060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
        "gpu1060": FakeSSH(run_results=[_result("userprofile", stdout="C:\\Users\\shardgrid")]),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
        ssh_factory=lambda worker: ssh[str(worker.worker_id)],
    )
    launcher._distribution_records = {
        ("job-0100", "gpu4060"): WorkerDistributionResult(
            worker_id="gpu4060",
            host="10.87.5.155",
            transport="scp",
            status=DistributionStatus.PASS,
            remote_snapshot_root="/var/tmp/shardgrid/jobs/../escape",
            control_checksum="same",
            remote_checksum="same",
            remote_job_id="job-0100",
            metadata_ready=True,
        ),
        ("job-0100", "gpu1060"): _distribution_records("job-0100")[("job-0100", "gpu1060")],
    }
    monkeypatch.setattr(
        "shardgrid.launchers.ssh._probe_remote_snapshot",
        lambda runtime, remote_root: RemoteSnapshotProbe(
            exists=True,
            empty=False,
            checksum="same",
            job_id="job-0100",
            top_level_entries=(
                "checkpoint",
                "code",
                "config",
                "diagnostics",
                "environment",
                "logs",
                "plan",
            ),
            metadata_ready=True,
        ),
    )

    result = launcher.cleanup(context)
    worker_fail = next(item for item in result.worker_results if item.worker_id == "gpu4060")

    assert result.status is LauncherResultStatus.PARTIAL
    assert worker_fail.status is LauncherResultStatus.FAILED
    assert worker_fail.failure is not None
    assert "escaped" in worker_fail.failure.message
