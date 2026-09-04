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
from shardgrid.common.enums import Health, JobState, PhysicalOS, RuntimeOS
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


def _status_failure(message: str) -> object:
    from shardgrid.common.enums import FailureStage
    from shardgrid.common.errors import make_failure_record

    return make_failure_record(
        stage=FailureStage.LAUNCH,
        host="10.87.5.155",
        worker_id="gpu4060",
        message=message,
        recommended_action="inspect failed rank state",
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
    phase: str = "launch",
    state: JobState = JobState.LAUNCHING,
) -> LauncherContext:
    snapshot_root = tmp_path / "job-0099"
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
                source_worker_id=source.worker_id,
                target_worker_id=target.worker_id,
                source_ip=source.ip or "",
                target_ip=target.ip or "",
                interface="eth0",
                tcp_reachable=True,
                measured_at="2026-08-28T11:00:00+00:00",
            )
            for source in workers
            for target in workers
            if source.worker_id != target.worker_id
        ],
        created_at="2026-08-28T11:00:00+00:00",
    )
    cluster_state = ResourceManager().build_cluster_state(
        workers,
        network_state=network,
        require_network=True,
        now=datetime(2026, 8, 28, 11, 0, tzinfo=UTC),
    )
    job = TrainingJob(
        job_id=as_job_id("job-0099"),
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
            log_path="/var/tmp/shardgrid/jobs/job-0099/logs/gpu4060/rank0.log",
        ),
        WorkerAssignment(
            worker_id=as_worker_id("gpu1060"),
            rank=1,
            local_rank=0,
            stage="stage1",
            gpu_index=0,
            launch_command="python examples/models/train_pipeline.py --rank 1",
            log_path="/var/tmp/shardgrid/jobs/job-0099/logs/gpu1060/rank1.log",
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
    return LauncherContext(
        job=job,
        execution_plan=plan,
        cluster_state=cluster_state,
        snapshot=snapshot,
        job_status=JobStatus(
            job_id=job.job_id,
            state=state,
            phase=phase,
            workers=[assignment.worker_id for assignment in assignments],
            assignments=assignments,
            runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
            backend=as_backend_name("nccl"),
            started_at="2026-08-28T12:00:00+00:00",
            loss_history=[1.0] if state is JobState.TRAINING else [],
            latest_loss=1.0 if state is JobState.TRAINING else None,
        ),
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


def _launch(launcher: SSHLauncher, context: LauncherContext) -> None:
    launcher._distribution_records = _distribution_records(str(context.job.job_id))
    result = launcher.launch(context)
    assert result.status is LauncherResultStatus.SUCCESS


def _train_log(checkpoint_path: str, *, final_loss: float = 0.25) -> str:
    train = {
        "loss_history": [1.0, 0.5, final_loss],
        "initial_loss": 1.0,
        "final_loss": final_loss,
        "checkpoint_path": checkpoint_path,
        "checkpoint_roundtrip_ok": True,
    }
    return (
        'STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage0"}\n'
        + "T074_TRAIN_EVIDENCE "
        + json.dumps(train, sort_keys=True)
    )


def _persisted_status(context) -> JobStatus:
    return JobStatus.from_dict(
        json.loads((Path(context.snapshot.root_path) / "job-status.json").read_text())
    )


@pytest.mark.integration
def test_stop_terminates_running_ranks_and_persists_stopped_status(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    checkpoint0 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank0.pt")
    checkpoint1 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank1.pt")
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[
                _result("launch", stdout="4100"),
                _result(
                    "stop",
                    stdout=_stop_stdout(4100, "stopped", ["SIGTERM"], 0.3),
                ),
            ],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout=_train_log(checkpoint0, final_loss=0.4)),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[
                _result("launch", stdout="4200"),
                _result(
                    "stop",
                    stdout=_stop_stdout(4200, "stopped", ["SIGTERM", "SIGKILL"], 0.8),
                ),
            ],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_train_log(checkpoint1, final_loss=0.25)),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.stop(context)
    persisted = _persisted_status(context)
    evidence0 = json.loads(
        (Path(context.snapshot.diagnostics_path) / "stop-gpu4060-rank0.json").read_text()
    )
    evidence1 = json.loads(
        (Path(context.snapshot.diagnostics_path) / "stop-gpu1060-rank1.json").read_text()
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert result.next_job_state is JobState.STOPPED
    assert persisted.state is JobState.STOPPED
    assert persisted.phase == "stopped"
    assert persisted.failure is None
    assert all(item.status == "stopped" for item in launcher.process_records())
    assert evidence0["final_state"] == "stopped"
    assert evidence0["action"] == "signal"
    assert evidence0["escalation_level"] == 1
    assert evidence1["final_state"] == "stopped"
    assert evidence1["escalation_level"] == 2


@pytest.mark.integration
def test_stop_preserves_completed_job_without_killing_dead_ranks(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    checkpoint0 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank0.pt")
    checkpoint1 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank1.pt")
    context = replace(
        context,
        job_status=replace(
            context.job_status,
            state=JobState.COMPLETED,
            phase="checkpoint",
            checkpoint_ref="checkpoint/rank1.pt",
            final_metrics={"final_loss": 0.25},
            latest_loss=0.25,
            loss_history=[1.0, 0.5, 0.25],
        ),
    )
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100", exit_code=1),
                _result("tail", stdout=_train_log(checkpoint0, final_loss=0.4)),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200", exit_code=1),
                _result("tail", stdout=_train_log(checkpoint1, final_loss=0.25)),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.stop(context)
    persisted = _persisted_status(context)
    evidence0 = json.loads(
        (Path(context.snapshot.diagnostics_path) / "stop-gpu4060-rank0.json").read_text()
    )

    assert result.status is LauncherResultStatus.NOOP
    assert result.next_job_state is JobState.COMPLETED
    assert persisted.state is JobState.COMPLETED
    assert evidence0["final_state"] == "completed"
    assert evidence0["action"] == "preserved"
    assert len(runtimes["gpu4060"].script_calls) == 1
    assert len(runtimes["gpu1060"].script_calls) == 1


@pytest.mark.integration
def test_stop_timeout_keeps_surviving_rank_failed_and_never_completed(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    checkpoint0 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank0.pt")
    checkpoint1 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank1.pt")
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[
                _result("launch", stdout="4100"),
                _result(
                    "stop",
                    stdout=_stop_stdout(4100, "stopped", ["SIGTERM"], 0.2),
                ),
            ],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout=_train_log(checkpoint0, final_loss=0.4)),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[
                _result("launch", stdout="4200"),
                _result(
                    "stop",
                    stdout=_stop_stdout(4200, "running", ["SIGTERM", "SIGKILL"], 15.0),
                ),
            ],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_train_log(checkpoint1, final_loss=0.25)),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.stop(context)
    persisted = _persisted_status(context)
    evidence1 = json.loads(
        (Path(context.snapshot.diagnostics_path) / "stop-gpu1060-rank1.json").read_text()
    )

    assert result.status is LauncherResultStatus.PARTIAL
    assert result.next_job_state is JobState.FAILED
    assert persisted.state is JobState.FAILED
    assert persisted.failure is not None
    assert persisted.phase == "stopping"
    assert persisted.checkpoint_ref is None
    assert evidence1["final_state"] == "running"
    assert evidence1["escalation_level"] == 2
    assert evidence1["failure"]["stage"] == "STOP"


@pytest.mark.integration
def test_stop_blocks_when_liveness_probe_is_transport_failure(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    checkpoint1 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank1.pt")
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100", stderr="ssh transport failed", exit_code=255),
                _result("tail", stdout="TRAIN_STEP_BEGIN\n"),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[
                _result("launch", stdout="4200"),
                _result(
                    "stop",
                    stdout=_stop_stdout(4200, "stopped", ["SIGTERM"], 0.2),
                ),
            ],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_train_log(checkpoint1, final_loss=0.25)),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.stop(context)
    persisted = _persisted_status(context)
    evidence0 = json.loads(
        (Path(context.snapshot.diagnostics_path) / "stop-gpu4060-rank0.json").read_text()
    )

    assert result.status is LauncherResultStatus.PARTIAL
    assert persisted.state is JobState.FAILED
    assert persisted.phase == "stopping"
    assert evidence0["final_state"] == "unknown"
    assert evidence0["action"] == "probe_failed"
    assert evidence0["process_probe"]["transport_status"] == "failed"
    assert "could not confirm remote pid 4100 liveness" in evidence0["message"]
    assert len(runtimes["gpu4060"].script_calls) == 1


@pytest.mark.integration
def test_stop_reuses_persisted_failed_job_pids_without_in_memory_launch_records(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, phase="launch", state=JobState.LAUNCHING)
    context = replace(
        context,
        job_status=replace(
            context.job_status,
            state=JobState.FAILED,
            phase="launch",
            failure=_status_failure("launch failed"),
            assignments=[
                replace(
                    context.job_status.assignments[0],
                    pid=36265,
                    status="running",
                    log_path="/var/tmp/shardgrid/jobs/job-0099/logs/gpu4060/rank0.log",
                ),
                replace(
                    context.job_status.assignments[1],
                    pid=26062,
                    status="running",
                    log_path="/var/tmp/shardgrid/jobs/job-0099/logs/gpu1060/rank1.log",
                ),
            ],
        ),
    )
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[
                _result(
                    "stop",
                    stdout=_stop_stdout(36265, "stopped", ["SIGTERM"], 0.3),
                ),
            ],
            run_results=[
                _result("kill -0 36265"),
                _result("tail", stdout="RUN_INFO\n"),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[
                _result(
                    "stop",
                    stdout=_stop_stdout(26062, "stopped", ["SIGTERM", "SIGKILL"], 0.9),
                ),
            ],
            run_results=[
                _result("kill -0 26062"),
                _result("tail", stdout="RUN_INFO\n"),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )

    result = launcher.stop(context)
    persisted = _persisted_status(context)

    assert result.status is LauncherResultStatus.SUCCESS
    assert result.next_job_state is JobState.FAILED
    assert persisted.state is JobState.FAILED
    assert persisted.phase == "launch"
    assert persisted.failure == context.job_status.failure
    assert all(item.status == "stopped" for item in launcher.process_records())
