from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
        self.run_kwargs: list[dict[str, object]] = []

    def run(self, command, **kwargs) -> ProcessResult:
        self.run_calls.append(command)
        self.run_kwargs.append(dict(kwargs))
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
    timed_out: bool = False,
) -> ProcessResult:
    return ProcessResult(
        args=command,
        recorded_command=command,
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
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


def _context(
    tmp_path: Path,
    *,
    phase: str = "launch",
    state: JobState = JobState.LAUNCHING,
) -> LauncherContext:
    snapshot_root = tmp_path / "job-0097"
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
        job_id=as_job_id("job-0097"),
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
            log_path="/var/tmp/shardgrid/jobs/job-0097/logs/gpu4060/rank0.log",
        ),
        WorkerAssignment(
            worker_id=as_worker_id("gpu1060"),
            rank=1,
            local_rank=0,
            stage="stage1",
            gpu_index=0,
            launch_command="python examples/models/train_pipeline.py --rank 1",
            log_path="/var/tmp/shardgrid/jobs/job-0097/logs/gpu1060/rank1.log",
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


def _plain_training_log(stage_id: str) -> str:
    if stage_id == "stage0":
        return "\n".join(
            [
                "TRAIN_STEP_BEGIN",
                "[RANK=0][HOST=ldj][TIME=21:54:55.419][MARKER=TRAIN_STEP_BEGIN]",
                "STAGE0_FORWARD_BEGIN",
                "[RANK=0][HOST=ldj][TIME=21:54:55.452][MARKER=STAGE0_FORWARD_BEGIN]",
                "STAGE0_FORWARD_END",
                "ACTIVATION_SEND_BEGIN",
                "ACTIVATION_SEND_END",
                "GRADIENT_RECV_BEGIN",
                "GRADIENT_RECV_END",
                "STAGE0_BACKWARD_BEGIN",
                "STAGE0_BACKWARD_END",
                "OPTIMIZER_STEP_END",
                "TRAIN_STEP_END",
            ]
        )
    return "\n".join(
        [
            "TRAIN_STEP_BEGIN",
            "[RANK=1][HOST=LAPTOP-5G3QUOGM][TIME=21:54:57.835][MARKER=TRAIN_STEP_BEGIN]",
            "ACTIVATION_RECV_BEGIN",
            "ACTIVATION_RECV_END",
            "STAGE1_FORWARD_BEGIN",
            "STAGE1_FORWARD_END",
            "LOSS_READY",
            "STAGE1_BACKWARD_BEGIN",
            "STAGE1_BACKWARD_END",
            "GRADIENT_SEND_BEGIN",
            "GRADIENT_SEND_END",
            "OPTIMIZER_STEP_END",
            "TRAIN_STEP_END",
        ]
    )


@pytest.mark.integration
def test_monitor_persists_completed_status_and_deduplicates_loss_history(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    checkpoint0 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank0.pt")
    checkpoint1 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank1.pt")
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

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert result.next_job_state is JobState.CHECKPOINTING
    assert persisted.state is JobState.CHECKPOINTING
    assert persisted.phase == "checkpoint"
    assert persisted.loss_history == [1.0, 0.5, 0.25]
    assert persisted.latest_loss == 0.25
    assert persisted.final_metrics["final_loss"] == 0.25
    assert persisted.checkpoint_ref is None


@pytest.mark.integration
def test_monitor_accepts_rank_local_checkpoint_evidence_without_rank0_final_loss(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    checkpoint0 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank0.pt")
    checkpoint1 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank1.pt")
    rank0_train = {
        "loss_history": [],
        "initial_loss": None,
        "final_loss": None,
        "checkpoint_path": checkpoint0,
        "checkpoint_roundtrip_ok": True,
        "param_update_ok": True,
        "optimizer_restore_ok": True,
        "param_restore_ok": True,
        "step_restore_ok": True,
    }
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100", exit_code=1),
                _result(
                    "tail",
                    stdout="T074_TRAIN_EVIDENCE " + json.dumps(rank0_train, sort_keys=True),
                ),
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

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert result.next_job_state is JobState.CHECKPOINTING
    assert persisted.state is JobState.CHECKPOINTING
    assert persisted.final_metrics["final_loss"] == 0.25
    assert persisted.checkpoint_ref is None


@pytest.mark.integration
def test_monitor_reports_running_rendezvous_without_faking_training(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="launch", state=JobState.LAUNCHING)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage0"}\n'),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage1"}\n'),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert persisted.state is JobState.RENDEZVOUS
    assert persisted.phase == "rendezvous"
    assert persisted.latest_loss is None


@pytest.mark.integration
def test_monitor_tracks_training_progress_and_is_idempotent(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    progress_log = (
        'STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage0"}\n'
        'T072_FORWARD_EVIDENCE {"loss": 1.0}\n'
        'T074_TRAIN_EVIDENCE {"loss_history": [1.0, 0.5], "final_loss": 0.5}\n'
    )
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout=progress_log),
                _result("kill -0 4100"),
                _result("tail", stdout=progress_log),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage1"}\n'),
                _result("kill -0 4200"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage1"}\n'),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    first = launcher.monitor(context)
    second = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert first.status is LauncherResultStatus.SUCCESS
    assert second.status is LauncherResultStatus.SUCCESS
    assert persisted.state is JobState.TRAINING
    assert persisted.loss_history == [1.0, 0.5]
    assert persisted.latest_loss == 0.5


@pytest.mark.integration
def test_monitor_uses_configured_remote_command_timeout(tmp_path: Path) -> None:
    base = _cluster_config()
    config = replace(
        base,
        ssh=replace(base.ssh, command_timeout_seconds=123),
    )
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout=_plain_training_log("stage0")),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_plain_training_log("stage1")),
            ],
        ),
    }
    launcher = SSHLauncher(
        config,
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)

    assert result.status is LauncherResultStatus.SUCCESS
    assert [call["timeout"] for call in runtimes["gpu4060"].run_kwargs] == [123.0, 123.0]
    assert [call["timeout"] for call in runtimes["gpu1060"].run_kwargs] == [123.0, 123.0]


@pytest.mark.integration
def test_monitor_keeps_alive_rank_running_when_log_read_times_out(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    prior = {
        "phase": "training",
        "rendezvous_ready": True,
        "training_started": True,
        "last_progress": "T072_FORWARD_EVIDENCE",
        "loss_history": [1.0],
        "latest_loss": 1.0,
    }
    (Path(context.snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").write_text(
        json.dumps(prior),
        encoding="utf-8",
    )
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stderr="timed out", exit_code=-1, timed_out=True),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_plain_training_log("stage1")),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )
    monitor_payload = json.loads(
        (Path(context.snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").read_text()
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert persisted.state is JobState.TRAINING
    assert persisted.failure is None
    assert monitor_payload["process_state"] == "alive"
    assert monitor_payload["log_state"] == "transient_timeout"
    assert monitor_payload["running"] is True
    assert monitor_payload["training_started"] is True
    assert monitor_payload["last_progress"] == "T072_FORWARD_EVIDENCE"
    assert "keeping last known progress" in monitor_payload["message"]


@pytest.mark.integration
def test_monitor_keeps_unknown_process_state_nonterminal_when_liveness_probe_times_out(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    prior = {
        "phase": "training",
        "rendezvous_ready": True,
        "training_started": True,
        "last_progress": "TRAIN_STEP_END",
        "loss_history": [1.0, 0.5],
        "latest_loss": 0.5,
    }
    (Path(context.snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").write_text(
        json.dumps(prior),
        encoding="utf-8",
    )
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100", stderr="timed out", exit_code=-1, timed_out=True),
                _result("tail", stderr="timed out", exit_code=-1, timed_out=True),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_plain_training_log("stage1")),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )
    monitor_payload = json.loads(
        (Path(context.snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").read_text()
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert persisted.state is JobState.TRAINING
    assert persisted.failure is None
    assert monitor_payload["process_state"] == "unknown"
    assert monitor_payload["log_state"] == "transient_timeout"
    assert monitor_payload["running"] is False
    assert monitor_payload["process_exit_known"] is False
    assert monitor_payload["last_progress"] == "TRAIN_STEP_END"
    assert "keeping last known progress" in monitor_payload["message"]


@pytest.mark.integration
def test_monitor_keeps_transport_probe_failure_as_unknown_not_exited(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100", stderr="ssh transport failed", exit_code=255),
                _result("tail", stdout=_plain_training_log("stage0")),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_plain_training_log("stage1")),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )
    monitor_payload = json.loads(
        (Path(context.snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").read_text()
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert persisted.state is JobState.TRAINING
    assert persisted.failure is None
    assert monitor_payload["status"] == "unknown"
    assert monitor_payload["process_state"] == "unknown"
    assert monitor_payload["process_exit_known"] is False
    assert monitor_payload["process_probe"]["transport_status"] == "failed"
    assert monitor_payload["process_probe"]["exit_code"] == 255
    assert monitor_payload["last_progress"] == "TRAIN_STEP_END"
    assert "keeping last known progress" in monitor_payload["message"]


@pytest.mark.integration
def test_monitor_marks_rank_crash_failed_without_losing_other_rank(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="rendezvous", state=JobState.RENDEZVOUS)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage0"}\n'),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200", exit_code=1),
                _result(
                    "tail",
                    stdout=(
                        'STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage1"}\n'
                        'T072_FORWARD_EVIDENCE {"loss": 1.0}\n'
                    ),
                ),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.PARTIAL
    assert persisted.state is JobState.FAILED
    assert persisted.failure is not None
    assert persisted.failure.stage is FailureStage.TRAIN
    assert "rank 1 exited" in persisted.failure.message
    assert result.worker_results[0].rank_results[0].pid == 4100


@pytest.mark.integration
def test_monitor_exit_without_checkpoint_does_not_mark_completed(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100", exit_code=1),
                _result(
                    "tail",
                    stdout=(
                        'T074_TRAIN_EVIDENCE '
                        '{"loss_history": [1.0, 0.4], "final_loss": 0.4}\n'
                    ),
                ),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200", exit_code=1),
                _result(
                    "tail",
                    stdout=(
                        'T074_TRAIN_EVIDENCE '
                        '{"loss_history": [1.0, 0.3], "final_loss": 0.3}\n'
                    ),
                ),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.FAILED
    assert persisted.state is JobState.FAILED
    assert persisted.checkpoint_ref is None


@pytest.mark.integration
def test_monitor_fails_rendezvous_timeout_from_config_based_threshold(tmp_path: Path) -> None:
    context = _context(tmp_path)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[_result("kill -0 4100"), _result("tail", stdout="")],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[_result("kill -0 4200"), _result("tail", stdout="")],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)
    for key, record in list(launcher._process_records.items()):
        launcher._process_records[key] = replace(
            record,
            launched_at=(datetime.now(tz=UTC) - timedelta(seconds=90)).isoformat(),
        )

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.FAILED
    assert persisted.state is JobState.FAILED
    assert persisted.failure is not None
    assert persisted.failure.stage is FailureStage.RENDEZVOUS


@pytest.mark.integration
def test_monitor_plain_training_markers_cancel_rendezvous_timeout(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="launch", state=JobState.LAUNCHING)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout=_plain_training_log("stage0")),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_plain_training_log("stage1")),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)
    for key, record in list(launcher._process_records.items()):
        launcher._process_records[key] = replace(
            record,
            launched_at=(datetime.now(tz=UTC) - timedelta(seconds=90)).isoformat(),
        )

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert persisted.state is JobState.TRAINING
    assert persisted.phase == "training"
    assert persisted.failure is None
    assert result.worker_results[0].rank_results[0].message == "running in training"
    assert result.worker_results[1].rank_results[0].message == "running in training"


@pytest.mark.integration
def test_monitor_training_progress_uses_last_progress_timestamp_not_launch_time(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout=_plain_training_log("stage0")),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_plain_training_log("stage1")),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)
    for key, record in list(launcher._process_records.items()):
        launcher._process_records[key] = replace(
            record,
            launched_at=(datetime.now(tz=UTC) - timedelta(seconds=300)).isoformat(),
        )

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.SUCCESS
    assert persisted.state is JobState.TRAINING
    assert persisted.failure is None


@pytest.mark.integration
def test_monitor_fails_training_stall_from_last_progress_timestamp(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    stale_at = (datetime.now(tz=UTC) - timedelta(seconds=180)).isoformat()
    prior = {
        "phase": "training",
        "rendezvous_ready": True,
        "training_started": True,
        "last_progress": "TRAIN_STEP_END",
        "last_update_timestamp": stale_at,
        "loss_history": [1.0, 0.5],
        "latest_loss": 0.5,
        "log_tail": "",
    }
    (Path(context.snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").write_text(
        json.dumps(prior),
        encoding="utf-8",
    )
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stderr="timed out", exit_code=-1, timed_out=True),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_plain_training_log("stage1")),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )
    monitor_payload = json.loads(
        (Path(context.snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").read_text()
    )

    assert result.status is LauncherResultStatus.PARTIAL
    assert persisted.state is JobState.FAILED
    assert persisted.failure is not None
    assert persisted.failure.stage is FailureStage.TRAIN
    assert monitor_payload["progress_changed"] is False
    assert monitor_payload["last_update_timestamp"] == stale_at


@pytest.mark.integration
def test_monitor_records_duplicate_terminal_marker_parse_diagnostic(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="training", state=JobState.TRAINING)
    checkpoint0 = str(Path(context.snapshot.root_path) / "checkpoint" / "rank0.pt")
    good = _train_log(checkpoint0, final_loss=0.4)
    duplicated = good + good.splitlines()[-1]
    duplicated_payload = duplicated.splitlines()[-1][len("T074_TRAIN_EVIDENCE ") :]
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100", exit_code=1),
                _result("tail", stdout=duplicated),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout=_plain_training_log("stage1")),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    launcher.monitor(context)
    monitor_payload = json.loads(
        (Path(context.snapshot.diagnostics_path) / "monitor-gpu4060-rank0.json").read_text()
    )

    assert monitor_payload["train"] is None
    assert len(monitor_payload["marker_parse_errors"]) == 1
    diagnostic = monitor_payload["marker_parse_errors"][0]
    assert diagnostic["worker_id"] == "gpu4060"
    assert diagnostic["rank"] == 0
    assert diagnostic["stage"] == "stage0"
    assert diagnostic["marker"] == "T074_TRAIN_EVIDENCE"
    assert diagnostic["error"] == "extra trailing data after JSON payload"
    assert diagnostic["payload_length"] == len(duplicated_payload)
    assert diagnostic["payload_prefix"] == duplicated_payload[:160]
    assert diagnostic["payload_suffix"] == duplicated_payload[-160:]


@pytest.mark.integration
def test_monitor_reports_lost_connection_without_faking_terminal_failure(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="launch", state=JobState.LAUNCHING)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_error=RuntimeError("ssh link dropped"),
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage1"}\n'),
            ],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    _launch(launcher, context)

    result = launcher.monitor(context)
    persisted = JobStatus.from_dict(
        json.loads((Path(context.snapshot.diagnostics_path) / "job-status.json").read_text())
    )

    assert result.status is LauncherResultStatus.PARTIAL
    assert result.worker_results[0].status is LauncherResultStatus.BLOCKED
    assert persisted.state is JobState.RENDEZVOUS
    assert persisted.failure is None


@pytest.mark.integration
def test_monitor_handles_more_than_two_ranks_without_rank_specific_logic(tmp_path: Path) -> None:
    config = replace(
        _cluster_config(),
        workers=[
            *_cluster_config().workers,
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
    context = _context(tmp_path)
    extra = WorkerAssignment(
        worker_id=as_worker_id("gpu3090"),
        rank=2,
        local_rank=0,
        stage="stage2",
        gpu_index=0,
        launch_command="python examples/models/train_pipeline.py --rank 2",
        log_path="/var/tmp/shardgrid/jobs/job-0097/logs/gpu3090/rank2.log",
    )
    context = replace(
        context,
        execution_plan=replace(
            context.execution_plan,
            world_size=3,
            workers=[*context.execution_plan.workers, extra],
        ),
        job=replace(context.job, requested_world_size=3),
        job_status=replace(
            context.job_status,
            workers=[*context.job_status.workers, as_worker_id("gpu3090")],
            assignments=[*context.job_status.assignments, extra],
            runtime_environment_refs={
                **context.job_status.runtime_environment_refs,
                "2": "env:gpu3090/shardgrid",
            },
        ),
        runtime_environment_refs={**context.runtime_environment_refs, "2": "env:gpu3090/shardgrid"},
    )
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("launch", stdout="4100")],
            run_results=[
                _result("kill -0 4100"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage0"}\n'),
            ],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("launch", stdout="4200")],
            run_results=[
                _result("kill -0 4200"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage1"}\n'),
            ],
        ),
        "gpu3090": FakeRuntime(
            script_results=[_result("launch", stdout="4300")],
            run_results=[
                _result("kill -0 4300"),
                _result("tail", stdout='STAGE_PLACEMENT_EVIDENCE {"stage_id":"stage2"}\n'),
            ],
        ),
    }
    launcher = SSHLauncher(
        config,
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )
    launcher._distribution_records = {
        (str(context.job.job_id), worker_id): WorkerDistributionResult(
            worker_id=worker_id,
            host="10.87.5.99" if worker_id == "gpu3090" else "10.87.5.155",
            transport="scp",
            status=DistributionStatus.PASS,
            remote_snapshot_root=f"/var/tmp/shardgrid/jobs/{context.job.job_id}",
            control_checksum="same",
            remote_checksum="same",
            remote_job_id=str(context.job.job_id),
            metadata_ready=True,
        )
        for worker_id in ("gpu4060", "gpu1060", "gpu3090")
    }
    assert launcher.launch(context).status is LauncherResultStatus.SUCCESS

    result = launcher.monitor(context)

    assert result.status is LauncherResultStatus.SUCCESS
    assert [item.rank_results[0].rank for item in result.worker_results] == [0, 1, 2]
