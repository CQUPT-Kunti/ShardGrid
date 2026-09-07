from __future__ import annotations

from pathlib import Path

from shardgrid.cli.app import main
from shardgrid.cli.commands import train as train_command
from shardgrid.common.enums import FailureStage, JobState
from shardgrid.common.errors import make_failure_record
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.control.job_manager import JobRunResult
from shardgrid.jobs.models import JobSnapshot, JobStatus, TrainingJob


def _write_cluster_config(root: Path) -> Path:
    examples = root / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    path = examples / "workers.yaml"
    path.write_text(
        f"""
control:
  machine_id: machine-a
  hostname: control-a.local
jobs_root: {(root / "jobs").resolve()}
ssh: {{}}
runtime:
  conda_environment: shardgrid
  conda_prefix: /opt/conda/envs/shardgrid
network: {{}}
backend_preference: {{}}
manual_override: {{}}
workers:
  - id: gpu4060
    machine_id: machine-c
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.0.0.10
    ssh_user: shardgrid
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_training_config(root: Path) -> Path:
    examples = root / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    path = examples / "train-minimal.yaml"
    path.write_text(
        """
job:
  name: train-minimal
  backend: ssh
  communication_backend: nccl
model:
  name: tiny-sequential
  type: minimal_sequential
resources:
  world_size: 1
artifacts:
  transport: auto
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _result(state: JobState, root: Path) -> JobRunResult:
    snapshot_root = root / "jobs" / "job-20260829-abc12345"
    snapshot = JobSnapshot(
        job_id=as_job_id("job-20260829-abc12345"),
        root_path=str(snapshot_root),
        code_path=str(snapshot_root / "code"),
        config_path=str(snapshot_root / "config"),
        plan_path=str(snapshot_root / "plan"),
        logs_path=str(snapshot_root / "logs"),
        environment_path=str(snapshot_root / "environment"),
        checkpoint_path=str(snapshot_root / "checkpoint"),
        diagnostics_path=str(snapshot_root / "diagnostics"),
    )
    job = TrainingJob(
        job_id=as_job_id("job-20260829-abc12345"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=1,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    status = JobStatus(
        job_id=job.job_id,
        state=state,
        phase="checkpoint" if state is JobState.COMPLETED else "training",
        backend=as_backend_name("nccl"),
        final_metrics={"final_loss": 0.25} if state is JobState.COMPLETED else {},
        checkpoint_ref="checkpoint/model.pt" if state is JobState.COMPLETED else None,
        failure=(
            None
            if state is not JobState.FAILED
            else make_failure_record(
                stage=FailureStage.TRAIN,
                host="10.0.0.10",
                message="training failed",
                recommended_action="inspect rank logs",
            )
        ),
    )
    return JobRunResult(job=job, status=status, snapshot=snapshot)


def test_train_uses_default_examples_cluster_config(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_cluster_config(tmp_path)
    _write_training_config(tmp_path)
    captured: dict[str, object] = {}

    class FakeManager:
        def __init__(self, config) -> None:
            captured["jobs_root"] = str(config.jobs_root)

        def run(self, config_path: str, *, dry_run: bool = False) -> JobRunResult:
            captured["dry_run"] = dry_run
            captured["config_path"] = config_path
            return _result(JobState.COMPLETED, tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_command, "JobManager", FakeManager)

    exit_code = main(["train", "examples/train-minimal.yaml"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured["dry_run"] is False
    assert captured["config_path"] == "examples/train-minimal.yaml"
    assert captured["jobs_root"] == str((tmp_path / "jobs").resolve())
    assert "Job: job-20260829-abc12345" in output
    assert "Backend: nccl" in output
    assert "State: COMPLETED" in output


def test_train_returns_nonzero_for_failed_job(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_cluster_config(tmp_path)
    _write_training_config(tmp_path)

    class FakeManager:
        def __init__(self, config) -> None:
            del config

        def run(self, config_path: str, *, dry_run: bool = False) -> JobRunResult:
            assert dry_run is False
            assert config_path == "examples/train-minimal.yaml"
            return _result(JobState.FAILED, tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_command, "JobManager", FakeManager)

    exit_code = main(["train", "examples/train-minimal.yaml"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "State: FAILED" in output
