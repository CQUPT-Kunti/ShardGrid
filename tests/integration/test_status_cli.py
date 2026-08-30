from __future__ import annotations

import json
from pathlib import Path

from shardgrid.cli.app import main
from shardgrid.cli.commands import status as status_command
from shardgrid.common.enums import FailureStage, JobState
from shardgrid.common.errors import make_failure_record
from shardgrid.common.models import as_backend_name, as_job_id, as_worker_id
from shardgrid.control.status_store import StatusStore
from shardgrid.jobs.models import JobStatus, TrainingJob
from shardgrid.planner.models import WorkerAssignment


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "workers.yaml"
    path.write_text(
        f"""
control:
  machine_id: machine-a
  hostname: control-a.local
jobs_root: {tmp_path / "jobs"}
ssh: {{}}
runtime:
  conda_environment: shardgrid
  conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
  default_wsl_distro: Ubuntu-22.04
network: {{}}
backend_preference: {{}}
manual_override:
  preferred_workers: []
  disabled_workers: []
  worker_address_overrides: {{}}
  rendezvous_port: null
workers:
  - id: gpu4060
    machine_id: machine-c
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.87.5.155
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
  - id: gpu1060
    machine_id: machine-d
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.87.5.15
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
""".strip(),
        encoding="utf-8",
    )
    return path


def _job(job_id: str) -> TrainingJob:
    return TrainingJob(
        job_id=as_job_id(job_id),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
        created_at="2026-08-29T12:00:00+00:00",
        updated_at="2026-08-29T12:00:00+00:00",
    )


def _assignments() -> list[WorkerAssignment]:
    return [
        WorkerAssignment(
            worker_id=as_worker_id("gpu4060"),
            rank=0,
            local_rank=0,
            stage="stage0",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            launch_command="python examples/models/train_pipeline.py --rank 0",
            log_path="logs/gpu4060/rank0-stage0/combined.log",
            status="running",
            pid=4100,
        ),
        WorkerAssignment(
            worker_id=as_worker_id("gpu1060"),
            rank=1,
            local_rank=0,
            stage="stage1",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            launch_command="python examples/models/train_pipeline.py --rank 1",
            log_path="logs/gpu1060/rank1-stage1/combined.log",
            status="running",
            pid=4101,
        ),
    ]


def _persist_status(
    tmp_path: Path,
    *,
    job_id: str,
    state: JobState,
    phase: str,
    loss_history: list[float] | None = None,
    latest_loss: float | None = None,
    final_metrics: dict[str, float] | None = None,
    checkpoint_ref: str | None = None,
    fallback_used: bool = False,
    failure_stage: FailureStage | None = None,
) -> JobStatus:
    store = StatusStore(tmp_path / "jobs")
    job = _job(job_id)
    store.create_initial_status(job)
    status = JobStatus(
        job_id=job.job_id,
        state=state,
        phase=phase,
        workers=[assignment.worker_id for assignment in _assignments()],
        assignments=_assignments(),
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        loss_history=list(loss_history or []),
        latest_loss=latest_loss,
        final_metrics=dict(final_metrics or {}),
        backend=as_backend_name("nccl"),
        fallback_used=fallback_used,
        started_at="2026-08-29T12:00:00+00:00",
        checkpoint_ref=checkpoint_ref,
        failure=(
            None
            if failure_stage is None
            else make_failure_record(
                stage=failure_stage,
                host="10.87.5.15",
                worker_id="gpu1060",
                command="torchrun --node_rank=1 train.py",
                exit_code=1,
                runtime_environment={"worker": "gpu1060", "runtime": "wsl2"},
                python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
                conda_environment="shardgrid",
                conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
                message=f"{phase} failed",
                recommended_action="inspect rank logs",
            )
        ),
    )
    return store.save(status)


def test_status_is_registered_as_real_command(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "status" in captured.out
    assert "Placeholder for status" not in captured.out


def test_status_requires_jobs_root_or_config(capsys) -> None:
    exit_code = main(["status", "job-0102"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "status requires --config or --jobs-root" in captured.out


def test_status_reports_not_found_with_nonzero_exit(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)

    exit_code = main(["--config", str(config), "status", "job-missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "job status not found" in captured.out


def test_status_renders_running_job_without_ssh_refresh(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)
    _persist_status(
        tmp_path,
        job_id="job-0102-running",
        state=JobState.TRAINING,
        phase="training",
        loss_history=[1.2, 0.8, 0.5],
        latest_loss=0.5,
    )

    exit_code = main(["--config", str(config), "status", "job-0102-running"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Job: job-0102-running" in captured.out
    assert "State: TRAINING" in captured.out
    assert "Phase: training" in captured.out
    assert "Backend: nccl" in captured.out
    assert "Fallback: false" in captured.out
    assert "Latest Loss: 0.5" in captured.out
    assert "Loss History: 3 points" in captured.out
    assert "gpu4060 | rank=0 | stage=stage0 | runtime=env:gpu4060/shardgrid" in captured.out
    assert "gpu1060 | rank=1 | stage=stage1 | runtime=env:gpu1060/shardgrid" in captured.out
    assert "Checkpoint: pending" in captured.out


def test_status_renders_completed_job_json(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)
    _persist_status(
        tmp_path,
        job_id="job-0102-completed",
        state=JobState.COMPLETED,
        phase="checkpoint",
        loss_history=[1.0, 0.5, 0.25],
        latest_loss=0.25,
        final_metrics={"final_loss": 0.25},
        checkpoint_ref="checkpoint/rank1.pt",
    )

    exit_code = main(
        ["--config", str(config), "--json", "status", "job-0102-completed"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["job_id"] == "job-0102-completed"
    assert payload["state"] == "completed"
    assert payload["phase"] == "checkpoint"
    assert payload["checkpoint_ref"] == "checkpoint/rank1.pt"
    assert payload["final_metrics"]["final_loss"] == 0.25
    assert payload["assignments"][1]["stage"] == "stage1"


def test_status_renders_failed_job_with_failure_record(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)
    _persist_status(
        tmp_path,
        job_id="job-0102-failed",
        state=JobState.FAILED,
        phase="training",
        fallback_used=True,
        failure_stage=FailureStage.TRAIN,
    )

    exit_code = main(["--config", str(config), "status", "job-0102-failed"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "State: FAILED" in captured.out
    assert "Fallback: true" in captured.out
    assert "Failure Stage: TRAIN" in captured.out
    assert "Failure Worker: gpu1060" in captured.out
    assert "Failure Exit Code: 1" in captured.out
    assert "Failure Runtime: {\"runtime\": \"wsl2\", \"worker\": \"gpu1060\"}" in captured.out
    assert "Recommended Action: inspect rank logs" in captured.out


def test_status_renders_stopped_job(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)
    _persist_status(
        tmp_path,
        job_id="job-0102-stopped",
        state=JobState.STOPPED,
        phase="stopped",
    )

    exit_code = main(["--config", str(config), "status", "job-0102-stopped"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "State: STOPPED" in captured.out
    assert "Latest Loss: pending" in captured.out


def test_status_watch_reloads_until_terminal_state(tmp_path: Path, monkeypatch, capsys) -> None:
    config = _config_path(tmp_path)
    job_id = "job-0102-watch"
    _persist_status(
        tmp_path,
        job_id=job_id,
        state=JobState.TRAINING,
        phase="training",
        loss_history=[1.0],
        latest_loss=1.0,
    )
    terminal = _persist_status(
        tmp_path,
        job_id=job_id,
        state=JobState.COMPLETED,
        phase="checkpoint",
        loss_history=[1.0, 0.5],
        latest_loss=0.5,
        final_metrics={"final_loss": 0.5},
        checkpoint_ref="checkpoint/final.pt",
    )
    first = _persist_status(
        tmp_path,
        job_id=job_id,
        state=JobState.TRAINING,
        phase="training",
        loss_history=[1.0],
        latest_loss=1.0,
    )

    loads = {"count": 0}
    real_load = status_command.StatusStore.load

    def fake_load(self, loaded_job_id):
        loads["count"] += 1
        if loads["count"] == 1:
            return first
        return terminal if str(loaded_job_id) == job_id else real_load(self, loaded_job_id)

    sleep_calls: list[float] = []

    monkeypatch.setattr(status_command.StatusStore, "load", fake_load)
    monkeypatch.setattr(status_command.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    exit_code = main(["--config", str(config), "status", job_id, "--watch"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert loads["count"] >= 2
    assert sleep_calls == [1.0]
    assert captured.out.count("Job: job-0102-watch") == 2
    assert "State: COMPLETED" in captured.out
