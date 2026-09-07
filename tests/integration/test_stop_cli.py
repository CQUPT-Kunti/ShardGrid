from __future__ import annotations

import json
from pathlib import Path

from shardgrid.cli.app import main
from shardgrid.cli.commands import stop as stop_command
from shardgrid.common.enums import FailureStage, JobState
from shardgrid.common.errors import make_failure_record
from shardgrid.common.models import as_backend_name, as_engine_name, as_job_id, as_worker_id
from shardgrid.control.status_store import StatusStore
from shardgrid.jobs.models import JobStatus, TrainingJob
from shardgrid.launchers.base import (
    LauncherOperation,
    LauncherResult,
    LauncherResultStatus,
    RankResult,
    WorkerResult,
)
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment


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


def _assignment(worker_id: str, rank: int, stage: str, pid: int) -> WorkerAssignment:
    return WorkerAssignment(
        worker_id=as_worker_id(worker_id),
        rank=rank,
        local_rank=0,
        stage=stage,
        gpu_index=0,
        launch_command=f"python examples/models/train_pipeline.py --rank {rank}",
        log_path=f"/var/tmp/shardgrid/jobs/job-0104/logs/{worker_id}/rank{rank}.log",
        status="running",
        pid=pid,
    )


def _assignments() -> list[WorkerAssignment]:
    return [
        _assignment("gpu4060", 0, "stage0", 4100),
        _assignment("gpu1060", 1, "stage1", 4200),
    ]


def _write_job(
    tmp_path: Path,
    *,
    job_id: str = "job-0104",
    state: JobState = JobState.TRAINING,
    phase: str = "training",
) -> Path:
    root = tmp_path / "jobs" / job_id
    for rel in ("code", "config", "plan", "logs", "checkpoint", "environment", "diagnostics"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    job = TrainingJob(
        job_id=as_job_id(job_id),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    assignments = _assignments()
    plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("nccl"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=assignments,
        snapshot_ref=str(root),
    )
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
        checkpoint_ref="checkpoint/final.pt" if state is JobState.COMPLETED else None,
        final_metrics={"final_loss": 0.25} if state is JobState.COMPLETED else {},
        failure=(
            make_failure_record(
                stage=FailureStage.TRAIN,
                host="10.87.5.15",
                worker_id="gpu1060",
                message="rank 1 failed",
                recommended_action="inspect rank logs",
            )
            if state is JobState.FAILED
            else None
        ),
    )
    (root / "plan" / "execution-plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    StatusStore(tmp_path / "jobs").save_path(root / "job-status.json", status)
    return root


def _stop_result(
    *,
    job_id: str,
    status: LauncherResultStatus,
    next_state: JobState,
    failure_stage: FailureStage | None = None,
) -> LauncherResult:
    success_or_noop = (
        LauncherResultStatus.NOOP
        if status is LauncherResultStatus.NOOP
        else LauncherResultStatus.SUCCESS
    )
    failure = (
        None
        if failure_stage is None
        else make_failure_record(
            stage=failure_stage,
            host="10.87.5.15",
            worker_id="gpu1060",
            message="stop failed",
            recommended_action="inspect stop evidence",
        )
    )
    return LauncherResult(
        operation=LauncherOperation.STOP,
        status=status,
        backend="ssh",
        job_id=job_id,
        worker_results=(
            WorkerResult(
                worker_id="gpu4060",
                status=success_or_noop,
                rank_results=(
                    RankResult(
                        rank=0,
                        worker_id="gpu4060",
                        stage="stage0",
                        pid=4100,
                        status=success_or_noop,
                        message=(
                            "already terminal"
                            if status is LauncherResultStatus.NOOP
                            else "stopped"
                        ),
                    ),
                ),
            ),
            WorkerResult(
                worker_id="gpu1060",
                status=(
                    LauncherResultStatus.SUCCESS
                    if status is LauncherResultStatus.SUCCESS
                    else status
                ),
                rank_results=(
                    RankResult(
                        rank=1,
                        worker_id="gpu1060",
                        stage="stage1",
                        pid=4200,
                        status=(
                            LauncherResultStatus.SUCCESS
                            if status is LauncherResultStatus.SUCCESS
                            else status
                        ),
                        failure=failure,
                        message="stop failed" if failure is not None else "stopped",
                    ),
                ),
                failure=failure,
            ),
        ),
        failure=failure,
        next_job_state=next_state,
    )


def test_stop_is_registered_as_real_command(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "stop" in captured.out
    assert "Placeholder for stop" not in captured.out


def test_stop_requires_jobs_root_or_config(capsys) -> None:
    exit_code = main(["stop", "job-0104", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "stop requires --config or --jobs-root" in captured.out


def test_stop_requires_confirmation_in_noninteractive_mode(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)
    _write_job(tmp_path)

    exit_code = main(["--config", str(config), "stop", "job-0104"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "use --yes to confirm stop in non-interactive mode" in captured.out


def test_stop_cancelled_by_user_does_not_call_launcher(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_job(tmp_path)
    called = {"stop": 0}

    monkeypatch.setattr(stop_command.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(stop_command.builtins, "input", lambda prompt="": "no")
    monkeypatch.setattr(
        stop_command,
        "_execute_stop",
        lambda *args, **kwargs: called.__setitem__("stop", called["stop"] + 1),
    )

    exit_code = main(["--config", str(config), "stop", "job-0104"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cancelled" in captured.out.lower()
    assert called["stop"] == 0


def test_stop_not_found_returns_nonzero(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)

    exit_code = main(["--config", str(config), "stop", "job-missing", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "job status not found" in captured.out


def test_stop_running_job_returns_structured_human_output_and_updates_status(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    root = _write_job(tmp_path)
    store = StatusStore(tmp_path / "jobs")
    store.reserve_resources("job-0104", _assignments())
    final_status = JobStatus(
        job_id=as_job_id("job-0104"),
        state=JobState.STOPPED,
        phase="stopped",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        assignments=_assignments(),
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
        backend=as_backend_name("nccl"),
        started_at="2026-08-29T12:00:00+00:00",
    )

    monkeypatch.setattr(
        stop_command,
        "_execute_stop",
        lambda context, status: (
            StatusStore(tmp_path / "jobs").save_path(
                root / "job-status.json",
                final_status,
            ),
            _stop_result(
                job_id=str(status.job_id),
                status=LauncherResultStatus.SUCCESS,
                next_state=JobState.STOPPED,
            ),
        )[1],
    )

    exit_code = main(["--config", str(config), "stop", "job-0104", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Job: job-0104" in captured.out
    assert "Previous State: TRAINING" in captured.out
    assert "Final State: STOPPED" in captured.out
    assert "Overall Stop Result: SUCCESS" in captured.out
    assert "Preserved Artifacts: yes" in captured.out
    assert "worker=gpu4060 rank=0 stage=stage0 pid=4100 status=SUCCESS" in captured.out
    assert "worker=gpu1060 rank=1 stage=stage1 pid=4200 status=SUCCESS" in captured.out
    persisted = StatusStore(tmp_path / "jobs").load_path(
        root / "job-status.json"
    )
    assert persisted.state is JobState.STOPPED
    assert persisted.phase == "stopped"
    assert persisted.finished_at is not None
    assert store.active_reservations() == []


def test_stop_noop_releases_only_target_job_reservation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_job(tmp_path, state=JobState.STOPPED, phase="stopped")
    store = StatusStore(tmp_path / "jobs")
    store.reserve_resources("job-0104", [_assignment("gpu4060", 0, "stage0", 4100)])
    store.reserve_resources("job-0200", [_assignment("gpu4060", 0, "stage0", 5100)])

    monkeypatch.setattr(
        stop_command,
        "_execute_stop",
        lambda context, status: _stop_result(
            job_id=str(status.job_id),
            status=LauncherResultStatus.NOOP,
            next_state=JobState.STOPPED,
        ),
    )

    exit_code = main(["--config", str(config), "stop", "job-0104", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Overall Stop Result: NOOP" in captured.out
    reservations = store.active_reservations()
    assert reservations == [
        {
            "job_id": "job-0200",
            "worker_id": "gpu4060",
            "rank": 0,
            "stage": "stage0",
            "gpu_index": 0,
            "estimated_peak_training_memory": None,
            "created_at": reservations[0]["created_at"],
        }
    ]


def test_stop_partial_does_not_release_reservations_when_job_not_stopped(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_job(tmp_path, state=JobState.TRAINING, phase="training")
    store = StatusStore(tmp_path / "jobs")
    store.reserve_resources("job-0104", _assignments())

    monkeypatch.setattr(
        stop_command,
        "_execute_stop",
        lambda context, status: _stop_result(
            job_id=str(status.job_id),
            status=LauncherResultStatus.PARTIAL,
            next_state=JobState.FAILED,
            failure_stage=FailureStage.STOP,
        ),
    )

    exit_code = main(["--config", str(config), "stop", "job-0104", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Overall Stop Result: PARTIAL" in captured.out
    assert all(item["job_id"] == "job-0104" for item in store.active_reservations())


def test_stop_completed_job_is_noop_and_preserves_completed_state(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_job(tmp_path, state=JobState.COMPLETED, phase="checkpoint")

    monkeypatch.setattr(
        stop_command,
        "_execute_stop",
        lambda context, status: _stop_result(
            job_id=str(status.job_id),
            status=LauncherResultStatus.NOOP,
            next_state=JobState.COMPLETED,
        ),
    )

    exit_code = main(["--config", str(config), "stop", "job-0104", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Previous State: COMPLETED" in captured.out
    assert "Final State: COMPLETED" in captured.out
    assert "Overall Stop Result: NOOP" in captured.out


def test_stop_json_output_includes_failure_and_per_rank_results(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_job(tmp_path, state=JobState.FAILED, phase="training")

    monkeypatch.setattr(
        stop_command,
        "_execute_stop",
        lambda context, status: _stop_result(
            job_id=str(status.job_id),
            status=LauncherResultStatus.PARTIAL,
            next_state=JobState.FAILED,
            failure_stage=FailureStage.STOP,
        ),
    )

    exit_code = main(["--json", "--config", str(config), "stop", "job-0104", "--yes"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["job_id"] == "job-0104"
    assert payload["previous_state"] == "failed"
    assert payload["final_state"] == "failed"
    assert payload["status"] == "partial"
    assert payload["preserved_artifacts"] is True
    assert payload["worker_results"][1]["rank_results"][0]["rank"] == 1
    assert payload["worker_results"][1]["failure"]["stage"] == "STOP"
