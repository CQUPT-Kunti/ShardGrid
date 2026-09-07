from __future__ import annotations

from pathlib import Path

import pytest

from shardgrid.cli.app import main
from shardgrid.cli.commands import run as run_command
from shardgrid.cli.commands import train as train_command
from shardgrid.cli.context import format_cli_error
from shardgrid.common.enums import FailureStage, JobState
from shardgrid.common.errors import make_failure_record
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.control.job_manager import JobRunResult
from shardgrid.jobs.models import JobSnapshot, JobStatus, TrainingJob


def _cluster_config(root: Path) -> Path:
    path = root / "workers.yaml"
    path.write_text(
        f"""
control:
  machine_id: control
  hostname: control.local
jobs_root: {(root / "jobs").resolve()}
ssh: {{}}
runtime:
  conda_environment: shardgrid
  conda_prefix: /opt/conda/envs/shardgrid
network: {{}}
backend_preference: {{}}
manual_override: {{}}
workers:
  - id: worker-a
    machine_id: machine-a
    physical_os: linux
    runtime_os: linux
    runtime: linux
    host: 127.0.0.1
    ssh_user: shardgrid
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _training_config(root: Path) -> Path:
    path = root / "train.yaml"
    path.write_text(
        """
job:
  name: train
  backend: ssh
  communication_backend: gloo
model:
  name: tiny
  type: minimal_sequential
resources:
  world_size: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _result(root: Path, state: JobState = JobState.COMPLETED) -> JobRunResult:
    snapshot = JobSnapshot(
        job_id=as_job_id("job-run-contract"),
        root_path=str(root / "jobs" / "job-run-contract"),
        code_path=str(root / "jobs" / "job-run-contract" / "code"),
        config_path=str(root / "jobs" / "job-run-contract" / "config"),
        plan_path=str(root / "jobs" / "job-run-contract" / "plan"),
        logs_path=str(root / "jobs" / "job-run-contract" / "logs"),
        environment_path=str(root / "jobs" / "job-run-contract" / "environment"),
        checkpoint_path=str(root / "jobs" / "job-run-contract" / "checkpoint"),
        diagnostics_path=str(root / "jobs" / "job-run-contract" / "diagnostics"),
    )
    job = TrainingJob(
        job_id=snapshot.job_id,
        config_path=str(root / "train.yaml"),
        model="tiny",
        requested_world_size=1,
        backend_preference=as_backend_name("gloo"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    status = JobStatus(
        job_id=job.job_id,
        state=state,
        phase="checkpoint" if state is JobState.COMPLETED else "capture",
        backend=as_backend_name("gloo"),
        final_metrics={"final_loss": 0.25} if state is JobState.COMPLETED else {},
        checkpoint_ref="checkpoint/model.pt" if state is JobState.COMPLETED else None,
        failure=(
            None
            if state is not JobState.FAILED
            else make_failure_record(
                stage=FailureStage.PLAN,
                host="control.local",
                message="capture failed before mutation",
                recommended_action="inspect diagnostics/capture.json",
                runtime_environment={"artifact_log": "diagnostics/capture.json"},
            )
        ),
    )
    return JobRunResult(job=job, status=status, snapshot=snapshot)


def test_run_preserves_entrypoint_argv_after_shardgrid_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeManager:
        def __init__(self, config) -> None:
            captured["jobs_root"] = str(config.jobs_root)

        def run_entrypoint(
            self,
            entrypoint: run_command.TrainingEntrypoint,
        ) -> JobRunResult:
            captured.update(
                {
                    "entrypoint": str(entrypoint.entrypoint),
                    "entrypoint_args": entrypoint.argv,
                    "dry_run": entrypoint.dry_run,
                    "json_output": entrypoint.json_output,
                    "cluster_config_path": entrypoint.cluster_config_path,
                }
            )
            return _result(tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_command, "JobManager", FakeManager)

    exit_code = main(
        [
            "--config",
            str(_cluster_config(tmp_path)),
            "run",
            "--dry-run",
            "--json",
            "train.py",
            "--config",
            "model.yaml",
            "--epochs",
            "2",
        ]
    )

    assert exit_code == 0
    assert captured["entrypoint"] == "train.py"
    assert captured["entrypoint_args"] == ("--config", "model.yaml", "--epochs", "2")
    assert captured["dry_run"] is True
    assert captured["json_output"] is True
    assert captured["cluster_config_path"] == tmp_path / "workers.yaml"


def test_run_requires_entrypoint_and_returns_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--config", str(_cluster_config(tmp_path)), "run"])
    error = capsys.readouterr().err

    assert exit_code == 2
    assert "entrypoint" in error.lower()


def test_run_returns_nonzero_and_displays_failure_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeManager:
        def __init__(self, config) -> None:
            del config

        def run_entrypoint(self, entrypoint: run_command.TrainingEntrypoint) -> JobRunResult:
            assert str(entrypoint.entrypoint) == "train.py"
            return _result(tmp_path, JobState.FAILED)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_command, "JobManager", FakeManager)

    exit_code = main(["--config", str(_cluster_config(tmp_path)), "run", "train.py"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Stage: PLAN" in output
    assert "capture failed before mutation" in output
    assert "diagnostics/capture.json" in output


def test_failure_display_contract_includes_stage_message_and_artifact_ref() -> None:
    failure = make_failure_record(
        stage=FailureStage.PLAN,
        host="control.local",
        message="capture failed before mutation",
        recommended_action="inspect diagnostics/capture.json",
        runtime_environment={"artifact_log": "diagnostics/capture.json"},
    )

    rendered = format_cli_error(failure.message, json_output=False)

    assert "capture failed before mutation" in rendered
    assert failure.stage is FailureStage.PLAN
    assert failure.runtime_environment["artifact_log"] == "diagnostics/capture.json"


def test_run_human_output_shows_structured_failure_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from shardgrid.common.enums import FailureCode

    class FakeManager:
        def __init__(self, config) -> None:
            del config

        def run_entrypoint(self, entrypoint: run_command.TrainingEntrypoint) -> JobRunResult:
            del entrypoint
            snapshot = JobSnapshot(
                job_id=as_job_id("job-run-structured"),
                root_path=str(tmp_path / "jobs" / "job-run-structured"),
                code_path=str(tmp_path / "jobs" / "job-run-structured" / "code"),
                config_path=str(tmp_path / "jobs" / "job-run-structured" / "config"),
                plan_path=str(tmp_path / "jobs" / "job-run-structured" / "plan"),
                logs_path=str(tmp_path / "jobs" / "job-run-structured" / "logs"),
                environment_path=str(tmp_path / "jobs" / "job-run-structured" / "environment"),
                checkpoint_path=str(tmp_path / "jobs" / "job-run-structured" / "checkpoint"),
                diagnostics_path=str(tmp_path / "jobs" / "job-run-structured" / "diagnostics"),
            )
            job = TrainingJob(
                job_id=snapshot.job_id,
                config_path=str(tmp_path / "train.yaml"),
                model="tiny",
                requested_world_size=1,
                backend_preference=as_backend_name("gloo"),
                runtime_environment_ref="env:cluster/shardgrid",
            )
            status = JobStatus(
                job_id=job.job_id,
                state=JobState.FAILED,
                phase="training",
                backend=as_backend_name("gloo"),
                final_metrics={},
                checkpoint_ref=None,
                failure=make_failure_record(
                    stage=FailureStage.TRAIN,
                    code=FailureCode.RUNTIME_FAILURE,
                    producer="ssh_launcher",
                    host="control.local",
                    rank=1,
                    message="worker training process failed",
                    recommended_action="inspect rank logs and rerun",
                    log_refs=("logs/train.rank1.stdout",),
                    artifact_refs=("checkpoint/model-state.pt",),
                    retryable=True,
                ),
            )
            return JobRunResult(job=job, status=status, snapshot=snapshot)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_command, "JobManager", FakeManager)

    exit_code = main(["--config", str(_cluster_config(tmp_path)), "run", "train.py"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Stage: TRAIN" in output
    assert "Code: RUNTIME_FAILURE" in output
    assert "Producer: ssh_launcher" in output
    assert "Retryable: YES" in output
    assert "Log Ref: logs/train.rank1.stdout" in output
    assert "Artifact Ref: checkpoint/model-state.pt" in output


def test_run_json_output_is_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json as jsonlib

    from shardgrid.common.enums import FailureCode

    class FakeManager:
        def __init__(self, config) -> None:
            del config

        def run_entrypoint(self, entrypoint: run_command.TrainingEntrypoint) -> JobRunResult:
            del entrypoint
            return _result(tmp_path, JobState.FAILED)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_command, "JobManager", FakeManager)

    exit_code = main(
        [
            "--config",
            str(_cluster_config(tmp_path)),
            "run",
            "--json",
            "train.py",
        ]
    )

    payload = jsonlib.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["failure"]["stage"] == "PLAN"
    assert payload["failure"]["code"] is None
    assert payload["failure"]["retryable"] is False
    assert payload["failure"]["producer"] is None
    assert payload["failure"]["log_refs"] == []
    assert payload["failure"]["artifact_refs"] == []


def test_run_json_output_preserves_structured_failure_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json as jsonlib

    from shardgrid.common.enums import FailureCode

    class FakeManager:
        def __init__(self, config) -> None:
            del config

        def run_entrypoint(self, entrypoint: run_command.TrainingEntrypoint) -> JobRunResult:
            del entrypoint
            snapshot = JobSnapshot(
                job_id=as_job_id("job-run-json-code"),
                root_path=str(tmp_path / "jobs" / "job-run-json-code"),
                code_path=str(tmp_path / "jobs" / "job-run-json-code" / "code"),
                config_path=str(tmp_path / "jobs" / "job-run-json-code" / "config"),
                plan_path=str(tmp_path / "jobs" / "job-run-json-code" / "plan"),
                logs_path=str(tmp_path / "jobs" / "job-run-json-code" / "logs"),
                environment_path=str(tmp_path / "jobs" / "job-run-json-code" / "environment"),
                checkpoint_path=str(tmp_path / "jobs" / "job-run-json-code" / "checkpoint"),
                diagnostics_path=str(tmp_path / "jobs" / "job-run-json-code" / "diagnostics"),
            )
            job = TrainingJob(
                job_id=snapshot.job_id,
                config_path=str(tmp_path / "train.yaml"),
                model="tiny",
                requested_world_size=1,
                backend_preference=as_backend_name("gloo"),
                runtime_environment_ref="env:cluster/shardgrid",
            )
            status = JobStatus(
                job_id=job.job_id,
                state=JobState.FAILED,
                phase="checkpoint",
                backend=as_backend_name("gloo"),
                final_metrics={},
                checkpoint_ref=None,
                failure=make_failure_record(
                    stage=FailureStage.CHECKPOINT,
                    code=FailureCode.RUNTIME_FAILURE,
                    producer="job_manager",
                    host="control.local",
                    message="checkpoint merge failed",
                    recommended_action="inspect checkpoint logs",
                    log_refs=("logs/checkpoint.stdout",),
                    artifact_refs=("checkpoint/model-state.pt",),
                    retryable=False,
                ),
            )
            return JobRunResult(job=job, status=status, snapshot=snapshot)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_command, "JobManager", FakeManager)

    exit_code = main(
        [
            "--config",
            str(_cluster_config(tmp_path)),
            "run",
            "--json",
            "train.py",
        ]
    )

    payload = jsonlib.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["failure"]["stage"] == "CHECKPOINT"
    assert payload["failure"]["code"] == "RUNTIME_FAILURE"
    assert payload["failure"]["producer"] == "job_manager"
    assert payload["failure"]["retryable"] is False
    assert payload["failure"]["log_refs"] == ["logs/checkpoint.stdout"]
    assert payload["failure"]["artifact_refs"] == ["checkpoint/model-state.pt"]


def test_existing_train_command_remains_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _cluster_config(tmp_path)
    training_path = _training_config(tmp_path)

    class FakeManager:
        def __init__(self, config) -> None:
            del config

        def run(self, config_path: str, *, dry_run: bool = False) -> JobRunResult:
            assert config_path == str(training_path)
            assert dry_run is False
            return _result(tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_command, "JobManager", FakeManager)

    exit_code = main(["--config", str(tmp_path / "workers.yaml"), "train", str(training_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Job: job-run-contract" in output
    assert "State: COMPLETED" in output
