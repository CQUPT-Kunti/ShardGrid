"""Structured stage-aware errors for ShardGrid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shardgrid.common.enums import FailureStage
from shardgrid.common.models import as_worker_id
from shardgrid.common.process import ProcessResult, redact_command, redact_text
from shardgrid.jobs.models import FailureRecord


@dataclass(frozen=True)
class StageError(RuntimeError):
    failure: FailureRecord

    def __str__(self) -> str:
        return f"{self.failure.stage.value} on {self.failure.host}: {self.failure.message}"



def make_failure_record(
    *,
    stage: FailureStage,
    host: str,
    message: str,
    recommended_action: str,
    worker_id: str | None = None,
    command: str | Sequence[str] | None = None,
    exit_code: int | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
    runtime_environment: dict[str, str] | None = None,
    python_executable: str | None = None,
    conda_environment: str | None = None,
    conda_prefix: str | None = None,
    retryable: bool = False,
    manual_action_required: bool = False,
    secrets: Sequence[str] = (),
) -> FailureRecord:
    rendered_command = None
    if command is not None:
        rendered_command = redact_command(command, secrets)
    redacted_runtime = {
        str(key): redact_text(str(value), secrets) or str(value)
        for key, value in ({} if runtime_environment is None else dict(runtime_environment)).items()
    }

    return FailureRecord(
        stage=stage,
        host=host,
        worker_id=None if worker_id is None else as_worker_id(worker_id),
        command=rendered_command,
        exit_code=exit_code,
        stdout_path=redact_text(stdout_path, secrets),
        stderr_path=redact_text(stderr_path, secrets),
        runtime_environment=redacted_runtime,
        python_executable=redact_text(python_executable, secrets),
        conda_environment=redact_text(conda_environment, secrets),
        conda_prefix=redact_text(conda_prefix, secrets),
        message=redact_text(message, secrets) or message,
        recommended_action=redact_text(recommended_action, secrets) or recommended_action,
        retryable=retryable,
        manual_action_required=manual_action_required,
    )



def failure_from_process_result(
    *,
    stage: FailureStage,
    host: str,
    result: ProcessResult,
    message: str,
    recommended_action: str,
    worker_id: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
    python_executable: str | None = None,
    conda_environment: str | None = None,
    conda_prefix: str | None = None,
    retryable: bool = False,
    manual_action_required: bool = False,
) -> FailureRecord:
    return FailureRecord(
        stage=stage,
        host=host,
        worker_id=None if worker_id is None else as_worker_id(worker_id),
        command=result.recorded_command,
        exit_code=result.exit_code,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        runtime_environment=dict(result.runtime_environment),
        python_executable=python_executable,
        conda_environment=conda_environment,
        conda_prefix=conda_prefix,
        message=message,
        recommended_action=recommended_action,
        retryable=retryable,
        manual_action_required=manual_action_required,
    )

def raise_stage_error(
    *,
    stage: FailureStage,
    host: str,
    message: str,
    recommended_action: str,
    worker_id: str | None = None,
    command: str | Sequence[str] | None = None,
    exit_code: int | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
    runtime_environment: dict[str, str] | None = None,
    python_executable: str | None = None,
    conda_environment: str | None = None,
    conda_prefix: str | None = None,
    retryable: bool = False,
    manual_action_required: bool = False,
    secrets: Sequence[str] = (),
) -> None:
    raise StageError(
        make_failure_record(
            stage=stage,
            host=host,
            message=message,
            recommended_action=recommended_action,
            worker_id=worker_id,
            command=command,
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            runtime_environment=runtime_environment,
            python_executable=python_executable,
            conda_environment=conda_environment,
            conda_prefix=conda_prefix,
            retryable=retryable,
            manual_action_required=manual_action_required,
            secrets=secrets,
        )
    )
