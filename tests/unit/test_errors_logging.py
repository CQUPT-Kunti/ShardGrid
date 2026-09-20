from __future__ import annotations

import json

from shardgrid.common.enums import FailureCode, FailureStage
from shardgrid.common.errors import (
    StageError,
    failure_from_process_result,
    make_failure_record,
    raise_stage_error,
)
from shardgrid.common.logging import (
    build_json_log,
    format_failure_diagnostics,
    format_json_log,
    redact_mapping,
)
from shardgrid.common.process import run_process

STAGES = [
    FailureStage.BOOTSTRAP,
    FailureStage.PROBE,
    FailureStage.NETWORK,
    FailureStage.RENDEZVOUS,
    FailureStage.TRAIN,
    FailureStage.CHECKPOINT,
]


def test_make_failure_record_covers_required_fields_and_redaction() -> None:
    failure = make_failure_record(
        stage=FailureStage.BOOTSTRAP,
        host="machine-a",
        command=["ssh", "user:token@example"],
        message="bootstrap failed",
        recommended_action="check ssh and dependencies",
        secrets=["token"],
    )

    assert failure.stage is FailureStage.BOOTSTRAP
    assert failure.host == "machine-a"
    assert failure.command is not None
    assert "token" not in failure.command
    assert failure.recommended_action == "check ssh and dependencies"



def test_failure_from_process_result_records_command_exit_code_and_paths() -> None:
    result = run_process(
        ["python3", "-c", "import sys; print('oops'); print('err', file=sys.stderr); sys.exit(7)"]
    )
    failure = failure_from_process_result(
        stage=FailureStage.PROBE,
        host="machine-c",
        result=result,
        message="probe command failed",
        recommended_action="inspect probe output",
        stdout_path="logs/probe.stdout",
        stderr_path="logs/probe.stderr",
    )

    assert failure.stage is FailureStage.PROBE
    assert failure.command == result.recorded_command
    assert failure.exit_code == 7
    assert failure.stdout_path == "logs/probe.stdout"
    assert failure.stderr_path == "logs/probe.stderr"



def test_stage_error_is_stage_aware() -> None:
    try:
        raise_stage_error(
            stage=FailureStage.NETWORK,
            host="machine-d",
            message="tcp reachability failed",
            recommended_action="verify firewall and routing",
        )
    except StageError as exc:
        assert exc.failure.stage is FailureStage.NETWORK
        assert "NETWORK" in str(exc)
    else:
        raise AssertionError("StageError was not raised")



def test_json_logging_and_mapping_redaction() -> None:
    failure = make_failure_record(
        stage=FailureStage.RENDEZVOUS,
        host="machine-c",
        message="rendezvous failed",
        recommended_action="inspect distributed init logs",
    )
    payload = build_json_log(
        event="job.failure",
        host="machine-c",
        stage=FailureStage.RENDEZVOUS.value,
        message="rendezvous failed",
        failure=failure,
        command=["torchrun", "--rdzv-endpoint", "10.0.0.1:29500", "--token", "secret"],
        extra={"token": "secret", "note": "keep"},
        secrets=["secret"],
    )
    rendered = format_json_log(
        event="job.failure",
        host="machine-c",
        stage=FailureStage.RENDEZVOUS.value,
        message="rendezvous failed",
        failure=failure,
        command=["torchrun", "--token", "secret"],
        extra={"token": "secret"},
        secrets=["secret"],
    )

    assert payload["failure"]["stage"] == FailureStage.RENDEZVOUS.value
    assert "secret" not in json.dumps(payload, sort_keys=True)
    assert "***" in rendered
    assert redact_mapping({"a": "secret"}, ["secret"])["a"] == "***"



def test_human_readable_diagnostics_include_expected_fields() -> None:
    failure = make_failure_record(
        stage=FailureStage.TRAIN,
        host="machine-c",
        message="loss became NaN",
        recommended_action="inspect model inputs and gradients",
        command="python train.py",
        exit_code=2,
        stdout_path="logs/train.stdout",
        stderr_path="logs/train.stderr",
        manual_action_required=True,
    )
    rendered = format_failure_diagnostics(failure)

    assert "stage: TRAIN" in rendered
    assert "host: machine-c" in rendered
    assert "recommended_action: inspect model inputs and gradients" in rendered
    assert "manual_action_required: true" in rendered



def test_required_failure_stages_are_supported() -> None:
    failures = [
        make_failure_record(
            stage=stage,
            host="machine-a",
            message=f"{stage.value} failed",
            recommended_action="inspect logs",
        )
        for stage in STAGES
    ]

    assert [failure.stage for failure in failures] == STAGES


def test_make_failure_record_persists_structured_metadata() -> None:
    failure = make_failure_record(
        stage=FailureStage.TRAIN,
        code=FailureCode.RUNTIME_FAILURE,
        producer="ssh_launcher",
        host="machine-c",
        rank=2,
        gpu_id="gpu1060",
        worker_id="gpu1060",
        message="worker training process failed",
        recommended_action="inspect worker logs and rerun",
        log_refs=["logs/train.rank2.stdout"],
        artifact_refs=["checkpoint/model-state.pt"],
        retryable=True,
    )

    assert failure.code is FailureCode.RUNTIME_FAILURE
    assert failure.producer == "ssh_launcher"
    assert failure.rank == 2
    assert failure.gpu_id == "gpu1060"
    assert failure.log_refs == ("logs/train.rank2.stdout",)
    assert failure.artifact_refs == ("checkpoint/model-state.pt",)
    assert failure.retryable is True


def test_make_failure_record_redacts_secrets_in_log_refs() -> None:
    failure = make_failure_record(
        stage=FailureStage.NETWORK,
        host="machine-c",
        message="ssh failed",
        recommended_action="verify connectivity",
        log_refs=["logs/ssh.token@example.stdout"],
        artifact_refs=["s3://bucket/secret-key/model-state.pt"],
        secrets=["token@example", "secret-key"],
    )

    assert "token@example" not in "".join(failure.log_refs)
    assert "secret-key" not in "".join(failure.artifact_refs)


def test_retryable_is_explicit_not_derived_from_message() -> None:
    retryable = make_failure_record(
        stage=FailureStage.PROBE,
        host="machine-c",
        message="probe out of memory for candidate",
        recommended_action="try next candidate",
        code=FailureCode.MEMORY_REJECT,
        retryable=True,
    )
    not_retryable = make_failure_record(
        stage=FailureStage.TRAIN,
        host="machine-c",
        message="formal training out of memory",
        recommended_action="reduce batch size or adjust placement",
        code=FailureCode.FORMAL_TRAINING_OOM,
        retryable=False,
    )

    assert retryable.retryable is True
    assert not_retryable.retryable is False
