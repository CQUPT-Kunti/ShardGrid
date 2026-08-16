from __future__ import annotations

import json

from shardgrid.common.enums import FailureStage
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
