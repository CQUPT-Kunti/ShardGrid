from __future__ import annotations

import json

from shardgrid.common.enums import FailureStage
from shardgrid.common.errors import make_failure_record
from shardgrid.common.logging import build_json_log, format_failure_diagnostics, redact_mapping
from shardgrid.common.process import redact_command, redact_text

SECRET = "TEST_PASSWORD_DO_NOT_LEAK"
TOKEN = "TEST_TOKEN_DO_NOT_LEAK"


def test_redact_text_and_command_mask_secret_values() -> None:
    rendered = redact_command(
        ["sshpass", "-p", SECRET, "ssh", "user@example", f"echo {TOKEN}"],
        secrets=[SECRET, TOKEN],
    )

    assert SECRET not in rendered
    assert TOKEN not in rendered
    assert "***" in rendered
    assert "sshpass" in rendered
    assert "user@example" in rendered
    assert redact_text(f"path/{SECRET}/file", [SECRET]) == "path/***/file"


def test_make_failure_record_redacts_message_command_paths_and_runtime_env() -> None:
    failure = make_failure_record(
        stage=FailureStage.BOOTSTRAP,
        host="worker-a",
        command=["sshpass", "-p", SECRET, "ssh", "user@example"],
        message=f"bootstrap failed with {SECRET}",
        recommended_action=f"rerun with token {TOKEN}",
        stdout_path=f"/tmp/{SECRET}/stdout.log",
        stderr_path=f"/tmp/{TOKEN}/stderr.log",
        runtime_environment={
            "API_TOKEN": TOKEN,
            "PASSWORD_PATH": f"/secrets/{SECRET}",
        },
        python_executable=f"/opt/{SECRET}/bin/python",
        conda_environment=f"env-{TOKEN}",
        conda_prefix=f"/envs/{SECRET}",
        secrets=[SECRET, TOKEN],
    )

    rendered = json.dumps(failure.to_dict(), sort_keys=True)
    assert SECRET not in rendered
    assert TOKEN not in rendered
    assert failure.command is not None and "sshpass" in failure.command
    assert failure.stdout_path is not None and failure.stdout_path.endswith("/stdout.log")
    assert failure.stderr_path is not None and failure.stderr_path.endswith("/stderr.log")
    assert failure.runtime_environment["API_TOKEN"] == "***"


def test_json_log_redacts_failure_and_preserves_useful_context() -> None:
    failure = make_failure_record(
        stage=FailureStage.PROBE,
        host="worker-b",
        command=["python", "-c", f"print('{SECRET}')"],
        message=f"probe failed: {SECRET}",
        recommended_action=f"check token {TOKEN}",
        secrets=[SECRET, TOKEN],
    )
    payload = build_json_log(
        event="doctor.failure",
        host="worker-b",
        stage=FailureStage.PROBE.value,
        message=f"probe failed: {SECRET}",
        failure=failure,
        command=["ssh", "user@example", f"echo {TOKEN}"],
        extra={
            "credential_env": TOKEN,
            "safe_key": "keep",
            "config_path": f"/configs/{SECRET}/worker.yaml",
        },
        secrets=[SECRET, TOKEN],
    )

    rendered = json.dumps(payload, sort_keys=True)
    assert SECRET not in rendered
    assert TOKEN not in rendered
    assert payload["event"] == "doctor.failure"
    assert payload["host"] == "worker-b"
    assert payload["stage"] == FailureStage.PROBE.value
    assert payload["extra"]["safe_key"] == "keep"
    assert payload["extra"]["config_path"].endswith("/worker.yaml")


def test_failure_diagnostics_redacts_command_and_path_but_keeps_stage_and_host() -> None:
    failure = make_failure_record(
        stage=FailureStage.TRAIN,
        host="worker-c",
        command=f"python train.py --token {TOKEN}",
        message=f"loss blew up after reading {SECRET}",
        recommended_action=f"rotate {TOKEN} and inspect logs",
        stdout_path=f"/tmp/{SECRET}/train.stdout",
        stderr_path=f"/tmp/{TOKEN}/train.stderr",
        exit_code=7,
        manual_action_required=True,
        secrets=[SECRET, TOKEN],
    )

    rendered = format_failure_diagnostics(failure, secrets=[SECRET, TOKEN])
    assert SECRET not in rendered
    assert TOKEN not in rendered
    assert "stage: TRAIN" in rendered
    assert "host: worker-c" in rendered
    assert "exit_code: 7" in rendered
    assert "manual_action_required: true" in rendered


def test_redact_mapping_masks_credential_like_values_without_erasing_keys() -> None:
    payload = redact_mapping(
        {
            "PASSWORD_ENV": SECRET,
            "token_ref": TOKEN,
            "path": f"/secure/{SECRET}/artifact.log",
            "safe": "ok",
        },
        secrets=[SECRET, TOKEN],
    )

    rendered = json.dumps(payload, sort_keys=True)
    assert SECRET not in rendered
    assert TOKEN not in rendered
    assert payload["PASSWORD_ENV"] == "***"
    assert payload["token_ref"] == "***"
    assert payload["path"].endswith("/artifact.log")
    assert payload["safe"] == "ok"
