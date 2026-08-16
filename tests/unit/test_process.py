from __future__ import annotations

import sys

import pytest

from shardgrid.common.process import ProcessTimeoutError, redact_command, run_process


def test_run_process_success() -> None:
    result = run_process([sys.executable, "-c", "print('ok')"])

    assert result.ok is True
    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"
    assert result.stderr == ""
    assert result.timed_out is False


def test_run_process_non_zero_exit() -> None:
    result = run_process([sys.executable, "-c", "import sys; sys.exit(3)"])

    assert result.ok is False
    assert result.exit_code == 3
    assert result.timed_out is False


def test_run_process_timeout() -> None:
    result = run_process(
        [sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.1
    )

    assert result.ok is False
    assert result.timed_out is True
    assert result.exit_code == -1


def test_run_process_timeout_with_check_raises() -> None:
    with pytest.raises(ProcessTimeoutError):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(1)"],
            timeout=0.1,
            check=True,
        )


def test_run_process_environment_propagation() -> None:
    result = run_process(
        [sys.executable, "-c", "import os; print(os.environ['SG_TEST_ENV'])"],
        env={"SG_TEST_ENV": "present"},
        runtime_environment={"manager": "conda", "name": "shardgrid-dev"},
    )

    assert result.stdout.strip() == "present"
    assert result.runtime_environment["manager"] == "conda"
    assert result.runtime_environment["name"] == "shardgrid-dev"


def test_run_process_output_capture_and_encoding() -> None:
    result = run_process(
        [sys.executable, "-c", "import sys; print('cafe'); print('err', file=sys.stderr)"]
    )

    assert "cafe" in result.stdout
    assert "err" in result.stderr


def test_redact_command_masks_secrets() -> None:
    rendered = redact_command(["ssh", "user:token@example"], secrets=["token"])

    assert "token" not in rendered
    assert "***" in rendered
