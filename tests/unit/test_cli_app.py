from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shardgrid.cli.app import main


def test_cli_help(capsys: Any) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "ShardGrid control-plane CLI" in captured.out
    assert "--config" in captured.out


def test_cli_unknown_command_returns_usage_error(capsys: Any) -> None:
    exit_code = main(["bogus-command"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid choice" in captured.err


def test_cli_invalid_global_option_returns_usage_error(capsys: Any) -> None:
    exit_code = main(["--not-a-real-option"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unrecognized arguments" in captured.err


def test_cli_json_error_behavior_for_missing_config(capsys: Any, tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    exit_code = main(["--json", "--config", str(missing)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 3
    assert payload["error"] == "FileNotFoundError"
    assert str(missing) in payload["message"]


def test_cli_json_success_output() -> None:
    assert main(["--json"]) == 0
