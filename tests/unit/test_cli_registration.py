from __future__ import annotations

import json
from typing import Any

from shardgrid.cli.app import main
from shardgrid.cli.commands import PLACEHOLDER_COMMANDS


def test_registered_commands_appear_in_help(capsys: Any) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    for command in PLACEHOLDER_COMMANDS:
        assert command in captured.out


def test_placeholder_command_help_uses_expected_name(capsys: Any) -> None:
    try:
        main(["doctor", "--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "usage: shardgrid doctor" in captured.out


def test_placeholder_command_behavior_is_explicit(capsys: Any) -> None:
    exit_code = main(["workers"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "placeholder command; implementation pending" in captured.out


def test_placeholder_command_json_behavior_is_explicit(capsys: Any) -> None:
    exit_code = main(["--json", "train"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["command"] == "train"
    assert payload["message"] == "placeholder command; implementation pending"
