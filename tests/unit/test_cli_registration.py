from __future__ import annotations

from typing import Any

from shardgrid.cli.app import main


def test_registered_commands_appear_in_help(capsys: Any) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "train" in captured.out
    assert "workers" in captured.out


def test_placeholder_command_help_uses_expected_name(capsys: Any) -> None:
    try:
        main(["doctor", "--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "usage: shardgrid doctor" in captured.out


def test_train_help_uses_expected_name(capsys: Any) -> None:
    try:
        main(["train", "--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "usage: shardgrid train" in captured.out
    assert "config_path" in captured.out


def test_train_without_config_path_is_usage_error(capsys: Any) -> None:
    exit_code = main(["train"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "the following arguments are required: config_path" in captured.err


def test_train_is_bound_to_real_handler() -> None:
    from shardgrid.cli.app import build_parser
    from shardgrid.cli.commands.train import run_train_command

    parser = build_parser()
    namespace = parser.parse_args(["train", "examples/train-minimal.yaml"])

    assert namespace.handler is run_train_command
