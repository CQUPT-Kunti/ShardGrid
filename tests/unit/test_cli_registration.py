from __future__ import annotations

import argparse
from typing import Any

from shardgrid.cli.app import main


def _subcommand_choices(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    raise AssertionError("parser has no subcommands")


def test_registered_commands_appear_in_help(capsys: Any) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "run" in captured.out
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


def test_run_is_bound_to_real_handler() -> None:
    from shardgrid.cli.app import build_parser
    from shardgrid.cli.commands.run import run_run_command

    parser = build_parser()
    namespace = parser.parse_args(["run", "train.py", "--config", "model.yaml"])

    assert namespace.handler is run_run_command
    assert namespace.entrypoint == "train.py"
    assert namespace.entrypoint_args == ["--config", "model.yaml"]


def test_build_parser_wires_train_registration(monkeypatch: Any) -> None:
    from shardgrid.cli import app

    calls = 0

    def fake_register_train_command(subparsers: argparse._SubParsersAction[Any]) -> None:
        nonlocal calls
        calls += 1
        subparsers.add_parser("train-probe")

    monkeypatch.setattr(app, "register_train_command", fake_register_train_command)

    parser = app.build_parser()

    assert calls == 1
    assert "train-probe" in _subcommand_choices(parser)


def test_run_command_is_registered() -> None:
    from shardgrid.cli.app import build_parser

    choices = _subcommand_choices(build_parser())

    assert "train" in choices
    assert "run" in choices
