"""CLI command registration helpers."""

from __future__ import annotations

import argparse
from typing import Any

from shardgrid.cli.context import EXIT_RUNTIME_ERROR

PLACEHOLDER_COMMANDS = (
    "workers",
    "probe",
    "network-test",
    "dist-test",
    "train",
    "status",
    "logs",
    "stop",
)


def register_placeholder_commands(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    for name in PLACEHOLDER_COMMANDS:
        parser = subparsers.add_parser(name, help=f"Placeholder for {name}")
        parser.set_defaults(handler=_placeholder_handler, command_name=name)


def _placeholder_handler(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        print(
            '{"ok": false, "command": "'
            + str(args.command_name)
            + '", "message": "placeholder command; implementation pending"}'
        )
    else:
        print(f"{args.command_name}: placeholder command; implementation pending")
    return EXIT_RUNTIME_ERROR
