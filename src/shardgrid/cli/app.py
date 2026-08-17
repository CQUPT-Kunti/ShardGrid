"""ShardGrid CLI root entrypoint."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Never

from shardgrid.cli.commands import register_placeholder_commands
from shardgrid.cli.commands.doctor import register_doctor_command
from shardgrid.cli.context import (
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE,
    format_cli_error,
    resolve_cli_context,
)


class CLIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError(message)


def build_parser() -> CLIArgumentParser:
    parser = CLIArgumentParser(prog="shardgrid", description="ShardGrid control-plane CLI")
    parser.add_argument("--config", help="Path to cluster configuration")
    parser.add_argument("--jobs-root", help="Override jobs root directory")
    parser.add_argument(
        "--conda-env",
        help="Override Conda environment name for Python/runtime commands",
    )
    parser.add_argument(
        "--conda-prefix",
        help="Override Conda environment prefix for Python/runtime commands",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    subparsers = parser.add_subparsers(dest="command")
    register_placeholder_commands(subparsers)
    register_doctor_command(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = list(sys.argv[1:] if argv is None else argv)

    try:
        namespace = parser.parse_args(args)
        context = resolve_cli_context(
            config=namespace.config,
            jobs_root=namespace.jobs_root,
            conda_env=namespace.conda_env,
            conda_prefix=namespace.conda_prefix,
            verbose=namespace.verbose,
            json_output=namespace.json,
        )
        context.load_config()
        namespace.context = context
        handler = getattr(namespace, "handler", None)
    except FileNotFoundError as error:
        json_output = "--json" in args
        print(format_cli_error(error, json_output=json_output), file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as error:
        json_output = "--json" in args
        print(format_cli_error(error, json_output=json_output), file=sys.stderr)
        return EXIT_USAGE if isinstance(error, ValueError) else EXIT_RUNTIME_ERROR

    if handler is not None:
        return handler(namespace)

    if context.json_output:
        print('{"ok": true, "command": "root"}')
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
