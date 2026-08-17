"""`shardgrid doctor` CLI command."""

from __future__ import annotations

import argparse
import json
from typing import Any

from shardgrid.control.doctor import ControlDoctorReport, run_control_doctor


def register_doctor_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser(
        "doctor", help="Run ShardGrid doctor checks on the control node"
    )
    parser.add_argument(
        "--target",
        choices=("control",),
        default="control",
        help="Doctor target (T032 supports control only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_doctor_command, command_name="doctor")


def _human_report(report: ControlDoctorReport) -> str:
    lines = [
        "ShardGrid doctor",
        f"target: {report.target}",
        f"host: {report.host} | os: {report.os_version} | timestamp: {report.timestamp}",
        f"conda: {report.environment.get('conda_executable') or 'not_found'} "
        f"envs: {', '.join(report.environment.get('conda_environments') or []) or 'none'} "
        f"active: {report.environment.get('active_environment') or 'none'}",
        f"selected environment: {report.environment.get('selected_environment') or 'none'}",
        f"python: {report.environment.get('python_executable') or 'not_found'} "
        f"({report.environment.get('python_version') or 'unknown'})",
    ]
    for check in report.checks:
        marker = {
            "ok": "OK",
            "degraded": "DEGRADED",
            "fail": "FAIL",
            "not_checked": "NOT CHECKED",
        }.get(check.status, check.status)
        lines.append(f"  [{marker}] {check.name}: {check.detail or ''}")
    lines.append(f"health: {report.health.value}")
    if report.manual_actions:
        lines.append("manual actions:")
        lines.extend(f"  - {action}" for action in report.manual_actions)
    return "\n".join(lines)


def run_doctor_command(args: argparse.Namespace) -> int:
    context = getattr(args, "context", None)
    config = getattr(context, "config", None)
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(context, "json_output", False)
    )
    report = run_control_doctor(config)
    if json_output:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(_human_report(report))
    return report.exit_code