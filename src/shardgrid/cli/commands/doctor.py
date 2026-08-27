"""`shardgrid doctor` CLI command."""

from __future__ import annotations

import argparse
import json
from typing import Any

from shardgrid.control.doctor import DoctorReport, run_doctor


def register_doctor_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser(
        "doctor", help="Run ShardGrid readiness checks on control and worker targets"
    )
    parser.add_argument(
        "--target",
        choices=("control", "workers", "all"),
        default="control",
        help="Doctor target",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe idempotent fixes only",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_doctor_command, command_name="doctor")


def _human_report(report: DoctorReport) -> str:
    lines = [
        "ShardGrid doctor",
        f"target: {report.target}",
        f"generated: {report.generated_at}",
    ]
    for subject in report.subjects:
        lines.append(
            f"{subject.subject} | host: {subject.host} | runtime: {subject.runtime} | "
            f"health: {subject.health.value}"
        )
        lines.append(
            f"  env: conda={subject.environment.get('conda_executable') or 'not_found'} "
            f"selected={subject.environment.get('selected_environment') or 'none'} "
            f"prefix={subject.environment.get('conda_prefix') or 'none'}"
        )
        lines.append(
            f"  python: {subject.environment.get('python_executable') or 'not_found'} "
            f"({subject.environment.get('python_version') or 'unknown'})"
        )
        for check in subject.checks:
            detail = check.detected_value
            if isinstance(detail, (dict, list)):
                detail = json.dumps(detail, sort_keys=True)
            if detail is None:
                detail = check.failure_reason or ""
            lines.append(f"  [{check.status}] {check.layer}.{check.name}: {detail}")
        if subject.manual_actions:
            lines.append("  manual actions:")
            lines.extend(f"    - {action}" for action in subject.manual_actions)
    lines.append(f"overall health: {report.health.value}")
    return "\n".join(lines)


def run_doctor_command(args: argparse.Namespace) -> int:
    context = getattr(args, "context", None)
    config = getattr(context, "config", None)
    json_output = bool(getattr(args, "json", False)) or bool(getattr(context, "json_output", False))
    report = run_doctor(getattr(args, "target", "control"), config=config, fix=bool(args.fix))
    if json_output:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(_human_report(report))
    return report.exit_code
