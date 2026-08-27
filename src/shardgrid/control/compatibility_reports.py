"""Shared compatibility report writer and validator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from shardgrid.common.enums import BackendStatus, FailureStage
from shardgrid.common.process import redact_command, redact_text
from shardgrid.engines.models import CompatibilitySpikeReport

_STATUS_ALIASES: dict[str, BackendStatus] = {
    "pass": BackendStatus.AVAILABLE,
    "available": BackendStatus.AVAILABLE,
    "success": BackendStatus.AVAILABLE,
    "fail": BackendStatus.FAILED,
    "failed": BackendStatus.FAILED,
    "blocked": BackendStatus.BLOCKED,
    "fallback": BackendStatus.FALLBACK_USED,
    "fallback_used": BackendStatus.FALLBACK_USED,
    "experimental": BackendStatus.EXPERIMENTAL,
    "not_checked": BackendStatus.NOT_CHECKED,
    "unchecked": BackendStatus.NOT_CHECKED,
    "unknown": BackendStatus.NOT_CHECKED,
}

_FAILURE_STATES = {BackendStatus.FAILED, BackendStatus.BLOCKED}
_CORE_FIELDS = (
    "report_id",
    "component",
    "stage",
    "machines_tested",
    "versions",
    "commands",
    "results",
    "logs_path",
    "status",
    "blockers",
    "decision",
    "recommended_next_action",
    "created_at",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_compatibility_status(status: BackendStatus | str) -> BackendStatus:
    if isinstance(status, BackendStatus):
        return status
    normalized = str(status).strip().lower().replace(" ", "_")
    if normalized in _STATUS_ALIASES:
        return _STATUS_ALIASES[normalized]
    return BackendStatus(str(status))


def _sanitize_text(value: Any, secrets: Sequence[str]) -> str:
    text = str(value)
    return redact_text(text, secrets) or text


def _sanitize_strings(values: Sequence[Any], secrets: Sequence[str]) -> list[str]:
    return [_sanitize_text(value, secrets) for value in values]


def _sanitize_commands(
    commands: Sequence[str | Sequence[str]],
    secrets: Sequence[str],
) -> list[str]:
    sanitized: list[str] = []
    for command in commands:
        sanitized_command = redact_command(command, secrets)
        sanitized.append(sanitized_command or _sanitize_text(command, secrets))
    return sanitized


def _sanitize_machine_fact(
    machine: Mapping[str, Any], secrets: Sequence[str]
) -> dict[str, str]:
    return {
        str(key): _sanitize_text(value, secrets)
        for key, value in machine.items()
    }


def _sanitize_versions(
    versions: Mapping[str, Any], secrets: Sequence[str]
) -> dict[str, str]:
    return {
        str(key): _sanitize_text(value, secrets)
        for key, value in versions.items()
    }


def _sanitize_stage(stage: FailureStage | str) -> FailureStage:
    return stage if isinstance(stage, FailureStage) else FailureStage(str(stage))


def build_compatibility_report(
    *,
    report_id: str,
    component: str,
    stage: FailureStage | str,
    status: BackendStatus | str,
    machines_tested: Sequence[str] = (),
    machine_facts: Sequence[Mapping[str, Any]] = (),
    versions: Mapping[str, Any] | None = None,
    commands: Sequence[str | Sequence[str]] = (),
    results: Sequence[str] = (),
    blockers: Sequence[str] = (),
    decision: str | None = None,
    recommended_next_action: str | None = None,
    logs_path: str | None = None,
    evidence_refs: Sequence[str] = (),
    preferred_path: str | None = None,
    actual_path: str | None = None,
    created_at: str | None = None,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    component_name = component.strip().lower()
    if not component_name:
        raise ValueError("component is required")
    status_value = normalize_compatibility_status(status)
    report = CompatibilitySpikeReport(
        report_id=report_id,
        component=component_name,
        stage=_sanitize_stage(stage),
        machines_tested=_sanitize_strings(machines_tested, secrets),
        versions=_sanitize_versions(versions or {}, secrets),
        commands=_sanitize_commands(commands, secrets),
        results=_sanitize_strings(results, secrets),
        logs_path=redact_text(logs_path, secrets),
        status=status_value,
        blockers=_sanitize_strings(blockers, secrets),
        decision=redact_text(decision, secrets),
        recommended_next_action=redact_text(recommended_next_action, secrets),
        created_at=created_at or _now(),
    )
    payload = report.to_dict()
    payload["machine_facts"] = [
        _sanitize_machine_fact(machine, secrets) for machine in machine_facts
    ]
    payload["evidence_refs"] = _sanitize_strings(evidence_refs, secrets)
    payload["preferred_path"] = redact_text(preferred_path, secrets)
    payload["actual_path"] = redact_text(actual_path, secrets)
    validate_compatibility_report(payload)
    return payload


def validate_compatibility_report(payload: Mapping[str, Any]) -> None:
    missing = [field for field in _CORE_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing compatibility report fields: {', '.join(missing)}")

    report = CompatibilitySpikeReport.from_dict(
        {field: payload[field] for field in _CORE_FIELDS}
    )
    if report.status in _FAILURE_STATES:
        if not report.machines_tested:
            raise ValueError("failed or blocked report requires machines_tested")
        if not report.versions:
            raise ValueError("failed or blocked report requires versions")
        if not report.commands:
            raise ValueError("failed or blocked report requires commands")
        if not report.results:
            raise ValueError("failed or blocked report requires results")
        if not report.blockers:
            raise ValueError("failed or blocked report requires blockers")
        if not report.recommended_next_action:
            raise ValueError(
                "failed or blocked report requires recommended_next_action"
            )

    if report.status == BackendStatus.FALLBACK_USED:
        preferred_path = payload.get("preferred_path")
        actual_path = payload.get("actual_path")
        if not preferred_path or not actual_path:
            raise ValueError("fallback report requires preferred_path and actual_path")
        if preferred_path == actual_path:
            raise ValueError("fallback report must distinguish preferred_path and actual_path")

    if report.status == BackendStatus.AVAILABLE:
        preferred_path = payload.get("preferred_path")
        actual_path = payload.get("actual_path")
        if preferred_path and actual_path and preferred_path != actual_path:
            raise ValueError("fallback success must not be reported as PASS")


def compatibility_report_from_spike_report(
    report: CompatibilitySpikeReport,
    *,
    machine_facts: Sequence[Mapping[str, Any]] = (),
    evidence_refs: Sequence[str] = (),
    preferred_path: str | None = None,
    actual_path: str | None = None,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    return build_compatibility_report(
        report_id=report.report_id,
        component=report.component,
        stage=report.stage,
        status=report.status,
        machines_tested=report.machines_tested,
        machine_facts=machine_facts,
        versions=report.versions,
        commands=report.commands,
        results=report.results,
        blockers=report.blockers,
        decision=report.decision,
        recommended_next_action=report.recommended_next_action,
        logs_path=report.logs_path,
        evidence_refs=evidence_refs,
        preferred_path=preferred_path,
        actual_path=actual_path,
        created_at=report.created_at,
        secrets=secrets,
    )


def write_compatibility_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    validate_compatibility_report(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def load_compatibility_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("compatibility report must decode to an object")
    validate_compatibility_report(payload)
    return dict(payload)
