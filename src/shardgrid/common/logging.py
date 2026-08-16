"""Structured logging and diagnostics helpers for ShardGrid."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence, cast

from shardgrid.common.process import redact_command
from shardgrid.jobs.models import FailureRecord


def _serialize(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value



def redact_mapping(data: Mapping[str, Any], secrets: Sequence[str] = ()) -> dict[str, Any]:
    redacted = _serialize(dict(data))
    rendered = json.dumps(redacted, sort_keys=True)
    for secret in secrets:
        if secret:
            rendered = rendered.replace(secret, "***")
    return cast(dict[str, Any], json.loads(rendered))



def build_json_log(
    *,
    event: str,
    host: str,
    stage: str,
    message: str,
    failure: FailureRecord | None = None,
    command: str | Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": event,
        "host": host,
        "stage": stage,
        "message": message,
    }
    if command is not None:
        payload["command"] = redact_command(command, secrets)
    if failure is not None:
        payload["failure"] = _serialize(failure)
    if extra is not None:
        payload["extra"] = redact_mapping(extra, secrets)
    return payload



def format_json_log(**kwargs: Any) -> str:
    return json.dumps(build_json_log(**kwargs), sort_keys=True)



def format_failure_diagnostics(failure: FailureRecord) -> str:
    lines = [
        f"stage: {failure.stage.value}",
        f"host: {failure.host}",
        f"message: {failure.message}",
        f"recommended_action: {failure.recommended_action}",
    ]
    if failure.command:
        lines.append(f"command: {failure.command}")
    if failure.exit_code is not None:
        lines.append(f"exit_code: {failure.exit_code}")
    if failure.stdout_path:
        lines.append(f"stdout_path: {failure.stdout_path}")
    if failure.stderr_path:
        lines.append(f"stderr_path: {failure.stderr_path}")
    if failure.manual_action_required:
        lines.append("manual_action_required: true")
    return "\n".join(lines)
