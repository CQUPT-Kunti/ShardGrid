"""Safe bootstrap execution helpers for ``doctor --fix``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardgrid.common.enums import Health
from shardgrid.common.manual_actions import classify_manual_action
from shardgrid.common.process import ProcessResult
from shardgrid.platforms.linux import LinuxPlatform
from shardgrid.transport.runtime import WSLRuntimeWrapper, wrap_wsl_direct_command


@dataclass(frozen=True)
class BootstrapExecution:
    target: str
    action: str
    before_state: dict[str, Any] | None
    execution: str
    after_verification: dict[str, Any] | None
    verified: bool = False
    failure_reason: str | None = None
    manual_action: str | None = None
    command: str | None = None
    commands_run: tuple[str, ...] = ()

    @property
    def effective_state(self) -> dict[str, Any] | None:
        return self.after_verification or self.before_state


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_bootstrap_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None


def _payload_health(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    health = payload.get("health")
    return None if health is None else str(health)


def _payload_manual_action(payload: dict[str, Any] | None) -> str | None:
    for action in (payload or {}).get("manual_actions", []):
        text = str(action)
        blocker = classify_manual_action(text)
        if blocker is not None:
            return text
    actions = (payload or {}).get("manual_actions") or []
    return str(actions[0]) if actions else None


def _payload_commands(payload: dict[str, Any] | None) -> tuple[str, ...]:
    return tuple(str(item) for item in ((payload or {}).get("commands_run") or []))


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _fix_execution(payload: dict[str, Any] | None, *, verified: bool) -> str:
    if not verified or payload is None:
        return "blocked"
    if _payload_health(payload) == Health.BLOCKED_MANUAL_ACTION.value:
        return "blocked"
    if _payload_manual_action(payload):
        return "blocked"
    return "executed"


def run_control_bootstrap(*, fix: bool) -> BootstrapExecution:
    platform = LinuxPlatform()
    script_path = _repo_root() / "scripts" / "bootstrap-linux.sh"
    before_step = platform.bootstrap_step(
        "bootstrap-check", ["bash", str(script_path), "--check", "--json"]
    )
    before = platform.run(before_step.command, timeout=60)
    before_payload = _parse_bootstrap_json(before.stdout)
    if before_payload is None:
        return BootstrapExecution(
            target="control",
            action="bootstrap-linux --check",
            before_state=None,
            execution="blocked",
            after_verification=None,
            verified=False,
            failure_reason=(
                before.stderr.strip()
                or before.stdout.strip()
                or "control bootstrap check failed"
            ),
            command=before.recorded_command,
        )
    if not fix:
        return BootstrapExecution(
            target="control",
            action="bootstrap-linux --check",
            before_state=before_payload,
            execution="skipped",
            after_verification=before_payload,
            verified=True,
            manual_action=_payload_manual_action(before_payload),
            command=before.recorded_command,
            commands_run=_payload_commands(before_payload),
        )
    if _payload_health(before_payload) in {
        Health.HEALTHY.value,
        Health.BLOCKED_MANUAL_ACTION.value,
    }:
        execution = (
            "skipped"
            if _payload_health(before_payload) == Health.HEALTHY.value
            else "blocked"
        )
        return BootstrapExecution(
            target="control",
            action="bootstrap-linux --install-deps",
            before_state=before_payload,
            execution=execution,
            after_verification=before_payload,
            verified=True,
            manual_action=_payload_manual_action(before_payload),
            command=before.recorded_command,
            commands_run=_payload_commands(before_payload),
        )
    fix_result = platform.run(
        platform.bootstrap_step(
            "bootstrap-fix",
            ["bash", str(script_path), "--install-deps", "--json"],
        ).command,
        timeout=120,
    )
    fixed_payload = _parse_bootstrap_json(fix_result.stdout)
    verify_step = platform.bootstrap_step(
        "bootstrap-verify", ["bash", str(script_path), "--check", "--json"]
    )
    verify = platform.run(
        verify_step.command,
        timeout=60,
    )
    verify_payload = _parse_bootstrap_json(verify.stdout)
    effective = verify_payload
    verified = verify_payload is not None
    return BootstrapExecution(
        target="control",
        action="bootstrap-linux --install-deps",
        before_state=before_payload,
        execution=_fix_execution(effective or fixed_payload, verified=verified),
        after_verification=effective or fixed_payload,
        verified=verified,
        failure_reason=(
            None
            if verified and _payload_health(effective) == Health.HEALTHY.value
            else _first_nonempty(
                verify.stderr,
                verify.stdout if not verified else None,
                fix_result.stderr,
                fix_result.stdout if fixed_payload is None else None,
                "control bootstrap verification failed",
            )
        ),
        manual_action=_payload_manual_action(effective or fixed_payload),
        command=fix_result.recorded_command,
        commands_run=_payload_commands(effective or fixed_payload),
    )


def _run_worker_bootstrap(
    wrapper: WSLRuntimeWrapper,
    *,
    peer_ip: str,
    expected_mtu: int,
    fix: bool,
) -> tuple[ProcessResult, dict[str, Any] | None]:
    script_path = _repo_root() / "scripts" / "bootstrap-wsl.sh"
    command = ["--fix-nccl-mtu-only", "--json"] if fix else ["--check", "--json"]
    payload = (
        f"SHARDGRID_NCCL_PEER_IP={peer_ip} "
        f"SHARDGRID_NCCL_MTU={expected_mtu} "
        "SHARDGRID_BOOTSTRAP_JSON=1 "
        "SHARDGRID_WSL_PERSIST_NCCL_MTU=0 "
        "bash -s -- " + " ".join(command)
    )
    remote_command = wrap_wsl_direct_command(
        wrapper.config.distro or "",
        wrapper.config.user or "root",
        payload,
    )
    result = wrapper.executor.run(
        remote_command,
        stdin=script_path.read_text(encoding="utf-8"),
        timeout=60,
    )
    parsed = _parse_bootstrap_json(result.stdout)
    if parsed is not None:
        return result, parsed
    latest = wrapper.run('cat "$HOME/.shardgrid/bootstrap/wsl-latest.json"', timeout=15)
    return result, _parse_bootstrap_json(latest.stdout)


def run_worker_runtime_bootstrap(
    wrapper: WSLRuntimeWrapper,
    *,
    peer_ip: str,
    expected_mtu: int,
    fix: bool,
) -> BootstrapExecution:
    before_result, before_payload = _run_worker_bootstrap(
        wrapper, peer_ip=peer_ip, expected_mtu=expected_mtu, fix=False
    )
    if before_payload is None:
        return BootstrapExecution(
            target=peer_ip,
            action="bootstrap-wsl --check",
            before_state=None,
            execution="blocked",
            after_verification=None,
            verified=False,
            failure_reason=(
                before_result.stderr.strip()
                or before_result.stdout.strip()
                or "remote bootstrap failed"
            ),
            command=before_result.recorded_command,
        )
    before_status = str(((before_payload.get("nccl_path_mtu") or {}).get("status")) or "")
    before_manual_action = _payload_manual_action(before_payload)
    if not fix or before_status.upper() == "PASS":
        return BootstrapExecution(
            target=peer_ip,
            action="bootstrap-wsl --fix-nccl-mtu-only",
            before_state=before_payload,
            execution="skipped",
            after_verification=before_payload,
            verified=True,
            manual_action=before_manual_action,
            command=before_result.recorded_command,
            commands_run=_payload_commands(before_payload),
        )
    if before_manual_action:
        return BootstrapExecution(
            target=peer_ip,
            action="bootstrap-wsl --fix-nccl-mtu-only",
            before_state=before_payload,
            execution="blocked",
            after_verification=before_payload,
            verified=True,
            manual_action=before_manual_action,
            command=before_result.recorded_command,
            commands_run=_payload_commands(before_payload),
        )
    fix_result, fix_payload = _run_worker_bootstrap(
        wrapper, peer_ip=peer_ip, expected_mtu=expected_mtu, fix=True
    )
    verify_result, verify_payload = _run_worker_bootstrap(
        wrapper, peer_ip=peer_ip, expected_mtu=expected_mtu, fix=False
    )
    effective = verify_payload
    verified = verify_payload is not None
    fallback = effective or fix_payload
    after_status = str((((effective or {}).get("nccl_path_mtu") or {}).get("status")) or "")
    return BootstrapExecution(
        target=peer_ip,
        action="bootstrap-wsl --fix-nccl-mtu-only",
        before_state=before_payload,
        execution=_fix_execution(fallback, verified=verified),
        after_verification=fallback,
        verified=verified,
        failure_reason=(
            None
            if verified and after_status.upper() == "PASS"
            else _first_nonempty(
                verify_result.stderr,
                verify_result.stdout if not verified else None,
                fix_result.stderr,
                fix_result.stdout if fix_payload is None else None,
                "remote bootstrap verification failed",
            )
        ),
        manual_action=_payload_manual_action(fallback),
        command=fix_result.recorded_command,
        commands_run=_payload_commands(fallback),
    )
