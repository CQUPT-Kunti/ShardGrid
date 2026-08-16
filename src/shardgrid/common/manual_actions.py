"""Manual-action safety boundaries for automation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shardgrid.common.enums import FailureStage, Health
from shardgrid.common.errors import make_failure_record
from shardgrid.jobs.models import FailureRecord
from shardgrid.platforms.base import ManualActionCheck


@dataclass(frozen=True)
class ManualActionRule:
    category: str
    tokens: tuple[str, ...]
    recommended_action: str


_RULES = (
    ManualActionRule(
        category="administrator_privilege",
        tokens=("sudo ", "run as administrator", "elevation required", "administrator privilege"),
        recommended_action=(
            "have an operator rerun this step with approved administrator privileges"
        ),
    ),
    ManualActionRule(
        category="reboot",
        tokens=("reboot", "restart required", "reboot required"),
        recommended_action="have an operator schedule and confirm the reboot before continuing",
    ),
    ManualActionRule(
        category="bios_modification",
        tokens=("bios", "uefi", "firmware setting"),
        recommended_action="have an operator review and apply the BIOS or firmware change manually",
    ),
    ManualActionRule(
        category="password_request",
        tokens=("password", "passphrase", "enter credentials"),
        recommended_action=(
            "have an operator provide credentials through an approved secure channel"
        ),
    ),
    ManualActionRule(
        category="risky_firewall_modification",
        tokens=("firewall", "ufw allow", "netsh advfirewall", "open port"),
        recommended_action=(
            "have an operator review the firewall change and apply it manually if approved"
        ),
    ),
)


@dataclass(frozen=True)
class ManualActionBlocker:
    category: str
    message: str
    recommended_action: str
    health: Health = Health.BLOCKED_MANUAL_ACTION
    requires_manual_action: bool = True

    def to_check(self) -> ManualActionCheck:
        return ManualActionCheck(
            allowed=False,
            reason=f"{self.category}: {self.message}",
            requires_manual_action=True,
        )

    def to_failure_record(
        self,
        *,
        stage: FailureStage,
        host: str,
        command: str | Sequence[str] | None = None,
    ) -> FailureRecord:
        return make_failure_record(
            stage=stage,
            host=host,
            message=self.message,
            recommended_action=self.recommended_action,
            command=command,
            manual_action_required=True,
        )


def classify_manual_action(action: str) -> ManualActionBlocker | None:
    normalized = action.casefold()
    for rule in _RULES:
        if any(token in normalized for token in rule.tokens):
            return ManualActionBlocker(
                category=rule.category,
                message=f"manual action required: {action}",
                recommended_action=rule.recommended_action,
            )
    return None


def validate_automation_action(action: str) -> ManualActionCheck:
    blocker = classify_manual_action(action)
    if blocker is None:
        return ManualActionCheck(allowed=True)
    return blocker.to_check()
