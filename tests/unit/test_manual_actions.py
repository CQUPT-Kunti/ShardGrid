from __future__ import annotations

import pytest

from shardgrid.common.enums import FailureStage, Health, JobState
from shardgrid.common.manual_actions import classify_manual_action, validate_automation_action
from shardgrid.common.models import as_job_id
from shardgrid.jobs.models import JobStatus


@pytest.mark.parametrize(
    ("action", "category"),
    [
        ("sudo apt install openssh-server", "administrator_privilege"),
        ("reboot required after package install", "reboot"),
        ("enable BIOS SR-IOV option", "bios_modification"),
        ("password required for remote login", "password_request"),
        ("netsh advfirewall firewall add rule", "risky_firewall_modification"),
        ("conda create -n shardgrid python", "conda_install_or_env_create"),
    ],
)
def test_manual_action_categories_are_blocked(action: str, category: str) -> None:
    blocker = classify_manual_action(action)

    assert blocker is not None
    assert blocker.category == category
    assert blocker.health is Health.BLOCKED_MANUAL_ACTION
    assert blocker.requires_manual_action is True


def test_safe_action_is_not_blocked() -> None:
    check = validate_automation_action("python3 --version")

    assert check.allowed is True
    assert check.requires_manual_action is False


def test_blocker_converts_to_manual_action_check_and_failure_record() -> None:
    blocker = classify_manual_action("reboot required after kernel update")
    assert blocker is not None

    check = blocker.to_check()
    failure = blocker.to_failure_record(
        stage=FailureStage.BOOTSTRAP,
        host="machine-a.local",
        command=["reboot"],
    )

    assert check.allowed is False
    assert check.requires_manual_action is True
    assert failure.manual_action_required is True
    assert failure.recommended_action == blocker.recommended_action
    assert failure.command == "reboot"


def test_blocked_state_cannot_be_marked_as_fake_pass() -> None:
    blocker = classify_manual_action("password required for SSH key import")
    assert blocker is not None

    failure = blocker.to_failure_record(
        stage=FailureStage.BOOTSTRAP,
        host="machine-c.local",
        command="ssh-copy-id worker",
    )
    status = JobStatus(
        job_id=as_job_id("job-0001"),
        state=JobState.FAILED,
        phase="bootstrap",
        failure=failure,
    )

    assert status.state.value == "failed"
    assert status.failure is not None
    assert status.failure.manual_action_required is True
    assert status.failure.recommended_action
