from __future__ import annotations

from types import SimpleNamespace

import pytest

from shardgrid.bootstrap import runner
from shardgrid.common.enums import Health


def _result(payload: dict[str, object], *, command: str = "cmd") -> SimpleNamespace:
    import json

    return SimpleNamespace(
        stdout=json.dumps(payload),
        stderr="",
        recorded_command=command,
    )


@pytest.mark.parametrize(
    "action",
    [
        "sudo apt-get install -y iperf3",
        "reboot required after package install",
        "enable BIOS virtualization option",
        "password required for remote login",
        "ufw allow 29500/tcp",
        "install conda into the runtime",
        "overwrite conda environment shardgrid",
    ],
)
def test_control_fix_stops_on_manual_action_categories(monkeypatch, action: str) -> None:
    payload = {
        "health": Health.BLOCKED_MANUAL_ACTION.value,
        "manual_actions": [action],
        "commands_run": ["check"],
    }

    monkeypatch.setattr(
        runner.LinuxPlatform,
        "run",
        lambda self, command, **kwargs: _result(payload),
    )

    result = runner.run_control_bootstrap(fix=True)

    assert result.execution == "blocked"
    assert result.verified is True
    assert result.manual_action == action
    assert result.effective_state == payload


def test_control_fix_without_verified_check_cannot_become_healthy(monkeypatch) -> None:
    responses = iter(
        [
            _result({"health": "degraded", "manual_actions": [], "commands_run": ["check"]}),
            _result({"health": "healthy", "manual_actions": [], "commands_run": ["fix"]}),
            SimpleNamespace(stdout="", stderr="verify failed", recorded_command="verify"),
        ]
    )

    monkeypatch.setattr(
        runner.LinuxPlatform,
        "run",
        lambda self, command, **kwargs: next(responses),
    )

    result = runner.run_control_bootstrap(fix=True)

    assert result.execution == "blocked"
    assert result.verified is False
    assert result.failure_reason == "verify failed"
    assert result.after_verification == {
        "health": "healthy",
        "manual_actions": [],
        "commands_run": ["fix"],
    }


def test_worker_fix_blocker_is_not_promoted_to_success(monkeypatch) -> None:
    states = iter(
        [
            (
                SimpleNamespace(stdout="", stderr="", recorded_command="check"),
                {
                    "health": "healthy",
                    "manual_actions": [],
                    "commands_run": ["check"],
                    "nccl_path_mtu": {"status": "NCCL_PATH_MTU_UNSAFE"},
                },
            ),
            (
                SimpleNamespace(stdout="", stderr="", recorded_command="fix"),
                {
                    "health": Health.BLOCKED_MANUAL_ACTION.value,
                    "manual_actions": [
                        "set interface eth0 MTU to 1500 for NCCL peer 10.0.0.2 "
                        "(requires root/CAP_NET_ADMIN)"
                    ],
                    "commands_run": ["fix"],
                    "nccl_path_mtu": {"status": "NCCL_PATH_MTU_UNSAFE"},
                },
            ),
            (
                SimpleNamespace(stdout="", stderr="", recorded_command="verify"),
                {
                    "health": Health.BLOCKED_MANUAL_ACTION.value,
                    "manual_actions": [
                        "set interface eth0 MTU to 1500 for NCCL peer 10.0.0.2 "
                        "(requires root/CAP_NET_ADMIN)"
                    ],
                    "commands_run": ["verify"],
                    "nccl_path_mtu": {"status": "NCCL_PATH_MTU_UNSAFE"},
                },
            ),
        ]
    )

    monkeypatch.setattr(runner, "_run_worker_bootstrap", lambda *args, **kwargs: next(states))

    result = runner.run_worker_runtime_bootstrap(
        SimpleNamespace(config=SimpleNamespace(distro="Ubuntu", user="shardgrid")),
        peer_ip="10.0.0.2",
        expected_mtu=1500,
        fix=True,
    )

    assert result.execution == "blocked"
    assert result.verified is True
    assert "requires root/CAP_NET_ADMIN" in (result.manual_action or "")
    assert result.effective_state is not None
    assert result.effective_state["health"] == Health.BLOCKED_MANUAL_ACTION.value


def test_worker_verified_safe_fix_clears_blocker_state(monkeypatch) -> None:
    states = iter(
        [
            (
                SimpleNamespace(stdout="", stderr="", recorded_command="check"),
                {
                    "health": "healthy",
                    "manual_actions": [],
                    "commands_run": ["check"],
                    "conda": {"selected_environment": "shardgrid"},
                    "nccl_path_mtu": {"status": "NCCL_PATH_MTU_UNSAFE"},
                },
            ),
            (
                SimpleNamespace(stdout="", stderr="", recorded_command="fix"),
                {
                    "health": "healthy",
                    "manual_actions": [],
                    "commands_run": ["fix"],
                    "conda": {"selected_environment": "shardgrid"},
                    "nccl_path_mtu": {"status": "PASS"},
                },
            ),
            (
                SimpleNamespace(stdout="", stderr="", recorded_command="verify"),
                {
                    "health": "healthy",
                    "manual_actions": [],
                    "commands_run": ["verify"],
                    "conda": {"selected_environment": "shardgrid"},
                    "nccl_path_mtu": {"status": "PASS"},
                },
            ),
        ]
    )

    monkeypatch.setattr(runner, "_run_worker_bootstrap", lambda *args, **kwargs: next(states))

    result = runner.run_worker_runtime_bootstrap(
        SimpleNamespace(config=SimpleNamespace(distro="Ubuntu", user="shardgrid")),
        peer_ip="10.0.0.2",
        expected_mtu=1500,
        fix=True,
    )

    assert result.execution == "executed"
    assert result.verified is True
    assert result.manual_action is None
    assert result.after_verification is not None
    assert result.after_verification["conda"]["selected_environment"] == "shardgrid"
    assert result.after_verification["nccl_path_mtu"]["status"] == "PASS"
