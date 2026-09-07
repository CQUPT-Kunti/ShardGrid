from __future__ import annotations

from types import SimpleNamespace

from shardgrid.bootstrap import runner
from shardgrid.common.enums import Health


def test_control_bootstrap_fix_is_noop_when_already_healthy(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(self, command, **kwargs):
        calls.append(tuple(command))
        return SimpleNamespace(
            stdout='{"health":"healthy","manual_actions":[],"commands_run":["check"]}',
            stderr="",
            recorded_command=" ".join(command),
        )

    monkeypatch.setattr(runner.LinuxPlatform, "run", fake_run)

    result = runner.run_control_bootstrap(fix=True)

    assert result.execution == "skipped"
    assert len(calls) == 1


def test_control_bootstrap_fix_stops_for_manual_action(monkeypatch) -> None:
    payload = (
        '{"health":"blocked_manual_action","manual_actions":'
        '["conda create -n shardgrid"],"commands_run":["check"]}'
    )

    def fake_run(self, command, **kwargs):
        return SimpleNamespace(stdout=payload, stderr="", recorded_command=" ".join(command))

    monkeypatch.setattr(runner.LinuxPlatform, "run", fake_run)

    result = runner.run_control_bootstrap(fix=True)

    assert result.execution == "blocked"
    assert "conda create" in (result.manual_action or "")
    assert result.verified is True


def test_control_bootstrap_fix_requires_successful_verification(monkeypatch) -> None:
    responses = iter(
        [
            SimpleNamespace(
                stdout='{"health":"degraded","manual_actions":[],"commands_run":["check"]}',
                stderr="",
                recorded_command="check",
            ),
            SimpleNamespace(
                stdout='{"health":"healthy","manual_actions":[],"commands_run":["fix"]}',
                stderr="",
                recorded_command="fix",
            ),
            SimpleNamespace(stdout="", stderr="verify failed", recorded_command="verify"),
        ]
    )

    def fake_run(self, command, **kwargs):
        return next(responses)

    monkeypatch.setattr(runner.LinuxPlatform, "run", fake_run)

    result = runner.run_control_bootstrap(fix=True)

    assert result.execution == "blocked"
    assert result.verified is False
    assert result.after_verification == {
        "health": "healthy",
        "manual_actions": [],
        "commands_run": ["fix"],
    }
    assert result.failure_reason == "verify failed"


def test_control_bootstrap_fix_preserves_existing_conda_reuse(monkeypatch) -> None:
    payload = (
        '{"health":"healthy","manual_actions":[],"commands_run":["check"],'
        '"conda":{"selected_environment":"shardgrid","selected_prefix":"/envs/shardgrid"}}'
    )

    def fake_run(self, command, **kwargs):
        return SimpleNamespace(stdout=payload, stderr="", recorded_command=" ".join(command))

    monkeypatch.setattr(runner.LinuxPlatform, "run", fake_run)

    result = runner.run_control_bootstrap(fix=True)

    assert result.execution == "skipped"
    assert result.after_verification["conda"]["selected_environment"] == "shardgrid"


def test_control_bootstrap_blocks_known_manual_categories(monkeypatch) -> None:
    actions = [
        "sudo apt-get install -y iperf3",
        "reboot required after package install",
        "enable BIOS virtualization option",
        "password required for remote login",
        "ufw allow 29500/tcp",
    ]
    for action in actions:
        payload = (
            '{"health":"blocked_manual_action","manual_actions":['
            f'"{action}"'
            '],"commands_run":["check"]}'
        )

        def fake_run(self, command, **kwargs):
            return SimpleNamespace(stdout=payload, stderr="", recorded_command=" ".join(command))

        monkeypatch.setattr(runner.LinuxPlatform, "run", fake_run)
        result = runner.run_control_bootstrap(fix=True)
        assert result.execution == "blocked"
        assert result.manual_action == action


def test_worker_runtime_bootstrap_rechecks_after_safe_fix(monkeypatch) -> None:
    states = iter(
        [
            {
                "health": "healthy",
                "manual_actions": [],
                "commands_run": ["check"],
                "nccl_path_mtu": {"status": "NCCL_PATH_MTU_UNSAFE"},
            },
            {
                "health": "healthy",
                "manual_actions": [],
                "commands_run": ["fix"],
                "nccl_path_mtu": {"status": "PASS"},
            },
            {
                "health": "healthy",
                "manual_actions": [],
                "commands_run": ["verify"],
                "nccl_path_mtu": {"status": "PASS"},
            },
        ]
    )

    def fake_run_worker_bootstrap(wrapper, *, peer_ip, expected_mtu, fix):
        payload = next(states)
        return SimpleNamespace(stdout="", stderr="", recorded_command="cmd"), payload

    monkeypatch.setattr(runner, "_run_worker_bootstrap", fake_run_worker_bootstrap)

    result = runner.run_worker_runtime_bootstrap(
        SimpleNamespace(config=SimpleNamespace(distro="Ubuntu", user="shardgrid")),
        peer_ip="10.0.0.2",
        expected_mtu=1500,
        fix=True,
    )

    assert result.execution == "executed"
    assert result.after_verification["nccl_path_mtu"]["status"] == "PASS"
    assert result.verified is True


def test_worker_runtime_bootstrap_blocks_manual_action_without_retry(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_run_worker_bootstrap(wrapper, *, peer_ip, expected_mtu, fix):
        calls.append(fix)
        payload = {
            "health": Health.BLOCKED_MANUAL_ACTION.value,
            "manual_actions": [
                "set interface eth3 MTU to 1500 "
                "for NCCL peer 10.0.0.2 (requires root/CAP_NET_ADMIN)"
            ],
            "commands_run": ["check"],
            "nccl_path_mtu": {"status": "NCCL_PATH_MTU_UNSAFE"},
        }
        return SimpleNamespace(stdout="", stderr="", recorded_command="cmd"), payload

    monkeypatch.setattr(runner, "_run_worker_bootstrap", fake_run_worker_bootstrap)

    result = runner.run_worker_runtime_bootstrap(
        SimpleNamespace(config=SimpleNamespace(distro="Ubuntu", user="shardgrid")),
        peer_ip="10.0.0.2",
        expected_mtu=1500,
        fix=True,
    )

    assert calls == [False]
    assert result.execution == "blocked"
    assert "requires root/CAP_NET_ADMIN" in (result.manual_action or "")


def test_worker_runtime_bootstrap_fix_does_not_pass_without_verify(monkeypatch) -> None:
    states = iter(
        [
            {
                "health": "healthy",
                "manual_actions": [],
                "commands_run": ["check"],
                "nccl_path_mtu": {"status": "NCCL_PATH_MTU_UNSAFE"},
            },
            {
                "health": "healthy",
                "manual_actions": [],
                "commands_run": ["fix"],
                "nccl_path_mtu": {"status": "PASS"},
            },
        ]
    )
    calls: list[bool] = []

    def fake_run_worker_bootstrap(wrapper, *, peer_ip, expected_mtu, fix):
        calls.append(fix)
        payload = next(states, None)
        if payload is None:
            return (
                SimpleNamespace(stdout="", stderr="verify missing", recorded_command="verify"),
                None,
            )
        return SimpleNamespace(stdout="", stderr="", recorded_command="cmd"), payload

    monkeypatch.setattr(runner, "_run_worker_bootstrap", fake_run_worker_bootstrap)

    result = runner.run_worker_runtime_bootstrap(
        SimpleNamespace(config=SimpleNamespace(distro="Ubuntu", user="shardgrid")),
        peer_ip="10.0.0.2",
        expected_mtu=1500,
        fix=True,
    )

    assert calls == [False, True, False]
    assert result.execution == "blocked"
    assert result.verified is False
    assert result.after_verification["commands_run"] == ["fix"]
    assert result.failure_reason == "verify missing"
