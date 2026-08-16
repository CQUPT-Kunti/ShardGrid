from __future__ import annotations

import sys

from shardgrid.platforms.base import BootstrapStep, FakePlatformAdapter, ManualActionCheck


def test_detect_contract() -> None:
    adapter = FakePlatformAdapter()

    payload = adapter.detect()

    assert payload["platform"] == "fake"
    assert payload["shell"] == "/bin/sh"


def test_run_contract() -> None:
    adapter = FakePlatformAdapter()

    result = adapter.run([sys.executable, "-c", "print('ok')"])

    assert result.ok is True
    assert result.stdout.strip() == "ok"


def test_path_contract() -> None:
    adapter = FakePlatformAdapter()

    path = adapter.path_join("jobs", "job-0001", "logs")

    assert path.endswith("logs")
    assert "job-0001" in path


def test_manual_action_contract() -> None:
    adapter = FakePlatformAdapter()

    allowed = adapter.validate_manual_action("safe:probe")
    blocked = adapter.validate_manual_action("manual:reboot required")

    assert allowed == ManualActionCheck(allowed=True)
    assert blocked.allowed is False
    assert blocked.requires_manual_action is True
    assert blocked.reason == "reboot required"


def test_bootstrap_step_contract() -> None:
    adapter = FakePlatformAdapter()

    step = adapter.bootstrap_step("python-check", [sys.executable, "--version"])

    assert step == BootstrapStep(name="python-check", command=(sys.executable, "--version"))
