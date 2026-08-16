from __future__ import annotations

import sys

from shardgrid.platforms.linux import LinuxPlatform
from shardgrid.platforms.windows import WindowsPlatform
from shardgrid.platforms.wsl import WSLPlatform


def test_linux_adapter_detect_run_and_path() -> None:
    adapter = LinuxPlatform()

    detected = adapter.detect()
    result = adapter.run([sys.executable, "-c", "print('ok')"])
    path = adapter.path_join("jobs", "job-0001", "logs")

    assert detected["platform"] == "linux"
    assert result.ok is True
    assert result.stdout.strip() == "ok"
    assert path == "jobs/job-0001/logs"


def test_windows_adapter_path_and_command_construction() -> None:
    adapter = WindowsPlatform()

    wrapped = adapter.wrap_command(["python", "-V"])
    path = adapter.path_join("jobs", "job-0001", "logs")

    assert wrapped[0] == "powershell.exe"
    assert wrapped[1] == "-Command"
    assert "python -V" in wrapped[2]
    assert path.endswith("logs")
    assert "\\" in path


def test_wsl_adapter_runtime_is_distinct_and_command_construction() -> None:
    adapter = WSLPlatform(distro="Ubuntu-22.04")

    detected = adapter.detect()
    wrapped = adapter.wrap_command(["python3", "-V"])
    path = adapter.path_join("jobs", "job-0001", "logs")

    assert detected["platform"] == "wsl2_linux"
    assert detected["distro"] == "Ubuntu-22.04"
    assert wrapped[:4] == ("wsl", "-d", "Ubuntu-22.04", "/bin/bash")
    assert wrapped[4] == "-lc"
    assert "python3 -V" in wrapped[5]
    assert path == "jobs/job-0001/logs"


def test_platform_manual_action_contract_is_shared() -> None:
    linux = LinuxPlatform()
    windows = WindowsPlatform()
    wsl = WSLPlatform()

    assert linux.validate_manual_action("manual:reboot").requires_manual_action is True
    assert windows.validate_manual_action("safe:probe").allowed is True
    assert wsl.validate_manual_action("manual:sudo required").allowed is False


def test_platform_bootstrap_steps_are_shell_specific() -> None:
    linux = LinuxPlatform()
    windows = WindowsPlatform()
    wsl = WSLPlatform(distro="Ubuntu")

    linux_step = linux.bootstrap_step("check", ["python3", "--version"])
    windows_step = windows.bootstrap_step("check", ["python", "--version"])
    wsl_step = wsl.bootstrap_step("check", ["python3", "--version"])

    assert linux_step.command == ("python3", "--version")
    assert windows_step.command[0] == "powershell.exe"
    assert wsl_step.command[0] == "wsl"
