from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from shardgrid.common.process import redact_command
from shardgrid.platforms.linux import LinuxPlatform
from shardgrid.platforms.windows import WindowsPlatform
from shardgrid.platforms.wsl import WSLPlatform


def test_linux_run_preserves_arguments_with_spaces() -> None:
    argument = "a b   c"

    result = LinuxPlatform().run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", argument]
    )

    assert result.ok is True
    assert result.stdout.strip() == argument


def test_linux_run_preserves_special_characters_without_shell_expansion() -> None:
    argument = "$HOME; echo injected | rev && whoami `id`"

    result = LinuxPlatform().run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", argument]
    )

    assert result.ok is True
    assert result.stdout.strip() == argument


def test_linux_run_quotes_round_trip() -> None:
    argument = "a' b\" c"

    result = LinuxPlatform().run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", argument]
    )

    assert result.ok is True
    assert result.stdout.strip() == argument


def test_linux_run_propagates_environment() -> None:
    result = LinuxPlatform().run(
        [sys.executable, "-c", "import os; print(os.environ['SG_TEST_ENV'])"],
        env={"SG_TEST_ENV": "from-config"},
    )

    assert result.ok is True
    assert result.stdout.strip() == "from-config"


def test_conda_python_executable_is_used_verbatim() -> None:
    result = LinuxPlatform().run([sys.executable, "-c", "import sys; print(sys.executable)"])

    assert result.ok is True
    assert result.stdout.strip() == sys.executable


def test_bootstrap_step_preserves_provided_python_executable() -> None:
    step = LinuxPlatform().bootstrap_step("python-check", [sys.executable, "--version"])

    assert step.command == (sys.executable, "--version")


def test_windows_wrap_command_serialization_is_deterministic() -> None:
    adapter = WindowsPlatform()

    first = adapter.wrap_command(["python", "-V"])
    second = adapter.wrap_command(["python", "-V"])

    assert first == second
    assert first[0] == "powershell.exe"
    assert first[1] == "-Command"
    assert first[2] == "python -V"


def test_windows_wrap_command_preserves_arguments_with_spaces() -> None:
    wrapped = WindowsPlatform().wrap_command(["python", "-c", "print('a b')"])

    assert wrapped[2] == "python -c print('a b')"


def test_windows_wrap_command_preserves_special_characters() -> None:
    wrapped = WindowsPlatform().wrap_command(["echo", "$env:USER; & whoami"])

    assert wrapped[2] == "echo $env:USER; & whoami"


def test_windows_run_forwards_env_cwd_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import shardgrid.platforms.windows as windows_module

    captured_command: tuple[object, ...] | None = None
    captured_kwargs: dict[str, object] = {}

    def fake_run(command: object, **kwargs: object) -> object:
        nonlocal captured_command
        if isinstance(command, tuple):
            captured_command = command
        captured_kwargs.update(kwargs)
        return windows_module.ProcessResult(
            args=(),
            recorded_command="",
            shell=False,
            cwd=None,
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            runtime_environment={},
        )

    monkeypatch.setattr(windows_module, "run_process", fake_run)
    adapter = WindowsPlatform()
    adapter.run(
        ["python", "-c", "print('ok')"],
        cwd="/tmp",
        env={"SG_TEST_ENV": "present"},
        timeout=5,
    )

    assert captured_command is not None
    assert captured_command[0] == "powershell.exe"
    assert captured_kwargs["env"] == {"SG_TEST_ENV": "present"}
    assert captured_kwargs["cwd"] == "/tmp"
    assert captured_kwargs["timeout"] == 5


def test_wsl_wrap_command_serialization_with_distro() -> None:
    wrapped = WSLPlatform(distro="Ubuntu-22.04").wrap_command(["python3", "-c", "print('a b')"])

    assert wrapped == (
        "wsl",
        "-d",
        "Ubuntu-22.04",
        "/bin/bash",
        "-lc",
        "python3 -c print('a b')",
    )


def test_wsl_wrap_command_serialization_without_distro() -> None:
    wrapped = WSLPlatform().wrap_command(["python3", "-V"])

    assert wrapped == ("wsl", "/bin/bash", "-lc", "python3 -V")


def test_wsl_run_forwards_env_cwd_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import shardgrid.platforms.wsl as wsl_module

    captured_command: tuple[object, ...] | None = None
    captured_kwargs: dict[str, object] = {}

    def fake_run(command: object, **kwargs: object) -> object:
        nonlocal captured_command
        if isinstance(command, tuple):
            captured_command = command
        captured_kwargs.update(kwargs)
        return wsl_module.ProcessResult(
            args=(),
            recorded_command="",
            shell=False,
            cwd=None,
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            runtime_environment={},
        )

    monkeypatch.setattr(wsl_module, "run_process", fake_run)
    adapter = WSLPlatform(distro="Ubuntu-22.04")
    adapter.run(
        ["python3", "-V"],
        cwd="/home/alice",
        env={"SG_TEST_ENV": "present"},
        timeout=7,
    )

    assert captured_command is not None
    assert captured_command[0] == "wsl"
    assert captured_kwargs["env"] == {"SG_TEST_ENV": "present"}
    assert captured_kwargs["cwd"] == "/home/alice"
    assert captured_kwargs["timeout"] == 7


def test_redact_command_round_trips_through_shell_lexing() -> None:
    command = ["python", "-c", "print('a b')", "arg with $HOME; spaces & such"]

    serialized = redact_command(command)

    assert serialized != " ".join(command)
    assert shlex.split(serialized) == command


def test_platform_and_process_modules_have_no_hardcoded_runtime_literals() -> None:
    module_dir = Path(__file__).resolve().parents[2] / "src" / "shardgrid"
    targets = [
        module_dir / "platforms" / "linux.py",
        module_dir / "platforms" / "windows.py",
        module_dir / "platforms" / "wsl.py",
        module_dir / "platforms" / "base.py",
        module_dir / "common" / "process.py",
    ]
    forbidden = [
        "conda activate",
        "conda run",
        "/home/",
        "/Users/",
        "C:\\Users",
        "C:/Users",
        "python.exe",
        "wsl.exe",
        "Ubuntu",
    ]

    for path in targets:
        text = path.read_text()
        for literal in forbidden:
            assert literal not in text, f"{path.name} contains hardcoded literal {literal!r}"