"""Granular runtime checks for Windows GPU Workers with WSL2 training runtimes.

Each check runs through a command runner and returns a :class:`CheckOutcome`
that preserves the real reason when a check fails.  The runner abstracts over
the PlatformAdapter + transport boundary so the same checks work locally,
over SSH, or with mocked command results in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from shardgrid.common.process import ProcessResult

Runner = Callable[[Sequence[str] | str], ProcessResult]


@dataclass(frozen=True)
class CheckOutcome:
    ok: bool
    value: str | None = None
    reason: str | None = None
    exit_code: int | None = None

    def failed(self, message: str) -> CheckOutcome:
        return CheckOutcome(
            ok=False,
            value=self.value,
            reason=message if self.reason is None else f"{message}: {self.reason}",
            exit_code=self.exit_code,
        )


def run_check(runner: Runner, command: str) -> ProcessResult:
    try:
        return runner([command])
    except Exception as exc:  # pragma: no cover - defensive runner boundary
        return ProcessResult(
            args=(command,),
            recorded_command=command,
            shell=False,
            cwd=None,
            exit_code=-1,
            stdout="",
            stderr=str(exc),
            timed_out=False,
            runtime_environment={},
        )


def _first_line(result: ProcessResult) -> str:
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else ""


def _outcome(result: ProcessResult) -> CheckOutcome:
    text = (result.stdout or result.stderr).strip()
    if result.ok and text:
        return CheckOutcome(ok=True, value=text, exit_code=result.exit_code)
    return CheckOutcome(
        ok=False,
        value=text or None,
        reason=f"exit {result.exit_code}" if not result.timed_out else "timed out",
        exit_code=result.exit_code,
    )


# --- Windows physical host checks -------------------------------------------

def probe_windows_os(runner: Runner) -> CheckOutcome:
    return _outcome(
        run_check(
            runner,
            "(Get-CimInstance Win32_OperatingSystem).Caption "
            "+ ' | ' + [System.Environment]::OSVersion.VersionString",
        )
    )


def probe_windows_openssh(runner: Runner) -> CheckOutcome:
    return _outcome(
        run_check(
            runner,
            "$ErrorActionPreference='SilentlyContinue'; "
            "if (Get-Command ssh) { (Get-Command ssh).Source } else { '' }",
        )
    )


def probe_windows_wsl_available(runner: Runner) -> CheckOutcome:
    return _outcome(run_check(runner, "wsl.exe --status"))


def probe_windows_nvidia_driver(runner: Runner) -> CheckOutcome:
    return _outcome(
        run_check(runner, "(Get-CimInstance Win32_VideoController).Name")
    )


# --- WSL2 training runtime checks -------------------------------------------

def probe_wsl_conda_executable(runner: Runner) -> CheckOutcome:
    return _outcome(run_check(runner, "command -v conda"))


def probe_wsl_conda_env_list(runner: Runner) -> CheckOutcome:
    return _outcome(run_check(runner, "conda env list"))


def probe_wsl_conda_active(runner: Runner) -> CheckOutcome:
    return _outcome(
        run_check(
            runner,
            'printf "%s|%s" "${CONDA_DEFAULT_ENV:-none}" "${CONDA_PREFIX:-none}"',
        )
    )


def probe_wsl_python_version(runner: Runner, python_executable: str) -> CheckOutcome:
    return _outcome(run_check(runner, f"{python_executable} --version 2>&1"))


def probe_wsl_torch(runner: Runner, python_executable: str) -> CheckOutcome:
    command = (
        "import torch; "
        "print('TORCH_VERSION=' + torch.__version__); "
        "print('CUDA_VERSION=' + str(torch.version.cuda)); "
        "print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))"
    )
    return _outcome(run_check(runner, f"{python_executable} -c '{command}'"))


def probe_wsl_nccl(runner: Runner, python_executable: str) -> CheckOutcome:
    command = (
        "import torch; "
        "if torch.cuda.is_available(): "
        "print('NCCL=' + '.'.join(str(x) for x in torch.cuda.nccl.version())) "
        "else: print('NCCL=not_available')"
    )
    return _outcome(run_check(runner, f"{python_executable} -c '{command}'"))


def probe_wsl_gloo(runner: Runner, python_executable: str) -> CheckOutcome:
    command = "import torch.distributed as d; print('GLOO=' + str(d.is_gloo_available()))"
    return _outcome(run_check(runner, f"{python_executable} -c '{command}'"))


def probe_wsl_gpu(runner: Runner, python_executable: str) -> CheckOutcome:
    command = (
        "import torch; "
        "p = torch.cuda.get_device_properties(0); "
        "print('GPU_NAME=' + p.name); "
        "print('GPU_CC=' + '.'.join(str(x) for x in torch.cuda.get_device_capability(0))); "
        "print('GPU_MEM_MB=' + str(p.total_memory // (1024 * 1024)))"
    )
    return _outcome(run_check(runner, f"{python_executable} -c '{command}'"))


def probe_wsl_interface(runner: Runner) -> CheckOutcome:
    return _outcome(
        run_check(runner, "hostname -I 2>/dev/null | awk '{print $1}'")
    )


def parse_env_list(output: str | None) -> list[str]:
    names: list[str] = []
    if not output:
        return names
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.append(stripped.split()[0])
    return names


def parse_active_env(output: str | None) -> tuple[str | None, str | None]:
    if not output:
        return None, None
    parts = output.split("|")
    environment = None if parts[0] == "none" else parts[0]
    prefix = None if len(parts) < 2 or parts[1] == "none" else parts[1]
    return environment, prefix


def parse_key_values(output: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not output:
        return result
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result