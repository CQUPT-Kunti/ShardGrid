"""Windows Worker -> WSL2 -> selected Conda environment command wrapper (T040).

Business code does not assemble WSL/Conda commands itself: it describes the
target command and lets :class:`WSLRuntimeWrapper` build the full
``Windows Worker -> configured WSL distro -> selected Conda environment``
execution chain.  The chain is executed through a host executor (an
``SSHTransport`` for remote Workers), so SSH and process-result behavior stay
in the existing primitives.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from shardgrid.common.config import RuntimeConfig, WorkerConfig
from shardgrid.common.enums import FailureStage
from shardgrid.common.errors import make_failure_record
from shardgrid.common.process import ProcessResult
from shardgrid.jobs.models import FailureRecord
from shardgrid.transport.remote_access import wrap_wsl_runtime_command

EXIT_CWD_NOT_FOUND = 66
EXIT_COMMAND_NOT_FOUND = 127


def wrap_wsl_direct_command(distro: str, user: str, command: str) -> str:
    """Direct ``wsl.exe`` invocation for stdin-fed payloads.

    Used by :meth:`WSLRuntimeWrapper.run_script`: the command must contain no
    ``$``, no parentheses, and no double quotes, so it survives the
    ssh -> cmd.exe -> wsl.exe -> bash chain without PowerShell involvement,
    which also keeps stdin flowing through to the Linux process.
    """
    return f'wsl.exe -d {distro} -u {user} -- /bin/bash -lc "{command}"'


class HostExecutor(Protocol):
    """Runs a Windows-side command and returns a ProcessResult."""

    def run(
        self,
        command: Sequence[str] | str,
        *,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
    ) -> ProcessResult: ...


@dataclass(frozen=True)
class WSLRuntimeConfig:
    distro: str | None = None
    user: str | None = None
    conda_executable: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None

    @classmethod
    def from_worker_and_runtime(
        cls, worker: WorkerConfig, runtime: RuntimeConfig
    ) -> WSLRuntimeConfig:
        return cls(
            distro=worker.runtime_distro or runtime.default_wsl_distro,
            user=worker.ssh_user,
            conda_executable=runtime.conda_executable,
            conda_environment=worker.conda_environment or runtime.conda_environment,
            conda_prefix=worker.conda_prefix or runtime.conda_prefix,
        )


class WSLRuntimeWrapper:
    """Builds and runs the WSL2 + selected Conda environment command chain."""

    def __init__(self, config: WSLRuntimeConfig, executor: HostExecutor) -> None:
        self.config = config
        self.executor = executor

    def _selection_preamble(self) -> list[str]:
        if self.config.conda_prefix:
            prefix_bin = shlex.quote(f"{self.config.conda_prefix}/bin")
            prefix = shlex.quote(self.config.conda_prefix)
            lines = [f"export PATH={prefix_bin}:\"$PATH\"", f"export CONDA_PREFIX={prefix}"]
            if self.config.conda_environment:
                lines.append(
                    f"export CONDA_DEFAULT_ENV={shlex.quote(self.config.conda_environment)}"
                )
            return lines
        return []

    def _command_prefix(self) -> str:
        if self.config.conda_prefix:
            return ""
        if self.config.conda_executable and self.config.conda_environment:
            return (
                f"{shlex.quote(self.config.conda_executable)} "
                f"run -n {shlex.quote(self.config.conda_environment)} -- "
            )
        return ""

    def build_payload(
        self,
        command: Sequence[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        parts = ["source ~/.bashrc >/dev/null 2>&1 || true"]
        parts.extend(self._selection_preamble())
        for key, value in (env or {}).items():
            parts.append(f"export {shlex.quote(key)}={shlex.quote(value)}")
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)} || exit {EXIT_CWD_NOT_FOUND}")
        command_line = shlex.join(command) if not isinstance(command, str) else command
        parts.append(f"{self._command_prefix()}{command_line}")
        return "; ".join(parts)

    def build_remote_command(self, payload: str) -> str:
        if not self.config.distro:
            raise ValueError("WSL distro is not configured")
        return wrap_wsl_runtime_command(
            self.config.distro,
            self.config.user or "root",
            payload,
        )

    def run(
        self,
        command: Sequence[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        secrets: Sequence[str] = (),
    ) -> ProcessResult:
        if not self.config.distro:
            raise ValueError("WSL distro is not configured")
        if not self.config.conda_prefix and not (
            self.config.conda_executable and self.config.conda_environment
        ):
            raise ValueError(
                "no Conda environment selected; configure conda_prefix or "
                "conda_executable + conda_environment"
            )
        payload = self.build_payload(command, cwd=cwd, env=env)
        remote_command = self.build_remote_command(payload)
        return self.executor.run(remote_command, timeout=timeout)

    def run_script(
        self,
        script: str,
        *,
        timeout: float | None = None,
        secrets: Sequence[str] = (),
    ) -> ProcessResult:
        """Run a Python script through the selected Conda python via stdin.

        The script is fed to ``<conda_prefix>/bin/python -`` on stdin using a
        direct ``wsl.exe`` invocation (no PowerShell layer), which preserves
        stdin and avoids both the Windows command-line length limit and any
        ``$VAR`` expansion across the chain.  Only the absolute conda python
        path is used, never Windows or system Python.  The payload contains no
        ``$``, parentheses, or double quotes by construction.
        """
        if not self.config.distro:
            raise ValueError("WSL distro is not configured")
        if not self.config.conda_prefix:
            raise ValueError("no Conda prefix configured for runtime script execution")
        python_executable = shlex.quote(f"{self.config.conda_prefix}/bin/python")
        parts = [
            "source ~/.bashrc >/dev/null 2>&1 || true",
            f"export CONDA_PREFIX={shlex.quote(self.config.conda_prefix)}",
        ]
        if self.config.conda_environment:
            parts.append(
                f"export CONDA_DEFAULT_ENV={shlex.quote(self.config.conda_environment)}"
            )
        parts.append(f"{python_executable} -")
        payload = "; ".join(parts)
        remote_command = wrap_wsl_direct_command(
            self.config.distro,
            self.config.user or "root",
            payload,
        )
        return self.executor.run(
            remote_command,
            stdin=script,
            timeout=timeout,
        )

    def classify_runtime_failure(
        self,
        result: ProcessResult,
        *,
        stage: FailureStage = FailureStage.LAUNCH,
        host: str | None = None,
        worker_id: str | None = None,
        conda_environment: str | None = None,
        conda_prefix: str | None = None,
    ) -> FailureRecord:
        if result.exit_code == EXIT_CWD_NOT_FOUND:
            message = "requested working directory does not exist in the WSL runtime"
            recommended_action = "fix the configured WSL working directory and retry"
        elif result.exit_code == EXIT_COMMAND_NOT_FOUND:
            message = "target command was not found in the selected WSL Conda environment"
            recommended_action = (
                "verify the command is installed in the selected Conda environment"
            )
        else:
            message = "runtime command failed with a non-zero exit code"
            recommended_action = "inspect stderr from the WSL runtime and retry"
        return make_failure_record(
            stage=stage,
            host=host or self.config.distro or "wsl-runtime",
            worker_id=worker_id,
            command=result.recorded_command,
            exit_code=result.exit_code,
            stderr_path=None,
            runtime_environment=dict(result.runtime_environment),
            conda_environment=conda_environment or self.config.conda_environment,
            conda_prefix=conda_prefix or self.config.conda_prefix,
            message=message,
            recommended_action=recommended_action,
            retryable=result.exit_code != EXIT_CWD_NOT_FOUND,
        )
