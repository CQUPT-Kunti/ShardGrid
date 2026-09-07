"""Minimal subprocess wrapper with stable result objects."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ProcessResult:
    args: tuple[str, ...] | str
    recorded_command: str
    shell: bool
    cwd: str | None
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    runtime_environment: dict[str, str]

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass(frozen=True)
class ProcessTimeoutError(RuntimeError):
    result: ProcessResult

    def __str__(self) -> str:
        return f"process timed out: {self.result.recorded_command}"


def _coerce_output(value: bytes | str | None, encoding: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(encoding, errors="replace")
    return value


def redact_text(text: str | None, secrets: Sequence[str] = ()) -> str | None:
    if text is None:
        return None
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "***")
    return redacted


def redact_command(command: Sequence[str] | str, secrets: Sequence[str] = ()) -> str:
    rendered = command if isinstance(command, str) else shlex.join(command)
    return redact_text(rendered, secrets) or rendered



def run_process(
    command: Sequence[str] | str,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    shell: bool = False,
    secrets: Sequence[str] = (),
    check: bool = False,
    encoding: str = "utf-8",
    runtime_environment: Mapping[str, str] | None = None,
    input: str | bytes | None = None,
) -> ProcessResult:
    completed = None
    recorded_command = redact_command(command, secrets)
    cwd_text = None if cwd is None else str(cwd)
    args = (
        tuple(command)
        if isinstance(command, Sequence) and not isinstance(command, str)
        else command
    )

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=None if env is None else dict(env),
            timeout=timeout,
            shell=shell,
            capture_output=True,
            text=True,
            encoding=encoding,
            errors="replace",
            check=False,
            input=input,
        )
        result = ProcessResult(
            args=args,
            recorded_command=recorded_command,
            shell=shell,
            cwd=cwd_text,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
            runtime_environment={} if runtime_environment is None else dict(runtime_environment),
        )
    except subprocess.TimeoutExpired as exc:
        result = ProcessResult(
            args=args,
            recorded_command=recorded_command,
            shell=shell,
            cwd=cwd_text,
            exit_code=-1,
            stdout=_coerce_output(exc.stdout, encoding),
            stderr=_coerce_output(exc.stderr, encoding),
            timed_out=True,
            runtime_environment={} if runtime_environment is None else dict(runtime_environment),
        )
        if check:
            raise ProcessTimeoutError(result) from exc
        return result

    if check and not result.ok:
        raise subprocess.CalledProcessError(
            returncode=result.exit_code,
            cmd=result.recorded_command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result
