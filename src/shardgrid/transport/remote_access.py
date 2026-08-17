"""Shared SSH remote-access check for Windows GPU Workers (T038/T039).

Both GPU Workers use exactly the same transport contract and check logic:
``Machine A -> SSHTransport -> Windows host -> WSL2 Ubuntu -> selected Conda
environment -> runtime Python``.  The check stops at identity/runtime evidence;
GPU depth probing belongs to later tasks.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from shardgrid.common.config import WorkerConfig
from shardgrid.common.enums import FailureStage
from shardgrid.common.process import ProcessResult
from shardgrid.transport.ssh import SSHTransport


@dataclass(frozen=True)
class RemoteRuntimeIdentity:
    windows_identity: str
    wsl_distro: str
    conda_executable: str
    conda_environment: str
    conda_prefix: str
    python_executable: str
    python_version: str


@dataclass(frozen=True)
class RemoteAccessResult:
    status: str
    worker_id: str
    host: str
    ssh_user: str
    transport: SSHTransport
    commands: tuple[str, ...]
    windows_identity: str | None = None
    wsl_distro: str | None = None
    runtime_identity: RemoteRuntimeIdentity | None = None
    failure_category: str | None = None
    failure_reason: str | None = None
    failure_record: dict[str, Any] | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None


def classify_connection_failure(
    result: ProcessResult, worker_label: str
) -> tuple[str, str, str]:
    text = f"{result.stdout}\n{result.stderr}".lower()
    if result.timed_out or "connection timed out" in text:
        return (
            "connection_timeout",
            f"SSH connection to the {worker_label} Worker timed out",
            "verify network reachability and the Worker SSH service, then rerun the check",
        )
    if "permission denied" in text:
        return (
            "authentication_failure",
            "SSH authentication failed before the Windows host command could run",
            "authorize Machine A for the Worker OpenSSH account and rerun the check",
        )
    if "host key verification failed" in text or "remote host identification has changed" in text:
        return (
            "known_host_failure",
            "SSH host-key validation failed",
            "fix the Machine A known_hosts entry for the Worker and rerun the check",
        )
    if (
        "no route to host" in text
        or "name or service not known" in text
        or "could not resolve hostname" in text
    ):
        return (
            "host_unreachable",
            f"Machine A could not reach the {worker_label} Worker host",
            "verify the configured Worker address from Machine A and rerun the check",
        )
    return (
        "remote_command_non_zero_exit",
        "SSH command failed before runtime validation completed",
        "inspect stderr and the Worker OpenSSH state, then rerun the check",
    )


def _parse_default_wsl_distro(output: str) -> str | None:
    for raw_line in output.splitlines():
        line = raw_line.replace("\x00", "").strip()
        if not line.startswith("*"):
            continue
        fields = line.replace("*", "", 1).split()
        if fields:
            return fields[0]
    return None


def _runtime_probe_script() -> str:
    return (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "if command -v conda >/dev/null 2>&1; then "
        "command -v conda; "
        "elif [ -x \"$HOME/miniconda3/bin/conda\" ]; then "
        "printf %s \"$HOME/miniconda3/bin/conda\"; "
        "elif [ -x \"$HOME/anaconda3/bin/conda\" ]; then "
        "printf %s \"$HOME/anaconda3/bin/conda\"; "
        "else exit 12; fi"
    )


def _runtime_active_script() -> str:
    return (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "printf '%s|%s' \"${CONDA_DEFAULT_ENV:-none}\" \"${CONDA_PREFIX:-none}\""
    )


def _env_list_script(conda_executable: str) -> str:
    return f"{conda_executable} env list"


def _python_version_script(python_executable: str) -> str:
    return f"{python_executable} --version 2>&1"


def _python_identity_script(python_executable: str) -> str:
    return f"{python_executable} -c 'import sys; print(sys.executable)'"


def wrap_wsl_runtime_command(distro: str, user: str, command: str) -> str:
    """Wrap a WSL runtime command so quoting survives every layer.

    The command is base64-encoded and decoded inside WSL before execution.
    Base64 contains no quotes, spaces, ``$``, or shell metacharacters, so the
    PowerShell -> wsl.exe -> bash -lc chain cannot corrupt the payload
    (``$PATH``, parentheses, and mixed quotes are all preserved literally).
    """
    payload_b64 = base64.b64encode(command.encode("utf-8")).decode("ascii")
    powershell = (
        f"$distro = '{distro}'; "
        f"$user = '{user}'; "
        f"$payload = '{payload_b64}'; "
        'wsl.exe -d $distro -u $user -- /bin/bash -lc '
        '"echo $payload | base64 -d | /bin/bash"; '
        "exit $LASTEXITCODE"
    )
    encoded = base64.b64encode(powershell.encode("utf-16le")).decode("ascii")
    return f"powershell -NoProfile -EncodedCommand {encoded}"


def _select_runtime_identity(
    *,
    preferred_environment: str | None,
    active_environment: str | None,
    active_prefix: str | None,
    env_list_output: str,
) -> tuple[str | None, str | None]:
    env_map: dict[str, str] = {}
    for line in env_list_output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            if parts[1] == "*":
                if len(parts) >= 3:
                    env_map[parts[0]] = parts[2]
            else:
                env_map[parts[0]] = parts[1]
    if preferred_environment and preferred_environment in env_map:
        return preferred_environment, env_map[preferred_environment]
    if "shardgrid" in env_map:
        return "shardgrid", env_map["shardgrid"]
    if active_environment and active_prefix and active_environment != "base":
        return active_environment, active_prefix
    return None, None


def run_remote_access_check(
    transport: SSHTransport,
    worker: WorkerConfig,
    *,
    worker_label: str,
    preferred_environment: str | None = None,
) -> RemoteAccessResult:
    commands: list[str] = []

    windows_identity = transport.run(["hostname"])
    commands.append(windows_identity.recorded_command)
    if not windows_identity.ok:
        category, message, recommended_action = classify_connection_failure(
            windows_identity, worker_label
        )
        failure = transport.to_failure_record(
            windows_identity,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message=message,
            recommended_action=recommended_action,
            conda_environment=worker.conda_environment,
            conda_prefix=worker.conda_prefix,
        )
        return RemoteAccessResult(
            status="BLOCKED",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            failure_category=category,
            failure_reason=message,
            failure_record=failure.to_dict(),
            stdout=windows_identity.stdout,
            stderr=windows_identity.stderr,
            exit_code=windows_identity.exit_code,
        )

    wsl_list = transport.run(["wsl.exe", "-l", "-v"])
    commands.append(wsl_list.recorded_command)
    if not wsl_list.ok:
        failure = transport.to_failure_record(
            wsl_list,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="Windows host is reachable but WSL is unavailable",
            recommended_action=(
                "verify WSL2 and the Ubuntu distro on the Worker, then rerun the check"
            ),
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=windows_identity.stdout.strip() or None,
            failure_category="wsl_unavailable",
            failure_reason="Windows host is reachable but WSL is unavailable",
            failure_record=failure.to_dict(),
            stdout=wsl_list.stdout,
            stderr=wsl_list.stderr,
            exit_code=wsl_list.exit_code,
        )

    distro = _parse_default_wsl_distro(wsl_list.stdout)
    if distro is None:
        failure = transport.to_failure_record(
            wsl_list,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="WSL is reachable but the default distro could not be determined",
            recommended_action=(
                "set or repair the default Ubuntu WSL distro on the Worker, "
                "then rerun the check"
            ),
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=windows_identity.stdout.strip() or None,
            failure_category="wsl_distro_unavailable",
            failure_reason="WSL is reachable but the default distro could not be determined",
            failure_record=failure.to_dict(),
            stdout=wsl_list.stdout,
            stderr=wsl_list.stderr,
            exit_code=wsl_list.exit_code,
        )

    runtime_command = wrap_wsl_runtime_command(
        distro,
        worker.ssh_user,
        _runtime_probe_script(),
    )
    runtime_conda = transport.run(runtime_command)
    commands.append(runtime_conda.recorded_command)
    if not runtime_conda.ok:
        failure = transport.to_failure_record(
            runtime_conda,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="WSL is reachable but Conda is unavailable in the training runtime",
            recommended_action=(
                "install or expose Conda in the WSL training runtime, "
                "then rerun the check"
            ),
            conda_environment=preferred_environment or worker.conda_environment,
            conda_prefix=worker.conda_prefix,
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=windows_identity.stdout.strip() or None,
            wsl_distro=distro,
            failure_category="conda_unavailable",
            failure_reason="WSL is reachable but Conda is unavailable in the training runtime",
            failure_record=failure.to_dict(),
            stdout=runtime_conda.stdout,
            stderr=runtime_conda.stderr,
            exit_code=runtime_conda.exit_code,
        )

    conda_executable = runtime_conda.stdout.strip()
    active_command = wrap_wsl_runtime_command(
        distro,
        worker.ssh_user,
        _runtime_active_script(),
    )
    runtime_active = transport.run(active_command)
    commands.append(runtime_active.recorded_command)
    if not runtime_active.ok:
        failure = transport.to_failure_record(
            runtime_active,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="WSL is reachable but the active Conda state could not be read",
            recommended_action="inspect the remote WSL Conda shell setup and rerun the check",
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=windows_identity.stdout.strip() or None,
            wsl_distro=distro,
            failure_category="remote_command_non_zero_exit",
            failure_reason="WSL is reachable but the active Conda state could not be read",
            failure_record=failure.to_dict(),
            stdout=runtime_active.stdout,
            stderr=runtime_active.stderr,
            exit_code=runtime_active.exit_code,
        )

    if "|" in runtime_active.stdout.strip():
        active_environment, active_prefix = runtime_active.stdout.strip().split("|", 1)
    else:
        active_environment, active_prefix = "none", "none"
    env_list_command = wrap_wsl_runtime_command(
        distro,
        worker.ssh_user,
        _env_list_script(conda_executable),
    )
    runtime_env_list = transport.run(env_list_command)
    commands.append(runtime_env_list.recorded_command)
    if not runtime_env_list.ok:
        failure = transport.to_failure_record(
            runtime_env_list,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="WSL is reachable but the Conda environment list could not be read",
            recommended_action="inspect the remote WSL Conda installation and rerun the check",
            conda_environment=preferred_environment or worker.conda_environment,
            conda_prefix=worker.conda_prefix,
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=windows_identity.stdout.strip() or None,
            wsl_distro=distro,
            failure_category="remote_command_non_zero_exit",
            failure_reason="WSL is reachable but the Conda environment list could not be read",
            failure_record=failure.to_dict(),
            stdout=runtime_env_list.stdout,
            stderr=runtime_env_list.stderr,
            exit_code=runtime_env_list.exit_code,
        )

    selected_environment, selected_prefix = _select_runtime_identity(
        preferred_environment=preferred_environment or worker.conda_environment,
        active_environment=None if active_environment == "none" else active_environment,
        active_prefix=None if active_prefix == "none" else active_prefix,
        env_list_output=runtime_env_list.stdout,
    )
    if selected_environment is None or selected_prefix is None:
        failure = transport.to_failure_record(
            runtime_env_list,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="WSL is reachable but the selected Conda environment is unavailable",
            recommended_action=(
                "create or restore the selected WSL Conda environment, "
                "then rerun the check"
            ),
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=windows_identity.stdout.strip() or None,
            wsl_distro=distro,
            failure_category="selected_conda_environment_unavailable",
            failure_reason="WSL is reachable but the selected Conda environment is unavailable",
            failure_record=failure.to_dict(),
            stdout=runtime_env_list.stdout,
            stderr=runtime_env_list.stderr,
            exit_code=runtime_env_list.exit_code,
        )

    python_executable = f"{selected_prefix}/bin/python"
    python_version_command = wrap_wsl_runtime_command(
        distro,
        worker.ssh_user,
        _python_version_script(python_executable),
    )
    runtime_python_version = transport.run(python_version_command)
    commands.append(runtime_python_version.recorded_command)
    if not runtime_python_version.ok:
        failure = transport.to_failure_record(
            runtime_python_version,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="WSL is reachable but the runtime Python could not be executed",
            recommended_action=(
                "repair the selected WSL Conda environment Python and rerun the check"
            ),
            conda_environment=selected_environment,
            conda_prefix=selected_prefix,
            python_executable=python_executable,
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=windows_identity.stdout.strip() or None,
            wsl_distro=distro,
            failure_category="remote_command_non_zero_exit",
            failure_reason="WSL is reachable but the runtime Python could not be executed",
            failure_record=failure.to_dict(),
            stdout=runtime_python_version.stdout,
            stderr=runtime_python_version.stderr,
            exit_code=runtime_python_version.exit_code,
        )

    python_identity_command = wrap_wsl_runtime_command(
        distro,
        worker.ssh_user,
        _python_identity_script(python_executable),
    )
    runtime_python_identity = transport.run(python_identity_command)
    commands.append(runtime_python_identity.recorded_command)
    if not runtime_python_identity.ok:
        failure = transport.to_failure_record(
            runtime_python_identity,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="WSL is reachable but the runtime Python identity could not be read",
            recommended_action=(
                "repair the selected WSL Conda environment Python and rerun the check"
            ),
            conda_environment=selected_environment,
            conda_prefix=selected_prefix,
            python_executable=python_executable,
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=windows_identity.stdout.strip() or None,
            wsl_distro=distro,
            failure_category="remote_command_non_zero_exit",
            failure_reason="WSL is reachable but the runtime Python identity could not be read",
            failure_record=failure.to_dict(),
            stdout=runtime_python_identity.stdout,
            stderr=runtime_python_identity.stderr,
            exit_code=runtime_python_identity.exit_code,
        )

    identity = RemoteRuntimeIdentity(
        windows_identity=(windows_identity.stdout.strip() or str(worker.host)),
        wsl_distro=distro,
        conda_executable=conda_executable,
        conda_environment=selected_environment,
        conda_prefix=selected_prefix,
        python_executable=runtime_python_identity.stdout.strip(),
        python_version=runtime_python_version.stdout.strip(),
    )
    if not identity.python_executable.startswith(identity.conda_prefix.rstrip("/") + "/"):
        failure = transport.to_failure_record(
            runtime_python_identity,
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message="Remote Python is not inside the selected WSL Conda environment",
            recommended_action=(
                "fix the WSL runtime selection so Python comes from the "
                "selected Conda prefix, then rerun the check"
            ),
            conda_environment=identity.conda_environment,
            conda_prefix=identity.conda_prefix,
            python_executable=identity.python_executable,
        )
        return RemoteAccessResult(
            status="FAIL",
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            transport=transport,
            commands=tuple(commands),
            windows_identity=identity.windows_identity,
            wsl_distro=identity.wsl_distro,
            failure_category="runtime_python_outside_selected_conda",
            failure_reason="Remote Python is not inside the selected WSL Conda environment",
            failure_record=failure.to_dict(),
            stdout=runtime_python_identity.stdout,
            stderr=runtime_python_identity.stderr,
            exit_code=runtime_python_identity.exit_code,
        )

    return RemoteAccessResult(
        status="PASS",
        worker_id=str(worker.worker_id),
        host=str(worker.host),
        ssh_user=worker.ssh_user,
        transport=transport,
        commands=tuple(commands),
        windows_identity=identity.windows_identity,
        wsl_distro=identity.wsl_distro,
        runtime_identity=identity,
        stdout=runtime_python_identity.stdout,
        stderr=runtime_python_identity.stderr,
        exit_code=runtime_python_identity.exit_code,
    )