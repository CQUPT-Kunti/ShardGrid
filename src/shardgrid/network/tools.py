"""Network tool discovery for the control node and WSL2 training runtimes (T042).

Discovery only reports whether tools are present and which versions were
detected.  It never guesses bandwidth, latency, or reachability, and it never
fabricates a NetworkState.  The same runner abstraction covers the Ubuntu
control node (``LinuxPlatform``) and remote WSL2 runtimes (the T040 runtime
wrapper).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from shardgrid.common.process import ProcessResult
from shardgrid.transport.runtime import WSLRuntimeWrapper

Runner = Callable[[Sequence[str] | str], ProcessResult]


@dataclass(frozen=True)
class ToolStatus:
    name: str
    status: str
    version: str | None = None
    reason: str | None = None
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "version": self.version,
            "reason": self.reason,
            "command": self.command,
        }


@dataclass(frozen=True)
class PythonHelperInfo:
    executable: str | None
    conda_environment: str | None
    conda_prefix: str | None
    conda_managed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "conda_environment": self.conda_environment,
            "conda_prefix": self.conda_prefix,
            "conda_managed": self.conda_managed,
        }


@dataclass(frozen=True)
class NetworkToolsReport:
    target: str
    ping: ToolStatus
    iperf3: ToolStatus
    tcp_socket: ToolStatus
    interfaces: tuple[str, ...]
    python_helper: PythonHelperInfo
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "ping": self.ping.to_dict(),
            "iperf3": self.iperf3.to_dict(),
            "tcp_socket": self.tcp_socket.to_dict(),
            "interfaces": list(self.interfaces),
            "python_helper": self.python_helper.to_dict(),
            "timestamp": self.timestamp,
        }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_iperf3_version(output: str | None) -> str | None:
    if not output:
        return None
    match = re.search(r"^iperf(?:\s+version)?\s+(\S+)", output)
    return match.group(1) if match else output.splitlines()[0] if output else None


def extract_ping_version(output: str | None) -> str | None:
    if not output:
        return None
    match = re.search(r"ping from iputils\s+(\S+)", output)
    return match.group(1) if match else output.splitlines()[0] if output else None


def _run(runner: Runner, command: Sequence[str] | str) -> ProcessResult:
    try:
        return runner(list(command) if not isinstance(command, str) else command)
    except Exception as exc:  # pragma: no cover - runner boundary
        return ProcessResult(
            args=(),
            recorded_command=str(command),
            shell=False,
            cwd=None,
            exit_code=-1,
            stdout="",
            stderr=str(exc),
            timed_out=False,
            runtime_environment={},
        )


def check_tool(
    runner: Runner,
    name: str,
    version_command: Sequence[str],
    *,
    version_parser: Callable[[str | None], str | None] | None = None,
) -> ToolStatus:
    command = " ".join(version_command)
    result = _run(runner, version_command)
    if result.ok:
        text = (result.stdout or result.stderr).strip()
        version = (
            version_parser(text)
            if version_parser
            else (text.splitlines()[0] if text else None)
        )
        return ToolStatus(name=name, status="available", version=version, command=command)
    reason = result.stderr.strip() or f"exit code {result.exit_code}"
    return ToolStatus(
        name=name,
        status="unavailable",
        reason=reason[:200],
        command=command,
    )


def check_tcp_socket(runner: Runner, python_command: Sequence[str]) -> ToolStatus:
    code = (
        "import socket; "
        "socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
        "print('tcp_socket_ok')"
    )
    command = [*python_command, "-c", code]
    result = _run(runner, command)
    if result.ok and "tcp_socket_ok" in result.stdout:
        return ToolStatus(
            name="tcp_socket",
            status="available",
            command=" ".join(command),
        )
    return ToolStatus(
        name="tcp_socket",
        status="unavailable",
        reason=(result.stderr or "tcp socket check failed").strip()[:200],
        command=" ".join(command),
    )


def discover_interfaces(runner: Runner, python_command: Sequence[str]) -> tuple[str, ...]:
    code = "import socket; print(' '.join(i[1] for i in socket.if_nameindex()))"
    result = _run(runner, [*python_command, "-c", code])
    if result.ok:
        return tuple(result.stdout.strip().split())
    return ()


def discover_network_tools(
    runner: Runner,
    *,
    target: str,
    python_command: Sequence[str],
    python_executable: str | None,
    conda_environment: str | None,
    conda_prefix: str | None,
) -> NetworkToolsReport:
    ping = check_tool(
        runner,
        "ping",
        ["ping", "-V"],
        version_parser=extract_ping_version,
    )
    iperf3 = check_tool(
        runner,
        "iperf3",
        ["iperf3", "--version"],
        version_parser=extract_iperf3_version,
    )
    tcp_socket = check_tcp_socket(runner, python_command)
    interfaces = discover_interfaces(runner, python_command)
    helper = PythonHelperInfo(
        executable=python_executable,
        conda_environment=conda_environment,
        conda_prefix=conda_prefix,
        conda_managed=bool(conda_prefix or conda_environment),
    )
    return NetworkToolsReport(
        target=target,
        ping=ping,
        iperf3=iperf3,
        tcp_socket=tcp_socket,
        interfaces=interfaces,
        python_helper=helper,
        timestamp=now_utc(),
    )


def discover_control_network_tools() -> NetworkToolsReport:
    """Live discovery on the Ubuntu control node (Machine A)."""
    from shardgrid.platforms.linux import LinuxPlatform
    from shardgrid.workers.environment_report import detect_conda

    conda_executable, conda_environment, conda_prefix = detect_conda()
    return discover_network_tools(
        LinuxPlatform().run,
        target="control",
        python_command=[sys.executable],
        python_executable=sys.executable,
        conda_environment=conda_environment,
        conda_prefix=conda_prefix,
    )


def discover_worker_network_tools(
    wrapper: WSLRuntimeWrapper,
    *,
    worker_id: str,
) -> NetworkToolsReport:
    """Live discovery inside a WSL2 training runtime through the T040 wrapper."""
    prefix = wrapper.config.conda_prefix
    environment = wrapper.config.conda_environment
    return discover_network_tools(
        lambda command: wrapper.run(command, timeout=60),
        target=f"worker:{worker_id}",
        python_command=["python"],
        python_executable=f"{prefix}/bin/python" if prefix else None,
        conda_environment=environment,
        conda_prefix=prefix,
    )