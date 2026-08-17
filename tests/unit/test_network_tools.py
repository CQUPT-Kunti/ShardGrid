from __future__ import annotations

from typing import Sequence

from shardgrid.common.process import ProcessResult
from shardgrid.network.tools import (
    check_tcp_socket,
    check_tool,
    discover_interfaces,
    discover_network_tools,
    extract_iperf3_version,
    extract_ping_version,
)


def _result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> ProcessResult:
    return ProcessResult(
        args=(),
        recorded_command="",
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        runtime_environment={},
    )


class FakeRunner:
    def __init__(self, responses: dict[str, ProcessResult]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def run(self, command: Sequence[str] | str) -> ProcessResult:
        text = command if isinstance(command, str) else " ".join(command)
        self.calls.append(text)
        for needle, result in self.responses.items():
            if needle in text:
                return result
        return _result(exit_code=127, stderr="command not found")


def test_ping_available_records_version() -> None:
    runner = FakeRunner(
        {"ping -V": _result(stdout="ping from iputils 20240117\n")}
    )

    status = check_tool(
        runner.run, "ping", ["ping", "-V"], version_parser=extract_ping_version
    )

    assert status.status == "available"
    assert status.version == "20240117"
    assert status.command == "ping -V"


def test_ping_unavailable_reports_reason() -> None:
    runner = FakeRunner(
        {"ping -V": _result(stderr="ping: command not found", exit_code=127)}
    )

    status = check_tool(
        runner.run, "ping", ["ping", "-V"], version_parser=extract_ping_version
    )

    assert status.status == "unavailable"
    assert status.reason is not None
    assert "command not found" in status.reason


def test_iperf3_available_records_version() -> None:
    runner = FakeRunner(
        {"iperf3 --version": _result(stdout="iperf 3.16 (cJSON 1.7.15)\n")}
    )

    status = check_tool(
        runner.run, "iperf3", ["iperf3", "--version"], version_parser=extract_iperf3_version
    )

    assert status.status == "available"
    assert status.version == "3.16"


def test_iperf3_unavailable_reports_reason() -> None:
    runner = FakeRunner(
        {"iperf3 --version": _result(stderr="iperf3: not found", exit_code=127)}
    )

    status = check_tool(
        runner.run, "iperf3", ["iperf3", "--version"], version_parser=extract_iperf3_version
    )

    assert status.status == "unavailable"
    assert status.reason is not None
    assert "not found" in status.reason


def test_version_parsing() -> None:
    assert extract_iperf3_version("iperf 3.16 (cJSON 1.7.15)") == "3.16"
    assert extract_iperf3_version("iperf version 3.20 (cJSON 1.7.15)") == "3.20"
    assert extract_ping_version("ping from iputils 20240117") == "20240117"
    assert extract_iperf3_version(None) is None
    assert extract_ping_version(None) is None


def test_tcp_socket_capability_available() -> None:
    runner = FakeRunner(
        {"tcp_socket_ok": _result(stdout="tcp_socket_ok\n")}
    )

    status = check_tcp_socket(runner.run, ["python"])

    assert status.status == "available"


def test_tcp_socket_capability_unavailable() -> None:
    runner = FakeRunner(
        {"tcp_socket_ok": _result(stderr="python: command not found", exit_code=127)}
    )

    status = check_tcp_socket(runner.run, ["python"])

    assert status.status == "unavailable"
    assert status.reason


def test_interface_discovery() -> None:
    runner = FakeRunner({"if_nameindex": _result(stdout="lo eth0\n")})

    interfaces = discover_interfaces(runner.run, ["python"])

    assert interfaces == ("lo", "eth0")


def test_interface_discovery_failure_returns_empty() -> None:
    runner = FakeRunner({"if_nameindex": _result(exit_code=1, stderr="boom")})

    assert discover_interfaces(runner.run, ["python"]) == ()


def test_discover_network_tools_records_conda_identity() -> None:
    runner = FakeRunner(
        {
            "ping -V": _result(stdout="ping from iputils 20240117"),
            "iperf3 --version": _result(stdout="iperf 3.16 (cJSON 1.7.15)"),
            "tcp_socket_ok": _result(stdout="tcp_socket_ok"),
            "if_nameindex": _result(stdout="lo eth0"),
        }
    )

    report = discover_network_tools(
        runner.run,
        target="worker:gpu1650",
        python_command=["python"],
        python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
    )

    assert report.target == "worker:gpu1650"
    assert report.ping.status == "available"
    assert report.iperf3.status == "available"
    assert report.tcp_socket.status == "available"
    assert report.interfaces == ("lo", "eth0")
    assert report.python_helper.conda_managed is True
    assert report.python_helper.conda_environment == "shardgrid"
    assert report.python_helper.conda_prefix == "/home/shardgrid/miniconda3/envs/shardgrid"
    payload = report.to_dict()
    assert payload["ping"]["status"] == "available"
    assert payload["python_helper"]["conda_managed"] is True


def test_discover_network_tools_missing_tools_never_guessed() -> None:
    runner = FakeRunner(
        {
            "ping -V": _result(stderr="ping: not found", exit_code=127),
            "iperf3 --version": _result(stderr="iperf3: not found", exit_code=127),
            "tcp_socket_ok": _result(stdout="tcp_socket_ok"),
            "if_nameindex": _result(stdout="lo"),
        }
    )

    report = discover_network_tools(
        runner.run,
        target="control",
        python_command=["python"],
        python_executable="/opt/conda/bin/python",
        conda_environment="base",
        conda_prefix="/opt/conda",
    )

    assert report.ping.status == "unavailable"
    assert report.iperf3.status == "unavailable"
    assert report.tcp_socket.status == "available"
    assert report.iperf3.version is None
    assert report.iperf3.reason