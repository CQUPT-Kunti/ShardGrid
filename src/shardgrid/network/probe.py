"""Bidirectional pairwise network probing between GPU Workers (T043).

For each direction (RTX 4060 -> GTX 1650 and reverse) this module measures
TCP reachability, ping latency, and real iperf3 throughput, plus the actual
interface and addresses, through the T040 runtime wrapper.  It never infers
bandwidth or latency from ping success or link speed, and it saves the raw
command output for T044 to build NetworkState.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from shardgrid.common.process import ProcessResult
from shardgrid.network.mtu import (
    DF_OVERSIZE_PAYLOAD_IPV4,
    DF_SAFE_PAYLOAD_IPV4,
    STATUS_PASS,
    STATUS_UNAVAILABLE,
    check_nccl_path_mtu,
    classify_df_ping,
)
from shardgrid.transport.runtime import WSLRuntimeWrapper

Runner = Callable[[Sequence[str] | str], ProcessResult]


@dataclass(frozen=True)
class LinkProbeResult:
    source_worker_id: str
    target_worker_id: str
    source_ip: str | None
    target_ip: str
    interface: str | None
    port: int
    tcp_reachable: bool
    latency_ms: float | None
    bandwidth_mbps: float | None
    interface_mtu: int | None
    expected_mtu: int | None
    mtu_status: str | None
    status: str
    failure_reason: str | None = None
    commands: tuple[str, ...] = ()
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_worker_id": self.source_worker_id,
            "target_worker_id": self.target_worker_id,
            "source_ip": self.source_ip,
            "target_ip": self.target_ip,
            "interface": self.interface,
            "port": self.port,
            "tcp_reachable": self.tcp_reachable,
            "latency_ms": self.latency_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "interface_mtu": self.interface_mtu,
            "expected_mtu": self.expected_mtu,
            "mtu_status": self.mtu_status,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "commands": list(self.commands),
            "raw_output": self.raw_output,
        }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def discover_source_ip(runner: Runner) -> str | None:
    result = _run(runner, ["hostname", "-I"])
    if not result.ok:
        return None
    for address in result.stdout.split():
        if not address.startswith("127."):
            return address
    return None


def discover_interface(runner: Runner, target_ip: str) -> str | None:
    result = _run(runner, ["ip", "route", "get", target_ip])
    if not result.ok:
        return None
    match = re.search(r"\bdev\s+(\S+)", result.stdout)
    return match.group(1) if match else None


def read_interface_link(runner: Runner, interface: str) -> str | None:
    result = _run(runner, ["ip", "link", "show", "dev", interface])
    return result.stdout.strip() if result.ok else None


def probe_tcp(runner: Runner, target_ip: str, port: int) -> tuple[bool, str]:
    code = (
        "import socket; "
        f"s=socket.create_connection(('{target_ip}',{port}), timeout=5); "
        "print('tcp_ok')"
    )
    result = _run(runner, ["python", "-c", code])
    raw = result.stdout.strip() or result.stderr.strip()
    return result.ok and "tcp_ok" in raw, raw


def parse_ping_latency(output: str) -> float | None:
    times = [float(value) for value in re.findall(r"time=([0-9.]+)\s*ms", output)]
    if not times:
        return None
    return round(sum(times) / len(times), 3)


def probe_ping(runner: Runner, target_ip: str, count: int = 3) -> tuple[float | None, str]:
    result = _run(runner, ["ping", "-c", str(count), target_ip])
    raw = result.stdout.strip() or result.stderr.strip()
    return parse_ping_latency(raw), raw


def probe_df_ping(runner: Runner, target_ip: str, payload_size: int) -> tuple[str, str]:
    result = _run(runner, ["ping", "-M", "do", "-c", "1", "-s", str(payload_size), target_ip])
    raw = result.stdout.strip() or result.stderr.strip()
    return classify_df_ping(raw, result.exit_code), raw


def parse_iperf3_json(output: str) -> float | None:
    try:
        payload = json.loads(output)
    except ValueError:
        return None
    end = payload.get("end") or {}
    sum_received = end.get("sum_received") or {}
    bits_per_second = sum_received.get("bits_per_second")
    if bits_per_second is None:
        return None
    return round(bits_per_second / 1_000_000.0, 3)


def probe_iperf3_client(
    runner: Runner,
    target_ip: str,
    port: int,
    duration: int = 5,
) -> tuple[float | None, str]:
    result = _run(
        runner,
        ["iperf3", "-c", target_ip, "-p", str(port), "-t", str(duration), "-J"],
    )
    raw = result.stdout.strip() or result.stderr.strip()
    if not result.ok:
        return None, raw
    return parse_iperf3_json(raw), raw


def start_iperf3_server(runner: Runner, port: int) -> ProcessResult:
    return _run(
        runner,
        [
            "nohup",
            "iperf3",
            "-s",
            "-p",
            str(port),
            ">",
            "/tmp/sg-iperf3-server.log",
            "2>&1",
            "&",
        ],
    )


def stop_iperf3_server(runner: Runner, port: int) -> ProcessResult:
    return _run(runner, ["pkill", "-f", f"iperf3 -s -p {port}"])


def probe_direction(
    source_runner: Runner,
    *,
    source_worker_id: str,
    target_worker_id: str,
    target_ip: str,
    tcp_port: int,
    iperf3_port: int,
    expected_mtu: int = 1500,
    ping_count: int = 3,
    iperf_duration: int = 5,
) -> LinkProbeResult:
    commands: list[str] = []
    failures: list[str] = []

    source_ip = discover_source_ip(source_runner)
    route_result = _run(source_runner, ["ip", "route", "get", target_ip])
    route_output = route_result.stdout.strip() or route_result.stderr.strip()
    interface = discover_interface(source_runner, target_ip)
    link_output = read_interface_link(source_runner, interface) if interface else None
    mtu_check = check_nccl_path_mtu(
        peer_ip=target_ip,
        route_output=route_output,
        link_output=link_output,
        expected_mtu=expected_mtu,
    )
    commands.extend(
        [
            "hostname -I",
            f"ip route get {target_ip}",
        ]
    )
    if interface:
        commands.append(f"ip link show dev {interface}")
    if mtu_check.failure_reason():
        failures.append(mtu_check.failure_reason() or "")

    df_1472_status, df_1472_raw = probe_df_ping(source_runner, target_ip, DF_SAFE_PAYLOAD_IPV4)
    df_1473_status, df_1473_raw = probe_df_ping(source_runner, target_ip, DF_OVERSIZE_PAYLOAD_IPV4)
    commands.extend(
        [
            f"ping -M do -c 1 -s {DF_SAFE_PAYLOAD_IPV4} {target_ip}",
            f"ping -M do -c 1 -s {DF_OVERSIZE_PAYLOAD_IPV4} {target_ip}",
        ]
    )
    if df_1472_status not in {STATUS_PASS, STATUS_UNAVAILABLE}:
        failures.append(f"df ping {DF_SAFE_PAYLOAD_IPV4} failed toward {target_ip}")
    if df_1473_status == STATUS_PASS:
        failures.append(
            f"df ping {DF_OVERSIZE_PAYLOAD_IPV4} unexpectedly passed toward {target_ip}"
        )

    tcp_reachable, tcp_raw = probe_tcp(source_runner, target_ip, tcp_port)
    commands.append(f"tcp {target_ip}:{tcp_port}")
    if not tcp_reachable:
        failures.append(f"tcp unreachable: {tcp_raw[:120]}")

    latency_ms, ping_raw = probe_ping(source_runner, target_ip, ping_count)
    commands.append(f"ping -c {ping_count} {target_ip}")
    if latency_ms is None:
        failures.append(f"ping latency unavailable: {ping_raw[:120]}")

    bandwidth_mbps, iperf_raw = probe_iperf3_client(
        source_runner, target_ip, iperf3_port, iperf_duration
    )
    commands.append(f"iperf3 -c {target_ip} -p {iperf3_port} -t {iperf_duration}")
    if bandwidth_mbps is None:
        failures.append(f"iperf3 throughput unavailable: {iperf_raw[:120]}")

    raw_output = "\n".join(
        part
        for part in (
            route_output,
            link_output or "",
            f"DF_PING_{DF_SAFE_PAYLOAD_IPV4}={df_1472_status} {df_1472_raw}".strip(),
            f"DF_PING_{DF_OVERSIZE_PAYLOAD_IPV4}={df_1473_status} {df_1473_raw}".strip(),
            tcp_raw,
            ping_raw,
            iperf_raw,
        )
        if part
    )
    if tcp_reachable and failures:
        status = "degraded"
    elif tcp_reachable:
        status = "ok"
    else:
        status = "unreachable"
    return LinkProbeResult(
        source_worker_id=source_worker_id,
        target_worker_id=target_worker_id,
        source_ip=source_ip,
        target_ip=target_ip,
        interface=interface,
        port=iperf3_port,
        tcp_reachable=tcp_reachable,
        latency_ms=latency_ms,
        bandwidth_mbps=bandwidth_mbps,
        interface_mtu=mtu_check.interface_mtu,
        expected_mtu=expected_mtu,
        mtu_status=mtu_check.status,
        status=status,
        failure_reason="; ".join(failures) if failures else None,
        commands=tuple(commands),
        raw_output=raw_output,
    )


def run_pairwise_probe(
    source_wrapper: WSLRuntimeWrapper,
    target_wrapper: WSLRuntimeWrapper,
    *,
    source_worker_id: str,
    target_worker_id: str,
    target_ip: str,
    tcp_port: int,
    iperf3_port: int,
    expected_mtu: int = 1500,
) -> LinkProbeResult:
    """Run one direction, managing the iperf3 server on the target."""
    start_iperf3_server(target_wrapper.run, iperf3_port)
    result = probe_direction(
        source_wrapper.run,
        source_worker_id=source_worker_id,
        target_worker_id=target_worker_id,
        target_ip=target_ip,
        tcp_port=tcp_port,
        iperf3_port=iperf3_port,
        expected_mtu=expected_mtu,
    )
    stop_iperf3_server(target_wrapper.run, iperf3_port)
    return result


def save_probe_output(result: LinkProbeResult, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"link-{result.source_worker_id}-{result.target_worker_id}-{now_utc()}.json"
    )
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    (directory / "latest.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True)
    )
    return path
