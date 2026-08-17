from __future__ import annotations

import json
from typing import Sequence

from shardgrid.common.process import ProcessResult
from shardgrid.network.probe import (
    parse_iperf3_json,
    parse_ping_latency,
    probe_direction,
    probe_iperf3_client,
    probe_tcp,
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


def _iperf_json(bits_per_second: float) -> str:
    return json.dumps({"end": {"sum_received": {"bits_per_second": bits_per_second}}})


def test_probe_direction_ok_with_live_values() -> None:
    runner = FakeRunner(
        {
            "hostname -I": _result(stdout="10.87.5.30 172.17.0.1\n"),
            "ip route get": _result(stdout="10.87.5.15 via 10.87.5.1 dev eth0 src 10.87.5.30\n"),
            "create_connection": _result(stdout="tcp_ok\n"),
            "ping -c": _result(
                stdout=(
                    "64 bytes from 10.87.5.15: icmp_seq=1 time=0.85 ms\n"
                    "64 bytes from 10.87.5.15: icmp_seq=2 time=0.91 ms\n"
                    "64 bytes from 10.87.5.15: icmp_seq=3 time=0.88 ms\n"
                )
            ),
            "iperf3 -c": _result(stdout=_iperf_json(940_000_000.0)),
        }
    )

    link = probe_direction(
        runner.run,
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
        target_ip="10.87.5.15",
        tcp_port=29500,
        iperf3_port=5201,
    )

    assert link.status == "ok"
    assert link.tcp_reachable is True
    assert link.latency_ms == 0.88
    assert link.bandwidth_mbps == 940.0
    assert link.interface == "eth0"
    assert link.source_ip == "10.87.5.30"
    assert link.failure_reason is None
    assert link.raw_output


def test_probe_direction_tcp_unreachable_is_unreachable() -> None:
    runner = FakeRunner(
        {
            "hostname -I": _result(stdout="10.87.5.30\n"),
            "ip route get": _result(stdout="dev eth0"),
            "create_connection": _result(
                stderr="ConnectionRefusedError", exit_code=1
            ),
        }
    )

    link = probe_direction(
        runner.run,
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
        target_ip="10.87.5.15",
        tcp_port=29500,
        iperf3_port=5201,
    )

    assert link.status == "unreachable"
    assert link.tcp_reachable is False
    assert link.failure_reason is not None
    assert "tcp unreachable" in link.failure_reason


def test_probe_direction_ping_blocked_keeps_latency_missing() -> None:
    runner = FakeRunner(
        {
            "hostname -I": _result(stdout="10.87.5.30\n"),
            "ip route get": _result(stdout="dev eth0"),
            "create_connection": _result(stdout="tcp_ok\n"),
            "ping -c": _result(stderr="Request timeout for icmp_seq 0", exit_code=1),
            "iperf3 -c": _result(stdout=_iperf_json(500_000_000.0)),
        }
    )

    link = probe_direction(
        runner.run,
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
        target_ip="10.87.5.15",
        tcp_port=29500,
        iperf3_port=5201,
    )

    assert link.status == "degraded"
    assert link.tcp_reachable is True
    assert link.latency_ms is None
    assert link.bandwidth_mbps == 500.0
    assert link.failure_reason is not None
    assert "ping latency unavailable" in link.failure_reason


def test_probe_direction_iperf3_failure_keeps_bandwidth_missing() -> None:
    runner = FakeRunner(
        {
            "hostname -I": _result(stdout="10.87.5.30\n"),
            "ip route get": _result(stdout="dev eth0"),
            "create_connection": _result(stdout="tcp_ok\n"),
            "ping -c": _result(stdout="64 bytes from 10.87.5.15: time=0.5 ms\n"),
            "iperf3 -c": _result(stderr="unable to connect to server", exit_code=1),
        }
    )

    link = probe_direction(
        runner.run,
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
        target_ip="10.87.5.15",
        tcp_port=29500,
        iperf3_port=5201,
    )

    assert link.status == "degraded"
    assert link.bandwidth_mbps is None
    assert link.failure_reason is not None
    assert "iperf3 throughput unavailable" in link.failure_reason


def test_ping_latency_parsing() -> None:
    assert parse_ping_latency("time=0.85 ms\ntime=0.91 ms\ntime=0.88 ms") == 0.88
    assert parse_ping_latency("no times here") is None


def test_iperf3_json_parsing() -> None:
    assert parse_iperf3_json(_iperf_json(940_000_000.0)) == 940.0
    assert parse_iperf3_json("not json") is None
    assert parse_iperf3_json('{"end":{}}') is None


def test_probe_tcp_and_client_helpers() -> None:
    runner = FakeRunner(
        {
            "create_connection": _result(stdout="tcp_ok\n"),
            "iperf3 -c": _result(stdout=_iperf_json(100_000_000.0)),
        }
    )

    reachable, raw = probe_tcp(runner.run, "10.87.5.15", 29500)
    assert reachable is True
    assert "tcp_ok" in raw

    mbps, iperf_raw = probe_iperf3_client(runner.run, "10.87.5.15", 5201, 2)
    assert mbps == 100.0
    assert iperf_raw


def test_live_pairwise_probe_records_real_result() -> None:
    """Real attempt (opt-in via multi_host marker)."""
    import json as _json
    from dataclasses import replace

    from shardgrid.common.config import load_cluster_config
    from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
    from shardgrid.transport.ssh import SSHOptions, SSHTransport

    config = load_cluster_config("examples/workers.yaml")
    address_book = _json.load(open("tests/address.json"))
    workers = {}
    for wid in ("gpu4060", "gpu1060"):
        worker = next(w for w in config.workers if str(w.worker_id) == wid)
        entry = next(
            e
            for e in address_book
            if wid == "gpu4060"
            and "RTX 4060" in str(e.get("gpu_model") or "")
            or wid == "gpu1060"
            and "GTX 1650" in str(e.get("gpu_model") or "")
        )
        worker = replace(worker, host=entry["ip"], ssh_user=entry["username"])
        transport = SSHTransport(
            SSHOptions.from_ssh_config(
                config.ssh, host=str(worker.host), user=worker.ssh_user, port=worker.ssh_port
            )
        )
        workers[wid] = (
            WSLRuntimeWrapper(
                WSLRuntimeConfig.from_worker_and_runtime(worker, config.runtime),
                transport,
            ),
            entry["ip"],
        )

    w4060, ip4060 = workers["gpu4060"]
    w1060, ip1060 = workers["gpu1060"]
    tcp_port = config.network.rendezvous_port
    iperf3_port = config.network.iperf3_port

    forward = probe_direction(
        w4060.run,
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
        target_ip=ip1060,
        tcp_port=tcp_port,
        iperf3_port=iperf3_port,
    )
    reverse = probe_direction(
        w1060.run,
        source_worker_id="gpu1060",
        target_worker_id="gpu4060",
        target_ip=ip4060,
        tcp_port=tcp_port,
        iperf3_port=iperf3_port,
    )

    for link in (forward, reverse):
        assert link.status in {"ok", "degraded", "unreachable"}
        assert link.commands
        assert link.raw_output
        if link.status != "unreachable":
            assert link.tcp_reachable is True