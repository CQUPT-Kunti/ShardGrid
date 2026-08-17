from __future__ import annotations

import json

from shardgrid.cli.app import main
from shardgrid.cli.commands import network_test as network_test_command
from shardgrid.network.probe import LinkProbeResult


def _probe(
    *,
    source: str,
    target: str,
    tcp: bool = True,
    latency: float | None = 0.9,
    bandwidth: float | None = 940.0,
    status: str = "ok",
    reason: str | None = None,
) -> LinkProbeResult:
    return LinkProbeResult(
        source_worker_id=source,
        target_worker_id=target,
        source_ip="10.87.5.30",
        target_ip="10.87.5.15",
        interface="eth0",
        port=5201,
        tcp_reachable=tcp,
        latency_ms=latency,
        bandwidth_mbps=bandwidth,
        status=status,
        failure_reason=reason,
        commands=("ping -c 3 10.87.5.15",),
        raw_output="raw",
    )


def test_network_test_is_registered_as_real_command(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "network-test" in captured.out
    assert "placeholder command" not in captured.out


def test_network_test_requires_config(capsys) -> None:
    exit_code = main(["network-test", "--all"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "requires a cluster config" in captured.out


def test_network_test_requires_mode(capsys) -> None:
    exit_code = main(
        ["--config", "examples/workers.yaml", "network-test"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "requires --all or --workers" in captured.out


def test_network_test_invalid_worker(monkeypatch, capsys) -> None:
    def fake_probe_pair(config, source, target) -> LinkProbeResult:
        raise ValueError("unknown worker id(s): nope")

    monkeypatch.setattr(network_test_command, "_probe_pair", fake_probe_pair)

    exit_code = main(
        ["--config", "examples/workers.yaml", "network-test", "--workers", "nope", "gpu1060"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown worker id" in captured.out


def test_network_test_json_output(monkeypatch, capsys) -> None:
    def fake_probe_pair(config, source, target) -> LinkProbeResult:
        return _probe(source=source, target=target)

    monkeypatch.setattr(network_test_command, "_probe_pair", fake_probe_pair)

    exit_code = main(
        [
            "--config",
            "examples/workers.yaml",
            "--json",
            "network-test",
            "--workers",
            "gpu4060",
            "gpu1060",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["network_id"] == "lan-a"
    assert len(payload["links"]) == 1
    link = payload["links"][0]
    assert link["source_worker_id"] == "gpu4060"
    assert link["target_worker_id"] == "gpu1060"
    assert link["tcp_reachable"] is True
    assert link["bandwidth_mbps"] == 940.0
    assert link["interface"] == "eth0"


def test_network_test_all_covers_both_directions(monkeypatch, capsys) -> None:
    def fake_probe_pair(config, source, target) -> LinkProbeResult:
        return _probe(source=source, target=target)

    monkeypatch.setattr(network_test_command, "_probe_pair", fake_probe_pair)

    exit_code = main(
        [
            "--config",
            "examples/workers.yaml",
            "--json",
            "network-test",
            "--all",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    links = payload["links"]
    assert exit_code == 0
    assert {link["source_worker_id"] for link in links} == {"gpu4060", "gpu1060"}
    assert {link["target_worker_id"] for link in links} == {"gpu1060", "gpu4060"}


def test_network_test_unreachable_link_returns_nonzero(monkeypatch, capsys) -> None:
    def fake_probe_pair(config, source, target) -> LinkProbeResult:
        return _probe(
            source=source,
            target=target,
            tcp=False,
            latency=9.3,
            bandwidth=None,
            status="unreachable",
            reason="tcp unreachable: connection timed out",
        )

    monkeypatch.setattr(network_test_command, "_probe_pair", fake_probe_pair)

    exit_code = main(
        [
            "--config",
            "examples/workers.yaml",
            "network-test",
            "--workers",
            "gpu4060",
            "gpu1060",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unreachable" in captured.out
    assert "connection timed out" in captured.out


def test_network_test_human_output(monkeypatch, capsys) -> None:
    def fake_probe_pair(config, source, target) -> LinkProbeResult:
        return _probe(source=source, target=target)

    monkeypatch.setattr(network_test_command, "_probe_pair", fake_probe_pair)

    exit_code = main(
        [
            "--config",
            "examples/workers.yaml",
            "network-test",
            "--workers",
            "gpu4060",
            "gpu1060",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "gpu4060 -> gpu1060" in captured.out
    assert "940.000" in captured.out
    assert "network state saved" in captured.out


def test_network_test_saves_network_state(monkeypatch, tmp_path, capsys) -> None:
    from pathlib import Path

    def fake_probe_pair(config, source, target) -> LinkProbeResult:
        return _probe(source=source, target=target)

    monkeypatch.setattr(network_test_command, "_probe_pair", fake_probe_pair)
    monkeypatch.setattr(network_test_command, "DEFAULT_STATE_DIR", str(tmp_path))

    main(["--config", "examples/workers.yaml", "network-test", "--workers", "gpu4060", "gpu1060"])
    capsys.readouterr()

    saved = [path for path in Path(tmp_path).glob("network-state-*.json")]
    assert saved
    payload = json.loads(saved[0].read_text())
    assert payload["links"][0]["source_worker_id"] == "gpu4060"