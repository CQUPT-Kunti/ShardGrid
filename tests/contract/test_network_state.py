from __future__ import annotations

from shardgrid.network.probe import LinkProbeResult
from shardgrid.network.state import (
    build_network_state,
    classify_link_health,
    link_from_probe,
    link_from_probe_dict,
    network_state_from_probe_results,
)
from shardgrid.resources.models import NetworkState


def _probe(
    *,
    source: str = "gpu4060",
    target: str = "gpu1060",
    tcp: bool = True,
    latency: float | None = 0.9,
    bandwidth: float | None = 940.0,
    status: str = "ok",
    reason: str | None = None,
    interface: str | None = "eth0",
) -> LinkProbeResult:
    return LinkProbeResult(
        source_worker_id=source,
        target_worker_id=target,
        source_ip="10.87.5.30",
        target_ip="10.87.5.15",
        interface=interface,
        port=5201,
        tcp_reachable=tcp,
        latency_ms=latency,
        bandwidth_mbps=bandwidth,
        status=status,
        failure_reason=reason,
        commands=("ping -c 3 10.87.5.15",),
        raw_output="raw",
    )


def test_link_from_probe_maps_fields() -> None:
    link = link_from_probe(_probe())

    assert str(link.source_worker_id) == "gpu4060"
    assert str(link.target_worker_id) == "gpu1060"
    assert link.interface == "eth0"
    assert link.tcp_reachable is True
    assert link.latency_ms == 0.9
    assert link.bandwidth_mbps == 940.0
    assert link.port == 5201
    assert link.measured_at
    assert link.failure_reason is None


def test_bidirectional_links_kept_separate() -> None:
    forward = link_from_probe(
        _probe(source="gpu4060", target="gpu1060", latency=0.9, bandwidth=940.0)
    )
    reverse = link_from_probe(
        _probe(
            source="gpu1060",
            target="gpu4060",
            latency=1.1,
            bandwidth=880.0,
            interface="eth0",
        )
    )

    assert str(forward.source_worker_id) == "gpu4060"
    assert str(reverse.source_worker_id) == "gpu1060"
    assert forward.bandwidth_mbps == 940.0
    assert reverse.bandwidth_mbps == 880.0
    assert forward.latency_ms != reverse.latency_ms


def test_failed_link_preserves_failure_reason() -> None:
    link = link_from_probe(
        _probe(
            tcp=False,
            latency=9.3,
            bandwidth=None,
            status="unreachable",
            reason="tcp unreachable: connection timed out",
        )
    )

    health = classify_link_health(link)
    assert health.name == "unreachable"
    assert health.tcp_reachable is False
    assert health.missing_measurement is True
    assert link.failure_reason == "tcp unreachable: connection timed out"


def test_missing_bandwidth_is_degraded_not_fabricated() -> None:
    link = link_from_probe(
        _probe(tcp=True, latency=1.0, bandwidth=None, status="degraded")
    )

    health = classify_link_health(link)
    assert health.name == "degraded"
    assert health.tcp_reachable is True
    assert link.bandwidth_mbps is None


def test_healthy_link_classified() -> None:
    health = classify_link_health(link_from_probe(_probe()))

    assert health.name == "healthy"
    assert health.degraded is False
    assert health.missing_measurement is False


def test_network_state_serialization_round_trip() -> None:
    state = network_state_from_probe_results(
        [
            _probe(source="gpu4060", target="gpu1060"),
            _probe(
                source="gpu1060",
                target="gpu4060",
                interface="eth0",
            ),
        ],
        network_id="lan-a",
        diagnostics_path="/var/tmp/shardgrid/network",
    )

    payload = state.to_dict()
    restored = NetworkState.from_dict(payload)

    assert restored == state
    assert payload["network_id"] == "lan-a"
    assert len(payload["links"]) == 2
    assert payload["diagnostics_path"] == "/var/tmp/shardgrid/network"
    assert set(payload["workers"]) == {"gpu4060", "gpu1060"}


def test_network_state_aggregates_workers_and_interfaces() -> None:
    state = build_network_state(
        [
            link_from_probe(_probe(source="gpu4060", target="gpu1060", interface="eth0")),
            link_from_probe(
                _probe(source="gpu1060", target="gpu4060", interface="eth0")
            ),
        ],
        network_id="lan-a",
    )

    assert set(str(worker) for worker in state.workers) == {"gpu4060", "gpu1060"}
    assert state.selected_interfaces == {"gpu4060": "eth0", "gpu1060": "eth0"}
    assert state.created_at


def test_link_from_probe_dict_rebuilds_from_saved_evidence() -> None:
    probe = _probe(
        tcp=False,
        latency=9.3,
        bandwidth=None,
        status="unreachable",
        reason="tcp unreachable: connection timed out",
    )
    saved = probe.to_dict()

    link = link_from_probe_dict(saved)

    assert link.tcp_reachable is False
    assert link.bandwidth_mbps is None
    assert link.failure_reason == "tcp unreachable: connection timed out"
    assert link.port == 5201