"""NetworkState aggregation from T043 pairwise probe evidence (T044).

Builds ``NetworkLink`` / ``NetworkState`` (existing models) from the raw
link-probe evidence saved by T043, without re-running iperf3.  Unreachable,
degraded, and missing measurements are preserved honestly; nothing is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from shardgrid.common.models import as_worker_id
from shardgrid.network.probe import LinkProbeResult
from shardgrid.resources.models import NetworkLink, NetworkState


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LinkHealth:
    name: str
    tcp_reachable: bool
    degraded: bool
    missing_measurement: bool
    reason: str | None = None


def classify_link_health(link: NetworkLink) -> LinkHealth:
    if not link.tcp_reachable:
        return LinkHealth(
            name="unreachable",
            tcp_reachable=False,
            degraded=False,
            missing_measurement=True,
            reason=link.failure_reason,
        )
    if link.bandwidth_mbps is None:
        return LinkHealth(
            name="degraded",
            tcp_reachable=True,
            degraded=True,
            missing_measurement=True,
            reason=link.failure_reason,
        )
    return LinkHealth(
        name="healthy",
        tcp_reachable=True,
        degraded=False,
        missing_measurement=False,
    )


def link_from_probe(
    result: LinkProbeResult,
    *,
    measured_at: str | None = None,
) -> NetworkLink:
    return NetworkLink(
        source_worker_id=as_worker_id(result.source_worker_id),
        target_worker_id=as_worker_id(result.target_worker_id),
        source_ip=result.source_ip or "",
        target_ip=result.target_ip,
        interface=result.interface or "",
        tcp_reachable=result.tcp_reachable,
        latency_ms=result.latency_ms,
        bandwidth_mbps=result.bandwidth_mbps,
        interface_mtu=result.interface_mtu,
        expected_mtu=result.expected_mtu,
        mtu_status=result.mtu_status,
        port=result.port,
        measured_at=measured_at or now_utc(),
        failure_reason=result.failure_reason,
    )


def link_from_probe_dict(data: dict[str, Any]) -> NetworkLink:
    return link_from_probe(
        LinkProbeResult(
            source_worker_id=str(data["source_worker_id"]),
            target_worker_id=str(data["target_worker_id"]),
            source_ip=data.get("source_ip"),
            target_ip=str(data["target_ip"]),
            interface=data.get("interface"),
            port=int(data.get("port", 29500)),
            tcp_reachable=bool(data.get("tcp_reachable", False)),
            latency_ms=data.get("latency_ms"),
            bandwidth_mbps=data.get("bandwidth_mbps"),
            interface_mtu=data.get("interface_mtu"),
            expected_mtu=data.get("expected_mtu"),
            mtu_status=data.get("mtu_status"),
            status=str(data.get("status", "unknown")),
            failure_reason=data.get("failure_reason"),
            commands=tuple(str(item) for item in data.get("commands", [])),
            raw_output=str(data.get("raw_output", "")),
        ),
        measured_at=data.get("measured_at"),
    )


def build_network_state(
    links: Sequence[NetworkLink],
    *,
    network_id: str,
    diagnostics_path: str | None = None,
) -> NetworkState:
    worker_ids = {
        str(link.source_worker_id) for link in links
    } | {str(link.target_worker_id) for link in links}
    selected_interfaces = {
        str(link.source_worker_id): link.interface
        for link in links
        if link.interface
    }
    return NetworkState(
        network_id=network_id,
        workers=[as_worker_id(worker_id) for worker_id in sorted(worker_ids)],
        links=list(links),
        created_at=now_utc(),
        selected_interfaces=selected_interfaces,
        diagnostics_path=diagnostics_path,
    )


def network_state_from_probe_results(
    results: Sequence[LinkProbeResult],
    *,
    network_id: str,
    diagnostics_path: str | None = None,
) -> NetworkState:
    links = [link_from_probe(result) for result in results]
    return build_network_state(
        links, network_id=network_id, diagnostics_path=diagnostics_path
    )
