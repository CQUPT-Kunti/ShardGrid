from __future__ import annotations

import pytest

from shardgrid.common.models import as_worker_id
from shardgrid.distributed.backend import (
    BackendError,
    build_backend_selection,
    select_backend,
)
from shardgrid.resources.models import NetworkLink, NetworkState


def _state(*, reachable: bool = True, interface: str = "eth0") -> NetworkState:
    return NetworkState(
        network_id="lan-a",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id("gpu4060"),
                target_worker_id=as_worker_id("gpu1060"),
                source_ip="10.87.5.155",
                target_ip="10.87.5.15",
                interface=interface,
                tcp_reachable=reachable,
                latency_ms=0.9,
                bandwidth_mbps=940.0,
            )
        ],
        selected_interfaces={"gpu4060": interface},
    )


def test_select_backend_nccl_first() -> None:
    assert select_backend("nccl") == "nccl"
    assert select_backend("auto") == "nccl"
    assert select_backend() == "nccl"


def test_select_backend_gloo_only_explicit() -> None:
    assert select_backend("gloo") == "gloo"


def test_select_backend_unsupported_rejected() -> None:
    with pytest.raises(BackendError, match="unsupported backend"):
        select_backend("mpi")


def test_build_selection_uses_network_state() -> None:
    selection = build_backend_selection(
        _state(),
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
        master_port=29500,
    )

    assert selection.backend == "nccl"
    assert selection.master_addr == "10.87.5.15"
    assert selection.master_port == 29500
    assert selection.interface == "eth0"
    assert selection.nccl_socket_ifname == "eth0"
    assert selection.gloo_socket_ifname == "eth0"
    assert selection.diagnostics is False


def test_diagnostics_mode_enabled() -> None:
    selection = build_backend_selection(
        _state(),
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
        diagnostics=True,
    )

    assert selection.diagnostics is True


def test_missing_network_state_rejected() -> None:
    state = NetworkState(network_id="empty")

    with pytest.raises(BackendError, match="no interface available"):
        build_backend_selection(
            state,
            source_worker_id="gpu4060",
            target_worker_id="gpu1060",
        )


def test_unreachable_link_rejected() -> None:
    with pytest.raises(BackendError, match="unreachable"):
        build_backend_selection(
            _state(reachable=False),
            source_worker_id="gpu4060",
            target_worker_id="gpu1060",
        )


def test_invalid_master_port_rejected() -> None:
    with pytest.raises(BackendError, match="master_port"):
        build_backend_selection(
            _state(),
            source_worker_id="gpu4060",
            target_worker_id="gpu1060",
            master_port=0,
        )


def test_invalid_master_address_rejected() -> None:
    state = _state()
    state.links[0] = NetworkLink(
        source_worker_id=as_worker_id("gpu4060"),
        target_worker_id=as_worker_id("gpu1060"),
        source_ip="",
        target_ip="",
        interface="eth0",
        tcp_reachable=True,
    )

    with pytest.raises(BackendError, match="master address"):
        build_backend_selection(
            state,
            source_worker_id="gpu4060",
            target_worker_id="gpu1060",
        )


def test_selection_is_deterministic() -> None:
    first = build_backend_selection(
        _state(),
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
    )
    second = build_backend_selection(
        _state(),
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
    )

    assert first == second


def test_same_rendezvous_for_both_ranks() -> None:
    forward = build_backend_selection(
        _state(),
        source_worker_id="gpu4060",
        target_worker_id="gpu1060",
    )
    reverse = build_backend_selection(
        NetworkState(
            network_id="lan-a",
            workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
            links=[
                NetworkLink(
                    source_worker_id=as_worker_id("gpu1060"),
                    target_worker_id=as_worker_id("gpu4060"),
                    source_ip="10.87.5.15",
                    target_ip="10.87.5.155",
                    interface="eth0",
                    tcp_reachable=True,
                )
            ],
            selected_interfaces={"gpu1060": "eth0"},
        ),
        source_worker_id="gpu1060",
        target_worker_id="gpu4060",
    )

    assert forward.master_addr == "10.87.5.15"
    assert reverse.master_addr == "10.87.5.155"
    assert forward.master_port == reverse.master_port == 29500
    assert forward.backend == reverse.backend == "nccl"