"""Distributed backend / interface / rendezvous selection (T048).

Selects NCCL first and only ever uses Gloo as an explicit later fallback (T050).
Rendezvous address, port, and socket interface all come from the existing
NetworkState and configuration; nothing is hard-coded.  In diagnostics mode
``NCCL_DEBUG=INFO`` is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shardgrid.resources.models import NetworkLink, NetworkState


class BackendError(ValueError):
    """Raised when backend/interface/rendezvous selection is invalid."""


@dataclass(frozen=True)
class BackendSelection:
    backend: str
    master_addr: str
    master_port: int
    interface: str
    nccl_socket_ifname: str
    gloo_socket_ifname: str
    diagnostics: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "interface": self.interface,
            "nccl_socket_ifname": self.nccl_socket_ifname,
            "gloo_socket_ifname": self.gloo_socket_ifname,
            "diagnostics": self.diagnostics,
        }


def select_backend(preference: str = "nccl") -> str:
    """NCCL first; Gloo only when explicitly requested as a fallback."""
    normalized = preference.lower()
    if normalized in ("nccl", "auto"):
        return "nccl"
    if normalized == "gloo":
        return "gloo"
    raise BackendError(
        f"unsupported backend {preference!r}; expected nccl, gloo, or auto"
    )


def _link_between(state: NetworkState, source: str, target: str) -> NetworkLink | None:
    for link in state.links:
        if str(link.source_worker_id) == source and str(link.target_worker_id) == target:
            return link
    return None


def select_interface(
    state: NetworkState,
    *,
    source_worker_id: str,
    target_worker_id: str,
) -> str:
    configured = state.selected_interfaces.get(source_worker_id)
    if configured:
        return configured
    link = _link_between(state, source_worker_id, target_worker_id)
    if link is not None and link.interface:
        return link.interface
    raise BackendError(
        f"no interface available in NetworkState for {source_worker_id}"
    )


def _master_address(
    state: NetworkState,
    *,
    source_worker_id: str,
    target_worker_id: str,
) -> str:
    link = _link_between(state, source_worker_id, target_worker_id)
    if link is None:
        raise BackendError(
            f"no NetworkLink between {source_worker_id} and {target_worker_id}"
        )
    if not link.tcp_reachable:
        raise BackendError(
            f"selected link {source_worker_id} -> {target_worker_id} is unreachable"
        )
    address = link.target_ip or link.source_ip
    if not address:
        raise BackendError("master address is empty in NetworkState")
    return address


def build_backend_selection(
    state: NetworkState,
    *,
    source_worker_id: str,
    target_worker_id: str,
    master_port: int = 29500,
    preference: str = "nccl",
    diagnostics: bool = False,
) -> BackendSelection:
    backend = select_backend(preference)
    if not (0 < master_port < 65536):
        raise BackendError(f"master_port must be in (0, 65536), got {master_port}")
    interface = select_interface(
        state, source_worker_id=source_worker_id, target_worker_id=target_worker_id
    )
    master_addr = _master_address(
        state, source_worker_id=source_worker_id, target_worker_id=target_worker_id
    )
    return BackendSelection(
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
        interface=interface,
        nccl_socket_ifname=interface,
        gloo_socket_ifname=interface,
        diagnostics=diagnostics,
    )