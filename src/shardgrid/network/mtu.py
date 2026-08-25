"""Helpers for NCCL path MTU checks on WSL2 multi-host paths."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_NCCL_MTU = 1500
DF_SAFE_PAYLOAD_IPV4 = 1472
DF_OVERSIZE_PAYLOAD_IPV4 = 1473

STATUS_PASS = "PASS"
STATUS_FAIL = "NCCL_PATH_MTU_UNSAFE"
STATUS_UNAVAILABLE = "MTU_PROBE_UNAVAILABLE"


@dataclass(frozen=True)
class NCCLPathMTUCheck:
    peer_ip: str
    interface: str | None
    interface_mtu: int | None
    expected_mtu: int
    status: str
    route_output: str

    def failure_reason(self) -> str | None:
        if self.status == STATUS_PASS:
            return None
        if self.interface is None:
            return f"{STATUS_FAIL}: no route interface for peer {self.peer_ip}"
        if self.interface_mtu is None:
            return f"{STATUS_FAIL}: could not read MTU for interface {self.interface}"
        return (
            f"{STATUS_FAIL}: peer={self.peer_ip} interface={self.interface} "
            f"interface_mtu={self.interface_mtu} expected_mtu={self.expected_mtu}"
        )


def parse_route_interface(route_output: str | None) -> str | None:
    if not route_output:
        return None
    match = re.search(r"\bdev\s+(\S+)", route_output)
    return match.group(1) if match else None


def parse_interface_mtu(link_output: str | None) -> int | None:
    if not link_output:
        return None
    match = re.search(r"\bmtu\s+(\d+)\b", link_output)
    return int(match.group(1)) if match else None


def check_nccl_path_mtu(
    *,
    peer_ip: str,
    route_output: str,
    link_output: str | None,
    expected_mtu: int = DEFAULT_NCCL_MTU,
) -> NCCLPathMTUCheck:
    interface = parse_route_interface(route_output)
    interface_mtu = parse_interface_mtu(link_output)
    status = (
        STATUS_PASS
        if interface is not None and interface_mtu == expected_mtu
        else STATUS_FAIL
    )
    return NCCLPathMTUCheck(
        peer_ip=peer_ip,
        interface=interface,
        interface_mtu=interface_mtu,
        expected_mtu=expected_mtu,
        status=status,
        route_output=route_output,
    )


def classify_df_ping(output: str | None, exit_code: int) -> str:
    text = (output or "").lower()
    if "command not found" in text or "operation not permitted" in text:
        return STATUS_UNAVAILABLE
    if exit_code == 0:
        return STATUS_PASS
    return "FAIL"
