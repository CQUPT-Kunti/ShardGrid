from __future__ import annotations

from shardgrid.network.mtu import (
    DEFAULT_NCCL_MTU,
    STATUS_FAIL,
    STATUS_PASS,
    check_nccl_path_mtu,
    parse_interface_mtu,
    parse_route_interface,
    parse_route_source_ip,
)


def test_parse_route_interface_reads_dynamic_dev() -> None:
    route = "10.87.5.15 dev eth3 src 10.87.5.155 uid 1000"
    assert parse_route_interface(route) == "eth3"


def test_parse_route_interface_returns_none_without_dev() -> None:
    assert parse_route_interface("10.87.5.15 src 10.87.5.155") is None


def test_parse_route_source_ip_reads_dynamic_src() -> None:
    route = "10.87.5.15 dev eth3 src 10.87.5.155 uid 1000"
    assert parse_route_source_ip(route) == "10.87.5.155"


def test_parse_interface_mtu_reads_numeric_value() -> None:
    link = "2: eth3: <BROADCAST> mtu 1500 qdisc mq state UP mode DEFAULT"
    assert parse_interface_mtu(link) == 1500


def test_check_nccl_path_mtu_passes_at_1500() -> None:
    check = check_nccl_path_mtu(
        peer_ip="10.87.5.15",
        route_output="10.87.5.15 dev eth3 src 10.87.5.155 uid 1000",
        link_output="2: eth3: <BROADCAST> mtu 1500 qdisc mq state UP mode DEFAULT",
    )
    assert check.status == STATUS_PASS
    assert check.interface == "eth3"
    assert check.interface_mtu == 1500
    assert check.expected_mtu == DEFAULT_NCCL_MTU
    assert check.failure_reason() is None


def test_check_nccl_path_mtu_fails_when_mtu_is_unsafe() -> None:
    check = check_nccl_path_mtu(
        peer_ip="10.87.5.15",
        route_output="10.87.5.15 dev eth3 src 10.87.5.155 uid 1000",
        link_output="2: eth3: <BROADCAST> mtu 2800 qdisc mq state UP mode DEFAULT",
    )
    assert check.status == STATUS_FAIL
    assert check.failure_reason() == (
        "NCCL_PATH_MTU_UNSAFE: peer=10.87.5.15 interface=eth3 "
        "interface_mtu=2800 expected_mtu=1500"
    )
