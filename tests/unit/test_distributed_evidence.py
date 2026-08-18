from __future__ import annotations

import json

from shardgrid.cli.commands.dist_test import _build_report
from shardgrid.distributed.collectives import (
    RankCollectiveResult,
    build_collectives_script,
    build_partial_collective_result,
)
from shardgrid.distributed.distributed_gate import GATE2_BLOCKED, evaluate_gate2
from shardgrid.distributed.fallback import FallbackEligibility


def _rank(
    rank: int,
    *,
    run_id: str = "run-123",
    worker_id: str | None = None,
    worker_ip: str | None = None,
    peer_ip: str | None = None,
) -> RankCollectiveResult:
    worker_id = worker_id or ("gpu4060" if rank == 0 else "gpu1060")
    worker_ip = worker_ip or ("10.87.5.155" if rank == 0 else "10.87.5.15")
    peer_ip = peer_ip or ("10.87.5.15" if rank == 0 else "10.87.5.155")
    result = {
        "run_id": run_id,
        "rank": rank,
        "worker_id": worker_id,
        "worker_host_ip": worker_ip,
        "peer_ip": peer_ip,
        "master_addr": "10.87.5.155",
        "master_port": 29500,
        "network_interface": "eth3" if rank == 0 else "eth0",
        "route_output": f"{peer_ip} dev {'eth3' if rank == 0 else 'eth0'} src {worker_ip} uid 1000",
        "port_range": "net.ipv4.ip_local_port_range = 44620 48715",
        "conda_environment": "shardgrid",
        "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
        "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        "python_version": "3.12.13",
        "torch_version": "2.7.1+cu118",
        "torch_cuda_version": "11.8",
        "gpu_name": (
            "NVIDIA GeForce RTX 4060"
            if rank == 0
            else "NVIDIA GeForce GTX 1650"
        ),
        "stages": [
            "BEFORE_INIT",
            "AFTER_INIT",
            "BEFORE_BROADCAST",
            "AFTER_BROADCAST",
            "BEFORE_SEND_RECV",
            "AFTER_SEND_RECV",
            "BEFORE_ALL_REDUCE",
            "AFTER_ALL_REDUCE",
        ],
        "last_stage": "AFTER_ALL_REDUCE",
        "init_ok": True,
        "broadcast_ok": True,
        "broadcast_tensor": [11.0, 22.0, 33.0, 44.0],
        "send_recv_ok": True,
        "send_recv_tensor": [5.0, 6.0, 7.0, 8.0],
        "all_reduce_ok": True,
        "all_reduce_tensor": [3.0, 3.0, 3.0, 3.0],
        "elapsed_s": 1.2,
    }
    return RankCollectiveResult(
        rank=rank,
        worker_id=worker_id,
        exit_code=0,
        timed_out=False,
        result=result,
        stdout="",
        stderr="",
    )


def test_collectives_script_routes_to_peer_not_master() -> None:
    script = build_collectives_script(
        worker_id="gpu4060",
        worker_ip="10.87.5.155",
        peer_ip="10.87.5.15",
        rank=0,
        world_size=2,
        master_addr="10.87.5.155",
        master_port=29500,
        backend="nccl",
        interface="eth3",
        run_id="run-123",
    )

    assert 'peer_ip = "10.87.5.15"' in script
    assert 'worker_ip = "10.87.5.155"' in script
    assert '["ip", "route", "get", peer_ip]' in script


def test_partial_collective_result_preserves_timeout_stage_and_route() -> None:
    bootstrap = {
        "run_id": "run-123",
        "rank": 0,
        "worker_host_ip": "10.87.5.155",
        "peer_ip": "10.87.5.15",
        "network_interface": "eth3",
        "route_output": "10.87.5.15 dev eth3 src 10.87.5.155 uid 1000",
        "port_range": "net.ipv4.ip_local_port_range = 44620 48715",
    }
    stdout = "\n".join(
        [
            f"COLLECTIVE_BOOTSTRAP {json.dumps(bootstrap, sort_keys=True)}",
            "BEFORE_INIT",
            "AFTER_INIT",
            "BEFORE_BROADCAST",
        ]
    )
    partial = build_partial_collective_result(
        stdout=stdout,
        stderr="",
        defaults={
            "rank": 0,
            "worker_id": "gpu4060",
            "master_addr": "10.87.5.155",
            "master_port": 29500,
        },
    )

    assert partial["peer_ip"] == "10.87.5.15"
    assert partial["route_output"].startswith("10.87.5.15 dev eth3")
    assert partial["stages"] == ["BEFORE_INIT", "AFTER_INIT", "BEFORE_BROADCAST"]
    assert partial["last_stage"] == "BEFORE_BROADCAST"


def test_build_report_threads_run_id_into_runtime_and_collectives() -> None:
    report = _build_report(
        run_id="run-123",
        timestamp="2026-08-18T12:00:00+00:00",
        requested_backend="nccl",
        state="NCCL SUCCESS",
        eligibility=FallbackEligibility(
            network_ok=True, rendezvous_ok=True, runtime_ok=True
        ),
        baseline={"interfaces": {"0": "eth3", "1": "eth0"}, "workers": {}},
        infos=[
            {
                "worker_id": "gpu4060",
                "rank": 0,
                "ip": "10.87.5.155",
                "hostname": "LDJ",
                "gpu_model": "RTX 4060",
            },
            {
                "worker_id": "gpu1060",
                "rank": 1,
                "ip": "10.87.5.15",
                "hostname": "LAPTOP-5G3QUOGM",
                "gpu_model": "GTX 1650",
            },
        ],
        captured={"nccl": (_rank(0), _rank(1))},
        master_addr="10.87.5.155",
        master_port=29500,
        elapsed_s=3.2,
        diagnostics_path="/tmp/dist-test.json",
    )

    assert report["run_id"] == "run-123"
    assert report["collectives"]["nccl"]["run_id"] == "run-123"
    assert report["runtime_evidence"]["run_id"] == "run-123"
    assert report["runtime_evidence"]["per_rank"]["0"]["run_id"] == "run-123"


def test_gate2_blocks_mixed_run_id_evidence() -> None:
    report = _build_report(
        run_id="run-123",
        timestamp="2026-08-18T12:00:00+00:00",
        requested_backend="nccl",
        state="NCCL SUCCESS",
        eligibility=FallbackEligibility(
            network_ok=True, rendezvous_ok=True, runtime_ok=True
        ),
        baseline={"interfaces": {"0": "eth3", "1": "eth0"}, "workers": {}},
        infos=[
            {
                "worker_id": "gpu4060",
                "rank": 0,
                "ip": "10.87.5.155",
                "hostname": "LDJ",
                "gpu_model": "RTX 4060",
            },
            {
                "worker_id": "gpu1060",
                "rank": 1,
                "ip": "10.87.5.15",
                "hostname": "LAPTOP-5G3QUOGM",
                "gpu_model": "GTX 1650",
            },
        ],
        captured={"nccl": (_rank(0), _rank(1, run_id="run-999"))},
        master_addr="10.87.5.155",
        master_port=29500,
        elapsed_s=3.2,
        diagnostics_path="/tmp/dist-test.json",
    )

    gate = evaluate_gate2(
        report,
        gate1_status="PASS",
        expected_workers=["gpu4060", "gpu1060"],
    )

    assert gate.status == GATE2_BLOCKED
    assert any("mixed run_id evidence" in problem for problem in gate.problems)
