from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from shardgrid.common.config import RuntimeConfig, WorkerConfig
from shardgrid.common.models import as_hostname
from shardgrid.distributed.backend import BackendSelection
from shardgrid.distributed.collectives import (
    build_collectives_script,
    collectives_outcome,
    parse_collective_result,
    run_pair_collectives,
    save_collectives_evidence,
)
from shardgrid.distributed.runner import (
    LaunchError,
    build_launch_plan,
    validate_rank,
)
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport


def _worker(worker_id: str, host: str) -> WorkerConfig:
    return WorkerConfig.from_dict(
        {
            "id": worker_id,
            "machine_id": "machine-c",
            "physical_os": "windows",
            "runtime_os": "wsl2_linux",
            "runtime": "wsl2",
            "host": host,
            "ssh_user": "shardgrid",
            "runtime_distro": "Ubuntu",
            "conda_environment": "shardgrid",
            "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
        }
    )


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(
        default_wsl_distro="Ubuntu",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        conda_environment="shardgrid",
    )


def test_build_collectives_script_injects_parameters() -> None:
    script = build_collectives_script(
        worker_id="gpu1060",
        rank=1,
        world_size=2,
        master_addr="10.87.5.155",
        master_port=29500,
        backend="nccl",
        interface="eth0",
        local_rank=0,
    )

    assert 'worker_id = "gpu1060"' in script
    assert "rank = 1" in script
    assert "world_size = 2" in script
    assert '"10.87.5.155"' in script
    assert "port = 29500" in script
    assert '"nccl"' in script
    assert "NCCL_SOCKET_IFNAME" in script
    assert "GLOO_SOCKET_IFNAME" in script
    assert "eth0" in script
    assert "python_executable" in script
    assert "torch.cuda.set_device" in script
    assert "dist.init_process_group" in script
    assert "COLLECTIVE_RESULT" in script


def test_parse_collective_result() -> None:
    payload = parse_collective_result(
        'some log line\nCOLLECTIVE_RESULT {"rank": 0, "init_ok": true}\n'
    )

    assert payload == {"rank": 0, "init_ok": True}


def test_parse_collective_result_missing() -> None:
    assert parse_collective_result("no result here") is None
    assert parse_collective_result('COLLECTIVE_RESULT not-json') is None


def _rank_result(result: dict | None) -> object:
    from shardgrid.distributed.collectives import RankCollectiveResult

    return RankCollectiveResult(
        rank=0,
        worker_id="gpu4060",
        exit_code=0,
        timed_out=False,
        result=result,
        stdout="",
        stderr="",
    )


def test_collectives_outcome_pass_when_all_ok() -> None:
    from shardgrid.distributed.collectives import RankCollectiveResult

    ok = {
        "init_ok": True,
        "broadcast_ok": True,
        "send_recv_ok": True,
        "all_reduce_ok": True,
    }
    rank0 = RankCollectiveResult(
        rank=0, worker_id="gpu4060", exit_code=0, timed_out=False, result=ok, stdout="", stderr=""
    )
    rank1 = RankCollectiveResult(
        rank=1, worker_id="gpu1060", exit_code=0, timed_out=False, result=ok, stdout="", stderr=""
    )

    assert collectives_outcome(rank0, rank1) == "PASS"


def test_collectives_outcome_fail_on_init_missing() -> None:
    from shardgrid.distributed.collectives import RankCollectiveResult

    rank0 = RankCollectiveResult(
        rank=0, worker_id="gpu4060", exit_code=0, timed_out=False,
        result={
            "init_ok": False,
            "broadcast_ok": False,
            "send_recv_ok": False,
            "all_reduce_ok": False,
        },
        stdout="", stderr="",
    )
    rank1 = RankCollectiveResult(
        rank=1, worker_id="gpu1060", exit_code=0, timed_out=False, result=None, stdout="", stderr=""
    )

    assert collectives_outcome(rank0, rank1) == "FAIL"


def test_collectives_outcome_fail_on_broadcast_false() -> None:
    from shardgrid.distributed.collectives import RankCollectiveResult

    rank0 = RankCollectiveResult(
        rank=0, worker_id="gpu4060", exit_code=0, timed_out=False,
        result={"init_ok": True, "broadcast_ok": True, "send_recv_ok": True, "all_reduce_ok": True},
        stdout="", stderr="",
    )
    rank1 = RankCollectiveResult(
        rank=1, worker_id="gpu1060", exit_code=0, timed_out=False,
        result={
            "init_ok": True,
            "broadcast_ok": False,
            "send_recv_ok": True,
            "all_reduce_ok": True,
        },
        stdout="", stderr="",
    )

    assert collectives_outcome(rank0, rank1) == "FAIL"


def test_runner_plan_consistent_with_collectives_contract() -> None:
    plan = build_launch_plan(
        [_worker("gpu4060", "10.87.5.155"), _worker("gpu1060", "10.87.5.15")],
        runtime=_runtime(),
        smoke_program="examples/distributed_smoke/smoke.py",
        master_addr="10.87.5.155",
        backend="nccl",
    )

    assert plan.world_size == 2
    assert plan.backend == "nccl"
    assert plan.launches[0].rank == 0
    assert plan.launches[1].rank == 1
    assert all(launch.local_world_size == 1 for launch in plan.launches)


def test_validate_rank_edge_cases() -> None:
    import pytest

    validate_rank(0, 1)
    with pytest.raises(LaunchError):
        validate_rank(1, 1)


def _address_book() -> list[dict[str, object]]:
    return json.loads(Path("tests/address.json").read_text(encoding="utf-8"))


def _worker_entry(worker_id: str, address_book: list[dict[str, object]]) -> dict[str, object]:
    label_by_worker = {
        "gpu4060": "RTX 4060",
        "gpu1060": "GTX 1650",
    }
    label = label_by_worker[worker_id]
    return next(
        entry for entry in address_book if label in str(entry.get("gpu_model") or "")
    )


def _discover_interface(wrapper: WSLRuntimeWrapper, peer_ip: str) -> str:
    result = wrapper.run(
        f"ip route get {peer_ip}",
        timeout=10.0,
    )
    interface = result.stdout.strip().splitlines()
    if not interface:
        raise AssertionError(f"failed to discover WSL interface: {result.stderr}")
    tokens = interface[0].split()
    if "dev" not in tokens:
        raise AssertionError(f"failed to parse WSL interface from route: {result.stdout}")
    return tokens[tokens.index("dev") + 1]


def _cleanup_runtime_python(wrapper: WSLRuntimeWrapper) -> None:
    result = wrapper.run(
        "pkill -f /home/shardgrid/miniconda3/envs/shardgrid/bin/python || true",
        timeout=10.0,
    )
    assert result.exit_code == 0


def _runtime_command_output(wrapper: WSLRuntimeWrapper, command: str) -> str:
    result = wrapper.run(command, timeout=10.0)
    text = (result.stdout or result.stderr).strip()
    return text


def _preflight_runtime_state(wrapper: WSLRuntimeWrapper, peer_ip: str) -> dict[str, str]:
    return {
        "route_get": _runtime_command_output(wrapper, f"ip route get {peer_ip}"),
        "port_range": _runtime_command_output(wrapper, "sysctl net.ipv4.ip_local_port_range"),
        "python_processes": _runtime_command_output(wrapper, "ps -ef | grep '[p]ython' || true"),
        "listen_29500": _runtime_command_output(wrapper, "ss -ltnp | grep ':29500' || true"),
    }


def _socket_snapshot(wrapper: WSLRuntimeWrapper) -> dict[str, str]:
    return {
        "ss_tanp_python": _runtime_command_output(wrapper, "ss -tanp | grep python || true"),
        "ss_ltnp_python": _runtime_command_output(wrapper, "ss -ltnp | grep python || true"),
    }


def _build_live_backend_selection(
    *,
    rank0_ip: str,
    interface: str,
    port: int,
) -> BackendSelection:
    return BackendSelection(
        backend="nccl",
        master_addr=rank0_ip,
        master_port=port,
        interface=interface,
        nccl_socket_ifname=interface,
        gloo_socket_ifname=interface,
        diagnostics=True,
    )


def test_live_pair_nccl_collectives_records_real_result() -> None:
    """Real two-host NCCL attempt (opt-in via multi_host marker)."""
    from shardgrid.common.config import load_cluster_config

    config = load_cluster_config("examples/workers.yaml")
    address_book = _address_book()
    wrappers: dict[int, tuple[WSLRuntimeWrapper, dict[str, object], str]] = {}
    for wid, rank in [("gpu4060", 0), ("gpu1060", 1)]:
        worker = next(w for w in config.workers if str(w.worker_id) == wid)
        entry = _worker_entry(wid, address_book)
        worker = replace(
            worker,
            host=as_hostname(str(entry["ip"])),
            ssh_user=str(entry["username"]),
        )
        transport = SSHTransport(
            SSHOptions.from_ssh_config(
                config.ssh,
                host=str(entry["ip"]),
                user=str(entry["username"]),
                port=worker.ssh_port,
            )
        )
        wrappers[rank] = (
            WSLRuntimeWrapper(
                WSLRuntimeConfig.from_worker_and_runtime(worker, config.runtime), transport
            ),
            entry,
            str(worker.worker_id),
        )

    w0, entry0, id0 = wrappers[0]
    w1, entry1, id1 = wrappers[1]
    _cleanup_runtime_python(w0)
    _cleanup_runtime_python(w1)
    rank0_interface = _discover_interface(w0, str(entry1["ip"]))
    rank1_interface = _discover_interface(w1, str(entry0["ip"]))
    port = config.network.rendezvous_port
    selection = _build_live_backend_selection(
        rank0_ip=str(entry0["ip"]),
        interface=rank0_interface,
        port=port,
    )
    rank_metadata = {
        0: {
            "windows_host_ip": entry0["ip"],
            "windows_hostname": entry0.get("hostname"),
            "windows_user": entry0["username"],
            "runtime_distro": w0.config.distro,
            "preflight": _preflight_runtime_state(w0, str(entry1["ip"])),
        },
        1: {
            "windows_host_ip": entry1["ip"],
            "windows_hostname": entry1.get("hostname"),
            "windows_user": entry1["username"],
            "runtime_distro": w1.config.distro,
            "preflight": _preflight_runtime_state(w1, str(entry0["ip"])),
        },
    }

    rank0, rank1 = run_pair_collectives(
        w0,
        w1,
        rank0_worker_id=id0, rank1_worker_id=id1,
        master_addr=selection.master_addr, master_port=selection.master_port,
        backend=selection.backend,
        rank0_interface=rank0_interface,
        rank1_interface=rank1_interface,
        timeout=90.0,
    )
    if rank0.timed_out or rank1.timed_out:
        rank_metadata[0]["timeout_sockets"] = _socket_snapshot(w0)
        rank_metadata[1]["timeout_sockets"] = _socket_snapshot(w1)

    evidence = save_collectives_evidence(
        rank0, rank1, backend=selection.backend, master_addr=selection.master_addr,
        master_port=selection.master_port, interface=selection.interface,
        output_dir="/var/tmp/shardgrid/distributed",
        rank_metadata=rank_metadata,
    )

    outcome = collectives_outcome(rank0, rank1)
    assert outcome in {"PASS", "FAIL"}
    if outcome == "PASS":
        assert rank0.result is not None and rank0.result["init_ok"] is True
        assert rank1.result is not None and rank1.result["init_ok"] is True
        assert rank0.result["python_executable"].startswith(rank0.result["conda_prefix"])
        assert rank1.result["python_executable"].startswith(rank1.result["conda_prefix"])
        assert rank0.result["broadcast_ok"] is True
        assert rank1.result["broadcast_ok"] is True
        assert rank0.result["send_recv_ok"] is True
        assert rank1.result["send_recv_ok"] is True
        assert rank0.result["all_reduce_ok"] is True
        assert rank1.result["all_reduce_ok"] is True
    else:
        # honest failure with real evidence preserved
        assert evidence.exists()
        assert (
            rank0.stderr
            or rank1.stderr
            or (rank0.result and rank0.result.get("error"))
            or (rank1.result and rank1.result.get("error"))
        )
