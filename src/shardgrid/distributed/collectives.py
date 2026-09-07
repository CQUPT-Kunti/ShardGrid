"""Real multi-host NCCL collectives harness (T049).

Builds the collectives script executed inside each Worker's WSL2 selected Conda
environment (via the T040 stdin wrapper), launches rank 0 on the RTX 4060
Worker and rank 1 on the GTX 1650 Worker, and parses the real per-rank
``COLLECTIVE_RESULT`` JSON. Parameters follow the T046 smoke contract
(rank / world-size / master-addr / master-port / backend / local-rank) and
preserve runtime evidence for the actual WSL Conda Python used on each rank.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shardgrid.transport.runtime import WSLRuntimeWrapper

_STAGE_MARKERS = (
    "BEFORE_INIT",
    "AFTER_INIT",
    "BEFORE_BROADCAST",
    "AFTER_BROADCAST",
    "BEFORE_SEND_RECV",
    "AFTER_SEND_RECV",
    "BEFORE_BARRIER",
    "AFTER_BARRIER",
    "BEFORE_ALL_REDUCE",
    "AFTER_ALL_REDUCE",
)
_BOOTSTRAP_PREFIX = "COLLECTIVE_BOOTSTRAP "
_RESULT_PREFIX = "COLLECTIVE_RESULT "

_COLLECTIVES_SCRIPT = """
import json
import os
import platform
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist

worker_id = "__WORKER_ID__"
rank = __RANK__
world_size = __WORLD_SIZE__
local_rank = __LOCAL_RANK__
master = "__MASTER__"
port = __PORT__
backend = "__BACKEND__"
interface = "__INTERFACE__"
worker_ip = "__WORKER_IP__"
peer_ip = "__PEER_IP__"
run_id = "__RUN_ID__"
start = time.time()
stages = []

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,BOOTSTRAP,NET,COLL"
os.environ["NCCL_SOCKET_IFNAME"] = f"={interface}"
os.environ["GLOO_SOCKET_IFNAME"] = interface
os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NET"] = "Socket"


def mark(stage: str) -> None:
    stages.append(stage)
    print(stage, flush=True)

out = {
    "run_id": run_id,
    "worker_id": worker_id,
    "hostname": socket.gethostname(),
    "worker_host_ip": worker_ip,
    "peer_ip": peer_ip,
    "rank": rank,
    "world_size": world_size,
    "local_rank": local_rank,
    "backend": backend,
    "network_interface": interface,
    "master_addr": master,
    "master_port": port,
    "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": str(torch.version.cuda),
    "cuda_available": bool(torch.cuda.is_available()),
    "current_device": None,
    "gpu_name": None,
    "effective_env": None,
    "route_output": None,
    "port_range": None,
    "stages": stages,
    "last_stage": None,
    "init_ok": False,
    "broadcast_ok": False,
    "broadcast_tensor": None,
    "send_recv_ok": False,
    "send_recv_tensor": None,
    "all_reduce_ok": False,
    "all_reduce_tensor": None,
    "elapsed_s": None,
    "error": None,
}

try:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False inside selected WSL Conda runtime")
    out["route_output"] = subprocess.check_output(
        ["ip", "route", "get", peer_ip],
        text=True,
    ).strip()
    out["port_range"] = subprocess.check_output(
        ["sysctl", "net.ipv4.ip_local_port_range"],
        text=True,
    ).strip()
    bootstrap = {
        "run_id": run_id,
        "rank": rank,
        "worker_host_ip": worker_ip,
        "peer_ip": peer_ip,
        "hostname": out["hostname"],
        "master_addr": master,
        "master_port": port,
        "backend": backend,
        "network_interface": interface,
        "route_output": out["route_output"],
        "port_range": out["port_range"],
        "cuda_device": f"cuda:{local_rank}",
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME"),
        "NCCL_SOCKET_FAMILY": os.environ.get("NCCL_SOCKET_FAMILY"),
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE"),
        "NCCL_NET": os.environ.get("NCCL_NET"),
    }
    print("COLLECTIVE_BOOTSTRAP " + json.dumps(bootstrap, sort_keys=True), flush=True)
    out["effective_env"] = {
        "run_id": run_id,
        "rank": rank,
        "interface": interface,
        "worker_host_ip": worker_ip,
        "peer_ip": peer_ip,
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME"),
        "NCCL_SOCKET_FAMILY": os.environ.get("NCCL_SOCKET_FAMILY"),
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE"),
        "NCCL_NET": os.environ.get("NCCL_NET"),
        "MASTER_ADDR": master,
        "MASTER_PORT": port,
    }
    print("NCCL_EFFECTIVE_ENV " + json.dumps(out["effective_env"], sort_keys=True), flush=True)
    device = torch.device(f"cuda:{local_rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
        out["current_device"] = int(torch.cuda.current_device())
    out["gpu_name"] = torch.cuda.get_device_name(local_rank)
    mark("BEFORE_INIT")
    init_kwargs = dict(
        backend=backend,
        init_method="tcp://%s:%d" % (master, port),
        rank=rank,
        world_size=world_size,
    )
    if backend == "nccl":
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    out["init_ok"] = True
    mark("AFTER_INIT")

    expected_broadcast = torch.tensor([11.0, 22.0, 33.0, 44.0], device=device)
    broadcast_tensor = (
        expected_broadcast.clone()
        if rank == 0
        else torch.zeros(4, dtype=torch.float32, device=device)
    )
    mark("BEFORE_BROADCAST")
    dist.broadcast(broadcast_tensor, src=0)
    torch.cuda.synchronize()
    mark("AFTER_BROADCAST")
    out["broadcast_tensor"] = broadcast_tensor.detach().cpu().tolist()
    out["broadcast_ok"] = bool(torch.equal(broadcast_tensor, expected_broadcast))

    expected_send = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device)
    mark("BEFORE_SEND_RECV")
    if rank == 0:
        dist.send(expected_send, dst=1)
        out["send_recv_tensor"] = expected_send.detach().cpu().tolist()
        out["send_recv_ok"] = True
    else:
        recv_tensor = torch.zeros(4, dtype=torch.float32, device=device)
        dist.recv(recv_tensor, src=0)
        torch.cuda.synchronize()
        out["send_recv_tensor"] = recv_tensor.detach().cpu().tolist()
        out["send_recv_ok"] = bool(torch.equal(recv_tensor, expected_send))
    torch.cuda.synchronize()
    mark("AFTER_SEND_RECV")
    mark("BEFORE_BARRIER")
    barrier_kwargs: dict[str, object] = {}
    if backend == "nccl":
        barrier_kwargs["device_ids"] = [local_rank]
    dist.barrier(**barrier_kwargs)
    torch.cuda.synchronize()
    mark("AFTER_BARRIER")

    all_reduce_tensor = torch.full((4,), float(rank + 1), dtype=torch.float32, device=device)
    mark("BEFORE_ALL_REDUCE")
    dist.all_reduce(all_reduce_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    mark("AFTER_ALL_REDUCE")
    expected_reduce = torch.full((4,), 3.0, dtype=torch.float32, device=device)
    out["all_reduce_tensor"] = all_reduce_tensor.detach().cpu().tolist()
    out["all_reduce_ok"] = bool(torch.equal(all_reduce_tensor, expected_reduce))
except Exception as exc:
    out["error"] = str(exc)
finally:
    out["elapsed_s"] = round(time.time() - start, 3)
    out["last_stage"] = stages[-1] if stages else None
    print("COLLECTIVE_RESULT " + json.dumps(out, sort_keys=True))
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
"""


@dataclass(frozen=True)
class RankCollectiveResult:
    rank: int
    worker_id: str
    exit_code: int | None
    timed_out: bool
    result: dict[str, Any] | None
    stdout: str
    stderr: str
    recorded_command: str | None = None


def build_collectives_script(
    *,
    worker_id: str,
    worker_ip: str,
    peer_ip: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    backend: str,
    interface: str,
    run_id: str,
    local_rank: int = 0,
) -> str:
    script = _COLLECTIVES_SCRIPT.replace("__WORKER_ID__", worker_id)
    script = script.replace("__INTERFACE__", interface)
    script = script.replace("__WORKER_IP__", worker_ip)
    script = script.replace("__PEER_IP__", peer_ip)
    script = script.replace("__RUN_ID__", run_id)
    script = script.replace("__RANK__", str(rank))
    script = script.replace("__WORLD_SIZE__", str(world_size))
    script = script.replace("__LOCAL_RANK__", str(local_rank))
    script = script.replace("__MASTER__", master_addr)
    script = script.replace("__PORT__", str(master_port))
    script = script.replace("__BACKEND__", backend)
    return script


def parse_collective_result(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(_RESULT_PREFIX):
            try:
                payload = json.loads(stripped[len(_RESULT_PREFIX) :])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def parse_collective_bootstrap(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(_BOOTSTRAP_PREFIX):
            try:
                payload = json.loads(stripped[len(_BOOTSTRAP_PREFIX) :])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def parse_collective_stages(*outputs: str) -> list[str]:
    stages: list[str] = []
    seen: set[str] = set()
    for output in outputs:
        for line in output.splitlines():
            marker = line.strip()
            if marker not in _STAGE_MARKERS or marker in seen:
                continue
            seen.add(marker)
            stages.append(marker)
    return stages


def build_partial_collective_result(
    *,
    stdout: str,
    stderr: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    payload = {**defaults}
    bootstrap = parse_collective_bootstrap(stdout)
    if bootstrap:
        payload.update(bootstrap)
    stages = parse_collective_stages(stdout, stderr)
    payload["stages"] = stages
    payload["last_stage"] = stages[-1] if stages else None
    return payload


def launch_rank(
    wrapper: WSLRuntimeWrapper,
    *,
    worker_id: str,
    worker_ip: str,
    peer_ip: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    backend: str,
    interface: str,
    run_id: str,
    timeout: float = 180.0,
) -> RankCollectiveResult:
    script = build_collectives_script(
        worker_id=worker_id,
        worker_ip=worker_ip,
        peer_ip=peer_ip,
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
        interface=interface,
        run_id=run_id,
        local_rank=0,
    )
    result = wrapper.run_script(script, timeout=timeout)
    partial = build_partial_collective_result(
        stdout=result.stdout,
        stderr=result.stderr,
        defaults={
            "run_id": run_id,
            "rank": rank,
            "worker_id": worker_id,
            "worker_host_ip": worker_ip,
            "peer_ip": peer_ip,
            "master_addr": master_addr,
            "master_port": master_port,
            "backend": backend,
            "network_interface": interface,
        },
    )
    parsed = parse_collective_result(result.stdout)
    if parsed is None:
        parsed = partial
    else:
        parsed = {
            **partial,
            **parsed,
            "stages": parsed.get("stages") or partial["stages"],
            "last_stage": parsed.get("last_stage") or partial["last_stage"],
            "route_output": parsed.get("route_output") or partial.get("route_output"),
            "port_range": parsed.get("port_range") or partial.get("port_range"),
        }
    return RankCollectiveResult(
        rank=rank,
        worker_id=worker_id,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        result=parsed,
        stdout=result.stdout,
        stderr=result.stderr,
        recorded_command=result.recorded_command,
    )


def run_pair_collectives(
    rank0_wrapper: WSLRuntimeWrapper,
    rank1_wrapper: WSLRuntimeWrapper,
    *,
    rank0_worker_id: str,
    rank1_worker_id: str,
    rank0_worker_ip: str,
    rank1_worker_ip: str,
    master_addr: str,
    master_port: int,
    backend: str,
    rank0_interface: str,
    rank1_interface: str,
    run_id: str,
    timeout: float = 180.0,
) -> tuple[RankCollectiveResult, RankCollectiveResult]:
    def run_rank0() -> None:
        nonlocal_holder["rank0"] = launch_rank(
            rank0_wrapper,
            worker_id=rank0_worker_id,
            worker_ip=rank0_worker_ip,
            peer_ip=rank1_worker_ip,
            rank=0,
            world_size=2,
            master_addr=master_addr,
            master_port=master_port,
            backend=backend,
            interface=rank0_interface,
            run_id=run_id,
            timeout=timeout,
        )

    def run_rank1() -> None:
        nonlocal_holder["rank1"] = launch_rank(
            rank1_wrapper,
            worker_id=rank1_worker_id,
            worker_ip=rank1_worker_ip,
            peer_ip=rank0_worker_ip,
            rank=1,
            world_size=2,
            master_addr=master_addr,
            master_port=master_port,
            backend=backend,
            interface=rank1_interface,
            run_id=run_id,
            timeout=timeout,
        )

    nonlocal_holder: dict[str, RankCollectiveResult | None] = {
        "rank0": None,
        "rank1": None,
    }
    threads = [
        threading.Thread(target=run_rank0, name="rank0"),
        threading.Thread(target=run_rank1, name="rank1"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rank0 = nonlocal_holder["rank0"]
    rank1 = nonlocal_holder["rank1"]
    assert rank0 is not None and rank1 is not None
    return rank0, rank1


def collectives_outcome(
    rank0: RankCollectiveResult, rank1: RankCollectiveResult
) -> str:
    """PASS only when both ranks initialized and all collectives are true."""
    results: list[dict[str, Any]] = []
    for result in (rank0.result, rank1.result):
        if result is None:
            return "FAIL"
        results.append(result)
    if not all(result.get("init_ok") is True for result in results):
        return "FAIL"
    checks = ("broadcast_ok", "send_recv_ok", "all_reduce_ok")
    if all(result.get(check) is True for result in results for check in checks):
        return "PASS"
    return "FAIL"


def save_collectives_evidence(
    rank0: RankCollectiveResult,
    rank1: RankCollectiveResult,
    *,
    run_id: str,
    backend: str,
    master_addr: str,
    master_port: int,
    interface: str,
    output_dir: str | Path,
    rank_metadata: dict[int, dict[str, Any]] | None = None,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "backend": backend,
        "master_addr": master_addr,
        "master_port": master_port,
        "interface": interface,
        "outcome": collectives_outcome(rank0, rank1),
        "ranks": [
            {
                "rank": rank0.rank,
                "worker_id": rank0.worker_id,
                "exit_code": rank0.exit_code,
                "timed_out": rank0.timed_out,
                "metadata": (rank_metadata or {}).get(rank0.rank, {}),
                "result": rank0.result,
                "recorded_command": rank0.recorded_command,
                "stdout": rank0.stdout[-4000:],
                "stderr": rank0.stderr[-8000:],
            },
            {
                "rank": rank1.rank,
                "worker_id": rank1.worker_id,
                "exit_code": rank1.exit_code,
                "timed_out": rank1.timed_out,
                "metadata": (rank_metadata or {}).get(rank1.rank, {}),
                "result": rank1.result,
                "recorded_command": rank1.recorded_command,
                "stdout": rank1.stdout[-4000:],
                "stderr": rank1.stderr[-8000:],
            },
        ],
    }
    path = directory / f"collectives-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (directory / "collectives-latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return path
