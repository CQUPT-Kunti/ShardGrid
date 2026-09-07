from __future__ import annotations

import json
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from shardgrid.common.config import RuntimeConfig, WorkerConfig, load_cluster_config
from shardgrid.common.models import as_hostname
from shardgrid.common.process import ProcessResult
from shardgrid.distributed.runner import build_launch_plan
from shardgrid.transport.runtime import (
    WSLRuntimeConfig,
    WSLRuntimeWrapper,
    wrap_wsl_direct_command,
)
from shardgrid.transport.ssh import SSHOptions, SSHTransport

_PREFLIGHT_PREFIX = "T107_PREFLIGHT "
_SMOKE_PREFIX = "T107_SMOKE "
_EVIDENCE_DIR = Path("/var/tmp/shardgrid/distributed")
_EVIDENCE_PATH = _EVIDENCE_DIR / "three-worker-smoke-latest.json"
_RUN_ID_PREFIX = "t107-three-worker-smoke"
_RENDEZVOUS_PREFIX = "T107_RENDEZVOUS "

_PREFLIGHT_SCRIPT = """
import json
import socket
import sys

payload = {
    "hostname": socket.gethostname(),
    "python_executable": sys.executable,
    "error": None,
}
try:
    import torch
    import torch.distributed as dist

    payload.update(
        {
            "torch_import": True,
            "torch_version": torch.__version__,
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nccl_available": bool(dist.is_nccl_available()),
            "gloo_available": bool(dist.is_gloo_available()),
        }
    )
except Exception as exc:
    payload["torch_import"] = False
    payload["error"] = str(exc)

print("T107_PREFLIGHT " + json.dumps(payload, sort_keys=True))
"""

_THREE_WORKER_SMOKE_TEMPLATE = """
import json
import os
import platform
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist

config = json.loads(__CONFIG_JSON__)
worker_id = config["worker_id"]
worker_ip = config["worker_ip"]
peer_ips = config["peer_ips"]
rank = int(config["rank"])
world_size = int(config["world_size"])
local_rank = int(config["local_rank"])
master_addr = config["master_addr"]
master_port = int(config["master_port"])
backend = config["backend"]
run_id = config["run_id"]

start = time.time()
routes = {}
stages = []
payload = {
    "run_id": run_id,
    "worker_id": worker_id,
    "hostname": socket.gethostname(),
    "physical_host": socket.gethostname(),
    "worker_host_ip": worker_ip,
    "peer_ips": peer_ips,
    "rank": rank,
    "world_size": world_size,
    "local_rank": local_rank,
    "local_world_size": 1,
    "backend": backend,
    "master_addr": master_addr,
    "master_port": master_port,
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": str(torch.version.cuda),
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "gpu_name": None,
    "cuda_device": None,
    "nccl_available": bool(dist.is_nccl_available()),
    "nccl_version": None,
    "network_interface": None,
    "route_diagnostics": routes,
    "mtu": None,
    "stages": stages,
    "init_ok": False,
    "broadcast_ok": False,
    "ring_ok": False,
    "all_reduce_ok": False,
    "barrier_ok": False,
    "destroy_ok": False,
    "broadcast_tensor": None,
    "ring_recv": None,
    "all_reduce_tensor": None,
    "expected_all_reduce": None,
    "shutdown_result": "NOT_RUN",
    "error": None,
    "elapsed_s": None,
}

def mark(stage):
    stages.append(stage)

def route_info(peer_ip):
    output = subprocess.check_output(["ip", "route", "get", peer_ip], text=True).strip()
    tokens = output.split()
    if "dev" not in tokens or "src" not in tokens:
        raise RuntimeError("unparseable route output: " + output)
    interface = tokens[tokens.index("dev") + 1]
    src_ip = tokens[tokens.index("src") + 1]
    mtu_output = subprocess.check_output(["ip", "link", "show", "dev", interface], text=True)
    mtu_tokens = mtu_output.split()
    mtu = int(mtu_tokens[mtu_tokens.index("mtu") + 1]) if "mtu" in mtu_tokens else None
    return {
        "route_output": output,
        "interface": interface,
        "src_ip": src_ip,
        "mtu": mtu,
    }

try:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False inside selected WSL Conda runtime")
    payload["gpu_name"] = torch.cuda.get_device_name(local_rank)
    try:
        import torch.cuda.nccl as nccl

        version = nccl.version()
        if isinstance(version, tuple):
            payload["nccl_version"] = ".".join(str(item) for item in version)
        else:
            payload["nccl_version"] = str(version)
    except Exception:
        payload["nccl_version"] = None
    for peer_ip in peer_ips:
        routes[peer_ip] = route_info(peer_ip)
    interfaces = {item["interface"] for item in routes.values()}
    if len(interfaces) != 1:
        raise RuntimeError("multiple runtime interfaces required: " + ",".join(sorted(interfaces)))
    interface = next(iter(interfaces))
    payload["network_interface"] = interface
    payload["mtu"] = routes[peer_ips[0]]["mtu"] if peer_ips else None
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,BOOTSTRAP,NET,COLL"
    os.environ["NCCL_SOCKET_IFNAME"] = "=" + interface
    os.environ["GLOO_SOCKET_IFNAME"] = interface
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_NET"] = "Socket"
    device = torch.device(f"cuda:{local_rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
        payload["cuda_device"] = int(torch.cuda.current_device())
    mark("before_init")
    init_kwargs = {
        "backend": backend,
        "init_method": f"tcp://{master_addr}:{master_port}",
        "rank": rank,
        "world_size": world_size,
    }
    if backend == "nccl":
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    payload["init_ok"] = True
    mark("after_init")

    expected_broadcast = torch.tensor([11.0, 22.0, 33.0], dtype=torch.float32, device=device)
    if rank == 0:
        broadcast_tensor = expected_broadcast.clone()
    else:
        broadcast_tensor = torch.zeros(3, dtype=torch.float32, device=device)
    dist.broadcast(broadcast_tensor, src=0)
    if backend == "nccl":
        torch.cuda.synchronize()
    payload["broadcast_tensor"] = broadcast_tensor.detach().cpu().tolist()
    payload["broadcast_ok"] = bool(torch.equal(broadcast_tensor, expected_broadcast))
    mark("after_broadcast")

    send_tensor = torch.tensor([float(rank)], dtype=torch.float32, device=device)
    recv_tensor = torch.full((1,), -1.0, dtype=torch.float32, device=device)
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size
    recv_req = dist.irecv(recv_tensor, src=prev_rank)
    send_req = dist.isend(send_tensor, dst=next_rank)
    send_req.wait()
    recv_req.wait()
    if backend == "nccl":
        torch.cuda.synchronize()
    payload["ring_recv"] = recv_tensor.detach().cpu().tolist()
    payload["ring_ok"] = bool(recv_tensor.item() == float(prev_rank))
    mark("after_ring")

    all_reduce_tensor = torch.full((1,), float(rank + 1), dtype=torch.float32, device=device)
    dist.all_reduce(all_reduce_tensor, op=dist.ReduceOp.SUM)
    if backend == "nccl":
        torch.cuda.synchronize()
    expected_total = float(world_size * (world_size + 1) // 2)
    payload["expected_all_reduce"] = [expected_total]
    payload["all_reduce_tensor"] = all_reduce_tensor.detach().cpu().tolist()
    payload["all_reduce_ok"] = bool(all_reduce_tensor.item() == expected_total)
    mark("after_all_reduce")

    barrier_kwargs = {}
    if backend == "nccl":
        barrier_kwargs["device_ids"] = [local_rank]
    dist.barrier(**barrier_kwargs)
    if backend == "nccl":
        torch.cuda.synchronize()
    payload["barrier_ok"] = True
    mark("after_barrier")
except Exception as exc:
    payload["error"] = str(exc)
finally:
    payload["elapsed_s"] = round(time.time() - start, 3)
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            payload["destroy_ok"] = True
            payload["shutdown_result"] = "PASS"
        except Exception as exc:
            payload["shutdown_result"] = "FAIL"
            payload["error"] = payload["error"] or str(exc)
    print("T107_SMOKE " + json.dumps(payload, sort_keys=True))
"""


def _worker(
    worker_id: str,
    host: str,
    *,
    runtime_distro: str = "Ubuntu-22.04",
) -> WorkerConfig:
    return WorkerConfig.from_dict(
        {
            "id": worker_id,
            "machine_id": f"machine-{worker_id}",
            "physical_os": "windows",
            "runtime_os": "wsl2_linux",
            "runtime": "wsl2",
            "host": host,
            "ssh_user": "shardgrid",
            "runtime_distro": runtime_distro,
            "conda_environment": "shardgrid",
            "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
            "local_world_size": 1,
        }
    )


def _process_result(
    *,
    args: str = "",
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> ProcessResult:
    return ProcessResult(
        args=args,
        recorded_command=args,
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        runtime_environment={},
    )


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(
        default_wsl_distro="Ubuntu-22.04",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
    )


def _parse_json_line(output: str, prefix: str) -> dict[str, Any] | None:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            try:
                payload = json.loads(stripped[len(prefix) :])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def _parse_port_owner(ss_output: str, ps_output: str) -> dict[str, Any] | None:
    match = re.search(r'users:\(\("(?P<process>[^"]+)",pid=(?P<pid>\d+),fd=', ss_output)
    if match is None:
        return None
    pid = int(match.group("pid"))
    process = match.group("process")
    ps_lines = [line.rstrip() for line in ps_output.splitlines() if line.strip()]
    cmd = ps_lines[0] if ps_lines else ""
    cwd = ps_lines[1] if len(ps_lines) > 1 else None
    return {
        "pid": pid,
        "process": process,
        "command": cmd,
        "cwd": cwd,
        "ss_output": ss_output.strip(),
    }


def _port_owner(wrapper: WSLRuntimeWrapper, port: int) -> dict[str, Any] | None:
    probe = wrapper.run(f"ss -ltnp | grep ':{port}' || true", timeout=10.0)
    ss_output = probe.stdout.strip()
    if not ss_output:
        return None
    match = re.search(r"pid=(\d+)", ss_output)
    if match is None:
        return {
            "pid": None,
            "process": None,
            "command": "",
            "cwd": None,
            "ss_output": ss_output,
        }
    pid = int(match.group(1))
    ps = wrapper.run(
        f"ps -p {pid} -o cmd=; readlink -f /proc/{pid}/cwd || true",
        timeout=10.0,
    )
    return _parse_port_owner(ss_output, ps.stdout)


def _owner_type(owner: dict[str, Any] | None) -> str:
    if owner is None:
        return "NONE"
    command = str(owner.get("command") or "")
    process = str(owner.get("process") or "")
    if process == "python" and "miniconda3/envs/shardgrid/bin/python -" in command:
        return "STALE_SHARDGRID"
    return "UNRELATED"


def _stop_precise_pid(wrapper: WSLRuntimeWrapper, pid: int) -> dict[str, Any]:
    script = (
        "import json, os, signal, time\n"
        f"pid = {int(pid)}\n"
        "def alive():\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "        return True\n"
        "    except ProcessLookupError:\n"
        "        return False\n"
        "    except PermissionError:\n"
        "        return True\n"
        "signals = []\n"
        "initial = alive()\n"
        "if initial:\n"
        "    try:\n"
        "        os.killpg(pid, signal.SIGTERM)\n"
        "    except OSError:\n"
        "        os.kill(pid, signal.SIGTERM)\n"
        "    signals.append('SIGTERM')\n"
        "    deadline = time.time() + 5.0\n"
        "    while time.time() < deadline and alive():\n"
        "        time.sleep(0.1)\n"
        "if alive():\n"
        "    try:\n"
        "        os.killpg(pid, signal.SIGKILL)\n"
        "    except OSError:\n"
        "        os.kill(pid, signal.SIGKILL)\n"
        "    signals.append('SIGKILL')\n"
        "    deadline = time.time() + 5.0\n"
        "    while time.time() < deadline and alive():\n"
        "        time.sleep(0.1)\n"
        "print(json.dumps(\n"
        "    {'pid': pid, 'signals': signals, 'stopped': not alive()},\n"
        "    sort_keys=True,\n"
        "))\n"
    )
    result = wrapper.run_script(script, timeout=15.0)
    if result.exit_code != 0 or result.timed_out:
        raise AssertionError(f"failed to stop stale pid {pid}: {result.stderr or result.stdout}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise AssertionError(f"invalid stop payload: {payload!r}")
    return payload


def _select_rendezvous_port(
    wrapper: WSLRuntimeWrapper,
    preferred_port: int,
) -> tuple[int, dict[str, Any] | None, str, bool]:
    owner = _port_owner(wrapper, preferred_port)
    owner_type = _owner_type(owner)
    cleaned = False
    if owner_type == "STALE_SHARDGRID":
        pid = owner.get("pid")
        if not isinstance(pid, int):
            raise AssertionError(f"stale port owner missing pid: {owner}")
        stopped = _stop_precise_pid(wrapper, pid)
        if stopped.get("stopped") is not True:
            raise AssertionError(f"stale rendezvous owner survived precise stop: {stopped}")
        cleaned = True
        owner = _port_owner(wrapper, preferred_port)
        owner_type = _owner_type(owner)
    if owner is None:
        return preferred_port, None, "NONE", cleaned
    if owner_type == "STALE_SHARDGRID":
        raise AssertionError(f"stale rendezvous owner still present after cleanup: {owner}")
    script = """
import json
import socket

preferred = __PREFERRED_PORT__
chosen = None
for port in range(preferred + 1, preferred + 101):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", port))
    except OSError:
        sock.close()
        continue
    chosen = port
    sock.close()
    break
if chosen is None:
    raise SystemExit("no free rendezvous port found")
print("__PREFIX__" + json.dumps({"master_port": chosen}, sort_keys=True))
"""
    script = script.replace("__PREFERRED_PORT__", str(preferred_port))
    script = script.replace("__PREFIX__", _RENDEZVOUS_PREFIX)
    result = wrapper.run_script(script, timeout=15.0)
    payload = _parse_json_line(result.stdout, _RENDEZVOUS_PREFIX)
    if result.exit_code != 0 or result.timed_out or payload is None:
        raise AssertionError(
            f"failed to allocate rendezvous port: {result.stderr or result.stdout}"
        )
    chosen = int(payload["master_port"])
    return chosen, owner, owner_type, cleaned


def _build_three_worker_smoke_script(config: dict[str, Any]) -> str:
    return _THREE_WORKER_SMOKE_TEMPLATE.replace(
        "__CONFIG_JSON__", repr(json.dumps(config, sort_keys=True))
    )


def _build_wrapper(config_path: str, worker_id: str) -> tuple[WSLRuntimeWrapper, WorkerConfig]:
    cluster = load_cluster_config(config_path)
    worker = next(item for item in cluster.workers if str(item.worker_id) == worker_id)
    resolved = replace(
        worker,
        host=as_hostname(str(worker.host)),
        ssh_user=str(worker.ssh_user),
    )
    transport = SSHTransport(
        SSHOptions.from_ssh_config(
            cluster.ssh,
            host=str(resolved.host),
            user=str(resolved.ssh_user),
            port=resolved.ssh_port,
        )
    )
    return (
        WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(resolved, cluster.runtime), transport
        ),
        resolved,
    )


def _route_info(wrapper: WSLRuntimeWrapper, peer_ip: str) -> dict[str, Any]:
    result = wrapper.run(f"ip route get {peer_ip}", timeout=10.0)
    output = result.stdout.strip()
    tokens = output.split()
    if not output or "dev" not in tokens or "src" not in tokens:
        raise AssertionError(f"failed to parse route to {peer_ip}: {result.stdout} {result.stderr}")
    interface = tokens[tokens.index("dev") + 1]
    source_ip = tokens[tokens.index("src") + 1]
    mtu_result = wrapper.run(f"ip link show dev {interface}", timeout=10.0)
    mtu_tokens = mtu_result.stdout.split()
    mtu = int(mtu_tokens[mtu_tokens.index("mtu") + 1]) if "mtu" in mtu_tokens else None
    return {
        "route_output": output,
        "interface": interface,
        "src_ip": source_ip,
        "mtu": mtu,
    }


def _run_mtu_fix(
    wrapper: WSLRuntimeWrapper,
    *,
    peer_ip: str,
    expected_mtu: int,
) -> dict[str, Any]:
    script = Path("scripts/bootstrap-wsl.sh").read_text(encoding="utf-8")
    payload = (
        f"SHARDGRID_NCCL_PEER_IP={peer_ip} "
        f"SHARDGRID_NCCL_MTU={expected_mtu} "
        "SHARDGRID_BOOTSTRAP_JSON=1 "
        "SHARDGRID_WSL_PERSIST_NCCL_MTU=0 "
        "bash -s -- --fix-nccl-mtu-only --json"
    )
    remote = wrap_wsl_direct_command(
        wrapper.config.distro or "",
        "root",
        payload,
    )
    result = wrapper.executor.run(remote, stdin=script, timeout=60.0)
    if result.exit_code not in (None, 0):
        raise AssertionError(
            f"mtu fix failed for {peer_ip}: "
            f"exit={result.exit_code} stdout={(result.stdout or '')[-800:]} "
            f"stderr={(result.stderr or '')[-800:]}"
        )
    try:
        payload_data = json.loads(result.stdout)
    except ValueError as exc:
        raise AssertionError(
            f"mtu fix returned non-json payload: {(result.stdout or '')[-800:]}"
        ) from exc
    if not isinstance(payload_data, dict):
        raise AssertionError(f"mtu fix returned unexpected payload: {payload_data!r}")
    return payload_data


def _run_preflight(wrapper: WSLRuntimeWrapper) -> dict[str, Any]:
    result = wrapper.run_script(_PREFLIGHT_SCRIPT, timeout=60.0)
    payload = _parse_json_line(result.stdout, _PREFLIGHT_PREFIX)
    if payload is None:
        raise AssertionError(
            "preflight produced no payload; "
            f"stdout tail={(result.stdout or '')[-800:]}; "
            f"stderr tail={(result.stderr or '')[-800:]}"
        )
    if result.exit_code not in (None, 0):
        raise AssertionError(f"preflight exit code {result.exit_code}: {result.stderr}")
    return payload


def _launch_rank(
    wrapper: WSLRuntimeWrapper,
    *,
    worker_id: str,
    worker_ip: str,
    peer_ips: list[str],
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    backend: str,
    run_id: str,
) -> dict[str, Any]:
    result = wrapper.run_script(
        _build_three_worker_smoke_script(
            {
                "worker_id": worker_id,
                "worker_ip": worker_ip,
                "peer_ips": peer_ips,
                "rank": rank,
                "world_size": world_size,
                "local_rank": 0,
                "master_addr": master_addr,
                "master_port": master_port,
                "backend": backend,
                "run_id": run_id,
            }
        ),
        timeout=120.0,
    )
    payload = _parse_json_line(result.stdout, _SMOKE_PREFIX)
    return {
        "rank": rank,
        "worker_id": worker_id,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "recorded_command": result.recorded_command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "result": payload,
    }


def _rank_ok(rank_result: dict[str, Any]) -> bool:
    result = rank_result.get("result") or {}
    return (
        rank_result.get("exit_code") == 0
        and not rank_result.get("timed_out")
        and result.get("init_ok") is True
        and result.get("broadcast_ok") is True
        and result.get("ring_ok") is True
        and result.get("all_reduce_ok") is True
        and result.get("barrier_ok") is True
        and result.get("destroy_ok") is True
    )


def test_three_worker_launch_plan_uses_world_size_three() -> None:
    plan = build_launch_plan(
        [
            _worker("gpu4060", "10.87.5.155"),
            _worker("gpu1060", "10.87.5.15"),
            _worker("gpu4060-cqupt", "10.87.5.214"),
        ],
        runtime=_runtime(),
        smoke_program="examples/distributed_smoke/smoke.py",
        master_addr="10.87.5.155",
        backend="nccl",
    )

    assert plan.world_size == 3
    assert [launch.rank for launch in plan.launches] == [0, 1, 2]
    assert [launch.local_world_size for launch in plan.launches] == [1, 1, 1]
    assert [launch.local_rank for launch in plan.launches] == [0, 0, 0]


def test_three_worker_smoke_script_embeds_expected_reduce_sum() -> None:
    script = _build_three_worker_smoke_script(
        {
            "worker_id": "gpu4060-cqupt",
            "worker_ip": "10.87.5.214",
            "peer_ips": ["10.87.5.155", "10.87.5.15"],
            "rank": 2,
            "world_size": 3,
            "local_rank": 0,
            "master_addr": "10.87.5.155",
            "master_port": 29500,
            "backend": "nccl",
            "run_id": _RUN_ID_PREFIX,
        }
    )

    assert '"world_size": 3' in script
    assert '"rank": 2' in script
    assert "world_size * (world_size + 1) // 2" in script
    assert "dist.all_reduce" in script
    assert "dist.irecv" in script
    assert "dist.isend" in script


def test_parse_port_owner_extracts_pid_process_and_command() -> None:
    owner = _parse_port_owner(
        'LISTEN 0 4096 *:29500 *:* users:(("python",pid=311,fd=17))',
        "/home/shardgrid/miniconda3/envs/shardgrid/bin/python -\n/mnt/c/Users/shardgrid\n",
    )

    assert owner == {
        "pid": 311,
        "process": "python",
        "command": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python -",
        "cwd": "/mnt/c/Users/shardgrid",
        "ss_output": 'LISTEN 0 4096 *:29500 *:* users:(("python",pid=311,fd=17))',
    }


class _FakeWrapper:
    def __init__(
        self,
        *,
        run_results: list[ProcessResult] | None = None,
        script_results: list[ProcessResult] | None = None,
    ) -> None:
        self.run_results = list(run_results or [])
        self.script_results = list(script_results or [])
        self.run_calls: list[str] = []
        self.script_calls: list[str] = []

    def run(self, command: str, **kwargs: Any) -> ProcessResult:
        self.run_calls.append(command)
        if not self.run_results:
            raise AssertionError("unexpected run call")
        return self.run_results.pop(0)

    def run_script(self, script: str, **kwargs: Any) -> ProcessResult:
        self.script_calls.append(script)
        if not self.script_results:
            raise AssertionError("unexpected run_script call")
        return self.script_results.pop(0)


def test_select_rendezvous_port_reuses_default_when_free() -> None:
    wrapper = _FakeWrapper(run_results=[_process_result(stdout="")])

    port, owner, owner_type, cleaned = _select_rendezvous_port(wrapper, 29500)

    assert port == 29500
    assert owner is None
    assert owner_type == "NONE"
    assert cleaned is False
    assert not wrapper.script_calls


def test_select_rendezvous_port_cleans_precise_stale_shardgrid_owner() -> None:
    wrapper = _FakeWrapper(
        run_results=[
            _process_result(stdout='LISTEN 0 4096 *:29500 *:* users:(("python",pid=311,fd=17))\n'),
            _process_result(
                stdout=(
                    "/home/shardgrid/miniconda3/envs/shardgrid/bin/python -\n"
                    "/mnt/c/Users/shardgrid\n"
                )
            ),
            _process_result(stdout=""),
        ],
        script_results=[
            _process_result(stdout='{"pid": 311, "signals": ["SIGTERM"], "stopped": true}\n')
        ],
    )

    port, owner, owner_type, cleaned = _select_rendezvous_port(wrapper, 29500)

    assert port == 29500
    assert owner is None
    assert owner_type == "NONE"
    assert cleaned is True
    assert len(wrapper.script_calls) == 1
    assert "pid = 311" in wrapper.script_calls[0]


def test_select_rendezvous_port_avoids_unrelated_owner_without_killing_it() -> None:
    wrapper = _FakeWrapper(
        run_results=[
            _process_result(stdout='LISTEN 0 4096 *:29500 *:* users:(("python",pid=9001,fd=17))\n'),
            _process_result(stdout="/usr/bin/python3 /opt/other/service.py\n/opt/other\n"),
        ],
        script_results=[
            _process_result(stdout='T107_RENDEZVOUS {"master_port": 29501}\n')
        ],
    )

    port, owner, owner_type, cleaned = _select_rendezvous_port(wrapper, 29500)

    assert port == 29501
    assert owner is not None and owner["pid"] == 9001
    assert owner_type == "UNRELATED"
    assert cleaned is False
    assert len(wrapper.script_calls) == 1
    assert "range(preferred + 1, preferred + 101)" in wrapper.script_calls[0]


@pytest.mark.hardware
def test_live_three_worker_smoke() -> None:
    run_id = (
        f"{_RUN_ID_PREFIX}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    config_path = "examples/workers.yaml"
    cluster = load_cluster_config(config_path)
    selected_ids = ["gpu4060", "gpu1060", "gpu4060-cqupt"]
    wrappers: dict[int, tuple[WSLRuntimeWrapper, WorkerConfig]] = {}
    workers: list[WorkerConfig] = []
    for rank, worker_id in enumerate(selected_ids):
        wrapper, worker = _build_wrapper(config_path, worker_id)
        wrappers[rank] = (wrapper, worker)
        workers.append(worker)

    plan = build_launch_plan(
        workers,
        runtime=cluster.runtime,
        smoke_program="examples/distributed_smoke/smoke.py",
        master_addr=str(workers[0].host),
        master_port=cluster.network.rendezvous_port,
        backend="nccl",
    )
    assert plan.world_size == 3
    assert [launch.rank for launch in plan.launches] == [0, 1, 2]
    assert all(launch.local_world_size == 1 for launch in plan.launches)
    assert all(launch.local_rank == 0 for launch in plan.launches)

    preflight: dict[str, dict[str, Any]] = {}
    mtu_fixes: dict[str, dict[str, Any]] = {}
    worker_routes: dict[str, dict[str, dict[str, Any]]] = {}
    expected_mtu = cluster.network.nccl_mtu
    for rank, worker_id in enumerate(selected_ids):
        wrapper, worker = wrappers[rank]
        preflight[worker_id] = _run_preflight(wrapper)
        pf = preflight[worker_id]
        assert pf.get("torch_import") is True, f"{worker_id} torch import failed: {pf}"
        assert pf.get("cuda_available") is True, f"{worker_id} cuda unavailable: {pf}"
        assert int(pf.get("device_count") or 0) >= 1, f"{worker_id} device_count < 1: {pf}"
        assert pf.get("nccl_available") is True, f"{worker_id} NCCL unavailable: {pf}"
        peers = [str(item.host) for item in workers if str(item.worker_id) != worker_id]
        worker_routes[worker_id] = {
            peer_ip: _route_info(wrapper, peer_ip)
            for peer_ip in peers
        }
        unsafe_routes = [
            route for route in worker_routes[worker_id].values() if route["mtu"] != expected_mtu
        ]
        if unsafe_routes:
            interfaces = {route["interface"] for route in unsafe_routes}
            assert len(interfaces) == 1, (
                f"{worker_id} requires multi-interface MTU fix: {unsafe_routes}"
            )
            mtu_fixes[worker_id] = _run_mtu_fix(
                wrapper,
                peer_ip=peers[0],
                expected_mtu=expected_mtu,
            )
            assert (
                ((mtu_fixes[worker_id].get("nccl_path_mtu") or {}).get("status")) == "PASS"
            ), mtu_fixes[worker_id]
            worker_routes[worker_id] = {
                peer_ip: _route_info(wrapper, peer_ip)
                for peer_ip in peers
            }
        assert worker_routes[worker_id], f"{worker_id} has no route diagnostics"

    all_mtu = {
        route["mtu"]
        for route_map in worker_routes.values()
        for route in route_map.values()
    }
    assert all_mtu == {expected_mtu}, f"unsafe MTU detected: {worker_routes}"
    selected_port, port_owner, owner_type, stale_cleanup = _select_rendezvous_port(
        wrappers[0][0],
        plan.master_port,
    )
    plan = build_launch_plan(
        workers,
        runtime=cluster.runtime,
        smoke_program="examples/distributed_smoke/smoke.py",
        master_addr=plan.master_addr,
        master_port=selected_port,
        backend=plan.backend,
    )

    results: dict[int, dict[str, Any]] = {}

    def launch(rank: int) -> None:
        wrapper, worker = wrappers[rank]
        peer_ips = [
            str(other.host)
            for other in workers
            if str(other.worker_id) != str(worker.worker_id)
        ]
        results[rank] = _launch_rank(
            wrapper,
            worker_id=str(worker.worker_id),
            worker_ip=str(worker.host),
            peer_ips=peer_ips,
            rank=rank,
            world_size=plan.world_size,
            master_addr=plan.master_addr,
            master_port=plan.master_port,
            backend=plan.backend,
            run_id=run_id,
        )

    threads = [
        threading.Thread(target=launch, args=(rank,), name=f"t107-rank-{rank}")
        for rank in (0, 1, 2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence = {
        "test": "test_live_three_worker_smoke",
        "task": "T107",
        "run_id": run_id,
        "backend": plan.backend,
        "master_addr": plan.master_addr,
        "master_port": plan.master_port,
        "port_29500_owner": port_owner,
        "port_29500_owner_type": owner_type,
        "stale_process_cleanup": stale_cleanup,
        "workers": [
            {
                "worker_id": str(worker.worker_id),
                "host": str(worker.host),
                "configured_distro": worker.runtime_distro,
                "preflight": preflight[str(worker.worker_id)],
                "mtu_fix": mtu_fixes.get(str(worker.worker_id)),
                "routes": worker_routes[str(worker.worker_id)],
            }
            for worker in workers
        ],
        "ranks": [results[rank] for rank in (0, 1, 2)],
        "outcome": "PASS" if all(_rank_ok(results[rank]) for rank in (0, 1, 2)) else "FAIL",
    }
    _EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")

    rank_payloads = []
    for rank in (0, 1, 2):
        payload = results[rank].get("result")
        assert payload is not None, (
            f"rank {rank} produced no smoke payload; "
            f"stdout tail={(results[rank]['stdout'] or '')[-800:]}; "
            f"stderr tail={(results[rank]['stderr'] or '')[-800:]}"
        )
        rank_payloads.append(payload)

    assert sorted(payload["rank"] for payload in rank_payloads) == [0, 1, 2]
    assert all(payload["world_size"] == 3 for payload in rank_payloads)
    assert all(payload["local_world_size"] == 1 for payload in rank_payloads)
    assert all(payload["local_rank"] == 0 for payload in rank_payloads)
    assert all(payload["backend"] == "nccl" for payload in rank_payloads)
    assert all(payload["master_addr"] == plan.master_addr for payload in rank_payloads)
    assert all(payload["master_port"] == plan.master_port for payload in rank_payloads)
    assert all(payload["barrier_ok"] is True for payload in rank_payloads)
    assert all(payload["broadcast_ok"] is True for payload in rank_payloads)
    assert all(payload["ring_ok"] is True for payload in rank_payloads)
    assert all(payload["all_reduce_ok"] is True for payload in rank_payloads)
    assert all(payload["destroy_ok"] is True for payload in rank_payloads)
    assert all(payload["expected_all_reduce"] == [6.0] for payload in rank_payloads)
    assert all(payload["all_reduce_tensor"] == [6.0] for payload in rank_payloads)
    assert all(payload["network_interface"] for payload in rank_payloads)
    assert all(_rank_ok(results[rank]) for rank in (0, 1, 2)), json.dumps(evidence, indent=2)
