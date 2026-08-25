"""NCCL P2P matching and API comparison diagnosis for T072."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path, PurePosixPath
from typing import Any

from tests.multi_host.test_activation_transfer import _discover_interface
from tests.multi_host.test_stage_placement import _build_wrapper

from shardgrid.transport.runtime import wrap_wsl_direct_command

EVENT_MARKER = "T072_P2P_EVIDENCE "

_P2P_SCRIPT = r'''
import json
import os
import socket
import time

import torch
import torch.distributed as dist

rank = __RANK__
world = 2
local_rank = 0
master_addr = "__MASTER_ADDR__"
master_port = __MASTER_PORT__
interface = "__INTERFACE__"
mode = "__MODE__"
shape = (2, 8, 128)
dtype = torch.float32
start = time.perf_counter()

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "P2P,NET"
os.environ["NCCL_DEBUG_FILE"] = f"/tmp/t072_p2p_{mode}_rank{rank}.nccl.log"
os.environ["NCCL_SOCKET_IFNAME"] = f"={interface}"
os.environ["GLOO_SOCKET_IFNAME"] = interface
os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NET"] = "Socket"

def mark(name: str) -> float:
    stamp = time.time()
    print(name, flush=True)
    return stamp

payload = {
    "mode": mode,
    "hostname": socket.gethostname(),
    "rank": rank,
    "world_size": world,
    "local_rank": local_rank,
    "device": f"cuda:{local_rank}",
    "shape": list(shape),
    "dtype": "float32",
    "peer": 1 if rank == 0 else 0,
    "send_begin": None,
    "send_end": None,
    "recv_begin": None,
    "recv_end": None,
    "tensor_ok": None,
    "preview": None,
    "elapsed_ms": None,
    "error": None,
}

try:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world,
    )
    print(
        f"COMM rank={dist.get_rank()} world_size={dist.get_world_size()} "
        f"device={device} peer={payload['peer']} mode={mode}",
        flush=True,
    )
    expected = torch.arange(2 * 8 * 128, dtype=dtype, device=device).reshape(shape)
    if mode == "send_recv":
        if rank == 0:
            payload["send_begin"] = mark("SEND_BEGIN")
            dist.send(expected.contiguous(), dst=1)
            torch.cuda.synchronize(device)
            payload["send_end"] = mark("SEND_END")
            payload["tensor_ok"] = True
        else:
            recv = torch.empty(shape, dtype=dtype, device=device)
            payload["recv_begin"] = mark("RECV_BEGIN")
            dist.recv(recv, src=0)
            torch.cuda.synchronize(device)
            payload["recv_end"] = mark("RECV_END")
            payload["tensor_ok"] = bool(torch.equal(recv, expected))
            payload["preview"] = [
                float(recv.flatten()[0].item()),
                float(recv.flatten()[-1].item()),
            ]
    elif mode == "batch":
        if rank == 0:
            payload["send_begin"] = mark("BATCH_SEND_BEGIN")
            works = dist.batch_isend_irecv([dist.P2POp(dist.isend, expected.contiguous(), 1)])
            for work in works:
                work.wait()
            torch.cuda.synchronize(device)
            payload["send_end"] = mark("BATCH_SEND_END")
            payload["tensor_ok"] = True
        else:
            recv = torch.empty(shape, dtype=dtype, device=device)
            payload["recv_begin"] = mark("BATCH_RECV_BEGIN")
            works = dist.batch_isend_irecv([dist.P2POp(dist.irecv, recv, 0)])
            for work in works:
                work.wait()
            torch.cuda.synchronize(device)
            payload["recv_end"] = mark("BATCH_RECV_END")
            payload["tensor_ok"] = bool(torch.equal(recv, expected))
            payload["preview"] = [
                float(recv.flatten()[0].item()),
                float(recv.flatten()[-1].item()),
            ]
    else:
        raise RuntimeError(f"unknown mode: {mode}")
except Exception as exc:
    payload["error"] = str(exc)
finally:
    payload["elapsed_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
    print("T072_P2P_EVIDENCE " + json.dumps(payload, sort_keys=True), flush=True)
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
'''


def _parse(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(EVENT_MARKER):
            try:
                payload = json.loads(stripped[len(EVENT_MARKER):])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def _build_p2p_script(
    *,
    rank: int,
    master_addr: str,
    master_port: int,
    interface: str,
    mode: str,
) -> str:
    script = _P2P_SCRIPT.replace("__RANK__", str(rank))
    script = script.replace("__MASTER_ADDR__", master_addr)
    script = script.replace("__MASTER_PORT__", str(master_port))
    script = script.replace("__INTERFACE__", interface)
    script = script.replace("__MODE__", mode)
    return script


def _read_remote_file(wrapper: Any, path: str) -> str:
    result = wrapper.run(f"test -f {path} && cat {path} || true", timeout=30)
    return result.stdout or ""


def _install_remote_script(wrapper: Any, script: str, *, mode: str) -> str:
    remote_root = PurePosixPath("/tmp/t072_p2p_api_compare")
    remote_path = str(remote_root / f"p2p_{mode}.py")
    init = wrap_wsl_direct_command(
        wrapper.config.distro,
        wrapper.config.user or "shardgrid",
        f"mkdir -p {remote_root}",
    )
    init_result = wrapper.executor.run(init, timeout=60)
    assert init_result.ok, (
        f"failed to initialize {remote_root}: "
        f"stdout={(init_result.stdout or '')[-400:]}; "
        f"stderr={(init_result.stderr or '')[-400:]}"
    )
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    installer = (
        "import base64\n"
        "from pathlib import Path\n\n"
        f"Path({remote_path!r}).write_bytes(base64.b64decode({encoded!r}))\n"
    )
    install_result = wrapper.run_script(installer, timeout=60)
    assert install_result.ok, (
        "failed to install remote p2p api comparison script: "
        f"stdout={(install_result.stdout or '')[-400:]}; "
        f"stderr={(install_result.stderr or '')[-400:]}"
    )
    return remote_path


def _run_mode(
    *,
    mode: str,
    master_port: int,
    w0: Any,
    w1: Any,
    ip0: str,
    ip1: str,
    iface0: str,
    iface1: str,
    output_dir: Path,
) -> dict[str, Any]:
    script0 = _build_p2p_script(
        rank=0,
        master_addr=ip0,
        master_port=master_port,
        interface=iface0,
        mode=mode,
    )
    script1 = _build_p2p_script(
        rank=1,
        master_addr=ip0,
        master_port=master_port,
        interface=iface1,
        mode=mode,
    )
    remote0 = _install_remote_script(w0, script0, mode=mode)
    remote1 = _install_remote_script(w1, script1, mode=mode)

    results: dict[int, Any] = {}

    def launch(wrapper: Any, rank: int, iface: str, remote_path: str) -> None:
        command = (
            f"NCCL_SOCKET_IFNAME={iface} "
            f"GLOO_SOCKET_IFNAME={iface} "
            "NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 NCCL_NET=Socket "
            f"python {remote_path}"
        )
        results[rank] = wrapper.run(command, timeout=180)

    threads = [
        threading.Thread(target=launch, args=(w0, 0, iface0, remote0)),
        threading.Thread(target=launch, args=(w1, 1, iface1, remote1)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rank0 = results[0]
    rank1 = results[1]
    payload0 = _parse(rank0.stdout or "")
    payload1 = _parse(rank1.stdout or "")
    nccl0 = _read_remote_file(w0, f"/tmp/t072_p2p_{mode}_rank0.nccl.log")
    nccl1 = _read_remote_file(w1, f"/tmp/t072_p2p_{mode}_rank1.nccl.log")

    summary = {
        "rank0": {
            "exit_code": rank0.exit_code,
            "timed_out": rank0.timed_out,
            "payload": payload0,
            "stdout": rank0.stdout,
            "stderr": rank0.stderr,
            "nccl_log": nccl0,
        },
        "rank1": {
            "exit_code": rank1.exit_code,
            "timed_out": rank1.timed_out,
            "payload": payload1,
            "stdout": rank1.stdout,
            "stderr": rank1.stderr,
            "nccl_log": nccl1,
        },
    }
    (output_dir / f"{mode}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / f"{mode}.rank0.stdout.txt").write_text(
        rank0.stdout or "", encoding="utf-8"
    )
    (output_dir / f"{mode}.rank0.stderr.txt").write_text(
        rank0.stderr or "", encoding="utf-8"
    )
    (output_dir / f"{mode}.rank1.stdout.txt").write_text(
        rank1.stdout or "", encoding="utf-8"
    )
    (output_dir / f"{mode}.rank1.stderr.txt").write_text(
        rank1.stderr or "", encoding="utf-8"
    )
    (output_dir / f"{mode}.rank0.nccl.log").write_text(nccl0, encoding="utf-8")
    (output_dir / f"{mode}.rank1.nccl.log").write_text(nccl1, encoding="utf-8")
    return summary


def test_live_t072_p2p_send_recv_diagnosis() -> None:
    output_dir = Path("/var/tmp/shardgrid/t072_p2p_api_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    w0, ip0, id0 = _build_wrapper("gpu4060")
    w1, ip1, id1 = _build_wrapper("gpu1060")
    assert id0 == "gpu4060" and id1 == "gpu1060"

    iface0 = _discover_interface(w0, ip1)
    iface1 = _discover_interface(w1, ip0)

    send_recv = _run_mode(
        mode="send_recv",
        master_port=29600,
        w0=w0,
        w1=w1,
        ip0=ip0,
        ip1=ip1,
        iface0=iface0,
        iface1=iface1,
        output_dir=output_dir,
    )
    batch = _run_mode(
        mode="batch",
        master_port=29601,
        w0=w0,
        w1=w1,
        ip0=ip0,
        ip1=ip1,
        iface0=iface0,
        iface1=iface1,
        output_dir=output_dir,
    )

    result = {"send_recv": send_recv, "batch": batch}
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
