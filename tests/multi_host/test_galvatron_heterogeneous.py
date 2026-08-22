"""Galvatron heterogeneous RTX 4060 + GTX 1650 behavior regression (T059).

T059 tests Galvatron's heterogeneous two-GPU behavior: rank placement on
physically different GPUs (RTX 4060 + GTX 1650), memory constraints per rank,
and explicit pipeline stage placement.  It reuses the T058 launch chain
(SSH -> WSL2 -> selected Conda, one rank per physical host) and records both
runtime evidence (GPU identity, VRAM, real NCCL collective) and planner
evidence (explicit ``pp_deg=2`` stage assignment per rank with a real memory
allocation against each GPU).

The outcome is one of:

- ``accepted``: both ranks placed correctly, real per-stage memory allocation
  succeeded on both GPUs, capability >= 7.5, and headroom is comfortable.
- ``rejected``: any rank failed GPU identity, NCCL, or the real memory
  allocation for its assigned stage.
- ``experimental``: placement works but only with tight headroom (or the
  smaller GPU carries more than half the stage budget).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Sequence

from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport

HET_PREFIX = "GALVATRON_HET_EVIDENCE "

HET_ACCEPTED = "accepted"
HET_REJECTED = "rejected"
HET_EXPERIMENTAL = "experimental"

MIN_CAPABILITY = (7, 5)

PROBE_SCRIPT = """
import json
import os
import platform
import socket
import subprocess
import sys

import torch
import torch.distributed as dist

worker_id = "__WORKER_ID__"
expected_gpu = "__EXPECTED_GPU__"
master = "__MASTER__"
port = __PORT__
stage_id = __STAGE_ID__
alloc_mb = __ALLOC_MB__

import galvatron

out = {
    "worker_id": worker_id,
    "expected_gpu": expected_gpu,
    "stage_id": stage_id,
    "galvatron_version": getattr(galvatron, "__version__", None),
    "global_rank": None,
    "world_size": None,
    "local_rank": None,
    "hostname": socket.gethostname(),
    "host_ip": None,
    "gpu_name": None,
    "capability": None,
    "gpu_total_vram_mb": None,
    "gpu_free_vram_mb": None,
    "alloc_mb": alloc_mb,
    "alloc_ok": False,
    "device": None,
    "backend": None,
    "collective_sum": None,
    "torch_version": torch.__version__,
    "torch_cuda_version": str(torch.version.cuda),
    "python_version": platform.python_version(),
    "error": None,
}

try:
    try:
        host_ip = subprocess.check_output(
            ["hostname", "-I"], text=True, timeout=10
        ).split()[0]
    except Exception:
        host_ip = None
    out["host_ip"] = host_ip

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    out["local_rank"] = local_rank
    out["global_rank"] = rank
    out["world_size"] = world

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    out["gpu_name"] = torch.cuda.get_device_name(local_rank)
    out["capability"] = ".".join(
        str(part) for part in torch.cuda.get_device_capability(local_rank)
    )
    out["gpu_total_vram_mb"] = (
        torch.cuda.get_device_properties(local_rank).total_memory // (2 ** 20)
    )
    out["gpu_free_vram_mb"] = torch.cuda.mem_get_info(local_rank)[0] // (2 ** 20)
    out["device"] = f"cuda:{local_rank}"

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master}:{port}",
        rank=rank,
        world_size=world,
    )
    out["backend"] = str(dist.get_backend())

    dist.barrier()
    tensor = torch.full((4,), float(rank + 1), dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    out["collective_sum"] = tensor.detach().cpu().tolist()

    if alloc_mb > 0:
        probe = torch.zeros(alloc_mb * (2 ** 20) // 4, dtype=torch.float32, device=device)
        torch.cuda.synchronize()
        out["alloc_ok"] = bool(torch.isfinite(probe).all().item())
        del probe
        torch.cuda.synchronize()
except Exception as exc:
    out["error"] = str(exc)
finally:
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    print("GALVATRON_HET_EVIDENCE " + json.dumps(out, sort_keys=True))
"""


def build_heterogeneous_probe_script(
    *,
    worker_id: str,
    expected_gpu: str,
    master_addr: str,
    master_port: int,
    stage_id: int,
    alloc_mb: int,
) -> str:
    script = PROBE_SCRIPT.replace("__WORKER_ID__", worker_id)
    script = script.replace("__EXPECTED_GPU__", expected_gpu)
    script = script.replace("__MASTER__", master_addr)
    script = script.replace("__PORT__", str(master_port))
    script = script.replace("__STAGE_ID__", str(stage_id))
    script = script.replace("__ALLOC_MB__", str(alloc_mb))
    return script


def parse_heterogeneous_evidence(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(HET_PREFIX):
            try:
                payload = json.loads(stripped[len(HET_PREFIX) :])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def _capability_tuple(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def evaluate_heterogeneous(
    evidences: Sequence[dict[str, Any] | None],
    *,
    min_capability: tuple[int, int] = MIN_CAPABILITY,
) -> tuple[str, list[str], list[str]]:
    """Decide heterogeneous acceptance from real per-rank evidence.

    Returns ``(status, problems, notes)`` where status is one of
    ``accepted`` / ``rejected`` / ``experimental``.
    """
    problems: list[str] = []
    notes: list[str] = []

    if len(evidences) != 2:
        return HET_REJECTED, [f"expected 2 rank evidences, got {len(evidences)}"], notes
    valid = [evidence for evidence in evidences if evidence is not None]
    if len(valid) != 2:
        missing = [i for i, evidence in enumerate(evidences) if evidence is None]
        return HET_REJECTED, [f"rank evidence missing for ranks {missing}"], notes

    hostnames = [evidence["hostname"] for evidence in valid]
    if len(set(hostnames)) != 2:
        problems.append(f"ranks not on two physical hosts: {hostnames}")

    stage_ids: list[int] = []
    total_allocated_mb = 0
    for evidence in valid:
        rank = evidence["global_rank"]
        if evidence.get("error"):
            problems.append(f"rank {rank} error: {evidence['error']}")
        if evidence.get("collective_sum") != [3.0, 3.0, 3.0, 3.0]:
            problems.append(f"rank {rank} collective {evidence.get('collective_sum')}")
        if evidence.get("backend") != "nccl":
            problems.append(f"rank {rank} backend {evidence.get('backend')}")
        expected = evidence.get("expected_gpu")
        if expected and expected not in (evidence.get("gpu_name") or ""):
            problems.append(
                f"rank {rank} gpu {evidence.get('gpu_name')!r} != expected {expected!r}"
            )
        capability = _capability_tuple(evidence.get("capability"))
        if capability is None or capability < min_capability:
            problems.append(
                f"rank {rank} capability {evidence.get('capability')} "
                f"< minimum {min_capability[0]}.{min_capability[1]}"
            )

        alloc_mb = evidence.get("alloc_mb") or 0
        total_allocated_mb += alloc_mb
        if alloc_mb > 0:
            if evidence.get("alloc_ok") is not True:
                problems.append(
                    f"rank {rank} stage allocation of {alloc_mb} MiB failed "
                    f"(free {evidence.get('gpu_free_vram_mb')} MiB on "
                    f"{evidence.get('gpu_name')})"
                )
            else:
                free = evidence.get("gpu_free_vram_mb") or 0
                headroom = 100.0 * (free - alloc_mb) / max(free, 1)
                if headroom < 30.0:
                    notes.append(
                        f"rank {rank} ({evidence.get('gpu_name')}) tight headroom: "
                        f"alloc {alloc_mb} MiB / free {free} MiB ({headroom:.0f}% left)"
                    )
        stage_ids.append(evidence.get("stage_id"))

    if len(set(stage_ids)) != 2 or sorted(stage_ids) != [0, 1]:
        problems.append(f"stage placement not one distinct stage per rank: {stage_ids}")
    else:
        notes.append(
            f"stage placement: pp_deg=2 -> stage {stage_ids[0]} and "
            f"{stage_ids[1]} on two physical hosts"
        )

    smaller_gpu_alloc = 0
    larger_gpu_alloc = 0
    for evidence in valid:
        alloc = evidence.get("alloc_mb") or 0
        total = evidence.get("gpu_total_vram_mb") or 0
        if total <= 5120:
            smaller_gpu_alloc += alloc
        else:
            larger_gpu_alloc += alloc
    if smaller_gpu_alloc > 0 and smaller_gpu_alloc > larger_gpu_alloc:
        notes.append(
            f"smaller GPU (<=5 GiB) carries more stage memory than the larger GPU "
            f"({smaller_gpu_alloc} > {larger_gpu_alloc} MiB)"
        )

    if problems:
        return HET_REJECTED, problems, notes

    risks: list[str] = []
    info_notes: list[str] = []
    for note in notes:
        if "tight headroom" in note or "smaller GPU" in note:
            risks.append(note)
        else:
            info_notes.append(note)
    if risks:
        return HET_EXPERIMENTAL, [], notes
    return HET_ACCEPTED, [], info_notes


def _address_entry(worker_id: str) -> dict[str, Any]:
    from shardgrid.common.config import load_cluster_config

    config = load_cluster_config("examples/workers.yaml")
    worker = next(w for w in config.workers if str(w.worker_id) == worker_id)
    expected_gpu = worker.labels.get("gpu", "")
    address_book = json.load(open("tests/address.json"))
    matches = [
        entry
        for entry in address_book
        if expected_gpu.replace(" ", "").upper()
        in str(entry.get("gpu_model") or "").replace(" ", "").upper()
    ]
    if not matches:
        raise ValueError(f"no address entry for worker {worker_id}")
    return {"config": config, "worker": worker, "entry": matches[0]}


def _build_live_wrapper(
    worker_id: str,
) -> tuple[WSLRuntimeWrapper, dict[str, Any], str]:
    info = _address_entry(worker_id)
    config = info["config"]
    worker = info["worker"]
    entry = info["entry"]
    ip = str(entry["ip"])
    resolved = replace(
        worker,
        host=ip,
        ssh_user=str(entry["username"]),
        runtime_distro="Ubuntu-22.04",
    )
    transport = SSHTransport(
        SSHOptions.from_ssh_config(
            config.ssh,
            host=ip,
            user=resolved.ssh_user,
            port=resolved.ssh_port,
        )
    )
    return (
        WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(resolved, config.runtime),
            transport,
        ),
        entry,
        str(resolved.worker_id),
    )


def _discover_interface(wrapper: WSLRuntimeWrapper, peer_ip: str) -> str | None:
    import re

    result = wrapper.run(f"ip route get {peer_ip}", timeout=10.0)
    if not result.ok:
        return None
    match = re.search(r"\bdev\s+(\S+)", (result.stdout or "").strip())
    return match.group(1) if match else None


def _install_probe(
    wrapper: WSLRuntimeWrapper, script: str, remote_path: str
) -> None:
    import base64

    from shardgrid.transport.runtime import wrap_wsl_direct_command

    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    command = f"echo {encoded} | base64 -d > {remote_path}"
    remote = wrap_wsl_direct_command(
        wrapper.config.distro,
        wrapper.config.user or "shardgrid",
        command,
    )
    result = wrapper.executor.run(remote, timeout=60)
    assert result.ok, f"probe install failed: {result.stderr or result.stdout}"


def _launch_probe(
    wrapper: WSLRuntimeWrapper,
    *,
    node_rank: int,
    master_addr: str,
    master_port: int,
    interface: str,
    remote_path: str,
    timeout: float = 180.0,
) -> Any:
    command = (
        f"RANK={node_rank} WORLD_SIZE=2 LOCAL_RANK=0 "
        f"MASTER_ADDR={master_addr} MASTER_PORT={master_port} "
        f"NCCL_SOCKET_IFNAME={interface} "
        f"GLOO_SOCKET_IFNAME={interface} "
        f"NCCL_SOCKET_FAMILY=AF_INET "
        f"NCCL_IB_DISABLE=1 "
        f"NCCL_NET=Socket "
        f"python {remote_path}"
    )
    return wrapper.run(command, timeout=timeout)


# ---------------------------------------------------------------------------
# Logic tests
# ---------------------------------------------------------------------------


def test_probe_script_injects_parameters() -> None:
    script = build_heterogeneous_probe_script(
        worker_id="gpu4060",
        expected_gpu="RTX 4060",
        master_addr="10.87.5.155",
        master_port=29500,
        stage_id=0,
        alloc_mb=1024,
    )
    assert 'worker_id = "gpu4060"' in script
    assert "stage_id = 0" in script
    assert "alloc_mb = 1024" in script
    assert "import galvatron" in script
    assert "GALVATRON_HET_EVIDENCE" in script


def test_parse_heterogeneous_evidence() -> None:
    payload = {
        "worker_id": "gpu4060",
        "global_rank": 0,
        "stage_id": 0,
        "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "capability": "8.9",
        "alloc_mb": 1024,
        "alloc_ok": True,
        "collective_sum": [3.0, 3.0, 3.0, 3.0],
        "backend": "nccl",
    }
    parsed = parse_heterogeneous_evidence(
        "noise\n" + HET_PREFIX + json.dumps(payload) + "\n"
    )
    assert parsed is not None
    assert parsed["stage_id"] == 0
    assert parsed["alloc_ok"] is True
    assert parse_heterogeneous_evidence("nothing") is None


def _evidence(
    rank: int,
    hostname: str,
    *,
    gpu: str = "NVIDIA GeForce RTX 4060 Laptop GPU",
    expected: str = "RTX 4060",
    capability: str = "8.9",
    total_vram_mb: int = 8188,
    free_vram_mb: int = 7000,
    alloc_mb: int = 2048,
    alloc_ok: bool = True,
    stage_id: int | None = None,
    collective: list[float] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "worker_id": f"gpu{rank}",
        "expected_gpu": expected,
        "stage_id": stage_id if stage_id is not None else rank,
        "global_rank": rank,
        "world_size": 2,
        "local_rank": 0,
        "hostname": hostname,
        "gpu_name": gpu,
        "capability": capability,
        "gpu_total_vram_mb": total_vram_mb,
        "gpu_free_vram_mb": free_vram_mb,
        "alloc_mb": alloc_mb,
        "alloc_ok": alloc_ok,
        "backend": "nccl",
        "collective_sum": collective or [3.0, 3.0, 3.0, 3.0],
        "error": error,
    }


def test_evaluate_accepted_balanced() -> None:
    status, problems, notes = evaluate_heterogeneous(
        [
            _evidence(0, "host-4060", free_vram_mb=7000, alloc_mb=1024),
            _evidence(
                1,
                "host-1650",
                gpu="NVIDIA GeForce GTX 1650",
                expected="GTX 1650",
                capability="7.5",
                total_vram_mb=4096,
                free_vram_mb=3500,
                alloc_mb=768,
            ),
        ]
    )
    assert status == HET_ACCEPTED, (problems, notes)
    assert problems == []


def test_evaluate_experimental_tight_headroom() -> None:
    status, problems, notes = evaluate_heterogeneous(
        [
            _evidence(0, "host-4060", free_vram_mb=3000, alloc_mb=2600),
            _evidence(
                1,
                "host-1650",
                gpu="NVIDIA GeForce GTX 1650",
                expected="GTX 1650",
                capability="7.5",
                total_vram_mb=4096,
                free_vram_mb=3500,
                alloc_mb=768,
            ),
        ]
    )
    assert status == HET_EXPERIMENTAL, (problems, notes)
    assert problems == []
    assert any("tight headroom" in note for note in notes)


def test_evaluate_rejected_oom() -> None:
    status, problems, notes = evaluate_heterogeneous(
        [
            _evidence(0, "host-4060", alloc_ok=True, alloc_mb=1024),
            _evidence(
                1,
                "host-1650",
                gpu="NVIDIA GeForce GTX 1650",
                expected="GTX 1650",
                capability="7.5",
                total_vram_mb=4096,
                free_vram_mb=3500,
                alloc_mb=4096,
                alloc_ok=False,
            ),
        ]
    )
    assert status == HET_REJECTED
    assert any("stage allocation" in problem for problem in problems)


def test_evaluate_rejected_gpu_mismatch() -> None:
    status, problems, notes = evaluate_heterogeneous(
        [
            _evidence(0, "host-4060"),
            _evidence(
                1,
                "host-1650",
                gpu="NVIDIA GeForce RTX 4060 Laptop GPU",
                expected="GTX 1650",
            ),
        ]
    )
    assert status == HET_REJECTED
    assert any("!= expected" in problem for problem in problems)


def test_evaluate_rejected_capability() -> None:
    status, problems, notes = evaluate_heterogeneous(
        [
            _evidence(0, "host-4060"),
            _evidence(
                1,
                "host-1650",
                gpu="NVIDIA GeForce GTX 1650",
                expected="GTX 1650",
                capability="6.1",
            ),
        ]
    )
    assert status == HET_REJECTED
    assert any("capability" in problem for problem in problems)


def test_evaluate_rejected_collective() -> None:
    status, problems, notes = evaluate_heterogeneous(
        [
            _evidence(0, "host-4060"),
            _evidence(1, "host-1650", collective=[1.0, 1.0, 1.0, 1.0]),
        ]
    )
    assert status == HET_REJECTED
    assert any("collective" in problem for problem in problems)


def test_evaluate_rejected_stage_duplicate() -> None:
    status, problems, notes = evaluate_heterogeneous(
        [
            _evidence(0, "host-4060", stage_id=0),
            _evidence(1, "host-1650", stage_id=0),
        ]
    )
    assert status == HET_REJECTED
    assert any("stage" in problem for problem in problems)


def test_evaluate_experimental_smaller_gpu_heavier() -> None:
    status, problems, notes = evaluate_heterogeneous(
        [
            _evidence(0, "host-4060", free_vram_mb=7000, alloc_mb=512),
            _evidence(
                1,
                "host-1650",
                gpu="NVIDIA GeForce GTX 1650",
                expected="GTX 1650",
                capability="7.5",
                total_vram_mb=4096,
                free_vram_mb=3500,
                alloc_mb=2048,
            ),
        ]
    )
    assert status == HET_EXPERIMENTAL, (problems, notes)
    assert any("smaller GPU" in note for note in notes)


def test_evaluate_missing_rank_is_rejected() -> None:
    status, problems, notes = evaluate_heterogeneous([_evidence(0, "host-4060"), None])
    assert status == HET_REJECTED
    assert any("missing" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------


def test_live_galvatron_heterogeneous() -> None:
    """Real heterogeneous RTX 4060 + GTX 1650 run with stage memory allocation."""
    import threading

    wrappers: dict[int, tuple[WSLRuntimeWrapper, dict[str, Any], str]] = {}
    for worker_id, rank in [("gpu4060", 0), ("gpu1060", 1)]:
        wrappers[rank] = _build_live_wrapper(worker_id)

    w0, entry0, id0 = wrappers[0]
    w1, entry1, id1 = wrappers[1]

    master_addr = str(entry0["ip"])
    master_port = 29500
    interface0 = _discover_interface(w0, str(entry1["ip"]))
    interface1 = _discover_interface(w1, str(entry0["ip"]))
    assert interface0, "no route interface on rank 0"
    assert interface1, "no route interface on rank 1"

    remote_path = "/tmp/galvatron_het_probe.py"
    script0 = build_heterogeneous_probe_script(
        worker_id=id0,
        expected_gpu="RTX 4060",
        master_addr=master_addr,
        master_port=master_port,
        stage_id=0,
        alloc_mb=2048,
    )
    script1 = build_heterogeneous_probe_script(
        worker_id=id1,
        expected_gpu="GTX 1650",
        master_addr=master_addr,
        master_port=master_port,
        stage_id=1,
        alloc_mb=1024,
    )
    _install_probe(w0, script0, remote_path)
    _install_probe(w1, script1, remote_path)

    for wrapper in (w0, w1):
        wrapper.run("pkill -9 -f galvatron_het_probe.py || true", timeout=15.0)

    results: dict[int, Any] = {}

    def launch(rank: int) -> None:
        wrapper, _, _ = wrappers[rank]
        results[rank] = _launch_probe(
            wrapper,
            node_rank=rank,
            master_addr=master_addr,
            master_port=master_port,
            interface=interface0 if rank == 0 else interface1,
            remote_path=remote_path,
            timeout=180.0,
        )

    threads = [
        threading.Thread(target=launch, args=(0,)),
        threading.Thread(target=launch, args=(1,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for wrapper in (w0, w1):
        wrapper.run("pkill -9 -f galvatron_het_probe.py || true", timeout=15.0)

    evidences = []
    for rank in (0, 1):
        result = results[rank]
        evidence = parse_heterogeneous_evidence(result.stdout or "")
        if evidence is None:
            assert False, (
                f"rank {rank} produced no heterogeneous evidence; "
                f"stdout tail: {(result.stdout or '')[-800:]}; "
                f"stderr tail: {(result.stderr or '')[-800:]}"
            )
        evidences.append(evidence)

    status, problems, notes = evaluate_heterogeneous(evidences)
    assert status in (HET_ACCEPTED, HET_EXPERIMENTAL), (
        f"heterogeneous {status}: {'; '.join(problems)}; "
        f"evidences: {json.dumps(evidences, indent=2)}"
    )

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "test": "test_live_galvatron_heterogeneous",
        "task": "T059",
        "heterogeneous": status,
        "problems": problems,
        "notes": notes,
        "ranks": evidences,
        "interfaces": {"0": interface0, "1": interface1},
        "master_addr": master_addr,
        "master_port": master_port,
    }
    path = os.path.join(output_dir, "galvatron-heterogeneous-latest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    assert os.path.exists(path)