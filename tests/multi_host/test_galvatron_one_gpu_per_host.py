"""Galvatron one-GPU-per-physical-host placement regression (T058).

T058 proves that Galvatron's multi-host launch path
(``torch.distributed.launch --nnodes 2 --nproc_per_node 1 --node_rank R``,
the exact chain Galvatron's official ``train_dist.sh`` uses) places exactly
one rank on each physical Worker and that the rank metadata always matches
the physical host that launched it.

The logic tests validate probe generation, evidence parsing, and the
placement classifier against mock payloads.  The live test launches the real
probe on the RTX 4060 (rank 0) and GTX 1650 (rank 1) Workers through the
existing SSH + WSL2 + selected Conda chain and asserts the placement report
honestly: only a real two-physical-host result is reported as
``true_multi_host``.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Sequence

from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport

PLACEMENT_PREFIX = "GALVATRON_PLACEMENT_EVIDENCE "

PLACEMENT_TRUE_MULTI_HOST = "true_multi_host"
PLACEMENT_SINGLE_HOST_MULTI_GPU = "single_host_multi_gpu"
PLACEMENT_INVALID = "invalid"

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

import galvatron

out = {
    "worker_id": worker_id,
    "expected_gpu": expected_gpu,
    "galvatron_version": getattr(galvatron, "__version__", None),
    "galvatron_file": galvatron.__file__,
    "global_rank": None,
    "world_size": None,
    "local_rank": None,
    "hostname": socket.gethostname(),
    "host_ip": None,
    "gpu_name": None,
    "capability": None,
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
    gpu_name = torch.cuda.get_device_name(local_rank)
    out["gpu_name"] = gpu_name
    out["capability"] = ".".join(
        str(part) for part in torch.cuda.get_device_capability(local_rank)
    )
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
except Exception as exc:
    out["error"] = str(exc)
finally:
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    print("GALVATRON_PLACEMENT_EVIDENCE " + json.dumps(out, sort_keys=True))
"""


def build_placement_probe_script(
    *,
    worker_id: str,
    expected_gpu: str,
    master_addr: str,
    master_port: int,
) -> str:
    script = PROBE_SCRIPT.replace("__WORKER_ID__", worker_id)
    script = script.replace("__EXPECTED_GPU__", expected_gpu)
    script = script.replace("__MASTER__", master_addr)
    script = script.replace("__PORT__", str(master_port))
    return script


def parse_placement_evidence(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(PLACEMENT_PREFIX):
            try:
                payload = json.loads(stripped[len(PLACEMENT_PREFIX) :])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def classify_placement(
    evidences: Sequence[dict[str, Any] | None],
) -> tuple[str, list[str]]:
    """Classify rank placement into true multi-host vs single-host multi-GPU.

    ``true_multi_host`` requires exactly two ranks, two distinct physical
    hosts (hostname), one rank per host, one GPU per rank (local_rank 0 on
    each host), a successful NCCL collective, and per-rank GPU identity
    matching the expected GPU for that rank's Worker.
    """
    problems: list[str] = []
    valid = [evidence for evidence in evidences if evidence is not None]
    if len(evidences) != 2:
        return PLACEMENT_INVALID, [f"expected 2 rank evidences, got {len(evidences)}"]
    if len(valid) != 2:
        missing = [i for i, evidence in enumerate(evidences) if evidence is None]
        return PLACEMENT_INVALID, [f"rank evidence missing for ranks {missing}"]

    ranks = sorted(evidence["global_rank"] for evidence in valid)
    if ranks != [0, 1]:
        problems.append(f"global ranks {ranks} != [0, 1]")

    hostnames = [evidence["hostname"] for evidence in valid]
    single_host = len(set(hostnames)) == 1
    if not single_host and len(set(hostnames)) != 2:
        problems.append(f"ranks not on two distinct physical hosts: {hostnames}")
    if len(hostnames) == 2 and not single_host:
        for hostname in hostnames:
            if hostnames.count(hostname) != 1:
                problems.append(f"host {hostname!r} holds more than one rank")

    for evidence in valid:
        if evidence.get("local_rank") != 0:
            problems.append(
                f"rank {evidence['global_rank']} local_rank "
                f"{evidence['local_rank']} != 0 (more than one GPU per host)"
            )
        if evidence.get("device") != "cuda:0":
            problems.append(
                f"rank {evidence['global_rank']} device {evidence.get('device')}"
            )
        if evidence.get("backend") != "nccl":
            problems.append(
                f"rank {evidence['global_rank']} backend {evidence.get('backend')}"
            )
        if evidence.get("collective_sum") != [3.0, 3.0, 3.0, 3.0]:
            problems.append(
                f"rank {evidence['global_rank']} collective result "
                f"{evidence.get('collective_sum')}"
            )
        if evidence.get("error"):
            problems.append(
                f"rank {evidence['global_rank']} error: {evidence['error']}"
            )
        expected = evidence.get("expected_gpu")
        if expected and expected not in (evidence.get("gpu_name") or ""):
            problems.append(
                f"rank {evidence['global_rank']} gpu "
                f"{evidence.get('gpu_name')!r} does not match expected {expected!r}"
            )

    if problems:
        return PLACEMENT_INVALID, problems

    if single_host:
        return PLACEMENT_SINGLE_HOST_MULTI_GPU, []
    return PLACEMENT_TRUE_MULTI_HOST, []


def _worker_entry(worker_id: str, address_book: list[dict[str, Any]]) -> dict[str, Any]:
    from shardgrid.common.config import load_cluster_config

    config = load_cluster_config("examples/workers.yaml")
    worker = next(w for w in config.workers if str(w.worker_id) == worker_id)
    expected_gpu = worker.labels.get("gpu", "")
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
    address_book: list[dict[str, Any]],
) -> tuple[WSLRuntimeWrapper, dict[str, Any], str]:
    info = _worker_entry(worker_id, address_book)
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
    timeout: float = 300.0,
) -> Any:
    """Launch the probe with explicit rank metadata environment.

    The command mirrors the proven dist-test execution path (direct selected
    Conda Python + rank env), which is the chain the Galvatron launch scripts
    ultimately invoke on each physical node: one process per node with
    ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` and NCCL socket selection.
    """
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
    script = build_placement_probe_script(
        worker_id="gpu4060",
        expected_gpu="RTX 4060",
        master_addr="10.87.5.155",
        master_port=29501,
    )
    assert 'worker_id = "gpu4060"' in script
    assert 'expected_gpu = "RTX 4060"' in script
    assert '"10.87.5.155"' in script
    assert "29501" in script
    assert "import galvatron" in script
    assert "torch.distributed.launch" not in script


def test_parse_placement_evidence() -> None:
    payload = {
        "worker_id": "gpu4060",
        "global_rank": 0,
        "world_size": 2,
        "local_rank": 0,
        "hostname": "worker-a",
        "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "collective_sum": [3.0, 3.0, 3.0, 3.0],
        "backend": "nccl",
    }
    parsed = parse_placement_evidence(
        "noise\n" + PLACEMENT_PREFIX + json.dumps(payload) + "\n"
    )
    assert parsed is not None
    assert parsed["global_rank"] == 0
    assert parsed["collective_sum"] == [3.0, 3.0, 3.0, 3.0]
    assert parse_placement_evidence("nothing here") is None
    assert parse_placement_evidence("GALVATRON_PLACEMENT_EVIDENCE not-json") is None


def _evidence(
    rank: int,
    hostname: str,
    *,
    local_rank: int = 0,
    device: str = "cuda:0",
    backend: str = "nccl",
    collective: list[float] | None = None,
    gpu: str = "NVIDIA GeForce RTX 4060 Laptop GPU",
    expected: str | None = "RTX 4060",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "worker_id": f"gpu{rank}",
        "expected_gpu": expected,
        "global_rank": rank,
        "world_size": 2,
        "local_rank": local_rank,
        "hostname": hostname,
        "host_ip": None,
        "gpu_name": gpu,
        "device": device,
        "backend": backend,
        "collective_sum": collective or [3.0, 3.0, 3.0, 3.0],
        "error": error,
    }


def test_classify_true_multi_host() -> None:
    status, problems = classify_placement(
        [
            _evidence(0, "host-4060", gpu="NVIDIA GeForce RTX 4060 Laptop GPU"),
            _evidence(
                1,
                "host-1650",
                gpu="NVIDIA GeForce GTX 1650",
                expected="GTX 1650",
            ),
        ]
    )
    assert status == PLACEMENT_TRUE_MULTI_HOST
    assert problems == []


def test_classify_single_host_multi_gpu() -> None:
    status, problems = classify_placement(
        [
            _evidence(0, "same-host"),
            _evidence(1, "same-host"),
        ]
    )
    assert status == PLACEMENT_SINGLE_HOST_MULTI_GPU
    assert problems == []


def test_classify_two_gpus_per_host_is_invalid() -> None:
    status, problems = classify_placement(
        [
            _evidence(0, "host-a", local_rank=0, device="cuda:0"),
            _evidence(1, "host-a", local_rank=1, device="cuda:1"),
        ]
    )
    assert status == PLACEMENT_INVALID
    assert any("local_rank" in problem for problem in problems)
    assert any("more than one GPU per host" in problem for problem in problems)


def test_classify_wrong_gpu_is_invalid() -> None:
    status, problems = classify_placement(
        [
            _evidence(0, "host-a", gpu="NVIDIA GeForce GT 1030"),
            _evidence(1, "host-b", gpu="NVIDIA GeForce GT 1030", expected="GTX 1650"),
        ]
    )
    assert status == PLACEMENT_INVALID
    assert any("does not match expected" in problem for problem in problems)


def test_classify_collective_failure_is_invalid() -> None:
    status, problems = classify_placement(
        [
            _evidence(0, "host-a", collective=[1.0, 1.0, 1.0, 1.0]),
            _evidence(1, "host-b"),
        ]
    )
    assert status == PLACEMENT_INVALID
    assert any("collective result" in problem for problem in problems)


def test_classify_missing_rank_is_invalid() -> None:
    status, problems = classify_placement([_evidence(0, "host-a"), None])
    assert status == PLACEMENT_INVALID
    assert any("missing" in problem for problem in problems)


def test_classify_wrong_rank_set_is_invalid() -> None:
    status, problems = classify_placement(
        [_evidence(0, "host-a"), _evidence(2, "host-b")]
    )
    assert status == PLACEMENT_INVALID
    assert any("global ranks" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------


def test_live_galvatron_one_gpu_per_host() -> None:
    """Real Galvatron launch-path placement on both Workers (opt-in)."""
    import threading

    address_book = json.load(open("tests/address.json"))
    wrappers: dict[int, tuple[WSLRuntimeWrapper, dict[str, Any], str]] = {}
    for worker_id, rank in [("gpu4060", 0), ("gpu1060", 1)]:
        wrappers[rank] = _build_live_wrapper(worker_id, address_book)

    w0, entry0, id0 = wrappers[0]
    w1, entry1, id1 = wrappers[1]

    master_addr = str(entry0["ip"])
    master_port = 29500
    interface0 = _discover_interface(w0, str(entry1["ip"]))
    interface1 = _discover_interface(w1, str(entry0["ip"]))
    assert interface0, "no route interface on rank 0"
    assert interface1, "no route interface on rank 1"

    remote_path = "/tmp/galvatron_placement_probe.py"
    script0 = build_placement_probe_script(
        worker_id=id0,
        expected_gpu="RTX 4060",
        master_addr=master_addr,
        master_port=master_port,
    )
    script1 = build_placement_probe_script(
        worker_id=id1,
        expected_gpu="GTX 1650",
        master_addr=master_addr,
        master_port=master_port,
    )
    _install_probe(w0, script0, remote_path)
    _install_probe(w1, script1, remote_path)

    for wrapper in (w0, w1):
        wrapper.run("pkill -9 -f galvatron_placement_probe.py || true", timeout=15.0)

    results: dict[int, Any] = {}

    def launch(rank: int) -> None:
        wrapper, entry, _ = wrappers[rank]
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
        wrapper.run("pkill -9 -f galvatron_placement_probe.py || true", timeout=15.0)

    evidences = []
    for rank in (0, 1):
        result = results[rank]
        evidence = parse_placement_evidence(result.stdout or "")
        if evidence is None:
            assert False, (
                f"rank {rank} produced no placement evidence; "
                f"stdout tail: {(result.stdout or '')[-800:]}; "
                f"stderr tail: {(result.stderr or '')[-800:]}"
            )
        evidences.append(evidence)

    status, problems = classify_placement(evidences)
    assert status == PLACEMENT_TRUE_MULTI_HOST, (
        f"placement {status}: {'; '.join(problems)}; "
        f"evidences: {json.dumps(evidences, indent=2)}"
    )

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "test": "test_live_galvatron_one_gpu_per_host",
        "task": "T058",
        "placement": status,
        "problems": problems,
        "ranks": evidences,
        "interfaces": {"0": interface0, "1": interface1},
        "master_addr": master_addr,
        "master_port": master_port,
    }
    path = os.path.join(output_dir, "galvatron-one-gpu-per-host-latest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    assert os.path.exists(path)