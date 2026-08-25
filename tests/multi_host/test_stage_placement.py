"""Two-host stage placement verification (T071).

Proves on real hardware that Stage0 lives on rank 0 (RTX 4060) and Stage1
lives on rank 1 (GTX 1650), that each rank really holds its own trainable
parameters on its local cuda:0, that the two parameter sets do not overlap,
and that their union covers the full T068/T069 model - i.e. a real split,
not a copy and not a single-host full-model placement.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any

from examples.models.minimal_transformer import (
    MinimalTransformerConfig,
    build_minimal_transformer,
)

from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport

EVENT_MARKER = "STAGE_PLACEMENT_EVIDENCE "

MODEL_CONFIG = MinimalTransformerConfig(
    vocab_size=1024, hidden_size=128, num_hidden_layers=2,
    num_attention_heads=4, max_seq_length=64,
)


def parse_placement_evidence(stdout: str) -> dict[str, Any] | None:
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


def stage_parameter_names_to_full_keys(names: list[str]) -> set[str]:
    """Map stage parameter names (stage0/stage1 domain) to full-model keys.

    ``block0.*``/``block1.*`` in the stage domain correspond to
    ``blocks.0.*``/``blocks.1.*`` in the full model domain; other names
    (embed, pos, ln, lm_head) are identical.
    """
    mapped: set[str] = set()
    for name in names:
        if name.startswith("block0."):
            mapped.add("blocks.0." + name[len("block0."):])
        elif name.startswith("block1."):
            mapped.add("blocks.1." + name[len("block1."):])
        else:
            mapped.add(name)
    return mapped


def full_model_parameter_keys() -> set[str]:
    full = build_minimal_transformer(MODEL_CONFIG, seed=42)
    return set(dict(full.named_parameters()).keys())


def expected_stage0_keys(full_keys: set[str]) -> set[str]:
    return {
        name for name in full_keys if name.startswith(("embed", "pos", "blocks.0"))
    }


def expected_stage1_keys(full_keys: set[str]) -> set[str]:
    return {
        name for name in full_keys
        if name.startswith(("blocks.1", "ln", "lm_head"))
    }


def _build_wrapper(worker_id: str) -> tuple[WSLRuntimeWrapper, str, str]:
    from shardgrid.common.config import load_cluster_config

    config = load_cluster_config("examples/workers.yaml")
    address_book = json.load(open("tests/address.json"))
    worker = next(w for w in config.workers if str(w.worker_id) == worker_id)
    gpu_label = worker.labels.get("gpu", "").upper().replace(" ", "")
    entry = next(
        e
        for e in address_book
        if gpu_label in str(e.get("gpu_model") or "").replace(" ", "").upper()
    )
    ip = str(entry["ip"])
    resolved = replace(
        worker,
        host=ip,
        ssh_user=str(entry["username"]),
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
        ip,
        str(resolved.worker_id),
    )


def _cleanup_t071_processes(wrapper: WSLRuntimeWrapper) -> None:
    remote_script = PurePosixPath("/tmp/t071/examples/models/train_pipeline.py")
    wrapper.run(f"pkill -9 -f '{remote_script}' || true", timeout=15.0)


def test_parse_placement_evidence() -> None:
    payload = {
        "hostname": "ldj",
        "rank": 0,
        "stage_id": "stage0",
        "parameter_count": 336384,
        "parameter_names": ["embed.weight", "block0.qkv.weight"],
        "parameter_devices": {"embed.weight": "cuda:0"},
    }
    parsed = parse_placement_evidence(
        "noise\n" + EVENT_MARKER + json.dumps(payload) + "\n"
    )
    assert parsed is not None
    assert parsed["stage_id"] == "stage0"
    assert parse_placement_evidence("nothing") is None


def test_parameter_name_mapping_and_expected_keys() -> None:
    full_keys = full_model_parameter_keys()
    stage0_names = sorted(
        name for name in full_keys if name.startswith(("embed", "pos", "blocks.0"))
    )
    stage1_names = sorted(
        name
        for name in full_keys
        if name.startswith(("blocks.1", "ln", "lm_head"))
    )
    # stage-domain names as the runner reports them
    stage0_domain = [
        name.replace("blocks.0.", "block0.") if name.startswith("blocks.0.") else name
        for name in stage0_names
    ]
    stage1_domain = [
        name.replace("blocks.1.", "block1.") if name.startswith("blocks.1.") else name
        for name in stage1_names
    ]
    mapped0 = stage_parameter_names_to_full_keys(stage0_domain)
    mapped1 = stage_parameter_names_to_full_keys(stage1_domain)
    assert mapped0 == expected_stage0_keys(full_keys)
    assert mapped1 == expected_stage1_keys(full_keys)
    assert not mapped0 & mapped1
    assert mapped0 | mapped1 == full_keys


def test_live_stage_placement_on_two_workers() -> None:
    """Real two-host stage placement (opt-in via multi_host marker)."""
    import base64
    import os
    import threading

    from shardgrid.transport.runtime import wrap_wsl_direct_command

    full_keys = full_model_parameter_keys()
    expected0 = expected_stage0_keys(full_keys)
    expected1 = expected_stage1_keys(full_keys)

    w0, ip0, id0 = _build_wrapper("gpu4060")
    w1, ip1, id1 = _build_wrapper("gpu1060")
    assert id0 == "gpu4060" and id1 == "gpu1060"

    import re as _re

    route0 = w0.run(f"ip route get {ip1}", timeout=10)
    iface0 = _re.search(r"\bdev\s+(\S+)", (route0.stdout or "").strip())
    route1 = w1.run(f"ip route get {ip0}", timeout=10)
    iface1 = _re.search(r"\bdev\s+(\S+)", (route1.stdout or "").strip())
    assert iface0 and iface1, "interface resolution failed"

    files = {
        "minimal_transformer.py": open(
            "examples/models/minimal_transformer.py", encoding="utf-8"
        ).read(),
        "stage0.py": open("examples/models/stage0.py", encoding="utf-8").read(),
        "stage1.py": open("examples/models/stage1.py", encoding="utf-8").read(),
        "train_pipeline.py": open(
            "examples/models/train_pipeline.py", encoding="utf-8"
        ).read(),
    }

    def install_file(wrapper: WSLRuntimeWrapper, path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        script = (
            "import base64\n"
            "from pathlib import Path\n\n"
            f"Path({path!r}).write_bytes(base64.b64decode({encoded!r}))\n"
        )
        result = wrapper.run_script(script, timeout=60)
        assert result.ok, (
            f"failed to install {path}: "
            f"stdout={(result.stdout or '')[-400:]}; "
            f"stderr={(result.stderr or '')[-400:]}"
        )

    for wrapper in (w0, w1):
        init = wrap_wsl_direct_command(
            wrapper.config.distro,
            wrapper.config.user or "shardgrid",
            "mkdir -p /tmp/t071/examples/models && "
            "touch /tmp/t071/__init__.py /tmp/t071/examples/__init__.py "
            "/tmp/t071/examples/models/__init__.py",
        )
        wrapper.executor.run(init, timeout=60)
        for name, content in files.items():
            install_file(
                wrapper,
                f"/tmp/t071/examples/models/{name}",
                content,
            )
        _cleanup_t071_processes(wrapper)

    results: dict[int, Any] = {}

    def launch(wrapper: WSLRuntimeWrapper, rank: int, iface: str) -> None:
        command = (
            f"RANK={rank} WORLD_SIZE=2 LOCAL_RANK=0 "
            f"MASTER_ADDR={ip0} MASTER_PORT=29500 "
            f"NCCL_SOCKET_IFNAME={iface} GLOO_SOCKET_IFNAME={iface} "
            f"NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 NCCL_NET=Socket "
            f"PYTHONPATH=/tmp/t071 python /tmp/t071/examples/models/train_pipeline.py"
        )
        results[rank] = wrapper.run(command, timeout=180)

    threads = [
        threading.Thread(target=launch, args=(w0, 0, iface0.group(1))),
        threading.Thread(target=launch, args=(w1, 1, iface1.group(1))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for wrapper in (w0, w1):
        _cleanup_t071_processes(wrapper)

    evidences: dict[int, dict[str, Any]] = {}
    for rank in (0, 1):
        result = results[rank]
        evidence = parse_placement_evidence(result.stdout or "")
        if evidence is None:
            assert False, (
                f"rank {rank} produced no placement evidence; "
                f"stdout tail: {(result.stdout or '')[-800:]}; "
                f"stderr tail: {(result.stderr or '')[-800:]}"
            )
        evidences[rank] = evidence

    rank0 = evidences[0]
    rank1 = evidences[1]

    assert rank0["hostname"] != rank1["hostname"]
    assert rank0["gpu_name"] != rank1["gpu_name"]
    assert rank0["stage_id"] == "stage0"
    assert rank1["stage_id"] == "stage1"
    assert rank0["rank"] == 0 and rank1["rank"] == 1
    assert rank0["world_size"] == 2 and rank1["world_size"] == 2

    for evidence, expected_stage, expected_gpu in (
        (rank0, "stage0", "RTX 4060"),
        (rank1, "stage1", "GTX 1650"),
    ):
        assert evidence["stage_id"] == expected_stage
        assert expected_gpu in evidence["gpu_name"]
        assert evidence["device"] == "cuda:0"
        assert evidence["parameter_count"] > 0
        assert evidence["trainable_parameter_count"] > 0
        assert evidence["trainable_parameter_count"] == evidence["parameter_count"]
        assert evidence["all_parameters_on_device"] is True
        assert evidence["stage_sanity_forward_ok"] is True
        for name, device in evidence["parameter_devices"].items():
            assert device == "cuda:0", f"parameter {name} on {device}"

    # ownership: disjoint + complete coverage of the full model
    keys0 = stage_parameter_names_to_full_keys(rank0["parameter_names"])
    keys1 = stage_parameter_names_to_full_keys(rank1["parameter_names"])
    assert keys0 == expected0, f"stage0 keys mismatch: {sorted(keys0 ^ expected0)}"
    assert keys1 == expected1, f"stage1 keys mismatch: {sorted(keys1 ^ expected1)}"
    assert not keys0 & keys1, "parameter sets overlap"
    assert keys0 | keys1 == full_keys, "parameter coverage incomplete"

    # neither rank holds the full model or a duplicate copy
    full_count = sum(parameter.numel() for parameter in
                     build_minimal_transformer(MODEL_CONFIG, seed=42).parameters())
    assert rank0["parameter_count"] < full_count
    assert rank1["parameter_count"] < full_count
    assert rank0["parameter_count"] + rank1["parameter_count"] == full_count

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "stage-placement-latest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "T071",
                "ranks": {"0": rank0, "1": rank1},
                "ownership": {
                    "disjoint": True,
                    "complete": True,
                    "full_model_parameter_count": full_count,
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    assert os.path.exists(path)
