"""Two-host backward + activation-gradient return verification (T073)."""

from __future__ import annotations

import base64
import json
import math
import re
import threading
from pathlib import PurePosixPath
from typing import Any

from tests.multi_host.test_stage_placement import _build_wrapper, parse_placement_evidence

BACKWARD_MARKER = "T073_BACKWARD_EVIDENCE "
_INTERFACE_RE = re.compile(r"\bdev\s+(\S+)")


def parse_backward_evidence(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(BACKWARD_MARKER):
            try:
                payload = json.loads(stripped[len(BACKWARD_MARKER):])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


def _discover_interface(wrapper: Any, peer_ip: str) -> str:
    route = wrapper.run(f"ip route get {peer_ip}", timeout=10)
    text = (route.stdout or "").strip()
    match = _INTERFACE_RE.search(text)
    if not match:
        raise AssertionError(f"interface resolution failed: {text!r}")
    return match.group(1)


def _cleanup_t073_processes(wrapper: Any) -> None:
    remote_script = PurePosixPath(
        "/home/shardgrid/Code/ShardGrid/examples/models/train_pipeline.py"
    )
    wrapper.run(f"pkill -9 -f '{remote_script}' || true", timeout=15.0)


def test_parse_backward_evidence() -> None:
    payload = {
        "rank": 1,
        "stage_id": "stage1",
        "loss": 7.0,
        "loss_isfinite": True,
        "stage1_backward_ok": True,
    }
    parsed = parse_backward_evidence(
        "noise\n" + BACKWARD_MARKER + json.dumps(payload) + "\n"
    )
    assert parsed is not None
    assert parsed["stage1_backward_ok"] is True
    assert parse_backward_evidence("nothing") is None


def test_live_backward_transfer_on_two_workers() -> None:
    """Real two-host backward + gradient return (opt-in)."""
    remote_root = "/home/shardgrid/Code/ShardGrid"
    files_to_stage = {
        "examples/models/train_pipeline.py": open(
            "examples/models/train_pipeline.py", encoding="utf-8"
        ).read(),
        "examples/models/stage0.py": open("examples/models/stage0.py", encoding="utf-8").read(),
        "examples/models/stage1.py": open("examples/models/stage1.py", encoding="utf-8").read(),
        "examples/models/minimal_transformer.py": open(
            "examples/models/minimal_transformer.py", encoding="utf-8"
        ).read(),
        "examples/models/static_parallel_plan.yaml": open(
            "examples/models/static_parallel_plan.yaml", encoding="utf-8"
        ).read(),
    }

    w0, ip0, id0 = _build_wrapper("gpu4060")
    w1, ip1, id1 = _build_wrapper("gpu1060")
    assert id0 == "gpu4060" and id1 == "gpu1060"
    iface0 = _discover_interface(w0, ip1)
    iface1 = _discover_interface(w1, ip0)

    for wrapper in (w0, w1):
        wrapper.run(
            f"mkdir -p {remote_root}/examples/models && touch "
            f"{remote_root}/examples/__init__.py {remote_root}/examples/models/__init__.py",
            timeout=30,
        )
        for rel, content in files_to_stage.items():
            target = f"{remote_root}/{rel}"
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            script = (
                "import base64\n"
                "from pathlib import Path\n"
                f"Path({target!r}).write_bytes(base64.b64decode({encoded!r}))\n"
            )
            result = wrapper.run_script(script, timeout=60)
            assert result.ok, f"failed to stage {rel}: {(result.stderr or result.stdout)[-300:]}"
        _cleanup_t073_processes(wrapper)

    results: dict[int, Any] = {}

    def launch(wrapper: Any, rank: int, iface: str) -> None:
        command = (
            f"RANK={rank} WORLD_SIZE=2 LOCAL_RANK=0 "
            f"MASTER_ADDR={ip0} MASTER_PORT=29773 "
            f"NCCL_SOCKET_IFNAME={iface} GLOO_SOCKET_IFNAME={iface} "
            f"NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 NCCL_NET=Socket "
            f"SHARDGRID_PIPELINE_TASK=t073 PYTHONUNBUFFERED=1 "
            f"PYTHONPATH={remote_root} /home/shardgrid/miniconda3/envs/shardgrid/bin/python "
            f"{remote_root}/examples/models/train_pipeline.py"
        )
        results[rank] = wrapper.run(command, timeout=90)

    threads = [
        threading.Thread(target=launch, args=(w0, 0, iface0)),
        threading.Thread(target=launch, args=(w1, 1, iface1)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for wrapper in (w0, w1):
        _cleanup_t073_processes(wrapper)

    rank0 = parse_backward_evidence(results[0].stdout or "")
    rank1 = parse_backward_evidence(results[1].stdout or "")
    place0 = parse_placement_evidence(results[0].stdout or "")
    place1 = parse_placement_evidence(results[1].stdout or "")
    assert rank0 is not None and rank1 is not None
    assert place0 is not None and place1 is not None
    assert results[0].exit_code == 0 and not results[0].timed_out
    assert results[1].exit_code == 0 and not results[1].timed_out

    assert rank0["stage0_forward_ok"] is True
    assert rank0["stage0_backward_ok"] is True
    assert rank0["activation_grad_isfinite"] is True
    assert rank0["stage0_gradients"]["all_params_have_grad"] is True
    assert rank0["stage0_gradients"]["all_grads_finite"] is True

    assert rank1["stage1_forward_ok"] is True
    assert rank1["loss_isfinite"] is True
    assert math.isfinite(rank1["loss"])
    assert rank1["stage1_backward_ok"] is True
    assert rank1["activation_grad_isfinite"] is True
    assert rank1["stage1_gradients"]["all_params_have_grad"] is True
    assert rank1["stage1_gradients"]["all_grads_finite"] is True

    assert rank0["gradient_return"]["shape"] == rank1["gradient_return"]["shape"] == [2, 8, 128]
    assert rank0["gradient_return"]["dtype"] == rank1["gradient_return"]["dtype"] == "float32"
    assert rank0["gradient_return"]["isfinite"] is True
    assert rank1["gradient_return"]["isfinite"] is True

    out0 = results[0].stdout or ""
    out1 = results[1].stdout or ""
    for marker in (
        "STAGE0_FORWARD_END",
        "ACTIVATION_SEND_END",
        "GRADIENT_RECV_BEGIN",
        "GRADIENT_RECV_END",
        "STAGE0_BACKWARD_BEGIN",
        "STAGE0_BACKWARD_END",
        "STAGE0_GRADIENTS_READY",
        "SHUTDOWN_END",
    ):
        assert marker in out0.splitlines(), f"missing rank0 marker {marker}"
    for marker in (
        "ACTIVATION_RECV_END",
        "STAGE1_FORWARD_END",
        "LOSS_READY",
        "STAGE1_BACKWARD_BEGIN",
        "STAGE1_BACKWARD_END",
        "ACTIVATION_GRADIENT_READY",
        "GRADIENT_SEND_BEGIN",
        "GRADIENT_SEND_END",
        "STAGE1_GRADIENTS_READY",
        "SHUTDOWN_END",
    ):
        assert marker in out1.splitlines(), f"missing rank1 marker {marker}"
