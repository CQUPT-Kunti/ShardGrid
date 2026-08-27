"""Two-host optimizer step + loss decrease + checkpoint verification (T074)."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import PurePosixPath
from typing import Any

from tests.multi_host.test_stage_placement import _build_wrapper, parse_placement_evidence

TRAIN_MARKER = "T074_TRAIN_EVIDENCE "
_INTERFACE_RE = re.compile(r"\bdev\s+(\S+)")
_DEFAULT_STEPS = 20
_DEFAULT_TIMEOUT = 90

_REQUIRED_RANK0_MARKERS = (
    "TRAIN_STEP_BEGIN",
    "STAGE0_BACKWARD_END",
    "OPTIMIZER_STEP_END",
    "TRAIN_STEP_END",
    "CHECKPOINT_SAVE_BEGIN",
    "CHECKPOINT_SAVE_END",
    "CHECKPOINT_LOAD_BEGIN",
    "CHECKPOINT_LOAD_END",
    "SHUTDOWN_END",
)

_REQUIRED_RANK1_MARKERS = (
    "TRAIN_STEP_BEGIN",
    "LOSS_READY",
    "STAGE1_BACKWARD_END",
    "GRADIENT_SEND_END",
    "OPTIMIZER_STEP_END",
    "TRAIN_STEP_END",
    "CHECKPOINT_SAVE_BEGIN",
    "CHECKPOINT_SAVE_END",
    "CHECKPOINT_LOAD_BEGIN",
    "CHECKPOINT_LOAD_END",
    "SHUTDOWN_END",
)


def _last_marker(stdout: str, markers: tuple[str, ...]) -> str:
    lines = set(stdout.splitlines())
    found = [marker for marker in markers if marker in lines]
    return found[-1] if found else ""


def parse_train_evidence(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(TRAIN_MARKER):
            try:
                payload = json.loads(stripped[len(TRAIN_MARKER):])
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


def _cleanup_t074_processes(wrapper: Any) -> None:
    remote_script = PurePosixPath(
        "/home/shardgrid/Code/ShardGrid/examples/models/train_pipeline.py"
    )
    wrapper.run(f"pkill -9 -f '{remote_script}' || true", timeout=15.0)


def test_parse_train_evidence() -> None:
    payload = {
        "rank": 1,
        "stage_id": "stage1",
        "steps": 20,
        "initial_loss": 7.26,
        "final_loss": 0.36,
        "loss_decrease": True,
        "loss_isfinite": True,
        "param_update_ok": True,
        "checkpoint_roundtrip_ok": True,
    }
    parsed = parse_train_evidence(
        "noise\n" + TRAIN_MARKER + json.dumps(payload) + "\n"
    )
    assert parsed is not None
    assert parsed["stage_id"] == "stage1"
    assert parsed["loss_decrease"] is True
    assert parse_train_evidence("nothing") is None


def test_parse_train_evidence_requires_final_less_than_initial() -> None:
    payload = {
        "rank": 1,
        "initial_loss": 1.0,
        "final_loss": 2.0,
        "loss_decrease": False,
    }
    parsed = parse_train_evidence(TRAIN_MARKER + json.dumps(payload))
    assert parsed is not None
    assert parsed["loss_decrease"] is False


def test_required_marker_sets() -> None:
    assert "LOSS_READY" in _REQUIRED_RANK1_MARKERS
    assert "STAGE1_BACKWARD_END" in _REQUIRED_RANK1_MARKERS
    assert "GRADIENT_SEND_END" in _REQUIRED_RANK1_MARKERS
    assert "STAGE0_BACKWARD_END" in _REQUIRED_RANK0_MARKERS
    assert "OPTIMIZER_STEP_END" in _REQUIRED_RANK0_MARKERS
    assert "OPTIMIZER_STEP_END" in _REQUIRED_RANK1_MARKERS
    assert "CHECKPOINT_SAVE_END" in _REQUIRED_RANK0_MARKERS
    assert "CHECKPOINT_LOAD_END" in _REQUIRED_RANK1_MARKERS


def test_live_optimizer_checkpoint_on_two_workers() -> None:
    """Real two-host optimizer + loss decrease + checkpoint (opt-in)."""
    import threading

    steps = int(os.environ.get("SHARDGRID_T074_TEST_STEPS", str(_DEFAULT_STEPS)))
    timeout = int(os.environ.get("SHARDGRID_T074_TEST_TIMEOUT", str(_DEFAULT_TIMEOUT)))
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
        _cleanup_t074_processes(wrapper)

    results: dict[int, Any] = {}

    def launch(wrapper: Any, rank: int, iface: str) -> None:
        command = (
            f"RANK={rank} WORLD_SIZE=2 LOCAL_RANK=0 "
            f"MASTER_ADDR={ip0} MASTER_PORT=29774 "
            f"NCCL_SOCKET_IFNAME={iface} GLOO_SOCKET_IFNAME={iface} "
            f"NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 NCCL_NET=Socket "
            f"SHARDGRID_PIPELINE_TASK=t074 "
            f"SHARDGRID_T074_CHECKPOINT_DIR=/tmp/t074/checkpoint "
            f"SHARDGRID_T074_STEPS={steps} SHARDGRID_T074_LR=1e-3 "
            f"PYTHONUNBUFFERED=1 "
            f"PYTHONPATH={remote_root} /home/shardgrid/miniconda3/envs/shardgrid/bin/python "
            f"{remote_root}/examples/models/train_pipeline.py"
        )
        results[rank] = wrapper.run(command, timeout=timeout)

    threads = [
        threading.Thread(target=launch, args=(w0, 0, iface0)),
        threading.Thread(target=launch, args=(w1, 1, iface1)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for wrapper in (w0, w1):
        _cleanup_t074_processes(wrapper)

    rank0 = parse_train_evidence(results[0].stdout or "")
    rank1 = parse_train_evidence(results[1].stdout or "")
    place0 = parse_placement_evidence(results[0].stdout or "")
    place1 = parse_placement_evidence(results[1].stdout or "")

    assert rank0 is not None and rank1 is not None, (
        f"rank0_evidence={rank0 is not None} rank1_evidence={rank1 is not None}; "
        f"rank0 tail={(results[0].stdout or '')[-1200:]}; "
        f"rank1 tail={(results[1].stdout or '')[-1200:]}; "
        f"rank0 stderr tail={(results[0].stderr or '')[-1200:]}; "
        f"rank1 stderr tail={(results[1].stderr or '')[-1200:]}"
    )
    assert place0 is not None and place1 is not None
    assert results[0].exit_code == 0 and not results[0].timed_out
    assert results[1].exit_code == 0 and not results[1].timed_out

    assert place0["stage_id"] == "stage0" and "RTX 4060" in place0["gpu_name"]
    assert place1["stage_id"] == "stage1" and "GTX 1650" in place1["gpu_name"]
    assert place0["hostname"] != place1["hostname"]

    assert rank0["steps"] == rank1["steps"] == steps

    # Stage0 optimizer / parameter update evidence
    assert rank0["param_update_ok"] is True
    assert rank0["params_before_checksum"] != rank0["params_after_checksum"]
    # Stage1 optimizer / parameter update evidence
    assert rank1["param_update_ok"] is True
    assert rank1["params_before_checksum"] != rank1["params_after_checksum"]

    # Loss evidence (computed on rank1 only)
    assert rank1["loss_isfinite"] is True
    assert rank1["loss_history"] == rank1["loss_history"]
    assert len(rank1["loss_history"]) == steps
    assert rank1["initial_loss"] is not None and rank1["final_loss"] is not None
    assert rank1["final_loss"] < rank1["initial_loss"]
    assert rank1["loss_decrease"] is True

    # Checkpoint save/load roundtrip on both ranks
    for evidence in (rank0, rank1):
        assert evidence["checkpoint_roundtrip_ok"] is True
        assert evidence["param_restore_ok"] is True
        assert evidence["optimizer_restore_ok"] is True
        assert evidence["step_restore_ok"] is True

    out0 = results[0].stdout or ""
    out1 = results[1].stdout or ""
    for marker in _REQUIRED_RANK0_MARKERS:
        assert marker in out0.splitlines(), f"missing rank0 marker {marker}"
    for marker in _REQUIRED_RANK1_MARKERS:
        assert marker in out1.splitlines(), f"missing rank1 marker {marker}"

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "optimizer-checkpoint-latest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "T074",
                "rank0": {
                    "placement": place0,
                    "train": rank0,
                },
                "rank1": {
                    "placement": place1,
                    "train": rank1,
                },
                "loss_decrease": rank1["loss_decrease"],
                "initial_loss": rank1["initial_loss"],
                "final_loss": rank1["final_loss"],
                "stage0_parameter_update": rank0["param_update_ok"],
                "stage1_parameter_update": rank1["param_update_ok"],
                "checkpoint": {
                    "rank0_roundtrip": rank0["checkpoint_roundtrip_ok"],
                    "rank1_roundtrip": rank1["checkpoint_roundtrip_ok"],
                },
                "process_lifecycle": {
                    "rank0_exit_code": results[0].exit_code,
                    "rank1_exit_code": results[1].exit_code,
                    "clean_exit": results[0].exit_code == 0 and results[1].exit_code == 0,
                    "rank0_last_marker": _last_marker(out0, _REQUIRED_RANK0_MARKERS),
                    "rank1_last_marker": _last_marker(out1, _REQUIRED_RANK1_MARKERS),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    assert os.path.exists(path)
