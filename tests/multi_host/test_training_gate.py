"""Formal Gate 3 acceptance on top of the real T074 training path (T075)."""

from __future__ import annotations

import json
import os
from typing import Any

from tests.multi_host.test_optimizer_checkpoint import (
    _REQUIRED_RANK0_MARKERS,
    _REQUIRED_RANK1_MARKERS,
    parse_train_evidence,
    test_live_optimizer_checkpoint_on_two_workers,
)


def _loss_drop_ratio(initial_loss: float, final_loss: float) -> float:
    if initial_loss <= 0:
        raise AssertionError(f"initial_loss must be positive, got {initial_loss!r}")
    return (initial_loss - final_loss) / initial_loss


def test_gate3_loss_drop_ratio_math() -> None:
    assert _loss_drop_ratio(10.0, 9.0) == 0.1


def test_gate3_marker_contract() -> None:
    assert "CHECKPOINT_SAVE_END" in _REQUIRED_RANK0_MARKERS
    assert "CHECKPOINT_LOAD_END" in _REQUIRED_RANK1_MARKERS
    assert "OPTIMIZER_STEP_END" in _REQUIRED_RANK0_MARKERS
    assert "OPTIMIZER_STEP_END" in _REQUIRED_RANK1_MARKERS


def test_gate3_parse_payload_contract() -> None:
    payload = {
        "rank": 1,
        "steps": 20,
        "initial_loss": 10.0,
        "final_loss": 9.0,
        "loss_decrease": True,
        "loss_isfinite": True,
        "param_update_ok": True,
        "checkpoint_roundtrip_ok": True,
    }
    parsed = parse_train_evidence("T074_TRAIN_EVIDENCE " + json.dumps(payload))
    assert parsed is not None
    assert _loss_drop_ratio(parsed["initial_loss"], parsed["final_loss"]) >= 0.05


def test_live_training_gate_on_two_workers() -> None:
    """Formal Gate 3 acceptance using the already-accepted T074 runtime."""
    test_live_optimizer_checkpoint_on_two_workers()

    evidence_path = os.path.join(
        os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or "/var/tmp/shardgrid/engines",
        "optimizer-checkpoint-latest.json",
    )
    with open(evidence_path, encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    rank0 = payload["rank0"]["train"]
    rank1 = payload["rank1"]["train"]
    place0 = payload["rank0"]["placement"]
    place1 = payload["rank1"]["placement"]

    assert place0["stage_id"] == "stage0"
    assert place1["stage_id"] == "stage1"
    assert place0["hostname"] != place1["hostname"]

    assert payload["process_lifecycle"]["clean_exit"] is True
    assert rank0["param_update_ok"] is True
    assert rank1["param_update_ok"] is True
    assert rank0["checkpoint_roundtrip_ok"] is True
    assert rank1["checkpoint_roundtrip_ok"] is True
    assert rank1["loss_isfinite"] is True
    assert rank1["loss_decrease"] is True
    assert len(rank1["loss_history"]) == rank1["steps"] >= 2
    assert rank1["initial_loss"] is not None
    assert rank1["final_loss"] is not None
    assert _loss_drop_ratio(rank1["initial_loss"], rank1["final_loss"]) >= 0.05

    gate_path = os.path.join(
        os.path.dirname(evidence_path),
        "training-gate-latest.json",
    )
    with open(gate_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "T075",
                "gate3_pass": True,
                "steps": rank1["steps"],
                "initial_loss": rank1["initial_loss"],
                "final_loss": rank1["final_loss"],
                "loss_drop_ratio": _loss_drop_ratio(
                    rank1["initial_loss"], rank1["final_loss"]
                ),
                "forward_pass": True,
                "backward_pass": True,
                "gradient_return_pass": True,
                "optimizer_pass": True,
                "checkpoint_pass": True,
                "stage0_parameter_update": rank0["param_update_ok"],
                "stage1_parameter_update": rank1["param_update_ok"],
                "rank0_checkpoint_roundtrip": rank0["checkpoint_roundtrip_ok"],
                "rank1_checkpoint_roundtrip": rank1["checkpoint_roundtrip_ok"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    assert os.path.exists(gate_path)
