"""Real SSH generic DAG hardware gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from shardgrid.common.config import load_cluster_config
from shardgrid.control.job_manager import JobManager


@pytest.mark.hardware
@pytest.mark.multi_host
def test_generic_dag_real_ssh_multihost_gate() -> None:
    repo = Path(__file__).resolve().parents[2]
    config = load_cluster_config(repo / "examples" / "workers.yaml")
    manager = JobManager(config, source_root=repo)

    result = manager.run(
        repo / "examples" / "train-generic-dag.yaml",
        min_selected_physical_hosts=2,
    )
    root = Path(result.snapshot.root_path)
    payload = _audit(root)

    assert result.status.state.value == "completed"
    assert result.execution_plan.world_size >= 2
    assert len({item["worker_id"] for item in payload["execution"]["workers"]}) >= 2
    assert len({item.get("machine_id") or item["host"] for item in payload["assignments"]}) >= 2
    assert int(payload["planning"]["selected_worker_count"]) >= 2
    assert payload["planner_worker_count_bounds"]["constraint_source"] == "hardware_gate"
    assert payload["planner_worker_count_bounds"]["min_selected_physical_hosts"] == 2
    assert payload["generic_runtime_used"] is True
    assert payload["legacy_runtime_used"] is False
    assert payload["full_model_materialized"] is False
    assert payload["remote_activation_edges"] > 0
    assert payload["remote_gradient_edges"] > 0
    assert min(payload["optimizer_steps"]) >= 20
    assert payload["loss_isfinite"] is True
    assert payload["required_shard_count"] == payload["received_shard_count"]
    assert payload["training_evidence"]["any_parameter_changed"] is True
    assert payload["training_evidence"]["all_trainable_workers_parameter_changed"] is True
    assert payload["model_state"]["parameter_changed"] is True
    assert payload["model_state"]["strict_load_missing_keys"] == []
    assert payload["model_state"]["strict_load_unexpected_keys"] == []
    assert payload["model_state"]["validation_forward_passed"] is True


def _audit(root: Path) -> dict[str, object]:
    execution = json.loads((root / "plan" / "execution-plan.json").read_text())
    metadata = json.loads((root / "diagnostics" / "snapshot-metadata.json").read_text())
    manifest = json.loads((root / "checkpoint" / "manifest.json").read_text())
    monitors = [
        json.loads(path.read_text())
        for path in sorted((root / "diagnostics").glob("monitor-*.json"))
    ]
    model_state = torch.load(
        root / "checkpoint" / "model-state.pt",
        map_location="cpu",
        weights_only=False,
    )
    trains = [monitor.get("train", {}) for monitor in monitors]
    launch = metadata["execution_plan_audit"]["launch_metadata"]
    return {
        "execution": execution,
        "assignments": metadata["execution_plan_audit"]["assignments"],
        "planning": metadata["execution_plan_audit"]["planning"],
        "planner_worker_count_bounds": launch["planning_evidence"][
            "planner_worker_count_bounds"
        ],
        "generic_runtime_used": all(train.get("generic_dag_runtime_used") for train in trains),
        "legacy_runtime_used": any(train.get("legacy_stage_runtime_used") for train in trains),
        "full_model_materialized": any(
            train.get("full_model_real_materialized") for train in trains
        ),
        "remote_activation_edges": sum(
            int(train.get("activation_remote_edges") or 0) for train in trains
        ),
        "remote_gradient_edges": sum(
            int(train.get("gradient_remote_edges") or 0) for train in trains
        ),
        "optimizer_steps": [int(train.get("optimizer_steps") or 0) for train in trains],
        "loss_isfinite": all(
            train.get("final_loss") is None or math.isfinite(float(train["final_loss"]))
            for train in trains
        ),
        "required_shard_count": int(manifest["required_shard_count"]),
        "received_shard_count": len(manifest["shards"]),
        "training_evidence": model_state["training_evidence"],
        "model_state": {
            "parameter_changed": model_state["parameter_changed"],
            "strict_load_missing_keys": model_state["strict_load_missing_keys"],
            "strict_load_unexpected_keys": model_state["strict_load_unexpected_keys"],
            "validation_forward_passed": model_state["validation_forward_passed"],
        },
    }
