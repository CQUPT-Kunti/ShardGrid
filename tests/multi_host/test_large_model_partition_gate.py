"""Large automatic model partition gate on real multi-host GPUs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.multi_host.large_model_gate import (
    estimate_large_model,
    execution_plan_payload,
    find_feasible_plan,
    jobs_root,
    large_b_parameters,
    live_worker_inventory,
    live_worker_memory,
    monitor_payloads,
    pytorch_pipeline_cluster_config,
    run_train,
    snapshot_metadata_payload,
    three_worker_model_candidates,
    write_training_config,
)


def test_estimate_large_model_uses_meta_materialization(monkeypatch) -> None:
    seen_devices: set[str] = set()
    from tests.multi_host import large_model_gate as gate

    original = gate.build_large_residual_transformer

    def recording_builder(*args, **kwargs):
        model = original(*args, **kwargs)
        seen_devices.update(str(parameter.device) for parameter in model.parameters())
        return model

    monkeypatch.setattr(gate, "build_large_residual_transformer", recording_builder)

    estimate = estimate_large_model(
        {
            "vocab_size": 128,
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_size": 64,
            "sequence_length": 8,
            "batch_size": 2,
            "memory_bank_rows": 16,
            "memory_bank_touch_rows": 2,
        }
    )

    assert seen_devices == {"meta"}
    assert int(estimate["parameters"]) > 0
    assert int(estimate["estimated_training_peak"]) > 0


@pytest.mark.hardware
@pytest.mark.multi_host
def test_large_b_single_gpu_infeasible_two_worker_training_passes(tmp_path: Path) -> None:
    if os.environ.get("SHARDGRID_RUN_LARGE_MODEL_HW") != "1":
        pytest.skip("set SHARDGRID_RUN_LARGE_MODEL_HW=1 to run the large model hardware gate")

    cluster_path = pytorch_pipeline_cluster_config(tmp_path)
    memory = live_worker_memory(cluster_path)
    parameters = large_b_parameters(memory)
    estimate = estimate_large_model(parameters)
    max_worker_memory = max(memory.values())
    assert int(estimate["estimated_training_peak"]) > max_worker_memory

    training_path = write_training_config(tmp_path, parameters, name="large-b")
    result = run_train(cluster_path, training_path, timeout=1200)
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)

    assert payload["state"] == "completed"
    assert payload["failure"] is None
    assert payload["plan_mode"] == "automatic"
    assert payload["engine"] == "pytorch_pipeline"
    assert payload["planning"]["partition_source"] == "automatic"
    assert payload["planning"]["selected_worker_count"] == "2"
    assert payload["planning"]["attempted_worker_counts"] == [2]
    assert payload["world_size"] == 2
    assert len(payload["assignments"]) == 2
    assert int(payload["planning"]["total_cross_worker_communication_bytes"]) > 0

    snapshot = Path(payload["snapshot_path"])
    metadata = json.loads((snapshot / "diagnostics" / "snapshot-metadata.json").read_text())
    manifest = json.loads((snapshot / "checkpoint" / "manifest.json").read_text())
    assert metadata["execution_plan_audit"]["planning"] == payload["planning"]
    assert manifest["partition_source"] == "automatic"
    assert manifest["selected_worker_count"] == "2"
    assert manifest["world_size"] == 2
    assert len(manifest["shards"]) == 2
    for shard in manifest["shards"]:
        assert shard["size_bytes"] > 0
    for monitor_path in (snapshot / "diagnostics").glob("monitor-*.json"):
        monitor = json.loads(monitor_path.read_text())
        placement = monitor["placement"]
        train = monitor["train"]
        assert train["checkpoint_mode"] == "metadata_only"
        assert train["model_type"] == "large_residual_transformer"
        assert train["partition_source"] == "automatic"
        assert train["selected_candidate_id"] == payload["planning"]["selected_candidate_id"]
        assert train["peak_gpu_memory_bytes"] > 0
        assert train["step_times"]
        assert train["loss_isfinite"] is True
        assert train["param_update_ok"] is True
        assert train["owned_module_paths"]
        assert placement["owned_module_paths"] == train["owned_module_paths"]
        assert train["materialized_parameter_bytes"] > 0
        assert placement["materialized_parameter_bytes"] == train["materialized_parameter_bytes"]
        assert train["full_model_materialized"] is False
        assert placement["full_model_materialized"] is False
        assert train["initial_weights_received_from_control"] is False
        assert placement["initial_weights_received_from_control"] is False
    assert len(list((snapshot / "diagnostics").glob("monitor-*.json"))) == 2

    dry_run = run_train(cluster_path, training_path, timeout=300, dry_run=True)
    assert dry_run.returncode == 0, dry_run.stderr or dry_run.stdout
    dry_payload = json.loads(dry_run.stdout)
    assert dry_payload["snapshot_path"]
    dry_snapshot = jobs_root(cluster_path) / str(dry_payload["job_id"])
    assert not list((dry_snapshot / "diagnostics").glob("monitor-*.json"))
    assert not list((dry_snapshot / "logs").glob("**/combined.log"))


@pytest.mark.hardware
@pytest.mark.multi_host
def test_large_residual_transformer_three_worker_training_passes(tmp_path: Path) -> None:
    if os.environ.get("SHARDGRID_RUN_THREE_WORKER_HW") != "1":
        pytest.skip("set SHARDGRID_RUN_THREE_WORKER_HW=1 to run the three-worker hardware gate")

    cluster_path = pytorch_pipeline_cluster_config(tmp_path)
    inventory = live_worker_inventory(cluster_path)
    memory = live_worker_memory(cluster_path)
    extra_env = {
        "SHARDGRID_AUTOMATIC_MIN_WORKERS": "3",
        "SHARDGRID_AUTOMATIC_STEPS": os.environ.get("SHARDGRID_THREE_WORKER_STEPS", "8"),
    }
    selected_parameters, selected_dry_run, scan_results = find_feasible_plan(
        tmp_path,
        config_path=cluster_path,
        name_prefix="three-worker-large",
        candidates=three_worker_model_candidates(memory),
        expected_worker_count="3",
        extra_env=extra_env,
    )

    assert selected_dry_run["planning"]["attempted_worker_counts"] == [3], json.dumps(
        {"inventory": inventory, "scan_results": scan_results},
        indent=2,
        sort_keys=True,
    )
    root = jobs_root(cluster_path)
    metadata = snapshot_metadata_payload(root, str(selected_dry_run["job_id"]))
    planning_evidence = metadata["execution_plan_audit"]["launch_metadata"]["planning_evidence"]
    assert planning_evidence["planner_worker_count_bounds"] == {
        "min_worker_count": 3,
        "max_worker_count": 3,
    } or planning_evidence["planner_worker_count_bounds"]["min_worker_count"] == 3
    assert planning_evidence["planner_worker_resources"]
    planner_workers = {
        str(item["worker_id"]): int(item["gpu_free_memory_mb"]) * 1024 * 1024
        for item in planning_evidence["planner_worker_resources"]
        if item.get("gpu_free_memory_mb") not in {None, 0}
    }
    execution = execution_plan_payload(root, str(selected_dry_run["job_id"]))
    assert execution["world_size"] == 3
    assert len(execution["workers"]) == 3
    assert len({item["worker_id"] for item in execution["workers"]}) == 3
    for assignment in execution["workers"]:
        assert int(assignment["estimated_peak_training_memory"]) < planner_workers[str(assignment["worker_id"])]

    smallest_worker = min(
        (item for item in inventory if item["gpu_free_memory_bytes"] is not None),
        key=lambda item: int(item["gpu_free_memory_bytes"]),
    )
    smallest_assignment = next(
        item for item in execution["workers"] if item["worker_id"] == smallest_worker["worker_id"]
    )
    assert int(smallest_assignment["estimated_peak_training_memory"]) < int(
        smallest_worker["gpu_free_memory_bytes"]
    )

    training_path = write_training_config(tmp_path, selected_parameters, name="three-worker-large")
    result = run_train(cluster_path, training_path, timeout=1800, extra_env=extra_env)
    assert result.returncode == 0, f"RUNTIME_MEMORY_MISMATCH\n{result.stderr or result.stdout}"
    payload = json.loads(result.stdout)

    assert payload["state"] == "completed"
    assert payload["failure"] is None
    assert payload["plan_mode"] == "automatic"
    assert payload["planning"]["selected_worker_count"] == "3"
    assert payload["planning"]["attempted_worker_counts"] == [3]
    assert payload["world_size"] == 3
    assert len(payload["assignments"]) == 3

    snapshot = Path(payload["snapshot_path"])
    monitors = monitor_payloads(snapshot.parent, str(payload["job_id"]))
    assert len(monitors) == 3
    assert len({monitor["placement"]["worker_id"] for monitor in monitors}) == 3
    for monitor in monitors:
        placement = monitor["placement"]
        train = monitor["train"]
        assert train["checkpoint_mode"] == "metadata_only"
        assert train["model_type"] == "large_residual_transformer"
        assert train["partition_source"] == "automatic"
        assert train["selected_candidate_id"] == payload["planning"]["selected_candidate_id"]
        assert train["peak_gpu_memory_bytes"] > 0
        assert train["cuda_training_peak_bytes"] == train["peak_gpu_memory_bytes"]
        assert train["steps"] >= 3
        assert train["step_times"]
        assert train["loss_isfinite"] is True
        assert train["distributed_initialized"] is True
        assert train["stage_materialized"] is True
        assert train["optimizer_step_completed"] is True
        assert train["param_update_ok"] is True
        assert train["activation_transfer_ok"] is True
        assert train["gradient_transfer_ok"] is True
        assert train["owned_module_paths"]
        assert placement["owned_module_paths"] == train["owned_module_paths"]
        assert int(train["pid"]) > 0
        assert int(placement["pid"]) > 0
        assert train["materialized_parameter_bytes"] > 0
        assert placement["materialized_parameter_bytes"] == train["materialized_parameter_bytes"]
        assert train["full_model_materialized"] is False
        assert placement["full_model_materialized"] is False
        assert train["initial_weights_received_from_control"] is False
        assert placement["initial_weights_received_from_control"] is False
        assert train["process_rss_before_materialization_bytes"] is not None
        assert train["process_rss_after_materialization_bytes"] is not None
        assert train["process_rss_before_materialization"] == train["process_rss_before_materialization_bytes"]
        assert train["process_rss_after_materialization"] == train["process_rss_after_materialization_bytes"]
        assert train["cuda_before_stage_move_bytes"] is not None
        assert train["cuda_after_stage_move_bytes"] is not None
        assert train["cuda_allocated_before_stage_to_device_bytes"] == train["cuda_before_stage_move_bytes"]
        assert train["cuda_allocated_after_stage_to_device_bytes"] == train["cuda_after_stage_move_bytes"]
