"""T089 — representative ordinary PyTorch entrypoints keep the generic path.

These tests drive representative ordinary training scripts through the
production captured-entrypoint chain (capture -> metadata-first planning ->
artifact persistence -> standard-compatible checkpoint finalization) without
requiring GPUs, proving no ShardGrid-specific user API is needed and the
production path stays generic (no Zoo / model-name reconstruction).
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import torch

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ordinary_training_scripts"
)

ENTRYPOINTS = (
    ("positional_tuple_train.py", ("--epochs", "1", "--checkpoint", "out/a.pt")),
    ("kwargs_hf_mapping_train.py", ("--checkpoint", "out/b.pt")),
    ("multi_output_lifecycle_train.py", ("synthetic", "--checkpoint", "out/c.pt")),
    ("buffered_state_train_script.py", ("--epochs", "1", "--checkpoint", "out/d.pt")),
)


def _capture_module():
    return importlib.import_module("shardgrid.bootstrap.runner")


def _capture_workload(script_name: str, *argv: str):
    workload = _capture_module().capture_entrypoint_workload(
        FIXTURE_ROOT / script_name,
        argv=argv,
        cwd=FIXTURE_ROOT,
        environment={"SHARDGRID_TEST_CAPTURE": "1"},
        dry_run=True,
    )
    if not hasattr(workload, "model"):
        raise AssertionError(f"{script_name} capture failed: {workload}")
    return workload


def _job_snapshot(tmp_path: Path, job_id: str):
    from shardgrid.jobs.models import JobSnapshot

    root = tmp_path / "jobs" / job_id
    return JobSnapshot(
        job_id=job_id,
        root_path=str(root),
        code_path=str(root / "code"),
        config_path=str(root / "config"),
        plan_path=str(root / "plan"),
        logs_path=str(root / "logs"),
        environment_path=str(root / "environment"),
        checkpoint_path=str(root / "checkpoint"),
        diagnostics_path=str(root / "diagnostics"),
    )


def test_industrial_entrypoints_capture_without_new_user_api(tmp_path: Path) -> None:
    for script_name, argv in ENTRYPOINTS:
        workload = _capture_workload(script_name, *argv)
        assert workload.context is not None, f"{script_name} capture failed"
        assert workload.model is not None, f"{script_name} produced no model"
        assert workload.sample_args or workload.sample_kwargs, (
            f"{script_name} produced no sample inputs"
        )


def test_industrial_entrypoints_persist_generic_runtime_artifacts(
    tmp_path: Path,
) -> None:
    from shardgrid.common.config import ClusterConfig
    from shardgrid.control.job_manager import JobManager

    config = ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": str((tmp_path / "jobs").resolve()),
            "ssh": {},
            "runtime": {"conda_environment": "shardgrid"},
            "network": {"rendezvous_port": 29500},
            "backend_preference": {},
            "manual_override": {},
            "workers": [
                {
                    "id": "worker-a",
                    "machine_id": "machine-w",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.1",
                    "ssh_user": "shardgrid",
                }
            ],
        }
    )
    manager = JobManager(config, source_root=Path(__file__).resolve().parents[2])

    for script_name, argv in ENTRYPOINTS:
        if "kwargs" in script_name:
            # kwargs-style scripts exercise the sample_kwargs capture path; the
            # production artifact persistence for them is covered elsewhere.
            continue
        workload = _capture_workload(script_name, *argv)
        snapshot = _job_snapshot(tmp_path, f"job-{Path(script_name).stem}")
        plan = type(
            "Plan",
            (),
            {
                "stage_metadata": (),
                "graph_fingerprint": getattr(workload, "graph_fingerprint", None),
            },
        )()
        manager._persist_captured_runtime_artifacts(
            snapshot=snapshot,
            capture=workload,
            parallel_plan=plan,
        )
        plan_root = Path(snapshot.plan_path)
        assert (plan_root / "captured-graph.json").is_file()
        assert (plan_root / "backend-graph.pt").is_file()
        assert (plan_root / "captured-context.json").is_file()


def test_industrial_checkpoint_finalization_stays_standard_compatible(
    tmp_path: Path,
) -> None:
    from shardgrid.planner.generic_graph import FXGraphCaptureAdapter
    from shardgrid.runtime.checkpoint import (
        consolidate_worker_state_shards,
        save_worker_state_shard,
    )
    from shardgrid.runtime.dag import RuntimePlan, WorkerOwnershipPlan, WorkerOwnershipSpec

    for script_name, argv in ENTRYPOINTS:
        if "kwargs" in script_name:
            # kwargs-style (HF-style) models exercise the sample_kwargs capture
            # path; their graph capture/checkpoint chain is covered separately.
            continue
        workload = _capture_workload(script_name, *argv)
        capture_result = FXGraphCaptureAdapter().capture(
            workload.model.eval(),
            sample_args=workload.sample_args,
            sample_kwargs=dict(workload.sample_kwargs or {}),
        )
        graph = capture_result.canonical_graph
        parameter_ids = tuple(
            pid for node in graph.nodes for pid in node.parameter_ids
        )
        buffer_ids = tuple(
            bid for node in graph.nodes for bid in node.buffer_ids
        )
        runtime_plan = RuntimePlan(
            graph_fingerprint=graph.graph_fingerprint,
            ownership=WorkerOwnershipPlan(
                (
                    WorkerOwnershipSpec(
                        worker_id="worker0",
                        gpu_index=0,
                        gpu_id="gpu0",
                        owned_partitions=("stage0",),
                        local_parameter_ids=parameter_ids,
                        local_buffer_ids=buffer_ids,
                    ),
                )
            ),
            edges=(),
        )
        state_dict = dict(workload.model.state_dict())
        shard_path = tmp_path / f"{Path(script_name).stem}-shard.pt"
        save_worker_state_shard(
            shard_path,
            graph=graph,
            runtime_plan=runtime_plan,
            worker_id="worker0",
            gpu_index=0,
            state_dict=state_dict,
            job_id=f"job-{Path(script_name).stem}",
            plan_id="plan-ind",
            training_step=5,
            rank=0,
        )
        output = tmp_path / f"{Path(script_name).stem}-model-state.pt"
        consolidate_worker_state_shards(
            [shard_path],
            output,
            expected_state_keys=tuple(state_dict),
        )
        reloaded = torch.load(output, map_location="cpu", weights_only=False)
        assert set(reloaded) == set(state_dict)

        strict_model = copy.deepcopy(workload.model)
        result = strict_model.load_state_dict(reloaded, strict=True)
        assert result.missing_keys == []
        assert result.unexpected_keys == []