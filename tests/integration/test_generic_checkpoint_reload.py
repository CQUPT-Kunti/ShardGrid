from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from shardgrid.planner.generic_graph import CanonicalGraphIR, capture_generic_graph
from shardgrid.runtime.checkpoint import consolidate_worker_state_shards, save_worker_state_shard
from shardgrid.runtime.dag import RuntimePlan, WorkerOwnershipPlan, WorkerOwnershipSpec

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ordinary_training_scripts"


def test_final_model_state_strict_loads_into_original_plain_pytorch_model(
    tmp_path: Path,
) -> None:
    fixture = _fixture_module("buffered_state_train")
    model = fixture.BufferedStateModel()
    fixture.train_one_step(model)
    features, _labels = fixture.build_batch()
    model.eval()
    graph = capture_generic_graph(model, sample_args=(features,))
    runtime_plan = _runtime_plan(graph)
    shard_paths = _save_worker_shards(tmp_path, graph, runtime_plan, model.state_dict())

    payload = consolidate_worker_state_shards(
        shard_paths,
        tmp_path / "model-state.pt",
        expected_state_keys=tuple(model.state_dict()),
        expected_graph_fingerprint=graph.graph_fingerprint,
        expected_plan_id="plan-reload",
        expected_training_step=1,
        runtime_plan=runtime_plan,
    )
    state = torch.load(tmp_path / "model-state.pt", map_location="cpu", weights_only=False)
    reloaded = fixture.BufferedStateModel()
    load_result = reloaded.load_state_dict(state, strict=True)

    assert set(state) == set(model.state_dict())
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert all(tuple(state[key].shape) == tuple(model.state_dict()[key].shape) for key in state)
    assert all(state[key].dtype == model.state_dict()[key].dtype for key in state)
    assert {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"} <= set(state)
    assert "optimizer_state_dict" not in state
    assert "scheduler_state_dict" not in state
    assert payload["training_evidence"]["any_parameter_changed"] is True


def _fixture_module(name: str) -> ModuleType:
    path = FIXTURE_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load fixture module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_plan(graph: CanonicalGraphIR) -> RuntimePlan:
    parameter_ids = tuple(use.parameter_id for use in graph.parameter_uses)
    buffer_ids = tuple(
        buffer_id for node in graph.nodes for buffer_id in node.buffer_ids
    )
    return RuntimePlan(
        graph_fingerprint=graph.graph_fingerprint,
        ownership=WorkerOwnershipPlan(
            (
                WorkerOwnershipSpec(
                    worker_id="worker0",
                    gpu_index=0,
                    gpu_id="gpu0",
                    owned_partitions=("stage0",),
                    local_parameter_ids=parameter_ids[:1],
                    local_buffer_ids=(),
                ),
                WorkerOwnershipSpec(
                    worker_id="worker1",
                    gpu_index=0,
                    gpu_id="gpu1",
                    owned_partitions=("stage1",),
                    local_parameter_ids=parameter_ids[1:],
                    local_buffer_ids=buffer_ids,
                ),
            )
        ),
        edges=(),
    )


def _save_worker_shards(
    tmp_path: Path,
    graph: CanonicalGraphIR,
    runtime_plan: RuntimePlan,
    state_dict: dict[str, Any],
) -> list[Path]:
    paths = []
    for rank, worker in enumerate(runtime_plan.ownership.workers):
        result = save_worker_state_shard(
            tmp_path / f"{worker.worker_id}.pt",
            graph=graph,
            runtime_plan=runtime_plan,
            worker_id=worker.worker_id,
            gpu_index=worker.gpu_index,
            state_dict=state_dict,
            job_id="job-reload",
            plan_id="plan-reload",
            training_step=1,
            rank=rank,
            metadata={
                "checked_parameter_count": len(worker.local_parameter_ids),
                "parameter_changed": bool(worker.local_parameter_ids),
            },
        )
        assert result.parameter_count == len(worker.local_parameter_ids)
        paths.append(Path(result.path))
    return paths
