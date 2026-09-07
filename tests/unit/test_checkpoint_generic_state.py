from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import nn

from shardgrid.control.job_manager import JobManager
from shardgrid.planner.generic_graph import CanonicalGraphIR, capture_generic_graph
from shardgrid.runtime.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    consolidate_worker_state_shards,
    save_worker_state_shard,
)
from shardgrid.runtime.dag import RuntimePlan, WorkerOwnershipPlan, WorkerOwnershipSpec


def _generic_fixture_module() -> ModuleType:
    module_name = "generic_training_models"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[1] / "fixtures" / "generic_training_models.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load fixture module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ordinary_case(name: str) -> Any:
    return _generic_fixture_module().generic_training_case(name)


def _parameter_ids(graph: CanonicalGraphIR) -> tuple[str, ...]:
    return tuple(use.parameter_id for use in graph.parameter_uses)


def _buffer_ids(graph: CanonicalGraphIR) -> tuple[str, ...]:
    return tuple(
        buffer_id for node in graph.nodes for buffer_id in node.buffer_ids
    )


def _runtime_plan(
    graph: CanonicalGraphIR,
    *,
    worker0_parameters: tuple[str, ...] = (),
    worker0_buffers: tuple[str, ...] = (),
    worker1_parameters: tuple[str, ...] = (),
    worker1_buffers: tuple[str, ...] = (),
) -> RuntimePlan:
    return RuntimePlan(
        graph_fingerprint=graph.graph_fingerprint,
        ownership=WorkerOwnershipPlan(
            (
                WorkerOwnershipSpec(
                    worker_id="worker0",
                    gpu_index=0,
                    gpu_id="gpu0",
                    owned_partitions=("stage0",),
                    local_parameter_ids=worker0_parameters,
                    local_buffer_ids=worker0_buffers,
                ),
                WorkerOwnershipSpec(
                    worker_id="worker1",
                    gpu_index=0,
                    gpu_id="gpu1",
                    owned_partitions=("stage1",),
                    local_parameter_ids=worker1_parameters,
                    local_buffer_ids=worker1_buffers,
                ),
            )
        ),
        edges=(),
    )


def _save_shards(
    tmp_path: Path,
    graph: CanonicalGraphIR,
    runtime_plan: RuntimePlan,
    state_dict: dict[str, torch.Tensor],
) -> list[Path]:
    paths = []
    for worker in runtime_plan.ownership.workers:
        path = tmp_path / f"{worker.worker_id}.pt"
        save_worker_state_shard(
            path,
            graph=graph,
            runtime_plan=runtime_plan,
            worker_id=worker.worker_id,
            gpu_index=worker.gpu_index,
            state_dict=state_dict,
            job_id="job-test",
            plan_id="plan-test",
            training_step=7,
        )
        paths.append(path)
    return paths


class BufferKeyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.proj = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.bn(x))


def test_checkpoint_shards_preserve_original_parameter_state_dict_keys(tmp_path) -> None:
    case = _ordinary_case("sequential")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    parameter_ids = _parameter_ids(graph)
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=parameter_ids[:2],
        worker1_parameters=parameter_ids[2:],
    )

    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())
    consolidated = consolidate_worker_state_shards(
        shard_paths,
        tmp_path / "model-state.pt",
        expected_state_keys=tuple(case.module.state_dict()),
    )

    assert set(consolidated["state_dict"]) == set(case.module.state_dict())
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        assert shard["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert all(
            entry["state_dict_key"] in case.module.state_dict()
            for entry in shard["parameters"]
        )


def test_checkpoint_shards_preserve_original_buffer_state_dict_keys(tmp_path) -> None:
    model = BufferKeyModel().eval()
    graph = capture_generic_graph(model, sample_args=(torch.randn(3, 4),))
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=_parameter_ids(graph),
        worker0_buffers=_buffer_ids(graph),
    )

    shard_paths = _save_shards(tmp_path, graph, runtime_plan, model.state_dict())
    consolidated = consolidate_worker_state_shards(
        shard_paths,
        tmp_path / "model-state.pt",
        expected_state_keys=tuple(model.state_dict()),
    )
    shard = torch.load(shard_paths[0], map_location="cpu", weights_only=False)

    assert {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"} <= set(
        consolidated["state_dict"]
    )
    assert [entry["state_dict_key"] for entry in shard["buffers"]] == [
        "bn.running_mean",
        "bn.running_var",
        "bn.num_batches_tracked",
    ]


def test_checkpoint_merge_rejects_duplicate_canonical_state_ids(tmp_path) -> None:
    case = _ordinary_case("sequential")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=("p0000",),
        worker1_parameters=("p0000",),
    )

    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())

    with pytest.raises(ValueError, match="duplicate checkpoint state id 'p0000'"):
        consolidate_worker_state_shards(shard_paths, tmp_path / "model-state.pt")


def test_checkpoint_merge_rejects_missing_expected_state_keys(tmp_path) -> None:
    model = BufferKeyModel().eval()
    graph = capture_generic_graph(model, sample_args=(torch.randn(3, 4),))
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=_parameter_ids(graph),
        worker0_buffers=_buffer_ids(graph)[:-1],
    )

    shard_paths = _save_shards(tmp_path, graph, runtime_plan, model.state_dict())

    with pytest.raises(ValueError, match="bn.num_batches_tracked"):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "model-state.pt",
            expected_state_keys=tuple(model.state_dict()),
        )


def test_consolidated_checkpoint_supports_strict_reload_for_plain_pytorch_model(
    tmp_path,
) -> None:
    source = _ordinary_case("sequential")
    graph = capture_generic_graph(source.module.eval(), sample_args=source.args)
    parameter_ids = _parameter_ids(graph)
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=parameter_ids[:2],
        worker1_parameters=parameter_ids[2:],
    )
    shard_paths = _save_shards(tmp_path, graph, runtime_plan, source.module.state_dict())

    consolidated = consolidate_worker_state_shards(
        shard_paths,
        tmp_path / "model-state.pt",
        expected_state_keys=tuple(source.module.state_dict()),
    )
    target = _ordinary_case("sequential")
    load_result = target.module.load_state_dict(consolidated["state_dict"], strict=True)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_tied_parameter_checkpoint_currently_keeps_only_canonical_state_key(
    tmp_path,
) -> None:
    case = _ordinary_case("shared_tied_parameter")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    runtime_plan = _runtime_plan(graph, worker0_parameters=_parameter_ids(graph))

    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())

    assert set(case.module.state_dict()) == {"embedding.weight", "decoder.weight"}
    with pytest.raises(ValueError, match="decoder.weight"):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "model-state.pt",
            expected_state_keys=tuple(case.module.state_dict()),
        )


def test_job_manager_consolidated_model_finalization_is_model_specific() -> None:
    consolidated_source = inspect.getsource(JobManager._write_consolidated_model)
    generic_source = inspect.getsource(JobManager._write_generic_dag_model_state)

    assert "training_config.model.type" in consolidated_source
    assert '"minimal_sequential"' in consolidated_source
    assert '"generic_dag"' in consolidated_source
    assert "MinimalTransformer" in consolidated_source
    assert "build_zoo_model" in generic_source
    assert "make_zoo_sample" in generic_source
    assert "load_state_dict" in generic_source
