from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from shardgrid.control.job_manager import JobManager
from shardgrid.planner.generic_graph import CanonicalGraphIR, capture_generic_graph
from shardgrid.runtime.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointContractError,
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
    *,
    rank_by_worker: dict[str, int] | None = None,
) -> list[Path]:
    paths = []
    rank_by_worker = rank_by_worker or {}
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
            rank=rank_by_worker.get(worker.worker_id),
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

    shard_paths = _save_shards(
        tmp_path,
        graph,
        runtime_plan,
        case.module.state_dict(),
        rank_by_worker={"worker0": 0, "worker1": 1},
    )
    consolidated = consolidate_worker_state_shards(
        shard_paths,
        tmp_path / "model-state.pt",
        expected_state_keys=tuple(case.module.state_dict()),
        expected_graph_fingerprint=graph.graph_fingerprint,
        expected_plan_id="plan-test",
        expected_training_step=7,
        runtime_plan=runtime_plan,
    )

    assert set(consolidated["state_dict"]) == set(case.module.state_dict())
    assert consolidated["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert consolidated["graph_fingerprint"] == graph.graph_fingerprint
    assert consolidated["plan_id"] == "plan-test"
    assert consolidated["training_step"] == 7
    assert [shard["rank"] for shard in consolidated["shards"]] == [0, 1]
    assert [shard["gpu_id"] for shard in consolidated["shards"]] == ["gpu0", "gpu1"]
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        assert shard["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert shard["job_id"] == "job-test"
        assert shard["graph_fingerprint"] == graph.graph_fingerprint
        assert shard["plan_id"] == "plan-test"
        assert shard["training_step"] == 7
        assert shard["worker_id"] in {"worker0", "worker1"}
        assert shard["gpu_index"] == 0
        assert shard["gpu_id"] in {"gpu0", "gpu1"}
        assert tuple(shard["owned_partition_ids"]) in {("stage0",), ("stage1",)}
        assert tuple(shard["ownership"]["owned_partitions"]) == tuple(
            shard["owned_partition_ids"]
        )
        assert {
            entry["canonical_id"] for entry in shard["parameters"]
        } == set(shard["ownership"]["local_parameter_ids"])
        assert shard["ownership"]["local_buffer_ids"] == ()
        assert shard["ownership"]["read_only_state_ids"] == ()
        assert all(
            entry["state_dict_key"] in case.module.state_dict()
            for entry in shard["parameters"]
        )
        for entry in shard["parameters"]:
            tensor = case.module.state_dict()[entry["state_dict_key"]]
            assert entry["canonical_id"].startswith("p")
            assert entry["shape"] == tuple(tensor.shape)
            assert entry["dtype"] == str(tensor.dtype).replace("torch.", "")


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
    assert {
        entry["canonical_id"] for entry in shard["buffers"]
    } == set(shard["ownership"]["local_buffer_ids"])
    for entry in shard["buffers"]:
        tensor = model.state_dict()[entry["state_dict_key"]]
        assert entry["canonical_id"].startswith("b")
        assert entry["shape"] == tuple(tensor.shape)
        assert entry["dtype"] == str(tensor.dtype).replace("torch.", "")


def test_checkpoint_merge_rejects_duplicate_canonical_state_ids(tmp_path) -> None:
    case = _ordinary_case("sequential")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=("p0000",),
        worker1_parameters=("p0000",),
    )

    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())

    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*duplicate checkpoint state id 'p0000'",
    ):
        consolidate_worker_state_shards(shard_paths, tmp_path / "model-state.pt")
    assert not (tmp_path / "model-state.pt").exists()


def test_checkpoint_merge_rejects_duplicate_original_state_dict_key(tmp_path) -> None:
    case = _ordinary_case("sequential")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    runtime_plan = _runtime_plan(graph, worker0_parameters=("p0000", "p0001"))
    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())
    shard = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    shard["parameters"][1]["state_dict_key"] = shard["parameters"][0]["state_dict_key"]
    torch.save(shard, shard_paths[0])

    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*duplicate checkpoint state key",
    ):
        consolidate_worker_state_shards(shard_paths, tmp_path / "model-state.pt")
    assert not (tmp_path / "model-state.pt").exists()


def test_checkpoint_merge_rejects_missing_expected_state_keys(tmp_path) -> None:
    model = BufferKeyModel().eval()
    graph = capture_generic_graph(model, sample_args=(torch.randn(3, 4),))
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=_parameter_ids(graph),
        worker0_buffers=_buffer_ids(graph)[:-1],
    )

    shard_paths = _save_shards(tmp_path, graph, runtime_plan, model.state_dict())

    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*bn.num_batches_tracked",
    ):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "model-state.pt",
            expected_state_keys=tuple(model.state_dict()),
        )
    assert not (tmp_path / "model-state.pt").exists()


def test_checkpoint_merge_rejects_shape_and_dtype_mismatches(tmp_path) -> None:
    case = _ordinary_case("sequential")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    runtime_plan = _runtime_plan(graph, worker0_parameters=("p0000", "p0001"))
    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())
    shape_bad = tmp_path / "shape-bad.pt"
    dtype_bad = tmp_path / "dtype-bad.pt"
    shard = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    shape_shard = {**shard, "parameters": list(shard["parameters"])}
    shape_shard["parameters"][0] = {**shape_shard["parameters"][0], "shape": (999,)}
    dtype_shard = {**shard, "parameters": list(shard["parameters"])}
    dtype_shard["parameters"][0] = {**dtype_shard["parameters"][0], "dtype": "float16"}
    torch.save(shape_shard, shape_bad)
    torch.save(dtype_shard, dtype_bad)

    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*shape mismatch",
    ):
        consolidate_worker_state_shards([shape_bad], tmp_path / "shape-state.pt")
    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*dtype mismatch",
    ):
        consolidate_worker_state_shards([dtype_bad], tmp_path / "dtype-state.pt")
    assert not (tmp_path / "shape-state.pt").exists()
    assert not (tmp_path / "dtype-state.pt").exists()


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
    loaded = torch.load(tmp_path / "model-state.pt", map_location="cpu", weights_only=False)
    assert set(loaded) == set(source.module.state_dict())
    file_load_result = target.module.load_state_dict(loaded, strict=True)
    assert file_load_result.missing_keys == []
    assert file_load_result.unexpected_keys == []


def test_checkpoint_merge_rejects_stale_metadata_and_ownership_mismatch(tmp_path) -> None:
    case = _ordinary_case("sequential")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    parameter_ids = _parameter_ids(graph)
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=parameter_ids[:2],
        worker1_parameters=parameter_ids[2:],
    )
    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())

    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*graph_fingerprint mismatch",
    ):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "stale-graph.pt",
            expected_graph_fingerprint="old-graph",
        )
    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*plan_id mismatch",
    ):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "stale-plan.pt",
            expected_plan_id="old-plan",
        )
    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*training_step mismatch",
    ):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "stale-step.pt",
            expected_training_step=999,
        )

    tampered = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    tampered["ownership"] = {**tampered["ownership"], "local_parameter_ids": ()}
    torch.save(tampered, shard_paths[0])
    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*ownership mismatch",
    ):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "ownership-bad.pt",
            runtime_plan=runtime_plan,
        )
    shard_paths = _save_shards(tmp_path / "fresh", graph, runtime_plan, case.module.state_dict())
    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*missing worker shards",
    ):
        consolidate_worker_state_shards(
            shard_paths[:1],
            tmp_path / "missing-worker.pt",
            runtime_plan=runtime_plan,
        )
    assert not (tmp_path / "stale-graph.pt").exists()
    assert not (tmp_path / "stale-plan.pt").exists()
    assert not (tmp_path / "stale-step.pt").exists()
    assert not (tmp_path / "ownership-bad.pt").exists()
    assert not (tmp_path / "missing-worker.pt").exists()


def test_tied_parameter_checkpoint_currently_keeps_only_canonical_state_key(
    tmp_path,
) -> None:
    case = _ordinary_case("shared_tied_parameter")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    runtime_plan = _runtime_plan(graph, worker0_parameters=_parameter_ids(graph))

    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())

    assert set(case.module.state_dict()) == {"embedding.weight", "decoder.weight"}
    assert len(_parameter_ids(graph)) == 1
    with pytest.raises(
        CheckpointContractError,
        match="CHECKPOINT_CONTRACT_FAILURE.*decoder.weight",
    ):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "model-state.pt",
            expected_state_keys=tuple(case.module.state_dict()),
        )
    assert not (tmp_path / "model-state.pt").exists()


def test_job_manager_consolidated_model_finalization_is_model_specific() -> None:
    consolidated_source = inspect.getsource(JobManager._write_consolidated_model)
    generic_source = inspect.getsource(JobManager._write_generic_model_state)

    assert "training_config.model.type" not in consolidated_source
    assert "build_zoo_model" not in generic_source
    assert "make_zoo_sample" not in generic_source
    assert "load_state_dict" not in generic_source


def test_job_manager_generic_checkpoint_finalization_writes_standard_state_dict(
    tmp_path,
) -> None:
    case = _ordinary_case("sequential")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    parameter_ids = _parameter_ids(graph)
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=parameter_ids[:2],
        worker1_parameters=parameter_ids[2:],
    )
    shard_paths = _save_shards(tmp_path, graph, runtime_plan, case.module.state_dict())
    manager = object.__new__(JobManager)
    shards = [
        {
            "local_path": str(path),
            "checkpoint_metadata": manager._generic_checkpoint_metadata_from_shard(path),
        }
        for path in shard_paths
    ]
    snapshot = SimpleNamespace(checkpoint_path=str(tmp_path / "checkpoint"))
    current = SimpleNamespace(job_id="job-test", final_metrics={"loss": 1.0})

    ref = manager._write_consolidated_model(
        snapshot=snapshot,
        training_config=SimpleNamespace(),
        current=current,
        shards=shards,
        manifest_ref="checkpoint/manifest.json",
        device="cpu",
    )

    model_state = torch.load(
        tmp_path / "checkpoint" / "model-state.pt",
        map_location="cpu",
        weights_only=False,
    )
    target = _ordinary_case("sequential")
    load_result = target.module.load_state_dict(model_state, strict=True)
    assert ref == "checkpoint/model-state.pt"
    assert set(model_state) == set(case.module.state_dict())
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
