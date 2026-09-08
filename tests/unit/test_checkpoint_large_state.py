"""T086 — large-state checkpoint finalization memory-safety regression.

The finalization contract requires that for models whose total logical state
exceeds the control-plane RAM budget, the control plane never holds all worker
shard tensors resident in memory at once.

This test uses bounded synthetic shards with a load-spy that tracks
simultaneously-resident tensor payload bytes through weakrefs, proving the
danger pattern:

    merged = {}
    for shard in shards:
        merged.update(torch.load(shard))
    torch.save(merged, "model-state.pt")

(which holds every real tensor resident until the end) is rejected.  This is
expected-red until T087 implements bounded/streaming finalization.
"""

from __future__ import annotations

import sys
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch

from shardgrid.planner.generic_graph import capture_generic_graph
from shardgrid.runtime.checkpoint import consolidate_worker_state_shards, save_worker_state_shard
from shardgrid.runtime.dag import RuntimePlan, WorkerOwnershipPlan, WorkerOwnershipSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_checkpoint_generic_state import BufferKeyModel  # noqa: E402


class _ResidentPayloadAccountant:
    """Spy tracking peak simultaneously-resident tensor payload bytes.

    ``torch.load`` results are accounted; when a tensor is garbage collected
    the weakref finalizer decrements the live count.  The peak reflects the
    maximum bytes of real state held in RAM at any instant.
    """

    def __init__(self) -> None:
        self._real_load = torch.load
        self.peak_bytes = 0
        self._live_bytes = 0
        self.loaded_paths: list[str] = []

    def __call__(self, path, *args, **kwargs):
        self.loaded_paths.append(str(Path(path).name))
        payload = self._real_load(path, *args, **kwargs)
        for tensor in self._iter_tensors(payload):
            self._account(tensor)
        return payload

    def _account(self, tensor: torch.Tensor) -> None:
        if tensor.device.type == "meta":
            return
        bytes_ = int(tensor.numel()) * max(tensor.element_size(), 1)
        self._live_bytes += bytes_
        self.peak_bytes = max(self.peak_bytes, self._live_bytes)
        weakref.finalize(tensor, self._release, bytes_)

    def _release(self, bytes_: int) -> None:
        self._live_bytes -= bytes_

    @staticmethod
    def _iter_tensors(payload: Any):
        if isinstance(payload, torch.Tensor):
            yield payload
            return
        if isinstance(payload, dict):
            for value in payload.values():
                yield from _ResidentPayloadAccountant._iter_tensors(value)
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                yield from _ResidentPayloadAccountant._iter_tensors(item)


def _runtime_plan(
    graph, *, worker0_parameters, worker1_parameters, worker0_buffers=(), worker1_buffers=()
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


def _parameter_ids(graph) -> tuple[str, ...]:
    ids = []
    for node in graph.nodes:
        ids.extend(node.parameter_ids)
    return tuple(ids)


def _buffer_ids(graph) -> tuple[str, ...]:
    ids = []
    for node in graph.nodes:
        ids.extend(node.buffer_ids)
    return tuple(ids)


def _save_shards(
    tmp_path: Path,
    graph,
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
            job_id="job-large",
            plan_id="plan-large",
            training_step=7,
            rank=int(worker.worker_id[-1]),
        )
        paths.append(path)
    return paths


def _total_state_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(
        int(tensor.numel()) * max(tensor.element_size(), 1)
        for tensor in state_dict.values()
        if isinstance(tensor, torch.Tensor)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T086 expected-red: consolidate_worker_state_shards merges every worker "
        "shard tensor into one in-memory state dict before saving; T087 implements "
        "bounded/streaming finalization."
    ),
)
def test_finalization_keeps_resident_payload_bounded_for_large_logical_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = BufferKeyModel()
    graph = capture_generic_graph(model.eval(), sample_args=(torch.randn(4, 4),))
    parameter_ids = _parameter_ids(graph)
    buffer_ids = _buffer_ids(graph)
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=parameter_ids[:2],
        worker1_parameters=parameter_ids[2:],
        worker0_buffers=buffer_ids,
    )
    state_dict = dict(model.state_dict())
    shard_paths = _save_shards(tmp_path, graph, runtime_plan, state_dict)

    accountant = _ResidentPayloadAccountant()
    monkeypatch.setattr(torch, "load", accountant)

    budget_bytes = 1024
    assert _total_state_bytes(state_dict) > budget_bytes

    consolidate_worker_state_shards(
        shard_paths,
        tmp_path / "model-state.pt",
        expected_state_keys=tuple(state_dict),
        expected_graph_fingerprint=graph.graph_fingerprint,
        expected_plan_id="plan-large",
        expected_training_step=7,
        runtime_plan=runtime_plan,
    )

    assert accountant.peak_bytes <= budget_bytes, (
        f"finalization held {accountant.peak_bytes} bytes resident while the "
        f"control-plane RAM budget is {budget_bytes}; full-state merge detected"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T086 expected-red: validation phase loads every shard tensor before "
        "validating; T087 must validate from manifests with bounded payload reads."
    ),
)
def test_validation_does_not_require_full_state_in_ram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = BufferKeyModel()
    graph = capture_generic_graph(model.eval(), sample_args=(torch.randn(4, 4),))
    parameter_ids = _parameter_ids(graph)
    buffer_ids = _buffer_ids(graph)
    runtime_plan = _runtime_plan(
        graph,
        worker0_parameters=parameter_ids[:2],
        worker1_parameters=parameter_ids[2:],
        worker0_buffers=buffer_ids,
    )
    state_dict = dict(model.state_dict())
    shard_paths = _save_shards(tmp_path, graph, runtime_plan, state_dict)

    accountant = _ResidentPayloadAccountant()
    monkeypatch.setattr(torch, "load", accountant)

    consolidate_worker_state_shards(
        shard_paths,
        tmp_path / "model-state.pt",
        expected_state_keys=tuple(state_dict),
        expected_graph_fingerprint=graph.graph_fingerprint,
        expected_plan_id="plan-large",
        expected_training_step=7,
        runtime_plan=runtime_plan,
    )

    per_shard_max = max(_total_state_bytes(state_dict) / 2, 0) + 16
    assert accountant.peak_bytes <= per_shard_max, (
        f"validation held {accountant.peak_bytes} bytes; at most a single shard "
        f"should be resident at a time (budget {per_shard_max:.0f} bytes)"
    )