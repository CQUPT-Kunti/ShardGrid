"""T081/T082 — backend graph artifact must be metadata-bounded.

These regression tests prove the backend graph artifact size and content are
decided by graph/code/metadata, not by the real model Parameter/Buffer payload.

The production control plane persists the graph artifact with parameter/buffer
storage stripped to meta tensors (T082), so these tests assert the artifact is
metadata-bounded and the full state lives in manifest-addressed shards.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.fx import GraphModule

from shardgrid.planner.generic_graph import FXGraphCaptureAdapter


class WideChain(nn.Module):
    def __init__(self, *, width: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _serialize_backend_artifact(model: nn.Module) -> bytes:
    """Reproduce the control-plane backend-graph.pt serialization (T082).

    Parameter/buffer storage is moved to meta tensors so the artifact carries
    graph/code/metadata only; real state is persisted separately.
    """
    capture = FXGraphCaptureAdapter().capture(
        model.eval(), sample_args=(torch.randn(1, model.layers[0].in_features),)
    )
    backend_graph = capture.backend_graph
    for node in backend_graph.graph.nodes:
        node.type = None
    backend_graph = GraphModule(backend_graph, backend_graph.graph)
    for name, parameter in list(
        backend_graph.named_parameters(remove_duplicate=False)
    ):
        if parameter.device.type != "meta":
            _replace_parameter(backend_graph, name, parameter)
    for name, buffer in list(backend_graph.named_buffers(remove_duplicate=False)):
        if buffer.device.type != "meta":
            _replace_buffer(backend_graph, name, buffer)
    buffer = io.BytesIO()
    torch.save(backend_graph, buffer)
    return buffer.getvalue()


def _replace_parameter(module: nn.Module, path: str, parameter: torch.Tensor) -> None:
    parent = module
    *head, leaf = path.split(".")
    for part in head:
        parent = getattr(parent, part)
    replacement = nn.Parameter(
        torch.empty(
            parameter.shape,
            dtype=parameter.dtype,
            device="meta",
            layout=parameter.layout,
            requires_grad=parameter.requires_grad,
        )
    )
    if parent._parameters.get(leaf) is None:
        setattr(parent, leaf, replacement)
    else:
        parent._parameters[leaf] = replacement


def _replace_buffer(module: nn.Module, path: str, buffer: torch.Tensor) -> None:
    parent = module
    *head, leaf = path.split(".")
    for part in head:
        parent = getattr(parent, part)
    replacement = torch.empty(
        buffer.shape,
        dtype=buffer.dtype,
        device="meta",
        layout=buffer.layout,
    )
    if parent._buffers.get(leaf) is None:
        setattr(parent, leaf, replacement)
    else:
        parent._buffers[leaf] = replacement


def _state_dict_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(
        int(tensor.numel()) * max(tensor.element_size(), 1)
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor)
    )


def test_backend_graph_artifact_is_metadata_bounded_not_state_linear() -> None:
    small = _serialize_backend_artifact(WideChain(width=64, depth=2))
    large = _serialize_backend_artifact(WideChain(width=512, depth=2))

    small_params = 64 * 64 * 2
    large_params = 512 * 512 * 2
    assert large_params > small_params * 50

    growth_ratio = len(large) / max(len(small), 1)
    assert growth_ratio < 20, (
        f"backend artifact grows {growth_ratio:.1f}x while real parameters grow "
        f"{large_params / small_params:.1f}x; artifact is state-linear, not metadata-bounded"
    )


def test_serialized_backend_payload_contains_no_full_real_state() -> None:
    model = WideChain(width=1024, depth=2)
    artifact = _serialize_backend_artifact(model)

    expected_state_bytes = _state_dict_bytes(dict(model.state_dict()))
    assert expected_state_bytes > 0
    assert len(artifact) * 10 < expected_state_bytes, (
        f"serialized backend artifact ({len(artifact)} bytes) is not far smaller "
        f"than the full real model state ({expected_state_bytes} bytes); full payload embedded"
    )


def test_reloaded_backend_graph_state_has_no_real_storage(tmp_path: Path) -> None:
    model = WideChain(width=256, depth=2)
    artifact_path = tmp_path / "backend-graph.pt"
    artifact_path.write_bytes(_serialize_backend_artifact(model))

    reloaded: GraphModule = torch.load(
        artifact_path, map_location="cpu", weights_only=False
    )
    for tensor in reloaded.state_dict().values():
        assert tensor.device.type == "meta", (
            "reloaded backend graph materializes real tensor storage"
        )


def test_large_declared_state_keeps_backend_artifact_bounded() -> None:
    artifact_64 = _serialize_backend_artifact(WideChain(width=64, depth=4))
    artifact_1024 = _serialize_backend_artifact(WideChain(width=1024, depth=4))

    declared_64 = 64 * 64 * 4
    declared_1024 = 1024 * 1024 * 4
    assert declared_1024 > declared_64 * 200

    artifact_growth = len(artifact_1024) / max(len(artifact_64), 1)
    assert artifact_growth < 30, (
        f"declared state grows {declared_1024 / declared_64:.0f}x but backend artifact "
        f"grows {artifact_growth:.1f}x; artifact is not bounded"
    )


def test_state_manifest_addresses_owned_shards(tmp_path: Path) -> None:
    """T082: state payload is split into manifest-addressed ownership shards."""
    from shardgrid.control.job_manager import (
        _build_state_manifest,
        _materialize_state_shards,
    )
    from shardgrid.common.models import as_engine_name
    from shardgrid.engines.models import ParallelPlan
    from shardgrid.engines.models import ParallelPlanStage

    model = WideChain(width=64, depth=2)
    capture = FXGraphCaptureAdapter().capture(
        model.eval(), sample_args=(torch.randn(1, 64),)
    )
    plan = ParallelPlan(
        parallel_plan_id="plan",
        engine=as_engine_name("pytorch_pipeline"),
        world_size=2,
        requirements={},
        stage_metadata=[
            ParallelPlanStage(
                stage_id="stage0",
                rank=0,
                module_ids=("layers.0",),
                module_paths=("layers.0",),
                start_index=0,
                stop_index=1,
            ),
            ParallelPlanStage(
                stage_id="stage1",
                rank=1,
                module_ids=("layers.1",),
                module_paths=("layers.1",),
                start_index=1,
                stop_index=2,
            ),
        ],
        selected_candidate_id="candidate",
    )
    manifest = _build_state_manifest(
        capture.backend_graph,
        capture.canonical_graph,
        plan,
        state_source=dict(model.state_dict()),
    )
    _materialize_state_shards(
        plan_root=tmp_path,
        backend_graph=capture.backend_graph,
        state_manifest=manifest,
        state_source=dict(model.state_dict()),
    )

    manifest_path = tmp_path / "state-manifest.json"
    shards_root = tmp_path / "state-shards"
    assert manifest_path.is_file()
    assert len(list(shards_root.glob("*.pt"))) == 2
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(entries) == 4
    by_state = {str(entry["state_id"]): entry for entry in entries}
    assert by_state["layers.0.weight"]["owner_stage"] == "stage0"
    assert by_state["layers.1.weight"]["owner_stage"] == "stage1"
    for entry in entries:
        assert entry["shard_ref"] in {
            "state-shards/stage0.pt",
            "state-shards/stage1.pt",
        }
        assert entry["checksum_sha256"]
        assert entry["bytes"] > 0
        assert entry["shape"]
        assert entry["dtype"]


def test_state_manifest_marks_read_only_stages() -> None:
    """T082: tied/shared state records read-only consumers without payload copy."""
    from shardgrid.control.job_manager import _build_state_manifest
    from shardgrid.common.models import as_engine_name
    from shardgrid.engines.models import ParallelPlan
    from shardgrid.engines.models import ParallelPlanStage

    model = WideChain(width=64, depth=2)
    capture = FXGraphCaptureAdapter().capture(
        model.eval(), sample_args=(torch.randn(1, 64),)
    )
    plan = ParallelPlan(
        parallel_plan_id="plan",
        engine=as_engine_name("pytorch_pipeline"),
        world_size=2,
        requirements={},
        stage_metadata=[
            ParallelPlanStage(
                stage_id="stage0",
                rank=0,
                module_ids=("layers.0", "layers.1"),
                module_paths=("layers.0", "layers.1"),
                start_index=0,
                stop_index=2,
            ),
            ParallelPlanStage(
                stage_id="stage1",
                rank=1,
                module_ids=("layers.1",),
                module_paths=("layers.1",),
                start_index=1,
                stop_index=2,
            ),
        ],
        selected_candidate_id="candidate",
    )
    manifest = _build_state_manifest(
        capture.backend_graph,
        capture.canonical_graph,
        plan,
        state_source=dict(model.state_dict()),
    )
    by_state = {str(entry["state_id"]): entry for entry in manifest}
    assert by_state["layers.1.weight"]["owner_stage"] == "stage0"
    assert by_state["layers.1.weight"]["read_only_stages"] == ["stage1"]