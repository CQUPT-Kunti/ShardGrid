"""T081 — backend graph artifact must be metadata-bounded.

These regression tests prove the backend graph artifact size and content are
decided by graph/code/metadata, not by the real model Parameter/Buffer payload.

T081 is regression-first: the current implementation still serializes the full
FX GraphModule (which carries real parameter storage), so these tests are
expected-red until T082 splits graph metadata from state payload artifacts.
"""

from __future__ import annotations

import io
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
    """Reproduce the control-plane backend-graph.pt serialization."""
    capture = FXGraphCaptureAdapter().capture(
        model.eval(), sample_args=(torch.randn(1, model.layers[0].in_features),)
    )
    backend_graph = capture.backend_graph
    for node in backend_graph.graph.nodes:
        node.type = None
    backend_graph = GraphModule(backend_graph, backend_graph.graph)
    buffer = io.BytesIO()
    torch.save(backend_graph, buffer)
    return buffer.getvalue()


def _state_dict_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(
        int(tensor.numel()) * max(tensor.element_size(), 1)
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T081 expected-red: backend-graph.pt still embeds full real parameter "
        "storage; T082 splits graph metadata from state payload artifacts."
    ),
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T081 expected-red: backend-graph.pt serialized payload still contains "
        "full real parameter storage; T082 removes payload from graph artifacts."
    ),
)
def test_serialized_backend_payload_contains_no_full_real_state() -> None:
    model = WideChain(width=128, depth=2)
    artifact = _serialize_backend_artifact(model)

    expected_state_bytes = _state_dict_bytes(dict(model.state_dict()))
    assert expected_state_bytes > 0
    assert len(artifact) < expected_state_bytes, (
        f"serialized backend artifact ({len(artifact)} bytes) is not smaller than "
        f"the full real model state ({expected_state_bytes} bytes); full payload embedded"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T081 expected-red: reloaded backend graph still materializes full real "
        "state instead of metadata-only graph; T082 splits the artifacts."
    ),
)
def test_reloaded_backend_graph_state_is_not_full_model_state(tmp_path: Path) -> None:
    model = WideChain(width=256, depth=2)
    artifact_path = tmp_path / "backend-graph.pt"
    artifact_path.write_bytes(_serialize_backend_artifact(model))

    reloaded: GraphModule = torch.load(
        artifact_path, map_location="cpu", weights_only=False
    )
    reloaded_state_bytes = _state_dict_bytes(dict(reloaded.state_dict()))
    full_state_bytes = _state_dict_bytes(dict(model.state_dict()))
    assert reloaded_state_bytes < full_state_bytes, (
        f"reloaded backend graph carries {reloaded_state_bytes} bytes of real state, "
        f"equal to the full model state ({full_state_bytes} bytes)"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T081 expected-red: declared state growth still inflates backend artifact "
        "proportionally; T082 persists state separately from graph metadata."
    ),
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