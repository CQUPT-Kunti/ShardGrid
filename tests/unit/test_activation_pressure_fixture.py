"""T088 — corrected GPU-packing stress fixture contract (hardware-independent).

Proves the corrected stress workload produces its memory band through real
activations (batch x sequence x width x depth, retained skip tensors) while
parameters and the backend graph artifact stay bounded, so CPU serialization /
backend artifact / SSH transfer are never the bottleneck.

These tests run without GPUs; the real multi-host packing loop is the opt-in
hardware test that feeds the T090 final hardware gate.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import torch

from shardgrid.planner.generic_graph import FXGraphCaptureAdapter

REPO = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO / "tests" / "fixtures" / "ordinary_training_scripts" / "activation_pressure_train.py"
)


def _load_fixture_module():
    spec = importlib.util.spec_from_file_location(
        "activation_pressure_train_contract", FIXTURE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrected_fixture_is_activation_driven_not_parameter_heavy() -> None:
    module = _load_fixture_module()
    model = module.ActivationPressureNet(width=256, depth=80, retention=4)
    parameter_bytes = sum(
        int(t.numel()) * max(t.element_size(), 1) for t in model.parameters()
    )
    activation_bytes = module.activation_footprint_bytes(
        width=256, depth=80, batch=8, sequence=1024, retention=4
    )
    assert activation_bytes > parameter_bytes * 8, (
        f"activation footprint {activation_bytes} must dominate parameters "
        f"{parameter_bytes}; fixture must not be parameter-heavy"
    )


def test_backend_artifact_stays_bounded_for_activation_pressure_fixture() -> None:
    module = _load_fixture_module()
    model = module.ActivationPressureNet(width=256, depth=80, retention=4)
    parameter_bytes = sum(
        int(t.numel()) * max(t.element_size(), 1) for t in model.parameters()
    )
    capture = FXGraphCaptureAdapter().capture(
        model.eval(), sample_args=(torch.randn(2, 8, 1024, 256),)
    )
    backend_graph = capture.backend_graph
    for node in backend_graph.graph.nodes:
        node.type = None
    from torch.fx import GraphModule

    backend_graph = GraphModule(backend_graph, backend_graph.graph)
    buffer = io.BytesIO()
    torch.save(backend_graph, buffer)
    artifact_bytes = len(buffer.getvalue())
    assert artifact_bytes < parameter_bytes * 4, (
        f"backend graph artifact {artifact_bytes} must stay bounded near "
        f"metadata size, not scale with the model state ({parameter_bytes})"
    )


def test_corrected_fixture_has_no_dummy_cuda_reserve_and_no_probe() -> None:
    source = FIXTURE.read_text(encoding="utf-8")
    assert "torch.empty" not in source
    assert 'device="cuda"' not in source
    assert "memory_probe" not in source
    assert "BandMLP" not in source


def test_band_calibrations_span_wide_activation_footprint_range() -> None:
    module = _load_fixture_module()
    footprints = {
        band: module.activation_footprint_bytes(
            width=width, depth=depth, batch=batch, sequence=sequence, retention=retention
        )
        for band, width, depth, retention, batch, sequence in (
            ("MEM_4G", 256, 160, 4, 8, 1024),
            ("MEM_3G", 256, 120, 4, 8, 1024),
            ("MEM_2G", 256, 80, 4, 8, 1024),
            ("MEM_1G", 256, 40, 4, 8, 1024),
            ("MEM_512M", 256, 20, 4, 8, 1024),
        )
    }
    assert footprints["MEM_4G"] > footprints["MEM_512M"] * 4, footprints
    assert footprints["MEM_4G"] > footprints["MEM_1G"], footprints