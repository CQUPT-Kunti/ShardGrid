from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from shardgrid.bootstrap import runner


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ordinary_training_scripts"
GB = 1024**3
CONTROL_PLANE_RAM_BUDGET = 16 * GB
DECLARED_STATE_BYTES = 30 * GB
DECLARED_STATE_CASES = (
    pytest.param(30 * GB, id="30G"),
    pytest.param(70 * GB, id="70G"),
    pytest.param(100 * GB, id="100G"),
)


class _StorageRequestRecorder:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.requests: list[int] = []
        self._original_empty = torch.empty
        monkeypatch.setattr(torch, "empty", self._empty)

    def _empty(self, *size: Any, **kwargs: Any) -> torch.Tensor:
        requested_numel = self._requested_numel(size)
        dtype = kwargs.get("dtype", torch.float32)
        element_size = self._original_empty((), dtype=dtype).element_size()
        requested_bytes = requested_numel * element_size
        self.requests.append(requested_bytes)
        if requested_bytes > CONTROL_PLANE_RAM_BUDGET:
            return self._original_empty((1,), dtype=dtype)
        return self._original_empty(*size, **kwargs)

    @staticmethod
    def _requested_numel(size: tuple[Any, ...]) -> int:
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            dims = tuple(size[0])
        else:
            dims = size
        numel = 1
        for dim in dims:
            if dim in ((), None):
                continue
            numel *= int(dim)
        return numel

    @property
    def oversize_requests(self) -> list[int]:
        return [value for value in self.requests if value > CONTROL_PLANE_RAM_BUDGET]


def test_materialization_detection_harness_prevents_real_large_cpu_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _StorageRequestRecorder(monkeypatch)

    tensor = torch.empty(DECLARED_STATE_BYTES // 4, dtype=torch.float32)

    assert tensor.numel() == 1
    assert recorder.oversize_requests
    assert recorder.oversize_requests[-1] >= DECLARED_STATE_BYTES


def test_ordinary_entrypoint_planning_does_not_request_full_cpu_parameter_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = _StorageRequestRecorder(monkeypatch)

    result = runner.capture_entrypoint(
        FIXTURE_ROOT / "large_declared_state_train.py",
        argv=("--checkpoint", str(tmp_path / "large-state.pt")),
        cwd=FIXTURE_ROOT,
        environment={
            "SHARDGRID_DECLARED_STATE_BYTES": str(DECLARED_STATE_BYTES),
            "SHARDGRID_CONTROL_PLANE_RAM_BUDGET_BYTES": str(CONTROL_PLANE_RAM_BUDGET),
        },
        dry_run=True,
    )

    assert not getattr(result, "ok", True) is False
    assert recorder.oversize_requests == []


@pytest.mark.parametrize("declared_state_bytes", DECLARED_STATE_CASES)
def test_declared_state_larger_than_control_plane_ram_reaches_bounded_planning_capture(
    declared_state_bytes: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = _StorageRequestRecorder(monkeypatch)
    checkpoint_path = tmp_path / f"declared-{declared_state_bytes}.pt"

    result = runner.capture_entrypoint(
        FIXTURE_ROOT / "large_declared_state_train.py",
        argv=("--checkpoint", str(checkpoint_path)),
        cwd=FIXTURE_ROOT,
        environment={
            "SHARDGRID_DECLARED_STATE_BYTES": str(declared_state_bytes),
            "SHARDGRID_CONTROL_PLANE_RAM_BUDGET_BYTES": str(CONTROL_PLANE_RAM_BUDGET),
        },
        dry_run=True,
    )

    assert not getattr(result, "ok", True) is False
    assert declared_state_bytes > CONTROL_PLANE_RAM_BUDGET
    assert recorder.oversize_requests == []
    assert result.parameter_count == 1
    assert result.graph_capture["backend"] == "shardgrid.metadata_declaration"
    assert "BOUNDED_METADATA_CAPTURE_WITHOUT_REAL_CPU_EXECUTION" in result.graph_capture[
        "diagnostics"
    ]
    assert not checkpoint_path.exists()


def test_dynamic_control_flow_fails_closed_before_training_mutation(tmp_path: Path) -> None:
    marker = tmp_path / "mutation"
    script = tmp_path / "dynamic_control_flow.py"
    script.write_text(
        f"""
from pathlib import Path

import torch
from torch import nn


class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.positive = nn.Linear(2, 1)
        self.negative = nn.Linear(2, 1)

    def forward(self, x):
        if x.sum() > 0:
            return self.positive(x)
        return self.negative(x)


model = DynamicModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss = model(torch.ones(1, 2)).sum()
loss.backward()
optimizer.step()
Path({str(marker)!r}).write_text("mutated", encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.capture_entrypoint(script, cwd=tmp_path, environment={}, dry_run=True)

    assert result.ok is False
    assert result.failure.stage == "graph_capture"
    assert result.failure.code == "DYNAMIC_CONTROL_FLOW_UNSUPPORTED"
    assert "real CPU execution" in result.failure.message
    assert not marker.exists()


def test_custom_forward_function_fails_closed_without_cpu_fallback(tmp_path: Path) -> None:
    marker = tmp_path / "mutation"
    script = tmp_path / "custom_op.py"
    script.write_text(
        f"""
from pathlib import Path

import torch
from torch import nn


def user_custom_op(x):
    Path({str(marker)!r}).write_text("forward-entered", encoding="utf-8")
    return x.sin()


class CustomOpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def forward(self, x):
        return user_custom_op(self.proj(x))


model = CustomOpModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = model(torch.ones(1, 2)).sum()
loss.backward()
optimizer.step()
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.capture_entrypoint(script, cwd=tmp_path, environment={}, dry_run=True)

    assert result.ok is False
    assert result.failure.stage == "graph_capture"
    assert result.failure.code == "CUSTOM_OP_UNSUPPORTED"
    assert "custom function" in result.failure.message
    assert not marker.exists()


def test_missing_tensor_metadata_fails_closed_before_forward(tmp_path: Path) -> None:
    marker = tmp_path / "forward"
    script = tmp_path / "missing_tensor_metadata.py"
    script.write_text(
        f"""
from pathlib import Path

import torch
from torch import nn


class MetadataModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 1)

    def forward(self, payload):
        Path({str(marker)!r}).write_text("forward-entered", encoding="utf-8")
        return self.proj(torch.ones(1, 2))


model = MetadataModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss = model({{"not_tensor": "value"}}).sum()
loss.backward()
optimizer.step()
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.capture_entrypoint(script, cwd=tmp_path, environment={}, dry_run=True)

    assert result.ok is False
    assert result.failure.stage == "capture"
    assert result.failure.code == "MISSING_REQUIRED_METADATA"
    assert "tensor input metadata" in result.failure.message
    assert not marker.exists()


def test_custom_optimizer_mutation_fails_closed_before_model_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "forward"
    script = tmp_path / "custom_optimizer.py"
    script.write_text(
        f"""
from pathlib import Path

import torch
from torch import nn


class HiddenMutationOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {{"lr": 0.1}})

    def step(self, closure=None):
        for group in self.param_groups:
            for param in group["params"]:
                param.data.add_(1.0)


class OptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 1)

    def forward(self, x):
        Path({str(marker)!r}).write_text("forward-entered", encoding="utf-8")
        return self.proj(x)


model = OptimizerModel()
optimizer = HiddenMutationOptimizer(model.parameters())
loss = model(torch.ones(1, 2)).sum()
loss.backward()
optimizer.step()
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.capture_entrypoint(script, cwd=tmp_path, environment={}, dry_run=True)

    assert result.ok is False
    assert result.failure.stage == "lifecycle_capture"
    assert result.failure.code == "HIDDEN_OPTIMIZER_MUTATION_UNSUPPORTED"
    assert not marker.exists()
