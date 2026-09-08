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
