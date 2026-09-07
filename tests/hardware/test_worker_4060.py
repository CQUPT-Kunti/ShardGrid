"""Real RTX 4060 single-GPU CUDA/PyTorch smoke test (T034).

This test is designed to run INSIDE the WSL2 training runtime of the RTX 4060
Worker, using the selected Conda environment.  It is opt-in: pytest skips it
unless ``--run-hardware`` is passed and ``SHARDGRID_ENABLE_HARDWARE_TESTS=1``
is set (see tests/conftest.py).

The test verifies the actual detected GPU is the target RTX 4060 and executes
a real CUDA tensor operation.  A JSON hardware finding (PASS / FAIL / BLOCKED)
is always written with the real evidence; it never infers success from
configuration or mock data.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

EXPECTED_GPU_SUBSTRING = "RTX 4060"
TARGET_WORKER_ID = "gpu4060"
TENSOR_SIZE = 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _findings_dir() -> Path:
    override = os.environ.get("SHARDGRID_HARDWARE_FINDINGS_DIR")
    return Path(override) if override else Path.home() / ".shardgrid" / "hardware-findings"


def _nvidia_smi() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return completed.stdout.strip()


def _write_finding(payload: dict[str, Any]) -> Path:
    directory = _findings_dir()
    directory.mkdir(parents=True, exist_ok=True)
    timestamped = directory / f"rtx4060-smoke-{_now()}.json"
    timestamped.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (directory / "rtx4060-latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return timestamped


def _base_finding() -> dict[str, Any]:
    return {
        "worker_id": TARGET_WORKER_ID,
        "timestamp": _now(),
        "physical_os": "windows",
        "runtime_os": "wsl2_linux",
        "host": platform.node(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "conda": {
            "executable": os.environ.get("CONDA_EXE"),
            "environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "prefix": os.environ.get("CONDA_PREFIX"),
        },
        "commands": [],
        "status": "BLOCKED",
        "reason": None,
        "details": {},
    }


def _finish(finding: dict[str, Any], message: str) -> None:
    path = _write_finding(finding)
    pytest.fail(f"{message} (finding: {path})")


def test_rtx4060_single_gpu_cuda_smoke() -> None:
    finding = _base_finding()
    finding["commands"].append(f"{sys.executable} -V")

    conda_prefix = finding["conda"]["prefix"]
    if not conda_prefix or not sys.executable.startswith(conda_prefix):
        finding["status"] = "FAIL"
        finding["reason"] = (
            "training Python is not inside the selected Conda environment "
            f"(CONDA_PREFIX={conda_prefix!r}, sys.executable={sys.executable!r})"
        )
        _finish(finding, finding["reason"])

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        finding["status"] = "BLOCKED"
        finding["reason"] = (
            f"PyTorch is not installed in the selected Conda environment: {exc}"
        )
        _finish(finding, finding["reason"])

    torch_version = torch.__version__
    torch_cuda_version = torch.version.cuda
    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    finding["details"]["torch"] = {
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "cuda_available": available,
        "device_count": device_count,
    }

    if not available or device_count == 0:
        finding["status"] = "FAIL"
        finding["reason"] = (
            f"CUDA is not available from the selected Conda environment: "
            f"torch.cuda.is_available()={available}, device_count={device_count}"
        )
        _finish(finding, finding["reason"])

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    capability_text = ".".join(str(part) for part in capability)
    finding["details"]["gpu"] = {
        "name": name,
        "compute_capability": capability_text,
        "total_vram_mb": properties.total_memory // (1024 * 1024),
        "current_device": device,
    }

    if EXPECTED_GPU_SUBSTRING not in name:
        finding["status"] = "FAIL"
        finding["reason"] = (
            f"detected GPU {name!r} is not the target {EXPECTED_GPU_SUBSTRING}"
        )
        _finish(finding, finding["reason"])

    left = torch.randn(TENSOR_SIZE, TENSOR_SIZE, device=f"cuda:{device}")
    right = torch.randn(TENSOR_SIZE, TENSOR_SIZE, device=f"cuda:{device}")
    product = left @ right
    torch.cuda.synchronize()
    result = product.cpu()
    finite = bool(torch.isfinite(result).all().item())
    finding["details"]["tensor_operation"] = {
        "op": f"{TENSOR_SIZE}x{TENSOR_SIZE} @ {TENSOR_SIZE}x{TENSOR_SIZE} matmul on CUDA",
        "result_shape": list(result.shape),
        "finite": finite,
        "synchronized": True,
    }
    if not finite:
        finding["status"] = "FAIL"
        finding["reason"] = "CUDA tensor operation returned non-finite values"
        _finish(finding, finding["reason"])

    nvidia_smi = _nvidia_smi()
    finding["details"]["nvidia_smi"] = nvidia_smi

    finding["status"] = "PASS"
    path = _write_finding(finding)
    print(f"RTX 4060 hardware smoke PASS; finding: {path}")