"""Formal Gate 1 single-GPU acceptance (T052).

Gate 1 proves that each physical GPU Worker independently passes a real
CUDA/PyTorch smoke test inside its own WSL2 selected Conda training runtime.
The gate is all-or-nothing: it PASSES only when BOTH Workers report a real
smoke PASS.

This module reuses the existing T040 ``WSLRuntimeWrapper`` (SSH -> WSL2 ->
selected Conda) and T041 probe conventions.  It does not reimplement SSH, WSL,
Conda detection, or GPU probing.  A single unified smoke script runs on both
Workers; the gate logic is one shared evaluator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from shardgrid.transport.runtime import WSLRuntimeWrapper

GATE_STATUS_PASS = "PASS"
GATE_STATUS_FAIL = "FAIL"
GATE_STATUS_BLOCKED = "BLOCKED"
GATE_STATUS_PENDING = "PENDING"

_GPU_SMOKE_SCRIPT = """
import json
import os
import platform
import socket
import subprocess
import sys
import time

worker_id = "__WORKER_ID__"
expected_gpu = "__EXPECTED_GPU__"
start = time.time()

out = {
    "worker_id": worker_id,
    "expected_gpu": expected_gpu,
    "hostname": socket.gethostname(),
    "physical_os": "windows",
    "runtime_os": "wsl2_linux",
    "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": None,
    "torch_cuda_version": None,
    "torch_import_ok": False,
    "cuda_available": False,
    "device_count": 0,
    "gpu_name": None,
    "gpu_match": False,
    "gpu_compute_capability": None,
    "gpu_total_vram_mb": None,
    "driver": None,
    "tensor_operation": None,
    "tensor_finite": None,
    "smoke_ok": False,
    "environment_ok": True,
    "elapsed_s": None,
    "error": None,
}

try:
    import torch

    out["torch_version"] = torch.__version__
    out["torch_cuda_version"] = str(torch.version.cuda)
    out["torch_import_ok"] = True
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("torch.cuda.is_available() is False or device_count == 0")
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["device_count"] = int(torch.cuda.device_count())
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    out["gpu_name"] = name
    out["gpu_compute_capability"] = ".".join(
        str(part) for part in torch.cuda.get_device_capability(device)
    )
    out["gpu_total_vram_mb"] = (
        torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
    )
    out["gpu_match"] = expected_gpu in name
    if not out["gpu_match"]:
        raise RuntimeError(
            "detected GPU %r does not match expected %r" % (name, expected_gpu)
        )
    left = torch.randn(1024, 1024, device="cuda")
    right = torch.randn(1024, 1024, device="cuda")
    product = left @ right
    torch.cuda.synchronize()
    result = product.cpu()
    out["tensor_operation"] = "1024x1024 @ 1024x1024 matmul on CUDA"
    out["tensor_finite"] = bool(torch.isfinite(result).all().item())
    if not out["tensor_finite"]:
        raise RuntimeError("CUDA tensor operation returned non-finite values")
    out["smoke_ok"] = True
except ModuleNotFoundError as exc:
    out["environment_ok"] = False
    out["error"] = str(exc)
except Exception as exc:
    out["error"] = str(exc)
finally:
    out["elapsed_s"] = round(time.time() - start, 3)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=30,
        ).strip()
        out["driver"] = smi
    except Exception as exc:
        out["driver"] = "unavailable: %s" % exc
    print("GPU_GATE_RESULT " + json.dumps(out, sort_keys=True))
"""


def build_gpu_smoke_script(*, worker_id: str, expected_gpu: str) -> str:
    script = _GPU_SMOKE_SCRIPT.replace("__WORKER_ID__", worker_id)
    script = script.replace("__EXPECTED_GPU__", expected_gpu)
    return script


def parse_gpu_gate_result(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU_GATE_RESULT "):
            try:
                payload = json.loads(stripped.split(" ", 1)[1])
            except ValueError:
                return None
            if isinstance(payload, dict):
                return payload
    return None


@dataclass(frozen=True)
class SingleGPUGateResult:
    worker_id: str
    expected_gpu: str
    exit_code: int | None
    timed_out: bool
    result: dict[str, Any] | None
    stdout: str
    stderr: str

    @property
    def status(self) -> str:
        """Derive the per-Worker gate status from the real smoke evidence."""
        if self.result is None:
            return GATE_STATUS_BLOCKED
        if "smoke_ok" not in self.result or "environment_ok" not in self.result:
            return GATE_STATUS_BLOCKED
        if self.result.get("environment_ok") is not True:
            return GATE_STATUS_BLOCKED
        if self.result.get("smoke_ok") is True:
            return GATE_STATUS_PASS
        return GATE_STATUS_FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "expected_gpu": self.expected_gpu,
            "status": self.status,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "result": self.result,
            "stderr_tail": self.stderr[-4000:],
            "stdout_tail": self.stdout[-4000:],
        }


def run_single_gpu_smoke(
    wrapper: WSLRuntimeWrapper,
    *,
    worker_id: str,
    expected_gpu: str,
    timeout: float = 120.0,
) -> SingleGPUGateResult:
    script = build_gpu_smoke_script(worker_id=worker_id, expected_gpu=expected_gpu)
    result = wrapper.run_script(script, timeout=timeout)
    return SingleGPUGateResult(
        worker_id=worker_id,
        expected_gpu=expected_gpu,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        result=parse_gpu_gate_result(result.stdout),
        stdout=result.stdout,
        stderr=result.stderr,
    )


def attempt_single_gpu_smoke(
    wrapper: WSLRuntimeWrapper,
    *,
    worker_id: str,
    expected_gpu: str,
    timeout: float = 120.0,
) -> SingleGPUGateResult:
    """Run the smoke, converting transport/wrapper failures into BLOCKED."""
    try:
        return run_single_gpu_smoke(
            wrapper, worker_id=worker_id, expected_gpu=expected_gpu, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001 - surfaced into gate evidence
        return SingleGPUGateResult(
            worker_id=worker_id,
            expected_gpu=expected_gpu,
            exit_code=None,
            timed_out=False,
            result=None,
            stdout="",
            stderr=str(exc),
        )


@dataclass(frozen=True)
class Gate1Status:
    status: str
    workers: tuple[SingleGPUGateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": "gate1-single-gpu",
            "status": self.status,
            "all_or_nothing": True,
            "workers": [worker.to_dict() for worker in self.workers],
        }


def evaluate_gate1(
    results: Sequence[SingleGPUGateResult],
) -> Gate1Status:
    """All-or-nothing Gate 1 decision.

    - PASS: every Worker passed its real smoke test.
    - FAIL: real execution happened and at least one Worker FAILED.
    - BLOCKED: no FAIL but at least one Worker could not complete the real test.
    - PENDING: no results yet.
    """
    if not results:
        return Gate1Status(status=GATE_STATUS_PENDING, workers=())
    statuses = [result.status for result in results]
    if all(status == GATE_STATUS_PASS for status in statuses):
        gate = GATE_STATUS_PASS
    elif any(status == GATE_STATUS_FAIL for status in statuses):
        gate = GATE_STATUS_FAIL
    elif any(status == GATE_STATUS_BLOCKED for status in statuses):
        gate = GATE_STATUS_BLOCKED
    else:
        gate = GATE_STATUS_PENDING
    return Gate1Status(status=gate, workers=tuple(results))


def save_gate1_evidence(
    status: Gate1Status,
    output_dir: str | Path,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = status.to_dict()
    payload["timestamp"] = timestamp
    path = directory / f"gate1-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (directory / "gate1-latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return path