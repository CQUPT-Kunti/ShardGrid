"""Formal Gate 1 single-GPU acceptance tests (T052).

The unit tests in this module validate the Gate 1 decision logic with mock
payloads.  The live test executes the real CUDA/PyTorch smoke on both physical
GPU Workers from Machine A (SSH -> WSL2 -> selected Conda) and asserts an
honest all-or-nothing Gate 1 result; it is opt-in via the ``hardware`` marker.
"""

from __future__ import annotations

import os
from typing import Any

from shardgrid.workers.single_gpu_gate import (
    GATE_STATUS_BLOCKED,
    GATE_STATUS_FAIL,
    GATE_STATUS_PASS,
    GATE_STATUS_PENDING,
    SingleGPUGateResult,
    attempt_single_gpu_smoke,
    build_gpu_smoke_script,
    evaluate_gate1,
    parse_gpu_gate_result,
    save_gate1_evidence,
)


def _payload(
    *,
    smoke_ok: bool = True,
    environment_ok: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "worker_id": "gpu4060",
        "expected_gpu": "RTX 4060",
        "cuda_available": smoke_ok,
        "device_count": 1,
        "gpu_name": "NVIDIA GeForce RTX 4060",
        "gpu_match": smoke_ok,
        "tensor_finite": smoke_ok,
        "tensor_operation": "1024x1024 @ 1024x1024 matmul on CUDA",
        "smoke_ok": smoke_ok,
        "environment_ok": environment_ok,
        "error": None if smoke_ok else "smoke failed",
    }
    payload.update(overrides)
    return payload


def _result(
    worker_id: str = "gpu4060",
    expected_gpu: str = "RTX 4060",
    *,
    result: dict[str, Any] | None = None,
    **payload_overrides: Any,
) -> SingleGPUGateResult:
    return SingleGPUGateResult(
        worker_id=worker_id,
        expected_gpu=expected_gpu,
        exit_code=0,
        timed_out=False,
        result=_payload(**payload_overrides) if result is None else result,
        stdout="",
        stderr="",
    )


def test_single_gpu_status_pass() -> None:
    result = _result()
    assert result.status == GATE_STATUS_PASS


def test_single_gpu_status_fail_on_cuda_unavailable() -> None:
    result = _result(
        smoke_ok=False,
        error="torch.cuda.is_available() is False or device_count == 0",
    )
    assert result.status == GATE_STATUS_FAIL


def test_single_gpu_status_fail_on_wrong_gpu_identity() -> None:
    result = _result(
        smoke_ok=False,
        gpu_match=False,
        gpu_name="NVIDIA GeForce GT 1030",
        error="detected GPU 'NVIDIA GeForce GT 1030' does not match expected",
    )
    assert result.status == GATE_STATUS_FAIL


def test_single_gpu_status_fail_on_tensor_operation_failure() -> None:
    result = _result(
        smoke_ok=False,
        tensor_finite=False,
        error="CUDA tensor operation returned non-finite values",
    )
    assert result.status == GATE_STATUS_FAIL


def test_single_gpu_status_blocked_on_missing_result() -> None:
    result = SingleGPUGateResult(
        worker_id="gpu4060",
        expected_gpu="RTX 4060",
        exit_code=1,
        timed_out=False,
        result=None,
        stdout="",
        stderr="",
    )
    assert result.status == GATE_STATUS_BLOCKED


def test_single_gpu_status_blocked_on_missing_runtime_environment() -> None:
    result = _result(environment_ok=False, error="No module named 'torch'")
    assert result.status == GATE_STATUS_BLOCKED


def test_single_gpu_status_blocked_on_incomplete_evidence() -> None:
    result = _result(result={"worker_id": "gpu4060"})
    assert result.status == GATE_STATUS_BLOCKED


def test_gate1_pass_when_both_workers_pass() -> None:
    gate = evaluate_gate1(
        [_result(), _result(worker_id="gpu1060", expected_gpu="GTX 1650")]
    )
    assert gate.status == GATE_STATUS_PASS


def test_gate1_fail_when_rtx4060_fails() -> None:
    gate = evaluate_gate1(
        [
            _result(smoke_ok=False),
            _result(worker_id="gpu1060", expected_gpu="GTX 1650"),
        ]
    )
    assert gate.status == GATE_STATUS_FAIL


def test_gate1_fail_when_gtx1650_fails() -> None:
    gate = evaluate_gate1(
        [
            _result(),
            _result(
                worker_id="gpu1060", expected_gpu="GTX 1650", smoke_ok=False
            ),
        ]
    )
    assert gate.status == GATE_STATUS_FAIL


def test_gate1_fail_is_all_or_nothing() -> None:
    gate = evaluate_gate1(
        [_result(smoke_ok=True), _result(worker_id="gpu1060", smoke_ok=False)]
    )
    assert gate.status == GATE_STATUS_FAIL
    assert gate.status != GATE_STATUS_PASS


def test_gate1_blocked_when_one_worker_blocked() -> None:
    blocked = SingleGPUGateResult(
        worker_id="gpu1060",
        expected_gpu="GTX 1650",
        exit_code=1,
        timed_out=False,
        result=None,
        stdout="",
        stderr="",
    )
    gate = evaluate_gate1([_result(), blocked])
    assert gate.status == GATE_STATUS_BLOCKED


def test_gate1_pending_when_no_results() -> None:
    gate = evaluate_gate1([])
    assert gate.status == GATE_STATUS_PENDING


def test_blocked_never_misjudged_pass() -> None:
    gate = evaluate_gate1(
        [
            _result(),
            _result(worker_id="gpu1060", environment_ok=False),
        ]
    )
    assert gate.status == GATE_STATUS_BLOCKED
    assert gate.status != GATE_STATUS_PASS


def test_pending_never_misjudged_pass() -> None:
    gate = evaluate_gate1([])
    assert gate.status == GATE_STATUS_PENDING
    assert gate.status != GATE_STATUS_PASS


def test_build_script_injects_worker_and_expected_gpu() -> None:
    script = build_gpu_smoke_script(worker_id="gpu4060", expected_gpu="RTX 4060")
    assert 'worker_id = "gpu4060"' in script
    assert 'expected_gpu = "RTX 4060"' in script
    assert "torch.randn(1024, 1024, device=\"cuda\")" in script
    assert "torch.cuda.synchronize()" in script
    assert "GPU_GATE_RESULT" in script


def test_parse_gpu_gate_result() -> None:
    payload = parse_gpu_gate_result(
        'log line\nGPU_GATE_RESULT {"worker_id": "gpu4060", "smoke_ok": true}\n'
    )
    assert payload == {"worker_id": "gpu4060", "smoke_ok": True}


def test_parse_gpu_gate_result_missing() -> None:
    assert parse_gpu_gate_result("no result here") is None
    assert parse_gpu_gate_result("GPU_GATE_RESULT not-json") is None


def test_gate1_status_serialization_and_evidence() -> None:
    gate = evaluate_gate1(
        [_result(), _result(worker_id="gpu1060", expected_gpu="GTX 1650")]
    )
    payload = gate.to_dict()
    assert payload["gate_id"] == "gate1-single-gpu"
    assert payload["status"] == GATE_STATUS_PASS
    assert len(payload["workers"]) == 2
    assert payload["workers"][0]["result"]["smoke_ok"] is True


def test_live_gate1_both_workers() -> None:
    """Real Gate 1 from Machine A: RTX 4060 + GTX 1650 (opt-in hardware marker)."""
    import json as _json
    from dataclasses import replace

    from shardgrid.common.config import load_cluster_config
    from shardgrid.common.models import as_hostname
    from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
    from shardgrid.transport.ssh import SSHOptions, SSHTransport

    config = load_cluster_config("examples/workers.yaml")
    address_book = _json.load(open("tests/address.json"))

    def build_wrapper(worker_id: str, expected_gpu: str) -> WSLRuntimeWrapper:
        worker = next(w for w in config.workers if str(w.worker_id) == worker_id)
        entry = next(
            e
            for e in address_book
            if expected_gpu.replace(" ", "")
            in str(e.get("gpu_model") or "").replace(" ", "")
        )
        worker = replace(
            worker,
            host=as_hostname(str(entry["ip"])),
            ssh_user=str(entry["username"]),
        )
        transport = SSHTransport(
            SSHOptions.from_ssh_config(
                config.ssh,
                host=str(entry["ip"]),
                user=worker.ssh_user,
                port=worker.ssh_port,
            )
        )
        return WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(worker, config.runtime), transport
        )

    targets = [("gpu4060", "RTX 4060"), ("gpu1060", "GTX 1650")]
    results = [
        attempt_single_gpu_smoke(
            build_wrapper(worker_id, expected_gpu),
            worker_id=worker_id,
            expected_gpu=expected_gpu,
            timeout=120.0,
        )
        for worker_id, expected_gpu in targets
    ]

    status = evaluate_gate1(results)
    output_dir = os.environ.get("SHARDGRID_GATE_EVIDENCE_DIR") or "/var/tmp/shardgrid/gates"
    evidence = save_gate1_evidence(status, output_dir)

    detail = "\n".join(
        f"{result.worker_id}: {result.status} stderr={result.stderr[-300:]!r}"
        for result in results
    )
    assert status.status == GATE_STATUS_PASS, (
        f"Gate 1 status={status.status}; evidence={evidence}\n{detail}"
    )
    for result in results:
        assert result.result is not None
        assert result.result["smoke_ok"] is True
        assert result.result["gpu_match"] is True
        assert result.result["cuda_available"] is True
        assert result.result["tensor_finite"] is True