"""Formal Gate 2 distributed acceptance tests (T053).

The unit tests validate the Gate 2 decision logic against dist-test reports
(controlled payloads).  The live test executes the real ``shardgrid dist-test``
on the two physical Workers and evaluates the gate on the real report; it is
opt-in via the ``multi_host`` marker.
"""

from __future__ import annotations

import json
from typing import Any

from shardgrid.distributed.distributed_gate import (
    EXPECTED_ALL_REDUCE,
    EXPECTED_BROADCAST,
    EXPECTED_SEND_RECV,
    GATE2_BLOCKED,
    GATE2_FAIL,
    GATE2_PASS,
    GATE2_PENDING,
    evaluate_gate2,
    save_gate2_evidence,
)


def _ok_map() -> dict[str, Any]:
    return {"rank0": True, "rank1": True}


def _collectives(
    *,
    process_ok: bool = True,
    broadcast_ok: bool = True,
    send_recv_ok: bool = True,
    all_reduce_ok: bool = True,
    tensor_ok: bool = True,
) -> dict[str, Any]:
    def section(
        ok: bool, expected: list[float]
    ) -> dict[str, Any]:
        return {
            "ok": _ok_map() if ok else {"rank0": False, "rank1": False},
            "tensor": (
                {"rank0": expected, "rank1": expected}
                if tensor_ok
                else {"rank0": [0.0], "rank1": [0.0]}
            ),
        }

    return {
        "run_id": "run-123",
        "process_group": {
            "ok": process_ok,
            "rank0_init": process_ok,
            "rank1_init": process_ok,
        },
        "broadcast": section(broadcast_ok, EXPECTED_BROADCAST),
        "send_recv": section(send_recv_ok, EXPECTED_SEND_RECV),
        "all_reduce": section(all_reduce_ok, EXPECTED_ALL_REDUCE),
        "elapsed_s": {"rank0": 10.0, "rank1": 10.0},
    }


def _per_rank(worker_id: str, gpu: str, interface: str) -> dict[str, Any]:
    return {
        "run_id": "run-123",
        "worker_id": worker_id,
        "conda_environment": "shardgrid",
        "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
        "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        "python_version": "3.12.13",
        "torch_version": "2.7.1+cu118",
        "torch_cuda_version": "11.8",
        "gpu_name": gpu,
        "network_interface": interface,
        "master_addr": "10.87.5.155",
        "master_port": 29500,
        "smoke_ok": True,
    }


def _report(
    *,
    backend_state: str = "NCCL SUCCESS",
    backend_actual: str = "nccl",
    workers: list[tuple[str, str]] | None = None,
    world_size: int = 2,
    local_world_size: int = 1,
    collectives: dict[str, Any] | None = None,
    include_runtime_evidence: bool = True,
    include_network: bool = True,
    nccl_failure_evidence: dict[str, Any] | None = None,
    include_gloo: bool = False,
    diagnostics_path: str | None = "/tmp/dist-test.json",
) -> dict[str, Any]:
    if workers is None:
        workers = [("gpu4060", "10.87.5.155"), ("gpu1060", "10.87.5.15")]
    used = collectives if collectives is not None else _collectives()
    collectives_payload: dict[str, Any] = {"nccl": _collectives()}
    if backend_actual == "gloo" or include_gloo:
        collectives_payload["gloo"] = used
    else:
        collectives_payload["nccl"] = used

    report: dict[str, Any] = {
        "run_id": "run-123",
        "backend_requested": "auto",
        "backend_state": backend_state,
        "backend_actual": backend_actual,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "workers": [
            {"worker_id": wid, "rank": rank, "ip": ip, "hostname": f"host-{rank}"}
            for rank, (wid, ip) in enumerate(workers)
        ],
        "network": (
            {
                "master_addr": "10.87.5.155",
                "master_port": 29500,
                "interfaces": {"0": "eth3", "1": "eth0"},
            }
            if include_network
            else {}
        ),
        "collectives": collectives_payload,
        "runtime_evidence": (
            {
                "run_id": "run-123",
                "conda_environment": "shardgrid",
                "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
                "python_version": "3.12.13",
                "torch_version": "2.7.1+cu118",
                "torch_cuda_version": "11.8",
                "per_rank": {
                    "0": _per_rank("gpu4060", "NVIDIA GeForce RTX 4060", "eth3"),
                    "1": _per_rank("gpu1060", "NVIDIA GeForce GTX 1650", "eth0"),
                },
            }
            if include_runtime_evidence
            else {}
        ),
        "nccl_failure_evidence": nccl_failure_evidence,
        "gloo_fallback": {"outcome": "PASS", "run_id": "run-123"} if include_gloo else None,
        "diagnostics_path": diagnostics_path,
    }
    return report


def _gate1_pass() -> str:
    return GATE2_PASS


def test_valid_nccl_evidence_gate2_pass() -> None:
    gate = evaluate_gate2(
        _report(), gate1_status=_gate1_pass(), expected_workers=["gpu4060", "gpu1060"]
    )
    assert gate.status == GATE2_PASS
    assert gate.problems == ()
    assert gate.backend_state == "NCCL SUCCESS"
    assert gate.backend_actual == "nccl"


def test_valid_gloo_fallback_evidence_gate2_pass() -> None:
    report = _report(
        backend_state="GLOO FALLBACK",
        backend_actual="gloo",
        nccl_failure_evidence={"outcome": "FAILED", "backend": "nccl", "error": "x"},
    )
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_PASS
    assert gate.backend_state == "GLOO FALLBACK"


def test_single_host_rejected() -> None:
    report = _report(workers=[("gpu4060", "10.87.5.155"), ("gpu1060", "10.87.5.155")])
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_BLOCKED
    assert any("two distinct physical hosts" in p for p in gate.problems)


def test_missing_gate1_blocked() -> None:
    for gate1 in (GATE2_FAIL, GATE2_BLOCKED, GATE2_PENDING):
        gate = evaluate_gate2(_report(), gate1_status=gate1)
        assert gate.status == GATE2_BLOCKED
        assert "Gate 1" in gate.problems[0]


def test_failed_process_group() -> None:
    report = _report(collectives=_collectives(process_ok=False))
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_FAIL
    assert "process group" in gate.problems[0]


def test_failed_collective() -> None:
    report = _report(collectives=_collectives(broadcast_ok=False))
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_FAIL
    assert "broadcast" in gate.problems[0]


def test_invalid_tensor_result() -> None:
    report = _report(collectives=_collectives(tensor_ok=False))
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_FAIL
    assert "tensor result invalid" in gate.problems[0]


def test_missing_runtime_evidence() -> None:
    report = _report(include_runtime_evidence=False)
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_BLOCKED
    assert any("evidence incomplete" in p for p in gate.problems)


def test_missing_network_evidence() -> None:
    report = _report(include_network=False)
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_BLOCKED
    assert any("evidence incomplete" in p for p in gate.problems)


def test_actual_backend_rank_evidence_can_satisfy_gate_when_top_level_is_stale() -> None:
    report = _report(
        backend_state="GLOO FALLBACK",
        backend_actual="gloo",
        nccl_failure_evidence={"outcome": "FAILED", "backend": "nccl", "error": "x"},
    )
    report["runtime_evidence"]["per_rank"]["0"].update(
        {
            "conda_environment": None,
            "conda_prefix": None,
            "python_executable": None,
            "python_version": None,
            "torch_version": None,
            "torch_cuda_version": None,
            "gpu_name": None,
        }
    )
    report["collectives"]["gloo"]["ranks"] = [
        _per_rank("gpu4060", "NVIDIA GeForce RTX 4060", "eth3") | {"rank": 0},
        _per_rank("gpu1060", "NVIDIA GeForce GTX 1650", "eth0") | {"rank": 1},
    ]
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_PASS


def test_missing_backend_label_blocked() -> None:
    report = _report()
    report["backend_state"] = None
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_FAIL
    assert "backend_state" in gate.problems[0]


def test_diagnostics_preserved() -> None:
    report = _report()
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_PASS


def test_missing_diagnostics_blocked() -> None:
    report = _report(diagnostics_path=None)
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_BLOCKED
    assert any("diagnostics evidence missing" in p for p in gate.problems)


def test_gloo_fallback_without_nccl_evidence() -> None:
    report = _report(
        backend_state="GLOO FALLBACK", backend_actual="gloo", nccl_failure_evidence=None
    )
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_FAIL
    assert "NCCL failure evidence" in gate.problems[0]


def test_invalid_backend_label_rejected() -> None:
    for state in ("NCCL FAILED", "GLOO PASS", "FALLBACK FAILED"):
        gate = evaluate_gate2(
            _report(backend_state=state, backend_actual="gloo"),
            gate1_status=_gate1_pass(),
        )
        assert gate.status == GATE2_FAIL
        assert "backend_state" in gate.problems[0]


def test_backend_not_allowed_blocked() -> None:
    gate = evaluate_gate2(
        _report(backend_state="FALLBACK NOT ALLOWED", backend_actual="none"),
        gate1_status=_gate1_pass(),
    )
    assert gate.status == GATE2_BLOCKED


def test_no_report_pending() -> None:
    gate = evaluate_gate2(None, gate1_status=_gate1_pass())
    assert gate.status == GATE2_PENDING


def test_missing_collectives_evidence() -> None:
    report = _report(collectives={})
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_BLOCKED
    assert any("collectives evidence" in p for p in gate.problems)


def test_expected_workers_mismatch() -> None:
    report = _report(workers=[("gpu4060", "10.87.5.155"), ("gpu4060e", "10.87.5.93")])
    gate = evaluate_gate2(
        report, gate1_status=_gate1_pass(), expected_workers=["gpu4060", "gpu1060"]
    )
    assert gate.status == GATE2_BLOCKED
    assert any("expected" in p for p in gate.problems)


def test_gate2_serialization_and_evidence(tmp_path) -> None:
    gate = evaluate_gate2(_report(), gate1_status=_gate1_pass())
    payload = gate.to_dict()
    assert payload["gate_id"] == "gate2-distributed"
    assert payload["status"] == GATE2_PASS
    path = save_gate2_evidence(gate, tmp_path)
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["status"] == GATE2_PASS


def test_mixed_run_id_evidence_blocked() -> None:
    report = _report()
    report["collectives"]["nccl"]["ranks"] = [
        _per_rank("gpu4060", "NVIDIA GeForce RTX 4060", "eth3") | {"rank": 0, "run_id": "run-123"},
        _per_rank("gpu1060", "NVIDIA GeForce GTX 1650", "eth0") | {"rank": 1, "run_id": "run-999"},
    ]
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_BLOCKED
    assert any("mixed run_id evidence" in p for p in gate.problems)


def test_missing_run_id_blocked() -> None:
    report = _report()
    report.pop("run_id")
    report["runtime_evidence"].pop("run_id", None)
    gate = evaluate_gate2(report, gate1_status=_gate1_pass())
    assert gate.status == GATE2_BLOCKED
    assert any("run_id missing" in p for p in gate.problems)


def test_live_gate2_real_pair() -> None:
    """Real Gate 2 from Machine A via dist-test (opt-in multi_host marker)."""
    import os

    from shardgrid.cli.app import main

    gate1_path = "/var/tmp/shardgrid/gates/gate1-latest.json"
    if not os.path.exists(gate1_path):
        raise AssertionError(
            "Gate 1 evidence missing; run the T052 live test first"
        )
    gate1_status = json.load(open(gate1_path))["status"]

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

    # Clean stale runtime processes before the run (documented T049 practice:
    # orphaned ranks hold the rendezvous port and poison subsequent attempts).
    for worker_id, expected_gpu in [
        ("gpu4060", "RTX 4060"),
        ("gpu1060", "GTX 1650"),
    ]:
        wrapper = build_wrapper(worker_id, expected_gpu)
        wrapper.run("pkill -9 -f miniconda3/envs/shardgrid/bin/python || true", timeout=15.0)

    report_path = "/var/tmp/shardgrid/gates/dist-test-gate2-live.json"
    exit_code = main(
        [
            "--config", "examples/workers.yaml",
            "dist-test", "--backend", "auto",
            "--workers", "gpu4060,gpu1060",
            "--save-report", report_path,
        ]
    )
    report = json.load(open(report_path))

    gate = evaluate_gate2(
        report,
        gate1_status=gate1_status,
        expected_workers=["gpu4060", "gpu1060"],
    )
    evidence = save_gate2_evidence(gate, "/var/tmp/shardgrid/gates")

    detail = "\n".join(f"- {problem}" for problem in gate.problems)
    assert exit_code == 0, f"dist-test exit code {exit_code}"
    assert gate.status == GATE2_PASS, (
        f"Gate 2 status={gate.status}; evidence={evidence}\n{detail}"
    )
    assert gate.backend_state in {"NCCL SUCCESS", "GLOO FALLBACK"}
    assert report["collectives"][gate.backend_actual]["process_group"]["ok"] is True
