from __future__ import annotations

import json

import pytest

from shardgrid.common.enums import BackendStatus, FailureStage
from shardgrid.control.compatibility_reports import (
    build_compatibility_report,
    compatibility_report_from_spike_report,
    load_compatibility_report,
    normalize_compatibility_status,
    validate_compatibility_report,
    write_compatibility_report,
)
from shardgrid.engines.models import CompatibilitySpikeReport


def _machine(worker_id: str) -> dict[str, str]:
    return {
        "worker_id": worker_id,
        "physical_os": "windows",
        "runtime_os": "wsl2_linux",
        "gpu": "NVIDIA GeForce GTX 1650" if worker_id == "gpu1060" else "RTX 4060",
        "conda_environment": "shardgrid",
        "python": "3.12.13",
        "torch": "2.7.1+cu118",
        "cuda": "11.8",
        "nccl": "2.21.5",
    }


def test_pass_report_round_trip(tmp_path) -> None:
    payload = build_compatibility_report(
        report_id="hw-pass",
        component="hardware",
        stage=FailureStage.PROBE,
        status="PASS",
        machines_tested=["gpu4060", "gpu1060"],
        machine_facts=[_machine("gpu4060"), _machine("gpu1060")],
        versions={"torch": "2.7.1+cu118", "cuda": "11.8"},
        commands=["nvidia-smi", "python -c import torch"],
        results=["both workers detected expected GPUs", "cuda available on both"],
        evidence_refs=["docs/operations/doctor-report.md"],
    )

    path = write_compatibility_report(tmp_path / "hardware.json", payload)
    loaded = load_compatibility_report(path)

    assert loaded["status"] == BackendStatus.AVAILABLE.value
    assert loaded["component"] == "hardware"
    assert loaded["machines_tested"] == ["gpu4060", "gpu1060"]
    assert len(loaded["machine_facts"]) == 2
    assert loaded["evidence_refs"] == ["docs/operations/doctor-report.md"]


def test_fail_report_requires_failure_evidence() -> None:
    payload = build_compatibility_report(
        report_id="k8s-fail",
        component="kubernetes",
        stage=FailureStage.SCHEDULE,
        status="FAIL",
        machines_tested=["gpu4060", "gpu1060"],
        versions={"kubernetes": "1.31.0", "nvidia-device-plugin": "0.16.0"},
        commands=["kubectl get nodes", "kubectl logs device-plugin"],
        results=["gpu plugin pod crashlooping"],
        blockers=["device plugin not healthy"],
        decision="kubernetes gpu gate failed",
        recommended_next_action="repair the device plugin before rerunning the gate",
        logs_path="/tmp/secret-token/device-plugin.log",
        evidence_refs=["/tmp/secret-token/device-plugin-summary.txt"],
        secrets=("secret-token",),
    )

    assert payload["status"] == BackendStatus.FAILED.value
    assert "secret-token" not in payload["logs_path"]
    assert "secret-token" not in payload["evidence_refs"][0]


def test_blocked_report_requires_next_action() -> None:
    with pytest.raises(ValueError, match="recommended_next_action"):
        build_compatibility_report(
            report_id="volcano-blocked",
            component="volcano",
            stage=FailureStage.SCHEDULE,
            status=BackendStatus.BLOCKED,
            machines_tested=["gpu4060"],
            versions={"volcano": "1.10.0"},
            commands=["kubectl get pods -n volcano-system"],
            results=["volcano not installed"],
            blockers=["operator has not installed volcano"],
            decision="volcano gate blocked",
        )


def test_fallback_report_does_not_claim_preferred_path_pass() -> None:
    payload = build_compatibility_report(
        report_id="network-fallback",
        component="network",
        stage=FailureStage.NETWORK,
        status="FALLBACK",
        machines_tested=["gpu4060", "gpu1060"],
        versions={"torch": "2.7.1+cu118", "nccl": "2.21.5"},
        commands=["dist-test --backend auto"],
        results=["nccl failed", "gloo retry passed"],
        blockers=["nccl path unavailable on current route"],
        decision="fallback backend accepted for the smoke gate",
        recommended_next_action="fix NCCL path before advertising NCCL as healthy",
        preferred_path="nccl",
        actual_path="gloo",
    )

    assert payload["status"] == BackendStatus.FALLBACK_USED.value
    assert payload["preferred_path"] == "nccl"
    assert payload["actual_path"] == "gloo"

    with pytest.raises(ValueError, match="fallback success must not be reported as PASS"):
        validate_compatibility_report(
            {
                **payload,
                "status": BackendStatus.AVAILABLE.value,
            }
        )


def test_experimental_report_uses_same_contract_for_engine() -> None:
    payload = build_compatibility_report(
        report_id="engine-exp",
        component="engine",
        stage=FailureStage.PROBE,
        status="EXPERIMENTAL",
        machines_tested=["gpu4060", "gpu1060"],
        machine_facts=[_machine("gpu4060"), _machine("gpu1060")],
        versions={"galvatron": "2.4.1", "torch": "2.7.1+cu118"},
        commands=["galvatron compatibility check"],
        results=["import and profiler checks passed", "heterogeneous setup still limited"],
        blockers=["smaller-gpu memory headroom remains tight"],
        decision="experimental support only",
        recommended_next_action="keep the static validation path as the supported route",
    )

    assert payload["status"] == BackendStatus.EXPERIMENTAL.value
    assert payload["component"] == "engine"


def test_unknown_status_does_not_become_pass() -> None:
    assert normalize_compatibility_status("unknown") is BackendStatus.NOT_CHECKED


def test_validate_missing_failure_context_rejected() -> None:
    payload = {
        "report_id": "broken",
        "component": "hami",
        "stage": FailureStage.GPU_SHARE.value,
        "machines_tested": [],
        "versions": {},
        "commands": [],
        "results": [],
        "logs_path": None,
        "status": BackendStatus.BLOCKED.value,
        "blockers": [],
        "decision": "hami blocked",
        "recommended_next_action": "install hami after kubernetes gate passes",
        "created_at": "2026-08-27T00:00:00+00:00",
        "machine_facts": [],
        "evidence_refs": [],
        "preferred_path": None,
        "actual_path": None,
    }
    with pytest.raises(ValueError, match="machines_tested"):
        validate_compatibility_report(payload)


def test_spike_report_wrapper_reuses_contract() -> None:
    spike = CompatibilitySpikeReport(
        report_id="galvatron-1",
        component="engine",
        stage=FailureStage.PROBE,
        machines_tested=["gpu4060"],
        versions={"galvatron": "2.4.1"},
        commands=["galvatron probe"],
        results=["available"],
        status=BackendStatus.AVAILABLE,
        created_at="2026-08-27T00:00:00+00:00",
    )

    payload = compatibility_report_from_spike_report(
        spike,
        evidence_refs=["/tmp/password-do-not-leak.log"],
        secrets=("password-do-not-leak",),
    )

    assert payload["report_id"] == "galvatron-1"
    assert payload["status"] == BackendStatus.AVAILABLE.value
    assert "password-do-not-leak" not in json.dumps(payload)
