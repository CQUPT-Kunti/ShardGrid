from __future__ import annotations

import json
from pathlib import Path

import pytest

from shardgrid.common.config import WorkerConfig
from shardgrid.common.enums import FailureStage, Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_machine_id, as_worker_id
from shardgrid.jobs.models import EnvironmentSnapshot, FailureRecord
from shardgrid.workers.environment_report import (
    EnvironmentReport,
    ReportScope,
    build_worker_report,
    load_environment_report,
    validate_environment_report,
    validate_environment_report_payload,
    write_environment_report,
)


def _worker_report() -> EnvironmentReport:
    return EnvironmentReport(
        report_id="worker-gpu4060",
        scope=ReportScope.WORKER,
        target="gpu4060",
        machine_id=as_machine_id("machine-c"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        timestamp="2026-08-16T00:00:00+00:00",
        health=Health.HEALTHY,
        conda_executable=r"C:\Users\shardgrid\miniconda3\Scripts\conda.exe",
        conda_environment="shardgrid-worker",
        conda_prefix="\\\\wsl.localhost\\Ubuntu-22.04\\home\\shardgrid\\miniconda3\\envs\\shardgrid-worker",
        python_executable="/home/shardgrid/miniconda3/envs/shardgrid-worker/bin/python",
        python_version="3.13.5",
        torch_version="2.5.1",
        torch_cuda_version="12.4",
        cuda_version="12.4",
        runtime_version="Ubuntu-22.04",
        commands=("wsl -d Ubuntu-22.04 /bin/bash -lc 'python -V'",),
        evidence_status="live",
    )


def test_worker_report_contract_preserves_physical_and_runtime_os() -> None:
    report = _worker_report()

    payload = report.to_dict()
    restored = EnvironmentReport.from_dict(payload)

    assert restored == report
    assert json.loads(json.dumps(payload)) == payload
    assert payload["scope"] == "worker"
    assert payload["physical_os"] == "windows"
    assert payload["runtime_os"] == "wsl2_linux"
    assert payload["conda_environment"] == "shardgrid-worker"


def test_control_report_contract_round_trip() -> None:
    report = EnvironmentReport(
        report_id="control-machine-a",
        scope=ReportScope.CONTROL,
        target="control:machine-a",
        machine_id=as_machine_id("machine-a"),
        hostname=as_hostname("control-a.local"),
        physical_os=PhysicalOS.LINUX,
        runtime_os=RuntimeOS.LINUX,
        timestamp="2026-08-16T00:00:00+00:00",
        health=Health.HEALTHY,
        conda_executable="/home/yangjilei/anaconda3/bin/conda",
        conda_environment="base",
        conda_prefix="/home/yangjilei/anaconda3",
        python_executable="/home/yangjilei/anaconda3/bin/python",
        python_version="3.13.5",
        commands=("conda --version",),
        components={
            "ssh": "OpenSSH_9.6p1",
            "git": "git version 2.43.0",
            "iperf3": "iperf 3.17.1",
        },
        evidence_status="live",
    )

    restored = EnvironmentReport.from_dict(report.to_dict())

    assert restored == report
    assert restored.scope == ReportScope.CONTROL
    assert restored.conda_environment == "base"


def test_environment_identity_round_trip() -> None:
    report = _worker_report()

    restored = EnvironmentReport.from_dict(report.to_dict())

    assert restored.conda_executable == report.conda_executable
    assert restored.conda_prefix == report.conda_prefix
    assert restored.python_executable == report.python_executable
    assert restored.python_version == "3.13.5"
    assert restored.torch_version == "2.5.1"
    assert restored.torch_cuda_version == "12.4"
    assert restored.cuda_version == "12.4"


def test_environment_report_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="machine_id"):
        EnvironmentReport.from_dict(
            {
                "report_id": "worker-gpu4060",
                "scope": "worker",
                "target": "gpu4060",
                "hostname": "machine-c.local",
                "physical_os": "windows",
                "runtime_os": "wsl2_linux",
                "timestamp": "2026-08-16T00:00:00+00:00",
                "health": "healthy",
            }
        )


def test_environment_report_records_failure_and_manual_action() -> None:
    failure = FailureRecord(
        stage=FailureStage.PROBE,
        host="machine-c.local",
        worker_id=as_worker_id("gpu4060"),
        command="wsl -d Ubuntu-22.04 /bin/bash -lc 'nvidia-smi'",
        exit_code=1,
        conda_environment="shardgrid-worker",
        message="CUDA not visible from selected Conda environment",
        recommended_action="install NVIDIA driver matching WSL2 requirements, then reboot",
        manual_action_required=True,
    )
    report = EnvironmentReport(
        report_id="worker-gpu4060",
        scope=ReportScope.WORKER,
        target="gpu4060",
        machine_id=as_machine_id("machine-c"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        timestamp="2026-08-16T00:00:00+00:00",
        health=Health.BLOCKED_MANUAL_ACTION,
        failure=failure,
        manual_actions=("reboot the Windows host after installing the NVIDIA driver",),
        evidence_status="pending",
    )

    restored = EnvironmentReport.from_dict(report.to_dict())

    assert restored == report
    assert restored.failure is not None
    assert restored.failure.stage == FailureStage.PROBE
    assert restored.failure.manual_action_required is True
    assert restored.health == Health.BLOCKED_MANUAL_ACTION
    assert restored.manual_actions == report.manual_actions


def test_environment_report_from_snapshot_reuses_existing_evidence() -> None:
    snapshot = EnvironmentSnapshot(
        snapshot_id="env-gpu4060",
        scope="worker:gpu4060",
        conda_executable="/opt/conda/bin/conda",
        conda_environment="shardgrid-worker",
        conda_prefix="/opt/conda/envs/shardgrid-worker",
        python_executable="/opt/conda/envs/shardgrid-worker/bin/python",
        python_version="3.13.5",
        torch_version="2.5.1",
        torch_cuda_version="12.4",
        cuda_version="12.4",
    )

    report = EnvironmentReport.from_snapshot(
        snapshot,
        scope=ReportScope.WORKER,
        target="gpu4060",
        machine_id=as_machine_id("machine-c"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        health=Health.HEALTHY,
        commands=("conda list", "python -V"),
    )

    assert report.conda_environment == "shardgrid-worker"
    assert report.torch_version == "2.5.1"
    assert report.evidence_status == "pending"
    assert report.components["snapshot_id"] == "env-gpu4060"


def test_environment_report_persists_and_loads(tmp_path: Path) -> None:
    report = _worker_report()

    written = write_environment_report(report, tmp_path)
    loaded = load_environment_report(written)

    assert written.exists()
    assert json.loads(written.read_text()) == report.to_dict()
    assert loaded == report


def test_environment_report_rejects_invalid_evidence_status() -> None:
    with pytest.raises(ValueError, match="evidence_status"):
        EnvironmentReport(
            report_id="control-machine-a",
            scope=ReportScope.CONTROL,
            target="control:machine-a",
            machine_id=as_machine_id("machine-a"),
            hostname=as_hostname("control-a.local"),
            physical_os=PhysicalOS.LINUX,
            runtime_os=RuntimeOS.LINUX,
            timestamp="2026-08-16T00:00:00+00:00",
            health=Health.HEALTHY,
            evidence_status="fabricated",
        )


def _worker_config(**overrides: str) -> WorkerConfig:
    payload: dict[str, str] = {
        "id": "gpu4060",
        "machine_id": "machine-c",
        "physical_os": "windows",
        "runtime_os": "wsl2_linux",
        "runtime": "wsl2",
        "host": "machine-c.local",
        "ssh_user": "shardgrid",
        "runtime_distro": "Ubuntu",
    }
    payload.update(overrides)
    return WorkerConfig.from_dict(payload)


def test_environment_report_pass_status_validates_against_schema() -> None:
    report = _worker_report()
    report = EnvironmentReport(
        report_id=report.report_id,
        scope=report.scope,
        target=report.target,
        machine_id=report.machine_id,
        hostname=report.hostname,
        physical_os=report.physical_os,
        runtime_os=report.runtime_os,
        timestamp=report.timestamp,
        health=Health.HEALTHY,
        conda_executable=report.conda_executable,
        conda_environment=report.conda_environment,
        conda_prefix=report.conda_prefix,
        python_executable=report.python_executable,
        python_version=report.python_version,
        torch_version=report.torch_version,
        torch_cuda_version=report.torch_cuda_version,
        cuda_version=report.cuda_version,
        runtime_version=report.runtime_version,
        gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        driver_version="566.07",
        gpu_total_memory_mb=8192,
        compute_capability="8.9",
        evidence_path="/tmp/rtx4060-latest.json",
        commands=report.commands,
        components={"hardware_findings_doc": "docs/operations/hardware-findings.md"},
        evidence_status="live",
    )

    validate_environment_report(report)
    assert EnvironmentReport.from_dict(report.to_dict()) == report


def test_environment_report_blocked_pending_status_validates_against_schema() -> None:
    failure = FailureRecord(
        stage=FailureStage.PROBE,
        host="10.87.5.15",
        worker_id=as_worker_id("gpu1650"),
        message="SSH public-key authentication denied",
        recommended_action="authorize Machine A public key on the Worker",
        manual_action_required=True,
    )
    report = build_worker_report(
        _worker_config(id="gpu1650", host="10.87.5.15"),
        distro="Ubuntu",
        health=Health.BLOCKED_MANUAL_ACTION,
        evidence_status="pending",
        runtime_facts={
            "conda_executable": "/home/shardgrid/miniconda3/bin/conda",
            "conda_environment": "shardgrid",
            "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
            "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            "python_version": "3.12.13",
            "torch_version": "2.7.1+cu118",
            "torch_cuda_version": "11.8",
            "cuda_version": "11.8",
            "gpu_name": "NVIDIA GeForce GTX 1650",
            "driver_version": "527.41",
        },
        commands=("wsl.exe -d Ubuntu -u shardgrid -- bash -lc 'python -V'",),
        manual_actions=(
            "authorize Machine A public key for shardgrid on 10.87.5.15, "
            "then re-run the smoke test",
        ),
        failure=failure,
        components={"runtime_facts_source": "docs/wsl-worker.md (T030 real verification)"},
    )

    validate_environment_report(report)
    restored = EnvironmentReport.from_dict(report.to_dict())

    assert restored == report
    assert restored.health == Health.BLOCKED_MANUAL_ACTION
    assert restored.evidence_status == "pending"
    assert restored.failure is not None
    assert restored.failure.manual_action_required is True
    assert restored.physical_os == PhysicalOS.WINDOWS
    assert restored.runtime_os == RuntimeOS.WSL2_LINUX
    assert restored.runtime_version == "Ubuntu"
    assert restored.python_executable == "/home/shardgrid/miniconda3/envs/shardgrid/bin/python"
    assert restored.torch_version == "2.7.1+cu118"


def test_environment_report_schema_rejects_missing_required_field() -> None:
    payload = _worker_report().to_dict()
    del payload["machine_id"]

    with pytest.raises(Exception, match="machine_id"):
        validate_environment_report_payload(payload)


def test_environment_report_schema_rejects_invalid_health_and_evidence_status() -> None:
    from shardgrid.common.serialization import SchemaValidationError

    payload = _worker_report().to_dict()
    payload["health"] = "fabricated"
    with pytest.raises(SchemaValidationError):
        validate_environment_report_payload(payload)

    payload = _worker_report().to_dict()
    payload["evidence_status"] = "fabricated"
    with pytest.raises(SchemaValidationError):
        validate_environment_report_payload(payload)