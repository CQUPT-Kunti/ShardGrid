from __future__ import annotations

import json
from pathlib import Path

import pytest

from shardgrid.common.enums import FailureStage, Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_machine_id, as_worker_id
from shardgrid.jobs.models import EnvironmentSnapshot, FailureRecord
from shardgrid.workers.environment_report import (
    EnvironmentReport,
    ReportScope,
    load_environment_report,
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