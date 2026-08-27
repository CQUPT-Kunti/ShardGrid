from __future__ import annotations

import json
from pathlib import Path

import pytest

from shardgrid.artifacts.metadata import (
    SnapshotMetadata,
    load_snapshot_metadata,
    validate_snapshot_metadata,
    write_snapshot_metadata,
)
from shardgrid.artifacts.store import ArtifactStore
from shardgrid.common.enums import FailureStage, Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_machine_id,
    as_worker_id,
)
from shardgrid.control.job_manager import create_training_job
from shardgrid.engines.models import ParallelPlan
from shardgrid.jobs.models import FailureRecord, JobStatus
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState
from shardgrid.workers.environment_report import EnvironmentReport, ReportScope

_SECRET = "TEST_PASSWORD_DO_NOT_LEAK"


def _snapshot(tmp_path: Path):
    store = ArtifactStore((tmp_path / "jobs").resolve())
    job = create_training_job(
        job_id=as_job_id("job-001"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    return job, store.create_snapshot(job)


def _parallel_plan() -> ParallelPlan:
    return ParallelPlan(
        parallel_plan_id="plan-1",
        engine=as_engine_name("galvatron"),
        engine_plan_path="/var/tmp/original-plan.json",
        model_name="tiny-sequential",
        world_size=2,
        stages=["stage0", "stage1"],
    )


def _execution_plan(job_id: str = "job-001") -> ExecutionPlan:
    return ExecutionPlan(
        job_id=as_job_id(job_id),
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("ssh"),
        world_size=2,
        master=MasterMetadata(address="10.0.0.1", port=29500),
        workers=[
            WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, stage="0"),
            WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1, stage="1"),
        ],
        conda_environment="shardgrid",
        python_executable="python",
        parallel_plan_ref="/var/tmp/original-plan.json",
    )


def _environment_report() -> EnvironmentReport:
    return EnvironmentReport(
        report_id="worker-gpu4060",
        scope=ReportScope.WORKER,
        target="gpu4060",
        machine_id=as_machine_id("machine-c"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        timestamp="2026-08-27T00:00:00+00:00",
        health=Health.HEALTHY,
        conda_executable="/opt/conda/bin/conda",
        conda_environment="shardgrid",
        conda_prefix="/opt/conda/envs/shardgrid",
        python_executable="/opt/conda/envs/shardgrid/bin/python",
        python_version="3.12.13",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        cuda_version="11.8",
        gpu_name="NVIDIA GeForce GTX 1650",
        components={"nccl": "2.21.5", "backend": "nccl"},
        evidence_status="live",
    )


def _network_state() -> NetworkState:
    return NetworkState(
        network_id="lan-a",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id("gpu4060"),
                target_worker_id=as_worker_id("gpu1060"),
                source_ip="10.0.0.1",
                target_ip="10.0.0.2",
                interface="eth0",
                tcp_reachable=True,
            ),
            NetworkLink(
                source_worker_id=as_worker_id("gpu1060"),
                target_worker_id=as_worker_id("gpu4060"),
                source_ip="10.0.0.2",
                target_ip="10.0.0.1",
                interface="eth0",
                tcp_reachable=True,
            ),
        ],
        diagnostics_path="/var/tmp/shardgrid/network/latest.json",
    )


def _completed_status() -> JobStatus:
    return JobStatus(
        job_id=as_job_id("job-001"),
        state=JobState.COMPLETED,
        phase="checkpoint",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        backend=as_backend_name("ssh"),
        checkpoint_ref="jobs/job-001/checkpoint/model.pt",
        final_metrics={"final_loss": 0.5},
        finished_at="2026-08-27T00:10:00+00:00",
    )


def _failed_status() -> JobStatus:
    return JobStatus(
        job_id=as_job_id("job-001"),
        state=JobState.FAILED,
        phase="launch",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        backend=as_backend_name("ssh"),
        failure=FailureRecord(
            stage=FailureStage.LAUNCH,
            host="machine-c.local",
            command=f"ssh --password {_SECRET}",
            message=f"failed to launch with {_SECRET}",
            recommended_action="inspect launch logs",
        ),
    )


def test_success_snapshot_metadata_round_trip_and_files(tmp_path: Path) -> None:
    job, snapshot = _snapshot(tmp_path)
    metadata = write_snapshot_metadata(
        snapshot=snapshot,
        job=job,
        config={
            "job": {"name": "train-minimal", "backend": "ssh"},
            "runtime_selection": {"selected_environment": "shardgrid"},
        },
        parallel_plan=_parallel_plan(),
        execution_plan=_execution_plan(),
        environment_report=_environment_report(),
        network_state=_network_state(),
        job_status=_completed_status(),
        checkpoint_metadata={"step": 20, "rank": 1, "status": "complete"},
    )

    loaded = load_snapshot_metadata(Path(snapshot.diagnostics_path) / "snapshot-metadata.json")

    assert isinstance(metadata, SnapshotMetadata)
    assert loaded == metadata
    assert json.loads(Path(metadata.config_path).read_text())["job"]["backend"] == "ssh"
    assert (
        json.loads(Path(metadata.original_parallel_plan_path).read_text())[
            "parallel_plan_id"
        ]
        == "plan-1"
    )
    assert json.loads(Path(metadata.execution_plan_path).read_text())["job_id"] == "job-001"
    assert (
        json.loads(Path(metadata.network_state_path).read_text())["diagnostics_path"]
        == "/var/tmp/shardgrid/network/latest.json"
    )
    assert json.loads(Path(metadata.checkpoint_metadata_path).read_text())["step"] == 20


def test_failed_snapshot_metadata_keeps_failure_evidence(tmp_path: Path) -> None:
    job, snapshot = _snapshot(tmp_path)
    metadata = write_snapshot_metadata(
        snapshot=snapshot,
        job=job,
        config={"job": {"name": "train-minimal"}},
        parallel_plan=_parallel_plan(),
        execution_plan=_execution_plan(),
        environment_report=_environment_report(),
        network_state=_network_state(),
        job_status=_failed_status(),
        secrets=[_SECRET],
    )

    assert metadata.failure_path is not None
    assert Path(metadata.failure_path).is_file() is True
    assert _SECRET not in Path(metadata.failure_path).read_text()
    assert _SECRET not in Path(metadata.job_status_path).read_text()


def test_secret_values_are_redacted_from_metadata_outputs(tmp_path: Path) -> None:
    job, snapshot = _snapshot(tmp_path)
    metadata = write_snapshot_metadata(
        snapshot=snapshot,
        job=job,
        config={"secret": _SECRET, "job": {"name": "train-minimal"}},
        parallel_plan=_parallel_plan(),
        execution_plan=_execution_plan(),
        environment_report=_environment_report(),
        network_state=_network_state(),
        job_status=_failed_status(),
        secrets=[_SECRET],
    )

    blob = "\n".join(
        [
            Path(metadata.config_path).read_text(),
            Path(metadata.job_status_path).read_text(),
            Path(metadata.failure_path or "").read_text(),
        ]
    )
    assert _SECRET not in blob


def test_metadata_validation_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        validate_snapshot_metadata({"job_id": "job-001", "snapshot_root": "/tmp/jobs/job-001"})


def test_metadata_validation_rejects_job_mismatch(tmp_path: Path) -> None:
    job, snapshot = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="execution plan job_id"):
        write_snapshot_metadata(
            snapshot=snapshot,
            job=job,
            config={"job": {"name": "train-minimal"}},
            parallel_plan=_parallel_plan(),
            execution_plan=_execution_plan("job-002"),
            environment_report=_environment_report(),
            network_state=_network_state(),
            job_status=_completed_status(),
            checkpoint_metadata={"step": 20, "status": "complete"},
        )


def test_completed_status_requires_checkpoint_metadata(tmp_path: Path) -> None:
    job, snapshot = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="checkpoint metadata"):
        write_snapshot_metadata(
            snapshot=snapshot,
            job=job,
            config={"job": {"name": "train-minimal"}},
            parallel_plan=_parallel_plan(),
            execution_plan=_execution_plan(),
            environment_report=_environment_report(),
            network_state=_network_state(),
            job_status=_completed_status(),
        )


def test_metadata_paths_stay_within_snapshot_root(tmp_path: Path) -> None:
    payload = {
        "job_id": "job-001",
        "snapshot_root": str((tmp_path / "jobs" / "job-001").resolve()),
        "config_path": str((tmp_path / "escape.json").resolve()),
        "original_parallel_plan_path": str((tmp_path / "jobs" / "job-001" / "plan.json").resolve()),
        "execution_plan_path": str((tmp_path / "jobs" / "job-001" / "exec.json").resolve()),
        "environment_report_path": str((tmp_path / "jobs" / "job-001" / "env.json").resolve()),
        "network_state_path": str((tmp_path / "jobs" / "job-001" / "net.json").resolve()),
        "checkpoint_metadata_path": str((tmp_path / "jobs" / "job-001" / "ckpt.json").resolve()),
        "job_status_path": str((tmp_path / "jobs" / "job-001" / "status.json").resolve()),
    }
    with pytest.raises(ValueError, match="escaped snapshot root"):
        validate_snapshot_metadata(payload)
