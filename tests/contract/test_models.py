from __future__ import annotations

import pytest

from shardgrid.common.enums import FailureStage, Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_worker_id,
)
from shardgrid.jobs.models import FailureRecord, JobStatus
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


def test_worker_resource_contract_round_trip() -> None:
    resource = WorkerResource(
        worker_id=as_worker_id("gpu4060"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        ip="192.168.1.30",
        gpu_name="RTX 4060",
        gpu_total_memory=8192,
        network_interface="eth0",
        health=Health.HEALTHY,
    )

    payload = resource.to_dict()

    assert WorkerResource.from_dict(payload) == resource
    assert payload["physical_os"] == "windows"
    assert payload["runtime_os"] == "wsl2_linux"


def test_network_state_contract_round_trip() -> None:
    state = NetworkState(
        network_id="lan-a",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id("gpu4060"),
                target_worker_id=as_worker_id("gpu1060"),
                source_ip="192.168.1.30",
                target_ip="192.168.1.31",
                interface="eth0",
                tcp_reachable=True,
                latency_ms=0.8,
                bandwidth_mbps=940.0,
            )
        ],
        selected_interfaces={"gpu4060": "eth0", "gpu1060": "eth0"},
    )

    assert NetworkState.from_dict(state.to_dict()) == state


def test_execution_plan_contract_round_trip_preserves_backend_labels() -> None:
    plan = ExecutionPlan(
        job_id=as_job_id("job-0001"),
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("nccl"),
        world_size=2,
        master=MasterMetadata(address="192.168.1.30", port=29500),
        workers=[
            WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0),
            WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1),
        ],
        labels={"backend_label": "nccl", "fallback": "false"},
    )

    payload = plan.to_dict()
    restored = ExecutionPlan.from_dict(payload)

    assert restored == plan
    assert restored.labels["backend_label"] == "nccl"


def test_execution_plan_contract_rejects_invalid_states() -> None:
    with pytest.raises(ValueError, match="duplicate ranks"):
        ExecutionPlan(
            job_id=as_job_id("job-0001"),
            engine=as_engine_name("torchrun"),
            backend=as_backend_name("nccl"),
            world_size=2,
            master=MasterMetadata(address="192.168.1.30", port=29500),
            workers=[
                WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0),
                WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=0),
            ],
        )

    with pytest.raises(ValueError, match="local_rank = 0"):
        ExecutionPlan(
            job_id=as_job_id("job-0001"),
            engine=as_engine_name("torchrun"),
            backend=as_backend_name("nccl"),
            world_size=1,
            master=MasterMetadata(address="192.168.1.30", port=29500),
            workers=[
                WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, local_rank=1)
            ],
        )


def test_job_status_contract_round_trip_and_lifecycle() -> None:
    status = JobStatus(
        job_id=as_job_id("job-0001"),
        state=JobState.CREATED,
        phase="created",
        backend=as_backend_name("nccl"),
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
    )

    status = status.transition_to(JobState.PROBING, phase="probe")
    status = status.transition_to(JobState.PLANNING, phase="plan")
    restored = JobStatus.from_dict(status.to_dict())

    assert restored == status
    assert restored.backend == as_backend_name("nccl")


def test_job_status_contract_rejects_illegal_failure_and_completion_states() -> None:
    with pytest.raises(ValueError, match="failure record"):
        JobStatus(job_id=as_job_id("job-0001"), state=JobState.FAILED, phase="launch")

    with pytest.raises(ValueError, match="checkpoint_ref"):
        JobStatus(job_id=as_job_id("job-0001"), state=JobState.COMPLETED, phase="done")

    status = JobStatus(job_id=as_job_id("job-0001"), state=JobState.CREATED, phase="created")
    with pytest.raises(ValueError, match="invalid job status transition"):
        status.transition_to(JobState.COMPLETED, checkpoint_ref="/tmp/ckpt")


def test_job_status_contract_failure_record_round_trip() -> None:
    failure = FailureRecord(
        stage=FailureStage.RENDEZVOUS,
        host="machine-c.local",
        worker_id=as_worker_id("gpu4060"),
        command="torchrun --rdzv-backend c10d",
        exit_code=1,
        stdout_path="/tmp/stdout.log",
        stderr_path="/tmp/stderr.log",
        message="rendezvous failed",
        recommended_action="check master address and port",
        retryable=True,
    )
    status = JobStatus(
        job_id=as_job_id("job-0001"),
        state=JobState.FAILED,
        phase="rendezvous",
        failure=failure,
    )

    restored = JobStatus.from_dict(status.to_dict())

    assert restored.failure == failure
