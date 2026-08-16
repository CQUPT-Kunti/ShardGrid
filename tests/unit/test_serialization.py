from __future__ import annotations

from pathlib import Path

import pytest

from shardgrid.common.enums import FailureStage, Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_worker_id,
)
from shardgrid.common.serialization import (
    SchemaValidationError,
    deserialize_json,
    deserialize_yaml,
    dump_json,
    dump_yaml,
    load_json,
    load_yaml,
    serialize_json,
    serialize_yaml,
    validate_execution_plan,
    validate_job_status,
    validate_schema_data,
)
from shardgrid.jobs.models import FailureRecord, JobStatus
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


def make_worker_resource() -> WorkerResource:
    return WorkerResource(
        worker_id=as_worker_id("gpu4060"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        ip="10.0.0.13",
        gpu_name="RTX 4060",
        gpu_total_memory=8188,
        gpu_free_memory=7680,
        gpu_utilization=12.5,
        compute_capability="8.9",
        driver_version="555.85",
        cuda_version="12.4",
        torch_version="2.5.1",
        torch_cuda_version="12.4",
        nccl_available=True,
        gloo_available=True,
        network_interface="eth0",
        network_bandwidth=940.5,
        network_latency=1.8,
        health=Health.HEALTHY,
        last_probe_at="2026-08-16T09:30:00Z",
    )


def make_network_state() -> NetworkState:
    return NetworkState(
        network_id="mvp-pair",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id("gpu4060"),
                target_worker_id=as_worker_id("gpu1060"),
                source_ip="10.0.0.13",
                target_ip="10.0.0.14",
                interface="eth0",
                tcp_reachable=True,
                latency_ms=2.1,
                bandwidth_mbps=930.0,
            )
        ],
        selected_interfaces={"gpu4060": "eth0", "gpu1060": "eth0"},
    )


def make_execution_plan() -> ExecutionPlan:
    return ExecutionPlan(
        job_id=as_job_id("job-0001"),
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("ssh"),
        world_size=2,
        master=MasterMetadata(address="10.0.0.13", port=29500),
        workers=[
            WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, stage="0"),
            WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1, stage="1"),
        ],
        labels={"backend_result": "gloo_fallback"},
    )


def make_job_status() -> JobStatus:
    return JobStatus(
        job_id=as_job_id("job-0001"),
        state=JobState.FAILED,
        phase="launch",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        backend=as_backend_name("ssh"),
        fallback_used=True,
        failure=FailureRecord(
            stage=FailureStage.LAUNCH,
            host="machine-c.local",
            worker_id=as_worker_id("gpu4060"),
            message="ssh launch failed",
            recommended_action="inspect remote logs and ssh access",
            retryable=True,
        ),
    )


def test_worker_resource_json_yaml_round_trip(tmp_path: Path) -> None:
    resource = make_worker_resource()

    json_text = serialize_json(resource)
    yaml_text = serialize_yaml(resource)

    assert deserialize_json(json_text, WorkerResource) == resource
    assert deserialize_yaml(yaml_text, WorkerResource) == resource

    json_path = dump_json(resource, tmp_path / "worker-resource.json")
    yaml_path = dump_yaml(resource, tmp_path / "worker-resource.yaml")

    assert load_json(json_path, WorkerResource) == resource
    assert load_yaml(yaml_path, WorkerResource) == resource


def test_network_state_json_yaml_round_trip() -> None:
    state = make_network_state()

    assert deserialize_json(serialize_json(state), NetworkState) == state
    assert deserialize_yaml(serialize_yaml(state), NetworkState) == state


def test_execution_plan_and_job_status_schema_validation_round_trip() -> None:
    plan = make_execution_plan()
    status = make_job_status()

    validate_execution_plan(plan)
    validate_job_status(status)

    assert deserialize_json(serialize_json(plan), ExecutionPlan) == plan
    assert deserialize_yaml(serialize_yaml(status), JobStatus) == status


def test_execution_plan_schema_rejects_duplicate_rank() -> None:
    payload = make_execution_plan().to_dict()
    payload["workers"][1]["rank"] = 0

    with pytest.raises(SchemaValidationError):
        validate_schema_data("execution_plan", payload)


def test_execution_plan_schema_rejects_invalid_local_rank() -> None:
    payload = make_execution_plan().to_dict()
    payload["workers"][0]["local_rank"] = 1

    with pytest.raises(SchemaValidationError):
        validate_schema_data("execution_plan", payload)


def test_execution_plan_schema_rejects_missing_master_metadata() -> None:
    payload = make_execution_plan().to_dict()
    payload.pop("master")

    with pytest.raises(SchemaValidationError):
        validate_schema_data("execution_plan", payload)


def test_job_status_schema_rejects_incomplete_failure_record() -> None:
    payload = make_job_status().to_dict()
    payload["failure"].pop("recommended_action")

    with pytest.raises(SchemaValidationError):
        validate_schema_data("job_status", payload)
