from __future__ import annotations

from pathlib import Path

from shardgrid.artifacts.store import ArtifactStore
from shardgrid.cli.app import main
from shardgrid.common.config import ClusterConfig
from shardgrid.common.enums import FailureStage, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.manual_actions import classify_manual_action
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_worker_id,
)
from shardgrid.common.serialization import dump_json, dump_yaml, validate_schema_data
from shardgrid.jobs.models import EnvironmentSnapshot, FailureRecord, JobStatus
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource
from shardgrid.workers.inventory import load_inventory


def test_foundation_gate_smoke(tmp_path: Path) -> None:
    config_payload = {
        "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
        "jobs_root": str(tmp_path / "jobs"),
        "ssh": {},
        "runtime": {},
        "network": {},
        "backend_preference": {},
        "manual_override": {
            "preferred_workers": ["gpu4060", "gpu1060"],
            "disabled_workers": [],
            "worker_address_overrides": {},
        },
        "workers": [
            {
                "id": "gpu4060",
                "machine_id": "machine-c",
                "physical_os": "windows",
                "runtime_os": "wsl2_linux",
                "runtime": "wsl2",
                "host": "machine-c.local",
                "ssh_user": "shardgrid",
                "local_world_size": 1,
            },
            {
                "id": "gpu1060",
                "machine_id": "machine-d",
                "physical_os": "windows",
                "runtime_os": "wsl2_linux",
                "runtime": "wsl2",
                "host": "machine-d.local",
                "ssh_user": "shardgrid",
                "local_world_size": 1,
            },
        ],
    }
    config = ClusterConfig.from_dict(config_payload)
    inventory = load_inventory(config)
    snapshot = ArtifactStore(config.jobs_root).snapshot_paths("job-0001").create()

    worker_resource = WorkerResource(
        worker_id=as_worker_id("gpu4060"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment="shardgrid-worker",
        python_executable="python",
        ip="192.168.1.30",
    )
    network_state = NetworkState(
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
            )
        ],
    )
    plan = ExecutionPlan(
        job_id=as_job_id("job-0001"),
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("nccl"),
        world_size=2,
        master=MasterMetadata(address="192.168.1.30", port=29500),
        workers=[
            WorkerAssignment(
                worker_id=as_worker_id("gpu4060"),
                rank=0,
                conda_environment="shardgrid-worker",
            ),
            WorkerAssignment(
                worker_id=as_worker_id("gpu1060"),
                rank=1,
                conda_environment="shardgrid-worker",
            ),
        ],
        conda_environment="shardgrid-worker",
        python_executable="python",
        snapshot_ref=str(snapshot.root),
    )
    failure = FailureRecord(
        stage=FailureStage.BOOTSTRAP,
        host="machine-a.local",
        message="manual action required: reboot required after package install",
        recommended_action="have an operator schedule and confirm the reboot before continuing",
        conda_environment="shardgrid-dev",
        manual_action_required=True,
    )
    status = JobStatus(
        job_id=as_job_id("job-0001"),
        state=JobState.FAILED,
        phase="bootstrap",
        failure=failure,
    )
    environment_snapshot = EnvironmentSnapshot(
        snapshot_id="env-job-0001",
        scope="job:job-0001",
        conda_environment="shardgrid-worker",
        python_executable="python",
        torch_version="2.5.1",
        cuda_version="12.4",
    )

    validate_schema_data("execution_plan", plan.to_dict())
    validate_schema_data("job_status", status.to_dict())

    assert len(inventory.enabled_workers()) == 2
    assert worker_resource.to_dict()["runtime_os"] == "wsl2_linux"
    assert worker_resource.to_dict()["conda_environment"] == "shardgrid-worker"
    assert NetworkState.from_dict(network_state.to_dict()) == network_state
    assert dump_json(plan, tmp_path / "execution-plan.json").is_file() is True
    assert dump_yaml(status, tmp_path / "job-status.yaml").is_file() is True
    assert environment_snapshot.to_dict()["environment_manager"] == "conda"
    assert snapshot.logs.is_dir() is True
    assert classify_manual_action("reboot required after package install") is not None
    assert main(["--json"]) == 0
