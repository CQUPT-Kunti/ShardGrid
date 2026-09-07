from __future__ import annotations

import json
from pathlib import Path

from shardgrid.common.enums import Health, MachineRole, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_machine_id, as_worker_id
from shardgrid.resources.models import GPUResource, NetworkLink, NetworkState, WorkerResource
from shardgrid.workers.models import ControlNode, Machine, Worker, WorkerRuntime


def test_worker_resource_round_trip_json() -> None:
    resource = WorkerResource(
        worker_id=as_worker_id("gpu4060"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        environment_manager="conda",
        conda_environment="shardgrid-worker",
        conda_prefix="/opt/conda/envs/shardgrid-worker",
        python_executable="python",
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

    payload = resource.to_dict()
    restored = WorkerResource.from_dict(json.loads(json.dumps(payload)))

    assert restored == resource
    assert restored.physical_os is PhysicalOS.WINDOWS
    assert restored.runtime_os is RuntimeOS.WSL2_LINUX
    assert restored.conda_environment == "shardgrid-worker"


def test_worker_defaults_to_one_gpu_per_physical_host() -> None:
    worker = Worker(
        worker_id=as_worker_id("gpu1060"),
        machine_id=as_machine_id("machine-d"),
        hostname=as_hostname("machine-d.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        host="machine-d.local",
        ssh_user_ref="shardgrid",
        runtime="wsl2",
        conda_environment="shardgrid-worker",
    )

    assert worker.local_world_size == 1
    assert worker.enabled is True
    assert worker.conda_environment == "shardgrid-worker"


def test_windows_and_wsl_runtime_os_remain_separate() -> None:
    worker = Worker(
        worker_id=as_worker_id("gpu4060"),
        machine_id=as_machine_id("machine-c"),
        hostname=as_hostname("machine-c.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        host="machine-c.local",
        ssh_user_ref="shardgrid",
        runtime="wsl2",
        runtime_distro="Ubuntu-22.04",
    )
    runtime = WorkerRuntime(
        worker_id=as_worker_id("gpu4060"),
        runtime_os=RuntimeOS.WSL2_LINUX,
        environment_manager="conda",
        conda_executable="/opt/conda/bin/conda",
        conda_environment="shardgrid-worker",
        conda_prefix="/opt/conda/envs/shardgrid-worker",
        conda_active=True,
        python_executable="python",
        python_version="3.13.5",
        torch_version="2.5.1",
        cuda_available=True,
        gloo_available=True,
        health=Health.HEALTHY,
    )

    assert worker.physical_os is PhysicalOS.WINDOWS
    assert worker.runtime_os is RuntimeOS.WSL2_LINUX
    assert runtime.runtime_os is RuntimeOS.WSL2_LINUX
    assert runtime.conda_environment == "shardgrid-worker"
    assert runtime.conda_active is True


def test_models_cover_control_gpu_and_network_records() -> None:
    machine = Machine(
        machine_id=as_machine_id("machine-a"),
        role=MachineRole.CONTROL,
        physical_os=PhysicalOS.LINUX,
        hostname=as_hostname("control-a.local"),
        configured_host="control-a.local",
        required_for_mvp=True,
    )
    control = ControlNode(
        machine_id=as_machine_id("machine-a"),
        hostname=as_hostname("control-a.local"),
        os_version="Ubuntu 24.04",
        python_version="3.13.5",
        ssh_available=True,
        git_available=True,
        iperf3_available=True,
        jobs_root=Path("/var/tmp/shardgrid/jobs"),
        disk_free_bytes=10_000_000,
        health=Health.HEALTHY,
        conda_environment="shardgrid-dev",
    )
    gpu = GPUResource(
        worker_id=as_worker_id("gpu4060"),
        gpu_name="RTX 4060",
        total_memory_mb=8188,
        free_memory_mb=7900,
        compute_capability="8.9",
        health=Health.HEALTHY,
    )
    link = NetworkLink(
        source_worker_id=as_worker_id("gpu4060"),
        target_worker_id=as_worker_id("gpu1060"),
        source_ip="10.0.0.13",
        target_ip="10.0.0.14",
        interface="eth0",
        tcp_reachable=True,
        latency_ms=2.1,
        bandwidth_mbps=930.0,
    )
    state = NetworkState(
        network_id="mvp-pair",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        links=[link],
        selected_interfaces={"gpu4060": "eth0", "gpu1060": "eth0"},
    )

    assert machine.role is MachineRole.CONTROL
    assert control.health is Health.HEALTHY
    assert control.conda_environment == "shardgrid-dev"
    assert gpu.gpu_index == 0
    assert state.links[0].tcp_reachable is True
    assert NetworkState.from_dict(state.to_dict()) == state
