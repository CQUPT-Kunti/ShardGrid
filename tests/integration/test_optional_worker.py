from __future__ import annotations

import json
from pathlib import Path

import yaml

from shardgrid.cli.app import main
from shardgrid.common.config import load_cluster_config
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.resources.models import WorkerResource
from shardgrid.workers.inventory import load_inventory
from shardgrid.workers.models import WorkerRuntime


def _write_config(tmp_path: Path, *, enabled: bool) -> Path:
    path = tmp_path / "workers.yaml"
    path.write_text(
        f"""
control:
  machine_id: machine-a
  hostname: control-a.local
jobs_root: /tmp/shardgrid/jobs
ssh: {{}}
runtime:
  conda_environment: shardgrid
  conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
network: {{}}
backend_preference: {{}}
manual_override:
  preferred_workers: []
  disabled_workers: []
  worker_address_overrides: {{}}
  rendezvous_port: null
workers:
  - id: gpu4060
    machine_id: machine-c
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.87.5.155
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
    local_world_size: 1
    enabled: true
    labels:
      gpu: rtx4060
  - id: gpu1060
    machine_id: machine-d
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.87.5.15
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
    local_world_size: 1
    enabled: true
    labels:
      gpu: gtx1650
  - id: gpu4060-cqupt
    machine_id: machine-e
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.87.5.214
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
    local_world_size: 1
    enabled: {str(enabled).lower()}
    labels:
      gpu: rtx4060
      host_identity: CQUPT
      optional: "true"
""".strip(),
        encoding="utf-8",
    )
    return path


def _runtime(worker_id: str, python_version: str = "Python 3.12.14") -> WorkerRuntime:
    return WorkerRuntime(
        worker_id=worker_id,
        runtime_os=RuntimeOS.WSL2_LINUX,
        runtime_version="Ubuntu-22.04",
        environment_manager="conda",
        conda_executable="/home/shardgrid/miniconda3/bin/conda",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        conda_active=True,
        python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        python_version=python_version,
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        cuda_available=True,
        nccl_available=True,
        gloo_available=True,
        path_style="posix",
        health=Health.HEALTHY,
    )


def _resource(
    worker_id: str,
    host: str,
    *,
    gpu_name: str,
    total_memory: int,
    free_memory: int,
    compute_capability: str,
    driver_version: str,
) -> WorkerResource:
    return WorkerResource(
        worker_id=worker_id,
        hostname=host,
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        ip=host,
        gpu_name=gpu_name,
        gpu_total_memory=total_memory,
        gpu_free_memory=free_memory,
        compute_capability=compute_capability,
        driver_version=driver_version,
        cuda_version="11.8",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        nccl_available=True,
        gloo_available=True,
        network_interface="eth0",
        health=Health.HEALTHY,
        last_probe_at="2026-09-02T04:26:12.934874+00:00",
    )


def _refresh_payload(worker_id: str) -> tuple[WorkerResource, WorkerRuntime, str, None, None, str, dict[str, str]]:
    if worker_id == "gpu4060":
        host = "10.87.5.155"
        return (
            _resource(
                worker_id,
                host,
                gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
                total_memory=8188,
                free_memory=7867,
                compute_capability="8.9",
                driver_version="566.07",
            ),
            _runtime(worker_id, "Python 3.12.13"),
            "REACHABLE",
            None,
            None,
            "LDJ",
            {
                "wsl_distro": "Ubuntu-22.04",
                "conda_executable": "/home/shardgrid/miniconda3/bin/conda",
                "conda_environment": "shardgrid",
                "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
                "python_version": "Python 3.12.13",
            },
        )
    if worker_id == "gpu1060":
        host = "10.87.5.15"
        return (
            _resource(
                worker_id,
                host,
                gpu_name="NVIDIA GeForce GTX 1650",
                total_memory=4096,
                free_memory=3403,
                compute_capability="7.5",
                driver_version="527.41",
            ),
            _runtime(worker_id, "Python 3.12.13"),
            "REACHABLE",
            None,
            None,
            "LAPTOP-5G3QUOGM",
            {
                "wsl_distro": "Ubuntu-22.04",
                "conda_executable": "/home/shardgrid/miniconda3/bin/conda",
                "conda_environment": "shardgrid",
                "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
                "python_version": "Python 3.12.13",
            },
        )
    host = "10.87.5.214"
    return (
        _resource(
            worker_id,
            host,
            gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            total_memory=8188,
            free_memory=7216,
            compute_capability="8.9",
            driver_version="566.07",
        ),
        _runtime(worker_id),
        "REACHABLE",
        None,
        None,
        "CQUPT",
        {
            "wsl_distro": "Ubuntu-22.04",
            "conda_executable": "/home/shardgrid/miniconda3/bin/conda",
            "conda_environment": "shardgrid",
            "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
            "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            "python_version": "Python 3.12.14",
        },
    )


def _cache_payload(cache_path: Path) -> list[WorkerResource]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return [WorkerResource.from_dict(item) for item in payload["resources"]]


def test_optional_worker_is_discoverable_eligible_and_disableable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    enabled_config = _write_config(tmp_path, enabled=True)
    cache_path = tmp_path / "inventory.json"
    monkeypatch.setattr("shardgrid.cli.commands.workers.DEFAULT_CACHE_PATH", cache_path)
    monkeypatch.setattr(
        "shardgrid.cli.commands.workers._refresh_worker",
        lambda _config, worker: _refresh_payload(str(worker.worker_id)),
    )

    exit_code = main(["--json", "--config", str(enabled_config), "workers", "--refresh"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["worker_count"] == 3
    enabled_entry = next(
        item for item in payload["workers"] if item["worker"]["worker_id"] == "gpu4060-cqupt"
    )
    assert enabled_entry["host_identity"] == "CQUPT"
    assert enabled_entry["worker"]["host"] == "10.87.5.214"
    assert enabled_entry["worker"]["runtime_os"] == "wsl2_linux"
    assert enabled_entry["runtime_identity"]["conda_prefix"] == "/home/shardgrid/miniconda3/envs/shardgrid"
    assert enabled_entry["resource"]["gpu_name"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert enabled_entry["health"] == "HEALTHY"
    assert enabled_entry["eligible"] is True

    cluster_enabled = ResourceManager().build_cluster_state(_cache_payload(cache_path))
    assert [entry.worker_id for entry in cluster_enabled.eligible_workers] == [
        "gpu1060",
        "gpu4060",
        "gpu4060-cqupt",
    ]

    disabled_config = _write_config(tmp_path, enabled=False)
    disabled_inventory = load_inventory(load_cluster_config(disabled_config))
    assert disabled_inventory.identify_worker("gpu4060-cqupt").enabled is False

    exit_code = main(["--json", "--config", str(disabled_config), "workers", "--refresh"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    disabled_entry = next(
        item for item in payload["workers"] if item["worker"]["worker_id"] == "gpu4060-cqupt"
    )
    assert disabled_entry["health"] == "UNAVAILABLE"
    assert disabled_entry["eligible"] is False
    assert disabled_entry["worker"]["enabled"] is False

    cluster_disabled = ResourceManager().build_cluster_state(_cache_payload(cache_path))
    assert [entry.worker_id for entry in cluster_disabled.eligible_workers] == [
        "gpu1060",
        "gpu4060",
        "gpu4060-cqupt",
    ]
    asserted = [
        str(worker.worker_id)
        for worker in disabled_inventory.enabled_workers()
    ]
    assert asserted == ["gpu4060", "gpu1060"]

