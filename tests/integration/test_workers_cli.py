from __future__ import annotations

import json
from pathlib import Path

from shardgrid.cli.app import main
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.resources.models import WorkerResource
from shardgrid.workers.models import Worker, WorkerRuntime


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "workers.yaml"
    path.write_text(
        """
control:
  machine_id: machine-a
  hostname: control-a.local
jobs_root: /tmp/shardgrid/jobs
ssh: {}
runtime:
  conda_environment: shardgrid
  conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
network: {}
backend_preference: {}
manual_override:
  preferred_workers: []
  disabled_workers: []
  worker_address_overrides: {}
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
    labels:
      gpu: gtx1650
""".strip(),
        encoding="utf-8",
    )
    return path


def _worker(worker_id: str, host: str) -> Worker:
    return Worker(
        worker_id=worker_id,
        machine_id=f"machine-{worker_id}",
        hostname=host,
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        host=host,
        ssh_user_ref="shardgrid",
        runtime="wsl2",
        runtime_distro="Ubuntu-22.04",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        enabled=True,
        health=Health.UNKNOWN,
    )


def _resource(
    worker_id: str,
    host: str,
    *,
    health: Health,
    gpu: str,
    last_probe_at: str,
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
        gpu_name=gpu,
        gpu_total_memory=8192 if "4060" in gpu else 4096,
        compute_capability="8.9" if "4060" in gpu else "7.5",
        driver_version="566.07",
        cuda_version="11.8",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        nccl_available=True,
        gloo_available=True,
        network_interface="eth3" if host.endswith("155") else "eth0",
        health=health,
        last_probe_at=last_probe_at,
    )


def _runtime(worker_id: str) -> WorkerRuntime:
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
        python_version="Python 3.12.13",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        cuda_available=True,
        nccl_available=True,
        gloo_available=True,
        path_style="posix",
        health=Health.HEALTHY,
    )


def test_workers_refresh_keeps_unhealthy_worker_visible(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    cache = tmp_path / "inventory.json"
    monkeypatch.setattr(
        "shardgrid.cli.commands.workers.DEFAULT_CACHE_PATH",
        cache,
    )

    def fake_refresh(cluster_config, worker_config):
        if str(worker_config.worker_id) == "gpu4060":
            return (
                _resource(
                    "gpu4060",
                    "10.87.5.155",
                    health=Health.HEALTHY,
                    gpu="NVIDIA GeForce RTX 4060 Laptop GPU",
                    last_probe_at="2026-08-27T06:00:00+00:00",
                ),
                _runtime("gpu4060"),
                "REACHABLE",
                None,
                None,
            )
        return (
            _resource(
                "gpu1060",
                "10.87.5.15",
                health=Health.FAILED,
                gpu="NVIDIA GeForce GTX 1650",
                last_probe_at="2026-08-27T06:00:00+00:00",
            ),
            _runtime("gpu1060"),
            "REACHABLE",
            "CUDA is not available from the selected Conda environment",
            {"message": "CUDA is not available from the selected Conda environment"},
        )

    monkeypatch.setattr("shardgrid.cli.commands.workers._refresh_worker", fake_refresh)

    exit_code = main(["--config", str(config), "workers", "--refresh"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Worker: gpu4060" in captured.out
    assert "Worker: gpu1060" in captured.out
    assert "Health: HEALTHY" in captured.out
    assert "Health: UNHEALTHY" in captured.out
    assert "CUDA is not available" in captured.out
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert len(payload["resources"]) == 2


def test_workers_require_healthy_returns_non_zero_but_keeps_full_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    cache = tmp_path / "inventory.json"
    monkeypatch.setattr("shardgrid.cli.commands.workers.DEFAULT_CACHE_PATH", cache)
    cache.write_text(
        json.dumps(
            {
                "resources": [
                    _resource(
                        "gpu4060",
                        "10.87.5.155",
                        health=Health.HEALTHY,
                        gpu="NVIDIA GeForce RTX 4060 Laptop GPU",
                        last_probe_at="2026-08-27T06:00:00+00:00",
                    ).to_dict(),
                    _resource(
                        "gpu1060",
                        "10.87.5.15",
                        health=Health.UNREACHABLE,
                        gpu="NVIDIA GeForce GTX 1650",
                        last_probe_at="2026-08-27T06:00:00+00:00",
                    ).to_dict(),
                ],
                "runtimes": [_runtime("gpu4060").to_dict(), _runtime("gpu1060").to_dict()],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config), "workers", "--require-healthy"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Worker: gpu4060" in captured.out
    assert "Worker: gpu1060" in captured.out
    assert "Health: UNREACHABLE" in captured.out


def test_workers_json_marks_stale_inventory_when_no_refresh(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    cache = tmp_path / "inventory.json"
    monkeypatch.setattr("shardgrid.cli.commands.workers.DEFAULT_CACHE_PATH", cache)
    cache.write_text(
        json.dumps(
            {
                "resources": [
                    _resource(
                        "gpu4060",
                        "10.87.5.155",
                        health=Health.HEALTHY,
                        gpu="NVIDIA GeForce RTX 4060 Laptop GPU",
                        last_probe_at="2026-08-20T06:00:00+00:00",
                    ).to_dict()
                ],
                "runtimes": [_runtime("gpu4060").to_dict()],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--json", "--config", str(config), "workers"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["worker_count"] == 2
    gpu4060 = next(item for item in payload["workers"] if item["worker"]["worker_id"] == "gpu4060")
    gpu1060 = next(item for item in payload["workers"] if item["worker"]["worker_id"] == "gpu1060")
    assert gpu4060["health"] == "STALE"
    assert gpu4060["stale"] is True
    assert gpu1060["health"] == "STALE"
    assert gpu1060["source"] == "config-only"
