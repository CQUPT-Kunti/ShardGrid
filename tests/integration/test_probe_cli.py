from __future__ import annotations

import json
from pathlib import Path

from shardgrid.cli.app import main
from shardgrid.common.enums import FailureStage, Health, PhysicalOS, RuntimeOS
from shardgrid.common.errors import make_failure_record
from shardgrid.jobs.models import FailureRecord
from shardgrid.resources.models import WorkerResource
from shardgrid.workers.models import WorkerRuntime


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


def _resource(
    worker_id: str,
    host: str,
    *,
    gpu: str,
    health: Health = Health.HEALTHY,
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
        gpu_total_memory=8188 if "4060" in gpu else 4096,
        gpu_free_memory=7000 if "4060" in gpu else 3200,
        compute_capability="8.9" if "4060" in gpu else "7.5",
        driver_version="566.07" if "4060" in gpu else "527.41",
        cuda_version="11.8",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        nccl_available=True,
        gloo_available=True,
        health=health,
    )


def _report(worker_id: str, host: str, gpu: str) -> dict[str, object]:
    runtime = _runtime(worker_id)
    resource = _resource(worker_id, host, gpu=gpu)
    return {
        "worker": {
            "worker_id": worker_id,
            "hostname": host,
            "configured_host": host,
            "physical_os": "windows",
            "runtime_os": "wsl2_linux",
            "reachability": "reachable",
        },
        "runtime": runtime,
        "resource": resource,
        "reachability": "reachable",
        "status": "PASS",
        "failure": None,
        "reason": None,
        "windows_identity": "LDJ" if worker_id == "gpu4060" else "LAPTOP-5G3QUOGM",
        "runtime_identity": {
            "wsl_distro": "Ubuntu-22.04",
            "conda_executable": "/home/shardgrid/miniconda3/bin/conda",
            "conda_environment": "shardgrid",
            "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
            "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            "python_version": "Python 3.12.13",
        },
        "gpu": {
            "name": gpu,
            "gpu_count": 1,
            "selected_gpu": 0,
            "total_memory_mb": resource.gpu_total_memory,
            "free_memory_mb": resource.gpu_free_memory,
            "compute_capability": resource.compute_capability,
            "driver_version": resource.driver_version,
        },
        "cuda": {
            "available": True,
            "runtime_version": "11.8",
            "torch_cuda_is_available": True,
        },
        "pytorch": {
            "version": "2.7.1+cu118",
            "python_version": "Python 3.12.13",
        },
        "backends": {
            "nccl_available": True,
            "nccl_version": "2.21.5",
            "gloo_available": True,
            "backend_capability": ["nccl", "gloo"],
        },
    }


def test_probe_json_reports_all_workers(monkeypatch, tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)

    def fake_probe(worker, config):
        if str(worker.worker_id) == "gpu4060":
            return monkeypatch_probe_dict_to_obj(
                _report(
                    "gpu4060",
                    "10.87.5.155",
                    "NVIDIA GeForce RTX 4060 Laptop GPU",
                )
            )
        return monkeypatch_probe_dict_to_obj(
            _report("gpu1060", "10.87.5.15", "NVIDIA GeForce GTX 1650")
        )

    monkeypatch.setattr("shardgrid.cli.commands.probe._probe_report", fake_probe)

    exit_code = main(["--json", "--config", str(config), "probe"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["worker_count"] == 2
    assert payload["workers"][0]["gpu"]["gpu_count"] == 1
    assert payload["workers"][0]["backends"]["nccl_version"] == "2.21.5"


def test_probe_worker_selects_single_target(monkeypatch, tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)
    seen: list[str] = []

    def fake_probe(worker, config):
        seen.append(str(worker.worker_id))
        return monkeypatch_probe_dict_to_obj(
            _report("gpu1060", "10.87.5.15", "NVIDIA GeForce GTX 1650")
        )

    monkeypatch.setattr("shardgrid.cli.commands.probe._probe_report", fake_probe)

    exit_code = main(["--config", str(config), "probe", "--worker", "gpu1060"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen == ["gpu1060"]
    assert "Worker: gpu1060" in captured.out
    assert "Worker: gpu4060" not in captured.out


def test_probe_unknown_worker_id_fails_cleanly(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)

    exit_code = main(["--config", str(config), "probe", "--worker", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown worker id: missing" in captured.out


def test_probe_failure_uses_stage_probe(monkeypatch, tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)
    failure = make_failure_record(
        stage=FailureStage.PROBE,
        host="10.87.5.155",
        worker_id="gpu4060",
        message="SSH connection to the Worker timed out",
        recommended_action="verify network reachability and rerun the probe",
    )
    failed = _report("gpu4060", "10.87.5.155", "NVIDIA GeForce RTX 4060 Laptop GPU")
    failed["status"] = "FAILED"
    failed["failure"] = failure
    failed["reason"] = failure.message

    monkeypatch.setattr(
        "shardgrid.cli.commands.probe._probe_report",
        lambda worker, config: monkeypatch_probe_dict_to_obj(failed),
    )

    exit_code = main(["--json", "--config", str(config), "probe", "--worker", "gpu4060"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["workers"][0]["failure"]["stage"] == "PROBE"
    assert payload["workers"][0]["status"] == "FAILED"
    assert payload["workers"][0]["reason"] == "SSH connection to the Worker timed out"


def monkeypatch_probe_dict_to_obj(payload: dict[str, object]):
    from shardgrid.cli.commands.probe import WorkerProbeReport

    failure = payload["failure"]
    if isinstance(failure, FailureRecord):
        failure_obj = failure
    elif isinstance(failure, dict):
        failure_obj = FailureRecord.from_dict(failure)
    else:
        failure_obj = None
    return WorkerProbeReport(
        worker=payload["worker"],
        runtime=payload["runtime"],
        resource=payload["resource"],
        reachability=payload["reachability"],
        status=payload["status"],
        failure=failure_obj,
        reason=payload["reason"],
        windows_identity=payload["windows_identity"],
        runtime_identity=payload["runtime_identity"],
        gpu=payload["gpu"],
        cuda=payload["cuda"],
        pytorch=payload["pytorch"],
        backends=payload["backends"],
    )
