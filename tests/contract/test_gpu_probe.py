from __future__ import annotations

import json
from typing import Sequence

from shardgrid.common.config import RuntimeConfig, WorkerConfig
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.process import ProcessResult
from shardgrid.resources.models import WorkerResource
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.workers.gpu_probe import probe_gpu


def _result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> ProcessResult:
    return ProcessResult(
        args=(),
        recorded_command="",
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        runtime_environment={},
    )


def _worker() -> WorkerConfig:
    return WorkerConfig.from_dict(
        {
            "id": "gpu4060",
            "machine_id": "machine-c",
            "physical_os": "windows",
            "runtime_os": "wsl2_linux",
            "runtime": "wsl2",
            "host": "10.87.5.155",
            "ssh_user": "shardgrid",
            "runtime_distro": "Ubuntu",
            "conda_environment": "shardgrid",
            "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
        }
    )


class FakeExecutor:
    def __init__(self, responses: list[ProcessResult]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []
        self.scripts: list[str] = []

    def run(
        self,
        command: Sequence[str] | str,
        *,
        stdin: str | bytes | None = None,
    ) -> ProcessResult:
        self.calls.append(command if isinstance(command, str) else " ".join(command))
        if isinstance(stdin, str):
            self.scripts.append(stdin)
        return self.responses.pop(0)


def _wrapper(responses: list[ProcessResult]) -> WSLRuntimeWrapper:
    runtime = RuntimeConfig(
        default_wsl_distro="Ubuntu",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        conda_environment="shardgrid",
    )
    config = WSLRuntimeConfig.from_worker_and_runtime(_worker(), runtime)
    return WSLRuntimeWrapper(config, FakeExecutor(responses))


def _full_payload() -> str:
    payload = {
        "torch": {
            "version": "2.7.1+cu118",
            "cuda_version": "11.8",
            "available": True,
            "device_count": 1,
            "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "total_memory_mb": 8192,
            "free_memory_mb": 6000,
            "utilization_percent": 12,
            "compute_capability": "8.9",
            "current_device": 0,
            "nccl": "2.21.5",
            "gloo": True,
        },
        "nvidia_smi": {
            "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "memory_total_mb": 8192,
            "memory_free_mb": 6000,
            "utilization_percent": 12,
            "driver_version": "566.07",
            "compute_capability": "8.9",
        },
    }
    return json.dumps(payload)


def test_gpu_probe_populates_worker_resource_and_runtime() -> None:
    worker = _worker()
    wrapper = _wrapper([_result(stdout=_full_payload() + "\n")])

    outcome = probe_gpu(wrapper, worker, probe_status="mock")

    assert outcome.health == Health.HEALTHY
    assert outcome.failures == ()
    resource = outcome.worker_resource
    assert resource.physical_os == PhysicalOS.WINDOWS
    assert resource.runtime_os == RuntimeOS.WSL2_LINUX
    assert resource.gpu_name == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert resource.gpu_total_memory == 8192
    assert resource.gpu_free_memory == 6000
    assert resource.gpu_utilization == 12
    assert resource.compute_capability == "8.9"
    assert resource.driver_version == "566.07"
    assert resource.cuda_version == "11.8"
    assert resource.torch_version == "2.7.1+cu118"
    assert resource.nccl_available is True
    assert resource.gloo_available is True
    runtime = outcome.worker_runtime
    assert runtime.conda_environment == "shardgrid"
    assert runtime.conda_prefix == "/home/shardgrid/miniconda3/envs/shardgrid"
    assert runtime.python_executable == "/home/shardgrid/miniconda3/envs/shardgrid/bin/python"
    assert runtime.cuda_available is True


def test_gpu_probe_worker_resource_serializes_and_round_trips() -> None:
    worker = _worker()
    wrapper = _wrapper([_result(stdout=_full_payload() + "\n")])

    outcome = probe_gpu(wrapper, worker, probe_status="mock")
    payload = outcome.worker_resource.to_dict()

    assert WorkerResource.from_dict(payload) == outcome.worker_resource
    assert payload["gpu_name"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert payload["physical_os"] == "windows"
    assert payload["runtime_os"] == "wsl2_linux"
    assert payload["conda_environment"] == "shardgrid"


def test_gpu_probe_torch_unavailable_keeps_smi_data_and_failure() -> None:
    payload = {
        "torch": {"error": "ModuleNotFoundError: No module named 'torch'"},
        "nvidia_smi": {
            "name": "NVIDIA GeForce GTX 1650",
            "memory_total_mb": 4096,
            "memory_free_mb": 3000,
            "utilization_percent": 0,
            "driver_version": "527.41",
            "compute_capability": "7.5",
        },
    }
    wrapper = _wrapper([_result(stdout=json.dumps(payload) + "\n")])

    outcome = probe_gpu(wrapper, _worker(), probe_status="mock")

    assert outcome.health == Health.FAILED
    assert any(f.check == "torch" for f in outcome.failures)
    assert outcome.worker_resource.gpu_name == "NVIDIA GeForce GTX 1650"
    assert outcome.worker_resource.driver_version == "527.41"
    assert outcome.worker_resource.torch_version is None


def test_gpu_probe_nvidia_smi_unavailable_keeps_torch_data_and_failure() -> None:
    payload = {
        "torch": {
            "version": "2.7.1+cu118",
            "cuda_version": "11.8",
            "available": True,
            "device_count": 1,
            "name": "NVIDIA GeForce GTX 1650",
            "total_memory_mb": 4096,
            "free_memory_mb": 3000,
            "utilization_percent": None,
            "compute_capability": "7.5",
            "current_device": 0,
            "nccl": "2.21.5",
            "gloo": True,
        },
        "nvidia_smi": {"error": "nvidia-smi: command not found"},
    }
    wrapper = _wrapper([_result(stdout=json.dumps(payload) + "\n")])

    outcome = probe_gpu(wrapper, _worker(), probe_status="mock")

    assert outcome.health == Health.FAILED
    assert any(f.check == "nvidia_smi" for f in outcome.failures)
    assert outcome.worker_resource.gpu_name == "NVIDIA GeForce GTX 1650"
    assert outcome.worker_resource.gpu_total_memory == 4096
    assert outcome.worker_resource.driver_version is None


def test_gpu_probe_cuda_unavailable_is_failed_honestly() -> None:
    payload = {
        "torch": {
            "version": "2.7.1+cu118",
            "cuda_version": None,
            "available": False,
            "device_count": 0,
        },
        "nvidia_smi": {
            "name": "NVIDIA GeForce GTX 1650",
            "memory_total_mb": 4096,
            "memory_free_mb": 3000,
            "utilization_percent": 0,
            "driver_version": "527.41",
            "compute_capability": "7.5",
        },
    }
    wrapper = _wrapper([_result(stdout=json.dumps(payload) + "\n")])

    outcome = probe_gpu(wrapper, _worker(), probe_status="mock")

    assert outcome.health == Health.FAILED
    assert any(f.check == "cuda" for f in outcome.failures)
    assert outcome.worker_resource.gpu_name == "NVIDIA GeForce GTX 1650"
    assert outcome.worker_runtime.cuda_available is False


def test_gpu_probe_nccl_unavailable_is_failed_honestly() -> None:
    payload = {
        "torch": {
            "version": "2.7.1+cu118",
            "cuda_version": "11.8",
            "available": True,
            "device_count": 1,
            "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "total_memory_mb": 8192,
            "free_memory_mb": 6000,
            "utilization_percent": 5,
            "compute_capability": "8.9",
            "current_device": 0,
            "nccl": None,
            "gloo": True,
        },
        "nvidia_smi": {
            "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "memory_total_mb": 8192,
            "memory_free_mb": 6000,
            "utilization_percent": 5,
            "driver_version": "566.07",
            "compute_capability": "8.9",
        },
    }
    wrapper = _wrapper([_result(stdout=json.dumps(payload) + "\n")])

    outcome = probe_gpu(wrapper, _worker(), probe_status="mock")

    assert outcome.health == Health.FAILED
    assert any(f.check == "nccl" for f in outcome.failures)
    assert outcome.worker_resource.nccl_available is False


def test_gpu_probe_no_gpu_visible_is_failed_honestly() -> None:
    payload = {
        "torch": {"error": "ModuleNotFoundError: No module named 'torch'"},
        "nvidia_smi": {"error": "nvidia-smi: command not found"},
    }
    wrapper = _wrapper([_result(stdout=json.dumps(payload) + "\n")])

    outcome = probe_gpu(wrapper, _worker(), probe_status="mock")

    assert outcome.health == Health.FAILED
    assert any(f.check == "gpu" for f in outcome.failures)
    assert outcome.worker_resource.gpu_name is None


def test_gpu_probe_unparseable_output_is_failed_honestly() -> None:
    wrapper = _wrapper([_result(stderr="boom", exit_code=1)])

    outcome = probe_gpu(wrapper, _worker(), probe_status="mock")

    assert outcome.health == Health.FAILED
    assert any(f.check == "gpu_probe_script" for f in outcome.failures)
    assert outcome.worker_resource.gpu_name is None