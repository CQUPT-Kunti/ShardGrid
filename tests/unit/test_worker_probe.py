from __future__ import annotations

from typing import Sequence

from shardgrid.common.config import WorkerConfig
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.process import ProcessResult
from shardgrid.workers.probe import probe_worker


def _result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    timed_out: bool = False,
) -> ProcessResult:
    return ProcessResult(
        args=(),
        recorded_command="",
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        runtime_environment={},
    )


def _worker(**overrides: str) -> WorkerConfig:
    payload: dict[str, str] = {
        "id": "gpu4060",
        "machine_id": "machine-c",
        "physical_os": "windows",
        "runtime_os": "wsl2_linux",
        "runtime": "wsl2",
        "host": "machine-c.local",
        "ssh_user": "shardgrid",
        "runtime_distro": "Ubuntu-22.04",
    }
    payload.update(overrides)
    return WorkerConfig.from_dict(payload)


class FakeRunner:
    """Stand-in for the PlatformAdapter runner; returns canned results."""

    def __init__(self, responses: list[tuple[str, ProcessResult]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def run(self, command: str | Sequence[str]) -> ProcessResult:
        text = command if isinstance(command, str) else " ".join(command)
        self.calls.append(text)
        for needle, result in self.responses:
            if needle in text:
                return result
        return _result(exit_code=1, stderr=f"no mock response for: {text}")


def _healthy_windows_runner() -> FakeRunner:
    return FakeRunner(
        [
            ("Win32_OperatingSystem", _result("Microsoft Windows 11 Pro | 10.0.22631")),
            ("Get-Command ssh", _result("C:\\Windows\\System32\\OpenSSH\\ssh.exe")),
            ("wsl.exe --status", _result("Default Distribution: Ubuntu-22.04")),
            ("Win32_VideoController", _result("NVIDIA GeForce RTX 4060")),
        ]
    )


def _healthy_wsl_runner() -> FakeRunner:
    return FakeRunner(
        [
            ("command -v conda", _result("/home/shardgrid/miniconda3/bin/conda")),
            (
                "conda env list",
                _result(
                    "base  /home/shardgrid/miniconda3\n"
                    "shardgrid-worker  /home/shardgrid/miniconda3/envs/shardgrid-worker\n"
                ),
            ),
            (
                "CONDA_DEFAULT_ENV",
                _result("shardgrid-worker|/home/shardgrid/miniconda3/envs/shardgrid-worker"),
            ),
            ("--version 2>&1", _result("Python 3.13.5")),
            (
                "TORCH_VERSION",
                _result("TORCH_VERSION=2.5.1\nCUDA_VERSION=12.4\nCUDA_AVAILABLE=True"),
            ),
            (
                "GPU_NAME",
                _result("GPU_NAME=NVIDIA GeForce RTX 4060\nGPU_CC=8.9\nGPU_MEM_MB=8192"),
            ),
            ("NCCL=", _result("NCCL=2.21.5")),
            ("GLOO=", _result("GLOO=True")),
            ("hostname -I", _result("192.168.1.30")),
        ]
    )


def _base_wsl_runner(cuda_available: bool = True) -> FakeRunner:
    torch_output = (
        "TORCH_VERSION=2.5.1\nCUDA_VERSION=12.4\nCUDA_AVAILABLE=True"
        if cuda_available
        else "TORCH_VERSION=2.5.1\nCUDA_VERSION=None\nCUDA_AVAILABLE=False"
    )
    responses: list[tuple[str, ProcessResult]] = [
        ("command -v conda", _result("/home/shardgrid/miniconda3/bin/conda")),
        ("conda env list", _result("base  /home/shardgrid/miniconda3\n")),
        ("CONDA_DEFAULT_ENV", _result("base|/home/shardgrid/miniconda3")),
        ("--version 2>&1", _result("Python 3.13.5")),
        ("TORCH_VERSION", _result(torch_output)),
        ("GLOO=", _result("GLOO=True")),
        ("hostname -I", _result("192.168.1.30")),
    ]
    if cuda_available:
        responses.insert(
            5,
            (
                "GPU_NAME",
                _result("GPU_NAME=NVIDIA GeForce RTX 4060\nGPU_CC=8.9\nGPU_MEM_MB=8192"),
            ),
        )
    return FakeRunner(responses)


def test_probe_distinguishes_windows_host_and_wsl_runtime() -> None:
    worker = _worker()
    windows = _healthy_windows_runner()
    wsl = _healthy_wsl_runner()

    result = probe_worker(worker, windows.run, wsl.run, probe_status="mock")

    assert result.health == Health.HEALTHY
    assert result.worker_resource.physical_os == PhysicalOS.WINDOWS
    assert result.worker_resource.runtime_os == RuntimeOS.WSL2_LINUX
    assert result.worker_runtime.runtime_os == RuntimeOS.WSL2_LINUX
    assert result.worker_runtime.runtime_version == "Ubuntu-22.04"
    assert result.windows_host.os_version == "Microsoft Windows 11 Pro | 10.0.22631"
    assert result.windows_host.wsl_available is True
    assert result.windows_host.nvidia_driver_visible is True
    assert result.windows_host.driver_name == "NVIDIA GeForce RTX 4060"
    assert result.worker_runtime.conda_environment == "shardgrid-worker"
    assert result.worker_runtime.conda_prefix == (
        "/home/shardgrid/miniconda3/envs/shardgrid-worker"
    )
    assert result.worker_runtime.python_version == "Python 3.13.5"
    assert result.worker_runtime.torch_version == "2.5.1"
    assert result.worker_runtime.cuda_available is True
    assert result.worker_runtime.nccl_available is True
    assert result.worker_runtime.gloo_available is True
    assert result.worker_resource.gpu_name == "NVIDIA GeForce RTX 4060"
    assert result.worker_resource.compute_capability == "8.9"
    assert result.worker_resource.gpu_total_memory == 8192
    assert result.worker_resource.network_interface == "192.168.1.30"


def test_probe_fails_honestly_when_wsl_is_missing() -> None:
    windows = FakeRunner(
        [
            ("Win32_OperatingSystem", _result("Microsoft Windows 11 Pro | 10.0.22631")),
            ("Get-Command ssh", _result("C:\\Windows\\System32\\OpenSSH\\ssh.exe")),
            ("wsl.exe --status", _result(stderr="wsl: command not found", exit_code=1)),
            ("Win32_VideoController", _result("NVIDIA GeForce RTX 4060")),
        ]
    )

    result = probe_worker(
        _worker(), windows.run, lambda command: _result(), probe_status="mock"
    )

    assert result.health == Health.FAILED
    assert result.windows_host.wsl_available is False
    assert any(f.check == "wsl_available" for f in result.failures)
    failure = next(f for f in result.failures if f.check == "wsl_available")
    assert "WSL2 is not available" in failure.message


def test_probe_fails_honestly_when_conda_is_missing() -> None:
    windows = _healthy_windows_runner()
    wsl = FakeRunner(
        [
            ("command -v conda", _result(exit_code=1, stderr="conda: not found")),
        ]
    )

    result = probe_worker(_worker(), windows.run, wsl.run, probe_status="mock")

    assert result.health == Health.FAILED
    assert result.worker_runtime.conda_executable is None
    assert any(f.check == "conda_executable" for f in result.failures)


def test_probe_fails_honestly_when_pytorch_is_missing() -> None:
    windows = _healthy_windows_runner()
    wsl = FakeRunner(
        [
            ("command -v conda", _result("/home/shardgrid/miniconda3/bin/conda")),
            ("conda env list", _result("base  /home/shardgrid/miniconda3\n")),
            ("CONDA_DEFAULT_ENV", _result("base|/home/shardgrid/miniconda3")),
            ("--version 2>&1", _result("Python 3.13.5")),
            (
                "TORCH_VERSION",
                _result(
                    stderr="ModuleNotFoundError: No module named 'torch'",
                    exit_code=1,
                ),
            ),
            ("GLOO=", _result("GLOO=False")),
            ("hostname -I", _result("192.168.1.30")),
        ]
    )

    result = probe_worker(_worker(), windows.run, wsl.run, probe_status="mock")

    assert result.health == Health.FAILED
    assert any(f.check == "pytorch" for f in result.failures)
    assert result.worker_runtime.torch_version is None
    assert result.worker_runtime.cuda_available is False


def test_probe_fails_honestly_when_cuda_is_unavailable() -> None:
    windows = _healthy_windows_runner()
    wsl = _base_wsl_runner(cuda_available=False)

    result = probe_worker(_worker(), windows.run, wsl.run, probe_status="mock")

    assert result.health == Health.FAILED
    assert result.worker_runtime.cuda_available is False
    assert any(f.check == "cuda" for f in result.failures)
    assert result.worker_resource.gpu_name is None


def test_probe_fails_honestly_when_nccl_is_unavailable() -> None:
    windows = _healthy_windows_runner()
    wsl = _base_wsl_runner()
    wsl.responses.insert(
        6,
        ("NCCL=", _result(stderr="RuntimeError: NCCL not available", exit_code=1)),
    )

    result = probe_worker(_worker(), windows.run, wsl.run, probe_status="mock")

    assert result.health == Health.FAILED
    assert any(f.check == "nccl" for f in result.failures)
    assert result.worker_runtime.nccl_available is False


def test_probe_fails_honestly_when_gpu_is_unavailable() -> None:
    windows = _healthy_windows_runner()
    wsl = _base_wsl_runner()
    wsl.responses.insert(
        5,
        ("GPU_NAME", _result(stderr="RuntimeError: Found no NVIDIA driver", exit_code=1)),
    )

    result = probe_worker(_worker(), windows.run, wsl.run, probe_status="mock")

    assert result.health == Health.FAILED
    assert any(f.check == "gpu" for f in result.failures)
    assert result.worker_resource.gpu_name is None


def test_probe_preserves_partial_failure_reasons() -> None:
    windows = _healthy_windows_runner()
    wsl = _base_wsl_runner()
    for index, (needle, _) in enumerate(wsl.responses):
        if needle == "hostname -I":
            wsl.responses[index] = (
                "hostname -I",
                _result(exit_code=1, stderr="hostname: -I: unknown option"),
            )
            break

    result = probe_worker(_worker(), windows.run, wsl.run, probe_status="mock")

    assert result.health == Health.FAILED
    assert any(f.check == "interface" for f in result.failures)
    interface_failure = next(f for f in result.failures if f.check == "interface")
    assert interface_failure.output is not None
    assert "hostname" in interface_failure.output
    assert result.worker_resource.network_interface is None
    assert result.worker_resource.gpu_name == "NVIDIA GeForce RTX 4060"