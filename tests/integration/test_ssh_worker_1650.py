from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from shardgrid.common.config import ClusterConfig, WorkerConfig, load_cluster_config
from shardgrid.common.process import ProcessResult
from shardgrid.transport.remote_access import run_remote_access_check
from shardgrid.transport.ssh import SSHOptions, SSHTransport

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "examples" / "workers.yaml"
ADDRESS_PATH = ROOT / "tests" / "address.json"
TARGET_WORKER_ID = "gpu1060"
TARGET_GPU_LABEL = "GTX 1650"
TARGET_HOST = "10.87.5.15"


def _result(
    *,
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


def _probe_output(*, python_executable: str) -> str:
    return json.dumps(
        {
            "python": {
                "executable": python_executable,
                "version": "Python 3.12.13",
            },
            "torch": {
                "version": "2.7.1+cu118",
                "cuda_version": "11.8",
                "available": True,
                "device_count": 1,
                "name": "NVIDIA GeForce GTX 1650",
                "total_memory_mb": 4096,
                "free_memory_mb": 3200,
                "utilization_percent": 7,
                "compute_capability": "7.5",
                "nccl": "2.21.5",
                "gloo": True,
            },
            "nvidia_smi": {
                "name": "NVIDIA GeForce GTX 1650",
                "memory_total_mb": 4096,
                "memory_free_mb": 3200,
                "utilization_percent": 7,
                "driver_version": "527.41",
                "compute_capability": "7.5",
            },
        },
        sort_keys=True,
    )


def _load_live_worker_config() -> tuple[ClusterConfig, WorkerConfig]:
    config = load_cluster_config(CONFIG_PATH)
    worker = next(
        candidate
        for candidate in config.workers
        if str(candidate.worker_id) == TARGET_WORKER_ID
    )
    address_book = json.loads(ADDRESS_PATH.read_text(encoding="utf-8"))
    address_entry = next(
        entry
        for entry in address_book
        if TARGET_GPU_LABEL in str(entry.get("gpu_model") or "")
    )
    return config, replace(
        worker,
        host=address_entry["ip"],
        ssh_user=address_entry["username"],
    )


def _build_transport(config: ClusterConfig, worker: WorkerConfig) -> SSHTransport:
    return SSHTransport(
        SSHOptions.from_ssh_config(
            config.ssh,
            host=str(worker.host),
            user=worker.ssh_user,
            port=worker.ssh_port,
        )
    )


def run_gtx1650_remote_access_check(
    transport: SSHTransport,
    worker: WorkerConfig,
    *,
    preferred_environment: str | None = None,
):
    return run_remote_access_check(
        transport,
        worker,
        worker_label=TARGET_GPU_LABEL,
        preferred_environment=preferred_environment,
    )


def test_remote_access_success_path_uses_ssh_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    config, worker = _load_live_worker_config()
    transport = _build_transport(config, worker)
    calls: list[str] = []
    responses = [
        _result(stdout="LAPTOP-5G3QUOGM\n"),
        _result(
            stdout="  NAME      STATE           VERSION\n"
            "* Ubuntu    Running         2\n"
        ),
        _result(
            stdout=_probe_output(
                python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python"
            )
        ),
    ]

    def fake_run(
        command: list[str] | tuple[str, ...] | str,
        *,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
    ) -> ProcessResult:
        del stdin, timeout
        calls.append(command if isinstance(command, str) else " ".join(command))
        return responses.pop(0)

    monkeypatch.setattr(transport, "run", fake_run)

    outcome = run_gtx1650_remote_access_check(transport, worker)

    assert isinstance(outcome.transport, SSHTransport)
    assert outcome.status == "PASS"
    assert outcome.windows_identity == "LAPTOP-5G3QUOGM"
    assert outcome.wsl_distro == "Ubuntu"
    assert outcome.runtime_identity is not None
    assert outcome.runtime_identity.conda_environment == "shardgrid"
    assert outcome.runtime_identity.python_executable.endswith("/bin/python")
    assert outcome.runtime_identity.python_executable.startswith(
        outcome.runtime_identity.conda_prefix
    )
    assert outcome.runtime_identity.python_version == "Python 3.12.13"
    assert outcome.gpu_probe_result is not None
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("stderr", "timed_out", "expected_category"),
    [
        (
            "ssh: connect to host 10.87.5.15 port 22: No route to host",
            False,
            "host_unreachable",
        ),
        (
            "ssh: connect to host 10.87.5.15 port 22: Connection timed out",
            True,
            "connection_timeout",
        ),
        (
            "shardgrid@10.87.5.15: Permission denied "
            "(publickey,password,keyboard-interactive).",
            False,
            "authentication_failure",
        ),
        ("Host key verification failed.", False, "known_host_failure"),
    ],
)
def test_remote_access_classifies_connection_failures(
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    timed_out: bool,
    expected_category: str,
) -> None:
    config, worker = _load_live_worker_config()
    transport = _build_transport(config, worker)

    def fake_run(
        command: list[str] | tuple[str, ...],
        *,
        timeout: float | None = None,
    ) -> ProcessResult:
        del command, timeout
        return _result(stderr=stderr, exit_code=255, timed_out=timed_out)

    monkeypatch.setattr(transport, "run", fake_run)

    outcome = run_gtx1650_remote_access_check(transport, worker)

    assert outcome.status == "BLOCKED"
    assert outcome.failure_category == expected_category
    assert outcome.failure_record is not None
    assert outcome.failure_record["stage"] == "PROBE"
    assert outcome.failure_record["host"] == str(worker.host)


def test_remote_access_distinguishes_windows_wsl_and_conda_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, worker = _load_live_worker_config()
    worker = replace(worker, conda_prefix=None)
    transport = _build_transport(config, worker)
    responses = [
        _result(stdout="LAPTOP-5G3QUOGM\n"),
        _result(stdout="  NAME      STATE           VERSION\n* Ubuntu    Running         2\n"),
        _result(stdout='{"status":"conda_unavailable"}', exit_code=12),
    ]

    def fake_run(
        command: list[str] | tuple[str, ...] | str,
        *,
        timeout: float | None = None,
    ) -> ProcessResult:
        del command, timeout
        return responses.pop(0)

    monkeypatch.setattr(transport, "run", fake_run)

    outcome = run_gtx1650_remote_access_check(transport, worker)

    assert outcome.status == "FAIL"
    assert outcome.windows_identity == "LAPTOP-5G3QUOGM"
    assert outcome.wsl_distro == "Ubuntu"
    assert outcome.failure_category == "conda_unavailable"
    assert outcome.failure_record is not None
    assert outcome.failure_record["message"] == (
        "WSL is reachable but Conda is unavailable in the training runtime"
    )


def test_remote_access_rejects_python_outside_selected_conda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, worker = _load_live_worker_config()
    transport = _build_transport(config, worker)
    responses = [
        _result(stdout="LAPTOP-5G3QUOGM\n"),
        _result(stdout="  NAME      STATE           VERSION\n* Ubuntu    Running         2\n"),
        _result(stdout=_probe_output(python_executable="/usr/bin/python3")),
    ]

    def fake_run(
        command: list[str] | tuple[str, ...] | str,
        *,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
    ) -> ProcessResult:
        del command, stdin, timeout
        return responses.pop(0)

    monkeypatch.setattr(transport, "run", fake_run)

    outcome = run_gtx1650_remote_access_check(transport, worker)

    assert outcome.status == "FAIL"
    assert outcome.failure_category == "runtime_python_outside_selected_conda"
    assert outcome.failure_record is not None
    assert outcome.failure_record["python_executable"] == "/usr/bin/python3"


def test_remote_access_distinguishes_runtime_command_timeout_from_ssh_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, worker = _load_live_worker_config()
    transport = _build_transport(config, worker)
    responses = [
        _result(stdout="LAPTOP-5G3QUOGM\n"),
        _result(stdout="  NAME      STATE           VERSION\n* Ubuntu    Running         2\n"),
        _result(stderr="timed out", exit_code=-1, timed_out=True),
    ]

    def fake_run(
        command: list[str] | tuple[str, ...] | str,
        *,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
    ) -> ProcessResult:
        del command, stdin, timeout
        return responses.pop(0)

    monkeypatch.setattr(transport, "run", fake_run)

    outcome = run_gtx1650_remote_access_check(transport, worker)

    assert outcome.status == "FAIL"
    assert outcome.windows_identity == "LAPTOP-5G3QUOGM"
    assert outcome.wsl_distro == "Ubuntu"
    assert outcome.failure_category == "runtime_probe_timeout"
    assert outcome.failure_record is not None
    assert outcome.failure_record["message"] == (
        "WSL is reachable but the structured runtime probe timed out"
    )


def test_live_remote_access_attempt_to_gtx1650_records_real_result() -> None:
    config, worker = _load_live_worker_config()
    transport = _build_transport(config, worker)

    outcome = run_gtx1650_remote_access_check(transport, worker)

    assert isinstance(outcome.transport, SSHTransport)
    assert outcome.worker_id == TARGET_WORKER_ID
    assert outcome.host == TARGET_HOST
    assert outcome.ssh_user == "shardgrid"
    assert outcome.commands
    assert outcome.status in {"PASS", "BLOCKED", "FAIL"}
    if outcome.status == "PASS":
        assert outcome.windows_identity
        assert outcome.wsl_distro
        assert outcome.runtime_identity is not None
        assert outcome.runtime_identity.conda_executable
        assert outcome.runtime_identity.conda_environment
        assert outcome.runtime_identity.conda_prefix
        assert outcome.runtime_identity.python_executable.startswith(
            outcome.runtime_identity.conda_prefix
        )
    else:
        assert outcome.failure_category is not None
        assert outcome.failure_reason is not None
        assert outcome.failure_record is not None
