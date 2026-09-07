from __future__ import annotations

import shlex
from typing import Sequence

import pytest

from shardgrid.common.config import RuntimeConfig, WorkerConfig
from shardgrid.common.enums import FailureStage
from shardgrid.common.process import ProcessResult
from shardgrid.transport.runtime import (
    EXIT_COMMAND_NOT_FOUND,
    EXIT_CWD_NOT_FOUND,
    WSLRuntimeConfig,
    WSLRuntimeWrapper,
)


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


def _worker(**overrides: str | None) -> WorkerConfig:
    payload: dict[str, str | None] = {
        "id": "gpu1650",
        "machine_id": "machine-d",
        "physical_os": "windows",
        "runtime_os": "wsl2_linux",
        "runtime": "wsl2",
        "host": "10.87.5.15",
        "ssh_user": "shardgrid",
        "runtime_distro": "Ubuntu",
    }
    payload.update(overrides)
    return WorkerConfig.from_dict(payload)


class FakeExecutor:
    def __init__(self, responses: list[ProcessResult]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []
        self.scripts: list[str] = []
        self.timeouts: list[float | None] = []

    def run(
        self,
        command: Sequence[str] | str,
        *,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
    ) -> ProcessResult:
        text = command if isinstance(command, str) else " ".join(command)
        self.calls.append(text)
        self.timeouts.append(timeout)
        if isinstance(stdin, str):
            self.scripts.append(stdin)
        return self.responses.pop(0)


def _wrapper(
    executor: FakeExecutor | None = None,
    **overrides: str | None,
) -> WSLRuntimeWrapper:
    defaults: dict[str, str | None] = {
        "distro": "Ubuntu",
        "user": "shardgrid",
        "conda_environment": "shardgrid",
        "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
    }
    defaults.update(overrides)
    config = WSLRuntimeConfig(**defaults)
    return WSLRuntimeWrapper(config, executor or FakeExecutor([_result()]))


def test_payload_uses_configured_conda_prefix_without_activate_or_run() -> None:
    wrapper = _wrapper()

    payload = wrapper.build_payload(["python", "-V"])

    assert "conda activate" not in payload
    assert "conda run" not in payload
    assert 'export PATH=/home/shardgrid/miniconda3/envs/shardgrid/bin:"$PATH"' in payload
    assert "export CONDA_DEFAULT_ENV=shardgrid" in payload
    assert payload.endswith("python -V")


def test_payload_uses_conda_run_when_only_env_and_executable_configured() -> None:
    wrapper = _wrapper(
        conda_prefix=None,
        conda_executable="/home/shardgrid/miniconda3/bin/conda",
    )

    payload = wrapper.build_payload(["python", "-V"])

    assert payload.endswith("conda run -n shardgrid -- python -V")


def test_payload_round_trips_args_with_spaces_and_special_characters() -> None:
    wrapper = _wrapper()
    command = ["python", "-c", "print('a b')", "arg with $HOME; spaces & such", 'q"uote']

    payload = wrapper.build_payload(command)

    joined = shlex.join(command)
    assert payload.endswith(joined)
    assert shlex.split(joined) == command


def test_payload_embeds_cwd_with_quoting() -> None:
    wrapper = _wrapper()

    payload = wrapper.build_payload(["python", "-V"], cwd="/home/shardgrid/my jobs")

    assert "cd '/home/shardgrid/my jobs' || exit 66" in payload


def test_payload_embeds_environment_variables() -> None:
    wrapper = _wrapper()

    payload = wrapper.build_payload(
        ["python", "-c", "import os; print(os.environ['K'])"],
        env={"K": "v with spaces", "NCCL_DEBUG": "INFO"},
    )

    assert "export K='v with spaces'" in payload
    assert "export NCCL_DEBUG=INFO" in payload


def test_remote_command_uses_powershell_encoded_wsl_wrapper() -> None:
    import base64

    wrapper = _wrapper()

    remote = wrapper.build_remote_command("echo ok")

    assert remote.startswith("powershell -NoProfile -EncodedCommand ")
    encoded = remote.split("EncodedCommand ", 1)[1]
    decoded = base64.b64decode(encoded).decode("utf-16le")
    assert "wsl.exe" in decoded
    assert "base64 -d" in decoded
    assert "exit $LASTEXITCODE" in decoded
    payload_b64 = base64.b64encode(b"echo ok").decode("ascii")
    assert payload_b64 in decoded


def test_run_preserves_stdout_stderr_and_exit_code() -> None:
    executor = FakeExecutor([_result(stdout="out line\n", stderr="err line\n", exit_code=3)])
    wrapper = _wrapper(executor=executor)

    result = wrapper.run(["python", "-V"])

    assert result.exit_code == 3
    assert result.stdout == "out line\n"
    assert result.stderr == "err line\n"
    assert result.ok is False
    assert executor.calls
    assert executor.calls[0].startswith("powershell -NoProfile")
    assert executor.timeouts == [None]


def test_run_requires_configured_distro() -> None:
    wrapper = _wrapper(distro=None)

    with pytest.raises(ValueError, match="distro"):
        wrapper.run(["python", "-V"])


def test_run_requires_selected_conda_environment() -> None:
    wrapper = _wrapper(conda_prefix=None, conda_environment=None, conda_executable=None)

    with pytest.raises(ValueError, match="Conda environment"):
        wrapper.run(["python", "-V"])


def test_classify_cwd_not_found() -> None:
    wrapper = _wrapper()

    failure = wrapper.classify_runtime_failure(
        _result(stderr="cd: no such file", exit_code=EXIT_CWD_NOT_FOUND),
        host="10.87.5.15",
    )

    assert failure.stage == FailureStage.LAUNCH
    assert failure.exit_code == EXIT_CWD_NOT_FOUND
    assert "working directory" in failure.message
    assert failure.retryable is False


def test_classify_command_not_found() -> None:
    wrapper = _wrapper()

    failure = wrapper.classify_runtime_failure(
        _result(stderr="command not found", exit_code=EXIT_COMMAND_NOT_FOUND),
        host="10.87.5.15",
    )

    assert failure.exit_code == EXIT_COMMAND_NOT_FOUND
    assert "command was not found" in failure.message
    assert failure.conda_environment == "shardgrid"
    assert failure.conda_prefix == "/home/shardgrid/miniconda3/envs/shardgrid"


def test_classify_generic_non_zero_exit() -> None:
    wrapper = _wrapper()

    failure = wrapper.classify_runtime_failure(
        _result(stderr="boom", exit_code=9),
        host="10.87.5.15",
    )

    assert failure.exit_code == 9
    assert "non-zero exit code" in failure.message
    assert failure.retryable is True


def test_config_from_worker_and_runtime_prefers_worker_values() -> None:
    worker = _worker(
        runtime_distro="Ubuntu",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
    )
    runtime = RuntimeConfig(
        default_wsl_distro="OtherDistro",
        conda_prefix="/fallback/prefix",
        conda_environment="fallback-env",
    )

    config = WSLRuntimeConfig.from_worker_and_runtime(worker, runtime)

    assert config.distro == "Ubuntu"
    assert config.conda_environment == "shardgrid"
    assert config.conda_prefix == "/home/shardgrid/miniconda3/envs/shardgrid"
    assert config.user == "shardgrid"


def test_config_from_worker_and_runtime_falls_back_to_runtime_defaults() -> None:
    worker = _worker(
        runtime_distro=None,
        conda_environment=None,
        conda_prefix=None,
    )
    runtime = RuntimeConfig(
        default_wsl_distro="Ubuntu",
        conda_prefix="/opt/conda/envs/shardgrid",
        conda_environment="shardgrid",
        conda_executable="/opt/conda/bin/conda",
    )

    config = WSLRuntimeConfig.from_worker_and_runtime(worker, runtime)

    assert config.distro == "Ubuntu"
    assert config.conda_environment == "shardgrid"
    assert config.conda_prefix == "/opt/conda/envs/shardgrid"
    assert config.conda_executable == "/opt/conda/bin/conda"

def test_run_script_feeds_script_via_stdin_and_uses_conda_python() -> None:
    executor = FakeExecutor([_result(stdout="3.12.13\n")])
    wrapper = _wrapper(executor=executor)

    result = wrapper.run_script(
        "import sys\nprint(sys.version.split()[0])\n",
        timeout=45.0,
    )

    assert result.stdout.strip() == "3.12.13"
    remote = executor.calls[0]
    assert remote.startswith("wsl.exe -d Ubuntu -u shardgrid -- /bin/bash -lc ")
    assert "/home/shardgrid/miniconda3/envs/shardgrid/bin/python -" in remote
    assert "powershell" not in remote
    assert executor.scripts == ["import sys\nprint(sys.version.split()[0])\n"]
    assert executor.timeouts == [45.0]


def test_run_script_requires_distro_and_prefix() -> None:
    with pytest.raises(ValueError, match="distro"):
        _wrapper(distro=None).run_script("print(1)")
    with pytest.raises(ValueError, match="Conda prefix"):
        _wrapper(conda_prefix=None).run_script("print(1)")
