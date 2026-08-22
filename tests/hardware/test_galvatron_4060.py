"""Galvatron compatibility workload on the RTX 4060 Worker (T056).

The logic tests here validate the spike status derivation, evidence parsing,
preflight rules, and result serialization with mock payloads.  The live test
executes the real workload on the physical RTX 4060 Worker from Machine A
through the existing SSH + WSL2 + selected Conda chain and asserts the
result honestly: only a real PASS is reported as PASS.
"""

from __future__ import annotations

import json
import os
from typing import Any

from shardgrid.engines.galvatron import (
    GALVATRON_OFFICIAL_REF,
    GALVATRON_OFFICIAL_REPO,
    SPIKE_STATUS_BLOCKED,
    SPIKE_STATUS_FAIL,
    SPIKE_STATUS_PASS,
    GalvatronSpikeResult,
    SpikeStepStatus,
    build_spike_evidence_script,
    derive_spike_status,
    load_galvatron_spike_evidence,
    parse_spike_evidence,
    preflight_install_blockers,
    save_galvatron_spike_evidence,
)

WORKER_ID = "gpu4060"
EXPECTED_GPU = "RTX 4060"
DEFAULT_PROXY_URL = "http://127.0.0.1:7890"


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "worker_id": WORKER_ID,
        "expected_gpu": EXPECTED_GPU,
        "conda_environment": "shardgrid",
        "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
        "python_executable": "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        "python_version": "3.12.13",
        "torch_version": "2.7.1+cu118",
        "torch_cuda_version": "11.8",
        "torch_cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "compute_capability": "8.9",
        "driver_version": "566.07",
        "galvatron_installed": True,
        "galvatron_version": "2.4.1",
        "galvatron_source": "/opt/checkout/Hetu-Galvatron/galvatron/__init__.py",
        "galvatron_import_error": None,
        "galvatron_submodules_ok": True,
        "diagnostics": [],
    }
    payload.update(overrides)
    return payload


def _result(**overrides: Any) -> GalvatronSpikeResult:
    result = GalvatronSpikeResult(
        run_id="galvatron-spike-abc123",
        worker_id=WORKER_ID,
        expected_gpu=EXPECTED_GPU,
        status=SPIKE_STATUS_PASS,
        started_at="2026-08-18T00:00:00+00:00",
        elapsed_s=1.0,
        **{k: v for k, v in overrides.items() if k not in {"status"}},
    )
    return result


def test_spike_evidence_script_injects_identity() -> None:
    script = build_spike_evidence_script(worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU)
    assert f'worker_id = "{WORKER_ID}"' in script
    assert f'expected_gpu = "{EXPECTED_GPU}"' in script
    assert "GALVATRON_SPIKE_EVIDENCE " in script
    assert "HardwareProfiler" in script
    assert "from galvatron.core.profiler import" in script


def test_parse_spike_evidence() -> None:
    payload = parse_spike_evidence(
        "log\nGALVATRON_SPIKE_EVIDENCE " + json.dumps(_payload()) + "\n"
    )
    assert payload is not None
    assert payload["worker_id"] == WORKER_ID
    assert payload["galvatron_version"] == "2.4.1"
    assert parse_spike_evidence("nothing here") is None
    assert parse_spike_evidence("GALVATRON_SPIKE_EVIDENCE not-json") is None


def test_derive_spike_status_rules() -> None:
    def step(name: str, status: SpikeStepStatus) -> Any:
        from shardgrid.engines.galvatron import GalvatronSpikeStep

        return GalvatronSpikeStep(name=name, status=status)

    assert derive_spike_status([step("a", SpikeStepStatus.PASS)]) == SPIKE_STATUS_PASS
    assert (
        derive_spike_status([step("a", SpikeStepStatus.PASS), step("b", SpikeStepStatus.FAIL)])
        == SPIKE_STATUS_FAIL
    )
    assert (
        derive_spike_status(
            [step("a", SpikeStepStatus.FAIL), step("b", SpikeStepStatus.BLOCKED)]
        )
        == SPIKE_STATUS_BLOCKED
    )
    assert (
        derive_spike_status([step("a", SpikeStepStatus.SKIPPED)])
        == SPIKE_STATUS_PASS
    )


def test_preflight_allows_fresh_dependencies() -> None:
    blockers = preflight_install_blockers(
        "Would install galvatron-2.4.1 numpy-1.26.4 transformers-4.49.0 "
        "scipy-1.13.1 hydra-core-1.3.2 pydantic-2.9.2 omegaconf-2.3.0",
        current_torch_version="2.7.1+cu118",
    )
    assert blockers == []


def test_preflight_blocks_torch_change() -> None:
    blockers = preflight_install_blockers(
        "Would install torch-2.8.0+cu126 torchvision-0.23.0",
        current_torch_version="2.7.1+cu118",
    )
    assert any("torch" in blocker for blocker in blockers)
    assert any("2.7.1" in blocker for blocker in blockers)


def test_preflight_blocks_cuda_stack_change() -> None:
    blockers = preflight_install_blockers(
        "Would install triton-3.2.0 nvidia-cuda-runtime-cu12-12.6.77",
        current_torch_version="2.7.1+cu118",
    )
    assert len(blockers) == 2
    assert any("triton" in blocker for blocker in blockers)
    assert any("nvidia-cuda" in blocker for blocker in blockers)


class _InstalledWrapper:
    """Already-installed Galvatron: run() only services checkout + profiler."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[Any] = []

    def run_script(self, script: str, *, timeout: float) -> Any:
        from shardgrid.common.process import ProcessResult

        return ProcessResult(
            args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
            stdout="GALVATRON_SPIKE_EVIDENCE " + json.dumps(self.payload),
            stderr="", timed_out=False, runtime_environment={},
        )

    def run(self, command: Any, *, timeout: float, env: Any = None) -> Any:
        from shardgrid.common.process import ProcessResult

        self.calls.append(command)
        rendered = " ".join(command) if isinstance(command, list) else str(command)
        if "test -f" in rendered:
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=1,
                stdout="", stderr="", timed_out=False, runtime_environment={},
            )
        if "ls-remote" in rendered:
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="deadbeef\tHEAD", stderr="", timed_out=False, runtime_environment={},
            )
        if "clone" in rendered:
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="cloned", stderr="", timed_out=False, runtime_environment={},
            )
        if "profile_hardware.py" in rendered:
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="profile_hardware OK", stderr="", timed_out=False,
                runtime_environment={},
            )
        raise AssertionError(f"unexpected command: {rendered}")


def test_gpu_identity_mismatch_is_fail() -> None:
    from shardgrid.engines.galvatron import run_galvatron_spike

    wrapper = _InstalledWrapper(_payload(gpu_name="NVIDIA GeForce GT 1030"))
    result = run_galvatron_spike(wrapper, worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU)
    assert result.status == SPIKE_STATUS_FAIL
    assert result.gpu_matched is False
    gpu_step = next(step for step in result.steps if step.name == "gpu_identity")
    assert gpu_step.status == SpikeStepStatus.FAIL


def test_cuda_unavailable_is_fail() -> None:
    from shardgrid.engines.galvatron import run_galvatron_spike

    wrapper = _InstalledWrapper(_payload(torch_cuda_available=False))
    result = run_galvatron_spike(wrapper, worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU)
    assert result.status == SPIKE_STATUS_FAIL
    cuda_step = next(step for step in result.steps if step.name == "cuda_visibility")
    assert cuda_step.status == SpikeStepStatus.FAIL


def test_installed_reused_with_profiler_pass() -> None:
    from shardgrid.engines.galvatron import run_galvatron_spike

    wrapper = _InstalledWrapper(_payload())
    result = run_galvatron_spike(wrapper, worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU)
    assert result.status == SPIKE_STATUS_PASS
    assert result.install_mode == "reused"
    assert result.galvatron_version == "2.4.1"
    assert "profile_hardware.py" in " ".join(
        " ".join(command) for command in wrapper.calls
    )
    profiler_step = next(
        step for step in result.steps if step.name == "profiler_runtime"
    )
    assert profiler_step.status == SpikeStepStatus.PASS


def test_install_disabled_and_missing_is_blocked() -> None:
    from shardgrid.engines.galvatron import run_galvatron_spike

    class _Wrapper:
        def run_script(self, script: str, *, timeout: float) -> Any:
            from shardgrid.common.process import ProcessResult

            payload = _payload(
                galvatron_installed=False,
                galvatron_version=None,
                galvatron_source=None,
            )
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="GALVATRON_SPIKE_EVIDENCE " + json.dumps(payload),
                stderr="", timed_out=False, runtime_environment={},
            )

        def run(self, command: Any, *, timeout: float, env: Any = None) -> Any:
            del env
            raise AssertionError("install disabled; no commands expected")

    result = run_galvatron_spike(
        _Wrapper(), worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU, install=False
    )
    assert result.status == SPIKE_STATUS_BLOCKED
    install_step = next(step for step in result.steps if step.name == "galvatron_install")
    assert install_step.status == SpikeStepStatus.BLOCKED


def test_clone_preflight_fail_fast_blocks_before_clone() -> None:
    from shardgrid.common.process import ProcessResult
    from shardgrid.engines.galvatron import run_galvatron_spike

    class _Wrapper:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run_script(self, script: str, *, timeout: float) -> ProcessResult:
            payload = _payload(
                galvatron_installed=False,
                galvatron_version=None,
                galvatron_source=None,
            )
            return ProcessResult(
                args=(),
                recorded_command="",
                shell=False,
                cwd=None,
                exit_code=0,
                stdout="GALVATRON_SPIKE_EVIDENCE " + json.dumps(payload),
                stderr="",
                timed_out=False,
                runtime_environment={},
            )

        def run(self, command: Any, *, timeout: float, env: Any = None) -> ProcessResult:
            rendered = " ".join(command) if isinstance(command, list) else str(command)
            self.calls.append(rendered)
            if "test -f" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=1,
                    stdout="", stderr="", timed_out=False, runtime_environment={},
                )
            if "ls-remote" in rendered:
                assert env is not None and env["https_proxy"] == DEFAULT_PROXY_URL
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=1,
                    stdout="", stderr="fatal: unable to access", timed_out=False,
                    runtime_environment={},
                )
            raise AssertionError(f"unexpected command: {rendered}")

    wrapper = _Wrapper()
    result = run_galvatron_spike(
        wrapper,
        worker_id=WORKER_ID,
        expected_gpu=EXPECTED_GPU,
        install=True,
        proxy_url=DEFAULT_PROXY_URL,
    )

    assert result.status == SPIKE_STATUS_BLOCKED
    install_step = next(step for step in result.steps if step.name == "galvatron_install")
    assert install_step.status == SpikeStepStatus.BLOCKED
    assert "not reachable" in (install_step.detail or "")
    assert all("git clone" not in call for call in wrapper.calls)


def test_missing_python_dep_is_auto_installed_and_continues() -> None:
    from shardgrid.common.process import ProcessResult
    from shardgrid.engines.galvatron import run_galvatron_spike

    class _Wrapper:
        def __init__(self) -> None:
            self.installed = False
            self.pydantic_installed = False
            self.pip_installs: list[str] = []

        def run_script(self, script: str, *, timeout: float) -> ProcessResult:
            if not self.installed:
                payload = _payload(
                    galvatron_installed=False,
                    galvatron_version=None,
                    galvatron_source=None,
                )
            elif not self.pydantic_installed:
                payload = _payload(
                    galvatron_installed=False,
                    galvatron_version=None,
                    galvatron_source=None,
                    galvatron_submodules_ok=False,
                    galvatron_import_error=(
                        "ModuleNotFoundError: No module named 'pydantic'"
                    ),
                )
            else:
                payload = _payload()
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="GALVATRON_SPIKE_EVIDENCE " + json.dumps(payload),
                stderr="", timed_out=False, runtime_environment={},
            )

        def run(self, command: Any, *, timeout: float, env: Any = None) -> ProcessResult:
            rendered = " ".join(command) if isinstance(command, list) else str(command)
            if "test -f" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=1,
                    stdout="", stderr="", timed_out=False, runtime_environment={},
                )
            if "ls-remote" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="refs/tags/v2.4.0", stderr="", timed_out=False,
                    runtime_environment={},
                )
            if "clone" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="cloned", stderr="", timed_out=False, runtime_environment={},
                )
            if "--dry-run" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="Would install torch-2.8.0+cu126 galvatron-2.4.1",
                    stderr="", timed_out=False, runtime_environment={},
                )
            if "--no-deps" in rendered:
                self.installed = True
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="Successfully installed hetu-galvatron-2.4.1",
                    stderr="", timed_out=False, runtime_environment={},
                )
            if "pip install pydantic" in rendered:
                self.pydantic_installed = True
                self.pip_installs.append("pydantic")
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="Successfully installed pydantic", stderr="",
                    timed_out=False, runtime_environment={},
                )
            if "profile_hardware.py" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="profile_hardware OK", stderr="", timed_out=False,
                    runtime_environment={},
                )
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="installed minimal deps", stderr="", timed_out=False,
                runtime_environment={},
            )

    wrapper = _Wrapper()
    result = run_galvatron_spike(
        wrapper,
        worker_id=WORKER_ID,
        expected_gpu=EXPECTED_GPU,
        install=True,
        proxy_url="http://127.0.0.1:7890",
    )
    assert result.status == SPIKE_STATUS_PASS
    assert "pydantic" in wrapper.pip_installs
    assert result.galvatron_version == "2.4.1"
    assert any(
        "auto-installed missing Python deps" in (step.detail or "")
        for step in result.steps
    )


def test_torch_stack_missing_module_is_not_auto_installed() -> None:
    from shardgrid.common.process import ProcessResult
    from shardgrid.engines.galvatron import run_galvatron_spike

    class _Wrapper:
        def __init__(self) -> None:
            self.installed = False
            self.calls: list[str] = []

        def run_script(self, script: str, *, timeout: float) -> ProcessResult:
            if not self.installed:
                payload = _payload(
                    galvatron_installed=False,
                    galvatron_version=None,
                    galvatron_source=None,
                )
            else:
                payload = _payload(
                    galvatron_installed=False,
                    galvatron_version=None,
                    galvatron_source=None,
                    galvatron_submodules_ok=False,
                    galvatron_import_error=(
                        "ModuleNotFoundError: No module named 'torchvision'"
                    ),
                )
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="GALVATRON_SPIKE_EVIDENCE " + json.dumps(payload),
                stderr="", timed_out=False, runtime_environment={},
            )

        def run(self, command: Any, *, timeout: float, env: Any = None) -> ProcessResult:
            rendered = " ".join(command) if isinstance(command, list) else str(command)
            self.calls.append(rendered)
            if "test -f" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=1,
                    stdout="", stderr="", timed_out=False, runtime_environment={},
                )
            if "ls-remote" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="refs/tags/v2.4.0", stderr="", timed_out=False,
                    runtime_environment={},
                )
            if "clone" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="cloned", stderr="", timed_out=False, runtime_environment={},
                )
            if "--dry-run" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="Would install torch-2.8.0+cu126 galvatron-2.4.1",
                    stderr="", timed_out=False, runtime_environment={},
                )
            if "--no-deps" in rendered:
                self.installed = True
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="Successfully installed hetu-galvatron-2.4.1",
                    stderr="", timed_out=False, runtime_environment={},
                )
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="installed minimal deps", stderr="", timed_out=False,
                runtime_environment={},
            )

    wrapper = _Wrapper()
    result = run_galvatron_spike(
        wrapper,
        worker_id=WORKER_ID,
        expected_gpu=EXPECTED_GPU,
        install=True,
        proxy_url="http://127.0.0.1:7890",
    )
    assert result.status == SPIKE_STATUS_BLOCKED
    assert not any("pip install torchvision" in call for call in wrapper.calls)
    blocked_step = next(
        step
        for step in result.steps
        if step.name == "galvatron_install"
        and step.status == SpikeStepStatus.BLOCKED
    )
    assert "torchvision" in (blocked_step.detail or "")


def test_spike_evidence_round_trip(tmp_path: Any) -> None:
    result = GalvatronSpikeResult(
        run_id="galvatron-spike-abc123",
        worker_id=WORKER_ID,
        expected_gpu=EXPECTED_GPU,
        status=SPIKE_STATUS_PASS,
        started_at="2026-08-18T00:00:00+00:00",
        elapsed_s=12.5,
        install_mode="reused",
        official_source=GALVATRON_OFFICIAL_REPO,
        requested_ref=GALVATRON_OFFICIAL_REF,
        resolved_commit="deadbeef",
        galvatron_version="2.4.1",
        galvatron_source="/opt/checkout/Hetu-Galvatron/galvatron/__init__.py",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        torch_cuda_available=True,
        gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        gpu_matched=True,
        steps=(),
        diagnostics=("note",),
    )
    saved = save_galvatron_spike_evidence(result, tmp_path)
    assert saved.name.startswith("galvatron-spike-rtx4060-")
    assert (tmp_path / "galvatron-spike-rtx4060-latest.json").exists()
    loaded = load_galvatron_spike_evidence(saved)
    assert loaded.status == SPIKE_STATUS_PASS
    assert loaded.requested_ref == GALVATRON_OFFICIAL_REF
    assert loaded.resolved_commit == "deadbeef"
    assert loaded.galvatron_version == "2.4.1"
    assert loaded.diagnostics == ("note",)


def test_live_galvatron_spike_rtx4060() -> None:
    """Real Galvatron workload on the physical RTX 4060 Worker (opt-in)."""
    from dataclasses import replace

    from shardgrid.common.config import load_cluster_config
    from shardgrid.common.models import as_hostname
    from shardgrid.engines.galvatron import run_galvatron_spike
    from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
    from shardgrid.transport.ssh import SSHOptions, SSHTransport

    config = load_cluster_config("examples/workers.yaml")
    address_book = json.load(open("tests/address.json"))
    worker = next(w for w in config.workers if str(w.worker_id) == WORKER_ID)
    entry = next(
        e
        for e in address_book
        if EXPECTED_GPU.replace(" ", "")
        in str(e.get("gpu_model") or "").replace(" ", "")
    )
    worker = replace(
        worker,
        host=as_hostname(str(entry["ip"])),
        ssh_user=str(entry["username"]),
    )
    transport = SSHTransport(
        SSHOptions.from_ssh_config(
            config.ssh,
            host=str(entry["ip"]),
            user=worker.ssh_user,
            port=worker.ssh_port,
        )
    )
    wrapper = WSLRuntimeWrapper(
        WSLRuntimeConfig.from_worker_and_runtime(worker, config.runtime), transport
    )
    result = run_galvatron_spike(
        wrapper,
        worker_id=WORKER_ID,
        expected_gpu=EXPECTED_GPU,
        install=True,
        proxy_url=os.environ.get("SHARDGRID_GALVATRON_PROXY_4060", DEFAULT_PROXY_URL),
    )
    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    path = save_galvatron_spike_evidence(result, output_dir)

    detail = "\n".join(
        f"{step.name}: {step.status.value} {step.detail or ''}"
        for step in result.steps
    )
    assert result.status in {
        SPIKE_STATUS_PASS,
        SPIKE_STATUS_FAIL,
        SPIKE_STATUS_BLOCKED,
    }, f"unexpected status {result.status}\n{detail}"
    assert result.gpu_matched is True, (
        f"expected {EXPECTED_GPU}, got {result.gpu_name!r}\n{detail}"
    )
    assert result.torch_cuda_available is True, f"CUDA unavailable\n{detail}"
    assert result.status != SPIKE_STATUS_PASS or result.galvatron_version, (
        f"PASS without Galvatron version\n{detail}"
    )
    assert path.exists()
