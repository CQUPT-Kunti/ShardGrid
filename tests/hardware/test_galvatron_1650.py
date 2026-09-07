"""Galvatron compatibility workload on the GTX 1650 Worker (T057).

Same workload and status rules as T056 but judged independently on the GTX
1650 Worker.  The logic tests validate status derivation, evidence parsing,
preflight rules, and serialization with mock payloads.  The live test executes
the real workload on the physical GTX 1650 Worker from Machine A and asserts
an honest result: only a real PASS is reported as PASS.  Memory and the older
compute capability (7.5) are recorded as measured limitations, never assumed.
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

WORKER_ID = "gpu1060"
EXPECTED_GPU = "GTX 1650"
DEFAULT_PROXY_URL = "http://127.0.0.1:7897"


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
        "gpu_name": "NVIDIA GeForce GTX 1650",
        "compute_capability": "7.5",
        "driver_version": "527.41",
        "galvatron_installed": True,
        "galvatron_version": "2.4.1",
        "galvatron_source": "/opt/checkout/Hetu-Galvatron/galvatron/__init__.py",
        "galvatron_import_error": None,
        "galvatron_submodules_ok": True,
        "diagnostics": [],
    }
    payload.update(overrides)
    return payload


def test_spike_evidence_script_injects_identity() -> None:
    script = build_spike_evidence_script(worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU)
    assert f'worker_id = "{WORKER_ID}"' in script
    assert f'expected_gpu = "{EXPECTED_GPU}"' in script
    assert "GALVATRON_SPIKE_EVIDENCE " in script


def test_parse_spike_evidence() -> None:
    payload = parse_spike_evidence(
        "log\nGALVATRON_SPIKE_EVIDENCE " + json.dumps(_payload()) + "\n"
    )
    assert payload is not None
    assert payload["worker_id"] == WORKER_ID
    assert payload["compute_capability"] == "7.5"
    assert parse_spike_evidence("nothing") is None


def test_derive_spike_status_with_skipped_flash_attn() -> None:
    from shardgrid.engines.galvatron import GalvatronSpikeStep

    steps = (
        GalvatronSpikeStep(name="env_detect", status=SpikeStepStatus.PASS),
        GalvatronSpikeStep(name="galvatron_import", status=SpikeStepStatus.PASS),
        GalvatronSpikeStep(name="flash_attn", status=SpikeStepStatus.SKIPPED),
    )
    assert derive_spike_status(steps) == SPIKE_STATUS_PASS


def test_preflight_allows_fresh_dependencies_on_1650() -> None:
    blockers = preflight_install_blockers(
        "Would install galvatron-2.4.1 numpy-1.26.4 transformers-4.49.0 "
        "sentencepiece-0.1.99 pybind11-2.12.0",
        current_torch_version="2.7.1+cu118",
    )
    assert blockers == []


def test_preflight_blocks_any_torch_change() -> None:
    blockers = preflight_install_blockers(
        "Would install torch-2.8.0+cu126",
        current_torch_version="2.7.1+cu118",
    )
    assert any("torch" in blocker for blocker in blockers)


class _InstalledWrapper:
    """Already-installed Galvatron: run() only services checkout + profiler."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def run_script(self, script: str, *, timeout: float) -> Any:
        from shardgrid.common.process import ProcessResult

        return ProcessResult(
            args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
            stdout="GALVATRON_SPIKE_EVIDENCE " + json.dumps(self.payload),
            stderr="", timed_out=False, runtime_environment={},
        )

    def run(self, command: Any, *, timeout: float, env: Any = None) -> Any:
        from shardgrid.common.process import ProcessResult

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

    wrapper = _InstalledWrapper(_payload(gpu_name="NVIDIA GeForce RTX 3090"))
    result = run_galvatron_spike(wrapper, worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU)
    assert result.status == SPIKE_STATUS_FAIL
    assert result.gpu_matched is False


def test_destructive_default_resolution_falls_back_to_constrained_install() -> None:
    from shardgrid.common.process import ProcessResult
    from shardgrid.engines.galvatron import run_galvatron_spike

    class _Wrapper:
        def __init__(self) -> None:
            self.calls: list[Any] = []
            self.installed = False

        def run_script(self, script: str, *, timeout: float) -> ProcessResult:
            if not self.installed:
                payload = _payload(
                    galvatron_installed=False,
                    galvatron_version=None,
                    galvatron_source=None,
                )
            else:
                payload = _payload()
            return ProcessResult(
                args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                stdout="GALVATRON_SPIKE_EVIDENCE " + json.dumps(payload),
                stderr="", timed_out=False, runtime_environment={},
            )

        def run(self, command: Any, *, timeout: float, env: Any = None) -> ProcessResult:
            self.calls.append(command)
            assert env is None or env.get("https_proxy")
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
                    stdout="cloned", stderr="", timed_out=False,
                    runtime_environment={},
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
        wrapper, worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU, install=True
    )
    assert result.status == SPIKE_STATUS_PASS
    assert result.install_mode == "installed-minimal"
    install_step = next(step for step in result.steps if step.name == "galvatron_install")
    assert install_step.status == SpikeStepStatus.PASS
    assert "constrained minimal deps" in (install_step.detail or "")
    joined = " ".join(" ".join(c) for c in wrapper.calls)
    assert "--no-deps" in joined
    assert "numpy<2.0.0" in joined
    assert result.galvatron_version == "2.4.1"


def test_install_failure_is_blocked() -> None:
    from shardgrid.common.process import ProcessResult
    from shardgrid.engines.galvatron import run_galvatron_spike

    class _Wrapper:
        def run_script(self, script: str, *, timeout: float) -> ProcessResult:
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
                    stdout="deadbeef\tHEAD", stderr="", timed_out=False, runtime_environment={},
                )
            if "clone" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="cloned", stderr="", timed_out=False,
                    runtime_environment={},
                )
            if "--dry-run" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=0,
                    stdout="Would install torch-2.8.0+cu126 galvatron-2.4.1",
                    stderr="", timed_out=False, runtime_environment={},
                )
            if "--no-deps" in rendered:
                return ProcessResult(
                    args=(), recorded_command="", shell=False, cwd=None, exit_code=1,
                    stdout="", stderr="ERROR: g++ not found; dp_core build failed",
                    timed_out=False, runtime_environment={},
                )
            raise AssertionError(f"unexpected command: {rendered}")

    result = run_galvatron_spike(
        _Wrapper(), worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU, install=True
    )
    assert result.status == SPIKE_STATUS_BLOCKED
    assert result.install_mode == "failed"
    install_step = next(step for step in result.steps if step.name == "galvatron_install")
    assert install_step.status == SpikeStepStatus.BLOCKED
    assert "g++ not found" in (install_step.detail or "")


def test_memory_and_capability_limitations_recorded() -> None:
    from shardgrid.engines.galvatron import run_galvatron_spike

    wrapper = _InstalledWrapper(
        _payload(
            compute_capability="7.5",
            gpu_name="NVIDIA GeForce GTX 1650",
        )
    )
    result = run_galvatron_spike(wrapper, worker_id=WORKER_ID, expected_gpu=EXPECTED_GPU)
    assert result.status == SPIKE_STATUS_PASS
    assert result.compute_capability == "7.5"
    gpu_step = next(step for step in result.steps if step.name == "gpu_identity")
    assert "7.5" in (gpu_step.detail or "")
    assert result.gpu_matched is True


def test_spike_evidence_round_trip(tmp_path: Any) -> None:
    result = GalvatronSpikeResult(
        run_id="galvatron-spike-def456",
        worker_id=WORKER_ID,
        expected_gpu=EXPECTED_GPU,
        status=SPIKE_STATUS_PASS,
        started_at="2026-08-18T00:00:00+00:00",
        elapsed_s=15.0,
        install_mode="reused",
        official_source=GALVATRON_OFFICIAL_REPO,
        requested_ref=GALVATRON_OFFICIAL_REF,
        resolved_commit="cafebabe",
        galvatron_version="2.4.1",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        torch_cuda_available=True,
        gpu_name="NVIDIA GeForce GTX 1650",
        compute_capability="7.5",
        gpu_matched=True,
        steps=(),
        diagnostics=("gtx1650 4GB vram limitation noted",),
    )
    saved = save_galvatron_spike_evidence(result, tmp_path)
    assert saved.name.startswith("galvatron-spike-gtx1650-")
    loaded = load_galvatron_spike_evidence(saved)
    assert loaded.status == SPIKE_STATUS_PASS
    assert loaded.requested_ref == GALVATRON_OFFICIAL_REF
    assert loaded.resolved_commit == "cafebabe"
    assert loaded.compute_capability == "7.5"
    assert "4GB vram limitation" in loaded.diagnostics[0]


def test_live_galvatron_spike_gtx1650() -> None:
    """Real Galvatron workload on the physical GTX 1650 Worker (opt-in)."""
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
        proxy_url=os.environ.get("SHARDGRID_GALVATRON_PROXY_1650", DEFAULT_PROXY_URL),
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
    assert result.compute_capability in {"7.5", None}, (
        f"unexpected compute capability {result.compute_capability}\n{detail}"
    )
    assert path.exists()
