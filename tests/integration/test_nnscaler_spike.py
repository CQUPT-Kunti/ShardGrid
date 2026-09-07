"""nnScaler compatibility spike tests (T064).

Logic tests validate the install-blocker detection and status derivation with
the real captured install plan from the official-source attempt.  The live
test records the environment state and the real blocker evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from shardgrid.engines.nnscaler import (
    SPIKE_STATUS_BLOCKED,
    SPIKE_STATUS_PASS,
    NnscalerSpikeResult,
    find_forbidden_package_changes,
    preflight_install_blockers,
    run_nnscaler_spike,
    save_spike_evidence,
)

# Real pip resolution plan captured from the official nnScaler install attempt
# on the RTX 4060 Worker (2026-08-22): it replaced torch 2.7.1+cu118 with
# torch 2.6.0 plus the nvidia-cu12 stack.
REAL_WOULD_INSTALL = (
    "contourpy-1.3.3 cycler-0.12.1 dill-0.4.1 fonttools-4.63.0 kiwisolver-1.5.0 "
    "matplotlib-3.11.1 more-itertools-11.1.0 nnscaler-0.8 "
    "nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 "
    "nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 "
    "nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 "
    "nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 "
    "nvidia-cusparse-cu12-12.3.1.170 nvidia-cusparselt-cu12-0.6.2 "
    "nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.4.127 "
    "nvidia-nvtx-cu12-12.4.127 pulp-3.3.2 pybind11-2.13.6 pyparsing-3.3.2 "
    "python-dateutil-2.9.0.post0 sympy-1.13.1 torch-2.6.0 triton-3.2.0"
)


def test_find_forbidden_package_changes() -> None:
    changes = find_forbidden_package_changes(REAL_WOULD_INSTALL)
    assert "torch-2.6.0" in changes
    assert "nvidia-cublas-cu12-12.4.5.8" in changes
    assert "triton-3.2.0" in changes
    assert "nnscaler-0.8" not in changes
    assert "matplotlib-3.11.1" not in changes


def test_preflight_blocks_torch_replacement() -> None:
    blockers = preflight_install_blockers(
        REAL_WOULD_INSTALL, current_torch_version="2.7.1+cu118"
    )
    assert any("torch-2.6.0" in blocker and "2.7.1+cu118" in blocker for blocker in blockers)


def test_preflight_blocks_cuda_stack_change() -> None:
    blockers = preflight_install_blockers(
        REAL_WOULD_INSTALL, current_torch_version="2.7.1+cu118"
    )
    assert any("nvidia-cublas-cu12" in blocker for blocker in blockers)
    assert any("nvidia-cudnn-cu12" in blocker for blocker in blockers)


def test_preflight_allows_clean_plan() -> None:
    blockers = preflight_install_blockers(
        "nnscaler-0.8 pulp-3.3.2 pybind11-2.13.6",
        current_torch_version="2.7.1+cu118",
    )
    assert blockers == []


def test_spike_blocked_with_real_evidence() -> None:
    result = run_nnscaler_spike(
        _FakeWrapper("2.7.1+cu118"),
        would_install_text=REAL_WOULD_INSTALL,
        nnscaler_installed=False,
        current_torch_version="2.7.1+cu118",
    )
    assert result.status == SPIKE_STATUS_BLOCKED
    assert result.torch_version == "2.7.1+cu118"
    assert result.blockers
    assert any("torch-2.6.0" in blocker for blocker in result.blockers)
    names = [step.name for step in result.steps]
    assert "install_preflight" in names
    preflight = next(step for step in result.steps if step.name == "install_preflight")
    assert preflight.status == "BLOCKED"


def test_spike_pass_when_installed_without_changes() -> None:
    result = run_nnscaler_spike(
        _FakeWrapper("2.7.1+cu118"),
        would_install_text="nnscaler-0.8",
        nnscaler_installed=True,
        current_torch_version="2.7.1+cu118",
    )
    assert result.status == SPIKE_STATUS_PASS


def test_save_spike_evidence_round_trip(tmp_path: object) -> None:
    import pathlib

    result = NnscalerSpikeResult(
        run_id="spike-1",
        status=SPIKE_STATUS_BLOCKED,
        nnscaler_installed=False,
        torch_version="2.7.1+cu118",
        blockers=["torch replacement"],
        started_at="2026-08-22T00:00:00+00:00",
    )
    saved = save_spike_evidence(result, pathlib.Path(str(tmp_path)))
    assert saved.name == "nnscaler-latest.json"
    loaded = json.loads(saved.read_text())
    assert loaded["status"] == SPIKE_STATUS_BLOCKED
    assert "torch replacement" in loaded["blockers"]


def test_live_nnscaler_spike_records_environment() -> None:
    """Real environment state + real captured install plan (opt-in)."""
    from shardgrid.common.config import load_cluster_config
    from shardgrid.common.models import as_hostname
    from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
    from shardgrid.transport.ssh import SSHOptions, SSHTransport

    config = load_cluster_config("examples/workers.yaml")
    address_book = json.load(open("tests/address.json"))
    worker = next(w for w in config.workers if str(w.worker_id) == "gpu4060")
    entry = next(
        e
        for e in address_book
        if "RTX4060" in str(e.get("gpu_model") or "").replace(" ", "").upper()
    )
    ip = str(entry["ip"])
    resolved = replace(
        worker,
        host=as_hostname(ip),
        ssh_user=str(entry["username"]),
        runtime_distro="Ubuntu-22.04",
    )
    transport = SSHTransport(
        SSHOptions.from_ssh_config(
            config.ssh,
            host=ip,
            user=resolved.ssh_user,
            port=resolved.ssh_port,
        )
    )
    wrapper = WSLRuntimeWrapper(
        WSLRuntimeConfig.from_worker_and_runtime(resolved, config.runtime),
        transport,
    )

    result = run_nnscaler_spike(
        wrapper,
        would_install_text=REAL_WOULD_INSTALL,
        nnscaler_installed=False,
    )

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    save_spike_evidence(result, output_dir)

    assert result.torch_version == "2.7.1+cu118", result.torch_version
    assert result.status == SPIKE_STATUS_BLOCKED
    assert result.blockers
    assert any("torch-2.6.0" in blocker for blocker in result.blockers)


class _FakeWrapper:
    def __init__(self, torch_version: str) -> None:
        self.torch_version = torch_version

    def run(self, command: str, *, timeout: float) -> object:
        from shardgrid.common.process import ProcessResult

        return ProcessResult(
            args=(command,),
            recorded_command=command,
            shell=False,
            cwd=None,
            exit_code=0,
            stdout=self.torch_version + "\n",
            stderr="",
            timed_out=False,
            runtime_environment={},
        )