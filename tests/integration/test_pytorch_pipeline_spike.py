"""PyTorch pipeline compatibility spike tests (T063).

Logic tests validate script building, marker parsing, status derivation, and
evidence round-trip with mock payloads.  The live test runs the real
two-physical-host torch.distributed.pipelining spike (RTX 4060 rank 0,
GTX 1650 rank 1, 1 GPU per host, GPipe 2 stages) through the existing
SSH + WSL2 + selected Conda chain and asserts the honest result: only a real
completed pipeline run is PASS.
"""

from __future__ import annotations

import json
from dataclasses import replace

from shardgrid.engines.pytorch_pipeline import (
    SPIKE_STATUS_BLOCKED,
    SPIKE_STATUS_FAIL,
    SPIKE_STATUS_PASS,
    PytorchPipelineSpikeResult,
    build_spike_script,
    derive_spike_status,
    parse_spike_markers,
    run_pytorch_pipeline_spike,
    save_spike_evidence,
)


def test_build_spike_script_contains_required_elements() -> None:
    script = build_spike_script()
    assert "torch.distributed.pipelining" in script
    assert "PipelineStage" in script
    assert "ScheduleGPipe" in script
    assert "PYTORCH_PIPELINE_STAGE_READY" in script
    assert "PYTORCH_PIPELINE_DONE" in script


def test_parse_spike_markers() -> None:
    markers = parse_spike_markers(
        "PYTORCH_PIPELINE_STAGE_READY rank=0\n",
        "PYTORCH_PIPELINE_STEP_OK rank=0 step=0\nPYTORCH_PIPELINE_DONE rank=0\n",
    )
    assert set(markers) == {
        "PYTORCH_PIPELINE_STAGE_READY",
        "PYTORCH_PIPELINE_STEP_OK",
        "PYTORCH_PIPELINE_DONE",
    }


def test_derive_status_pass_when_all_markers() -> None:
    markers = ["PYTORCH_PIPELINE_STAGE_READY", "PYTORCH_PIPELINE_STEP_OK",
               "PYTORCH_PIPELINE_DONE"]
    status, blockers = derive_spike_status(markers, timed_out=False, timeout=180.0)
    assert status == SPIKE_STATUS_PASS
    assert blockers == []


def test_derive_status_fail_without_all_markers() -> None:
    status, blockers = derive_spike_status(
        ["PYTORCH_PIPELINE_STAGE_READY"], timed_out=False, timeout=180.0
    )
    assert status == SPIKE_STATUS_FAIL


def test_derive_status_blocked_on_timeout_after_ready() -> None:
    status, blockers = derive_spike_status(
        ["PYTORCH_PIPELINE_STAGE_READY"], timed_out=True, timeout=180.0
    )
    assert status == SPIKE_STATUS_BLOCKED
    assert any("did not finish" in blocker for blocker in blockers)


def test_save_spike_evidence_round_trip(tmp_path: object) -> None:
    import pathlib

    result = PytorchPipelineSpikeResult(
        run_id="spike-1",
        status=SPIKE_STATUS_PASS,
        torch_version="2.7.1+cu118",
        diagnostics=["timeout=180.0s"],
        started_at="2026-08-22T00:00:00+00:00",
    )
    saved = save_spike_evidence(result, pathlib.Path(str(tmp_path)))
    assert saved.name == "pytorch-pipeline-latest.json"
    loaded = json.loads(saved.read_text())
    assert loaded["status"] == SPIKE_STATUS_PASS
    assert loaded["torch_version"] == "2.7.1+cu118"


def test_live_pytorch_pipeline_spike() -> None:
    """Real two-host torch.distributed.pipelining spike (opt-in)."""
    import os
    import re

    from shardgrid.common.config import load_cluster_config
    from shardgrid.common.models import as_hostname
    from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
    from shardgrid.transport.ssh import SSHOptions, SSHTransport

    config = load_cluster_config("examples/workers.yaml")
    address_book = json.load(open("tests/address.json"))
    wrappers: dict[int, tuple[WSLRuntimeWrapper, str]] = {}

    def resolve(worker_id: str) -> tuple[dict[str, object], str]:
        worker = next(w for w in config.workers if str(w.worker_id) == worker_id)
        gpu_label = worker.labels.get("gpu", "").upper().replace(" ", "")
        entry = next(
            e
            for e in address_book
            if gpu_label in str(e.get("gpu_model") or "").replace(" ", "").upper()
        )
        return {"worker": worker, "entry": entry}, str(entry["ip"])

    for worker_id, rank in [("gpu4060", 0), ("gpu1060", 1)]:
        info, ip = resolve(worker_id)
        worker = info["worker"]
        entry = info["entry"]
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
        wrappers[rank] = (wrapper, ip)

    for wrapper, _ in wrappers.values():
        wrapper.run("pkill -9 -f pytorch_pipeline_spike.py || true", timeout=15.0)

    w0, ip0 = wrappers[0]
    w1, ip1 = wrappers[1]
    route0 = w0.run(f"ip route get {ip1}", timeout=10)
    match0 = re.search(r"\bdev\s+(\S+)", (route0.stdout or "").strip())
    iface0 = match0.group(1) if match0 else ""
    route1 = w1.run(f"ip route get {ip0}", timeout=10)
    match1 = re.search(r"\bdev\s+(\S+)", (route1.stdout or "").strip())
    iface1 = match1.group(1) if match1 else ""
    assert iface0 and iface1, f"interfaces: {iface0!r} / {iface1!r}"

    result = run_pytorch_pipeline_spike(
        w0,
        w1,
        rank0_worker_ip=ip0,
        rank1_worker_ip=ip1,
        rank0_interface=iface0,
        rank1_interface=iface1,
        master_port=29500,
        timeout=180.0,
    )

    for wrapper, _ in wrappers.values():
        wrapper.run("pkill -9 -f pytorch_pipeline_spike.py || true", timeout=15.0)

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    save_spike_evidence(result, output_dir)

    detail = "\n".join(
        f"{step.name}: {step.status} {step.detail or ''}" for step in result.steps
    )
    # Acceptance: PyTorch's mature pipeline API is usable before any custom
    # model-parallel code; on this environment the real run must be PASS.
    assert result.status == SPIKE_STATUS_PASS, (
        f"pytorch pipeline {result.status}\n{detail}"
    )
    assert result.torch_version, f"PASS without torch version\n{detail}"