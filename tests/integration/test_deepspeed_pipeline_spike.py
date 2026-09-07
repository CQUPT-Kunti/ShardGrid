"""DeepSpeed Pipeline compatibility spike tests (T062).

Logic tests validate script building, marker parsing, status derivation, and
evidence round-trip with mock payloads.  The live test runs the real
two-physical-host DeepSpeed pipeline spike (RTX 4060 rank 0, GTX 1650 rank 1,
1 GPU per host, pp=2) through the existing SSH + WSL2 + selected Conda chain
and asserts the honest result: only a real completed pipeline run is PASS.
"""

from __future__ import annotations

import json
from dataclasses import replace

from shardgrid.engines.deepspeed_pipeline import (
    SPIKE_STATUS_BLOCKED,
    SPIKE_STATUS_FAIL,
    SPIKE_STATUS_PASS,
    DeepspeedSpikeResult,
    build_spike_script,
    derive_spike_status,
    parse_spike_marker,
    parse_spike_results,
    run_deepspeed_spike,
    save_spike_evidence,
)


def test_build_spike_script_contains_required_elements() -> None:
    script = build_spike_script()
    assert "PipelineModule" in script
    assert "train_batch" in script
    assert "deepspeed.init_distributed" in script
    assert "DEEPSPEED_PIPELINE_INIT_OK" in script
    assert "DEEPSPEED_PIPELINE_DONE" in script
    assert "STAGES = 2" in script


def test_parse_spike_marker() -> None:
    assert parse_spike_marker("a DEEPSPEED_PIPELINE_INIT_OK b", "DEEPSPEED_PIPELINE_INIT_OK")
    assert not parse_spike_marker("nothing", "DEEPSPEED_PIPELINE_DONE")


def test_parse_spike_results() -> None:
    markers = parse_spike_results(
        "DEEPSPEED_PIPELINE_INIT_OK rank=0\n",
        "DEEPSPEED_PIPELINE_STEP_OK rank=0 step=0\n",
    )
    assert "DEEPSPEED_PIPELINE_INIT_OK" in markers
    assert "DEEPSPEED_PIPELINE_STEP_OK" in markers
    assert "DEEPSPEED_PIPELINE_DONE" not in markers


def test_derive_status_pass_when_all_markers() -> None:
    markers = [
        "DEEPSPEED_PIPELINE_INIT_OK",
        "DEEPSPEED_PIPELINE_STEP_OK",
        "DEEPSPEED_PIPELINE_DONE",
    ]
    status, blockers = derive_spike_status(markers, timed_out=False, timeout=120.0)
    assert status == SPIKE_STATUS_PASS
    assert blockers == []


def test_derive_status_blocked_on_timeout_after_init() -> None:
    markers = ["DEEPSPEED_PIPELINE_INIT_OK"]
    status, blockers = derive_spike_status(markers, timed_out=True, timeout=120.0)
    assert status == SPIKE_STATUS_BLOCKED
    assert any("train_batch" in blocker for blocker in blockers)
    assert any("isend/irecv" in blocker for blocker in blockers)


def test_derive_status_fail_on_timeout_before_init() -> None:
    status, blockers = derive_spike_status([], timed_out=True, timeout=120.0)
    assert status == SPIKE_STATUS_FAIL


def test_derive_status_fail_without_all_markers() -> None:
    status, blockers = derive_spike_status(
        ["DEEPSPEED_PIPELINE_INIT_OK", "DEEPSPEED_PIPELINE_STEP_OK"],
        timed_out=False,
        timeout=120.0,
    )
    assert status == SPIKE_STATUS_FAIL


def test_save_spike_evidence_round_trip(tmp_path: object) -> None:
    import pathlib

    result = DeepspeedSpikeResult(
        run_id="spike-1",
        status=SPIKE_STATUS_BLOCKED,
        deepspeed_version="0.19.5",
        torch_version="2.7.1+cu118",
        blockers=["train_batch deadlock"],
        diagnostics=["timeout=120.0s"],
        started_at="2026-08-22T00:00:00+00:00",
    )
    saved = save_spike_evidence(result, pathlib.Path(str(tmp_path)))
    assert saved.name == "deepspeed-pipeline-latest.json"
    loaded = json.loads(saved.read_text())
    assert loaded["status"] == SPIKE_STATUS_BLOCKED
    assert loaded["deepspeed_version"] == "0.19.5"
    assert "train_batch deadlock" in loaded["blockers"]


def test_live_deepspeed_pipeline_spike() -> None:
    """Real two-host DeepSpeed pipeline spike (opt-in via integration marker)."""
    import os

    from shardgrid.common.config import load_cluster_config
    from shardgrid.common.models import as_hostname
    from shardgrid.network.probe import discover_interface
    from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
    from shardgrid.transport.ssh import SSHOptions, SSHTransport

    config = load_cluster_config("examples/workers.yaml")
    address_book = json.load(open("tests/address.json"))
    wrappers: dict[int, tuple[WSLRuntimeWrapper, str, str]] = {}

    def resolve(worker_id: str) -> dict[str, object]:
        worker = next(w for w in config.workers if str(w.worker_id) == worker_id)
        gpu_label = worker.labels.get("gpu", "").upper().replace(" ", "")
        entry = next(
            e
            for e in address_book
            if gpu_label in str(e.get("gpu_model") or "").replace(" ", "").upper()
        )
        return {"worker": worker, "entry": entry, "ip": str(entry["ip"])}

    for worker_id, rank in [("gpu4060", 0), ("gpu1060", 1)]:
        info = resolve(worker_id)
        worker = info["worker"]
        entry = info["entry"]
        ip = info["ip"]
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
        wrappers[rank] = (wrapper, "", ip)

    for wrapper, _, _ in wrappers.values():
        wrapper.run("pkill -9 -f deepspeed_pipeline_spike.py || true", timeout=15.0)

    w0, _, ip0 = wrappers[0]
    w1, _, ip1 = wrappers[1]
    iface0 = discover_interface(w0.run, ip1) or ""
    iface1 = discover_interface(w1.run, ip0) or ""
    assert iface0 and iface1, f"interfaces: {iface0!r} / {iface1!r}"
    result = run_deepspeed_spike(
        w0,
        w1,
        rank0_worker_ip=ip0,
        rank1_worker_ip=ip1,
        rank0_interface=iface0,
        rank1_interface=iface1,
        master_port=29500,
        timeout=90.0,
    )

    for wrapper, _, _ in wrappers.values():
        wrapper.run("pkill -9 -f deepspeed_pipeline_spike.py || true", timeout=15.0)

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    save_spike_evidence(result, output_dir)

    detail = "\n".join(
        f"{step.name}: {step.status} {step.detail or ''}" for step in result.steps
    )
    # Acceptance: only a real completed pipeline is PASS; on the current WSL2
    # two-host environment the observed honest outcome is BLOCKED (train_batch
    # deadlock) with evidence, or PASS if DeepSpeed actually completes.
    assert result.status in (SPIKE_STATUS_PASS, SPIKE_STATUS_BLOCKED), (
        f"unexpected status {result.status}\n{detail}"
    )
    if result.status == SPIKE_STATUS_BLOCKED:
        assert any("train_batch" in blocker for blocker in result.blockers), (
            f"BLOCKED without train_batch blocker note\n{detail}"
        )
    else:
        assert result.deepspeed_version, f"PASS without deepspeed version\n{detail}"
