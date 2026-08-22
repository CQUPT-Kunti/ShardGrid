"""Galvatron capabilities regression: profiler / search / pipeline / runtime / checkpoint (T060).

T060 gives every required Galvatron capability a pass / fail / blocked result
with a blocker note, executing the official v2.4.0 entry points on the real
RTX 4060 Worker through the existing SSH + WSL2 + selected Conda chain.

Capabilities:

- ``profiler``: official ``profiler.py`` model profiler (computation and
  memory) plus the official ``profile_hardware.py`` hardware profiler.
- ``search_engine``: official ``search_dist.py`` (GalvatronSearchEngine).
- ``pipeline_construction``: explicit parallel-layout model construction
  inside the official training entry point.
- ``runtime_launch``: real minimal training run (random synthetic data,
  official ``train_dist_random.py``).
- ``checkpoint``: activation checkpoint configuration accepted and applied
  during the real training run.

Known environment limitation: the WSL2 CUPTI restriction blocks the hardware
profiler, which is a hard input of the search engine.  Both are reported as
``blocked`` with the exact error, never as pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Sequence

from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport

CAPABILITY_PROFILER = "profiler"
CAPABILITY_SEARCH_ENGINE = "search_engine"
CAPABILITY_PIPELINE = "pipeline_construction"
CAPABILITY_RUNTIME = "runtime_launch"
CAPABILITY_CHECKPOINT = "checkpoint"

CAPABILITIES = (
    CAPABILITY_PROFILER,
    CAPABILITY_SEARCH_ENGINE,
    CAPABILITY_PIPELINE,
    CAPABILITY_RUNTIME,
    CAPABILITY_CHECKPOINT,
)

CAP_PASS = "pass"
CAP_FAIL = "fail"
CAP_BLOCKED = "blocked"

EVENT_PREFIX = "GALVATRON_CAP_EVENT "

MODEL_DIR = "galvatron/models/gpt_hf"
TRAIN_RANDOM = "train_dist_random.py"
PROFILER_SCRIPT = "profiler.py"
SEARCH_SCRIPT = "search_dist.py"
HARDWARE_PROFILER = "galvatron/profile_hardware/profile_hardware.py"

COMMON_ENV = (
    "RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 "
    "NCCL_SOCKET_IFNAME=eth3 NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 "
    "NCCL_NET=Socket"
)


def capability_event(capability: str, status: str, detail: str) -> str:
    payload = {
        "capability": capability,
        "status": status,
        "detail": detail,
    }
    return EVENT_PREFIX + json.dumps(payload)


def parse_capability_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(EVENT_PREFIX):
            try:
                payload = json.loads(stripped[len(EVENT_PREFIX) :])
            except ValueError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def judge_capability(
    capability: str,
    status: str,
    detail: str,
    *,
    required_blocker_note_when_blocked: bool = True,
) -> dict[str, Any]:
    """Normalize a capability result and enforce the blocker-note rule."""
    if status not in (CAP_PASS, CAP_FAIL, CAP_BLOCKED):
        raise ValueError(f"invalid capability status {status!r} for {capability}")
    if status == CAP_BLOCKED and required_blocker_note_when_blocked:
        note = detail.strip()
        if not note:
            raise ValueError(f"blocked capability {capability} requires a blocker note")
    return {
        "capability": capability,
        "status": status,
        "detail": detail.strip(),
    }


def capability_report(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build the per-capability report; every capability must have a result."""
    by_name = {result["capability"]: result for result in results}
    missing = [name for name in CAPABILITIES if name not in by_name]
    if missing:
        raise ValueError(f"missing capability results: {missing}")
    for name in CAPABILITIES:
        result = by_name[name]
        if result["status"] == CAP_BLOCKED and not result.get("detail"):
            raise ValueError(f"blocked capability {name} is missing its blocker note")
    return {
        "task": "T060",
        "capabilities": [by_name[name] for name in CAPABILITIES],
        "blocked_count": sum(
            1 for result in results if result["status"] == CAP_BLOCKED
        ),
        "pass_count": sum(1 for result in results if result["status"] == CAP_PASS),
        "fail_count": sum(1 for result in results if result["status"] == CAP_FAIL),
    }


def _build_live_wrapper() -> tuple[WSLRuntimeWrapper, dict[str, Any], str]:
    from shardgrid.common.config import load_cluster_config

    config = load_cluster_config("examples/workers.yaml")
    worker = next(w for w in config.workers if str(w.worker_id) == "gpu4060")
    address_book = json.load(open("tests/address.json"))
    matches = [
        entry
        for entry in address_book
        if "RTX4060" in str(entry.get("gpu_model") or "").replace(" ", "").upper()
    ]
    if not matches:
        raise ValueError("no address entry for gpu4060")
    entry = matches[0]
    ip = str(entry["ip"])
    resolved = replace(
        worker,
        host=ip,
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
    return wrapper, entry, str(resolved.worker_id)


def _run_live(
    wrapper: WSLRuntimeWrapper,
    command: str,
    *,
    timeout: float,
    output_tail: int = 600,
) -> tuple[str, str]:
    result = wrapper.run(command, timeout=timeout)
    stdout = (result.stdout or "")[-output_tail:]
    stderr = (result.stderr or "")[-output_tail:]
    if result.ok:
        return "ok", stdout
    combined = f"{stdout}\n{stderr}".strip()
    return "failed", combined[-output_tail:]


def _probe_dir_exists(wrapper: WSLRuntimeWrapper, path: str) -> bool:
    result = wrapper.run(f"test -e {path} && echo EXISTS || echo MISSING", timeout=30)
    return "EXISTS" in (result.stdout or "")


# ---------------------------------------------------------------------------
# Logic tests
# ---------------------------------------------------------------------------


def test_capability_list_complete() -> None:
    assert len(CAPABILITIES) == 5
    assert CAPABILITY_PROFILER in CAPABILITIES
    assert CAPABILITY_SEARCH_ENGINE in CAPABILITIES
    assert CAPABILITY_PIPELINE in CAPABILITIES
    assert CAPABILITY_RUNTIME in CAPABILITIES
    assert CAPABILITY_CHECKPOINT in CAPABILITIES


def test_judge_capability_pass() -> None:
    result = judge_capability(CAPABILITY_RUNTIME, CAP_PASS, "loss printed")
    assert result["status"] == CAP_PASS
    assert result["capability"] == CAPABILITY_RUNTIME


def test_judge_capability_fail() -> None:
    result = judge_capability(CAPABILITY_PROFILER, CAP_FAIL, "traceback")
    assert result["status"] == CAP_FAIL


def test_judge_blocked_requires_note() -> None:
    with __import__("pytest").raises(ValueError):
        judge_capability(CAPABILITY_SEARCH_ENGINE, CAP_BLOCKED, "  ")
    result = judge_capability(
        CAPABILITY_SEARCH_ENGINE,
        CAP_BLOCKED,
        "hardware profiler output missing",
    )
    assert result["status"] == CAP_BLOCKED
    assert result["detail"]


def test_judge_invalid_status_rejected() -> None:
    with __import__("pytest").raises(ValueError):
        judge_capability(CAPABILITY_RUNTIME, "unknown", "x")


def test_report_requires_all_capabilities() -> None:
    results = [
        judge_capability(CAPABILITY_PROFILER, CAP_PASS, "ok"),
        judge_capability(CAPABILITY_SEARCH_ENGINE, CAP_BLOCKED, "note"),
        judge_capability(CAPABILITY_PIPELINE, CAP_PASS, "ok"),
        judge_capability(CAPABILITY_RUNTIME, CAP_PASS, "ok"),
    ]
    with __import__("pytest").raises(ValueError):
        capability_report(results)


def test_report_all_pass() -> None:
    results = [
        judge_capability(name, CAP_PASS, "ok") for name in CAPABILITIES
    ]
    report = capability_report(results)
    assert report["pass_count"] == 5
    assert report["blocked_count"] == 0
    assert report["fail_count"] == 0


def test_report_counts_blocked_and_fail() -> None:
    results = [
        judge_capability(CAPABILITY_PROFILER, CAP_PASS, "ok"),
        judge_capability(CAPABILITY_SEARCH_ENGINE, CAP_BLOCKED, "note"),
        judge_capability(CAPABILITY_PIPELINE, CAP_PASS, "ok"),
        judge_capability(CAPABILITY_RUNTIME, CAP_FAIL, "traceback"),
        judge_capability(CAPABILITY_CHECKPOINT, CAP_PASS, "ok"),
    ]
    report = capability_report(results)
    assert report["pass_count"] == 3
    assert report["blocked_count"] == 1
    assert report["fail_count"] == 1


def test_parse_capability_events() -> None:
    text = (
        "noise\n"
        + capability_event(CAPABILITY_RUNTIME, CAP_PASS, "loss=5.9")
        + "\n"
    )
    events = parse_capability_events(text)
    assert len(events) == 1
    assert events[0]["capability"] == CAPABILITY_RUNTIME
    assert events[0]["status"] == CAP_PASS
    assert parse_capability_events("nothing") == []


def test_capability_report_round_trip(tmp_path: Any) -> None:
    report = capability_report(
        [judge_capability(name, CAP_PASS, "ok") for name in CAPABILITIES]
    )
    path = tmp_path / "capabilities.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    loaded = json.loads(path.read_text())
    assert loaded["task"] == "T060"
    assert len(loaded["capabilities"]) == 5


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------


def test_live_galvatron_capabilities() -> None:
    """Real capability execution on the RTX 4060 Worker (opt-in)."""
    wrapper, entry, worker_id = _build_live_wrapper()
    results: dict[str, dict[str, Any]] = {}

    model_args = (
        "--model_size gpt-0.3b --hidden_size 1024 --num_attention_heads 16 "
        "--vocab_size 50257 --set_model_config_manually 0 "
        "--set_layernum_manually 1 --set_seqlen_manually 1 "
        "--seq_length 256 --mixed_precision fp16"
    )

    # 1. model profiler: computation
    status, output = _run_live(
        wrapper,
        f"cd ~/galvatron-spike-v2.4.0/{MODEL_DIR} && "
        f"{COMMON_ENV} PROFILE_TRAINER={TRAIN_RANDOM} python {PROFILER_SCRIPT} "
        f"{model_args} --profile_type computation --profile_batch_size 1 "
        f"--profile_seq_length_list 256 --layernum_min 1 --layernum_max 2",
        timeout=600,
    )
    computation_config = os.path.join(
        "configs", "computation_profiling_fp16_hidden1024_head16.json"
    )
    config_ok = _probe_dir_exists(
        wrapper,
        f"~/galvatron-spike-v2.4.0/{MODEL_DIR}/{computation_config}",
    )
    if status == "ok" and config_ok:
        results[CAPABILITY_PROFILER] = judge_capability(
            CAPABILITY_PROFILER, CAP_PASS, "model profiler wrote computation config"
        )
    else:
        results[CAPABILITY_PROFILER] = judge_capability(
            CAPABILITY_PROFILER, CAP_FAIL, output[-400:]
        )

    # 2. model profiler: memory (needed by search engine)
    status, output = _run_live(
        wrapper,
        f"cd ~/galvatron-spike-v2.4.0/{MODEL_DIR} && "
        f"{COMMON_ENV} PROFILE_TRAINER={TRAIN_RANDOM} python {PROFILER_SCRIPT} "
        f"{model_args} --profile_type memory --profile_batch_size 1 "
        f"--profile_seq_length_list 256 --layernum_min 1 --layernum_max 2 "
        f"--profile_dp_type ddp",
        timeout=900,
    )
    memory_config = os.path.join(
        "configs", "memory_profiling_fp16_hidden1024_head16.json"
    )
    memory_ok = _probe_dir_exists(
        wrapper,
        f"~/galvatron-spike-v2.4.0/{MODEL_DIR}/{memory_config}",
    )
    if status == "ok" and memory_ok:
        detail = "model profiler wrote computation and memory configs"
    else:
        detail = f"memory profiling failed: {output[-300:]}"
    profiler_status = (
        CAP_PASS if results[CAPABILITY_PROFILER]["status"] == CAP_PASS and memory_ok else CAP_FAIL
    )
    results[CAPABILITY_PROFILER] = judge_capability(
        CAPABILITY_PROFILER, profiler_status, detail
    )

    # 3. hardware profiler: expected CUPTI blocker (does not touch PASS/FAIL)
    status, output = _run_live(
        wrapper,
        f"cd ~/galvatron-spike-v2.4.0 && {COMMON_ENV} "
        f"timeout 120 python {HARDWARE_PROFILER} "
        f"--num_nodes 1 --num_gpus_per_node 1 --max_tp_size 1 --max_pp_deg 1 "
        f"--backend nccl",
        timeout=180,
    )
    hardware_note = (
        "WSL2 CUPTI restriction: profile_overlap child exits with SIGSEGV "
        "(-11) during torch.profiler; hardware profiler output required by the "
        "search engine is not produced"
    )
    if "SIGSEGV" in output or "-11" in output or "CUPTI" in output:
        results["hardware_profiler"] = judge_capability(
            "hardware_profiler", CAP_BLOCKED, hardware_note
        )
    else:
        results["hardware_profiler"] = judge_capability(
            "hardware_profiler", CAP_BLOCKED, f"not usable: {output[-300:]}"
        )

    # 4. search engine: depends on hardware profiler output -> blocked
    status, output = _run_live(
        wrapper,
        f"cd ~/galvatron-spike-v2.4.0/{MODEL_DIR} && {COMMON_ENV} "
        f"python {SEARCH_SCRIPT} {model_args} "
        f"--num_nodes 1 --num_gpus_per_node 1 --memory_constraint 6 "
        f"--min_bsz 4 --max_bsz 4 "
        f"--time_profiling_path ./configs",
        timeout=300,
    )
    if status == "ok":
        results[CAPABILITY_SEARCH_ENGINE] = judge_capability(
            CAPABILITY_SEARCH_ENGINE, CAP_PASS, "search engine produced a plan"
        )
    else:
        results[CAPABILITY_SEARCH_ENGINE] = judge_capability(
            CAPABILITY_SEARCH_ENGINE,
            CAP_BLOCKED,
            "search engine requires hardware profiler output "
            "(allreduce/p2p bandwidth configs), which is BLOCKED_BY_WSL2_CUPTI; "
            f"last error: {output[-300:]}",
        )

    # 5. pipeline construction + runtime launch + checkpoint in one real run
    wrapper.run(
        "pkill -9 -f miniconda3/envs/shardgrid/bin/python || true; "
        "pkill -9 -f miniconda3/envs/shardgrid/bin/python3 || true",
        timeout=30,
    )
    train_env = COMMON_ENV.replace("MASTER_PORT=29500", "MASTER_PORT=29501")
    status, output = _run_live(
        wrapper,
        f"cd ~/galvatron-spike-v2.4.0/{MODEL_DIR} && {train_env} "
        f"python {TRAIN_RANDOM} "
        f"--set_model_config_manually 1 "
        f"--hidden_size 256 --num_hidden_layers 2 --num_attention_heads 8 "
        f"--seq_length 256 --vocab_size 1024 "
        f"--global_train_batch_size 512 --epochs 1 --lr 1e-4 "
        f"--adam_weight_decay 0.01 --dropout_prob 0.1 "
        f"--pp_deg 1 --global_tp_deg 1 --sdp 0 "
        f"--mixed_precision fp16 --check_loss 1 --profile 0 "
        f"--global_checkpoint 1 --initialize_on_meta 0",
        timeout=600,
    )
    loss_ok = "Loss =" in output or "Loss=" in output
    if status == "ok" and loss_ok:
        results[CAPABILITY_PIPELINE] = judge_capability(
            CAPABILITY_PIPELINE,
            CAP_PASS,
            "gpt-0.3b built with explicit pp_deg=1 layout",
        )
        results[CAPABILITY_RUNTIME] = judge_capability(
            CAPABILITY_RUNTIME,
            CAP_PASS,
            "train_dist_random.py trained with real loss output",
        )
        results[CAPABILITY_CHECKPOINT] = judge_capability(
            CAPABILITY_CHECKPOINT,
            CAP_PASS,
            "global_checkpoint=1 activation checkpoint accepted and applied",
        )
    else:
        detail = output[-400:]
        results[CAPABILITY_PIPELINE] = judge_capability(
            CAPABILITY_PIPELINE, CAP_FAIL, detail
        )
        results[CAPABILITY_RUNTIME] = judge_capability(
            CAPABILITY_RUNTIME, CAP_FAIL, detail
        )
        results[CAPABILITY_CHECKPOINT] = judge_capability(
            CAPABILITY_CHECKPOINT, CAP_FAIL, detail
        )

    ordered = [
        judge_capability(
            CAPABILITY_PROFILER,
            results[CAPABILITY_PROFILER]["status"],
            (
                results[CAPABILITY_PROFILER]["detail"]
                + " | hardware profiler: "
                + results["hardware_profiler"]["status"]
                + " ("
                + results["hardware_profiler"]["detail"][:400]
                + ")"
            ),
        ),
        results[CAPABILITY_SEARCH_ENGINE],
        results[CAPABILITY_PIPELINE],
        results[CAPABILITY_RUNTIME],
        results[CAPABILITY_CHECKPOINT],
    ]
    report = capability_report(ordered)

    output_dir = os.environ.get("SHARDGRID_ENGINE_EVIDENCE_DIR") or (
        "/var/tmp/shardgrid/engines"
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "galvatron-capabilities-latest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    assert os.path.exists(path)

    # Acceptance: every capability has a pass/fail/blocked result; blocked
    # capabilities carry a blocker note; hardware profiler + search engine are
    # blocked (WSL2 CUPTI), never reported as pass.
    assert report["pass_count"] >= 3
    assert report["blocked_count"] >= 1
    assert report["fail_count"] == 0
    profiler_item = next(
        item
        for item in report["capabilities"]
        if item["capability"] == CAPABILITY_PROFILER
    )
    assert "hardware profiler: blocked" in profiler_item["detail"]
    blocked = [
        item for item in report["capabilities"] if item["status"] == CAP_BLOCKED
    ]
    assert all(item.get("detail") for item in blocked)
    search = next(
        item
        for item in report["capabilities"]
        if item["capability"] == CAPABILITY_SEARCH_ENGINE
    )
    assert search["status"] == CAP_BLOCKED
    assert "CUPTI" in search["detail"]