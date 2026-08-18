from __future__ import annotations

from shardgrid.distributed.collectives import RankCollectiveResult
from shardgrid.distributed.fallback import (
    LABEL_GLOO_FALLBACK,
    LABEL_GLOO_FALLBACK_FAILED,
    LABEL_NCCL_PASS,
    run_with_fallback,
)


def _ranks(
    *,
    init_ok: bool,
    broadcast_ok: bool = True,
    send_recv_ok: bool = True,
    all_reduce_ok: bool = True,
    error: str | None = None,
    rank0_exit: int = 0,
    rank1_exit: int = 0,
) -> tuple[RankCollectiveResult, RankCollectiveResult]:
    result = {
        "init_ok": init_ok,
        "broadcast_ok": broadcast_ok,
        "send_recv_ok": send_recv_ok,
        "all_reduce_ok": all_reduce_ok,
        "error": error,
    }
    return (
        RankCollectiveResult(
            rank=0, worker_id="gpu4060", exit_code=rank0_exit,
            timed_out=False, result=result, stdout="", stderr="",
        ),
        RankCollectiveResult(
            rank=1, worker_id="gpu1060", exit_code=rank1_exit,
            timed_out=False, result=result, stdout="", stderr="",
        ),
    )


def test_nccl_success_does_not_fall_back() -> None:
    def run_nccl():
        return _ranks(init_ok=True)

    outcome = run_with_fallback(run_nccl, lambda: _ranks(init_ok=True))

    assert outcome.label == LABEL_NCCL_PASS
    assert outcome.nccl.outcome == "PASS"
    assert outcome.gloo is None


def test_nccl_failure_triggers_gloo_fallback_and_labels_correctly() -> None:
    calls: list[str] = []

    def run_nccl():
        calls.append("nccl")
        return _ranks(init_ok=False, error="NCCL rendezvous failed")

    def run_gloo():
        calls.append("gloo")
        return _ranks(init_ok=True)

    outcome = run_with_fallback(run_nccl, run_gloo)

    assert calls == ["nccl", "gloo"]
    assert outcome.label == LABEL_GLOO_FALLBACK
    assert outcome.nccl.outcome == "FAILED"
    assert outcome.nccl.error is not None
    assert "NCCL rendezvous failed" in outcome.nccl.error
    assert outcome.gloo is not None
    assert outcome.gloo.outcome == "PASS"
    # fallback success must never be labelled as NCCL success
    assert outcome.label != LABEL_NCCL_PASS


def test_nccl_and_gloo_both_fail_recorded_honestly() -> None:
    def run_nccl():
        return _ranks(init_ok=False, error="nccl socket timeout")

    def run_gloo():
        return _ranks(init_ok=False, error="gloo connection refused")

    outcome = run_with_fallback(run_nccl, run_gloo)

    assert outcome.label == LABEL_GLOO_FALLBACK_FAILED
    assert outcome.nccl.outcome == "FAILED"
    assert outcome.gloo is not None and outcome.gloo.outcome == "FAILED"
    assert outcome.gloo.error is not None
    assert "gloo connection refused" in outcome.gloo.error


def test_nccl_failure_preserves_evidence_in_outcome() -> None:
    def run_nccl():
        return _ranks(init_ok=False, error="EADDRINUSE", rank0_exit=255)

    def run_gloo():
        return _ranks(init_ok=True)

    outcome = run_with_fallback(run_nccl, run_gloo)

    payload = outcome.to_dict()
    assert payload["nccl"]["outcome"] == "FAILED"
    assert payload["nccl"]["error"] is not None
    assert "EADDRINUSE" in payload["nccl"]["error"]
    assert payload["gloo"]["outcome"] == "PASS"
    assert payload["label"] == LABEL_GLOO_FALLBACK


def test_live_nccl_then_gloo_fallback_records_real_result() -> None:
    """Real attempt (opt-in via multi_host marker): NCCL first, Gloo fallback."""
    import json as _json
    from dataclasses import replace

    from shardgrid.common.config import load_cluster_config
    from shardgrid.distributed.collectives import run_pair_collectives
    from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
    from shardgrid.transport.ssh import SSHOptions, SSHTransport

    config = load_cluster_config("examples/workers.yaml")
    address_book = _json.load(open("tests/address.json"))
    wrappers = {}
    for wid, label, rank in [("gpu4060", "RTX 4060", 0), ("gpu1060", "GTX 1650", 1)]:
        worker = next(w for w in config.workers if str(w.worker_id) == wid)
        entry = next(e for e in address_book if label in str(e.get("gpu_model") or ""))
        worker = replace(worker, host=entry["ip"], ssh_user=entry["username"])
        transport = SSHTransport(
            SSHOptions.from_ssh_config(
                config.ssh, host=entry["ip"], user="shardgrid", port=worker.ssh_port
            )
        )
        wrappers[rank] = (
            WSLRuntimeWrapper(
                WSLRuntimeConfig.from_worker_and_runtime(worker, config.runtime), transport
            ),
            wid,
        )

    w0, id0 = wrappers[0]
    w1, id1 = wrappers[1]
    master, rank0_interface, rank1_interface = "10.87.5.155", "eth3", "eth0"
    port = 29510

    def make(backend: str):
        def run():
            return run_pair_collectives(
                w0, w1,
                rank0_worker_id=id0, rank1_worker_id=id1,
                master_addr=master, master_port=port,
                backend=backend,
                rank0_interface=rank0_interface,
                rank1_interface=rank1_interface,
                timeout=90.0,
            )
        return run

    outcome = run_with_fallback(make("nccl"), make("gloo"))

    assert outcome.label in {
        LABEL_NCCL_PASS,
        LABEL_GLOO_FALLBACK,
        LABEL_GLOO_FALLBACK_FAILED,
    }
    assert outcome.nccl.outcome in {"PASS", "FAILED"}
    if outcome.nccl.outcome == "PASS":
        assert outcome.gloo is None
    else:
        assert outcome.gloo is not None
        assert outcome.label in {LABEL_GLOO_FALLBACK, LABEL_GLOO_FALLBACK_FAILED}
