from __future__ import annotations

import json

import pytest
from examples.distributed_smoke.smoke import (
    SmokeArgumentError,
    SmokeArguments,
    collect_runtime_evidence,
    init_and_run,
    main,
    parse_args,
    validate_arguments,
)


def _args(
    *,
    rank: int = 0,
    world_size: int = 2,
    master_addr: str = "10.87.5.155",
    master_port: int = 29500,
    backend: str = "gloo",
    local_rank: int = 0,
) -> SmokeArguments:
    return SmokeArguments(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
        local_rank=local_rank,
    )


def test_parse_args_valid(monkeypatch) -> None:
    parsed = parse_args(
        [
            "--rank",
            "1",
            "--world-size",
            "2",
            "--master-addr",
            "10.87.5.155",
            "--master-port",
            "29500",
            "--backend",
            "nccl",
            "--local-rank",
            "1",
        ]
    )

    assert parsed.rank == 1
    assert parsed.world_size == 2
    assert parsed.master_addr == "10.87.5.155"
    assert parsed.master_port == 29500
    assert parsed.backend == "nccl"
    assert parsed.local_rank == 1


def test_invalid_rank_rejected() -> None:
    with pytest.raises(SmokeArgumentError, match="rank"):
        validate_arguments(_args(rank=2, world_size=2))
    with pytest.raises(SmokeArgumentError, match="rank"):
        validate_arguments(_args(rank=-1, world_size=2))


def test_invalid_world_size_rejected() -> None:
    with pytest.raises(SmokeArgumentError, match="world_size"):
        validate_arguments(_args(world_size=0))


def test_invalid_backend_rejected() -> None:
    with pytest.raises(SmokeArgumentError, match="backend"):
        validate_arguments(_args(backend="mpi"))


def test_missing_master_address_rejected() -> None:
    with pytest.raises(SmokeArgumentError, match="master_addr"):
        validate_arguments(_args(master_addr=""))


def test_invalid_master_port_rejected() -> None:
    with pytest.raises(SmokeArgumentError, match="master_port"):
        validate_arguments(_args(master_port=0))
    with pytest.raises(SmokeArgumentError, match="master_port"):
        validate_arguments(_args(master_port=70000))


def test_runtime_evidence_contains_expected_fields(monkeypatch) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "shardgrid")
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/shardgrid")
    monkeypatch.setattr(
        "examples.distributed_smoke.smoke._torch_versions",
        lambda: ("2.7.1+cu118", "11.8"),
    )

    evidence = collect_runtime_evidence(_args())

    assert evidence["conda_environment"] == "shardgrid"
    assert evidence["conda_prefix"] == "/opt/conda/envs/shardgrid"
    assert evidence["python_executable"]
    assert evidence["python_version"]
    assert evidence["torch_version"] == "2.7.1+cu118"
    assert evidence["torch_cuda_version"] == "11.8"
    assert evidence["backend"] == "gloo"
    assert evidence["rank"] == 0
    assert evidence["world_size"] == 2
    assert evidence["local_rank"] == 0


def test_init_and_run_success_calls_init_barrier_destroy(monkeypatch) -> None:
    calls: list[str] = []

    def fake_init(backend, init_method, rank, world_size) -> None:
        calls.append(f"init:{backend}:{init_method}:{rank}:{world_size}")

    def fake_barrier() -> None:
        calls.append("barrier")

    def fake_destroy() -> None:
        calls.append("destroy")

    monkeypatch.setattr(
        "examples.distributed_smoke.smoke._torch_versions",
        lambda: ("2.7.1", None),
    )

    result = init_and_run(
        _args(),
        init_fn=fake_init,
        barrier_fn=fake_barrier,
        is_initialized_fn=lambda: True,
        destroy_fn=fake_destroy,
    )

    assert result["ok"] is True
    assert result["stage"] == "ok"
    assert result["evidence"]["rank"] == 0
    assert "init:gloo:tcp://10.87.5.155:29500:0:2" in calls
    assert "barrier" in calls
    assert "destroy" in calls


def test_init_and_run_init_failure_is_structured(monkeypatch) -> None:
    def fake_init(backend, init_method, rank, world_size) -> None:
        raise RuntimeError("connection refused")

    result = init_and_run(_args(), init_fn=fake_init, is_initialized_fn=lambda: False)

    assert result["ok"] is False
    assert result["stage"] == "init"
    assert "connection refused" in result["error"]
    assert result["evidence"]["backend"] == "gloo"


def test_init_and_run_cleans_up_on_barrier_failure(monkeypatch) -> None:
    calls: list[str] = []

    def fake_barrier() -> None:
        calls.append("barrier")
        raise RuntimeError("barrier failed")

    def fake_destroy() -> None:
        calls.append("destroy")

    with pytest.raises(RuntimeError, match="barrier failed"):
        init_and_run(
            _args(),
            init_fn=lambda **kwargs: None,
            barrier_fn=fake_barrier,
            is_initialized_fn=lambda: True,
            destroy_fn=fake_destroy,
        )

    assert "destroy" in calls


def test_main_success_outputs_json(monkeypatch, capsys) -> None:
    def fake_init_and_run(smoke_args):
        return {
            "ok": True,
            "stage": "ok",
            "evidence": collect_runtime_evidence(smoke_args),
        }

    monkeypatch.setattr(
        "examples.distributed_smoke.smoke.init_and_run", fake_init_and_run
    )
    monkeypatch.setattr(
        "examples.distributed_smoke.smoke._torch_versions",
        lambda: ("2.7.1", "11.8"),
    )

    exit_code = main(
        [
            "--rank",
            "0",
            "--world-size",
            "2",
            "--master-addr",
            "10.87.5.155",
            "--backend",
            "gloo",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["evidence"]["rank"] == 0


def test_main_invalid_arguments_fail_explicitly(capsys) -> None:
    exit_code = main(
        ["--rank", "5", "--world-size", "2", "--master-addr", "10.87.5.155"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "distributed-smoke" in captured.err
    assert "rank" in captured.err