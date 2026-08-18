from __future__ import annotations

import shlex

import pytest

from shardgrid.common.config import RuntimeConfig, WorkerConfig
from shardgrid.distributed.runner import (
    LaunchError,
    build_launch_plan,
    remote_launch_command,
    validate_rank,
)
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper

SMOKE = "examples/distributed_smoke/smoke.py"


def _worker(
    *,
    worker_id: str = "gpu4060",
    host: str = "10.87.5.155",
    conda_prefix: str | None = "/home/shardgrid/miniconda3/envs/shardgrid",
    conda_environment: str | None = "shardgrid",
    runtime_distro: str | None = "Ubuntu",
) -> WorkerConfig:
    payload: dict[str, str | None] = {
        "id": worker_id,
        "machine_id": "machine-c",
        "physical_os": "windows",
        "runtime_os": "wsl2_linux",
        "runtime": "wsl2",
        "host": host,
        "ssh_user": "shardgrid",
        "runtime_distro": runtime_distro,
        "conda_environment": conda_environment,
        "conda_prefix": conda_prefix,
    }
    return WorkerConfig.from_dict(payload)


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(
        default_wsl_distro="Ubuntu",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        conda_environment="shardgrid",
    )


def test_rank0_command_generation() -> None:
    plan = build_launch_plan(
        [
            _worker(worker_id="gpu4060", host="10.87.5.155"),
            _worker(worker_id="gpu1060", host="10.87.5.15"),
        ],
        runtime=_runtime(),
        smoke_program=SMOKE,
        master_addr="10.87.5.155",
        master_port=29500,
        backend="gloo",
    )

    rank0 = plan.launches[0]
    assert rank0.worker_id == "gpu4060"
    assert rank0.rank == 0
    assert rank0.local_rank == 0
    assert rank0.local_world_size == 1
    assert rank0.python_executable == "/home/shardgrid/miniconda3/envs/shardgrid/bin/python"
    assert rank0.argv[:2] == (
        "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        SMOKE,
    )
    assert "--rank" in rank0.argv
    assert rank0.argv[rank0.argv.index("--rank") + 1] == "0"
    assert "--master-addr" in rank0.argv
    assert "10.87.5.155" in rank0.argv


def test_rank1_command_generation() -> None:
    plan = build_launch_plan(
        [_worker(worker_id="gpu4060"), _worker(worker_id="gpu1060")],
        runtime=_runtime(),
        smoke_program=SMOKE,
        master_addr="10.87.5.155",
    )

    rank1 = plan.launches[1]
    assert rank1.worker_id == "gpu1060"
    assert rank1.rank == 1
    assert rank1.local_rank == 0
    assert rank1.world_size == 2
    rank_index = rank1.argv.index("--rank")
    world_index = rank1.argv.index("--world-size")
    assert rank1.argv[rank_index + 1] == "1"
    assert rank1.argv[world_index + 1] == "2"


def test_rendezvous_parameters_consistent_across_ranks() -> None:
    plan = build_launch_plan(
        [_worker(worker_id="gpu4060"), _worker(worker_id="gpu1060")],
        runtime=_runtime(),
        smoke_program=SMOKE,
        master_addr="10.87.5.155",
        master_port=29500,
        backend="nccl",
    )

    assert plan.world_size == 2
    assert plan.master_addr == "10.87.5.155"
    assert plan.backend == "nccl"
    for launch in plan.launches:
        assert "10.87.5.155" in launch.argv
        assert "29500" in launch.argv
        assert "nccl" in launch.argv


def test_each_worker_uses_its_own_conda_runtime() -> None:
    plan = build_launch_plan(
        [
            _worker(
                worker_id="gpu4060",
                conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            ),
            _worker(
                worker_id="gpu1060",
                conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            ),
        ],
        runtime=_runtime(),
        smoke_program=SMOKE,
        master_addr="10.87.5.155",
    )

    assert plan.launches[0].conda_environment == "shardgrid"
    assert plan.launches[1].conda_prefix == "/home/shardgrid/miniconda3/envs/shardgrid"
    assert plan.launches[0].python_executable.startswith(
        plan.launches[0].conda_prefix or ""
    )


def test_log_paths_are_independent_per_rank() -> None:
    plan = build_launch_plan(
        [_worker(worker_id="gpu4060"), _worker(worker_id="gpu1060")],
        runtime=_runtime(),
        smoke_program=SMOKE,
        master_addr="10.87.5.155",
        log_dir="/var/tmp/shardgrid/logs",
    )

    assert plan.launches[0].log_path == "/var/tmp/shardgrid/logs/rank-0.log"
    assert plan.launches[1].log_path == "/var/tmp/shardgrid/logs/rank-1.log"
    assert plan.launches[0].log_path != plan.launches[1].log_path


def test_no_manual_rank_startup() -> None:
    plan = build_launch_plan(
        [_worker(worker_id="gpu4060"), _worker(worker_id="gpu1060")],
        runtime=_runtime(),
        smoke_program=SMOKE,
        master_addr="10.87.5.155",
    )

    assert len(plan.launches) == 2
    # both ranks are fully specified; no user must supply rank/world-size manually
    for launch in plan.launches:
        assert "--rank" in launch.argv
        assert "--world-size" in launch.argv


def test_invalid_rank_rejected() -> None:
    with pytest.raises(LaunchError, match="rank"):
        validate_rank(2, 2)
    with pytest.raises(LaunchError, match="world_size"):
        validate_rank(0, 0)


def test_empty_workers_rejected() -> None:
    with pytest.raises(LaunchError, match="at least one worker"):
        build_launch_plan([], runtime=_runtime(), smoke_program=SMOKE)


def test_missing_conda_runtime_rejected() -> None:
    with pytest.raises(LaunchError, match="Conda prefix"):
        build_launch_plan(
            [
                _worker(worker_id="gpu4060", conda_prefix=None, conda_environment=None),
            ],
            runtime=RuntimeConfig(),
            smoke_program=SMOKE,
            master_addr="10.87.5.155",
        )


def test_invalid_rendezvous_rejected() -> None:
    with pytest.raises(LaunchError, match="master_port"):
        build_launch_plan(
            [_worker(worker_id="gpu4060")],
            runtime=_runtime(),
            smoke_program=SMOKE,
            master_port=0,
        )
    with pytest.raises(LaunchError, match="backend"):
        build_launch_plan(
            [_worker(worker_id="gpu4060")],
            runtime=_runtime(),
            smoke_program=SMOKE,
            backend="mpi",
        )


def test_command_serialization_round_trips() -> None:
    plan = build_launch_plan(
        [_worker(worker_id="gpu4060")],
        runtime=_runtime(),
        smoke_program=SMOKE,
        master_addr="10.87.5.155",
    )

    argv = plan.launches[0].argv
    serialized = shlex.join(argv)
    assert shlex.split(serialized) == list(argv)


def test_remote_launch_command_uses_t040_wrapper() -> None:
    plan = build_launch_plan(
        [_worker(worker_id="gpu4060")],
        runtime=_runtime(),
        smoke_program=SMOKE,
        master_addr="10.87.5.155",
    )
    launch = plan.launches[0]
    wrapper = WSLRuntimeWrapper(
        WSLRuntimeConfig(
            distro="Ubuntu",
            user="shardgrid",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        ),
        None,  # type: ignore[arg-type]
    )

    import base64
    import re

    remote = remote_launch_command(wrapper, launch)

    assert remote.startswith("powershell -NoProfile -EncodedCommand ")
    encoded = remote.split("EncodedCommand ", 1)[1]
    decoded = base64.b64decode(encoded).decode("utf-16le")
    assert "wsl.exe" in decoded
    payload_b64 = re.search(r"\$payload = '([A-Za-z0-9+/=]+)'", decoded)
    assert payload_b64 is not None
    payload = base64.b64decode(payload_b64.group(1)).decode("utf-8")
    assert "distributed_smoke/smoke.py" in payload