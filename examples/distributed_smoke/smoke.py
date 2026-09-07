"""Minimal, parameterized torch.distributed smoke program (T046).

This program uses only official PyTorch Distributed APIs.  It validates launch
arguments, initializes a process group over ``tcp://`` rendezvous, records
runtime evidence (Conda/Python/PyTorch/backend/rank), and destroys the process
group on exit.  The four torch.distributed call sites are thin module-level
functions so tests can mock init/barrier/destroy without importing torch on a
machine that has no torch installed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

VALID_BACKENDS = ("nccl", "gloo")


class SmokeArgumentError(ValueError):
    """Raised when distributed smoke arguments are invalid."""


@dataclass(frozen=True)
class SmokeArguments:
    rank: int
    world_size: int
    master_addr: str
    master_port: int
    backend: str
    local_rank: int = 0


def parse_args(argv: Sequence[str] | None = None) -> SmokeArguments:
    parser = argparse.ArgumentParser(
        prog="distributed-smoke",
        description="ShardGrid minimal torch.distributed smoke program",
    )
    parser.add_argument("--rank", type=int, required=True, help="global rank")
    parser.add_argument("--world-size", type=int, required=True, help="world size")
    parser.add_argument("--master-addr", required=True, help="master address")
    parser.add_argument(
        "--master-port", type=int, default=29500, help="master port"
    )
    parser.add_argument(
        "--backend", choices=list(VALID_BACKENDS), default="gloo", help="backend"
    )
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parsed = parser.parse_args(argv)
    return validate_arguments(
        SmokeArguments(
            rank=parsed.rank,
            world_size=parsed.world_size,
            master_addr=parsed.master_addr,
            master_port=parsed.master_port,
            backend=parsed.backend,
            local_rank=parsed.local_rank,
        )
    )


def validate_arguments(smoke_args: SmokeArguments) -> SmokeArguments:
    if smoke_args.world_size < 1:
        raise SmokeArgumentError(f"world_size must be >= 1, got {smoke_args.world_size}")
    if smoke_args.rank < 0 or smoke_args.rank >= smoke_args.world_size:
        raise SmokeArgumentError(
            f"rank must be in [0, world_size), got rank={smoke_args.rank} "
            f"world_size={smoke_args.world_size}"
        )
    if not smoke_args.master_addr:
        raise SmokeArgumentError("master_addr must not be empty")
    if not (0 < smoke_args.master_port < 65536):
        raise SmokeArgumentError(
            f"master_port must be in (0, 65536), got {smoke_args.master_port}"
        )
    if smoke_args.backend not in VALID_BACKENDS:
        raise SmokeArgumentError(
            f"backend must be one of {VALID_BACKENDS}, got {smoke_args.backend}"
        )
    if smoke_args.local_rank < 0:
        raise SmokeArgumentError(
            f"local_rank must be >= 0, got {smoke_args.local_rank}"
        )
    return smoke_args


def _torch_versions() -> tuple[str | None, str | None]:
    try:
        import torch  # type: ignore[import-not-found]

        return torch.__version__, str(torch.version.cuda)
    except Exception:
        return None, None


def collect_runtime_evidence(smoke_args: SmokeArguments) -> dict[str, Any]:
    torch_version, torch_cuda_version = _torch_versions()
    return {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "backend": smoke_args.backend,
        "rank": smoke_args.rank,
        "world_size": smoke_args.world_size,
        "local_rank": smoke_args.local_rank,
        "master_addr": smoke_args.master_addr,
        "master_port": smoke_args.master_port,
    }


def _init_process_group(
    backend: str,
    init_method: str,
    rank: int,
    world_size: int,
) -> None:
    import torch.distributed as dist  # type: ignore[import-not-found]

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def _barrier() -> None:
    import torch.distributed as dist  # type: ignore[import-not-found]

    dist.barrier()


def _is_initialized() -> bool:
    import torch.distributed as dist  # type: ignore[import-not-found]

    return bool(dist.is_initialized())


def _destroy_process_group() -> None:
    import torch.distributed as dist  # type: ignore[import-not-found]

    dist.destroy_process_group()


def init_and_run(
    smoke_args: SmokeArguments,
    *,
    init_fn: Callable[..., Any] = _init_process_group,
    barrier_fn: Callable[[], Any] = _barrier,
    is_initialized_fn: Callable[[], bool] = _is_initialized,
    destroy_fn: Callable[[], Any] = _destroy_process_group,
) -> dict[str, Any]:
    evidence = collect_runtime_evidence(smoke_args)
    init_method = f"tcp://{smoke_args.master_addr}:{smoke_args.master_port}"
    try:
        init_fn(
            backend=smoke_args.backend,
            init_method=init_method,
            rank=smoke_args.rank,
            world_size=smoke_args.world_size,
        )
    except Exception as exc:
        return {"ok": False, "stage": "init", "error": str(exc), "evidence": evidence}
    try:
        barrier_fn()
        return {"ok": True, "stage": "ok", "evidence": evidence}
    finally:
        if is_initialized_fn():
            destroy_fn()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        smoke_args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    except SmokeArgumentError as exc:
        print(f"distributed-smoke: {exc}", file=sys.stderr)
        return 2
    result = init_and_run(smoke_args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())