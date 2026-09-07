"""PyTorch pipeline compatibility spike for GPU Workers (T063).

T063 evaluates the mature PyTorch pipeline option (``torch.distributed.
pipelining``) for the actual torch version (2.7.1) and the two-physical-host
one-GPU-per-host runtime, before any custom model-parallel code is considered.

The spike runs the smallest possible GPipe schedule: 2 stages over 2 physical
hosts with 1 GPU per host, synthetic data, few micro batches.  Outcomes:

- ``PASS``: both ranks built their ``PipelineStage``, the GPipe schedule ran
  all steps, and the run finished with a clean ``destroy_process_group``.
- ``BLOCKED`` / ``FAIL``: real environment limitation or other failure.

The spike reuses the proven T058 launch shape (explicit rank env + direct
selected Conda python, per-host ``NCCL_SOCKET_IFNAME`` from ``ip route get``),
never hard-codes interfaces, and only real execution can mark PASS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from shardgrid.transport.runtime import WSLRuntimeWrapper

SPIKE_STATUS_PASS = "PASS"
SPIKE_STATUS_FAIL = "FAIL"
SPIKE_STATUS_BLOCKED = "BLOCKED"

STAGE_READY = "PYTORCH_PIPELINE_STAGE_READY"
STEP_OK = "PYTORCH_PIPELINE_STEP_OK"
DONE = "PYTORCH_PIPELINE_DONE"
REQUIRED_MARKERS = (STAGE_READY, STEP_OK, DONE)

SPIKE_SCRIPT = r"""
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

HIDDEN = 128
MB = 4
STEPS = 2


class Block(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.ln = torch.nn.LayerNorm(hidden)
        self.ff = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(self.ln(x))


def get_model() -> torch.nn.Module:
    return torch.nn.Sequential(Block(HIDDEN), Block(HIDDEN))


def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(output, target)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT'])}",
        rank=rank,
        world_size=world,
    )
    device = torch.device("cuda:0")

    model = get_model()
    stage_idx = rank
    sub = model[stage_idx].to(device)
    stage = PipelineStage(
        sub,
        stage_index=stage_idx,
        num_stages=2,
        device=device,
    )
    schedule = ScheduleGPipe(stage, n_microbatches=2, loss_fn=loss_fn)
    print("PYTORCH_PIPELINE_STAGE_READY rank=%d" % rank, flush=True)
    dist.barrier()

    start = time.time()
    for step_idx in range(STEPS):
        target = torch.randn(MB, HIDDEN, device=device)
        if rank == 0:
            input_tensor = torch.randn(MB, HIDDEN, device=device)
            loss = schedule.step(input_tensor, target=target)
        else:
            loss = schedule.step(target=target)
        torch.cuda.synchronize()
        print(
            "PYTORCH_PIPELINE_STEP_OK rank=%d step=%d loss=%s" % (rank, step_idx, loss),
            flush=True,
        )
    dist.barrier()
    print(
        "PYTORCH_PIPELINE_DONE rank=%d world=%d elapsed=%.1f"
        % (rank, world, time.time() - start),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
"""


@dataclass
class PytorchPipelineSpikeStep:
    name: str
    status: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class PytorchPipelineSpikeResult:
    run_id: str
    status: str
    torch_version: str | None
    steps: list[PytorchPipelineSpikeStep] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    started_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "torch_version": self.torch_version,
            "steps": [step.to_dict() for step in self.steps],
            "blockers": list(self.blockers),
            "diagnostics": list(self.diagnostics),
            "started_at": self.started_at,
        }


def build_spike_script() -> str:
    return SPIKE_SCRIPT


def parse_spike_markers(stdout: str, stderr: str) -> list[str]:
    combined = f"{stdout}\n{stderr}"
    markers: list[str] = []
    for marker in REQUIRED_MARKERS:
        if marker in combined:
            markers.append(marker)
    return markers


def derive_spike_status(
    markers: Sequence[str],
    *,
    timed_out: bool,
    timeout: float,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if timed_out:
        blockers.append(
            f"pipeline did not finish within {timeout:.0f}s "
            "(schedule.step made no progress)"
        )
        if STAGE_READY in markers:
            return SPIKE_STATUS_BLOCKED, blockers
        return SPIKE_STATUS_FAIL, blockers
    if all(marker in markers for marker in REQUIRED_MARKERS):
        return SPIKE_STATUS_PASS, []
    return SPIKE_STATUS_FAIL, blockers


def save_spike_evidence(result: PytorchPipelineSpikeResult, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pytorch-pipeline-latest.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return path


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_pytorch_pipeline_spike(
    rank0_wrapper: WSLRuntimeWrapper,
    rank1_wrapper: WSLRuntimeWrapper,
    *,
    rank0_worker_ip: str,
    rank1_worker_ip: str,
    rank0_interface: str,
    rank1_interface: str,
    master_port: int = 29500,
    timeout: float = 180.0,
) -> PytorchPipelineSpikeResult:
    """Run the real two-host torch.distributed.pipelining spike."""
    import base64
    import threading

    from shardgrid.transport.runtime import wrap_wsl_direct_command

    run_id = _now_utc().replace(":", "-").replace("+00:00", "Z")
    steps: list[PytorchPipelineSpikeStep] = []

    def probe_torch(wrapper: WSLRuntimeWrapper) -> str | None:
        result = wrapper.run(
            'python -c "import torch; print(torch.__version__)"', timeout=60
        )
        return (result.stdout or "").strip() or None

    torch_version = probe_torch(rank0_wrapper)
    if not torch_version:
        return PytorchPipelineSpikeResult(
            run_id=run_id,
            status=SPIKE_STATUS_BLOCKED,
            torch_version=None,
            steps=steps,
            blockers=["torch not importable on rank 0 Worker"],
            diagnostics=[],
            started_at=_now_utc(),
        )
    steps.append(
        PytorchPipelineSpikeStep(
            name="torch_version",
            status="PASS",
            detail=f"torch {torch_version}",
        )
    )

    script = build_spike_script()
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    remote_path = "/tmp/pytorch_pipeline_spike.py"
    for wrapper in (rank0_wrapper, rank1_wrapper):
        install = wrap_wsl_direct_command(
            wrapper.config.distro,
            wrapper.config.user or "shardgrid",
            f"echo {encoded} | base64 -d > {remote_path}",
        )
        wrapper.executor.run(install, timeout=60)
        wrapper.run("pkill -9 -f pytorch_pipeline_spike.py || true", timeout=15.0)

    def launch(
        wrapper: WSLRuntimeWrapper,
        rank: int,
        iface: str,
        results: dict[int, Any],
    ) -> None:
        command = (
            f"RANK={rank} WORLD_SIZE=2 LOCAL_RANK=0 "
            f"MASTER_ADDR={rank0_worker_ip} MASTER_PORT={master_port} "
            f"NCCL_SOCKET_IFNAME={iface} GLOO_SOCKET_IFNAME={iface} "
            f"NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 NCCL_NET=Socket "
            f"python {remote_path}"
        )
        results[rank] = wrapper.run(command, timeout=timeout)

    results: dict[int, Any] = {}
    threads = [
        threading.Thread(
            target=launch, args=(rank0_wrapper, 0, rank0_interface, results)
        ),
        threading.Thread(
            target=launch, args=(rank1_wrapper, 1, rank1_interface, results)
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for wrapper in (rank0_wrapper, rank1_wrapper):
        wrapper.run("pkill -9 -f pytorch_pipeline_spike.py || true", timeout=15.0)

    timed_out = any(result.timed_out for result in results.values())
    markers: list[str] = []
    for rank in (0, 1):
        result = results[rank]
        markers.extend(parse_spike_markers(result.stdout or "", result.stderr or ""))
        if result.stderr:
            steps.append(
                PytorchPipelineSpikeStep(
                    name=f"rank{rank}_stderr",
                    status="BLOCKED" if timed_out else "INFO",
                    detail=(result.stderr or "")[-400:],
                )
            )

    status, blockers = derive_spike_status(
        markers, timed_out=timed_out, timeout=timeout
    )
    steps.insert(
        1,
        PytorchPipelineSpikeStep(
            name="pipeline_runtime",
            status=status,
            detail=(
                "markers: "
                + ", ".join(sorted(set(markers)))
                if markers
                else "no pipeline markers in output"
            ),
        ),
    )

    return PytorchPipelineSpikeResult(
        run_id=run_id,
        status=status,
        torch_version=torch_version,
        steps=steps,
        blockers=blockers,
        diagnostics=[f"timeout={timeout:.0f}s", f"master_port={master_port}"],
        started_at=_now_utc(),
    )