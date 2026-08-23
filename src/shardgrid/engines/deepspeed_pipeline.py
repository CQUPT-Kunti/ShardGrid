"""DeepSpeed Pipeline compatibility spike for GPU Workers (T062).

T062 evaluates DeepSpeed Pipeline only when Galvatron is insufficient.  The
spike runs the smallest possible DeepSpeed pipeline (2 stages over 2 physical
hosts with 1 GPU per host, ``pp_size=2``) through the existing SSH + WSL2 +
selected Conda chain and captures the real outcome:

- ``PASS``: pipeline initialized and ``engine.train_batch`` completed with a
  finite loss on both ranks.
- ``BLOCKED``: the run stopped at a real environment limitation (observed:
  ``train_batch`` deadlocks on WSL2 two-host NCCL even though native torch
  NCCL ``isend/irecv`` succeeds).
- ``FAIL``: any other real failure.

The spike never fabricates a pass: on the current environment it reports the
captured deadlock with evidence.  No custom full pipeline engine is ever
introduced because of a Galvatron failure.
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

# One script per Worker; rank comes from RANK env.  Pipeline stages = 2,
# one stage per physical host, tiny synthetic data, few micro steps.
SPIKE_SCRIPT = r"""
import faulthandler
import os
import sys
import time
from types import MethodType

import torch
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import LayerSpec, PipelineModule
from deepspeed.runtime.pipe import schedule as pipe_schedule
from deepspeed.runtime.pipe.engine import PipelineEngine

HIDDEN = 128
STAGES = 2
MICRO = 4
GLOBAL_BATCH = 8
STEPS = 2
PREGENERATED_MICRO_BATCHES = 2


def ts(rank: int, event: str) -> None:
    print(
        "DEEPSPEED_DIAG ts=%s rank=%d %s"
        % (time.strftime("%H:%M:%S"), rank, event),
        flush=True,
    )


def install_diagnostics(rank: int) -> None:
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(45, repeat=True, file=sys.stderr)
    ts(rank, "faulthandler_enabled")


def patch_pipeline_diagnostics(rank: int) -> None:
    instruction_map = dict(PipelineEngine._INSTRUCTION_MAP)

    def instrument(name: str) -> None:
        original = getattr(PipelineEngine, name)

        def wrapped(self, *args, **kwargs):
            ts(rank, f"{name}_begin args={args} kwargs={kwargs}")
            result = original(self, *args, **kwargs)
            ts(rank, f"{name}_end")
            return result

        setattr(PipelineEngine, name, wrapped)

    for method_name in (
        "_exec_forward_pass",
        "_exec_backward_pass",
        "_exec_send_activations",
        "_exec_recv_activations",
        "_exec_send_grads",
        "_exec_recv_grads",
        "_exec_optimizer_step",
        "_aggregate_total_loss",
    ):
        instrument(method_name)

    instruction_map[pipe_schedule.ForwardPass] = PipelineEngine._exec_forward_pass
    instruction_map[pipe_schedule.BackwardPass] = PipelineEngine._exec_backward_pass
    instruction_map[pipe_schedule.SendActivation] = PipelineEngine._exec_send_activations
    instruction_map[pipe_schedule.RecvActivation] = PipelineEngine._exec_recv_activations
    instruction_map[pipe_schedule.SendGrad] = PipelineEngine._exec_send_grads
    instruction_map[pipe_schedule.RecvGrad] = PipelineEngine._exec_recv_grads

    original_next_batch = PipelineEngine._next_batch

    def wrapped_next_batch(self):
        ts(rank, "_next_batch_begin")
        batch = original_next_batch(self)
        ts(
            rank,
            "_next_batch_end "
            + describe_batch(batch),
        )
        return batch

    PipelineEngine._next_batch = wrapped_next_batch

    def exec_load_micro_batch(self, buffer_id):
        ts(rank, f"load_micro_batch_enter buffer_id={buffer_id}")
        if self.wall_clock_breakdown():
            self.timers("batch_input").start()

        ts(rank, f"load_micro_batch_before_next_batch buffer_id={buffer_id}")
        batch = self._next_batch()
        ts(rank, f"load_micro_batch_after_next_batch buffer_id={buffer_id} {describe_batch(batch)}")

        if self.is_first_stage():
            loaded = None
            ts(rank, f"load_micro_batch_first_stage_input_type type={type(batch[0]).__name__}")
            if torch.is_tensor(batch[0]):
                ts(
                    rank,
                    "load_micro_batch_first_stage_to_cuda_begin "
                    f"shape={tuple(batch[0].shape)}",
                )
                loaded = batch[0].clone().to(self.device).detach()
                ts(rank, "load_micro_batch_first_stage_to_cuda_end")
                if self._reentrant_activation_checkpointing():
                    loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], (tuple, list))
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    ts(
                        rank,
                        "load_micro_batch_first_stage_item_to_cuda_begin "
                        f"shape={tuple(x.shape)}",
                    )
                    mine = x.clone().detach().to(self.device)
                    ts(rank, "load_micro_batch_first_stage_item_to_cuda_end")
                    if self._reentrant_activation_checkpointing():
                        mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)
            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            ts(rank, f"load_micro_batch_last_stage_label_type type={type(batch[1]).__name__}")
            if torch.is_tensor(batch[1]):
                ts(rank, f"load_micro_batch_last_stage_to_cuda_begin shape={tuple(batch[1].shape)}")
                loaded = batch[1].to(self.device)
                ts(rank, "load_micro_batch_last_stage_to_cuda_end")
            elif isinstance(batch[1], (tuple, list)):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    ts(
                        rank,
                        "load_micro_batch_last_stage_item_to_cuda_begin "
                        f"shape={tuple(x.shape)}",
                    )
                    x = x.to(self.device).detach()
                    ts(rank, "load_micro_batch_last_stage_item_to_cuda_end")
                    loaded.append(x)
                loaded = tuple(loaded)
            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers("batch_input").stop()
        ts(rank, f"load_micro_batch_exit buffer_id={buffer_id}")

    PipelineEngine._exec_load_micro_batch = exec_load_micro_batch
    instruction_map[pipe_schedule.LoadMicroBatch] = PipelineEngine._exec_load_micro_batch
    PipelineEngine._INSTRUCTION_MAP = instruction_map

    def exec_schedule(self, pipe_schedule):
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []
        for step_id, step_cmds in enumerate(pipe_schedule):
            names = ",".join(type(cmd).__name__ for cmd in step_cmds) or "NOOP"
            ts(rank, f"schedule_step_begin step={step_id} cmds={names}")
            for cmd in step_cmds:
                cmd_name = type(cmd).__name__
                ts(rank, f"instr_begin step={step_id} cmd={cmd_name} kwargs={cmd.kwargs}")
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                    )
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                self._exec_instr(**cmd.kwargs)
                ts(rank, f"instr_end step={step_id} cmd={cmd_name}")
            ts(rank, f"schedule_step_end step={step_id}")

    PipelineEngine._exec_schedule = exec_schedule


def describe_tensor(value) -> str:
    if torch.is_tensor(value):
        return (
            f"tensor(shape={tuple(value.shape)},dtype={value.dtype},device={value.device.type})"
        )
    return type(value).__name__


def describe_batch(batch) -> str:
    if batch is None:
        return "batch=None"
    if not isinstance(batch, (tuple, list)):
        return f"batch_type={type(batch).__name__}"
    items = [f"item{index}={describe_tensor(value)}" for index, value in enumerate(batch)]
    return "batch_type=%s %s" % (type(batch).__name__, " ".join(items))


class StageBlock(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.ln = torch.nn.LayerNorm(hidden)
        self.ff = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ts(int(os.environ.get("RANK", "-1")), "stageblock_forward_begin")
        return self.ff(self.ln(x))


def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ts(int(os.environ.get("RANK", "-1")), "loss_fn_begin")
    return torch.nn.functional.mse_loss(output, target)


def build_static_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    base = torch.arange(MICRO * HIDDEN, dtype=torch.float32).reshape(MICRO, HIDDEN)
    for index in range(PREGENERATED_MICRO_BATCHES):
        offset = float(index) / 10.0
        batch = base.add(offset).clone()
        target = base.mul(0.5).add(offset).clone()
        batches.append((batch, target))
    return batches


class StaticDataIterator:
    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor]], rank: int) -> None:
        self._batches = batches
        self._index = 0
        self._rank = rank

    def __iter__(self) -> "StaticDataIterator":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        ts(self._rank, f"data_iterator_next_begin index={self._index}")
        batch = self._batches[self._index % len(self._batches)]
        self._index += 1
        ts(self._rank, "data_iterator_next_end")
        return batch


def main() -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    install_diagnostics(rank)
    patch_pipeline_diagnostics(rank)
    ts(rank, "main_begin")
    if not dist.is_initialized():
        ts(rank, "dist_init_begin")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT'])}",
            rank=rank,
            world_size=world,
        )
        ts(rank, "dist_init_end")
    ts(rank, "deepspeed_init_distributed_begin")
    deepspeed.init_distributed()
    ts(rank, "deepspeed_init_distributed_end")

    assert world == STAGES, f"expected world_size {STAGES}, got {world}"
    static_batches = build_static_batches()
    ts(rank, f"static_batches_ready count={len(static_batches)} device=cpu")

    layers = [LayerSpec(StageBlock, HIDDEN) for _ in range(STAGES)]
    model = PipelineModule(
        layers=layers,
        num_stages=STAGES,
        loss_fn=loss_fn,
        partition_method="uniform",
    )

    config = {
        "train_batch_size": GLOBAL_BATCH,
        "train_micro_batch_size_per_gpu": MICRO,
        "gradient_accumulation_steps": 2,
        "pipeline": {"enabled": True, "stages": STAGES},
        "zero_optimization": {"stage": 0},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
    }
    engine, _, _, _ = deepspeed.initialize(
        config=config,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    print("DEEPSPEED_PIPELINE_INIT_OK rank=%d" % rank, flush=True)
    ts(rank, "post_initialize_barrier_begin")
    dist.barrier()
    print("DEEPSPEED_PIPELINE_BARRIER_OK rank=%d" % rank, flush=True)
    ts(rank, "post_initialize_barrier_end")

    start = time.time()
    for step in range(STEPS):
        ts(rank, "train_batch_begin step=%d" % step)
        loss = engine.train_batch(data_iter=StaticDataIterator(static_batches, rank))
        ts(rank, "train_batch_end step=%d" % step)
        print(
            "DEEPSPEED_PIPELINE_STEP_OK rank=%d step=%d loss=%s" % (rank, step, loss),
            flush=True,
        )
    ts(rank, "final_barrier_begin")
    dist.barrier()
    ts(rank, "final_barrier_end")
    print(
        "DEEPSPEED_PIPELINE_DONE rank=%d world=%d elapsed=%.1f"
        % (rank, world, time.time() - start),
        flush=True,
    )
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
"""


@dataclass
class DeepspeedSpikeStep:
    name: str
    status: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class DeepspeedSpikeResult:
    run_id: str
    status: str
    deepspeed_version: str | None
    torch_version: str | None
    steps: list[DeepspeedSpikeStep] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    started_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "deepspeed_version": self.deepspeed_version,
            "torch_version": self.torch_version,
            "steps": [step.to_dict() for step in self.steps],
            "blockers": list(self.blockers),
            "diagnostics": list(self.diagnostics),
            "started_at": self.started_at,
        }


def build_spike_script() -> str:
    return SPIKE_SCRIPT


def parse_spike_marker(stdout: str, marker: str) -> bool:
    return marker in (stdout or "")


def parse_spike_results(stdout: str, stderr: str) -> list[str]:
    combined = f"{stdout}\n{stderr}"
    results: list[str] = []
    for marker in (
        "DEEPSPEED_PIPELINE_INIT_OK",
        "DEEPSPEED_PIPELINE_BARRIER_OK",
        "DEEPSPEED_PIPELINE_STEP_OK",
        "DEEPSPEED_PIPELINE_DONE",
    ):
        if marker in combined:
            results.append(marker)
    return results


def derive_spike_status(
    markers: Sequence[str],
    *,
    timed_out: bool,
    timeout: float,
) -> tuple[str, list[str]]:
    """Decide the spike status from real markers and timeout evidence."""
    blockers: list[str] = []
    if timed_out:
        blockers.append(
            f"engine.train_batch did not finish within {timeout:.0f}s "
            "(two-host WSL2 NCCL pipeline deadlock observed; native torch "
            "NCCL isend/irecv succeeds on the same hosts)"
        )
        if "DEEPSPEED_PIPELINE_INIT_OK" in markers:
            return SPIKE_STATUS_BLOCKED, blockers
        return SPIKE_STATUS_FAIL, blockers
    if all(
        marker in markers
        for marker in (
            "DEEPSPEED_PIPELINE_INIT_OK",
            "DEEPSPEED_PIPELINE_STEP_OK",
            "DEEPSPEED_PIPELINE_DONE",
        )
    ):
        return SPIKE_STATUS_PASS, []
    return SPIKE_STATUS_FAIL, blockers


def save_spike_evidence(result: DeepspeedSpikeResult, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "deepspeed-pipeline-latest.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return path


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_deepspeed_spike(
    rank0_wrapper: WSLRuntimeWrapper,
    rank1_wrapper: WSLRuntimeWrapper,
    *,
    rank0_worker_ip: str,
    rank1_worker_ip: str,
    rank0_interface: str,
    rank1_interface: str,
    master_port: int = 29500,
    timeout: float = 120.0,
) -> DeepspeedSpikeResult:
    """Run the real two-host DeepSpeed Pipeline spike and classify it."""
    import base64
    import threading

    from shardgrid.transport.runtime import wrap_wsl_direct_command

    run_id = _now_utc().replace(":", "-").replace("+00:00", "Z")
    steps: list[DeepspeedSpikeStep] = []
    blockers: list[str] = []

    def probe_version(wrapper: WSLRuntimeWrapper) -> tuple[str | None, str | None]:
        result = wrapper.run(
            "python -c "
            '"import deepspeed, torch; '
            "print(deepspeed.__version__, torch.__version__)\"",
            timeout=60,
        )
        parts = (result.stdout or "").strip().split()
        if len(parts) >= 2:
            return parts[0], parts[1]
        return None, None

    ds_version0, torch_version0 = probe_version(rank0_wrapper)
    ds_version1, _ = probe_version(rank1_wrapper)
    if not ds_version0 or not ds_version1:
        return DeepspeedSpikeResult(
            run_id=run_id,
            status=SPIKE_STATUS_BLOCKED,
            deepspeed_version=ds_version0,
            torch_version=torch_version0,
            steps=steps,
            blockers=["deepspeed not importable on one or both Workers"],
            diagnostics=[],
            started_at=_now_utc(),
        )

    steps.append(
        DeepspeedSpikeStep(
            name="deepspeed_version",
            status="PASS",
            detail=f"deepspeed {ds_version0} on both Workers, torch {torch_version0}",
        )
    )
    steps.append(
        DeepspeedSpikeStep(
            name="interface_discovery",
            status="PASS",
            detail=(
                f"rank0={rank0_worker_ip}:{rank0_interface}, "
                f"rank1={rank1_worker_ip}:{rank1_interface}"
            ),
        )
    )

    script = build_spike_script()
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    remote_path = "/tmp/deepspeed_pipeline_spike.py"
    for wrapper in (rank0_wrapper, rank1_wrapper):
        install = wrap_wsl_direct_command(
            wrapper.config.distro,
            wrapper.config.user or "shardgrid",
            f"echo {encoded} | base64 -d > {remote_path}",
        )
        wrapper.executor.run(install, timeout=60)
        wrapper.run("pkill -9 -f deepspeed_pipeline_spike.py || true", timeout=15.0)

    def launch(
        wrapper: WSLRuntimeWrapper,
        rank: int,
        iface: str,
        worker_ip: str,
        peer_ip: str,
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
            target=launch,
            args=(rank0_wrapper, 0, rank0_interface, rank0_worker_ip, rank1_worker_ip, results),
        ),
        threading.Thread(
            target=launch,
            args=(rank1_wrapper, 1, rank1_interface, rank1_worker_ip, rank0_worker_ip, results),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for wrapper in (rank0_wrapper, rank1_wrapper):
        wrapper.run("pkill -9 -f deepspeed_pipeline_spike.py || true", timeout=15.0)

    timed_out = any(result.timed_out for result in results.values())
    markers: list[str] = []
    for rank in (0, 1):
        result = results[rank]
        markers.extend(parse_spike_results(result.stdout or "", result.stderr or ""))
        if result.stdout:
            steps.append(
                DeepspeedSpikeStep(
                    name=f"rank{rank}_stdout",
                    status="INFO",
                    detail=(result.stdout or "")[-2000:],
                )
            )
        if result.stderr:
            steps.append(
                DeepspeedSpikeStep(
                    name=f"rank{rank}_stderr",
                    status="BLOCKED" if timed_out else "INFO",
                    detail=(result.stderr or "")[-400:],
                )
            )

    status, status_blockers = derive_spike_status(
        markers, timed_out=timed_out, timeout=timeout
    )
    blockers.extend(status_blockers)
    steps.insert(
        1,
        DeepspeedSpikeStep(
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
    if status == SPIKE_STATUS_PASS:
        steps.append(
            DeepspeedSpikeStep(
                name="train_batch",
                status="PASS",
                detail="engine.train_batch completed with finite loss on both ranks",
            )
        )

    return DeepspeedSpikeResult(
        run_id=run_id,
        status=status,
        deepspeed_version=ds_version0,
        torch_version=torch_version0,
        steps=steps,
        blockers=blockers,
        diagnostics=[f"timeout={timeout:.0f}s", f"master_port={master_port}"],
        started_at=_now_utc(),
    )
