"""Two-host runner for the supported model (T071/T072).

Each physical Worker launches this script with explicit rank environment and
loads exactly its own stage of the T069/T070 two-stage split onto its local
``cuda:0``:

- rank 0 (RTX 4060): Stage0 (embed/pos/block0)
- rank 1 (GTX 1650): Stage1 (block1/ln/lm_head)

The script always reports a ``STAGE_PLACEMENT_EVIDENCE`` JSON line with
hostname, rank/world_size, GPU name, CUDA device, stage id, parameter counts,
trainable parameter count, the full parameter-name ownership list, and the
per-parameter device.

`SHARDGRID_PIPELINE_TASK=t071`:
    placement-only verification for T071

`SHARDGRID_PIPELINE_TASK=t072`:
    real two-host forward for T072
    rank0: input -> Stage0 -> activation -> send
    rank1: recv activation -> Stage1 -> logits -> loss

`SHARDGRID_PIPELINE_TASK=t074`:
    real multi-iteration training for T074
    rank0: zero_grad -> Stage0 forward -> send activation -> recv grad ->
           Stage0 backward -> optimizer.step
    rank1: recv activation -> Stage1 forward -> loss -> Stage1 backward ->
           send grad -> optimizer.step
    Each rank optimizes only its own stage parameters, records the loss
    history (rank1), proves parameter updates, and saves/reloads a per-rank
    checkpoint (model state + optimizer state + step/iteration).

Usage (per Worker, rank from environment):

    RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=<rank0 ip> MASTER_PORT=29500 \\
        NCCL_SOCKET_IFNAME=<iface> python examples/models/train_pipeline.py
"""

from __future__ import annotations

import ctypes
import faulthandler
import hashlib
import inspect
import json
import math
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from examples.models.minimal_transformer import MinimalTransformerConfig
from examples.models.stage0 import build_stage0
from examples.models.stage1 import build_stage1

EVENT_MARKER = "STAGE_PLACEMENT_EVIDENCE "
FORWARD_MARKER = "T072_FORWARD_EVIDENCE "
BACKWARD_MARKER = "T073_BACKWARD_EVIDENCE "
TRAIN_MARKER = "T074_TRAIN_EVIDENCE "
TASK_ENV = "SHARDGRID_PIPELINE_TASK"
TASK_T071 = "t071"
TASK_T072 = "t072"
TASK_T073 = "t073"
TASK_T074 = "t074"
T072_BATCH_SIZE = 2
T072_SEQ_LENGTH = 8
T074_DEFAULT_STEPS = 20
T074_DEFAULT_LR = 1e-3
PLAN_PATH = Path(__file__).resolve().with_name("static_parallel_plan.yaml")

MODEL_CONFIG = MinimalTransformerConfig(
    vocab_size=1024, hidden_size=128, num_hidden_layers=2,
    num_attention_heads=4, max_seq_length=64,
)
_LOG_HANDLES: list[object] = []
_CUDA_RUNTIME = None


class _TeeStream:
    def __init__(self, *streams: object) -> None:
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        first = self._streams[0] if self._streams else None
        return bool(first is not None and hasattr(first, "isatty") and first.isatty())


def _emit_line(message: str) -> None:
    sys.__stdout__.write(message + "\n")
    sys.__stdout__.flush()
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _configure_rank_log(rank: int) -> Path:
    path_text = os.environ.get("SHARDGRID_LOG_PATH", "").strip()
    if path_text:
        path = Path(path_text)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        path = Path(f"/tmp/t072_model_entry_rank{rank}_full_{stamp}.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8", buffering=1)
    _LOG_HANDLES.append(handle)
    sys.stdout = _TeeStream(sys.__stdout__, handle)
    sys.stderr = _TeeStream(sys.__stderr__, handle)
    print(f"T072_LOG_PATH {path}", flush=True)
    return path


def _log_running_file_info(rank: int) -> None:
    _mark("RUNNING_FILE_INFO")
    _debug_breakpoint(rank, "RUNNING_FILE_INFO")
    running_file = str(Path(__file__).resolve())
    cwd = str(Path.cwd())
    repo_root = str(Path(__file__).resolve().parents[2])
    _emit_line(f"RUNNING_FILE={running_file}")
    _diag_line(
        "running_file_info",
        rank=rank,
        running_file=running_file,
        cwd=cwd,
        repo_root=repo_root,
    )
def _torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype name {name!r}") from error


def _load_t072_runtime_plan() -> dict[str, object]:
    raw = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("static plan must be a mapping")
    stages_raw = raw.get("stages") or []
    if not isinstance(stages_raw, list):
        raise ValueError("static plan stages must be a list")
    boundary = ((raw.get("tensor_boundary") or {}).get("activation") or {})
    if not isinstance(boundary, dict):
        raise ValueError("activation boundary metadata missing from static plan")
    stage_by_rank = {
        int(stage["rank"]): {
            "id": str(stage["id"]),
            "rank": int(stage["rank"]),
            "parameter_count": int(stage.get("parameter_count", 0)),
            "worker_id": stage.get("worker_id"),
        }
        for stage in stages_raw
    }
    stage_by_id = {stage["id"]: stage for stage in stage_by_rank.values()}
    producer_id = str(boundary["producer_stage"])
    consumer_id = str(boundary["consumer_stage"])
    if os.environ.get("SHARDGRID_T072_SWAP_ROLES", "").strip() == "1":
        stage_by_rank = {
            0: {**stage_by_id["stage1"], "rank": 0},
            1: {**stage_by_id["stage0"], "rank": 1},
        }
        stage_by_id = {stage["id"]: stage for stage in stage_by_rank.values()}
    producer_rank = int(stage_by_id[producer_id]["rank"])
    consumer_rank = int(stage_by_id[consumer_id]["rank"])
    shape_tokens = tuple(boundary["shape"])
    shape = tuple(
        T072_BATCH_SIZE if token == "batch"
        else T072_SEQ_LENGTH if token == "seq"
        else int(token)
        for token in shape_tokens
    )
    return {
        "world_size": int(raw["world_size"]),
        "stage_by_rank": stage_by_rank,
        "producer_id": producer_id,
        "consumer_id": consumer_id,
        "producer_rank": producer_rank,
        "consumer_rank": consumer_rank,
        "activation_shape": shape,
        "activation_dtype_name": str(boundary["dtype"]),
        "activation_dtype": _torch_dtype_from_name(str(boundary["dtype"])),
    }


def _dtype_name(dtype: torch.dtype) -> str:
    text = str(dtype)
    return text.replace("torch.", "")


def _timestamp() -> float:
    return time.time()


def _mark(name: str) -> float:
    stamp = _timestamp()
    rank = os.environ.get("RANK", "?")
    host = socket.gethostname()
    _emit_line(name)
    time_text = time.strftime("%H:%M:%S", time.localtime(stamp))
    millis = int((stamp % 1) * 1000)
    _emit_line(
        f"[RANK={rank}][HOST={host}]"
        f"[TIME={time_text}.{millis:03d}][MARKER={name}]"
    )
    return stamp


def _debug_breakpoint(rank: int, marker: str) -> float:
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    filename = Path(caller.f_code.co_filename).name if caller is not None else "unknown"
    lineno = caller.f_lineno if caller is not None else 0
    stamp = _timestamp()
    _emit_line(
        f"[RANK={rank}][{stamp:.6f}][{marker}][{filename}:{lineno}]"
        f"[HOST={socket.gethostname()}]"
    )
    del frame
    return stamp


def _rank_mark(rank: int, name: str) -> float:
    stamp = _timestamp()
    print(
        f"[RANK{rank}][{stamp:.6f}][{name}] host={socket.gethostname()}",
        flush=True,
    )
    return stamp


def _fail_stage(rank: int, stage: str, error: BaseException) -> None:
    payload = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": _timestamp(),
    }
    _emit_line("FAIL_STAGE " + json.dumps(payload, sort_keys=True))


def _diag_line(event: str, **extra: object) -> float:
    stamp = _timestamp()
    payload = {"event": event, "timestamp": stamp}
    payload.update(extra)
    _emit_line("T072_DIAG " + json.dumps(payload, sort_keys=True))
    return stamp


def _runtime_info(rank: int, local_rank: int) -> None:
    _mark("RUN_INFO")
    _debug_breakpoint(rank, "RUN_INFO")
    _mark("RUNTIME_INFO_BEGIN")
    _debug_breakpoint(rank, "RUNTIME_INFO_BEGIN")
    payload: dict[str, object] = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "running_file": str(Path(__file__).resolve()),
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        payload.update(
            {
                "device_count": int(torch.cuda.device_count()),
                "current_device": int(torch.cuda.current_device()),
                "gpu_name": torch.cuda.get_device_name(local_rank),
            }
        )
    _diag_line("runtime_info", **payload)
    _mark("RUNTIME_INFO_END")
    _debug_breakpoint(rank, "RUNTIME_INFO_END")


def _activation_basics(hidden: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(hidden.shape),
        "dtype": _dtype_name(hidden.dtype),
        "device": str(hidden.device),
        "numel": int(hidden.numel()),
        "requires_grad": bool(hidden.requires_grad),
        "is_contiguous": bool(hidden.is_contiguous()),
        "stride": list(hidden.stride()),
        "grad_fn": type(hidden.grad_fn).__name__ if hidden.grad_fn is not None else None,
        "is_leaf": bool(hidden.is_leaf),
        "data_ptr": int(hidden.data_ptr()),
    }


def _current_stream_info(device: torch.device) -> dict[str, object]:
    stream = torch.cuda.current_stream(device=device)
    return {
        "device": str(device),
        "current_device": int(torch.cuda.current_device()),
        "stream_repr": str(stream),
        "stream_type": type(stream).__name__,
        "stream_cuda_stream": int(getattr(stream, "cuda_stream", 0)),
        "stream_priority": int(getattr(stream, "priority", 0)),
    }


def _op_log(
    *,
    rank: int,
    op_id: str,
    stage: str,
    device: torch.device,
    tensor: torch.Tensor,
) -> None:
    payload = {
        "rank": rank,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "op_id": op_id,
        "stage": stage,
    }
    payload.update(_current_stream_info(device))
    payload.update(_activation_basics(tensor))
    _emit_line("[OP_TRACE] " + json.dumps(payload, sort_keys=True))


def _tensor_checksum(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu", copy=True).contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _model_param_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        cpu = parameter.detach().to("cpu", copy=True).contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _state_dict_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(_state_dict_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            return False
        return all(_state_dict_equal(a, b) for a, b in zip(left, right))
    return left == right


def _activation_check(name: str, tensor: torch.Tensor) -> dict[str, object]:
    detached = tensor.detach()
    return {
        "name": name,
        "shape": list(detached.shape),
        "dtype": _dtype_name(detached.dtype),
        "device": str(detached.device),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()),
        "abs_sum": float(detached.abs().sum().item()),
        "checksum": _tensor_checksum(detached),
    }


def _activation_probe(name: str, tensor: torch.Tensor) -> dict[str, object]:
    _mark(f"{name.upper()}_BEGIN")
    payload = _activation_check(name, tensor)
    _diag_line(name, **payload)
    _mark(f"{name.upper()}_END")
    return payload


def _numpy_dtype_for_torch(dtype: torch.dtype) -> np.dtype:
    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int64: np.int64,
        torch.int32: np.int32,
    }
    try:
        return np.dtype(mapping[dtype])
    except KeyError as error:
        raise ValueError(f"unsupported numpy conversion dtype {dtype!r}") from error


def _cuda_runtime() -> ctypes.CDLL:
    global _CUDA_RUNTIME
    if _CUDA_RUNTIME is not None:
        return _CUDA_RUNTIME
    pattern = Path(sys.prefix).glob(
        "lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"
    )
    try:
        library_path = next(pattern)
    except StopIteration as error:
        raise RuntimeError("libcudart.so not found in current environment") from error
    runtime = ctypes.CDLL(str(library_path))
    runtime.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    runtime.cudaMemcpy.restype = ctypes.c_int
    runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
    runtime.cudaGetErrorString.restype = ctypes.c_char_p
    _CUDA_RUNTIME = runtime
    return runtime


def _tensor_to_cpu(source: torch.Tensor) -> torch.Tensor:
    detached = source.detach()
    if detached.device.type != "cuda":
        return detached.to("cpu", copy=True)
    contiguous = detached.contiguous()
    numpy_array = np.empty(tuple(contiguous.shape), dtype=_numpy_dtype_for_torch(contiguous.dtype))
    runtime = _cuda_runtime()
    error = runtime.cudaMemcpy(
        ctypes.c_void_p(numpy_array.ctypes.data),
        ctypes.c_void_p(contiguous.data_ptr()),
        ctypes.c_size_t(contiguous.numel() * contiguous.element_size()),
        ctypes.c_int(2),
    )
    if error != 0:
        message = runtime.cudaGetErrorString(error)
        text = message.decode("utf-8", errors="replace") if message else f"cuda error {error}"
        raise RuntimeError(f"cudaMemcpy DeviceToHost failed: {text}")
    return torch.from_numpy(numpy_array)


def _capture_root(rank: int) -> Path:
    root_text = os.environ.get("SHARDGRID_T072_CAPTURE_ROOT", "").strip()
    if root_text:
        root = Path(root_text) / f"rank{rank}"
    else:
        root = Path("/tmp/t072_activation_raw") / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tensor_capture_payload(
    tensor: torch.Tensor,
    *,
    cpu_tensor_source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object], list[float]]:
    detached = tensor.detach()
    cpu_source = detached if cpu_tensor_source is None else cpu_tensor_source.detach()
    cpu_tensor = _tensor_to_cpu(cpu_source)
    flat = cpu_tensor.reshape(-1)
    sample = [float(flat[index].item()) for index in range(min(100, int(flat.numel())))]
    stats = {
        "min": float(cpu_tensor.min().item()),
        "max": float(cpu_tensor.max().item()),
        "mean": float(cpu_tensor.mean().item()),
        "std": float(cpu_tensor.std().item()),
        "sum": float(cpu_tensor.sum().item()),
        "numel": int(cpu_tensor.numel()),
    }
    metadata = {
        "shape": list(detached.shape),
        "dtype": _dtype_name(detached.dtype),
        "numpy_dtype": str(cpu_tensor.numpy().dtype),
        "device": str(detached.device),
        "stride": list(detached.stride()),
        "contiguous": bool(detached.is_contiguous()),
        "requires_grad": bool(detached.requires_grad),
        "stats": stats,
    }
    return cpu_tensor, metadata, sample


def _save_tensor_capture(
    rank: int,
    base_name: str,
    tensor: torch.Tensor,
    *,
    cpu_tensor_source: torch.Tensor | None = None,
) -> None:
    root = _capture_root(rank)
    cpu_tensor, metadata, sample = _tensor_capture_payload(
        tensor,
        cpu_tensor_source=cpu_tensor_source,
    )
    torch.save(
        {
            "tensor": cpu_tensor,
            "shape": metadata["shape"],
            "dtype": metadata["dtype"],
            "device": metadata["device"],
            "stride": metadata["stride"],
            "stats": metadata["stats"],
            "sample": sample,
        },
        root / f"{base_name}.pt",
    )
    np.save(root / f"{base_name}.npy", cpu_tensor.numpy())
    (root / f"{base_name}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sample_lines = [f"{index},{value:.6f}" for index, value in enumerate(sample)]
    (root / f"{base_name}_sample.txt").write_text(
        "\n".join(sample_lines) + ("\n" if sample_lines else ""),
        encoding="utf-8",
    )


def _write_gpu_read_result(
    rank: int,
    *,
    success: bool,
    error: str,
    read_values: list[float],
) -> None:
    (_capture_root(rank) / "gpu_read_result.json").write_text(
        json.dumps(
            {
                "success": success,
                "error": error,
                "read_values": read_values,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _log_model_object_info(rank: int, model: torch.nn.Module) -> dict[str, object]:
    _mark("MODEL_OBJECT_INFO_BEGIN")
    _debug_breakpoint(rank, "MODEL_OBJECT_INFO_BEGIN")
    try:
        params = list(model.named_parameters())
        dtype_names = sorted({_dtype_name(param.dtype) for _, param in params})
        cpu_parameter_found = any(str(param.device) == "cpu" for _, param in params)
        total_numel = sum(int(param.numel()) for _, param in params)
        total_bytes = sum(int(param.numel() * param.element_size()) for _, param in params)
        sample = [
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": _dtype_name(param.dtype),
                "device": str(param.device),
                "requires_grad": bool(param.requires_grad),
                "is_leaf": bool(param.is_leaf),
            }
            for name, param in params[:20]
        ]
        _diag_line(
            "model_object_info",
            rank=rank,
            hostname=socket.gethostname(),
            model_type=str(type(model)),
            model_class=model.__class__.__name__,
            model_module=model.__module__,
            model_repr=repr(model),
            parameter_sample=sample,
            parameter_count=total_numel,
            parameter_bytes=total_bytes,
            cpu_parameter_found=cpu_parameter_found,
            dtype_names=dtype_names,
            mixed_dtype=bool(len(dtype_names) > 1),
        )
    except Exception as error:
        _fail_stage(rank, "model_object_info", error)
        raise
    _mark("MODEL_OBJECT_INFO_END")
    _debug_breakpoint(rank, "MODEL_OBJECT_INFO_END")
    return {
        "model_type": model.__class__.__name__,
        "parameter_count": total_numel,
        "cpu_parameter_found": cpu_parameter_found,
        "hook_count": "",
    }


def _log_model_structure(rank: int, model: torch.nn.Module) -> None:
    _mark("MODEL_STRUCTURE_BEGIN")
    _debug_breakpoint(rank, "MODEL_STRUCTURE_BEGIN")
    try:
        modules = [
            {"name": name, "type": type(module).__name__}
            for name, module in list(model.named_modules())[:50]
        ]
        _diag_line("model_structure", rank=rank, modules=modules)
    except Exception as error:
        _fail_stage(rank, "model_structure", error)
        raise
    _mark("MODEL_STRUCTURE_END")
    _debug_breakpoint(rank, "MODEL_STRUCTURE_END")


def _log_model_hook_info(rank: int, model: torch.nn.Module) -> str:
    _mark("MODEL_HOOK_INFO_BEGIN")
    _debug_breakpoint(rank, "MODEL_HOOK_INFO_BEGIN")
    try:
        forward_hooks = list(getattr(model, "_forward_hooks", {}).values())
        forward_pre_hooks = list(getattr(model, "_forward_pre_hooks", {}).values())
        backward_hooks = list(getattr(model, "_backward_hooks", {}).values())
        hook_count = (
            f"forward={len(forward_hooks)},pre={len(forward_pre_hooks)},"
            f"backward={len(backward_hooks)}"
        )
        _diag_line(
            "model_hook_info",
            rank=rank,
            forward_hook_count=len(forward_hooks),
            pre_hook_count=len(forward_pre_hooks),
            backward_hook_count=len(backward_hooks),
            forward_hook_types=[type(hook).__name__ for hook in forward_hooks],
            pre_hook_types=[type(hook).__name__ for hook in forward_pre_hooks],
            backward_hook_types=[type(hook).__name__ for hook in backward_hooks],
        )
    except Exception as error:
        _fail_stage(rank, "model_hook_info", error)
        raise
    _mark("MODEL_HOOK_INFO_END")
    _debug_breakpoint(rank, "MODEL_HOOK_INFO_END")
    return hook_count


def _log_model_input_info(rank: int, hidden: torch.Tensor) -> None:
    _mark("MODEL_INPUT_INFO_BEGIN")
    _debug_breakpoint(rank, "MODEL_INPUT_INFO_BEGIN")
    try:
        _diag_line("model_input_info", rank=rank, **_activation_basics(hidden))
    except Exception as error:
        _fail_stage(rank, "model_input_info", error)
        raise
    _mark("MODEL_INPUT_INFO_END")
    _debug_breakpoint(rank, "MODEL_INPUT_INFO_END")


def _log_model_forward_attribute(rank: int, model: torch.nn.Module) -> str:
    _mark("MODEL_FORWARD_INFO_BEGIN")
    _debug_breakpoint(rank, "MODEL_FORWARD_INFO_BEGIN")
    try:
        forward = model.forward
        _diag_line(
            "model_forward_info",
            rank=rank,
            forward_repr=repr(forward),
            forward_type=str(type(forward)),
            forward_module=getattr(forward, "__module__", None),
            forward_qualname=getattr(forward, "__qualname__", None),
        )
        forward_type = str(type(forward))
    except Exception as error:
        _fail_stage(rank, "model_forward_attribute", error)
        raise
    _mark("MODEL_FORWARD_INFO_END")
    _debug_breakpoint(rank, "MODEL_FORWARD_INFO_END")
    return forward_type


def _log_model_call_mode(rank: int, model: torch.nn.Module) -> None:
    _mark("MODEL_RUNTIME_STATE_BEGIN")
    _debug_breakpoint(rank, "MODEL_RUNTIME_STATE_BEGIN")
    try:
        gpu_autocast = (
            bool(torch.is_autocast_enabled("cuda"))
            if hasattr(torch, "is_autocast_enabled")
            else False
        )
        cpu_autocast = (
            bool(torch.is_autocast_enabled("cpu"))
            if hasattr(torch, "is_autocast_enabled")
            else False
        )
        _diag_line(
            "model_call_mode",
            rank=rank,
            grad_enabled=bool(torch.is_grad_enabled()),
            inference_mode_enabled=bool(torch.is_inference_mode_enabled()),
            autocast_enabled=bool(torch.is_autocast_enabled()),
            autocast_cpu_enabled=cpu_autocast,
            autocast_gpu_enabled=gpu_autocast,
            cuda_current_device=int(torch.cuda.current_device()),
            cuda_device_name=torch.cuda.get_device_name(),
            cuda_memory_allocated=int(torch.cuda.memory_allocated()),
            cuda_memory_reserved=int(torch.cuda.memory_reserved()),
        )
    except Exception as error:
        _fail_stage(rank, "model_call_mode", error)
        raise
    _mark("MODEL_RUNTIME_STATE_END")
    _debug_breakpoint(rank, "MODEL_RUNTIME_STATE_END")
def _log_model_call_impl(rank: int, model: torch.nn.Module) -> None:
    _mark("MODEL_CALL_IMPL_BEGIN")
    _debug_breakpoint(rank, "MODEL_CALL_IMPL_BEGIN")
    try:
        call_impl = model.__call__
        _diag_line(
            "model_call_impl",
            rank=rank,
            call_repr=repr(call_impl),
            call_type=str(type(call_impl)),
        )
    except Exception as error:
        _fail_stage(rank, "model_call_impl", error)
        raise
    _mark("MODEL_CALL_IMPL_END")
    _debug_breakpoint(rank, "MODEL_CALL_IMPL_END")
def _log_role_assignment(
    rank: int,
    stage_id: str,
    local_rank: int,
    device: torch.device,
) -> None:
    gpu_name = (
        torch.cuda.get_device_name(local_rank)
        if torch.cuda.is_available()
        else "unavailable"
    )
    _mark("ROLE_ASSIGNMENT")
    _debug_breakpoint(rank, "ROLE_ASSIGNMENT")
    _emit_line(f"[RANK={rank}] GPU={gpu_name} ROLE={stage_id.upper()}")
    _diag_line(
        "role_assignment",
        rank=rank,
        hostname=socket.gethostname(),
        gpu_name=gpu_name,
        role=stage_id,
        device=str(device),
    )


def _log_model_device_info(
    rank: int,
    stage_id: str,
    model: torch.nn.Module,
    named: dict[str, torch.nn.Parameter],
    device: torch.device,
) -> None:
    parameter_devices = {
        name: str(parameter.device)
        for name, parameter in list(named.items())[:20]
    }
    _mark("MODEL_DEVICE_INFO")
    _debug_breakpoint(rank, "MODEL_DEVICE_INFO")
    _diag_line(
        "model_device_info",
        rank=rank,
        stage_id=stage_id,
        model_type=f"{type(model).__module__}.{type(model).__qualname__}",
        device=str(device),
        parameter_devices=parameter_devices,
        all_parameters_on_device=all(
            str(parameter.device) == str(device)
            for parameter in named.values()
        ),
        parameter_count=sum(parameter.numel() for parameter in named.values()),
    )


def _log_model_final_type(rank: int, model: torch.nn.Module) -> None:
    _mark("MODEL_INFO")
    _debug_breakpoint(rank, "MODEL_INFO")
    _mark("MODEL_FINAL_TYPE")
    _debug_breakpoint(rank, "MODEL_FINAL_TYPE")
    _diag_line(
        "model_final_type",
        rank=rank,
        model_type=f"{type(model).__module__}.{type(model).__qualname__}",
        model_class=str(model.__class__),
        model_forward=repr(model.forward),
    )


def _stream_details(device: torch.device) -> dict[str, object]:
    stream = torch.cuda.current_stream(device)
    return {
        "device": str(device),
        "current_device": int(torch.cuda.current_device()),
        "device_count": int(torch.cuda.device_count()),
        "stream_repr": str(stream),
        "stream_id": int(stream.cuda_stream),
        "default_stream_repr": str(torch.cuda.default_stream(device)),
        "default_stream_id": int(torch.cuda.default_stream(device).cuda_stream),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", ""),
        "memory_allocated": int(torch.cuda.memory_allocated(device)),
        "memory_reserved": int(torch.cuda.memory_reserved(device)),
    }


def _recv_activation_ready(
    *,
    hidden: torch.Tensor,
    src: int,
    device: torch.device,
) -> dict[str, float]:
    wait_begin = _mark("ACTIVATION_RECV_WAIT_BEGIN")
    work = dist.irecv(hidden, src=src)
    work.wait()
    wait_end = _mark("ACTIVATION_RECV_WAIT_END")
    return {
        "recv_wait_begin": wait_begin,
        "recv_wait_end": wait_end,
    }


def _build_t072_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(T072_BATCH_SIZE * T072_SEQ_LENGTH, dtype=torch.long)
    input_ids = base.reshape(T072_BATCH_SIZE, T072_SEQ_LENGTH) % MODEL_CONFIG.vocab_size
    labels = (input_ids + 1) % MODEL_CONFIG.vocab_size
    return input_ids.to(device), labels.to(device)


def _placement_evidence(
    *,
    rank: int,
    world: int,
    local_rank: int,
    device: torch.device,
    stage_id: str,
    named: dict[str, torch.nn.Parameter],
    sanity_ok: bool,
) -> dict[str, object]:
    parameter_count = sum(parameter.numel() for parameter in named.values())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in named.values()
        if parameter.requires_grad
    )
    return {
        "hostname": socket.gethostname(),
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "device": str(device),
        "stage_id": stage_id,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "parameter_names": sorted(named.keys()),
        "parameter_devices": {
            name: str(parameter.device) for name, parameter in named.items()
        },
        "all_parameters_on_device": all(
            parameter.device == device for parameter in named.values()
        ),
        "stage_sanity_forward_ok": sanity_ok,
    }


def _run_t072(
    *,
    rank: int,
    world: int,
    local_rank: int,
    device: torch.device,
    stage_id: str,
    model: torch.nn.Module,
    parameter_count: int,
    runtime_plan: dict[str, object],
) -> dict[str, object]:
    producer_rank = int(runtime_plan["producer_rank"])
    consumer_rank = int(runtime_plan["consumer_rank"])
    activation_shape = tuple(int(item) for item in runtime_plan["activation_shape"])
    activation_dtype = runtime_plan["activation_dtype"]
    activation_dtype_name = str(runtime_plan["activation_dtype_name"])

    if rank == producer_rank:
        input_ids, _ = _build_t072_inputs(device)
        _mark("T072_STAGE0_FORWARD_BEGIN")
        stage0_forward_begin = _mark("STAGE0_FORWARD_BEGIN")
        hidden = model(input_ids)
        stage0_forward_end = _mark("STAGE0_FORWARD_END")
        _mark("T072_STAGE0_FORWARD_END")
        hidden = hidden.detach().contiguous()
        activation_stats = _activation_basics(hidden)
        _diag_line("activation_send_metadata", rank=rank, **activation_stats)
        _mark("ACTIVATION_SEND_BEGIN")
        send_begin = _timestamp()
        dist.send(hidden, dst=consumer_rank)
        send_end = _timestamp()
        _mark("ACTIVATION_SEND_END")
        return {
            "hostname": socket.gethostname(),
            "rank": rank,
            "world_size": world,
            "local_rank": local_rank,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            "device": str(device),
            "stage_id": stage_id,
            "parameter_count": parameter_count,
            "input_shape": list(input_ids.shape),
            "input_dtype": _dtype_name(input_ids.dtype),
            "output_shape": list(hidden.shape),
            "output_dtype": _dtype_name(hidden.dtype),
            "stage0_forward_ok": bool(torch.isfinite(hidden).all().item()),
            "lifecycle": {
                "stage0_forward_begin": stage0_forward_begin,
                "stage0_forward_end": stage0_forward_end,
                "activation_send_begin": send_begin,
                "activation_send_end": send_end,
            },
            "activation_transfer": {
                "sender_rank": producer_rank,
                "receiver_rank": consumer_rank,
                "shape": list(hidden.shape),
                "dtype": _dtype_name(hidden.dtype),
                "numel": int(hidden.numel()),
                "send_begin": send_begin,
                "send_end": send_end,
                "send_check": activation_stats,
            },
        }

    hidden = torch.empty(
        activation_shape,
        dtype=activation_dtype,
        device=device,
    )
    _mark("ACTIVATION_RECV_BEGIN")
    recv_begin = _timestamp()
    dist.recv(hidden, src=producer_rank)
    recv_end = _timestamp()
    _mark("ACTIVATION_RECV_END")
    activation_stats = _activation_basics(hidden)
    _diag_line("activation_recv_metadata", rank=rank, **activation_stats)
    labels = _build_t072_inputs(device)[1]

    _mark("STAGE1_FORWARD_BEGIN")
    stage1_forward_begin = _timestamp()
    logits = model(hidden)
    stage1_forward_end = _timestamp()
    _mark("STAGE1_FORWARD_END")

    _mark("FORWARD_OUTPUT_BEGIN")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )
    loss_value = float(loss.item())
    loss_isfinite = bool(torch.isfinite(loss).item())
    _diag_line(
        "forward_output_metadata",
        rank=rank,
        output_shape=list(logits.shape),
        output_dtype=_dtype_name(logits.dtype),
        target_shape=list(labels.shape),
        target_dtype=_dtype_name(labels.dtype),
        loss=loss_value,
        loss_isfinite=loss_isfinite,
    )
    loss_ready = _mark("FORWARD_OUTPUT_END")
    return {
        "hostname": socket.gethostname(),
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "device": str(device),
        "stage_id": stage_id,
        "parameter_count": parameter_count,
        "input_shape": list(hidden.shape),
        "input_dtype": _dtype_name(hidden.dtype),
        "output_shape": list(logits.shape),
        "output_dtype": _dtype_name(logits.dtype),
        "target_shape": list(labels.shape),
        "target_dtype": _dtype_name(labels.dtype),
        "loss": loss_value,
        "loss_isfinite": loss_isfinite,
        "stage1_forward_ok": bool(torch.isfinite(logits).all().item()),
        "lifecycle": {
            "activation_recv_begin": recv_begin,
            "activation_recv_end": recv_end,
            "stage1_forward_begin": stage1_forward_begin,
            "stage1_forward_end": stage1_forward_end,
            "loss_ready": loss_ready,
        },
        "activation_transfer": {
            "sender_rank": producer_rank,
            "receiver_rank": consumer_rank,
            "shape": list(hidden.shape),
            "dtype": activation_dtype_name,
            "numel": int(hidden.numel()),
            "recv_begin": recv_begin,
            "recv_end": recv_end,
            "stats": activation_stats,
        },
    }


def _grad_stats(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "isfinite": bool(torch.isfinite(tensor).all().item()),
    }


def _parameter_grad_summary(model: torch.nn.Module) -> dict[str, object]:
    grads = [(name, param.grad) for name, param in model.named_parameters()]
    present = [name for name, grad in grads if grad is not None]
    finite = [
        name for name, grad in grads
        if grad is not None and bool(torch.isfinite(grad).all().item())
    ]
    return {
        "gradient_param_count": len(present),
        "all_params_have_grad": len(present) == len(grads),
        "all_grads_finite": len(finite) == len(grads),
        "parameter_samples": present[:10],
    }


def _run_t073(
    *,
    rank: int,
    world: int,
    local_rank: int,
    device: torch.device,
    stage_id: str,
    model: torch.nn.Module,
    parameter_count: int,
    runtime_plan: dict[str, object],
) -> dict[str, object]:
    producer_rank = int(runtime_plan["producer_rank"])
    consumer_rank = int(runtime_plan["consumer_rank"])
    activation_shape = tuple(int(item) for item in runtime_plan["activation_shape"])
    activation_dtype = runtime_plan["activation_dtype"]
    activation_dtype_name = str(runtime_plan["activation_dtype_name"])

    if rank == producer_rank:
        input_ids, _ = _build_t072_inputs(device)
        _mark("STAGE0_FORWARD_BEGIN")
        stage0_forward_begin = _timestamp()
        hidden = model(input_ids)
        stage0_forward_end = _timestamp()
        _mark("STAGE0_FORWARD_END")

        send_hidden = hidden.detach().contiguous()
        activation_stats = _activation_basics(send_hidden)
        _diag_line("activation_send_metadata", rank=rank, **activation_stats)
        _mark("ACTIVATION_SEND_BEGIN")
        activation_send_begin = _timestamp()
        dist.send(send_hidden, dst=consumer_rank)
        activation_send_end = _timestamp()
        _mark("ACTIVATION_SEND_END")

        grad_hidden = torch.empty(
            activation_shape,
            dtype=activation_dtype,
            device=device,
        )
        _mark("GRADIENT_RECV_BEGIN")
        gradient_recv_begin = _timestamp()
        dist.recv(grad_hidden, src=consumer_rank)
        gradient_recv_end = _timestamp()
        _mark("GRADIENT_RECV_END")

        grad_stats = _grad_stats(grad_hidden)
        _diag_line("activation_gradient_recv_metadata", rank=rank, **grad_stats)

        _mark("STAGE0_BACKWARD_BEGIN")
        stage0_backward_begin = _timestamp()
        hidden.backward(grad_hidden)
        stage0_backward_end = _timestamp()
        _mark("STAGE0_BACKWARD_END")

        stage0_grad_summary = _parameter_grad_summary(model)
        _mark("STAGE0_GRADIENTS_READY")
        return {
            "hostname": socket.gethostname(),
            "rank": rank,
            "world_size": world,
            "local_rank": local_rank,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            "device": str(device),
            "stage_id": stage_id,
            "parameter_count": parameter_count,
            "input_shape": list(input_ids.shape),
            "input_dtype": _dtype_name(input_ids.dtype),
            "activation_shape": list(hidden.shape),
            "activation_dtype": _dtype_name(hidden.dtype),
            "stage0_forward_ok": bool(torch.isfinite(hidden).all().item()),
            "stage0_backward_ok": True,
            "activation_grad_isfinite": bool(grad_stats["isfinite"]),
            "lifecycle": {
                "stage0_forward_begin": stage0_forward_begin,
                "stage0_forward_end": stage0_forward_end,
                "activation_send_begin": activation_send_begin,
                "activation_send_end": activation_send_end,
                "gradient_recv_begin": gradient_recv_begin,
                "gradient_recv_end": gradient_recv_end,
                "stage0_backward_begin": stage0_backward_begin,
                "stage0_backward_end": stage0_backward_end,
            },
            "activation_transfer": {
                "sender_rank": producer_rank,
                "receiver_rank": consumer_rank,
                "shape": list(send_hidden.shape),
                "dtype": activation_dtype_name,
                "numel": int(send_hidden.numel()),
            },
            "gradient_return": {
                "sender_rank": consumer_rank,
                "receiver_rank": producer_rank,
                "shape": grad_stats["shape"],
                "dtype": grad_stats["dtype"],
                "isfinite": grad_stats["isfinite"],
            },
            "stage0_gradients": stage0_grad_summary,
        }

    hidden = torch.empty(
        activation_shape,
        dtype=activation_dtype,
        device=device,
    )
    _mark("ACTIVATION_RECV_BEGIN")
    activation_recv_begin = _timestamp()
    dist.recv(hidden, src=producer_rank)
    activation_recv_end = _timestamp()
    _mark("ACTIVATION_RECV_END")
    activation_stats = _activation_basics(hidden)
    _diag_line("activation_recv_metadata", rank=rank, **activation_stats)

    hidden.requires_grad_(True)
    labels = _build_t072_inputs(device)[1]

    _mark("STAGE1_FORWARD_BEGIN")
    stage1_forward_begin = _timestamp()
    logits = model(hidden)
    stage1_forward_end = _timestamp()
    _mark("STAGE1_FORWARD_END")

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )
    loss_value = float(loss.item())
    loss_isfinite = bool(torch.isfinite(loss).item())
    _mark("LOSS_READY")

    _mark("STAGE1_BACKWARD_BEGIN")
    stage1_backward_begin = _timestamp()
    loss.backward()
    stage1_backward_end = _timestamp()
    _mark("STAGE1_BACKWARD_END")

    if hidden.grad is None:
        raise RuntimeError("Stage1 backward completed but hidden.grad is missing")
    grad_hidden = hidden.grad.detach().contiguous()
    grad_stats = _grad_stats(grad_hidden)
    _diag_line("activation_gradient_send_metadata", rank=rank, **grad_stats)
    _mark("ACTIVATION_GRADIENT_READY")

    _mark("GRADIENT_SEND_BEGIN")
    gradient_send_begin = _timestamp()
    dist.send(grad_hidden, dst=producer_rank)
    gradient_send_end = _timestamp()
    _mark("GRADIENT_SEND_END")

    stage1_grad_summary = _parameter_grad_summary(model)
    _mark("STAGE1_GRADIENTS_READY")
    return {
        "hostname": socket.gethostname(),
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "device": str(device),
        "stage_id": stage_id,
        "parameter_count": parameter_count,
        "input_shape": list(hidden.shape),
        "input_dtype": _dtype_name(hidden.dtype),
        "output_shape": list(logits.shape),
        "output_dtype": _dtype_name(logits.dtype),
        "target_shape": list(labels.shape),
        "target_dtype": _dtype_name(labels.dtype),
        "loss": loss_value,
        "loss_isfinite": loss_isfinite,
        "stage1_forward_ok": bool(torch.isfinite(logits).all().item()),
        "stage1_backward_ok": True,
        "activation_grad_isfinite": bool(grad_stats["isfinite"]),
        "lifecycle": {
            "activation_recv_begin": activation_recv_begin,
            "activation_recv_end": activation_recv_end,
            "stage1_forward_begin": stage1_forward_begin,
            "stage1_forward_end": stage1_forward_end,
            "stage1_backward_begin": stage1_backward_begin,
            "stage1_backward_end": stage1_backward_end,
            "gradient_send_begin": gradient_send_begin,
            "gradient_send_end": gradient_send_end,
        },
        "activation_transfer": {
            "sender_rank": producer_rank,
            "receiver_rank": consumer_rank,
            "shape": list(hidden.shape),
            "dtype": activation_dtype_name,
            "numel": int(hidden.numel()),
        },
        "gradient_return": {
            "sender_rank": consumer_rank,
            "receiver_rank": producer_rank,
            "shape": grad_stats["shape"],
            "dtype": grad_stats["dtype"],
            "isfinite": grad_stats["isfinite"],
        },
        "stage1_gradients": stage1_grad_summary,
    }


def _model_config_metadata() -> dict[str, object]:
    return {
        "vocab_size": MODEL_CONFIG.vocab_size,
        "hidden_size": MODEL_CONFIG.hidden_size,
        "num_hidden_layers": MODEL_CONFIG.num_hidden_layers,
        "num_attention_heads": MODEL_CONFIG.num_attention_heads,
        "max_seq_length": MODEL_CONFIG.max_seq_length,
        "dropout": MODEL_CONFIG.dropout,
    }


def _run_t074(
    *,
    rank: int,
    world: int,
    local_rank: int,
    device: torch.device,
    stage_id: str,
    model: torch.nn.Module,
    parameter_count: int,
    runtime_plan: dict[str, object],
) -> dict[str, object]:
    producer_rank = int(runtime_plan["producer_rank"])
    consumer_rank = int(runtime_plan["consumer_rank"])
    producer_id = str(runtime_plan["producer_id"])
    consumer_id = str(runtime_plan["consumer_id"])
    activation_shape = tuple(int(item) for item in runtime_plan["activation_shape"])
    activation_dtype = runtime_plan["activation_dtype"]
    activation_dtype_name = str(runtime_plan["activation_dtype_name"])

    steps = int(os.environ.get("SHARDGRID_T074_STEPS", str(T074_DEFAULT_STEPS)))
    learning_rate = float(os.environ.get("SHARDGRID_T074_LR", str(T074_DEFAULT_LR)))
    checkpoint_dir = Path(
        os.environ.get("SHARDGRID_T074_CHECKPOINT_DIR", "/tmp/t074/checkpoint")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_rank{rank}.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss_history: list[float] = []
    initial_loss: float | None = None
    final_loss: float | None = None
    params_before = _model_param_checksum(model)
    lifecycle: dict[str, object] = {
        "train_step_begin": [],
        "optimizer_step_end": [],
        "train_step_end": [],
    }
    if rank == producer_rank:
        lifecycle["stage0_backward_end"] = []
    else:
        lifecycle["loss_ready"] = []
        lifecycle["stage1_backward_end"] = []
        lifecycle["gradient_send_end"] = []

    for _ in range(1, steps + 1):
        _mark("TRAIN_STEP_BEGIN")
        lifecycle["train_step_begin"].append(_timestamp())
        optimizer.zero_grad(set_to_none=True)

        if rank == producer_rank:
            input_ids, _ = _build_t072_inputs(device)
            _mark("STAGE0_FORWARD_BEGIN")
            hidden = model(input_ids)
            _mark("STAGE0_FORWARD_END")
            send_hidden = hidden.detach().contiguous()
            _mark("ACTIVATION_SEND_BEGIN")
            dist.send(send_hidden, dst=consumer_rank)
            _mark("ACTIVATION_SEND_END")
            grad_hidden = torch.empty(
                activation_shape,
                dtype=activation_dtype,
                device=device,
            )
            _mark("GRADIENT_RECV_BEGIN")
            dist.recv(grad_hidden, src=consumer_rank)
            _mark("GRADIENT_RECV_END")
            _mark("STAGE0_BACKWARD_BEGIN")
            hidden.backward(grad_hidden)
            _mark("STAGE0_BACKWARD_END")
            lifecycle["stage0_backward_end"].append(_timestamp())
        else:
            hidden = torch.empty(
                activation_shape,
                dtype=activation_dtype,
                device=device,
            )
            _mark("ACTIVATION_RECV_BEGIN")
            dist.recv(hidden, src=producer_rank)
            _mark("ACTIVATION_RECV_END")
            hidden.requires_grad_(True)
            labels = _build_t072_inputs(device)[1]
            _mark("STAGE1_FORWARD_BEGIN")
            logits = model(hidden)
            _mark("STAGE1_FORWARD_END")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            loss_value = float(loss.item())
            loss_history.append(loss_value)
            if initial_loss is None:
                initial_loss = loss_value
            _mark("LOSS_READY")
            lifecycle["loss_ready"].append(_timestamp())
            _mark("STAGE1_BACKWARD_BEGIN")
            loss.backward()
            _mark("STAGE1_BACKWARD_END")
            lifecycle["stage1_backward_end"].append(_timestamp())
            if hidden.grad is None:
                raise RuntimeError(
                    "Stage1 backward completed but hidden.grad is missing"
                )
            grad_hidden = hidden.grad.detach().contiguous()
            _mark("GRADIENT_SEND_BEGIN")
            dist.send(grad_hidden, dst=producer_rank)
            _mark("GRADIENT_SEND_END")
            lifecycle["gradient_send_end"].append(_timestamp())

        optimizer.step()
        _mark("OPTIMIZER_STEP_END")
        lifecycle["optimizer_step_end"].append(_timestamp())
        _mark("TRAIN_STEP_END")
        lifecycle["train_step_end"].append(_timestamp())

    if loss_history:
        final_loss = loss_history[-1]
    params_after = _model_param_checksum(model)
    param_update_ok = params_before != params_after

    _mark("CHECKPOINT_SAVE_BEGIN")
    checkpoint_save_begin = _timestamp()
    checkpoint_payload = {
        "checkpoint_version": 1,
        "task": "t074",
        "rank": rank,
        "world_size": world,
        "stage_id": stage_id,
        "step": steps,
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "metadata": {
            "producer_id": producer_id,
            "consumer_id": consumer_id,
            "producer_rank": producer_rank,
            "consumer_rank": consumer_rank,
            "activation_shape": list(activation_shape),
            "activation_dtype": activation_dtype_name,
            "model_config": _model_config_metadata(),
            "learning_rate": learning_rate,
            "steps": steps,
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_save_end = _timestamp()
    _mark("CHECKPOINT_SAVE_END")

    for parameter in model.parameters():
        parameter.data.zero_()
    fresh_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    _mark("CHECKPOINT_LOAD_BEGIN")
    checkpoint_load_begin = _timestamp()
    loaded = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(loaded["model_state_dict"])
    fresh_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    restored_step = int(loaded["step"])
    checkpoint_load_end = _timestamp()
    _mark("CHECKPOINT_LOAD_END")

    saved_state = checkpoint_payload["model_state_dict"]
    current_state = model.state_dict()
    param_restore_ok = all(
        torch.equal(current_state[name].detach().cpu(), saved_state[name])
        for name in saved_state
    )
    optimizer_restore_ok = _state_dict_equal(
        fresh_optimizer.state_dict(), optimizer.state_dict()
    )
    step_restore_ok = restored_step == steps
    checkpoint_roundtrip_ok = bool(
        param_restore_ok and optimizer_restore_ok and step_restore_ok
    )

    loss_isfinite = all(math.isfinite(value) for value in loss_history)
    loss_decrease = bool(
        initial_loss is not None
        and final_loss is not None
        and final_loss < initial_loss
    )

    return {
        "hostname": socket.gethostname(),
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "device": str(device),
        "stage_id": stage_id,
        "parameter_count": parameter_count,
        "steps": steps,
        "loss_history": loss_history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_isfinite": loss_isfinite,
        "loss_decrease": loss_decrease,
        "params_before_checksum": params_before,
        "params_after_checksum": params_after,
        "param_update_ok": param_update_ok,
        "param_restore_ok": param_restore_ok,
        "optimizer_restore_ok": optimizer_restore_ok,
        "step_restore_ok": step_restore_ok,
        "checkpoint_roundtrip_ok": checkpoint_roundtrip_ok,
        "checkpoint_path": str(checkpoint_path),
        "lifecycle": {
            **lifecycle,
            "checkpoint_save_begin": checkpoint_save_begin,
            "checkpoint_save_end": checkpoint_save_end,
            "checkpoint_load_begin": checkpoint_load_begin,
            "checkpoint_load_end": checkpoint_load_end,
        },
    }


def _shutdown_distributed(*, local_rank: int) -> None:
    if not dist.is_initialized():
        return
    _mark("SHUTDOWN_BEGIN")
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    _mark("SHUTDOWN_END")


def main() -> None:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    rank = int(os.environ["RANK"])
    _configure_rank_log(rank)
    _log_running_file_info(rank)
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _runtime_info(rank, local_rank)
    runtime_plan = _load_t072_runtime_plan()
    stage_by_rank = runtime_plan["stage_by_rank"]

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=(
                f"tcp://{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT'])}"
            ),
            rank=rank,
            world_size=world,
        )

    if world != int(runtime_plan["world_size"]):
        raise ValueError(
            "WORLD_SIZE=%d does not match static plan world_size=%d"
            % (world, int(runtime_plan["world_size"]))
        )
    stage = stage_by_rank.get(rank)
    if stage is None:
        raise ValueError(f"rank {rank} is not assigned in the static plan")
    stage_id = str(stage["id"])
    if stage_id == "stage0":
        model = build_stage0(MODEL_CONFIG, seed=42)
    elif stage_id == "stage1":
        model = build_stage1(MODEL_CONFIG, seed=42)
    else:
        raise ValueError(f"unsupported stage {stage_id!r} in the example runner")

    model.to(device)
    named = dict(model.named_parameters())
    _log_model_final_type(rank, model)
    _log_role_assignment(rank, stage_id, local_rank, device)
    _log_model_device_info(rank, stage_id, model, named, device)
    parameter_count = sum(parameter.numel() for parameter in named.values())
    task = os.environ.get(TASK_ENV, TASK_T071).strip().lower()

    if task in {TASK_T072, TASK_T073, TASK_T074}:
        sanity_ok = True
    elif stage_id == "stage0":
        sample = torch.randint(
            0, MODEL_CONFIG.vocab_size, (1, 8), dtype=torch.long, device=device
        )
        hidden = model(sample)
        sanity_ok = bool(torch.isfinite(hidden).all().item())
    else:
        # Stage1 consumes an activation; a local sanity tensor is enough to
        # prove it executes on its GPU (transfer is T072).
        hidden = torch.randn(
            1, 8, MODEL_CONFIG.hidden_size, dtype=torch.float32, device=device
        )
        logits = model(hidden)
        sanity_ok = bool(torch.isfinite(logits).all().item())

    placement = _placement_evidence(
        rank=rank,
        world=world,
        local_rank=local_rank,
        device=device,
        stage_id=stage_id,
        named=named,
        sanity_ok=sanity_ok,
    )

    if task == TASK_T071:
        print("T071_PROGRESS before_barrier rank=%d" % rank, flush=True)
        dist.barrier()
        print("T071_PROGRESS after_barrier rank=%d" % rank, flush=True)
        print(EVENT_MARKER + json.dumps(placement, sort_keys=True), flush=True)
    elif task == TASK_T072:
        print(EVENT_MARKER + json.dumps(placement, sort_keys=True), flush=True)
        forward = _run_t072(
            rank=rank,
            world=world,
            local_rank=local_rank,
            device=device,
            stage_id=stage_id,
            model=model,
            parameter_count=parameter_count,
            runtime_plan=runtime_plan,
        )
        print(FORWARD_MARKER + json.dumps(forward, sort_keys=True), flush=True)
        _shutdown_distributed(local_rank=local_rank)
    elif task == TASK_T073:
        print(EVENT_MARKER + json.dumps(placement, sort_keys=True), flush=True)
        backward = _run_t073(
            rank=rank,
            world=world,
            local_rank=local_rank,
            device=device,
            stage_id=stage_id,
            model=model,
            parameter_count=parameter_count,
            runtime_plan=runtime_plan,
        )
        print(BACKWARD_MARKER + json.dumps(backward, sort_keys=True), flush=True)
        _shutdown_distributed(local_rank=local_rank)
    elif task == TASK_T074:
        print(EVENT_MARKER + json.dumps(placement, sort_keys=True), flush=True)
        train = _run_t074(
            rank=rank,
            world=world,
            local_rank=local_rank,
            device=device,
            stage_id=stage_id,
            model=model,
            parameter_count=parameter_count,
            runtime_plan=runtime_plan,
        )
        print(TRAIN_MARKER + json.dumps(train, sort_keys=True), flush=True)
        _shutdown_distributed(local_rank=local_rank)
    else:
        raise ValueError(
            f"unsupported {TASK_ENV}={task!r}; expected "
            f"{TASK_T071!r}, {TASK_T072!r}, {TASK_T073!r}, or {TASK_T074!r}"
        )
    try:
        if dist.is_initialized():
            _shutdown_distributed(local_rank=local_rank)
    except Exception:
        pass


if __name__ == "__main__":
    main()
