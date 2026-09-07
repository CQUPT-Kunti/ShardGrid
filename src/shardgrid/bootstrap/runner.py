"""In-process user entrypoint bootstrap and first-step capture."""

from __future__ import annotations

import os
import runpy
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class CaptureFailure:
    stage: str
    code: str
    message: str
    artifact_log_ref: str | None = None


@dataclass(frozen=True)
class CapturedTrainingContext:
    entrypoint_path: str
    argv: tuple[str, ...]
    cwd: str
    environment_summary: dict[str, str]
    capture_backend: str
    capture_backend_version: str
    model_identity: dict[str, str]
    parameter_count: int
    buffer_count: int
    first_batch_structure: dict[str, Any]
    model_call: dict[str, Any]
    tensor_metadata: dict[str, dict[str, Any]]
    optimizer: dict[str, Any] | None
    scheduler: dict[str, Any] | None
    lifecycle: dict[str, bool]
    model_output_structure: dict[str, Any]
    state_dict_key_to_canonical_state_id: dict[str, str]
    distributed_mutation_observed_before_capture: bool = False


@dataclass(frozen=True)
class CaptureResult:
    ok: bool
    context: CapturedTrainingContext | None = None
    failure: CaptureFailure | None = None


@dataclass(frozen=True)
class CapturedEntrypointWorkload:
    context: CapturedTrainingContext
    model: object
    sample_args: tuple[object, ...]
    sample_kwargs: dict[str, object]


@dataclass
class _CaptureState:
    entrypoint_path: Path
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    model: Any | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    first_batch: Any = None
    output: Any = None
    optimizer: dict[str, Any] | None = None
    scheduler: dict[str, Any] | None = None
    backward_called: bool = False
    optimizer_step_called: bool = False
    scheduler_step_called: bool = False
    distributed_mutation: bool = False


class _CaptureComplete(RuntimeError):
    pass


def capture_entrypoint(
    entrypoint: str | Path,
    *,
    argv: tuple[str, ...] = (),
    cwd: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    dry_run: bool = True,
) -> CapturedTrainingContext | CaptureResult:
    result = _capture_entrypoint_state(
        entrypoint,
        argv=argv,
        cwd=cwd,
        environment=environment,
        dry_run=dry_run,
    )
    if isinstance(result, CaptureResult):
        return result
    return _build_context(result)


def capture_entrypoint_workload(
    entrypoint: str | Path,
    *,
    argv: tuple[str, ...] = (),
    cwd: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    dry_run: bool = True,
) -> CapturedEntrypointWorkload | CaptureResult:
    result = _capture_entrypoint_state(
        entrypoint,
        argv=argv,
        cwd=cwd,
        environment=environment,
        dry_run=dry_run,
    )
    if isinstance(result, CaptureResult):
        return result
    if result.model is None:
        return CaptureResult(
            ok=False,
            failure=CaptureFailure(
                stage="capture",
                code="NO_MODEL_CALL_CAPTURED",
                message="entrypoint completed without an observable torch.nn.Module call",
                artifact_log_ref="diagnostics/capture.json",
            ),
        )
    return CapturedEntrypointWorkload(
        context=_build_context(result),
        model=result.model,
        sample_args=result.args,
        sample_kwargs=dict(result.kwargs),
    )


def _capture_entrypoint_state(
    entrypoint: str | Path,
    *,
    argv: tuple[str, ...],
    cwd: str | Path | None,
    environment: Mapping[str, str] | None,
    dry_run: bool,
) -> _CaptureState | CaptureResult:
    root = Path.cwd() if cwd is None else Path(cwd)
    path = Path(entrypoint)
    if not path.is_absolute():
        path = root / path
    state = _CaptureState(
        entrypoint_path=path,
        argv=tuple(argv),
        cwd=root,
        environment=dict(environment or {}),
    )
    try:
        with _entrypoint_process_context(path, state.argv, root, state.environment):
            with _torch_capture_hooks(state, dry_run=dry_run):
                runpy.run_path(str(path), run_name="__main__")
    except _CaptureComplete:
        return state
    except BaseException as exc:
        if state.model is not None:
            return state
        return CaptureResult(
            ok=False,
            failure=CaptureFailure(
                stage="capture",
                code=exc.__class__.__name__,
                message=str(exc),
                artifact_log_ref="diagnostics/capture.json",
            ),
        )
    return CaptureResult(
        ok=False,
        failure=CaptureFailure(
            stage="capture",
            code="NO_MODEL_CALL_CAPTURED",
            message="entrypoint completed without an observable torch.nn.Module call",
            artifact_log_ref="diagnostics/capture.json",
        ),
    )


@contextmanager
def _entrypoint_process_context(
    entrypoint: Path,
    argv: tuple[str, ...],
    cwd: Path,
    environment: Mapping[str, str],
) -> Iterator[None]:
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_env = os.environ.copy()
    sys.argv = [str(entrypoint), *argv]
    os.chdir(cwd)
    os.environ.update({str(key): str(value) for key, value in environment.items()})
    try:
        yield
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)


@contextmanager
def _torch_capture_hooks(state: _CaptureState, *, dry_run: bool) -> Iterator[None]:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    original_module_call = nn.Module.__call__
    original_dataloader_iter = torch.utils.data.DataLoader.__iter__
    original_optimizer_init = Optimizer.__init__
    original_scheduler_init = LRScheduler.__init__
    original_scheduler_step = LRScheduler.step
    original_backward = torch.Tensor.backward
    original_init_process_group = dist.init_process_group
    optimizer_step_methods = {
        cls: cls.step
        for cls in (Optimizer, torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)
    }

    def module_call(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        output = original_module_call(module, *args, **kwargs)
        if state.model is None and _looks_like_user_model(module):
            state.model = module
            state.args = tuple(args)
            state.kwargs = dict(kwargs)
            state.output = output
        return output

    def dataloader_iter(loader: torch.utils.data.DataLoader) -> Any:
        return _CaptureIterator(original_dataloader_iter(loader), state)

    def optimizer_init(optimizer: Optimizer, params: Any, defaults: dict[str, Any]) -> None:
        original_optimizer_init(optimizer, params, defaults)
        state.optimizer = _optimizer_metadata(optimizer)

    def optimizer_step(optimizer: Optimizer, *args: Any, **kwargs: Any) -> Any:
        state.optimizer_step_called = True
        state.optimizer = _optimizer_metadata(optimizer)
        if dry_run:
            return None
        original_step = optimizer_step_methods.get(optimizer.__class__)
        if original_step is None:
            original_step = optimizer_step_methods[Optimizer]
        return original_step(optimizer, *args, **kwargs)

    def scheduler_init(scheduler: LRScheduler, *args: Any, **kwargs: Any) -> None:
        original_scheduler_init(scheduler, *args, **kwargs)
        state.scheduler = {"class_name": scheduler.__class__.__name__}

    def scheduler_step(scheduler: LRScheduler, *args: Any, **kwargs: Any) -> Any:
        state.scheduler_step_called = True
        state.scheduler = {"class_name": scheduler.__class__.__name__}
        if dry_run and state.model is not None and state.optimizer_step_called:
            raise _CaptureComplete
        return original_scheduler_step(scheduler, *args, **kwargs)

    def backward(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        state.backward_called = True
        return original_backward(tensor, *args, **kwargs)

    def init_process_group(*args: Any, **kwargs: Any) -> Any:
        state.distributed_mutation = True
        if dry_run:
            raise _CaptureComplete
        return original_init_process_group(*args, **kwargs)

    nn.Module.__call__ = module_call
    torch.utils.data.DataLoader.__iter__ = dataloader_iter
    Optimizer.__init__ = optimizer_init
    for optimizer_cls in optimizer_step_methods:
        optimizer_cls.step = optimizer_step
    LRScheduler.__init__ = scheduler_init
    LRScheduler.step = scheduler_step
    torch.Tensor.backward = backward
    dist.init_process_group = init_process_group
    try:
        yield
    finally:
        nn.Module.__call__ = original_module_call
        torch.utils.data.DataLoader.__iter__ = original_dataloader_iter
        Optimizer.__init__ = original_optimizer_init
        for optimizer_cls, step in optimizer_step_methods.items():
            optimizer_cls.step = step
        LRScheduler.__init__ = original_scheduler_init
        LRScheduler.step = original_scheduler_step
        torch.Tensor.backward = original_backward
        dist.init_process_group = original_init_process_group


class _CaptureIterator:
    def __init__(self, iterator: Any, state: _CaptureState) -> None:
        self._iterator = iterator
        self._state = state

    def __iter__(self) -> "_CaptureIterator":
        return self

    def __next__(self) -> Any:
        item = next(self._iterator)
        if self._state.first_batch is None:
            self._state.first_batch = deepcopy(item)
        return item


def _looks_like_user_model(module: Any) -> bool:
    import torch

    return module.__class__.__module__ in {"__main__", "<run_path>"} and any(
        isinstance(item, torch.nn.Parameter) for item in module.parameters(recurse=True)
    )


def _build_context(state: _CaptureState) -> CapturedTrainingContext:
    if state.model is None:
        raise ValueError("cannot build capture context without a model")
    return CapturedTrainingContext(
        entrypoint_path=str(state.entrypoint_path),
        argv=state.argv,
        cwd=str(state.cwd),
        environment_summary={str(key): "present" for key in state.environment},
        capture_backend="torch-monkeypatch",
        capture_backend_version="v1",
        model_identity={
            "class_name": state.model.__class__.__name__,
            "module_name": state.model.__class__.__module__,
        },
        parameter_count=sum(parameter.numel() for parameter in state.model.parameters()),
        buffer_count=sum(buffer.numel() for buffer in state.model.buffers()),
        first_batch_structure=_structure(
            state.first_batch if state.first_batch is not None else state.args
        ),
        model_call={
            "args": _structure(state.args),
            "kwargs": _structure(state.kwargs),
        },
        tensor_metadata=_tensor_metadata(state.args, state.kwargs),
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        lifecycle={
            "loss_computed_before_backward": state.backward_called,
            "backward_called_before_optimizer_step": (
                state.backward_called and state.optimizer_step_called
            ),
            "scheduler_step_observed": state.scheduler_step_called,
        },
        model_output_structure=_structure(state.output),
        state_dict_key_to_canonical_state_id={
            key: f"state:{index:04d}"
            for index, key in enumerate(state.model.state_dict().keys())
        },
        distributed_mutation_observed_before_capture=state.distributed_mutation,
    )


def _optimizer_metadata(optimizer: Any) -> dict[str, Any]:
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                "parameter_count": len(group.get("params", ())),
                "hyperparameters": {
                    key: value
                    for key, value in group.items()
                    if key != "params" and isinstance(value, (str, int, float, bool))
                },
            }
        )
    return {"class_name": optimizer.__class__.__name__, "param_groups": groups}


def _structure(value: Any) -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_structure(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_structure(item) for item in value]}
    if isinstance(value, Mapping):
        return {
            "kind": "dict",
            "fields": {str(key): _structure(item) for key, item in value.items()},
        }
    return {"kind": type(value).__name__, "repr": repr(value)}


def _tensor_metadata(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(args):
        _collect_tensor_metadata(f"arg{index}", value, metadata)
    for key, value in kwargs.items():
        _collect_tensor_metadata(f"kwarg.{key}", value, metadata)
    return metadata


def _collect_tensor_metadata(
    prefix: str,
    value: Any,
    metadata: dict[str, dict[str, Any]],
) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        metadata[prefix] = _structure(value)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _collect_tensor_metadata(f"{prefix}.{index}", item, metadata)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _collect_tensor_metadata(f"{prefix}.{key}", item, metadata)
