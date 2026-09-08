"""In-process user entrypoint bootstrap and first-step capture."""

from __future__ import annotations

import ast
import json
import os
import runpy
import sys
import textwrap
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from shardgrid.common.enums import Health
from shardgrid.common.manual_actions import classify_manual_action
from shardgrid.common.process import ProcessResult
from shardgrid.platforms.linux import LinuxPlatform
from shardgrid.transport.runtime import WSLRuntimeWrapper, wrap_wsl_direct_command


@dataclass(frozen=True)
class BootstrapExecution:
    target: str
    action: str
    before_state: dict[str, Any] | None
    execution: str
    after_verification: dict[str, Any] | None
    verified: bool = False
    failure_reason: str | None = None
    manual_action: str | None = None
    command: str | None = None
    commands_run: tuple[str, ...] = ()

    @property
    def effective_state(self) -> dict[str, Any] | None:
        return self.after_verification or self.before_state


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_bootstrap_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None


def _payload_health(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    health = payload.get("health")
    return None if health is None else str(health)


def _payload_manual_action(payload: dict[str, Any] | None) -> str | None:
    for action in (payload or {}).get("manual_actions", []):
        text = str(action)
        blocker = classify_manual_action(text)
        if blocker is not None:
            return text
    actions = (payload or {}).get("manual_actions") or []
    return str(actions[0]) if actions else None


def _payload_commands(payload: dict[str, Any] | None) -> tuple[str, ...]:
    return tuple(str(item) for item in ((payload or {}).get("commands_run") or []))


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _fix_execution(payload: dict[str, Any] | None, *, verified: bool) -> str:
    if not verified or payload is None:
        return "blocked"
    if _payload_health(payload) == Health.BLOCKED_MANUAL_ACTION.value:
        return "blocked"
    if _payload_manual_action(payload):
        return "blocked"
    return "executed"


def run_control_bootstrap(*, fix: bool) -> BootstrapExecution:
    platform = LinuxPlatform()
    script_path = _repo_root() / "scripts" / "bootstrap-linux.sh"
    before_step = platform.bootstrap_step(
        "bootstrap-check", ["bash", str(script_path), "--check", "--json"]
    )
    before = platform.run(before_step.command, timeout=60)
    before_payload = _parse_bootstrap_json(before.stdout)
    if before_payload is None:
        return BootstrapExecution(
            target="control",
            action="bootstrap-linux --check",
            before_state=None,
            execution="blocked",
            after_verification=None,
            verified=False,
            failure_reason=(
                before.stderr.strip()
                or before.stdout.strip()
                or "control bootstrap check failed"
            ),
            command=before.recorded_command,
        )
    if not fix:
        return BootstrapExecution(
            target="control",
            action="bootstrap-linux --check",
            before_state=before_payload,
            execution="skipped",
            after_verification=before_payload,
            verified=True,
            manual_action=_payload_manual_action(before_payload),
            command=before.recorded_command,
            commands_run=_payload_commands(before_payload),
        )
    if _payload_health(before_payload) in {
        Health.HEALTHY.value,
        Health.BLOCKED_MANUAL_ACTION.value,
    }:
        execution = (
            "skipped"
            if _payload_health(before_payload) == Health.HEALTHY.value
            else "blocked"
        )
        return BootstrapExecution(
            target="control",
            action="bootstrap-linux --install-deps",
            before_state=before_payload,
            execution=execution,
            after_verification=before_payload,
            verified=True,
            manual_action=_payload_manual_action(before_payload),
            command=before.recorded_command,
            commands_run=_payload_commands(before_payload),
        )
    fix_result = platform.run(
        platform.bootstrap_step(
            "bootstrap-fix",
            ["bash", str(script_path), "--install-deps", "--json"],
        ).command,
        timeout=120,
    )
    fixed_payload = _parse_bootstrap_json(fix_result.stdout)
    verify = platform.run(
        platform.bootstrap_step(
            "bootstrap-verify",
            ["bash", str(script_path), "--check", "--json"],
        ).command,
        timeout=60,
    )
    verify_payload = _parse_bootstrap_json(verify.stdout)
    effective = verify_payload
    verified = verify_payload is not None
    return BootstrapExecution(
        target="control",
        action="bootstrap-linux --install-deps",
        before_state=before_payload,
        execution=_fix_execution(effective or fixed_payload, verified=verified),
        after_verification=effective or fixed_payload,
        verified=verified,
        failure_reason=(
            None
            if verified and _payload_health(effective) == Health.HEALTHY.value
            else _first_nonempty(
                verify.stderr,
                verify.stdout if not verified else None,
                fix_result.stderr,
                fix_result.stdout if fixed_payload is None else None,
                "control bootstrap verification failed",
            )
        ),
        manual_action=_payload_manual_action(effective or fixed_payload),
        command=fix_result.recorded_command,
        commands_run=_payload_commands(effective or fixed_payload),
    )


def _run_worker_bootstrap(
    wrapper: WSLRuntimeWrapper,
    *,
    peer_ip: str,
    expected_mtu: int,
    fix: bool,
) -> tuple[ProcessResult, dict[str, Any] | None]:
    script_path = _repo_root() / "scripts" / "bootstrap-wsl.sh"
    command = ["--fix-nccl-mtu-only", "--json"] if fix else ["--check", "--json"]
    payload = (
        f"SHARDGRID_NCCL_PEER_IP={peer_ip} "
        f"SHARDGRID_NCCL_MTU={expected_mtu} "
        "SHARDGRID_BOOTSTRAP_JSON=1 "
        "SHARDGRID_WSL_PERSIST_NCCL_MTU=0 "
        "bash -s -- " + " ".join(command)
    )
    remote_command = wrap_wsl_direct_command(
        wrapper.config.distro or "",
        wrapper.config.user or "root",
        payload,
    )
    result = wrapper.executor.run(
        remote_command,
        stdin=script_path.read_text(encoding="utf-8"),
        timeout=60,
    )
    parsed = _parse_bootstrap_json(result.stdout)
    if parsed is not None:
        return result, parsed
    latest = wrapper.run('cat "$HOME/.shardgrid/bootstrap/wsl-latest.json"', timeout=15)
    return result, _parse_bootstrap_json(latest.stdout)


def run_worker_runtime_bootstrap(
    wrapper: WSLRuntimeWrapper,
    *,
    peer_ip: str,
    expected_mtu: int,
    fix: bool,
) -> BootstrapExecution:
    before_result, before_payload = _run_worker_bootstrap(
        wrapper, peer_ip=peer_ip, expected_mtu=expected_mtu, fix=False
    )
    if before_payload is None:
        return BootstrapExecution(
            target=peer_ip,
            action="bootstrap-wsl --check",
            before_state=None,
            execution="blocked",
            after_verification=None,
            verified=False,
            failure_reason=(
                before_result.stderr.strip()
                or before_result.stdout.strip()
                or "remote bootstrap failed"
            ),
            command=before_result.recorded_command,
        )
    before_status = str(((before_payload.get("nccl_path_mtu") or {}).get("status")) or "")
    before_manual_action = _payload_manual_action(before_payload)
    if not fix or before_status.upper() == "PASS":
        return BootstrapExecution(
            target=peer_ip,
            action="bootstrap-wsl --fix-nccl-mtu-only",
            before_state=before_payload,
            execution="skipped",
            after_verification=before_payload,
            verified=True,
            manual_action=before_manual_action,
            command=before_result.recorded_command,
            commands_run=_payload_commands(before_payload),
        )
    if before_manual_action:
        return BootstrapExecution(
            target=peer_ip,
            action="bootstrap-wsl --fix-nccl-mtu-only",
            before_state=before_payload,
            execution="blocked",
            after_verification=before_payload,
            verified=True,
            manual_action=before_manual_action,
            command=before_result.recorded_command,
            commands_run=_payload_commands(before_payload),
        )
    fix_result, fix_payload = _run_worker_bootstrap(
        wrapper, peer_ip=peer_ip, expected_mtu=expected_mtu, fix=True
    )
    verify_result, verify_payload = _run_worker_bootstrap(
        wrapper, peer_ip=peer_ip, expected_mtu=expected_mtu, fix=False
    )
    effective = verify_payload
    verified = verify_payload is not None
    fallback = effective or fix_payload
    after_status = str((((effective or {}).get("nccl_path_mtu") or {}).get("status")) or "")
    return BootstrapExecution(
        target=peer_ip,
        action="bootstrap-wsl --fix-nccl-mtu-only",
        before_state=before_payload,
        execution=_fix_execution(fallback, verified=verified),
        after_verification=fallback,
        verified=verified,
        failure_reason=(
            None
            if verified and after_status.upper() == "PASS"
            else _first_nonempty(
                verify_result.stderr,
                verify_result.stdout if not verified else None,
                fix_result.stderr,
                fix_result.stdout if fix_payload is None else None,
                "remote bootstrap verification failed",
            )
        ),
        manual_action=_payload_manual_action(fallback),
        command=fix_result.recorded_command,
        commands_run=_payload_commands(fallback),
    )


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
    lifecycle: dict[str, Any]
    model_output_structure: dict[str, Any]
    state_dict_key_to_canonical_state_id: dict[str, str]
    distributed_mutation_observed_before_capture: bool = False
    graph_capture: dict[str, Any] = field(default_factory=dict)


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
    optimizer_ref: Any | None = None
    optimizer: dict[str, Any] | None = None
    scheduler: dict[str, Any] | None = None
    backward_call_count: int = 0
    optimizer_step_called: bool = False
    scheduler_step_called: bool = False
    distributed_mutation: bool = False
    graph_capture_backend: str | None = None
    graph_capture_diagnostics: tuple[str, ...] = ()
    failure: CaptureFailure | None = None
    amp_autocast_observed: bool = False
    amp_scaler_observed: bool = False


class _CaptureComplete(RuntimeError):
    pass


class _CaptureFailed(RuntimeError):
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
    except _CaptureFailed:
        return CaptureResult(ok=False, failure=state.failure)
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
    if state.model is not None:
        return state
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
    original_amp_autocast = torch.amp.autocast
    original_cuda_amp_autocast = torch.cuda.amp.autocast
    original_amp_grad_scaler_init = torch.amp.GradScaler.__init__
    original_cuda_amp_grad_scaler_init = torch.cuda.amp.GradScaler.__init__
    original_torch_empty = torch.empty
    optimizer_step_methods = {
        cls: cls.step
        for cls in (Optimizer, torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)
    }
    control_plane_ram_budget = _control_plane_ram_budget_bytes()

    def module_call(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        if dry_run and state.model is None and _looks_like_user_model(module):
            state.model = module
            state.args = tuple(args)
            state.kwargs = dict(kwargs)
            state.output = None
            _capture_declared_graph_or_fail(state)
            raise _CaptureComplete
        output = original_module_call(module, *args, **kwargs)
        if state.model is None and _looks_like_user_model(module):
            state.model = module
            state.args = tuple(args)
            state.kwargs = dict(kwargs)
            state.output = output
            _capture_graph_or_fail(state)
        return output

    def dataloader_iter(loader: torch.utils.data.DataLoader) -> Any:
        return _CaptureIterator(original_dataloader_iter(loader), state)

    def optimizer_init(optimizer: Optimizer, params: Any, defaults: dict[str, Any]) -> None:
        if not optimizer.__class__.__module__.startswith("torch.optim"):
            state.failure = CaptureFailure(
                stage="lifecycle_capture",
                code="HIDDEN_OPTIMIZER_MUTATION_UNSUPPORTED",
                message=(
                    "custom optimizer step semantics cannot be safely captured before "
                    "distributed mutation"
                ),
                artifact_log_ref="diagnostics/capture.json",
            )
            raise _CaptureFailed
        original_optimizer_init(optimizer, params, defaults)
        state.optimizer_ref = optimizer
        state.optimizer = _optimizer_metadata(optimizer, state.model)

    def optimizer_step(optimizer: Optimizer, *args: Any, **kwargs: Any) -> Any:
        state.optimizer_step_called = True
        state.optimizer_ref = optimizer
        state.optimizer = _optimizer_metadata(optimizer, state.model)
        if dry_run:
            return None
        original_step = optimizer_step_methods.get(optimizer.__class__)
        if original_step is None:
            original_step = optimizer_step_methods[Optimizer]
        return original_step(optimizer, *args, **kwargs)

    def scheduler_init(scheduler: LRScheduler, *args: Any, **kwargs: Any) -> None:
        original_scheduler_init(scheduler, *args, **kwargs)
        state.scheduler = _scheduler_metadata(scheduler)

    def scheduler_step(scheduler: LRScheduler, *args: Any, **kwargs: Any) -> Any:
        state.scheduler_step_called = True
        state.scheduler = _scheduler_metadata(scheduler)
        if dry_run and state.model is not None and state.optimizer_step_called:
            raise _CaptureComplete
        return original_scheduler_step(scheduler, *args, **kwargs)

    def backward(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        state.backward_call_count += 1
        return original_backward(tensor, *args, **kwargs)

    def init_process_group(*args: Any, **kwargs: Any) -> Any:
        state.distributed_mutation = True
        if dry_run:
            raise _CaptureComplete
        return original_init_process_group(*args, **kwargs)

    def amp_autocast(*args: Any, **kwargs: Any) -> Any:
        state.amp_autocast_observed = True
        return original_amp_autocast(*args, **kwargs)

    def cuda_amp_autocast(*args: Any, **kwargs: Any) -> Any:
        state.amp_autocast_observed = True
        return original_cuda_amp_autocast(*args, **kwargs)

    def amp_grad_scaler_init(scaler: Any, *args: Any, **kwargs: Any) -> None:
        state.amp_scaler_observed = True
        original_amp_grad_scaler_init(scaler, *args, **kwargs)

    def cuda_amp_grad_scaler_init(scaler: Any, *args: Any, **kwargs: Any) -> None:
        state.amp_scaler_observed = True
        original_cuda_amp_grad_scaler_init(scaler, *args, **kwargs)

    def bounded_empty(*size: Any, **kwargs: Any) -> Any:
        requested_bytes = _requested_tensor_storage_bytes(
            size,
            dtype=kwargs.get("dtype"),
            original_empty=original_torch_empty,
        )
        if dry_run and control_plane_ram_budget and requested_bytes > control_plane_ram_budget:
            bounded_kwargs = dict(kwargs)
            bounded_kwargs.pop("device", None)
            return original_torch_empty((1,), **bounded_kwargs)
        return original_torch_empty(*size, **kwargs)

    nn.Module.__call__ = module_call
    torch.utils.data.DataLoader.__iter__ = dataloader_iter
    Optimizer.__init__ = optimizer_init
    for optimizer_cls in optimizer_step_methods:
        optimizer_cls.step = optimizer_step
    LRScheduler.__init__ = scheduler_init
    LRScheduler.step = scheduler_step
    torch.Tensor.backward = backward
    dist.init_process_group = init_process_group
    torch.amp.autocast = amp_autocast
    torch.cuda.amp.autocast = cuda_amp_autocast
    torch.amp.GradScaler.__init__ = amp_grad_scaler_init
    torch.cuda.amp.GradScaler.__init__ = cuda_amp_grad_scaler_init
    torch.empty = bounded_empty
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
        torch.amp.autocast = original_amp_autocast
        torch.cuda.amp.autocast = original_cuda_amp_autocast
        torch.amp.GradScaler.__init__ = original_amp_grad_scaler_init
        torch.cuda.amp.GradScaler.__init__ = original_cuda_amp_grad_scaler_init
        torch.empty = original_torch_empty


def _control_plane_ram_budget_bytes() -> int | None:
    value = os.environ.get("SHARDGRID_CONTROL_PLANE_RAM_BUDGET_BYTES")
    if value is None:
        return None
    try:
        budget = int(value)
    except ValueError:
        return None
    return budget if budget > 0 else None


def _requested_tensor_storage_bytes(
    size: tuple[Any, ...],
    *,
    dtype: Any,
    original_empty: Any,
) -> int:
    import torch

    dims: tuple[Any, ...]
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        dims = tuple(size[0])
    else:
        dims = size
    numel = 1
    for dim in dims:
        if dim == ():
            continue
        try:
            numel *= int(dim)
        except (TypeError, ValueError):
            return 0
    tensor_dtype = dtype or torch.float32
    return numel * original_empty((), dtype=tensor_dtype).element_size()


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
        optimizer=(
            _optimizer_metadata(state.optimizer_ref, state.model)
            if state.optimizer_ref is not None
            else state.optimizer
        ),
        scheduler=state.scheduler,
        lifecycle={
            "loss_computed_before_backward": state.backward_call_count > 0,
            "backward_called_before_optimizer_step": (
                state.backward_call_count > 0 and state.optimizer_step_called
            ),
            "scheduler_step_observed": state.scheduler_step_called,
            "gradient_accumulation_observed": state.backward_call_count > 1,
            "backward_call_count": state.backward_call_count,
            "amp_autocast_observed": state.amp_autocast_observed,
            "amp_scaler_observed": state.amp_scaler_observed,
        },
        model_output_structure=_structure(state.output),
        state_dict_key_to_canonical_state_id={
            key: f"state:{index:04d}"
            for index, key in enumerate(state.model.state_dict().keys())
        },
        distributed_mutation_observed_before_capture=state.distributed_mutation,
        graph_capture={
            "backend": state.graph_capture_backend,
            "diagnostics": list(state.graph_capture_diagnostics),
        },
    )


def _capture_declared_graph_or_fail(state: _CaptureState) -> None:
    if state.model is None:
        raise ValueError("cannot capture graph metadata without a model")
    if not _tensor_metadata(state.args, state.kwargs):
        state.failure = CaptureFailure(
            stage="capture",
            code="MISSING_REQUIRED_METADATA",
            message="model call did not expose tensor input metadata required for planning",
            artifact_log_ref="diagnostics/capture.json",
        )
        raise _CaptureFailed
    if _forward_contains_python_control_flow(state.model):
        state.failure = CaptureFailure(
            stage="graph_capture",
            code="DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
            message=(
                "model forward contains Python control flow that cannot be proven "
                "safe without real CPU execution"
            ),
            artifact_log_ref="diagnostics/capture.json",
        )
        raise _CaptureFailed
    if _forward_contains_custom_function_call(state.model):
        state.failure = CaptureFailure(
            stage="graph_capture",
            code="CUSTOM_OP_UNSUPPORTED",
            message=(
                "model forward calls a non-module custom function that cannot be "
                "proven safe without real CPU execution"
            ),
            artifact_log_ref="diagnostics/capture.json",
        )
        raise _CaptureFailed
    module_count = sum(1 for _ in state.model.named_modules())
    state_count = sum(1 for _ in state.model.state_dict().keys())
    state.graph_capture_backend = "shardgrid.metadata_declaration"
    state.graph_capture_diagnostics = (
        "BOUNDED_METADATA_CAPTURE_WITHOUT_REAL_CPU_EXECUTION",
        f"MODULE_COUNT:{module_count}",
        f"STATE_OBJECT_COUNT:{state_count}",
    )


def _forward_contains_python_control_flow(model: Any) -> bool:
    import inspect

    try:
        source = inspect.getsource(model.forward)
    except (OSError, TypeError):
        return True
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            return True
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Call) for child in ast.walk(node.test)
        ):
            return True
    return False


def _forward_contains_custom_function_call(model: Any) -> bool:
    tree = _forward_ast(model)
    if tree is None:
        return True
    allowed_names = {"getattr", "len", "max", "min", "range", "sum"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        for child in ast.walk(node.value):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id not in allowed_names:
                    return True
    return False


def _forward_ast(model: Any) -> ast.AST | None:
    import inspect

    try:
        source = inspect.getsource(model.forward)
    except (OSError, TypeError):
        return None
    try:
        return ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None


def _capture_graph_or_fail(state: _CaptureState) -> None:
    from shardgrid.planner.generic_graph import (
        GraphCaptureUnsupported,
        capture_generic_graph_with_backend_fallback,
    )

    try:
        result = capture_generic_graph_with_backend_fallback(
            state.model,
            sample_args=state.args,
            sample_kwargs=state.kwargs,
        )
    except GraphCaptureUnsupported as exc:
        state.failure = CaptureFailure(
            stage="graph_capture",
            code=exc.code,
            message=str(exc),
            artifact_log_ref="diagnostics/capture.json",
        )
        raise _CaptureFailed from exc
    state.graph_capture_backend = result.canonical_graph.capture_backend
    state.graph_capture_diagnostics = tuple(result.diagnostics)


def _optimizer_metadata(optimizer: Any, model: Any | None = None) -> dict[str, Any]:
    parameter_keys_by_object = _parameter_keys_by_object(model)
    state_ids_by_key = _state_ids_by_key(model)
    groups = []
    for group in optimizer.param_groups:
        parameter_keys = [
            key
            for parameter in group.get("params", ())
            for key in parameter_keys_by_object.get(id(parameter), ())
        ]
        groups.append(
            {
                "parameter_count": len(group.get("params", ())),
                "parameter_keys": parameter_keys,
                "canonical_state_ids": [
                    state_ids_by_key[key]
                    for key in parameter_keys
                    if key in state_ids_by_key
                ],
                "hyperparameters": {
                    key: value
                    for key, value in group.items()
                    if key != "params" and isinstance(value, (str, int, float, bool))
                },
            }
        )
    return {"class_name": optimizer.__class__.__name__, "param_groups": groups}


def _scheduler_metadata(scheduler: Any) -> dict[str, Any]:
    return {
        "class_name": scheduler.__class__.__name__,
        "last_epoch": getattr(scheduler, "last_epoch", None),
    }


def _parameter_keys_by_object(model: Any | None) -> dict[int, list[str]]:
    if model is None:
        return {}
    keys: dict[int, list[str]] = {}
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = model.named_parameters()
    for key, parameter in named_parameters:
        keys.setdefault(id(parameter), []).append(str(key))
    return keys


def _state_ids_by_key(model: Any | None) -> dict[str, str]:
    if model is None:
        return {}
    return {
        key: f"state:{index:04d}"
        for index, key in enumerate(model.state_dict().keys())
    }


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
