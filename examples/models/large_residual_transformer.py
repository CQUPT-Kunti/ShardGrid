"""Configurable residual Transformer workload for large automatic-plan gates."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class LargeResidualTransformerConfig:
    vocab_size: int = 2048
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    ffn_size: int = 2048
    sequence_length: int = 16
    batch_size: int = 4
    memory_bank_rows: int = 0
    memory_bank_touch_rows: int = 1

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | None = None,
    ) -> "LargeResidualTransformerConfig":
        payload = dict(values or {})
        return cls(
            vocab_size=int(payload.get("vocab_size", cls.vocab_size)),
            hidden_size=int(payload.get("hidden_size", cls.hidden_size)),
            num_layers=int(payload.get("num_layers", cls.num_layers)),
            num_heads=int(payload.get("num_heads", cls.num_heads)),
            ffn_size=int(payload.get("ffn_size", cls.ffn_size)),
            sequence_length=int(payload.get("sequence_length", cls.sequence_length)),
            batch_size=int(payload.get("batch_size", cls.batch_size)),
            memory_bank_rows=int(payload.get("memory_bank_rows", cls.memory_bank_rows)),
            memory_bank_touch_rows=int(
                payload.get("memory_bank_touch_rows", cls.memory_bank_touch_rows)
            ),
        )

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        for name in (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_heads",
            "ffn_size",
            "sequence_length",
            "batch_size",
            "memory_bank_touch_rows",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.memory_bank_rows < 0:
            raise ValueError("memory_bank_rows must be >= 0")
        if self.memory_bank_rows and self.memory_bank_touch_rows > self.memory_bank_rows:
            raise ValueError("memory_bank_touch_rows must be <= memory_bank_rows")


class MemoryPressureAdapter(nn.Module):
    def __init__(self, config: LargeResidualTransformerConfig) -> None:
        super().__init__()
        self.touch_rows = config.memory_bank_touch_rows
        if config.memory_bank_rows:
            self.weight = nn.Parameter(torch.empty(config.memory_bank_rows, config.hidden_size))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            return x
        bias = self.weight[: self.touch_rows].mean(dim=0).view(1, 1, -1)
        return x + bias.to(dtype=x.dtype) * 0.001


def _merge_attention_qkv(
    *,
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, seq, hidden_times_three = qkv.shape
    hidden = hidden_times_three // 3
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    return (attn @ v).transpose(1, 2).reshape(batch, seq, hidden)


class SelfAttention(nn.Module):
    def __init__(self, config: LargeResidualTransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = _merge_attention_qkv(
            qkv=self.qkv(x),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        return self.out_proj(merged)


class TransformerBlock(nn.Module):
    def __init__(self, config: LargeResidualTransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_size),
            nn.GELU(),
            nn.Linear(config.ffn_size, config.hidden_size),
        )
        self.memory_pressure = MemoryPressureAdapter(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return self.memory_pressure(x + self.ffn(self.norm2(x)))


class LargeResidualTransformer(nn.Module):
    def __init__(
        self,
        config: LargeResidualTransformerConfig | None = None,
        *,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = config or LargeResidualTransformerConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.position_embedding = nn.Embedding(
            self.config.sequence_length,
            self.config.hidden_size,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.num_layers)
        )
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.output_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq = input_ids.shape[-1]
        positions = torch.arange(seq, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        long_skip = x
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index % 2 == 1:
                x = x + long_skip
                long_skip = x
        return self.output_head(self.norm(x))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def large_residual_module_paths(
    config: LargeResidualTransformerConfig | None = None,
) -> tuple[str, ...]:
    cfg = config or LargeResidualTransformerConfig()
    paths = ["token_embedding", "position_embedding"]
    for index in range(cfg.num_layers):
        paths.extend(
            (
                f"blocks.{index}.norm1",
                f"blocks.{index}.attn.qkv",
                f"blocks.{index}.attn.out_proj",
                f"blocks.{index}.norm2",
                f"blocks.{index}.ffn.0",
                f"blocks.{index}.ffn.1",
                f"blocks.{index}.ffn.2",
                f"blocks.{index}.memory_pressure",
            )
        )
    paths.extend(("norm", "output_head"))
    return tuple(paths)


@dataclass
class _StageState:
    input_ids: torch.Tensor | None = None
    x: torch.Tensor | None = None
    long_skip: torch.Tensor | None = None
    norm1_out: torch.Tensor | None = None
    attn_merged: torch.Tensor | None = None
    norm2_out: torch.Tensor | None = None
    ffn_hidden: torch.Tensor | None = None


_STATE_ORDER = (
    "input_ids",
    "x",
    "long_skip",
    "norm1_out",
    "attn_merged",
    "norm2_out",
    "ffn_hidden",
)


def _ordered_state_names(names: set[str]) -> tuple[str, ...]:
    return tuple(name for name in _STATE_ORDER if name in names)


def _path_uses(path: str) -> set[str]:
    if path == "token_embedding":
        return {"input_ids"}
    if path == "position_embedding":
        return {"x"}
    if path in {"norm", "output_head"}:
        return {"x"}
    if path.endswith("norm1"):
        return {"x"}
    if path.endswith("attn.qkv"):
        return {"norm1_out"}
    if path.endswith("attn.out_proj"):
        return {"x", "attn_merged"}
    if path.endswith("norm2"):
        return {"x"}
    if path == "ffn.0" or path.endswith("ffn.0"):
        return {"norm2_out"}
    if path.endswith(("ffn.1", "ffn.2")):
        return {"ffn_hidden"}
    if path.endswith("memory_pressure"):
        names = {"x", "ffn_hidden"}
        block_index = int(path.split(".")[1])
        if block_index % 2 == 1:
            names.add("long_skip")
        return names
    raise ValueError(f"unsupported stage module path {path!r}")


def _path_defs(path: str) -> set[str]:
    if path == "token_embedding":
        return {"x"}
    if path == "position_embedding":
        return {"x", "long_skip"}
    if path in {"norm", "output_head"}:
        return {"x"}
    if path.endswith("norm1"):
        return {"norm1_out"}
    if path.endswith("attn.qkv"):
        return {"attn_merged"}
    if path.endswith("attn.out_proj"):
        return {"x"}
    if path.endswith("norm2"):
        return {"norm2_out"}
    if path.endswith(("ffn.0", "ffn.1", "ffn.2")):
        return {"ffn_hidden"}
    if path.endswith("memory_pressure"):
        names = {"x"}
        block_index = int(path.split(".")[1])
        if block_index % 2 == 1:
            names.add("long_skip")
        return names
    raise ValueError(f"unsupported stage module path {path!r}")


def _default_boundary_state_names_for_start_path(start_path: str) -> tuple[str, ...]:
    if start_path == "token_embedding":
        return ("input_ids",)
    if start_path in {"position_embedding", "norm", "output_head"}:
        return ("x",)
    if start_path.endswith(("norm1", "norm2")):
        return ("x", "long_skip")
    if start_path.endswith("attn.qkv"):
        return ("x", "long_skip", "norm1_out")
    if start_path.endswith("attn.out_proj"):
        return ("x", "long_skip", "attn_merged")
    if start_path.endswith("ffn.0"):
        return ("x", "long_skip", "norm2_out")
    if start_path.endswith(("ffn.1", "ffn.2", "memory_pressure")):
        return ("x", "long_skip", "ffn_hidden")
    raise ValueError(f"unsupported stage start path {start_path!r}")


def required_boundary_state_names(
    module_paths: Sequence[str],
    required_outputs: Sequence[str] | None,
) -> tuple[str, ...]:
    if not module_paths:
        raise ValueError("module_paths must not be empty")
    live = set(required_outputs or ("x",))
    for path in reversed(tuple(module_paths)):
        live = (live - _path_defs(path)) | _path_uses(path)
    return _ordered_state_names(live)


class LargeResidualTransformerStage(nn.Module):
    def __init__(
        self,
        config: LargeResidualTransformerConfig,
        *,
        stage_id: str,
        module_paths: Sequence[str],
        next_stage_start_path: str | None,
        input_state_names: Sequence[str] | None = None,
        output_state_names: Sequence[str] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not module_paths:
            raise ValueError("module_paths must not be empty")
        self.config = config
        self.stage_id = stage_id
        self.module_paths = tuple(str(path) for path in module_paths)
        self.next_stage_start_path = next_stage_start_path
        self.input_state_names = tuple(
            input_state_names or _default_boundary_state_names_for_start_path(self.module_paths[0])
        )
        self.output_state_names = tuple(
            output_state_names
            or (
                ("x",)
                if next_stage_start_path is None
                else _default_boundary_state_names_for_start_path(next_stage_start_path)
            )
        )
        path_order = {
            path: index for index, path in enumerate(large_residual_module_paths(config))
        }
        indices = [path_order[path] for path in self.module_paths]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("stage module_paths must be one contiguous slice")
        torch.manual_seed(seed)
        self.blocks = nn.ModuleDict()
        self._path_modules: dict[str, nn.Module] = {}
        for path in self.module_paths:
            self._path_modules[path] = self._register_path_module(path)

    def forward(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        state = self._decode_inputs(args)
        for path in self.module_paths:
            self._apply_path(path, state)
        return self._encode_outputs(state)

    def parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def _register_path_module(self, path: str) -> nn.Module:
        if path == "token_embedding":
            module = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.token_embedding = module
            return module
        if path == "position_embedding":
            module = nn.Embedding(self.config.sequence_length, self.config.hidden_size)
            self.position_embedding = module
            return module
        if path == "norm":
            module = nn.LayerNorm(self.config.hidden_size)
            self.norm = module
            return module
        if path == "output_head":
            module = nn.Linear(self.config.hidden_size, self.config.vocab_size)
            self.output_head = module
            return module
        if not path.startswith("blocks."):
            raise ValueError(f"unsupported stage module path {path!r}")
        _blocks, index, section, *rest = path.split(".")
        block = self.blocks[index] if index in self.blocks else None
        if block is None:
            block = nn.Module()
            self.blocks[index] = block
        if section == "norm1":
            module = nn.LayerNorm(self.config.hidden_size)
            block.norm1 = module
            return module
        if section == "norm2":
            module = nn.LayerNorm(self.config.hidden_size)
            block.norm2 = module
            return module
        if section == "memory_pressure":
            module = MemoryPressureAdapter(self.config)
            block.memory_pressure = module
            return module
        if section == "attn":
            attn = getattr(block, "attn", None)
            if attn is None:
                attn = nn.Module()
                block.attn = attn
            leaf = rest[0]
            if leaf == "qkv":
                module = nn.Linear(self.config.hidden_size, self.config.hidden_size * 3)
                attn.qkv = module
                return module
            if leaf == "out_proj":
                module = nn.Linear(self.config.hidden_size, self.config.hidden_size)
                attn.out_proj = module
                return module
        if section == "ffn":
            ffn = getattr(block, "ffn", None)
            if ffn is None:
                ffn = nn.ModuleDict()
                block.ffn = ffn
            leaf = rest[0]
            if leaf == "0":
                module = nn.Linear(self.config.hidden_size, self.config.ffn_size)
            elif leaf == "1":
                module = nn.GELU()
            elif leaf == "2":
                module = nn.Linear(self.config.ffn_size, self.config.hidden_size)
            else:
                raise ValueError(f"unsupported FFN path {path!r}")
            ffn[leaf] = module
            return module
        raise ValueError(f"unsupported stage module path {path!r}")

    def _decode_inputs(self, args: Sequence[torch.Tensor]) -> _StageState:
        values = tuple(args)
        _expect_arity(self.module_paths[0], values, len(self.input_state_names))
        state = _StageState()
        for name, value in zip(self.input_state_names, values, strict=True):
            setattr(state, name, value)
        return state

    def _apply_path(self, path: str, state: _StageState) -> None:
        module = self._path_modules[path]
        if path == "token_embedding":
            state.x = module(_require(state.input_ids, "input_ids"))
            return
        if path == "position_embedding":
            x = _require(state.x, "x")
            positions = torch.arange(x.shape[1], device=x.device)
            state.x = x + module(positions)
            state.long_skip = state.x
            return
        if path == "norm":
            state.x = module(_require(state.x, "x"))
            return
        if path == "output_head":
            state.x = module(_require(state.x, "x"))
            return
        parts = path.split(".")
        block_index = int(parts[1])
        leaf = ".".join(parts[2:])
        if leaf == "norm1":
            state.norm1_out = module(_require(state.x, "x"))
            return
        if leaf == "attn.qkv":
            state.attn_merged = _merge_attention_qkv(
                qkv=module(_require(state.norm1_out, "norm1_out")),
                num_heads=self.config.num_heads,
                head_dim=self.config.hidden_size // self.config.num_heads,
            )
            return
        if leaf == "attn.out_proj":
            state.x = _require(state.x, "x") + module(
                _require(state.attn_merged, "attn_merged")
            )
            return
        if leaf == "norm2":
            state.norm2_out = module(_require(state.x, "x"))
            return
        if leaf == "ffn.0":
            state.ffn_hidden = module(_require(state.norm2_out, "norm2_out"))
            return
        if leaf in {"ffn.1", "ffn.2"}:
            state.ffn_hidden = module(_require(state.ffn_hidden, "ffn_hidden"))
            return
        if leaf == "memory_pressure":
            state.x = module(_require(state.x, "x") + _require(state.ffn_hidden, "ffn_hidden"))
            if block_index % 2 == 1:
                state.x = _require(state.x, "x") + _require(state.long_skip, "long_skip")
                state.long_skip = state.x
            return
        raise ValueError(f"unsupported stage module path {path!r}")

    def _encode_outputs(self, state: _StageState) -> torch.Tensor | tuple[torch.Tensor, ...]:
        values = tuple(_require(getattr(state, name), name) for name in self.output_state_names)
        if len(values) == 1:
            return values[0]
        return values


def build_large_residual_transformer(
    config: LargeResidualTransformerConfig | None = None,
    *,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> LargeResidualTransformer:
    context = nullcontext() if device is None else torch.device(device)
    with context:
        return LargeResidualTransformer(config=config, seed=seed)


def build_large_residual_transformer_stage(
    config: LargeResidualTransformerConfig,
    *,
    stage_metadata: Any,
    next_stage_start_path: str | None,
    input_state_names: Sequence[str] | None = None,
    output_state_names: Sequence[str] | None = None,
    seed: int = 42,
) -> LargeResidualTransformerStage:
    return LargeResidualTransformerStage(
        config,
        stage_id=str(stage_metadata.stage_id),
        module_paths=tuple(str(path) for path in stage_metadata.module_paths),
        next_stage_start_path=next_stage_start_path,
        input_state_names=input_state_names,
        output_state_names=output_state_names,
        seed=seed,
    )


def make_large_residual_stage_inputs(
    config: LargeResidualTransformerConfig,
    start_path: str,
    *,
    state_names: Sequence[str] | None = None,
    seed: int = 42,
    step: int = 0,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, ...]:
    effective_batch_size = config.batch_size if batch_size is None else batch_size
    shape = (effective_batch_size, config.sequence_length, config.hidden_size)
    names = tuple(state_names or _default_boundary_state_names_for_start_path(start_path))
    ffn_hidden_width = (
        config.hidden_size if start_path.endswith("memory_pressure") else config.ffn_size
    )
    if start_path == "token_embedding":
        inputs, _targets = make_large_residual_batch(
            config,
            seed=seed,
            step=step,
            batch_size=effective_batch_size,
            device=device,
        )
        return (inputs,)
    tensors: dict[str, torch.Tensor] = {
        "x": torch.zeros(shape, device=device),
        "long_skip": torch.zeros(shape, device=device),
        "norm1_out": torch.zeros(shape, device=device),
        "attn_merged": torch.zeros(shape, device=device),
        "norm2_out": torch.zeros(shape, device=device),
        "ffn_hidden": torch.zeros(
            effective_batch_size,
            config.sequence_length,
            ffn_hidden_width,
            device=device,
        ),
    }
    if "input_ids" in names:
        raise ValueError("input_ids stage inputs must start at token_embedding")
    return tuple(tensors[name] for name in names)


def make_large_residual_batch(
    config: LargeResidualTransformerConfig | None = None,
    *,
    seed: int = 42,
    step: int = 0,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = config or LargeResidualTransformerConfig()
    shape = (cfg.batch_size if batch_size is None else batch_size, cfg.sequence_length)
    if str(device) == "meta":
        inputs = torch.zeros(shape, dtype=torch.long, device=device)
        targets = torch.zeros(shape, dtype=torch.long, device=device)
        return inputs, targets
    generator = torch.Generator(device="cpu" if str(device) == "cpu" else str(device))
    generator.manual_seed(seed + step)
    inputs = torch.randint(
        0,
        cfg.vocab_size,
        shape,
        generator=generator,
        device=device,
    )
    targets = torch.randint(
        0,
        cfg.vocab_size,
        shape,
        generator=generator,
        device=device,
    )
    return inputs, targets


def train_step(
    model: LargeResidualTransformer,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    inputs, targets = batch
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss


def _expect_arity(path: str, values: Sequence[torch.Tensor], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{path} expects {expected} inputs, got {len(values)}")


def _require(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"missing stage state value {name}")
    return value
