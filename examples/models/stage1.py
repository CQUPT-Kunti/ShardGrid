"""Stage 1 of the supported minimal Transformer (T069).

Stage 1 owns the second Transformer block, the final layer norm, and the
output head.  Its input IS the activation interface from Stage 0: a hidden
tensor of shape ``(batch, seq, hidden_size)``.  ``backward_from_loss`` is
the explicit gradient interface: it computes the loss from the logits and
runs ``backward`` so gradients reach both Stage 1 parameters and the input
activation tensor (Stage 0's gradient interface).
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import traceback
from pathlib import Path

import torch
from examples.models.minimal_transformer import (
    AttentionBlock,
    MinimalTransformerConfig,
    set_seed,
)
from torch import nn


def _diag_enabled() -> bool:
    return os.environ.get("SHARDGRID_T072_STAGE1_DIAG", "").strip() == "1"


def _diag_event(event: str, start: float, **extra: object) -> None:
    payload = {
        "event": event,
        "timestamp": time.time(),
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }
    payload.update(extra)
    print("T072_STAGE1_PROFILE " + json.dumps(payload, sort_keys=True), flush=True)


def _diag_mark(name: str) -> None:
    print(name, flush=True)


def _rank_text() -> str:
    return os.environ.get("RANK", "?")


def _stage1_marker(name: str, **extra: object) -> None:
    payload = {
        "marker": name,
        "rank": _rank_text(),
        "hostname": socket.gethostname(),
        "timestamp": time.time(),
    }
    payload.update(extra)
    print("T072_STAGE1_REMOTE " + json.dumps(payload, sort_keys=True), flush=True)
    if name == "STAGE1_FORWARD_ENTER":
        flag = os.environ.get("SHARDGRID_T072_FORWARD_ENTER_FLAG", "").strip()
        if flag:
            try:
                Path(flag).write_text(str(payload["timestamp"]), encoding="utf-8")
            except Exception:
                pass


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _tensor_state(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
        "is_leaf": bool(tensor.is_leaf),
    }


def _diag_stage(
    name: str,
    start: float,
    tensor: torch.Tensor | None = None,
    **extra: object,
) -> None:
    payload = dict(extra)
    if tensor is not None:
        payload.update(_tensor_state(tensor))
    _diag_event(name, start, **payload)


def _diag_fail(stage: str, error: BaseException, start: float) -> None:
    _diag_event(
        "fail",
        start,
        stage=stage,
        exception_type=type(error).__name__,
        exception_message=str(error),
        traceback=traceback.format_exc(),
    )


def _forward_error(module_name: str, error: BaseException) -> None:
    _stage1_marker(
        "FORWARD_ERROR",
        module_name=module_name,
        exception_type=type(error).__name__,
        exception_message=str(error),
        traceback=traceback.format_exc(),
    )


def _sync_if_cuda(tensor: torch.Tensor) -> None:
    _ = tensor


class Stage1(nn.Module):
    """Second Transformer block + layer norm + output head (logits output)."""

    def __init__(
        self,
        config: MinimalTransformerConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MinimalTransformerConfig()
        if seed is not None:
            set_seed(seed)
        self.block1 = AttentionBlock(self.config)
        self.ln = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Activation interface: hidden activations -> logits.

        ``hidden`` must have shape ``(batch, seq_length, hidden_size)`` with
        the model dtype; returns logits of shape
        ``(batch, seq_length, vocab_size)``.
        """
        try:
            _stage1_marker("STAGE1_FORWARD_ENTER", **_tensor_state(hidden))
        except Exception as error:
            _stage1_marker(
                "ERROR_STAGE",
                stage="stage1_forward_enter",
                exception_type=type(error).__name__,
                exception_message=str(error),
                traceback=traceback.format_exc(),
            )
            raise
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be 3-D, got shape {tuple(hidden.shape)}")
        expected_hidden = self.config.hidden_size
        if hidden.shape[-1] != expected_hidden:
            raise ValueError(
                f"hidden last dim {hidden.shape[-1]} != hidden_size {expected_hidden}"
            )
        try:
            block_index = 0
            _stage1_marker(f"BLOCK_{block_index}_BEGIN", **_tensor_state(hidden))
            hidden_ln1 = self.block1.ln1(hidden)

            try:
                _stage1_marker(f"ATTENTION_{block_index}_BEGIN", **_tensor_state(hidden_ln1))
                _stage1_marker(f"BLOCK_{block_index}_ATTENTION_BEGIN", **_tensor_state(hidden_ln1))
                qkv = self.block1.qkv(hidden_ln1)
                head_dim = self.config.hidden_size // self.config.num_attention_heads
                q, k, v = qkv.chunk(3, dim=-1)
                batch, seq, _ = q.shape
                q = q.view(batch, seq, self.config.num_attention_heads, head_dim).transpose(1, 2)
                k = k.view(batch, seq, self.config.num_attention_heads, head_dim).transpose(1, 2)
                v = v.view(batch, seq, self.config.num_attention_heads, head_dim).transpose(1, 2)
                attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
                causal = torch.tril(
                    torch.ones(seq, seq, dtype=attn.dtype, device=attn.device)
                ).view(1, 1, seq, seq)
                attn = attn.masked_fill(causal == 0, float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                out = (attn @ v).transpose(1, 2).reshape(batch, seq, self.config.hidden_size)
                x = hidden + self.block1.proj(out)
                _stage1_marker(f"ATTENTION_{block_index}_END", **_tensor_state(x))
                _stage1_marker(f"BLOCK_{block_index}_ATTENTION_END", **_tensor_state(x))
            except Exception as error:
                _forward_error(f"ATTENTION_{block_index}", error)
                raise

            try:
                _stage1_marker(f"MLP_{block_index}_BEGIN", **_tensor_state(x))
                _stage1_marker(f"BLOCK_{block_index}_MLP_BEGIN", **_tensor_state(x))
                mlp_input = self.block1.ln2(x)
                x = x + self.block1.ffn(mlp_input)
                _stage1_marker(f"MLP_{block_index}_END", **_tensor_state(x))
                _stage1_marker(f"BLOCK_{block_index}_MLP_END", **_tensor_state(x))
            except Exception as error:
                _forward_error(f"MLP_{block_index}", error)
                raise

            _stage1_marker(f"BLOCK_{block_index}_END", **_tensor_state(x))

            try:
                _stage1_marker("LN_BEGIN", **_tensor_state(x))
                x = self.ln(x)
                _stage1_marker("LN_END", **_tensor_state(x))
            except Exception as error:
                _forward_error("LN", error)
                raise

            try:
                _stage1_marker("LM_HEAD_BEGIN", **_tensor_state(x))
                logits = self.lm_head(x)
                _stage1_marker("LM_HEAD_END", **_tensor_state(logits))
            except Exception as error:
                _forward_error("LM_HEAD", error)
                raise

            _stage1_marker("STAGE1_FORWARD_RETURN_BEGIN", **_tensor_state(logits))
            _stage1_marker("STAGE1_FORWARD_RETURN", **_tensor_state(logits))
            return logits
        except Exception as error:
            _forward_error("FORWARD", error)
            raise

    def loss_and_backward(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """Gradient interface: logits -> loss -> backward through to hidden.

        Returns the scalar loss; after the call, ``hidden.grad`` carries the
        gradient for Stage 0's activation interface (when ``hidden``
        requires grad) and all Stage 1 parameters have gradients.
        """
        logits = self(hidden)
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        return loss


def build_stage1(
    config: MinimalTransformerConfig | None = None,
    *,
    seed: int = 42,
) -> Stage1:
    """Build Stage 1 with deterministic initialization."""
    return Stage1(config=config, seed=seed)


def stage1_from_full_model(full_model: torch.nn.Module, *, seed: int = 42) -> Stage1:
    """Build Stage 1 carrying the weights of a full MinimalTransformer."""
    stage1 = build_stage1(full_model.config, seed=seed)
    stage1.block1.load_state_dict(full_model.blocks[1].state_dict())
    stage1.ln.load_state_dict(full_model.ln.state_dict())
    stage1.lm_head.load_state_dict(full_model.lm_head.state_dict())
    return stage1
