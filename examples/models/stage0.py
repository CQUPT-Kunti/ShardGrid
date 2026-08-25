"""Stage 0 of the supported minimal Transformer (T069).

Stage 0 owns the token/position embeddings and the first Transformer block.
Its output IS the activation interface to Stage 1: a hidden tensor of shape
``(batch, seq, hidden_size)`` with dtype matching the model dtype.

``forward`` is the activation interface; gradients flow back through the
standard autograd graph (Stage 1's backward produces the gradient for this
stage's output tensor, which is this stage's gradient interface).
"""

from __future__ import annotations

import torch
from examples.models.minimal_transformer import (
    AttentionBlock,
    MinimalTransformerConfig,
    set_seed,
)
from torch import nn


class Stage0(nn.Module):
    """Embedding + position + first Transformer block (activation output)."""

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
        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos = nn.Embedding(self.config.max_seq_length, self.config.hidden_size)
        self.block0 = AttentionBlock(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Activation interface: token ids -> hidden activations.

        Returns a tensor of shape ``(batch, seq_length, hidden_size)`` whose
        dtype is the model dtype (this is what Stage 1 consumes).
        """
        seq = input_ids.shape[-1]
        if seq > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {seq} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )
        positions = torch.arange(seq, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(positions)
        return self.block0(x)

    def activation_shape(self, batch_size: int, seq_length: int) -> tuple[int, ...]:
        return (batch_size, seq_length, self.config.hidden_size)

    def activation_dtype(self) -> torch.dtype:
        parameter = next(self.parameters())
        return parameter.dtype


def build_stage0(
    config: MinimalTransformerConfig | None = None,
    *,
    seed: int = 42,
) -> Stage0:
    """Build Stage 0 with deterministic initialization."""
    return Stage0(config=config, seed=seed)


def stage0_from_full_model(full_model: torch.nn.Module, *, seed: int = 42) -> Stage0:
    """Build Stage 0 carrying the weights of a full MinimalTransformer.

    Lets a previously trained full model become the initialization for the
    two-stage split without retraining.
    """
    stage0 = build_stage0(full_model.config, seed=seed)
    stage0.embed.load_state_dict(full_model.embed.state_dict())
    stage0.pos.load_state_dict(full_model.pos.state_dict())
    stage0.block0.load_state_dict(full_model.blocks[0].state_dict())
    return stage0