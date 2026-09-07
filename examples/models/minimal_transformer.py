"""Deterministic minimal Transformer for ShardGrid MVP training (T068).

This is the explicitly supported MVP model: repeatable (fixed-seed
construction), very small (fits the GTX 1650 4 GiB comfortably), with a
recorded loss history.  It is a plain PyTorch model with no Galvatron
dependency so it can run through any selected ParallelEngine adapter.

Usage:

    from examples.models.minimal_transformer import (
        build_minimal_transformer, set_seed, train_step,
    )

    model = build_minimal_transformer(seed=42)
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = ...  # (input_ids [B, S], labels [B, S])
    loss = train_step(model, batch, optimizer)   # appends to model.loss_history
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class MinimalTransformerConfig:
    vocab_size: int = 1024
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    max_seq_length: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")


def set_seed(seed: int) -> None:
    """Deterministic seeding for repeatable model construction and data."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AttentionBlock(nn.Module):
    def __init__(self, config: MinimalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.ln1(x)
        qkv = self.qkv(hidden)
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
        x = x + self.proj(out)
        x = x + self.ffn(self.ln2(x))
        return x


class MinimalTransformer(nn.Module):
    """Very small causal Transformer with an exposed loss history."""

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
        self.blocks = nn.ModuleList(
            [AttentionBlock(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.ln = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.loss_history: list[float] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq = input_ids.shape[-1]
        if seq > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {seq} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )
        positions = torch.arange(seq, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln(x))

    def record_loss(self, loss: torch.Tensor) -> None:
        self.loss_history.append(float(loss.detach().cpu().item()))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_minimal_transformer(
    config: MinimalTransformerConfig | None = None,
    *,
    seed: int = 42,
) -> MinimalTransformer:
    """Build the supported model with deterministic initialization."""
    return MinimalTransformer(config=config, seed=seed)


def train_step(
    model: MinimalTransformer,
    batch: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """One forward/backward/optimizer step; appends the loss to history."""
    input_ids, labels = batch
    logits = model(input_ids)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.record_loss(loss)
    return loss


def train_iterations(
    model: MinimalTransformer,
    batches: Iterable[Sequence[torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int = 1,
) -> list[float]:
    """Run ``steps`` training steps and return the recorded loss history."""
    iterator = iter(batches)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        train_step(model, batch, optimizer)
    return list(model.loss_history)