"""Residual/skip full-model fixture for automatic partition testing (T110)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PartitionStressConfig:
    input_dim: int = 16
    hidden_dim: int = 32
    bottleneck_dim: int = 24
    output_dim: int = 8


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual


class SkipFusion(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fuse = nn.Linear(width * 2, width)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        merged = torch.cat((x, skip), dim=-1)
        return self.act(self.fuse(merged))


class PartitionStressModel(nn.Module):
    def __init__(
        self,
        config: PartitionStressConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            _set_seed(seed)
        self.config = config or PartitionStressConfig()
        width = self.config.hidden_dim
        self.input_proj = nn.Linear(self.config.input_dim, width)
        self.encoder1 = ResidualBlock(width)
        self.encoder2 = ResidualBlock(width)
        self.down_proj = nn.Linear(width, self.config.bottleneck_dim)
        self.bottleneck = ResidualBlock(self.config.bottleneck_dim)
        self.up_proj = nn.Linear(self.config.bottleneck_dim, width)
        self.skip_fusion2 = SkipFusion(width)
        self.decoder1 = ResidualBlock(width)
        self.skip_fusion1 = SkipFusion(width)
        self.decoder2 = ResidualBlock(width)
        self.output_head = nn.Linear(width, self.config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc0 = self.input_proj(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        bottleneck = self.bottleneck(self.down_proj(enc2))
        up = self.up_proj(bottleneck)
        dec = self.skip_fusion2(up, enc2)
        dec = self.decoder1(dec)
        dec = self.skip_fusion1(dec, enc1)
        dec = self.decoder2(dec)
        return self.output_head(dec)


def build_partition_stress_model(
    config: PartitionStressConfig | None = None,
    *,
    seed: int = 42,
) -> PartitionStressModel:
    return PartitionStressModel(config=config, seed=seed)


def make_training_batch(
    config: PartitionStressConfig | None = None,
    *,
    batch_size: int = 4,
    seed: int = 42,
    step: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = config or PartitionStressConfig()
    generator = torch.Generator(device=device if device != "cpu" else "cpu")
    generator.manual_seed(seed + step)
    inputs = torch.randn(
        (batch_size, cfg.input_dim), generator=generator, device=device
    )
    targets = torch.randn(
        (batch_size, cfg.output_dim), generator=generator, device=device
    )
    return inputs, targets


def train_step(
    model: PartitionStressModel,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    inputs, targets = batch
    predictions = model(inputs)
    criterion = loss_fn or nn.MSELoss()
    loss = criterion(predictions, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss
