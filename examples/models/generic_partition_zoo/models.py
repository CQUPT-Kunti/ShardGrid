"""Graph-diverse toy models; factories only, no partition/stage builders."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv2(self.conv1(x)) + self.proj(x))


class MiniResNet(nn.Module):
    def __init__(self, base_channels: int = 8, blocks: int = 3) -> None:
        super().__init__()
        self.stem = ConvBlock(3, base_channels)
        layers: list[nn.Module] = []
        channels = base_channels
        for index in range(blocks):
            out_channels = channels * 2 if index == 1 else channels
            layers.append(ResidualBlock(channels, out_channels))
            channels = out_channels
        self.blocks = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.pool(x).flatten(1))


class MiniUNet(nn.Module):
    def __init__(self, base_channels: int = 8) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels)
        self.down = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.down(skip1))
        x = self.bottleneck(self.down(skip2))
        x = self.dec2(torch.cat([self.up2(x), skip2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))
        return self.out(x)


class MiniDenseNet(nn.Module):
    def __init__(self, width: int = 16, growth_rate: int = 8, layers: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(width, width)
        self.layers = nn.ModuleList(
            [nn.Linear(width + index * growth_rate, growth_rate) for index in range(layers)]
        )
        self.head = nn.Linear(width + layers * growth_rate, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [torch.relu(self.input(x))]
        for layer in self.layers:
            features.append(torch.relu(layer(torch.cat(features, dim=-1))))
        return self.head(torch.cat(features, dim=-1))


class MiniInception(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.stem = ConvBlock(3, channels)
        self.branch1 = nn.Conv2d(channels, channels, 1)
        self.branch3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels, channels, 1),
        )
        self.head = nn.Conv2d(channels * 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        branches = [self.branch1(x), self.branch3(x), self.branch5(x), self.pool(x)]
        return self.head(torch.cat(branches, dim=1))


class TinyAttentionBlock(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        attn = torch.softmax(self.q(y) @ self.k(y).transpose(-1, -2) / (y.shape[-1] ** 0.5), dim=-1)
        x = x + self.proj(attn @ self.v(y))
        return x + self.mlp(self.ln2(x))


class MiniViT(nn.Module):
    def __init__(self, hidden: int = 64, blocks: int = 2) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, hidden, 4, stride=4)
        self.blocks = nn.ModuleList([TinyAttentionBlock(hidden) for _ in range(blocks)])
        self.head = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))


class MiniEncoderDecoder(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.src_embed = nn.Embedding(vocab, hidden)
        self.encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.tgt_embed = nn.Embedding(vocab, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(self.src_embed(src)).mean(dim=1)
        target = self.tgt_embed(tgt)
        repeated = memory.unsqueeze(1).expand(-1, target.shape[1], -1)
        return self.head(self.decoder(torch.cat([target, repeated], dim=-1)))


class MultiInputNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.meta = nn.Linear(5, 8)
        self.head = nn.Linear(8 * 32 * 32 + 8, 3)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.image(image), self.meta(metadata)], dim=-1))


class MultiOutputNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(16, 16)
        self.classifier = nn.Linear(16, 4)
        self.regressor = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.relu(self.backbone(x))
        return self.classifier(features), self.regressor(features)


class ResidualMLPDAG(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.a = nn.Linear(width, width)
        self.b = nn.Linear(width, width)
        self.c = nn.Linear(width, width)
        self.head = nn.Linear(width * 2, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.a(x))
        b = torch.relu(self.b(a))
        c = torch.relu(self.c(a + b))
        return self.head(torch.cat([b, c], dim=-1))


def build_zoo_model(name: str, **kwargs: Any) -> nn.Module:
    models = {
        "mini_resnet": MiniResNet,
        "mini_unet": MiniUNet,
        "mini_densenet": MiniDenseNet,
        "mini_inception": MiniInception,
        "mini_vit": MiniViT,
        "mini_encoder_decoder": MiniEncoderDecoder,
        "multi_input": MultiInputNet,
        "multi_output": MultiOutputNet,
        "residual_mlp_dag": ResidualMLPDAG,
    }
    try:
        return models[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown generic partition zoo model {name!r}") from exc


def make_zoo_sample(name: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if name in {"mini_resnet", "mini_unet", "mini_inception", "mini_vit"}:
        return (torch.randn(1, 3, 32, 32),), {}
    if name == "mini_encoder_decoder":
        return (torch.randint(0, 32, (1, 8)), torch.randint(0, 32, (1, 6))), {}
    if name == "multi_input":
        return (torch.randn(1, 3, 32, 32), torch.randn(1, 5)), {}
    return (torch.randn(1, 16),), {}
