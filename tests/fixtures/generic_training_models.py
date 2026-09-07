from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GenericTrainingCase:
    name: str
    module: nn.Module
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] | None = None


class SequentialMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.block = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(x)
        return self.head(hidden + self.block(hidden))


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(6, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.features(image).flatten(1)
        return self.head(hidden)


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Embedding(32, 8)
        self.position = nn.Parameter(torch.zeros(1, 6, 8))
        self.attn = nn.MultiheadAttention(8, num_heads=2, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(8)
        self.ffn = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        self.norm2 = nn.LayerNorm(8)
        self.head = nn.Linear(8, 5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token(input_ids) + self.position[:, : input_ids.shape[1], :]
        attended, _weights = self.attn(hidden, hidden, hidden, need_weights=False)
        hidden = self.norm1(hidden + attended)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return self.head(hidden[:, 0])


class KeywordAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(8, num_heads=2, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 2)

    def forward(self, *, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden, _weights = self.attn(
            tokens,
            tokens,
            tokens,
            key_padding_mask=mask,
            need_weights=False,
        )
        return self.head(self.norm(tokens + hidden).mean(dim=1))


class TinyUNetLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(12, 4, kernel_size=3, padding=1)
        self.out = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skip = F.relu(self.enc1(image))
        encoded = F.relu(self.enc2(F.avg_pool2d(skip, kernel_size=2)))
        up = F.interpolate(encoded, size=skip.shape[-2:], mode="nearest")
        decoded = F.relu(self.dec1(torch.cat([up, skip], dim=1)))
        return self.out(decoded)


class DenseConnectionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(6, 8)
        self.layer1 = nn.Linear(8, 8)
        self.layer2 = nn.Linear(16, 8)
        self.head = nn.Linear(24, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = F.relu(self.input(x))
        h1 = F.relu(self.layer1(h0))
        h2 = F.relu(self.layer2(torch.cat([h0, h1], dim=-1)))
        return self.head(torch.cat([h0, h1, h2], dim=-1))


class MultiBranchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(5, 7)
        self.right = nn.Linear(5, 7)
        self.gate = nn.Linear(5, 7)
        self.head = nn.Linear(14, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        left = F.relu(self.left(features))
        right = torch.sigmoid(self.gate(features)) * F.relu(self.right(features))
        return self.head(torch.cat([left, right], dim=-1))


class SharedModuleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(8, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.relu(self.shared(x))
        second = F.relu(self.shared(first))
        return self.head(first + second)


class TiedParameterNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 6)
        self.decoder = nn.Linear(6, 16, bias=False)
        self.decoder.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids).mean(dim=1)
        return self.decoder(hidden)


def _cases() -> dict[str, Callable[[], GenericTrainingCase]]:
    return {
        "sequential": lambda: GenericTrainingCase(
            "sequential",
            SequentialMLP(),
            (torch.randn(3, 8),),
        ),
        "residual": lambda: GenericTrainingCase(
            "residual",
            ResidualMLP(),
            (torch.randn(3, 8),),
        ),
        "cnn": lambda: GenericTrainingCase(
            "cnn",
            TinyCNN(),
            (torch.randn(2, 3, 8, 8),),
        ),
        "transformer": lambda: GenericTrainingCase(
            "transformer",
            TinyTransformer(),
            (torch.randint(0, 32, (2, 6), dtype=torch.long),),
        ),
        "attention": lambda: GenericTrainingCase(
            "attention",
            KeywordAttention(),
            kwargs={
                "tokens": torch.randn(2, 4, 8),
                "mask": torch.tensor([[False, False, False, True], [False, False, True, True]]),
            },
        ),
        "unet_like": lambda: GenericTrainingCase(
            "unet_like",
            TinyUNetLike(),
            (torch.randn(2, 1, 8, 8),),
        ),
        "dense": lambda: GenericTrainingCase(
            "dense",
            DenseConnectionMLP(),
            (torch.randn(3, 6),),
        ),
        "multi_branch": lambda: GenericTrainingCase(
            "multi_branch",
            MultiBranchNet(),
            (torch.randn(3, 5),),
        ),
        "shared_module": lambda: GenericTrainingCase(
            "shared_module",
            SharedModuleNet(),
            (torch.randn(3, 8),),
        ),
        "shared_tied_parameter": lambda: GenericTrainingCase(
            "shared_tied_parameter",
            TiedParameterNet(),
            (torch.randint(0, 16, (2, 5), dtype=torch.long),),
        ),
    }


def generic_training_case_names() -> tuple[str, ...]:
    return tuple(_cases())


def generic_training_case(name: str) -> GenericTrainingCase:
    return _cases()[name]()


def iter_generic_training_cases() -> Iterator[GenericTrainingCase]:
    for name in generic_training_case_names():
        yield generic_training_case(name)


def output_to_loss(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.float().pow(2).mean()
    if isinstance(output, Mapping):
        losses = [output_to_loss(value) for value in output.values()]
    elif isinstance(output, (list, tuple)):
        losses = [output_to_loss(value) for value in output]
    else:
        raise TypeError(f"unsupported output type {type(output)!r}")
    if not losses:
        raise ValueError("cannot build loss from empty output")
    return torch.stack(losses).sum()


def run_cpu_forward_backward(case: GenericTrainingCase) -> torch.Tensor:
    torch.manual_seed(42)
    case.module.train()
    case.module.zero_grad(set_to_none=True)
    kwargs = {} if case.kwargs is None else dict(case.kwargs)
    output = case.module(*case.args, **kwargs)
    loss = output_to_loss(output)
    loss.backward()
    trainable = [parameter for parameter in case.module.parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError(f"{case.name} has no trainable parameters")
    if not any(parameter.grad is not None for parameter in trainable):
        raise AssertionError(f"{case.name} produced no parameter gradients")
    return loss.detach()
