from __future__ import annotations

import torch
from torch import nn


class BufferedStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(features))


def build_batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(49)
    return torch.randn(6, 4), torch.tensor([0, 1, 0, 1, 1, 0])


def train_one_step(model: BufferedStateModel) -> None:
    features, labels = build_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()(model(features), labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
