from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BufferedStateTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(features))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=Path("buffered-checkpoint.pt"))
    args = parser.parse_args()

    torch.manual_seed(49)
    features = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 0, 1, 1, 0])
    loader = DataLoader(TensorDataset(features, labels), batch_size=3)
    model = BufferedStateTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    final_loss = 0.0
    for _epoch in range(args.epochs):
        for batch in loader:
            x, y = batch
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, args.checkpoint)
    print(f"loss={final_loss:.6f}")
    print(f"checkpoint={args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())