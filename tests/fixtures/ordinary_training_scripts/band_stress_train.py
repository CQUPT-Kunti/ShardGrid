from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BandMLP(nn.Module):
    """Non-zoo MLP whose parameter footprint scales with ``width``.

    Used by T066 memory packing stress to probe progressively smaller
    memory bands (4G -> 3G -> 2G -> 1G -> 512M) on real GPUs.
    """

    def __init__(self, width: int, depth: int = 4) -> None:
        super().__init__()
        layers = []
        in_features = width
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)
        self.width = width

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--checkpoint", type=Path, default=Path("band-checkpoint.pt"))
    args = parser.parse_args()

    torch.manual_seed(49)
    batch = 4
    features = torch.randn(batch, args.width)
    labels = torch.randint(0, 2, (batch,))
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch)
    model = BandMLP(args.width, args.depth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    final_loss = 0.0
    for _step in range(args.steps):
        for x, y in loader:
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