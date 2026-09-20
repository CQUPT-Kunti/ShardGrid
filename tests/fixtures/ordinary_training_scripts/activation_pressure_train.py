from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ActivationPressureNet(nn.Module):
    """Non-zoo model whose GPU training-memory pressure comes from activations.

    Parameters stay small (O(width^2 * depth)); the memory bands are produced
    by real forward/backward activation storage:

    * activation volume scales with batch x sequence x width x depth;
    * retained skip tensors extend activation liveness across the DAG
      (multi-consumer fan-in at the end);
    * optimizer/backward intermediates come from real training steps.

    This replaces the old parameter-heavy stress fixture whose artifact size
    (and CPU serialization/SSH cost) became the bottleneck before GPU packing
    was ever exercised.
    """

    def __init__(self, width: int = 256, depth: int = 64, retention: int = 8) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.retention = retention
        self.proj = nn.Linear(256, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, 10)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.proj(features))
        skips: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            x = torch.relu(layer(x))
            if index % self.retention == 0:
                skips.append(x)
        if skips:
            x = x + torch.stack(skips).sum(dim=0)
        return self.head(x)


def activation_footprint_bytes(
    width: int, depth: int, batch: int, sequence: int, retention: int
) -> int:
    """Estimate the real activation tensor volume kept for backward.

    Each layer keeps its output for backward (batch x sequence x width x 4B),
    retained skips stay alive across the whole DAG, and the final fan-in adds
    another copy.
    """
    per_layer = batch * sequence * width * 4
    retained = (depth // retention) * batch * sequence * width * 4
    return per_layer * depth + retained + batch * sequence * width * 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--retention", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--checkpoint", type=Path, default=Path("activation-checkpoint.pt"))
    args = parser.parse_args()

    torch.manual_seed(49)
    features = torch.randn(args.batch, args.sequence, 256)
    labels = torch.randint(0, 10, (args.batch, args.sequence))
    loader = DataLoader(TensorDataset(features, labels), batch_size=args.batch)
    model = ActivationPressureNet(args.width, args.depth, args.retention)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    final_loss = 0.0
    for _step in range(args.steps):
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.permute(0, 2, 1), y)
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