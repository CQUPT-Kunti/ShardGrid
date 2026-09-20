from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


class MultiOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(3, 4)
        self.right = nn.Linear(3, 4)
        self.head = nn.Linear(8, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        left = torch.relu(self.left(features))
        right = torch.relu(self.right(features))
        prediction = self.head(torch.cat([left, right], dim=-1))
        return prediction, [left, right]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="synthetic")
    parser.add_argument("--checkpoint", type=Path, default=Path("lifecycle-checkpoint.pt"))
    args = parser.parse_args()

    torch.manual_seed(13)
    batches = [
        [torch.randn(2, 3), torch.randn(2, 1)],
        [torch.randn(2, 3), torch.randn(2, 1)],
    ]
    model = MultiOutputModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss_fn = nn.MSELoss()

    final_loss = 0.0
    for features, labels in batches:
        optimizer.zero_grad(set_to_none=True)
        prediction, aux = model(features)
        loss = loss_fn(prediction, labels) + 0.01 * sum(item.square().mean() for item in aux)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.detach())

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": args.dataset,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        args.checkpoint,
    )
    print(f"loss={final_loss:.6f}")
    print(f"lr={scheduler.get_last_lr()[0]:.6f}")
    print(f"checkpoint={args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
