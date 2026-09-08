from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn


class LargeDeclaredStateModel(nn.Module):
    def __init__(self, declared_state_bytes: int) -> None:
        super().__init__()
        if declared_state_bytes <= 0:
            raise ValueError("declared_state_bytes must be positive")
        elements = max(1, declared_state_bytes // torch.empty((), dtype=torch.float32).element_size())
        self.weight = nn.Parameter(torch.empty(elements, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight.reshape(-1)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("large-declared-state.pt"))
    args = parser.parse_args()

    declared_state_bytes = int(os.environ.get("SHARDGRID_DECLARED_STATE_BYTES", "32"))
    model = LargeDeclaredStateModel(declared_state_bytes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.ones(2, 2)

    optimizer.zero_grad(set_to_none=True)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
