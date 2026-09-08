from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn


EVENT_DIR = Path(os.environ["SHARDGRID_CAPTURE_SENTINEL_DIR"])


def _mark(name: str, value: str = "1") -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    (EVENT_DIR / name).write_text(value, encoding="utf-8")


class CaptureExecutionSentinelModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _mark("forward_entered")
        return self.proj(x)


model = CaptureExecutionSentinelModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
model.proj.weight.register_hook(lambda grad: (_mark("backward_grad_computed"), grad)[1])

before = model.proj.weight.detach().clone()
optimizer.zero_grad(set_to_none=True)
loss = model(torch.ones(2, 2)).sum()
loss.backward()
optimizer.step()
after = model.proj.weight.detach()

_mark("optimizer_changed", str(not torch.equal(before, after)))
