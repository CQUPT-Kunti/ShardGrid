from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class MappingDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "tokens": torch.tensor([(index + offset) % 16 for offset in range(5)]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool),
            "labels": torch.tensor(index % 3),
            "metadata": {"row_id": f"row-{index}"},
        }


class KeywordClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 6)
        self.proj = nn.Linear(6, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden = hidden * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        logits = self.proj(pooled)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = nn.functional.cross_entropy(logits, labels)
        return output


def _collate(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "input_ids": torch.stack([row["tokens"] for row in rows]),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows]),
        "labels": torch.stack([row["labels"] for row in rows]),
        "metadata": {"row_ids": [row["metadata"]["row_id"] for row in rows]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model.yaml")
    parser.add_argument("--checkpoint", type=Path, default=Path("mapping-checkpoint.pt"))
    args = parser.parse_args()

    torch.manual_seed(11)
    model = KeywordClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loader = DataLoader(MappingDataset(), batch_size=2, collate_fn=_collate)

    final_loss = 0.0
    for batch in loader:
        metadata = batch.pop("metadata")
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        assert metadata["row_ids"]

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": args.config, "model": model.state_dict()}, args.checkpoint)
    print(f"loss={final_loss:.6f}")
    print(f"checkpoint={args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
