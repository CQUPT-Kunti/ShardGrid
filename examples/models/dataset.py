"""Deterministic synthetic next-token dataset for the MVP model (T068).

A fixed-seed synthetic token sequence dataset (no downloads, no external
files) that works with :class:`examples.models.minimal_transformer
.MinimalTransformer` for both local single-GPU verification and later
two-host training.  Deterministic across runs for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from examples.models.minimal_transformer import set_seed
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class NextTokenDatasetConfig:
    vocab_size: int = 1024
    seq_length: int = 64
    num_samples: int = 256
    seed: int = 42


class NextTokenDataset(Dataset[torch.Tensor]):
    """Synthetic token-sequence dataset with deterministic content."""

    def __init__(
        self,
        config: NextTokenDatasetConfig | None = None,
        *,
        seed: int = 42,
    ) -> None:
        config = config or NextTokenDatasetConfig(seed=seed)
        set_seed(seed)
        self.config = config
        tokens = torch.randint(
            0,
            config.vocab_size,
            (config.num_samples, config.seq_length),
            dtype=torch.long,
        )
        self.tokens = tokens

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tokens[index]


def collate_next_token(
    batch: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate token sequences into (input_ids, labels) with a left shift.

    The last token is predicted from the first ``seq - 1`` tokens; the first
    label position is a sentinel (ignored in training).
    """
    sequences = torch.stack(batch, dim=0)
    input_ids = sequences[:, :-1].contiguous()
    labels = sequences[:, 1:].contiguous()
    return input_ids, labels


def make_dataloader(
    config: NextTokenDatasetConfig | None = None,
    *,
    batch_size: int = 4,
    seed: int = 42,
) -> tuple[DataLoader[torch.Tensor], NextTokenDataset]:
    """Create the deterministic dataloader and its backing dataset."""
    dataset = NextTokenDataset(config, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_next_token,
        drop_last=True,
    )
    return loader, dataset


def iterate_batches(
    loader: DataLoader[torch.Tensor],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (input_ids, labels) batches from the loader."""
    for batch in loader:
        yield batch


def example_batch(
    *,
    batch_size: int = 4,
    vocab_size: int = 1024,
    seq_length: int = 64,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """A single deterministic (input_ids, labels) batch for quick checks."""
    set_seed(seed)
    sequences = torch.randint(
        0, vocab_size, (batch_size, seq_length), dtype=torch.long, device=device
    )
    return sequences[:, :-1].contiguous(), sequences[:, 1:].contiguous()