"""Minimal transformer + dataset tests (T068).

Logic tests verify shape, determinism, parameter count, dataset behavior,
and parameter updates on CPU.  The hardware test runs real single-GPU
training on a Worker and confirms parameters actually update.
"""

from __future__ import annotations

import pytest
import torch
from examples.models.dataset import (
    NextTokenDataset,
    NextTokenDatasetConfig,
    collate_next_token,
    example_batch,
    make_dataloader,
)
from examples.models.minimal_transformer import (
    MinimalTransformer,
    MinimalTransformerConfig,
    build_minimal_transformer,
    set_seed,
    train_step,
)


def test_model_build_and_shapes() -> None:
    config = MinimalTransformerConfig(
        vocab_size=1024, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, max_seq_length=64,
    )
    model = build_minimal_transformer(config, seed=42)
    assert isinstance(model, MinimalTransformer)
    assert len(model.blocks) == 2
    batch = example_batch(device="cpu")
    logits = model(batch[0])
    assert logits.shape == (4, 63, 1024)
    assert model.parameter_count() > 0


def test_model_fits_gtx1650_memory() -> None:
    model = build_minimal_transformer(seed=42)
    # fp32 parameters + optimizer state fit in 4 GiB easily
    assert model.parameter_count() < 5_000_000


def test_model_is_deterministic() -> None:
    batch = example_batch(seed=7)
    first = build_minimal_transformer(seed=42)(batch[0])
    second = build_minimal_transformer(seed=42)(batch[0])
    assert torch.equal(first, second)
    different = build_minimal_transformer(seed=43)(batch[0])
    assert not torch.equal(first, different)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        MinimalTransformerConfig(vocab_size=0)
    with pytest.raises(ValueError):
        MinimalTransformerConfig(hidden_size=128, num_attention_heads=3)


def test_sequence_length_enforced() -> None:
    model = build_minimal_transformer(seed=42)
    long_sequence = torch.randint(0, 1024, (1, 128), dtype=torch.long)
    with pytest.raises(ValueError):
        model(long_sequence)


def test_dataset_deterministic() -> None:
    dataset_a = NextTokenDataset(NextTokenDatasetConfig(seed=42), seed=42)
    dataset_b = NextTokenDataset(NextTokenDatasetConfig(seed=42), seed=42)
    assert torch.equal(dataset_a.tokens, dataset_b.tokens)
    assert len(dataset_a) == 256
    assert dataset_a.tokens.shape == (256, 64)


def test_collate_next_token_shapes() -> None:
    dataset = NextTokenDataset(NextTokenDatasetConfig(num_samples=8), seed=42)
    batch = [dataset[i] for i in range(4)]
    input_ids, labels = collate_next_token(batch)
    assert input_ids.shape == (4, 63)
    assert labels.shape == (4, 63)
    assert torch.equal(input_ids[:, 1:], labels[:, :-1])


def test_dataloader_repeatable() -> None:
    loader_a, _ = make_dataloader(batch_size=4, seed=42)
    loader_b, _ = make_dataloader(batch_size=4, seed=42)
    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))
    assert torch.equal(batch_a[0], batch_b[0])
    assert torch.equal(batch_a[1], batch_b[1])


def test_training_updates_parameters_on_cpu() -> None:
    set_seed(42)
    model = build_minimal_transformer(seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.clone() for parameter in model.parameters()]
    loader, _ = make_dataloader(batch_size=4, seed=42)
    for batch in loader:
        train_step(model, batch, optimizer)
        break
    assert len(model.loss_history) == 1
    changed = any(
        not torch.equal(b, parameter)
        for b, parameter in zip(before, model.parameters())
    )
    assert changed, "parameters must update after a training step"


def test_training_loss_is_finite_and_recorded() -> None:
    model = build_minimal_transformer(seed=42)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    input_ids, labels = example_batch(seed=42)
    loss = train_step(model, (input_ids, labels), optimizer)
    assert torch.isfinite(loss).item()
    assert len(model.loss_history) == 1
    assert torch.isfinite(torch.tensor(model.loss_history[0])).item()


def test_hardware_single_gpu_training_updates_parameters() -> None:
    """Real single-GPU training on a Worker (opt-in via hardware marker)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = build_minimal_transformer(seed=42).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.clone() for parameter in model.parameters()]
    loader, _ = make_dataloader(batch_size=8, seed=42)
    loss_value: float | None = None
    for input_ids, labels in loader:
        loss = train_step(model, (input_ids.to("cuda"), labels.to("cuda")), optimizer)
        loss_value = float(loss.detach().cpu().item())
        break
    assert loss_value is not None and torch.isfinite(torch.tensor(loss_value)).item()
    changed = any(
        not torch.equal(b, parameter)
        for b, parameter in zip(before, model.parameters())
    )
    assert changed, "parameters must update after a CUDA training step"
    assert len(model.loss_history) == 1