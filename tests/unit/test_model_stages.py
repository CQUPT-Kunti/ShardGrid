"""Stage0 / Stage1 split tests (T069).

Verifies: each stage builds and executes independently, the activation
interface (hidden tensor) shape/dtype is explicit and matches between
stages, the gradient interface (hidden.grad) flows from Stage 1 back to
Stage 0's output, and the two stages together are equivalent to the full
T068 model.
"""

from __future__ import annotations

import torch
from examples.models.dataset import example_batch
from examples.models.minimal_transformer import (
    MinimalTransformerConfig,
    build_minimal_transformer,
)
from examples.models.stage0 import build_stage0, stage0_from_full_model
from examples.models.stage1 import build_stage1, stage1_from_full_model

CONFIG = MinimalTransformerConfig(
    vocab_size=1024, hidden_size=128, num_hidden_layers=2,
    num_attention_heads=4, max_seq_length=64,
)


def test_two_real_stages_exist() -> None:
    stage0 = build_stage0(CONFIG, seed=42)
    stage1 = build_stage1(CONFIG, seed=42)
    assert isinstance(stage0, torch.nn.Module)
    assert isinstance(stage1, torch.nn.Module)
    assert stage0.block0 is not None
    assert stage1.block1 is not None
    assert stage1.lm_head is not None


def test_stage0_executes_independently() -> None:
    stage0 = build_stage0(CONFIG, seed=42)
    input_ids, _ = example_batch(seed=42)
    hidden = stage0(input_ids)
    assert hidden.shape == stage0.activation_shape(4, 63)
    assert hidden.dtype == stage0.activation_dtype() == torch.float32
    assert torch.isfinite(hidden).all().item()


def test_stage1_executes_independently() -> None:
    stage1 = build_stage1(CONFIG, seed=42)
    hidden = torch.randn(4, 63, CONFIG.hidden_size)
    logits = stage1(hidden)
    assert logits.shape == (4, 63, CONFIG.vocab_size)
    assert torch.isfinite(logits).all().item()


def test_activation_interface_shape_and_dtype_explicit() -> None:
    stage0 = build_stage0(CONFIG, seed=42)
    stage1 = build_stage1(CONFIG, seed=42)
    input_ids, labels = example_batch(seed=42)
    hidden = stage0(input_ids)
    assert hidden.shape == (4, 63, 128)
    assert hidden.dtype == torch.float32
    logits = stage1(hidden)
    assert logits.shape == (4, 63, CONFIG.vocab_size)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]), labels.reshape(-1)
    )
    assert torch.isfinite(loss).item()


def test_gradient_interface_flows_from_stage1_to_stage0_activation() -> None:
    stage1 = build_stage1(CONFIG, seed=42)
    _, labels = example_batch(seed=42)
    hidden = torch.randn(4, 63, CONFIG.hidden_size, requires_grad=True)
    loss = stage1.loss_and_backward(hidden, labels)
    assert torch.isfinite(loss).item()
    assert hidden.grad is not None
    assert hidden.grad.shape == (4, 63, CONFIG.hidden_size)
    assert hidden.grad.dtype == torch.float32
    assert torch.isfinite(hidden.grad).all().item()
    # all stage1 parameters must have gradients
    for parameter in stage1.parameters():
        assert parameter.grad is not None


def test_stage0_backward_produces_parameter_gradients() -> None:
    stage0 = build_stage0(CONFIG, seed=42)
    input_ids, _ = example_batch(seed=42)
    hidden = stage0(input_ids)
    hidden.sum().backward()
    for parameter in stage0.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()


def test_stages_together_equivalent_to_full_model() -> None:
    full = build_minimal_transformer(CONFIG, seed=42)
    stage0 = stage0_from_full_model(full, seed=42)
    stage1 = stage1_from_full_model(full, seed=42)
    input_ids, _ = example_batch(seed=42)
    with torch.no_grad():
        split_logits = stage1(stage0(input_ids))
        full_logits = full(input_ids)
    assert torch.allclose(split_logits, full_logits, atol=1e-6)


def test_stage_dtype_preserved_across_interface() -> None:
    stage0 = build_stage0(CONFIG, seed=42).half()
    stage1 = build_stage1(CONFIG, seed=42).half()
    input_ids, _ = example_batch(seed=42)
    hidden = stage0(input_ids)
    assert hidden.dtype == torch.float16
    logits = stage1(hidden)
    assert logits.dtype == torch.float16
    assert logits.shape == (4, 63, CONFIG.vocab_size)


def test_stage1_rejects_wrong_activation_shape() -> None:
    stage1 = build_stage1(CONFIG, seed=42)
    bad = torch.randn(4, 63, 64)
    try:
        stage1(bad)
    except ValueError as error:
        assert "hidden_size" in str(error)
    else:
        raise AssertionError("wrong activation shape must raise ValueError")