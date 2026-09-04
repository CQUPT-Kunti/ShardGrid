from __future__ import annotations

from types import SimpleNamespace

import torch

from examples.models.large_residual_transformer import (
    LargeResidualTransformer,
    LargeResidualTransformerConfig,
    build_large_residual_transformer,
    build_large_residual_transformer_stage,
    large_residual_module_paths,
    make_large_residual_batch,
    required_boundary_state_names,
    train_step,
)


def _small_config() -> LargeResidualTransformerConfig:
    return LargeResidualTransformerConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=4,
        num_heads=4,
        ffn_size=64,
        sequence_length=8,
        batch_size=2,
        memory_bank_rows=16,
        memory_bank_touch_rows=2,
    )


def _three_layer_config() -> LargeResidualTransformerConfig:
    return LargeResidualTransformerConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=3,
        num_heads=4,
        ffn_size=64,
        sequence_length=8,
        batch_size=2,
        memory_bank_rows=16,
        memory_bank_touch_rows=2,
    )


def _stage_meta(stage_id: str, module_paths: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(stage_id=stage_id, module_paths=module_paths)


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def test_large_residual_transformer_is_deterministic() -> None:
    config = _small_config()
    batch = make_large_residual_batch(config, seed=7)
    first = build_large_residual_transformer(config, seed=42)(batch[0])
    second = build_large_residual_transformer(config, seed=42)(batch[0])
    assert isinstance(build_large_residual_transformer(config), LargeResidualTransformer)
    assert torch.equal(first, second)


def test_large_residual_dataset_is_deterministic_and_external_to_model() -> None:
    config = _small_config()
    first = make_large_residual_batch(config, seed=11, step=3)
    second = make_large_residual_batch(config, seed=11, step=3)
    different = make_large_residual_batch(config, seed=12, step=3)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[0], different[0])


def test_large_residual_transformer_trains_one_cpu_step() -> None:
    config = _small_config()
    model = build_large_residual_transformer(config, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_step(model, make_large_residual_batch(config, seed=42), optimizer)
    assert torch.isfinite(loss).item()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_stage_local_builder_only_materializes_owned_parameters() -> None:
    config = _small_config()
    paths = large_residual_module_paths(config)
    stage0_paths = paths[:18]
    stage1_paths = paths[18:]

    stage0 = build_large_residual_transformer_stage(
        config,
        stage_metadata=_stage_meta("stage0", stage0_paths),
        next_stage_start_path=stage1_paths[0],
        seed=42,
    )
    stage1 = build_large_residual_transformer_stage(
        config,
        stage_metadata=_stage_meta("stage1", stage1_paths),
        next_stage_start_path=None,
        seed=42,
    )

    stage0_names = set(stage0.state_dict().keys())
    stage1_names = set(stage1.state_dict().keys())

    assert "blocks.2.memory_pressure.weight" not in stage0_names
    assert "blocks.3.memory_pressure.weight" not in stage0_names
    assert "blocks.0.memory_pressure.weight" not in stage1_names
    assert "blocks.1.memory_pressure.weight" not in stage1_names
    assert "token_embedding.weight" in stage0_names
    assert "output_head.weight" in stage1_names


def test_stage_local_parameter_bytes_are_smaller_than_full_model() -> None:
    config = _small_config()
    paths = large_residual_module_paths(config)
    split = 18
    full = build_large_residual_transformer(config, seed=42)
    stage0 = build_large_residual_transformer_stage(
        config,
        stage_metadata=_stage_meta("stage0", paths[:split]),
        next_stage_start_path=paths[split],
        seed=42,
    )
    stage1 = build_large_residual_transformer_stage(
        config,
        stage_metadata=_stage_meta("stage1", paths[split:]),
        next_stage_start_path=None,
        seed=42,
    )

    full_bytes = _parameter_bytes(full)
    stage0_bytes = _parameter_bytes(stage0)
    stage1_bytes = _parameter_bytes(stage1)

    assert 0 < stage0_bytes < full_bytes
    assert 0 < stage1_bytes < full_bytes
    assert stage0_bytes + stage1_bytes == full_bytes


def test_stage_local_chain_preserves_residual_and_long_skip_semantics() -> None:
    config = _small_config()
    full = build_large_residual_transformer(config, seed=42)
    paths = large_residual_module_paths(config)
    split = 10
    stage0 = build_large_residual_transformer_stage(
        config,
        stage_metadata=_stage_meta("stage0", paths[:split]),
        next_stage_start_path=paths[split],
        seed=42,
    )
    stage1 = build_large_residual_transformer_stage(
        config,
        stage_metadata=_stage_meta("stage1", paths[split:]),
        next_stage_start_path=None,
        seed=42,
    )

    full_state = full.state_dict()
    stage0.load_state_dict(
        {name: tensor for name, tensor in full_state.items() if name in stage0.state_dict()},
        strict=True,
    )
    stage1.load_state_dict(
        {name: tensor for name, tensor in full_state.items() if name in stage1.state_dict()},
        strict=True,
    )

    inputs, _targets = make_large_residual_batch(config, seed=9, step=1)
    with torch.no_grad():
        expected = full(inputs)
        boundary = stage0(inputs)
        assert isinstance(boundary, tuple)
        assert len(boundary) == 2
        actual = stage1(*boundary)

    assert boundary[0].shape == (config.batch_size, config.sequence_length, config.hidden_size)
    assert boundary[1].shape == boundary[0].shape
    assert actual.shape == (config.batch_size, config.sequence_length, config.vocab_size)
    assert torch.allclose(actual, expected)


def test_large_full_builder_can_materialize_meta_parameters() -> None:
    config = _small_config()
    model = build_large_residual_transformer(config, seed=42, device="meta")
    assert {str(parameter.device) for parameter in model.parameters()} == {"meta"}


def test_required_boundary_state_names_drop_unused_long_skip_for_attn_split() -> None:
    config = _three_layer_config()
    paths = large_residual_module_paths(config)
    stage1_paths = paths[19:]

    assert stage1_paths[0] == "blocks.2.attn.qkv"
    assert required_boundary_state_names(stage1_paths, ("x",)) == ("x", "norm1_out")


def test_stage_boundary_backward_keeps_gradients_for_split_19() -> None:
    config = _three_layer_config()
    full = build_large_residual_transformer(config, seed=42)
    paths = large_residual_module_paths(config)
    split = 19
    stage1_inputs = required_boundary_state_names(paths[split:], ("x",))
    stage0 = build_large_residual_transformer_stage(
        config,
        stage_metadata=_stage_meta("stage0", paths[:split]),
        next_stage_start_path=paths[split],
        input_state_names=("input_ids",),
        output_state_names=stage1_inputs,
        seed=42,
    )
    stage1 = build_large_residual_transformer_stage(
        config,
        stage_metadata=_stage_meta("stage1", paths[split:]),
        next_stage_start_path=None,
        input_state_names=stage1_inputs,
        output_state_names=("x",),
        seed=42,
    )

    full_state = full.state_dict()
    stage0.load_state_dict(
        {name: tensor for name, tensor in full_state.items() if name in stage0.state_dict()},
        strict=True,
    )
    stage1.load_state_dict(
        {name: tensor for name, tensor in full_state.items() if name in stage1.state_dict()},
        strict=True,
    )

    inputs, targets = make_large_residual_batch(config, seed=9, step=1)
    boundary = stage0(inputs)
    assert isinstance(boundary, tuple)
    assert len(boundary) == 2
    for tensor in boundary:
        tensor.retain_grad()
    logits = stage1(*boundary)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    loss.backward()

    assert all(tensor.grad is not None for tensor in boundary)
