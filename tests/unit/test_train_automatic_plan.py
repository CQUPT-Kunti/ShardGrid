from __future__ import annotations

from types import SimpleNamespace

import torch

from examples.models.train_automatic_plan import (
    _automatic_batch_sizes,
    _build_large_stage_module,
)
from shardgrid.common.config import TrainingConfig
from shardgrid.common.models import as_engine_name
from shardgrid.engines.models import ParallelPlan, ParallelPlanStage


def _training_config(model_type: str):
    return SimpleNamespace(model=SimpleNamespace(type=model_type))


def _large_training_config() -> TrainingConfig:
    return TrainingConfig.from_dict(
        {
            "job": {
                "name": "large-auto",
                "backend": "ssh",
                "communication_backend": "nccl",
            },
            "model": {
                "name": "large-auto",
                "type": "large_residual_transformer",
                "stage_count": 2,
                "parameters": {
                    "vocab_size": 128,
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "ffn_size": 64,
                    "sequence_length": 8,
                    "batch_size": 2,
                    "memory_bank_rows": 16,
                    "memory_bank_touch_rows": 2,
                },
            },
            "resources": {"world_size": 2, "preferred_workers": ["gpu-a", "gpu-b"]},
            "planning": {"mode": "automatic"},
        }
    )


def _large_parallel_plan() -> ParallelPlan:
    return ParallelPlan(
        parallel_plan_id="large-auto-plan",
        engine=as_engine_name("galvatron"),
        model_name="large-auto",
        world_size=2,
        stages=["stage0", "stage1"],
        partition_source="automatic",
        stage_metadata=[
            ParallelPlanStage(
                stage_id="stage0",
                rank=0,
                module_ids=tuple(f"m{index}" for index in range(10)),
                module_paths=(
                    "token_embedding",
                    "position_embedding",
                    "blocks.0.norm1",
                    "blocks.0.attn.qkv",
                    "blocks.0.attn.out_proj",
                    "blocks.0.norm2",
                    "blocks.0.ffn.0",
                    "blocks.0.ffn.1",
                    "blocks.0.ffn.2",
                    "blocks.0.memory_pressure",
                ),
                start_index=0,
                stop_index=10,
            ),
            ParallelPlanStage(
                stage_id="stage1",
                rank=1,
                module_ids=tuple(f"m{index}" for index in range(10, 20)),
                module_paths=(
                    "blocks.1.norm1",
                    "blocks.1.attn.qkv",
                    "blocks.1.attn.out_proj",
                    "blocks.1.norm2",
                    "blocks.1.ffn.0",
                    "blocks.1.ffn.1",
                    "blocks.1.ffn.2",
                    "blocks.1.memory_pressure",
                    "norm",
                    "output_head",
                ),
                start_index=10,
                stop_index=20,
            ),
        ],
    )


def test_automatic_batch_sizes_scale_with_microbatches_for_hf_style() -> None:
    total, sample = _automatic_batch_sizes(_training_config("hf_style"), microbatches=3)
    assert (total, sample) == (6, 2)


def test_automatic_batch_sizes_scale_with_microbatches_for_minimal_transformer() -> None:
    total, sample = _automatic_batch_sizes(
        _training_config("minimal_sequential"),
        microbatches=3,
    )
    assert (total, sample) == (3, 1)


def test_large_runtime_stage_build_does_not_call_full_model_builder(
    monkeypatch,
) -> None:
    def fail_full_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("full model builder must not run on the large automatic worker path")

    monkeypatch.setattr(
        "examples.models.train_automatic_plan.build_large_residual_transformer",
        fail_full_builder,
    )

    module, sample_inputs, sample_outputs, evidence = _build_large_stage_module(
        _large_training_config(),
        _large_parallel_plan(),
        stage_index=0,
        device=torch.device("cpu"),
        sample_batch_size=1,
    )

    assert module.__class__.__name__ == "LargeResidualTransformerStage"
    assert len(sample_inputs) == 1
    assert isinstance(sample_outputs, tuple)
    assert evidence["full_model_materialized"] is False
    assert evidence["owned_module_paths"][0] == "token_embedding"
    assert evidence["process_rss_before_materialization"] == evidence[
        "process_rss_before_materialization_bytes"
    ]
    assert evidence["process_rss_after_materialization"] == evidence[
        "process_rss_after_materialization_bytes"
    ]
    assert evidence["cuda_before_stage_move_bytes"] == evidence[
        "cuda_allocated_before_stage_to_device_bytes"
    ]
    assert evidence["cuda_after_stage_move_bytes"] == evidence[
        "cuda_allocated_after_stage_to_device_bytes"
    ]


def test_large_runtime_stage_build_prunes_unused_boundary_state() -> None:
    module, sample_inputs, sample_outputs, _evidence = _build_large_stage_module(
        _large_training_config(),
        _large_parallel_plan(),
        stage_index=1,
        device=torch.device("cpu"),
        sample_batch_size=1,
    )

    assert module.__class__.__name__ == "LargeResidualTransformerStage"
    assert len(sample_inputs) == 2
    assert isinstance(sample_outputs, torch.Tensor)
