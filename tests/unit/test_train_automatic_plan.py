from __future__ import annotations

from types import SimpleNamespace

from examples.models.train_automatic_plan import _automatic_batch_sizes


def _training_config(model_type: str):
    return SimpleNamespace(model=SimpleNamespace(type=model_type))


def test_automatic_batch_sizes_scale_with_microbatches_for_hf_style() -> None:
    total, sample = _automatic_batch_sizes(_training_config("hf_style"), microbatches=3)
    assert (total, sample) == (6, 2)


def test_automatic_batch_sizes_scale_with_microbatches_for_minimal_transformer() -> None:
    total, sample = _automatic_batch_sizes(_training_config("minimal_sequential"), microbatches=3)
    assert (total, sample) == (3, 1)
