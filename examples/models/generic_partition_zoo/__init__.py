"""Small graph-diverse models for generic automatic partition validation."""

from .models import (
    MiniDenseNet,
    MiniEncoderDecoder,
    MiniInception,
    MiniResNet,
    MiniUNet,
    MiniViT,
    MultiInputNet,
    MultiOutputNet,
    ResidualMLPDAG,
    build_zoo_model,
    make_zoo_sample,
)

__all__ = [
    "MiniDenseNet",
    "MiniEncoderDecoder",
    "MiniInception",
    "MiniResNet",
    "MiniUNet",
    "MiniViT",
    "MultiInputNet",
    "MultiOutputNet",
    "ResidualMLPDAG",
    "build_zoo_model",
    "make_zoo_sample",
]
