"""
Model factory for experiment configuration.
"""

import torch.nn as nn

from .mlp import MLP
from .cnn import SimpleCNN
from .vit import SimpleViT


def build_model(
    model_type: str,
    num_classes: int = 100,
) -> nn.Module:
    """
    Build one of the experiment architectures.

    Parameters
    ----------
    model_type:
        One of {"mlp", "cnn", "vit"}.

    num_classes:
        Number of output classes.
    """

    model_type = model_type.lower()

    if model_type == "mlp":
        return MLP(num_classes=num_classes)

    if model_type == "cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_type == "vit":
        return SimpleViT(num_classes=num_classes)

    raise ValueError(
        f"Unknown model_type: {model_type}. "
        "Expected one of: mlp, cnn, vit."
    )