"""
Fully-connected models used in the Manifold MuOn experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Three-layer MLP used for CIFAR experiments.

    Architecture:
        input -> hidden -> hidden -> num_classes

    For CIFAR-10:
        input_dim   = 3 * 32 * 32 = 3072
        hidden_dim  = 128
        num_classes = 10
    """

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dim: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CIFAR images: (batch, 3, 32, 32) -> (batch, 3072)
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)