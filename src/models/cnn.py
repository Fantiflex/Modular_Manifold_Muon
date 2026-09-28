"""
Convolutional model used in the Manifold MuOn experiments.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-sized images.

    Architecture:
        Conv 3 -> 64
        Conv 64 -> 128
        Conv 128 -> 256
        Adaptive average pooling
        Linear 256 -> num_classes
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            256,
            num_classes,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)