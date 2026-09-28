"""
Evaluation utilities for classification experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute classification accuracy in percent.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0

    return 100.0 * correct / total