"""
Training utilities for classification experiments.
"""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def linear_decay_lr(
    initial_lr: float,
    step: int,
    total_steps: int,
) -> float:
    """
    Linear learning-rate decay used in the original experiments.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    return initial_lr * (1.0 - step / total_steps)


def train_one_epoch_euclidean(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    initial_lr: float,
    start_step: int,
    total_steps: int,
) -> tuple[float, int]:
    """
    Train one epoch using a standard PyTorch optimizer.

    Returns
    -------
    average_loss:
        Mean training loss over the epoch.

    step:
        Updated global optimization step.
    """

    model.train()

    running_loss = 0.0
    step = start_step

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Linear LR decay, matching the original notebook.
        lr = linear_decay_lr(
            initial_lr=initial_lr,
            step=step,
            total_steps=total_steps,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        running_loss += loss.item()
        step += 1

    average_loss = running_loss / len(dataloader)

    return average_loss, step