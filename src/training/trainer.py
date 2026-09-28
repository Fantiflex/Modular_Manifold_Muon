"""
Training utilities for classification experiments.
"""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collections.abc import Mapping

from src.optimizers import GlobalizedRiemannianLBFGS


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



def manifold_parameter_step(
    model: nn.Module,
    manifold_optimizers: Mapping[
        torch.nn.Parameter,
        GlobalizedRiemannianLBFGS,
    ],
    lr: float,
) -> None:
    """
    Update model parameters after backward().

    Parameters registered in ``manifold_optimizers`` are updated with
    globalized Riemannian L-BFGS.

    Other parameters are updated with a standard Euclidean gradient step.

    4D convolutional tensors are flattened to matrices before the
    manifold update and reshaped afterwards.
    """

    with torch.no_grad():
        for parameter in model.parameters():

            if parameter.grad is None:
                continue

            # ---------------------------------------------------------
            # Manifold-constrained parameter
            # ---------------------------------------------------------
            if parameter in manifold_optimizers:
                optimizer = manifold_optimizers[parameter]

                # Keep the current experiment learning rate.
                optimizer.eta = lr

                if parameter.ndim == 2:
                    W = parameter.data
                    G = parameter.grad

                    # update() completes the curvature pair generated
                    # by the previous step.
                    optimizer.update(G)

                    W_new = optimizer.step(W, G)

                    parameter.data.copy_(W_new)

                elif parameter.ndim == 4:
                    shape = parameter.shape

                    W = parameter.data.view(shape[0], -1)
                    G = parameter.grad.view(shape[0], -1)

                    optimizer.update(G)

                    W_new = optimizer.step(W, G)

                    parameter.data.copy_(W_new.view(shape))

                else:
                    raise ValueError(
                        "Manifold parameters must be 2D or 4D, "
                        f"got shape {tuple(parameter.shape)}."
                    )

            # ---------------------------------------------------------
            # Euclidean parameter
            # ---------------------------------------------------------
            else:
                parameter.data.add_(
                    parameter.grad,
                    alpha=-lr,
                )



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