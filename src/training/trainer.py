"""
Training utilities for classification experiments.
"""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time


from collections.abc import Mapping

from src.optimizers import (
    RiemannianLBFGS,
    GlobalizedRiemannianLBFGS,
    ManifoldMuon,
)

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



def initialize_manifold_optimizers(
    model: nn.Module,
    eta: float,
    history: int = 10,
    optimizer_name: str = "globalized_rlbfgs",
) -> dict[torch.nn.Parameter, object]:
    """
    Project matrix-like parameters onto the Stiefel manifold and
    create one globalized Riemannian L-BFGS optimizer per parameter.

    2D parameters are treated directly as matrices.
    4D convolutional kernels are flattened to matrices before
    projection, then reshaped back.

    Other parameters remain Euclidean.
    """
    optimizer_name = optimizer_name.lower()

    optimizer_classes = {
        "rlbfgs": RiemannianLBFGS,
        "globalized_rlbfgs": GlobalizedRiemannianLBFGS,
        "dual_ascent": ManifoldMuon, 
    }

    if optimizer_name not in optimizer_classes:
        raise ValueError(
            f"Unknown manifold optimizer: {optimizer_name}. "
            f"Expected one of {list(optimizer_classes)}."
        )

    optimizer_class = optimizer_classes[optimizer_name]
    manifold_optimizers = {}

    with torch.no_grad():
        for parameter in model.parameters():

            if not parameter.requires_grad:
                continue

            if parameter.ndim == 2:
                optimizer = optimizer_class(
                    eta=eta,
                    history=history,
                )

                # A zero-gradient step performs the initial
                # projection/retraction onto the manifold.
                zero_grad = torch.zeros_like(parameter.data)

                projected = optimizer.step(
                    parameter.data,
                    zero_grad,
                )

                parameter.data.copy_(projected)

                manifold_optimizers[parameter] = optimizer

            elif parameter.ndim == 4:
                shape = parameter.shape

                W = parameter.data.view(shape[0], -1)
                zero_grad = torch.zeros_like(W)

                optimizer = optimizer_class(
                    eta=eta,
                    history=history,
                )

                projected = optimizer.step(
                    W,
                    zero_grad,
                )

                parameter.data.copy_(
                    projected.view(shape)
                )

                manifold_optimizers[parameter] = optimizer

    return manifold_optimizers

def summarize_manifold_optimizer_stats(
    manifold_optimizers,
) -> dict[str, float]:
    """
    Aggregate optimizer diagnostics across all manifold parameters.
    """

    accepted = 0
    rejected = 0
    all_margins = []
    all_omegas = []

    for optimizer in manifold_optimizers.values():
        accepted += getattr(optimizer, "accepted_updates", 0)
        rejected += getattr(optimizer, "rejected_updates", 0)
        all_margins.extend(
            getattr(optimizer, "cautious_margins", [])
        )
        all_omegas.extend(
            getattr(optimizer, "omega_values", [])
        )

    total = accepted + rejected

    acceptance_rate = (
        accepted / total
        if total > 0
        else 0.0
    )

    mean_margin = (
        sum(all_margins) / len(all_margins)
        if all_margins
        else None
    )
    mean_omega = (
        sum(all_omegas) / len(all_omegas)
        if all_omegas
        else None
    )

    min_omega = min(all_omegas) if all_omegas else None
    max_omega = max(all_omegas) if all_omegas else None

    return {
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "acceptance_rate": acceptance_rate,
        "mean_cautious_margin": mean_margin,
        "mean_omega": mean_omega,
        "min_omega": min_omega,
        "max_omega": max_omega,
    }

def train_one_epoch_manifold(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    manifold_optimizers: Mapping[
        torch.nn.Parameter,
        GlobalizedRiemannianLBFGS,
    ],
    device: torch.device,
    initial_lr: float,
    start_step: int,
    total_steps: int,
) -> tuple[float, int]:
    """
    Train one epoch using globalized Riemannian L-BFGS
    on manifold parameters and gradient descent on the others.
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
        model.zero_grad()
        loss.backward()

        lr = linear_decay_lr(
            initial_lr=initial_lr,
            step=step,
            total_steps=total_steps,
        )

        manifold_parameter_step(
            model=model,
            manifold_optimizers=manifold_optimizers,
            lr=lr,
        )

        running_loss += loss.item()
        step += 1

    average_loss = running_loss / len(dataloader)

    return average_loss, step




def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    initial_lr: float,
    device: torch.device,
    mode: str = "manifold",
    weight_decay: float = 0.0,
    history: int = 10,
    manifold_optimizer: str = "globalized_rlbfgs",
) -> tuple[
    nn.Module,
    list[float],
    list[float],
    dict[str, float],
]:
    """
    Train a classification model.

    Parameters
    ----------
    model:
        Neural network to train.

    train_loader:
        Training DataLoader.

    epochs:
        Number of epochs.

    initial_lr:
        Initial learning rate / manifold step size.

    device:
        Device used for training.

    mode:
        Either "manifold" or "adamw".

    weight_decay:
        Weight decay used by AdamW.

    history:
        L-BFGS memory size.

    Returns
    -------
    model:
        Trained model.

    epoch_losses:
        Mean loss for each epoch.

    epoch_times:
        Runtime of each epoch.
    """

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_steps = epochs * len(train_loader)
    step = 0

    epoch_losses = []
    epoch_times = []

    mode = mode.lower()

    if mode == "manifold":
        manifold_optimizers = initialize_manifold_optimizers(
            model=model,
            eta=initial_lr,
            history=history,
            optimizer_name=manifold_optimizer,
        )

    elif mode == "adamw":
        manifold_optimizers = None

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown training mode: {mode}. "
            "Expected 'manifold' or 'adamw'."
        )

    for epoch in range(epochs):
        start_time = time.time()

        if mode == "manifold":
            epoch_loss, step = train_one_epoch_manifold(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                manifold_optimizers=manifold_optimizers,
                device=device,
                initial_lr=initial_lr,
                start_step=step,
                total_steps=total_steps,
            )

        else:
            epoch_loss, step = train_one_epoch_euclidean(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                initial_lr=initial_lr,
                start_step=step,
                total_steps=total_steps,
            )

        epoch_time = time.time() - start_time

        epoch_losses.append(epoch_loss)
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
    if mode == "manifold":
        optimizer_stats = summarize_manifold_optimizer_stats(
            manifold_optimizers
        )
    else:
        optimizer_stats = {}


    return model, epoch_losses, epoch_times, optimizer_stats

