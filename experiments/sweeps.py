"""
Hyperparameter sweep utilities.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import build_model
from src.training import (
    evaluate_accuracy,
    train_model,
)


DEFAULT_SEEDS = [0, 15, 42, 65]


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_eta_seed_sweep(
    model_type: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    etas,
    device: torch.device,
    epochs: int = 5,
    seeds=None,
    mode: str = "manifold",
    num_classes: int = 100,
    weight_decay: float = 0.05,
    history: int = 10,
) -> dict[float, list[float]]:
    """
    Run training for several eta values and random seeds.

    Returns
    -------
    dict
        Mapping:

            eta -> [test_accuracy_seed_1, ...]
    """

    if seeds is None:
        seeds = DEFAULT_SEEDS

    results = {}

    for eta in etas:
        accuracies = []

        print(f"\n========== ETA = {eta} ==========")

        for seed in seeds:
            print(f"-- Seed {seed} --")

            set_seed(seed)

            model = build_model(
                model_type=model_type,
                num_classes=num_classes,
            )

            model, _, _ = train_model(
                model=model,
                train_loader=train_loader,
                epochs=epochs,
                initial_lr=eta,
                device=device,
                mode=mode,
                weight_decay=weight_decay,
                history=history,
            )

            test_accuracy = evaluate_accuracy(
                model=model,
                dataloader=test_loader,
                device=device,
            )

            print(
                f"Test accuracy: "
                f"{test_accuracy:.2f}%"
            )

            accuracies.append(test_accuracy)

        results[eta] = accuracies

    return results