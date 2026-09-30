"""
Hyperparameter sweep utilities.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import build_model
from src.training import (
    evaluate_accuracy,
    train_model,
)


DEFAULT_SEEDS = [0, 15, 42]


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
    manifold_optimizer: str = "globalized_rlbfgs",
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

            model, _, _, _ = train_model(
                model=model,
                train_loader=train_loader,
                epochs=epochs,
                initial_lr=eta,
                device=device,
                mode=mode,
                manifold_optimizer=manifold_optimizer,
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


def run_optimizer_sweep(
    *,
    model_type: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizers: list[str],
    etas: list[float],
    seeds: list[int],
    device: torch.device,
    epochs: int,
    num_classes: int,
    batch_size: int,
    history: int = 10,
    weight_decay: float = 0.05,
    checkpoint_path: str | Path | None = None,
) -> list[dict]:
    """
    Run a reproducible optimizer x eta x seed sweep.

    Each returned row contains training performance, runtime,
    and manifold-optimizer diagnostics.
    """

    results = []

    total_runs = (
        len(optimizers)
        * len(etas)
        * len(seeds)
    )

    run_index = 0

    for optimizer_name in optimizers:
        for eta in etas:
            for seed in seeds:
                run_index += 1

                print(
                    "\n"
                    f"========== RUN {run_index}/{total_runs} =========="
                )
                print(f"Optimizer: {optimizer_name}")
                print(f"Eta: {eta}")
                print(f"Seed: {seed}")

                set_seed(seed)

                model = build_model(
                    model_type=model_type,
                    num_classes=num_classes,
                )

                if optimizer_name == "adamw":
                    mode = "adamw"
                    manifold_optimizer = "globalized_rlbfgs"
                else:
                    mode = "manifold"
                    manifold_optimizer = optimizer_name

                try:
                    (
                        model,
                        epoch_losses,
                        epoch_times,
                        optimizer_stats,
                    ) = train_model(
                        model=model,
                        train_loader=train_loader,
                        epochs=epochs,
                        initial_lr=eta,
                        device=device,
                        mode=mode,
                        manifold_optimizer=manifold_optimizer,
                        weight_decay=weight_decay,
                        history=history,
                    )

                    train_accuracy = evaluate_accuracy(
                        model=model,
                        dataloader=train_loader,
                        device=device,
                    )

                    test_accuracy = evaluate_accuracy(
                        model=model,
                        dataloader=test_loader,
                        device=device,
                    )

                    row = {
                        "model": model_type,
                        "optimizer": optimizer_name,
                        "lr": eta,
                        "seed": seed,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "history": history,
                        "status": "ok",
                        "error": "",
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                        "final_loss": epoch_losses[-1],
                        "total_time_s": sum(epoch_times),
                        "accepted_updates": optimizer_stats.get(
                            "accepted_updates",
                            0,
                        ),
                        "rejected_updates": optimizer_stats.get(
                            "rejected_updates",
                            0,
                        ),
                        "acceptance_rate": optimizer_stats.get(
                            "acceptance_rate",
                            0.0,
                        ),
                        "mean_cautious_margin": optimizer_stats.get(
                            "mean_cautious_margin"
                        ),
                        "mean_omega": optimizer_stats.get(
                            "mean_omega"
                        ),
                        "min_omega": optimizer_stats.get(
                            "min_omega"
                        ),
                        "max_omega": optimizer_stats.get(
                            "max_omega"
                        ),
                    }

                    print(
                        f"Train accuracy: "
                        f"{train_accuracy:.2f}%"
                    )
                    print(
                        f"Test accuracy: "
                        f"{test_accuracy:.2f}%"
                    )
                    print(
                        f"Final loss: "
                        f"{epoch_losses[-1]:.4f}"
                    )
                    print(
                        f"Total time: "
                        f"{sum(epoch_times):.2f}s"
                    )

                except Exception as exc:
                    print(
                        f"\nFAILED: optimizer={optimizer_name}, "
                        f"eta={eta}, seed={seed}"
                    )
                    print(
                        f"{type(exc).__name__}: {exc}"
                    )

                    row = {
                        "model": model_type,
                        "optimizer": optimizer_name,
                        "lr": eta,
                        "seed": seed,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "history": history,
                        "status": "failed",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "train_accuracy": None,
                        "test_accuracy": None,
                        "final_loss": None,
                        "total_time_s": None,
                        "accepted_updates": None,
                        "rejected_updates": None,
                        "acceptance_rate": None,
                        "mean_cautious_margin": None,
                        "mean_omega": None,
                        "min_omega": None,
                        "max_omega": None,
                    }

                results.append(row)

                if checkpoint_path is not None:
                    save_sweep_results(
                        results,
                        checkpoint_path,
                    )

    return results


def save_sweep_results(
    results: list[dict],
    output_path: str | Path,
) -> None:
    """
    Save sweep results as a CSV file.
    """
    if not results:
        raise ValueError(
            "Cannot save an empty sweep."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nSaved sweep results to: "
        f"{output_path}"
    )