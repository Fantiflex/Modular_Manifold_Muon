from __future__ import annotations

import torch

from experiments.sweeps import (
    run_optimizer_sweep,
    save_sweep_results,
)
from src.data import create_cifar10_dataloaders


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    batch_size = 1024
    epochs = 3
    history = 10

    seeds = [42]
    etas = [0.1]

    optimizers = [
        "dual_ascent",
        "rlbfgs",
        "globalized_rlbfgs",
    ]

    train_loader, test_loader = create_cifar10_dataloaders(
        batch_size=batch_size,
        root="data",
    )

    results = run_optimizer_sweep(
        model_type="mlp",
        train_loader=train_loader,
        test_loader=test_loader,
        optimizers=optimizers,
        etas=etas,
        seeds=seeds,
        device=device,
        epochs=epochs,
        num_classes=10,
        batch_size=batch_size,
        history=history,
        checkpoint_path=(
            "results/mlp_timing_benchmark.csv"
        ),
    )

    save_sweep_results(
        results,
        "results/mlp_timing_benchmark.csv",
    )


if __name__ == "__main__":
    main()