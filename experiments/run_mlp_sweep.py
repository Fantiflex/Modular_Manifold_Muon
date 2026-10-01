from __future__ import annotations

import torch

from experiments.sweeps import (
    run_optimizer_sweep,
    save_sweep_results,
)
from src.data import create_cifar10_dataloaders

def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    batch_size = 1024
    epochs = 3
    history = 10

    seeds = [42]

    etas = [0.1]

    optimizers = [
        
        "dual_ascent",
    ]

    train_loader, test_loader = (
        create_cifar10_dataloaders(
            root="data",
            batch_size=batch_size,
        )
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
    )

    save_sweep_results(
        results,
        "results/mlp_optimizer_sweep.csv",
    )


if __name__ == "__main__":
    main()