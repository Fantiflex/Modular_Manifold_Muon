"""
Run a single training experiment from the command line.
"""

import argparse

import torch

from src.data import create_cifar100_dataloaders
from src.models import build_model
from src.training import evaluate_accuracy, train_model
from experiments.sweeps import set_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model on CIFAR-100."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "vit"],
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="globalized_rlbfgs",
        choices=["adamw", "rlbfgs", "globalized_rlbfgs"],
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--history",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    train_loader, test_loader = create_cifar100_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
    )

    model = build_model(
        model_type=args.model,
        num_classes=100,
    )

    if args.optimizer == "adamw":
        mode = "adamw"
        manifold_optimizer = "globalized_rlbfgs"
    else:
        mode = "manifold"
        manifold_optimizer = args.optimizer

    model, epoch_losses, epoch_times, optimizer_stats = train_model(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        initial_lr=args.lr,
        device=device,
        mode=mode,
        weight_decay=args.weight_decay,
        history=args.history,
        manifold_optimizer=manifold_optimizer,
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

    print("\n========== RESULTS ==========")
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Final training loss: {epoch_losses[-1]:.4f}")
    print(f"Total training time: {sum(epoch_times):.2f}s")
    if optimizer_stats:
        print(
            f"Accepted curvature updates: "
            f"{optimizer_stats['accepted_updates']}"
        )
        print(
            f"Rejected curvature updates: "
            f"{optimizer_stats['rejected_updates']}"
        )
        print(
            f"Curvature acceptance rate: "
            f"{100 * optimizer_stats['acceptance_rate']:.2f}%"
        )


if __name__ == "__main__":
    main()