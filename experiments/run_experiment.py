"""
Run a single training experiment from the command line.
"""

import argparse

import torch

from src.data import create_cifar100_dataloaders
from src.models import build_model
from src.training import evaluate_accuracy, train_model


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
        "--mode",
        type=str,
        default="manifold",
        choices=["manifold", "adamw"],
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

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    train_loader, test_loader = create_cifar100_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
    )

    model = build_model(
        model_type=args.model,
        num_classes=100,
    )

    model, epoch_losses, epoch_times = train_model(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        initial_lr=args.lr,
        device=device,
        mode=args.mode,
        weight_decay=args.weight_decay,
        history=args.history,
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


if __name__ == "__main__":
    main()