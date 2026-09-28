"""
Run a single training experiment from the command line.
"""

import argparse

import torch

from src.models import build_model
from src.training import evaluate_accuracy, train_model


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--num-classes",
        type=int,
        default=100,
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

    model = build_model(
        model_type=args.model,
        num_classes=args.num_classes,
    )

    # DataLoaders will be plugged in next.
    raise NotImplementedError(
        "Dataset loading is not yet connected. "
        "Next step: add CIFAR data utilities."
    )


if __name__ == "__main__":
    main()