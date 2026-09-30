"""
CIFAR data-loading utilities.
"""

from pathlib import Path

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


CIFAR_MEAN = (
    0.49139968,
    0.48215827,
    0.44653124,
)

CIFAR_STD = (
    0.24703233,
    0.24348505,
    0.26158768,
)

def create_cifar10_dataloaders(
    root: str | Path = "./data",
    batch_size: int = 1024,
    num_workers: int = 0,
    download: bool = True,
):
    """
    Create CIFAR-10 train and test dataloaders.

    Uses the same preprocessing as the original experiments:
    ToTensor + fixed channel-wise normalization.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR_MEAN,
                CIFAR_STD,
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=True,
        transform=transform,
        download=download,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=False,
        transform=transform,
        download=download,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader



def create_cifar100_dataloaders(
    root: str | Path = "./data",
    batch_size: int = 1024,
    num_workers: int = 0,
    download: bool = True,
):
    """
    Create CIFAR-100 train and test dataloaders.

    This reproduces the preprocessing used in the original experiments:
    ToTensor + fixed channel-wise normalization.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR_MEAN,
                CIFAR_STD,
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root=str(root),
        train=True,
        transform=transform,
        download=download,
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=str(root),
        train=False,
        transform=transform,
        download=download,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader