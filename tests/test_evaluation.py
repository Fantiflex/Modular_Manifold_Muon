import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training import evaluate_accuracy


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x


def test_evaluate_accuracy():
    logits = torch.tensor([
        [10.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
        [0.0, 10.0],
    ])

    labels = torch.tensor([
        0,
        1,
        0,
        1,
    ])

    loader = DataLoader(
        TensorDataset(logits, labels),
        batch_size=2,
    )

    model = DummyModel()

    accuracy = evaluate_accuracy(
        model,
        loader,
        torch.device("cpu"),
    )

    assert accuracy == 100.0