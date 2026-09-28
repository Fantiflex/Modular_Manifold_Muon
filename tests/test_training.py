from src.training import linear_decay_lr
import torch

from src.optimizers import GlobalizedRiemannianLBFGS
from src.training import manifold_parameter_step
from src.training import (
    initialize_manifold_optimizers,
    linear_decay_lr,
    manifold_parameter_step,
)




def test_manifold_parameter_step_preserves_stiefel():
    model = torch.nn.Linear(
        4,
        10,
        bias=False,
        dtype=torch.float64,
    )

    # Linear weight has shape (10, 4), so it can have
    # orthonormal columns.
    with torch.no_grad():
        Q, _ = torch.linalg.qr(
            torch.randn(
                10,
                4,
                dtype=torch.float64,
            )
        )
        model.weight.copy_(Q)

    optimizer = GlobalizedRiemannianLBFGS(
        eta=0.05,
        history=5,
    )

    manifold_optimizers = {
        model.weight: optimizer,
    }

    x = torch.randn(
        8,
        4,
        dtype=torch.float64,
    )

    loss = model(x).pow(2).mean()

    model.zero_grad()
    loss.backward()

    manifold_parameter_step(
        model=model,
        manifold_optimizers=manifold_optimizers,
        lr=0.05,
    )

    W = model.weight.detach()

    identity = torch.eye(
        W.shape[1],
        dtype=W.dtype,
    )

    assert torch.allclose(
        W.T @ W,
        identity,
        atol=1e-6,
        rtol=1e-6,
    )

def test_linear_decay_lr_start():
    lr = linear_decay_lr(
        initial_lr=0.1,
        step=0,
        total_steps=100,
    )

    assert lr == 0.1


def test_linear_decay_lr_halfway():
    lr = linear_decay_lr(
        initial_lr=0.1,
        step=50,
        total_steps=100,
    )

    assert abs(lr - 0.05) < 1e-12


def test_linear_decay_lr_end():
    lr = linear_decay_lr(
        initial_lr=0.1,
        step=100,
        total_steps=100,
    )

    assert abs(lr) < 1e-12



def test_initialize_manifold_optimizers():
    model = torch.nn.Sequential(
        torch.nn.Linear(
            4,
            10,
            bias=True,
            dtype=torch.float64,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            10,
            6,
            bias=False,
            dtype=torch.float64,
        ),
    )

    manifold_optimizers = initialize_manifold_optimizers(
        model=model,
        eta=0.05,
        history=5,
    )

    # Only the two matrix weights should be manifold parameters.
    assert len(manifold_optimizers) == 2

    for parameter in manifold_optimizers:
        W = parameter.detach()

        if W.shape[0] >= W.shape[1]:
            identity = torch.eye(
                W.shape[1],
                dtype=W.dtype,
            )

            gram = W.T @ W

        else:
            identity = torch.eye(
                W.shape[0],
                dtype=W.dtype,
            )

            gram = W @ W.T

        assert torch.allclose(
            gram,
            identity,
            atol=1e-6,
            rtol=1e-6,
        )



def test_train_model_adamw_runs():
    from torch.utils.data import DataLoader, TensorDataset

    from src.training import train_model

    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=4,
    )

    model = torch.nn.Linear(4, 2)

    model, losses, times = train_model(
        model=model,
        train_loader=loader,
        epochs=2,
        initial_lr=0.01,
        device=torch.device("cpu"),
        mode="adamw",
    )

    assert len(losses) == 2
    assert len(times) == 2
    assert all(torch.isfinite(torch.tensor(losses)))