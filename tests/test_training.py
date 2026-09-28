from src.training import linear_decay_lr
import torch

from src.optimizers import GlobalizedRiemannianLBFGS
from src.training import manifold_parameter_step


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