import torch

from src.optimizers import (
    RiemannianLBFGS,
    GlobalizedRiemannianLBFGS,
)


def random_stiefel(n: int, p: int) -> torch.Tensor:
    """Generate an n x p matrix with orthonormal columns."""
    A = torch.randn(n, p, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q


def assert_on_stiefel(W: torch.Tensor, atol: float = 1e-8):
    """Check W^T W = I."""
    p = W.shape[1]

    identity = torch.eye(
        p,
        dtype=W.dtype,
        device=W.device,
    )

    assert torch.allclose(
        W.T @ W,
        identity,
        atol=atol,
    )


def test_vanilla_step_stays_on_stiefel():
    W = random_stiefel(10, 4)
    G = torch.randn_like(W)

    opt = RiemannianLBFGS(
        eta=0.1,
        history=5,
    )

    W_new = opt.step(W, G)

    assert W_new.shape == W.shape
    assert_on_stiefel(W_new)


def test_globalized_step_stays_on_stiefel():
    W = random_stiefel(10, 4)
    G = torch.randn_like(W)

    opt = GlobalizedRiemannianLBFGS(
        eta=0.1,
        history=5,
    )

    W_new = opt.step(W, G)

    assert W_new.shape == W.shape
    assert_on_stiefel(W_new)


def test_globalized_rejects_bad_curvature_pair():
    """
    If the gradient does not change between W_k and W_{k+1},
    the cautious curvature condition should not accept the pair
    in this constructed case.
    """

    W = random_stiefel(10, 4)

    # Zero gradient gives zero search direction.
    G = torch.zeros_like(W)

    opt = GlobalizedRiemannianLBFGS(
        eta=0.1,
        history=5,
    )

    opt.step(W, G)

    accepted = opt.update(G)

    assert accepted is False
    assert len(opt.pairs) == 0


def test_globalized_history_is_bounded():
    """
    The L-BFGS memory must never exceed `history`.
    """

    opt = GlobalizedRiemannianLBFGS(
        eta=0.05,
        history=2,
    )

    W = random_stiefel(10, 4)

    for _ in range(5):

        G = torch.randn_like(W)

        W_new = opt.step(W, G)

        # Use a perturbed gradient at the new point.
        G_new = G + 0.1 * torch.randn_like(G)

        opt.update(G_new)

        W = W_new

        assert len(opt.pairs) <= 2


def test_vanilla_initial_direction_is_descent():
    optimizer = RiemannianLBFGS(
        eta=0.1,
        history=10,
    )

    q = torch.randn(5, 3)

    h = optimizer.two_loops(q)

    step = -h

    inner_product = torch.sum(step * q)

    assert inner_product < 0



def test_globalized_initial_direction_is_descent():
    optimizer = GlobalizedRiemannianLBFGS(
        eta=0.1,
        history=10,
    )

    q = torch.randn(5, 3)

    h = optimizer.two_loop(
        q,
        gamma=1.0,
    )

    step = -h

    inner_product = torch.sum(step * q)

    assert inner_product < 0