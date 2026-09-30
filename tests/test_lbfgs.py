import torch

from src.optimizers import (
    RiemannianLBFGS,
    GlobalizedRiemannianLBFGS,
)

from src.geometry.stiefel import (
    tangent_proj,
    transport_by_projection,
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




def test_vanilla_cached_step_matches_applied_displacement():
    torch.manual_seed(0)

    optimizer = RiemannianLBFGS(
        eta=0.1,
        history=10,
    )

    W = torch.linalg.qr(
        torch.randn(5, 3)
    ).Q

    G = torch.randn_like(W)

    optimizer.step(W, G)

    assert optimizer.last is not None

    d = optimizer.last["d"]
    step_vec = optimizer.last["step_vec"]

    expected_step = -optimizer.eta * d

    assert torch.allclose(
        step_vec,
        expected_step,
        atol=1e-6,
    )


def test_vanilla_curvature_displacement_is_transported_step():
    torch.manual_seed(0)

    optimizer = RiemannianLBFGS(
        eta=0.05,
        history=10,
    )

    W = torch.linalg.qr(
        torch.randn(5, 3)
    ).Q

    G = torch.randn_like(W)

    W_new = optimizer.step(W, G)

    W_old = optimizer.last["W"]
    step_vec = optimizer.last["step_vec"]

    expected_s = transport_by_projection(
        W_old,
        W_new,
        step_vec,
    )

    # Construct a new gradient that produces
    # positive curvature in the step direction.
    G_new = G + step_vec

    accepted = optimizer.update(G_new)

    if accepted:
        stored_s = optimizer.S[-1]

        assert torch.allclose(
            stored_s,
            expected_s,
            atol=1e-5,
        )


def test_globalized_cached_step_has_fixed_frobenius_norm():
    torch.manual_seed(0)

    eta = 0.1

    optimizer = GlobalizedRiemannianLBFGS(
        eta=eta,
        history=10,
    )

    W = torch.linalg.qr(
        torch.randn(5, 3)
    ).Q

    G = torch.randn_like(W)

    optimizer.step(W, G)

    assert optimizer._pending is not None

    _, _, step_vec, _ = optimizer._pending

    assert torch.allclose(
        step_vec.norm(),
        torch.tensor(
            eta,
            dtype=step_vec.dtype,
            device=step_vec.device,
        ),
        atol=1e-5,
    )




def test_globalized_curvature_displacement_is_transported_step():
    torch.manual_seed(0)

    optimizer = GlobalizedRiemannianLBFGS(
        eta=0.05,
        history=10,
    )

    W = torch.linalg.qr(
        torch.randn(5, 3)
    ).Q

    G = torch.randn_like(W)

    W_new = optimizer.step(W, G)

    W_old, _, step_vec, _ = optimizer._pending

    expected_s = transport_by_projection(
        W_old,
        W_new,
        step_vec,
    )

    G_new = G + step_vec

    accepted = optimizer.update(G_new)

    if accepted:
        stored_s, _ = optimizer.pairs[-1]

        assert torch.allclose(
            stored_s,
            expected_s,
            atol=1e-5,
        )



def assert_tangent(W, Z, atol=1e-5):
    residual = W.T @ Z + Z.T @ W

    assert torch.allclose(
        residual,
        torch.zeros_like(residual),
        atol=atol,
    )


def test_vanilla_y_matches_gradient_difference():
    torch.manual_seed(0)

    optimizer = RiemannianLBFGS(
        eta=0.05,
        history=10,
    )

    W = torch.linalg.qr(
        torch.randn(5, 3)
    ).Q

    G = torch.randn_like(W)

    W_new = optimizer.step(W, G)

    W_old = optimizer.last["W"]
    g_old = optimizer.last["g"]
    step_vec = optimizer.last["step_vec"]

    expected_s = transport_by_projection(
        W_old,
        W_new,
        step_vec,
    )

    transported_g = transport_by_projection(
        W_old,
        W_new,
        g_old,
    )

    # Construct G_new so that:
    #
    # g_new = transported_g + expected_s
    #
    # Therefore:
    #
    # y = g_new - transported_g = expected_s
    #
    G_new = transported_g + expected_s

    expected_g_new = tangent_proj(
        W_new,
        G_new,
    )

    expected_y = (
        expected_g_new
        - transported_g
    )

    accepted = optimizer.update(G_new)

    assert accepted

    stored_s = optimizer.S[-1]
    stored_y = optimizer.Y[-1]

    assert torch.allclose(
        stored_s,
        expected_s,
        atol=1e-5,
    )

    assert torch.allclose(
        stored_y,
        expected_y,
        atol=1e-5,
    )
    assert_tangent(W_new, stored_s)
    assert_tangent(W_new, stored_y)



def test_globalized_y_matches_gradient_difference():
    torch.manual_seed(0)

    optimizer = GlobalizedRiemannianLBFGS(
        eta=0.05,
        history=10,
    )

    W = torch.linalg.qr(
        torch.randn(5, 3)
    ).Q

    G = torch.randn_like(W)

    W_new = optimizer.step(W, G)

    (
        W_old,
        g_old,
        step_vec,
        _,
    ) = optimizer._pending

    expected_s = transport_by_projection(
        W_old,
        W_new,
        step_vec,
    )

    transported_g = transport_by_projection(
        W_old,
        W_new,
        g_old,
    )

    G_new = transported_g + expected_s

    expected_g_new = tangent_proj(
        W_new,
        G_new,
    )

    expected_y = (
        expected_g_new
        - transported_g
    )

    accepted = optimizer.update(G_new)

    assert accepted

    stored_s, stored_y = optimizer.pairs[-1]

    assert torch.allclose(
        stored_s,
        expected_s,
        atol=1e-5,
    )

    assert torch.allclose(
        stored_y,
        expected_y,
        atol=1e-5,
    )
    assert_tangent(W_new, stored_s)
    assert_tangent(W_new, stored_y)


def test_globalized_memory_stays_in_current_tangent_space():
    torch.manual_seed(0)

    optimizer = GlobalizedRiemannianLBFGS(
        eta=0.05,
        history=10,
    )

    W0 = torch.linalg.qr(
        torch.randn(5, 3)
    ).Q

    G0 = torch.randn_like(W0)

    # First step
    W1 = optimizer.step(W0, G0)

    W_old, g_old, step_vec, _ = optimizer._pending

    s0 = transport_by_projection(
        W_old,
        W1,
        step_vec,
    )

    g0_transported = transport_by_projection(
        W_old,
        W1,
        g_old,
    )

    G1 = g0_transported + s0

    assert optimizer.update(G1)

    # Second step
    W2 = optimizer.step(W1, G1)

    W_old, g_old, step_vec, _ = optimizer._pending

    s1 = transport_by_projection(
        W_old,
        W2,
        step_vec,
    )

    g1_transported = transport_by_projection(
        W_old,
        W2,
        g_old,
    )

    G2 = g1_transported + s1

    assert optimizer.update(G2)

    # Every stored pair must now live in T_{W2}M.
    for s, y in optimizer.pairs:
        assert_tangent(W2, s)
        assert_tangent(W2, y)



def test_globalized_gamma_matches_standard_lbfgs_scaling():
    optimizer = GlobalizedRiemannianLBFGS(
        eta=0.1,
        history=10,
    )

    s = torch.tensor([[2.0, 0.0]])
    y = torch.tensor([[1.0, 0.0]])

    optimizer.pairs = [(s, y)]

    gamma = optimizer._gamma_from_memory()

    expected = (
        torch.sum(s * y)
        / torch.sum(y * y)
    ).item()

    assert abs(gamma - expected) < 1e-12


def test_globalized_gamma_respects_lower_bound():
    optimizer = GlobalizedRiemannianLBFGS(
        eta=0.1,
        history=10,
        c0=1e-4,
        c1=1.0,
    )

    s = torch.tensor([[1e-10, 0.0]])
    y = torch.tensor([[1.0, 0.0]])

    optimizer.pairs = [(s, y)]

    gamma = optimizer._gamma_from_memory()

    assert gamma >= 1e-4



def test_globalized_gamma_respects_upper_bound():
    optimizer = GlobalizedRiemannianLBFGS(
        eta=0.1,
        history=10,
        c0=1e-4,
        c1=1.0,
    )

    s = torch.tensor([[1e6, 0.0]])
    y = torch.tensor([[1.0, 0.0]])

    optimizer.pairs = [(s, y)]

    gamma = optimizer._gamma_from_memory()

    assert gamma <= 1e4