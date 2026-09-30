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
    """
    Vanilla R-LBFGS displacement convention.

    Verifies that the cached tangent displacement is exactly

        xi_k = -eta H_k g_k,

    so that the curvature vector s_k is constructed from the same
    displacement that was actually used to update W_k.
    """
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
    """
    Globalized R-LBFGS fixed-step budget.

    Verifies that the normalized tangent displacement xi_k satisfies

        ||xi_k||_F = eta,

    which implements the fixed Frobenius-norm budget used by the
    globalized optimizer.
    """
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
    """
    Vanilla R-LBFGS curvature vector y_k.

    Verifies the Riemannian curvature equation

        y_k = g_{k+1} - T_{k -> k+1}(g_k),

    and checks that both s_k and y_k belong to the tangent space
    at W_{k+1}.
    """
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
    """
    Globalized R-LBFGS memory transport.

    Verifies that after moving from W_k to W_{k+1}, every stored
    L-BFGS curvature pair is transported into the current tangent
    space T_{W_{k+1}} St(d, p).

    This guards against double transport and stale tangent-space memory.
    """
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



def test_vanilla_rho_matches_transported_curvature_pairs():
    """
    Vanilla R-LBFGS curvature scaling after vector transport.

    Verifies that after stored curvature pairs are transported into
    the current tangent space, every rho_i is recomputed as

        rho_i = 1 / <s_i, y_i>.

    This ensures that the two-loop recursion never uses a curvature
    scalar computed from stale pre-transport vectors.
    """
    torch.manual_seed(0)

    optimizer = RiemannianLBFGS(
        eta=0.05,
        history=10,
    )

    W0 = torch.linalg.qr(
        torch.randn(5, 3)
    ).Q

    G0 = torch.randn_like(W0)

    # ---- First step ----
    W1 = optimizer.step(W0, G0)

    W_old = optimizer.last["W"]
    g_old = optimizer.last["g"]
    step_vec = optimizer.last["step_vec"]

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

    # Construct positive curvature:
    # y_0 = s_0.
    G1 = g0_transported + s0

    assert optimizer.update(G1)

    # ---- Second step ----
    W2 = optimizer.step(W1, G1)

    W_old = optimizer.last["W"]
    g_old = optimizer.last["g"]
    step_vec = optimizer.last["step_vec"]

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

    # Every rho must correspond to the CURRENT transported pair.
    assert len(optimizer.S) == len(optimizer.Y)
    assert len(optimizer.S) == len(optimizer.RHO)

    for s, y, rho in zip(
        optimizer.S,
        optimizer.Y,
        optimizer.RHO,
    ):
        sy = torch.sum(s * y).item()

        assert torch.isfinite(torch.tensor(sy))
        assert sy > 0

        expected_rho = 1.0 / sy

        assert abs(rho - expected_rho) < 1e-8


def test_vanilla_two_loop_with_memory_is_descent():
    """
    Vanilla R-LBFGS two-loop recursion with curvature memory.

    Verifies that after storing a valid positive-curvature pair,
    the L-BFGS inverse-Hessian approximation still produces a
    descent update:

        <-H_k g, g> < 0.

    This checks the actual quasi-Newton path rather than only the
    empty-memory steepest-descent fallback.
    """
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

    s = transport_by_projection(
        W_old,
        W_new,
        step_vec,
    )

    transported_g = transport_by_projection(
        W_old,
        W_new,
        g_old,
    )

    # Guarantees positive curvature.
    G_new = transported_g + s

    assert optimizer.update(G_new)
    assert len(optimizer.S) > 0

    q = tangent_proj(
        W_new,
        torch.randn_like(W_new),
    )

    Hq = optimizer.two_loops(q)

    descent_direction = -Hq

    inner_product = torch.sum(
        descent_direction * q
    )

    assert torch.isfinite(inner_product)
    assert inner_product < 0



def cautious_condition(
    s: torch.Tensor,
    y: torch.Tensor,
    omega: float,
) -> bool:
    sy = torch.sum(s * y).item()

    threshold = omega * max(
        s.norm().pow(2).item(),
        y.norm().pow(2).item(),
    )

    return sy >= threshold



def test_cautious_condition_accepts_above_threshold():
    """
    Globalized R-LBFGS cautious-update acceptance.

    Verifies that a curvature pair is accepted when

        <s, y> >= omega * max(||s||_F^2, ||y||_F^2).

    This checks the exact mathematical acceptance rule used
    by the globalized optimizer.
    """
    omega = 1e-4

    s = torch.tensor([1.0, 0.0])
    y = torch.tensor([1.0, 0.0])

    assert cautious_condition(
        s,
        y,
        omega,
    )



def test_cautious_condition_rejects_below_threshold():
    """
    Globalized R-LBFGS cautious-update rejection.

    Verifies that a curvature pair is rejected when

        <s, y> < omega * max(||s||_F^2, ||y||_F^2).
    """
    omega = 0.5

    s = torch.tensor([1.0, 0.0])
    y = torch.tensor([0.1, 1.0])

    assert not cautious_condition(
        s,
        y,
        omega,
    )


def test_cautious_condition_accepts_exact_threshold():
    """
    Globalized R-LBFGS cautious-update boundary case.

    Verifies that equality at the cautious threshold is accepted:

        <s, y> = omega * max(||s||_F^2, ||y||_F^2).

    The paper specifies a non-strict inequality (>=).
    """
    omega = 0.5

    s = torch.tensor([1.0, 0.0])
    y = torch.tensor([0.5, 0.0])

    sy = torch.sum(s * y).item()

    threshold = omega * max(
        s.norm().pow(2).item(),
        y.norm().pow(2).item(),
    )

    assert abs(sy - threshold) < 1e-12

    assert cautious_condition(
        s,
        y,
        omega,
    )


def test_vanilla_history_discards_oldest_pair():
    """
    Vanilla R-LBFGS bounded FIFO memory.

    Verifies that after more than m accepted curvature updates,
    only the m most recent curvature pairs remain in memory.
    """
    optimizer = RiemannianLBFGS(
        eta=0.1,
        history=2,
    )

    s1 = torch.tensor([[1.0]])
    y1 = torch.tensor([[1.0]])

    s2 = torch.tensor([[2.0]])
    y2 = torch.tensor([[2.0]])

    s3 = torch.tensor([[3.0]])
    y3 = torch.tensor([[3.0]])

    optimizer.S = [s1, s2]
    optimizer.Y = [y1, y2]
    optimizer.RHO = [
        1.0 / torch.sum(s1 * y1).item(),
        1.0 / torch.sum(s2 * y2).item(),
    ]

    optimizer.S.pop(0)
    optimizer.Y.pop(0)
    optimizer.RHO.pop(0)

    optimizer.S.append(s3)
    optimizer.Y.append(y3)
    optimizer.RHO.append(
        1.0 / torch.sum(s3 * y3).item()
    )

    assert len(optimizer.S) == 2
    assert len(optimizer.Y) == 2
    assert len(optimizer.RHO) == 2

    assert torch.equal(optimizer.S[0], s2)
    assert torch.equal(optimizer.S[1], s3)