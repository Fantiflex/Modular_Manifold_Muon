import torch

from src.geometry.stiefel import (
    tangent_proj,
    polar_retraction,
    retract_stiefel_shape_preserving,
    transport_by_projection,
)


def random_stiefel(n: int, p: int) -> torch.Tensor:
    """
    Generate a random n x p matrix with orthonormal columns.
    Requires n >= p.
    """
    assert n >= p

    A = torch.randn(n, p, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A, mode="reduced")

    return Q


def test_tangent_projection():
    """
    Stiefel tangent-space projection.

    Verifies that tangent_proj(W, Z) satisfies the defining
    tangent-space constraint

        W^T Z + Z^T W = 0.
    """

    W = random_stiefel(10, 4)
    Z = torch.randn_like(W)

    Z_tangent = tangent_proj(W, Z)

    residual = (
        W.T @ Z_tangent
        + Z_tangent.T @ W
    )

    assert torch.allclose(
        residual,
        torch.zeros_like(residual),
        atol=1e-8,
    )


def test_polar_retraction():
    """
    Polar retraction onto the Stiefel manifold.

    Verifies that the retracted matrix has orthonormal columns:

        W^T W = I.
    """

    X = torch.randn(
        10,
        4,
        dtype=torch.float64,
    )

    W = polar_retraction(X)

    gram = W.T @ W

    identity = torch.eye(
        4,
        dtype=W.dtype,
    )

    assert torch.allclose(
        gram,
        identity,
        atol=1e-8,
    )


def test_shape_preserving_retraction():
    """
    Retraction must preserve the original tensor shape.
    """

    X = torch.randn(
        10,
        4,
        dtype=torch.float64,
    )

    Xi = 0.01 * torch.randn_like(X)

    W = retract_stiefel_shape_preserving(
        X,
        Xi,
    )

    assert W.shape == X.shape

    gram = W.T @ W

    identity = torch.eye(
        4,
        dtype=W.dtype,
    )

    assert torch.allclose(
        gram,
        identity,
        atol=1e-8,
    )


def test_vector_transport_is_tangent():
    """
    Projection-based vector transport.

    Verifies that a tangent vector transported from W_k to W_{k+1}
    belongs to the destination tangent space:

        W_{k+1}^T Z + Z^T W_{k+1} = 0.
    """

    W_old = random_stiefel(10, 4)

    step = 0.01 * torch.randn_like(W_old)

    W_new = retract_stiefel_shape_preserving(
        W_old,
        step,
    )

    Xi = tangent_proj(
        W_old,
        torch.randn_like(W_old),
    )

    transported = transport_by_projection(
        W_old,
        W_new,
        Xi,
    )

    residual = (
        W_new.T @ transported
        + transported.T @ W_new
    )

    assert torch.allclose(
        residual,
        torch.zeros_like(residual),
        atol=1e-8,
    )