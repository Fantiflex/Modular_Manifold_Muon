"""
Geometry utilities for optimization on the Stiefel manifold.

This module centralizes the geometric operations used by the
Riemannian L-BFGS optimizers:

    - tangent-space projection
    - polar / QR retractions
    - vector transport by projection
    - Frobenius inner product
    - shape-preserving operations for convolutional weights

For tensors with more than two dimensions (e.g. Conv2d weights),
all dimensions except the first one are flattened before applying
the matrix operation, then the original shape is restored.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------

def as_matrix(X: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """
    Represent a tensor as a 2D matrix.

    The first dimension is preserved and all remaining dimensions
    are flattened.

    Examples
    --------
    Linear weight:
        (out_features, in_features)
        -> unchanged

    Conv2d weight:
        (out_channels, in_channels, kH, kW)
        -> (out_channels, in_channels * kH * kW)

    Parameters
    ----------
    X:
        Input tensor.

    Returns
    -------
    X_mat:
        2D matrix representation of X.

    original_shape:
        Original tensor shape, used to restore the tensor later.
    """
    if X.ndim < 2:
        raise ValueError(
            f"Expected a tensor with at least 2 dimensions, got {X.ndim}."
        )

    original_shape = X.shape

    if X.ndim == 2:
        return X, original_shape

    return X.reshape(X.shape[0], -1), original_shape


def from_matrix(
    X_mat: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    """
    Restore a matrix produced by :func:`as_matrix` to its original shape.
    """
    return X_mat.reshape(original_shape)


# ---------------------------------------------------------------------------
# Basic linear algebra
# ---------------------------------------------------------------------------

def sym(X: torch.Tensor) -> torch.Tensor:
    """
    Return the symmetric part of a square matrix.

        Sym(X) = 1/2 (X + X^T)
    """
    return 0.5 * (X + X.mT)


def frob_inner(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Frobenius inner product.

        <X, Y>_F = sum_ij X_ij Y_ij

    This implementation works for both matrices and higher-dimensional
    tensors of identical shape.
    """
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have the same shape, got {X.shape} and {Y.shape}."
        )

    return torch.sum(X * Y)


# ---------------------------------------------------------------------------
# Tangent-space projection
# ---------------------------------------------------------------------------

def tangent_proj(
    W: torch.Tensor,
    Z: torch.Tensor,
) -> torch.Tensor:
    """
    Project an ambient tensor Z onto the tangent space of the
    Stiefel manifold at W.

    For a matrix W with orthonormal columns, the tangent-space condition is

        W^T Δ + Δ^T W = 0.

    The orthogonal projection is

        Proj_W(Z) = Z - W Sym(W^T Z).

    Wide matrices are handled by applying the corresponding operation
    in the transposed representation.

    Higher-dimensional tensors, such as Conv2d kernels, are flattened
    to matrices before projection and restored afterwards.

    Parameters
    ----------
    W:
        Point on the Stiefel manifold.

    Z:
        Ambient direction with the same shape as W.

    Returns
    -------
    torch.Tensor
        Tangent vector at W with the same shape as W.
    """
    if W.shape != Z.shape:
        raise ValueError(
            f"W and Z must have the same shape, got {W.shape} and {Z.shape}."
        )

    W_mat, original_shape = as_matrix(W)
    Z_mat, _ = as_matrix(Z)

    n, p = W_mat.shape

    if n >= p:
        # Standard Stiefel representation: orthonormal columns.
        WTZ = W_mat.mT @ Z_mat
        projected = Z_mat - W_mat @ sym(WTZ)

    else:
        # Wide matrix: work with the transpose, whose columns can be
        # orthonormal, then transpose the result back.
        W_t = W_mat.mT
        Z_t = Z_mat.mT

        WTZ = W_t.mT @ Z_t
        projected_t = Z_t - W_t @ sym(WTZ)

        projected = projected_t.mT

    return from_matrix(projected, original_shape)


# ---------------------------------------------------------------------------
# Retractions
# ---------------------------------------------------------------------------

def polar_retraction(X: torch.Tensor) -> torch.Tensor:
    """
    Project a matrix onto the Stiefel manifold using its polar factor.

    If

        X = U Σ V^T,

    then the closest Stiefel point in Frobenius norm is

        U V^T.

    Both tall and wide matrices are supported.

    Parameters
    ----------
    X:
        2D matrix to retract.

    Returns
    -------
    torch.Tensor
        Polar factor with the same matrix shape as X.
    """
    if X.ndim != 2:
        raise ValueError(
            "polar_retraction expects a 2D matrix. "
            "Use retract_stiefel_shape_preserving for arbitrary weight tensors."
        )

    U, _, Vh = torch.linalg.svd(X, full_matrices=False)

    return U @ Vh


def retract_qr(
    X: torch.Tensor,
    Xi: torch.Tensor,
) -> torch.Tensor:
    """
    QR-based Stiefel retraction.

    Computes

        R_X(Xi) = qf(X + Xi),

    where qf denotes the Q factor of a QR decomposition.

    This function is intended for 2D matrices.
    """
    if X.ndim != 2 or Xi.ndim != 2:
        raise ValueError("retract_qr expects 2D matrices.")

    if X.shape != Xi.shape:
        raise ValueError(
            f"X and Xi must have the same shape, got {X.shape} and {Xi.shape}."
        )

    Y = X + Xi

    # For wide matrices, perform QR on the transpose.
    if Y.shape[0] < Y.shape[1]:
        return retract_qr(Y.mT, torch.zeros_like(Y.mT)).mT

    Q, R = torch.linalg.qr(Y, mode="reduced")

    # Fix QR sign ambiguity to make the retraction deterministic.
    diagonal = torch.diagonal(R)
    signs = torch.where(
        diagonal >= 0,
        torch.ones_like(diagonal),
        -torch.ones_like(diagonal),
    )

    return Q * signs.unsqueeze(0)


@torch.no_grad()
def project_stiefel_keep_shape_qr(
    W: torch.Tensor,
) -> torch.Tensor:
    """
    Project a weight tensor onto the Stiefel manifold using QR while
    preserving its original shape.

    Useful for both Linear and Conv2d weights.
    """
    W_mat, original_shape = as_matrix(W)

    m, n = W_mat.shape

    if m >= n:
        Q, R = torch.linalg.qr(W_mat, mode="reduced")

        diagonal = torch.diagonal(R)
        signs = torch.where(
            diagonal >= 0,
            torch.ones_like(diagonal),
            -torch.ones_like(diagonal),
        )

        W_projected = Q * signs.unsqueeze(0)

    else:
        # Wide matrix: project the transpose.
        Q, R = torch.linalg.qr(W_mat.mT, mode="reduced")

        diagonal = torch.diagonal(R)
        signs = torch.where(
            diagonal >= 0,
            torch.ones_like(diagonal),
            -torch.ones_like(diagonal),
        )

        W_projected = (Q * signs.unsqueeze(0)).mT

    return from_matrix(W_projected, original_shape)


@torch.no_grad()
def retract_stiefel_shape_preserving(
    X: torch.Tensor,
    Xi: torch.Tensor,
) -> torch.Tensor:
    """
    Retract X + Xi onto the Stiefel manifold while preserving the
    original tensor shape.

    Higher-dimensional tensors are flattened to matrices before the
    polar/SVD projection and reshaped afterwards.

    Parameters
    ----------
    X:
        Current point.

    Xi:
        Tangent update.

    Returns
    -------
    torch.Tensor
        Retracted tensor with the same shape as X.
    """
    if X.shape != Xi.shape:
        raise ValueError(
            f"X and Xi must have the same shape, got {X.shape} and {Xi.shape}."
        )

    Y = X + Xi

    Y_mat, original_shape = as_matrix(Y)

    try:
        U, _, Vh = torch.linalg.svd(
            Y_mat,
            full_matrices=False,
        )

    except torch._C._LinAlgError:
        if not torch.isfinite(Y_mat).all():
            raise

        # Numerical fallback:
        # compute the same polar factor in higher precision.
        Y_high_precision = Y_mat.to(torch.float64)

        U, _, Vh = torch.linalg.svd(
            Y_high_precision,
            full_matrices=False,
        )

        U = U.to(Y_mat.dtype)
        Vh = Vh.to(Y_mat.dtype)
    Y_retracted = U @ Vh

    return from_matrix(Y_retracted, original_shape)


# ---------------------------------------------------------------------------
# Vector transport
# ---------------------------------------------------------------------------

def transport_by_projection(
    W_old: torch.Tensor,
    W_new: torch.Tensor,
    Xi: torch.Tensor,
) -> torch.Tensor:
    """
    Transport a tangent vector from T_{W_old}M to T_{W_new}M by
    projecting it onto the new tangent space.

        T_{W_old -> W_new}(Xi) = Proj_{W_new}(Xi)

    This transport is simple and robust, although it is not isometric.

    W_old is kept in the signature because it makes the geometric
    operation explicit and allows a different transport rule to be
    introduced later without changing the optimizer API.
    """
    if W_old.shape != W_new.shape or W_new.shape != Xi.shape:
        raise ValueError(
            "W_old, W_new and Xi must all have the same shape."
        )

    return tangent_proj(W_new, Xi)


# Alias used by some of the original experimental notebooks.
transport_proj = transport_by_projection