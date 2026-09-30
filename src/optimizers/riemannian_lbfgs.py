"""
Riemannian L-BFGS optimizer on the Stiefel manifold.

This module implements the vanilla Riemannian L-BFGS method used in
the Manifold Muon experiments.

Geometry operations are defined separately in src.geometry.
"""

from __future__ import annotations

import torch

from src.geometry.stiefel import (
    frob_inner,
    polar_retraction,
    tangent_proj,
    transport_by_projection,
)

from src.geometry.matrix_sign import msign


class RiemannianLBFGS:
    """
    Limited-memory Riemannian BFGS optimizer on the Stiefel manifold.

    The implementation uses:

    - Euclidean / Frobenius metric
    - tangent-space projection
    - polar retraction
    - vector transport by projection
    - limited-memory BFGS two-loop recursion

    Parameters
    ----------
    eta : float
        Step size.

    history : int
        Maximum number of curvature pairs stored by L-BFGS.

    eps_curv : float
        Minimum accepted curvature <s_k, y_k>.

    use_polar_impl : bool
        If True, use polar_retraction().
        Otherwise, use the matrix-sign implementation.
    """

    def __init__(
        self,
        eta: float = 0.1,
        history: int = 10,
        eps_curv: float = 1e-12,
        use_polar_impl: bool = True,
    ):
        self.eta = eta
        self.m = history
        self.eps_curv = eps_curv
        self.use_polar_impl = use_polar_impl

        # L-BFGS history
        self.S = []
        self.Y = []
        self.RHO = []

        # Information cached between step() and update()
        self.last = None

    # ------------------------------------------------------------------
    # Retraction
    # ------------------------------------------------------------------

    def _retract(self, X: torch.Tensor) -> torch.Tensor:
        """
        Retract X back onto the Stiefel manifold.
        """
        if self.use_polar_impl:
            return polar_retraction(X)

        return msign(X)

    # ------------------------------------------------------------------
    # L-BFGS two-loop recursion
    # ------------------------------------------------------------------

    def two_loops(self, q: torch.Tensor) -> torch.Tensor:
        """
        Apply the L-BFGS inverse-Hessian approximation to q.

        Parameters
        ----------
        q :
            Current Riemannian gradient.

        Returns
        -------
        torch.Tensor
            L-BFGS search quantity.
        """

        # No curvature information yet:
        # fall back to the behavior used in the original notebook.
        if len(self.S) == 0:
            return -q

        alpha = []

        # --------------------------------------------------------------
        # Backward loop
        # --------------------------------------------------------------

        for s, y, rho in reversed(
            list(zip(self.S, self.Y, self.RHO))
        ):
            a = rho * frob_inner(s, q)

            alpha.append(a)

            q = q - a * y

        # --------------------------------------------------------------
        # Initial inverse-Hessian scaling
        #
        # H_0 = gamma I
        # --------------------------------------------------------------

        y_last = self.Y[-1]
        s_last = self.S[-1]

        sy = frob_inner(s_last, y_last)
        yy = frob_inner(y_last, y_last)

        if yy > 0:
            gamma = sy / yy
        else:
            gamma = 1.0

        r = gamma * q

        # --------------------------------------------------------------
        # Forward loop
        # --------------------------------------------------------------

        for (s, y, rho), a in zip(
            zip(self.S, self.Y, self.RHO),
            reversed(alpha),
        ):
            beta = rho * frob_inner(y, r)

            r = r + s * (a - beta)

        return r

    # ------------------------------------------------------------------
    # Optimization step
    # ------------------------------------------------------------------

    def step(
        self,
        W: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take one Riemannian L-BFGS step.

        Parameters
        ----------
        W :
            Current point on the Stiefel manifold.

        G :
            Raw Euclidean gradient evaluated at W.

        Returns
        -------
        torch.Tensor
            New point on the Stiefel manifold.
        """

        if W.shape != G.shape:
            raise ValueError(
                "W and G must have the same shape. "
                f"Got W={W.shape}, G={G.shape}."
            )

        # --------------------------------------------------------------
        # Handle wide matrices
        # --------------------------------------------------------------

        should_transpose = W.shape[0] < W.shape[1]

        if should_transpose:
            W = W.T
            G = G.T

        # --------------------------------------------------------------
        # Euclidean gradient -> Riemannian gradient
        # --------------------------------------------------------------

        g = tangent_proj(W, G)

        # --------------------------------------------------------------
        # L-BFGS search quantity
        # --------------------------------------------------------------

        d = self.two_loops(g)

        # Numerical safeguard:
        # ensure the direction is tangent.
        d = tangent_proj(W, d)

        # --------------------------------------------------------------
        # Ambient update + retraction
        # --------------------------------------------------------------

        W_new = W - self.eta * d

        W_new = self._retract(W_new)

        # --------------------------------------------------------------
        # Save information needed by update()
        # --------------------------------------------------------------

        self.last = {
            "W": W.detach().clone(),
            "W_new": W_new.detach().clone(),
            "g": g.detach().clone(),
            "d": d.detach().clone(),
            "should_transpose": should_transpose,
        }

        if should_transpose:
            return W_new.T

        return W_new

    # ------------------------------------------------------------------
    # Curvature update
    # ------------------------------------------------------------------

    def update(self, G_new: torch.Tensor) -> bool:
        """
        Update the L-BFGS curvature history.

        This must be called after step(), once the gradient at the new
        point W_{k+1} has been computed.

        Parameters
        ----------
        G_new :
            Euclidean gradient evaluated at the new point.

        Returns
        -------
        bool
            True if the curvature pair was accepted.
            False if it failed the curvature safeguard.
        """

        if self.last is None:
            raise RuntimeError(
                "update() must be called after step()."
            )

        W = self.last["W"]
        W_new = self.last["W_new"]

        d = self.last["d"]
        g = self.last["g"]

        should_transpose = self.last["should_transpose"]

        if should_transpose:
            G_new = G_new.T

        # --------------------------------------------------------------
        # Gradient at the new point
        # --------------------------------------------------------------

        g_new = tangent_proj(W_new, G_new)

        # --------------------------------------------------------------
        # Transport quantities to T_{W_new} M
        # --------------------------------------------------------------

        s = transport_by_projection(
            W,
            W_new,
            self.eta * d,
        )

        transported_g = transport_by_projection(
            W,
            W_new,
            g,
        )

        y = g_new - transported_g

        # --------------------------------------------------------------
        # Curvature safeguard
        # --------------------------------------------------------------

        sy = frob_inner(s, y)

        if (
            not torch.isfinite(sy)
            or sy <= self.eps_curv
        ):
            self.last = None
            return False

        # --------------------------------------------------------------
        # Maintain limited-memory history
        # --------------------------------------------------------------

        if len(self.S) == self.m:
            self.S.pop(0)
            self.Y.pop(0)
            self.RHO.pop(0)

        self.S.append(
            s.detach().clone()
        )

        self.Y.append(
            y.detach().clone()
        )

        self.RHO.append(
            1.0 / sy
        )

        self.last = None

        return True