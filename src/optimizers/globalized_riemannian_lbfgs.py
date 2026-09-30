"""
Globalized Riemannian L-BFGS optimizer on the Stiefel manifold.

Compared with vanilla Riemannian L-BFGS, this implementation uses a
cautious-update mechanism that only accepts curvature pairs satisfying

    <s_k, y_k> >= omega_k * max(||s_k||^2, ||y_k||^2)

where

    omega_k = min(c0, c1 * ||grad f(W_k)||^c2).

The implementation follows the globalized variant used in the original
research notebooks.
"""

from __future__ import annotations

import torch

from src.geometry.stiefel import (
    tangent_proj,
    transport_by_projection,
    retract_stiefel_shape_preserving,
)


class GlobalizedRiemannianLBFGS:
    """
    Globalized limited-memory Riemannian BFGS optimizer.

    Parameters
    ----------
    eta : float
        Step size.

    history : int
        Maximum number of curvature pairs stored.

    c0, c1 : float
        Constants controlling the cautious-update threshold.

    c2 : float or None
        Exponent used in the cautious-update threshold.
        If None:

            c2 = 1 / (2 * history + 3)
    """

    def __init__(
        self,
        eta: float = 0.1,
        history: int = 10,
        c0: float = 1e-4,
        c1: float = 1.0,
        c2: float | None = None,
    ):
        self.eta = eta
        self.history = history

        self.c0 = c0
        self.c1 = c1

        self.c2 = (
            1.0 / (2 * history + 3)
            if c2 is None
            else c2
        )

        # Curvature pairs (s_i, y_i), represented in the current
        # tangent space.
        self.pairs: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []

        # Information saved between step() and update().
        self._pending = None

    # ------------------------------------------------------------------
    # L-BFGS two-loop recursion
    # ------------------------------------------------------------------

    def two_loop(
        self,
        G: torch.Tensor,
        gamma: float,
        eps: float = 1e-16,
    ) -> torch.Tensor:
        """
        Apply the L-BFGS two-loop recursion.

        The stored curvature pairs are assumed to live in the current
        tangent space.
        """

        q = G.clone()

        alphas = []

        # Backward loop
        for s, y in reversed(self.pairs):

            ys = torch.dot(
                y.flatten(),
                s.flatten(),
            ).item()

            alpha = (
                torch.dot(
                    s.flatten(),
                    q.flatten(),
                ).item()
                / (ys + eps)
            )

            alphas.append(alpha)

            q = q - alpha * y

        # Initial inverse-Hessian approximation:
        #
        # H_0 = gamma I
        r = gamma * q

        # Forward loop
        for (s, y), alpha in zip(
            self.pairs,
            reversed(alphas),
        ):

            ys = torch.dot(
                y.flatten(),
                s.flatten(),
            ).item()

            beta = (
                torch.dot(
                    y.flatten(),
                    r.flatten(),
                ).item()
                / (ys + eps)
            )

            r = r + (alpha - beta) * s

        # Search direction
        return -r

    # ------------------------------------------------------------------
    # Initial inverse-Hessian scaling
    # ------------------------------------------------------------------

    def _gamma_from_memory(
        self,
        default: float = 1.0,
    ) -> float:
        """
        Compute the scalar initial inverse-Hessian approximation.

        Uses the most recent curvature pair.
        """

        if not self.pairs:
            return default

        s, y = self.pairs[-1]

        sty = torch.dot(
            s.flatten(),
            y.flatten(),
        ).item()

        yy = torch.dot(
            y.flatten(),
            y.flatten(),
        ).item()

        if sty > 0:
            gamma = sty / (yy + 1e-16)

        elif yy > 0:
            gamma = (
                s.norm()
                / (y.norm() + 1e-16)
            ).item()

        else:
            gamma = default

        # Same bounded scaling used in the research notebook.
        omega = min(
            self.c0,
            self.c1 * (1.0 ** self.c2),
        )

        return max(
            omega,
            min(gamma, 1.0 / omega),
        )

    # ------------------------------------------------------------------
    # Optimization step
    # ------------------------------------------------------------------

    def step(
        self,
        W: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one globalized Riemannian L-BFGS step.
        """

        if W.shape != G.shape:
            raise ValueError(
                "W and G must have identical shapes. "
                f"Got W={W.shape}, G={G.shape}."
            )

        # Euclidean gradient -> Riemannian gradient
        G_riem = tangent_proj(W, G)

        # Initial inverse-Hessian scaling
        gamma = self._gamma_from_memory(
            default=1.0
        )

        # Quasi-Newton search direction
        P = self.two_loop(
            G_riem,
            gamma,
        )

        # --------------------------------------------------------------
        # Fixed-length step
        #
        # This preserves the behavior of the original notebook:
        #
        #     eta * P / ||P||
        # --------------------------------------------------------------

        if P.norm().item() == 0:
            step_vec = self.eta * P

        else:
            step_vec = (
                self.eta
                * P
                / (P.norm() + 1e-16)
            )

        # Retraction onto Stiefel
        W_new = retract_stiefel_shape_preserving(
            W,
            step_vec,
        )

        # Save information required for cautious update.
        self._pending = (
            W.detach().clone(),
            G_riem.detach().clone(),
            step_vec.detach().clone(),
            W_new.detach().clone(),
        )

        return W_new

    # ------------------------------------------------------------------
    # Cautious curvature update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        G_new: torch.Tensor,
    ) -> bool:
        """
        Build and conditionally accept a new curvature pair.

        Returns
        -------
        bool
            True if the pair satisfies the cautious-update condition.
        """

        if self._pending is None:
            return False

        (
            W_old,
            G_old_riem,
            step_vec,
            W_new,
        ) = self._pending

        # --------------------------------------------------------------
        # Construct s_k
        # --------------------------------------------------------------

        s = transport_by_projection(
            W_old,
            W_new,
            step_vec,
        )

        # --------------------------------------------------------------
        # Construct y_k
        # --------------------------------------------------------------

        G_new_riem = tangent_proj(
            W_new,
            G_new,
        )

        transported_old_gradient = (
            transport_by_projection(
                W_old,
                W_new,
                G_old_riem,
            )
        )

        y = (
            G_new_riem
            - transported_old_gradient
        )

        # --------------------------------------------------------------
        # Cautious-update threshold
        # --------------------------------------------------------------

        omega = min(
            self.c0,
            self.c1
            * (
                G_old_riem.norm().item()
                ** self.c2
            ),
        )

        sy = torch.dot(
            s.flatten(),
            y.flatten(),
        ).item()

        threshold = omega * max(
            s.norm().pow(2).item(),
            y.norm().pow(2).item(),
        )

        # Reject degenerate curvature pairs explicitly.
        nondegenerate = (
            s.norm().item() > 1e-12
            and y.norm().item() > 1e-12
        )

        accepted = nondegenerate and sy >= threshold

        if accepted:
            self.pairs.append(
                (
                    s.detach(),
                    y.detach(),
                )
            )

            if len(self.pairs) > self.history:
                self.pairs.pop(0)
            # --------------------------------------------------------------
        # Transport the entire L-BFGS memory into the new tangent space
        # --------------------------------------------------------------

        self.pairs = [
            (
                transport_by_projection(
                    W_old,
                    W_new,
                    s_i,
                ),
                transport_by_projection(
                    W_old,
                    W_new,
                    y_i,
                ),
            )
            for s_i, y_i in self.pairs
        ]

        self._pending = None

        return accepted