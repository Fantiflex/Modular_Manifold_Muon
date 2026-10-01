import math
import torch
from src.geometry.bernstein_msign import msign


@torch.no_grad()
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6):
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    # Ascend on the dual problem to find the update direction A
    for step in range(steps):
        # Update the candidate direction A
        A = msign(G + 2 * W @ Lambda)
        # Measure deviation of A from the tangent space:
        H = W.T @ A + A.T @ W
        # Check the stopping criterion
        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break
        # Update the dual variable
        Lambda -= alpha * (1 - step / steps) * H
    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    return new_W.T if should_tranpose else new_W



class ManifoldMuon:
    """
    Adapter around Bernstein's Manifold MuOn update rule so it can be
    used through the same interface as the R-LBFGS optimizers.
    """

    def __init__(
        self,
        eta: float = 0.1,
        history: int = 10,
        alpha: float = 0.01,
        steps: int = 100,
        tol: float = 1e-6,
    ):
        self.eta = eta

        # Kept only for API compatibility with the sweep pipeline.
        self.history = history

        self.alpha = alpha
        self.steps = steps
        self.tol = tol

    def update(self, G: torch.Tensor) -> None:
        """
        Manifold MuOn does not maintain curvature memory.
        """
        return None

    def step(
        self,
        W: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        return manifold_muon(
            W=W,
            G=G,
            eta=self.eta,
            alpha=self.alpha,
            steps=self.steps,
            tol=self.tol,
        )