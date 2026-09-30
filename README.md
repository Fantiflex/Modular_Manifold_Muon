# Modular Manifold MuOn
Exploring Riemannian quasi-Newton methods as efficient inner solvers for Manifold MuOn.

This project replaces the original dual-ascent inner optimization with
Riemannian L-BFGS and a globalized cautious-update variant on the Stiefel manifold.
The goal is to reduce optimization overhead while preserving constrained geometry.

---
## Key Contributions

- Riemannian L-BFGS implementation for Stiefel-constrained parameters
- Globalized L-BFGS with cautious curvature updates
- Integration with MLP, CNN, and ViT experiments
- Runtime, convergence, and learning-rate robustness comparisons

## Overview

Manifold MuOn constrains matrix-valued neural network parameters to structured manifolds, such as the **Stiefel manifold**, in order to control their geometry during optimization.

While this can provide useful optimization properties, the original formulation relies on an iterative inner solver that can become computationally expensive and sensitive to hyperparameters.

This project investigates an alternative approach based on:

- **Riemannian L-BFGS**
- a **globalized / cautious L-BFGS variant**
- Stiefel-manifold projections and retractions
- tangent-space gradient transport
- controlled step geometry
- comparison with Euclidean and manifold-based baselines

The main objective is to reduce optimization overhead while preserving the geometric constraints imposed on neural network weight matrices.

---

## Research idea

For a matrix parameter \(W\), optimization is performed while maintaining a Stiefel-type constraint such as

\[
W^\top W = I
\]

or its row-orthogonal counterpart depending on the matrix shape.

Rather than solving an expensive constrained inner problem at every update, the project maintains a limited-memory approximation of curvature information in the tangent space.

The globalized Riemannian L-BFGS optimizer additionally uses cautious curvature updates to reject unstable or degenerate history pairs.

## Method

Euclidean gradient
→ Tangent-space projection
→ Riemannian L-BFGS
→ Cautious curvature update
→ Retraction onto the Stiefel manifold