# Modular Manifold MuOn

A modular PyTorch implementation for experimenting with **manifold-constrained optimization** in neural networks, with a particular focus on replacing expensive inner solvers in Manifold MuOn with **Riemannian quasi-Newton methods**.

This repository accompanies my research on making manifold-based optimization more computationally practical for deep learning.

---

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

At a high level:

```text
Euclidean gradient
        ↓
Tangent-space projection
        ↓
Riemannian L-BFGS direction
        ↓
Manifold retraction
        ↓
Updated constrained parameter