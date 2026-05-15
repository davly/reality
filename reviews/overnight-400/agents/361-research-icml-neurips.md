# 361 — research-icml-neurips (ICML/NeurIPS 2025 math primitives)

## Headline
ICML 2025 (Vancouver, July) and NeurIPS 2025 (San Diego, December) crystallised four pure-math
fronts reality should track: orthogonalised/spectral-norm gradient methods (Muon family),
flow-matching↔Schrödinger-bridge↔OT unification, conformal prediction beyond exchangeability,
and Riemannian variance-reduced optimisation.

## Top papers (sorted by impact × fit)

### 1. Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data (ICML 2025)
Bernstein, Newhouse et al. Provides a complete characterisation of the implicit bias of
p-norm normalised steepest descent (NSD) and momentum steepest descent (NMD) under multiclass
cross-entropy. Spectral Descent and **Muon** (orthogonalised-momentum step via Newton–Schulz
polar factor) are shown to converge to spectral-norm max-margin classifiers, distinct from
Adam/AdamW (∞-norm) or SGD (Frobenius). The math is pure linear algebra: NS iteration
`X ← 1.5X − 0.5XX*X` polishes a matrix toward its orthogonal polar factor in 5 steps.
Reality fit: `optim` should expose Newton–Schulz polar factor as a primitive (linalg) and
add a steepest-descent-under-norm helper family. arXiv:2502.04404.

### 2. Towards Understanding of Orthogonalization in Muon (ICML 2025)
Liu et al. Decomposes Muon = SSD (stochastic spectral descent) + heavy-ball momentum + 5-step
Newton–Schulz. Proves improved nonconvex convergence rates and shows Muon optimises under
spectral-norm trust-region constraints. Companion paper "MUON Optimizes Under Spectral Norm
Constraints" (OPT-ML 2025) makes this rigorous. Reality fit: `linalg` already has QR/SVD;
add NS polar iteration with documented quintic convergence and tight numeric pin.

### 3. SOAP: Improving and Stabilizing Shampoo using Adam (ICLR/ICML 2025 line)
Vyas, Janson et al. Establishes formal equivalence between Shampoo (1/2 power) and Adafactor
run in Shampoo's preconditioner eigenbasis. Practical algorithm: rotate gradients into
Kronecker-factor eigenbasis, run Adam there. ~20% wall-clock and iteration win over Shampoo.
Pure-math primitive: Kronecker-factored preconditioner with periodic eigendecomposition.
Reality fit: `optim` Shampoo skeleton + `linalg` Kronecker-product helpers and stable
symmetric eigensolver (already partially via Cholesky).

### 4. MARS: Unleashing Variance Reduction for Training Large Models (ICML 2025)
Yuan, Liu et al. A unified framework reconciling preconditioned gradient methods with
variance reduction via scaled stochastic recursive momentum (SRM). Drop-in instances on
top of AdamW, Lion, Shampoo. Math: scaled-recursive-momentum estimator g_t = ∇f(x_t) +
β(g_{t-1} − ∇f(x_{t-1})) with adaptive scaling. Reality fit: `optim` could ship a generic
SRM helper that wraps any base optimiser — pure recurrence, no allocs.

### 5. Why Diffusion Models Don't Memorize (NeurIPS 2025 Best Paper)
George, Abbe et al. Two-timescale analysis of score-matching dynamics. Early time t_gen
(good samples) scales linearly with N_train; memorisation onset t_mem stays roughly constant.
Implicit dynamical regularisation arises from gradient-flow geometry on the score manifold.
Reality fit: not a primitive but anchors a future `prob/diffusion` slot — needs score
function, OU forward process, reverse SDE Euler–Maruyama, and Tweedie's formula
(x̂_0 = x_t + σ²∇log p_t).

### 6. Gated Attention (NeurIPS 2025 Best Paper)
Qiu et al. Adds a sigmoid gate after scaled-dot-product attention; eliminates "attention
sinks" and stabilises long-context training. Math primitive: σ(W_g x) ⊙ Attention(Q,K,V).
Reality fit: marginal — out of scope for `reality`. Note for downstream `aicore`.

### 7. Schrödinger Bridge as Robustified Optimal Transport Flows (NeurIPS 2025)
Shows diffusion models implicitly solve a distributionally robust variant of the
Kantorovich dual. Concretely: SB = entropy-regularised OT with KL penalty around reference
Brownian motion; diffusion = SB with Gaussian endpoint. Provides closed-form drift
b(x,t) = ∇ log Φ(x,t) where Φ solves backward Kolmogorov. Reality fit: enables a clean
`prob/ot` slot — Sinkhorn iterations, Wasserstein-2, entropy-regularised OT solver.

### 8. Statistical Analysis of Sinkhorn Iterations for Two-Sample Schrödinger Bridge (NeurIPS 2025)
First sharp rates for Sinkhorn-bridge estimator. Establishes Õ(n^{-1/2}) sample complexity
in W_2 under sub-Gaussian marginals. Math content: matrix-balancing convergence (Sinkhorn-
Knopp), entropic regularisation, plug-in OT. Reality fit: Sinkhorn iteration is a 30-line
pure-math primitive (positive matrix scaling); add to `prob` or new `transport` slot.

### 9. Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein
Barycentres (NeurIPS 2025)
Howard et al. Iterative Markovian Fitting (IMF) for SB barycentres on tree-structured
cost graphs, with explicit recursion against the Iterative Proportional Fitting (IPF /
Sinkhorn) baseline. Pure-math primitive: free-support W_2 barycentre via fixed-point
iteration. Reality fit: reusable W_2 barycentre helper in `prob`/`transport`.

### 10. Conformal Prediction under Lévy–Prokhorov Distribution Shifts (NeurIPS 2025)
Provides finite-sample coverage bounds when test distribution is within a Lévy–Prokhorov
ball of calibration. Pure-math: LP metric d_LP(P,Q) = inf{ε : P(A) ≤ Q(A^ε)+ε ∀A}, plus
worst-case quantile inflation 1−α → 1−α+ε. Reality fit: `prob` already has empirical
quantile + Kolmogorov–Smirnov; add split conformal predictor (~50 LOC) and LP-shift
inflation as a pure-math helper.

### 11. Non-exchangeable Conformal Prediction with Optimal Transport (NeurIPS 2025)
Couples conformal calibration with OT-based reweighting under covariate shift. Computes
sample weights w_i = dQ/dP via entropic OT plan, then runs weighted-quantile conformal.
Reality fit: pairs with #8 (Sinkhorn) — same primitive answers two papers.

### 12. SERENA: Stochastic Recursive Variance-Reduced Gradient on Riemannian Manifolds
(ICML 2025)
Unifies R-SRVRG, R-SVRRM, R-Hybrid-SGD with a single recursion built on parallel transport
and exponential map. Math primitives required: Exp_x, Log_x, parallel transport Γ_{x→y}.
Reality fit: `geometry` already has quaternions/SDF — extend with explicit charts for the
Stiefel and Grassmann manifolds (St(n,p), Gr(n,p)) and add their Exp/Log maps. Cleanest
candidate for a new `riemann` package.

### 13. Preconditioned Riemannian Gradient Descent for Low-Multilinear-Rank Tensor
Completion (ICML 2025)
PRGD on the Tucker-rank manifold, with explicit metric tensor from leverage-score
weighting. Faster than vanilla RGD at same per-iter cost. Reality fit: tensor primitives
(Tucker / HOSVD) absent from `linalg`; this is the canonical use case to add them.

### 14. Fishers for Free: Squared-Gradient-Accumulator FIM Approximation (ICML 2025 Spotlight)
"Squisher" — recycles Adam's v_t (squared-gradient EMA) as a diagonal Fisher estimate, free
of extra forward/backward passes. Bias-corrected version matches empirical Fisher to within
3% on standard benchmarks. Pure-math primitive: diag(v_t)·(1−β_2^t)^{-1}. Reality fit:
trivially expressed as a one-liner in `optim`; document as "natural-gradient adapter".

### 15. Implicit Riemannian Optimism for Min-Max Problems (ICML 2025)
Inexact-implicit-update online learning on Hadamard (non-positively-curved) manifolds.
Required primitives: Riemannian distance, geodesic interpolation, sectional curvature.
Reality fit: defers to a future `riemann` package; not urgent.

### 16. Energy Matching: Unifying Flow Matching and Energy-Based Models (NeurIPS 2025)
Joint loss `L = L_FM + λ L_EBM`; closes the gap between deterministic flow matching and
likelihood-based EBMs by sharing the score parameterisation. Math primitive set:
conditional vector field u_t(x|x_1) on optimal-transport probability path, plus contrastive
divergence. Reality fit: would seed a `prob/flow` slot — but flow matching is a parameteric
recipe, not a pure-math primitive; defer.

### 17. Neural Stochastic Flows: Solver-Free SDE Inference (NeurIPS 2025)
Direct learning of the transition density p(x_t | x_s) of an SDE without numerical
integration. The math hook: closed-form Markov-bridge densities and Feynman–Kac integrals.
Reality fit: `chaos` already has ODE solvers (RK4) — adding Euler–Maruyama and
Milstein for SDEs is the obvious extension.

## Reality slot recommendations

Concrete additions, ranked by signal:

- **`linalg/polar.go`** — Newton–Schulz polar-factor iteration (5-step quintic). Used by
  Muon (#1, #2), spectral steepest descent, polar decomposition. Single clean primitive.
- **`linalg/kron.go`** — Kronecker product / vec / unvec helpers. Required for SOAP-style
  preconditioners (#3) and tensor algebra (#13).
- **`prob/transport/sinkhorn.go`** — Sinkhorn–Knopp matrix balancing for entropic OT.
  Drives #7, #8, #9, #11. ~30 LOC, well-conditioned, perfect for a golden file.
- **`prob/transport/wasserstein.go`** — closed-form W_2 between Gaussians + 1-D empirical
  W_p; barycentre fixed-point iteration.
- **`prob/conformal.go`** — split-conformal predictor with empirical quantile and weighted-
  quantile variant; LP-shift inflation helper. ~80 LOC, ties to existing `prob` quantile
  primitives. Drives #10, #11.
- **`chaos/sde.go`** — Euler–Maruyama and Milstein integrators. Pure recurrence over
  existing PRNG; complements RK4 in `chaos`. Anchors #17 and the OU forward process used
  by every diffusion paper.
- **`optim/srm.go`** — scaled-recursive-momentum wrapper (variance reduction). Generic
  decorator over any base optimiser; #4.
- **`optim/preconditioned/`** — Shampoo / SOAP scaffold (Kronecker-factored, periodic
  eigendecomposition). Larger work item; sequence after `linalg/kron.go`.
- **`linalg/randomised.go`** — randomised SVD (Halko–Martinsson–Tropp) and CountSketch.
  RandNLA is a recurring theme across #3, #13, #14 and broader 2024–2025 RandNLA literature
  (Frangella thesis, Stanford 2025); not an ICML paper itself but the missing infrastructure
  for several.
- **(optional) `riemann/`** — new package with Stiefel/Grassmann/Hadamard charts,
  Exp/Log/parallel-transport. Earns its keep only if `optim` grows manifold variants
  (#12, #13, #15). Defer until a consumer asks.

Distinguishing from slot 351 (broad 2025 survey): this slot is conference-locked; only
NeurIPS-25 and ICML-25 papers were considered, and the recommendations are indexed against
specific paper IDs above. Slot 351 should pull in non-conference 2025 work (RandNLA
monographs, Mahoney Berkeley course, Frangella thesis, Mamba-3 OpenReview).

## Sources
- ICML 2025 Awards page — https://icml.cc/virtual/2025/awards_detail
- NeurIPS 2025 Best Paper Awards announcement — https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/
- NeurIPS 2025 Awards page — https://neurips.cc/virtual/2025/awards_detail
- Muon implicit bias (ICML 2025) — https://icml.cc/virtual/2025/47634
- Muon orthogonalisation analysis (ICML 2025) — https://icml.cc/virtual/2025/47855
- MUON spectral norm constraints (OPT-ML 2025) — https://opt-ml.org/papers/2025/paper137.pdf
- SOAP — https://openreview.net/forum?id=IDxZhXrpNf and https://arxiv.org/abs/2409.11321
- MARS (ICML 2025) — https://icml.cc/virtual/2025/poster/45479
- SB as Robustified OT Flows (NeurIPS 2025) — https://neurips.cc/virtual/2025/loc/san-diego/132762
- SB Matching Tree-Structured (NeurIPS 2025) — https://neurips.cc/virtual/2025/poster/119196
- Sinkhorn Statistical Analysis (NeurIPS 2025) — https://neurips.cc/virtual/2025/poster/119183
- Conformal under LP shift (NeurIPS 2025) — https://neurips.cc/virtual/2025/loc/san-diego/poster/120230
- Non-exchangeable Conformal + OT (NeurIPS 2025) — https://neurips.cc/virtual/2025/loc/san-diego/poster/118951
- SERENA Riemannian VR (ICML 2025) — https://icml.cc/virtual/2025/poster/46244
- Preconditioned RGD tensor completion (ICML 2025) — https://icml.cc/virtual/2025/poster/44012
- Fishers for Free (ICML 2025) — https://icml.cc/virtual/2025/poster/44175 and https://arxiv.org/abs/2507.18807
- Implicit Riemannian Optimism — https://openreview.net/forum?id=Mz4J6GRZso
- Energy Matching (NeurIPS 2025) — https://neurips.cc/virtual/2025/poster/117591
- Neural Stochastic Flows (NeurIPS 2025) — https://neurips.cc/virtual/2025/poster/118182
- ICML 2025 outstanding papers digest — https://gonzoml.substack.com/p/the-icml-2025-outstanding-papers
- NeurIPS 2025 best papers digest — https://www.theneuron.ai/explainer-articles/the-best-papers-at-neurips-2025-explained/
