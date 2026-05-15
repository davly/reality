# 364 — research-arxiv-math-na (recent math.NA preprints)

## Headline
arXiv math.NA 2025-2026 churn focuses on randomized NLA error bounds, mixed-precision GMRES, structure-preserving / pseudo-symplectic neural integrators, entropy-stable DG, and PINN/DeepONet rigorous error theory; concrete primitives are usable in `linalg`, `signal`, `chaos`, `calculus` packages.

## Top papers

### 1. Palitta & Portaro — Improved error bounds for randomized SVD (arXiv:2408.04503v3, May 2025)
Tightens the classical Halko-Martinsson-Tropp bound for the gap-revealing case: when singular values exhibit pronounced clusters/gaps, the new bound captures the rate at which Range(A) is recovered far better than Frobenius-norm worst-case. Authors also propose a row-space-augmented randomized SVD plus a sub-sampling variant that approaches HMT accuracy at lower flop count. Direct relevance to `linalg.PCA` and any randomized truncated SVD primitive: gap-aware bounds let users pick oversampling parameter `p` deterministically rather than heuristically. Pairs with the broader 2512.05286 survey of randomized matrix/tensor decomp (Tucker, ID, CUR, t-SVD).

### 2. Connecting Randomized Iterative Methods with Krylov Subspaces (arXiv:2505.20602, May 2025)
Bridges block-randomized Kaczmarz / coordinate-descent variants with classical Krylov theory, proving linear convergence in expectation with a contraction factor that strictly decreases as the look-back window grows; finite termination at window=rank(A) absent round-off. Significance: gives a unified Krylov-style spectral analysis for a family of methods that had previously been studied with martingale arguments only. Useful guidance if `reality` ever ships a sparse Kaczmarz/CGLS solver — sets expected convergence behaviour rigorously.

### 3. AK-SLRL: Adaptive Krylov restart via single-life RL (arXiv:2502.00227, Feb 2025)
Trains a one-shot RL controller to choose GMRES restart `m` per-problem on the fly; reports 5-30× wall-clock improvement over fixed-restart on systems requiring >1k iterations. Outside the zero-deps remit of `reality` but a clear data point that adaptive restart selection is an open optimization frontier. A purely heuristic adaptive-`m` rule (no RL) is plausible for an in-house GMRES.

### 4. Mixed-precision sketching for least-squares + GMRES-IR (arXiv:2410.06319, late 2024 / 2025 finite-precision analysis)
Performs full backward-error analysis of a mixed-precision sketched-QR preconditioner inside GMRES-based iterative refinement. Gives explicit precision-budget formulas: how low you can drop the sketch precision while still guaranteeing the preconditioner reduces κ to a prescribed level. Companion: arXiv:2503.03456v2 (mixed-precision Sylvester solver). Acta Numerica 2024 mixed-precision survey + arXiv:2412.19322 cover the ecosystem. For `reality`, where everything is float64, these are background — but the formulas inform what is safe in float32 fast-paths.

### 5. Two new ERK classes for stiff/oscillatory ODEs — MVERK and SVERK (arXiv:2210.00685, updated 2025)
Explicit modified- and simplified-version exponential Runge-Kutta methods derived from new order-condition trees that avoid expensive φ-function evaluations on full Jacobians. Plus arXiv:2506.04416 ETD-RK analysis with sharper steady-state preservation bounds, and arXiv:2510.21381 extending exponential integrators to non-homogeneous BCs via boundary-data smooth extension. Survey: arXiv:2507.04024 "Exploring Exponential Runge-Kutta Methods". Concrete fit for `chaos` and `calculus`: a high-order exponential integrator alongside RK4 covers stiff-but-oscillatory regimes (e.g., Van der Pol at high μ).

### 6. Explicit symplectic integrators for nonseparable Hamiltonians (arXiv:2504.12567, April 2025)
Closes a long-standing gap: classical leapfrog / Stormer-Verlet require H = T(p) + V(q) separability. Authors propose an extended-phase-space splitting plus a corrector that yields explicit symplectic schemes for general H(p,q) without implicit solves. Related: arXiv:2506.07072 efficient structure-preserving exponential integrator (energy-preserving via averaged-vector-field × exponential), arXiv:2502.20212 Pseudo-Symplectic Neural Network with explicit pseudo-symplectic core, arXiv:2504.01307 multi-invariant-preserving integrator for multi-symplectic PDE forms. Direct relevance: `chaos` Lorenz/Hamiltonian flows would benefit from a true symplectic option (currently RK4 drifts energy).

### 7. Non-oscillatory entropy-stable DG (arXiv:2410.16729, accepted 2025) + entropy-stable DGSEM on curvilinear hybrids (arXiv:2507.04334, July 2025)
NOES-DG: artificial viscosity controlled directly by entropy production, paired with integrating-factor SSP-RK time stepping; suppresses Gibbs without flux limiters. Curvilinear-hybrid DGSEM (Chan/Gassner et al.) extends entropy-conservative SBP operators to mixed tet/prism/hex meshes — the missing piece for general unstructured high-order conservation-law solvers. Plus arXiv:2603.18978 affordable high-order entropy-stable methods for nonconservative systems. `reality` has no PDE solver today, but if a `pde` package emerges these are SOTA reference designs.

### 8. PINN convergence/coercivity theory (arXiv:2506.13554 + arXiv:2406.09217)
Two complementary rigorous frameworks. Doumèche-Boyer-Biau follow-ups (arXiv:2305.01240) show classical PINN training is overfitting-prone and ridge-regularized empirical risk is risk-consistent for linear and nonlinear PDE residuals. arXiv:2406.09217 gives a "consistent PINN" loss + sharp error control for elliptic PDEs proving optimal recovery in restricted NN spaces. arXiv:2506.13554 unifies via operator coercivity: residual minimization in Sobolev norms ⇒ convergence in energy and uniform norms under mild regularity. Out-of-scope for `reality` (no NN), but feeds reality slot policy: any future NN-PDE wrapper should ship coercivity-checked loss.

### 9. DeepONet error decomposition + scaling laws (arXiv:2602.21910 + arXiv:2410.00357)
Branch-trunk-mode error decomposition: across a battery of PDE benchmarks the branch (basis-coefficient learner) dominates approximation error at large inner dimension, while trunk learns a near-sufficient basis quickly. Scaling-law paper derives explicit relationships between approximation/generalization error vs network size and training-set size, and identifies regimes where DeepONets beat the curse of dimensionality (algebraic in 1/ε). Useful if any future `reality` slot ships an operator-learning interface — it dictates where to invest training compute.

### 10. Geometrically-informed AMG for high-order FE systems (arXiv:2512.15121, Dec 2025)
Hybridizes geometric and algebraic multigrid: at the top of the hierarchy uses geometry-aware p-coarsening (degree reduction) before falling back to pure-algebraic h-coarsening. Standard black-box AMG plateaus on high-order discretizations; GIAMG restores grid-independent iteration counts. Plus arXiv:2412.08186 genetic-programming auto-tuning of AMG smoothers. Background for any future iterative-solver layer; not directly portable into the dependency-free `reality` because production AMG needs sparse-matrix infrastructure beyond stdlib.

### 11. Higher-order quadrature transport for QMC + sparse grids (transport-map QMC; Springer Stat Comput 2025; arXiv:2308.10081 + 2025 extensions)
Pushes lattice/sparse-grid points through transport maps so transformed samples inherit better-than-N^{-1/2} convergence on mixture distributions and non-product target measures. Crucial for high-d Bayesian integrals. Companion empirical work shows QMC achieves the optimal convergence rate even on non-smooth NN integrands. Pairs with `prob` package — currently relies on plain MC; QMC + Sobol sequences with regularity-aware weights are an upgrade path with no extra deps.

### 12. Enhanced gradient-recovery a-posteriori estimator (arXiv:2503.19701, March 2025) + Prager-Synge non-conforming bounds (arXiv:2506.23381)
arXiv:2503.19701 splits the recovery estimator into two terms — direct vs post-processed gradient difference and recovered-gradient residual — and proves both reliability and efficiency under standard regularity, fixing known counterexamples to plain ZZ-style estimators on anisotropic meshes. arXiv:2506.23381 gives stabilization-free a posteriori bounds for non-conforming FE via a new Prager-Synge reformulation. Background for any future `fem` package; reality has none, so this is informational.

### 13. Connecting points: parametric / data-driven SymCLaw (arXiv:2601.21080, Jan 2026)
Late-2025 follow-up that introduces a parametric family of hyperbolic conservation laws preserving conservation, entropy stability, and hyperbolicity *by construction* — closing a gap where prior data-driven approaches lost one of these guarantees. arXiv:2507.01795 NESCFN (neural entropy-stable conservative flux network) is the NN realization. Together they define a template: encode invariants in the parameterization rather than the loss.

## Reality slot recommendations

- **`linalg`**: gap-aware error bound from Palitta-Portaro (arXiv:2408.04503) is implementable in pure Go alongside the existing PCA/randomized-SVD primitive — adds a `RandomizedSVDGapBound(σ, k, p, q)` that returns the predicted approximation error before running the algorithm.
- **`linalg`**: investigate adding randomized block-Krylov for `f(A)b` (arXiv:2502.01888) as low-rank function-of-matrix support; pure-stdlib doable.
- **`chaos`**: ship a true symplectic integrator (Stormer-Verlet for separable; arXiv:2504.12567 explicit nonseparable variant for general H). `RK4` will keep being the workhorse but Hamiltonian flows want energy preservation.
- **`chaos` / `calculus`**: add an exponential time-differencing RK (ETD-RK4 of Cox-Matthews + Krogstad updates) alongside RK4. ERK survey arXiv:2507.04024 documents the accuracy/cost trade for stiff-but-mildly-nonlinear ODEs that RK4 handles poorly.
- **`prob`**: replace plain Monte Carlo `IntegrateMC` with optional Sobol/Halton QMC + transport-map weighting; uses no external deps. Reference: arXiv:2308.10081 + 2025 extensions.
- **`signal`**: arXiv:2502.01888 randomized block-Krylov also speeds spectral-density estimation via low-rank approximations of large covariance matrices.
- Out of scope but worth a `CONTEXT.md` note: PDE/FE/AMG/PINN slots remain unaddressed and the literature is moving fast — defer until aicore demands.

## Sources
- arXiv:2408.04503v3 — Palitta & Portaro, improved randomized SVD bounds. https://arxiv.org/abs/2408.04503
- arXiv:2512.05286 — Randomized algorithms for low-rank matrix and tensor decompositions (survey). https://arxiv.org/abs/2512.05286
- arXiv:2505.20602 — Connecting randomized iterative methods with Krylov subspaces. https://arxiv.org/abs/2505.20602
- arXiv:2502.00227 — AK-SLRL adaptive Krylov via RL. https://arxiv.org/abs/2502.00227
- arXiv:2410.06319 — Mixed-precision sketching for LS + GMRES-IR. https://arxiv.org/abs/2410.06319
- arXiv:2412.19322 — Mixed-precision numerics in scientific apps survey. https://arxiv.org/html/2412.19322v1
- arXiv:2503.03456 — Mixed-precision Sylvester. https://www.arxiv.org/pdf/2503.03456v2
- arXiv:2210.00685 — MVERK / SVERK exponential RK classes. https://arxiv.org/html/2210.00685
- arXiv:2507.04024 — Exploring exponential RK methods (survey). https://arxiv.org/html/2507.04024v1
- arXiv:2506.04416 — ETD-RK analysis. https://arxiv.org/pdf/2506.04416
- arXiv:2510.21381 — Exponential integrators for parabolic PDEs with non-homogeneous BCs. https://arxiv.org/abs/2510.21381
- arXiv:2504.12567 — Explicit symplectic integrators for nonseparable Hamiltonians. https://arxiv.org/pdf/2504.12567
- arXiv:2506.07072 — Structure-preserving exponential integrator. https://arxiv.org/html/2506.07072
- arXiv:2502.20212 — Pseudo-Symplectic Neural Network. https://arxiv.org/abs/2502.20212
- arXiv:2504.01307 — Semi-analytical multi-invariant-preserving integrator. https://arxiv.org/pdf/2504.01307
- arXiv:2410.16729 — Non-oscillatory entropy-stable DG schemes. https://arxiv.org/abs/2410.16729
- arXiv:2507.04334 — Entropy stable high-order DGSEM on curvilinear hybrid meshes. https://arxiv.org/html/2507.04334v1
- arXiv:2603.18978 — Affordable HO entropy-stable / well-balanced methods. https://arxiv.org/html/2603.18978
- arXiv:2406.09217 — Convergence and error control of consistent PINNs for elliptic PDEs. https://arxiv.org/abs/2406.09217
- arXiv:2506.13554 — Coercive operator analysis of PINNs. https://arxiv.org/html/2506.13554
- arXiv:2305.01240 — On the convergence of PINNs (Doumèche-Biau-Boyer). https://arxiv.org/abs/2305.01240
- arXiv:2602.21910 — DeepONet error = branch + trunk + mode decomposition. https://arxiv.org/html/2602.21910
- arXiv:2410.00357 — Neural scaling laws for DeepONet / deep ReLU. https://arxiv.org/html/2410.00357
- arXiv:2512.15121 — Geometrically-informed AMG for high-order FE. https://arxiv.org/abs/2512.15121
- arXiv:2412.08186 — GP-tuned AMG preconditioner design. https://arxiv.org/html/2412.08186
- arXiv:2503.19701 — Enhanced gradient-recovery a posteriori estimator. https://arxiv.org/abs/2503.19701
- arXiv:2506.23381 — Stabilization-free Prager-Synge a posteriori bounds (non-conforming FE). https://arxiv.org/html/2506.23381
- arXiv:2502.01888 — Randomized block-Krylov for low-rank f(A) approximation. https://arxiv.org/abs/2502.01888
- arXiv:2308.10081 + 2025 follow-ups — Transporting higher-order quadrature rules (QMC + sparse grids). https://arxiv.org/abs/2308.10081
- arXiv:2601.21080 — Parametric hyperbolic conservation laws (SymCLaw). https://arxiv.org/pdf/2601.21080
- arXiv:2507.01795 — Neural entropy-stable conservative flux NN. https://arxiv.org/html/2507.01795
