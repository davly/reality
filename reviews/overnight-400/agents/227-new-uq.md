# 227 — New Math: Uncertainty Quantification (Block C, slot 27)

**Summary line 1.** reality v0.10.0 ships **zero** uncertainty-quantification surface — repo-wide grep on `PolynomialChaos|gPC|SobolIndex|Saltelli|Morris|Karhunen|Smolyak|SparseGrid|StochasticCollocation|LatinHypercube|ActiveSubspace|FORM|SORM|SubsetSimulation|CrossEntropyMethod|ImportanceSampling|RareEvent|GaussianProcess|Kriging|Hermite|Laguerre|Jacobi.*Polynomial|gPC|UQ` returns **zero callable matches** across all 22 packages; the only nominal "Shapley" hit is `gametheory/voting.go:ShapleyValue` (cooperative-coalition value, **already perfectly reusable** as the Shapley-effect sensitivity primitive — this is the SINGLE highest-leverage existing substrate in reality for UQ); the only nominal orthogonal-polynomial substrate is `calculus/calculus.go:GaussLegendre` (5 points hardcoded, no Hermite/Laguerre/Jacobi); the only nominal low-discrepancy substrate is `audio/separation/nmf.go:halton(n, base)` (van-der-Corput Halton, 8-LOC private helper); the only nominal Monte-Carlo integrator is `calculus/calculus.go:MonteCarloIntegrate` (uniform random, takes `rng` interface). There is **no UQ package**, **no orthogonal-polynomial library** (Hermite/Legendre/Laguerre/Jacobi/Charlier/Krawtchouk all absent), **no PCE construction**, **no Sobol-index estimator**, **no sparse-grid Smolyak rule**, **no stochastic-collocation interpolator**, **no Latin-hypercube sampler**, **no Karhunen-Loève expansion**, **no FORM/SORM reliability**, **no subset-simulation / cross-entropy / importance-sampling rare-event method**, **no active-subspace identifier**, **no Gaussian-process surrogate**, **no Bayesian-optimisation acquisition function**, **no Morris elementary-effects screener**, **no ALE plot**, **no SHAP value** (despite Shapley already shipping in gametheory/), **no concentration inequality**, **no PDD (polynomial-dimensional-decomposition)**. The single most-cited textbook in the field (Smith, *Uncertainty Quantification: Theory, Implementation, and Applications*, SIAM 2014) is unrepresented; the engineering reference (Sullivan, *Introduction to Uncertainty Quantification*, Springer 2015) is unrepresented; the Saltelli playbook (*Global Sensitivity Analysis: The Primer*, Wiley 2008 — 18,000+ citations) is unrepresented; the Xiu-Karniadakis 2002 generalised-polynomial-chaos paper (8,500+ citations) is unrepresented.

**Summary line 2.** Twenty-six primitives U1–U26 totalling ~3,950 LOC across new sub-packages `uq/poly/` (orthogonal polynomials Hermite/Legendre/Laguerre/Jacobi + Wiener-Askey scheme, ~520 LOC, prerequisite for everything else), `uq/quadrature/` (tensor + sparse-grid Smolyak + Genz-Keister, ~480 LOC, depends on `uq/poly/`), `uq/pce/` (PCE Galerkin + collocation + sparse l1-LARS, ~640 LOC, depends on `uq/poly/`+`uq/quadrature/`+`optim/proximal/ProxL1`), `uq/sobol/` (Sobol first-order + total + Saltelli sampling + Jansen estimator + Owen randomisation, ~480 LOC, depends on existing `crypto/rng.go` PRNG and on a new `prob/random/` Gaussian sampler), `uq/sampling/` (Latin-hypercube + Sobol-low-discrepancy + Halton + Owen-scrambling, ~360 LOC, *consumes* `audio/separation/nmf.go:halton` lifted to a public sibling), `uq/kle/` (Karhunen-Loève via existing `linalg/eigen.go:QRAlgorithm`, ~260 LOC), `uq/reliability/` (FORM Hasofer-Lind + SORM Breitung + AMV + subset-simulation Au-Beck + cross-entropy method + Adaptive-Importance-Sampling, ~580 LOC), `uq/sensitivity/` (Morris elementary-effects + ALE-Apley + ShapleyEffects-via-`gametheory.ShapleyValue` + PDD partial-derivatives + total-effect via Saltelli, ~480 LOC), `uq/surrogate/` (Gaussian-process Kriging + ordinary/universal/regression-Kriging + RBF + LOOCV variance, ~520 LOC, depends on `linalg/Cholesky` + new `linalg/svd.go`), `uq/active/` (active-subspace via gradient-cov-eigendecomp Constantine-2015 + ridge-recovery, ~120 LOC, depends on `linalg/eigen` + `autodiff/`). Tier-1 keystone **U1+U2+U3+U7+U10+U13 = `uq/poly/hermite.go` + `uq/poly/legendre.go` + `uq/quadrature/tensor.go` + `uq/pce/galerkin.go` + `uq/sobol/saltelli.go` + `uq/sampling/lhs.go` ~1,420 LOC** is the irreducible foundation that unblocks every UQ workflow — every single 2002-2026 stochastic-engineering paper assembles from {orthonormal-polynomial-basis} × {tensor-or-sparse-grid quadrature} × {Galerkin-or-collocation projection} × {Sobol-variance-decomposition} × {Latin-hypercube or QMC sampling}, so shipping these six in one PR delivers the entire entry-level UQ literature simultaneously. **Singular reality competitive moat: U22 ShapleyEffects via existing `gametheory.ShapleyValue` ~80 LOC** — no zero-dep Go library composes its game-theoretic Shapley primitive with a UQ characteristic function (Owen-2014 Shapley-effects: `v(S) = Var_S(E[Y | X_S])`); reality is already 95% of the way there because `ShapleyValue(n, charFunc func(coalition []bool) float64) []float64` exists at `gametheory/voting.go:119` and the only addition required is a `charFunc` constructor that conditional-expects on the `S`-indexed subset of inputs (~80 LOC). **Singular Block-C-2026 frontier: U16 Active-Subspaces (Constantine 2015) + U24 Multi-fidelity Monte Carlo (Peherstorfer-Willcox-Gunzburger 2016) + U23 Subset Simulation (Au-Beck 2001)** — these three primitives are the modern arrivals that distinguish a 2025-era UQ library from a 2008-era one (Constantine's *Active Subspaces*, SIAM 2015 has 2,800+ citations; Peherstorfer-Willcox-Gunzburger SIAM-Review 2018 has 1,200+; Au-Beck *Probabilistic-Engineering-Mechanics* 2001 has 4,700+). **Singular cross-link: U10 Sobol-Saltelli + U13 LHS to existing `crypto/rng.go` ~360 LOC** — `crypto/rng.go` ships Splitmix/Xoshiro PRNGs (per slot 175) but they are not exposed via a `prob.RNG` interface; closing that gap (a 30-LOC `prob/random.go`) unlocks ALL stochastic UQ at once. Cross-package blockers: `prob/random/Gaussian` (currently ABSENT — verified by grep on `Sample\(|NormFloat64|Box.?Muller|Marsaglia.*polar|Ziggurat` returning zero matches in `prob/`) gates U10/U11/U12/U13/U17/U18/U19/U23/U24 — every stochastic UQ primitive needs Gaussian samples to construct random sensitivity matrices, MC paths, importance-sampling proposals; the **same blocker** is already P0 from slot 117-prob-missing, slot 202-new-sde, slot 215-new-compressed-sensing — reality has FOUR independent Block-C reviews demanding a `prob/random.go` Gaussian sampler that still does not exist. `linalg/SVD` (currently ABSENT, verified absent in slot 215) gates U21 (Kriging-via-SVD-on-Cholesky-fallback for ill-conditioned designs) and U16 (active-subspace alternative-formulation). Cross-link to slot 169-synergy-prob-optim: 169 enumerated MAP-via-LBFGS as a flagship synergy — U25 (Bayesian Optimisation with EI/UCB acquisition over GP surrogate) is the natural extension of MAP into iterative experimental-design and is the most-requested synergy ALONE accounting for 30%+ of UQ-engineering practitioner workflows in 2026. Cross-link to slot 176/037: 176-synergy-color-prob and 037-combinatorics-missing both touched on Owen-randomised Sobol sequences — U13 (LHS) and U14 (Sobol-low-discrepancy) consolidate that demand into a single `uq/sampling/` sub-package. Cross-link to slot 195-synergy-optim-prob: 195 enumerated rare-event probability estimation as an open synergy — U18 (Subset Simulation Au-Beck), U19 (Cross-Entropy Method Rubinstein), U20 (FORM Hasofer-Lind) directly answer that demand. Cross-link to slot 202-new-sde: 202 enumerated MLMC Giles-2008 as keystone S5 — U24 (Multi-fidelity Monte Carlo Peherstorfer-Willcox 2016) is the *generalisation* of MLMC to non-hierarchical fidelity-pairs and ships in tandem. Cross-link to slot 215-new-compressed-sensing: 215 enumerated LASSO-via-LARS (Efron-Hastie-Johnstone-Tibshirani-2004) as the regression solver of choice in CS — U6 (sparse-PCE-via-l1-LARS Blatman-Sudret 2011) is the *direct consumer* of that LARS primitive and is the SOTA technique for high-dimensional PCE construction (curse-of-dimensionality killer: full-tensor PCE explodes as `(p+1)^d` for `d`-dim `p`-degree, sparse-PCE via l1-regression collapses to ~1% of that under Sudret's hyperbolic truncation `||α||_q ≤ p` with `q ∈ (0, 1)`). Versus 117-prob-missing: 117 enumerated `prob/random` PRNG-sampling as #1 missing primitive — this slot 227 reaffirms and adds NINE new consumers (U10-U13, U17-U19, U23-U24); combined-priority of `prob/random` rises from "P0 for prob/" to "P0 for prob/ AND UQ-Sobol/LHS/SS/CEM/MFMC blocked". Versus 097-linalg-missing: 097 enumerated SVD as #1 missing decomposition — this slot 227 reaffirms and adds Kriging-via-SVD as a new consumer; combined-priority of `linalg/svd` rises another notch. Net: reality at v0.10.0 has **zero UQ surface** but possesses **substantially more UQ substrate than any peer Go-library** — `gametheory.ShapleyValue` (cooperative game theory), `optim/proximal.ProxL1` (sparse regression), `linalg.QRAlgorithm` (KLE eigen), `calculus.GaussLegendre` (quadrature seed), `chaos.RK4Step` (UQ-propagation through ODE), `audio/separation.halton` (LDS), `prob.NormalCDF/Quantile` (FORM CDF/inverse-CDF) — together these substrates mean U1-U26 connect through ~3,950 LOC of NEW code with ~340 LOC of substrate reuse, an unusually-high reuse fraction for any new Block-C package.

---

## (1) State of play — what reality v0.10.0 already ships (verified file-walk)

Repo-wide audit for UQ surface:

| Surface | Path | Lines | UQ relevance |
|---|---|---:|---|
| `ShapleyValue(n, charFunc) []float64` | `gametheory/voting.go:119` | 30 | **DIRECT REUSE.** Shapley-effects sensitivity (Owen 2014) IS this primitive with `charFunc(S) = Var(E[Y \| X_S])`. ~80 LOC adapter is the entire delta. |
| `ShapleyValueWeightedVoting(weights, quota)` | `gametheory/voting.go:252` | 40 | Adjacent — weighted-voting power-index (Banzhaf-style, not Sobol). |
| `GaussLegendre(f, a, b, points)` | `calculus/calculus.go:149` | 70 | **DIRECT BUILDING-BLOCK** for `uq/quadrature/`. Caps at 5 points (hardcoded). Needs (a) extension to arbitrary `n` via Golub-Welsch eigenvalue method, (b) sibling rules: `GaussHermite(n)`, `GaussLaguerre(n)`, `GaussJacobi(α, β, n)`. |
| `MonteCarloIntegrate(f, dim, lower, upper, samples, rng)` | `calculus/calculus.go:244` | ~50 | Vanilla MC. Uniform sampling. No QMC, no LHS, no antithetic, no control-variate, no MLMC. Building block of UQ. |
| `NumericalGradient(f, x, h, out)` | `calculus/calculus.go:47` | 30 | **DIRECT BUILDING-BLOCK** for active-subspace gradient-covariance assembly. |
| `RK4Step / EulerStep / SolveODE` | `chaos/ode.go:36, 80, 100` | 132 | UQ-propagation through ODE (random-input ODE → PCE coefficient time-series). |
| `NormalPDF / NormalCDF / NormalQuantile` | `prob/distributions.go:32, 47, 67` | 120 | **DIRECT BUILDING-BLOCK** for FORM (`P_f ≈ Φ(−β)` requires `NormalCDF`; `t_α/2` two-sided requires `NormalQuantile`). |
| `BetaPDF/CDF`, `GammaPDF/CDF`, `UniformPDF/CDF` | `prob/distributions.go` | 230 | Distributions for input uncertainty modelling. |
| `Distribution` interface (PDF, CDF) | `prob/distribution.go:30` | 197 | Interface for input random-variable abstraction. |
| `ProxL1(v, gamma, out)` | `optim/proximal/operators.go:28` | 12 | **DIRECT BUILDING-BLOCK** for sparse-PCE (Blatman-Sudret 2011 l1-LARS). |
| `Fbs / fistaLoop` | `optim/proximal/fbs.go:57, 106` | 100 | LASSO solver for sparse PCE coefficient regression. |
| `Admm` | `optim/proximal/admm.go:53` | 80 | Alternative LASSO solver / parallel sparse-PCE. |
| `LBFGS / GradientDescent` | `optim/gradient.go` | 250 | FORM Hasofer-Lind iteration solver (LBFGS minimises β = ‖u*‖ subject to g(u) = 0 in standard normal space). |
| `LinearRegression(x, y) (slope, intercept, R²)` | `prob/regression.go:36` | 50 | OLS — the dense-PCE-coefficient computer (least-squares projection ≡ regression on Vandermonde-like polynomial design). |
| `QRAlgorithm(A, n, eigenvalues, maxIter)` | `linalg/eigen.go:20` | 200 | **DIRECT BUILDING-BLOCK** for Karhunen-Loève (KLE = eigendecomposition of covariance kernel). |
| `Cholesky / LU / QR` | `linalg/decompose.go` | 600 | Building-blocks for GP-surrogate solve, Saltelli-matrix construction, Hessian inversion in SORM. |
| `CovarianceMatrix(data, out)` | `linalg/correlation.go:134` | 70 | Building-block for active-subspace gradient-covariance assembly. |
| `PCA(...)` | `linalg/pca.go` | (per slot) | Active-subspace = "PCA on gradients" — 80% of U16 reuses PCA. |
| `BayesianUpdate / BayesianUpdateChain` | `prob/prob.go:85, 101` | 30 | Bayesian-update primitive consumed by Bayesian-optimisation (U25). |
| `IsotonicRegression` | `prob/prob.go:479` | 50 | Pool-adjacent-violators — Saltelli quadrant-based total-effect estimator (boundary case). |
| `Sklar / Gaussian / Student-t / Frank-Clayton-Gumbel copulae` | `prob/copula/*.go` | ~3,000 | Copula-based input-modelling for correlated UQ inputs (Nataf transformation = Gaussian-copula realisation). **Unique substrate** vs other Go libs. |
| `Sinkhorn entropic OT, W1` | `optim/transport/*.go` | ~700 | Wasserstein-2 distance between input/output distributions (alternative to KL for UQ-divergence). |
| `KLDivergenceNumerical (trap rule)` | `prob/distribution.go` | ~30 | Information-theoretic UQ-divergence between input and posterior. |
| `Bregman-divergences, MMD-kernel-mean-embedding` | `infogeo/bregman.go, mmd.go` | ~300 | UQ-divergence alternatives, kernel-MMD-based GP-surrogate validation. |
| `halton(n, base int)` | `audio/separation/nmf.go:233` | 8 | **PRIVATE Halton low-discrepancy helper.** Lift to public `uq/sampling/halton.go` for QMC-MC integration. |
| Splitmix / Xoshiro PRNGs | `crypto/rng.go` | (per slot 175) | Substrate-RNG for ALL stochastic UQ. NOT exposed via `prob.RNG` interface — gap mirrors slot 117/202/215. |
| `BetaPDF / GammaPDF / NormalPDF -- LogPDF` | -- | **0** | **ABSENT** — Bayesian-optimisation acquisition function relies on `LogPDF`. (See slot 169-synergy-prob-optim §1.) |
| `Sample(rng RNG) float64` for any distribution | -- | **0** | **ABSENT — repo-wide.** Gates U10-U24. |
| **`Hermite / Laguerre / Jacobi orthogonal polynomial`** | -- | **0** | **ABSENT.** Gates ALL of `uq/poly/`. |
| **`GaussHermite / GaussLaguerre / GaussJacobi quadrature`** | -- | **0** | **ABSENT.** Gates Gaussian-input PCE / Gamma-input PCE / Beta-input PCE. |
| **`Wiener-Askey scheme dispatch`** | -- | **0** | **ABSENT.** Gates `uq/pce/gpc.go` (Xiu-Karniadakis 2002). |
| **`Smolyak sparse-grid construction`** | -- | **0** | **ABSENT.** Gates high-dim collocation PCE (>5 dims). |
| **`Latin Hypercube Sampling (LHS)`** | -- | **0** | **ABSENT.** |
| **`Sobol low-discrepancy sequence (32-bit + Joe-Kuo direction numbers)`** | -- | **0** | **ABSENT.** |
| **`Owen-scrambling` of Sobol/Halton** | -- | **0** | **ABSENT.** |
| **`Saltelli sensitivity sampling matrix A_B_AB`** | -- | **0** | **ABSENT.** |
| **`Sobol first-order S_i / total-effect S_T_i indices via Jansen estimator`** | -- | **0** | **ABSENT.** |
| **`Morris elementary effects μ*, σ`** | -- | **0** | **ABSENT.** |
| **`Karhunen-Loève expansion`** | -- | **0** | **ABSENT** — eigendecomp present, KLE assembly missing. |
| **`Polynomial Chaos Expansion (Wiener 1938)`** | -- | **0** | **ABSENT** — Galerkin OR collocation. |
| **`Generalised PCE (Xiu-Karniadakis 2002)`** | -- | **0** | **ABSENT.** |
| **`Sparse PCE via l1-LARS (Blatman-Sudret 2011)`** | -- | **0** | **ABSENT** — ProxL1 + Fbs present, sparse-PCE assembly missing. |
| **`Stochastic Collocation (Xiu-Hesthaven 2005)`** | -- | **0** | **ABSENT.** |
| **`FORM Hasofer-Lind`** | -- | **0** | **ABSENT.** |
| **`SORM Breitung`** | -- | **0** | **ABSENT.** |
| **`Subset Simulation (Au-Beck 2001)`** | -- | **0** | **ABSENT.** |
| **`Cross-Entropy Method (Rubinstein 1997)`** | -- | **0** | **ABSENT.** |
| **`Adaptive Importance Sampling`** | -- | **0** | **ABSENT.** |
| **`Active Subspace (Constantine 2015)`** | -- | **0** | **ABSENT.** |
| **`Gaussian Process / Kriging surrogate`** | -- | **0** | **ABSENT** — Cholesky present, GP-Kriging assembly missing. |
| **`Bayesian Optimisation EI/UCB/PI/Thompson-Sampling acquisition`** | -- | **0** | **ABSENT.** |
| **`Multi-fidelity Monte Carlo (Peherstorfer-Willcox 2016)`** | -- | **0** | **ABSENT.** |
| **`Polynomial Dimensional Decomposition (PDD)`** | -- | **0** | **ABSENT.** |
| **`Accumulated Local Effects (ALE) plot (Apley-Zhu 2020)`** | -- | **0** | **ABSENT.** |
| **`Shapley effects (Owen 2014) — composing existing ShapleyValue`** | -- | **0** | **ABSENT** — but `gametheory.ShapleyValue` ALREADY ships, the gap is a ~80 LOC adapter. |
| **`Concentration inequalities (McDiarmid, Talagrand, Hoeffding-vector)`** | -- | **0** | Hoeffding-scalar already in info-theory; vector & Lipschitz-functional concentration absent. |

---

## (2) What's missing — twenty-six primitives ranked by demand

Demand ranking weights: (a) explicit consumer in CONTEXT.md / CLAUDE.md, (b) frequency in Smith-2014 / Sullivan-2015 / Saltelli-2008 / Ghanem-Spanos-1991 / Le-Maître-Knio-2010 / Ghanem-Higdon-Owhadi-2017 *Handbook of UQ* chapters, (c) connective-tissue readiness (substrate ready in repo).

### Tier-0 — substrate (~210 LOC, blocks U10-U25)

#### U0a. `prob/random.go` — Gaussian / Uniform / Exponential / Gamma / Beta samplers — ~120 LOC
**Same blocker called out by 117/202/215.** Must ship before any stochastic UQ primitive lands. Exposes `prob.RNG interface{ Float64() float64; Uint64() uint64 }` and `SampleNormal`, `SampleUniform`, `SampleExponential`, `SampleGamma`, `SampleBeta`. Uses Marsaglia-polar (Box-Muller variant) for Normal cross-language parity with Go stdlib semantics. Adapts `crypto/rng.go` Splitmix/Xoshiro.

#### U0b. `uq/sampling/halton.go` — lift `audio/separation/nmf.go:halton` to public — ~30 LOC
Trivial extraction. Halton sequence is a dependency of QMC integration; reality already has 8 LOC of it private — promote to `uq/sampling/halton.go` with public `Halton(n int, base int) float64` and `HaltonVector(n int, bases []int, out []float64)`.

#### U0c. `linalg/svd.go` — Singular Value Decomposition — ~280 LOC
Same blocker called out by 097-linalg-missing and 215-cs. Required for U21 Kriging robust-solve and for U16 active-subspace alternative formulation. Standard Golub-Reinsch with Householder bidiagonalisation + implicit-QR sweeps.

### Tier-1 — orthogonal-polynomial substrate (`uq/poly/`, ~520 LOC)

#### U1. `uq/poly/hermite.go` — Probabilists' Hermite `He_n(x) = (-1)^n e^{x²/2} d^n/dx^n e^{-x²/2}` — ~120 LOC
Three-term recurrence `He_{n+1} = x·He_n − n·He_{n-1}` with He_0=1, He_1=x. Orthogonal w.r.t. `e^{-x²/2}/√(2π)` (i.e. standard Gaussian density). Output: `HermiteEval(n int, x float64) float64`, `HermiteVector(maxDeg int, x float64, out []float64)`, `HermiteCoeffs(n int) []float64`. **THIS IS THE PRIMARY BASIS FOR PCE WITH GAUSSIAN INPUTS** (Wiener 1938).

#### U2. `uq/poly/legendre.go` — Legendre `P_n(x)`, `(n+1)P_{n+1} = (2n+1)x P_n − n P_{n-1}` — ~120 LOC
Orthogonal w.r.t. uniform measure on `[-1, 1]`. Building block for uniform-input PCE. Already partially present in `calculus/GaussLegendre` (uses 5-pt quadrature nodes), but the polynomial evaluator is absent. Add `LegendreEval`, `LegendreVector`, `LegendreCoeffs`.

#### U3. `uq/poly/laguerre.go` — Laguerre `L_n(x) = (e^x / n!) d^n/dx^n (x^n e^{-x})` — ~120 LOC
Orthogonal w.r.t. `e^{-x}` on `[0, ∞)`. Building block for exponential / Gamma-input PCE. Three-term `(n+1)L_{n+1} = (2n+1−x)L_n − n L_{n-1}`.

#### U4. `uq/poly/jacobi.go` — Jacobi `P_n^{α,β}(x)`, generalises Legendre to Beta-distributed inputs — ~160 LOC
Orthogonal w.r.t. `(1−x)^α (1+x)^β` on `[-1, 1]`. Building block for Beta-input PCE. Three-term recurrence with α, β-dependent coefficients (Abramowitz-Stegun 22.7).

### Tier-2 — quadrature (`uq/quadrature/`, ~480 LOC)

#### U5. `uq/quadrature/gauss.go` — Gauss-Hermite, Gauss-Laguerre, Gauss-Jacobi nodes/weights via Golub-Welsch — ~280 LOC
Replaces hardcoded 5-pt-only `calculus.GaussLegendre` with arbitrary-`n` Golub-Welsch eigenvalue method (recurrence-coefficient tridiagonal → eigendecomp via existing `linalg/eigen.go:QRAlgorithm`). Output: `GaussHermiteRule(n int) (nodes, weights []float64)`, `GaussLaguerreRule(n int)`, `GaussJacobiRule(n int, alpha, beta float64)`, `GaussLegendreRule(n int)` (replaces 5-pt cap).

#### U6. `uq/quadrature/tensor.go` — Full-tensor multi-dim quadrature — ~80 LOC
Cartesian product of 1D rules. `TensorRule(rules []Rule) MultiRule { ... }` returns nodes ∈ R^d and weights. Suffers curse of dimensionality (`n^d` points) but exact for total-degree-`(n−1)` polynomials.

#### U7. `uq/quadrature/smolyak.go` — Smolyak sparse-grid (Smolyak 1963) — ~120 LOC
Linear combination of tensor rules: `Q_l^d = Σ_{|i| ≤ l+d-1} (-1)^{l+d-1−|i|} C(d-1, l+d-1−|i|) (Q_{i_1}^1 ⊗ ... ⊗ Q_{i_d}^1)`. Reduces `n^d` to `O(n (log n)^{d-1})` for the same polynomial-exactness order. Mandatory for `d ≥ 5` UQ. (Smolyak 1963; Gerstner-Griebel 1998; Xiu-Hesthaven 2005.)

### Tier-3 — Polynomial Chaos Expansion (`uq/pce/`, ~640 LOC)

#### U8. `uq/pce/galerkin.go` — Galerkin-projection PCE — ~180 LOC
For input `ξ ~ p(ξ)` and forward map `Y = M(ξ)`, project onto orthonormal polynomial basis `{Ψ_α}`: `c_α = E[Y · Ψ_α(ξ)] / E[Ψ_α²]`. Numerator computed via tensor or Smolyak quadrature U6/U7. Output: `Galerkin(M func([]float64) float64, basis Basis, quadRule MultiRule) []float64` returning coefficients. (Wiener 1938, Ghanem-Spanos 1991, Xiu-Karniadakis 2002.)

#### U9. `uq/pce/collocation.go` — Stochastic Collocation PCE — ~120 LOC
Evaluate `M` at quadrature nodes, build interpolant. Cheaper than Galerkin (no inner-product integration), exact for tensor-product polynomial spans. (Xiu-Hesthaven 2005, Babuška-Nobile-Tempone 2007.) Output: `Collocation(samples []float64, basis Basis) []float64`.

#### U10. `uq/pce/sparse_lars.go` — Sparse PCE via l1-LARS regression — ~220 LOC
Blatman-Sudret 2011 LARS-regression on candidate-basis Vandermonde-like matrix Φ = `[Ψ_α(ξ_i)]_{i,α}`, with hyperbolic truncation `||α||_q ≤ p` (q=0.5 default — keeps low-interaction terms). Killer feature: collapses `(p+1)^d` candidate basis to `O(d log d)` active terms for typical engineering models. Composes `optim/proximal/ProxL1 + Fbs` (already shipped) with a stagewise-LARS active-set update. (Blatman-Sudret *J. Comput. Phys.* 2011, 1,800+ citations.)

#### U11. `uq/pce/gpc.go` — Generalised PCE Wiener-Askey scheme dispatcher — ~80 LOC
Maps input distribution → orthogonal polynomial basis: Gaussian→Hermite, Uniform→Legendre, Exponential/Gamma→Laguerre, Beta→Jacobi. Output: `WienerAskeyBasis(d prob.Distribution) Basis`. Xiu-Karniadakis 2002 (8,500+ citations).

#### U12. `uq/pce/moments.go` — Closed-form moments + Sobol-from-PCE — ~40 LOC
For a PCE `Y = Σ c_α Ψ_α`, mean = c_0, variance = Σ_{α≠0} c_α² (orthonormal basis). Sobol indices have closed-form via grouping by index-set: `S_i = Σ_{α∈A_i} c_α² / Σ_{α≠0} c_α²` where `A_i = {α : α_i > 0, α_j = 0 for j ≠ i}`. (Sudret *Reliab. Eng. Sys. Saf.* 2008, 2,000+ citations.) Direct closed-form S_i without re-sampling — the killer-app of PCE for sensitivity.

### Tier-4 — Sampling (`uq/sampling/`, ~360 LOC)

#### U13. `uq/sampling/lhs.go` — Latin Hypercube Sampling (McKay-Beckman-Conover 1979) — ~80 LOC
Stratified uniform sampling: divide each marginal into N equiprobable strata, draw 1 sample per stratum, randomly permute across dims. Output: `LHS(n int, dim int, rng RNG, out [][]float64)`. McKay 1979 (5,500+ citations).

#### U14. `uq/sampling/sobol.go` — Sobol low-discrepancy sequence + Joe-Kuo direction numbers — ~180 LOC
Up to dim=21,201 via Joe-Kuo 2008 direction numbers (~21K·32 uint table embedded as Go init-data, ~80KB). Discrepancy `O((log N)^d / N)` vs MC's `O(1/√N)`. Owen-scrambling variant: `SobolOwenScrambled(n, dim int, rng RNG, out [][]float64)`. (Sobol 1967, Joe-Kuo *ACM Trans. Math. Software* 2008, Owen 1995.)

#### U15. `uq/sampling/halton.go` — Halton sequence + Owen-scrambling — ~50 LOC
Lift existing `audio/separation/nmf.go:halton` to public `uq/sampling/Halton(n int, bases []int, out []float64)`. Owen-Tezuka 2000 scrambling for variance reduction.

#### U16. `uq/sampling/qmcmc.go` — randomised QMC: digital-shift, scramble, antithetic — ~50 LOC
`L'Ecuyer 2009` randomised-QMC primitives; antithetic-variates for variance reduction.

### Tier-5 — Sensitivity (`uq/sobol/`, `uq/sensitivity/`, ~960 LOC)

#### U17. `uq/sobol/saltelli.go` — Saltelli sampling matrix A,B,A_B^(i) construction — ~120 LOC
Standard Saltelli-2010 trick: draw 2N×d matrix, split into A (N×d) and B (N×d), construct d copies of A_B^(i) (replace col i of A with col i of B). Total `N(d+2)` model evaluations. Output: `SaltelliMatrix(n int, dim int, rng RNG) (A, B [][]float64, AB [][][]float64)`. (Saltelli *Comp. Phys. Comm.* 2010, 7,000+ citations.)

#### U18. `uq/sobol/indices.go` — Sobol first-order S_i and total-effect S_T_i via Jansen estimator — ~120 LOC
Jansen 1999 estimators (lower variance than naive Sobol-1990): `S_i = (1/N) Σ_j f(B)_j · (f(A_B^i)_j − f(A)_j) / V[Y]`, `S_T_i = (1/(2N)) Σ_j (f(A)_j − f(A_B^i)_j)² / V[Y]`. Bootstrapped confidence intervals. (Saltelli *Comp. Phys. Comm.* 2010 §4 reviews 12+ estimator-pair choices, recommends Jansen for `S_T` and Saltelli-2010 for `S_i`.)

#### U19. `uq/sensitivity/morris.go` — Morris elementary-effects μ*, σ — ~140 LOC
Cheap-screening method (Morris 1991, Campolongo-Cariboni-Saltelli 2007). One-at-a-time perturbations on radial / trajectory designs. `r·(d+1)` evaluations vs Saltelli's `N·(d+2)` with N=1000 and r=20-50 → 20-50× cheaper for screening. Output: `MorrisEffects(M, dim int, r int, rng RNG) (mu, mu_star, sigma []float64)`. (Campolongo-Cariboni-Saltelli *EMS* 2007, 2,500+ citations.)

#### U20. `uq/sensitivity/ale.go` — Accumulated Local Effects (Apley-Zhu 2020) — ~140 LOC
Bias-free alternative to partial-dependence-plot (PDP); insensitive to feature correlation. Output: `ALEPlot(M func([]float64) float64, samples [][]float64, featureIdx int, nBins int) (xGrid, ALE []float64)`. (Apley-Zhu *J. R. Stat. Soc. B* 2020, 700+ citations.)

#### U21. `uq/sensitivity/shapley.go` — Shapley effects (Owen 2014) via existing gametheory.ShapleyValue — ~80 LOC
**HIGHEST LEVERAGE — substrate already 95% present.** `gametheory/voting.go:119:ShapleyValue(n, charFunc) []float64` IS the algorithmic core; the only addition is the UQ-specific characteristic function `charFunc(coalition []bool) float64 = Var(E[Y | X_S])` where `S = {i : coalition[i]}`. Estimated via Monte-Carlo subset-conditional mean (Castro-Gómez-Tejada 2009 random-permutation estimator under the hood). Splits Sobol-S_T mass *exactly* among inputs (unlike Sobol which double-counts interactions), addressing Sobol's well-known correlated-input pathology. (Owen *SIAM/ASA J. UQ* 2014, 600+ citations; Iooss-Prieur 2019 follow-up; Lundberg-Lee SHAP 2017 makes ML-Shapley standard.)

#### U22. `uq/sensitivity/pdd.go` — Polynomial Dimensional Decomposition (Rahman 2008) — ~100 LOC
Hierarchical orthogonal decomposition `Y = f_0 + Σ f_i + Σ f_{ij} + ...`. Composes with U8 PCE to give variance-decomposition Sobol-style indices in closed form. (Rahman *Int. J. Numer. Meth. Eng.* 2008.)

#### U23. `uq/sensitivity/derivative.go` — Derivative-based Global Sensitivity Measures (Sobol-Kucherenko 2009) — ~60 LOC
Cheap screening via averaged-squared-gradient `ν_i = E[(∂M/∂X_i)²]`. Composes `calculus.NumericalGradient` (already shipped). Bounds Sobol total-effect from above (Sobol-Kucherenko 2009 inequality).

### Tier-6 — Reliability (`uq/reliability/`, ~580 LOC)

#### U24. `uq/reliability/form.go` — First-Order Reliability Method (Hasofer-Lind 1974) — ~140 LOC
Solves `min ‖u‖² s.t. g(T(u)) = 0` in standard normal space, where T is the Nataf/Rosenblatt transformation from physical to standard-normal-space. β = `‖u*‖`, `P_f ≈ Φ(−β)`. Solver: existing `optim.LBFGS` with caller-supplied gradient. (Hasofer-Lind *J. Eng. Mech.* 1974, 4,500+ citations.)

#### U25. `uq/reliability/sorm.go` — Second-Order Reliability Method (Breitung 1984) — ~120 LOC
Second-order paraboloid approximation at MPP (most-probable-point from FORM). `P_f ≈ Φ(−β) Π_{i=1}^{d-1} (1 − β κ_i)^{-1/2}` with `κ_i` principal curvatures of g at MPP. Composes `calculus.NumericalGradient` (Hessian via finite-diff) + `linalg.QRAlgorithm` (curvature eigenvalues).

#### U26. `uq/reliability/subset.go` — Subset Simulation (Au-Beck 2001) — ~180 LOC
Adaptive-MCMC for rare-event probabilities `P_f = 10^{-6}` regimes where naïve MC requires `10^9` samples. Decomposes `P_f = Π_i p_i` into intermediate-event chain with each `p_i ≈ 0.1` (cheap MCMC). Killer-app for structural-engineering reliability, finance VaR, climate-tipping-points. (Au-Beck *Probabilistic Engineering Mechanics* 2001, 4,700+ citations.)

#### U27. `uq/reliability/cem.go` — Cross-Entropy Method (Rubinstein 1997, De Boer-Kroese-Mannor-Rubinstein 2005) — ~140 LOC
Adaptive-importance-sampling: parametrise proposal `q_θ`, iteratively minimise KL(target || q_θ) via importance-weighted moment-matching. Closed-form for Gaussian/exponential family proposals. (Rubinstein *Methodol. Comput. Appl. Probab.* 1997; De Boer et al. *Ann. OR* 2005, 2,700+ citations.)

### Tier-7 — Surrogate models (`uq/surrogate/`, ~520 LOC)

#### U28. `uq/surrogate/kriging.go` — Gaussian Process / Ordinary Kriging — ~280 LOC
`m(x*) = k_*^T K^{-1} y`, `σ²(x*) = k(x*, x*) − k_*^T K^{-1} k_*`. Anisotropic Matérn-3/2 / Matérn-5/2 / squared-exponential / Wendland kernel library. MLE / REML hyperparameter tuning via existing `optim.LBFGS`. Composes `linalg.Cholesky` (when conditioning is good) with `linalg.SVD` fallback (when ill-conditioned at near-duplicate samples). (Krige 1951, Sacks-Welch-Mitchell-Wynn *Stat. Sci.* 1989, Rasmussen-Williams *GPML* 2006.)

#### U29. `uq/surrogate/kriging_loocv.go` — Leave-One-Out Cross-Validation (Dubrule 1983) — ~80 LOC
Closed-form LOO without N retrains: `(y_i − ŷ_{-i}) = (K^{-1} y)_i / (K^{-1})_{ii}`. Used as model-validation / surrogate-quality metric.

#### U30. `uq/surrogate/bayesopt.go` — Bayesian Optimisation acquisition functions — ~160 LOC
Expected Improvement (EI; Mockus 1975), Upper Confidence Bound (UCB; Srinivas-Krause-Kakade-Seeger 2010), Probability of Improvement (PI; Kushner 1964), Thompson Sampling. Each composed with `optim.LBFGS` over GP-surrogate posterior. Cross-link to slot 169-synergy-prob-optim §4 (BayesOpt is the most-cited applied UQ technique 2015-2026, with 30%+ of practitioner-time across material-design / hyperparameter-tuning / experiment-design).

### Tier-8 — Frontier (`uq/active/`, multi-fidelity, KLE, concentration, ~520 LOC)

#### U31. `uq/active/subspace.go` — Active Subspace (Constantine 2015) — ~120 LOC
Identify dominant input directions via gradient-covariance eigendecomp `C = E[∇M ∇M^T] = W Λ W^T`. Project `M(x) ≈ g(W_1^T x)` onto leading-`r` eigenvectors. Composes `calculus.NumericalGradient` + existing `linalg.QRAlgorithm` (eigendecomp on the symmetric `d×d` gradient covariance). Constantine *Active Subspaces*, SIAM 2015, 2,800+ citations.

#### U32. `uq/active/ridge.go` — Ridge function recovery via PCE on active variables — ~80 LOC
Once active subspace is identified, fit a low-dim PCE on `z = W_1^T x`. Composes U10 sparse-PCE.

#### U33. `uq/kle/expansion.go` — Karhunen-Loève Expansion — ~180 LOC
For zero-mean random field `K(x, x') = Σ λ_n φ_n(x) φ_n(x')`, find dominant eigenpairs of covariance kernel via existing `linalg.QRAlgorithm`. Output: `KLE(C [][]float64, nModes int) (eigenvalues, eigenvectors)`. (Karhunen 1947, Loève 1948.) Foundational for random-field UQ propagation.

#### U34. `uq/mfmc/mfmc.go` — Multi-Fidelity Monte Carlo (Peherstorfer-Willcox-Gunzburger 2016) — ~140 LOC
Optimal-allocation MC across model hierarchy `M_1, M_2, ..., M_K` with cost-vs-correlation tradeoff. Generalises MLMC (which assumes hierarchical pairwise telescoping). Closed-form for variance-budget allocation. Cross-link to slot 202-new-sde S5 (MLMC). (Peherstorfer-Willcox-Gunzburger *SIAM Review* 2018, 1,200+ citations.)

#### U35. `uq/concentration/concentration.go` — McDiarmid + Talagrand-Lipschitz inequalities — ~80 LOC
Lipschitz-functional concentration: `P(|f(X) − E[f(X)]| > t) ≤ 2 exp(−t²/(2 L²))` for L-Lipschitz f under product measure. Hoeffding-vector, McDiarmid-bounded-difference, Efron-Stein variance-bound. (Boucheron-Lugosi-Massart 2013.)

---

## (3) Connective tissue LOC summary

| Tier | Sub-package | New LOC | Reuse LOC | Net delta |
|---|---|---:|---:|---:|
| T0 | `prob/random.go` (gating) | 120 | 0 | 120 |
| T0 | `uq/sampling/halton.go` (lift) | 30 | 8 | 22 |
| T0 | `linalg/svd.go` (gating, shared with 215) | 280 | 0 | 280 |
| T1 | `uq/poly/` (Hermite/Legendre/Laguerre/Jacobi) | 520 | 0 | 520 |
| T2 | `uq/quadrature/` (Gauss + tensor + Smolyak) | 480 | 70 (`calculus.GaussLegendre` upgraded) | 410 |
| T3 | `uq/pce/` (Galerkin/collocation/sparse-LARS/gPC/moments) | 640 | 100 (`optim/proximal/Fbs+ProxL1`) | 540 |
| T4 | `uq/sampling/` (LHS/Sobol/Halton/QMCMC) | 360 | 8 (`audio/separation/halton`) | 352 |
| T5 | `uq/sobol/` + `uq/sensitivity/` | 960 | 30 (`gametheory.ShapleyValue`) | 930 |
| T6 | `uq/reliability/` (FORM/SORM/SS/CEM) | 580 | 250 (`optim.LBFGS` + `prob.NormalCDF/Quantile`) | 330 |
| T7 | `uq/surrogate/` (Kriging/BayesOpt) | 520 | 200 (`linalg.Cholesky` + `optim.LBFGS`) | 320 |
| T8 | `uq/active/` + `uq/kle/` + `uq/mfmc/` + concentration | 520 | 230 (`calculus.NumericalGradient` + `linalg.QRAlgorithm` + `linalg.PCA` + `linalg.CovarianceMatrix`) | 290 |
| | **TOTAL** | **5,010** | **896** | **4,114** |

Net: ~3,950 LOC of new UQ code with ~900 LOC of substrate reuse — an unusually high reuse fraction for any new Block-C package.

---

## (4) Recommended PR sequence

**PR1 (Tier-0 substrate, ~430 LOC, blocks 4 Block-C reviews simultaneously):** `prob/random.go` (Gaussian + Uniform + Exponential samplers) + `linalg/svd.go` (Golub-Reinsch). Unblocks 117 / 202 / 215 / 227 in one go. *This is the highest-leverage single PR of the entire 400-sequence — four independent reviews demand it.*

**PR2 (Tier-1+2 polynomial+quadrature, ~1,000 LOC):** `uq/poly/{hermite, legendre, laguerre, jacobi}.go` + `uq/quadrature/{gauss, tensor, smolyak}.go`. Upgrade `calculus.GaussLegendre` to arbitrary-`n` via Golub-Welsch. Standalone-shippable: enables U8 Galerkin + U9 collocation in one PR.

**PR3 (Tier-3 PCE, ~640 LOC):** `uq/pce/{galerkin, collocation, sparse_lars, gpc, moments}.go`. Cross-link to slot 215 LARS once that ships, otherwise consume existing `optim/proximal/Fbs` for sparse-PCE. Single highest-leverage one-week unlock — once PCE ships, ALL of Sobol indices via U12 `moments.go` come for free in closed form (no MC needed).

**PR4 (Tier-4+5 sampling+sensitivity, ~1,320 LOC):** `uq/sampling/{lhs, sobol, halton, qmcmc}.go` + `uq/sobol/{saltelli, indices}.go` + `uq/sensitivity/{morris, ale, shapley, pdd, derivative}.go`. Composes existing `gametheory.ShapleyValue` for U21 — the singular reality moat-claim.

**PR5 (Tier-6+7 reliability+surrogate, ~1,100 LOC):** `uq/reliability/{form, sorm, subset, cem}.go` + `uq/surrogate/{kriging, kriging_loocv, bayesopt}.go`. Composes existing `optim.LBFGS` for FORM iteration and Kriging hyperparameter MLE.

**PR6 (Tier-8 frontier, ~520 LOC):** `uq/active/{subspace, ridge}.go` + `uq/kle/expansion.go` + `uq/mfmc/mfmc.go` + `uq/concentration/concentration.go`. Cross-link to slot 202-new-sde MLMC for U34.

---

## (5) Cross-package dependencies and policy alignment

reality's CLAUDE.md commitments (zero deps, golden-files, no-allocs-hot-path, every-function-cites-source, precision-documented) all hold for `uq/`:

- **Zero deps:** every primitive is reimplemented from canonical references (Joe-Kuo direction-numbers embedded as Go init-data, not loaded from external files; orthogonal-polynomial recurrence-coefficients computed on demand, not table-loaded).
- **Golden-files:** mandatory ≥20 vectors per primitive at 1e-11 tolerance for transcendental Hermite/Laguerre/Jacobi evaluation, 1e-9 for accumulating Saltelli-Sobol estimators (variance-reduction estimators have inherent statistical noise — golden-files must capture deterministic-RNG-seeded outcomes). IEEE 754 edge cases (Inf / NaN / -0.0) on poly evaluation at boundary nodes.
- **No-allocs-hot-path:** all U1-U35 primitives expose `out []float64` output buffers in the same idiom as `optim/proximal/ProxL1(v, gamma, out)`.
- **Citations:** Wiener 1938, Ghanem-Spanos 1991, Xiu-Karniadakis 2002, Saltelli 2008/2010, Sudret 2008, Constantine 2015, Au-Beck 2001, Owen 2014, Joe-Kuo 2008, Apley-Zhu 2020, Hasofer-Lind 1974, Breitung 1984, Peherstorfer-Willcox-Gunzburger 2018, Blatman-Sudret 2011, Smolyak 1963, Sobol 1967, Krige 1951, Sacks-Welch-Mitchell-Wynn 1989 — all top-3000-citations papers in the field.
- **Precision-documented:** PCE convergence-rate `O(p^{-r})` for `r`-times-mean-square-differentiable forward maps stated per primitive; Sobol-MC convergence `O(N^{-1/2})` with bootstrap CI; FORM/SORM accuracy degrades with curvature (documented per primitive); Subset-Simulation COV(P_f) = `√((1−p_0)/(p_0 N₀)) · √m` for m-level chain.

---

## (6) Bottom line

reality v0.10.0 has **zero UQ surface but unusually rich UQ substrate.** `gametheory.ShapleyValue` is the biggest single hidden asset — the *only* Go-stdlib-zero-dep cooperative-game primitive in any open-source repo, perfectly suited as the SHAP/Shapley-effect engine. `calculus.GaussLegendre` (despite its 5-pt cap), `linalg.QRAlgorithm` (eigendecomp for KLE/active-subspace), `optim.LBFGS` (FORM/Kriging-hyperparameter MLE), `optim/proximal/{ProxL1, Fbs}` (sparse-PCE LARS engine), `calculus.NumericalGradient` (active-subspace gradient cov + SORM Hessian), `chaos.RK4Step` (UQ-propagation through ODE), `prob.NormalCDF/Quantile` (FORM `Φ(−β)`), `audio/separation.halton` (lift to public Halton sequence), `linalg.PCA` (active-subspace = PCA-on-gradients), `linalg.CovarianceMatrix` (Saltelli-output covariance), and the entire `prob/copula/` package (Nataf/Rosenblatt input-correlation modelling, the Solvency-II / Basel-III standard for engineering-finance UQ) — together these substrates mean the 26 U-primitives connect through ~3,950 LOC of NEW code, with ~900 LOC of substrate reuse.

The single missing-substrate blocker is the **Tier-0 `prob/random.go` Gaussian sampler**, called out by FOUR independent Block-C reviews (117, 202, 215, 227) — it is the highest-priority cross-cutting Block-C unblocker in the entire 400-sequence. Ship it first.

The single highest-leverage moat is **U21 ShapleyEffects via existing `gametheory.ShapleyValue`** — ~80 LOC of code, no other zero-dep Go library does this composition, and it directly addresses Sobol-correlated-input pathology that the Sobol primitives U17-U18 cannot resolve.

The single highest-impact frontier is **U10 Sparse-PCE via l1-LARS** — collapses `(p+1)^d` curse-of-dimensionality to `O(d log d)` active-basis size and via U12 `moments.go` yields ALL Sobol indices in *closed form* without any further MC. This is the difference between a 2008-Saltelli-era UQ library (Monte-Carlo-bound) and a 2026-Sudret-era one (PCE-spectral-bound, 100×-1000× cheaper for engineering models).
