# 264 | new-mlmc — Multilevel Monte Carlo: Giles construction, antithetic, randomized

**Summary line 1.** reality v0.10.0 ships **zero** Multilevel Monte Carlo (MLMC) surface — repo-wide grep on `MLMC|MultiLevel|MultilevelMC|Multi.Level.Monte|Giles|telescop|Rhee.Glynn|MIMC|MultiFidelity|MFMC|continuation.MLMC|antithetic.*MC|antithetic.*level|N_l|optimal.allocation|level.*hierarchy` returns **zero callable matches** across all 22 packages and `chaos|prob|calculus|optim|orbital|signal` substrates; closest tangential surface is the 25-LOC `MonteCarloIntegrate(f, dim, lower, upper, samples, rng)` at `calculus/calculus.go:244` (a single-level uniform MC with O(N^{−½}) convergence — the *natural* MLMC injection-point but ships no level-hierarchy / variance-decay / cost-allocation / telescoping logic) and the deterministic `chaos/ode.go::RK4Step` + `EulerStep` (the *substrate* a level-pair coarse/fine simulator would build on). The 202-new-sde slot already enumerated S5 MLMC as the single highest-leverage cutting-edge SDE primitive (~300 LOC) but bound it to SDE simulation only; slot 263-new-quasi-mc Q25 enumerated MultilevelQMC (Kuo-Schwab-Sloan-2012) as a 120-LOC consumer of MLMC + Sobol; slot 227-new-uq U24 enumerated multi-fidelity Monte Carlo (Peherstorfer-Willcox-Gunzburger 2016) as the *non-hierarchical* generalisation of MLMC; slot 215-new-compressed-sensing did not touch MLMC. Net: across 4 prior Block-C reviews (202, 215, 227, 263) the MLMC umbrella is *named* in three of them (S5, U24, Q25) but **no slot has scoped the MLMC theory + algorithm corpus on its own**. This slot 264 fills that gap. The single most-cited reference in the field — Giles-2008-Oper-Res-56:607 *Multilevel Monte Carlo Path Simulation* (~3,800 citations) and the 2015-Acta-Numer-24:259 follow-up *Multilevel Monte Carlo methods* (~1,400 citations) — is unrepresented in any reality test/doc/comment; the antithetic-MLMC frontier (Giles-Szpruch-2014-Ann-Appl-Probab-24:1585 ~580 citations) is unrepresented; the randomized-MLMC unbiased-estimator frontier (Rhee-Glynn-2015-Oper-Res-63:1026 ~520 citations) is unrepresented; the Multi-Index Monte Carlo (MIMC) frontier (Haji-Ali-Nobile-Tempone-2016-Numer-Math-132:767 ~280 citations) is unrepresented; the Continuation-MLMC adaptive-budget extension (Collier-Haji-Ali-Nobile-von-Schwerin-Tempone-2015-BIT-55:399 ~140 citations) is unrepresented; the multilevel-Bayesian-inverse-problem extension (Hoang-Schwab-Stuart-2013-Inverse-Problems-29:085010 ~260 citations) is unrepresented; the rule-of-three convergence theorem (Giles-2015 Theorem 1: complexity is O(ε⁻²) iff α ≥ ½ min(β, γ−2)) is unrepresented as a verifiable invariant pin.

**Summary line 2.** Twenty-six primitives **L1–L26** totalling **~3,200 LOC pure new code + ~80 LOC shared substrate** (the same `prob/random/normal.go ~80 LOC` Gaussian-sampler-substrate-pool already P0-blocking 117/202/215/227/259-263; co-shipping with 202-S5 amortises away the substrate cost) split across (a) ~440 LOC **Tier-1 Giles-2008 keystone** in new sub-package `mlmc/` (L1 `mlmc/estimator.go::Estimator{LevelSampler, OptimalN, MaxLevels, Tolerance}` ~180 LOC the canonical Giles-2008 Algorithm 1 driver with on-the-fly level-extension `while bias > ε/√2 do L+=1`; L2 `mlmc/estimator.go::SampleLevel(l, dN int, sampler LevelSampler) (sumP, sumP2 float64)` ~80 LOC the per-level batch-sampler that returns first/second-moments of the difference Δ_ℓ = P_ℓ − P_{ℓ−1} with paired coarse/fine RNG state; L3 `mlmc/estimator.go::OptimalAllocation(V, C []float64, eps float64) []int` ~60 LOC Giles-2008 §2.3 closed-form `N_ℓ = ⌈2·ε⁻²·√(V_ℓ·C_ℓ)·Σ_k √(V_k/C_k)⌉` with the leading-`2` factor the standard 50%-budget split between bias and variance error; L4 `mlmc/estimator.go::BiasExtrapolation(meanY []float64, M float64) float64` ~40 LOC Giles-2008 §2.5 weak-convergence-rate-α extrapolation `bias ≤ |E[Y_L]|·M^α/(M^α − 1)` for richardson-style remaining-bias cap; L5 `mlmc/levelsampler.go::LevelSampler interface{ SampleLevel(l int, rng RNG) (Pf, Pc float64) }` ~80 LOC the canonical interface every consumer (SDE-paths, PDE-FEM-meshes, Bayesian-inverse-MCMC-chains) implements once), (b) ~360 LOC **Tier-2 antithetic-MLMC Giles-Szpruch-2014** (L6 `mlmc/antithetic.go::AntitheticLevelSampler` ~120 LOC the wrapper that for each fine-path on level ℓ generates BOTH the standard fine-path `P_ℓ` AND the *swap-Brownian-pair* antithetic fine-path `P̃_ℓ` and uses `Y_ℓ = ½(P_ℓ + P̃_ℓ) − P_{ℓ−1}` — gives Milstein-strong-order without computing Lévy-area corrections for non-commutative-noise multi-D SDEs, the 2014 paper's keystone result; L7 `mlmc/antithetic.go::TruncatedAntithetic` ~60 LOC truncate-pair-when-degenerate variant for jump-diffusions; L8 `mlmc/antithetic.go::SymmetricAntithetic` ~80 LOC `Y_ℓ = ¼(P_ℓ + P̃_ℓ + 2P_{ℓ−1})` symmetric-quadrature when both coarse and fine are antithetised; L9 `mlmc/antithetic.go::PairBrownianIncrements(fineDt, coarseDt float64, fineN int, rng RNG, fineW, coarseW [][]float64)` ~100 LOC the cache-coherent paired-increment generator that ensures sum-over-fine-equals-coarse `coarseW[k] = sum(fineW[k·M:(k+1)·M])` — the SINGLE most-error-prone implementation detail in any MLMC code: independent regeneration of coarse/fine breaks the variance-decay), (c) ~360 LOC **Tier-3 randomized MLMC Rhee-Glynn-2015** (L10 `mlmc/randomized.go::RandomizedEstimator{LevelDist, LevelSampler}` ~140 LOC the Rhee-Glynn-2015 unbiased-MLMC: at each MC trial draw a random level `L ~ p_L` then return `Δ_L / p_L` — UNBIASED estimator with finite-variance iff `Σ_ℓ V_ℓ/p_ℓ < ∞`; L11 `mlmc/randomized.go::OptimalLevelDist(V, C []float64) []float64` ~80 LOC the optimal-sampling-distribution `p_ℓ ∝ √(V_ℓ/C_ℓ)` minimising work-normalised-variance product; L12 `mlmc/randomized.go::SingleTermRandomized(sampler, dist, rng) float64` ~60 LOC the per-MC-trial single-term-estimator that bypasses the level-hierarchy budget-allocation entirely — useful when bias-tolerance ε is unknown a priori and the MLMC budget will be set by wall-clock not by ε; L13 `mlmc/randomized.go::DoubleRandomized(...)` ~80 LOC McLeish-2011 + Rhee-Glynn-2015 §4 double-randomization for refresh-noisy-objectives like SGD-of-stochastic-targets — connects MLMC with stochastic-gradient methods, see slot 220 stochastic-opt cross-link), (d) ~360 LOC **Tier-4 continuation MLMC Collier-Haji-Ali-Nobile-von-Schwerin-Tempone-2015** (L14 `mlmc/continuation.go::ContinuationDriver{Tolerances []float64, ...}` ~160 LOC the continuation-MLMC adaptive-driver that runs MLMC at a sequence of decreasing tolerances `ε_0 > ε_1 > ... > ε_T` reusing prior-level samples — saves ~30% over restarting MLMC at each ε; L15 `mlmc/continuation.go::AdaptiveTolerance(stats LevelStats, target float64) []float64` ~80 LOC the tolerance-schedule selector `ε_{k+1} = max(ε/2, ε_k/r)` with adaptive ratio r; L16 `mlmc/continuation.go::AdaptiveLevelExtension(maxBias, eps float64, alpha float64) (newL int, ok bool)` ~60 LOC the adaptive-L-selector for unknown-α regimes — estimate α from observed |E[Y_ℓ]| log-log slope; L17 `mlmc/continuation.go::PostStratifiedEstimator` ~60 LOC the Heinrich-Sindambiwe-2009 post-stratified MLMC which combines stratification across levels with the standard MLMC budget), (e) ~480 LOC **Tier-5 multi-index MC Haji-Ali-Nobile-Tempone-2016** (L18 `mlmc/mimc.go::MIMCEstimator{IndexSet, ...}` ~160 LOC the MIMC driver replacing 1-D level-index `ℓ` with multi-D index `α ∈ N^d` using *mixed differences* `Δ_α = ∏_i (P_{α_i} − P_{α_i−1})` — KEY INNOVATION: tensor-product index-sets share the variance-decay across dimensions, breaking the curse-of-dimensionality for d-D PDEs where each dim has its own discretisation level; L19 `mlmc/mimc.go::TotalDegreeIndexSet(L int) [][]int` ~80 LOC the |α|≤L total-degree truncation; L20 `mlmc/mimc.go::HyperbolicCrossIndexSet(L int) [][]int` ~80 LOC the optimal index-set for analytic functions; L21 `mlmc/mimc.go::AdaptiveProfit(stats IndexStats) []int` ~80 LOC the adaptive index-extension by max-profit-marker `profit_α = √(V_α/C_α)`; L22 `mlmc/mimc.go::OptimalMIMCAllocation(V, C [][]float64) [][]int` ~80 LOC the multi-D analog of L3), (f) ~480 LOC **Tier-6 multilevel-QMC + multi-fidelity MC** (L23 `mlmc/qmc.go::MultilevelQMC` ~160 LOC Kuo-Schwab-Sloan-2012-Found-Comp-Math-13:1245 — combine MLMC with QMC at each level using randomised-shift lattice or Sobol with Owen-scrambling — convergence rate O(N^{−1+δ}) per level vs MC's O(N^{−½}), so total cost drops below O(ε⁻²) under sufficient regularity; CROSS-LINK to 263-Q25; L24 `mlmc/mfmc.go::MultiFidelityMC` ~200 LOC Peherstorfer-Willcox-Gunzburger-2016-SIAM-J-Sci-Comput-38:A3163 — non-hierarchical multi-fidelity MC where models are NOT in a discretisation-hierarchy but in a model-fidelity-hierarchy (e.g. high-fidelity full Navier-Stokes vs low-fidelity Reynolds-averaged); KEY INNOVATION: estimator `Y = sum_i α_i (P_i − E[P_i])` with optimal correlation-weights α_i computed from sample-covariances; CROSS-LINK to 227-U24; L25 `mlmc/mfmc.go::ApproximateControlVariates` ~120 LOC Gorodetsky-Geraci-Eldred-Jakeman-2020-J-Comput-Phys-408:109257 ACV the modern generalisation that handles disconnected-fidelity-graphs with the optimal allocation solving a small QP — reality has `optim/proximal/admm.go` for the QP), (g) ~360 LOC **Tier-7 SDE/PDE/Bayes consumers** (L26 `mlmc/consumers/sde.go::SDELevelSampler{Drift, Diffusion, Scheme, Payoff, M}` ~120 LOC THE consumer for slot 202-S5 — converts an SDE problem `dX = μ dt + σ dW` plus payoff into an MLMC-ready LevelSampler; L27 `mlmc/consumers/pde.go::PDELevelSampler{Mesh, Solver, QoI, RefineFactor}` ~120 LOC the consumer for stochastic-PDE problems where each level is a successively-refined FEM mesh — discretisation-error decays as h^p for p-th-order FEM; L28 `mlmc/consumers/bayes.go::BayesianInverseLevelSampler{Likelihood, Prior, MCMCChain, ChainLength}` ~120 LOC Hoang-Schwab-Stuart-2013 multilevel-Bayesian-inverse-problems where each level is a successively-refined forward-model — KEY: total cost of posterior-MCMC drops by an order of magnitude vs single-level).

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for MLMC surface:

| Surface | Path | Lines | MLMC relevance |
|---|---|---:|---|
| `MonteCarloIntegrate(f, dim, lower, upper, samples, rng)` | `calculus/calculus.go:244` | ~30 | **THE INJECTION POINT.** Vanilla single-level MC with O(N^{−½}). No level-hierarchy / variance-decay / cost-allocation. Natural caller-side replacement: `MLMCIntegrate(levelSampler, eps, ...)`. |
| `RK4Step / EulerStep / SolveODE` | `chaos/ode.go:36, 80, 100` | 132 | The deterministic ODE-substrate that an SDE-MLMC consumer (L26) builds on; coarse-fine pair is two calls at different `dt`. |
| `NormalPDF / NormalCDF / NormalQuantile` | `prob/distributions.go:32, 47, 67` | 120 | CDF building-block for inverse-CDF Gaussian sampling but NO direct sampler. |
| `Sample(rng RNG) float64` for any distribution | -- | **0** | **ABSENT — repo-wide.** The same P0 blocker as 117/202/215/227/263. Gates every MLMC primitive that needs Brownian increments. |
| **Multilevel Monte Carlo (Giles 2008)** | -- | **0** | **ABSENT.** ~3,800 citations unrepresented. |
| **Optimal-N_l allocation** `N_l ∝ √(V_l/C_l)` | -- | **0** | **ABSENT.** |
| **Bias-extrapolation** `\|E[Y_L]\| M^α / (M^α − 1)` | -- | **0** | **ABSENT.** |
| **Antithetic MLMC (Giles-Szpruch 2014)** | -- | **0** | **ABSENT.** |
| **Paired Brownian increments** | -- | **0** | **ABSENT.** Single most-error-prone MLMC implementation detail. |
| **Randomized MLMC (Rhee-Glynn 2015)** | -- | **0** | **ABSENT.** Unbiased single-term estimator. |
| **Continuation MLMC (Collier et al. 2015)** | -- | **0** | **ABSENT.** Adaptive-tolerance schedule. |
| **Multi-Index MC / MIMC (Haji-Ali-Nobile-Tempone 2016)** | -- | **0** | **ABSENT.** Tensor-product level hierarchies. |
| **Multilevel QMC (Kuo-Schwab-Sloan 2012)** | -- | **0** | **ABSENT.** Cross-link to 263-Q25. |
| **Multi-fidelity MC (Peherstorfer-Willcox-Gunzburger 2016)** | -- | **0** | **ABSENT.** Cross-link to 227-U24. |
| **Approximate Control Variates (Gorodetsky et al. 2020)** | -- | **0** | **ABSENT.** Modern non-hierarchical multi-fidelity. |
| **MLMC for Bayesian inverse problems (Hoang-Schwab-Stuart 2013)** | -- | **0** | **ABSENT.** |
| **Adaptive level extension** | -- | **0** | **ABSENT.** |
| **Rule-of-three convergence theorem α ≥ ½min(β, γ−2)** | -- | **0** | **ABSENT** as a verifiable pin. |

**Bottom line:** zero MLMC surface. The `MonteCarloIntegrate` signature is the only natural injection point in the repo. `chaos/ode.go` provides the ODE substrate that an SDE-consumer builds on. Everything else is greenfield.

---

## 1. Theory crash-course (the math worth shipping)

### 1.1 The telescoping identity

Let `P` be the quantity-of-interest (e.g. option payoff, FEM solution L²-norm, posterior mean). Standard MC estimates `E[P]` via N i.i.d. samples at cost-per-sample C. For an SDE/PDE/integral with discretisation parameter h (timestep, mesh-size), the discrete approximation `P_h` has bias `E[P_h] − E[P] = O(h^β_weak)` and cost-per-sample `O(h^{−γ})`. Standard MC at tolerance ε needs `N = O(ε^{−2})` samples each at cost `O(ε^{−γ/β_weak})` ⇒ **total cost O(ε^{−2−γ/β_weak})**, e.g. O(ε^{−3}) for Euler-Maruyama (β_weak=1, γ=1).

Giles-2008 telescopes:

```
E[P_L] = E[P_0] + sum_{l=1}^L E[P_l − P_{l−1}]
```

with each level using h_l = h_0 · M^{−l} (typically M=2 or M=4). The variance V_l = Var(P_l − P_{l−1}) **DECAYS** with l for any strong-order-q scheme: V_l = O(h_l^{β}) for β = 2q. This is the magic — V_l dropping fast enough means the fine-level samples (which are expensive) need few replications.

### 1.2 Optimal allocation

Minimising total cost subject to total-variance ≤ ε²/2 yields the Lagrange-multiplier optimum

```
N_l = ⌈ 2 ε^{−2} sqrt(V_l C_l) sum_k sqrt(V_k / C_k) ⌉
```

(see Giles-2008 §2.3 derivation). Note this is identical structurally to Rao-Cramer / inverse-variance pooling — variance-and-cost proportional matching.

### 1.3 The complexity theorem (Rule of Three)

Giles-2015-Acta-Numer Theorem 1: under bias `|E[P_l − P]| = O(2^{−αl})`, variance `V_l = O(2^{−βl})`, cost-per-sample `C_l = O(2^{γl})`, plus `α ≥ ½ min(β, γ−2)`, MLMC achieves total cost

```
total cost = O(ε^{−2})              if β > γ
           = O(ε^{−2} log²ε)        if β = γ
           = O(ε^{−2−(γ−β)/α})      if β < γ
```

For Euler-Maruyama on a Lipschitz-payoff SDE: α=1, β=1, γ=1 ⇒ β = γ, log² regime. For Milstein: α=1, β=2, γ=1 ⇒ β > γ, full O(ε^{−2}). **This is the canonical pin: golden-file fits the log-log slope of total-cost-vs-ε against ε^{−2} ± 0.1 for Milstein-MLMC and ε^{−3} ± 0.1 for plain MC.**

### 1.4 Antithetic MLMC (Giles-Szpruch 2014)

For multi-D SDEs with non-commutative diffusion `σ(x)·dW`, Milstein needs Lévy-area iterated-integrals which are EXPENSIVE to simulate. The 2014 paper observes: if you average the standard fine-path `P_l^f` with a "twin" fine-path `P̃_l^f` that uses the SAME coarse Brownian-increments `ΔW^c` but with the fine-increments swapped (e.g. for M=2 swap (ΔW₁, ΔW₂) → (ΔW₂, ΔW₁)), the Lévy-area errors CANCEL in expectation while the variance-decay survives. Result: Milstein-strong-order MLMC without Lévy areas. ~3x speedup for multi-D SDE pricing.

### 1.5 Randomized MLMC (Rhee-Glynn 2015)

Standard MLMC has finite L (bias never zero). The trick:
- Define p_l = sampling-prob for level l, with sum p_l = 1
- For each MC-trial: draw random level L ~ p, then return `Δ_L / p_L`
- E of single-trial estimator = sum_l p_l · E[Δ_L]/p_L = sum_l E[Δ_l] = E[P]

UNBIASED, no truncation. Optimal `p_l ∝ sqrt(V_l/C_l)` minimises work-normalised variance. Finite-variance iff `sum_l V_l/p_l < ∞` (requires β > γ from §1.3). Useful when bias-tolerance is unknown and budget is set by wall-clock.

### 1.6 Multi-Index MC (Haji-Ali-Nobile-Tempone 2016)

For d-D PDE-with-stochastic-coefficient: the level "l" splits into per-dimension levels α = (α_1, ..., α_d). Mixed-difference operator

```
Δ_α P = ∏_{i=1}^d (P^{α_i} − P^{α_i−1})  applied as tensor-product
```

Telescoping over a downward-closed index-set `I` recovers `E[P]` with bias-from-truncation. Optimal index-set is hyperbolic-cross or total-degree depending on smoothness. Critical for stochastic-PDE problems where MLMC's curse-of-dim makes single-level d-D MC catastrophic.

### 1.7 Multi-fidelity (Peherstorfer-Willcox-Gunzburger 2016)

Drop the assumption that levels form a discretisation-hierarchy. Instead, k different models (e.g. full-NS, RANS, surrogate-NN, polynomial-regression) with different costs C_i and correlations ρ_i with the high-fidelity. Optimal estimator

```
Y = P_1 + sum_{i=2}^k α_i (P_i − P_i^{prev-correlation-model})
```

with α_i = ρ_i · σ_1/σ_i. Allocates more samples to cheap-correlated models. Goes beyond MLMC: even "wrong" surrogates (low ρ_i) help if very cheap.

---

## 2. Twenty-eight primitives L1–L28 by tier

### Tier-0 substrate (~80 LOC NEW shared with 202/263)

#### L0a. `prob/random/normal.go::SampleNormal(rng RNG) float64` — ~80 LOC
**Same blocker as 117/202/215/227/259/260/261/262/263.** Co-ship once, amortise across all. Marsaglia-polar default; cross-language Ziggurat opt-in for golden-file parity.

### Tier-1 Giles-2008 keystone (~440 LOC)

| ID | File | LOC | Pin |
|----|------|----:|-----|
| L1 | `mlmc/estimator.go::Estimator` | 180 | Giles-2008 Algorithm 1 driver, on-the-fly L extension |
| L2 | `mlmc/estimator.go::SampleLevel(l, dN, sampler) (sumP, sumP2)` | 80 | Per-level batched sampler with paired RNG |
| L3 | `mlmc/estimator.go::OptimalAllocation(V, C, eps) []int` | 60 | Closed-form `N_l = 2 ε^{−2} sqrt(V_l C_l) Σ sqrt(V_k/C_k)` |
| L4 | `mlmc/estimator.go::BiasExtrapolation(meanY, M) float64` | 40 | Richardson `bias ≤ \|E[Y_L]\| M^α/(M^α−1)` |
| L5 | `mlmc/levelsampler.go::LevelSampler` interface | 80 | Canonical interface every consumer implements |

### Tier-2 Antithetic MLMC (Giles-Szpruch 2014, ~360 LOC)

| ID | File | LOC | Pin |
|----|------|----:|-----|
| L6 | `mlmc/antithetic.go::AntitheticLevelSampler` | 120 | Swap-pair Brownian-increments, half-sum estimator |
| L7 | `mlmc/antithetic.go::TruncatedAntithetic` | 60 | Truncate when antithetic pair degenerate (jumps) |
| L8 | `mlmc/antithetic.go::SymmetricAntithetic` | 80 | Both coarse and fine antithetised |
| L9 | `mlmc/antithetic.go::PairBrownianIncrements(...)` | 100 | THE keystone — coarse[k] = sum(fine[k·M:(k+1)·M]) |

### Tier-3 Randomized MLMC (Rhee-Glynn 2015, ~360 LOC)

| ID | File | LOC | Pin |
|----|------|----:|-----|
| L10 | `mlmc/randomized.go::RandomizedEstimator` | 140 | Unbiased single-term `Δ_L / p_L` |
| L11 | `mlmc/randomized.go::OptimalLevelDist(V, C) []float64` | 80 | `p_l ∝ sqrt(V_l/C_l)` |
| L12 | `mlmc/randomized.go::SingleTermRandomized(...)` | 60 | Bypass level-budget allocation entirely |
| L13 | `mlmc/randomized.go::DoubleRandomized(...)` | 80 | McLeish-2011 noisy-objective extension |

### Tier-4 Continuation MLMC (Collier et al. 2015, ~360 LOC)

| ID | File | LOC | Pin |
|----|------|----:|-----|
| L14 | `mlmc/continuation.go::ContinuationDriver` | 160 | Adaptive-tolerance schedule reusing samples |
| L15 | `mlmc/continuation.go::AdaptiveTolerance(...)` | 80 | `ε_{k+1} = max(ε/2, ε_k/r)` |
| L16 | `mlmc/continuation.go::AdaptiveLevelExtension(...)` | 60 | Estimate α from log-log slope |
| L17 | `mlmc/continuation.go::PostStratifiedEstimator` | 60 | Heinrich-Sindambiwe-2009 |

### Tier-5 Multi-Index MC (Haji-Ali-Nobile-Tempone 2016, ~480 LOC)

| ID | File | LOC | Pin |
|----|------|----:|-----|
| L18 | `mlmc/mimc.go::MIMCEstimator` | 160 | Multi-D index α with mixed-differences |
| L19 | `mlmc/mimc.go::TotalDegreeIndexSet(L) [][]int` | 80 | `\|α\| ≤ L` truncation |
| L20 | `mlmc/mimc.go::HyperbolicCrossIndexSet(L) [][]int` | 80 | Optimal for analytic functions |
| L21 | `mlmc/mimc.go::AdaptiveProfit(stats) []int` | 80 | Adaptive-extension by max-profit-marker |
| L22 | `mlmc/mimc.go::OptimalMIMCAllocation(V, C) [][]int` | 80 | Multi-D analog of L3 |

### Tier-6 Multilevel-QMC + Multi-Fidelity (~480 LOC)

| ID | File | LOC | Pin |
|----|------|----:|-----|
| L23 | `mlmc/qmc.go::MultilevelQMC` | 160 | Kuo-Schwab-Sloan-2012; CROSS-LINK 263-Q25 |
| L24 | `mlmc/mfmc.go::MultiFidelityMC` | 200 | Peherstorfer-Willcox-Gunzburger 2016; CROSS-LINK 227-U24 |
| L25 | `mlmc/mfmc.go::ApproximateControlVariates` | 120 | Gorodetsky-Geraci-Eldred-Jakeman 2020 |

### Tier-7 SDE/PDE/Bayes consumers (~360 LOC)

| ID | File | LOC | Pin |
|----|------|----:|-----|
| L26 | `mlmc/consumers/sde.go::SDELevelSampler` | 120 | Cross-link 202-S5 |
| L27 | `mlmc/consumers/pde.go::PDELevelSampler` | 120 | FEM-mesh refinement; opens stochastic-PDE corpus |
| L28 | `mlmc/consumers/bayes.go::BayesianInverseLevelSampler` | 120 | Hoang-Schwab-Stuart 2013 |

**Total:** L0a substrate ~80 LOC + L1-L28 primitives ~3,200 LOC = **~3,280 LOC**.

---

## 3. Connective tissue per primitive

| Primitive | Calls | Called-by |
|-----------|-------|-----------|
| L1 Estimator | L2, L3, L4 internally | top-level user driver |
| L2 SampleLevel | L5 LevelSampler.SampleLevel | L1 |
| L3 OptimalAllocation | none | L1 |
| L4 BiasExtrapolation | none | L1 (termination check) |
| L5 LevelSampler interface | RNG | L26, L27, L28 (consumer impls), L1 (driver) |
| L6 AntitheticLevelSampler | L9 PairBrownianIncrements + user-supplied scheme | L1 |
| L9 PairBrownianIncrements | L0a SampleNormal | L6, L7, L8 |
| L10 RandomizedEstimator | L11, L12 | top-level (alternative to L1) |
| L11 OptimalLevelDist | none | L10 |
| L14 ContinuationDriver | L1 internally | top-level (wraps L1) |
| L18 MIMCEstimator | L19/L20/L21, L22 | top-level |
| L23 MultilevelQMC | L1 + 263-Q2 Sobol + 263-Q12 OwenScramble | top-level |
| L24 MultiFidelityMC | L0a + correlation matrices | top-level |
| L26 SDELevelSampler | 202-S1 Euler-Maruyama OR 202-S2 Milstein | L1, L6 |
| L27 PDELevelSampler | external FEM solver | L1 |
| L28 BayesianInverseLevelSampler | external MCMC | L1 |

**Cross-package edges:**

| Edge | LOC of glue | What it unlocks |
|------|------------:|-----------------|
| `mlmc/ → calculus/MonteCarloIntegrate` | 0 — already callable | Compare MC vs MLMC for educational integrals |
| `mlmc/ → chaos/ode.go` (RK4 substrate for SDE) | 0 — already callable | Stochastic-perturbation MLMC studies |
| `mlmc/ → 202-sde/EulerMaruyama,Milstein` | 0 — both consume LevelSampler | THE primary consumer of MLMC |
| `mlmc/ → prob/random/normal.go` (L0a) | 0 | Brownian increment generation |
| `mlmc/ → 263-qmc/{Sobol,OwenScramble}` | 0 — direct consumer of Sequence interface | L23 MultilevelQMC |
| `mlmc/ → optim/proximal/admm.go` | ~30 — adapter | L25 ACV multi-fidelity QP |
| `mlmc/ → 227-uq/sobol/SobolIndices` | ~40 — adapter | MLMC-Sobol-indices for UQ |
| `mlmc/consumers/pde.go → ?` | external FEM stub | OPEN: reality has no FEM solver yet |

---

## 4. Three architectural recommendations

**F1. Ship `mlmc/` as a sibling package to `prob/`, NOT inside `prob/` or `calculus/`.** Rationale: MLMC is structurally an algorithm-with-budget, not a probability primitive. Its consumers span SDE (chaos/sde), PDE (future), Bayesian-inverse (future), generic-MC. Placing under `prob/` confuses distribution-theory with budget-allocation; placing under `calculus/` ties it to integration when most consumers are non-integral. Sibling-package matches the architectural pattern of `optim/proximal/`.

**F2. Establish `mlmc.LevelSampler` as the canonical consumer interface.** Single 5-method interface:

```go
type LevelSampler interface {
    // SampleLevel returns coarse and fine payoffs for level l using paired RNG state.
    // For l == 0, Pf is the level-0 sample and Pc is 0 (treated as P_-1 = 0).
    // For l >= 1, returns (P_l, P_{l-1}) with paired Brownian increments.
    SampleLevel(l int, rng RNG) (Pf, Pc float64)

    // CostPerSample returns the relative cost-per-sample at level l.
    // Used by OptimalAllocation. Typically 2^l for SDE-with-Δt = 2^{-l}.
    CostPerSample(l int) float64

    // RefinementFactor returns M (e.g. 2 or 4 for SDE).
    // Used by BiasExtrapolation for Richardson convergence.
    RefinementFactor() int
}
```

Every consumer (SDE, PDE, Bayes, generic-integral) implements exactly this. Avoids the API explosion of `MLMCEstimateSDE`, `MLMCEstimatePDE`, etc. Pattern-matches `optim/proximal.ProxOp` interface.

**F3. Pin convergence-complexity claims via golden files, not just function correctness.** Total-cost-as-function-of-ε for plain MC is `O(ε^{-3})`; MLMC + Milstein is `O(ε^{-2})`; MLMC + Euler-Maruyama is `O(ε^{-2} log²ε)`. **Golden file ships GBM E[max(S_T − K, 0)] (call option) at ε ∈ {1/16, 1/32, 1/64, 1/128, 1/256} and validates the log-log slope:**

| Method | Expected slope | Tolerance |
|--------|----------------|-----------|
| Plain MC + Euler-Maruyama | −3.0 | ±0.15 |
| MLMC + Euler-Maruyama | −2.0 (modulo log² correction) | ±0.15 |
| MLMC + Milstein | −2.0 | ±0.10 |
| Antithetic MLMC + Euler-Maruyama | −2.0 (no Lévy areas) | ±0.10 |

This is the cross-language parity contract that proves the MLMC implementation actually delivers the claimed complexity, not just numerically reasonable answers. Giles-2008 Figure 5 / Giles-2015 Figure 3.1 are the canonical citations.

---

## 5. Risks and gotchas

- **G1. Brownian-increment-pairing.** MLMC pairs coarse + fine paths on the SAME Brownian path. Naïve regen breaks variance-decay. Ship `BrownianIncrementCache` shared between coarse and fine simulators (L9). **#1 source of MLMC bugs in the wild.**
- **G2. Variance-decay assumption is scheme-dependent.** Euler-Maruyama gives β=1 ⇒ β=γ ⇒ log²-regime; Milstein gives β=2 ⇒ β>γ ⇒ optimal-regime. Document per-scheme; expose `Estimator.RecommendedScheme(payoff PayoffType)`. For non-Lipschitz payoffs (digital options) Euler-Maruyama gives β=½ ⇒ β<γ ⇒ MLMC is *worse* than plain MC unless antithetic; LOUD-FAIL if Estimator detects β<γ−2 over 5+ levels.
- **G3. Antithetic-pair degeneracy for jumps.** When a level contains a jump (Poisson Bernoulli), the antithetic-pair averaging breaks because the jump indicator differs between paths. Ship L7 TruncatedAntithetic that detects and handles.
- **G4. Optimal allocation N_l rounding.** N_l = ⌈x⌉ over-spends; the standard fix is "round-and-truncate-largest-deficit" (Giles-2015 §2.4). Document or ship.
- **G5. Bias-extrapolation needs estimated α.** Richardson's `bias ≤ |E[Y_L]| M^α/(M^α−1)` requires α known. For unknown α, fit log-log slope from |E[Y_l]| vs l (L16 AdaptiveLevelExtension). Pin against Giles-2008 §6 numerics.
- **G6. Continuation MLMC sample-reuse subtle.** When extending L by 1, prior samples on levels 0..L are still valid IF the optimal-N_l for new ε ≤ existing-N_l (most of the time true since variance estimates are now better). Document the "samples are immutable; only N_l grows" invariant.
- **G7. MIMC index-set must be downward-closed.** Out-of-order index addition breaks the telescoping. Encode invariant in the index-set primitive, not in the driver.
- **G8. Multi-fidelity correlation matrix ill-conditioning.** For highly-correlated low-fidelity models, the Σ matrix is near-singular and the optimal-α weights blow up. Ship pseudo-inverse fallback via SVD (CROSS-LINK to 097-linalg-missing SVD).
- **G9. Rule-of-three boundary case β=γ has the log² penalty.** Document loudly that "MLMC + Euler is O(ε^{−2} log²ε), NOT O(ε^{−2})" — many users assume the latter. The log² is only ~3-4x at typical ε so practically fine, but the asymptotic slope claim must be honest.

---

## 6. Cross-language parity targets

Eight pinned tests covering the variance-decay-rate, optimal-allocation, antithetic-cancellation, and unbiased-randomized claims:

| Test | Pin | Tolerance | Reference |
|------|-----|-----------|-----------|
| `TestMLMC_VarianceDecay_GBM_Milstein` | log-log slope of V_l vs l = −2.0 (β=2) | ±0.1 on slope | Giles-2008 §6 |
| `TestMLMC_VarianceDecay_GBM_EulerMaruyama` | log-log slope = −1.0 (β=1) | ±0.1 on slope | Giles-2008 §6 |
| `TestMLMC_TotalComplexity_GBM_Milstein` | total-cost slope vs ε = −2.0 | ±0.15 | Giles-2008 Fig 5 |
| `TestMLMC_TotalComplexity_GBM_EulerMaruyama` | total-cost slope = −2.0 modulo log² correction | ±0.2 | Giles-2008 Fig 5 |
| `TestAntitheticMLMC_VarianceDecay_2D_NonCommutative` | β=2 without Lévy areas | ±0.15 | Giles-Szpruch 2014 Thm 4.10 |
| `TestRandomizedMLMC_Unbiased_GBM_Call` | empirical mean within ε of BS-closed-form, NO truncation | 95% CI coverage | Rhee-Glynn 2015 §3 |
| `TestOptimalAllocation_MatchesGilesFormula` | N_l = ⌈2 ε^{−2} sqrt(V_l C_l) sum sqrt(V_k/C_k)⌉ | bit-exact | Giles-2008 §2.3 |
| `TestMIMC_HyperbolicCrossDownwardClosed` | invariant pin: every α ∈ I has all sub-multi-indices in I | bit-exact | Haji-Ali-Nobile-Tempone 2016 |

---

## 7. Verdict

**Ship Tier-1 (Giles-2008 keystone, ~440 LOC):** L1-L5 in one PR. Pair with 202-S5 SDE-MLMC consumer (which is L26 here) — that's the *only* consumer reality has substrate for today, but it's also the canonical demo.

**Ship Tier-2 next sprint (~360 LOC):** L6-L9 antithetic. The keystone result — Milstein-strong-order in multi-D without Lévy areas — is a genuine modern (post-2014) capability that no zero-dep Go library has.

**Defer-but-design Tier-3-4 (~720 LOC):** L10-L17 randomized + continuation. Both are research-frontier extensions with a 2-3 paper depth. Continuation has higher industrial pull (real-world workflows often refine ε); randomized has higher elegance (unbiased estimator).

**Defer until consumer pulls Tier-5-7 (~1,320 LOC):** L18-L28. MIMC requires a multi-D PDE consumer that reality does not have today. Multi-fidelity requires a model-hierarchy that reality does not have. Bayesian-inverse requires an MCMC backend that reality does not have. Each lands when the *consumer* lands; today they're scope-creep.

**Single-highest-leverage 1-day project:** L1 + L2 + L3 + L4 + L5 + L26 (~480 LOC including the SDE consumer wrapper) on top of 202-S0a/S1/S2 substrate. This delivers MLMC-for-SDE-Asian-options with Milstein and the canonical Giles-2008 Figure 5 reproduction, ALL of the convergence-complexity pin claims of §6, and a directly-callable surface for any consumer who has an SDE problem.

**Single-highest-leverage cutting-edge piece:** L6 + L9 antithetic-MLMC. The 2014 Giles-Szpruch result is the textbook example of "modern variance-reduction beats clever-numerics" — getting Milstein-rate without Lévy-areas is a 3-5x wall-clock saving that the entire competitive landscape (QuantLib, Numerix, Bloomberg-models) has implemented since 2015 but no zero-dep cross-language Go/Python/C++/C# library ships today. **Antithetic MLMC is the flagship deliverable for this slot.**

**Cross-slot ship-once amortisations:**
- L0a Gaussian sampler: ship with 202-S0a (zero net cost here)
- L23 MultilevelQMC: ship with 263-Q25 (was already designated cross-link; no double-implementation)
- L24 MFMC: ship with 227-U24 (was already designated cross-link; no double-implementation)
- L26 SDELevelSampler: ship with 202-S5 (the SDE-side wrapper that 202 named but bound to its own scope)

After ship-once amortisation, **net-new MLMC LOC for slot 264 is ~2,500 LOC** of pure-MLMC algorithm + framework, pure connective tissue, no novel substrate required beyond the 80-LOC sampler that 9+ slots demand.
