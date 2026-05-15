# 195 | synergy-optim-prob (noisy-optim angle)

**Topic:** optim × prob — SGD-as-Langevin, stochastic approximation, MCMC-as-optimisation, sample-complexity bounds.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** the **noisy-optim half** of the optim×prob join (169 owns the deterministic fit-distribution-to-data half: MLE/EM/MAP/VI/BO). This review covers what *requires* injected noise, finite-sample stochastic gradients, or interpreting an SDE as a sampler.

## Two-line summary

`optim/` ships ZERO stochastic-gradient anything (no SGD, no Adam, no SGLD, no SVRG, no Robbins-Monro `1/k` schedule); the only RNG-aware functions in the entire repo are `optim.GeneticAlgorithm` (BLX-α evolutionary) and `optim.SimulatedAnnealing` (Boltzmann acceptance with geometric cooling — already 80 % of the way to a Metropolis sampler but exposed as a minimiser, not a sampler), while `prob/` ships ZERO `Sample(rng)` API anywhere (verified — no `Float64()`-taking RNG interface in `distribution.go`/`distributions.go`/`prob.go`/`markov.go`; the entire MCMC-from-Gibbs-measure pipeline is **wholly absent**: no Metropolis-Hastings, no Hamiltonian-Monte-Carlo, no Gibbs sampler, no slice sampler, no NUTS, no thinning, no ESS/R-hat diagnostics, no Welford streaming variance for control variates, no Polyak-Ruppert tail-averaging, no Robbins-Siegmund convergence verification). **Twenty-two synergy primitives (N1-N22) totalling ~3,150 LOC of pure connective tissue** stand up the entire SDE/SA/SG-MCMC/CMA-ES/IGO canon on existing `optim.SimulatedAnnealing`'s temperature-schedule + Box-Muller normal-from-`Float64()` (already inlined in `optim/genetic.go:58-65`); cheapest one-day standalone is **N1 RobbinsMonroSGD + N2 LangevinNoiseInjection (~280 LOC)** which converts the existing `GradientDescent` loop into a Welling-Teh-2011 SGLD sampler with a one-line addition `x[i] -= lr*g[i] + sqrt(2*lr*T)*normal()`; highest-leverage one-week unlock is **N9 CMA-ES (~420 LOC)** because Hansen's covariance-matrix-adaptation evolution-strategy is **the** derivative-free optimiser the repo is missing and it natural-gradient-descends on a Gaussian search distribution (cross-link to 153 infogeo for the Fisher-information matrix); the architectural keystone is **N3 RNGSampler interface (~80 LOC)** — a single `Float64() float64`-method abstraction matching `optim.GeneticAlgorithm`'s precedent — gating every Tier-2/3 primitive (N4-N22) and unlocking 169-S14 (PriorSamples) at zero marginal cost.

---

## 0. State of play (verified file-walk)

### `optim/` stochastic surface = SA + GA only

Two `*math/rand.Rand`-or-`interface{Float64()float64}`-aware functions in the entire package:

- `optim/metaheuristic.go:38-93` — `SimulatedAnnealing(f, x0, neighbor, temp0, cooling, maxIter, rng) → (best, bestF)`. **Boltzmann acceptance `P = exp(−Δ/T)`** is mathematically the Metropolis criterion; the cooling factor is geometric `T ← α·T`. Five lines from being a Markov-chain Monte Carlo sampler at fixed T (just remove the cooling step and emit `current` at every iteration instead of `best`).
- `optim/genetic.go:44-178` — `GeneticAlgorithm` BLX-α + Gaussian-mutation; uses inlined Box-Muller `sqrt(-2 log u1) cos(2π u2)` per `genetic.go:58-65`. **The only Gaussian sampler in the repo lives here**, hidden behind no public API.

ZERO matches across `optim/` for any of: `SGD`, `Adam`, `Adagrad`, `RMSprop`, `Lion`, `Lookahead`, `Langevin`, `SGLD`, `SGHMC`, `MALA`, `Metropolis`, `HMC`, `NUTS`, `Gibbs`, `slice`, `RobbinsMonro`, `KieferWolfowitz`, `PolyakRuppert`, `Polyak`, `SVRG`, `SAGA`, `SAG`, `MISO`, `IGO`, `CMA`, `xNES`, `sNES`, `cross.entropy`, `CEM`, `Frank.?Wolfe`, `mirror.descent`, `proximal.point`. Verified by grep.

### `prob/` stochastic surface = ZERO

ZERO matches across `prob/` for `Sample`, `RNG`, `*rand.Rand`, `MetropolisHastings`, `HamiltonianMonteCarlo`, `Gibbs`, `slice`, `MCMC`, `R-hat`, `ESS`, `effective.sample.size`, `thinning`, `burn.in`, `MALA`, `NUTS`, `importance.sample`, `rejection.sample`, `inverse.transform`, `Marsaglia`, `Cheng`, `Knuth.poisson`, `Box.Muller` (outside the inlined copy in `optim/genetic.go`), `Welford`, `Kahan`, `running.variance`, `control.variate`. The seven `Distribution`s have only `PDF/CDF`; `prob/markov.go:99-139:MarkovSimulate` uses a hard-coded LCG for "deterministic" sampling but exposes no RNG abstraction. **Markov chain Monte Carlo is wholly absent from a package called `prob/` that ships `MarkovSteadyState`** — the irony is exactly the gap.

### Cross-coupling: zero today (same as 169)

```
$ grep -r "github.com/davly/reality/prob"  optim/ ; echo "---"
$ grep -r "github.com/davly/reality/optim" prob/
---
(no matches in either direction)
```

---

## 1. The conceptual unlock — `SimulatedAnnealing` is already 80 % of MCMC

The Kirkpatrick-Gelatt-Vecchi-1983 acceptance test `P_accept = min(1, exp(−Δf/T))` for symmetric proposals is **literally the Metropolis 1953 criterion**; the only structural difference in `optim/metaheuristic.go` is (a) the cooling factor multiplies `T` toward zero instead of staying at the equilibrium `T*`, and (b) the function returns `bestX` instead of the trajectory `[]x_t`. Three additions convert it into a sampler:

1. **`MetropolisHastings(f → -log p, x0, proposal, T, n_burn, n_samples, rng) → []state`** at fixed T (~60 LOC, 80 % code-reuse from `SimulatedAnnealing`).
2. **`SimulatedAnnealingTemperatureSchedules { Geometric, Logarithmic, Cauchy, ... }`** — log-cooling `T_k = T_0/log(k)` is the Hajek-1988 condition for asymptotic global-optimum probability 1 (~30 LOC).
3. **Trajectory emission** — return `(samples, accept_rate)` not `(best, bestF)`.

This is the single highest-leverage observation in the review: **the repo already ships a Metropolis sampler under a different name**. Promoting it from "minimiser" to "sampler+minimiser" (with the function as `−log p`) is one PR.

The deeper Robbins-Monro 1951 bridge: **stochastic gradient descent with `lr_k = a/k` and finite-variance noise converges almost surely** (Σ a_k = ∞, Σ a_k² < ∞ — Robbins-Monro conditions). This is the same reason geometric-cooling SA converges to the global optimum: the temperature schedule satisfies the equivalent annealing condition. **Robbins-Monro 1951 + Welling-Teh 2011 + Hajek 1988 are the same theorem on three different SDEs.** Reality should pin this identity.

---

## 2. Synergy primitives (N1-N22, ~3,150 LOC pure glue, noisy-optim axis only)

Numbered ascending by composition-depth. Each lists (capability, composition of existing primitives, LOC).

### Tier 1 — ships today against v0.10.0 (no new infrastructure beyond N3 RNG)

**N1 RobbinsMonroSGD(grad_stoch, x0, a0, alpha, n_iter, rng) → x_T** [~150 LOC]
Robbins-Monro 1951 stochastic approximation `x_{k+1} = x_k − (a_0/(k+1)^α) · ĝ(x_k, ξ_k)` with `α ∈ (0.5, 1]`. Pure variant of `optim.GradientDescent` with (i) `lr` replaced by per-step schedule `a_k = a_0 · k^{−α}`, (ii) `grad` replaced by `grad_stoch(x, sample_idx, out)` for finite-sum problems. Saturates **R-CONVERGENCE-CONDITION** pin: Robbins-Monro 1951 conditions Σ a_k = ∞, Σ a_k² < ∞ verified at compile-time (`alpha > 0.5 && alpha <= 1.0` panic).

**N2 LangevinNoiseInjection(grad, x0, lr, T, n_iter, rng) → []x_t** [~130 LOC]
Welling-Teh 2011 SGLD: `x_{k+1} = x_k − (lr/2) ∇U(x_k) + sqrt(lr · T) η_k` with `η_k ~ N(0, I)`. Stationary distribution is Gibbs `π(x) ∝ exp(−U(x)/T)` (cross-link 180 stat-mech). One-line modification of `optim.GradientDescent` + Box-Muller normal sampler from `optim/genetic.go:58-65`. **Three identities pin the 3/3 R-MUTUAL-CROSS-VALIDATION:**
1. Stationary distribution at T=1 = `exp(−U)` measured by histogram on N=10⁵ post-burn-in samples vs analytic `Z⁻¹ exp(−U)` on `U(x) = x²/2` (pinned to 1e-3 KL).
2. Mean and variance of post-burn-in samples = Normal(0, 1) sufficient stats (pinned to 1e-2 with N=10⁴).
3. Asymptotic mode at T → 0 = `optim.LBFGS` solution on the same U (pinned to 1e-6 with `T_k = 1/log(k+e)` Hajek schedule).

**N3 RNGSampler interface** [~80 LOC, **architectural keystone**]
Single `interface{ Float64() float64 }` lifted from `optim.GeneticAlgorithm`'s anonymous parameter into a public `optim.RNG` (or `prob.RNG`) type with `*math/rand.Rand` adapter. Gates every primitive in this review and 169-S14 simultaneously. Per CLAUDE.md "Reimplement from first principles", `*math/rand.Rand` is stdlib, not a dependency.

**N4 MetropolisHastings(logp, x0, proposalSigma, n_burn, n_samples, rng) → []x_t** [~120 LOC]
Lift `optim.SimulatedAnnealing` to fixed T (= 1; absorb T into `logp`); replace `bestX` return with `samples[]`; emit acceptance rate as second return. Symmetric Gaussian proposal `q(x' | x) = N(x, σ²I)` makes the acceptance ratio `min(1, exp(logp(x') − logp(x)))` — already the form in `metaheuristic.go:73`. **Crown-jewel observation: `MetropolisHastings = SimulatedAnnealing` with cooling=1 and trajectory-emission.**

**N5 SimulatedAnnealingHajek(f, x0, neighbor, T0, n_iter, rng) → x_best** [~30 LOC]
Hajek-1988 logarithmic schedule `T_k = c / log(k+1)` with `c ≥ Δ_max` (max barrier height). Promotes `optim.SimulatedAnnealing`'s geometric cooling from "trades-off-heuristic" to "asymptotically-global-optimal-with-probability-1". Single-function addition; one new schedule type.

**N6 CrossEntropyMethod(f, n_dim, mu0, sigma0, popSize, eliteFrac, n_iter, rng) → x_best** [~150 LOC]
Rubinstein 1997 CEM for global optim: sample N candidates from `N(μ, Σ)`, keep top ρN by fitness, re-fit `(μ, Σ)` to elites, iterate. Pure compositional sibling of `optim.GeneticAlgorithm` with **Gaussian search distribution** (instead of population) and **MLE-on-elites update** (instead of crossover+mutation). Closed-form moment update is the M-step from a one-component GMM (cross-link 169-S5 EM-GMM). De Boer-Kroese-Mannor-Rubinstein 2005 tutorial.

**N7 KieferWolfowitzSA(f, x0, a0, c0, n_iter, rng) → x_T** [~110 LOC]
Kiefer-Wolfowitz 1952 — stochastic approximation when ∇f is unavailable; finite-difference gradient `ĝ_i = (f(x + c·e_i) − f(x − c·e_i))/(2c)` with `c_k → 0`. Composes `f` (caller-supplied) + Robbins-Monro `a_k → 0` schedule + central differences. Convergence requires `a_k/c_k² → 0` and Σ a_k = ∞ (different bookkeeping than RM 1951).

**N8 PolyakRuppertAveraging(sgdIter func() []float64, n_total) → x_avg** [~50 LOC]
Polyak-1990 / Ruppert-1988 tail-averaging: with `lr_k = a/k^α`, `α ∈ (0.5, 1)`, the average `x̄_T = (1/T) Σ_{k=1}^T x_k` achieves CLT with **optimal asymptotic variance** (the inverse Fisher information at optimum, equal to Cramér-Rao bound). Two-line wrapper around any iterator-returning SGD primitive. Saturates **R-CRAMER-RAO-PIN**: PR averaging on a logistic regression converges to MLE with variance equal to inverse Fisher (cross-link 153-S2 FisherFromDistribution).

**N9 CMA-ES(f, x0, sigma0, popSize, n_gen, rng) → x_best** [~420 LOC]
Hansen 2001 CMA-ES — covariance-matrix-adaptation evolution strategy. Per-generation: sample `λ` candidates from `N(m, σ²C)`, weight-rank-sort top μ, update `m, σ, C` via cumulation paths `p_σ, p_c` and rank-μ + rank-1 updates. **Information-geometric interpretation (Akimoto et al. 2010): CMA-ES is natural-gradient ascent on `E_{N(m, σ²C)}[−f]` w.r.t. the Fisher metric on Gaussian-with-mean-and-cov.** Composes:
- `linalg.Cholesky` (sample from `N(m, σ²C)` via `m + σ·L·z` with `z ~ N(0,I)`).
- `linalg.Eigen` (occasional decomposition for numerical stability of `C`).
- `optim.RNG` (N3) for `z`.
- Box-Muller from `optim/genetic.go:58-65`.

The de-facto-best black-box optimiser since 2001; ship as the canonical metaheuristic.

**N10 InformationGeometricOptimization(f, family, theta0, lr, popSize, n_iter, rng) → theta_T** [~290 LOC]
Ollivier-Arnold-Auger-Hansen 2017 IGO — natural-gradient ascent on `J(θ) = E_{p_θ}[w ∘ f]` where `w` is a quantile-based shaping function. CMA-ES (N9) is `IGO ∘ Gaussian-with-mean-and-cov` applied to a specific `(η, lr)` schedule; xNES, sNES, REINFORCE-with-baseline are the same IGO kernel applied to other parametric families. Composes `prob.Distribution.LogPDF` (169-S8), `prob.NaturalGradient` (153-S2 Fisher inverse), and `optim.GradientDescent` outer-loop.

**N11 NaturalEvolutionStrategies(f, mu0, sigma0, popSize, lr, n_iter, rng) → mu_best** [~180 LOC]
Wierstra-Schaul-Glasmachers-Sun-Peters-Schmidhuber 2014 NES — Fisher-rescaled gradient-of-search-distribution variant. Special case of IGO (N10) with diagonal Gaussian. Lower fixed-cost than CMA-ES; useful when `dim > 100` makes full `C` impractical.

### Tier 2 — needs N3 RNG + 169-S8 LogPDF

**N12 SG-HMC (Stochastic Gradient Hamiltonian Monte Carlo)** [~210 LOC]
Chen-Fox-Guestrin 2014 — SGHMC corrects the noise in the SGD Hamiltonian by adding friction:
```
v_{k+1} = (1−γlr)v_k − lr · ∇U(x_k) + sqrt(2lr·γ·T) η_k
x_{k+1} = x_k + lr · v_{k+1}
```
Underdamped Langevin = Hamiltonian + friction + Brownian noise. Composes N2 SGLD + momentum bookkeeping + Box-Muller. Stationary distribution is `exp(−H/T)` with `H = U + ‖v‖²/2` marginalised to `exp(−U/T)` in x. Saturates **R-DETAILED-BALANCE pin** at T = 1 (acceptance rate via Metropolis correction agrees to 1e-3 with no-correction limit lr → 0).

**N13 SG-NUTS (No-U-Turn-Sampler with stochastic gradients)** [~340 LOC]
Hoffman-Gelman 2014 NUTS adapted to mini-batch gradients. Doubling tree built via leapfrog integrator + U-turn detection; trajectory length adaptive. Composes N12 SGHMC kernel + binary-tree recursion + slice-sampling acceptance. Heaviest single primitive; gates production-grade Bayesian inference. Cross-link 173-prob-VI-via-stein.

**N14 SVRG (Stochastic Variance-Reduced Gradient)** [~180 LOC]
Johnson-Zhang 2013: epoch-based variance reduction. Per epoch keep snapshot `x̃` with full gradient `μ̃ = (1/N)Σ ∇f_i(x̃)`; per inner iter `x_{k+1} = x_k − lr · (∇f_{i}(x_k) − ∇f_{i}(x̃) + μ̃)`. Composes `optim.GradientDescent` outer loop + N3 RNG for sample index `i`. Crown achievement: linear convergence on smooth strongly-convex finite-sum problems where vanilla SGD gets `O(1/k)`.

**N15 SAGA (Stochastic Average Gradient Accelerated)** [~170 LOC]
Defazio-Bach-Lacoste-Julien 2014 — table-of-past-gradients SVRG variant; storage `O(N·d)` but no full-gradient passes after init. Same composition as N14 with table maintenance. Lands as sibling primitive showing the time-vs-space tradeoff in variance-reduced SGD.

**N16 Adam(grad_stoch, x0, lr, beta1, beta2, eps, n_iter, rng) → x_T** [~140 LOC]
Kingma-Ba 2015 Adam — adaptive-moment SGD with bias correction. **Robbins-Monro 1951 interpretation: `m̂_k/√(v̂_k)` is a per-coordinate Robbins-Monro update with adaptive step `lr/√(v̂_k)` that satisfies the RM conditions in expectation** (Reddi-Kale-Kumar 2018 caveat: Adam without amsgrad-correction can fail to converge on certain non-convex problems; document this). Composes per-coordinate moment buffers + bias correction. **The single SGD variant the repo cannot ethically omit**; cross-link 163-A12 (the autodiff-side mention).

**N17 AdamW(grad_stoch, x0, lr, beta1, beta2, eps, weightDecay, n_iter, rng) → x_T** [~30 LOC]
Loshchilov-Hutter 2019 — decouples L2 regularisation from gradient. Three-line modification of N16; document the precision hazard (vanilla "Adam + L2" is mathematically not Adam-on-MAP).

**N18 Adagrad / RMSprop** [~80 LOC each, ~160 LOC total]
Duchi-Hazan-Singer 2011 / Hinton 2012 — historical / EMA-of-squared-gradient adaptive step. Two siblings of N16 with different `v_k` recurrences. Lands together for completeness; explicit Robbins-Monro framing in docstrings.

**N19 ImportanceSamplingFiniteSum(grad_per_sample, x0, lr, weights, n_iter, rng) → x_T** [~130 LOC]
Zhao-Zhang 2015 / Needell-Srebro-Ward 2014 importance-sampling SGD: sample index `i` with probability proportional to `‖∇f_i‖` (or its Lipschitz upper-bound `L_i`); reweight by `1/(N·p_i)` to maintain unbiasedness. Provably faster than uniform sampling when Lipschitz constants vary across samples. Composes `prob.CategoricalSample` (new 30-LOC inverse-CDF over weights) + N1 Robbins-Monro outer loop.

**N20 ControlVariateSGD(grad_per_sample, baseline_grad, x0, lr, n_iter, rng) → x_T** [~110 LOC]
General control-variate variance reduction: replace `∇f_i(x)` with `∇f_i(x) − c·(b_i(x) − E[b_i(x)])` where `c = Cov(∇f, b)/Var(b)`. Composes `prob.WelfordVariance` (new 40-LOC streaming) + N1. Generalises SVRG (where `b = ∇f_i(x̃)` is the snapshot baseline).

**N21 OrnsteinUhlenbeckSGD-asymptotic-analysis** [~150 LOC test-pin only]
Mandt-Hoffman-Blei 2017 — small-step SGD on a quadratic loss `L(x) = ½ xᵀHx` with mini-batch noise covariance `B/N` converges to an Ornstein-Uhlenbeck process `dx = −Hx dt + sqrt(lr · B/N) dW`, with stationary distribution `N(0, lr·H⁻¹·B/(2N))`. **Pin** as a saturation test: run N1 RobbinsMonroSGD on a quadratic `L(x) = xᵀHx/2` with controlled per-sample noise; verify the stationary covariance matches the OU formula to 1e-3 over 10⁵ steps. Cross-link 191 chaos-control (OU is the Mandel-Sevcik-Stratonovich linear SDE).

**N22 PAC-BayesGeneralizationBound(prior, posterior, N, delta) → (bound, evidence)** [~210 LOC]
Catoni 2007 / Dziugaite-Roy 2017 PAC-Bayes bound: `Pr[ E[L_test] ≤ E_q[L_train] + sqrt((KL(q‖p) + log(2N/δ))/(2N)) ] ≥ 1−δ`. Composes `prob.KLDivergenceNumerical` (already shipped per `distribution.go`) + posterior `q` from BBVI/SGLD trajectory + prior `p`. Provides a **non-vacuous test-loss certificate** from training samples without held-out data — the right finale for the SGD-as-Langevin pipeline because the SGLD posterior is the natural `q`. Cross-link 189 MDL.

---

## 3. Composition graph (DAG)

```
N3 RNGSampler ──── architectural keystone ──── gates ALL Tier 1 + Tier 2
 │
 ├── N1 RobbinsMonroSGD                                                 ┐
 │    ├── N7 KieferWolfowitz   (FD-gradient variant)                    │
 │    ├── N8 PolyakRuppert     (averaging wrapper)                      │
 │    ├── N14 SVRG             (variance-reduced)                       │
 │    ├── N15 SAGA             (variance-reduced + table)               │
 │    ├── N16 Adam ── N17 AdamW                                         │── consumer-side
 │    ├── N18 Adagrad / RMSprop                                         │   loops on
 │    ├── N19 ImportanceSamplingSGD                                     │   stochastic
 │    └── N20 ControlVariateSGD                                         │   gradient
 │                                                                      ┘
 ├── N2 LangevinNoiseInjection (SGLD, Welling-Teh 2011)                 ┐
 │    ├── N12 SG-HMC                                                    │── samplers from
 │    └── N13 SG-NUTS                                                   │   Gibbs measure
 │                                                                      ┘
 ├── N4 MetropolisHastings ─── lifts SimulatedAnnealing to fixed T      ┐
 │    └── N5 SAHajek         ─── log-cooling promotion                  │── trajectory
 │                                                                      ┘   primitives
 │
 ├── N6 CrossEntropyMethod ─── Gaussian-search-dist global optim        ┐
 │    ├── N9 CMA-ES        (full-cov natural gradient)                  │── evolutionary
 │    ├── N10 IGO         (natural-grad on any expon. family)           │   / IGO
 │    └── N11 NES          (diagonal-cov NES)                           ┘
 │
 ├── N21 OU-asymptotic-analysis ─── stationary-covariance pin
 │
 └── N22 PAC-BayesBound ─── certificate-from-SGLD-trajectory ────────── crown-jewel composition
```

---

## 4. Saturation pins this review unlocks

Per recent saturation pattern (audio-onset 3-detector 6a55bb4, copula×autodiff 365368a, NGramDice 85a80db):

- **R-MUTUAL-CROSS-VALIDATION 3/3 on N2 SGLD stationary distribution:** histogram-from-trajectory × analytic `Z⁻¹ exp(−U)` × `optim.LBFGS` mode-finder all agree on a 1D Gaussian / 2D banana / mixture-of-two-Gaussians benchmark. Three orthogonal sampling/optimisation paths converging to the same `(mean, cov, mode)` is the canonical R-MUTUAL idiom on the SDE-as-sampler axis.
- **R-IDENTITY-OF-SDE-LIMITS 3/3:** Robbins-Monro 1951 SA × Welling-Teh 2011 SGLD × Hajek 1988 SA-with-log-cooling all share the same Σ a_k = ∞, Σ a_k² < ∞ + cooling rate condition. Pin one Lyapunov function on the harmonic oscillator U = x²/2 and verify all three primitives' trajectories satisfy the same convergence-in-distribution bound to 1e-2 in W2 distance.
- **R-MANDT-OU-STATIONARY-COVARIANCE pin (N21):** small-step SGD on quadratic loss with controlled mini-batch noise covariance vs analytic OU `Σ_∞ = lr·B/(2N)·H⁻¹` agrees to 1e-3 over 10⁵ steps. **First pin in the repo of an SDE-limit theorem.**
- **R-CMA-ES-NATURAL-GRADIENT pin (N9):** Akimoto et al. 2010 identity — CMA-ES rank-μ update equals Fisher-rescaled gradient ascent on `E_{N(m, σ²C)}[w∘f]` to 1e-9 on a quadratic. Cross-validates 153-S2 FisherFromDistribution.
- **R-PAC-BAYES-CERTIFICATE pin (N22):** Dziugaite-Roy 2017 — non-vacuous PAC-Bayes bound on a 1-hidden-layer logistic regression with SGLD posterior agrees with held-out test error to within 5 % on MNIST-3-vs-5 binary subset (cross-language pin via golden-file).

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| N1 | RobbinsMonroSGD | 150 | 1 | N3 |
| N2 | LangevinNoiseInjection (SGLD) | 130 | 1 | N3 |
| N3 | RNGSampler interface | 80 | 1 | — (keystone) |
| N4 | MetropolisHastings | 120 | 1 | N3 |
| N5 | SimulatedAnnealingHajek | 30 | 1 | — |
| N6 | CrossEntropyMethod | 150 | 1 | N3 |
| N7 | KieferWolfowitzSA | 110 | 1 | N3 |
| N8 | PolyakRuppertAveraging | 50 | 1 | N1 |
| N9 | CMA-ES | 420 | 1 | N3, linalg.Cholesky, linalg.Eigen |
| N10 | IGO (Ollivier 2017) | 290 | 1 | 169-S8 LogPDF, 153-S2 Fisher |
| N11 | NES (Wierstra 2014) | 180 | 1 | N3, 153-S2 Fisher |
| N12 | SG-HMC | 210 | 2 | N2 |
| N13 | SG-NUTS | 340 | 2 | N12 |
| N14 | SVRG | 180 | 2 | N1 |
| N15 | SAGA | 170 | 2 | N1 |
| N16 | Adam | 140 | 2 | N1 |
| N17 | AdamW | 30 | 2 | N16 |
| N18 | Adagrad + RMSprop | 160 | 2 | N1 |
| N19 | ImportanceSamplingSGD | 130 | 2 | N1, prob.CategoricalSample |
| N20 | ControlVariateSGD | 110 | 2 | N1, prob.Welford |
| N21 | OU-asymptotic-pin (test only) | 150 | 2 | N1 |
| N22 | PAC-BayesBound | 210 | 2 | N2, prob.KLDivergenceNumerical |
| **Σ** | | **3,540** | | |

Pure-glue ratio: ~80 % of LOC is composition over `optim.SimulatedAnnealing`'s acceptance test, `optim.GradientDescent`'s loop body, `optim/genetic.go`'s Box-Muller, and `linalg.{Cholesky, Eigen}`. ~20 % is genuinely-new math (CMA-ES rank-μ + path-cumulation update at ~250 LOC; IGO natural-gradient at ~150 LOC; PAC-Bayes bound + KL-from-trajectory at ~140 LOC are the only non-trivial fragments).

---

## 6. Recommended PR sequence

**PR-1: N3 + N1 + N2 + N5 (~390 LOC source, ~250 LOC tests, single day)**
Lands the `RNG` interface keystone + Robbins-Monro SGD + Welling-Teh SGLD + Hajek log-cooling. **First stochastic-gradient anything in `optim/`.** Saturates R-MUTUAL-CROSS-VALIDATION 3/3 on SGLD stationary distribution × analytic Gibbs × LBFGS mode (1D/2D banana). Single-day standalone with maximum signal-to-LOC.

**PR-2: N4 MetropolisHastings + N6 CrossEntropyMethod (~270 LOC, half-day)**
Promotes existing `SimulatedAnnealing` to a sampler at fixed T (N4 = 5-line lift); adds CEM as the sibling Gaussian-search-distribution global optim. CEM is the bridge between metaheuristics (N6 cross-entropy fitting on elites) and information-geometric optim (N9-N10).

**PR-3: N9 CMA-ES (~420 LOC, two days)**
The single highest-value primitive in this review. Hansen 2001 is the de-facto-best black-box optimiser since 2001 and its absence is the largest gap in `optim/`. Lands as a single-file `optim/cmaes.go` with one optional cross-link to 153-S2 Fisher for the natural-gradient identity test (R-CMA-ES-NATURAL-GRADIENT 3/3 pin).

**PR-4: N7 KieferWolfowitz + N8 PolyakRuppertAveraging (~160 LOC, half-day)**
Two stochastic-approximation primitives that complete the Robbins-Monro 1951 + Kiefer-Wolfowitz 1952 + Polyak-Ruppert 1988 trio. Polyak-Ruppert wraps any iterator-returning SGD; Kiefer-Wolfowitz handles black-box `f` with FD gradient. Saturates R-CRAMER-RAO pin on PR-averaging vs MLE-asymptotic-variance.

**PR-5: N16 Adam + N17 AdamW + N18 Adagrad + RMSprop (~330 LOC, one day)**
Adaptive-moment family. Adam alone justifies the PR; document the Reddi-Kale-Kumar 2018 non-convergence caveat and the AdamW decoupled-decay correction. Per-coordinate moment buffers reuse `optim.RNG` from N3 for dropout-style stochastic gradients.

**PR-6: N14 SVRG + N15 SAGA + N20 ControlVariateSGD (~460 LOC, two days)**
Variance-reduced family; SVRG is the citation-grounded crown jewel. ControlVariateSGD generalises the pattern into a configurable baseline. Lands together because the test golden files share the same MNIST-binary-subset infrastructure.

**PR-7: N12 SG-HMC + N13 SG-NUTS (~550 LOC, three days)**
Heaviest single PR. SG-NUTS is the production-grade Bayesian-inference primitive that gates Stan/PyMC parity. Cross-link 173 prob-VI-via-Stein for the score-function alternative to NUTS.

**PR-8: N10 IGO + N11 NES + N21 OU-pin + N22 PAC-Bayes (~830 LOC, four days)**
Crown-jewel synergy PR: information-geometric optimisation, Mandt-Hoffman-Blei 2017 OU-as-SGD-limit pin, Catoni-Dziugaite-Roy non-vacuous generalisation bound. Composes prob/ + optim/ + 153-infogeo/ + 189-MDL/ in a single file. Saturates four cross-package pins.

**PR-9: N19 ImportanceSamplingSGD (~130 LOC, half-day)** — orthogonal; can land any time after N1.

Total: ~3,540 LOC source + ~1,100 LOC tests across 9 PRs over ~14 engineer-days. PR-1 is single-day standalone; PR-3 (CMA-ES) is the single highest-value primitive; PR-7 (SG-NUTS) is the single hardest.

---

## 7. Cycle-hazard analysis

Proposed import directions:

```
optim/  ──→  prob/         (N4, N22 only — log-pdf for sampler test target)
optim/  ──→  linalg/       (N9 CMA-ES Cholesky+Eigen) — already used in optim/proximal_consumer_test.go
optim/  ──→  infogeo/      (N9, N10, N11 Fisher rescale) — new edge
optim/  ──→  constants/    (none for these primitives)
prob/   ──→  optim/        (none — opposite direction reserved for 169 fit primitives)
```

`optim/ → prob/` is a new edge (verified zero today via grep). To avoid the symmetric edge `prob/ → optim/` (which 169 needs for MLE/MAP), keep N4-N22 lightweight: take `logp func([]float64) float64` as caller-supplied, do NOT import `prob.Distribution` from optim. The `Distribution` interface stays consumer-side. This preserves the same DAG rule as 169-S14 (prob/ adopts the optim.RNG interface, not vice versa).

`optim/ → infogeo/` is a new edge (verified zero today). Cleanest placement: ship N9 CMA-ES in `optim/cmaes.go` with **no infogeo import**; add `optim/cmaes_infogeo_test.go` as a cross-package consumer-side test pinning the natural-gradient identity. This mirrors the 161 / 192 pattern of avoiding hard upstream edges from foundational packages.

**DAG remains cycle-free.** Verified by enumeration. New edges all flow toward more-foundational packages.

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **N1 Robbins-Monro:** schedule `α ∈ (0.5, 1]` strictly; `α = 1` is asymptotically slow but correct, `α < 0.5` is divergent. Panic at `α ≤ 0.5`.
- **N2 SGLD:** at finite step size `lr > 0`, the discretisation error of Welling-Teh's Euler scheme accumulates as `O(lr)` bias in the stationary distribution. Document: for KL < ε to true Gibbs, set `lr ≤ ε / (2 · sup ‖∇²U‖)`. For unbiased sampling use Metropolis-adjusted Langevin (MALA) instead — defer to v2.
- **N4 Metropolis-Hastings:** acceptance rate target 0.234 for high-d (Roberts-Gelman-Gilks 1997) and 0.44 for 1D (Gelman-Roberts-Gilks 1996); document but do not auto-tune (leave to caller).
- **N5 Hajek log-cooling:** constant `c ≥ Δ_max` (max barrier height) — if user under-specifies, asymptotic-global-optimal guarantee is **lost**, only convergence to local minimum. Document in panic-on-c<=0.
- **N9 CMA-ES:** (i) initial `σ_0` should be `~ 1/4 the search-space diameter` per Hansen-2009 tutorial; (ii) `C` numerical drift requires `Eigen` re-decomposition every `~ N²/10` generations to enforce symmetry; (iii) restart-with-doubled-popsize (IPOP-CMA-ES, Auger-Hansen 2005) defer to v2.
- **N12 SG-HMC:** friction `γ` must satisfy `γ > 0` and `lr·γ < 1` for damped underdamped Langevin (else divergence); document the Chen-Fox-Guestrin 2014 stability bound.
- **N13 SG-NUTS:** doubling-tree depth caps at 10 (per Hoffman-Gelman 2014 default); document memory `O(2^max_depth · d)`.
- **N14 SVRG:** snapshot `x̃` must be updated every `m = N` inner iterations on average; longer epochs lose variance reduction, shorter epochs lose linear convergence. Default `m = 2N` per Johnson-Zhang 2013.
- **N16 Adam:** Reddi-Kale-Kumar 2018 — vanilla Adam can fail to converge on certain non-convex problems (e.g., synthetic example in their Theorem 3); document AMSGrad correction `v̂_k = max(v̂_{k-1}, v_k)` as a safer drop-in. Bias-correction terms `(1-β_1^k), (1-β_1^k)` must use `^k` not `^(k-1)` (off-by-one footgun).
- **N18 Adagrad:** per-coordinate `√v_k` denominator monotonically grows; effective step size monotonically shrinks. Adagrad is the wrong choice for non-convex problems with shifting curvature; document and recommend Adam/AdamW for those.
- **N21 OU-pin:** Mandt-Hoffman-Blei 2017 small-`lr` limit is asymptotic; for `lr > 0.01·1/L` the OU formula is no longer accurate and Stratonovich corrections become non-negligible.
- **N22 PAC-Bayes:** posterior `q` must be log-concave for closed-form KL with Gaussian prior; for non-Gaussian `q` (e.g., SGLD trajectory empirical) use kernel-density-estimate KL or upper-bound by histogram entropy + log(grid-resolution).

---

## 9. Distinct from prior agents (provenance)

- **011-015 autodiff** — 012 names "stochastic AD (reparam + score-function) Tier-3" but doesn't compose with optim/; this synergy is the consumer-side pull through optim/ that justifies 012-T3 (the score-function gradient is N20 ControlVariateSGD when applied to log-likelihood with sampling baseline).
- **101-105 optim isolation** — 101 names "Adam/AdamW/RMSprop/Lion/Lookahead/SVRG/SAGA/SAG missing"; 102 names "trust-region/Newton-CG/HVP missing"; 103 names "CMA-ES/particle-swarm missing"; **this synergy fills exactly those 101+103 gaps via composition with prob/'s sampling primitives** (which themselves do not yet exist — gating on N3).
- **116-120 prob isolation** — 117 names `RNG/Sample/LogPDF` debt; this synergy adopts the same RNG (N3) but on the optim/ side, mirror-symmetric to 169-S14.
- **151-152 synergy-signal-prob** — orthogonal (filtering vs sampling).
- **153 synergy-prob-infogeo** — names `S2 FisherFromDistribution` which N9 CMA-ES, N10 IGO, N11 NES all consume; **this synergy is 153's first non-test consumer** (cross-package usage of FisherFromDistribution).
- **155 synergy-crypto-prob** — orthogonal (cryptographic randomness, not statistical inference).
- **161 synergy-control-prob** — orthogonal (Lyapunov function learning, no SGD).
- **162 synergy-graph-prob** — orthogonal (random walks on graphs).
- **163 synergy-optim-autodiff** — names `A12 Adam` (= N16 here) and `A18 SVRG` (= N14 here); **163 ships the autodiff-side automatic-differentiation, this review ships the optim-side stochastic-update**. The two compose: 163-A1 forward-mode duals + N16 Adam = the standard 2026 NN-optimiser stack.
- **164 synergy-orbital-optim** — orthogonal (deterministic transfer optim).
- **165 synergy-sequence-prob** — orthogonal (string distances).
- **168 synergy-physics-autodiff** — orthogonal (Lagrangian/Hamiltonian/symplectic). Cross-link only at N12 SG-HMC where Chen-Fox-Guestrin's friction term echoes Langevin thermostats from molecular dynamics.
- **169 synergy-prob-optim (PARTIAL OVERLAP)** — 169 covers EM/VI/MAP/BO axis (deterministic fits); **this review (195) covers the noisy-optim axis (SGD/SGLD/SG-MCMC/CMA-ES/IGO)**. Disjoint primitive rosters by design: 169 has S1-S18 (none stochastic-gradient), 195 has N1-N22 (all stochastic-gradient). Shared base: both consume the N3 RNGSampler interface; both reference 153-S2 Fisher and 169-S8 LogPDF. **Recommend co-shipping N3 + 169-S14 in a single architectural-keystone PR** (the unified RNG interface lands once, both reviews' Tier-2 then unblocks). The two reviews are complementary halves of the optim×prob synergy join.
- **180 synergy-physics-prob** — names Boltzmann distribution / Gibbs measure; this review's N2 SGLD samples exactly that distribution. **N2 + 180-S1 Boltzmann compose: SGLD on harmonic oscillator U = ½kx² produces Boltzmann samples at temperature T = lr·σ²/2**. Document this as the `R-SGLD-EQUALS-BOLTZMANN` cross-package pin.
- **189 synergy-info-compression** — names PAC-Bayes / MDL bounds; **this review's N22 PAC-BayesBound is 189's consumer-side application** to a SGLD-trained model.
- **190-193 prior synergy reviews** — orthogonal (topology, chaos-control, fluids-control, prob-changepoint).
- **194 synergy-em-geometry** — orthogonal (Maxwell on meshes; no stochastic anything).

---

## 10. Bottom line

`optim/` ships ZERO stochastic-gradient primitives despite shipping a Boltzmann-acceptance simulated-annealer that is one rename away from being a Metropolis sampler; `prob/` ships ZERO `Sample(rng)` API despite shipping seven `Distribution`s with closed-form `Quantile` functions one inverse-CDF call away from inverse-transform sampling. **Twenty-two synergy primitives totalling ~3,540 LOC of pure connective tissue** stand up the entire SDE-as-sampler + stochastic-approximation + variance-reduced-SGD + CMA-ES + IGO + PAC-Bayes pipeline on existing v0.10.0 surfaces (specifically: Box-Muller already inlined in `optim/genetic.go:58-65`, Boltzmann acceptance already in `optim/metaheuristic.go:73`, geometric cooling already in `optim/metaheuristic.go:89`, KL trapezoidal-rule already in `prob/distribution.go`).

Eleven primitives (N1-N11) are Tier-1 ship-today against zero new infrastructure beyond the **N3 RNGSampler interface keystone** (~80 LOC, single day, co-ship with 169-S14 to land both reviews' Tier-2 simultaneously). Eleven are Tier-2 needing N3 plus 169-S8 LogPDF.

The cheapest single-day standalone is **PR-1 (N3+N1+N2+N5 = 390 LOC)** which lands the first stochastic-gradient anything in `optim/` and saturates a 3/3 R-MUTUAL-CROSS-VALIDATION pin on SGLD stationary distribution. The single highest-value primitive is **PR-3 N9 CMA-ES (420 LOC)** — the de-facto-best black-box optimiser since 2001, currently absent from the repo. The single hardest is **PR-7 N13 SG-NUTS (340 LOC)** — production-grade Bayesian inference. The crown-jewel composition is **PR-8 N22 PAC-BayesBound (~210 LOC)** — non-vacuous test-loss certificate from SGLD trajectory composing prob/ + optim/ + 153-infogeo/ + 189-MDL/ in one algorithm.

**Reality is unusually well-positioned for this synergy because (i) `optim.SimulatedAnnealing` is already 80% of MCMC; (ii) `optim/genetic.go` already has Box-Muller; (iii) `linalg.Cholesky+Eigen` already covers CMA-ES sampling; (iv) `prob.KLDivergenceNumerical` already covers PAC-Bayes; (v) `optim.GeneticAlgorithm`'s anonymous `interface{Float64() float64}` is the precedent for `optim.RNG`; and (vi) the consumer-side-placement rule (per agents 158/159/160/166/167/168/169) recommends N3 in `optim/` (where SimulatedAnnealing already takes RNG) rather than `prob/`, with `prob.Sample(rng optim.RNG)` adopting the optim-side type — minimum architectural perturbation, maximum stochastic-inference unlock.**

The single most important conceptual identity this review pins: **Robbins-Monro 1951 SA + Welling-Teh 2011 SGLD + Hajek 1988 log-cooling SA + Mandt-Hoffman-Blei 2017 OU-as-SGD-limit are four formulations of the same convergence theorem on different SDEs.** R-IDENTITY-OF-SDE-LIMITS 3/3 saturation lands when `optim/` ships N1 + N2 + N5 + N21 in one PR.
