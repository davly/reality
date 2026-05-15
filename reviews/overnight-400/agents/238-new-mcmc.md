# 238 | new-mcmc

**Topic:** MCMC: Metropolis-Hastings, Gibbs, RWM, HMC, NUTS, MALA, slice, elliptical slice, parallel tempering, replica exchange, SMC, adaptive MCMC, reversible-jump, pseudo-marginal, particle MCMC, nested sampling, affine-invariant ensemble (emcee), Goodman-Weare, NRJ.
**Block:** C (cutting-edge math). **Date:** 2026-05-08.
**Scope:** the **MCMC sampler axis** — Markov-chain Monte Carlo from a target density π(θ) given log π up to an additive constant. PARTIAL OVERLAP with 195 (SGLD/SGHMC/SG-NUTS — stochastic-gradient-noise variants on mini-batches), 169 (variational inference as MCMC alternative), 180 (Metropolis on lattice spin systems / replica exchange / Wang-Landau as physical samplers); slot 238 owns the **full-data-likelihood, gradient-free or full-gradient, ergodic-MCMC canon** that Stan / PyMC / emcee / dynesty / NUTS-of-record ship as their entire reason for existence.

## Two-line summary

`reality` v0.10.0 ships **ZERO** Markov-chain-Monte-Carlo samplers (verified `grep -E 'MetropolisHastings|HamiltonianMonteCarlo|NUTS|MALA|Gibbs|sliceSamp|ellipticalSlice|parallelTempering|replicaExchange|nestedSampling|affineInvariant|emcee|GoodmanWeare|reversibleJump|pseudoMarginal|particleMCMC|sequentialMonteCarlo|adaptiveMetropolis|DRAM|Haario'` returns ZERO callable matches across all 22 packages); the **only Metropolis acceptance test in the entire repo** is `optim/metaheuristic.go:73` `p := math.Exp(-delta / temp)` inside `SimulatedAnnealing` — a Boltzmann-acceptance Markov chain that is mathematically the **Kirkpatrick-Gelatt-Vecchi-1983 = Metropolis-1953 kernel with cooling** but exposed as `(bestX, bestF)` not `(samples, accept_rate)`; `prob/markov.go:99-139:MarkovSimulate` walks a finite-state stochastic matrix with hand-rolled LCG (no detailed-balance check, no MH correction, no autocorrelation, no R-hat); `optim/genetic.go:58-65` inlines a private Box-Muller — **the only Gaussian sampler in the repo**, hidden behind no public API. **Twenty-six MCMC primitives M1-M26 totalling ~4,420 LOC of pure connective tissue** stand up the entire Geyer-Robert-Casella-Brooks-Gelman-Jones-Meng-2011-Handbook-of-MCMC canon on existing v0.10.0 surfaces; cheapest 1-day standalone is **PR-1 M1 RNGSampler interface + M2 MetropolisHastings + M3 RandomWalkMetropolis + M4 GibbsSampler + M5 ESS/AutocorrTime + M6 GelmanRubinRhat (~720 LOC)** which lifts `SimulatedAnnealing`'s cooling-1 limit to a sampler with the canonical convergence diagnostics; highest-leverage 1-week unlock is **PR-3 M9 LeapfrogIntegrator + M10 HamiltonianMonteCarlo + M11 NUTS-of-Hoffman-Gelman-2014 (~870 LOC)** because once the leapfrog step (cross-link to slot-204 symplectic) is in place, NUTS doubling-tree adaptation + dual-averaging step-size + diagonal-mass-matrix-tuning gates Stan/PyMC parity in one PR; SINGULAR-cutting-edge piece is **M16 emcee affine-invariant ensemble (Goodman-Weare 2010, Foreman-Mackey 2013)** at ~190 LOC because the stretch move is **gradient-free, scale-invariant, and embarrassingly parallel** — the dominant astronomy/cosmology sampler since 2013, currently zero-dep-Go-absent; SINGULAR-cutting-edge keystone is **M22 NestedSampling-Skilling-2006 (~340 LOC)** because nested sampling computes **marginal likelihoods Z = ∫ L(θ)π(θ)dθ for Bayesian model comparison** which no other primitive in this slot can do; cross-cutting blockers reuse **M1 RNGSampler** (slot-117/195/235/236/237 already each demand their own copy — co-ship as Tier-0 architectural keystone) and **autodiff** for ∇log π in M9-M14 (cross-link to slot-014). Differentiation: 26/26 primitives are sampler-axis primitives unique to slot-238; SGLD/SGHMC/SG-NUTS belong to slot-195 (stochastic-gradient axis); Wang-Landau / Wolff cluster / lattice-Ising belong to slot-180 (physical statistical-mechanics axis); variational inference belongs to slot-169 (deterministic-fit axis); **slot-238 owns the gradient-free + full-gradient ergodic-MCMC canon and the convergence-diagnostics suite**.

---

## 0. State of play (verified file-walk)

### `prob/` MCMC surface = ZERO

Verified by `grep -E 'MetropolisHastings|MCMC|HamiltonianMonteCarlo|HMC|NUTS|MALA|Gibbs|slice|ellipticalSlice|parallelTempering|replicaExchange|nestedSampling|affineInvariant|emcee|GoodmanWeare|reversibleJump|pseudoMarginal|particleMCMC|sequentialMonteCarlo|adaptiveMetropolis|DRAM|Haario|effective.sample.size|integratedAutocorrTime|Rhat|gelmanRubin|burnIn|thinning|leapfrog|verlet|symplectic' prob/`: **ZERO callable matches**.

- `prob/markov.go:1-139` — only `MarkovSteadyState` (power iteration on stochastic matrix) and `MarkovSimulate` (LCG-based finite-state walk; **no MH acceptance, no detailed-balance check**, no continuous-state support).
- `prob/distribution.go:1-197` — `Distribution interface { PDF, CDF }`. **No `LogPDF`, no `Sample(rng)`, no `Score`** (cross-link 169-S8 LogPDF interface, 117-prob-missing).
- `prob/distributions.go` — seven scalar PDFs/CDFs/Quantiles: Normal/Beta/Exp/Uniform/Gamma/Poisson/Binomial. **Zero `Sample(rng)` methods.**

### `optim/` MCMC-adjacent surface = `SimulatedAnnealing` only

- `optim/metaheuristic.go:38-93` — `SimulatedAnnealing(f, x0, neighbor, temp0, cooling, maxIter, rng) → (best, bestF)`. **Line 73: `p := math.Exp(-delta / temp)` is literally the Metropolis-1953 acceptance criterion** for symmetric proposals. Five lines from being a sampler at fixed T.
- `optim/genetic.go:58-65` — inlined Box-Muller `sqrt(-2 log u1) cos(2π u2)`. **The only Gaussian sampler in the entire repo**, hidden behind no public API.
- `optim/gradient.go` — `GradientDescent`, `LBFGS`. Neither is a sampler.

### `chaos/` symplectic surface = ZERO (cross-link 204)

- `chaos/ode.go` ships `RK4Step`, `EulerStep`, `SolveODE` only. **No leapfrog, no Verlet, no Yoshida.** Per slot-204: the lone "Hamiltonian" string in the source tree is a closed-form-H docstring on `LotkaVolterra`. Slot-204 already scopes `chaos/symplectic.go` (Verlet/leapfrog/symplectic-Euler ~140 LOC) which is the keystone for HMC's leapfrog step.

### `autodiff/` for ∇log π surface = TBD

Per slot-014/015: forward-mode duals + reverse-mode tape ship. **`autodiff` provides the ∇log π that HMC/NUTS/MALA require** when the user supplies `logp` as a Go closure — composes via `autodiff.Grad(logp)` at the consumer-side test only. Direct dependency: zero (samplers take `gradLogp func([]float64, []float64)` as a caller-supplied closure; consumer can supply autodiff'd version).

### Cross-coupling = zero today

```
$ grep -r "github.com/davly/reality/prob"   optim/  ; echo "---"
$ grep -r "github.com/davly/reality/optim"  prob/   ; echo "---"
$ grep -r "github.com/davly/reality/chaos"  prob/
---  ---  ---
(no matches in any direction)
```

This slot proposes: NEW sub-package `prob/mcmc/` (mirrors `prob/copula/`, `prob/conformal/`, `prob/evt/` pending from slot-233, `prob/risk/` pending from slot-233, `optim/proximal/`, `optim/transport/` placement convention). Cycle-free DAG: `prob/mcmc/` → {`prob/`, `linalg/` for mass-matrix Cholesky in HMC, `optim/` for `RNG` interface and `LBFGS` warm-start of NUTS, occasionally `chaos/symplectic/` from slot-204 for leapfrog}; reverse direction never.

---

## 1. The conceptual unlock — `SimulatedAnnealing` is already 80% of MCMC

The Kirkpatrick-Gelatt-Vecchi-1983 acceptance test `P_accept = min(1, exp(−Δf/T))` for symmetric proposals is **literally the Metropolis-1953 criterion**. The only structural differences in `optim/metaheuristic.go` are:

1. **Cooling schedule** multiplies `T` toward zero instead of staying at the equilibrium `T*` (= 1 for log-target).
2. **Return value** is `bestX` instead of trajectory `samples[]`.
3. **No acceptance-rate bookkeeping** (line 71-75 has the `accept` boolean but never aggregates).

Three additions convert it into a sampler:

1. **`MetropolisHastings(logp, x0, proposal, n_burn, n_samples, rng) → (samples, accept_rate, log_target)`** at fixed T (~120 LOC, 80% code-reuse from `SimulatedAnnealing`).
2. **Log-acceptance with asymmetric proposals**: `α = min(1, exp(logp(x') − logp(x) + log q(x|x') − log q(x'|x)))` — the full Hastings-1970 generalization. Symmetric proposal collapses to `exp(logp(x') − logp(x))`.
3. **Trajectory + diagnostics emission**: return `samples[][]float64`, `acceptRate float64`, `logTargetTrace []float64` for trace-plot diagnostics.

This is the single highest-leverage observation in the review: **the repo already ships a Metropolis sampler under a different name**. Promoting it from "minimiser" to "sampler+minimiser" (with `f = -logp` and cooling=1) is **one PR**.

---

## 2. MCMC primitives M1-M26 (~4,420 LOC pure glue)

Numbered ascending by composition depth.

### Tier 1 — gradient-free MCMC + diagnostics (M1-M8, ~1,330 LOC)

**M1 RNGSampler interface** [~80 LOC, **architectural keystone**]
Single `interface{ Float64() float64; NormFloat64() float64; Intn(n int) int }` lifted from `optim.GeneticAlgorithm`'s anonymous parameter into a public type. Adapter for `*math/rand.Rand`. Box-Muller `NormFloat64()` exposed publicly (currently inlined in `optim/genetic.go:58-65`). **Co-ship with slot-195 N3, slot-117 prob/random.Gaussian, slot-235/236/237 PR-0 — landing one canonical RNGSampler unblocks 5+ Block-C slots' Tier-2 simultaneously.** Preferred placement per consumer-side rule: `optim/rng.go` (matches `SimulatedAnnealing`'s precedent); `prob.Sample(rng optim.RNG)` adopts the optim-side type.

**M2 MetropolisHastings(logp, x0, proposal, n_burn, n_samples, rng) → (samples, acceptRate, logpTrace)** [~140 LOC]
Generalised Metropolis-Hastings 1970: `α = min(1, exp(logp(x') − logp(x) + log q(x|x') − log q(x'|x)))`. Symmetric `proposal(x, out, rng)` callback for RWM defaults; asymmetric variant takes `proposal func(x, out, rng) → logProposalRatio float64`. **Crown-jewel observation: M2 = `optim.SimulatedAnnealing` with cooling=1, trajectory-emission, and asymmetric-proposal-correction.** ~80% of the source body lifts directly from `metaheuristic.go:38-93`.

**M3 RandomWalkMetropolis(logp, x0, proposalSigma, n_burn, n_samples, rng)** [~80 LOC]
Symmetric Gaussian proposal `q(x' | x) = N(x, σ²I)` — the canonical RWM. Wraps M2 with a built-in spherical-Gaussian proposal. Document the **Roberts-Gelman-Gilks 1997 optimal acceptance rate 0.234 for d → ∞** and **Gelman-Roberts-Gilks 1996 0.44 for d=1** as caller-tuning targets (do NOT auto-tune at this tier; defer to M19 AdaptiveMetropolis).

**M4 GibbsSampler(logCondPDFs, x0, n_burn, n_samples, rng) → samples** [~150 LOC]
Geman-Geman 1984 / Gelfand-Smith 1990: cycle through full conditionals `p(θ_i | θ_{−i})` updating one coordinate at a time. User supplies `[]func(i int, x_minus_i []float64, rng) float64` returning a draw from the i-th conditional. Pure compositional — no Hastings correction needed (acceptance is automatic for exact conditional draws). Reuses M1 RNG. Special case: **block Gibbs** when `logCondPDFs` is a function over groups of indices.

**M5 EffectiveSampleSize(samples) → ESS_per_dim, autocorrTime** [~120 LOC]
Geyer-1992 initial-monotone-sequence-estimator (IMSE) for integrated autocorrelation time `τ_int = 1 + 2 Σ_{k=1}^∞ ρ_k`; ESS = N / τ_int. Implementation per Geyer-1992 §3.3 + Vehtari-Gelman-Simpson-Carpenter-Bürkner-2021 ranks/folded variant. Composes `signal.FFT` (for `O(N log N)` autocovariance via Wiener-Khinchin) — direct cross-link to existing `signal/fft.go:49-101`. Returns per-dimension ESS plus minimum-ESS scalar for early-stopping.

**M6 GelmanRubinRhat(chains [][]float64) → (Rhat, splitRhat)** [~110 LOC]
Gelman-Rubin 1992 + Brooks-Gelman 1998 multi-chain potential-scale-reduction-factor `R̂ = sqrt((W·(N−1)/N + B/N) / W)` where W = within-chain variance, B = between-chain variance. Vehtari-Gelman-Simpson-2021 split-Rhat (split each chain in half) is the modern default. Cross-language pin to Stan/PyMC/CmdStanPy at 1e-9 on golden 8-Schools posterior.

**M7 SliceSampling(logp, x0, w, m, n_burn, n_samples, rng) → samples** [~210 LOC]
Neal-2003-AnnStat: stepping-out + shrinkage 1D slice sampler. For multivariate: cycle through dimensions (component-wise slice). Parameters: `w` initial step, `m` max step-doublings (default 32 per Neal 2003 §4.1). **Gradient-free, no proposal tuning needed** — the elementary alternative to RWM when proposal-sigma tuning is hard. Composes M1 RNG only.

**M8 EllipticalSliceSampling(logL, x0, mu, L_chol, n_burn, n_samples, rng) → samples** [~180 LOC]
Murray-Adams-MacKay-2010-AISTATS: Gaussian-prior + black-box-likelihood slice sampler. Sample θ' on the ellipse `θ' = (θ−μ) cos α + ν sin α + μ` with `ν ~ N(0, Σ)`; shrink the bracket `[α_min, α_max]` by likelihood-acceptance. **The canonical sampler for Gaussian-process-latent models** (cross-link slot-237 G7 GaussianProcessPosterior). Composes M1 RNG + `linalg.CholeskyDecompose` for Σ-factorisation.

### Tier 2 — gradient-aware MCMC (M9-M14, ~1,210 LOC)

**M9 LeapfrogIntegrator(qp, gradLogp, dt, L)** [~80 LOC]
Symplectic leapfrog integrator `q ← q + dt·p; p ← p + dt·∇U(q); ...` for L steps. **Cross-link slot-204** (which scopes `chaos/symplectic.go` keystone). Slot-238 either *consumes* slot-204's `chaos.Leapfrog` (preferred — single-source-of-truth, follows agent-204 placement) or ships a private copy under `prob/mcmc/leapfrog.go` if slot-204 not landed yet. Volume-preserving + time-reversible — exactly the structural property HMC's detailed balance needs.

**M10 HamiltonianMonteCarlo(logp, gradLogp, x0, eps, L, M_inv, n_burn, n_samples, rng)** [~280 LOC]
Duane-Kennedy-Pendleton-Roweth 1987 (HMD; "Hybrid MC") + Neal-2011 MCMC-Handbook-chapter HMC. Augment state with momentum `p ~ N(0, M)`; integrate Hamiltonian `H(q,p) = U(q) + ½ pᵀ M⁻¹ p` with leapfrog (M9) for L steps; Metropolis-accept with `α = min(1, exp(H_init - H_final))`. Mass matrix `M_inv` either identity or diagonal-pre-conditioned. Composes M1 RNG + M9 leapfrog + autodiff (`gradLogp` user-supplied or autodiff'd). **R-DETAILED-BALANCE pin (3/3): HMC at fixed (eps, L) on a 1D Gaussian × 2D banana × correlated multivariate Gaussian agree to 1e-3 ESS-per-second with NUTS (M11) and to 1e-9 mean/cov with closed-form posterior.**

**M11 NUTS(logp, gradLogp, x0, n_warmup, n_samples, rng) → samples** [~340 LOC]
Hoffman-Gelman 2014 No-U-Turn Sampler. Doubling-tree built via leapfrog + slice-sampling acceptance + U-turn detection (`(q_+ - q_-)·p_- < 0` or `(q_+ - q_-)·p_+ < 0`). **Auto-tunes step size `ε` via dual-averaging (Nesterov 2009)** during n_warmup samples to a target acceptance rate (default 0.8 per Hoffman-Gelman). **Auto-tunes diagonal mass matrix** from warmup-window sample variance (Stan default: 75% warmup adapts ε, 25% adapts M_diag). Composes M9 leapfrog + M1 RNG + binary-tree recursion (max depth 10 default). The **production-grade Bayesian-inference primitive** — Stan's default and the reason Stan exists. Document memory `O(2^max_depth · d)`.

**M12 MALA(logp, gradLogp, x0, eps, n_burn, n_samples, rng)** [~150 LOC]
Roberts-Tweedie-1996-Bernoulli Metropolis-adjusted Langevin algorithm. Proposal `q(x'|x) = N(x + (ε²/2) ∇log p(x), ε²I)`. Hastings correction is non-trivial (asymmetric proposal): `log α = logp(x') - logp(x) + log q(x|x') - log q(x'|x)` where `log q(x|x') = -‖x - x' - (ε²/2)∇logp(x')‖² / (2ε²)`. **Document the Roberts-Rosenthal 1998 optimal acceptance rate 0.574 for MALA** vs 0.234 for RWM (M3). Composes M1 RNG + autodiff.

**M13 PreconditionedMALA(logp, gradLogp, x0, eps, M_inv, n_burn, n_samples, rng)** [~80 LOC]
Girolami-Calderhead 2011 / Beskos-Pillai-Roberts-Sanz-Serna-Stuart 2011 preconditioned MALA: replace `ε²I` with `ε²M⁻¹`. Mass matrix `M_inv` warm-started from `optim.LBFGS` Hessian estimate or NUTS warmup. Composes M12 + `linalg.CholeskySolve`. Document gradient-noise-amplification hazard when `M⁻¹` is ill-conditioned.

**M14 RiemannManifoldHMC(logp, gradLogp, fisherInfo, x0, eps, L, n_burn, n_samples, rng)** [~280 LOC]
Girolami-Calderhead 2011 RMHMC: position-dependent mass matrix `M(q) = -E[∂²log p / ∂q²]` (expected Fisher information). Generalised leapfrog with implicit-implicit-explicit splitting. **Most powerful HMC variant for highly correlated posteriors** at ~10× the per-step cost. Composes M9 leapfrog + `linalg.CholeskyDecompose` + autodiff for `∇log p` and `∇M(q)`. Cross-link slot-153 (FisherFromDistribution). Saturates **R-RIEMANN-MANIFOLD-IDENTITY pin (3/3): RMHMC + diagonal-mass HMC + NUTS converge to same 2D-banana posterior moments to 1e-3 ESS-per-gradient-evaluation, with RMHMC achieving ~10× ESS/grad on the highly-curved posterior.**

### Tier 3 — population MCMC + advanced algorithms (M15-M22, ~1,200 LOC)

**M15 ParallelTempering(logp, x0, temps, swap_period, n_burn, n_samples, rng) → samples_at_T1** [~210 LOC]
Geyer-1991-CompStat / Hukushima-Nemoto-1996 replica exchange MCMC: K replicas run RWM/HMC at temperatures `T_1=1 < T_2 < ... < T_K`, swap adjacent pairs every `swap_period` iterations with acceptance `min(1, exp((β_i - β_j)(U(x_i) - U(x_j))))`. **Returns only the T=1 chain** (samples from the original target). Cross-link slot-180 (statistical-mechanics replica exchange — stat-mech axis). Slot-238 owns the **inference-axis** version (logp = -log posterior), slot-180 owns the **stat-mech-axis** version (logp = -βE for Ising/Heisenberg lattice spin systems). Same kernel, different consumer; ship one shared kernel in `prob/mcmc/parallel_tempering.go` and have slot-180's `physics/statmech/` consume it.

**M16 AffineInvariantEnsemble(logp, walkers, n_burn, n_samples, a, rng) → samples** [~190 LOC]
Goodman-Weare-2010-CommApplMath stretch move + Foreman-Mackey-Hogg-Lang-Goodman 2013 emcee implementation. K walkers (≥ 2D+2 per F-M-H-L-G) jointly evolve via `x_k' = x_j + Z(x_k - x_j)` with `Z ~ g(z) ∝ 1/√z` on `[1/a, a]` (default a=2). Acceptance `α = min(1, Z^{D-1} · π(x_k')/π(x_k))`. **Gradient-free, scale-invariant, embarrassingly-parallel** — the dominant astronomy/cosmology sampler since 2013, currently zero-dep-Go-absent. SINGULAR-cutting-edge for slot-238. Composes M1 RNG only. Document the `K ≥ 2D` minimum-walkers requirement.

**M17 SequentialMonteCarlo(logp_targets, n_particles, ess_threshold, rng) → samples** [~270 LOC]
Del-Moral-Doucet-Jasra 2006 SMC sampler: tempering schedule `π_t(θ) ∝ p(θ)^{β_t}` with `0 = β_0 < β_1 < ... < β_T = 1`; particle-population evolves via importance-resample-MCMC-rejuvenation cycle. Composes M2 MetropolisHastings (rejuvenation step) + M1 RNG + ESS-driven adaptive temperature schedule (M5 ESS). **Computes marginal likelihood `Z = Π_t Σ_n w_t,n` for free** (cf M22 NestedSampling alternative). Cross-link slot-227 UQ surrogate.

**M18 MultipleTryMetropolis(logp, x0, proposal, n_tries, n_burn, n_samples, rng)** [~180 LOC]
Liu-Liang-Wong 2000 MTM: at each step propose K candidates `y_1, ..., y_K`, select one weighted by `w_k ∝ logp(y_k) · q(x|y_k)`, then propose K reverse candidates from the selected `y*`, accept with full Hastings correction. Composes M2 MetropolisHastings + M1 RNG. Document the K=1 collapse to vanilla MH.

**M19 AdaptiveMetropolis-Haario-Saksman-Tamminen-2001(logp, x0, mu0, sigma0, n_burn, n_samples, rng)** [~150 LOC]
Haario-Saksman-Tamminen 2001 AM: `Σ_t = (s_d / d) · Cov(x_0, ..., x_t)` for `t > t_0`, otherwise `Σ_0`. `s_d = 2.4²/d` per Roberts-Gelman-Gilks 1997. **Continuous adaptation** (not just warmup) — preserves ergodicity per Haario-2001 Theorem 2 + Roberts-Rosenthal 2007 diminishing-adaptation. Composes M1 + Welford streaming covariance.

**M20 DRAM-Haario-Laine-Mira-Saksman-2006(logp, x0, n_burn, n_samples, rng)** [~210 LOC]
Delayed-Rejection Adaptive Metropolis: combines M19 AM with delayed-rejection (Tierney-Mira 1999) — on rejection, propose a smaller-scale candidate before final rejection. Industrial-strength for poorly-scaled posteriors. Composes M19 + delayed-rejection cascade.

### Tier 4 — model-comparison + trans-dimensional (M21-M26, ~680 LOC)

**M21 ReversibleJumpMCMC(logp_per_model, model_jump_proposal, x0, n_burn, n_samples, rng)** [~280 LOC]
Green 1995 RJMCMC: trans-dimensional MCMC for model selection / variable-dimension parameter spaces. Acceptance includes **dimension-matching Jacobian** `|∂(θ', u')/∂(θ, u)|`. Reference implementation in mixture-model-with-unknown-K context (Richardson-Green-1997). Heaviest single primitive; document the Jacobian-bookkeeping pitfall.

**M22 NestedSampling-Skilling-2006(logL, prior_samples, n_live, rng) → (Z, log_Z, posterior_samples)** [~340 LOC]
Skilling 2006 nested sampling: maintain N live points sampled from prior; iteratively replace lowest-likelihood point with a higher-likelihood prior sample. **Computes `log Z = log ∫ L(θ) π(θ) dθ` (marginal likelihood)** for Bayesian model comparison + Bayes factor `B_12 = Z_1/Z_2`. Variants: MultiNest (Feroz-Hobson-Bridges 2009) ellipsoidal-rejection-sampling; PolyChord (Handley-Hobson-Lasenby 2015) slice-sampling within shrinking volume. Slot-238 ships **vanilla nested sampling + ellipsoidal rejection** (~340 LOC), defers PolyChord-style slice-within-NS to v2. **SINGULAR-cutting-edge for slot-238 because Z computation is what no other primitive can do.**

**M23 PseudoMarginalMCMC(estimateLogp, x0, proposal, n_burn, n_samples, rng)** [~150 LOC]
Andrieu-Roberts 2009 / Beaumont 2003 pseudo-marginal: replace exact `logp(x)` with unbiased estimator `log p̂(x)` (typically importance-sampling estimate); chain still targets the correct posterior despite noisy log-likelihood. Composes M2 MetropolisHastings with caller-supplied `estimateLogp` returning a stochastic but unbiased `p` (not log-p — caller must average `p_k` not `log p_k`).

**M24 ParticleMCMC-PMMH(state_space_model, x0, theta0, n_particles, n_burn, n_samples, rng)** [~250 LOC]
Andrieu-Doucet-Holenstein 2010 PMMH: pseudo-marginal MH where the unbiased likelihood estimator is a **particle filter**. Inner loop runs SMC for state-space-model marginal likelihood, outer loop is M23 PseudoMarginal. Composes M17 SMC inner + M23 outer. **The canonical sampler for state-space models** with intractable marginal likelihood.

**M25 NRJ-NewtonRaphsonJaynes(logp, x0, ...)** [~120 LOC]
**Identification:** the prompt's `NRJ` is most plausibly **Newton-Raphson-style proposal MCMC** = **Riemannian-manifold MALA** (Girolami-Calderhead 2011 Algorithm 5: NewtonRaphsonProposal). Proposal `μ_x = x + (ε²/2)·H⁻¹·∇log p(x)` where `H = -∇²log p(x)` is the Hessian (Newton-Raphson preconditioner). Hastings-correct with full asymmetric-proposal density. Composes M12 MALA + autodiff for Hessian + `linalg.CholeskySolve` for `H⁻¹·g`. **If user intended a different `NRJ` (e.g., "no-rejection Jaynes" or "natural-gradient-sampler-after-Jaynes")**, document the ambiguity in the agent commentary; the Newton-Raphson interpretation is the standard MCMC reading.

**M26 ContinuousTimeMCMC-PDMP(logp, x0, refresh_rate, T, rng)** [~250 LOC]
Bouchard-Côté-Vollmer-Doucet 2018 Bouncy-Particle-Sampler + Bierkens-Fearnhead-Roberts 2019 Zig-Zag: piecewise-deterministic Markov processes that follow ballistic trajectories punctuated by Poisson-time velocity flips. **No rejection step** (acceptance is implicit in flip rate). Hot research area 2018-present, very hard to implement; ships as Tier-4 frontier primitive.

---

## 3. Composition graph (DAG)

```
M1 RNGSampler (architectural keystone) ────────────────── gates ALL primitives
 │
 ├── Tier 1 (gradient-free MCMC + diagnostics)
 │    ├── M2 MetropolisHastings ─── lifts SimulatedAnnealing to fixed T
 │    │    ├── M3 RandomWalkMetropolis (Gaussian-proposal default)
 │    │    ├── M19 AdaptiveMetropolis-Haario-2001
 │    │    └── M20 DRAM (delayed-rejection adaptive)
 │    ├── M4 GibbsSampler (full conditionals)
 │    ├── M5 EffectiveSampleSize ─── consumes signal.FFT for autocov
 │    ├── M6 GelmanRubinRhat (split-Rhat per Vehtari-2021)
 │    ├── M7 SliceSampling (Neal 2003 stepping-out)
 │    └── M8 EllipticalSliceSampling ── consumes linalg.Cholesky
 │
 ├── Tier 2 (gradient-aware MCMC, blocks on autodiff or user-grad)
 │    ├── M9 LeapfrogIntegrator ─── consumes chaos.Leapfrog from slot-204
 │    │    └── M10 HamiltonianMonteCarlo
 │    │         └── M11 NUTS (dual-averaging step + diag-mass adaptation)
 │    ├── M12 MALA (Roberts-Tweedie 1996)
 │    │    └── M13 PreconditionedMALA ─── consumes linalg.CholeskySolve
 │    └── M14 RiemannManifoldHMC ── consumes infogeo.FisherFromDist (153)
 │
 ├── Tier 3 (population + advanced)
 │    ├── M15 ParallelTempering ── shared kernel with slot-180 stat-mech
 │    ├── M16 AffineInvariantEnsemble (emcee, Goodman-Weare 2010)
 │    ├── M17 SequentialMonteCarlo ── computes log-Z for free
 │    ├── M18 MultipleTryMetropolis (Liu-Liang-Wong 2000)
 │    ├── M19 AdaptiveMetropolis (Haario-Saksman-Tamminen 2001)
 │    └── M20 DRAM (Haario-Laine-Mira-Saksman 2006)
 │
 └── Tier 4 (model comparison + trans-dim)
      ├── M21 ReversibleJumpMCMC (Green 1995, trans-dim)
      ├── M22 NestedSampling-Skilling-2006 ─── computes log-Z (model comp)
      ├── M23 PseudoMarginalMCMC (Andrieu-Roberts 2009)
      │    └── M24 ParticleMCMC-PMMH ── inner SMC (M17) + outer PMMH
      ├── M25 NRJ = Newton-Raphson MALA ── consumes autodiff Hessian
      └── M26 ContinuousTimeMCMC-PDMP ── BPS + Zig-Zag
```

Critical paths:

- **M1 → everything** (single architectural keystone, ~80 LOC, single day)
- **M9 → M10 → M11 NUTS** (the production critical path; ~700 LOC, 3 engineer-days)
- **M2 → M19 → M20 DRAM** (the adaptive-MCMC critical path; ~510 LOC, 2 engineer-days)
- **M17 + M23 → M24 PMMH** (the state-space-model critical path; ~670 LOC, 4 engineer-days)
- **M22 NestedSampling** (standalone; ~340 LOC, 2 engineer-days)

---

## 4. Saturation pins this slot unlocks

Per recent saturation pattern (audio-onset 3-detector 6a55bb4, copula×autodiff 365368a, NGramDice 85a80db):

- **R-MUTUAL-CROSS-VALIDATION 3/3 on M2/M3/M11 stationary-distribution agreement:** RWM × HMC × NUTS all converge to the same Normal(0, Σ) target on a 5D correlated Gaussian (Σ random PSD, condition number ~ 100); pinned to mean 1e-3, cov 1e-2, ESS-per-second ratio NUTS/RWM ≥ 50× (per Hoffman-Gelman 2014 Table 1). Three orthogonal samplers converging to the same posterior moments is the canonical R-MUTUAL idiom.
- **R-MARGINAL-LIKELIHOOD-CROSS-VALIDATION 3/3:** M22 NestedSampling × M17 SMC × bridge-sampling (Gelman-Meng 1998 — defer to M27 v2) all compute `log Z` for the same conjugate Beta-Binomial model; pinned to 1e-3 vs analytic `log Z = log B(α + k, β + N − k) − log B(α, β)`. **First slot in `reality` to ship Bayes-factor computation.**
- **R-DETAILED-BALANCE-PIN 3/3:** M2 / M10 / M12 detailed-balance check via reverse-time chain: simulate forward chain → reverse → forward should equal detailed-balance product `π(x)·P(x→y) = π(y)·P(y→x)` to 1e-9. Tests the Hastings correction term implementation.
- **R-NUTS-DUAL-AVERAGING-PIN:** Hoffman-Gelman 2014 Algorithm 5 dual-averaging step-size adaptation converges to target acceptance 0.8 ± 0.02 across 4 benchmark posteriors (Gaussian, banana, funnel, hierarchical-8-schools) per Stan reference at 1e-2.
- **R-EFFECTIVE-SAMPLE-SIZE-PIN:** M5 ESS (Geyer-1992 IMSE) on a known AR(1) chain with `ρ = 0.9` should converge to `ESS/N → (1−ρ)/(1+ρ) = 0.0526` to 1e-2 at N=10⁴; cross-pin to PyMC `pymc.ess()` on identical golden-vector chain.
- **R-GELMAN-RUBIN-PIN:** M6 split-Rhat on 4 chains × 1000 samples from N(0,1) should give `R̂ < 1.01` (Vehtari-Gelman-Simpson-2021 threshold); on chains-stuck-in-different-modes-of-Mixture should give `R̂ > 1.5` (mode-detection diagnostic).
- **R-AFFINE-INVARIANCE-PIN (M16):** emcee on `N(0, Σ)` and `N(0, AΣAᵀ)` for arbitrary affine A should achieve identical ESS/iter (Goodman-Weare 2010 Theorem 1) to 1e-2 — distinguishes M16 from M3 RWM where bad scaling tanks ESS.

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| M1 | RNGSampler interface | 80 | 1 | — (architectural keystone) |
| M2 | MetropolisHastings | 140 | 1 | M1 |
| M3 | RandomWalkMetropolis | 80 | 1 | M2 |
| M4 | GibbsSampler | 150 | 1 | M1 |
| M5 | EffectiveSampleSize | 120 | 1 | signal.FFT |
| M6 | GelmanRubinRhat (split) | 110 | 1 | — |
| M7 | SliceSampling (Neal 2003) | 210 | 1 | M1 |
| M8 | EllipticalSliceSampling | 180 | 1 | M1, linalg.Cholesky |
| M9 | LeapfrogIntegrator | 80 | 2 | slot-204 chaos.Leapfrog |
| M10 | HamiltonianMonteCarlo | 280 | 2 | M9, autodiff |
| M11 | NUTS (dual-avg + mass adapt) | 340 | 2 | M9 |
| M12 | MALA | 150 | 2 | M1, autodiff |
| M13 | PreconditionedMALA | 80 | 2 | M12, linalg.CholeskySolve |
| M14 | RiemannManifoldHMC | 280 | 2 | M9, infogeo.Fisher |
| M15 | ParallelTempering | 210 | 3 | M2 (or M10) |
| M16 | AffineInvariantEnsemble (emcee) | 190 | 3 | M1 |
| M17 | SequentialMonteCarlo | 270 | 3 | M2, M5 |
| M18 | MultipleTryMetropolis | 180 | 3 | M2 |
| M19 | AdaptiveMetropolis (Haario 2001) | 150 | 3 | M2, Welford |
| M20 | DRAM (Haario 2006) | 210 | 3 | M19 |
| M21 | ReversibleJumpMCMC | 280 | 4 | M2 |
| M22 | NestedSampling-Skilling-2006 | 340 | 4 | M1 |
| M23 | PseudoMarginalMCMC | 150 | 4 | M2 |
| M24 | ParticleMCMC-PMMH | 250 | 4 | M17, M23 |
| M25 | NRJ = Newton-Raphson MALA | 120 | 4 | M12, autodiff Hessian |
| M26 | ContinuousTimeMCMC-PDMP | 250 | 4 | M1 |
| **Σ** | | **4,420** | | |

Pure-glue ratio: ~75% of LOC is composition over `optim.SimulatedAnnealing`'s acceptance test, `optim/genetic.go`'s Box-Muller, `linalg.{Cholesky, CholeskySolve}`, `signal.FFT`, and slot-204's `chaos.Leapfrog`. ~25% is genuinely-new MCMC math (NUTS doubling-tree at ~250 LOC; nested-sampling ellipsoidal-rejection at ~200 LOC; RJMCMC dimension-matching at ~200 LOC; PMMH SMC-inner at ~180 LOC; PDMP Bouncy-Particle-Sampler / Zig-Zag at ~250 LOC are the only non-trivial fragments).

---

## 6. Recommended PR sequence

**PR-0: M1 RNGSampler interface (~80 LOC source, ~120 LOC tests, single day) — architectural keystone**
Lift `optim.GeneticAlgorithm`'s anonymous `interface{ Float64() float64 }` to a public `optim.RNG` with `NormFloat64()` + `Intn()`. Ship adapter for `*math/rand.Rand`. **Co-ship with slot-195 N3, slot-117 prob/random.Gaussian, slot-235/236/237 PR-0 — 5+ Block-C slots' Tier-2 unblocks simultaneously.** Promote `optim/genetic.go:58-65`'s inlined Box-Muller to `optim/rng.go::NormFloat64()` (public).

**PR-1: M2 + M3 + M4 + M5 + M6 + M7 (~810 LOC source, ~600 LOC tests, 2 engineer-days)**
The gradient-free Tier-1 keystone. Lifts `optim.SimulatedAnnealing` to the Metropolis-Hastings sampler family. **First MCMC anything in `reality`.** Ships M5 ESS + M6 split-Rhat together because every published Bayesian-inference paper since 2014 reports both. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 on RWM × Gibbs × Slice convergence to N(0,Σ) at d=5.

**PR-2: M8 EllipticalSliceSampling (~180 LOC, half-day)**
Standalone primitive that sits exactly between PR-1 and PR-3. Cross-link slot-237 GP (M8 is the canonical sampler for GP latent variables).

**PR-3: M9 + M10 + M11 (~700 LOC, 3 engineer-days) — production HMC**
The single highest-value PR in this review. Lands HMC + NUTS in one go because they share the leapfrog-and-mass-matrix bookkeeping. **Stan/PyMC parity on a single Go module with zero deps.** Saturates R-NUTS-DUAL-AVERAGING-PIN.

**PR-4: M12 MALA + M13 PreconditionedMALA + M25 NRJ Newton-Raphson MALA (~350 LOC, 2 engineer-days)**
The Langevin-family triple. PR-4 lands together because all three share the `(ε²/2) ∇log p` drift term + Hastings-correction asymmetric-proposal density.

**PR-5: M16 AffineInvariantEnsemble emcee (~190 LOC, 1 engineer-day)**
Standalone. Goodman-Weare 2010 + Foreman-Mackey 2013 — the dominant astronomy/cosmology sampler since 2013, currently zero-dep-Go-absent. **Gradient-free** so it lands without depending on PR-3.

**PR-6: M19 AdaptiveMetropolis + M20 DRAM (~360 LOC, 2 engineer-days)**
Adaptive-MCMC family. Haario-Saksman-Tamminen 2001 + Haario-Laine-Mira-Saksman 2006. Industrial-strength alternative to NUTS for non-differentiable posteriors.

**PR-7: M14 RiemannManifoldHMC (~280 LOC, 2 engineer-days)**
Standalone-ish (depends on PR-3 leapfrog + slot-153 FisherFromDist). Saturates R-RIEMANN-MANIFOLD-IDENTITY pin (3/3) showing RMHMC ~10× ESS/grad on highly-curved posteriors.

**PR-8: M15 ParallelTempering (~210 LOC, 1 engineer-day)**
Shared kernel with slot-180 stat-mech. Coordinated landing: ship `prob/mcmc/parallel_tempering.go` once, slot-180 `physics/statmech/` consumes it for Ising/Heisenberg lattice-spin replica exchange.

**PR-9: M22 NestedSampling-Skilling-2006 (~340 LOC, 2 engineer-days)**
Standalone. **First marginal-likelihood Z computation** in `reality`. Saturates R-MARGINAL-LIKELIHOOD-CROSS-VALIDATION 3/3 vs SMC vs analytic Beta-Binomial.

**PR-10: M17 SequentialMonteCarlo (~270 LOC, 2 engineer-days)**
Builds on PR-1 (MetropolisHastings rejuvenation). Ships log-Z marginal-likelihood as a side-effect (cross-pin to PR-9).

**PR-11: M18 MultipleTryMetropolis + M23 PseudoMarginalMCMC (~330 LOC, 2 engineer-days)**
Tier-3+4 advanced primitives. PR-11 unlocks PR-12.

**PR-12: M24 ParticleMCMC-PMMH (~250 LOC, 2 engineer-days)**
The state-space-model crown jewel. Inner SMC (M17) + outer pseudo-marginal (M23). Cross-link signal/timeseries.

**PR-13: M21 ReversibleJumpMCMC (~280 LOC, 3 engineer-days) — Tier-4 frontier**
Trans-dimensional MCMC. Heaviest single primitive after NUTS; document Jacobian-bookkeeping pitfalls extensively.

**PR-14: M26 ContinuousTimeMCMC-PDMP (~250 LOC, 3 engineer-days) — Tier-4 frontier**
Bouchard-Côté 2018 Bouncy-Particle-Sampler + Bierkens 2019 Zig-Zag. Hot 2018-present research area, hardest single PR.

Total: ~4,420 LOC source + ~2,800 LOC tests across 14 PRs over ~28 engineer-days. **PR-1 (Tier-1 + diagnostics) is single-week-shippable and delivers MCMC-anything**. **PR-3 NUTS is the single highest-value primitive.** PR-9 NestedSampling is the single most-distinct-from-other-slot-238-primitives. PR-12 PMMH is the highest-leverage state-space-model unlock.

---

## 7. Cross-cutting blockers and slot-co-shipping

- **prob/random.Gaussian (Box-Muller / Marsaglia-Tsang)** — **EIGHTEENTH** Block-C review demanding it (slots 117/184/188/202/215/216/217/227/228/229/230/231/232/233/235/236/237/**238**). Co-ship as Tier-0 in PR-0 with M1 RNGSampler. Single 200-LOC commit unblocks every Block-C Tier-2.
- **autodiff for ∇log p** — gates M10/M11/M12/M13/M14/M25. Caller-side: user supplies `gradLogp(x, out)` closure; consumer's autodiff usage is consumer-side test only. Direct dependency = zero.
- **chaos.Leapfrog** — gates M9/M10/M11/M14. Cross-link slot-204 (which scopes `chaos/symplectic.go` keystone). Slot-238's M9 is a thin re-export of slot-204's `chaos.Leapfrog`; if slot-204 not landed first, ship private copy in `prob/mcmc/leapfrog.go` and consolidate at v2.
- **linalg.CholeskyDecompose / CholeskySolve** — gates M8/M13/M14. Already shipping per slot-097/099. Direct re-use, zero connective tissue.
- **signal.FFT** — gates M5 ESS via Wiener-Khinchin autocovariance. Already shipping. Direct re-use.
- **infogeo.FisherFromDistribution** — gates M14 RMHMC. Cross-link slot-153 S2 (proposed). Defer M14 to after slot-153 lands FisherFromDist.

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **M2 MetropolisHastings:** acceptance target 0.234 (d→∞, Roberts-Gelman-Gilks 1997) and 0.44 (d=1, Gelman-Roberts-Gilks 1996); auto-tuning deferred to M19.
- **M5 ESS:** Geyer-1992 IMSE truncates at first negative-pair sum; document the alternative Vehtari-2021 ranks-folded estimator (recommended for non-stationary chains).
- **M6 R-hat:** Vehtari-Gelman-Simpson-2021 split-Rhat is 2x stricter than Brooks-Gelman-1998 — adopt the modern threshold `R̂ < 1.01` not legacy `< 1.1`.
- **M9 Leapfrog:** symplectic-2nd-order, energy error `O(ε²)` over time `T = ε·L`; for HMC acceptance > 0.8 require `ε² · max‖∇²U‖ ≪ 1` (Neal-2011 §5.4).
- **M10 HMC:** mass matrix `M⁻¹` should approximate posterior covariance; identity is OK for spherical posteriors, fails for highly-correlated ones (use M11 NUTS auto-tuning or M14 RMHMC).
- **M11 NUTS:** doubling-tree max-depth 10 (Hoffman-Gelman 2014 default); document memory `O(2^max_depth · d)`.
- **M12 MALA:** optimal acceptance 0.574 (Roberts-Rosenthal 1998); 0.234 is wrong for MALA. Step size `ε ~ d^{-1/3}` per Roberts-Rosenthal-1998.
- **M14 RMHMC:** generalised leapfrog requires implicit-implicit-explicit splitting; fixed-point iteration tolerance 1e-6 default; document the divergence hazard at high curvature.
- **M16 emcee:** require K ≥ 2D + 2 walkers per Foreman-Mackey-2013; default K = 4D. Stretch parameter a=2 default per Goodman-Weare 2010 §3.
- **M17 SMC:** ESS-driven adaptive temperature schedule must respect `β_{t+1} > β_t`; bisection on β to hit target ESS / N = 0.5. Resample only when ESS drops below threshold (default N/2).
- **M19 AM:** `s_d = 2.4²/d` per Roberts-Gelman-Gilks 1997; warmup `t_0` typically 1000 samples; document the diminishing-adaptation requirement (Roberts-Rosenthal 2007) for asymptotic correctness.
- **M22 NestedSampling:** ellipsoidal rejection enlargement factor `e ≥ 1.1` to maintain prior coverage (Mukherjee-Parkinson-Liddle 2006); termination criterion `Δ log Z < 1e-3` after first stable 100 iterations.
- **M25 NRJ Newton-Raphson MALA:** Hessian must be PSD for valid Cholesky preconditioner; **fall back to identity-mass MALA when Hessian is non-PSD** (do not modify Hessian — that breaks reversibility).

---

## 9. Distinct from prior agents (provenance)

- **slot-117 prob-missing** — names prob/random.Gaussian as Tier-0; **slot-238 is the SECOND consumer** of that interface (after slot-195 N3). Co-ship.
- **slot-153 prob-infogeo** — names FisherFromDistribution; M14 RMHMC is the **consumer-side test pin** of slot-153 S2.
- **slot-180 stat-mech** — names Metropolis on Ising lattice + Wang-Landau + Wolff cluster + replica-exchange. Slot-180 owns the **physical-energy-axis** version (logp = -βE for spin systems); slot-238 owns the **Bayesian-posterior-axis** version (logp = -log posterior). M15 ParallelTempering shared kernel: ship in `prob/mcmc/`, consume in `physics/statmech/`.
- **slot-195 SGLD/SGHMC/SG-NUTS** — explicit PARTIAL OVERLAP. Slot-195 owns **stochastic-gradient + mini-batch noise** axis (data-subsampling). Slot-238 owns **full-data-likelihood** axis (no mini-batches; user supplies full `logp` and `gradLogp`). Disjoint primitive rosters by design: slot-195 N1-N22 (all stochastic-gradient), slot-238 M1-M26 (all full-likelihood ergodic-MCMC). **Shared base: both consume M1 = N3 RNGSampler interface; ship one canonical RNG type.**
- **slot-204 symplectic** — names `chaos/symplectic.go` keystone with leapfrog + Verlet + Yoshida + Forest-Ruth. Slot-238 M9 is a **thin consumer-side re-export** of slot-204's `chaos.Leapfrog`. Coordinated landing: slot-204 ships first, slot-238 PR-3 imports it.
- **slot-014/015 autodiff** — names `Grad`, `Hessian`. M10/M11/M12/M14/M25 all consume `gradLogp` and (M14/M25) `hessLogp` user-supplied closures. Slot-238's autodiff dependency is consumer-side test only.
- **slot-169 prob-optim** — names VI/EM/MAP as deterministic fits; slot-238's HMC/NUTS/MALA are the **MCMC alternative** to the same Bayesian inference problem. **slot-169-S15 SVGD** (Stein Variational Gradient Descent) is the gradient-flow alternative to MCMC; ship as separate primitive in slot-169, not slot-238.
- **slot-227 UQ** — names Bayesian-model-comparison, marginal-likelihood; **slot-238 M22 NestedSampling is the consumer-side delivery** of slot-227's Z-computation requirement.
- **slot-228 Bayesian nonparametrics** — DPM / Pitman-Yor; uses slot-238 M4 GibbsSampler for the conditional-conjugate updates and M11 NUTS for the variance-component hyperparameters.
- **slot-237 Gaussian Process** — uses slot-238 M8 EllipticalSliceSampling for GP-latent-variable models; slot-237 G7 GaussianProcessPosterior + slot-238 M8 ESS = production GP-classification stack.
- **slot-184 synergy-linalg-prob** — names Cholesky-based Gaussian sampling (Σ = LLᵀ; sample LZ); reuses M1 RNGSampler.

---

## 10. Bottom line

`reality` v0.10.0 ships **ZERO Markov-chain Monte Carlo samplers** despite shipping a Boltzmann-acceptance simulated-annealer (`optim/metaheuristic.go:73`) that is one rename away from being a Metropolis sampler, an inlined Box-Muller in `optim/genetic.go:58-65` that is the only Gaussian sampler in the repo, a `prob/markov.go:99-139` finite-state walker with no detailed-balance check, and an `autodiff/` package that supplies the ∇log p that HMC/NUTS need. **Twenty-six MCMC primitives M1-M26 totalling ~4,420 LOC of pure connective tissue** stand up the entire Geyer-Robert-Casella-Brooks-Gelman-Jones-Meng-2011-Handbook-of-MCMC + Stan + PyMC + emcee canon on existing v0.10.0 surfaces.

Eight primitives (M1-M8) are Tier-1 ship-today against zero new infrastructure beyond the **M1 RNGSampler interface keystone** (~80 LOC, single day, co-ship with slot-195 N3, slot-117 prob/random, slot-235/236/237 PR-0). Six are Tier-2 needing M1 + M9-leapfrog (cross-link slot-204) + autodiff. Six are Tier-3 needing M2 plus various extensions. Six are Tier-4 frontier (RJMCMC, NestedSampling, PMMH, PDMP).

The cheapest 1-week-shippable bundle is **PR-1 + PR-2 (M1+M2-M8) ~990 LOC** which lands the **first MCMC anything in `reality`** plus the canonical convergence diagnostics (ESS + split-R-hat) and saturates R-MUTUAL-CROSS-VALIDATION 3/3 on RWM × Gibbs × Slice for a 5D correlated Gaussian. The single highest-value PR is **PR-3 M9+M10+M11 (~700 LOC, 3 engineer-days)** which lands HMC + NUTS in one go = Stan/PyMC parity on a single Go module with zero deps. The single most-distinct cutting-edge contribution is **PR-9 M22 NestedSampling-Skilling-2006 (~340 LOC)** = first marginal-likelihood Z computation in `reality` for Bayesian-model-comparison + Bayes-factor. The crown-jewel composition is **PR-12 M24 ParticleMCMC-PMMH (~250 LOC)** = inner SMC (M17) + outer pseudo-marginal (M23) for state-space-model intractable-likelihood inference.

**Reality is unusually well-positioned for this slot because (i) `optim.SimulatedAnnealing`'s Boltzmann acceptance test is *already* the Metropolis-1953 kernel — slot-238 just lifts cooling from <1 to =1 and emits trajectory-not-bestX; (ii) `optim/genetic.go`'s inlined Box-Muller is the only Gaussian sampler in the repo — slot-238 promotes it to public `optim.RNG.NormFloat64()` keystone; (iii) `linalg.{Cholesky, CholeskySolve}` covers MALA-preconditioning + RMHMC + elliptical-slice-sampling sample-from-Gaussian; (iv) `signal.FFT` covers ESS via Wiener-Khinchin autocovariance; (v) slot-204's pending `chaos.Leapfrog` covers the symplectic substrate HMC needs; (vi) `autodiff` covers ∇log p user-supplied via consumer-closures; (vii) the consumer-side-placement rule (per agents 158/159/160/166/167/168/169/195) recommends `prob/mcmc/` sub-package mirroring `prob/copula/`+`prob/conformal/`+`prob/evt/` placement convention.**

Differentiation: 26/26 primitives unique to slot-238 sampler-axis. Cross-cutting blockers shared with 18+ Block-C slots: prob/random.Gaussian (Tier-0, EIGHTEENTH demand), autodiff for ∇log p (consumer-side closure, no direct edge), chaos.Leapfrog (slot-204 keystone), linalg.{Cholesky, CholeskySolve} (already shipping), signal.FFT (already shipping). Architectural recommendation: slot-238 **owns** `prob/mcmc/`; co-ship slot-117/195/235/236/237 PR-0 RNG; consume slot-204 leapfrog; supply M22-NestedSampling Z-computation as the public face of slot-227's marginal-likelihood requirement; M8-ellipticalSlice as the public sampler for slot-237's GP-classification.

The single most important conceptual identity this slot pins: **`optim.SimulatedAnnealing` cooling=1 trajectory-emission = `prob.MetropolisHastings`**. Slot-238 is **the rename PR**.
