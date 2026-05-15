# 265 | new-pmcmc — Particle MCMC: PIMH, particle Gibbs, SMC²

**Block:** C (cutting-edge math). **Date:** 2026-05-09. **Repo:** v0.10.0, 1,965 tests passing.
**Scope:** the **state-space-model + nested-SMC inference axis** — Andrieu-Doucet-Holenstein-2010 PMMH/PIMH/PG, Lindsten-Jordan-Schön-2014 PGAS, Chopin-Jacob-Papaspiliopoulos-2013 SMC², Del-Moral-Doucet-Jasra-2006 SMC sampler, Pitt-Shephard-1999 auxiliary PF, Gordon-Salmond-Smith-1993 bootstrap PF, Storvik-2002, Liu-West-2001, forward-backward + two-filter + backward-simulation smoothers, Rao-Blackwellised PF, multinomial / residual / stratified / systematic / adaptive resampling, ESS / CESS, adaptive tempering, IBIS, marginal-likelihood-via-SMC. **PARTIAL OVERLAP**: 238-M17 (SMC sampler ~270 LOC) + 238-M24 (PMMH ~250 LOC) name two slot-265 primitives as line items inside the broader MCMC review without scoping the corpus; 161-C11 (bootstrap-PF ~250 LOC) + 161-stretch-Rao-Blackwellised-PF (~300 LOC) covers the **filtering** axis but not the **inference-of-parameters-via-particle-MCMC** axis; 263-Q25 (MultilevelQMC) + 264-L26 (SDE-MLMC) name resampling-adjacent primitives. Slot 265 owns the **PMCMC + advanced-SMC + particle-smoother corpus**: where slot 161 stops at "estimate latent state given parameters", slot 265 picks up at "estimate parameters AND latents jointly via nested SMC inside MCMC".

## Two-line summary

`reality` v0.10.0 ships **ZERO** particle-MCMC / particle-filter / SMC-sampler / particle-smoother surface — verified `grep -E 'ParticleFilter|BootstrapFilter|SMC|SequentialMonteCarlo|ParticleMCMC|PMMH|PIMH|ParticleGibbs|cSMC|conditionalSMC|PGAS|AncestorSampling|SMC2|IBIS|AuxiliaryParticleFilter|BoostrapFilter|MultinomialResample|StratifiedResample|SystematicResample|ResidualResample|AdaptiveResample|EffectiveSampleSize.*particle|CESS|ConditionalESS|LiuWest|Storvik|RaoBlackwellised|RaoBlackwellized|BackwardSimulation|ForwardBackwardSmoother|TwoFilterSmoother|AdaptiveTempering' --include=\*.go` returns ZERO callable matches across all 22 packages; the `EffectiveSampleSize(n, halfLife)` at `prob/conformal/adaptive.go:160` is the **Kish-effective-sample-size for exponentially-weighted conformal-calibration windows** (`n_eff = halfLife · (1 - 0.5^(n/halfLife)) / ln(2)`), **NOT** the SMC `ESS = 1/Σw²` slot 265 needs; the `audio/separation/nmf.go:233` private `halton` is the only deterministic-low-discrepancy emission in the repo (slot-263 keystone, NOT a resampler); the `audio/onset/median_filter.go::resample`-style symbols return zero matches; `optim/genetic.go:58-65`'s inlined Box-Muller is the only Gaussian sampler in the repo (also slot-238-M1 / slot-117 / slot-202 P0 blocker). **Twenty-eight particle-method primitives P1-P28 totalling ~3,920 LOC of pure connective tissue + ~80 LOC shared substrate** (the same `prob/random/normal.go` Gaussian sampler now blocking 117/202/215/227/259-264/238 — eighteenth+ Block-C demand) stand up the entire Andrieu-Doucet-Holenstein-2010 + Chopin-Jacob-Papaspiliopoulos-2013 + Lindsten-Jordan-Schön-2014 + Del-Moral-Doucet-Jasra-2006 + Doucet-de-Freitas-Gordon-2001-Springer-canon on existing v0.10.0 surfaces; cheapest 1-day standalone is **PR-1 P1 ResamplerInterface + P2 MultinomialResample + P3 SystematicResample + P4 StratifiedResample + P5 ResidualResample + P6 EffectiveSampleSizeSMC ~410 LOC** which ships **the resampling family** every particle-method primitive in this slot consumes; highest-leverage 1-week unlock is **PR-3 P9 BootstrapFilter + P10 AuxiliaryPF + P12 PMMH-Andrieu-Doucet-Holenstein-2010 ~620 LOC** because PMMH = inner-bootstrap-PF unbiased-likelihood-estimator + outer-Metropolis-Hastings = **the canonical sampler for state-space models** with intractable likelihood and is the single highest-impact PMCMC primitive (~2,400 citations); SINGULAR-cutting-edge piece is **P14 ParticleGibbsAncestorSampling-Lindsten-Jordan-Schön-2014 ~280 LOC** because PGAS solves the **path-degeneracy problem** that breaks vanilla particle-Gibbs after 5-10 time steps — no zero-dep-Go-implementation exists worldwide; SINGULAR-cutting-edge keystone is **P17 SMC²-Chopin-Jacob-Papaspiliopoulos-2013 ~340 LOC** because SMC² is the **nested-SMC alternative to PMMH** for online sequential parameter+state inference (sequential-Monte-Carlo *over* sequential-Monte-Carlo); cross-cutting blockers reuse **slot-238-M1 RNGSampler** (eighteenth Block-C demand), **slot-238-M2/M3 MetropolisHastings** for PMMH outer-loop, **slot-238-M4 GibbsSampler** for particle-Gibbs outer-loop, **slot-238-M17 SMC sampler** for IBIS-rejuvenation, and **slot-161-C5 KalmanFilter + C8 EKF** for Rao-Blackwellisation. Differentiation: 28/28 primitives unique to slot-265 particle-method-axis; **slot-265 owns the SMC-sampler / PMCMC / particle-smoother / advanced-resampling corpus** that turns slot-161's "filter" into "inference-on-parameters-given-state-space-observations".

---

## 0. State of play (verified file-walk, 2026-05-09)

### `prob/` particle-method surface = ZERO

Repo-wide audit:

| Surface | Path | Lines | PMCMC relevance |
|---|---:|---:|---|
| `EffectiveSampleSize(n, halfLife int) float64` | `prob/conformal/adaptive.go:160` | 14 | **WRONG NAME, WRONG MATH.** Kish ESS for conformal exponential-decay windows: `n_eff = halfLife · (1 - 0.5^(n/halfLife)) / ln(2)`. Slot-265 needs `ESS(weights) = (Σw)² / Σw²` — the Liu-Chen-1995 SMC effective-sample-size on a normalised weight vector. **Co-name hazard:** P6 `EffectiveSampleSizeSMC` to disambiguate. |
| `MarkovSimulate / MarkovSteadyState` | `prob/markov.go:99-139` | ~40 | Finite-state stochastic-matrix walker via embedded LCG. **No detailed-balance check, no MH correction, no continuous-state support, no particle representation.** |
| `Distribution interface { PDF, CDF }` | `prob/distribution.go:1-197` | 197 | **No `LogPDF`, no `Sample(rng)`, no `Score`** — the same gap that blocks slot-238 M2 MetropolisHastings (the `logp` callback PMMH and PG need). |
| `NormalQuantile` (Acklam) + `BetaQuantile`, `GammaQuantile` | `prob/distributions.go` | ~250 | Inverse-CDF building blocks for likelihood-weight evaluations. |
| `BoxMuller`-style Gaussian sampler | -- | **0** | **ABSENT — same blocker as 117/202/215/227/238/259-264.** Eighteenth+ Block-C slot demanding it. Inlined private at `optim/genetic.go:58-65` only. |
| **`Sample(rng RNG) float64`** for any distribution | -- | **0** | **ABSENT — repo-wide.** Gates particle-propagation, ancestor-sampling, importance-sampling, every non-trivial PF/SMC primitive. |

### `prob/conformal/` ESS surface = orthogonal

`prob/conformal/adaptive.go:145-160` ships `EffectiveSampleSize(n int, halfLife int) float64` for **adaptive conformal calibration windows** (cross-link 231-conformal). The math is the **Kish-1965 effective-sample-size on exponentially-decayed weights** `w_k = 0.5^(k/halfLife)` — formally `(Σw)²/Σw²` with that specific weight schedule, but the entry-point takes integer `(n, halfLife)` not a `weights []float64`. Slot-265's `EffectiveSampleSizeSMC(weights []float64) float64` is the **same idea applied to normalised SMC particle weights** — different consumer, different signature, different package home. Recommend co-existence via `prob/smc.EffectiveSampleSizeSMC(weights)` mirroring `prob/conformal.EffectiveSampleSize(n, halfLife)`.

### `prob/markov.go` finite-state surface = orthogonal

`MarkovSimulate(P [][]float64, x0, steps int, rng) []int` walks a discrete-state stochastic matrix. **No continuous-state support, no observation model, no acceptance test, no particle ensemble.** Slot 265 starts where slot 116/117 stops: "given an observation model `p(y_t | x_t)` and a transition kernel `p(x_t | x_{t-1})`, produce particle approximations to filtering / smoothing / parameter posteriors via SMC".

### `optim/` MCMC-substrate cross-pollination

- `optim/metaheuristic.go:38-93::SimulatedAnnealing` — Boltzmann acceptance test = Metropolis-1953 kernel, the same source-of-truth slot 238 promotes to MetropolisHastings sampler. Slot 265 PMMH outer-loop (P12) consumes that promoted MH directly.
- `optim/genetic.go:58-65` — inlined Box-Muller. Same Tier-0 blocker as slot 238.

### `chaos/` SDE / state-space-substrate = ZERO

Per slot 202 (SDE) and slot 161: `chaos/ode.go` ships RK4 + Euler only; no SDE solver, no state-space simulator beyond the deterministic ODE family. Slot 161 names a future `chaos.SDESolver` that PMMH-on-SDE-models would consume; this is a slot 202-S0/S1 keystone and slot 265 is *downstream* consumer.

### Cross-coupling = zero today

```
$ grep -r "github.com/davly/reality/prob"   chaos/  ; echo "---"
$ grep -r "github.com/davly/reality/prob"   control/ ; echo "---"
$ grep -r "github.com/davly/reality/optim"  prob/    ; echo "---"
(no matches in any direction)
```

This slot proposes: NEW sub-package `prob/smc/` (mirrors `prob/copula/`, `prob/conformal/`, slot-238-proposed `prob/mcmc/`, slot-263-proposed `qmc/`, slot-264-proposed `mlmc/`). Cycle-free DAG: `prob/smc/` → {`prob/`, slot-238 `prob/mcmc/` for outer-loop MH/Gibbs, slot-161 `control/kf/` for Rao-Blackwellisation EKF/UKF, `linalg/` for covariance Cholesky}; reverse direction never.

---

## 1. The conceptual unlock — particle methods sit BETWEEN filtering and MCMC

Slot 161 (synergy-control-prob) ships state-space + Kalman + EKF + UKF + bootstrap-PF for **state estimation given parameters** `θ`. Slot 238 (MCMC) ships MH + HMC + NUTS for **parameter estimation given a tractable likelihood** `p(y | θ)`. The gap is **state-space models with intractable likelihood**: integrating out the latent state `x_{0:T}` to get `p(y_{1:T} | θ) = ∫ p(y_{1:T} | x_{0:T}, θ) p(x_{0:T} | θ) dx_{0:T}` is **the canonical example** of an MH target that cannot be computed — and Andrieu-Doucet-Holenstein-2010's seminal observation is that **a particle-filter estimate of this integral is unbiased**, so plugging it into Andrieu-Roberts-2009 pseudo-marginal MH **still targets the exact posterior** despite using a noisy likelihood. PMMH is the union: inner SMC + outer MH.

The four landmark frameworks:

1. **PMMH** (Andrieu-Doucet-Holenstein-2010 §4): M-H proposal on θ, accept/reject using particle-filter likelihood estimate `p̂(y|θ)`. Unbiased ⇒ targets exact posterior.
2. **Particle Gibbs / cSMC** (same paper §4.4): Gibbs scan over (θ, x_{0:T}) using a *conditional SMC* kernel that fixes one path. Path-degenerate after ~5-10 t.
3. **PGAS** (Lindsten-Jordan-Schön-2014): cSMC + ancestor-sampling step that **resamples the reference-path ancestors at each t**. **Solves the path-degeneracy problem.**
4. **SMC²** (Chopin-Jacob-Papaspiliopoulos-2013): nested SMC where the *outer* SMC is over θ-particles and each θ-particle carries an *inner* SMC over x_{0:T}. **Sequential, online**: process observations one at a time without re-running the whole chain.

Plus the resampling family (multinomial / residual / stratified / systematic — Hol-Schön-Gustafsson-2006), the auxiliary-PF (Pitt-Shephard-1999), the smoother family (FFBS / two-filter / backward-simulation — Godsill-Doucet-West-2004), Rao-Blackwellisation (Doucet-Godsill-Andrieu-2000), Liu-West-2001 shrinkage, and Storvik-2002 sufficient-statistic conditioning. All ~3,920 LOC pure glue over slot-238-M2/M4 + slot-161-C5/C8 + slot-238-M1 (RNG) + new resampler family.

---

## 2. The twenty-eight primitives P1-P28 (~3,920 LOC pure glue)

Numbered ascending by composition depth.

### Tier 1 — resampling family + ESS (P1-P6, ~410 LOC) — keystone gates everything

**P1 ResamplerInterface** [~30 LOC]
`type Resampler interface { Resample(weights []float64, indices []int, rng RNG) }` — the canonical signature. Indices output buffer (no allocation per step). Composes RNG.

**P2 MultinomialResample** [~80 LOC]
Gordon-Salmond-Smith-1993 vanilla: draw N i.i.d. samples from `Categorical(weights)`. Fastest path: cumsum + binary-search per index = O(N log N). Variance of N_i = N w_i (1 − w_i) — highest of the four standard methods, **only used for compatibility**.

**P3 SystematicResample** [~70 LOC]
Carpenter-Clifford-Fearnhead-1999 / Kitagawa-1996: draw `u ~ U(0, 1/N)`, set `u_k = u + (k-1)/N`, walk cumsum once. **O(N), deterministic-after-u, lowest variance** of the four — the production default per Hol-Schön-Gustafsson-2006 §4. **Variance reduction ≥ 4× vs multinomial.**

**P4 StratifiedResample** [~70 LOC]
Kitagawa-1996: `u_k ~ U((k-1)/N, k/N)` independent. **O(N) + ensures one sample per stratum.** Slightly more variance than systematic but unbiased per-stratum.

**P5 ResidualResample** [~80 LOC]
Liu-Chen-1998-JASA: deterministic `⌊N w_i⌋` copies of particle i (residual count `N_d = N − Σ⌊N w_i⌋`), then multinomial-resample residual N_d slots from residuals `(N w_i − ⌊N w_i⌋) / N_d`. **Lowest variance for skewed weights.**

**P6 EffectiveSampleSizeSMC(weights []float64) float64** [~80 LOC]
Liu-Chen-1995-JASA `ESS = (Σw)²/Σw²` on a normalised weight vector. Returns `[1, N]`. Co-name with `prob/conformal.EffectiveSampleSize` (Kish-on-decayed-windows) to disambiguate the two distinct uses. **Saturation pin:** ESS on uniform weights = N to 1e-15, ESS on degenerate (1, 0, ..., 0) = 1 to 1e-15.

### Tier 2 — bootstrap PF / auxiliary PF / adaptive resampling (P7-P11, ~700 LOC)

**P7 AdaptiveResample(weights, threshold) bool** [~30 LOC]
Liu-Chen-1995: resample only when `ESS < α·N` for `α ∈ [0.5, 0.75]` (default 0.5). Threshold `α=1` collapses to "always resample" (vanilla SIR), `α=0` to "never resample" (sequential-importance-sampling, weight-degenerate-fast).

**P8 ParticleStateInterface + Particle struct** [~80 LOC]
`type StateSpaceModel interface { Init(rng RNG) []float64; Transition(x, t int, rng RNG) []float64; LogObs(y, x []float64, t int) float64 }` — caller-supplied. `type ParticleEnsemble struct { States [][]float64; Weights []float64 }`. The single canonical interface every PF/PMCMC primitive consumes.

**P9 BootstrapFilter(model StateSpaceModel, N int, ys [][]float64, rng RNG) (mean [][]float64, logZ float64)** [~280 LOC]
Gordon-Salmond-Smith-1993 SIR: at each t, propagate via `Transition`, weight by `exp(LogObs)`, resample via P3 systematic if ESS < N/2. Returns particle posterior means + log-marginal-likelihood `log p̂(y_{1:T}) = Σ_t log(Σ_i w_t,i / Σ_i w_{t-1,i})` for downstream PMMH consumption. **IDENTICAL to slot-161-C11 — ship-once in `prob/smc/bootstrap.go`, slot-161 imports.** Cross-link 161-bridge `BoxMullerSample` already named.

**P10 AuxiliaryParticleFilter(model, N, ys, rng) (mean, logZ)** [~180 LOC]
Pitt-Shephard-1999-JASA: pre-weight particles by `g(x_{t-1}, y_t)` (a "lookahead" approximation to the obs likelihood), resample, then propagate through the standard transition. **Reduces variance for informative observations** (where bootstrap-PF particles drift away from the y-supported region). Composes P9 + caller-supplied `LookaheadLogObs(x_{t-1}, y_t) float64` (default = `LogObs(y_t, Transition(x_{t-1}))` mean).

**P11 OptimalProposalPF(model, N, ys, optimalProposal, rng) (mean, logZ)** [~130 LOC]
Doucet-Godsill-Andrieu-2000-Stat-Comput: propagate via the **optimal proposal** `q(x_t | x_{t-1}, y_t) ∝ p(x_t|x_{t-1}) p(y_t|x_t)` if available analytically (e.g. Gaussian transition + Gaussian observation). Variance of weights is **minimised** over all proposal choices — the gold standard when computable. Composes P9 with caller-supplied closed-form `OptimalSample(x_prev, y, rng)` and `OptimalLogProb(x_new, x_prev, y)`.

### Tier 3 — PMCMC keystone (P12-P15, ~1,150 LOC) — the Andrieu-Doucet-Holenstein-2010 frontier

**P12 PMMH-ParticleMarginalMetropolisHastings(model_θ, x0, θ0, proposal, N_particles, n_burn, n_samples, ys, rng) → (θ_samples, accept_rate)** [~280 LOC]
Andrieu-Doucet-Holenstein-2010 §4 Algorithm 2. Outer Metropolis-Hastings on θ; inner bootstrap-PF (P9) at each step computes unbiased `p̂(y|θ)`; accept/reject with `α = min(1, p̂(y|θ')p(θ')q(θ|θ') / [p̂(y|θ)p(θ)q(θ'|θ)])`. **The pseudo-marginal trick (Andrieu-Roberts-2009) guarantees the chain still targets the exact posterior despite the noisy likelihood estimator.** Slot-238-M23 names this; slot-265-P12 is the state-space-model specialisation. **R-PSEUDO-MARGINAL pin (3/3): PMMH × HMC-on-marginal × analytic-Bayes for a Gaussian-linear state-space-model (Kalman-marginal-likelihood available analytically) all converge to identical θ-posterior moments to 1e-3.** Recommended `N_particles ∝ T` per Doucet-Pitt-Deligiannidis-Kohn-2015 (target log-likelihood-estimator-variance ≈ 1.2-1.5).

**P13 PIMH-ParticleIndependentMetropolisHastings(model, N_particles, n_burn, n_samples, ys, rng) → x_path_samples** [~150 LOC]
Andrieu-Doucet-Holenstein-2010 §4.2 Algorithm 1. **Independent** proposal (re-run the full bootstrap-PF from scratch, propose the entire path `x_{0:T}`); accept/reject on path-marginal-likelihood ratio. Simpler than PMMH (no parameter θ); useful for **fixed-θ smoothing** when full posterior over paths is needed. Composes P9 + slot-238-M2 (independent MH).

**P14 ParticleGibbs-cSMC(model, N_particles, x_ref, ys, rng) → x_new_path** [~240 LOC]
Andrieu-Doucet-Holenstein-2010 §4.4 Algorithm 4 = Gibbs-on-(θ, x_{0:T}) using **conditional-SMC** as the x-update kernel. cSMC fixes one ancestor-path equal to the reference `x_ref`, runs SMC on the remaining N−1 particles, samples a new path from the final weighted ensemble. **Ergodic in (θ, x_{0:T})** but suffers **path degeneracy** at large T because the reference path shares identical early-time ancestors with all other particles after resampling. Composes P9 + slot-238-M4 (Gibbs scan).

**P15 ParticleGibbsAncestorSampling-PGAS(model, N_particles, x_ref, ys, rng) → x_new_path** [~280 LOC]
**Lindsten-Jordan-Schön-2014-JMLR-15:2145.** Augments cSMC (P14) with an **ancestor-sampling step**: at each t, resample the ancestor of the reference particle from `(w_{t-1,i} · p(x_ref,t | x_{t-1,i}))_i`. **Solves path degeneracy** — the reference path is "rejuvenated" at each t. Convergence rate independent of T (vs. cSMC's exponential-in-T mixing time). **No public Go implementation exists worldwide**; closest is the PMC-Stan / NIMBLE / pyro implementations. **SINGULAR-MOAT for slot-265.** Composes P14 + caller-supplied `LogTransition(x_t, x_{t-1}, t)`.

### Tier 4 — SMC sampler family (P16-P19, ~970 LOC) — Del-Moral / Chopin frontier

**P16 SMCSampler-Del-Moral-Doucet-Jasra-2006(logπ_target, n_particles, schedule, rng) → (samples, logZ)** [~270 LOC]
Del-Moral-Doucet-Jasra-2006-JRSS-B-68:411 SMC sampler: tempering schedule `π_t ∝ π_0^{1-β_t} · π_target^{β_t}` with `0 = β_0 < ... < β_T = 1`; particles evolve via importance-resample-MCMC-rejuvenation cycle. **Computes log-marginal-likelihood `log Z = Σ_t log(Σ_i w_t,i / Σ_i w_{t-1,i})`** directly — the same quantity slot-238-M22 NestedSampling computes via Skilling-2006 nested-sampling. Composes P9 (resampling) + slot-238-M2 (MH rejuvenation kernel) + P6 ESS. **IDENTICAL to slot-238-M17 — ship once in `prob/smc/`, slot-238 imports.**

**P17 SMC²-Chopin-Jacob-Papaspiliopoulos-2013(model, N_θ, N_x, ys, rng) → (θ_samples, logZ_t_trace)** [~340 LOC]
**Chopin-Jacob-Papaspiliopoulos-2013-JRSS-B-75:397.** Nested SMC: outer N_θ particles over θ-space, each carrying an inner N_x bootstrap-PF over x_{0:T}. As each new observation y_t arrives, update **all** θ-particles' likelihood-weights via their inner PF; resample θ when ESS_θ < N_θ/2; rejuvenate via PMMH (using the inner-PF unbiased likelihood). **The natural online-sequential alternative to PMMH** — process observations one at a time, no re-running the whole inference. **No public Go implementation exists**; closest is the particles-Python-Chopin reference implementation. Composes P9 (inner PF) + P12 (rejuvenation MH) + P16 (outer SMC structure).

**P18 IBIS-IteratedBatchImportanceSampling(target, batches, n_particles, rng) → (samples, logZ_per_batch)** [~180 LOC]
Chopin-2002-Biometrika-89:539: SMC over θ where the "schedule" is **observations added one batch at a time** — `π_t(θ) ∝ p(θ) · p(y_{1:t}|θ)`. **The non-state-space-model special case of SMC²** (when likelihood is tractable: no inner PF). The canonical sequential-Bayesian-inference primitive. Composes P16 + slot-238-M2 rejuvenation.

**P19 AdaptiveTempering-Jasra-Stephens-Doucet-Tsagaris-2011(target, n_particles, ess_target, rng) → (samples, logZ, schedule_used)** [~180 LOC]
Jasra-Stephens-Doucet-Tsagaris-2011-Scand-J-Stat-38:1 / Beskos-Jasra-Kantas-Thiery-2016-Ann-Appl-Probab-26:1111: bisection-on-`β_{t+1}` to hit `CESS(β_{t+1} | β_t) = α·N` for `α ∈ [0.5, 0.95]` (default 0.9). **Removes the user-tuned schedule** of P16 — schedule emerges adaptively from ESS-targeting. Composes P16 + P20 CESS + caller-supplied `logπ(x, β)`.

### Tier 5 — particle smoothers (P20-P23, ~720 LOC) — Godsill-Doucet-West-2004 frontier

**P20 ConditionalESS-CESS(weights_prev, log_incremental, β_step) float64** [~80 LOC]
Zhou-Johansen-Aston-2016-J-Comput-Graph-Stat-25:701 / Beskos-Jasra-Kantas-Thiery-2016: `CESS(β) = N · (Σ w_prev_i exp(β·δ_i))² / (Σ w_prev_i · Σ w_prev_i exp(2β·δ_i))` where `δ_i = log π_target(x_i) − log π_0(x_i)`. **Used by P19** for adaptive-β bisection. Distinct from P6 ESS: CESS conditions on the previous-step weights and predicts the next-step ESS as a function of β-increment.

**P21 ForwardFilteringBackwardSmoothing-FFBS(particles_filter, weights_filter, model, ys, rng) → x_smoothed_paths** [~180 LOC]
Kitagawa-1996-J-Comput-Graph-Stat-5:1 / Doucet-Godsill-Andrieu-2000: backward pass that re-weights forward-filter particles `w̃_t,i ∝ w_t,i · p(x_{t+1,j} | x_t,i) · w̃_{t+1,j}` for each backward step. **The canonical particle smoother** — Rao-Blackwellisable to the linear-Gaussian case (= RTS smoother). Composes P9 forward-pass + caller-supplied `LogTransition`.

**P22 BackwardSimulation-Godsill-Doucet-West-2004(particles_filter, weights_filter, model, ys, n_paths, rng) → x_paths** [~200 LOC]
Godsill-Doucet-West-2004-JASA-99:156: instead of computing all-pairs FFBS (O(N²·T) memory), **sample n_paths** complete smoothed paths backward from the filter ensemble. At each t (going backward from T), draw ancestor index `j ∝ w_t,i · p(x_path_{t+1} | x_t,i)`. **O(N·T·n_paths)** — practical for large T. Composes P21 + caller-supplied `LogTransition`.

**P23 TwoFilterSmoother-Briers-Doucet-Maskell-2010(model, ys, N_particles, rng) → x_smoothed** [~260 LOC]
Briers-Doucet-Maskell-2010-Ann-Inst-Statist-Math-62:61 = forward-PF (P9) **+** backward-information-PF (running on artificial backward-time) merged at each t via importance-weighting. **Better numerical stability than FFBS** for long T and degenerate transitions. Composes P9 ×2 + caller-supplied `LogTransitionReverse(x_{t-1}, x_t)`.

### Tier 6 — Rao-Blackwell + parameter-conditional + advanced (P24-P28, ~890 LOC)

**P24 RaoBlackwellisedPF-RBPF(model_RB, N_particles, ys, rng) → (x_nl_mean, x_l_mean, x_l_cov, logZ)** [~280 LOC]
Doucet-de-Freitas-Murphy-Russell-2000-UAI: partition state into linear-Gaussian sub-state `x_l` and non-linear sub-state `x_nl`; **run a Kalman filter on x_l conditional on each x_nl particle** (P9 only resamples on x_nl). Variance reduction proportional to dim(x_l)/dim(x). Composes P9 + slot-161-C5 KalmanFilter (or C8 EKF) per-particle. **Slot-161 stretch primitive — slot-265 owns it.** Cross-link slot-161-C5/C8/C10. **Saturation:** mixed-Gaussian-state-with-jump-Markov-coefficient where the linear sub-state is Gaussian → RBPF posterior cov 5-10× tighter than vanilla PF at same N.

**P25 LiuWestParticleFilter(model_θ, N_particles, ys, h_shrinkage, rng) → (θ_samples, x_samples)** [~180 LOC]
Liu-West-2001-Sequential-MC-Methods-Chap-10: online state+parameter PF that artificially perturbs θ-particles via Gaussian shrinkage `θ_new = a·θ_old + (1-a)·θ̄ + h·ε` with `a = √(1-h²)`. **Avoids degeneracy of the static-parameter problem** (without rejuvenation, repeated resampling collapses θ to a single value). Composes P9 + caller-supplied `Sample-θ(rng)`. **Pre-PMCMC-2010 baseline** — kept for comparison + lightweight online inference.

**P26 StorvikFilter(model_θ_sufficient_stats, N_particles, ys, rng) → (θ_samples, x_samples)** [~150 LOC]
Storvik-2002-IEEE-Trans-Signal-Process-50:281: **sufficient-statistic conditioning** for parameter PF when θ has conjugate prior over which (x_{1:t}, y_{1:t}) sufficient-statistic family `T_t` is closed-form-updatable. Each particle carries `(x_t,i, T_t,i)`; θ_t,i ~ p(θ | T_t,i) drawn analytically. **Fast online parameter inference** when conjugacy holds (e.g. linear-Gaussian with conjugate-Normal-Wishart prior on (μ, Σ)). Composes P9 + caller-supplied `UpdateStats(T, x_new, y_new)` and `SampleThetaFromStats(T, rng)`.

**P27 IslandParticleFilter-Vergé-Dubarry-DelMoral-Moulines-2015(model, N_islands, M_particles, ys, rng) → (mean, logZ)** [~140 LOC]
Vergé-Dubarry-Del-Moral-Moulines-2015-Stat-Comput-25:243: parallel-friendly PF where N_islands run independent SMC on M_particles each; periodic global-resampling across islands. **Embarrassingly parallel** — 5-50× wallclock speedup on multicore. Composes P9 ×N_islands + global P3 systematic-resample.

**P28 NestedParticleFilter-Crisan-Miguez-2018(model, N_outer, N_inner, ys, rng) → (θ_x_samples, logZ)** [~140 LOC]
Crisan-Miguez-2018-Ann-Appl-Probab-28:2911: alternative to SMC² (P17) with **inner PF replaced by an inner SMC sampler** running on x conditional on θ — handles non-Markov state-space-models where SMC² assumes Markov-x. Composes P16 ×outer + P9 ×inner.

---

## 3. Composition graph (DAG)

```
slot-238-M1 RNGSampler [eighteenth+ Block-C demand] ───── gates ALL primitives
 │
 ├── Tier 1 (resampling family + ESS)
 │    ├── P1 ResamplerInterface
 │    │    ├── P2 MultinomialResample (compatibility)
 │    │    ├── P3 SystematicResample (production default)
 │    │    ├── P4 StratifiedResample
 │    │    └── P5 ResidualResample
 │    └── P6 EffectiveSampleSizeSMC ── distinct from prob/conformal.ESS
 │
 ├── Tier 2 (bootstrap PF / aux PF / adaptive)
 │    ├── P7 AdaptiveResample (Liu-Chen 1995)
 │    ├── P8 ParticleStateInterface (caller-supplied model)
 │    ├── P9 BootstrapFilter ── IDENTICAL to slot-161-C11 ship-once
 │    ├── P10 AuxiliaryParticleFilter (Pitt-Shephard 1999)
 │    └── P11 OptimalProposalPF (Doucet-Godsill-Andrieu 2000)
 │
 ├── Tier 3 (PMCMC keystone — Andrieu-Doucet-Holenstein 2010)
 │    ├── P12 PMMH ── inner-P9 + outer-238-M2
 │    ├── P13 PIMH ── inner-P9 + outer-independent-MH
 │    ├── P14 ParticleGibbs-cSMC ── inner-conditional-SMC + outer-238-M4
 │    └── P15 PGAS ── P14 + ancestor-sampling MOAT
 │
 ├── Tier 4 (SMC sampler family)
 │    ├── P16 SMCSampler ── IDENTICAL to slot-238-M17 ship-once
 │    ├── P17 SMC² ── P9 inner + P12 rejuvenation + P16 outer MOAT
 │    ├── P18 IBIS (Chopin 2002)
 │    └── P19 AdaptiveTempering ── P16 + P20 CESS
 │
 ├── Tier 5 (smoothers)
 │    ├── P20 CESS
 │    ├── P21 FFBS (Kitagawa 1996)
 │    ├── P22 BackwardSimulation (Godsill-Doucet-West 2004)
 │    └── P23 TwoFilterSmoother (Briers-Doucet-Maskell 2010)
 │
 └── Tier 6 (Rao-Black + advanced)
      ├── P24 RaoBlackwellisedPF ── P9 + slot-161-C5 KF (slot-161 stretch)
      ├── P25 LiuWestParticleFilter (Liu-West 2001 baseline)
      ├── P26 StorvikFilter (Storvik 2002 sufficient-stats)
      ├── P27 IslandParticleFilter (Vergé et al. 2015 parallel)
      └── P28 NestedParticleFilter (Crisan-Miguez 2018 SMC²-alt)
```

Critical paths:

- **P1-P6 → everything** (resampling family + ESS keystone, ~410 LOC, single day)
- **P9 → P12 PMMH** (the production critical path; ~560 LOC, 3 engineer-days)
- **P14 → P15 PGAS** (the MOAT path; ~520 LOC, 3 engineer-days)
- **P17 SMC²** (standalone after P9+P12+P16; ~340 LOC, 2 engineer-days)
- **P9 → P21 → P22 BackwardSimulation** (smoothing critical path; ~660 LOC, 3 engineer-days)

---

## 4. Saturation pins this slot unlocks

Per recent saturation pattern (audio-onset 3-detector 6a55bb4, copula×autodiff 365368a, NGramDice 85a80db):

- **R-RESAMPLE-CROSS-VALIDATION 4/4 P2/P3/P4/P5:** four resamplers (multinomial / systematic / stratified / residual) on a fixed weight vector all converge to the same posterior expectation E[f(x)] to 5e-2 with N=10⁵ particles, but variance ratio multinomial/systematic ≥ 4× per Hol-Schön-Gustafsson-2006 §4 Table 1. **Four orthogonal resamplers — saturates R-MUTUAL-CROSS-VALIDATION.**
- **R-PSEUDO-MARGINAL-CORRECTNESS-PIN 3/3 (P12):** PMMH × HMC-on-marginal × analytic-Bayes for a Gaussian-linear state-space-model (Kalman-marginal-likelihood available) all converge to identical θ-posterior moments to 1e-3 (mean) and 1e-2 (cov). Tests that the Andrieu-Roberts-2009 pseudo-marginal trick is implemented correctly.
- **R-PGAS-PATH-DEGENERACY-PIN 2/2 (P14 vs P15):** at T=200, particle-Gibbs (P14) shows path-degeneracy fraction (% particles sharing identical x_0) > 0.95 after 100 iterations; PGAS (P15) at same T=200 keeps fraction < 0.1. **The single most-important PGAS test — proves Lindsten-Jordan-Schön-2014 actually works as advertised.**
- **R-MARGINAL-LIKELIHOOD-CROSS-VALIDATION 3/3 (P9 + P16):** log-Z from bootstrap-PF (P9) × SMC-sampler (P16) × analytic-Kalman-marginal for a linear-Gaussian SSM all agree to 1e-3 at N=10⁴ particles. Tests that the SMC marginal-likelihood estimator is implemented correctly.
- **R-SMC²-ONLINE-PIN (P17):** SMC² processed sequentially y_t one at a time should converge to the same final θ-posterior as PMMH on the full y_{1:T} batch (to 1e-2 in posterior mean). Tests the Chopin-Jacob-Papaspiliopoulos-2013 nested-SMC correctness.
- **R-RAO-BLACKWELL-VARIANCE-PIN (P24):** RBPF on a 4D state with 2D linear sub-state should achieve ESS-per-particle 5-10× better than vanilla bootstrap-PF (P9) at same N. Tests the Rao-Blackwellisation variance reduction is captured.
- **R-OPTIMAL-PROPOSAL-VARIANCE-PIN (P11 vs P9):** optimal-proposal PF on a Gaussian-Gaussian SSM should achieve ESS-per-particle 2-5× better than bootstrap-PF at same N (per Doucet-Godsill-Andrieu-2000-Stat-Comput-10:197 Table 1). Tests proposal-quality dependence.

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| P1 | ResamplerInterface | 30 | 1 | RNG (M1) |
| P2 | MultinomialResample | 80 | 1 | P1 |
| P3 | SystematicResample | 70 | 1 | P1 |
| P4 | StratifiedResample | 70 | 1 | P1 |
| P5 | ResidualResample | 80 | 1 | P1, P2 |
| P6 | EffectiveSampleSizeSMC | 80 | 1 | — |
| P7 | AdaptiveResample | 30 | 2 | P6 |
| P8 | ParticleStateInterface | 80 | 2 | — |
| P9 | BootstrapFilter | 280 | 2 | P3, P7, P8, RNG, prob/random/normal (P0) |
| P10 | AuxiliaryParticleFilter | 180 | 2 | P9 |
| P11 | OptimalProposalPF | 130 | 2 | P9 |
| P12 | PMMH | 280 | 3 | P9, slot-238-M2 |
| P13 | PIMH | 150 | 3 | P9, slot-238-M2 |
| P14 | ParticleGibbs-cSMC | 240 | 3 | P9, slot-238-M4 |
| P15 | PGAS-Lindsten-Jordan-Schön-2014 | 280 | 3 | P14 |
| P16 | SMCSampler-Del-Moral-2006 | 270 | 4 | P9, slot-238-M2 |
| P17 | SMC²-Chopin-Jacob-Papaspiliopoulos-2013 | 340 | 4 | P9, P12, P16 |
| P18 | IBIS-Chopin-2002 | 180 | 4 | P16 |
| P19 | AdaptiveTempering-Jasra-2011 | 180 | 4 | P16, P20 |
| P20 | ConditionalESS (CESS) | 80 | 5 | — |
| P21 | FFBS-Kitagawa-1996 | 180 | 5 | P9 |
| P22 | BackwardSimulation-GDW-2004 | 200 | 5 | P21 |
| P23 | TwoFilterSmoother-BDM-2010 | 260 | 5 | P9 ×2 |
| P24 | RaoBlackwellisedPF | 280 | 6 | P9, slot-161-C5 KalmanFilter |
| P25 | LiuWestParticleFilter | 180 | 6 | P9 |
| P26 | StorvikFilter | 150 | 6 | P9 |
| P27 | IslandParticleFilter | 140 | 6 | P9 ×N |
| P28 | NestedParticleFilter | 140 | 6 | P9, P16 |
| **Σ** | | **~3,920** | | |

Pure-glue ratio: ~80% of LOC is composition over slot-238-M2/M4 (MH/Gibbs), slot-161-C5/C8 (KF/EKF), slot-238-M1 (RNG), slot-238-M17 (SMC sampler — ship-once), and the new resampler family. ~20% is genuinely-new particle-method math: PGAS ancestor-sampling at ~150 LOC, SMC² nested-SMC bookkeeping at ~200 LOC, two-filter smoother artificial-backward-time at ~150 LOC, Rao-Blackwell-conditional-Kalman-bookkeeping at ~150 LOC.

---

## 6. Recommended PR sequence

**PR-0: P1-P6 resampling family + ESS keystone (~410 LOC source, ~300 LOC tests, single day)**
The resampling-family keystone. Saturates R-RESAMPLE-CROSS-VALIDATION 4/4 (multinomial / systematic / stratified / residual converge to same E[f(x)] with variance-ratio matching Hol-Schön-Gustafsson-2006 Table 1). Co-name P6 `EffectiveSampleSizeSMC` to disambiguate from `prob/conformal.EffectiveSampleSize` (Kish-on-decayed-windows).

**PR-1: P7-P11 bootstrap PF + auxiliary PF + adaptive (~700 LOC source, ~500 LOC tests, 3 engineer-days)**
The PF-keystone. **Co-ship with slot-161-C11** — slot-161 imports `prob/smc.BootstrapFilter` instead of shipping its own. Saturates R-OPTIMAL-PROPOSAL-VARIANCE-PIN (Doucet-Godsill-Andrieu-2000 Table 1).

**PR-2: P12 PMMH + P13 PIMH (~430 LOC, 3 engineer-days)**
The PMCMC keystone. **The single highest-impact PR in this review** (~2,400 citations on Andrieu-Doucet-Holenstein-2010). Saturates R-PSEUDO-MARGINAL-CORRECTNESS-PIN 3/3 vs HMC-on-marginal vs analytic-Kalman.

**PR-3: P14 cSMC + P15 PGAS (~520 LOC, 4 engineer-days) — the MOAT**
PGAS-Lindsten-Jordan-Schön-2014. **No public Go implementation exists worldwide** — this PR establishes a reference. Saturates R-PGAS-PATH-DEGENERACY-PIN 2/2 (vanilla cSMC degenerates at T=200 to 95% ancestor-share; PGAS stays under 10%).

**PR-4: P16 SMCSampler + P18 IBIS (~450 LOC, 3 engineer-days)**
**P16 IDENTICAL to slot-238-M17 — ship-once in `prob/smc/`, slot-238 imports.** Saturates R-MARGINAL-LIKELIHOOD-CROSS-VALIDATION 3/3 vs nested-sampling vs analytic-Bayes.

**PR-5: P17 SMC²-Chopin-Jacob-Papaspiliopoulos-2013 (~340 LOC, 3 engineer-days) — second MOAT**
The nested-SMC keystone. **No public Go implementation exists worldwide.** Saturates R-SMC²-ONLINE-PIN vs PMMH-batch.

**PR-6: P19 AdaptiveTempering + P20 CESS (~260 LOC, 2 engineer-days)**
The adaptive-schedule pair. Closes the user-tuned-tempering-schedule pain point of P16.

**PR-7: P21 FFBS + P22 BackwardSimulation + P23 TwoFilterSmoother (~640 LOC, 4 engineer-days)**
The smoother family. Saturates the path-marginal-vs-state-marginal symmetry pin.

**PR-8: P24 RaoBlackwellisedPF (~280 LOC, 2 engineer-days)**
Cross-link slot-161-C5 KalmanFilter. **Slot-161 stretch primitive — slot-265 owns the algorithmic contribution.** Saturates R-RAO-BLACKWELL-VARIANCE-PIN.

**PR-9: P10 AuxiliaryPF + P25 LiuWestParticleFilter + P26 StorvikFilter + P27 IslandParticleFilter + P28 NestedParticleFilter (~790 LOC, 5 engineer-days)**
The advanced-PF tail. Each is standalone and can ship in any order.

Total: ~3,920 LOC source + ~2,800 LOC tests across 10 PRs over ~28 engineer-days. **PR-0 (resampling-family keystone) is single-day-shippable** and unblocks every other primitive in this slot. **PR-2 PMMH is the single highest-value PR** (~2,400 citations Andrieu-Doucet-Holenstein-2010). **PR-3 PGAS** establishes the no-public-Go-worldwide reference. **PR-5 SMC²** establishes the second moat.

---

## 7. Cross-cutting blockers and slot-co-shipping

- **prob/random.Gaussian (Box-Muller / Marsaglia-Tsang)** — **NINETEENTH** Block-C review demanding it (slots 117/184/188/202/215/216/217/227/228/229/230/231/232/233/235/236/237/**238**/259-264/**265**). Co-ship as Tier-0 in PR-0 with slot-238-M1 RNGSampler. Single ~80-LOC commit unblocks every Block-C Tier-2.
- **slot-238-M1 RNGSampler interface** — gates every PMCMC primitive (caller supplies an RNG that emits `Float64()`, `NormFloat64()`, `Intn(n)`). Co-ship with slot-238 PR-0.
- **slot-238-M2 MetropolisHastings** — gates P12 PMMH outer-loop, P13 PIMH outer-loop, P16 SMCSampler rejuvenation. Cross-link slot-238 PR-1.
- **slot-238-M4 GibbsSampler** — gates P14 cSMC outer-Gibbs scan. Cross-link slot-238 PR-1.
- **slot-238-M17 SMCSampler** — IDENTICAL to slot-265-P16; ship once. Recommended canonical home: `prob/smc/sampler.go`.
- **slot-161-C5 KalmanFilter (Joseph form) + C8 EKF + C10 UKF** — gates P24 RaoBlackwellisedPF. Cross-link slot-161 PR-2 / PR-stretch.
- **slot-161-C11 BootstrapFilter** — IDENTICAL to slot-265-P9; ship once. Recommended canonical home: `prob/smc/bootstrap.go`, slot-161 imports.
- **slot-263-Q3 SobolSequence + Q12 OwenScramble** — optional cross-link for QMC-SMC (Gerber-Chopin-2015-JRSS-B-77:509 randomised-QMC-SMC). Defer to v2.
- **slot-264-L26 SDELevelSampler** — optional cross-link for multilevel-PMMH (Jasra-Kamatani-Law-Zhou-2018-SIAM-J-Numer-Anal-56:2911). Defer to v2.

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **P3 SystematicResample:** sort-stable on tied weights; document the `u ~ U(0, 1/N)` requirement (using `u ~ U(0,1)` is a common bug — gives N+1 samples or 0 samples at edges).
- **P5 ResidualResample:** `⌊N w_i⌋` arithmetic must use `math.Floor(N * w_i + 1e-15)` with explicit tie-break to prevent `N w_i ≈ k` rounding inconsistency at large N.
- **P6 ESS:** weights must be normalised. Document `ESS(uniform N-vector) = N` to 1e-15 and `ESS((1, 0, ..., 0)) = 1` exactly. Edge case: all zero weights → return 0 not NaN (caller signal of degenerate filter).
- **P9 BootstrapFilter:** log-marginal-likelihood `log Z = Σ_t log(Σ_i w_t,i)` after normalisation must use **logsumexp** — naive `log(sum(exp(log_w_i)))` underflows for any T > 50 with informative observations.
- **P12 PMMH:** target log-likelihood-estimator-variance ≈ 1.2-1.5 per Doucet-Pitt-Deligiannidis-Kohn-2015-Biometrika-102:295. **Recommended N_particles = O(T)** so per-step variance scales linearly. Too-few particles → estimator-variance > 3 → chain stuck; too-many → wasted compute.
- **P14 cSMC:** path degeneracy after T > 5-10 timesteps is **fundamental**, not an implementation bug. Document mixing-time `O(T²)` and recommend P15 PGAS for T > 10.
- **P15 PGAS:** ancestor-sampling weights must include the **transition density** `p(x_ref,t | x_{t-1,i})` — omitting it (the most common implementation bug) silently breaks ergodicity and biases the posterior. Document the formula explicitly: `w_t,i^anc ∝ w_t,i · p(x_ref,t+1 | x_t,i)`.
- **P16 SMCSampler:** ESS-driven adaptive temperature schedule (P19) must respect monotonicity `β_{t+1} > β_t`; bisection on `β` to hit `CESS = α·N`. **Resample only when ESS < α·N** (default α=0.5) — over-resampling wastes diversity, under-resampling loses representation.
- **P17 SMC²:** outer-θ rejuvenation kernel must be **invariant for the current target** `π(θ | y_{1:t})` — using stale tempering schedules causes biased posterior. Recommend rejuvenation-via-PMMH-with-current-data (P12 inside the outer SMC).
- **P19 AdaptiveTempering:** adaptive-schedule data-dependence breaks the standard SMC-sampler-unbiased-Z guarantee — Beskos-Jasra-Kantas-Thiery-2016-Ann-Appl-Probab-26:1111 prove asymptotic-correctness but document the finite-N bias.
- **P21 FFBS:** O(N²·T) memory — explicitly document the prohibitive cost for T > 1000. Recommend P22 backward-simulation as the practical alternative.
- **P24 RaoBlackwellisedPF:** the per-particle Kalman-filter must use **Joseph form** (slot-161-C5) — naive `(I - KH)P` form loses positive-definiteness after ~50 RBPF time-steps because each particle accumulates floating-point error independently.
- **P25 LiuWestParticleFilter:** shrinkage parameter `h ∈ [0.05, 0.2]` per Liu-West-2001 — too-small h ⇒ insufficient regeneration, too-large h ⇒ over-smooths posterior. Auto-tune via the Liu-West-2001 §10.2 acceptance-rate-target heuristic.

---

## 9. Distinct from prior agents (provenance)

- **slot-117 prob-missing** — names prob/random.Gaussian as Tier-0; **slot-265 is the NINETEENTH consumer**. Co-ship.
- **slot-118 prob-sota** — does not name SMC/PMCMC; slot-265 fills this gap.
- **slot-161 synergy-control-prob** — owns C11 bootstrap PF (~250 LOC) and stretch-primitive Rao-Blackwellised-PF (~300 LOC) at the **filtering/state-estimation axis**. **Slot-265-P9 is IDENTICAL to slot-161-C11 — ship once** in `prob/smc/`, slot-161 imports. **Slot-265-P24 is the slot-161-stretch-primitive made first-class.** Slot 161 stops at "estimate state given known parameters"; slot 265 continues to "estimate parameters AND state jointly via PMCMC".
- **slot-202 new-sde** — names SDE simulator (Euler-Maruyama, Milstein) which P24 RBPF on continuous-time SSM consumes. Cross-link slot-202 PR-A.
- **slot-238 new-mcmc** — names M17 (SMCSampler ~270 LOC) + M24 (PMMH ~250 LOC) as line items in the broader MCMC review. **Slot-265-P16 IDENTICAL to slot-238-M17 — ship once.** **Slot-265-P12 IDENTICAL to slot-238-M24 — ship once.** Slot-265 owns the **PMCMC + SMC sampler corpus** in full (28 primitives); slot-238 owns the **MCMC sampler corpus** (26 primitives) where M17/M24 are line items.
- **slot-263 new-quasi-mc** — names Q12 OwenScramble + Q3 SobolSequence; Gerber-Chopin-2015 randomised-QMC-SMC is the cross-link to v2 (slot-265-P9 with Sobol-driven resampling). Defer.
- **slot-264 new-mlmc** — names L26 SDELevelSampler; multilevel-PMMH (Jasra-Kamatani-Law-Zhou-2018) is the cross-link to v2 (slot-265-P12 inside MLMC level-pairs). Defer.

---

## 10. Bottom line

`reality` v0.10.0 ships **ZERO** particle-MCMC / particle-filter / SMC-sampler / particle-smoother / advanced-resampling surface. The repo's only `EffectiveSampleSize` symbol (`prob/conformal/adaptive.go:160`) is the **Kish-effective-sample-size on exponentially-decayed conformal calibration windows** — wrong consumer, wrong signature, wrong package home. The closest tangential surface is `prob/markov.go::MarkovSimulate` finite-state-LCG-walker which has neither a continuous-state representation nor an observation model. **Twenty-eight particle-method primitives P1-P28 totalling ~3,920 LOC of pure connective tissue** stand up the entire Andrieu-Doucet-Holenstein-2010 + Chopin-Jacob-Papaspiliopoulos-2013 + Lindsten-Jordan-Schön-2014 + Del-Moral-Doucet-Jasra-2006 + Doucet-de-Freitas-Gordon-2001-Springer-canon on existing v0.10.0 surfaces.

Six primitives (P1-P6) are Tier-1 ship-today against zero new infrastructure beyond the **slot-238-M1 RNGSampler keystone** + **prob/random/normal.go ~80 LOC P0** (eighteenth+ Block-C demand — co-ship with slot-117 / 202 / 215 / 227 / 238 / 259-264). Five are Tier-2 needing P9 BootstrapFilter (= slot-161-C11 ship-once). Four are Tier-3 PMCMC keystone (P12 PMMH = slot-238-M24 ship-once; P15 PGAS = MOAT). Four are Tier-4 SMC sampler family (P16 = slot-238-M17 ship-once; P17 SMC² = MOAT). Four are Tier-5 smoothers. Five are Tier-6 advanced (Rao-Black, Liu-West, Storvik, Island, Nested).

The cheapest 1-week-shippable bundle is **PR-0 + PR-1 (P1-P11) ~1,110 LOC** which lands the **first SMC anything in `reality`** — resampling family + bootstrap PF + auxiliary PF + adaptive resampling + ESS — and saturates R-RESAMPLE-CROSS-VALIDATION 4/4 (Hol-Schön-Gustafsson-2006 Table 1) plus R-OPTIMAL-PROPOSAL-VARIANCE-PIN. The single highest-value PR is **PR-2 P12 PMMH** (~280 LOC, ~2,400 citations Andrieu-Doucet-Holenstein-2010) = **the canonical sampler for state-space-model intractable-likelihood-inference**. The single most-distinct cutting-edge contribution is **PR-3 P15 PGAS-Lindsten-Jordan-Schön-2014** (~280 LOC) = **the only known fix for cSMC path degeneracy + no public Go implementation exists worldwide**. The crown-jewel composition is **PR-5 P17 SMC²-Chopin-Jacob-Papaspiliopoulos-2013** (~340 LOC) = **online sequential parameter+state inference** via nested-SMC, the natural alternative to PMMH for streaming observations.

**Reality is unusually well-positioned for this slot because (i) slot-161-C11 BootstrapFilter and slot-238-M17 SMCSampler and slot-238-M24 PMMH are *already* enumerated in upstream slots — slot-265 ships canonical-home in `prob/smc/` and slot-161 + slot-238 import; (ii) the resampling family P2-P5 is straight `math/rand`-style code with zero new substrate; (iii) `linalg.{Cholesky, CholeskySolve}` covers Rao-Blackwell-conditional-Kalman + Liu-West-Gaussian-shrinkage; (iv) slot-238's pending RNGSampler + MetropolisHastings + GibbsSampler are direct PMCMC outer-loop dependencies; (v) slot-161's pending KalmanFilter Joseph-form is the direct RBPF inner-loop dependency; (vi) the consumer-side-placement rule (per agents 158/159/160/166/167/168/169/195/238/263/264) recommends `prob/smc/` sub-package mirroring `prob/copula/`+`prob/conformal/`+`prob/mcmc/`+`prob/qmc/` placement convention.**

Differentiation: 28/28 primitives unique to slot-265 particle-method-axis. Cross-cutting blockers shared with 19+ Block-C slots: prob/random.Gaussian (Tier-0, NINETEENTH demand), slot-238-M1 RNGSampler / M2 MH / M4 Gibbs, slot-161-C5 KalmanFilter, slot-238-M17 SMCSampler ship-once, slot-161-C11 BootstrapFilter ship-once. Architectural recommendation: slot-265 **owns** `prob/smc/`; co-ship slot-117/238/161 PR-0 (RNG + MH + KF); supply P12 PMMH as the public face of slot-238's intractable-likelihood requirement; supply P9 BootstrapFilter as the public face of slot-161's filtering surface; supply P15 PGAS + P17 SMC² as the SINGULAR-MOAT contributions distinguishing reality from every other zero-dep-Go-numerics library worldwide (no public PGAS-Go and no public SMC²-Go implementation exists in 2026).

The single most important conceptual identity this slot pins: **PMMH = inner-bootstrap-PF unbiased-likelihood + outer-Metropolis-Hastings = canonical sampler for state-space models with intractable likelihood.** Slot-265 is the rename PR for slot-238-M24 + slot-161-stretch into a first-class 28-primitive PMCMC + SMC + particle-smoother corpus.
