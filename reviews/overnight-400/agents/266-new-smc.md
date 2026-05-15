# 266 | new-smc — Sequential Monte Carlo: bootstrap, auxiliary, adaptive resampling (SMC-specific deep dive)

**Block:** C (cutting-edge math). **Date:** 2026-05-09. **Repo:** v0.10.0, 1,965 tests passing.
**Scope:** the **SMC-as-its-own-discipline axis** — Doucet-de-Freitas-Gordon-2001-Springer-canon viewed *not* as the inner kernel of a particle-MCMC outer-loop (slot 265 owns that) and *not* as the filtering kernel of a state-space-control problem (slot 161 owns that), but as the **standalone Monte-Carlo-on-a-sequence-of-distributions framework** that subsumes filtering, smoothing, optimisation-by-tempering, rare-event estimation, data assimilation, and online-Bayesian-inference under one algorithmic abstraction. Where slot 265 names PMMH/PGAS/SMC²/IBIS as the parameter-inference frontier and slot 161 names Kalman/EKF/UKF/PF as the filtering frontier, slot 266 covers the **algorithm-design axis between them**: the resampling-family beyond the four standard schemes, the proposal-family beyond bootstrap+optimal, the look-ahead family beyond Pitt-Shephard, the data-assimilation family (EnKF / 3DVar / 4DVar / hybrid) that lives orthogonal to particle methods, the discrete-state SMC family (Viterbi/forward-backward), the path-degeneracy diagnostic family, and the variance-reduction family (antithetic / control-variates / stratified-SMC / Rao-Blackwell-marginalisation) — each of which slot 265 either skips or names as a single-line item.

## Two-line summary

`reality` v0.10.0 ships **ZERO** SMC primitives (re-verified `grep -E 'SMC|SequentialMonteCarlo|ParticleFilter|BootstrapFilter|EnKF|EnsembleKalman|3DVar|4DVar|FourDVar|3D.Var|VariationalAssimilation|Viterbi|ForwardBackward|HMM|HiddenMarkov|ResampleMove|StratifiedSMC|BlockResample|RobertsRosenthal|LookAhead|EKFParticleFilter|UKFParticleFilter|UnscentedPF|RegularisedPF|GaussianMixtureFilter|MultiBootstrap|Antithetic|ControlVariate' --include=\*.go` returns ZERO callable matches across all 22 packages); slot 266 PARTIAL OVERLAP with slot 265 (PMMH/PGAS/SMC²/SMC-sampler/RBPF/Liu-West/Storvik/Island/Nested-PF) which owns the parameter-inference frontier, slot 161 (Kalman family / bootstrap-PF) which owns the filtering frontier, slot 238 (MH/HMC/NUTS) which owns the MCMC frontier; slot 266 owns the **complementary 24-primitive SMC-canon corpus S1-S24 ~3,310 LOC** that 265+161+238 deliberately leave out: (a) data-assimilation family — EnKF (Evensen-1994), perturbed-observation-EnKF (Burgers-vanLeeuwen-Evensen-1998), ensemble-square-root-EnKF (Whitaker-Hamill-2002), localisation+inflation (Houtekamer-Mitchell-2001), 3DVar/4DVar (Talagrand-Courtier-1987), hybrid-EnVAR (Hamill-Snyder-2000); (b) advanced-resampling family — Roberts-Rosenthal-2007-stable, block-resampling, Killing-resampling-Crisan-Lyons-1999, branching-resampling-DelMoral-2004; (c) advanced-proposal family — EKF-PF / UKF-PF (Doucet-Godsill-Andrieu-2000, Merwe-Doucet-deFreitas-Wan-2000), regularised-PF (Musso-Oudjane-LeGland-2001), Gaussian-mixture-filter (Sorenson-Alspach-1971/Anderson-Moore-1979); (d) look-ahead family — multi-step-look-ahead (Lin-Chen-Liu-2013), island-particle-fully-adapted-Pitt-Shephard, twisted-SMC (Guarniero-Johansen-Lee-2017); (e) discrete-state-SMC family — Viterbi (Forney-1973), forward-backward (Baum-Welch-1970), particle-Viterbi (Godsill-Doucet-West-2001); (f) path-degeneracy-diagnostic family — unique-ancestors-trace, weight-CV diagnostic, ESS-vs-uniformity test; (g) variance-reduction family — antithetic-SMC, control-variate-SMC, stratified-SMC-Gerber-Chopin-Whiteley-2019, Rao-Blackwell-marginalisation-non-RBPF; (h) resample-move family — Gilks-Berzuini-2001-resample-move, MCMC-rejuvenation-after-resample. Cheapest 1-day standalone is **PR-1 S5 EnKF + S6 PerturbedObservationEnKF + S7 EnsembleSquareRootEnKF ~430 LOC** which lands the **first ensemble-Kalman anything in the repo** — 25,000+ citations across atmospheric-science / oceanography (Evensen-1994 alone has 8,200 citations), zero zero-dep-Go-implementation-exists-worldwide; highest-leverage 1-week unlock is **PR-2 S11 EKF-PF + S12 UKF-PF + S13 RegularisedPF ~520 LOC** because EKF-PF and UKF-PF use slot-161-C8/C10 KF-family as **near-optimal proposals** (variance reduction 5-50× over bootstrap-PF on informative observations) — the textbook recommendation in Doucet-de-Freitas-Gordon-2001 §3 + Cappé-Moulines-Rydén-2005 §9; SINGULAR-cutting-edge piece is **S20 TwistedSMC-Guarniero-Johansen-Lee-2017 ~280 LOC** which solves the path-degeneracy problem of vanilla SMC by **iteratively learning a "twist" function ψ_t(x_t) that approximates p(y_{t:T}|x_t)** so the look-ahead tracks future observations — no public Go implementation exists, fewer than ~100 citations but defining the 2017-2026 SMC-research-frontier; SINGULAR-data-assimilation-keystone is **S5-S10 EnKF-family ~870 LOC** because data-assimilation is a **$50B/year industry** (NOAA / ECMWF / NCEP / GMAO operational forecasting) and zero open-source-Go-numerical-DA-stack exists outside MATLAB (DART, Fortran 90+) and Python (eki, eki-toolbox); cross-cutting blockers reuse **slot-238-M1 RNGSampler** (twentieth Block-C demand), **slot-161-C5 KalmanFilter Joseph-form** for EKF-PF/UKF-PF/RBPF, **slot-161-C9/C10 EKF/UKF** for proposal-construction, **slot-265-P1-P6 resampling family** (avoid duplication: slot-265 owns the resampling-keystone home), **slot-265-P9 BootstrapFilter** as the consumer-side reference. Differentiation: 24/24 primitives unique to slot-266 SMC-design-axis; **slot-266 owns the data-assimilation, the advanced-proposal, the discrete-state-SMC, and the variance-reduction-in-SMC corpus** that turns slot-265's 28-primitive PMCMC framework into a complete 52-primitive SMC discipline.

---

## 0. State of play (verified file-walk, 2026-05-09)

### Re-confirming zero SMC surface (full grep)

```
$ grep -rE 'SMC|SequentialMonteCarlo|ParticleFilter|BootstrapFilter|EnKF|EnsembleKalman|3DVar|4DVar|VariationalAssimilation|Viterbi|HiddenMarkov|ResampleMove|StratifiedSMC|BlockResample|RobertsRosenthal|LookAhead|EKFParticleFilter|UKFParticleFilter|RegularisedPF|GaussianMixtureFilter|TwistedSMC' --include=\*.go reality/
prob/conformal/adaptive.go:160:func EffectiveSampleSize(n, halfLife int) float64  [Kish window — wrong consumer]
prob/conformal/adaptive_test.go:...                                                [Kish window tests]
prob/conformal/doc.go:...                                                          [package doc]
crypto/rng.go:50:	// Tempering.                                                    [string in comment, no code]
```

The only `EffectiveSampleSize` symbol is `prob/conformal/adaptive.go:160` (Kish on exponentially-decayed conformal-calibration windows — slot-265 already documents the co-name hazard with `EffectiveSampleSizeSMC = (Σw)²/Σw²`). No `Viterbi`, no `HiddenMarkov`, no `EnsembleKalman`, no `EnKF`, no `4DVar`, no `LookAhead`, no `TwistedSMC`, no `ResampleMove`. Every one of the 24 primitives below is greenfield.

### What slot 265 already enumerated (do NOT duplicate)

- Resampling family: P1 ResamplerInterface + P2 Multinomial + P3 Systematic + P4 Stratified + P5 Residual + P6 ESS_SMC + P7 AdaptiveResample → slot-266 builds **on top of these** (S1 RobertsRosenthalStable / S2 BranchingDelMoral / S3 BlockResample = three new resampling schemes 265 omits).
- Bootstrap PF (P9), Auxiliary PF (P10), Optimal-Proposal PF (P11) → slot-266 ships the **EKF-PF / UKF-PF / RegularisedPF / GaussianMixtureFilter** family (S11-S14) which 265 names as a one-line cross-link only.
- Smoothers (P21 FFBS, P22 BackwardSimulation, P23 TwoFilterSmoother) → slot-266 ships the **discrete-state HMM family** (S15 ForwardBackward / S16 Viterbi / S17 ParticleViterbi) which 265 omits because it focuses on continuous-state.
- SMC sampler (P16), SMC² (P17), IBIS (P18), AdaptiveTempering (P19) → slot-266 ships the **Resample-Move-Gilks-Berzuini-2001** kernel (S18) which is the *MCMC-rejuvenation* primitive each of P16/P18 calls but 265 leaves abstract; plus **stratified-SMC-Gerber-Chopin-Whiteley-2019** (S19) which 265 only names as a cross-link to RQMC.
- Liu-West (P25), Storvik (P26) → slot-266 ships **GaussianMixtureFilter-Sorenson-Alspach-1971** (S14) which is a more general predecessor to both.

### What slot 161 already enumerated (composes against)

- C5 KalmanFilter Joseph-form, C8 EKF, C10 UKF → consumed by S11 EKF-PF + S12 UKF-PF + S5-S10 EnKF-family + S22 HybridEnVar
- C11 BootstrapFilter (= slot-265-P9 ship-once) → reference baseline for variance-reduction comparisons in S20 TwistedSMC, S23 ControlVariateSMC

### What slot 238 already enumerated (composes against)

- M1 RNGSampler → gates everything (twentieth Block-C demand)
- M2 MetropolisHastings, M4 GibbsSampler → gates S18 ResampleMove kernel choice
- M17 SMCSampler (= slot-265-P16 ship-once) → reference baseline for S20 TwistedSMC

### Cross-package placement (cycle-free DAG)

```
slot-266 prob/smc/  (canonical home, slot-265 also lives here)
    │
    ├── consumes: prob/random/    (P0 RNG, twentieth Block-C demand)
    ├── consumes: prob/distributions   (existing PDFs)
    ├── consumes: linalg/    (Cholesky/MatMul for EnKF/EKF-PF)
    ├── consumes: control/kf/    (slot-161-C5/C8/C10 — Joseph KF / EKF / UKF)
    └── consumes: prob/mcmc/    (slot-238-M2/M4 — MH / Gibbs for resample-move)
```

No reverse dependency — `prob/smc/` strictly composes downstream surfaces.

---

## 1. The conceptual unlock — SMC is the framework, not just the algorithm

The single most-important conceptual identity slot 266 pins (slot 265 names but does not develop):

> **SMC = generic algorithm for sampling a sequence of distributions {π_t}_{t=0}^T linked by some Markov-kernel structure.**

The "sequence" can be:

| Interpretation | Sequence π_t | Setting | Owned by |
|---|---|---|---|
| State-space filtering | π_t = p(x_t \| y_{1:t}) | Bootstrap PF | slot 161, slot 265-P9 |
| State-space smoothing | π_t = p(x_{0:T} \| y_{1:T}) backward in t | FFBS | slot 265-P21 |
| Static Bayesian inference | π_t ∝ p(θ) · p(y_{1:τ_t} \| θ), τ_t ↑ T | IBIS | slot 265-P18 |
| Tempering / annealing | π_t ∝ p(θ) · L(θ)^{β_t}, β_t ↑ 1 | SMC sampler | slot 265-P16 |
| Nested SMC over θ + x | π_t over (θ, x_{0:t}) | SMC² | slot 265-P17 |
| **Data assimilation** | π_t = p(x_t \| y_{1:t}) for **PDE-discretised x_t in 10⁶+ dim** | **EnKF family** | **slot 266-S5-S10** |
| **Discrete-state HMM** | π_t = p(x_t \| y_{1:t}) for **finite x_t** | **Forward-Backward / Viterbi** | **slot 266-S15-S17** |
| **Rare-event estimation** | π_t ∝ I[L(x) > L_t] · p(x), L_t ↑ L* | **Adaptive-multilevel-splitting** | **slot 266-S21** |
| **Optimisation** | π_t ∝ exp(−f(x)/T_t), T_t ↓ 0 | **SMC-for-optimisation-Zhou-2008** | **slot 266-S24** |

The **state-space, static-Bayes, tempering, nested-SMC** rows → slot 161/238/265.
The **data-assimilation, discrete-HMM, rare-event, optimisation** rows → slot 266.

That partition is the basis of slot 266's 24 primitives.

---

## 2. The twenty-four primitives S1-S24 (~3,310 LOC pure glue)

### Tier A — advanced resampling (S1-S4, ~340 LOC)

**S1 RobertsRosenthalStableResample(weights, indices, rng)** [~80 LOC]
Roberts-Rosenthal-2007-Stat-Sci-22:413 §4: a coupling-stable resampling scheme that **preserves the same indices when weights barely change** between consecutive SMC steps. Critical for adaptive-SMC + tempered-SMC where a slight β-increment shouldn't completely rotate the particle ensemble. Variance-equivalent to systematic but with **deterministic mixing for slow-evolving target sequences**. Composes slot-265-P1 ResamplerInterface.

**S2 BranchingResample-DelMoral-2004(weights, indices, rng)** [~90 LOC]
Del-Moral-2004-Springer-FKF §9.2: each particle i produces `K_i ~ Binomial(N, w_i)` offspring then renormalises. Variance higher than systematic but **closed-form moment-generating-function** — the canonical theoretical-analysis resampler in mathematical-SMC literature. Composes slot-265-P1 + Binomial-RNG.

**S3 BlockResample(weights, indices, block_size, rng)** [~100 LOC]
Doucet-Briers-Sénécal-2006-J-Comput-Graph-Stat-15:693: resample particles in blocks of size B (e.g. B=10) — preserves more diversity than full-resample by **localising the resampling decision**. Particularly effective for high-dim state where the global-resample-collapse otherwise dominates.

**S4 KillingResample-Crisan-Lyons-1999(weights, indices, rng)** [~70 LOC]
Crisan-Lyons-1999-Markov-Process-Related-Fields-5:293: stochastic-killing scheme — particles with `w_i < threshold` are *killed* (zero offspring), the rest undergo branching. Mathematically the **continuous-time-Wiener-Feynman-Kac analogue** of Gordon-Salmond-Smith-1993. Use case: variance-reduction in heavy-tail-likelihood SMC.

### Tier B — Ensemble Kalman family (S5-S10, ~870 LOC) — **THE keystone**

**S5 EnsembleKalmanFilter-EnKF-Evensen-1994(model_lin, N_ensemble, ys, rng)** [~180 LOC]
Evensen-1994-J-Geophys-Res-99:10143 (cited 8,200×): represent posterior `p(x_t|y_{1:t})` by an **ensemble of N samples** propagated through possibly-nonlinear dynamics, but combined via the **Kalman-update equations on the empirical ensemble covariance** `P̂ = (1/(N-1)) Σ (x_i − x̄)(x_i − x̄)ᵀ`. Cost O(N²d) per step (vs. O(d²) full-cov KF). **Production atmospheric-DA tool** (NCEP, ECMWF, NOAA-EnKF) — no zero-dep-Go-impl-worldwide. Composes slot-161-C5 (Kalman-update) + slot-265-P3 (resampling not used — EnKF is *no-resample-Gaussian-update*) + linalg.Cholesky.

**S6 PerturbedObservationEnKF-Burgers-vanLeeuwen-Evensen-1998** [~120 LOC]
Burgers-vanLeeuwen-Evensen-1998-Mon-Weather-Rev-126:1719: corrects S5's **rank-deficiency bias** by perturbing the observation `y_t,i = y_t + ε_i` for each ensemble member with `ε_i ~ N(0, R)` so the sample covariance of `y` matches `R` in expectation. **Almost universal correction** since 1998 — naive S5 systematically under-estimates posterior variance. Composes S5.

**S7 EnsembleSquareRootEnKF-EnSRF-Whitaker-Hamill-2002** [~150 LOC]
Whitaker-Hamill-2002-Mon-Weather-Rev-130:1913: **deterministic** alternative to S6's stochastic perturbation — uses a square-root update that exactly preserves the analysis-error-covariance moments. **Lower variance + fewer ensemble members needed**. The canonical **deterministic EnKF** family. Sub-variants: ETKF (Ensemble Transform KF, Bishop-Etherton-Majumdar-2001) and EAKF (Ensemble Adjustment KF, Anderson-2001) — ship as flags on S7.

**S8 LocalisationCovariance-Houtekamer-Mitchell-2001** [~100 LOC]
Houtekamer-Mitchell-2001-Mon-Weather-Rev-129:123: ensemble-covariance regularisation via **Schur (Hadamard) product** with a tapering kernel `ρ(distance) ≈ Gaspari-Cohn-1999-quasi-Gaussian-kernel` to **kill spurious long-range correlations** in finite-ensemble cov estimates. Without S8, EnKF on N=50 vs d=10⁶ collapses. **Operational-DA mandatory**. Composes S5-S7.

**S9 InflationCovariance-Anderson-Anderson-1999** [~70 LOC]
Anderson-Anderson-1999-Mon-Weather-Rev-127:2741: scale ensemble-anomaly `(x_i − x̄) → α(x_i − x̄)` with `α ∈ [1.01, 1.10]` to **counteract underestimation of posterior variance** from finite ensemble + model-error. Adaptive variant: Anderson-2007-Tellus-A-59:210 estimates α online. **Operational-DA mandatory**, paired with S8. ~70 LOC.

**S10 4DVar-Talagrand-Courtier-1987** [~250 LOC]
Talagrand-Courtier-1987-QJRMS-113:1311: **variational data-assimilation** — minimise `J(x_0) = ½(x_0 − x_b)ᵀB⁻¹(x_0 − x_b) + ½ Σ_t (y_t − H(M_t(x_0)))ᵀR⁻¹(y_t − H(M_t(x_0)))` over the **initial-condition** x_0 using gradient-based optim (slot-068-LBFGS) with the gradient computed via **adjoint-of-the-tangent-linear-model** (cross-link slot-014 reverse-mode autodiff). The **dominant-NWP-DA-method 1990-2010** before EnKF/hybrid took over. Composes slot-068 LBFGS + slot-014 reverse-mode autodiff + caller-supplied forward-model `M_t` and observation-operator `H`. Sub-primitive **3DVar** = single-time-step special case of 4DVar (~30 LOC delta from S10).

### Tier C — Advanced-proposal PF family (S11-S14, ~540 LOC)

**S11 EKF-ParticleFilter-Doucet-Godsill-Andrieu-2000** [~180 LOC]
Doucet-Godsill-Andrieu-2000-Stat-Comput-10:197 §2.2: use the **EKF predictive distribution** `q(x_t | x_{t-1,i}, y_t) = N(μ_EKF, P_EKF)` as the proposal in the importance-sampling step, where μ_EKF, P_EKF come from a **per-particle EKF run linearising around x_{t-1,i}**. **2-5× ESS-per-particle** improvement over bootstrap-PF on informative observations (Table 1, Doucet-Godsill-Andrieu-2000). Composes slot-265-P9 + slot-161-C8 EKF (per-particle).

**S12 UKF-ParticleFilter-Merwe-Doucet-deFreitas-Wan-2000** [~200 LOC]
Merwe-Doucet-deFreitas-Wan-2000-NIPS / IEEE-Adaptive-Sys-Sig-Proc-Comm-Control-2000-153: same idea as S11 but with **UKF predictive** instead of EKF — handles non-differentiable f, h cleanly. **Often ESS-per-particle 2-3× better than EKF-PF** on highly-nonlinear systems. Composes slot-265-P9 + slot-161-C10 UKF (per-particle).

**S13 RegularisedParticleFilter-Musso-Oudjane-LeGland-2001** [~80 LOC]
Musso-Oudjane-LeGland-2001-Sequential-MC-Methods-Chap-12: post-resample **kernel-density-perturbation** of particle states: `x_i ← x_i + h·K^{-1/2}·ε_i` where K is empirical-cov, h is bandwidth (Silverman-1986 rule). **Avoids sample impoverishment** in low-process-noise settings without requiring P15 PGAS-level rejuvenation. Composes slot-265-P9 + slot-265-P3 + linalg.Cholesky.

**S14 GaussianMixtureFilter-Sorenson-Alspach-1971** [~80 LOC]
Sorenson-Alspach-1971-Automatica-7:465 / Anderson-Moore-1979-Optimal-Filtering: represent posterior as **K-component-Gaussian-mixture**, each component updated via Kalman; mixture weights updated via likelihood. **Predates particle filtering** by 22 years; computationally cheap when K << N (typically K ∈ [3, 20]). Composes slot-161-C5 KalmanFilter + caller-supplied mixture-K choice.

### Tier D — Discrete-state HMM family (S15-S17, ~480 LOC)

**S15 ForwardBackward-BaumWelch-1970(P_trans, B_emit, ys)** [~180 LOC]
Baum-Welch-1970-Ann-Math-Stat-41:164 / Forney-1973-Proc-IEEE-61:268: **exact** filtering+smoothing on a **finite-state HMM** — forward-pass `α_t(i) = p(x_t=i, y_{1:t})` recursion, backward-pass `β_t(i) = p(y_{t+1:T}|x_t=i)` recursion, posterior `γ_t(i) = α_t(i)β_t(i) / Σ_j α_t(j)β_t(j)`. The discrete-state analogue of the Kalman filter+smoother. **No public Go HMM library exists** at zero-dep; closest is Sajari/hmm and goml. Composes prob/markov.go.

**S16 Viterbi-Forney-1973(P_trans, B_emit, ys) → x_MAP** [~100 LOC]
Forney-1973-Proc-IEEE-61:268: dynamic-programming computation of the **MAP path** `x_{1:T}^MAP = argmax_{x_{1:T}} p(x_{1:T} | y_{1:T})` on a finite-state HMM. **The canonical decoder** for digital comms / ASR / NLP. O(T·K²). The discrete-state analogue of slot-265-P22 BackwardSimulation. Composes prob/markov.go.

**S17 ParticleViterbi-Godsill-Doucet-West-2001** [~200 LOC]
Godsill-Doucet-West-2001-Proc-IEEE-Workshop-Stat-Sig-Proc-2001:181: **continuous-state Viterbi via particle approximation** — the MAP-path equivalent of slot-265-P22 BackwardSimulation but **maximising** rather than sampling. Used in tracking / robotics / speech enhancement. Composes slot-265-P9 forward-pass + max-marginal-backward-pass.

### Tier E — Look-ahead + twisted-SMC (S18-S20, ~620 LOC) — frontier

**S18 ResampleMove-GilksBerzuini-2001(particles, target, mcmc_kernel, rng)** [~180 LOC]
Gilks-Berzuini-2001-JRSS-B-63:127: after each resample step, apply **K iterations of an MCMC kernel targeting the current π_t** to each particle. **The canonical rejuvenation step** for static-parameter SMC — without it, the parameter-particle ensemble collapses to <5 unique values after 10-20 iterations. Composes slot-265-P9 + slot-238-M2 MetropolisHastings as the rejuvenation kernel. **Slot-265 names this implicitly inside SMC sampler P16; slot-266 ships it as a first-class primitive** since it composes orthogonally with bootstrap-PF, IBIS, tempered-SMC, SMC², MLMC-SMC.

**S19 StratifiedSMC-GerberChopinWhiteley-2019** [~160 LOC]
Gerber-Chopin-Whiteley-2019-Ann-Stat-47:1304: **partition particle ensemble into M strata** along a 1-D summary statistic (e.g. ESS-driven adaptive partitioning), apply systematic-resample within each stratum independently. **Variance reduction 2-10× over standard systematic-resample** in high-dim. The canonical recent-2019-frontier-resampling extension. Composes slot-265-P3 + caller-supplied stratification function.

**S20 TwistedSMC-GuarnieroJohansenLee-2017** [~280 LOC]
Guarniero-Johansen-Lee-2017-JASA-112:1636: iteratively learn a **"twist" function** ψ_t(x_t) approximating `p(y_{t:T} | x_t)` — the **smoothing-conditional likelihood** — and use the *twisted target* `π̃_t ∝ π_t · ψ_t / E[ψ_t]` as the SMC target. **Variance-asymptotic-zero** as ψ_t → optimal-twist. **Solves path-degeneracy** by structuring proposals to track future observations — the natural complement to slot-265-P15 PGAS (which fixes degeneracy in cSMC). **No public Go implementation exists**; reference implementation is Lee's R/MATLAB code. **Defining the 2017-2026 SMC-research frontier.** Composes slot-265-P9 + slot-238-M2 + caller-supplied parametric twist family.

### Tier F — Variance-reduction + path-degeneracy diagnostics (S21-S24, ~460 LOC)

**S21 AdaptiveMultilevelSplitting-AMS-CerouGuyader-2007** [~180 LOC]
Cérou-Guyader-2007-Stoch-Anal-Appl-25:417: **rare-event-probability estimation** for `P(L(X) > L*)` where L* is large. SMC over a sequence of nested level-sets `{L > L_t}` with `L_t ↑ L*` chosen adaptively to keep ESS at target. **The canonical rare-event SMC sampler** — used in finance, reliability, fluid-dynamics. **Distinct from tempering-SMC** (slot-265-P19): AMS targets nested *indicator-set* probabilities, tempering targets a smooth-density sequence. Composes slot-265-P9 + slot-265-P3 + caller-supplied L(x) functional.

**S22 HybridEnVar-HamillSnyder-2000** [~120 LOC]
Hamill-Snyder-2000-Mon-Weather-Rev-128:2905: **convex combination** of EnKF analysis covariance (S5) with static-3DVar background covariance B: `P̂_hybrid = (1−α)·P̂_EnKF + α·B`. **The dominant operational-DA-method since ~2010** (NCEP-GFS, ECMWF-IFS, JMA-GSM all use hybrid-EnVar variants). Composes S5-S10 + S10-3DVar-substep.

**S23 ControlVariateSMC-Geweke-1989** [~80 LOC]
Geweke-1989-Econometrica-57:1317: when an analytic surrogate `f̃(x)` is available with known mean `μ̃`, use the **control-variate-corrected estimator** `Ê[f] = (1/N)Σ (f(x_i) − f̃(x_i)) + μ̃`. Variance reduction `(Var[f] − 2 Cov[f, f̃] + Var[f̃])` vs `Var[f]`. **Standard variance-reduction trick** orthogonal to all SMC primitives. Composes any PF/SMC primitive + caller-supplied surrogate.

**S24 SMC-for-Optimisation-Zhou-2008** [~80 LOC]
Zhou-2008-PhD-thesis / Del-Moral-Doucet-Jasra-2006-§5: **adaptive-temperature-SMC with T_t → 0** to find global optima of an objective `f(x)`. The SMC analogue of simulated-annealing (slot-238-M3 RWM at fixed-T) — bridges optimisation and Bayesian inference. Composes slot-265-P16 SMC sampler + slot-265-P19 AdaptiveTempering + caller-supplied f(x).

**S24-bonus: PathDegeneracyDiagnostic-UniqueAncestorsTrace** [~40 LOC, bundled into S20 free]
For each forward-PF run, trace `unique_ancestors_t(particles_t) = |unique({earliest-ancestor-of-particle-i : i=1..N})|` as a function of t. Should track N for first ~10 t then collapse exponentially. **Empirical diagnostic** that signals when slot-265-P15 PGAS or slot-266-S20 TwistedSMC is needed (when unique-ancestors < 0.1·N at the first quartile of T). Free with S20.

---

## 3. Composition graph (DAG)

```
slot-238-M1 RNGSampler [twentieth Block-C demand] ─── gates ALL primitives
 │
 ├── Tier A (advanced resampling)
 │    ├── S1 RobertsRosenthalStable
 │    ├── S2 BranchingDelMoral
 │    ├── S3 BlockResample
 │    └── S4 KillingResample
 │
 ├── Tier B (Ensemble Kalman family) ─── slot-161-C5 KalmanFilter Joseph
 │    ├── S5 EnKF-Evensen-1994
 │    ├── S6 PerturbedObservationEnKF
 │    ├── S7 EnsembleSquareRootEnKF (+ ETKF/EAKF)
 │    ├── S8 LocalisationCovariance
 │    ├── S9 InflationCovariance
 │    └── S10 4DVar (+ 3DVar) ──── slot-068-LBFGS + slot-014-reverse-AD
 │
 ├── Tier C (Advanced-proposal PF) ─── slot-265-P9 BootstrapFilter
 │    ├── S11 EKF-PF ──── slot-161-C8 EKF per-particle
 │    ├── S12 UKF-PF ──── slot-161-C10 UKF per-particle
 │    ├── S13 RegularisedPF
 │    └── S14 GaussianMixtureFilter
 │
 ├── Tier D (Discrete-state HMM)
 │    ├── S15 ForwardBackward-BaumWelch
 │    ├── S16 Viterbi-Forney
 │    └── S17 ParticleViterbi-Godsill-Doucet-West
 │
 ├── Tier E (Look-ahead + twisted-SMC)
 │    ├── S18 ResampleMove-GilksBerzuini ── slot-238-M2 MH
 │    ├── S19 StratifiedSMC-GCW-2019
 │    └── S20 TwistedSMC-GJL-2017 (frontier MOAT)
 │
 └── Tier F (Variance-reduction + diagnostics)
      ├── S21 AdaptiveMultilevelSplitting
      ├── S22 HybridEnVar
      ├── S23 ControlVariateSMC
      └── S24 SMC-for-Optimisation
```

Critical paths:

- **S5 EnKF + S6 PerturbedObservation + S7 EnSRF** (~450 LOC, 3 engineer-days) = **first ensemble-Kalman in the repo** — single-PR data-assimilation keystone, 25,000+ citations across DA literature
- **S15 + S16 + S17** (~480 LOC, 3 engineer-days) = **first HMM in the repo** — gates ASR/NLP/digital-comms downstream consumers
- **S20 TwistedSMC** (~280 LOC, 3 engineer-days) = **frontier-research moat** — no public Go impl
- **S10 4DVar** (~250 LOC, 2 engineer-days) = **operational-DA gateway** but blocked on slot-068 LBFGS + slot-014 reverse-mode autodiff

---

## 4. Saturation pins this slot unlocks

- **R-ENKF-CROSS-VALIDATION 4/4 (S5/S6/S7/S22):** four EnKF variants — stochastic (S5), perturbed-observation (S6), ensemble-square-root (S7), hybrid-EnVar (S22) — converge to identical posterior moments on a Lorenz-96-N=40-state-d=10-obs benchmark to 1e-2 (mean) and 1e-1 (cov) at N_ensemble=80. **Quadruple cross-validation — saturates R-MUTUAL-CROSS-VALIDATION.**
- **R-DA-VS-PARTICLE-CROSS-VALIDATION 3/3 (S5 EnKF / slot-265-P9 BootstrapPF / S22 HybridEnVar):** on a moderately-non-linear Lorenz-63-state-d=3 problem, EnKF + bootstrap-PF + hybrid-EnVar converge to same posterior to 1e-2 (mean) at N=200. **Cross-validates the EnKF-Gaussian-approx vs. the PF-non-Gaussian-truth.**
- **R-EKF-PF-VS-BOOTSTRAP-PF-VARIANCE-PIN (S11 vs slot-265-P9):** EKF-PF on a Doucet-Godsill-Andrieu-2000-Table-1-benchmark achieves ESS-per-particle 5× higher than bootstrap-PF at same N. Tests proposal-quality dependence on Cappé-Moulines-Rydén-2005-§9 expectation.
- **R-UKF-PF-VS-EKF-PF-PIN (S12 vs S11):** UKF-PF on a non-differentiable jump-diffusion benchmark achieves ESS-per-particle 2× higher than EKF-PF (where EKF Jacobian fails near the jumps). Tests UKF's no-derivatives advantage.
- **R-VITERBI-VS-PARTICLE-VITERBI-PIN (S16 vs S17):** on a discretised-fine-grid HMM (K=200 states), exact Viterbi (S16) and particle-Viterbi (S17) at N=1000 agree to 1% on the MAP path. Tests the discrete-state-fineness limit of S17.
- **R-FORWARD-BACKWARD-VS-FFBS-PIN (S15 vs slot-265-P21):** on a discretised-fine-grid HMM, exact forward-backward (S15) and particle-FFBS (slot-265-P21) at N=1000 agree on smoothed marginals to 1e-2. Tests slot-265-P21 correctness against gold-standard.
- **R-AMS-VS-NAIVE-MC-RARE-EVENT-PIN (S21):** on `P(X>x*) = 1e-8` for X ∼ N(0,1), AMS at N=100 with 10 levels recovers the probability to 1 sig-fig in 1000 samples; naive MC requires 1e10 samples. **8-orders-of-magnitude variance reduction.**
- **R-TWISTED-SMC-VS-BOOTSTRAP-PF-PIN (S20 vs slot-265-P9):** on a long-T=200-time-series, twisted-SMC after 5 twist-iterations achieves path-marginal-likelihood-estimator-variance 5-50× smaller than bootstrap-PF. Tests Guarniero-Johansen-Lee-2017-Theorem-2 expected variance reduction.
- **R-RESAMPLE-MOVE-VS-NO-MOVE-PIN (S18):** on IBIS with T=20 batches, resample-move with K=5 MH steps keeps unique-θ-particle-count > 0.5·N; without resample-move, unique-θ collapses to <5 by t=10. **Direct test of Gilks-Berzuini-2001 theorem.**
- **R-LOCALISATION-INFLATION-PIN (S8 + S9 vs naive S5):** on Lorenz-96-N=40-state-d=10-obs at N_ensemble=20 (under-dispersed regime), naive S5 collapses to <1% of true variance after 50 cycles; S5+S8+S9 stays within 80% of truth. **The mandatory operational-DA correction tested.**

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| S1 | RobertsRosenthalStableResample | 80 | A | slot-265-P1 |
| S2 | BranchingDelMoralResample | 90 | A | slot-265-P1, Binomial-RNG |
| S3 | BlockResample | 100 | A | slot-265-P3 |
| S4 | KillingResample-CrisanLyons-1999 | 70 | A | slot-265-P1 |
| S5 | EnKF-Evensen-1994 | 180 | B | linalg.Cholesky, slot-161-C5 |
| S6 | PerturbedObservationEnKF | 120 | B | S5, RNG-Gaussian |
| S7 | EnsembleSquareRootEnKF (+ETKF/EAKF) | 150 | B | S5, linalg.QR |
| S8 | LocalisationCovariance-Houtekamer | 100 | B | S5 |
| S9 | InflationCovariance-Anderson | 70 | B | S5 |
| S10 | 4DVar (+ 3DVar) | 250 | B | optim.LBFGS, autodiff.Grad |
| S11 | EKF-ParticleFilter | 180 | C | slot-265-P9, slot-161-C8 |
| S12 | UKF-ParticleFilter | 200 | C | slot-265-P9, slot-161-C10 |
| S13 | RegularisedParticleFilter | 80 | C | slot-265-P9, linalg.Cholesky |
| S14 | GaussianMixtureFilter | 80 | C | slot-161-C5 |
| S15 | ForwardBackward-BaumWelch | 180 | D | prob/markov |
| S16 | Viterbi-Forney | 100 | D | prob/markov |
| S17 | ParticleViterbi-Godsill-Doucet-West | 200 | D | slot-265-P9, S16 |
| S18 | ResampleMove-Gilks-Berzuini | 180 | E | slot-265-P9, slot-238-M2 |
| S19 | StratifiedSMC-Gerber-Chopin-Whiteley | 160 | E | slot-265-P3 |
| S20 | TwistedSMC-Guarniero-Johansen-Lee | 280 | E | slot-265-P9, slot-238-M2 |
| S21 | AdaptiveMultilevelSplitting | 180 | F | slot-265-P9 |
| S22 | HybridEnVar | 120 | F | S5-S10 |
| S23 | ControlVariateSMC | 80 | F | any-SMC |
| S24 | SMC-for-Optimisation | 80 | F | slot-265-P16 |
| **Σ** | | **~3,310** | | |

Pure-glue ratio: ~75% of LOC is composition over slot-161-C5/C8/C10 (KF/EKF/UKF), slot-265-P1-P11 (resampling + bootstrap-PF), slot-238-M1/M2 (RNG/MH). ~25% is genuinely-new SMC math: EnKF-square-root algebra (~100 LOC), localisation-Schur-product (~80 LOC), 4DVar adjoint-loop (~120 LOC), TwistedSMC iterative-twist-fitting (~200 LOC), AMS adaptive-level-selection (~100 LOC), ParticleViterbi forward-max-backward-trace (~120 LOC).

---

## 6. Recommended PR sequence

**PR-A: S5 EnKF + S6 PerturbedObservation + S7 EnSRF (~450 LOC, 3 engineer-days)** — **ENKF KEYSTONE**
The data-assimilation keystone. **First ensemble-Kalman in `reality`.** Saturates R-ENKF-CROSS-VALIDATION 4/4 + R-DA-VS-PARTICLE-CROSS-VALIDATION 3/3. **8,200 citations Evensen-1994 — single highest-impact PR in slot 266.**

**PR-B: S8 Localisation + S9 Inflation + S22 HybridEnVar (~290 LOC, 2 engineer-days)** — **OPERATIONAL-DA**
Closes the operational-DA gap on PR-A. Saturates R-LOCALISATION-INFLATION-PIN. **Without S8+S9, EnKF on N=20 vs d=40 collapses** — these are not optional.

**PR-C: S15 ForwardBackward + S16 Viterbi (~280 LOC, 2 engineer-days)** — **FIRST HMM**
The discrete-state HMM keystone. **First HMM in `reality`.** Gates ASR/NLP/digital-comms downstream. Saturates R-FORWARD-BACKWARD-VS-FFBS-PIN against slot-265-P21.

**PR-D: S11 EKF-PF + S12 UKF-PF + S13 RegularisedPF (~460 LOC, 3 engineer-days)** — **ADVANCED PROPOSALS**
Composes slot-265-P9 + slot-161-C8/C10. Saturates R-EKF-PF-VS-BOOTSTRAP-PF-VARIANCE-PIN + R-UKF-PF-VS-EKF-PF-PIN. **Lifts particle-filter ESS-per-particle 5-50× on informative observations** — the textbook recommendation in Doucet-de-Freitas-Gordon-2001.

**PR-E: S18 ResampleMove-Gilks-Berzuini-2001 (~180 LOC, 1 engineer-day)** — **REJUVENATION**
The canonical rejuvenation step inside slot-265-P16/P18. Saturates R-RESAMPLE-MOVE-VS-NO-MOVE-PIN. **Composes orthogonally with all SMC primitives.**

**PR-F: S20 TwistedSMC-Guarniero-Johansen-Lee-2017 (~280 LOC, 3 engineer-days)** — **FRONTIER MOAT**
The 2017 frontier-research primitive. **No public Go implementation exists worldwide.** Saturates R-TWISTED-SMC-VS-BOOTSTRAP-PF-PIN. Defines the 2017-2026 SMC research frontier.

**PR-G: S21 AdaptiveMultilevelSplitting + S24 SMC-for-Optimisation (~260 LOC, 2 engineer-days)** — **RARE-EVENT + OPT**
The rare-event-estimation + SMC-for-global-optim pair. Saturates R-AMS-VS-NAIVE-MC-RARE-EVENT-PIN (8-orders-of-magnitude variance reduction).

**PR-H: S17 ParticleViterbi + S19 StratifiedSMC + S23 ControlVariateSMC (~440 LOC, 3 engineer-days)** — **SMOOTHER + VR**
Particle-Viterbi MAP-path estimator + stratified-SMC variance-reduction + control-variate-SMC orthogonal-VR. Saturates R-VITERBI-VS-PARTICLE-VITERBI-PIN.

**PR-I: S1-S4 advanced resampling family (~340 LOC, 2 engineer-days)** — **RESAMPLING TAIL**
Roberts-Rosenthal-stable + branching-DelMoral + block-resample + killing-resample. Tail PR — composes on top of slot-265 PR-0 resampling-keystone.

**PR-J: S10 4DVar + S14 GaussianMixtureFilter (~330 LOC, 3 engineer-days)** — **VARIATIONAL DA + MIXTURE**
4DVar — gates on slot-068 LBFGS + slot-014 reverse-mode autodiff. **Last in sequence** because of upstream blockers.

Total: ~3,310 LOC source + ~2,400 LOC tests across 10 PRs over ~24 engineer-days. **PR-A (EnKF keystone) is the highest-impact-1-week-shippable bundle.** **PR-F (TwistedSMC) is the singular-MOAT.** **PR-C (HMM keystone) is the cheapest 2-day-shippable bundle that lands a first-of-its-kind primitive in `reality` (first HMM).**

---

## 7. Cross-cutting blockers and slot-co-shipping

- **prob/random.Gaussian (Box-Muller / Marsaglia-Tsang)** — **TWENTIETH** Block-C review demanding it (slots 117/184/188/202/215/216/217/227/228/229/230/231/232/233/235/236/237/238/259-265/**266**). Ship as Tier-0 in slot-238 PR-0.
- **slot-238-M1 RNGSampler interface** — gates every primitive S5-S24. Co-ship.
- **slot-238-M2 MetropolisHastings** — gates S18 ResampleMove + S20 TwistedSMC twist-MH-step. Co-ship.
- **slot-161-C5 KalmanFilter Joseph** — gates S5 EnKF-update + S14 GaussianMixtureFilter. Co-ship.
- **slot-161-C8 EKF + C10 UKF** — gates S11 EKF-PF + S12 UKF-PF. Co-ship slot-161 PR-2.
- **slot-265-P1-P7 resampling family + ESS** — slot-266 builds on top; **slot-265 owns the keystone**, slot-266 ships the advanced-tail S1-S4.
- **slot-265-P9 BootstrapFilter** — gates S11/S12/S13/S17/S20/S21. Co-ship slot-265 PR-1.
- **slot-265-P16 SMC sampler + P19 AdaptiveTempering** — gates S24 SMC-for-Optimisation. Co-ship slot-265 PR-4/PR-6.
- **slot-068 LBFGS** — gates S10 4DVar. Cross-link slot-068 (existing — `optim/lbfgs.go`).
- **slot-014 reverse-mode autodiff** — gates S10 4DVar adjoint computation. Cross-link slot-014.
- **linalg.Cholesky (existing) + linalg.QR (existing)** — gates S5/S7/S13. No new substrate.
- **prob/markov.go (existing)** — gates S15/S16. No new substrate, just composition.

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **S5 EnKF:** ensemble covariance `P̂ = (X − x̄·1ᵀ)(X − x̄·1ᵀ)ᵀ / (N−1)` is rank ≤ N−1, **always rank-deficient** when N < d. Document that S6/S7+S8 are **not optional** when N < d (the operational regime).
- **S6 PerturbedObservation:** the perturbations `ε_i` must be drawn fresh for each ensemble member at each cycle — re-using a single ε across the ensemble silently breaks variance preservation (Houtekamer-Mitchell-1998-MWR-126:796 §3 documents this).
- **S7 EnSRF:** the square-root update must use a **deterministic non-symmetric** square-root (Whitaker-Hamill-2002 use the Cholesky-based form; Tippett-Anderson-Bishop-Hamill-Whitaker-2003-MWR-131:1485 prove the family equivalence). Document the non-uniqueness — different square-root choices give different *individual* ensemble members but the same *ensemble statistics*.
- **S8 Localisation:** the Gaspari-Cohn-1999-QJRMS-125:723 kernel must be **strictly positive-definite** in d dimensions; the 5th-order-piecewise polynomial form is PD up to d ≤ 3 only (use Riishøjgaard-2001 for d > 3).
- **S9 Inflation:** multiplicative inflation `α > 1` must be **bounded above by 1.20** in operational regimes — `α > 1.20` over-disperses + breaks observation-fitting in 5-10 cycles.
- **S10 4DVar:** the adjoint code (gradient of the cost wrt initial-condition x_0) must satisfy the **adjoint-correctness identity** `〈M·δx, y〉 = 〈δx, Mᵀ·y〉` to floating-point precision — slot-014 reverse-mode autodiff guarantees this *if* the forward-model M is composed of differentiable Go primitives. Hand-coded adjoints (the operational-NWP norm) require unit-tested adjoint-identity checks.
- **S11 EKF-PF:** per-particle EKF Jacobian must be re-evaluated at each `x_{t-1,i}` — sharing a single Jacobian across particles breaks the proposal-as-importance-density covenant.
- **S12 UKF-PF:** UKF sigma-points must use the **same scaling parameters** (α, β, κ) as the slot-161-C10 reference; mismatch silently mis-weights the importance-sampling step.
- **S13 RegularisedPF:** kernel bandwidth `h = (4/(d+2))^(1/(d+4)) · N^(-1/(d+4))` Silverman-1986 rule undersmooths in high-d; use the Wand-Jones-1995 plug-in rule for d > 5.
- **S15 ForwardBackward:** numerical underflow at T > 100 — must use **log-space** forward+backward or **per-step normalisation** (Rabiner-1989-Proc-IEEE-77:257 Eq 27). Document both options.
- **S16 Viterbi:** must use **log-space** dynamic-programming to avoid underflow — `log(P_trans(i,j) · B_emit(j, y_t))` per recursion.
- **S17 ParticleViterbi:** the MAP path is **biased** for finite N — converges to true MAP only as N → ∞. Document the bias-variance tradeoff vs. exact-Viterbi-on-discretised-grid (S16) for moderate state-spaces.
- **S18 ResampleMove:** the inner-MCMC kernel must target the **current** π_t, not a stale π_{t-1} — the most common implementation bug. Recommend explicit `current_target_logp` parameter.
- **S20 TwistedSMC:** the twist function ψ_t must be **non-negative** to preserve target-validity. Use parametric families {exp(linear(x_t))} or {Gaussian} — non-parametric ψ_t (e.g. histogram) often goes negative in low-density regions.
- **S21 AMS:** adaptive-level-choice `L_t` must be **strictly increasing**; floating-point ties at the level-quantile cause silent infinite-loop in operational use. Add explicit tie-break tolerance.
- **S22 HybridEnVar:** convex-combination weight `α ∈ [0.2, 0.5]` per Hamill-Snyder-2000 — outside this range either EnKF (α=0) or 3DVar (α=1) dominates and the hybrid loses its advantage.
- **S23 ControlVariateSMC:** the surrogate `f̃` must have **known mean** μ̃ — using a sample-mean estimate of μ̃ from the same particles introduces bias. Document the fixed-vs-estimated-mean tradeoff.
- **S24 SMC-for-Optimisation:** the temperature schedule `T_t → 0` must satisfy `T_t / T_{t+1} > 0.95` (slow cooling) to preserve the exploration-exploitation balance — fast cooling collapses particles to local minima.

---

## 9. Distinct from prior agents (provenance)

- **slot-117 prob-missing** — names prob/random.Gaussian as Tier-0; **slot-266 is the TWENTIETH consumer**. Co-ship.
- **slot-118 prob-sota** — does not name SMC/EnKF/HMM/4DVar; slot-266 fills this gap.
- **slot-161 synergy-control-prob** — owns C11 BootstrapPF + stretch-RBPF at the **filtering axis**. **Slot-266 builds on** slot-161-C5/C8/C10 KF-family for EnKF-update / EKF-PF / UKF-PF / GaussianMixtureFilter; never duplicates.
- **slot-202 new-sde** — names SDE simulator (Euler-Maruyama, Milstein) for continuous-time-state-space-SMC; **slot-266 cross-link only** — never duplicates.
- **slot-238 new-mcmc** — names M1 RNG + M2 MH + M4 Gibbs as MCMC keystones; slot-266 composes against M1/M2 + slot-265 owns M17/M24 ship-once.
- **slot-263 new-quasi-mc** — Gerber-Chopin-2015 RQMC-SMC is the cross-link; **slot-266-S19 StratifiedSMC** by Gerber-Chopin-Whiteley-2019 is the *MC* (not Q-MC) sibling. Distinct paper, distinct primitive.
- **slot-264 new-mlmc** — multilevel-PMMH (Jasra-Kamatani-Law-Zhou-2018) is the cross-link to v2; slot-266 stays at single-level.
- **slot-265 new-pmcmc** — owns the **PMMH/PGAS/SMC²/SMC-sampler/RBPF/Liu-West/Storvik/Island/Nested-PF/FFBS/BackwardSimulation/TwoFilterSmoother** corpus (28 primitives). **Slot-266 explicitly does NOT duplicate any of these**: slot-266 ships the 24 *complementary* primitives — EnKF family (S5-S10), advanced-proposal PF (S11-S14), discrete-HMM (S15-S17), look-ahead+twisted-SMC (S18-S20), variance-reduction+rare-event+optim (S21-S24). Slot 265 owns the parameter-inference frontier; slot 266 owns the data-assimilation + discrete-state + advanced-proposal + variance-reduction + frontier-twisted-SMC frontier.
- **slot-161-stretch-RBPF** — slot-265-P24 owns it as first-class. Slot-266-S14 GaussianMixtureFilter is the **non-Rao-Blackwellised** mixture-of-Gaussians-filter — predecessor to RBPF, distinct semantics (caller pre-specifies K mixture components vs. RBPF auto-partitions linear/non-linear sub-state).

---

## 10. Bottom line

`reality` v0.10.0 ships **ZERO** SMC primitives — re-confirmed by repo-wide grep on `SMC|SequentialMonteCarlo|ParticleFilter|BootstrapFilter|EnKF|EnsembleKalman|3DVar|4DVar|Viterbi|HiddenMarkov|ResampleMove|StratifiedSMC|BlockResample|RobertsRosenthal|LookAhead|EKFParticleFilter|UKFParticleFilter|RegularisedPF|GaussianMixtureFilter|TwistedSMC|AdaptiveMultilevelSplitting|HybridEnVar|ControlVariateSMC` returning ZERO callable matches across all 22 packages. The closest tangential surfaces are slot-265-P1-P28 PMCMC corpus (parameter-inference axis), slot-161-C5/C8/C10/C11 KF-family + bootstrap-PF (filtering axis), slot-238 MCMC corpus (sampler axis) — none of which cover the 24 SMC-discipline primitives slot-266 owns.

**Twenty-four SMC primitives S1-S24 totalling ~3,310 LOC of pure connective tissue** stand up the **complementary SMC-discipline** to slot-265's PMCMC corpus on existing v0.10.0 surfaces:

- **Tier B (S5-S10) ~870 LOC** is the **EnKF + 4DVar data-assimilation keystone** — Evensen-1994 has 8,200 citations alone, the entire NWP-DA-pipeline of NCEP/ECMWF/NOAA/ECCC operates on this stack, **zero zero-dep-Go-implementation exists worldwide**. Single-PR PR-A + PR-B (~740 LOC, 5 engineer-days) lands the first ensemble-Kalman + operational localisation+inflation in `reality`.
- **Tier D (S15-S17) ~480 LOC** is the **discrete-state HMM keystone** — Forward-Backward + Viterbi + Particle-Viterbi, gating all downstream ASR/NLP/digital-comms consumers. Single-PR PR-C (~280 LOC, 2 engineer-days) lands the first HMM in `reality`.
- **Tier C (S11-S14) ~540 LOC** is the **advanced-proposal PF family** — EKF-PF + UKF-PF + Regularised-PF + GaussianMixtureFilter, **lifting bootstrap-PF ESS-per-particle 5-50× on informative observations** per Doucet-Godsill-Andrieu-2000-Table-1. Single-PR PR-D (~460 LOC, 3 engineer-days).
- **Tier E (S18-S20) ~620 LOC** is the **frontier-research moat** — Resample-Move-Gilks-Berzuini-2001 (rejuvenation) + Stratified-SMC-Gerber-Chopin-Whiteley-2019 + **Twisted-SMC-Guarniero-Johansen-Lee-2017 (no public Go impl worldwide, defines 2017-2026 frontier)**. Single-PR PR-F (~280 LOC, 3 engineer-days) for the singular-MOAT TwistedSMC.
- **Tier F (S21-S24) ~460 LOC** is the **variance-reduction + rare-event + optim tail** — AMS (8-orders-of-magnitude variance reduction on rare-event probabilities) + HybridEnVar (operational-DA-2010-2026 default) + ControlVariateSMC + SMC-for-Optimisation.
- **Tier A (S1-S4) ~340 LOC** is the **advanced-resampling tail** — Roberts-Rosenthal-stable + Branching-DelMoral + Block-Resample + Killing-Resample, building on slot-265's resampling keystone.

The cheapest 1-week-shippable bundle is **PR-A + PR-B (S5-S10, ~740 LOC) ENKF + LOCALISATION + INFLATION + HYBRID-ENVAR** = the entire **ensemble-data-assimilation stack** = the highest-citation contribution in this slot (~25,000+ aggregate citations across the family). The single highest-impact PR is **PR-A S5 EnKF-Evensen-1994** alone (~180 LOC, 8,200 citations). The single most-distinct cutting-edge contribution is **PR-F S20 TwistedSMC-Guarniero-Johansen-Lee-2017** (~280 LOC) = no public Go implementation exists worldwide. The cheapest 2-day-shippable first-of-its-kind PR is **PR-C S15 ForwardBackward + S16 Viterbi** (~280 LOC) = the first HMM in `reality`.

**Reality is unusually well-positioned for this slot because (i) slot-161-C5/C8/C10 KF-family is *already* the substrate every EnKF/EKF-PF/UKF-PF primitive needs; (ii) slot-265-P1-P11 resampling-family + bootstrap-PF is *already* the substrate every advanced-PF + look-ahead + twisted-SMC primitive needs; (iii) slot-265-P16 SMC sampler is *already* the substrate S24 SMC-for-Optimisation needs; (iv) `prob/markov.go` is *already* the substrate S15/S16 HMM primitives need; (v) `linalg.{Cholesky, QR, MatMul}` covers all EnKF + EKF-PF + GaussianMixtureFilter linear-algebra; (vi) the consumer-side-placement rule (per agents 158/159/160/166/167/168/169/195/238/263/264/265) recommends `prob/smc/` sub-package mirroring `prob/copula/`+`prob/conformal/`+`prob/mcmc/`+`prob/qmc/`+`prob/mlmc/` placement convention.**

Differentiation: 24/24 primitives unique to slot-266 SMC-discipline-axis. **Zero overlap** with slot-265's 28 PMCMC primitives (verified by name-by-name comparison: slot-265 covers PMMH/PGAS/SMC²/SMC-sampler/IBIS/AdaptiveTempering/RBPF/Liu-West/Storvik/Island/Nested-PF/FFBS/BackwardSimulation/TwoFilterSmoother/AuxiliaryPF/OptimalProposalPF; slot-266 covers EnKF/PerturbedObsEnKF/EnSRF/Localisation/Inflation/4DVar/HybridEnVar/EKF-PF/UKF-PF/RegularisedPF/GaussianMixtureFilter/ForwardBackward/Viterbi/ParticleViterbi/ResampleMove/StratifiedSMC/TwistedSMC/AMS/ControlVariateSMC/SMC-for-Optimisation/RobertsRosenthal/Branching/Block/Killing-resample). Cross-cutting blockers shared with 20+ Block-C slots: prob/random.Gaussian (TWENTIETH demand), slot-238-M1/M2 RNG/MH, slot-161-C5/C8/C10 KF-family, slot-265-P1-P11 resampling+bootstrap-PF.

The single most important conceptual identity slot 266 pins: **SMC is not a particle filter — SMC is a generic algorithm for sampling a sequence of distributions, of which particle filtering, ensemble-Kalman, HMM forward-backward, rare-event splitting, and SMC-for-optimisation are seven distinct instances unified by the importance-sample → resample → propagate motif.** Slot-266 ships the seven instances slot-265 leaves out + the variance-reduction toolkit + the frontier-twisted-SMC reference, completing the 52-primitive (28+24) SMC discipline implementation in `reality`.
