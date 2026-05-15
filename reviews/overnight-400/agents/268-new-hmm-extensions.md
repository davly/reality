# 268 | new-hmm-extensions — Hidden Markov extensions: HSMM, IO-HMM, factorial HMM, switching state-space, HDP-HMM, HCRF, CT-HMM

**Block:** C (cutting-edge math). **Date:** 2026-05-09. **Repo:** v0.10.0, 1,965 tests passing, 22 packages.

## Two-line summary

`reality` v0.10.0 ships **ZERO** HMM surface — repo-wide grep on `HMM|HiddenMarkov|Viterbi|ForwardBackward|BaumWelch|HSMM|HDP.HMM|FactorialHMM|IOHMM|HCRF|SLDS|SwitchingState|TripletMarkov|PairwiseMarkov|CT.?HMM|MMPP` returns ZERO callable matches across all 22 packages; the only Markov surface is `prob/markov.go::MarkovSteadyState` (power-iteration on stochastic matrix, 70 LOC) + `prob/markov.go::MarkovSimulate` (LCG-driven trajectory, 70 LOC) — pure state-only, no notion of *emission* conditional on latent state, which is exactly the missing piece between a Markov chain and an HMM. PARTIAL OVERLAP with **slot 165** (`synergy-sequence-prob`: proposed standard HMM forward/backward/Viterbi/Baum-Welch ~520 LOC + LogSumExp toolbelt 80 LOC + Pair HMM 330 LOC + linear-chain CRF 250 LOC, all named-not-shipped) and **slot 266 Tier-D** (`new-smc`: proposed S15 ForwardBackward 180 + S16 Viterbi 100 + S17 ParticleViterbi 200 LOC, gated on slot 265 PR-1 bootstrap-PF, named-not-shipped) and **slot 173-Q15** (MMPP-2 Heffes-Lucantoni for *queueing* mean-rate + cv²_a closed form 220 LOC — already counted, distinct consumer not HMM-inference). Slot 268 owns the **HMM-extensions axis** that 165 + 266 deliberately leave out: 165 ships only the standard discrete-state HMM canon; 266 ships only the discrete-state HMM as a tier inside SMC (S15/S16/S17). Neither addresses **HSMM (Ferguson-1980) explicit-duration**, **IO-HMM (Bengio-Frasconi-1995) input-conditioned emissions/transitions**, **Factorial HMM (Ghahramani-Jordan-1997) K parallel chains**, **Coupled HMM (Brand-Oliver-Pentland-1997) cross-influencing chains**, **Switching State-Space (Ghahramani-Hinton-2000) mixture-of-Kalman-filters**, **SLDS (Pavlovic-Rehg-MacCormick-2000) discrete-state-gated linear dynamics**, **Hierarchical HMM (Fine-Singer-Tishby-1998) recursive-state HHMM**, **Variational Bayesian HMM (MacKay-1997 / Beal-2003)**, **Infinite HMM / HDP-HMM (Beal-Ghahramani-Rasmussen-2002 / Teh-Jordan-Beal-Blei-2006)**, **Sticky HDP-HMM (Fox-Sudderth-Jordan-Willsky-2008)**, **CT-HMM (Liu-Yeh-Tao-2015 / Metzner-Horenko-Schuette-2007) continuous-time generator-matrix HMM**, **Triplet Markov chains (Pieczynski-2003)**, **Pairwise Markov chains (Pieczynski-2002)**, **HMM with missing observations (Yeh-Chiang-1997)**, **Mixed HMM (Altman-2007) random-effects HMM**, **Discriminative HMM / Maximum-Margin HMM (Sha-Saul-2007)**, **HCRF (Quattoni-Wang-Morency-Collins-Darrell-2007)**, **Latent Dirichlet HMM (Andrews-Vigliocco-2010)**, **Multi-channel/Multi-source HMM (Brand-Oliver-Pentland-1997 / Dupont-Luettin-2000)**, **Forward-with-covariates / EM under covariates**. Slot 268 enumerates **22 HMM-extension primitives H1-H22 totalling ~3,720 LOC** of pure connective tissue on top of slot 165 + slot 228 (HDP) + slot 266 (forward-backward+Viterbi). The **cheapest 1-day standalone** is **PR-1 H1 HSMM-Yu-2010 explicit-duration forward-backward ~280 LOC** which lands the **first explicit-duration HMM in the repo** (Yu-2010-Artif-Intell-174:215 has 950 citations — the canonical HSMM survey post-Ferguson). The **highest-leverage 1-week unlock** is **PR-2 H6 SLDS-Pavlovic-Rehg-MacCormick-2000 + H5 SwitchingStateSpace-Ghahramani-Hinton-2000 ~620 LOC** which composes slot-161-C5 KalmanFilter + slot-165-HMM-forward-backward + Rao-Blackwellisation across discrete-mode and continuous-state — the **canonical economics/finance/tracking model** (3000+ citations Hamilton-1989-Econometrica regime-switching → Kim-Nelson-1999-MIT-Press SLDS textbook). The **SINGULAR cutting-edge piece** is **H10 StickyHDP-HMM-Fox-Sudderth-Jordan-Willsky-2008-ICML ~360 LOC** which adds a self-transition stickiness `κ` to HDP-HMM solving the fast-state-switching pathology — **no public Go implementation exists worldwide**, reference is Fox's MATLAB code; the natural composition is slot-228-B19 HDP + slot-165-HMM via direct-assignment Gibbs (Teh-2006). The **SINGULAR architectural identity**: **HMM = Markov-chain × emission-distribution; HSMM = HMM × duration-distribution; IO-HMM = HMM × input-modulation; Factorial HMM = HMM^K (tensor product); HHMM = HMM nested into states; SLDS = HMM × Kalman-filter (per-mode); HDP-HMM = HMM × DP-prior-on-rows-of-A**. Each of these is a strict generalisation of the standard HMM and reuses the forward-backward DP scaffold in slot-165 + log-space toolbelt in slot-165-§2. Cross-cutting blockers reuse **slot-165-§1 standard HMM** (gates H1-H8, H11-H22), **slot-165-§2 LogSumExp + Log1mExp + Log1pExp** (gates everything in log-space DP), **slot-228 HDP-CRF sampler B19** (gates H9 iHMM + H10 StickyHDP-HMM), **slot-161-C5 KalmanFilter** (gates H5 SSSM + H6 SLDS), **slot-238-M1 RNG + M2 MetropolisHastings + M4 GibbsSampler** (gates H8 VB-HMM + H9 iHMM + H10 StickyHDP-HMM), **slot-265-P9 BootstrapPF** (gates H6 SLDS-PF + H21 PF-HMM). Differentiation: 22/22 primitives unique to slot-268 HMM-extension axis, ZERO duplication with slot 165 (which owns the *standard* HMM keystone) and ZERO duplication with slot 266 Tier-D (which owns the *single discrete HMM* as one of seven SMC-target instances).

---

## 0. State of play (verified 2026-05-09 against v0.10.0)

### Repo-wide grep returns ZERO HMM-extension surface

```
$ grep -rE 'HMM|HiddenMarkov|HSMM|FactorialHMM|IOHMM|CoupledHMM|HHMM|SLDS|SwitchingStateSpace|VariationalBayesHMM|InfiniteHMM|HDP.HMM|StickyHDP|TripletMarkov|PairwiseMarkov|CT.?HMM|MaxMarginHMM|HCRF|LatentDirichletHMM|MMPP' --include=\*.go reality/
[ZERO callable matches in all 22 packages — only doc-comment/test references inside changepoint/audio/sequence]
```

### Adjacent surfaces that exist (substrate for slot 268)

| Surface | Path | Slot 268 use |
|---|---|---|
| `prob.MarkovSteadyState(P, n)` | `prob/markov.go:31-70` | π for HMM/HSMM/IOHMM init |
| `prob.MarkovSimulate(P, n, s0, T)` | `prob/markov.go:99-139` | trajectory sampler — LCG, deterministic |
| `prob.Distribution` interface (PDF/CDF) | `prob/distribution.go:24-35` | per-state emission distribution |
| `prob.NormalDist / BetaDist / ExponentialDist / UniformDist` | `prob/distribution.go` | concrete emissions for Gaussian-HMM |
| `prob.LogGamma` | `prob/mathutil.go:13-65` | substrate for negative-binomial duration in HSMM |
| `prob.PoissonPMF / PoissonCDF` | `prob/distributions.go` | Poisson duration in HSMM, MMPP rate |
| `prob.GammaPDF / GammaCDF` | `prob/distributions.go` | gamma-duration HSMM, CT-HMM holding-time |
| `prob.BayesianUpdate` | `prob/prob.go:85` | adjacent — IO-HMM not direct |
| `LogSumExp` (private) | `changepoint/bocpd.go:294-305` | **must promote to `prob/mathutil.go`** per slot 165 §2 |
| `prob/conformal/adaptive.go:160 EffectiveSampleSize` | Kish on decayed windows | wrong consumer (slot-265 / slot-266 documented) |
| Slot 161 C5 KalmanFilter | proposed-not-shipped | gates H5 SSSM + H6 SLDS |
| Slot 165 §1 HMM forward-backward-Viterbi-Baum-Welch | proposed-not-shipped | gates H1-H22 |
| Slot 165 §2 LogSumExp/Log1mExp/Log1pExp | proposed-not-shipped | gates log-space DP |
| Slot 228 B19 HDP-CRF sampler | proposed-not-shipped | gates H9 iHMM + H10 StickyHDP-HMM |
| Slot 238 M1 RNG / M2 MH / M4 Gibbs | proposed-not-shipped | gates Bayesian-inference H8/H9/H10/H17 |
| Slot 265 P9 BootstrapPF | proposed-not-shipped | gates H21 particle-HMM |
| Slot 266 S15 ForwardBackward / S16 Viterbi | proposed-not-shipped | duplicate of slot-165 §1 — ship-once |
| Slot 173 Q15 MMPP-2 Heffes-Lucantoni | proposed-not-shipped | distinct *queueing* consumer, not HMM-inference |

### Cross-import edges (cycle-free DAG)

```
prob/hmm/             (canonical home, slot-165 §1 ships base, slot-268 extends)
   │
   ├── consumes: prob/                      (Distribution, MarkovSteadyState, LogGamma)
   ├── consumes: prob/mathutil.go LogSumExp (slot-165 §2 promote-from-bocpd)
   ├── consumes: prob/random/               (slot-238-M1 Box-Muller, twentieth Block-C demand)
   ├── consumes: prob/mcmc/                 (slot-238-M2 MH + M4 Gibbs)
   ├── consumes: prob/bnp/                  (slot-228-B19 HDP for H9/H10)
   ├── consumes: control/kf/                (slot-161-C5 KalmanFilter for H5/H6)
   └── consumes: prob/smc/                  (slot-265-P9 BootstrapPF for H21)
```

No reverse dependency. `prob/hmm/` strictly composes downstream surfaces.

---

## 1. The architectural identity

The single most-important conceptual identity slot 268 pins:

> **Standard HMM** = `(MarkovChain, EmissionPerState)`. Every HMM extension is one of:
> - **HMM × Duration** → HSMM (H1) / Variable-Duration (H2).
> - **HMM × Input** → IO-HMM (H3) / forward-with-covariates (H17).
> - **HMM^K (tensor product over independent chains)** → Factorial HMM (H4).
> - **HMM × HMM (cross-influencing chains)** → Coupled HMM (H7).
> - **HMM nested into states** → Hierarchical HMM (H11).
> - **HMM × Kalman** → Switching State-Space (H5) / SLDS (H6).
> - **HMM × DP-prior-on-A-rows** → Infinite HMM / HDP-HMM (H9) / Sticky HDP-HMM (H10).
> - **HMM with continuous time** → CT-HMM (H12) / MMPP-N (H13).
> - **Joint Markov over (state, observation)** → Pairwise / Triplet Markov chain (H14/H15).
> - **HMM with missing data** → H16.
> - **HMM with random effects across subjects** → Mixed HMM (H18).
> - **HMM trained discriminatively** → Max-Margin HMM (H19) / HCRF (H20).
> - **HMM × LDA** → Latent Dirichlet HMM (H22).
> - **HMM × particles** → Particle HMM (H21).

That partition is the basis of slot 268's 22 primitives. Each generalisation reuses the **forward-backward DP scaffold** from slot-165 §1: only the local factor structure changes.

---

## 2. The twenty-two primitives H1-H22 (~3,720 LOC pure connective tissue)

Each entry: capability + reference / composition / LOC / cross-link / blocking-flag.

### Tier A — Duration-explicit HMM (H1-H2, ~360 LOC)

**H1 — HSMM (Hidden Semi-Markov Model) Ferguson-1980 + Yu-2010** [~280 LOC]
Ferguson-1980-CVPR / Yu-2010-Artif-Intell-174:215 (950 citations). State `s_t` has **explicit duration** `d_s ~ p_s(d)` (geometric / Poisson / negative-binomial / non-parametric); standard HMM is special case `p_s(d) = (1-a_ss) a_ss^{d-1}` (geometric). DP recursion (Yu-2010 §3): `α_t(j) = Σ_i Σ_d α_{t-d}(i) · a_ij · p_j(d) · Π_{τ=t-d+1}^t b_j(o_τ)`. Cost `O(T·N²·D_max)`. API surface: `type HSMM { N, A, Pi, Emit, Duration, DMax }; Forward / Viterbi / BaumWelch`. Composes slot-165 §1 + §2 LogSumExp. Pin R-HSMM-VS-HMM-WHEN-GEOMETRIC 3/3.

**H2 — Variable-Duration HMM Levinson-1986 + Russell-Moore-1985** [~80 LOC]
Levinson-1986-Comput-Speech-Lang-1:29 / Russell-Moore-1985-ICASSP-10. Strict variant of H1 with state-specific maximum duration but parametric form (gamma / log-normal). Closed-form duration-density evaluation 80 LOC delta on H1.

### Tier B — Input-conditioned (H3, ~280 LOC)

**H3 — IO-HMM (Input-Output HMM) Bengio-Frasconi-1995** [~280 LOC]
IEEE-TNN-7:1231 (1100 citations). Both transitions and emissions conditioned on input sequence `u_{1:T}`: `p(s_t | s_{t-1}, u_t)` softmax-gated + `p(o_t | s_t, u_t)`. Forward-backward identical to HMM with per-time `A_t = A(u_t)`, `B_t = B(u_t)`. API: callbacks `TransLogP(input, sPrev) []float64`, `EmitLogP(input, s, obs)`. Train via slot-014 reverse-AD + slot-068 LBFGS. Pin R-IO-HMM-RECOVERS-HMM-WHEN-CONSTANT-INPUT 3/3.

### Tier C — Parallel chains (H4, ~360 LOC)

**H4 — Factorial HMM Ghahramani-Jordan-1997** [~360 LOC]
Mach-Learn-29:245 (1850 citations). **K parallel independent state chains** `s_t^(1..K)` with joint emission. Exact `O(T·N^{2K})` intractable for K≥4. (a) Exact for K small (~100 LOC); (b) structured-mean-field VI (Ghahramani-Jordan §5) factorises `q(s^(1:K)) = Π_k q(s^(k))`, alternates forward-backward on each chain with effective emissions (~260 LOC). Pin R-FACTORIAL-EXACT-VS-MEANFIELD 3/3.

### Tier D — Switching state-space (H5-H6, ~620 LOC)

**H5 — Switching State-Space Model Ghahramani-Hinton-2000** [~280 LOC]
Neural-Comput-12:831 (1500 citations). Discrete switch `s_t ∈ {1..M}` over M linear-Gaussian SSMs; each mode runs its own Kalman; observation = mixture-of-experts `p(o_t | s_t=m, x_t^(m))`. Variational inference alternates HMM forward-backward over `s_t` with per-mode Kalman filtering. Composes slot-161-C5 + slot-165 §1.

**H6 — SLDS (Switching Linear Dynamical System) Pavlovic-Rehg-MacCormick-2000** [~340 LOC]
NIPS-13 / Kim-Nelson-1999-MIT-Press *State-Space Models with Regime Switching* (3000+ citations). Continuous state has memory across mode switches: `x_t = A_{s_t}x_{t-1} + B_{s_t}ε_t`, `y_t = C_{s_t}x_t + D_{s_t}η_t`. Inference is exponentially-growing Gaussian mixture; production approach is **GPB2** (Gaussian-Pseudo-Bayesian order 2, collapse `M²→M`) or **IMM** (Bar-Shalom-1988). RBPF alternative: slot-265-P24 with discrete-mode in particles + Kalman per particle. Pin R-SLDS-VS-KF-WHEN-SINGLE-MODE 3/3 + R-SLDS-GPB2-VS-RBPF 3/3.

### Tier E — Coupled / Hierarchical (H7, H11, ~430 LOC)

**H7 — Coupled HMM Brand-Oliver-Pentland-1997** [~180 LOC]
CVPR-94. K HMMs whose transitions depend on all chains' previous states: `p(s_t^(k) | s_{t-1}^(1:K))`. Exact `N^{K·2}` intractable; N-Heads algorithm (Brand-1997 §3.2) reduces to `O(T·K·N²)` via Loopy BP. Composes slot-256 BP + slot-165 §1.

**H11 — Hierarchical HMM Fine-Singer-Tishby-1998** [~250 LOC]
Mach-Learn-32:41 (1100 citations). Each parent-state emits an entire sub-HMM trajectory; recursive nesting. Inference: generalised inside-outside `O(T³·N·L)` or flattened HMM (Murphy-Paskin-2001) mapping HHMM to flat HMM with `Π N_i` joint states.

### Tier F — Bayesian / Nonparametric (H8-H10, H17, ~990 LOC)

**H8 — Variational Bayesian HMM MacKay-1997 / Beal-2003** [~220 LOC]
MacKay-1997 / Beal-2003-PhD-Gatsby-Ch3. Dirichlet prior on each row of A + base prior on emissions. VB-E uses expected log-transition `E[log A_ij] = ψ(γ_ij) − ψ(Σ_j γ_ij)` (digamma). Provides ELBO-based automatic model selection (low-usage states prune). Needs `prob.Digamma` shared with slot 228 VB-DP (~30 LOC).

**H9 — Infinite HMM / HDP-HMM Beal-Ghahramani-Rasmussen-2002 + Teh-Jordan-Beal-Blei-2006** [~280 LOC]
HDP-HMM 4500 citations (JASA-101:1566). Each row `A_i ~ DP(α, β)` with `β ~ DP(γ, H)` global stick-breaking over infinite states. Direct-assignment Gibbs (Teh-2006 §5) with new-state stick-breaking marginalisation. Composes slot-228-B19 HDP-CRF + slot-165 §1 + slot-238-M4 Gibbs. Pin R-IHMM-VS-HMM-AS-ALPHA→0 3/3.

**H10 — Sticky HDP-HMM Fox-Sudderth-Jordan-Willsky-2008** [~360 LOC] **MOAT**
ICML-25:312 / Ann-Stat-39:1020. HDP-HMM with `a_ii ~ Beta(α+κ, ...)` self-transition stickiness — solves vanilla HDP-HMM fast-state-switching pathology (Beal-2002 oversamples short-duration states). Default `κ ∈ [0.5α, 5α]`. κ learned via auxiliary-Beta-binomial posterior (Fox-2011 §6). **No public Go implementation worldwide**; reference is Fox's MATLAB `HDPHMM_HDPSLDS`. Composes H9 + auxiliary-variable Gibbs. Pin R-STICKY-VS-IHMM-AS-KAPPA→0 3/3.

**H17 — Forward-with-covariates / EM-under-covariates Altman-2007** [~130 LOC]
JASA-102:201. Parametric-only special case of H3: `a_{ij}(u_t) = softmax(β_{ij}^T u_t)`, `b_j(u_t, o_t) = N(u_t β_j, σ²_j)` with closed-form M-step.

### Tier G — Continuous-time (H12-H13, ~380 LOC)

**H12 — CT-HMM Liu-Yeh-Tao-2015 + Metzner-Horenko-Schuette-2007** [~260 LOC]
NeurIPS-28 / Phys-Rev-E-76. State `s(t)` evolves CTMC with generator Q; transition `P(Δt) = exp(Q·Δt)`. Forward `α_k(j) = Σ_i α_{k-1}(i)·[exp(Q·(t_k−t_{k−1}))]_{ij}·b_j(o_k)`. Disease-progression / EHR / RNA-seq trajectory. EM Q-update via Liu-Yeh-Tao §3 closed-form (Pyle-2008). Composes `linalg.MatExp` + slot-165 §1+§2. Pin R-CTHMM-VS-HMM-WHEN-UNIFORM-DT 3/3.

**H13 — MMPP-N HMM-inference (Heffes-Lucantoni-1986)** [~120 LOC]
slot-173-Q15 ships queueing-stationary closed-form (cv²_a). slot-268-H13 ships **per-event posterior decoding + EM training**: given Poisson arrival times, infer hidden modulating CTMC via CT-HMM forward-backward with `Poisson(λ_{s(t)}·Δt)` emission. Pure synergy on H12. Pin R-MMPP-VS-Q15 3/3.

### Tier H — Pairwise / Triplet Markov chains (H14-H15, ~280 LOC)

**H14 — Pairwise Markov chains Pieczynski-2002** [~140 LOC]
Int-J-Approx-Reason-29:175. Joint `(s_t, o_t)` Markov (HMM assumes only `s_t` Markov, `o_t` iid given `s_t`). Allows `o_t` autocorrelation. Forward-backward on extended state `z_t = (s_t, o_{t-1})`.

**H15 — Triplet Markov chains Pieczynski-2003** [~140 LOC]
LNCS-2683. Auxiliary unobserved process `u_t` such that `(s_t, u_t, o_t)` joint Markov. Subsumes HSMM (with `u_t` = duration-since-state-entry).

### Tier I — Missing data / mixed effects (H16, H18, ~280 LOC)

**H16 — HMM with missing observations Yeh-Chiang-1997** [~130 LOC]
CSDA-25:233. Missing `o_t` ⟹ `b_j(o_t) = 1` in forward (marginalises). MAR-aware Baum-Welch with explicit `missing` mask; MNAR biases EM. Pin R-MISSING-OBS-VS-FULL 3/3.

**H18 — Mixed HMM Altman-2007** [~150 LOC]
Stat-Med-26:1135 / Maruotti-2011 (650 citations). Subject-level random effects: `A^{(i)}_{kk'} = a_{kk'}·exp(b_{kk'}^{(i)})`, `b^{(i)} ~ N(0,Σ)`. Numerical-quadrature E-step (Gauss-Hermite slot-018) + outer Baum-Welch.

### Tier J — Discriminative training (H19-H20, ~480 LOC)

**H19 — Max-Margin HMM / M³N Sha-Saul-2007 + Taskar-Guestrin-Koller-2003** [~220 LOC]
NIPS-19 / NIPS-16. Replace generative Baum-Welch with structured-perceptron / structured-SVM: `max_w min_{y'} margin(y_i, y', x_i, w)`. Viterbi reused as max-violator. Composes slot-068 LBFGS + slot-165 §1 Viterbi.

**H20 — HCRF Quattoni-Wang-Morency-Collins-Darrell-2007** [~260 LOC]
IEEE-PAMI-29:1848 (1700 citations). CRF with hidden states between input `x` and label `y`: `p(y|x) = Σ_h p(y,h|x) ∝ Σ_h exp(Σ_t f(y, h_t, h_{t-1}, x_t)·w)`. Two-pass forward-backward (outer y, inner h). Gesture/sign-language/handwriting. Composes slot-165 §7 LinearChainCRF + hidden-state marginalisation. Pin R-HCRF-VS-CRF-WHEN-FIXED-HIDDEN 3/3.

### Tier K — Latent-Dirichlet / Multi-channel / Particle-HMM (H21-H22, ~580 LOC)

**H21 — Particle HMM Andrieu-Doucet-Tadic-2005** [~180 LOC]
Ann-Stat-33:1834. HMM with continuous/factorial/infinite state space; exact forward-backward replaced by bootstrap-PF (slot-265-P9) + FFBS smoothing (slot-265-P21). Distinct from slot-266-S17 ParticleViterbi (max-path) — H21 is the smoothing consumer.

**H22 — LD-HMM (Andrews-Vigliocco-2010) + Multi-channel HMM (Brand-Oliver-Pentland-1997 / Dupont-Luettin-2000)** [~400 LOC]
Top-Cogn-Sci-2:101 / IEEE-Trans-Multimedia-2:141. LD-HMM marries LDA topic model with HMM dynamics. Multi-channel HMM observes synchronous streams (audio+lip-movement, audio+text) sharing latent state; factored emission `p(o^{(c)}_t | s_t)` per channel. ~200 LOC LD-HMM (defer to dedicated topic-model slot) + ~200 LOC multi-channel.

---

## 3. Composition graph (DAG)

```
slot-238-M1 RNG ─── gates Bayesian-inference-HMMs
slot-165 §2 LogSumExp + Log1mExp ─── gates ALL log-space DP
slot-165 §1 standard HMM ─── gates H1-H4, H7-H8, H11-H22
   │
   ├── H1 HSMM ─── adds duration distribution
   ├── H2 Variable-Duration HMM ─── parametric H1
   ├── H3 IO-HMM ─── adds input modulation
   ├── H4 Factorial HMM ─── tensor-product chains + structured-MF VI
   ├── H7 Coupled HMM ─── slot-256 belief propagation
   ├── H11 Hierarchical HMM ─── nested HMMs
   ├── H17 Forward-with-covariates ─── parametric H3
   ├── H14/H15 Pairwise/Triplet MC ─── extended state
   ├── H16 missing-obs HMM ─── direct extension
   ├── H18 Mixed HMM ─── slot-018 Gauss-Hermite
   ├── H19 Max-Margin HMM ─── slot-068 LBFGS, structured-SVM
   ├── H20 HCRF ─── slot-165 §7 LinearChainCRF + hidden-states
   ├── H22 LD-HMM / Multi-channel ─── factored emission
   │
   ├── slot-161-C5 KalmanFilter
   │      ├── H5 Switching-State-Space ─── per-mode Kalman
   │      └── H6 SLDS ─── GPB2 / IMM / RBPF
   │
   ├── slot-228-B19 HDP-CRF
   │      ├── H9 iHMM / HDP-HMM ─── direct-assignment Gibbs
   │      └── H10 Sticky HDP-HMM ─── + κ self-transition (MOAT)
   │
   ├── linalg.MatExp
   │      ├── H12 CT-HMM ─── exp(Q·Δt)
   │      └── H13 MMPP-N ─── Poisson emission on H12
   │
   └── slot-265-P9 BootstrapPF + P21 FFBS
          └── H21 Particle HMM ─── intractable-state-space HMM
```

---

## 4. Saturation pins this slot unlocks

- **R-HSMM-VS-HMM-WHEN-GEOMETRIC 3/3 (H1):** HSMM with `p_s(d) = (1-q_s) q_s^{d-1}` collapses to plain HMM with `a_ss = q_s` — forward-prob agreement to 1e-12.
- **R-IO-HMM-RECOVERS-HMM-WHEN-CONSTANT-INPUT 3/3 (H3):** IO-HMM with input-independent transition/emission matches HMM forward to 1e-12.
- **R-FACTORIAL-EXACT-VS-MEANFIELD 3/3 (H4):** for K=2, N=3, T=20, exact joint forward-backward (9 joint states tractable) and mean-field VI agree on marginal posteriors to 1e-2 KL after 50 iters.
- **R-SLDS-VS-KF-WHEN-SINGLE-MODE 3/3 (H5/H6):** SLDS/SSSM with M=1 mode collapses to plain Kalman filter — state estimate agreement to 1e-12.
- **R-SLDS-GPB2-VS-RBPF 3/3 (H6):** GPB2 and Rao-Blackwellised PF (slot-265-P24) agree on marginal-mode-posterior to 1e-2 at N=2000 particles.
- **R-IHMM-VS-HMM-AS-ALPHA→0 3/3 (H9):** as DP concentration α→0 the HDP-HMM collapses to a single-state HMM (degenerate cluster); empirically state-count → 1.
- **R-STICKY-VS-IHMM-AS-KAPPA→0 3/3 (H10):** as κ→0 sticky HDP-HMM collapses to vanilla HDP-HMM (H9); posterior state-self-transition rate matches.
- **R-CTHMM-VS-HMM-WHEN-UNIFORM-DT 3/3 (H12):** CT-HMM observed at uniform time grid `Δt=1` collapses to discrete HMM with `A = exp(Q)`; forward-prob agreement to 1e-10.
- **R-MMPP-VS-Q15 3/3 (H13 vs slot-173-Q15):** MMPP-2 stationary-mean-rate from H13 matches slot-173-Q15 closed-form `π·λ` to 1e-9.
- **R-MISSING-OBS-VS-FULL 3/3 (H16):** with all observations present, missing-obs HMM forward-prob matches plain HMM to floating-point.
- **R-HCRF-VS-CRF-WHEN-FIXED-HIDDEN 3/3 (H20):** HCRF with degenerate single-hidden-state collapses to plain LinearChainCRF.

---

## 5. LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|----:|------|-----------|
| H1 | HSMM (Yu-2010) | 280 | A | 165§1, 165§2 |
| H2 | Variable-Duration HMM | 80 | A | H1 |
| H3 | IO-HMM (Bengio-Frasconi) | 280 | B | 165§1, 014 reverse-AD, 068 LBFGS |
| H4 | Factorial HMM | 360 | C | 165§1, 165§2 |
| H5 | Switching State-Space | 280 | D | 161-C5 KF, 165§1 |
| H6 | SLDS (Pavlovic-Rehg) | 340 | D | 161-C5, 165§1, 265-P24 RBPF |
| H7 | Coupled HMM | 180 | E | 256 BP, 165§1 |
| H8 | Variational Bayesian HMM | 220 | F | 165§1, Digamma |
| H9 | iHMM / HDP-HMM | 280 | F | 228-B19 HDP, 165§1, 238-M4 |
| H10 | Sticky HDP-HMM (MOAT) | 360 | F | H9 + auxiliary-Beta |
| H11 | Hierarchical HMM | 250 | E | 165§1, flat-HMM expansion |
| H12 | CT-HMM | 260 | G | linalg.MatExp, 165§1, 165§2 |
| H13 | MMPP-N HMM-inference | 120 | G | H12, slot-173-Q15 cross-link |
| H14 | Pairwise Markov chain | 140 | H | 165§1 (extended state) |
| H15 | Triplet Markov chain | 140 | H | H14 |
| H16 | HMM with missing obs | 130 | I | 165§1 |
| H17 | Forward-with-covariates | 130 | F | H3 (parametric special case) |
| H18 | Mixed HMM | 150 | I | 165§1, 018 Gauss-Hermite |
| H19 | Max-Margin HMM | 220 | J | 165§1, 068 LBFGS |
| H20 | HCRF | 260 | J | 165§7 LinearChainCRF |
| H21 | Particle HMM | 180 | K | 265-P9, 265-P21 FFBS |
| H22 | LD-HMM + multi-channel | 400 | K | factored emission, future-LDA |
| **Σ** | | **~3,720** | | |

Pure-glue ratio: **~70%** of LOC is pure composition over slot-165 §1 forward-backward + slot-228-B19 HDP + slot-161-C5 KalmanFilter + slot-265-P9 BootstrapPF. **~30%** is genuinely-new HMM math: HSMM duration-DP recursion (~150 LOC), structured-MF for factorial-HMM (~200 LOC), GPB2 mode-collapse for SLDS (~140 LOC), Sticky HDP-HMM κ-posterior auxiliary-Beta sampler (~120 LOC), CT-HMM matrix-exponential forward (~80 LOC), HCRF two-pass forward-backward (~120 LOC).

---

## 6. Recommended PR sequence

| PR | Primitives | LOC | Days | Theme |
|----|-----------|----:|----:|-------|
| 1 | H1 HSMM Yu-2010 | 280 | 1 | FIRST explicit-duration HMM, 950 cit. |
| 2 | H5 SSSM + H6 SLDS GPB2 | 620 | 4 | Econometrics/tracking, Hamilton-1989 / Kim-Nelson-1999 textbook. Composes slot-161-C5. |
| 3 | H10 Sticky HDP-HMM Fox-2008 | 360 | 3 | **MOAT** — no public Go impl worldwide. Composes slot-228-B19 + H9 + Beta-binomial κ. |
| 4 | H3 IO-HMM Bengio-Frasconi-1995 | 280 | 2 | Input-conditioned, slot-014 reverse-AD + slot-068 LBFGS. |
| 5 | H4 Factorial HMM Ghahramani-Jordan-1997 | 360 | 2 | Parallel chains + structured-MF VI, 1850 cit. |
| 6 | H12 CT-HMM Liu-Yeh-Tao-2015 | 260 | 2 | Continuous-time, EHR/disease-progression. Needs `linalg.MatExp`. |
| 7 | H8 Variational-Bayes HMM | 220 | 1 | Automatic model selection via ELBO. |
| 8 | H9 iHMM + H13 MMPP-N + H16 missing-obs | 530 | 3 | Infinite-state + CT-Poisson + missing-data. |
| 9 | H7 Coupled + H11 HHMM + H14/H15 Pairwise/Triplet | 710 | 4 | Structural extensions; slot-256 BP for H7. |
| 10 | H19 Max-Margin + H20 HCRF + H21 Particle-HMM + H22 LD-HMM/multi-channel | 1060 | 6 | Discriminative + particle + multi-channel tail. |
| 11 | H2 + H17 + H18 | 360 | 2 | Variant tail. |

Total ~3,720 LOC source + ~2,800 LOC tests, ~11 PRs / ~30 engineer-days. **PR-1 (HSMM) cheapest 1-day first-of-its-kind. PR-3 (Sticky HDP-HMM) singular MOAT. PR-2 (SLDS) highest-citation 4-day bundle.**

---

## 7. Cross-cutting blockers

- **Slot-165 §1 standard HMM** — **gates all of slot-268**. Co-ship slot-165 PR-1 first, then slot-268 PR-1 follows immediately (1-day delta).
- **Slot-165 §2 LogSumExp + Log1mExp + Log1pExp** — promote-from-`changepoint/bocpd.go` (private). 80 LOC, blocks every log-space DP.
- **Slot-228-B19 HDP-CRF sampler** — gates H9 iHMM + H10 Sticky HDP-HMM.
- **Slot-161-C5 KalmanFilter Joseph-form** — gates H5 SSSM + H6 SLDS.
- **Slot-161-C8/C10 EKF/UKF** — gates H6 SLDS-EKF/UKF variants (deferred).
- **Slot-265-P9 BootstrapPF + P21 FFBS** — gates H21 Particle-HMM.
- **Slot-265-P24 RBPF** — gates H6 SLDS-RBPF variant.
- **Slot-238-M1 RNG** — gates Bayesian-inference HMMs (TWENTY-SECOND+ Block-C demand).
- **Slot-238-M2 MetropolisHastings + M4 Gibbs** — gates H8/H9/H10/H17 Bayesian inference.
- **Slot-014 reverse-mode autodiff** — gates H3 IO-HMM gradient + H17 covariate-Baum-Welch.
- **Slot-068 LBFGS** — already in `optim/` (existing). Gates H3 + H19.
- **Slot-018 Gauss-Hermite quadrature** — gates H18 Mixed HMM random-effects E-step.
- **Slot-256 belief-propagation** — gates H7 Coupled HMM N-Heads.
- **`linalg.MatExp`** — gates H12 CT-HMM. Verify shipped (slot 095); +120 LOC if not.
- **`prob.Digamma`** — gates H8 VB-HMM. Verify shipped; +30 LOC if not, shared with slot-228 VB-DP.

---

## 8. Precision hazards (compressed)

- **H1 HSMM:** `D_max` truncation tradeoff `O(T·N²·D_max)` cost vs long-duration bias; Yu-2010 §4 right-censoring for boundary segments; log-space mandatory at T>100.
- **H3 IO-HMM:** `transitionLogProb` callback must return proper log-prob vector (stochastic-row violation silently breaks Baum-Welch).
- **H4 Factorial:** exact `O(T·N^{2K})` blows up at `N^K > 256`; structured-MF posterior `q = Π_k q^{(k)}` strictly looser than tree-VI (Wainwright-Jordan-2008).
- **H5/H6 SSSM/SLDS:** GPB2 collapses `M^2 → M` Gaussian moments; per-particle Kalman covariance must be per-particle in RBPF.
- **H8 VB-HMM:** Dirichlet prior `α=1` default uninformative; `α<<1` near-deterministic, `α>>1` uniform-collapse.
- **H9 iHMM:** direct-assignment Gibbs slow-mixing for large α; use slot-228-B12 split-merge for production.
- **H10 Sticky HDP-HMM:** `κ ∈ [0.5α, 5α]` per Fox-2011 §6; outside range collapses to H9 or single-state. Auxiliary-Beta-binomial `m_jj` count off-by-one is the canonical bug.
- **H12 CT-HMM:** `MatExp(Q·Δt)` Padé-13/13 + scaling-and-squaring (Higham-2009); Krylov-Arnoldi for dim>100; EM Q-update needs `Q·exp(Q·Δt)` order-of-magnitude more expensive — cache.
- **H13 MMPP-N:** rate posterior must use mode-marginal `α_i(t)·β_i(t)` not just forward.
- **H16 missing-obs:** MAR contract only; MNAR biases EM.
- **H18 Mixed HMM:** Gauss-Hermite `n_q ∈ [10,30]`.
- **H19 Max-Margin:** structured-perceptron requires explicit `C·||w||²` regulariser (Sha-Saul-2007 §3).
- **H20 HCRF:** two-pass forward-backward; log-space throughout.
- **H21 Particle HMM:** FFBS-backward mandatory at T>50 for path-degeneracy mitigation.

---

## 9. Distinct from prior agents

- **slot-165 §1** ships standard HMM keystone (forward/backward/Viterbi/Baum-Welch). slot-268 = extensions layer on top — H1 HSMM is the next primitive after slot-165 §1.
- **slot-266 Tier-D** ships S15-S17 = IDENTICAL semantics to slot-165 §1 (slot-266 acknowledges ship-once in §0); slot-268 builds on top.
- **slot-228-B19** ships HDP-CRF; slot-268 H9/H10 consume without duplication.
- **slot-161-C5/C8/C10** ships Kalman/EKF/UKF; slot-268 H5/H6 consume without duplication.
- **slot-265-P9/P21** ships BootstrapPF/FFBS; slot-268 H21 consumes without duplication.
- **slot-173-Q15** ships MMPP-2 closed-form queueing-mean-rate (Heffes-Lucantoni); slot-268-H13 ships HMM-inference consumer — distinct surface, shared parametrisation.
- **slot-238-M1/M2/M4** ships RNG/MH/Gibbs; slot-268 composes without duplication.
- **slot-117 prob-missing** named HMM as missing; slot-268 fills the extension axis.

22 of 22 primitives unique to slot-268 HMM-extensions axis. Zero duplication.

---

## 10. Bottom line

`reality` v0.10.0 ships **ZERO** HMM-extension surface. Closest tangential surfaces are `prob/markov.go` (state-only, no emission) + slot-165 §1 *proposed* standard HMM (gates everything here) + slot-266 Tier-D *proposed* discrete-HMM (identical to 165 §1, ship-once).

**22 HMM-extension primitives H1-H22 totalling ~3,720 LOC of pure connective tissue** stand up the complete HMM-extension discipline on top of slot-165's standard HMM keystone. Tier A duration (HSMM 360), B input (IO-HMM 280), C parallel (Factorial 360), D switching state-space (SSSM+SLDS 620), E structural (Coupled+HHMM 430), F Bayesian/nonparametric (VB+iHMM+Sticky-HDP+covariate 990), G continuous-time (CT-HMM+MMPP 380), H joint-Markov (Pairwise+Triplet 280), I missing/mixed (280), J discriminative (Max-Margin+HCRF 480), K particle/multi-channel (580).

Cheapest 1-day-shippable bundle: **PR-1 H1 HSMM-Yu-2010 ~280 LOC = first explicit-duration HMM in `reality`**. Highest-impact PR: **PR-2 H6 SLDS ~340 LOC** (3000+ citations Hamilton/Kim-Nelson textbook). Most-distinct cutting-edge: **PR-3 H10 Sticky HDP-HMM Fox-2008 ~360 LOC** — no public Go implementation worldwide.

Reality is well-positioned: slot-165 §1 + slot-228-B19 HDP + slot-161-C5 KalmanFilter + slot-265-P9/P21 PF+FFBS + slot-238-M1/M2/M4 MCMC + `prob.Distribution` interface + `prob.MarkovSteadyState` + slot-165 §2 LogSumExp are *exactly* the substrate every H1-H22 primitive needs. Recommended placement: `prob/hmm/` sub-package with `hsmm.go`, `iohmm.go`, `factorial.go`, `slds.go`, `hdp.go`, `cthmm.go`, `pairwise.go`, `missing.go`, `mixed.go`, `maxmargin.go`, `hcrf.go`, `particle.go`, `multichannel.go`.

Conceptual identity: **standard HMM is the meet-in-the-middle of nineteen generalisations**, each adding one structural element (duration, input, parallel-chains, switch-on-Kalman, hidden-CRF, infinite-states, continuous-time, joint-state-observation, missing-mask, random-effects, discriminative-loss, particle-approximation). slot-268 ships these generalisations + saturation-tests reducing each back to the standard HMM in degenerate limits, completing the **full HMM discipline implementation** (1 keystone slot-165 §1 + 22 extensions slot-268, with H21 spanning slot-265, H13 cross-linking slot-173, H22 cross-linking future-LDA).
