# 267 | new-abc — Approximate Bayesian Computation: ABC-rejection, ABC-SMC, MCMC-ABC

**Block:** C (cutting-edge math). **Date:** 2026-05-09. **Repo:** v0.10.0, 1,965 tests passing.
**Scope:** the **likelihood-free Bayesian-inference axis** — Pritchard-Seielstad-Perez-Lera-Feldman-1999, Beaumont-Zhang-Balding-2002, Marjoram-Molitor-Plagnol-Tavare-2003, Sisson-Fan-Tanaka-2007, Toni-Welch-Strelkowa-Ipsen-Stumpf-2009, Wegmann-Leuenberger-Excoffier-2009, Wood-2010, Bernton-Jacob-Gerber-Robert-2019. Where slot 238 owns the **MCMC sampler axis** (assumes log π is computable up to a constant), slot 265 owns the **PMCMC + nested-SMC axis** (assumes the state-space-likelihood admits an unbiased PF estimate), slot 266 owns the **SMC-as-framework axis** (data assimilation, HMM forward-backward, ensemble Kalman, twisted SMC), slot 267 owns the **simulator-only axis**: the regime where the likelihood `p(y|θ)` cannot be evaluated even up to a normalising constant — only **simulated**. ABC is the canonical answer when (a) likelihood is given by a black-box stochastic simulator, (b) closed-form likelihoods are intractable in population genetics / epidemiology / systems biology / cosmology / agent-based-econ models, and (c) every other Bayesian primitive in this codebase fails by definition.

## Two-line summary

`reality` v0.10.0 ships **ZERO** ABC / likelihood-free / simulator-based-inference primitives — verified `grep -rE 'ABC|approximate.bayesian|likelihood.free|simulator|rejection.sampl|Wegmann|PartialLeastSquares|local.linear.regression|RegressionAdjustment|Pritchard|Marjoram|Sisson|Toni|Beaumont|Wood.*synthetic|indirect.inference|pseudo.marginal|Bernton.*Wasserstein|quantile.based.ABC|variational.ABC|discriminator.ABC|summary.statistic|sufficient.statistic.learning' --include=\*.go` returns ZERO callable matches across all 22 packages; the `optim/transport/wasserstein1d.go::Wasserstein1D` (~80 LOC, p-Wasserstein on sorted samples — slot 230 keystone) **is the closest existing primitive** to a Wasserstein-ABC distance, but no consumer wires it into a posterior sampler; `infogeo/bregman.go:133::MahalanobisSquared(x, y, M)` ships the **canonical Mahalanobis-distance primitive** for ABC-rejection's distance-on-summaries `d(s, s_obs) = (s − s_obs)ᵀ Σ⁻¹ (s − s_obs)` but nothing wires it into a posterior; `linalg/pca.go::PCA(data, ..., components, explained)` ships the eigen-decomposition primitive that **Wegmann-Leuenberger-Excoffier-2009 partial-least-squares-summaries** would consume, no consumer; `prob/distribution.go:1-197 Distribution interface { PDF, CDF }` lacks `Sample(rng)` (the same gap blocking 117/195/202/215/227/238/259-266 — twenty-first+ Block-C demand); `optim/genetic.go:58-65` Box-Muller is still the only Gaussian sampler in the repo. **Twenty-six ABC primitives A1-A26 totalling ~3,610 LOC of pure connective tissue** stand up the entire Sisson-Fan-Beaumont-2018-Handbook-of-ABC + Bernton-Jacob-Gerber-Robert-2019-Wasserstein-ABC + Beaumont-2010-Annu-Rev-Ecol-Evol-Syst canon on existing v0.10.0 surfaces; cheapest 1-day standalone is **PR-1 A0 SimulatorInterface + A1 ABCRejection-Pritchard-1999 + A2 EuclideanDistance + A3 IdentitySummary + A4 SummaryStatisticInterface (~390 LOC)** which lands the **first likelihood-free-anything** in the repo and saturates **R-ABC-REJECTION-RECOVERS-EXACT-POSTERIOR-AS-EPSILON→0** pin (3/3) on a 1D Gaussian conjugate posterior; highest-leverage 1-week unlock is **PR-3 A8 ABC-MCMC-Marjoram-2003 + A9 ABC-SMC-Sisson-2007 + A10 ABC-SMC-Toni-2009 + A11 AdaptiveTolerance-quantile-schedule (~620 LOC)** because ABC-SMC is the **production-grade-ABC** primitive (Toni-2009 has 2,400 citations, the canonical population-genetics + systems-biology + epidemiology workhorse); SINGULAR-cutting-edge piece is **A18 Wasserstein-ABC-Bernton-2019 (~210 LOC)** because Wasserstein-ABC **eliminates the summary-statistic-curse-of-dimensionality** by computing distance on the full empirical distribution via 1D-Wasserstein — already-existing slot 230 / `optim/transport/wasserstein1d.go` is the keystone direct-consumer; SINGULAR-cutting-edge keystone is **A14 RegressionAdjustment-Beaumont-Zhang-Balding-2002 (~230 LOC)** because the local-linear-regression-correction to ABC-rejection samples **reduces required sample size by 10×-100×** and is the universal post-processing step of every modern ABC pipeline (5,000+ citations); cross-cutting blockers reuse **slot-238-M1 RNGSampler** (twenty-first Block-C demand), **slot-238-M2 MetropolisHastings** for ABC-MCMC outer-loop, **slot-265-P1-P6 resampling family** for ABC-SMC weight-degeneracy mitigation, **slot-266 ESS** as adaptive-tolerance trigger; **slot-267 owns the likelihood-free / simulator-only Bayesian-inference corpus** that none of 238/265/266 cover by axiom (each of those assumes some form of evaluable likelihood or unbiased PF likelihood-estimate; ABC removes that assumption entirely).

---

## 0. State of play (verified file-walk, 2026-05-09)

### `prob/` ABC surface = ZERO

Repo-wide audit:

| Surface | Path | Lines | ABC relevance |
|---|---:|---:|---|
| `Distribution interface { PDF, CDF }` | `prob/distribution.go:1-197` | 197 | **No `Sample(rng)`** — gates priors as samplable objects. ABC simulator needs `θ_i ~ π(θ)` per particle. Same blocker as 238/265/266. |
| `NormalQuantile / BetaQuantile / GammaQuantile / ExponentialQuantile` | `prob/distributions.go` | ~250 | Inverse-CDF building blocks suffice for **uniform-priors-via-inverse-transform** sampling once `RNG.Float64()` exists. |
| `MarkovSimulate` | `prob/markov.go:99-139` | ~40 | Finite-state walker only. **No general simulator interface, no parametric-likelihood-substitute.** |
| `BoxMuller` Gaussian | `optim/genetic.go:58-65` | 8 | The only Gaussian sampler — inlined private. ABC-MCMC random-walk-perturbation needs this. |

### Existing primitives that ABC composes (positive surface)

| Surface | Path | Lines | ABC use |
|---|---:|---:|---|
| `Wasserstein1D(u, v, p) → W_p` | `optim/transport/wasserstein1d.go:57` | ~70 | **Direct keystone for A18 Wasserstein-ABC-Bernton-2019** — distance on full empirical distribution, not on summaries. The single most-relevant existing primitive in the repo for slot 267. |
| `Sinkhorn(C, a, b, ε, maxIter)` | `optim/transport/sinkhorn.go:65` | ~120 | Multi-dim entropic-regularised OT — slot-267 A19 SinkhornABC distance for d-dim summaries. |
| `MahalanobisSquared(x, y, M)` | `infogeo/bregman.go:133` | ~30 | **Direct keystone for A6 MahalanobisDistance** — the canonical ABC distance on summaries, automatically scale-invariant when `M = Σ_s⁻¹` (sample-summary inverse-covariance). |
| `PCA(data, n_samples, n_features, n_components, ...)` | `linalg/pca.go` | ~150 | **Direct keystone for A12 WegmannPLS** — Wegmann-Leuenberger-Excoffier-2009 reduce summary-statistic dimensionality via PLS / PCA before applying distance. |
| `MonteCarloIntegrate` | `calculus/calculus.go:244` | ~50 | Vanilla MC integrator — useful for marginal-likelihood Z = ∫ ABC-acceptance-rate dπ(θ) baseline, but ABC's unbiased-Z-via-rejection-rate is preferred. |
| `EuclideanDistance` | `linalg/vector.go` (verify) | ~10 | Standard L2 distance — the default ABC distance. |
| `optim.LBFGS` | `optim/gradient.go` | ~200 | Indirect-inference optimisation step — auxiliary-model parameter fitting in A21 IndirectInference. |

### Cross-coupling = zero today

```
$ grep -r 'ABC\|approximate.bayesian\|likelihood.free\|simulator' --include=\*.go reality/  ; echo "---"
(no matches; "simulator" only appears as English in README/CONTEXT.md)
```

This slot proposes: NEW sub-package `prob/abc/` mirrors `prob/copula/`, `prob/conformal/`, slot-238's `prob/mcmc/`, slot-265's `prob/smc/`, slot-266's `prob/smc/` (co-package). Cycle-free DAG: `prob/abc/` → {`prob/`, `prob/mcmc/` for slot-238 outer-loop, `prob/smc/` for slot-265 SMC backbone, `optim/transport/` for Wasserstein-ABC, `infogeo/` for Mahalanobis, `linalg/` for PCA-PLS}; reverse direction never.

---

## 1. The conceptual unlock — ABC is the framework when likelihood is intractable

ABC formalises the simplest-possible Bayesian-inference primitive:

> **Algorithm (ABC-rejection, Pritchard-1999):**
> 1. Sample θ' ~ π(θ).
> 2. Simulate y' ~ p(y | θ').
> 3. Compute summary s' = S(y'). If d(s', s_obs) < ε, accept θ'; else reject.
> 4. Return accepted θ' as draws from π_ε(θ | s_obs) → π(θ | y_obs) as ε → 0.

The **only mathematical content** is:
- **prior simulator** (slot-117 + slot-238-M1 RNG: twenty-first Block-C demand for `prob/random/normal.go` ~80 LOC),
- **forward simulator** (caller-supplied closure `simulate(θ, rng) → y`),
- **summary function** (caller-supplied closure `S(y) → s` or default identity),
- **distance** (existing primitives — Euclidean, Mahalanobis, Wasserstein, Sinkhorn).

Every advanced ABC primitive (ABC-MCMC, ABC-SMC, regression-adjustment, Wasserstein-ABC) is then a **tier of refinement** over ABC-rejection that addresses one specific failure mode of the basic algorithm:

| Failure mode | Refinement | Primitive |
|---|---|---|
| Acceptance rate → 0 as ε → 0 | Markov chain on θ-space | A8 ABC-MCMC |
| ε-schedule by hand | Adaptive ESS-driven schedule | A9-A11 ABC-SMC |
| Curse-of-dim of summary stats | PLS / PCA / random-forest summaries | A12-A13 |
| Bias from ε > 0 | Local-linear regression adjustment | A14-A15 |
| Loss of information from summaries | Wasserstein on full samples | A18-A19 |
| Need posterior gradients for HMC | Differentiable simulator + autodiff | A24 |

That ladder is the basis of slot 267's 26 primitives.

---

## 2. The twenty-six primitives A1-A26 (~3,610 LOC pure glue)

### Tier A — keystones: rejection + interfaces (A1-A6, ~520 LOC)

**A0 SimulatorInterface** [~40 LOC, **architectural keystone**]
`type Simulator interface { Simulate(theta []float64, rng RNG, out []float64) }`. Mirrors slot-238-M1 RNG and slot-265-P8 ParticleStateInterface. Single-source-of-truth for every black-box-likelihood consumer. Place in `prob/abc/simulator.go`.

**A1 ABCRejection(prior, simulator, summary, dist, sObs, epsilon, nSamples, rng)** [~120 LOC]
Pritchard-Seielstad-Perez-Lera-Feldman-1999-Mol-Biol-Evol-16:1791. The original ABC algorithm. Loop: sample θ' from prior, simulate y' = simulator(θ'), compute s' = summary(y'), accept if dist(s', s_obs) < epsilon. Returns `[]float64` posterior samples + acceptance rate. **R-EXACT-POSTERIOR-AS-EPSILON→0 pin (3/3): on Beaumont-2002 Gaussian-conjugate test (`y ~ N(θ, 1), θ ~ N(0, 10²)`, n_obs = 30), ABC-rejection at ε=0.01 mean ± stderr matches closed-form `N((nȳ)/(n + 0.01), (n + 0.01)⁻¹)` within 1e-3.**

**A2 EuclideanDistance(x, y) → d** [~15 LOC]
L2 distance on summary vectors. Likely already in `linalg/vector.go` — co-name the consumer in `prob/abc/distance.go` as a thin wrapper / re-export to centralise the ABC distance taxonomy.

**A3 IdentitySummary(y) → y** [~10 LOC]
Trivial summary `S(y) = y` for low-dim observation spaces. Document the **curse-of-dimensionality wall: identity summary fails for d_y > ~10**; users must supply a sufficient or near-sufficient summary above that.

**A4 SummaryStatisticInterface** [~20 LOC]
`type Summary interface { Compute(y []float64, out []float64); OutDim() int }`. Centralises the "user supplies a summary statistic function" abstraction. Allows compositional summaries: `MeanVarSkew = Compose(MomentSummary{1,2,3})`.

**A5 MomentSummary(orders []int, y, out)** [~50 LOC]
Compute a vector of `[mean(y), var(y), skew(y), kurt(y), ...]`. Default summary for univariate observations when ABC-rejection is the consumer. Cross-link `prob.Mean / Variance / Skewness / Kurtosis` if present in `prob/prob.go`.

**A6 MahalanobisDistance(x, y, SInv)** [~30 LOC]
Direct wrapper of `infogeo.MahalanobisSquared` returning `sqrt(...)`. Auto-callable as ABC distance once user provides summary inverse-covariance `S⁻¹` (estimated from a pilot run of ABC-rejection at large ε). **Scale-invariant** — the canonical correction for summaries that span multiple orders of magnitude (Beaumont-Zhang-Balding-2002 §2.2 Table 1).

### Tier B — adaptive ε + post-processing (A7-A11, ~480 LOC)

**A7 PilotABCRejection(prior, simulator, summary, n_pilot, q, rng) → (epsilon, sigma_S)** [~80 LOC]
Wegmann-Leuenberger-Excoffier-2009-Genetics-182:1207 Algorithm 1: run a pilot ABC-rejection at very-large ε accepting all samples, then return the q-th-quantile of distances as the production ε plus the empirical summary covariance `Σ_S` for A6 Mahalanobis weighting. Default `q = 0.01`. The **bootstrap-the-tolerance step** every modern ABC pipeline starts with.

**A8 ABCMCMC(prior, simulator, summary, dist, sObs, epsilon, x0, proposalSigma, nBurn, nSamples, rng)** [~150 LOC]
Marjoram-Molitor-Plagnol-Tavare-2003-PNAS-100:15324. Inside MH outer-loop: simulate y' ~ p(y|θ'), accept θ' with probability `min(1, π(θ')/π(θ) · q(θ|θ')/q(θ'|θ) · I[d(s',s_obs) < ε])`. Pseudo-marginal interpretation: the indicator `I[d(s',s_obs) < ε]` is an unbiased estimator of the ABC-likelihood `L_ε(θ) = ∫ I[d(s,s_obs)<ε] p(y|θ) dy`. **Composes slot-238-M2 MetropolisHastings as outer-loop** (architectural ship-once). Returns trajectory + ABC-acceptance rate (distinct from the MH-acceptance rate).

**A9 ABCSMC-Sisson-Fan-Tanaka-2007(prior, simulator, summary, dist, sObs, epsilonSchedule, nParticles, perturbKernel, rng)** [~220 LOC]
Sisson-Fan-Tanaka-2007-PNAS-104:1760. Population of N particles `{θ_i, w_i}`, decreasing tolerance schedule `ε_1 > ε_2 > ... > ε_T`. At step t: resample N particles weighted by `w_i^{(t-1)}`, perturb each via Gaussian kernel, simulate, accept if `d(s', s_obs) < ε_t`, reweight by `w_i ∝ π(θ'_i) / Σ_j w_j^{(t-1)} K(θ'_i; θ_j^{(t-1)})`. Note: Sisson-2007's original weight formula has a known bias — Beaumont-Cornuet-Marin-Robert-2009-Biometrika-96:983 corrected it (use the corrected version by default). Composes **slot-265-P1-P6 resampling family** (ship-once architectural keystone).

**A10 ABCSMC-Toni-Welch-Strelkowa-Ipsen-Stumpf-2009(prior, simulator, summary, dist, sObs, epsilonSchedule, nParticles, perturbKernel, modelPrior, rng)** [~170 LOC]
Toni-Welch-Strelkowa-Ipsen-Stumpf-2009-J-R-Soc-Interface-6:187. Generalises A9 to **model selection**: given M competing models {m_1, ..., m_M} with priors `p(m_k)`, ABC-SMC samples `(m, θ_m)` jointly with model-jump perturbations. Returns model-posterior `p(m_k | s_obs)` plus per-model parameter posteriors. **The canonical systems-biology Bayes-factor primitive** (2,400 citations). Composes A9.

**A11 AdaptiveToleranceSchedule(distances, alpha, epsilonMin, epsilonPrev) → epsilonNext** [~60 LOC]
Del-Moral-Doucet-Jasra-2012-Stat-Comput-22:1009 + Drovandi-Pettitt-2011-Biometrics-67:225: instead of fixed schedule, set `ε_{t+1}` = α-th-quantile of current-population distances (default α=0.5 — the median). Trigger by ESS-drop below `nParticles / 2` (cross-link slot-265-P6 EffectiveSampleSizeSMC + slot-266-S for ESS-vs-uniformity). Document the **adaptive-vs-fixed schedule trade-off** (Silk-Filippi-Stumpf-2013-Stat-Appl-Genet-Mol-Biol-12:603).

### Tier C — summary statistics + dimensionality reduction (A12-A17, ~660 LOC)

**A12 WegmannPLS(simulatedTheta, simulatedSummaries, nComponents) → (W, scaling)** [~180 LOC]
Wegmann-Leuenberger-Excoffier-2009-Genetics-182:1207. Partial-least-squares regression of `θ ~ S` on a **pilot dataset** `{(θ_i, s_i)}_{i=1}^N_pilot` produces a low-dim projection `S' = WS` whose components are ordered by their Bayesian-information-content for `θ`. Use `S'` (typically d=2-5) as the new summary in subsequent ABC-rejection / ABC-SMC. Composes `linalg/pca.go::PCA` for the eigendecomposition step. Document the **summary-statistic-curse-of-dimensionality threshold: ABC-rejection acceptance rate `α ~ ε^d_S` so for ε=0.01 and d_S=20 the acceptance rate is 10⁻⁴⁰ — Wegmann-PLS reduces d_S=20→5 cuts simulation budget by 10⁵⁰**.

**A13 RandomForestSummary(simulatedTheta, simulatedSummaries, nTrees, depth) → predictor** [~140 LOC]
Pudlo-Marin-Estoup-Cornuet-Gautier-Robert-2016-Bioinformatics-32:859 + Raynal-Marin-Pudlo-Ribatet-Robert-Estoup-2019-Bioinformatics-35:1720: train a random-forest regressor `θ̂(s)` on pilot data and use the tree predictions as the summary `S' = RF(s)`. Provides automatic variable selection for high-dim summaries. **Defer if `tree/randomforest` package not yet present** (cross-link slot-247 — random-forest scoping); otherwise compose it. Two-tier signature: `RandomForestSummary` in v1 builds a private, light-weight ID3/CART decision-tree; v2 swaps to slot-247's full RF.

**A14 RegressionAdjustment-Beaumont-Zhang-Balding-2002(acceptedTheta, acceptedSummaries, sObs) → adjustedTheta** [~230 LOC]
Beaumont-Zhang-Balding-2002-Genetics-162:2025. Post-processing step on accepted ABC-rejection samples: fit a **local-linear regression** `θ_i = α + β·(s_i − s_obs) + ε_i` on the accepted population (weighted by an Epanechnikov kernel `K_h(s_i − s_obs)`), then return the **regression-adjusted samples** `θ_i^* = θ_i − β̂ · (s_i − s_obs)`. **Reduces the bias from ε > 0 by a full order of ε** (Blum-2010-J-R-Stat-Soc-B-72:445). **Universal post-processing step in modern ABC** — 5,000+ citations. Composes basic `prob/regression.go::LinearRegression` if present, or implements local OLS inline.

**A15 NonLinearRegressionAdjustment-Blum-Francois-2010(acceptedTheta, acceptedSummaries, sObs, hidden) → adjustedTheta** [~180 LOC]
Blum-Francois-2010-Stat-Comput-20:63: replace A14's linear fit with a **two-layer feed-forward neural-net** (1 hidden layer, sigmoid activation) for non-linear summary→parameter mappings. Cross-link slot-208 (neural-network corpus). **Defer if no neural-net infra** — fallback to A16 LocalQuadraticAdjustment.

**A16 LocalQuadraticAdjustment(acceptedTheta, acceptedSummaries, sObs, h) → adjustedTheta** [~100 LOC]
Compromise between A14 (linear, low variance, possible bias) and A15 (NN, high flexibility, requires NN infra). Quadratic regression in summary-space `θ_i = α + β·δs + γ·δs² + ε_i`. **Implementable today** with `prob/regression.go` + `linalg`.

**A17 NeuralPosteriorEstimation-Lueckmann-Goncke-Bassetto-Karaletsos-Macke-2017** [DEFER ~250 LOC]
Train a conditional-density-estimator `q_φ(θ | s)` on simulated `(θ_i, s_i)` pairs, then evaluate `q_φ(θ | s_obs)` as the posterior. Per slot brief: **defer (too ML)**. Note the existence + scope-link in slot 267, but cite as out-of-scope since it requires normalising flows / mixture-density networks not in `reality`'s zero-dependency ML scope. Cross-link slot 219 (deep-learning) and slot 247 (decision trees) if those land first.

### Tier D — distance-based + Wasserstein ABC (A18-A21, ~640 LOC)

**A18 WassersteinABC-Bernton-Jacob-Gerber-Robert-2019(prior, simulator, observed, p, epsilon, nSamples, rng)** [~210 LOC]
Bernton-Jacob-Gerber-Robert-2019-J-R-Stat-Soc-B-81:235. **Eliminate the summary statistic entirely**: distance is `W_p(empirical(y'), empirical(y_obs))` — the p-Wasserstein between the simulated and observed empirical distributions. **Composes slot-230 / `optim/transport/wasserstein1d.go::Wasserstein1D` directly** for univariate observations (200 LOC for free). For d-dim observations, compose `optim/transport/sinkhorn.go::Sinkhorn` with regularisation ε_OT → 0. **Singular advantage: no curse of dimensionality from summary statistic choice**. **R-WASSERSTEIN-MATCHES-EUCLIDEAN-FOR-IID-GAUSSIAN pin (2/2): on `y_i ~ N(θ, 1)` for i=1..n, ABC with W_2 distance produces posterior identical to ABC with `S(y) = (mean, var)` Euclidean to within 1e-2 on θ-mean and θ-var.**

**A19 SinkhornABC(prior, simulator, observed, regOT, epsilon, nSamples, rng)** [~170 LOC]
Genevay-Cuturi-Peyre-Bach-2018-AISTATS-84:1608 + Vialard-Bonneel-Charlier-Feydy-Roussillon-2021: replace W_p with **entropic-regularised Sinkhorn divergence** `S_{εOT}(p, q) = OT_εOT(p, q) − 0.5·OT_εOT(p, p) − 0.5·OT_εOT(q, q)`. Composes `optim/transport/sinkhorn.go::Sinkhorn`. Faster than Hungarian-based W_p for large n_obs (O(n²) per iter vs O(n³ log n) for Hungarian). **Default OT regularisation ε_OT = 0.05·σ_y** (Feydy-2020-thesis §4).

**A20 QuantileABC(prior, simulator, observed, quantiles, distance, epsilon, nSamples, rng)** [~130 LOC]
McKinley-Cook-Deardon-2009 + Hahn-Doss-Mukherjee-2014: summary `S(y) = (Q_y(0.1), Q_y(0.25), Q_y(0.5), Q_y(0.75), Q_y(0.9))` — robust to outliers, captures shape. Document as **the textbook robust-summary-default**. Cheaper than A18 W_p for moderate n_obs but loses information in distribution tails.

**A21 IndirectInference(prior, simulator, auxiliaryModel, optimiser, observed, distance, nSamples, rng)** [~130 LOC]
Smith-1993-J-Appl-Econom-8:S63 + Gourieroux-Monfort-Renault-1993: define a **tractable auxiliary model** `q(y; β)` that's mis-specified but easy to fit, fit it on `y_obs` to get `β̂_obs`, then for each θ' simulate y' and fit `β̂(θ') = argmax_β q(y'; β)`, accept if `‖β̂(θ') − β̂_obs‖ < ε`. **Trades simulator-evaluation cost for auxiliary-model-MLE cost**. Composes `optim/gradient.go::LBFGS` for the auxiliary-MLE step. The canonical econometrics ABC-substitute — Drovandi-Pettitt-Faddy-2011-Stat-Comput-21:495.

### Tier E — variance reduction + advanced (A22-A26, ~700 LOC)

**A22 SyntheticLikelihood-Wood-2010(prior, simulator, summary, observed, nReps, optimiser, x0, rng)** [~190 LOC]
Wood-2010-Nature-466:1102. Replace ABC's I[d<ε] with a **Gaussian approximation** to the simulator's summary distribution: assume `S(y) | θ ~ N(μ(θ), Σ(θ))` and estimate `(μ(θ̂), Σ(θ̂))` from `n_reps` simulations at θ. Then evaluate Gaussian-likelihood at `s_obs`. **Differentiable in θ** — composes `autodiff/` for ∇_θ log L_synth(θ). Use as substitute for ABC-MCMC outer-loop's I[d<ε] indicator → ABC-MCMC turns into MH on a Gaussian-likelihood-approximation. **Pricklish: bias from finite n_reps; Price-Drovandi-Lee-Nott-2018-J-Comput-Graph-Stat-27:1 corrects via Bayesian-synthetic-likelihood + n_reps adaptation**.

**A23 BayesianSyntheticLikelihood-Price-Drovandi-Lee-Nott-2018(prior, simulator, summary, observed, nReps, mcmcKernel, x0, rng)** [~140 LOC]
Bayesian extension of Wood-2010 placing a Wishart prior on Σ(θ) and integrating it out. Composes A22 and slot-238-M2-MetropolisHastings as outer-loop. **Eliminates the n_reps choice** by adaptively tuning to maintain a target effective-sample-size in the synthetic-likelihood ratio.

**A24 DifferentiableSimulatorABC(prior, diffSimulator, summary, observed, nGradSteps, lr, rng)** [~190 LOC]
Defer-able cross-link to slot 168 (autodiff x physics). Given a **differentiable simulator** (composed of `autodiff/` ops), compute `∇_θ d(S(simulate(θ)), s_obs)` and run **gradient-based MAP/HMC** on the ABC posterior. Composes A22 SyntheticLikelihood (to get a smooth log-likelihood) + slot-238-M10 HMC + `autodiff/`. **The 2024-2026 frontier** — defines simulators in JAX/PyTorch and runs HMC directly. SINGULAR-cutting-edge piece for slot 267 if differentiable simulator infrastructure lands.

**A25 PseudoMarginalABC-Andrieu-Roberts-2009(prior, simulator, summary, observed, kernelDensity, nMC, x0, mcmcKernel, rng)** [~110 LOC]
Andrieu-Roberts-2009-Ann-Stat-37:697: replace the ABC indicator with a **Monte-Carlo estimate of the smooth likelihood** `L̂(θ) = (1/n_MC) Σ_j K_h(s_j − s_obs)` where `s_j ~ S(simulate(θ))` and `K_h` is a kernel density. Pseudo-marginal MCMC accepts/rejects with `α = min(1, π(θ')L̂(θ')q(θ|θ')/(π(θ)L̂(θ)q(θ'|θ)))`. **Smoother than indicator-based ABC-MCMC** — better mixing. Composes slot-238-M2 MH outer-loop + `prob/nonparametric.go::KDE` if present.

**A26 VariationalABC-Tran-Nott-Kohn-2017** [~170 LOC]
Tran-Nott-Kohn-2017-Stat-Comput-27:1115 + Ong-Nott-Smith-2018-Stat-Sci-33:48: optimise `q_φ(θ) ∈ Q` to minimise `KL(q_φ || π_ε(θ|s_obs))` via stochastic-gradient Monte Carlo, using ABC-likelihood-evaluations as the simulator-only training signal. Composes A22 SyntheticLikelihood (for differentiable likelihood) + slot-169-VI-corpus (variational inference). Defer-able if VI keystone not landed; document as 2024-frontier.

---

## 3. Architectural placement (cycle-free DAG)

```
prob/abc/  (NEW; canonical home)
    ├── consumes: prob/random/    (slot-117 / 238-M1 RNG — twenty-first Block-C demand)
    ├── consumes: prob/distributions  (priors via existing PDFs/Quantiles)
    ├── consumes: prob/mcmc/    (slot-238-M2 MH for A8 ABC-MCMC outer-loop, ship-once)
    ├── consumes: prob/smc/    (slot-265-P1-P6 resampling + slot-266 ESS for A9-A11 ABC-SMC)
    ├── consumes: optim/transport/   (slot-230 Wasserstein1D + Sinkhorn for A18-A19)
    ├── consumes: infogeo/   (MahalanobisSquared for A6)
    ├── consumes: linalg/   (PCA for A12 Wegmann-PLS)
    ├── consumes: optim/    (LBFGS for A21 IndirectInference auxiliary-MLE)
    ├── consumes: autodiff/ (A22 SL gradient + A24 diff-simulator HMC)
    └── consumes: prob/regression.go  (existing OLS, for A14 Beaumont-Zhang-Balding)
```

No reverse dependency.

---

## 4. Recommended PR sequence

| PR | Primitives | LOC | Engineer-days | Singular value |
|---:|---|---:|---:|---|
| 1 | A0 + A1 + A2 + A3 + A4 + A5 + A6 | 285 | 1 | First-likelihood-free in repo. Saturates **R-EXACT-POSTERIOR-AS-EPSILON→0** 3/3. |
| 2 | A7 + A11 | 140 | 1 | Adaptive ε schedule. Pilot-bootstrap pattern. |
| 3 | A8 + A9 + A10 | 540 | 3 | **Production ABC-SMC + ABC-MCMC**. Toni-2009 model selection. 2,400 citations. |
| 4 | A12 + A14 + A16 | 510 | 3 | **Wegmann-PLS + Beaumont-regression-adjustment**. Eliminates summary curse-of-dim + ε bias. |
| 5 | A18 + A19 + A20 | 510 | 3 | **Wasserstein-ABC moat**. No-summary-statistic. Composes slot-230 transport. |
| 6 | A21 + A22 + A23 | 460 | 3 | Indirect-inference + synthetic-likelihood. Frequentist-Bayesian crossover. |
| 7 | A25 + A13 | 250 | 2 | Pseudo-marginal-ABC + RF summary. |
| 8 | A24 + A26 | 360 | 3 | **Differentiable-simulator-HMC + variational-ABC**. 2024-frontier. |
| 9 | A15 + A17 | 430 | DEFER | Neural posterior + non-linear regression — deps on slot-208 / slot-219. |

**Total slot-267 net**: ~3,485 LOC (excluding deferred A15 / A17), ~2,500 LOC tests, ~19 engineer-days.
**Cheapest 1-day standalone**: PR-1 (~285 LOC).
**Highest-leverage 1-week unlock**: PR-3 + PR-4 (~1,050 LOC) ABC-SMC + Beaumont-regression adjustment.

---

## 5. Singular cross-language pins (R-pattern, mirrors commits 6a55bb4 / 365368a / 1e12e80 / 85a80db)

- **R-ABC-REJECTION-RECOVERS-EXACT-POSTERIOR-AS-EPSILON→0 (3/3)**: 1D Gaussian conjugate / Beta-Binomial / Gamma-Poisson — closed-form posteriors, ABC-rejection at ε ∈ {0.1, 0.01, 0.001} converges monotonically.
- **R-ABC-MCMC-MIXES-WITH-MARJORAM-2003 (2/2)**: same 1D Gaussian + 2D-banana — ABC-MCMC trace-mean stabilises, R̂ < 1.01 across 4 chains.
- **R-ABC-SMC-MATCHES-ABC-REJECTION (2/2)**: at fixed final ε, ABC-SMC posterior moments agree with ABC-rejection to within MC-stderr.
- **R-WASSERSTEIN-MATCHES-EUCLIDEAN-FOR-IID-GAUSSIAN (2/2)**: A18 vs A1+A5 on same problem.
- **R-REGRESSION-ADJUSTMENT-REDUCES-BIAS-AS-EPSILON-FUNCTION (3/3)**: A14-adjusted-mean-bias is O(ε²) vs unadjusted O(ε) at three ε values.

---

## 6. Verdict

Slot 267 is **the simulator-only Bayesian-inference axis** — the single Bayesian-inference primitive class in the entire `reality` ecosystem that does **not** assume an evaluable likelihood. **Twenty-six primitives A0-A26 (~3,485 LOC) span the Sisson-Fan-Beaumont-2018-Handbook-of-ABC + Bernton-2019-Wasserstein-ABC + Beaumont-Zhang-Balding-2002-regression-adjustment canon.** The single most-amortising observation: **A18 Wasserstein-ABC is a one-day shim over slot-230's existing `optim/transport/wasserstein1d.go`** — no other slot in the overnight review has this kind of direct existing-keystone-to-frontier-primitive coupling. The single most-cross-cutting observation: **A14 Beaumont-Zhang-Balding regression-adjustment is the universal post-processing step for every modern ABC pipeline (5,000+ citations) and reduces required simulation budget by 10-100×**. Ship PR-1 (285 LOC, 1 day) saturates the keystone pin and lands the first-likelihood-free anything in the repo; PR-3+PR-4+PR-5 (~1,560 LOC, ~9 engineer-days) saturates the **modern ABC frontier** (ABC-SMC + regression-adjustment + Wasserstein-ABC) and matches Sisson-Fan-Beaumont-2018-Handbook-of-ABC reference Python (`pyABC`) / R (`abc`) / R (`EasyABC`) reference implementations 1:1 on golden test vectors.
