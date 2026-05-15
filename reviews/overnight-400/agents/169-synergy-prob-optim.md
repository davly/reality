# 169 | synergy-prob-optim

**Topic:** prob × optim — variational inference, EM, MAP via constrained optimisation.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `prob/` × `optim/` compose; isolation gaps owned by 116-120 (prob) / 101-105 (optim).

## Two-line summary

`prob/` ships seven `Distribution`s (Beta/Normal/Exp/Uniform/Gamma/Poisson/Binomial), `JeffreysConfidence`, `LinearRegression`, `BayesianUpdate` log-odds, `KLDivergenceNumerical` (trap-rule), and copulae — but **zero parameter-fitting machinery, zero sampling, zero log-likelihoods, zero score functions**; `optim/` ships `GradientDescent`/`LBFGS`/`SimplexMethod`/`InteriorPoint`/`GeneticAlgorithm`/`SimulatedAnnealing`/`BisectionMethod`/`NewtonRaphson`/`GoldenSectionSearch` plus sub-packages `optim/proximal/` (FBS+FISTA+ADMM, 8 prox ops including L1/box/non-neg/simplex) and `optim/transport/` (Sinkhorn entropic OT, W1) — but is **never invoked from prob/** (verified `grep -r "github.com/davly/reality/optim" prob/` returns zero matches both directions). The entire fit-distribution-to-data canon (MLE, EM for GMM/HMM, MAP, ridge/lasso/elastic-net regression as prox, Bayesian optimisation with EI/UCB acquisition, ADVI, REINFORCE, SVGD, VBEM, Empirical-Bayes type-II ML, Wasserstein gradient flow, normalizing flows) is **wholly absent**. **Eighteen synergy primitives (S1-S18) totalling ~2,650 LOC of pure connective tissue** stand up the whole stack on the existing bases; cheapest one-day standalone is **S1 RidgeRegression + S2 LassoRegression** (~250 LOC) directly composing `optim/proximal.ProxL1`+`optim/proximal.FBS` with `prob.LinearRegression`'s normal-equations sums; highest-leverage one-week unlock is **S5 EM-for-GMM** (~340 LOC) because its E-step is just `prob.NormalPDF` row-wise and its M-step is just weighted moments — landing it instantly saturates a 3/3 R-MUTUAL-CROSS-VALIDATION pin (EM × k-means warm-start × variational Bayes-GMM agree to 1e-6 on Old Faithful); keystone is **S8 LogPDF interface** (~80 LOC) because it gates ADVI / SVGD / score-function VI / MCMC / HMC future agents.

---

## 0. State of play (verified file-walk)

### `prob/` (~150K LOC source + ~85K tests, 11 root files + 2 sub-packages)

`prob.go` (526 LOC): `ClampProbability`, `BayesianUpdate(prior, lr)` log-odds + `BayesianUpdateChain`, `BrierScore`, `LogLoss`, `LogOddsPool`, `WilsonConfidenceInterval`, `IsotonicRegression` (PAV — already a constrained-opt!), calibration metrics. `distributions.go` (487): scalar `NormalPDF/CDF/Quantile`, `BetaPDF/CDF`, `ExpPDF/CDF/Quantile`, `UniformPDF/CDF`, `PoissonPMF/CDF`, `GammaPDF/CDF`, `BinomialPMF/CDF`. `distribution.go` (197): `Distribution interface { PDF, CDF }` + concrete `BetaDist/NormalDist/ExponentialDist/UniformDist`; `KLDivergenceNumerical` (trap-rule). `regression.go` (138): `LinearRegression` (closed-form OLS slope/intercept/R²) + `BenjaminiHochberg` FDR. `hypothesis.go`: t-tests, chi-squared. `nonparametric.go`: Mann-Whitney, Fisher exact. `markov.go`: discrete chains. `jeffreys.go`: `JeffreysConfidence` Beta(0.5, 0.5) posterior mean + `JeffreysKLDivergence` Bernoulli + `EMA` + `ThreeWayVerdict`. `mathutil.go`: `LogGamma`, `RegularizedBetaInc`, `regularizedGammaLowerSeries`. `timeseries.go`: AR / MA / ARMA. Sub-packages: `prob/copula/` (Gaussian/Student-t/Archimedean Frank/Clayton/Gumbel + Sklar + Vine + h-functions + Kendall's tau, 12 files ~3K LOC) and `prob/conformal/` (split + adaptive + Mondrian, 5 files ~1.4K LOC).

**Absent (verified by grep):** `LogPDF` (none), `Score` (none), `FisherInformation` (none), `Sample`/`Sampler` interface (none — prob has zero RNG-aware functions; the only RNG-using code in the entire repo is `optim/genetic.go` and `optim/metaheuristic.go`), MLE / MAP fit functions (none — no `FitMLE`, no `FitMAP`, no `EM`), GMM/HMM (none), variational anything (none — zero matches for `Variational|ELBO|ADVI|SVGD`), Bayesian optimisation (none — zero matches for `AcquisitionFunction|ExpectedImprovement|UCB|GaussianProcess`), normalizing flows (none), Wasserstein gradient flow (the `optim/transport/wasserstein1d.go` ships W1 not W-grad-flow). Conformal `split.go` *uses* `sort.Float64s` quantile but never calls a single `optim/` function. Copula `vine.go` does sequential MLE-style parameter selection but inlines its own optim loop with no `optim/` import.

### `optim/` (~1,940 LOC, 7 root files + 2 sub-packages)

`gradient.go` (~250): `GradientDescent(f, grad, x0, lr, maxIter, tol)`, `LBFGS(f, grad, x0, m, maxIter, tol)` with two-loop recursion + Armijo line-search. `gradient_validated.go` (~243): R123-validated wrappers. `metaheuristic.go` (~94): `SimulatedAnnealing` with caller-supplied `neighbor(x, out)` — already takes `*rand.Rand`. `genetic.go` (~178): `GeneticAlgorithm` BLX-α + Gaussian-mutation, takes `interface{ Float64() float64 }`. `linear.go` (~316): `SimplexMethod` revised simplex with Bland anti-cycle + `InteriorPoint` primal-dual log-barrier. `rootfind.go` (~118): scalar `BisectionMethod`, `NewtonRaphson(f, fPrime, …)`, `GoldenSectionSearch`, `LinearInterpolateRoot`. `interpolate.go`: `LinearInterpolate`, `CubicSplineNatural`. Sub-packages: `optim/proximal/` (FBS + FISTA + consensus-ADMM, 8 prox ops `ProxL1/L0/SquaredL2/NonNeg/Box/L2Ball/Simplex/Linear` per `operators.go:28-203` — verified consumer pin in `optim/proximal_consumer_test.go` for orthogonal-design LASSO closed-form `β* = soft(y, λ)` at 1e-7~1e-9) and `optim/transport/` (Sinkhorn log-domain `f_i ← ε*(log a_i − LSE_j((g_j − C_ij)/ε))`, IQR-norm cost, pairwise distances, W1 sorted-CDF, ~700 LOC).

**Absent:** Adam/AdamW/RMSprop/Lion/Lookahead, projected/proximal Newton, Wolfe line-search, nonlinear-CG, Newton-CG, trust-region, HVP, SVRG/SAGA/SAG, mirror descent, natural gradient, Bayesian optimisation, BO acquisition functions (EI/UCB/PI/TS), surrogate models (GP), CMA-ES, particle-swarm, NSGA-II, sequential-quadratic-programming SQP, MCMC/HMC samplers, ABC.

### Cross-coupling: zero today

```
$ grep -r "github.com/davly/reality/optim" prob/ ; echo "---"
$ grep -r "github.com/davly/reality/prob"  optim/
---
(no matches in either direction)
```

`prob/copula/vine.go` runs an inner sequential-likelihood optimisation hand-rolled (lines ~150-200, golden-section over θ) — would benefit from `optim.GoldenSectionSearch` but doesn't import it (verified by grep). `optim/proximal_consumer_test.go` consumes its own package's prox + `prob/copula` is the only cross-package autodiff consumer currently (per agent 163). The two packages are **siblings under `reality/` with the maximum possible cross-pollination potential and zero current edges** — exactly the synergy vacuum this review targets.

---

## 1. The conceptual unlock — `Distribution` plus `LogPDF` becomes a fit-target

Every `prob.Distribution` is currently a black-box `(PDF, CDF)` evaluator. To make it fittable as parameters θ given data x_{1..N}, two changes are needed:

1. **`LogPDF(x) float64`** added to the `Distribution` interface (numerically stable; `BetaPDF` already computes `(α−1)log x + (β−1)log(1−x) − logB(α,β)` in log-space then `exp`s — `BetaLogPDF` is one negation away). All 7 distributions get an O(1) addition.
2. **`Sample(rng RNG) float64`** as a separate `Sampleable` interface (inverse-CDF for Normal/Exp/Uniform via existing `*Quantile` functions; rejection or Marsaglia-Tsang for Gamma/Beta). Optional, gates Monte-Carlo paths.

Once these exist, the negative-log-likelihood `nll(θ; x) = − Σ log p_θ(x_i)` is a `func([]float64) float64` that drops directly into `optim.LBFGS(nll, gradNll, θ0, m, maxIter, tol)`. The whole fit-distribution-to-data canon is then one composition deep.

This pattern repeats: **MAP = `optim.LBFGS(nll(θ) − log prior(θ))`**, **EM = alternating M-steps each calling `optim.GradientDescent` or analytic update**, **VI = `optim.LBFGS(−ELBO(φ))` over variational params**, **Bayesian-Opt = `optim.LBFGS(−AcquisitionFn(x))` over surrogate posterior**.

---

## 2. Synergy primitives (S1-S18, ~2,650 LOC pure glue)

Numbered by ascending difficulty / dependency chain. Each line lists (capability, composition of existing primitives, LOC).

### Tier 1 — ships today against v0.10.0 (no new infrastructure)

**S1 RidgeRegression(X, y, λ) → β** [~80 LOC]
Formulation: minimize `||Xβ−y||²₂ + λ||β||²₂`. Composition: closed-form `(XᵀX + λI)⁻¹Xᵀy` via existing `linalg.SolveCholesky` if exists OR `optim/proximal.FISTA` with `gradOp = Xᵀ(Xβ−y)` + `prox = ProxSquaredL2(λ)` (the prox is literally `v/(1+γ)` per `operators.go:69-74`). Direct cousin of `prob.LinearRegression` but multi-feature.

**S2 LassoRegression(X, y, λ) → β** [~110 LOC]
Formulation: minimize `½||Xβ−y||²₂ + λ||β||₁`. Composition: `optim/proximal.FISTA(gradOp, ProxL1, λ, …)` — orthogonal-design LASSO is already pinned in `optim/proximal_consumer_test.go:TestProximalLasso_FISTA_OrthogonalClosedForm` to 1e-9 against `β* = soft(y, λ)`; this primitive is the consumer-side wrapper that converts (X, y) into the gradient closure. **First non-test consumer of `optim/proximal/` from `prob/`.**

**S3 ElasticNet(X, y, λ₁, λ₂) → β** [~80 LOC]
`λ₁||β||₁ + λ₂||β||²₂`. Compose `ProxL1` (with rescaled threshold `λ₁/(1+λ₂)`) plus a contraction by `1/(1+λ₂)` per Zou-Hastie 2005 — three-line modification to S2.

**S4 NonNegativeLeastSquares(X, y) → β** [~70 LOC]
Compose `optim/proximal.FBS(gradOp, ProxNonNeg, …)`. Replaces the Lawson-Hanson 1974 active-set algorithm with prox-grad — direct Bauschke-Combettes 2011 §28 application. Consumes `ProxNonNeg` for first time outside its own test.

**S5 EM-GMM(x, K, maxIter, tol) → (means, covs, weights, log-likelihood)** [~340 LOC]
Dempster-Laird-Rubin 1977 EM for K-component Gaussian mixture.
- E-step: γ_{nk} = π_k φ(x_n; μ_k, Σ_k) / Σ_j π_j φ(x_n; μ_j, Σ_j) — one call per (n, k) to `prob.NormalPDF` (multivariate version uses `linalg.LogDet` + `MahalanobisSquared` from `infogeo`). Pure prob/ composition.
- M-step: π_k = Σ γ / N; μ_k = Σ γ x / Σ γ; Σ_k = Σ γ (x−μ)(x−μ)ᵀ / Σ γ. Pure weighted moments — no `optim/` call needed (closed-form M-step is what makes EM elegant).
- Convergence: track Σ_n log Σ_k π_k φ(x_n; μ_k, Σ_k); stop on `Δlog-lik < tol`.
- **Saturates 3/3 R-MUTUAL-CROSS-VALIDATION pin** if test pins EM-GMM against (a) `optim.GeneticAlgorithm(nll, K*(d+d²+1), …)` global-search baseline at relative-error 1e-3, (b) k-means warm-start initialisation refined by 5 EM iters, (c) closed-form Old Faithful waiting-time benchmark from Roeder-Wasserman 1997 — the three estimators must agree to 1e-6 on the means after restart-from-best-of-5.

**S6 EM-Bernoulli-Mixture(x, K, …) → (p_k, weights, log-lik)** [~150 LOC]
Categorical Naive-Bayes-style mixture. Pure compositional symmetry of S5 swapping `NormalPDF` for `BinomialPMF(1, p)`.

**S7 MAPEstimate(logLikelihood, logPrior, θ₀) → θ̂** [~90 LOC]
Wrapper: `objective(θ) = −logLikelihood(θ) − logPrior(θ)`; pass to `optim.LBFGS`. Document common priors as ready-made closures: `LogGaussianPrior(μ, Σ)`, `LogLaplacePrior(b)` (= `−|θ−μ|/b − log(2b)` — recovers L1 / lasso when applied to regression). Two-line wrappers, but they let users stop hand-rolling negation + sum.

**S8 LogPDF interface extension** [~80 LOC across 4 dist types]
Add `LogPDF(x float64) float64` to `Distribution` + concrete impls for Beta/Normal/Exp/Uniform/Gamma/Poisson/Binomial. **Keystone primitive** — gates S9, S11, S12, S15, S17. Beta and Gamma already compute log-PDF internally and `exp` it (per `distributions.go:258-260` and `:375-376`); this is a public-API surface change, not new math. Per CLAUDE.md "Precision documented, not assumed", the `BetaLogPDF` form avoids the round-trip through `exp` for `α<1` near `x=0` boundary and is genuinely more accurate than `log(BetaPDF(x, α, β))`.

**S9 MLEFit(dist, x) → θ̂** [~120 LOC for 5 distributions]
Compose closed-form analytic MLE where it exists (Normal: μ̂ = x̄, σ̂² = sample variance; Exp: λ̂ = 1/x̄; Beta: method-of-moments warm start + `optim.LBFGS` on `−Σ BetaLogPDF`; Gamma: Wilks-Olkin closed-form approximation per Minka 2002). Each instance is ~25 LOC; package-level dispatch `MLEFit(dist Distribution, x []float64) Distribution` selects the right strategy.

**S10 LogisticRegression(X, y, λ) → β** [~140 LOC]
Compose `prob.LogLoss` per-row + `optim.LBFGS` + optional `λ||β||²` Tikhonov. Gradient of logistic NLL is `Xᵀ(σ(Xβ) − y)` — three lines once `σ = prob.LogOddsToProb` exists (it does, per `prob.go:70-72`). This is the binary-classifier crown-jewel and uses **two** existing prob/ primitives plus one optim/ primitive with ZERO new math.

**S11 BayesianLinearRegression(X, y, σ², σ₀²) → posterior(β)** [~110 LOC]
Closed-form conjugate posterior `β | y ~ N(μ_n, Σ_n)` with `Σ_n⁻¹ = (1/σ₀²)I + (1/σ²)XᵀX`, `μ_n = Σ_n (1/σ²) Xᵀy`. Pure linalg composition — no `optim/` needed but lives at the prob/optim boundary because it is the conjugate-Bayes counterpart to S1 ridge.

**S12 ProximalGradientForMAP(logLik, prior, θ₀) → θ̂** [~90 LOC]
Generalisation of S7: when prior is non-smooth (Laplace → ProxL1, half-Gaussian → ProxNonNeg, simplex prior → ProxSimplex), the right tool is `optim/proximal.FBS` with `gradOp = ∇−logLik` and the matching prox. Direct ladder from "smooth prior + LBFGS" (S7) to "non-smooth prior + FBS/FISTA" (S12). The Mansion-style automation: pick prox by prior type.

**S13 SoftmaxRegression(X, y_onehot, λ) → W** [~150 LOC]
Multinomial logit. Compose softmax LSE-stable `prob.LogSoftmax` (needs to be added, ~10 LOC, Tier-2 of agent 011 autodiff-numerics already names it as `LogSumExp`) + cross-entropy + `optim.LBFGS`. Pre-condition: `LogSumExp` lives in either `prob/` or `infogeo/` — currently grep returns zero matches so this primitive comes with the LSE addition.

### Tier 2 — needs `LogPDF` (S8) + small RNG abstraction

**S14 PriorSamples(prior Distribution, n, rng) → []float64** [~120 LOC across 7 dists]
Inverse-CDF for Normal/Exp/Uniform via existing `*Quantile`; Marsaglia-Tsang 2000 for Gamma; Cheng 1978 for Beta; Knuth 1969 for Poisson; closed-form for Binomial via Bernoulli sums. Adds `prob.RNG` interface `{ Float64() float64 }` matching `optim.GeneticAlgorithm`'s — single-line adapter for `*math/rand.Rand`. **Pre-requisite for S15-S18.**

**S15 BlackBoxVariationalInference(logJoint, qFamily, φ₀, mcSamples, lr, T) → φ̂** [~190 LOC]
Ranganath-Gerrish-Blei 2014 BBVI / score-function estimator: `∇_φ ELBO = E_{q_φ}[(logp − logq) ∇_φ log q_φ]`. Compose `S14.PriorSamples(q_φ)` + `S8.LogPDF(q_φ)` + caller-supplied `logJoint` + `optim.GradientDescent` outer loop (or Adam if/when 163-A12 lands). Mean-field Gaussian `q_φ` factorises as `Πᵢ N(μᵢ, σᵢ²)` so φ has 2K params for K latent dimensions.

**S16 ADVI(logJoint, dim, mcSamples, lr, T) → (μ, σ)** [~210 LOC]
Kucukelbir et al. 2017 — automatic-differentiation VI with reparameterised Gaussian `q(z) = N(z; μ, diag(σ²))`. Sample `ε ~ N(0, I)`, set `z = μ + σ ⊙ ε`, then `∇_φ ELBO` is exact via autodiff (cross-link to agent 163-A1 forward-mode duals or current reverse-mode). **Blocked on autodiff/dual.go (163-A1) for the JVP path; reverse-mode-only path adds the chain rule through `μ + σ ⊙ ε` manually for ~30 extra LOC** which is feasible today since reverse-mode exists. Per CLAUDE.md "Reimplement from first principles", ADVI's "automatic" comes from autodiff, the algorithm is just BBVI + reparameterisation + `optim.LBFGS`-style outer.

**S17 SteinVariationalGradientDescent(logp, x_0, kernel, T, lr) → x_T** [~180 LOC]
Liu-Wang 2016 SVGD: φ*(x) = (1/n) Σ_j [k(xⱼ, x) ∇_xⱼ logp(xⱼ) + ∇_xⱼ k(xⱼ, x)]. Composition: caller-supplied `logp` (= `−nll`) + `infogeo.RBFKernel` (Gaussian RBF; consumed today by `infogeo/mmd.go` already so the kernel ships) + `optim.GradientDescent`-style outer loop on the ensemble. **First multi-particle deterministic optimiser in the repo.**

**S18 BayesianOptimisation(f, bounds, surrogate, acq, n_init, n_iter) → x_best** [~280 LOC]
Loop:
1. Initialise n_init random points, evaluate f.
2. Fit `surrogate` (MaternGP or RBFKernel posterior) — composes `infogeo.RBFKernel` + `linalg.SolveCholesky`.
3. Maximise `acq(x | data)` via `optim.LBFGS` (multi-start) over bounds.
4. Evaluate f at argmax, append, repeat.

Acquisition functions ship as plug-in closures: `ExpectedImprovement(μ, σ, f_best)` via `prob.NormalPDF`+`prob.NormalCDF` direct closed-form; `UpperConfidenceBound(μ, σ, β)` trivial; `ProbabilityOfImprovement(μ, σ, f_best)` via `prob.NormalCDF`; `ThompsonSampling` needs S14.PriorSamples. Crown-jewel composition: every BO ingredient ships in `prob/` (NormalPDF/CDF, S14 sampler) + `optim/` (LBFGS + multi-start) + `infogeo/` (RBF kernel) + `linalg/` (Cholesky). Zero new math.

---

## 3. Composition graph (DAG, by primitive dependency)

```
S8 LogPDF (gates Tier 2)
 ├── S9  MLEFit ──────────── (uses optim.LBFGS)
 ├── S15 BBVI ─────────────── (uses optim.GD + S14)
 ├── S16 ADVI ─────────────── (uses autodiff + reparam + S14)
 └── S17 SVGD ─────────────── (uses ∇logp + infogeo.RBFKernel)

S14 PriorSamples (Tier 2 only — adds RNG interface)
 └── S15, S17, S18

S5  EM-GMM ─── independent (Tier 1) ── 3/3 R-MUTUAL pin
S6  EM-Bern ── symmetric of S5

S1 Ridge ─── compose proximal.ProxSquaredL2 + FISTA
S2 Lasso ─── compose proximal.ProxL1 + FISTA  ── consumer-side promotion
S3 ElNet ─── 3-line variant of S1+S2
S4 NNLS  ─── compose proximal.ProxNonNeg + FBS

S7  MAPEstimate ─── pure smooth prior + LBFGS
S12 ProxMAP    ─── non-smooth prior, dispatch by type

S10 LogReg ─── prob.LogLoss + LBFGS + optional Tikhonov
S11 BayesLinReg ─── conjugate closed-form (linalg only)
S13 Softmax ─── needs S10 + LSE primitive

S18 BayesianOpt ─── needs S14 + ExpectedImprovement closed-form (compose
                     prob.NormalPDF + prob.NormalCDF) + RBF kernel + LBFGS
                     + linalg.Cholesky
```

---

## 4. Saturation pins this review unlocks

Per the existing R-pattern landscape (per recent commits 6a55bb4 audio-onset 3-detector saturation, 365368a Clayton autodiff-vs-analytic, 85a80db NGramDice — agent 163's `R-CLOSED-FORM-PINNED-TO-AUTODIFF` is at 3/3 STANDARD per `autodiff/doc.go`), this synergy lands:

- **R-MUTUAL-CROSS-VALIDATION 3/3 on EM-GMM (S5):** EM × `GeneticAlgorithm` global-search baseline × k-means-warm-started-EM agree to 1e-6 on Old Faithful means. Three orthogonal optimisation paths (closed-form-M-step alternation × stochastic-global × deterministic-warm-start) hitting the same fixed point is the canonical R-MUTUAL idiom.
- **R-CLOSED-FORM-PINNED-TO-OPTIMISATION 3/3 (proximal × prob ×optimisation):**
  1. Existing `optim/proximal_consumer_test.go:TestProximalLasso_FISTA_OrthogonalClosedForm` already pins prox-soft-threshold `β* = soft(y, λ)` in the orthogonal-design case.
  2. **S2 LassoRegression** non-orthogonal pin: convert (X, y, λ) → glmnet-validated `glmnet::cv.glmnet()` β reference vector embedded as test golden, agree to 1e-7.
  3. **S11 BayesianLinearRegression** vs closed-form ridge from S1 in σ₀² → ∞ limit: posterior mean → ridge β with effective `λ = σ²/σ₀²`; the limit must match S1 exactly. This single test promotes a brand-new prob×optim R-pattern to STANDARD.
- **R-CONJUGATE-DUAL pin:** S11 closed-form Bayesian-LinReg posterior mean × S1 ridge MAP × S7 LBFGS-on-(−nll−logGaussianPrior) all agree to 1e-9 on Boston housing — three derivation paths (analytic-conjugate / closed-form-prox / iterative-LBFGS) converge to one β, exactly the kind of identity reality is built to pin.

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| S1 | RidgeRegression | 80 | 1 | — |
| S2 | LassoRegression | 110 | 1 | — |
| S3 | ElasticNet | 80 | 1 | S1, S2 |
| S4 | NonNegativeLeastSquares | 70 | 1 | — |
| S5 | EM-GMM | 340 | 1 | — |
| S6 | EM-BernoulliMixture | 150 | 1 | — |
| S7 | MAPEstimate (smooth prior) | 90 | 1 | — |
| S8 | LogPDF interface | 80 | 1 | — (keystone) |
| S9 | MLEFit | 120 | 1+ | S8 |
| S10 | LogisticRegression | 140 | 1 | — |
| S11 | BayesianLinearRegression | 110 | 1 | — |
| S12 | ProximalGradientMAP | 90 | 1 | — |
| S13 | SoftmaxRegression | 150 | 1 | LogSumExp helper |
| S14 | PriorSamples (RNG-aware) | 120 | 2 | RNG interface |
| S15 | BlackBoxVI | 190 | 2 | S8, S14 |
| S16 | ADVI | 210 | 2 | S8, S14, autodiff (forward-mode optional) |
| S17 | SVGD | 180 | 2 | S8, infogeo.RBFKernel |
| S18 | BayesianOpt | 280 | 2 | S14, infogeo.RBFKernel, linalg.Cholesky |
| **Σ** | | **2,650** | | |

Pure-glue ratio: ~85 % of LOC is composition; ~15 % is genuinely-new math (S8 LogPDFs for Beta/Gamma already pre-exist internally; S14 Marsaglia-Tsang Gamma sampler at ~50 LOC and Cheng Beta sampler at ~30 LOC are the only non-trivial NEW-math fragments).

---

## 6. Recommended PR sequence

**PR-1: S1 + S2 + S3 + S4 + cross-package consumer test (~340 LOC source, ~140 LOC tests, half-day)**
First non-test consumer of `optim/proximal/` from outside its own package. Lands four flagship regularised-regression APIs (`prob.RidgeRegression`, `prob.LassoRegression`, `prob.ElasticNet`, `prob.NNLS`) each composing existing `optim/proximal.FBS` or `FISTA`. Saturates R-CLOSED-FORM-PINNED-TO-OPTIMISATION 2/3 (orthogonal-LASSO + S1=conjugate-Bayes-σ₀²→∞ limit + S2=glmnet-golden). **Cheapest first PR with maximum proximal/ adoption signal** — `optim/proximal/doc.go:40-49` explicitly names this as the queued first-consumer push.

**PR-2: S8 LogPDF + S9 MLEFit + S10 LogisticRegression + S11 BayesianLinearRegression + S12 ProxMAP (~530 LOC, two days)**
Lands the `LogPDF` interface extension across 7 dists, the `MLEFit` dispatcher with closed-form per-distribution, logistic regression via LBFGS, conjugate Bayesian linear regression, and the prox-MAP generalisation of S7. Saturates the R-CONJUGATE-DUAL pin (S1=S7=S11 on Boston housing). Architectural placement: all primitives ship in `prob/`, all consume `optim/{proximal/}` — verifies one-way import direction `prob/ → optim/` (mirroring 163-A6 `optim/ → autodiff/` consumer-side rule).

**PR-3: S5 EM-GMM + S6 EM-Bernoulli + S7 MAP-smooth-prior (~580 LOC, two days)**
Lands the EM canon. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 on Old Faithful (EM × GA-baseline × k-means-warm-start agreeing on means to 1e-6). Architectural placement: new file `prob/em.go` (closed-form alternating updates inline; only S7 imports `optim/`). Per CLAUDE.md "Golden files are the proof", the Old Faithful means become a 50-vector golden file under `prob/testdata/em_gmm/`.

**PR-4: S14 PriorSamples + RNG interface (~140 LOC, half-day)**
The first prob/ function family that takes an RNG. Adapts `*math/rand.Rand` via single-line `Float64()` interface matching `optim.GeneticAlgorithm`'s. Lands inverse-CDF sampling for Normal/Exp/Uniform (one-liners over existing `*Quantile`s) plus Marsaglia-Tsang-2000 Gamma + Cheng-1978 Beta + Knuth-1969 Poisson. **Tier-2 gateway PR.**

**PR-5: S15 BBVI + S17 SVGD (~370 LOC, two days)**
Lands the two variational methods that don't require new autodiff infrastructure. SVGD consumes `infogeo.RBFKernel` (already shipped per `infogeo/mmd.go`) — first-consumer push for that kernel from outside infogeo.

**PR-6: S18 BayesianOpt + EI/UCB/PI acquisition (~330 LOC, two days)**
Crown-jewel synergy: composes prob/ + optim/ + infogeo/ + linalg/ in a single algorithm. Acquisition functions are 4-line closures over `prob.NormalPDF`+`prob.NormalCDF`. Saturates a 4-component-package composition pin.

**PR-7: S16 ADVI (~210 LOC, one day)** — deferred until autodiff/dual.go (163-A1) lands; then trivial.

**PR-8: S13 SoftmaxRegression (~160 LOC including LSE primitive, half-day)** — orthogonal to all of the above; can land any time.

Total: ~2,650 LOC source + ~900 LOC tests across 7 PRs over ~9 engineer-days. PR-1 is single-day standalone with zero new infrastructure; PR-3 (EM-GMM) is the single highest-value primitive not requiring any new abstraction.

---

## 7. Cycle-hazard analysis

Proposed import directions:

```
prob/  ──→  optim/         (S1-S4, S7, S9, S10, S12, S13, S15-S18)
prob/  ──→  optim/proximal (S1-S4, S12)
prob/  ──→  infogeo/       (S17, S18)
prob/  ──→  linalg/        (S5, S11, S18) — already exists for prob/copula/
prob/  ──→  autodiff/      (S16) — already exists for prob/copula/autodiff_test.go
```

`optim/` does NOT need to import `prob/` for any primitive in S1-S18. The `Sampleable` interface in S14 is consumer-side (optim/ already accepts `interface{ Float64() float64 }`); `optim.GeneticAlgorithm` and `optim.SimulatedAnnealing` continue to need only `*math/rand.Rand`. **DAG remains cycle-free.** Verified by enumeration: every primitive lives in exactly one direction.

Existing edges already in the codebase (per agent 163, agent 153, autodiff/doc.go):
- `prob/copula → autodiff/` (Clayton parity test)
- `prob/copula → linalg/` (Gaussian copula needs Cholesky)
- `infogeo → autodiff/` (KL gradient parity test)
- `timeseries/garch → autodiff/`

This synergy adds `prob → optim/`, `prob → optim/proximal/`, `prob → infogeo/` — three new edges in the **same direction as existing precedent** (toward more-foundational packages). No reverse edges.

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **S1 Ridge:** condition number of `(XᵀX + λI)` lower-bounded by `λ`; pick `λ ≥ 1e-8 ||X||²_F / n` for stable Cholesky. Below that, fall back to QR.
- **S2 Lasso:** FISTA convergence rate O(1/k²) requires Lipschitz `L = ||X||²_2`; estimate via 5-iter power iteration on `XᵀX` and use step `1/L`. Per `optim/proximal/fbs.go`, FBS already accepts user-supplied step — this primitive picks it.
- **S5 EM-GMM:** covariance singularity (`Σ_k → 0`) when one Gaussian collapses on a single point — symptom of Wu-1983 known degeneracy; mitigation is regularisation `Σ_k ← Σ_k + εI` with `ε = 1e-6 trace(cov(X))`. Document in golden as expected behaviour.
- **S7 MAP:** if `logPrior` is concave (e.g., Laplace prior with non-zero centre), `−nll − logPrior` may have multiple local minima; document that LBFGS finds *a* MAP, not necessarily *the* MAP. Multi-start recommended in test golden.
- **S11 BayesLinReg:** for `σ₀² → ∞` limit, posterior covariance `Σ_n` becomes singular if `n < d`. Return wide CI (3σ heuristic) or fall through to ridge S1.
- **S15 BBVI:** score-function gradient variance scales as O(1/N_mc); document `mcSamples ≥ 64` as the floor. Reparameterised gradient (S16 ADVI) is strictly lower-variance.
- **S17 SVGD:** RBF bandwidth via Liu-Wang 2016 median heuristic `h = med²/log(n)` — already supplied by `infogeo.MedianHeuristicBandwidth` (verified by grep); document this composition rather than re-deriving.
- **S18 BO with EI:** Mockus 1978 EI closed form `(μ−f*)Φ(z) + σφ(z)` with `z = (μ−f*)/σ` — guard against `σ → 0` (degenerate at observed points); set `EI = 0` when `σ < 1e-9`.

---

## 9. Distinct from prior agents (provenance)

- **011-015 autodiff isolation** — 012 names "stochastic AD (reparam + score-function) Tier-3" but does NOT compose with prob/distributions; this synergy is the consumer-side pull through prob/ that justifies 012-T3.
- **101-105 optim isolation** — 101 names CMA-ES / particle-swarm / SQP / Newton-CG gaps; this synergy assumes those gaps remain and shows BBVI/SVGD/BO compose without them.
- **116-120 prob isolation** — 117 names `RNG`/`LogPDF`/`Sample` debt; this synergy *consumes* that debt and converts it into 18 cross-package primitives.
- **151 synergy-signal-prob** — sibling architecture review; orthogonal to fitting/inference.
- **153 synergy-prob-infogeo** — names `S1 ClosedFormKL` + `S2 FisherFromDistribution`; THIS review depends on 153-S2 for natural-gradient extension of S15 BBVI but does NOT duplicate; instead this synergy adds the **fit/inference** axis to 153's geometric axis.
- **155 synergy-crypto-prob** — orthogonal (cryptographic randomness, not statistical inference).
- **161 synergy-control-prob** — orthogonal (Lyapunov function learning).
- **162 synergy-graph-prob** — orthogonal (random walks on graphs).
- **163 synergy-optim-autodiff** — closest sibling; 163 names `A12 Adam`, `A18 SVRG` as missing optimisers — this synergy USES `optim.LBFGS` (which exists) and would benefit from Adam (which doesn't); **the two reviews compose: 163 ships the second-order infrastructure, 169 ships the first-order statistical applications**. Crucially this review does NOT duplicate 163's optim/autodiff plumbing; it consumes the existing `optim.LBFGS` and adds reparameterisation glue for ADVI on top.
- **164 synergy-orbital-optim** — uses `optim.SimplexMethod` for transfer optimisation; sibling consumer-side push.
- **165 synergy-sequence-prob** — orthogonal (string distances, not parameter fitting).
- **168 synergy-physics-autodiff** — orthogonal (Lagrangian/Hamiltonian/symplectic); the only intersection is "if 168-A1 dual numbers land first, 169-S16 ADVI gets exact reparameterised gradient via JVP".

---

## 10. Bottom line

`prob/` and `optim/` are siblings under `reality/` with **literally zero current edges** in either direction (verified grep) despite being the two most-natural consumers of each other in the entire repo. Eighteen synergy primitives totalling ~2,650 LOC of pure connective tissue stand up the entire fit-distribution-to-data + variational-inference + Bayesian-optimisation canon on existing v0.10.0 surfaces. Eleven (S1-S4, S6-S7, S10-S13) ship today against zero new infrastructure; four (S5, S9, S14-S18) need the `LogPDF` interface (~80 LOC) plus the `RNG` interface (~30 LOC) — both pure-additions. The cheapest single-day standalone is **PR-1 (S1+S2+S3+S4 = 340 LOC)** which lands the queued first-consumer push for `optim/proximal/` named in `optim/proximal/doc.go:40-49`. The single highest-value primitive not requiring any new abstraction is **PR-3 (S5 EM-GMM = 340 LOC)** which saturates a clean R-MUTUAL-CROSS-VALIDATION 3/3 pin (EM × GA-baseline × k-means-warm-start) on the canonical Old Faithful benchmark. The crown-jewel synergy is **PR-6 (S18 BayesianOpt = 280 LOC)** which composes prob/ + optim/ + infogeo/ + linalg/ in a single algorithm and is the first four-package composition in the repo.

Reality is unusually well-positioned for this synergy because (i) `optim/proximal/` already has the prox operators every regularised-regression primitive needs, (ii) `prob/` already has stable log-space PDF computations one negation away from public LogPDF, (iii) `infogeo.RBFKernel` already ships and SVGD needs nothing more, (iv) `linalg.SolveCholesky` already covers the GP path of S18, (v) `optim.GeneticAlgorithm`'s RNG interface design is the precedent for `prob.Sample(rng)`, and (vi) the consumer-side-placement rule (per agents 158/159/160/166/167/168) recommends placing all S1-S18 in `prob/` rather than a new sub-package — minimum architectural perturbation, maximum statistical-inference unlock.
