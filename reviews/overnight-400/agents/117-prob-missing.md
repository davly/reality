# 117 | prob-missing — canonical primitives absent from `prob/`

Scope: enumerate distributions / copulas / processes / Bayesian objects / samplers / tests / IC absent from `C:\limitless\foundation\reality\prob\` and `prob/copula/`. 116 catalogued the numerical bugs; this report catalogues the **API surface gaps**. Web crosswalk: scipy.stats v1.13, statsmodels 0.14, Distributions.jl 0.25, R `stats` 4.4 + `actuar` 3.3, `copula` 1.1.4, `hawkes` 0.0.5, `KFAS` 1.5.

---

## Roster present today

**Top-level `prob/distributions.go`:** Normal {PDF, CDF, Quantile}, Exponential {PDF, CDF, Quantile}, Uniform {PDF, CDF}, Beta {PDF, CDF}, Poisson {PMF, CDF}, Gamma {PDF, CDF}, Binomial {PMF, CDF}.
**`prob/distribution.go`:** Distribution interface + NewBetaDist / NewNormalDist / NewExponentialDist / NewUniformDist + KLDivergenceNumerical.
**`prob/mathutil.go`:** RegularizedBetaInc, regularizedGammaLowerSeries, unexported `studentTCDF`, `chiSquaredCDF`.
**`prob/hypothesis.go`:** TTestOneSample, TTestTwoSample (Welch), ChiSquaredTest.
**`prob/nonparametric.go`:** FisherExactTest, MannWhitneyU.
**`prob/regression.go`:** LinearRegression, BenjaminiHochberg.
**`prob/timeseries.go`:** ExponentialSmoothing, HoltLinear, ARIMA.
**`prob/markov.go`:** MarkovSteadyState, MarkovSimulate.
**`prob/jeffreys.go`:** JeffreysConfidence, ThreeWayVerdict, EMA, JeffreysKLDivergence, QualityWeightedDominance.
**`prob/prob.go`:** ClampProbability, ConfidenceFromPValue, ProbToLogOdds, LogOddsToProb, BayesianUpdate{,Chain}, BrierScore{,Batch}, LogLoss{,Batch}, LogOddsPool, WilsonConfidenceInterval, SimpleAverage, WeightedAverage, Median, TrimmedMean, ECE, MCE, ReliabilityDiagram, IsotonicRegression.
**`prob/copula/`:** Clayton/Gumbel CDF + h-fn + PDF + LogPDF, GaussianCopulaCDF (n-D via Bivariate/Trivariate ≤3), StudentTCopulaCDF, StudentT{CDF,Quantile} (duplicated from `prob/`), KendallTau, Sklar, D-vine.
**`prob/conformal/`:** split / adaptive / mondrian conformal prediction.

**Roster gap-set delta vs 116:** 116 already named missing Student-t public, F, chi-squared PDF/quantile, NegBin, geometric, multinomial, Dirichlet, general n-D MVN, Box-Muller, Marsaglia ziggurat, erfcinv, logsumexp, expm1. This report is **everything else** and ranks by canonical-library prevalence (scipy.stats / Distributions.jl / R `stats`).

---

## Tier 1 — Must-ship (table-stakes; every canonical library has them)

### T1.1 Discrete distribution backbone (~600 LOC)

Beyond 116's named misses (NegBin, geometric, multinomial, categorical, Dirichlet):

- **Hypergeometric** (PMF/CDF/Quantile) — sampling-without-replacement, exact small-sample tests, Fisher-exact's continuous parent. scipy `hypergeom`, R `dhyper`. Closed-form via `comb(K,k)·comb(N−K,n−k)/comb(N,n)`; lgamma path for log-PMF. ~80 LOC.
- **Skellam** (PMF/CDF) — difference of two Poissons, `S = X − Y`, X∼Poi(μ₁), Y∼Poi(μ₂); PMF involves modified Bessel `I_|k|(2√(μ₁μ₂))`. Used in sports analytics, queueing differences. scipy `skellam`. ~60 LOC + a Bessel I primitive (modified Bessel functions are missing repo-wide; coordinate with future `signal/special` or stand up here).
- **Beta-binomial** (PMF/CDF) — over-dispersed binomial with Beta-distributed `p`. PMF = `C(n,k) · B(k+α, n−k+β) / B(α,β)`. Bayesian conjugate posterior predictive for Bernoulli. R `dbbinom` (extraDistr). ~50 LOC.
- **Zero-inflated Poisson (ZIP) / NB (ZINB)** — mixture `π·δ_0 + (1−π)·Pois(λ)` / NB. Standard for count regression with excess zeros (insurance claims, ecology). statsmodels `ZeroInflatedPoisson`. ~60 LOC each.
- **Discrete uniform** (PMF/CDF/Quantile) — scipy `randint`. ~30 LOC. (Currently only continuous Uniform exists.)
- **Bernoulli** as standalone (PMF/CDF/Sampler) — degenerate of Binomial(n=1) but every library exposes it standalone for Bayesian logistic / categorical aliasing. ~30 LOC.
- **Zipf / power-law / Yule-Simon** — `P(k) ∝ k^{−s}` with cutoff. Citation-network / Zipf's-law / heavy-tail regression. scipy `zipf`, R `VGAM::dzipf`. ~70 LOC. Useful baseline for `gametheory`/`graph` consumers.
- **Logarithmic series** — Poisson-stopped-sum primitive, compounds with Poisson to give NB. scipy `logser`. ~40 LOC.
- **Conway-Maxwell-Poisson (CMP)** — 2-parameter generalization of Poisson allowing under/over-dispersion. R `compoisson`. ~80 LOC. Lower-priority but widely cited 2024-era count model.

### T1.2 Continuous distribution backbone (~900 LOC)

**Heavy-tail / extreme-value** (these are the practical 2025-era bread-and-butter for risk; Pistachio aicore reasoning needs them):

- **Pareto** (PDF/CDF/Quantile/LogPDF) — type-I (Pareto distribution). Power-law tails. scipy `pareto`, R `actuar::dpareto`. ~50 LOC.
- **Generalized Pareto (GPD)** — Peaks-Over-Threshold tail for extremes; the second leg (after GEV) of the Fisher-Tippett tail-equivalence theorem. scipy `genpareto`. ~60 LOC.
- **Lognormal** (PDF/CDF/Quantile/LogPDF) — log of normal; multiplicative noise / financial price. scipy `lognorm`. ~40 LOC. Trivial via NormalCDF.
- **Weibull** (PDF/CDF/Quantile/LogPDF) — reliability / survival / wind-speed. scipy `weibull_min`. ~50 LOC.
- **Gumbel / Fréchet / GEV** (extreme-value type I/II/III) — block-maxima, Fisher-Tippett. scipy `gumbel_r`, `gumbel_l`, `genextreme`. ~120 LOC.
- **Cauchy** (PDF/CDF/Quantile/LogPDF) — heavy-tail Student-t(df=1); robust statistics; MCMC proposals. scipy `cauchy`. ~30 LOC. Note no moments — important docstring cite.
- **Logistic** (PDF/CDF/Quantile/LogPDF) — logit's parent; sigmoid `1/(1+e^{-x})` is its CDF. Used everywhere in ML calibration; reality currently exposes `LogOddsToProb` but not the underlying Logistic distribution. scipy `logistic`. ~30 LOC.
- **Half-Normal / Half-Cauchy / Half-t** — folded distributions; standard weakly-informative priors for scale parameters in Bayesian models (Gelman 2006). scipy `halfnorm`, `halfcauchy`. ~40 LOC each.
- **Truncated Normal / Truncated t** — common likelihood for censored data and rejection-sampling envelopes. scipy `truncnorm`, `truncpareto`. ~80 LOC. **Inverse-CDF sampler is non-trivial:** must compute `Φ⁻¹(Φ(a) + u·(Φ(b)−Φ(a)))` with tail-stable arithmetic for `a` deep in the tail (use erfcinv path, not Acklam→1.15e-9).
- **Inverse-Gamma** (PDF/CDF/Quantile/LogPDF) — conjugate prior for Normal-known-mean variance. `1/X` where `X ~ Gamma`. scipy `invgamma`, R `MCMCpack::dinvgamma`. ~50 LOC. Linchpin of T3.1.
- **Chi (not chi-squared) / Rayleigh / Maxwell-Boltzmann** — radial Gaussian magnitudes. scipy `chi`, `rayleigh`, `maxwell`. ~40 LOC each.
- **Generalized Gaussian / Exponential power** (β-stable density `e^{-|x|^β}`) — Laplace and Normal as β=1, β=2. scipy `gennorm`. ~40 LOC.
- **Laplace** (PDF/CDF/Quantile/LogPDF) — double-exponential; LASSO MAP prior; robust regression. scipy `laplace`. ~30 LOC.
- **Triangular** — minimum-information distribution from (min, mode, max). scipy `triang`. ~30 LOC.

### T1.3 Sampling primitives (~400 LOC)

Today **zero** samplers exist anywhere in `prob/`. Every distribution should expose `XxxSample(rng)` and `XxxSampleN(rng, n, out)` (caller-supplied buffer per CLAUDE.md rule 3). Foundations needed first:

- **Inverse-CDF sampler** generic helper (`InvCDFSample(quantile, rng)`) — works for any distribution exposing a Quantile.
- **Box-Muller** (polar / classic) for Normal — closed form, two-at-a-time. ~30 LOC.
- **Marsaglia ziggurat** for Normal & Exponential — ~7× faster than Box-Muller for batch sampling. ~150 LOC + 256-row precomputed table.
- **Rejection sampling** (`RejectionSample(target, envelope, M, rng)`) — generic.
- **Adaptive Rejection Sampling (ARS)** — Gilks-Wild 1992 for log-concave densities. Standard MCMC building-block. ~120 LOC.
- **Importance sampling** with effective-sample-size diagnostic — `ImportanceSample(target, proposal, n, rng) → (samples, weights, ESS)`. ~80 LOC.
- **Gamma sampler** — Marsaglia-Tsang squeeze 2000 (the canonical algorithm; ~50 LOC). Underlies Beta, Dirichlet, Chi-squared, Inverse-Gamma, Wishart samplers — none of these can be sampled until Gamma sampler exists.
- **Beta sampler** — two Gamma samples ratio. ~10 LOC once Gamma exists.
- **Dirichlet sampler** — k Gamma samples normalised. ~15 LOC.
- **Categorical/Multinomial sampler** — Walker's alias method (O(1) per draw after O(k) setup). ~80 LOC.
- **Poisson sampler** — Knuth small-λ + PTRS large-λ (Hörmann 1993). ~70 LOC.
- **Binomial sampler** — BTPE (Kachitvichyanukul-Schmeiser 1988). ~80 LOC.
- **Student-t sampler** — Bailey 1994 (Normal/√(χ²/ν)). ~20 LOC once Gamma sampler exists.
- **Stable-distribution sampler** — Chambers-Mallows-Stuck 1976 (CMS) for α-stable family. ~50 LOC. The **only** tractable path to Lévy stable simulation; without this T2.4 below is a paper exercise.

### T1.4 erfcinv + logsumexp + log1mexp + log1pexp (~100 LOC)

Core building blocks that should be in `prob/mathutil.go`:

- **`Erfcinv(x)`** — Blair-Edwards-Johnson 1976 rational approximation, full-double accuracy. Required for tail-stable Normal quantile, truncated-normal sampling, t-distribution quantile bracket. scipy `erfcinv`. ~80 LOC.
- **`LogSumExp(xs ...float64)`** + `LogSumExp2(a,b)` — `m + log(Σ exp(xᵢ-m))` Kahan-shifted. Per 116 LOW.A finding, currently absent everywhere; Archimedean copula PDF, all mixture densities, and every future MCMC log-target need it. ~30 LOC.
- **`Log1mExp(x)`** — `log(1 - exp(x))` for `x ≤ 0`. Mächler 2012 dispatch on `x > -log 2 ? log(-expm1(x)) : log1p(-exp(x))`. ~15 LOC.
- **`Log1pExp(x)`** — `log(1 + exp(x))` = softplus, branched on `x` to avoid overflow. ~10 LOC.
- **`Logaddexp` / `Logsubexp`** — sibling primitives. ~15 LOC.

These five primitives unblock log-space everywhere: log-likelihood evaluation, mixture-model EM, HMM forward-backward in log-space, MCMC log-targets, copula log-PDFs.

### T1.5 Hypothesis tests beyond t/χ²/Mann-Whitney/Fisher (~500 LOC)

- **Kolmogorov-Smirnov** one-sample + two-sample with exact small-n + asymptotic tail. scipy `kstest`, `ks_2samp`. ~150 LOC. Surprising omission — KS is the textbook GoF test alongside χ². The KS distribution itself (Kolmogorov dist) should be exposed alongside.
- **Anderson-Darling** — weighted KS giving more power in tails. Critical for normality testing. scipy `anderson`, `anderson_ksamp`. ~120 LOC.
- **Cramér-von Mises** — alternative L²-distance GoF. scipy `cramervonmises`. ~80 LOC.
- **Shapiro-Wilk** — best small-sample normality test. R `shapiro.test`. ~150 LOC (uses Royston 1995 polynomial coefficients for `n ≤ 5000`).
- **Jarque-Bera** — moment-based normality. scipy `jarque_bera`. ~30 LOC.
- **Levene / Bartlett / Fligner-Killeen** — equal-variance tests. scipy `levene`, `bartlett`, `fligner`. ~150 LOC.
- **Kruskal-Wallis** — non-parametric ANOVA. scipy `kruskal`. ~80 LOC.
- **Friedman / Wilcoxon signed-rank** — paired non-parametric. scipy `wilcoxon`, `friedmanchisquare`. ~100 LOC.
- **Permutation test** generic helper. scipy `permutation_test`. ~80 LOC.
- **Bootstrap** generic — percentile / BCa / studentized intervals. scipy `bootstrap`. ~150 LOC.
- **F-test for variance ratio** — once F-distribution lands per 116. ~30 LOC.
- **One-way / two-way ANOVA** — depends on F-distribution. scipy `f_oneway`. ~150 LOC.
- **Welch's ANOVA** — heteroscedastic ANOVA. ~80 LOC.

### T1.6 Information criteria (~80 LOC)

- **AIC** = `2k − 2 ln L` ; **BIC** = `k ln n − 2 ln L` ; **AICc** small-sample correction ; **DIC** Bayesian deviance ; **WAIC** Watanabe-Akaike (requires per-observation log-likelihoods). statsmodels exposes all five. ~80 LOC trivial wrappers but they're the model-selection lingua franca.
- **Cross-entropy / KL / JS / Hellinger / TV / Wasserstein-1** between empirical samples — currently only `KLDivergenceNumerical` exists. ~150 LOC.

---

## Tier 2 — Should-ship (frontier-relevant; one or two libraries have them)

### T2.1 Multivariate roster beyond 116 named (~700 LOC)

- **Multivariate Student-t** (PDF/CDF/Quantile/Sampler) — Pistachio's robust-to-outlier MVN replacement. scipy `multivariate_t`. ~120 LOC. CDF via Genz-Bretz QMC (algorithm `mvtdst`).
- **Wishart** + **Inverse-Wishart** (PDF/Sampler) — covariance-matrix priors. Bartlett decomposition for sampling (Smith-Hocking 1972). scipy `wishart`, `invwishart`. ~150 LOC.
- **LKJ correlation distribution** — Lewandowski-Kurowicka-Joe 2009 prior over correlation matrices; Stan / PyMC / brms standard. Vine-based onion construction. ~100 LOC. Goes naturally next to `prob/copula/vine.go`.
- **Matrix-Normal** + **Matrix-t** — matrix-variate analogs of MVN/MVT for vec-Kronecker covariance. statsmodels `matrix_normal`. ~80 LOC.
- **Multinomial-Dirichlet (Pólya / DCM)** — compound Dirichlet-Multinomial used in topic models, language modeling. ~60 LOC.
- **Generalized inverse Gaussian (GIG)** — three-parameter (λ,χ,ψ) on R⁺; mixing distribution for Generalized Hyperbolic family. Sampling via Hörmann-Leydold 2014 ratio-of-uniforms. R `GIGrvg`. ~150 LOC.
- **Generalized Hyperbolic (GHyp)** — full 5-parameter family covering NIG, Variance-Gamma, hyperbolic, Student-t, Laplace, Cauchy, Normal as special / limiting cases. Density involves modified Bessel `K_λ`. R `ghyp`, `fBasics::dghyp`. ~200 LOC + Bessel K primitive.
- **Normal-Inverse-Gaussian (NIG)** — special case of GHyp with `λ = -1/2`; standard heavy-tail asset-return model (Barndorff-Nielsen 1997). R `fBasics::dnig`. ~80 LOC once GIG/Bessel exist.
- **Variance-Gamma (VG)** — special case of GHyp; Madan-Carr-Chang option pricing model. ~80 LOC.
- **Tweedie** family — exponential dispersion family `EDM(μ, φ, p)` covering Normal (p=0), Poisson (p=1), Gamma (p=2), Inverse-Gaussian (p=3), compound-Poisson-Gamma (1<p<2). Standard for insurance pure-premium GLMs. R `tweedie`, `statmod::dtweedie`. ~250 LOC (density via Dunn-Smyth 2005 series-vs-Fourier dispatch — non-trivial; density evaluation is the hard part).
- **Lévy stable α-stable** family — 4-parameter (α, β, μ, σ) family closed under sums; α=2 Normal, α=1, β=0 Cauchy, α=1/2, β=1 Lévy. PDF via Nolan 1997 Zolotarev integral; CDF likewise. Sampling via CMS-1976 (T1.3 above). R `stabledist`. ~300 LOC PDF/CDF + ~50 LOC sampler.
- **Lévy distribution** (one-sided stable, α=1/2) — heavy-tail unbounded right-tail; reflected-Brownian first-passage. scipy `levy`. ~30 LOC.
- **Inverse-Gaussian (Wald)** — first-passage time of Brownian-with-drift; popular insurance / reaction-time model. scipy `invgauss`. ~50 LOC.

### T2.2 Copula family completion (~600 LOC)

Today: Clayton, Gumbel, Gaussian, Student-t, D-vine. Missing per MASTER_PLAN line 117:

- **Frank** Archimedean — symmetric tail-independence, only Archimedean spanning all dependence (τ ∈ [-1, 1]). Generator `−log((e^{-θu} − 1)/(e^{-θ} − 1))`. ~80 LOC fits next to `archimedean.go`.
- **Joe** Archimedean — upper-tail-dependent only, no lower-tail. Generator `−log(1 − (1−u)^θ)`. ~80 LOC.
- **Ali-Mikhail-Haq (AMH)** Archimedean — bounded `τ ∈ [-0.18, 1/3]`; light-tail. Generator `log((1 − θ(1−u))/u)`. ~70 LOC.
- **BB1, BB6, BB7, BB8** two-parameter Archimedean — flexible tail-asymmetry; standard in financial copula modeling (Joe 1997). ~250 LOC.
- **Plackett** — single-parameter, useful for ordinal-data copula. ~60 LOC.
- **Galambos / Hüsler-Reiss / Tawn** extreme-value copulas — pair-copula counterparts to GEV margins. ~150 LOC.
- **Empirical / Bernstein / Beta copula** — non-parametric copula construction from ranks. ~100 LOC.
- **Vine completion** — `prob/copula/vine.go` has D-vine; missing **C-vine** (canonical) and **R-vine** (regular) constructions. Aas-Czado-Frigessi-Bakken 2009 paper covers all three; D-vine is the linear case, C-vine the star, R-vine the general tree-of-trees structure. R `VineCopula` is the reference. ~250 LOC including R-vine matrix encoding (Dißmann-Brechmann-Czado-Kurowicka 2013).
- **Hierarchical / Nested Archimedean (HAC)** copulas — Joe 1997, Hofert 2008. R `HAC`. ~150 LOC.

### T2.3 Stochastic processes (none today; ~1500 LOC)

`prob/markov.go` has discrete-time Markov chains only. Continuous-time / point-process / SDE families entirely absent:

- **Homogeneous Poisson process** — `Sample(rate, T)` returning event times via exponential inter-arrivals. ~30 LOC.
- **Inhomogeneous Poisson process** — time-varying intensity λ(t); thinning algorithm (Lewis-Shedler 1979). ~60 LOC.
- **Cox process (doubly-stochastic Poisson)** — Poisson with random intensity drawn from a separate stochastic process. ~80 LOC composing inhomogeneous-Poisson + arbitrary intensity sampler.
- **Hawkes self-exciting process** — λ(t) = μ + Σ φ(t−tᵢ) with exponential-kernel `φ(s) = α e^{-β s}` (Ogata 1981 thinning). Simulation, log-likelihood, MLE. R `hawkes`, Python `tick`. ~250 LOC for univariate exp-kernel; multivariate Hawkes adds matrix `α_{ij}, β_{ij}` and is ~150 LOC more.
- **Compound Poisson** — `S = Σ_{i=1}^{N} X_i` with N ~ Poisson, Xᵢ iid jump distribution. Insurance aggregate-claims standard. ~60 LOC.
- **Brownian motion** — sample-path simulation via cumulative iid Normal increments; supports general drift μ and diffusion σ. ~50 LOC.
- **Geometric Brownian motion** — Black-Scholes asset price `S(t) = S₀ exp((μ − σ²/2)t + σW(t))`. ~30 LOC.
- **Fractional Brownian motion (fBm)** — Hurst-parameterised long-range-dependent self-similar process. Cholesky-based exact sim or Davies-Harte FFT-based fast sim (O(n log n)). ~150 LOC.
- **Ornstein-Uhlenbeck** — mean-reverting Gaussian SDE `dX = θ(μ−X)dt + σdW`. Exact-discretization sampler (no Euler error). ~40 LOC.
- **Cox-Ingersoll-Ross (CIR)** — square-root mean-reverting `dr = κ(θ − r)dt + σ√r dW`. Non-central-chi-squared exact sampler (Glasserman 2003). ~80 LOC. Term-structure / vol modeling consumer.
- **Vasicek** — Gaussian short-rate. ~30 LOC, sibling of OU.
- **Geometric / arithmetic Lévy processes** — jump-diffusion `dX = μdt + σdW + dJ` with J a compound Poisson or pure-jump α-stable. ~120 LOC composing T2.1 stable + T2.3 compound-Poisson.
- **Bridge sampling** for Brownian / OU / CIR — conditional simulation given start/end (linchpin of MCMC for SDEs). ~80 LOC.
- **Euler-Maruyama / Milstein / Runge-Kutta-Maruyama** generic SDE integrators. ~150 LOC.

### T2.4 Bayesian / non-parametric process priors (~700 LOC)

- **Beta-Binomial conjugate** — `posterior = Beta(α + s, β + n − s)`. ~20 LOC. Standard textbook example.
- **Normal-Inverse-Gamma (NIG)** conjugate for Normal-unknown-mean-and-variance — `posterior = NIG(μ', λ', α', β')` with closed-form updates (Murphy 2007 cheat-sheet). ~80 LOC. Linchpin of `prob/changepoint/bocpd.go` in sibling package; today BOCPD reimplements this.
- **Normal-Wishart / Normal-Inverse-Wishart** — multivariate analog. ~80 LOC.
- **Dirichlet-Multinomial / Dirichlet-Categorical** conjugate. ~30 LOC.
- **Gamma-Poisson** conjugate (Gamma prior, Poisson likelihood, NB posterior predictive). ~30 LOC.
- **Beta-Geometric / Beta-NegBin** conjugates. ~40 LOC.
- **Conjugate-prior catalog** — single registry mapping `(likelihood family, parameter) → posterior update closure`. ~150 LOC. Murphy 2007 lists 14 standard pairs.
- **Dirichlet Process (DP)** — stick-breaking / Chinese-restaurant-process / Pólya-urn representations. Basis of every non-parametric Bayesian model. Sethuraman 1994. ~150 LOC.
- **Pitman-Yor process (PYP)** — two-parameter generalization of DP with power-law cluster sizes. Standard for language models. ~100 LOC.
- **Indian Buffet Process (IBP)** — non-parametric prior over binary feature matrices. Griffiths-Ghahramani 2011. ~100 LOC.
- **Beta Process** + **Gamma Process** — completely-random-measure Bayesian non-parametric primitives. ~150 LOC.

### T2.5 MCMC samplers (~800 LOC)

Today: zero MCMC infrastructure. Even basic Metropolis-Hastings absent. (116 confirmed.)

- **Metropolis-Hastings (MH)** — generic with arbitrary proposal kernel + acceptance-rate diagnostic + thinning + burn-in. ~120 LOC.
- **Random-walk MH** — Gaussian proposal with adaptive step-size (Roberts-Rosenthal 2009 optimal-scaling 0.234). ~80 LOC.
- **Gibbs sampler** — coordinate-wise update with user-supplied conditional samplers. ~60 LOC scaffold.
- **Slice sampler** — Neal 2003 univariate + step-out / shrink-in. ~120 LOC.
- **Hamiltonian Monte Carlo (HMC)** — Duane-Kennedy-Pendleton-Roweth 1987; uses leapfrog integrator + Metropolis correction. Requires gradient of log-target — composable with `autodiff`. ~150 LOC.
- **No-U-Turn-Sampler (NUTS)** — Hoffman-Gelman 2014; auto-tunes HMC trajectory length via doubling-and-stopping rule. ~250 LOC. Stan / PyMC default.
- **Affine-invariant ensemble (emcee)** — Goodman-Weare 2010 / Foreman-Mackey 2013. Gradient-free, no tuning. ~120 LOC.
- **Parallel-tempering / Replica exchange** — multi-chain temperature ladder for multimodal targets. ~100 LOC composing MH.
- **Sequential Monte Carlo (SMC)** — Del Moral-Doucet-Jasra 2006 particle-filter for static-parameter inference + Liu-Chen 1998 particle-filter for state-space; resampling (multinomial / systematic / stratified) primitives. ~250 LOC.
- **Convergence diagnostics** — R̂ (Gelman-Rubin), ESS (effective sample size, Geyer 1992 IAT or autocorrelation-based), Heidelberger-Welch, Geweke. ~150 LOC.

### T2.6 Numerical integration of densities — quadrature for CDFs / moments (~200 LOC)

`prob/distribution.go` has `KLDivergenceNumerical` via what looks like trapezoidal. Real generic CDF/moment computation needs adaptive Gauss-Kronrod (017's Tier 1 in `calculus`). Cross-package coupling: once `calculus.AdaptiveGaussKronrod` lands, expose `prob.CDFFromPDF(pdf, x)`, `prob.MomentNumerical(pdf, n)` as one-liners. Today every distribution hand-derives its CDF; many of T1.2/T2.1 above (NIG, GHyp, Variance-Gamma, GIG, Stable) have no closed-form CDF and **must** route through quadrature.

---

## Tier 3 — Nice-to-have (frontier; specialty consumers)

### T3.1 ABC / Likelihood-free inference (~300 LOC)
Pritchard 1999 / Beaumont-Zhang-Balding 2002. Used when likelihood is intractable but simulation is cheap (genetics, epidemiology). ABC-rejection / ABC-MCMC / ABC-SMC. R `abc`, Python `ELFI`.

### T3.2 Exact / Almost-exact small-sample CDFs
- **Permutation distribution** for two-sample tests with exact p-values up to n=20.
- **Fisher's exact** generalized to k×m contingency (Mehta-Patel 1983 network algorithm). Today only 2×2 in `nonparametric.go:42`.
- **Exact KS** distribution for small n (Marsaglia-Tsang-Wang 2003).

### T3.3 Spatial / functional data primitives
- **Gaussian Process** — kernel-parameterised; predictive mean/variance; conjugate-noise marginal likelihood. RBF / Matérn / RationalQuadratic / Periodic / Linear kernels. ~250 LOC.
- **Spatial point processes** — Strauss / Matérn / Thomas cluster processes, Ripley's K function, pair-correlation function. R `spatstat`. ~200 LOC.

### T3.4 Operational / specialised
- **Beta regression** (Ferrari-Cribari-Neto 2004) for proportion data.
- **Dirichlet regression**.
- **Generalized linear models (GLM) family** — link-function table (logit/probit/cloglog/identity/log/inverse) + IRLS solver. statsmodels `GLM`. ~300 LOC, but coordinates with `regression.go` + `optim/`.
- **Quantile regression** (Koenker-Bassett 1978) — linear-program-based; coordinates with `optim/simplex`.
- **Robust regression** (Huber-M / Tukey-bisquare). ~150 LOC.
- **Mixed-effects models** (Pinheiro-Bates 2000). ~400 LOC.

### T3.5 Information geometry (already partial in `infogeo/`)
Not strictly `prob/` scope — Fisher information / natural gradient / ParametricFamily already live in `infogeo/`. Cross-link recommendation: `prob.FisherInformationFromLogPDF(family)` should compose `infogeo` primitives.

---

## Cross-package coupling notes

- **`autodiff` ↔ `prob`** — HMC/NUTS gradients should flow through `autodiff`. Today every `XxxLogPDF` will need to be expressed as `autodiff.Variable` to be HMC-compatible. The pattern is correct in `copula/pdf.go:73` (`ClaytonLogPDFFn` is autodiff-friendly per `copula/autodiff_test.go`); replicate package-wide. **Coordinate with 014 autodiff-api `autodiff.Func` recommendation.**
- **`calculus` ↔ `prob`** — adaptive quadrature unblocks generic CDF/moment computation for distributions without closed-form CDFs (NIG, GHyp, Stable, Tweedie, Cox-process intensities). Wait for 017's Tier 1 AdaptiveGaussKronrod before shipping T2.1's exotic CDFs.
- **`linalg` ↔ `prob`** — MVN sampler needs Cholesky (already in `linalg/cholesky.go`); Wishart Bartlett needs Cholesky factors. Confirmed available.
- **`optim` ↔ `prob`** — MLE/MAP fitting helpers (`prob.FitNormal(data) → (μ, σ, ll)`, `prob.FitBeta(data) → (α, β, ll)`, etc.) should compose `optim/lbfgs`. Currently absent — every consumer rolls its own. ~300 LOC of one-liners across the distribution roster.
- **`signal` ↔ `prob`** — modified Bessel functions `I_ν` and `K_ν` (needed for Skellam, NIG, GHyp, GIG densities) are absent repo-wide. Standing them up here is fine but `signal/special` (per slot 124-ish?) might be the future home; coordinate.

---

## Best-in-class references (web crosswalk dated 2026-05-07)

| Area | Reference library | Reason |
|------|-------------------|--------|
| Discrete + continuous roster | `scipy.stats` v1.13 | 100+ named distributions, exhaustive {pdf, logpdf, cdf, sf, logcdf, logsf, ppf, isf, rvs, mean, var, skew, kurt, entropy, fit} — this is the API to mirror. |
| Multivariate / mixture | `Distributions.jl` 0.25 | Type-system-driven; cleanest API for MVT, Wishart, LKJ, Mixture. |
| Copulas | R `copula` 1.1.4 + `VineCopula` 2.5 | Full Archimedean, EV, vine roster. |
| Stochastic processes | `tick` 0.7 (Hawkes) + Python `stochastic` + R `yuima` | Hawkes / SDE / point-process simulation. |
| Bayesian non-parametric | Python `numpyro` + Stan | DP / PYP / IBP / GP. |
| MCMC | Stan + PyMC v5 | NUTS / HMC / SMC reference. |
| Hypothesis tests | `scipy.stats` + `statsmodels.stats` | 60+ tests including all named in T1.5. |

---

## Triage — single-sprint highest-leverage commit

**Add the seven log-space primitives (T1.4: erfcinv, logsumexp, log1mexp, log1pexp, logaddexp, logsubexp) plus the Gamma sampler (T1.3 Marsaglia-Tsang) plus the Inverse-Gamma distribution (T1.2).** ~400 LOC total.

This single commit unblocks:
- Every `XxxLogPDF` / `XxxLogSF` per 116's HIGH finding (logsumexp + log1mexp).
- Every truncated / boundary-tail computation (erfcinv).
- Every Beta / Dirichlet / Wishart / Chi² / Inverse-Gamma sampler (Gamma sampler is the dependency).
- The Normal-Inverse-Gamma conjugate prior (T2.4) which BOCPD already needs (slot 023's R1).
- Half the Tier-1 sampler roster (T1.3) reduces to one-liners.

After that, sprint ordering is **T1.4 → T1.3 → T1.2 → T1.1 → T1.5 → T1.6 → T2.\*** with T2.3 (stochastic processes) and T2.5 (MCMC) being the two big "v0.12" milestones once the foundation is in.

---

## Recap — items absent from `prob/` and `prob/copula/` per topic prompt

- Skellam ✓ T1.1
- Zero-inflated Poisson / NB ✓ T1.1
- Beta-binomial ✓ T1.1
- Hypergeometric ✓ T1.1
- Categorical / Multinomial / Geometric / NegBin ✓ (already named by 116; samplers T1.3)
- Power-law / Zipf ✓ T1.1
- Inverse-gamma ✓ T1.2
- NIG ✓ T2.1
- Lévy stable / α-stable ✓ T2.1
- Cauchy ✓ T1.2
- Pareto / Weibull / Lognormal / Gumbel / Fréchet / GEV / Logistic ✓ T1.2
- GIG / Generalized Hyperbolic ✓ T2.1
- Tweedie ✓ T2.1
- Multivariate-t / Wishart / inverse-Wishart / Multinomial-Dirichlet ✓ T2.1
- Gaussian / Student-t copula ✓ (present in `prob/copula/`)
- Frank / Joe / AMH / BB1-8 / vines C/R ✓ T2.2
- Poisson process / Cox / Hawkes / Compound Poisson ✓ T2.3
- Brownian / fBm / OU / CIR / Lévy processes ✓ T2.3
- Conjugate-prior catalog / DP / IBP / Pitman-Yor ✓ T2.4
- Inverse-CDF / Box-Muller / Ziggurat / Rejection / Importance ✓ T1.3
- MH / Gibbs / HMC / NUTS / SMC ✓ T2.5
- Mann-Whitney ✓ (present at `nonparametric.go:120`)
- Welch ✓ (present at `hypothesis.go:94`)
- KS / AD / CvM / Shapiro-Wilk / Kruskal-Wallis ✓ T1.5
- AIC / BIC / DIC / WAIC ✓ T1.6

**Total Tier 1 ≈ 2,580 LOC**, **Tier 2 ≈ 4,300 LOC**, **Tier 3 ≈ 1,300 LOC**. Tier 1 is the v0.11 target; Tier 2 splits across v0.12 (T2.1, T2.2) and v0.13 (T2.3, T2.4, T2.5); Tier 3 is post-v1.0.

Report ends; ~370 lines.
