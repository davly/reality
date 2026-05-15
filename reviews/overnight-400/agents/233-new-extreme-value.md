# 233 | new-extreme-value — Extreme Value Theory canon absent from `reality`

**Summary (2 lines):** Reality v0.10.0 ships **zero** extreme-value primitives — no GEV, no GPD, no Hill/Pickands/DEdH estimator, no return-level, no POT machinery, no tail index, no mean-excess plot, no VaR/ES; the Fisher–Tippett–Gnedenko trichotomy (Gumbel/Fréchet/Weibull-bounded) and Balkema–de Haan–Pickands threshold theorem are the missing twin pillars of risk math. Slot 233 scopes a new `prob/evt/` sub-package (~1,640 LOC source + ~960 LOC tests) covering 18 primitives E1–E18 plus a 2-primitive `prob/risk/` cross-link (VaR/CVaR/ES — composes with slot 222 risk-aware bandits and slot 117 Pareto/Weibull/GEV distribution roster) for v0.11 — the second-most overdue Block-C slot in `prob/` family after slot 232 robust-stats.

---

## 0. Verified ABSENT (repo-wide grep)

`Grep` over `C:\limitless\foundation\reality\` returns **zero matches** for: `GEV`, `Generalized Extreme Value`, `Fisher.Tippett`, `Gnedenko`, `Pickands`, `Hill estimator`, `Dekkers.Einmahl.de.Haan`, `Balkema.de.Haan`, `peaks over threshold`, `POT`, `Generalized Pareto`, `GPD`, `genextreme`, `genpareto`, `tailIndex`, `tail.index`, `extremeValueIndex`, `mean excess`, `meanExcess`, `meplot`, `paretoQQ`, `returnLevel`, `return.level`, `returnPeriod`, `exceedanceProbability`, `blockMaxima`, `annualMaxSeries`, `Hosking`, `probabilityWeightedMoments`, `PWM`, `LMoments`, `L.moments`, `tailDependence`, `lambda_U`, `lambda_L`, `λ_U`, `λ_L`, `expectedShortfall`, `Expected.Shortfall`, `CVaR`, `ConditionalVaR`, `valueAtRisk`, `VaR`, `spectralRiskMeasure`, `subexponential`, `regularlyVarying`, `regular.variation`, `tailEstimator`, `Coles 2001`, `Embrechts.Klüppelberg.Mikosch`.

**The 47 grep hits for `tail|VaR|threshold|exceedance` from CLAUDE.md context survey are all unrelated:** `audio/onset/peak_picking.go` thresholds are spectral-flux peaks; `optim/proximal/*` thresholds are L1-soft-threshold; `audio/segmentation/onset_offset.go` thresholds are amplitude detectors; `chaos/analysis.go` Lyapunov tail is exponent decay; `prob/conformal/*` quantile is rank-based finite-sample correction (NOT extreme-value tail). **None touch EVT.**

The two near-misses worth naming explicitly:

- `prob/distributions.go:17` and `prob/distribution.go:21` consumer-doc comments **literally name "extreme value analysis (Exponential)"** as a Sentinel use case — but only Exponential ships, which is precisely the Gumbel-domain-of-attraction *light-tail* benchmark, NOT a Fisher–Tippett primitive. The doc-string promise is unfulfilled by 2024-era code.
- `prob/copula/archimedean.go:32` cites Gumbel 1960 "Distributions des valeurs extrêmes en deux dimensions" — the **bivariate-extreme-value-copula** literature seed — but reality ships only the Gumbel *copula*, never the Gumbel *univariate distribution* (the Gumbel CDF `exp(-exp(-(x-μ)/σ))` is absent; only the copula's Archimedean generator `(-ln u)^θ` is present). The Gumbel/Fréchet/Weibull triple — and the GEV unifying parameterisation `H_ξ(x) = exp(-(1+ξz)^{-1/ξ})` — is the precise 2025-era gap.

Slot 117 §T1.2 already lists Gumbel/Fréchet/GEV/Pareto/GPD/Weibull as ~120+50+60+50 LOC bullets but stops at PDF/CDF/Quantile/LogPDF surface; slot 233 is the **estimation, threshold-selection, return-level, tail-index** layer that 117 doesn't scope — the half of EVT canon that lives in `evir`/`extRemes`/`POT`/`ismev` R packages and Python `pyextremes`/`scipy.stats.genextreme,genpareto`. **117 owns the distribution roster; 233 owns the inference + risk-quantification machinery.**

---

## 1. The conceptual unlock — the two-pillar Fisher–Tippett / Pickands–Balkema–de Haan duality IS the framework

EVT is governed by **two limit theorems** that together cover every tail-quantification regime:

| Theorem | What it says | Estimation regime | Primitive |
|---|---|---|---|
| **Fisher–Tippett–Gnedenko (1928, 1943)** | Maxima of `n` iid rvs, suitably normalised, converge in distribution to one of three families: Gumbel (light), Fréchet (heavy / power-law), Weibull (bounded above) — unified as **GEV** with shape ξ ≷ 0. | **Block maxima**: take annual / monthly maximum, fit GEV by MLE or PWM. | E1–E5 |
| **Pickands–Balkema–de Haan (1974, 1975)** | Conditional excess `X − u \| X > u` converges to **Generalized Pareto Distribution (GPD)** as threshold `u → x_F`. | **Peaks-Over-Threshold (POT)**: pick threshold, fit GPD by MLE or PWM. | E6–E10 |

The two are **dual** under the equivalence `ξ_GEV = ξ_GPD` — the shape parameter is the **tail index** and is identical across the two parameterisations:

- ξ > 0: heavy tail (Fréchet domain) — Pareto, Cauchy, Student-t, log-gamma, Burr.
- ξ = 0: light tail (Gumbel domain) — Normal, Exponential, Lognormal, Gamma, Weibull (the *distribution*).
- ξ < 0: bounded tail (Weibull domain — note: NOT the Weibull distribution; the ξ<0-GEV is sometimes called *reversed Weibull*) — Beta, Uniform.

**ξ is the single number that controls all tail behaviour.** Estimating ξ from data is the **central computational problem** of EVT and admits four canonical estimators (Hill 1975 / Pickands 1975 / Dekkers–Einmahl–de Haan 1989 / Probability-Weighted Moments — Hosking–Wallis 1985, 1987), each with different bias-variance tradeoffs and admissibility regions. **A repo without ξ-estimators cannot compute the 100-year flood.**

The risk-quantification chain:

```
Data → fit GEV/GPD → ξ̂, μ̂, σ̂ → return level z_T = μ + (σ/ξ)((-log(1-1/T))^(-ξ) - 1)
                                → return period T(z) = 1/(1 - GEV(z; θ̂))
                                → exceedance prob P(X > u) = ζ_u · (1+ξ(z-u)/σ̃)^(-1/ξ)
                                → VaR_α / CVaR_α / Expected Shortfall E[X | X > VaR_α]
```

Every primitive in slot 233 is one node of this chain. **It is the only chain reality lacks for "what is the worst-case loss?"**

---

## 2. The 18 primitives E1–E18 + 2 risk extensions

### Cluster A — GEV distribution + estimation (~440 LOC)

- **E1 — `GEVPDF / GEVCDF / GEVQuantile / GEVLogPDF`** (`prob/evt/gev.go`, ~120 LOC) — three-parameter (μ, σ, ξ); ξ=0 limit handled by Gumbel branch (`exp(-exp(-z))`); ξ≠0 uses `(1+ξz)^(-1/ξ)` with domain check `1+ξz > 0`. Match scipy `genextreme` (note scipy uses opposite-sign convention `c = -ξ` — pin tests against R `extRemes::devd` and Python `pyextremes` which both use Coles convention ξ).
- **E2 — `GEVFitMLE(data) → (μ̂, σ̂, ξ̂, ll, fisher_info)`** (~120 LOC) — Maximum Likelihood via L-BFGS on profile log-likelihood; standard pitfall is unbounded likelihood for ξ ≤ -0.5 (Smith 1985), handle by constrained optimisation `ξ ∈ (-0.5, ∞)`. Fisher information matrix yields delta-method standard errors.
- **E3 — `GEVFitPWM(data) → (μ̂, σ̂, ξ̂)`** (~80 LOC) — Probability-Weighted Moments (Hosking-Wallis 1985); β_r = (1/n) Σ ((i-1)(i-2)…(i-r)) / ((n-1)(n-2)…(n-r)) · X_(i). Closed-form for ξ via β_0, β_1, β_2 and root-finding on `(2β_1 - β_0)/(3β_2 - β_0) = (1 - 2^{-ξ})/(1 - 3^{-ξ})`. **PWM is preferred when n < 50** (Hosking-Wallis-Wood 1985) — MLE is asymptotically efficient but has finite-sample bias.
- **E4 — `LMoments(data, order) → (l_1, l_2, l_3, l_4, t_3, t_4)`** (~60 LOC) — Hosking 1990 L-moments (linear combinations of order statistics) with L-skewness `t_3 = l_3/l_2` and L-kurtosis `t_4 = l_4/l_2`. Foundational primitive: PWM is just a re-parameterisation of L-moments. Useful far beyond GEV — every Tier-1/Tier-2 distribution in slot 117 §T1.2 admits an L-moment fitter (Hosking 1996 textbook covers 22 distributions).
- **E5 — `GEVReturnLevel(μ, σ, ξ, T) → z_T`** + **`GEVReturnLevelCI(data, T, α, method) → (z_T, lower, upper)`** (~60 LOC) — z_T = μ + (σ/ξ)((-log(1-1/T))^(-ξ) - 1) for ξ≠0; z_T = μ - σ·log(-log(1-1/T)) for ξ=0. CI via delta-method (cheap) or **profile-likelihood** (correct asymmetric coverage near boundary ξ → 0, Coles 2001 §3.3.3) — profile likelihood is the industry standard for hydrology. This is the "100-year flood" function.

### Cluster B — Generalized Pareto (POT) + threshold tools (~440 LOC)

- **E6 — `GPDPDF / GPDCDF / GPDQuantile / GPDLogPDF`** (`prob/evt/gpd.go`, ~80 LOC) — two-parameter (σ, ξ); 1 - (1+ξx/σ)^(-1/ξ) for ξ≠0, 1 - exp(-x/σ) for ξ=0. Match scipy `genpareto`.
- **E7 — `GPDFitMLE(excesses) → (σ̂, ξ̂, ll, fisher_info)`** (~80 LOC) — same numerical pathology as E2 at ξ ≤ -0.5; for ξ near 0 the Hessian becomes ill-conditioned (Grimshaw 1993) — use Grimshaw's reparameterisation `θ = -ξ/σ`, profile out one parameter, root-find on the other. **Industry-standard MLE is Grimshaw 1993** in `evir::gpd` R package.
- **E8 — `GPDFitPWM(excesses) → (σ̂, ξ̂)`** (~60 LOC) — Hosking-Wallis 1987; ξ̂ = 2 - β_0/(β_0 - 2β_1), σ̂ = 2β_0β_1/(β_0 - 2β_1). Closed-form, fast, robust at small n.
- **E9 — `MeanExcess(data) → (thresholds, meanExcess)`** + `MeanExcessPlot` (~80 LOC) — for each threshold `u`, compute `e(u) = mean(X_i - u | X_i > u)`. **Linear in u above optimum threshold ⇔ GPD assumption holds** (Davison-Smith 1990); the slope equals ξ/(1-ξ). Threshold-selection diagnostic. Cross-link to slot 165 (sequence-prob synergy, robust mean estimator).
- **E10 — `ParameterStabilityPlot(data, thresholds) → (σ̃(u), ξ̂(u))`** (~80 LOC) — Coles 2001 §4.3.4: re-fit GPD at varying thresholds; plot σ̃(u) = σ_u - ξu (modified scale) and ξ̂(u). Above the true threshold both should be approximately constant. Standard diagnostic alongside E9.
- **E11 — `POTReturnLevel(σ, ξ, ζ_u, u, ny, T) → z_T`** (~40 LOC) — z_T = u + (σ/ξ)((T·n_y·ζ_u)^ξ - 1); ζ_u = P(X>u) is rate of exceedance per observation, n_y is observations-per-year. The **POT alternative to E5** — typically lower variance because it uses more data than block-maxima.

### Cluster C — Tail-index estimators (~360 LOC)

- **E12 — `HillEstimator(data, k) → ξ̂_Hill`** + **`HillPlot`** (~80 LOC) — Hill 1975: for **heavy tails only (ξ > 0)**, `ξ̂_Hill(k) = (1/k) Σ_{i=1}^k log(X_(n-i+1)/X_(n-k))` where X_(i) is order statistic. Plot ξ̂(k) vs k and pick "stable region" between bias (k too large) and variance (k too small). The most-cited tail-index estimator (~5000 citations).
- **E13 — `PickandsEstimator(data, k) → ξ̂_Pickands`** (~50 LOC) — Pickands 1975: ξ̂_P = (1/log 2) · log((X_(n-k) - X_(n-2k))/(X_(n-2k) - X_(n-4k))). **Works for all ξ ∈ ℝ** (unlike Hill which is ξ>0-only) but high variance. Standard alternative when ξ sign is unknown.
- **E14 — `DekkersEinmahlDeHaanEstimator(data, k) → ξ̂_DEdH`** (~60 LOC) — DEdH 1989 "moment estimator": combines Hill log-moments H_n^(1) and H_n^(2) into ξ̂ = H_n^(1) + 1 - 0.5/(1 - (H_n^(1))²/H_n^(2)). **Works for all ξ**, lower variance than Pickands. Modern industry default.
- **E15 — `SmithEstimator(data, k) → ξ̂_Smith`** (~40 LOC) — Smith 1987 MLE-based; bias-corrected Hill. Optional but standard in `evir`.
- **E16 — `BiasCorrectedHill(data, k, method) → ξ̂_corrected`** (~60 LOC) — Beirlant–Dierckx–Goegebeur–Matthys 1999 jackknife-style bias correction; the canonical bias-corrected Hill used in 2010s+ research.
- **E17 — `OptimalK(data, estimator, method) → k*`** (~70 LOC) — automated selection of order-statistic count k via:
  - Drees–Kaufmann 1998 sequential procedure
  - Reiss–Thomas 2007 minimum-distance heuristic
  - Bootstrap subsampling (Danielsson-de Haan-Peng-de Vries 2001)
  Without this primitive, the Hill plot is "Hill's horror" — pick k by eye. With it, the workflow is automatable.

### Cluster D — Multivariate / spatial extremes + tail dependence (~280 LOC)

- **E18 — `TailDependenceCoefficient(C; "upper"|"lower") → λ_U / λ_L`** (~60 LOC) — λ_U = lim_{u→1} P(F_2(X_2)>u | F_1(X_1)>u). For Gumbel copula λ_U = 2 - 2^(1/θ) (already in `prob/copula/archimedean.go:175` as `GumbelUpperTailDependence` — slot 233 generalises to all copula families: Clayton λ_L = 2^(-1/θ); Gaussian λ = 0 unless ρ=1; Student-t λ_U = λ_L = 2 t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ))); Frank λ = 0; Joe upper-only). 60 LOC of dispatch over `prob/copula/archimedean.go::ArchimedeanFamily`.
- **`ExtremeValueCopula(family, data) → fitted_C`** (deferred to v2, slot 117 §T2.2 covers Galambos / Hüsler-Reiss / Tawn copulas) — non-parametric Pickands dependence function `A: [0,1] → [0,1]` estimable via Capéraà-Fougères-Genest 1997 estimator. ~120 LOC; defer to slot-117-§T2.2 ownership.
- **`MultivariateGEV(data, model) → (μ̂_d, σ̂_d, ξ̂_d, dep)`** (deferred) — Tawn 1988 logistic / negative-logistic dependence; `extRemes::fevd.mvn` reference. Defer.

### Cluster E — Risk measures (cross-link to slot 222 risk-aware bandits) (~120 LOC)

- **`VaR(data, α) → q`** and **`VaR_parametric(distribution, params, α) → q`** (~30 LOC, `prob/risk/var.go`) — Value-at-Risk = quantile_α; trivial wrapper for empirical (sorted) and parametric (any Distribution). Useful only because the *naming* is the lingua franca of finance.
- **`CVaR(data, α)` / `ExpectedShortfall(data, α)` / `ES(data, α)` (aliases)** (~30 LOC) — CVaR_α = E[X | X > VaR_α] = (1/(1-α)) ∫_α^1 VaR_u du. Empirical (mean of top (1-α) fraction) and parametric branches; for GPD-tail, closed form ES = (VaR + σ - ξu)/(1-ξ). **The canonical risk-aware-bandit reward functional** — slot 222 risk-aware-bandit cluster F (line 178) names "CVaR-UCB" and "Mean-variance" but no shared CVaR primitive exists; landing 233 unblocks 222.
- **`SpectralRiskMeasure(data, φ)` / `DistortionRiskMeasure(data, g)`** (~30 LOC) — Acerbi 2002; φ-weighted average quantiles. Generalisation covering VaR (φ = δ_α), CVaR (φ = uniform on [α,1]), exponential / power-law spectral functions.
- **`ProfileLikelihoodCI(data, fit, parameter, α) → (lower, upper)`** (~30 LOC) — generic — used by E5 return-level CI and E7/E8 GPD shape CI. Composes with `optim/lbfgs` for the inner profile optimisation.

---

## 3. Substrate audit — what exists, what's reused

| Substrate | Path | Reuse for slot 233 |
|---|---|---|
| Order statistics | absent (slot 232 PR-0 lands `prob/orderstats.go` 120 LOC) | Hill / Pickands / DEdH all need sorted order statistics — **shared with slot 232** PR-0 |
| Random number generation | absent (slot 230/231/232 PR-0 lands `prob/random.go` 280 LOC) | E17 bootstrap-k selection, return-level bootstrap CI — **shared cross-cutting** |
| L-BFGS / numerical optimisation | `optim/lbfgs.go` ✓ | E2 GEV-MLE, E7 GPD-MLE outer, E22 profile-likelihood inner |
| Bracketed root-finding | `optim/bisection.go` (slot 116 confirms exists) | E3 PWM ξ root-find on `(1-2^{-ξ})/(1-3^{-ξ})` |
| `optim/proximal/operators.go` | exists | unused — EVT is unconstrained |
| Copula tail dependence | `prob/copula/archimedean.go:173-181` `GumbelUpperTailDependence` ✓ | E18 generalises across families; one helper to lift |
| Quantile / inverse CDF infra | `prob/distributions.go` Normal/Beta/Exponential ✓ | parametric VaR for these three; need GEV/GPD/Pareto/Lognormal/Weibull from slot 117 §T1.2 |
| Profile likelihood (none anywhere) | absent | new in slot 233 — also useful for slot 117 generic MLE |
| Adaptive Gauss-Kronrod quadrature | absent (slot 017 §Tier-1) | spectral risk measure CDF integral — **shared with slot 117 §T2.6** |
| Bessel `K_λ` (none) | absent | NIG/Generalized Hyperbolic family in slot 117 §T2.1 — slot 233 does NOT need (chooses GEV/GPD branch) |
| Hill estimator log-moments | none | new — ~25 LOC primitive shared by E12/E14/E15/E16 |
| Block-maxima extractor | none | new — ~20 LOC `BlockMaxima(data, blockSize) → maxima[]` shared by E2/E5 |
| Threshold exceedance extractor | none | new — ~20 LOC `Exceedances(data, u) → (excesses, count, rate)` shared by E7/E8/E11 |

**Substrate completeness:** ~70% — the missing 30% is order-statistics / bootstrap / profile-likelihood, all of which are pulled in by adjacent slots (232 robust-stats, 230 FDR for bootstrap, 117 prob-missing for distribution roster). **The single net-new EVT primitive is the Hill log-moment primitive (~25 LOC); everything else composes.**

---

## 4. Cross-language parity targets

| Primitive | Reference | Tolerance | Notes |
|---|---|---|---|
| GEV PDF/CDF | scipy `genextreme.pdf/cdf` (note sign convention) | 1e-12 | exact algebra, no quadrature |
| GPD PDF/CDF | scipy `genpareto.pdf/cdf` | 1e-12 | exact |
| GEV-PWM | R `lmom::pelgev` (Hosking authoritative) | 1e-12 | exact algebra after L-moments |
| GPD-PWM | R `lmom::pelgpa` | 1e-12 | exact algebra |
| GEV-MLE | R `extRemes::fevd` + Python `pyextremes` | 1e-9 | iterative; shared seed for QQ-plots |
| GPD-MLE Grimshaw | R `evir::gpd` | 1e-9 | Grimshaw 1993 reparam |
| Hill estimator | R `evir::hill`, Python `arch.unitroot.hill` | 1e-12 | closed-form sum of log-ratios |
| Pickands | R `evir::pickands` | 1e-12 | closed-form |
| DEdH | R `evir::moment`, Python `pyextremes.tail_index` | 1e-12 | closed-form moment combination |
| L-moments | R `lmom::samlmu` (Hosking package) | 1e-12 | exact PWM |
| Mean-excess | R `evir::meplot`, Python `pyextremes.mrl_plot` | 1e-12 | sort+sweep |
| Return level (delta CI) | R `extRemes::return.level` | 1e-9 | delta-method Hessian |
| Return level (profile CI) | R `extRemes::ci.fevd` | 1e-6 | inner-loop optimiser convergence |
| Tail dependence λ_U | R `copula::lambda` | 1e-12 | closed-form per family |
| VaR (empirical) | R `PerformanceAnalytics::VaR` | 1e-12 | sort+index |
| CVaR / ES | R `PerformanceAnalytics::ES` | 1e-12 | sort+mean of tail |
| Spectral risk | R `actuar` | 1e-9 | numerical integration |

**Cross-language pinning UNUSUALLY-THICK** because R `evir` (Pfaff-McNeil 2018) + R `extRemes` (Gilleland-Katz NCAR canonical 2024) + R `lmom` (Hosking 2024 — original-author maintained) + Python `pyextremes` (Bocharov 2024) + Python `arch` are all reference-grade implementations. Direct deterministic-equivalence achievable closed-form for half the primitives; shared-seed exact for randomised half.

**R-MUTUAL-CROSS-VALIDATION-3/3 candidates:**

1. **Triple-shape-estimator-Pareto-data**: ξ̂_Hill (E12) ≈ ξ̂_Pickands (E13) ≈ ξ̂_DEdH (E14) on Pareto(α) data with k = √n converge to 1/α at rate O(1/√k) — three estimators agree to 5e-2 at n=10^5.
2. **Triple-fit-method-equivalence**: GEV-MLE (E2), GEV-PWM (E3), GEV-LMoments (E4) on Gumbel(0,1) data converge to (μ̂, σ̂, ξ̂) = (0, 1, 0) with all three estimators agreeing to 1e-3 at n=10^4 — diagnostic for implementation correctness.
3. **Triple-return-level-equivalence**: 100-year level computed via (a) GEV block-maxima E5, (b) GPD-POT E11, (c) empirical (1-1/T)-quantile of 100-year sample agree to 5e-2 on simulated heavy-tail data — demonstrates Fisher-Tippett ⇔ Pickands-BdH duality numerically.

---

## 5. PR cadence — ~13-engineer-day v0.11 milestone

| PR | Primitives | LOC source | LOC test | Days | Rationale |
|---|---|---|---|---|---|
| PR-0 | shared `prob/random.go` + `prob/orderstats.go` | 400 | 120 | 1.5 | **FOURTEENTH Block-C review demanding it** (slots 117/184/188/202/215/216/217/227/228/229/230/231/232/233); shared across slots; should be Tier-0 cross-cutting |
| PR-1 | E1 GEV {PDF/CDF/Quantile/LogPDF} + E6 GPD {PDF/CDF/Quantile/LogPDF} | 200 | 120 | 1 | distribution roster — overlaps slot 117 §T1.2 (cite who lands first); cheapest day-1-shippable |
| PR-2 | E4 L-moments + E3 GEV-PWM + E8 GPD-PWM | 200 | 120 | 1.5 | closed-form fitters; pinned 1e-12 against R `lmom` (Hosking authoritative) |
| PR-3 | E2 GEV-MLE + E7 GPD-MLE Grimshaw | 200 | 140 | 2 | iterative L-BFGS fitters; **architectural keystone** — E2/E7 unblock E5/E11 return levels |
| PR-4 | E9 mean-excess + E10 parameter-stability + E17 OptimalK | 220 | 140 | 1.5 | threshold-selection diagnostics; Drees-Kaufmann 1998 + Reiss-Thomas 2007 |
| PR-5 | E12 Hill + E13 Pickands + E14 DEdH + E15 Smith | 230 | 140 | 1.5 | tail-index quartet; closed-form, pinned 1e-12 |
| PR-6 | E16 bias-corrected Hill + Beirlant-Dierckx-Goegebeur-Matthys 1999 | 60 | 60 | 0.5 | bias correction layer |
| PR-7 | E5 GEV return-level + E11 POT return-level + E22 profile-likelihood CI | 130 | 100 | 1 | the **"100-year flood" PR** — the user-facing primitive |
| PR-8 | E18 tail-dependence dispatch over copula families + bootstrap return-level CI | 90 | 60 | 0.5 | composes `prob/copula/`; closes the bivariate gap |
| PR-9 | `prob/risk/`: VaR + CVaR/ES + SpectralRiskMeasure | 90 | 60 | 0.5 | cross-link to slot 222 (risk-aware bandits — CVaR-UCB now has a shared primitive) |
| PR-10 | `prob/evt/doc.go` + golden-file test corpus + 22-vector minimum per primitive | (in tests) | 100 | 1 | per CLAUDE.md golden-file rule |

**Total ~1,640 LOC source + ~960 LOC tests + ~400 LOC PR-0 substrate over ~13 engineer-days.**

---

## 6. Singular cutting-edge piece, singular moat, singular architectural witness

- **SINGULAR CUTTING-EDGE PIECE — E22 profile-likelihood return-level CI** (~30 LOC composing E2/E5/`optim/lbfgs`). The 2025-state-of-the-art for hydrology / climate / insurance is **profile likelihood, NOT delta-method** — Coles 2001 §3.3.3 textbook treatment, but every practitioner uses it via R `extRemes::ci.fevd(method="proflik")` because it correctly handles asymmetric coverage near ξ → 0 (delta method gives nonsensical CI when the Fisher information matrix is near-singular at the Gumbel boundary). **No zero-dep Go implementation exists.** The 30 LOC is a profile-out-one-parameter loop wrapping E2 — but the *correctness* matters: hydrologists building 100-year flood control infrastructure need the asymmetric CI, not the symmetric one.

- **SINGULAR MOAT — E14 Dekkers-Einmahl-de Haan moment estimator + E16 bias-corrected Hill**. The DEdH 1989 estimator is the **only ξ-estimator that works for all ξ ∈ ℝ with finite variance** — Hill is heavy-tail-only (ξ>0), Pickands has high variance, MLE requires distributional assumption. Bias-corrected Hill (Beirlant–Dierckx–Goegebeur–Matthys 1999) is the **2010s+ research default** because raw Hill is biased toward ξ when k is moderate. **No zero-dep Go alternative; Python `arch.unitroot.hill` ships only raw Hill; R `evir::moment` is the closest.** Slot 233 is the only zero-dep Go landing of the post-1989 tail-index canon.

- **SINGULAR ARCHITECTURAL WITNESS — `prob/distributions.go:17` and `prob/distribution.go:21` consumer-doc comments literally name "extreme value analysis (Exponential)" as a Sentinel use case** but only ship the Exponential PDF/CDF/Quantile — the Gumbel-domain-of-attraction *light-tail* benchmark, NOT a Fisher–Tippett primitive. **The 2024-author wrote the doc-string promise but stopped at the Exponential — the entire EVT canon is unbuilt.** Slot 233 IS the doc-string promise's two-year-overdue follow-up. Likewise `prob/copula/archimedean.go:32` cites Gumbel 1960 "Distributions des valeurs extrêmes en deux dimensions" — the bivariate-EVT seed paper — but reality ships the Gumbel *copula*, never the Gumbel *univariate distribution*. **The Gumbel copula without the Gumbel distribution is "the bivariate extreme without the univariate extreme" — slot 233 closes the half-built bridge.**

---

## 7. Cross-link map

- **slot 117 (prob-missing) §T1.2** owns Pareto/Weibull/Lognormal/Gumbel/Fréchet/GEV/Logistic/Cauchy distribution roster (~120 LOC). **Slot 233 owns the inference + risk-quantification layer.** Recommend slot 117 lands distribution PDFs/CDFs/Quantiles first (PR-1 here); slot 233 PR-2 onwards composes.
- **slot 222 (new-bandits) cluster F line 178** lists "CVaR-UCB" and "Mean-variance" as risk-aware bandits but no shared CVaR primitive exists. **Slot 233 PR-9 lands `prob/risk/CVaR` which slot 222 imports.** Bidirectional cross-link.
- **slot 232 (new-robust-stats) PR-0** lands `prob/orderstats.go` — slot 233 reuses (Hill / Pickands / DEdH / mean-excess / VaR / CVaR are all order-statistic-based). **Tier-0 cross-cutting infrastructure landing as a package across two slots.**
- **slot 230 (new-fdr) F22 e-BH + F23 always-valid** — independent (FDR is not EVT). No direct cross-link.
- **slot 231 (new-conformal) C12 conformal-CDF** — distribution-free vs distribution-fitted: orthogonal. No direct cross-link.
- **slot 228 (bayes-nonparametric)** — Bayesian hierarchical EVT (Coles 2001 §9 is the textbook frontier, Cooley-Nychka-Naveau 2007 is the citation engine) is deferred to "EVT v2"; slot 228 BNP machinery (DPM, hierarchical) is the prerequisite. Slot 233 ships frequentist-only canon.
- **slot 119 (prob-api)** — `Distribution` interface should accommodate GEV/GPD as additional implementations of the existing `NewBetaDist / NewNormalDist / NewExponentialDist / NewUniformDist` pattern: `NewGEVDist(μ, σ, ξ)`, `NewGPDDist(σ, ξ)`. Coordinate.
- **slot 116 (prob-numerics)** — log-space primitives `LogSumExp`, `Log1mExp` (slot 117 §T1.4) needed for GEV/GPD log-PDF tail-stable arithmetic (`(1+ξz)^(-1/ξ)` overflows for large `z`; rewrite via `exp(-(1/ξ) log1p(ξz))`). Wait for slot 117 §T1.4 to land first.
- **slot 165 (sequence-prob synergy)** — orthogonal; sequence ops don't compute extremes.
- **slot 169 (prob-optim synergy)** — direct cross-link: every MLE in slot 233 is L-BFGS over a log-likelihood; the `Distribution.LogLikelihood` ↔ `optim.Func` adapter goes through this slot.
- **slot 188 (prob-linalg synergy)** — Fisher information matrix inversion for delta-method CI; `linalg.CholeskySolve` already exists.
- **slot 124 (signal-special)** — modified Bessel functions absent; slot 233 deliberately avoids the GHyp/NIG/Variance-Gamma branch (slot 117 §T2.1 owns) which needs them. Pure GEV/GPD canon does NOT need Bessel.
- **slot 014 (autodiff-api)** — GEV/GPD log-likelihoods should be autodiff-friendly for HMC/NUTS in MCMC fits (per slot 117 §T2.5 long-term roadmap). Mark as v2.

---

## 8. Topic-prompt recap — every named area covered

| Topic-prompt area | Slot 233 primitive |
|---|---|
| Extreme value theory (EVT): rare event statistics | E1–E18 (entire package) |
| Generalized Extreme Value distribution (GEV) — Fisher-Tippett-Gnedenko | E1 |
| GEV three families: Gumbel / Fréchet / Weibull | E1 ξ=0 / ξ>0 / ξ<0 branches |
| Block maxima → fit GEV via MLE or PWM | E2 (MLE), E3 (PWM), `BlockMaxima(...)` extractor |
| Peaks Over Threshold (POT) → fit GPD | E6, E7 (MLE Grimshaw), E8 (PWM), `Exceedances(...)` extractor |
| Threshold selection: parameter stability / mean excess | E9 mean-excess, E10 parameter-stability |
| Hill estimator: tail index γ for heavy tails (Hill 1975) | E12 |
| Pickands estimator (1975) | E13 |
| Dekkers-Einmahl-de Haan estimator | E14 |
| Maximum Likelihood for GEV / GPD | E2, E7 |
| Probability-Weighted Moments (Hosking-Wallis 1985, 1987) | E3, E8 |
| L-moments | E4 |
| Return levels: r-year return value | E5 (block-maxima), E11 (POT) |
| Return periods, exceedance probability | E5/E11 + `ReturnPeriod(z)` / `ExceedanceProb(u)` helpers |
| Stationary vs non-stationary GEV (with covariates, time) | DEFERRED to v2 (regression+EVT) — Coles 2001 §6, slot 247 |
| Multivariate / spatial extremes (Tawn 1988, Coles 2001) | DEFERRED — slot 117 §T2.2 EV copulas |
| Copulas for tail dependence (cross-link to prob/copula) | E18 |
| Tail dependence coefficient λ_U, λ_L | E18 |
| Pareto tail / power-law fitting | composes E12 Hill + slot 117 Pareto distribution |
| Fréchet vs subexponential vs heavy-tail | doc.go taxonomy + ξ>0 GEV |
| Extreme value index estimation: Hill, moment, Pickands | E12, E13, E14 (moment = DEdH) |
| Bias correction in tail estimation | E15 Smith, E16 Beirlant-Dierckx-Goegebeur-Matthys |
| Bootstrap for return levels | PR-8 bootstrap-CI helper |
| Profile likelihood CI for return levels | E22 (PR-7) |
| Threshold-uncertainty quantification | E10 + E17 OptimalK + bootstrap-k Danielsson 2001 |
| Operational risk: VaR, CVaR (ES) — cross-link to finance | `prob/risk/` (PR-9) — composes slot 222 |
| Risk-aware bandits (cross-link to 222 CVaR) | PR-9 lands shared `CVaR` primitive |
| Conditional VaR / ES | `prob/risk/CVaR` (alias `ES`) |
| Spectral risk measures | `SpectralRiskMeasure` (PR-9) |
| Climate / hydrology: 100-year flood | E5 + E22 = the canonical user-facing primitive |

---

## 9. Closing — one-line architectural witness

`reality v0.10.0` ships the Gumbel **copula** (`prob/copula/archimedean.go`) but not the Gumbel **distribution**; cites Gumbel 1960 "Distributions des valeurs extrêmes" but ships zero univariate-extremes; promises "extreme value analysis (Exponential)" in `prob/distributions.go:17` but the Exponential is the *Gumbel-domain-of-attraction benchmark*, not a Fisher–Tippett primitive. **The half-built bridge from Gumbel-the-copula to Gumbel-the-extreme-value-distribution is the most diagnostic missing-tissue pattern in `prob/`.** Slot 233 closes the bridge end-to-end with the 18-primitive E1–E18 + 4-primitive `prob/risk/` canon — the second-most overdue Block-C slot in the prob-family after slot 232 robust-stats, and the only zero-dep Go landing of the post-1989 tail-index estimation canon (Hill / Pickands / DEdH / Smith / bias-corrected Hill / DKW + Reiss-Thomas + bootstrap k-selection).

Differentiation: **18/18 primitives unique to slot 233**; cross-cutting blocker `prob/random.go + prob/orderstats.go` (~400 LOC) shared with thirteen prior Block-C reviews — correctly scoped Tier-0 cross-cutting infrastructure.

Report ends; ~340 lines.
