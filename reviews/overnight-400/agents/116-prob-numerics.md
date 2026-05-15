# 116 | prob-numerics — log-pdf/log-cdf stability, tail accuracy, regularized incomplete beta/gamma

Audit of `C:\limitless\foundation\reality\prob\` (top-level + `prob/copula/`). Scope per MASTER_PLAN line 116: log-pdf/log-cdf stability under tiny probabilities, CDF/SF tail accuracy, quantile robustness, regularized incomplete beta `I_x(a,b)`, regularized incomplete gamma `P(a,x)`, erf/erfc/erfcinv extremes, log1p/expm1/logsumexp usage, sampler entropy preservation, and the named distribution roster.

Files inspected (+lines): `mathutil.go` 225, `distributions.go` 486, `distribution.go` 196, `hypothesis.go` 191, `prob.go` 525, `jeffreys.go` 174, `copula/archimedean.go` 181, `copula/pdf.go` 169, `copula/hfunctions.go` 131, `copula/studentt.go` 240, `copula/gaussian.go` 358. Distribution roster confirmed missing: Student-t (only in copula/, unexported in prob/), F, chi-squared (CDF only, no PDF/quantile), negative-binomial, geometric, multinomial, Dirichlet, MVN (only bivariate/trivariate in copula/). No Box-Muller, Marsaglia ziggurat, or any sampler. No erfcinv. No logsumexp. No expm1. log1p used in exactly one site (`copula/pdf.go:88`).

---

## CRITICAL: regularized incomplete gamma uses series-only, no continued-fraction branch

**Location:** `mathutil.go:200-225` `regularizedGammaLowerSeries`.

The function uses **only** the Pearson series (DLMF 8.7.1). Numerical Recipes §6.2 and DLMF 8.9 are explicit: the series converges efficiently only when `x < a + 1`; for `x > a + 1` the **continued fraction for Q(a,x) = 1 − P(a,x)** must be used instead. Otherwise the series suffers catastrophic cancellation: it sums O(x^n / Γ(a+n)) terms whose individual magnitudes far exceed the final value, so the right tail is contaminated by O(1) cancellation noise. With `eps = 1e-14` and `maxIter = 200`, large `(a,x)` either silently exit on the relative-tolerance check at a wrong sum, or simply hit max-iter for `x ≫ a`.

**Direct fallout — three production paths affected:**

1. `chiSquaredCDF(x, df)` at `mathutil.go:187` is `regularizedGammaLowerSeries(df/2, x/2)`. Used by `ChiSquaredTest` at `hypothesis.go:186` to compute `pValue = 1 - chiSquaredCDF(chiSq, df)`. For the strongest/most-significant rejections — exactly where p-values matter most (tiny right-tail probabilities) — `chiSquaredCDF` is being asked for `1 - tiny`, then subtracted from 1 in `ChiSquaredTest`. **Double tail loss:** the series is computing the wrong thing in the regime where it matters, then the result is subtracted from 1 to get the very small number that was lost in rounding. `ChiSquaredTest`'s docstring claim of `~1e-12` precision is wrong by orders of magnitude for chiSq ≫ df.

2. `GammaCDF(x, k, theta)` at `distributions.go:397` — same issue, fixed-shape gamma right-tail. For `x/theta ≫ k` the CDF is ~1 and the function returns `1 − ε` with ε wrong.

3. `PoissonCDF(k, lambda)` at `distributions.go:339` is `1.0 - regularizedGammaLowerSeries(k+1, lambda)`. The CDF identity `P(X≤k) = Q(k+1, λ)`. Critically the code computes `1 - P(k+1,λ)` to get `Q`, but **always** routes through the series. For `λ ≫ k+1` (left tail of Poisson — the tail that is small) we want a small number; computing it as `1 − P` where `P ≈ 1` loses all leading digits. For `λ ≪ k+1` (right tail, where CDF ≈ 1) the `1 − P` form is fine, but the series converges slowly. The correct branching: if `λ < k+1`, use series → `1 − series`; otherwise use continued fraction for `Q(k+1, λ)` directly. Currently neither path is correct in its dual regime.

**Fix:** add `regularizedGammaUpperCF(a, x)` (Lentz on DLMF 8.9.2 continued fraction), then dispatch in `regularizedGammaP(a,x)` by `x < a+1 ? series : 1 - upperCF`. Mirrors the dispatch already correctly implemented for `RegularizedBetaInc` symmetry-flip at `mathutil.go:72`.

**Severity:** high. Touches every chi-squared p-value and every Poisson/Gamma CDF call. Cross-language golden files (per CLAUDE.md design rule 1) will not catch this if all golden vectors avoided the broken regime.

---

## HIGH: NormalCDF tail loss for very negative `(x − μ)/σ`

**Location:** `distributions.go:47-52`.

```go
return 0.5 * math.Erfc(-(x-mu)/(sigma*math.Sqrt2))
```

For `x ≪ μ` (left tail), `-(x−μ)/(σ√2)` is large positive, so `erfc` returns a tiny positive number — fine. For `x ≫ μ` (right tail) the argument is large negative, `erfc(-z)` for large positive `z` returns a number very close to 2, multiplied by 0.5 yields ≈1; **the survival function `1 − CDF` then loses all significant digits by cancellation**. There is no public `NormalSF`/`NormalLogCDF`/`NormalLogSF` to give the caller a tail-accurate path. This is the standard "always provide both `cdf` and `sf`" rule (R `pnorm(x, lower.tail=FALSE)`, SciPy `norm.sf`) and reality lacks it across **every** distribution in the package.

**Fix:** add `NormalSF(x,μ,σ) = 0.5 * math.Erfc((x-μ)/(σ*√2))` (no negation, no `1−`). Plus `NormalLogCDF`, `NormalLogSF` using `log(erfc)` with asymptotic expansion `log(erfc(z)) ≈ -z² - log(z√π)` for `z > ~6` to avoid `log(0)`. Same applies to `ExponentialCDF` (`distributions.go:157` returns `1 - exp(-λx)` — should expose `ExponentialSF(x,λ) = exp(-λx)` for tail accuracy; left tail OK but no `1 - exp` form for `λx → 0` either, where `expm1` is the right primitive).

---

## HIGH: BetaPDF / GammaPDF / PoissonPMF / BinomialPMF return PDF only; no LogPDF/LogPMF

**Location:** `distributions.go:229-261, 359-377, 303-315, 434-457`.

Every density function computes `math.Exp(logPDF)` at the last step. For Bayesian inference, MCMC, mixture models, importance sampling, and any chain of multiplicative density evaluations, callers want **log-density** without the trip through `exp`. The package internally computes `logPDF` then exponentiates; the natural exposure is to expose `BetaLogPDF`, `GammaLogPDF`, `PoissonLogPMF`, `BinomialLogPMF` directly. `prob/copula/pdf.go` got this right with `ClaytonLogPDFFn` / `GumbelLogPDFFn` (`pdf.go:73, 130`); the top-level distributions did not. For tiny densities (deep tails of Gaussian, Beta near boundaries, Poisson with large `k`) the PDF underflows to 0 while the log-PDF remains finite. Today reality forces the underflow.

**Fix:** alongside each `XxxPDF`/`XxxPMF`, add `XxxLogPDF`/`XxxLogPMF`. Refactor `XxxPDF(x,...) = math.Exp(XxxLogPDF(x,...))`. Zero risk of behavior change; pure additive.

---

## HIGH: NormalQuantile precision capped at 1.15e-9 — no Newton refinement

**Location:** `distributions.go:67-127` (Acklam 2004).

Acklam's docstring claims max relative error **1.15e-9**, which the comment at line 64 honestly reproduces. SciPy / Boost / R's `qnorm` deliver full-double accuracy (~2 ulp) by following Acklam with one Halley/Newton step using the analytic CDF. With `NormalCDF` already in the package and `dΦ/dz = NormalPDF(z)`, one Newton iteration `z' = z - (Φ(z) - p)/φ(z)` lifts accuracy to ~1e-15 in ~12 ns. Quantile callers (used by `copula/gaussian.go:72` for **every** Gaussian copula CDF query at probit-transform; used by `copula/studentt.go:74` for the t-quantile bracket; will be used by every future MVN sampler) inherit Acklam's 1e-9 cap as a **floor**, not a ceiling.

**Fix:** add one Halley step at end of `standardNormalQuantile`. ~6 LOC.

---

## MEDIUM: betaCF / regularizedBetaInc duplicated across packages

**Locations:**
- `mathutil.go:59-145`  `RegularizedBetaInc` + `betaCF` (exported, used by `BetaCDF`, `BinomialCDF`, `studentTCDF`).
- `copula/studentt.go:175-240`  `regularizedBetaInc` + `betaContinuedFraction` (unexported, byte-equivalent algorithm).
- `copula/studentt.go:120-165`  `standardNormalQuantileLocal` (Acklam coefficients duplicated from `distributions.go:79-103`).

Three disconnected reasons for the duplication ("avoid circular import", "self-contained package"), but reality is now sitting on **two** `Acklam` and **two** `Lentz-betaCF` implementations whose coefficients and tolerances must be kept in lockstep across silent updates. The next golden-file refresh that touches one but not the other yields a bisect-only flake. Promote `studentTCDF` from unexported in `prob` to exported `StudentTCDF`/`StudentTLogPDF`/`StudentTPDF`/`StudentTQuantile`, then `copula/` imports `prob` (it already does at `copula/gaussian.go:7`).

**Severity:** medium today, high once a third caller materializes (e.g. F-distribution PDF will need t and χ² and beta).

---

## MEDIUM: betaCF non-convergence is silent

**Location:** `mathutil.go:142-145` and `copula/studentt.go:238-240`.

```go
// Did not converge — return best estimate.
return f
```

Both implementations swallow non-convergence and return whatever value was last computed. For pathological `(x, a, b)` (e.g., `a` or `b` very large with `x` near the convergence boundary `(a+1)/(a+b+2)`) the continued fraction can need >200 iterations; the `f` returned is potentially off by orders of magnitude. No `math.NaN()`, no logged warning, no error. Since this routine underlies **every** chi-squared via `studentTCDF`, every Beta CDF, every binomial CDF, and every Welch t-test p-value, a silent miss is a silent wrong p-value. NumRecipes 3rd ed §6.4 explicitly recommends bumping `MAXIT` to 1000 for Beta and emitting a warning at exhaustion.

**Fix:** raise `maxIter` to 1000 (memory cost zero, runtime cost negligible at the rare exhaustion case), and on exhaustion return `math.NaN()` rather than `f`, matching the package's documented "Failure mode: returns NaN" pattern at `RegularizedBetaInc:55`.

---

## MEDIUM: Beta near-boundary log-PDF cancellation

**Location:** `distributions.go:259`.

```go
logPDF := (alpha-1)*math.Log(x) + (beta-1)*math.Log(1-x) - logB
```

For `x` very close to 1 (e.g. `x = 1 - 1e-15`), `math.Log(1-x)` loses precision: `1-x` is computed in double-precision *before* the log, so the small subtraction is the floor on accuracy, not the log. The right primitive is `math.Log1p(-x)` (= `log(1−x)`) which is monotone-stable. Same issue would bite `BetaCDF` if it ever needed `log(1−x)`; today it doesn't because `RegularizedBetaInc` uses raw `math.Log(1-x)` at `mathutil.go:77` — same flaw, same fix. `BinomialPMF` line 455 also has `math.Log(1-p)` for `p → 1`; should be `math.Log1p(-p)`.

**Fix:** mechanical replace `math.Log(1-x) → math.Log1p(-x)` in `distributions.go:259, mathutil.go:77`, and `distributions.go:455`.

---

## MEDIUM: ExponentialCDF numerical loss for small `λx`

**Location:** `distributions.go:164` `1 - math.Exp(-lambda*x)`.

For `λx → 0` this returns 0 instead of `λx − (λx)²/2 + …`. The right primitive is `-math.Expm1(-λx)` (= `1 − exp(−λx)`). Affects the entire exponential left tail and any reliability/survival computation done as `1 − S`.

**Fix:** `return -math.Expm1(-lambda*x)`.

---

## MEDIUM: chiSquaredCDF returns 0 for `x == 0` even when `df < 2`

**Location:** `mathutil.go:188-191`.

```go
if x <= 0 || df <= 0 { return 0 }
return regularizedGammaLowerSeries(df/2.0, x/2.0)
```

For `df = 1` and `x = 0` the chi-squared CDF is 0 — that's correct. But `regularizedGammaLowerSeries` itself at `mathutil.go:204` returns 0 for `x == 0` while `LogGamma(a)` at `mathutil.go:212` is computed only when `x > 0`. Looks safe, but combined with `chiSquaredTest`'s `1 - chiSquaredCDF` semantics, a `chiSq = 0` test (perfect fit) returns p-value 1 — fine — and the path through 0 is handled. **However**: for `df = 1, x = +1e-300` (subnormal), `regularizedGammaLowerSeries` computes `a*math.Log(x) - LogGamma(a)` = `0.5 * (-690) - LogGamma(0.5)` ≈ −346, then `math.Exp(lnPrefix) * sum` underflows to 0. Acceptable here, but the structural rule "subnormal-aware density" is violated: the function silently truncates to 0 instead of returning the smallest representable positive double. Density consumers (rejection sampling, importance-sampling weights) detect this as zero density at non-zero support — wrong.

**Lower priority** because the practical regime is rare.

---

## MEDIUM: StudentTQuantile bisection uses linear convergence

**Location:** `copula/studentt.go:60-111`.

`StudentTQuantile` brackets via Acklam's normal approximation, then **bisects** with `eps = 1e-10`. Bisection halves the bracket per step, requiring ~50 iterations to reach 1e-15 from a width-1 bracket. The function caps at `maxIter = 200` and `eps = 1e-10`, so users get ~10 digits of precision when 15 are available. With the analytic `StudentTCDF` already costing one beta-CF evaluation per step, the same cost in a Newton-Halley step using `StudentTPDF` (not currently exposed but trivial: `gamma((df+1)/2) / (sqrt(πdf) * gamma(df/2)) * (1 + x²/df)^(-(df+1)/2)`) yields cubic convergence — ~5 steps to full precision instead of ~50.

**Fix:** add Newton step `x' = x - (CDF(x) - p)/PDF(x)` at the end of bisection. Same algorithmic improvement applies to `NormalQuantile` per the earlier finding. Acceptance bracket already in place; just one analytic update step.

---

## LOW: Welch t-test p-value clipping suspicious

**Location:** `hypothesis.go:65-68, 142-145`.

```go
pValue = 2.0 * (1.0 - studentTCDF(math.Abs(tStat), df))
if pValue > 1.0 { pValue = 1.0 }
```

The clamp to ≤1 hides numerical bug rather than fix it: if `studentTCDF(|t|) < 0.5` ever happens (rounding), `pValue > 1` is the symptom; clamping to 1 silently produces the result a user expects. The right behavior is to detect that and return `2*studentTCDF(-|t|)` instead — more numerically stable for the small-p regime since it computes the small tail directly rather than `1 - large`. Same issue at line 142.

**Fix:** `pValue = 2.0 * studentTCDF(-math.Abs(tStat), df)`. No clamp needed; small-p directly computed.

---

## LOW: chiSquaredTest p-value clipping at 0

**Location:** `hypothesis.go:186-190`.

```go
pValue = 1.0 - chiSquaredCDF(chiSq, df)
if pValue < 0 { pValue = 0 }
```

`pValue < 0` only happens because `chiSquaredCDF` is buggy (Critical issue above) and over-shoots `1`. The clamp masks the bug. Once the upper-CF dispatch is added, `pValue = chiSquaredSF(chiSq, df)` returns the small tail directly without subtraction; clamp becomes unnecessary.

---

## LOW: ClampProbability default range [0.01, 0.99]

**Location:** `prob.go:12-15`.

`MinProb = 0.01`, `MaxProb = 0.99` is far too aggressive. Modern Bayesian/forecasting code routinely operates in `(1e-300, 1 - 1e-15)`. The constants are documented as guarding against `log(0)` in log-odds, but `1e-300` is fine in `math.Log`. Hard-clamping `0.999 → 0.99` destroys 2 orders of magnitude of confidence-interval information at every aggregation step (`LogOddsPool`, `WeightedAverage`, `BayesianUpdateChain`). Recommendation: `MinProb = 1e-15`, `MaxProb = 1 - 1e-15`. CLAUDE.md's "absolute certainty is never warranted" is preserved while the routine doesn't quietly crush legitimate posteriors.

**Severity:** design — out of strict numerics scope but mentioned because it ripples through every probability that survives this package.

---

## LOW: Archimedean copula `math.Pow(x, -theta)` overflow path

**Location:** `copula/archimedean.go:86, 122`, `copula/pdf.go:55-56, 81-82`, `copula/hfunctions.go:62-63, 109-110`.

For Clayton with `theta = 50` and `u = 1e-3`, `math.Pow(u, -theta) = 1e150`, while `vNegT` for `v = 0.5` is `~10^15`; the sum is dominated by `uNegT` and `vNegT` is lost in rounding. The closed-form is mathematically stable but its float64 implementation isn't. The log-form (e.g. `ClaytonLogPDFFn` at `pdf.go:88`) doesn't help because it still computes `math.Pow(u, -theta)` raw — log gets taken **after** the sum. Right approach: factor out the dominant `max(uNegT, vNegT)` before summing, equivalent to logsumexp on `[-θ log u, -θ log v]`. Today the package has **no** `logsumexp` helper anywhere; adding one is a small, high-leverage change that this and several other Archimedean / mixture-density paths would consume.

**Fix:** add `prob.LogSumExp(xs ...float64) float64` (Kahan-shifted formula `m + log(Σ exp(xᵢ - m))`); rewrite Archimedean PDFs/h-fns/CDFs in log space using it. Concretely, Clayton's `s = u^{-θ} + v^{-θ} - 1` becomes `log(s) = logsumexp(-θ log u, -θ log v) + log(1 - exp(...))` etc. — moderate refactor but worth it for `theta > 20`.

---

## LOW: BetaPDF boundary case `alpha = 1, beta = 1` returns 0 instead of 1

**Location:** `distributions.go:236-255`. Walking through: `x = 0, alpha = 1, beta = 1` →  `alpha == 1` branch → returns `beta = 1`. ✓ `x = 1, alpha = 1, beta = 1` → `beta == 1` branch → returns `alpha = 1`. ✓ Looks correct on inspection. **No bug here**, kept in audit trail because the boundary cascade reads brittle and tests should pin it.

---

## Missing / not-implemented

Per MASTER_PLAN line 116 distributional roster, the following are entirely absent from `prob/`:

- **Student-t** PDF/CDF/quantile (only via unexported `studentTCDF` in `mathutil.go:160` and via duplicated `StudentTCDF`/`StudentTQuantile` in `copula/studentt.go`). Promote to public.
- **F-distribution** (chi-squared ratio) — needed for ANOVA. None.
- **Chi-squared** PDF, quantile (only CDF unexported).
- **Negative-binomial**, **Geometric**, **Multinomial**, **Dirichlet** — none.
- **Multivariate normal** — bivariate / trivariate CDFs only via `copula/gaussian.go`; no general n-variate (Genz QMC deferred to v2 per `gaussian.go:36`); no MVN PDF, no MVN log-PDF, no MVN sampler.
- **Box-Muller, Marsaglia ziggurat** — no samplers exist anywhere in the package.
- **erfcinv** — needed for any quantile that goes through erfc; not present.
- **logsumexp** — not present (one site uses `log1p`, none use logsumexp).
- **expm1** — not used anywhere; needed for `ExponentialCDF`, log-survival functions, `log(1 − exp(x))` style routines.

---

## Triage summary (priority order)

1. **CRITICAL — `regularizedGammaLowerSeries` series-only.** Add upper-CF branch. Touches χ², Gamma, Poisson CDFs. ~40 LOC. Refresh chi-squared / Poisson / gamma golden vectors with right-tail samples that today's series gets wrong.
2. **HIGH — Add `XxxSF` / `XxxLogPDF` / `XxxLogSF` for every distribution.** Tail accuracy + zero-underflow log-density. Currently zero such functions exist. ~150 LOC additive.
3. **HIGH — Newton/Halley refinement on `NormalQuantile` and bisection-based `StudentTQuantile`.** Lifts both from ~1e-9/1e-10 to ~1e-15. ~10 LOC each.
4. **HIGH — Promote `studentTCDF` and Acklam coefficients to single public source.** Eliminates duplication between `prob` and `prob/copula`. ~30 LOC delete in `copula/studentt.go`.
5. **MEDIUM — `betaCF` exhaustion path returns NaN, raise maxIter to 1000.** Prevents silent wrong p-values. ~6 LOC.
6. **MEDIUM — `math.Log1p(-x)`, `-math.Expm1(-x)` substitutions.** Mechanical, ~4 sites.
7. **MEDIUM — `prob.LogSumExp` helper + Archimedean copula log-space refactor.** Stabilizes Clayton/Gumbel for high theta + small u/v.
8. **LOW — Hypothesis-test p-value cleanup (`2*CDF(-|t|)`, drop clamps once roots fixed).**
9. **LOW — Reconsider `MinProb=0.01` floor.**

## Single highest-leverage commit

**Add `regularizedGammaUpperCF` + dispatch in a unified `regularizedGammaP`/`regularizedGammaQ` pair, and route `chiSquaredSF` / `gammaSF` / `poissonSF` through `Q` directly rather than `1 − P`.** This single change:

- Makes `ChiSquaredTest` produce correct small p-values (the regime where it matters).
- Makes `PoissonCDF` / `GammaCDF` correct in their right tail.
- Eliminates the `if pValue < 0 { pValue = 0 }` band-aid in `hypothesis.go:188`.
- Costs ~50 LOC + golden vectors regenerated on the right-tail regime.
- Aligns with reality's existing pattern in `RegularizedBetaInc` which already does the symmetry-flip dispatch correctly (`mathutil.go:72`).

Without this, every chi-squared p-value below ~1e-12 is wrong by orders of magnitude and the package's documented `~1e-12` precision claim at `hypothesis.go:165` and `mathutil.go:185` is structurally false.

---

## Cross-references

- `mathutil.go:59-145` `RegularizedBetaInc` is **structurally correct** (DLMF 8.17.22, Lentz, symmetry-flip) — good template for the gamma fix.
- `copula/gaussian.go:149-196` `BivariateNormalCDF` (Drezner-Wesolowsky GL10) is **best-in-class numerics in the package** — no findings.
- `copula/pdf.go:73-92` `ClaytonLogPDFFn` is **the only correct log-density form in `prob/`** and the right model for the missing top-level `BetaLogPDF` etc.

Report ends; ~370 lines.
