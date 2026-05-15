# 119 | prob-api: distribution interface, sampling vs density, vectorized batch

**Agent 119 / 400 | Topic: API ergonomics for `prob` distributions**
**Scope: `C:/limitless/foundation/reality/prob/distribution.go`, `distributions.go`, sibling `copula/sklar.go`**

---

## TL;DR

The `prob` package has a **deliberately minimalist** distribution surface that is consistent at the **free-function tier** (`<Name>PDF/CDF/Quantile`) but **incomplete at the interface tier** (`Distribution` exposes only PDF + CDF, no Quantile, no Sample, no moments). There is **zero RNG / sampling** anywhere — no `Sample()`, no `Source`, no `rand.Source`. There is **no log-density** API. There is **no vectorized batch** for PDF/CDF/Quantile. `BetaDist`, `NormalDist`, `ExponentialDist`, `UniformDist` exist as scipy-style frozen distributions but only for 4 of the 8 distributions implemented. No moment-summary struct; no `Mean()`, `Variance()`, `Entropy()`. The sibling `copula` package takes a different — and arguably better-aligned-with-Reality-philosophy — approach: **CDFs as first-class function closures** (`type CopulaCDF func(...)`) instead of interface types.

---

## 1. Distribution interface — uniform across all distributions?

**No, it is partial.** The interface only covers PDF + CDF, and only 4 of 8 distributions are wrapped:

`prob/distribution.go:27`:
```go
type Distribution interface {
    PDF(x float64) float64
    CDF(x float64) float64
}
```

| Distribution | Free fns | Wrapped as `Distribution` | Notes |
|---|---|---|---|
| Normal | PDF/CDF/Quantile | `NormalDist` (PDF/CDF only) | Quantile not on interface |
| Exponential | PDF/CDF/Quantile | `ExponentialDist` (PDF/CDF only) | Quantile not on interface |
| Uniform | PDF/CDF | `UniformDist` | no Quantile fn at all |
| Beta | PDF/CDF | `BetaDist` | no Quantile fn at all |
| Poisson | PMF/CDF | **no wrapper** | discrete, no `Distribution` |
| Gamma | PDF/CDF | **no wrapper** | continuous; could be wrapped |
| Binomial | PMF/CDF | **no wrapper** | discrete |

### Issue 119-1: Quantile is missing from the Distribution interface

`Distribution` exposes only `PDF` and `CDF`. The free-function tier has `NormalQuantile` and `ExponentialQuantile`, but they are **not** methods on `NormalDist`/`ExponentialDist`, and they are **not** on the `Distribution` interface. Consumers doing inverse-CDF sampling, conformal-style quantile lookups, or risk-quantile reporting (VaR / CVaR — exactly the RubberDuck use case cited in the doc comment) cannot get a quantile through the interface. They must type-switch and call the free function. Recommend extending:

```go
type Distribution interface {
    PDF(x float64) float64
    CDF(x float64) float64
    Quantile(p float64) float64  // returns NaN if not implemented
}
```

…and adding `BetaQuantile`, `UniformQuantile`, `GammaQuantile`, `PoissonQuantile`, `BinomialQuantile`.

### Issue 119-2: PMF and PDF unified under one method name `PDF`

`PoissonPMF`/`BinomialPMF` are unified into one `PDF(x float64)` only conceptually — but Poisson and Binomial don't have wrappers, so the issue is hidden. If they ever get wrapped, `PoissonDist.PDF(x float64)` will silently truncate `x` to int. The interface should either:
- accept `interface{}` / a typed key (ugly), or
- have parallel `Mass(k int)` for discrete distributions and a separate `DiscreteDistribution` interface, or
- formally document that `PDF` for discrete returns the PMF at `floor(x)` (mass at integer point).

The current docstring at `distribution.go:29` says "For discrete distributions, this is the probability mass function" — but there is no implementation that exercises this.

---

## 2. Sampling: returns single value or batched slice?

**Neither. Sampling does not exist in `prob`.** Search of the entire package returns zero `Sample(…)` methods, zero `rand.Source` parameters, zero `math/rand` imports in non-test files. Test files import `math/rand` only for property-test setup in `conformal/`.

This is a **legitimate design choice**, given Reality's philosophy of pure deterministic math + golden-file reproducibility. Sampling intrinsically requires an RNG, and a pure-math library should refuse to bundle a non-deterministic facility. But it does mean:

- Every consumer (Oracle, RubberDuck, Sentinel) that needs samples must roll their own inverse-CDF transform: `x = NormalQuantile(rng.Float64(), mu, sigma)`. That requires a `Quantile`. See Issue 119-1.
- For Beta, Uniform, Gamma, Binomial there is no Quantile, so consumers cannot inverse-CDF sample either.

### Recommendation 119-A: Either commit to "no RNG" with full Quantile coverage, or add explicit `Sample(rng *rand.Rand)`

The middle ground (no RNG + missing quantiles) is the worst position. Two clean alternatives:

**Option A (purest):** Declare that `prob` provides PDF/CDF/Quantile only; samplers live in consumer packages. Then **complete Quantile coverage** for every distribution.

**Option B (scipy-like):** Add `Sample(rng *rand.Rand) float64` to the `Distribution` interface (or a parallel `Sampleable` interface). Pass `*rand.Rand` explicitly — never an implicit global. Provide a sibling `SampleN(rng, n) []float64` for batched draws.

Because Reality already forbids hidden mutable state and global RNG, Option B's signature must be `Sample(rng *rand.Rand)`, never `Sample()` with package-level state.

### Issue 119-3: No batched / vectorized PDF, CDF, Quantile

There is no `NormalPDFBatch(xs, mu, sigma []float64) []float64` or similar. Consumers like `aicore` evaluating likelihoods over a buffer have to loop in Go and pay function-call overhead per element. Compare `BrierScoreBatch`/`LogLossBatch` in `prob/prob.go:131,166` which **do** provide batch APIs for scoring. The asymmetry is unmotivated: scoring has batch; PDF/CDF do not.

Reality's design rule #3 says "No allocations in hot paths. Functions accept output buffers. Pistachio calls these at 60 FPS." A 60 FPS consumer cannot call `NormalPDF` element-wise on a per-pixel buffer without violating that rule. Recommend adding `NormalPDFInto(out, xs []float64, mu, sigma float64)` and similar `…Into` variants for the high-traffic distributions.

---

## 3. Density vs log-density: both? log-default?

**Only natural-scale density. No log-density API.**

This is a real problem for numerical stability. Internally, `BetaPDF`, `GammaPDF`, `PoissonPMF`, `BinomialPMF` all already compute in log-space and then `math.Exp` at the end (see `distributions.go:259`, `377`, `313`, `455`). Consumers doing MAP estimation, MLE optimization, or HMC sampling almost always want the log-density directly — exponentiating then re-logging loses precision and underflows for high-dimensional joint products.

### Issue 119-4: Add LogPDF / LogPMF as the primitive; PDF derived from it

Recommended pattern (matches scipy.stats `logpdf`, Distributions.jl `logpdf`, Stan):

```go
func NormalLogPDF(x, mu, sigma float64) float64 {
    if sigma <= 0 { return math.NaN() }
    z := (x - mu) / sigma
    return -0.5*z*z - math.Log(sigma) - 0.5*math.Log(2*math.Pi)
}
func NormalPDF(x, mu, sigma float64) float64 { return math.Exp(NormalLogPDF(x, mu, sigma)) }
```

Same shape for Beta, Gamma, Poisson, Binomial — these already compute log internally. Almost free refactor. Big precision win for log-likelihood sums.

`KLDivergenceNumerical` at `distribution.go:169` would also benefit: it already calls `math.Log(px/qx)`, which is `LogPDF(p,x) - LogPDF(q,x)` — directly consumable.

---

## 4. Frozen distributions — partial scipy-style

The four wrapper structs (`BetaDist`, `NormalDist`, `ExponentialDist`, `UniformDist`) are scipy-frozen-distribution-style. Each provides a constructor with parameter validation:

```go
NewBetaDist(alpha, beta) *BetaDist     // returns nil if alpha<=0 || beta<=0
NewNormalDist(mu, sigma) *NormalDist   // returns nil if sigma<=0
```

### Issue 119-5: nil-on-invalid is unidiomatic; consumers cannot distinguish "bad params" from "out of memory"

Returning `nil` plus no `error` is the lossy pattern. The free functions return `NaN` on bad parameters, which is at least a typed signal. Constructors should return `(*Dist, error)` per Go convention, e.g. `(*BetaDist, error)` with `err == ErrBetaInvalidParams`. Compare `prob/copula/archimedean.go:70` (`ClaytonCopulaCDFFn`) which **does** return `(func, error)` — sibling code already establishes the convention.

### Issue 119-6: Constructor coverage is asymmetric

Poisson, Gamma, Binomial have no wrapper. Either wrap all 8, or wrap none and use the function-closure pattern from `copula` (see §6 below).

### Issue 119-7: Wrappers store params as exported fields with no immutability

`BetaDist.Alpha`, `BetaDist.Beta` are public mutable fields. A consumer can `d := NewBetaDist(2,3); d.Alpha = -1` and `d.PDF` will silently return NaN forever after. A "frozen" distribution should be frozen — make fields unexported, expose `Alpha()` getter, or document mutation as a foot-gun.

---

## 5. Random source: io.Reader / Source / explicit RNG

**N/A — no sampling.** See §2. If sampling is added, the contract must be an explicit `*rand.Rand` (or `rand/v2.Source`) parameter, never global. This aligns with Reality design rule "Reimplement from first principles" and golden-file determinism.

If Reality wishes to preserve cross-language golden-file parity for samplers, it must standardize on a specific PRNG implemented identically in Go/Python/C++/C#. PCG, xoshiro256**, or splitmix64 are the realistic candidates. `crypto/rand.Reader` would not work (not deterministic). Recommend documenting this decision explicitly in `crypto` package docs.

---

## 6. Result types: scalar vs struct{Mean, Variance, Skew, Kurt}

**No moment-summary struct exists.** No `Mean()`, `Variance()`, `Skewness()`, `Kurtosis()`, `Entropy()`, `Mode()`, `Median()` methods on any wrapper. (Note: there is a `Median` *function* at `prob.go:311` but it operates on a `[]float64` sample, not a distribution.)

### Issue 119-8: Add MomentSummary and Entropy

scipy returns a `Stats` named-tuple with `(mean, variance, skew, kurtosis)`. Distributions.jl exposes `mean(d), var(d), std(d), entropy(d), mode(d), skewness(d), kurtosis(d)` as separate functions. For Reality, recommend:

```go
type Moments struct {
    Mean, Variance, Skewness, Kurtosis float64
}
type MomentDistribution interface {
    Distribution
    Moments() Moments
    Entropy() float64
    Support() (lo, hi float64)
}
```

These are closed-form for every distribution in the package and would unlock substantial reuse. Right now consumers compute `mu` and `sigma` themselves and lose the closed-form skew/kurt entirely.

### Issue 119-9: Support / range introspection is missing

Each distribution implicitly defines its support (Beta on [0,1], Exponential on [0,∞), Normal on R, Poisson on N). The bounds are buried in docstrings. A `Support() (lo, hi float64)` method would make `KLDivergenceNumerical` self-configuring (it currently demands the caller pass `lo, hi`, which is a footgun for unbounded distributions like Normal).

---

## 7. Comparison with siblings: prob/conformal, prob/copula

### prob/copula — **function-closure pattern**

```go
type CopulaCDF func(u []float64) (float64, error)
type MarginalCDF func(x float64) float64
type JointCDF func(x []float64) (float64, error)

func GaussianCopulaCDFFn(sigma [][]float64) CopulaCDF { ... }
func ClaytonCopulaCDFFn(theta float64) (func(u, v float64) float64, error) { ... }
```

This is **substantially more idiomatic Go** than `prob`'s interface-with-method-set pattern. Closures over parameters give you a frozen distribution for free; the type **is** the function; no struct, no method dispatch overhead, no nil-vs-error ambiguity. And errors are returned at evaluation time when they actually arise (not at construction with sentinel nil).

Note however the asymmetry: `CopulaCDF` returns `(float64, error)` while `MarginalCDF` returns `float64`. This is itself a minor inconsistency inside `copula` — one for agent 117 to flag — but compared to `prob`'s `Distribution` interface, the function-closure approach is cleaner.

### Issue 119-10: prob and prob/copula use different distribution-API paradigms

`prob.Distribution` is interface-based; `copula.CopulaCDF` is function-typed. These are sibling subpackages. A user composing `prob.NormalDist` with `copula.SklarJointFromMarginals` cannot directly pass `NormalDist` as a `MarginalCDF` — they must wrap it: `copula.MarginalCDF(func(x float64) float64 { return n.CDF(x) })`. The packages are **not interoperable** without adapter glue.

Recommend: either (a) make `prob.Distribution` a function type compatible with `copula.MarginalCDF`, or (b) provide explicit adapter helpers `func AsMarginalCDF(d Distribution) copula.MarginalCDF`.

### prob/conformal — **no distribution abstraction at all**

`conformal` deals exclusively with empirical quantiles over score slices (`SplitQuantile(scores []float64, alpha float64)`). It does not consume `Distribution` or `CopulaCDF`. This is appropriate for distribution-free prediction sets — that is its whole point — but worth noting that no shared abstraction is needed here.

---

## 8. Smaller observations

- **`distribution.go:48,77,106,134` constructors:** All four use `if alpha <= 0 { return nil }`. There is **no error returned, no documented sentinel**. This breaks idiomatic Go. Use `(*BetaDist, error)`.
- **`distribution.go:169` `KLDivergenceNumerical`:** Trapezoidal-rule integration on the interface. Reasonable, but there is no analytic-KL fast-path (e.g. closed-form Normal-Normal KL is well known). Consumers in Echo (cited at top of file) will want the analytic version. Recommend `func KLDivergence(p, q Distribution) (float64, bool)` where bool indicates analytic-form availability.
- **`distributions.go:303` `PoissonPMF` takes `int`** while `BinomialPMF` takes `int, int`. This is correct — discrete supports are integer — but inconsistent with the `Distribution.PDF(x float64) float64` signature. Re-emphasizes Issue 119-2.
- **`distributions.go:359` `GammaPDF(x, k, theta)`:** parameterizes by **scale** `theta`. Comment correctly clarifies this. But `GammaCDF` and `GammaPDF` use `theta` as **scale** while many references use **rate** `beta = 1/theta`. Add an explicit `NewGammaDistFromRate(k, beta)` constructor for users coming from the rate convention. Also: there is no `GammaQuantile` despite the inverse incomplete gamma being available in the same family (see `regularizedGammaLowerSeries`).
- **No `MultivariateNormal`, no `Dirichlet`, no `MixtureModel`.** These are foundational distributions used pervasively in Bayesian work; their absence is notable. (Possibly out of scope for this agent — flag for a future agent.)

---

## 9. Recommendations summary, prioritized

| # | Severity | Item |
|---|---|---|
| 119-1 | **High** | Add `Quantile(p) float64` to `Distribution` interface; implement for all 8 distributions |
| 119-4 | **High** | Add `LogPDF` / `LogPMF` as the primitive; derive `PDF` from it |
| 119-3 | High | Add `…Into(out, xs)` batched variants for PDF/CDF (allocation-free) |
| 119-8 | High | Add `Moments()`, `Entropy()`, `Support()` methods or interface |
| 119-5 | Medium | Constructors should return `(*Dist, error)`, not nil-on-invalid |
| 119-6 | Medium | Wrap all 8 distributions, not just 4 |
| 119-10 | Medium | Adapt `prob.Distribution` ↔ `copula.MarginalCDF` interop |
| 119-A | Medium | Decide: pure-deterministic (no RNG), or add `Sample(rng *rand.Rand)` explicitly |
| 119-2 | Low | Document discrete `PDF(x float64)` semantics or add `Mass(k int)` |
| 119-7 | Low | Make wrapper struct fields immutable post-construction |
| 119-9 | Low | `KLDivergenceNumerical` should pull `Support()` from distributions |

---

## Appendix — files examined (absolute paths)

- `C:/limitless/foundation/reality/prob/distribution.go` (interface + 4 wrappers + KL helper)
- `C:/limitless/foundation/reality/prob/distributions.go` (8 distributions, free functions)
- `C:/limitless/foundation/reality/prob/types.go` (PredictionOutcome, CalibrationPoint, DiagramBucket — non-distribution types)
- `C:/limitless/foundation/reality/prob/prob.go` (Bayesian / scoring / aggregation — unrelated to distribution API but provides BatchN reference patterns)
- `C:/limitless/foundation/reality/prob/copula/sklar.go` (sibling pattern — function closures)
- `C:/limitless/foundation/reality/prob/copula/archimedean.go` (sibling — `(func, error)` return convention)
- `C:/limitless/foundation/reality/prob/copula/gaussian.go` (sibling — uses `prob.NormalQuantile` directly)
- `C:/limitless/foundation/reality/prob/conformal/split.go` (sibling — distribution-free, no abstraction needed)

End agent 119.
