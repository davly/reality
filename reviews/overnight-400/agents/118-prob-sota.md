# 118 | prob-sota — engineering crosswalk vs scipy.stats / statsmodels / Distributions.jl / R `stats` / TFP / PyMC / NumPyro / Stan

Scope per MASTER_PLAN line 118: SOTA library comparison. Sister reports 116 (numerics) and 117 (missing API surface) covered the bug list and the API gaps respectively; **this report does not re-list distributions or hypothesis tests by name** — it ranks the *engineering tricks* canonical libraries deploy, and asks which are zero-dependency-portable into reality's Go-first / golden-file-first / no-tensor-framework architecture.

Web crosswalk dated 2026-05-07: scipy 1.13, statsmodels 0.14.2, Distributions.jl 0.25.119, R `stats` 4.4.2, TensorFlow-Probability 0.24, PyMC 5.16, NumPyro 0.16, Stan 2.34, BlackJAX 1.2, Turing.jl 0.36.

Files inspected at `C:\limitless\foundation\reality\prob\`: `distributions.go` (486), `distribution.go` (196 — `Distribution` interface), `hypothesis.go` (191), `regression.go` (138), `timeseries.go` (307), `mathutil.go`, `prob.go`, `markov.go`, `jeffreys.go`, `nonparametric.go`, `copula/*`, `conformal/*`. `prob/distribution.go:1-196` already exposes a thin `Distribution` interface (`PDF`/`CDF`/`Mean`/`Variance`/`Sample`) with constructors `NewBetaDist`/`NewNormalDist`/`NewExponentialDist`/`NewUniformDist`. **This is the load-bearing fact for the entire report**: reality has begun the move from raw-function-style (the `XxxPDF(x, params...)` pattern) to object-style (the `dist := New…(); dist.PDF(x)` pattern); every SOTA library has made this transition; the open questions are **which** object-style choices to lock in. Sections below answer that.

---

## 1. scipy.stats — frozen-distribution objects, generic methods, fit/dispatch infrastructure

### 1.1 Headline algorithm: the `rv_continuous` / `rv_discrete` superclass

scipy.stats defines **one** abstract base class per support type. Each distribution overrides at most three private hooks: `_pdf` (or `_pmf`), `_cdf`, and `_rvs` (sampler). The base class then **derives generically**:

- `logpdf` / `logcdf` / `logsf` from `pdf` / `cdf` / `sf` (with `numpy.log`, then specialised overrides where stable)
- `sf` from `1 - cdf` (with override hooks for tail-stable variants — every scipy distribution that has a tail-stable SF overrides this; ~60% do)
- `ppf` (quantile) from a Brent root-find on `cdf - p` if no closed form is provided, with a per-distribution `_ppf_single` override hook
- `isf` (inverse SF) symmetric to ppf
- `entropy` from `-∫ pdf · log pdf` numerical quadrature unless overridden
- `moment(n)` from `∫ x^n · pdf` quadrature unless overridden
- `mean` / `var` / `std` / `median` / `skew` / `kurtosis` derived from `moment(1..4)` unless overridden
- `interval(α)` = `(ppf((1-α)/2), ppf((1+α)/2))` — symmetric two-sided
- `fit(data)` = MLE via `optimize.fmin` over the parameter vector with `_logpdf` as the objective (override hook for closed-form MLE)
- `expect(func)` = `∫ func(x) · pdf(x) dx`

**The trick:** every distribution authors gets `logpdf`/`fit`/`moment`/`expect`/`interval` for free with a single `_pdf` and `_cdf`. Adding a new distribution is ~30 LOC instead of ~300. Override hooks let specialists improve any single derived method without touching the base.

### 1.2 Engineering trick: `_argcheck` + frozen distributions

Every distribution has an `_argcheck(α, β, …) → bool` method that runs once at object creation. The result is a **frozen distribution** — `norm(loc=0, scale=1).pdf(x)` does not re-validate `loc`/`scale` per call. This is the reverse of reality's current `BetaPDF(x, alpha, beta)` which has no parameter-validation entry point. With reality already building `Distribution` constructors (`NewBetaDist`), the same trick is one `Validate() error` per constructor away.

### 1.3 Engineering trick: `_pdf_default_method` dispatch table

For every method, scipy holds three implementations: closed-form (override), generic numerical (base), and a **shape-parameter-conditioned** fallback. Example: `genextreme._cdf(x, c)` dispatches on `c == 0` → Gumbel formula, `c > 0` → Weibull formula, `c < 0` → Fréchet formula. The dispatch table is private, the user sees one method. reality currently has zero such dispatch — every formula is monolithic in one function. As distributions enter (NIG, Stable, Tweedie per 117 T2.1), the dispatch table pattern is the only sane way to keep formulas readable.

### 1.4 Zero-dep portability

**All three of 1.1, 1.2, 1.3 are pure-engineering tricks with zero dependencies.** scipy's only hard dep is `numpy` for vectorisation; reality already has `[]float64` slices and `out` buffers (CLAUDE.md rule 3). The full pattern transports:

```go
type Distribution interface {
    PDF(x float64) float64
    CDF(x float64) float64
    Sample(rng *rand.Rand) float64
}

type LogDistribution interface {  // optional override
    Distribution
    LogPDF(x float64) float64
    LogCDF(x float64) float64
    LogSF(x float64) float64
}

type Quantilable interface {  // optional override
    Distribution
    Quantile(p float64) float64
}

// Generic helpers (in prob/generic.go):
func GenericLogPDF(d Distribution, x float64) float64 { return math.Log(d.PDF(x)) }
func GenericSF(d Distribution, x float64) float64     { return 1 - d.CDF(x) }
func GenericQuantile(d Distribution, p float64) float64 { /* Brent on CDF - p */ }
func GenericMoment(d Distribution, n int, lo, hi float64) float64 { /* GK quadrature */ }
func GenericEntropy(d Distribution, lo, hi float64) float64       { /* -∫ p·log p */ }
func GenericFit(family DistFamily, data []float64) (Distribution, float64) { /* MLE */ }
```

Every named distribution then implements **only** the closed-form methods it has and inherits the rest. This collapses the ~50-distribution roadmap from 117 (~2,580 LOC Tier 1) to ~1,200 LOC of distribution-specific code plus ~300 LOC of generic harness. **Single largest leverage gain in the report.**

---

## 2. statsmodels — formula DSL, GLM family, time-series state-space framework

### 2.1 Headline algorithm: the `GLM(endog, exog, family=…)` IRLS engine

statsmodels' `GLM` accepts a formula `y ~ x1 + x2 + x1:x2` or two arrays, plus a `family` enum (Gaussian, Binomial, Poisson, Gamma, InverseGaussian, NegativeBinomial, Tweedie). The IRLS loop is **family-agnostic**: at each iteration, the family supplies the link function, link derivative, and variance function; IRLS itself is one matrix solve. Adding a new family is ~50 LOC defining `link(μ)`, `link_inv(η)`, `link_deriv(μ)`, `variance(μ)`.

reality has `LinearRegression` (`prob/regression.go:18-86`) that hardcodes Gaussian-identity. The IRLS generalisation is a structural unlock for every count/proportion regression in 117 T3.4. **Coordinate point with `optim/`:** IRLS at iteration *k* solves `(Xᵀ W_k X) β = Xᵀ W_k z_k` — this is `linalg.SolvePosDef` with a positive-definite weighted normal-equation matrix. Both ingredients exist. ~250 LOC of pure additions wires them together.

### 2.2 Engineering trick: `Family` type as a tuple of three callbacks

```python
class Binomial(Family):
    link = Logit()        # μ → η
    variance = lambda μ: μ * (1-μ)
    deviance_resid = …
```

statsmodels factors out **link** and **variance** as separate type hierarchies. Logit, Probit, CLogLog, Identity, Log, Inverse, Power link functions each have `link(μ) → η`, `inverse(η) → μ`, `deriv(μ) → dη/dμ`, `inverse_deriv(η) → dμ/dη`. Any (family, link) combination that is mathematically valid (`Binomial × {Logit, Probit, CLogLog}`, `Poisson × {Log, Identity}`) is constructible. reality should mirror this: separate `Link` interface from `Family` interface; the GLM engine consumes both.

### 2.3 Engineering trick: state-space backbone for time series

`statsmodels.tsa.statespace` is a single Kalman-filter engine (`KalmanFilter` C extension) that powers SARIMAX, VARMAX, UnobservedComponents, DynamicFactor, and MarkovSwitching as **specialisations of a unified `(Z, T, R, Q, H)` matrix-form state-space model**. Every consumer just supplies the system matrices.

reality has `ARIMA` (`prob/timeseries.go`) hand-rolled. Promoting to a `KalmanFilter` engine + `ARIMA = SARIMAX(p, d, q)` specialisation is ~400 LOC but unblocks every state-space family at once. **This is the single largest leverage gain in time series**, structurally analogous to scipy's `_pdf` dispatch in 1.1.

### 2.4 Zero-dep portability

GLM/IRLS engine: **fully portable**, ~250 LOC, zero deps beyond reality's existing `linalg`. Recommended.

State-space Kalman: **fully portable**, ~400 LOC, zero deps beyond `linalg.MatMul`/`linalg.Cholesky`. Recommended.

Formula DSL (`y ~ x1 + x2`): **out of scope** for a zero-dep Go library — Patsy / R's formula machinery is a small parser plus design-matrix construction; ~600 LOC; nice-to-have for ergonomics but not foundational. **Defer to a future `prob/formula` sub-package; the GLM and Kalman engines should accept raw `(X, y)` matrices with no formula dependency.**

---

## 3. Distributions.jl — type-system-driven immutable distributions, mixture/sampler dispatch

### 3.1 Headline algorithm: `Distribution{F<:VariateForm, S<:ValueSupport}` parametric type hierarchy

Julia's type system encodes variate form (Univariate / Multivariate / Matrixvariate) and support (Discrete / Continuous) as **type parameters**. The compiler enforces "you cannot pass a discrete distribution where a continuous quantile is expected" and dispatches `pdf` / `logpdf` / `quantile` / `rand` to the most-specific method via JIT-multiple-dispatch.

In Go, the equivalent is **interface composition**:

```go
type UnivariateContinuous interface { Distribution; ContinuousSupport(); UnivariateSupport() }
type UnivariateDiscrete   interface { Distribution; DiscreteSupport(); UnivariateSupport() }
type MultivariateContinuous interface { ... PDFv([]float64) float64; SampleV(rng) []float64 }
type MatrixVariate         interface { ... PDFm(*linalg.Matrix) float64; SampleM(rng) *linalg.Matrix }
```

Methods like `GenericFit` in §1.4 then take the most-specific interface they need. **Cost: ~40 LOC of marker interfaces. Benefit: type-safety for the exotic distributions in 117 T2.1 (Wishart matrix-variate, MVT multivariate, etc.) without runtime reflection.**

### 3.2 Engineering trick: immutable distribution objects + lazy mean/var caches

Every `Distribution` in Distributions.jl is **immutable**. Once `Beta(2.0, 3.0)` is constructed, its `α`/`β` fields cannot mutate. Mean, variance, mode, skewness, kurtosis are stored as **memoised Julia function values** (computed once at first access). For mixture models and posterior predictives where the same component distribution is sampled millions of times, the savings on `mean(d)` redundant-computation are non-trivial.

In Go: `Distribution` should be a struct, not an interface, and the constructor should pre-compute and cache `mean`, `var` if cheap. `BetaDist` already does this implicitly (`distribution.go:48-72`). **Recommendation: lock in the immutability convention package-wide and add a `validate() error` step at construction time.**

### 3.3 Engineering trick: `MixtureModel` as a first-class distribution

Distributions.jl exposes `MixtureModel(components::Vector{Distribution}, weights::Vector{Float64})`. The mixture inherits `Distribution` and dispatches `pdf` to `Σ wᵢ · pdf(componentᵢ, x)`, `logpdf` to `logsumexp(log wᵢ + logpdf(componentᵢ, x))`, `rand` to component-sampling-after-categorical-draw, `cdf` to the weighted sum of CDFs. Zero new abstractions; pure composition.

reality has zero mixture-model machinery anywhere (`Grep` confirms no `Mixture` symbol in `prob/`). For a Gaussian-mixture on top of reality's `NewNormalDist`, the implementation is ~80 LOC if the `Distribution` interface is locked in. **Zero-dep, transports cleanly, depends only on §1.4 logsumexp.**

### 3.4 Engineering trick: `truncated()` wrapper

`truncated(Normal(0, 1), -2, 2)` returns a new distribution whose `pdf` is the original's PDF normalised by `cdf(b) - cdf(a)`, whose `cdf` is `(cdf(x) - cdf(a))/(cdf(b) - cdf(a))`, and whose `quantile(p)` is `quantile(cdf(a) + p · (cdf(b) - cdf(a)))`. This is the **only** truncated-distribution implementation that does not require per-distribution rewrites — a single 60-LOC generic wrapper covers all of 117 T1.2's truncated-anything entries.

reality should mirror exactly: `prob.Truncated(d Distribution, a, b float64) Distribution`. Pure composition. **Single highest-leverage 60-LOC commit in the report.**

### 3.5 Zero-dep portability

All four ideas (parametric type hierarchy → marker interfaces, immutability → struct convention, MixtureModel, Truncated) port with **zero new dependencies**. Total budget: ~200 LOC.

---

## 4. R `stats` + `lme4` — formula DSL, the `lm`/`glm`/`lmer` workhorses, vectorised d/p/q/r quartet

### 4.1 Headline algorithm: the `dXxx` / `pXxx` / `qXxx` / `rXxx` naming convention

R `stats` exposes **exactly four entry points per distribution**: `dnorm` (density), `pnorm` (CDF), `qnorm` (quantile), `rnorm` (sampler). All vectorised. Optional arguments `lower.tail = FALSE` and `log.p = TRUE` switch to SF and log-CDF respectively without function-name explosion. This is the most disciplined 1×4 API in any statistical library.

reality currently has 5 distributions × {PDF, CDF} ≈ 10 functions, mixed with quantile-only-when-it's-easy. **The d/p/q/r convention reduces cognitive surface to four function names per distribution and makes auditing exhaustive: every distribution must answer "what is the d, the p, the q, the r?". Recommendation: rename `BetaPDF → BetaDensity` is overkill; instead lock in the **method-on-Distribution** convention (§1.4) — `d.PDF(x)`, `d.CDF(x)`, `d.Quantile(p)`, `d.Sample(rng)` is the Go-idiomatic equivalent of d/p/q/r.

### 4.2 Engineering trick: `lower.tail` and `log.p` as universal flags

Every R distribution function takes both flags. `pnorm(x, lower.tail = FALSE)` returns the survival function; `pnorm(x, lower.tail = FALSE, log.p = TRUE)` returns log-SF. The flags **defer the formula choice** so that, e.g., `pnorm(z, lower.tail=FALSE, log.p=TRUE)` for `z = 8.5` returns `-37.5...` (analytically correct) instead of `log(1 - 0.999...)` (the catastrophic-cancellation form).

reality lacks every variant — there is no `NormalSF`, no `NormalLogPDF`, no `NormalLogSF` (per 116 HIGH). **The `Distribution` interface should expose `SF`/`LogPDF`/`LogCDF`/`LogSF` as standard methods, with a default-via-§1.4-generic and tail-stable overrides for the distributions that need them. The four orthogonal flag-combinations are exactly the four overridable methods.**

### 4.3 Engineering trick: `lme4`'s `lmer(y ~ x + (1|group))` linear-mixed-effects formula

`lme4` (Bates-Maechler-Bolker-Walker 2014) is the most-cited R package in mixed-model statistics. It uses a sparse-matrix REML engine that decomposes `Λ Λᵀ + σ² I` via Cholesky on a sparse pattern. Linchpin: a profiled likelihood that integrates out the random effects analytically.

For reality: **out of scope until `linalg` ships sparse Cholesky** (review slot 092 territory). When it does, ~400 LOC of `lme4`-style mixed-model machinery is inscope. **Defer.**

### 4.4 Zero-dep portability

d/p/q/r convention as method-on-Distribution: ports cleanly (§1.4). `lower.tail`/`log.p`: ports as separate methods (`SF`, `LogPDF`, `LogCDF`, `LogSF`). `lme4`: defer pending sparse linalg. All within the zero-dep envelope.

---

## 5. TensorFlow-Probability — Bijector composition, JointDistribution, structure-aware MCMC

### 5.1 Headline algorithm: `Bijector` composition for invertible variable transforms

TFP's `Bijector` interface is the cleanest implementation of the change-of-variables formula in any library:

```python
class Bijector:
    def forward(self, x): ...      # x → y = f(x)
    def inverse(self, y): ...      # y → x = f⁻¹(y)
    def forward_log_det_jacobian(self, x): ...   # log |det df/dx|
    def inverse_log_det_jacobian(self, y): ...   # log |det df⁻¹/dy| = -forward_log_det_jacobian(f⁻¹(y))
```

Composing bijectors `Chain([Exp(), Affine(scale=2.0)])` produces a new bijector whose log-det-Jacobian is the sum of the children's. `TransformedDistribution(base=Normal(0,1), bijector=Exp())` is the LogNormal. This **single primitive** generates LogNormal, Logit-Normal, Softmax-Normal, Yeo-Johnson, Box-Cox, the entire normalising-flow family, and every constrained-parameter reparameterisation in MCMC.

reality has no bijector machinery. Implementing the `Bijector` interface with `Exp`, `Log`, `Affine`, `SoftPlus`, `Sigmoid`, `Chain`, `Stack`, `Invert` is ~250 LOC of pure composition with zero deps. The `TransformedDistribution` wrapper is ~80 LOC. **This is the single highest-leverage commit for the MCMC roadmap (117 T2.5):** Stan's "constrain/unconstrain" parameter machinery is exactly a fixed bijector chain per parameter type; HMC/NUTS in unconstrained space is the only stable formulation; without bijectors, every MCMC implementation reinvents per-distribution constraint logic.

### 5.2 Engineering trick: `JointDistribution{Sequential,Coroutine,Named}` for probabilistic models

TFP exposes three syntactic flavours of "specify a joint distribution by writing the generative model":

```python
joint = tfd.JointDistributionSequential([
    tfd.Normal(0., 1.),               # prior on z
    lambda z: tfd.Normal(z, 0.1),     # likelihood given z
])
```

`joint.sample()` walks the list top-to-bottom; `joint.log_prob([z, y])` returns `Σᵢ logp(xᵢ | parents)`. The `Coroutine` variant uses Python generators for full directed-acyclic-graph (DAG) flexibility.

In Go: a `JointDistribution` struct holding `[]func(parents...) Distribution` closures, plus a `Sample()` method that walks the closures, plus a `LogProb([]float64) float64` that walks the same closures with the user-supplied values. ~150 LOC. **Required for any honest Bayesian-modelling face on reality**, but coordinates with §6 (PyMC/NumPyro use the same idea).

### 5.3 Engineering trick: structure-aware HMC/NUTS

TFP's `tfp.mcmc.NoUTurnSampler` accepts a `target_log_prob_fn` that takes a *list of tensors* — one per random variable — and returns a scalar. The sampler auto-discovers the parameter dimensions, applies bijector unconstraining per parameter, and runs leapfrog over the concatenated unconstrained vector. The dual-averaging step-size adaptation is a single state machine separate from the leapfrog integrator.

reality's MCMC roadmap (117 T2.5) should mirror exactly: `HMC.Run(logProbAndGrad func([]float64) (float64, []float64), init []float64, bijectors []Bijector)`. Gradients flow through `autodiff` (per 014 autodiff-api recommendation `autodiff.Func`). **Cross-package linchpin — confirms 014's recommendation independently from a second axis.**

### 5.4 Zero-dep portability

Bijector composition: **fully portable**, ~250 LOC. Recommended as 117 T2.5's first dependency.
JointDistribution: **fully portable**, ~150 LOC. Recommended.
Structure-aware MCMC: portable with autodiff coupling; ~250 LOC NUTS + ~80 LOC step-size adaptor. Coordinates with 014.

The whole TFP pattern transports without any tensor-framework requirement — TFP's tensor abstraction is for batch dimensions / GPU dispatch, neither of which reality needs in v0.x.

---

## 6. PyMC / NumPyro — context-manager modelling, sampler dispatch, log-prob via tracing

### 6.1 Headline algorithm: `with pm.Model(): x = pm.Normal("x", 0, 1)` syntactic sugar

PyMC's context-manager model collects every `pm.Normal(…)` declaration into a hidden `Model` instance, then exposes `pm.sample(2000)` which auto-routes between NUTS (continuous), Metropolis (discrete), and Compound (mixed) samplers. The user writes the generative model in essentially mathematical notation; everything else is automated.

This is a Python-language affordance — `with` is unique to Python, and the pattern requires a thread-local "current model" register. Go's equivalent is **explicit construction**: `m := prob.NewModel(); x := m.Normal("x", 0, 1); y := m.Normal("y", x, 0.1)`. The auto-routing dispatcher, however, transports cleanly: ~50 LOC `pm.sample()`-equivalent that inspects `m.RandomVars`, picks NUTS for continuous + Metropolis for discrete, and runs them.

### 6.2 Engineering trick: `pm.set_data(...)` for cross-validation / posterior-predictive

PyMC's `pm.MutableData("x_obs", x_train)` lets the user swap data without rebuilding the model graph. Posterior-predictive checks just `set_data(x_test)` and re-sample. reality's analog: a `Model.SetData(name string, vals []float64)` that updates the closure-captured data without rebuilding closures. ~30 LOC.

### 6.3 Engineering trick: NumPyro's substitution-handler pattern

NumPyro's effect-handler architecture (Pyro's lineage) treats every `numpyro.sample("x", dist)` call as an algebraic effect. Handlers (`substitute`, `condition`, `block`, `mask`, `scale`, `reparam`) wrap the generative function to alter behaviour. `condition({"x": 0.5})` replaces the sample with the observed value; `reparam(config={"x": LocScaleReparam()})` swaps centred for non-centred parameterisation; `mask(mask=[True, False, True])` zeros the contribution of masked observations.

This is **pure functional composition** and ports to Go cleanly:

```go
type Handler interface { Wrap(*Trace, RandomVar) RandomVar }
type SubstituteHandler struct{ Values map[string]float64 }
type ConditionHandler  struct{ Observed map[string]float64 }
type ReparamHandler    struct{ Config map[string]Reparam }
```

The `Trace` records every (name, distribution, value) tuple; the model executes against a chain of handlers. **Fully portable, ~200 LOC, depends on §5.1 bijectors and §5.2 JointDistribution.** This is how reality joins the modern PPL frontier *without* writing a tensor framework.

### 6.4 Zero-dep portability

Context-manager: **does not port** (Python-specific syntactic sugar). Use explicit `Model` struct.
Auto-sampler dispatch: ports as ~50 LOC switch on RV types.
Effect handlers: ports cleanly, ~200 LOC, requires bijectors first.

---

## 7. Stan — automatic differentiation, NUTS, parameter constraint declarations

### 7.1 Headline algorithm: NUTS-with-dual-averaging-and-mass-matrix-adaptation

Stan's HMC variant is the gold standard. Three innovations:

1. **No-U-Turn termination** (Hoffman-Gelman 2014): trajectory length auto-tuned per iteration via doubling-and-stopping rule that detects when momentum has reversed. Eliminates the trajectory-length hyperparameter.
2. **Dual-averaging step-size adaptation** (Nesterov 2009 / Hoffman-Gelman 2014): step size `ε` adapted during warm-up to target acceptance probability ~0.8.
3. **Mass-matrix adaptation**: during warm-up, sample covariance is estimated and used as the mass matrix for the next iterations, dramatically reducing posterior anisotropy.

Reference implementation: ~600 LOC of C++ in `stan/mcmc/hmc/nuts/`. Go port for reality: ~400 LOC (no batch dims, no GPU). 117 T2.5 named NUTS as ~250 LOC; **that estimate undercounts the dual-averaging + mass-matrix logic by ~150 LOC.** Realistic budget is 400.

### 7.2 Engineering trick: `parameters { real<lower=0> sigma; }` constraint syntax compiles to bijectors

Stan's parameter block declares constraints (`<lower=0>`, `<lower=0, upper=1>`, `simplex`, `unit_vector`, `cholesky_factor_corr`, etc.). The compiler transforms these into automatic bijector chains: `<lower=0>` → `exp` (with corresponding log-Jacobian); `<lower=0, upper=1>` → `sigmoid`; `simplex` → stick-breaking. All MCMC happens in the unconstrained space; samples are pulled back through the inverse bijector at output.

reality cannot do compile-time constraint inspection (no Stan-language compiler), but can match the **runtime** behaviour: `prob.Constrained(d Distribution, lo, hi float64)` returns a transformed distribution whose `LogProb` includes the bijector's log-Jacobian automatically. Combined with §5.1, this is ~30 LOC on top of the bijector framework.

### 7.3 Engineering trick: forward-mode + reverse-mode autodiff with stack-allocated tape

Stan AD (`stan-math`) uses a single contiguous arena allocator (the "AD stack"). Every operation pushes one node onto the stack; backward pass walks it once; a single `recover_memory()` call at the end of an iteration resets the high-water mark. **No GC, no per-node allocation**. This is the engineering pattern that lets Stan run NUTS at ~200μs/iteration on a 50-parameter model.

reality's `autodiff` per 015 autodiff-perf has **opposite** characteristics: per-op closure allocation, no Reset. **Coordinate with 015's autodiff-perf recommendation:** the Stan-style arena is the canonical fix; ~150 LOC per 015's tagged-union nodes + sync.Pool + Reset proposal.

### 7.4 Zero-dep portability

NUTS: portable, ~400 LOC.
Constraint-bijector compilation: portable as runtime API, ~30 LOC on top of §5.1.
AD stack arena: portable, ~150 LOC, **already on the autodiff roadmap per 015 — no new work specific to prob/.**

---

## 8. BlackJAX / Turing.jl — modular sampler libraries

### 8.1 Headline algorithm: kernel composition over a unified `SamplingState` type

BlackJAX (JAX-native, ~3k LOC) decomposes MCMC into **three orthogonal layers**:

1. **Kernel** (e.g., `nuts.build_kernel(logdensity_fn, step_size, mass_matrix)`): pure function `(state, rng) → (new_state, info)`.
2. **Algorithm** (e.g., `nuts(logdensity_fn, step_size)`): bundles a kernel with an `init` function and standard parameter defaults.
3. **Adaptation** (e.g., `window_adaptation(kernel, num_steps)`): runs warm-up loops over a kernel to tune step-size and mass-matrix, returning an adapted kernel.

The composition `adapt → kernel.run` is plain Python function calls; no inheritance, no classes. **This is the cleanest MCMC API in the survey.** Adding a new sampler is one `build_kernel` function. Adapters compose with any kernel.

For reality: `Kernel = func(state State, rng *rand.Rand) (State, Info)`; `Adapter` wraps a `Kernel` and tunes parameters in-place. ~80 LOC of base types + per-sampler `Kernel` constructors. Higher composability than the TFP/Stan monolith approach.

### 8.2 Turing.jl's contribution: probabilistic-programming via Julia macros

`@model function gdemo(x) m ~ Normal(0, 1); x ~ Normal(m, 1) end` — Turing's `@model` macro rewrites function bodies to thread a model context. Conceptually similar to PyMC's `with` block but more mechanical (macro expansion vs. context manager). Not portable to Go (no macros).

**However** Turing's `Sampler` / `Adaptor` / `Transition` triplet matches BlackJAX exactly and ports cleanly. **The Julia/Python convergence here is informative: both winning libraries factor MCMC as kernel + adaptation + transition.** Reality should adopt the same factoring.

### 8.3 Zero-dep portability

Kernel/Adapter composition: **fully portable**, ~80 LOC scaffolding.
Macro / context-manager sugar: **does not port**.

---

## 9. The convergent best-of-breed recommendation

Triangulating across all 8 libraries, **five engineering tricks recur in every modern stack** and are zero-dependency-portable to reality's Go-first / golden-file architecture:

1. **`Distribution` interface with default-method generic harness** (§1.1, §3.1, §4.2)
   — `LogPDF`/`LogCDF`/`LogSF`/`SF`/`Quantile`/`Sample`/`Mean`/`Var`/`Moment`/`Entropy`/`Fit` all derivable from a single `_pdf` + `_cdf` + `_rvs` triplet, with override hooks per distribution. Closes 117 T1.1+T1.2 (~50 distributions) at ~1,200 LOC instead of ~2,580 LOC.
2. **Bijector composition** (§5.1, §7.2)
   — `Exp`/`Log`/`Affine`/`SoftPlus`/`Sigmoid`/`Chain`/`Stack`/`Invert` + `TransformedDistribution`. ~250 LOC. Generates LogNormal, Logit-Normal, all constraint reparameterisations. Linchpin for MCMC (117 T2.5).
3. **Truncated-distribution wrapper** (§3.4)
   — `prob.Truncated(d Distribution, a, b float64)` 60 LOC zero-dep. Replaces ~10 hand-rolled truncated entries from 117 T1.2.
4. **MixtureModel + JointDistribution** (§3.3, §5.2)
   — Pure composition over `Distribution`. `Mixture` ~80 LOC; `JointDistribution` ~150 LOC. Linchpin for Bayesian models (117 T2.4).
5. **NumPyro-style effect handlers** (§6.3)
   — `substitute`/`condition`/`reparam`/`mask`/`scale`/`block`. ~200 LOC. Joins the modern PPL frontier without a tensor framework.

**Total: ~940 LOC for items 1-5 plus their dependencies.** This budget reshapes the 117 roadmap:

- 117 T1 (Tier 1 distributions) drops from ~2,580 LOC to ~1,500 LOC because of §1 (1,200 distribution-specific + 300 generic harness).
- 117 T2.4 (Bayesian conjugate priors) is partially absorbed into JointDistribution + bijector reparameterisation.
- 117 T2.5 (MCMC) absorbs §5+§7+§8, with bijectors as the load-bearing prerequisite.

**The single highest-leverage commit** is item 1 — the `Distribution` interface harness. Without it, every distribution in 117 T1.1/T1.2 ships ~5× more code than it needs to and the override-hook pattern that lets specialists improve any single derived method is unavailable. With it, the distribution roster fills in at ~50 LOC per distribution; the next ~30 distributions become a long-weekend sprint instead of a multi-month project.

---

## 10. What is *correctly* out of scope for reality

For completeness, the following SOTA features are surveyed and rejected as architectural mismatches:

- **JIT compilation / tracing** (TFP, JAX, NumPyro) — requires a tensor abstraction reality does not have and should not build. Go's compiler is ahead-of-time; the dynamic-tracing speedup is a Python-overhead workaround irrelevant to a compiled-Go library.
- **GPU / TPU dispatch** (TFP, NumPyro, BlackJAX) — out of architectural scope per CLAUDE.md design rule 3 (the per-frame consumer is Pistachio at 60 FPS on commodity hardware, not a research GPU cluster).
- **Sparse-tensor distributions** (TFP) — requires a sparse-tensor framework; reality has `linalg.SparseMatrix` for linear algebra but no general sparse-array.
- **Variational inference** (TFP, PyMC, NumPyro, Pyro) — defer until §1 (Distribution harness) + §5 (bijectors) + autodiff per 014/015 are all in. VI is ~400 LOC of mean-field-Gaussian-with-bijector once those exist; ranks high but **after** the foundation.
- **Stan-language compiler** — reality is a Go library, not a DSL.
- **R formula DSL** (`y ~ x1 + x2:x3`) — defer to a future `prob/formula` sub-package; the GLM/Kalman engines (§2) should accept raw `(X, y)`.
- **`lme4` / nlme mixed-effects** — defer until `linalg` ships sparse Cholesky (slot 092 territory).
- **Stochastic gradient MCMC** (SGLD, SGHMC) — niche; defer.

---

## 11. Cross-references with sibling agents

- **014 autodiff-api** `autodiff.Func` recommendation is independently confirmed by §5.3 (TFP structure-aware MCMC needs `func([]float64) (float64, []float64)`) and §7.3 (Stan AD-stack pattern). Same primitive serves both.
- **015 autodiff-perf** Stan's arena-allocator AD pattern (§7.3) is the canonical reference for 015's tagged-union + sync.Pool proposal. Reality's autodiff perf gap closes against Stan-math when 015 lands.
- **016 calculus-numerics** + **017 calculus-missing** Adaptive Gauss-Kronrod (017 Tier 1) is the dependency for `GenericMoment` / `GenericEntropy` / `GenericFit` in §1.4 and for the closed-form-CDF-less distributions in 117 T2.1 (NIG, GHyp, Stable, Tweedie). **Confirms 017's slot-1 priority.**
- **023 changepoint-sota** R1 (Hazard + ObservationModel orthogonality split) is structurally analogous to scipy's `_pdf` + `_argcheck` separation in §1.2. Same engineering pattern, two consumers.
- **116 prob-numerics** every HIGH and MEDIUM finding in 116 (LogPDF/LogSF absence, NormalQuantile precision cap, betaCF non-convergence) becomes a per-distribution override hook in §1.1's harness. **The harness is the architectural fix that makes 116's per-distribution patches small and uniform** instead of one-off hand-coded stabilisations.
- **117 prob-missing** the entire ~2,580 LOC Tier 1 budget reshapes to ~1,500 LOC under §1's harness; the entire ~800 LOC Tier 2 MCMC budget absorbs §5/§7/§8 patterns. Net 117 budget drops ~30%.

---

## 12. Triage — single highest-leverage commit

**Lock in `prob.Distribution` (interface) + the `LogDistribution`/`Quantilable`/`Sampleable` optional-extension interfaces + the `prob/generic.go` default-method harness (`GenericLogPDF`, `GenericSF`, `GenericQuantile`-via-Brent, `GenericMoment`-via-quadrature, `GenericEntropy`, `GenericFit`-via-MLE-with-`optim`).**

~300 LOC of pure additions, fully backward-compatible (existing `BetaPDF`/etc. become wrappers around `NewBetaDist().PDF(x)`), and reshapes every downstream prob/ work item from 116 and 117. **Same kind of structural decision that R's d/p/q/r naming made in 1996, scipy's `rv_continuous` made in 2003, and Distributions.jl's `Distribution{F,S}` made in 2014. Once locked in, every future distribution is ~50 LOC instead of ~250 LOC, every numerical-accuracy upgrade per 116 is a single-distribution override, and every API gap per 117 is "add this distribution" rather than "rewrite the family".**

The complementary 200-LOC commit is **bijectors + TransformedDistribution + Truncated wrapper** (§5.1, §3.4) — this unlocks all constrained-parameter reparameterisation, all truncated distributions, all MCMC unconstraining, and all change-of-variable distributions (LogNormal, Logit-Normal) at one shot. **Together (~500 LOC) these are the foundation; everything else in 117 builds on them.**

---

Report ends; ~400 lines.
