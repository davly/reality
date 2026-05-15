# 139 | timeseries-api

**Agent:** 139 of 400
**Topic:** timeseries: irregular vs regular sampling, multivariate, exogenous regressors
**Scope:** API ergonomics for the existing timeseries surface (`timeseries/garch`, `timeseries/dcc`, `prob/timeseries.go`) and the API contracts needed for the gap-list 137 / 138 enumerated. Not a duplication of 136 (numerics), 137 (missing models), or 138 (SOTA libraries).
**Date:** 2026-05-08

## TL;DR

The five concrete API decisions worth pinning **before** the Tier-1 stack from 137 lands:

1. **Time axis is implicit `index ∈ [0, N)`.** Callers pass `[]float64` only — no `time.Time`, no `dt`. Irregular sampling is delegated to **resampling helpers in `signal/`** (`signal.LinearResample`, `signal.AsofJoin`) that produce a regular grid before the model sees it. This keeps every model's API a function of a single `[]float64` (univariate) or row-major `[]float64` + `k` (multivariate) and avoids `time.Time` infecting the entire forecasting layer.
2. **Multivariate is row-major flat slice + dimension count `k`** — exactly the existing `dcc.SampleQbar(zSeries []float64, n, k int, out []float64)` shape. `[][]float64` slice-of-slices is rejected (no contiguous memory, allocation-pattern violation, no `linalg.Matrix` interop). All future multivariate models (VAR, BEKK, multivariate Kalman) inherit this shape.
3. **Exogenous regressors enter as a separate `(X []float64, k int)` matrix argument**, never bundled with `y`. Convention: `Fit(y []float64, X []float64, k int, ...)` where `k=0` (or `nil` X) means "no regressors". This matches statsmodels' SARIMAX `exog` and sktime's `X` second-argument idiom.
4. **The convergent SOTA `(Fit, Predict, Update)` triplet from 138 is realised in Go as a `Model` struct with three methods**: `Fit(y) (Fitted, error)`, `Fitted.Predict(h int) Forecast`, `Fitted.Update(yNew) Fitted`. Not a Go interface (no virtual dispatch tax in hot loops). Each forecaster ships its own `Model` and `Fitted` type; the **method-name contract** is the API uniformity, not a polymorphic interface.
5. **Forecast output is a `Forecast` struct, not `[]float64`.** Even the cheapest model (Holt linear) returns `Forecast{Point, Lower, Upper []float64; H int}`. Probabilistic by default; degenerate models leave `Lower=Upper=Point`. ~30 LOC of struct change now is much cheaper than retrofitting after every model lands.

The current `prob/timeseries.go` and `timeseries/garch`/`timeseries/dcc` API surface is **inconsistent** along all five axes. The five decisions above are zero-to-low-LOC API changes that pin the convergent SOTA pattern (138 §1) for every model added afterward.

---

## Current API audit — five inconsistencies

### A1 — `prob.ExponentialSmoothing` returns smoothed history; no forecast

```go
func ExponentialSmoothing(data []float64, alpha float64, out []float64)
```

`out[t]` is the **in-sample smoothed** value, not a forecast. To produce a forecast the caller must read `out[N-1]` and replicate it `h` times — the ES h-step-ahead forecast is constant at the last level. The function silently does nothing about this; the caller is expected to know the model. **Forecast horizon is implicit and the user must reconstruct it.**

### A2 — `prob.HoltLinear` has horizon as a parameter but no forecast object

```go
func HoltLinear(data []float64, alpha, beta float64, horizon int, out []float64)
```

Better than A1: `out[N..N+horizon-1]` holds the forecast. But `out` is a single `[]float64` that **mixes** smoothed history (`out[0..N-1]`) and forecast (`out[N..N+horizon-1]`) into one slice. There is no way for the caller to get just the forecast without index arithmetic. The trend, level, and any uncertainty estimate are dropped — the function is an output-only point estimate.

### A3 — `prob.ARIMA` returns AR/MA coefficients only — no model state, no forecast, no integration

```go
func ARIMA(data []float64, p, d, q int) ([]float64, error)
```

Returns a flat `[]float64` of length `p+q`. The differencing is in-sample only; **the function never forecasts** because there is no Box-Jenkins integration step preserved. The caller cannot recover the level series. The signature is the functional inverse of every other library's ARIMA: returns an opaque coefficient list instead of a fitted model that can predict. Per 136-N5 the MA estimator is biased anyway; the API does not even expose what the math would let it compute.

### A4 — `garch.Model` has parameters but `Fit` returns a fresh `Model` instead of a `FittedModel`

```go
func Fit(eps []float64, init Model, cfg FitConfig) (Model, FitResult, error)
func (m Model) Filter(eps, sigma2, z []float64) error
func (m Model) ForecastVariance(eps2T, sigma2T float64, h int) ([]float64, error)
```

The `Model` struct holds **only the parameters** — Omega, Alpha, Beta, UncondVar. There is no `FittedModel` that holds the most-recent `(eps²_T, sigma²_T)` pair needed for `ForecastVariance`. So callers that want to forecast must call `Filter` first, capture the last value, and pass it to `ForecastVariance`. This is the **stateless-model + stateful-history split** from R::tseries — not wrong, but it forces the caller to remember the contract and is easy to mis-use. Every other modern library (statsmodels, statsforecast, darts, sktime) wraps fitted state behind a single `Fitted` object.

### A5 — `dcc.Params.FilterSeries` writes a flattened `[]float64` of correlation matrices

```go
func (p Params) FilterSeries(zSeries []float64, n int, rSeries []float64) error
```

`rSeries` has length `n*k*k`, row-major. Caller must index `rSeries[t*k*k + i*k + j]` to get `R_t[i,j]`. This is consistent with `dcc.SampleQbar` (multivariate convention), but **not** with `garch.Filter` which writes two parallel slices `sigma2[]` and `z[]`. The `signal/`-style "scalar sequence as `[]float64`" idiom and the `linalg/`-style "row-major flat matrix" idiom are both used in the same package without a documented convention.

---

## Decision 1 — Time axis: implicit index, not `time.Time`

### The choice

Every forecaster takes `[]float64` and treats indices as `0, 1, 2, ..., N-1`. The library does not know wall-clock time. Sampling rate, period, season length, and irregular timestamps are **caller-resolved**, with helpers in `signal/` for the regular-from-irregular conversion.

### Why

1. **Zero-dep rule.** `time.Time` is in `stdlib`, so it would not violate Go's "stdlib only" stance — but cross-language goldens (Python, C++, C#) cannot interchange `time.Time` values. JSON test vectors carry `[]float64` cleanly; embedding timestamps would require a date-format spec that varies across the four target languages. **Indices in `[0, N)` are the only data type that round-trips losslessly across all four languages.**
2. **Period vs sample-rate is a model parameter, not a data parameter.** Seasonal models (SARIMA, ETS) need to know the **period in samples** (`m=12` for monthly, `m=4` for quarterly, `m=24` for hourly), not the calendar. Once you have a regular grid, the period is just an int.
3. **Mixed-cadence series are common.** Tick data, sensor data, and event streams arrive irregularly; nearly all classical TS models require regular sampling. **Resampling is the boundary** — make it explicit in caller code, not implicit in model code.
4. **Aligns with `signal/`.** `signal.FFT`, `signal.Convolve`, `signal.MovingAverage`, `signal.ExponentialMovingAverage` all take `[]float64` and treat samples as `dt = 1`. Time-series should match.

### The API surface

Add to `signal/` (sibling package, not `timeseries/`, because the math is signal-processing):

```go
// signal.LinearResample resamples (t, y) onto a regular grid of n points
// at uniform spacing using linear interpolation. Pre-condition: t is
// strictly increasing, len(t) == len(y), n >= 2. Out-of-range indices
// extrapolate with the boundary value.
func LinearResample(t, y []float64, tStart, tStep float64, n int, out []float64) error

// signal.AsofJoin joins a primary time series (tA, yA) with a secondary
// series (tB, yB) onto the primary's grid, holding the most-recent value
// of yB at or before each tA[i]. Pre-condition: tA, tB strictly
// increasing. Output written to outB, length len(tA).
func AsofJoin(tA, yA, tB, yB, outB []float64) error
```

These are 30-50 LOC each, no allocation, sit alongside `signal.MovingAverage`. Tier-1 cost is small and they unblock the whole "real-world data → regular grid" pipeline without infecting any model.

### What this does NOT do

- Does **not** introduce a `TimeSeries` container type (rejected — see Decision 2).
- Does **not** track a `time.Duration` per model.
- Does **not** ship a calendar / business-day / holiday system. **(Prophet's holiday calendar is a per-model bolt-on accepting a `[]int` of indices, not a global concept.)**

### Counter-argument considered

darts' `TimeSeries` (a typed, indexed, multivariate-aware container — 138 §darts) is genuinely useful for hierarchical / panel workflows. But it is **heavy** (~2,000 LOC of bookkeeping) and re-implements `linalg.Matrix` for a different purpose. The judgment here: defer the container until a multivariate-with-static-covariates use case actually shows up; in v1.0, `[]float64` plus index arithmetic plus `signal/` resamplers cover ≥95% of cases.

---

## Decision 2 — Multivariate: row-major flat slice + `k`

### The choice

Multivariate inputs are flat row-major `[]float64` plus a separate `k int` for dimension. Time index in major position: `data[t*k + j]` is dimension `j` at time `t`.

### Why

1. **Already the convention** in `dcc.SampleQbar` and `dcc.FilterSeries`. Pinning it for the rest of the package keeps `timeseries/` internally consistent.
2. **Matches `linalg/`.** `linalg.MatMul(A, aRows, aCols, B, bCols, out)` is row-major flat; future VAR / BEKK / multivariate Kalman models can call into `linalg` without conversion.
3. **Zero allocation.** `[][]float64` slice-of-slices forces one Go slice header per row (16 bytes × n rows = 16 KB/4096 rows wasted), defeats SIMD-friendly stride-1 inner loops, and makes the workspace-pattern (see CLAUDE.md rule 3) unimplementable.
4. **Cross-language interchange.** JSON test vectors store flat arrays; goldens for multivariate Kalman / BEKK / DCC are easier to write and validate when the on-disk shape and the in-memory shape match.

### Convention to spell out in `package timeseries` doc

```
A multivariate series of length n with dimension k is stored as a row-major
slice of length n*k. The element at time t and dimension j is at
data[t*k + j]. This matches dcc.SampleQbar / dcc.FilterSeries / linalg.MatMul.
```

### The k=1 univariate case

A `[]float64` of length `n` is the special case `k=1` row-major. **Public APIs should NOT require the caller to pass `k=1` — provide a convenience overload that takes plain `[]float64`.** Pattern:

```go
// arima.Fit (univariate)
func Fit(y []float64, p, d, q int, cfg FitConfig) (FittedARIMA, error)

// var.Fit (multivariate VAR)
func Fit(yMulti []float64, n, k, p int, cfg FitConfig) (FittedVAR, error)
```

This avoids the `[][]float64{{y}}` ugliness while keeping the multivariate interface consistent.

### Counter-argument considered

A typed `Matrix` struct (like `gonum.mat.Dense`) would carry `(rows, cols, data)` together and prevent the "caller forgot to multiply n*k correctly" bug. But the entire `reality` codebase has chosen flat slices — `linalg.MatMul` does not return a `*Matrix`, it returns `[]float64`. **Adding a `Matrix` type only in `timeseries/` would split the convention without paying back the cost.**

---

## Decision 3 — Exogenous regressors as a separate `(X, k)` matrix argument

### The choice

Forecasters that accept exogenous regressors take `X` as a separate argument, never bundled with `y`. Signature template:

```go
func Fit(y []float64, X []float64, kX int, p, d, q int, cfg FitConfig) (FittedARIMAX, error)
```

`X` has length `n*kX` row-major (Decision 2 convention), where `kX` is the exogenous dimension. `kX=0` means "no regressors"; passing `X=nil` or `len(X)=0` is the documented signal.

### Why

1. **statsmodels.SARIMAX precedent.** `sm.tsa.SARIMAX(endog=y, exog=X, order=(p,d,q))` — endogenous and exogenous are different tensors with different lengths potentially (X may extend past y for known-future regressors).
2. **sktime / fable convention.** `fit(y, X=None)`, `predict(fh, X=None)`. Same shape across 40+ forecasters.
3. **Forecast time, X must extend into the future.** ARIMAX forecasting at horizon `h` requires `X[N..N+h-1]`. If X were bundled with y, the bookkeeping is awkward; separated, the forecast call cleanly takes its own `Xfuture`:

```go
func (f FittedARIMAX) Predict(h int, Xfuture []float64) Forecast
```

4. **Aligns with darts' `past_covariates` / `future_covariates` distinction.** future_covariates (known beyond `t` — e.g., scheduled holidays, planned promotions) and past_covariates (known up to `t`, must be provided again at predict time) are first-class concepts in darts. The Go signature handles both: `Xtraining` for past-or-future covariates known during fit, `Xfuture` for the same regressor extended into the prediction horizon.

### What this does NOT do

- Does **not** auto-standardise X. Caller must demean / scale before fit. The library does not own a feature-engineering pipeline.
- Does **not** distinguish past from future covariates at the type level. Both are `[]float64` row-major; the user-facing distinction is which method consumes them.
- Does **not** ship a one-hot encoder for categorical regressors. Caller passes pre-encoded `[]float64`.

### Static covariates (per-series, time-invariant)

Out of scope for v1.0. statsmodels and darts both support per-series static metadata (e.g., "store ID, region"); the use case is hierarchical forecasting, which is Tier 3 (137 T3.3 MinT). Defer until hierarchical lands.

---

## Decision 4 — `(Fit, Predict, Update)` realised as struct + methods

### The choice

Each forecaster ships **two structs**: a `Model` (parameters + config, may be the zero value) and a `Fitted` (parameters + state needed to forecast and update). Three methods:

```go
type ARIMA struct{ P, D, Q int }            // parameters / hyperparameters
type FittedARIMA struct {
    P, D, Q int
    AR, MA  []float64
    Sigma2  float64
    LastObs []float64    // state for forecasting + update
    DiffHistory []float64 // for re-integrating differencing
}

func (m ARIMA) Fit(y []float64, cfg FitConfig) (FittedARIMA, error)
func (f FittedARIMA) Predict(h int) Forecast
func (f FittedARIMA) Update(yNew []float64) FittedARIMA
```

Same template for `garch.GARCH`, `ets.ETS`, `theta.Theta`, etc.

### Why two structs (Model + Fitted) instead of one

- `Model` is the **specification** (p, d, q, max iter, learning rate). Cheap to construct, value semantics, no hidden state.
- `Fitted` is the **artefact** of `Fit`. Carries everything needed for `Predict` and `Update`. Immutable from the caller's perspective except via the explicit `Update` method (which returns a new `Fitted`, value semantics).

The split prevents the "did I call Fit yet?" bug class. A `FittedARIMA` cannot be created without `Fit` returning one. A `Model` cannot Predict — it has no state.

### Why `Update` returns a new `Fitted` rather than mutating

- **Value semantics dominate the codebase** (CLAUDE.md rule 3, no allocations in hot paths but also no hidden mutation). Methods on value receivers is the existing convention (`garch.Model.Filter`, `garch.Model.LogLikelihood`).
- **Allocation cost is small** because `Fitted` is small (~10 fields, the slices are reused by reference if the math allows). A future allocation-free `UpdateInPlace(workspace *Workspace)` can be added if profiling demands it; the safe-by-default API is the value-returning version.

### Why not a Go interface

```go
// REJECTED:
type Forecaster interface {
    Fit(y []float64) (FittedForecaster, error)
}
type FittedForecaster interface {
    Predict(h int) Forecast
    Update(yNew []float64) FittedForecaster
}
```

- **Virtual dispatch is non-trivial overhead** in tight loops. statsforecast's measured 20-100× speedup over statsmodels comes partly from JIT specialisation; Go's analog is direct concrete-type calls.
- **Goldens are per-concrete-model**. The interface adds zero testing benefit because every model is tested against its own concrete golden file at 1e-9.
- **No polymorphic ensemble use case in v1.0.** Forecast combination (T3.4 in 137) eventually wants `[]Forecaster`, but that's the only use site — and even there, a `[]func(y []float64) Forecast` closure slice is sufficient.

The **method-name contract** (`Fit`/`Predict`/`Update`) is the API uniformity. Documentation pins it. Linters could enforce it (a simple AST check). **No interface needed.**

### Migration path for existing GARCH

```go
// CURRENT
type Model struct { Omega, Alpha, Beta, UncondVar float64 }
func Fit(eps []float64, init Model, cfg FitConfig) (Model, FitResult, error)
func (m Model) Filter(eps, sigma2, z []float64) error
func (m Model) ForecastVariance(eps2T, sigma2T float64, h int) ([]float64, error)

// PROPOSED
type GARCH struct { /* hyperparameters; (1,1) order is implicit */ }
type FittedGARCH struct {
    Omega, Alpha, Beta, UncondVar float64
    LastEps2, LastSigma2          float64    // forecast state
    Result                        FitResult
}
func (g GARCH) Fit(eps []float64, init FittedGARCH, cfg FitConfig) (FittedGARCH, error)
func (f FittedGARCH) Filter(eps, sigma2, z []float64) error
func (f FittedGARCH) Forecast(h int) Forecast
func (f FittedGARCH) Update(epsNew []float64) FittedGARCH
```

Net change: ~30 LOC of struct rename + a new `Forecast` (Decision 5) + an `Update` method that runs the recursion forward. The hot-path math is unchanged.

---

## Decision 5 — `Forecast` is a struct, not `[]float64`

### The choice

Every forecaster returns:

```go
// timeseries/forecast.go
type Forecast struct {
    H       int          // horizon
    Point   []float64    // length H, point estimates (median or mean depending on model)
    Lower   []float64    // length H, lower predictive bound at Level
    Upper   []float64    // length H, upper predictive bound at Level
    Level   float64      // confidence level for Lower/Upper, e.g. 0.95
    Samples []float64    // optional, length nSamples*H row-major; nil if model is deterministic
    NSamples int         // 0 if Samples is nil
}
```

### Why probabilistic-by-default

darts and the modern Hyndman canon (138 §R::forecast) treat probabilistic output as the primary type; point forecasts are the degenerate case. For models with closed-form prediction intervals (ARIMA via Kalman, ETS via innovation form) producing `Lower`/`Upper` is **free at fit time** — the math is already there.

For models that don't have closed-form intervals (Holt linear, simple ES, naive baselines), `Lower=Upper=Point` and `Level=0` is a documented "no uncertainty quantified" signal.

### Why one `Level` instead of multiple

darts allows `quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]` — five quantiles per timestep. For v1.0, **one symmetric interval is enough** and is the de facto industry default (`forecast::forecast()` returns 80% + 95% by default; statsmodels `get_forecast().conf_int(alpha=0.05)` returns one). Multiple-quantile support is a v2.0 extension via `Samples`:

```go
// Caller can compute any quantile from samples post-hoc:
samples := f.Forecast(12).Samples  // nSamples*H
q90 := percentile(samples, 0.9, h)  // user-side
```

### Why include `Samples` at all

Two reasons:
1. **Bayesian models** (BSTS, particle filter) produce posterior samples natively; flattening them to a point + interval throws away information.
2. **Forecast combination / ensembling** (137 T3.4) wants raw samples to compute weighted-quantile combinations; means + intervals don't combine cleanly.

For v1.0, `Samples=nil` is the common case. The struct field is there as a forward-compatible extension point.

### Allocation: who owns the slices?

Either:
- **Allocate on every `Predict` call** — clean but garbage-y.
- **Caller pre-allocates and passes `out *Forecast`** — workspace pattern, allocation-free.

Recommend **both**: `Predict(h int) Forecast` (allocating, ergonomic) and `PredictInto(h int, out *Forecast)` (allocation-free, hot-path). The workspace pattern is already in `garch.Filter` (caller-supplied `sigma2, z`) and `linalg.MatMul`. Forecasters should follow.

---

## Comparison with sibling: `signal/`

`signal/` is the closest sibling package and serves as the API style reference. Patterns:

| Pattern | `signal/` | Apply to `timeseries/`? |
|---|---|---|
| Take `[]float64`, treat indices as samples | yes (`FFT`, `Convolve`, `MovingAverage`) | **yes** (Decision 1) |
| Workspace via caller-supplied `out []float64` | yes (`FFT(real, imag)`, `Convolve(s, k, out)`) | **yes** (already done in `garch.Filter`, extend to `Forecast`) |
| Zero heap allocations in hot path | strict (CLAUDE.md rule 3) | **must adopt**; current `Filter` allocates two N-slices per `LogLikelihood` call (136-N13) |
| Stateless free functions | dominant (`signal.FFT`, `signal.Convolve`) | **partial** — forecasters need state (`Fitted` struct), but the methods on `Fitted` should be stateless reads of the struct |
| Struct types only when state is unavoidable | rare (`signal/` has no struct types currently) | **inevitable** for forecasters; minimise to `Model` + `Fitted` |
| Panic on contract violation, error on data validation | mixed (panic on `n!=power-of-2`, no error returns) | **error returns dominate `garch/`** — keep this; do not panic |

**Three divergences between current `timeseries/` and `signal/` style:**
1. `garch.Filter` returns `error`, `signal.FFT` panics. **Keep error returns** — time-series users handle data validation more carefully than signal users.
2. `garch.Model` has methods, `signal/` has free functions. **Justified** because forecasters need state.
3. `garch.LogLikelihood` allocates two N-slices internally; `signal.MovingAverage` is allocation-free. **Fix** (136-N13) — accept a `*Workspace` arg.

---

## Comparison with sibling: `chaos/`

`chaos/ode.go` ships `RK4Step(state []float64, dt float64, derivs func([]float64) []float64) []float64` — taking a closure. This is the **callback pattern** for "I'll integrate, you tell me the dynamics".

For time series, the analogous pattern would be a generic Kalman filter taking F/H/Q/R closures:

```go
// REJECTED for timeseries/kalman/:
func Filter(y []float64,
    transition func(x []float64) []float64,
    observation func(x []float64) []float64,
    Q, R []float64) ([]float64, error)
```

This works for nonlinear EKF / UKF but is overkill for **linear** Kalman, where F, H, Q, R are matrices. The 138 architectural recommendation (one Kalman primitive, statsmodels-style) suggests **a struct of matrices**:

```go
type Kalman struct {
    F, H, Q, R []float64
    Dx, Dz     int
    P, x       []float64    // state + covariance, mutated by Predict / Update
}

func (k *Kalman) Predict() error
func (k *Kalman) Update(z []float64) error
```

Pointer receiver on `Kalman` — mutation is the point of the filter, and the in-place workspace pattern is more important here than value semantics. **This is a documented exception** to the value-semantics convention; justify in the doc.

---

## Convention summary — five-line API spec for `timeseries/`

```
1. Sampling:    []float64, indices in [0, N). Use signal.LinearResample/AsofJoin
                to produce a regular grid before passing to a forecaster.
2. Multivariate: row-major flat []float64 + dimension k. data[t*k+j].
3. Exogenous:   separate (X []float64, kX int); Xfuture at Predict time.
4. Lifecycle:   ModelStruct + FittedStruct + (Fit, Predict, Update) methods.
                Method-name contract; no Go interface.
5. Output:      Forecast{Point, Lower, Upper, Level, Samples} struct.
                PredictInto(h, *out) for allocation-free hot path.
```

Every model added under 137's Tier-1/2/3 plan should obey this. **Pin this in `timeseries/doc.go` (currently absent — only sub-package doc.go files exist) before T1.1 Kalman lands.**

---

## What to deprecate / move on landing

When the new API spec lands, the existing primitives should be repositioned:

| Current | After spec | Why |
|---|---|---|
| `prob.ExponentialSmoothing` | `timeseries/ets.SES` (proper Fitted + Predict + Update) | A1 fix |
| `prob.HoltLinear` | `timeseries/ets.Holt` (same triplet + Forecast struct) | A2 fix |
| `prob.ARIMA` | **deprecated**; replaced by `timeseries/arima.ARIMA` (137 T1.2) | A3 fix; current is a stub anyway |
| `prob.levinsonDurbin` (private) | `timeseries/acf.YuleWalker` (returns AR + PACF + per-order error variance) | 136-N6 + 137 T1.5 |
| `garch.Model.Filter` (allocates) | `garch.FittedGARCH.Filter(eps, sigma2, z, *Workspace)` | A4 + 136-N13 |
| `garch.Model.ForecastVariance(eps2T, sigma2T, h)` | `garch.FittedGARCH.Forecast(h)` (state inside Fitted) | A4 fix |
| `dcc.Params` / `dcc.Update` / `dcc.FilterSeries` | `dcc.DCC` / `dcc.FittedDCC.Update` / `dcc.FittedDCC.FilterSeries` | A5 fix |

**Total LOC delta: ~200**, almost entirely renames + struct splits. The hot-path math is unchanged. The deprecation in `prob/timeseries.go` keeps ExponentialSmoothing and HoltLinear as thin wrappers that forward to `timeseries/ets/` so existing callers (Pulse, Oracle, Horizon, RubberDuck per the prob/timeseries.go header) do not break.

---

## R-pattern saturation

- **R-API-CONSISTENCY-WITHIN-PACKAGE: 1/3.** `garch/` is internally consistent (one struct, methods on it, error returns, error sentinels). `dcc/` is internally consistent. `prob/timeseries.go` is internally **inconsistent** (ExponentialSmoothing returns void with output buffer; HoltLinear mixes history and forecast; ARIMA returns coefficients only). Cross-package consistency between `garch/` and `dcc/` and `prob/timeseries.go` is poor.
- **R-FIT-PREDICT-UPDATE-TRIPLET: 0/3.** No model in the repo currently has `(Fit, Predict, Update)`. `garch.Fit` exists; `Predict` / `Update` do not. ARIMA / ETS / DCC have nothing matching the triplet.
- **R-PROBABILISTIC-FORECAST-OUTPUT: 0/3.** No forecaster returns prediction intervals. `garch.ForecastVariance` returns variance — closest analog — but no `(Point, Lower, Upper)` envelope.
- **R-EXOGENOUS-REGRESSOR-API: 0/3.** No model accepts exogenous regressors at all.
- **R-IRREGULAR-SAMPLING-API: 0/3.** No resampling helpers in `signal/`; no model handles irregular timestamps.

---

## Two-line summary

The five API decisions to pin before the Tier-1 stack from 137 lands: time axis is implicit `[0, N)` (resamplers in `signal/`), multivariate is row-major flat slice + `k` (matches `dcc/` and `linalg/`), exogenous regressors are a separate `(X, kX)` argument with a separate `Xfuture` for prediction, lifecycle is `ModelStruct + FittedStruct + (Fit, Predict, Update)` methods (no Go interface), and forecast output is a `Forecast{Point, Lower, Upper, Level, Samples}` struct allocation-friendly via `PredictInto`. Current `prob.ExponentialSmoothing/HoltLinear/ARIMA` and `garch.Model.Filter/ForecastVariance` and `dcc.Params.FilterSeries` are inconsistent along all five axes; ~200 LOC of renames + struct splits realigns them and pins the SOTA convergence pattern for every model added afterward.

---

## Progress

- 2026-05-08: 139-timeseries-api report complete; five API decisions pinned (time axis, multivariate layout, exogenous regressors, Fit/Predict/Update lifecycle, Forecast struct); five current-API inconsistencies catalogued; ~200 LOC migration plan; sibling-package comparison (signal/, chaos/) documented.
