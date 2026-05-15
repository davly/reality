# 136 | timeseries-numerics

**Agent:** 136 of 400
**Topic:** timeseries: ARIMA likelihood stability, state-space filter conditioning
**Scope:** `C:/limitless/foundation/reality/timeseries/` — `garch/garch.go` (175 LOC), `garch/fit.go` (243 LOC), `garch/doc.go`, `dcc/dcc.go` (159 LOC), `dcc/doc.go`, plus `prob/timeseries.go` (308 LOC) where the package's only ARIMA implementation actually lives.
**Date:** 2026-05-08

## TL;DR

`timeseries/` is **not** a time-series package — it is a univariate-GARCH(1,1) +
DCC-correlation pair plus three forecasting toys (ExponentialSmoothing /
HoltLinear / a one-pass ARIMA) wedged into `prob/timeseries.go`. Every named
audit topic except "ARIMA likelihood stability" reduces to "this code does not
exist": **no Kalman filter, no innovations algorithm, no ACF/PACF, no
Yule-Walker beyond a 31-line Levinson-Durbin helper, no Welch, no periodogram,
no STL/X-11/X-13, no ADF/KPSS/PP, no cross-correlation.** What does exist is
mostly numerically clean (analytic-gradient pinned to autodiff at 1e-9, log2pi
hard-coded, softmax stationarity cage, NaN/Inf UncondVar fallback) but contains
two real numerical concerns and one shape-of-objective concern, plus the
prob/ARIMA implementation is a pedagogical stub that should not be cited as
the package's ARIMA primitive.

---

## Inventory: what actually exists

```
timeseries/
  garch/   garch.go fit.go doc.go + 3 _test.go (43 tests)
  dcc/     dcc.go doc.go + 2 _test.go
prob/
  timeseries.go   ExponentialSmoothing, HoltLinear, ARIMA, levinsonDurbin
```

### `timeseries/garch` — GARCH(1,1) only

| File | Function | Notes |
|---|---|---|
| garch.go | `Model.Validate()` | Omega>0, Alpha/Beta≥0, Alpha+Beta<1; rejects NaN/Inf |
| garch.go | `Model.Filter(eps, sigma2, z)` | Forward recursion, NaN/Inf UncondVar fallback |
| garch.go | `Model.LogLikelihood(eps)` | Sum of `-0.5*(log2pi + log s2 + e²/s2)`; allocates 2 N-slices |
| garch.go | `Model.ForecastVariance(eps2T, sigma2T, h)` | Linear recursion + closed-form geometric tail |
| garch.go | `Model.Simulate(shocks, eps, sigma2)` | Pre-drawn N(0,1) shocks; deterministic given input |
| fit.go | `Fit(eps, init, cfg)` | Tikhonov-regularised gradient descent in unconstrained `theta` |
| fit.go | `unpack(theta)` | exp(θ_ω), softmax(θ_α,θ_β,θ_s) |
| fit.go | `negLogLikGrad(eps, m, theta, tikh)` | Mean nll + analytic gradient pinned to autodiff |

### `timeseries/dcc` — DCC correlation

| Function | Notes |
|---|---|
| `EngleDefault()` | α=0.05, β=0.93 industry defaults, no Qbar/K |
| `Params.Validate()` | α+β<1, K≥1, len(Qbar)=K² |
| `SampleQbar(zSeries, n, k, out)` | (1/n) Σ z zᵀ — biased MLE estimator |
| `Params.Update(z, Q, qOut)` | Q ← (1-α-β)Q̄ + α·zzᵀ + β·Q |
| `CorrelationFromQ(Q, k, rOut)` | R = D⁻¹/² Q D⁻¹/² |
| `Params.FilterSeries(z, n, rSeries)` | Multi-step recursion, init Q=Q̄ |

### `prob/timeseries.go` — three forecasting toys

| Function | Status |
|---|---|
| `ExponentialSmoothing(data, α, out)` | Textbook SES; clamps α ∈ (0,1] silently |
| `HoltLinear(data, α, β, h, out)` | Textbook double-exp; clamps both rates silently |
| `ARIMA(data, p, d, q)` | **Pedagogical stub** — see N5 below; not a true ARIMA fitter |
| `levinsonDurbin(autocorr, p)` | Reflection-coefficient clamp `[-1,1]`; no PACF return |
| `arimaAutocovariance(data, mean, k)` | Unused dead code (private) |
| `sigmoid(x)` | Belongs elsewhere; out of topic scope |

**Test count:** GARCH 43 tests + DCC tests, all passing. No `testdata/` golden
files for either subpackage (verified absence via Glob `timeseries/**/testdata/**`).

---

## Out of scope because absent

The audit topic enumerates the canonical time-series primitives. None of the
following exist anywhere under `timeseries/`, and grep across the repo confirms
the only adjacent code lives in unrelated packages (`signal/fft.go` for FFT,
`audio/pitch/autocorrelation.go` for pitch-detection autocorrelation):

- **State-space / Kalman filter / smoother** — no `Kalman*`, no Joseph-form
  covariance update, no square-root variant, no information filter, no RTS
  smoother. `signal/` does not have these either.
- **Innovations algorithm (Brockwell-Davis)** — absent. Would be the standard
  exact-likelihood path for ARMA before reaching for state-space; not present.
- **ACF / PACF (sample autocorrelation, partial autocorrelation)** — only
  inline computation inside `ARIMA()` (lines 173-180 of `prob/timeseries.go`),
  and a private `arimaAutocovariance` that is dead code. No exported
  `prob.ACF` / `prob.PACF` / `signal.ACF` exists.
- **Yule-Walker / Durbin-Levinson** — `levinsonDurbin` is implemented but is
  **unexported**, takes `autocorr` not data, returns AR coefficients only
  (no PACF, no innovation variance series, no order selection by FPE/AIC).
- **Spectral analysis: Welch, periodogram, multitaper** — absent. `signal/`
  ships FFT and PowerSpectrum but no segmenting/averaging/multitaper.
- **Decomposition: STL, X-11, X-13ARIMA-SEATS, classical / additive / mult.** — absent.
- **Unit-root tests: ADF, KPSS, Phillips-Perron, DF-GLS, Zivot-Andrews** — absent.
- **Cross-correlation, lagged correlation, CCF** — absent (signal has
  `Convolve` but no normalised CCF).
- **Trend tests: Mann-Kendall, seasonal Mann-Kendall, Theil-Sen** — absent
  (Theil-Sen exists nowhere in the repo per `grep -i theil`).
- **State-space subspace ID (SSI/N4SID), Whittle likelihood, Burg AR, MLE-AR
  via Hannan-Rissanen** — absent.

Each of these is in scope for a zero-dep Go math library and is canonical in
statsmodels.tsa, R::stats / R::forecast, MATLAB Econometrics Toolbox.
**Topic coverage of the actual codebase: ~5%.**

---

## Numerical findings on what exists

### N1 — GARCH(1,1) log-likelihood: log2pi present, but no per-step Kahan; no log-sum-exp guard

`garch.go:101` and `fit.go:177` both define
`const log2pi = 1.8378770664093454835606594728112` correctly to 16 digits.
`LogLikelihood` accumulates the sum naively (`ll -= 0.5*(...)`) with no
compensated summation — at N=10⁶ daily observations the accumulated
roundoff is `~N·ε·|term|` ≈ 10⁶·2.22e-16·5 ≈ 1e-9 absolute, which is
well below the 1e-7 default `AbsTol` of the Fit loop. Not a bug at the
target scale. **Not worth fixing** unless a future consumer pushes N>10⁸.

`fit.go` works with the **mean** nll (`nll *= invN` at line 215), which is
the correct decision for keeping the learning rate sample-size-independent
and is documented in the comment at lines 185-187. This is a quiet good
practice that the comment block deserves credit for — most textbook
references default to the sum and break LR transferability.

There is **no log-sum-exp** anywhere in the package because the GARCH
Gaussian likelihood does not need one (single-mixture, no weighting); flag
this only because the topic prompt enumerates "MLE numerical stability"
and the absence here is correct, not a gap.

### N2 — `negLogLikGrad` analytic gradient is pinned to autodiff at 1e-9 — this is the saturation witness

`autodiff_test.go:TestNegLogLikGrad_AutodiffEquivalence` runs the same
GARCH(1,1) forward graph through both the hand-rolled analytic gradient
in `fit.go` and a reverse-mode autodiff tape; per-coordinate gradient
agreement is asserted at `1e-9`. The forward graph in the autodiff
companion (`negLogLikGradAutodiff`) **treats the initial unconditional
variance as a constant** at lines 134-141 — comment is honest about it
("ignores the gradient path through the initial condition, which is
standard for QMLE on long series since the initial-condition contribution
decays geometrically with beta < 1"). For GARCH(1,1) with β<1 the
contribution decays geometrically and at the test's N=200 with β=0.90 the
truncation is ~0.90^200 ≈ 10⁻⁹ which sits at the test's tolerance floor.
**This is a saturated R-CLOSED-FORM-PINNED-TO-AUTODIFF cell at 3/3** in
the cross-package consumer pattern that 011-autodiff-numerics flagged as
the top autodiff pattern.

### N3 — `Filter` recursion has no overflow guard; for shock-cluster series sigma² can blow up before `Validate` catches it

The forward recursion `s2 = Omega + Alpha*prevEps² + Beta*prevS2` will
silently produce `+Inf` if a single eps is so large that `prevEps²`
overflows (any `|eps|>1e154` in float64). Subsequent `z[i] = e/sqrt(s2)`
becomes `0/0=NaN` and `LogLikelihood` returns NaN with no error. This is
**defensive-error territory**, not a bug for any real financial-returns
series (eps is typically 0.001-0.05) but the `Validate` call at line 56
checks the **model**, not the **data**. A `math.IsInf(prevEps, 0)` check
or an `eps²` saturation clamp would make `Filter` total. Not a topic-1
priority but the absence is worth noting because the prompt specifically
asks about "filter conditioning."

### N4 — `Filter`'s NaN/Inf `UncondVar` fallback is correct; tests pin it

`garch.go:68-71` — `if prevS2 <= 0 || math.IsNaN(prevS2) || math.IsInf(prevS2, 0)`
falls back to the implied `Omega/(1-Alpha-Beta)`. The same pattern is in
`Simulate` at line 160. Three tests pin it
(`TestFilter_NegativeUncondVar_FallsBackToImplied`,
`TestFilter_NaNUncondVar_FallsBackToImplied`,
`TestFilter_InfUncondVar_FallsBackToImplied`). Solid.

### N5 — `prob.ARIMA` is a pedagogical stub, not a real ARIMA fitter — and its MA estimation is wrong

`prob/timeseries.go:125-224`. Three problems compounding:

1. **MA coefficients computed from residual autocorrelation directly**
   (line 218: `coefficients[p+k-1] = sum / (float64(n) * resVar)`). This
   is the method-of-moments AR-residual heuristic, which is **biased and
   inconsistent** for true ARMA processes — there is no Hannan-Rissanen
   two-stage, no innovations-algorithm MA estimation, no MLE. Box-Jenkins
   1970 §6.3 explicitly disclaims this approach.
2. **Differencing is in-sample only** — series shrinks by `d` each pass,
   no integration on forecast (`prob.ARIMA` returns coefficients only,
   never forecasts). Caller cannot reconstruct the level series.
3. **Levinson-Durbin reflection coefficients are clamped `[-1,1]`**
   (`prob/timeseries.go:250-255`). This silently masks non-positive-definite
   sample autocovariance matrices — the standard fix is to abort the
   recursion and return the prefix that did fit, not to truncate the PARCOR
   value into the unit interval. The clamp keeps the recursion from
   diverging numerically but produces a **wrong AR fit** that the caller
   has no way to detect.

The function should either be rewritten to use a real ARMA likelihood
(state-space + Kalman filter — see "absent" list above) or be marked
deprecated and moved out of `prob`. As-is, it is the only thing the repo
calls "ARIMA" and it does not deserve the name.

### N6 — `levinsonDurbin` returns AR coefficients but discards the byproduct PACF and the per-order prediction error variance

Lines 233-277 of `prob/timeseries.go`. The Levinson-Durbin recursion
produces, at each step `i`:
- the AR(i) coefficient vector (kept, returned),
- the i-th partial autocorrelation `lambda` (dropped),
- the i-th prediction error variance `e *= (1 - lambda*lambda)` (dropped).

A canonical Levinson implementation returns all three in one pass — the
PACF is the principal output for ACF/PACF diagnostic plots and the error
variance is the input to AIC/BIC order selection. **Both come for free
from the recursion already running; just extending the return shape closes
the gap.** This is a 12-LOC fix that would unblock a future `ACF`/`PACF`/
`AROrderSelect` API at zero math cost.

### N7 — `levinsonDurbin` comment at lines 271-273 contradicts its body

Comment: "no negation needed". Body: `result := make([]float64, p); copy(result, a); return result`. Correct — but the comment claims this is because Levinson-Durbin already produces the prediction filter sign. Cross-checking against Brockwell-Davis 5.1.1, the standard recursion produces coefficients `phi_i` such that `x_t = Σ phi_i x_{t-i} + eps_t`; this is consistent with the body's no-negation. So the body is right. The comment is right too, but the comment's three-line "Negate to match standard AR convention... no negation needed" reads ambiguously enough that it sounds like a TODO. **Tighten the comment to one line; do not change the math.**

### N8 — `dcc.SampleQbar` uses biased MLE divisor `1/n`, not unbiased `1/(n-1)`

`dcc.go:72`. `SampleQbar` divides by `n`, not `n-1`. For DCC's role as a
**target** in the recursion `Q_t = (1-α-β)Q̄ + αzzᵀ + βQ_{t-1}` this is
arguably correct — Engle 2002 §3 uses the MLE divisor — but the docstring
at line 49 silently asserts the formula without naming the bias choice.
**Documentation gap, not a bug.** A one-line note "(biased MLE; multiply
by n/(n-1) for an unbiased target)" closes it.

### N9 — `dcc.CorrelationFromQ` checks diagonal positivity but not symmetry / PSD

`dcc.go:107-125`. The function asserts `Q[i,i] > 0` (line 114) but does
not assert `Q` is symmetric, does not assert `Q` is positive semi-definite.
For DCC the recursion `Q = (1-α-β)Q̄ + α·zzᵀ + βQ_{t-1}` with α,β≥0,
α+β<1, Q̄ PSD, zzᵀ PSD, Q_{t-1} PSD preserves PSD by induction (convex
combination of PSD matrices), so the check is structurally unnecessary —
**but** the function is exported, callable on caller-supplied `Q`, and
will silently produce a non-Hermitian "correlation" matrix R if the input
is non-symmetric. A `panic`-or-`error` on `|Q[i,j]-Q[j,i]| > tol` is the
defensive check. **Low priority** — DCC's internal callers always pass
the recursion output.

### N10 — `dcc.Update` recurrence preserves PSD analytically but float roundoff can produce R[i,j] slightly outside `[-1, 1]`

The DCC recursion is mathematically a convex combination of PSD matrices
(see N9), but float roundoff in `(1-α-β)*Qbar[i,j] + α*zi*z[j] + β*Q[i,j]`
can land R[i,j] outside `[-1, 1]` by `~ε·k` after the diagonal
normalisation. No clamp anywhere in `CorrelationFromQ`. Downstream
consumers who reach for `acos(R[i,j])` (e.g., to compute angular
correlation distance) get NaN. **2-LOC fix:** clamp `rOut[i*k+j]` to
`[-1, 1]` after the off-diagonal computation, only for `i != j`. Pinned
diagonal at 1.0 (line 121 implicitly does this via Q[i,i]/sqrt(Q[i,i])²=1
but only up to one ULP).

### N11 — Tikhonov regularisation in `Fit` is on the unconstrained theta, not the GARCH parameters — this is documented but worth a numerical sanity note

`fit.go:240`: `nll += 0.5 * tikh * (theta[0]² + theta[1]² + theta[2]² + theta[3]²)`.
With `theta_omega = log(omega)`, `theta_a = log(alpha/slack)`,
`theta_b = log(beta/slack)`, `theta_s = 0`, the penalty pulls
`omega → 1`, `alpha = beta = slack = 1/3`. This is **not a Bayesian
prior on financial-returns parameters** — `omega → 1` is enormous (typical
financial omega is 1e-6). At the default `tikh=1e-4` and `n=5000`, the
data nll dominates the penalty by ~5 orders of magnitude so the bias is
negligible, but at small n (close to the n>=50 floor) the penalty pulls
the optimum significantly. The doc-comment says "stabilises ill-posed
calibration" without quantifying what "small" tikh means relative to data.
**This is more an API doc gap than a numerical bug**; the math is consistent
with itself. Recommend either a doc-block note "for n >> 100·sqrt(tikh^-1)
the penalty is negligible" or a switch to penalising `(omega - omega_0)²`
in the original parameterisation.

### N12 — `Fit` gradient descent is fixed-LR with no line search

`fit.go:103-137`. Single fixed step `theta[i] -= lr * grad[i]`. No backtracking, no Armijo / Wolfe condition, no momentum, no Adam. Convergence
on the test fixture (`TestFit_RecoversApproximateParameters`) takes up to
2000 iterations on N=5000 data — slow but acceptable as a demo. The
log-likelihood is **not monotone** under fixed-LR descent in this
parameterisation — there is no Lyapunov-style "cost decreased this step"
guard. The L-BFGS in `optim/lbfgs.go` would be the cleaner choice for a
4-parameter unconstrained problem; **the Fit comment block at fit.go:34-53
even names "Newton-CG with adjoint gradients" as the target method, but
the implementation is plain GD.** The autodiff test (`autodiff_test.go`)
explicitly says "the analytic gradient stays in production for speed
(no tape allocation per Fit iteration)" — that argument applies equally
to L-BFGS, which would only need the gradient that's already computed.
**Recommend: switch Fit to call `optim.LBFGS` with `negLogLikGrad` as
the user function.** ~30 LOC of rewiring; would close the iter-count gap
by a factor of 10-50× and make convergence robust to parameter scaling.

### N13 — `Filter` allocates two N-slices per `LogLikelihood` call

`garch.go:96-97`: `sigma2 := make([]float64, n); z := make([]float64, n)`.
For every `Fit` iteration this is two `n*8`-byte allocs (at n=5000 that's
80 KB of garbage per iteration × 2000 iterations = 160 MB churn). Workspace
buffer pattern (a-la `signal.FFT(real, imag)` and the topic 026/029 chaos
RK4 workspace finding) closes this. `LogLikelihood` could accept an
`out *Workspace` arg or compute the recursion inline without allocating.
This is a CLAUDE.md rule-3 violation in a hot path. **Owned by perf
agent if there is one; flagging here only because it intersects with the
N12 fix.**

---

## Priorities

If `timeseries/` is to grow into the package the audit prompt assumes,
the gap-closing order is roughly:

1. **Build a state-space/Kalman primitive in `timeseries/kalman/`** —
   one `Kalman{F, H, Q, R, P, x}` struct with `Predict()` / `Update(z)`
   methods, Joseph-form covariance, square-root variant for ill-conditioned
   `P`. ~250 LOC. Unblocks: exact ARMA likelihood, dynamic linear models,
   structural time-series, every state-space book chapter.
2. **Replace `prob.ARIMA` with `timeseries/arima.ARIMA`** built on the
   Kalman primitive, including proper differencing inversion for forecast
   and exact MLE via Whittle or Kalman likelihood. Mark `prob.ARIMA`
   deprecated, do not delete. ~400 LOC.
3. **Promote `levinsonDurbin` to public `timeseries/yulewalker.YuleWalker`**
   that returns AR coeffs, PACF, and per-order residual variance in one
   pass. Wire `prob.ACF`/`prob.PACF` to it. ~80 LOC. (N6.)
4. **Add `signal.Welch` / `signal.Periodogram`** in the signal package
   (not timeseries), backed by existing `signal.FFT`. ~120 LOC. Unblocks
   spectral diagnostics, ARMA whiteness tests, change-point spectra.
5. **Add unit-root tests** `ADF`, `KPSS`, `PP` in `timeseries/unitroot/`
   (the ADF critical values are non-trivial — MacKinnon 2010 polynomial
   surface, not just hard-coded constants). ~200 LOC.
6. **Switch `garch.Fit` to L-BFGS** via `optim/lbfgs` (N12). ~30 LOC of
   rewiring + delete the GD loop. Cleanest first commit.
7. **Add allocation-free `garch.Filter`** + `garch.LogLikelihood(eps,
   workspace *Workspace)` (N13). ~30 LOC.
8. **Cross-correlation `signal.CCF` / `signal.LaggedCorrelation`**
   in signal (~40 LOC). Closes a topic gap with minimal scope creep.
9. **Decomposition** (STL etc.) is the largest absent layer at ~600+ LOC
   and probably warrants its own scope discussion before implementation.

Items 6/7/3 could land tonight; items 1/2/4/5 are weekend-scale work each.

---

## What was looked at but is not a bug

- `garch.ForecastVariance` closed-form geometric-decay tail is exactly correct
  (test `TestForecastVariance_ClosedForm` pins it to 1e-12, sane).
- `garch.Simulate` initial-shock fallback (lines 158-162) when `UncondVar≤0`
  uses the implied long-run variance for **both** `prevS2` and `prevEps2` —
  this is the correct stationarity-consistent initialisation.
- `dcc.FilterSeries` initialises `Q` from `Qbar` (line 145), not from the
  identity, which is the standard DCC initialisation choice.
- `dcc.Update` aliasing safety (`Q` and `qOut` may alias, doc claim at line
  85) is **correct** because the inner loop never reads `qOut[i,j]` — every
  RHS reference is to `Q[i,j]` and `Qbar[i,j]`. Pinned by inspection.
- The `sigmoid` helper in `prob/timeseries.go:300-307` uses the
  positive-x / negative-x branching that prevents `exp(|x|)` overflow —
  textbook-correct.

---

## Saturation

- **R-CLOSED-FORM-PINNED-TO-AUTODIFF: 3/3 saturated** for GARCH negLogLikGrad
  (autodiff_test.go pins analytic gradient to autodiff at 1e-9). Already
  named in the recent commit log (3b8413a area).
- **R-IEEE-754-EDGE-CASES: 1/3** for GARCH (Validate covers NaN/Inf for
  Omega/Alpha/Beta, Filter covers NaN/Inf UncondVar; missing: shock-cluster
  overflow N3, denormal eps).

---

## Two-line summary

`timeseries/` is GARCH(1,1)+DCC and `prob.ARIMA` is a stub: the topic prompt
enumerates Kalman/ACF/PACF/Welch/STL/ADF/KPSS but **none exist**, so 95% of
this audit is gap-mapping rather than bug-hunting. What exists is mostly
clean — analytic-gradient-pinned-to-autodiff at 1e-9, NaN/Inf UncondVar
fallback, log2pi to 16 digits — with two real numerical concerns
(prob.ARIMA MA estimation is biased/inconsistent, and Levinson-Durbin
PARCOR clamping silently masks non-PSD autocovariance) and a workspace-pattern
violation in Fit (two N-slice allocs per LogLikelihood call); switching Fit
to L-BFGS and exposing PACF + per-order residual variance from the existing
recursion are the two zero-risk closes worth landing tonight.
