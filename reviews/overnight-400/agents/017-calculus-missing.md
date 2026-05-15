# 017 | calculus-missing

**Scope.** What `reality/calculus` *should* ship that it does not. Agent 016 already audited the existing 5-function surface (`NumericalDerivative`, `NumericalGradient`, `TrapezoidalRule`, `SimpsonsRule`, `GaussLegendre 2-5pt`, `MonteCarloIntegrate`) and the package-doc misclaims; this report is the additive axis only.

**TL;DR.** `calculus` covers the *first three rows* of a numerical-methods textbook (Newton-Cotes + low-order Gauss + flat MC) and stops there. Of the ~28 canonical algorithm families a "numerical differentiation and integration" library is expected to ship in 2026, **the package implements 5 (~18%)**. The single most consequential missing primitive is **adaptive Gauss-Kronrod (G7-K15)** — it is the QUADPACK `qag` workhorse, the SciPy default `quad`, and the Boost.Math `gauss_kronrod` default, and it is the *only* general-purpose scalar quadrature that ships an honest error estimate alongside the value. Without it (or Romberg), the package cannot satisfy CLAUDE.md design rule "Precision documented, not assumed" for any non-polynomial integrand. Tier 1 below is **6 algorithms (~600 LOC, ~120 golden vectors)** that close the headline gaps named in the topic; Tier 2 adds **9 algorithms (~900 LOC)** for special-weight, oscillatory, and singular integrals; Tier 3 is **6 advanced topics (~1,200 LOC)** for spectral / multi-D / IFT-based work that may belong in sibling packages. Notable side findings: zero downstream code in the repo currently imports `calculus.*` (only its own tests), so the API contract is still negotiable; `calculus` overlaps `chaos/ode.go` (RK4) and `optim/rootfind.go` (Newton/bisection/Brent) but `CLAUDE.md` *advertises* both as living in `calculus` — Tier 1 should resolve that boundary explicitly, not by re-shipping.

---

## 1. Decision frame

Three axioms drive what belongs in `reality/calculus` versus a sibling:

1. **Reality is a math library, not a solver framework.** Anything that requires assembling a sparse matrix, dispatching on stiffness, or owning an event loop belongs in `optim/`, `chaos/`, or a new sibling. Single-call quadrature/differentiation routines belong here.
2. **Every primitive must be golden-file-testable from `math/big`.** This excludes pseudo-random methods that depend on a specific RNG, but `MonteCarloIntegrate` already accepts an injected `rng interface{ Float64() float64 }` so deterministic Sobol/Halton sequences are fine.
3. **Convergence theorems are part of the contract.** A method without a published error formula on a stated function class (e.g. "Romberg gives O(h^{2k}) for C^{2k} integrands") fails CLAUDE.md design rule 5 ("Precision documented, not assumed") and should not ship.

These axioms cleave cleanly: Tier 1 is the QUADPACK / Boost.Math.Quadrature scalar adaptive core; Tier 2 is the special-class quadrature surface (oscillatory, singular, semi-infinite); Tier 3 is multi-D / spectral / adjoint territory that legitimately overlaps `optim/`, `signal/`, and `autodiff/`.

---

## 2. Catalog of missing primitives (master list)

Cross-referencing the topic prompt with QUADPACK, Boost.Math.Quadrature 1.84, scipy.integrate 1.13, GSL 2.7, ALGLIB, and Mathematica 13's `NIntegrate` method registry yields the following gap matrix. "Where canonical" cites a representative reference implementation; "LOC" is a Go-port estimate from reading the canonical source.

| # | Algorithm | Where canonical | What it solves | LOC | Tier |
|---|---|---|---|---|---|
| 1 | Adaptive Gauss-Kronrod (G7-K15) | QUADPACK `qag/qags`, Boost `gauss_kronrod` | General scalar quadrature with error estimate | ~180 | 1 |
| 2 | Romberg (Richardson on trapezoid) | NR §4.3, GSL `gsl_integration_romberg` | Smooth integrands, exponential convergence | ~80 | 1 |
| 3 | Adaptive Simpson with ε-recursion | Lyness 1969, McKeeman recursive | Drop-in upgrade of `SimpsonsRule` | ~60 | 1 |
| 4 | Tanh-sinh (double-exponential) | Takahasi-Mori 1974, Boost `tanh_sinh` | Endpoint singularities, semi-infinite | ~140 | 1 |
| 5 | Richardson extrapolation (general) | Burden & Faires §4.2 | Reusable error-killer | ~40 | 1 |
| 6 | Five- and seven-point central / one-sided diff | Fornberg 1988 | O(h⁴), O(h⁶) derivatives | ~80 | 1 |
| 7 | Complex-step differentiation | Squire-Trapp 1998, Martins 2003 | O(eps) derivative, no cancellation | ~30 | 1 |
| 8 | Gauss-Legendre arbitrary n | Golub-Welsch 1969 | Removes the {2..5} clamp | ~120 | 2 |
| 9 | Gauss-Hermite | Stroud-Secrest 1966 | ∫ e^{-x²} f(x) dx on (-∞,∞) | ~90 | 2 |
| 10 | Gauss-Laguerre | Stroud-Secrest 1966 | ∫₀^∞ e^{-x} f(x) dx | ~90 | 2 |
| 11 | Gauss-Chebyshev (1st & 2nd kind) | Abramowitz §25.4 | ∫ f(x)/√(1-x²) dx, closed-form nodes | ~60 | 2 |
| 12 | Gauss-Lobatto / Kronrod-Lobatto | Gander-Gautschi 2000 | Endpoints included; adaptive variant | ~120 | 2 |
| 13 | Clenshaw-Curtis (FFT-based) | Trefethen 2008 | n×log n, near-Gauss accuracy | ~140 | 2 |
| 14 | Filon quadrature | Filon 1928 | Oscillatory ∫ f(x) cos(ωx) dx | ~80 | 2 |
| 15 | Levin collocation | Levin 1996 | Highly oscillatory, ω-independent cost | ~150 | 2 |
| 16 | IMT / Sidi / mIMT transformations | IMT 1970, Sidi 1993 | Endpoint singularities pre-quadrature | ~110 | 2 |
| 17 | Higher-derivative finite differences (n ≥ 2) | Fornberg 1988 weights | f''(x), f'''(x), arbitrary order | ~70 | 2 |
| 18 | Spectral differentiation (Chebyshev / Fourier) | Trefethen *Spectral Methods* | Exponentially-accurate f' on grid | ~130 | 3 |
| 19 | Sparse-grid (Smolyak) cubature | Smolyak 1963, Gerstner-Griebank 1998 | High-D smooth integrands | ~250 | 3 |
| 20 | Adaptive cubature (multi-D h-adaptive) | Berntsen-Espelid-Genz 1991 (Cubature) | Low-D rough integrands | ~280 | 3 |
| 21 | QMC (Sobol / Halton / lattice) | Sobol 1967, Niederreiter | Deterministic sub-MC stderr | ~180 | 3 |
| 22 | Importance-sampled MC + VEGAS | Lepage 1978 | High-D peaked integrands | ~200 | 3 |
| 23 | Pruess-Crockett step-size selection | Pruess-Fulton 1986 | Auto-h for finite-diff | ~60 | 3 |
| 24 | Adaptive RK45 / DOPRI5 | Dormand-Prince 1980 | Already in `chaos/` — flag, do not duplicate | — | — |

24 distinct primitives; `calculus` ships 5. Six (#1, #2, #3, #5, #6, #7) are textbook table-stakes that close the topic prompt's named gaps and are each ≤180 LOC.

---

## 3. Tier 1 — must-ship (≈ 6 primitives, ≈ 600 LOC)

The minimum surface for `calculus` to call itself a numerical-methods library rather than a textbook chapter.

### 3.1 `AdaptiveGaussKronrod(f, a, b, tol) (val, errEst, evals, ok)` ⭐ flagship

The single highest-leverage addition. G7-K15 means the 7-point Gauss rule is embedded in the 15-point Kronrod rule, so one set of evaluations gives both the value (K15) and an error estimate (|G7−K15|·factor). Adaptive subdivision with worst-error-first heap mirrors QUADPACK `qag`.

- **Why it's the QUADPACK staple.** The Gauss-Kronrod construction (Kronrod 1964) was specifically designed so that *adding* n+1 points to an n-point Gauss rule gives a 2n+1-point rule of degree 3n+1, while reusing the Gauss evaluations — i.e. you get an error estimate **for free in evaluations**. No other scalar method has this property.
- **Higher-order pairs to consider.** G10-K21, G15-K31, G20-K41, G25-K51, G30-K61 — QUADPACK exposes all six. Recommend shipping G7-K15 (cheap, default), G15-K31 (smooth analytic), G30-K61 (heavy oscillation). Node tables are precomputed constants — golden-file friendly.
- **Reference impl.** Boost `boost/math/quadrature/gauss_kronrod.hpp` is ~400 LOC C++; Go port is ~180 LOC because Go's `container/heap` removes a lot of boilerplate.
- **Tolerance contract.** Return `(val, errEst, evals, ok)` where `ok = errEst ≤ max(absTol, relTol·|val|)`. This is what Mathematica's `NIntegrate[..., "Method"->"GaussKronrod"]` does. Agent 016 noted `MonteCarloIntegrate` returns no stderr — this primitive sets the precedent that all adaptive integrators in the package ship error estimates.

### 3.2 `Romberg(f, a, b, maxK, tol) (val, errEst, ok)`

Trapezoid + Richardson extrapolation. For smooth integrands this is exponentially convergent in K (number of refinement levels). Drop-in upgrade for `TrapezoidalRule` whenever the integrand is C^∞.

- **Why it ships alongside Gauss-Kronrod.** Romberg is the right tool when you can refine the trapezoid rule cheaply (e.g. tabulated data already sampled on a 2^k grid); Gauss-Kronrod is the right tool when you can pay for free-form evaluation. Both belong.
- **Reference.** GSL `gsl_integration_romberg`; NR §4.3.
- **LOC.** ~80, dominated by the Neville-style extrapolation table.

### 3.3 `AdaptiveSimpson(f, a, b, tol, maxDepth) (val, errEst, ok)`

ε-recursive Simpson's rule. For each interval, compare `S(a,b)` against `S(a,m) + S(m,b)`; if difference < 15·tol, accept; else recurse on each half. McKeeman's classic 30-line implementation.

- **Why it's a Tier 1 add despite being "just" Simpson.** The current `SimpsonsRule(f, a, b, n)` requires the caller to *choose* `n` blind — it has no way to know whether n=100 was enough. An adaptive variant returns when the error is below a stated tolerance, satisfying CLAUDE.md rule 5.
- **Reference.** Lyness 1969 "Notes on the Adaptive Simpson Quadrature Routine."
- **LOC.** ~60.

### 3.4 `TanhSinh(f, a, b, tol) (val, errEst, ok)`

Double-exponential transformation. Uses `x = tanh(π/2 · sinh(t))` to map a finite interval to (-∞,∞), then trapezoid on the transformed grid. **The right tool for endpoint singularities** like `∫₀^1 ln(x)/√x dx` where Gauss-Kronrod struggles.

- **Why it's Tier 1, not Tier 2.** Endpoint singularities are extremely common in physics integrands (Coulomb potential, blackbody, radiative transfer). Without it, `calculus` users fall back to symbolic regularization or hand-coded substitutions.
- **Variants.** Sinh-sinh for (-∞,∞), exp-sinh for [a,∞). Boost ships all three; recommend tanh-sinh + a `Method` enum field.
- **Reference.** Takahasi-Mori 1974, Bailey 2005 "A Comparison of Three High-Precision Quadrature Schemes."
- **LOC.** ~140 including precomputed abscissae table.

### 3.5 `Richardson(eval func(h float64) float64, h, ratio float64, k int) float64`

Generic Richardson extrapolation. Given a sequence `eval(h), eval(h/r), eval(h/r²), ...`, kill the leading error term to get O(h^{p+k}) convergence. Used internally by Romberg and externally by anyone refining a finite-difference derivative.

- **Why it's a publicly-named primitive.** Agent 016 lists it as missing; the optim and infogeo packages would consume it directly for adaptive step refinement.
- **Reference.** Burden & Faires §4.2; Joyce 1971.
- **LOC.** ~40, including a `RichardsonTable` helper for the Neville triangle.

### 3.6 `NumericalDerivative5pt(f, x, h)`, `NumericalDerivative7pt(f, x, h)`, `NumericalDerivativeForward(f, x, h)`, `NumericalDerivative2nd(f, x, h)`, `NumericalDerivativeForwardOrder(f, x, h, order)`

The finite-difference family. 5-point central is `(-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h))/(12h)`, O(h⁴). 7-point is O(h⁶). One-sided versions for boundary/log/sqrt domains. Second derivative for Newton's method.

- **Why grouped.** All share the same Fornberg-1988 coefficient-generation algorithm; ship one private `fornbergWeights(order, stencil)` helper and expose six wrappers.
- **Reference.** Fornberg 1988 *Math. Comp.* 51:699-706 — generates arbitrary-order finite-difference coefficients on arbitrary grids in ~30 lines.
- **LOC.** ~80 total.

### 3.7 `ComplexStepDerivative(f func(complex128) complex128, x, h float64) float64`

`f'(x) ≈ Im(f(x + ih)) / h`. **Exact to machine precision** with no cancellation, because there is no subtraction of nearly-equal real numbers. The catch: `f` must be differentiable in the complex sense, which restricts to non-`abs`/`max`/`min` integrands.

- **Why it ships in Tier 1.** It is the *only* finite-difference-grade method that does not lose half the bits to cancellation. For consumers who can write their integrand in complex arithmetic (most physics, most prob distributions), it is strictly superior to central differences.
- **Cross-package.** Agent 011's autodiff audit flagged the complex-step path as missing; this is the function that would land it in `calculus` for the cases where a tape is overkill.
- **Reference.** Squire-Trapp 1998 *SIAM Review* 40(1):110-112; Martins-Sturdza-Alonso 2003.
- **LOC.** ~30 including a `ComplexStepGradient` n-D wrapper.

**Tier 1 totals.** 6 algorithm families, ~610 LOC, ~120 new golden vectors (CLAUDE.md target ≥20 each), 0 new dependencies. Closes every algorithm named in the topic prompt's "Integration" and "Differentiation" lists except oscillatory and special-weight (Tier 2).

---

## 4. Tier 2 — special-class quadrature (≈ 9 primitives, ≈ 900 LOC)

Required for any consumer whose integrand has known structure (oscillatory, singular, semi-infinite, weight-bearing). All have published error theorems and golden-file vectors from `math/big` Hermite/Laguerre node generation.

| Add | Why | Closes |
|---|---|---|
| `GaussLegendreN(n)` (Golub-Welsch) | Removes the `{2..5}` clamp | Agent 016 §4 silent-clamp bug |
| `GaussHermite(f, n)` | Statistical-mechanics, prob.gauss-quadrature MGFs | Topic "special weight functions" |
| `GaussLaguerre(f, n)` | Reliability/queue Laplace integrals | Topic |
| `GaussChebyshev(f, n, kind)` | Closed-form nodes, no eigenproblem | Topic |
| `GaussLobatto(f, n)` + adaptive | Endpoint enforcement; FE assembly | Topic |
| `ClenshawCurtis(f, a, b, n)` | FFT-based, near-Gauss accuracy at n×log n | Topic |
| `Filon(f, a, b, omega, n)` | Oscillatory ∫ f cos(ωx) dx, ω-stable | Topic |
| `Levin(f, g, a, b, n)` | Highly oscillatory ω-independent cost | Topic |
| `IMT/Sidi/mIMT` transforms | Pre-quadrature endpoint regularization | Topic |
| `FiniteDiffWeights(stencil, order)` (Fornberg) | Generates arbitrary FD coefs | Higher-order derivs |

**Boundary call.** `Filon` and `Levin` overlap `signal/` (oscillatory analysis); recommendation is to keep them in `calculus` because they are quadrature methods that happen to specialize on oscillation, not signal-processing operators. The signal package can re-export.

**LOC.** ~900 total. The Gauss-* family shares a Golub-Welsch eigensolver (~150 LOC borrowed from `linalg/`).

---

## 5. Tier 3 — multi-D, spectral, adjoint (≈ 6 primitives, ≈ 1,200 LOC)

Lower priority because either (a) the use case is narrower, (b) the boundary with `optim/`, `signal/`, or `autodiff/` is genuinely fuzzy, or (c) the implementation cost is large enough to deserve its own package.

| Add | Sibling that may be the right home | Notes |
|---|---|---|
| **Sparse-grid (Smolyak) cubature** | `calculus` | Smolyak nodes; saves *exponential* work vs full tensor product up to ~d=10. Gerstner-Griebank 1998 is the canonical Go-portable pseudocode. |
| **Adaptive cubature (Berntsen-Espelid-Genz)** | `calculus` | The h-adaptive multi-D analog of Gauss-Kronrod. Cubature.jl is the reference. ~280 LOC. |
| **QMC (Sobol / Halton / lattice)** | `calculus` (or new `qmc` sub) | Deterministic O((log N)^d / N) instead of O(1/√N). Strictly better than `MonteCarloIntegrate` for d ≤ ~40. The CONTEXT.md zero-dep rule is satisfied; direction-number tables are public. |
| **Importance-sampled MC + VEGAS** | `calculus` | Lepage 1978. Adaptive grid refinement on integrand magnitude. ~200 LOC. |
| **Spectral differentiation** | likely `signal/` (FFT-shared) | Fourier diff for periodic data, Chebyshev diff for non-periodic. Trefethen *Spectral Methods*. Belongs adjacent to FFT, not finite differences. |
| **Pruess-Crockett auto-step** | `calculus` | Wrapper that probes `h` to find the cancellation/truncation crossover. ~60 LOC. The fix for agent 016's "wrong h scaling for large x" finding. |

**Explicit non-additions:**
- **RK4 / RK45 / DOPRI5.** Already in `chaos/ode.go`. The CLAUDE.md package table lists "RK4" under `calculus` but it is not implemented there; the fix is to **delete the misclaim from CLAUDE.md and the package doc-comment**, not to ship a duplicate. Agent 016 §1 also flagged this.
- **Root finding (bisection, Newton, Brent).** Already in `optim/rootfind.go`. Same fix: correct the package-doc reference. The `calculus.go` doc-comment lines 1-13 advertises RubberDuck and Pistachio consumers using "Gauss-Legendre, Monte Carlo" — the Newton claim should move to `optim/`.

---

## 6. Cross-cutting: convergence-tracking interface

Once Tier 1 ships, `calculus` will have at least four functions whose return type is `(value float64, errEst float64, ok bool)`. Standardize this as a public type now:

```go
// QuadResult is the canonical return for any error-estimating quadrature.
type QuadResult struct {
    Value   float64
    ErrEst  float64   // estimated absolute error
    Evals   int       // number of f-evaluations spent
    Reason  Status    // Converged | MaxDepth | NoProgress | NaN | Inf
}

type Status uint8
const (
    Converged Status = iota
    MaxDepth
    NoProgress
    NaNDetected
    InfDetected
)
```

Every Tier 1 and Tier 2 adaptive primitive returns `QuadResult`. Non-adaptive primitives (`TrapezoidalRule`, `SimpsonsRule`, `GaussLegendreN`) keep their existing scalar return for backwards compatibility but gain a `*WithError(...) QuadResult` companion. This sets the contract before consumer packages start importing — there are currently zero consumers (verified via repo-wide grep), so the API window is open.

**Boost.Math.Quadrature parallel.** Boost's `gauss_kronrod::integrate(...)` takes `&error_estimate, &L1_norm` by reference; SciPy's `quad` returns `(value, abserr)`. Either is conventional; the struct form composes better with Go and matches the `signal.FFTResult` precedent.

---

## 7. Consumer-driven prioritization

Repo-wide grep for `calculus.NumericalDerivative|calculus.NumericalGradient|calculus.TrapezoidalRule|calculus.SimpsonsRule|calculus.GaussLegendre|calculus.MonteCarlo` returns **zero non-test hits**. Every doc-comment-listed consumer (Oracle, Causal, RubberDuck, Pistachio, Horizon) is *projected*, not actual.

The implication for prioritization: ship Tier 1 *before* writing the consumer code, because the consumer code is what would hard-code the API. Specifically:

- **Pistachio** (60 FPS particle sim) needs allocation-free `RK4Step` (already in `chaos/`) and an allocation-free Gauss-Legendre. Tier 1 adaptive Gauss-Kronrod won't be used here; Tier 2 `GaussLegendreN` with cached weight tables will.
- **RubberDuck** (option pricing) needs **exactly the Tier 1 + Tier 2 quadrature stack**: adaptive Gauss-Kronrod for vanilla, Gauss-Hermite for Black-Scholes-derived expectations, tanh-sinh for at-the-money singular Greeks, complex-step for fast Greek calculation without an autodiff tape. This is the consumer that justifies Tier 1 most strongly.
- **Causal** (counterfactual gradient) needs `NumericalGradient` with per-component step (already noted in 016 §3) and `ComplexStepDerivative` for the cases where the model is differentiable in the complex sense.
- **Horizon** (forecasting trend derivatives) needs `Richardson` and `NumericalDerivative5pt` for noise-tolerant slope estimation — Tier 1 #5 and #6.
- **Oracle** (sensitivity analysis) needs Tier 1 #5/6/7 plus eventual Hessian (which agent 012 already routed to `autodiff` — fine to leave there).

Every named consumer is satisfied by Tier 1 + the four Tier 2 Gauss-* additions.

---

## 8. Web research crosswalk

Six external libraries cross-referenced for what reality should ship:

| Library | What it ships that reality lacks | What reality should *not* copy |
|---|---|---|
| **QUADPACK** (Fortran 1983) | qag/qags (Gauss-Kronrod), qawf (Filon-Fourier), qawc (Cauchy PV), qawo (oscillatory), qawe (singularity-aware), qng (non-adaptive Gauss-Kronrod) | Workspace-array calling convention (Fortran-era; Go can use slices + heap directly) |
| **scipy.integrate** (Python, 2024) | `quad` (QUADPACK wrap), `dblquad`, `tplquad`, `nquad`, `fixed_quad`, `quadrature` (adaptive Romberg), `romberg`, `simpson`, `quad_vec`, `solve_ivp` | Heavy reliance on FORTRAN libs; reality should reimplement |
| **Boost.Math.Quadrature** (C++, 1.84) | `gauss_kronrod`, `tanh_sinh`, `exp_sinh`, `sinh_sinh`, `trapezoidal` (with caching), `gauss<n>` for n ∈ {7,10,15,20,25,30}, `naive_monte_carlo` (with stderr), `wavelet_transforms` | Template-heavy API; Go generics are sufficient |
| **GSL** (C, 2.7) | `gsl_integration_qag/qags/qagi/qagiu/qagil/qagp/qawc/qawf/qawo/qaws`, `gsl_integration_romberg`, `gsl_integration_glfixed`, `gsl_integration_cquad` | GPL'd source — must reimplement, not port |
| **Mathematica `NIntegrate`** | `GaussKronrod`, `DoubleExponential`, `MonteCarlo`, `QuasiMonteCarlo`, `LocalAdaptive`, `GlobalAdaptive`, `LevinRule`, `OscillatorySelection`, `MultidimensionalRule`, `Trapezoidal`, `Simpson`, `Romberg`, `ClenshawCurtis`, `TaiKronrod`, `NewtonCotesRule` | Symbolic dispatch logic (out of scope) |
| **Cubature.jl / SciPy `quad_vec`** | Berntsen-Espelid-Genz multi-D h-adaptive | None; this is a clean port target |

Three observations from the crosswalk:

1. **Adaptive Gauss-Kronrod is unanimous as the default scalar routine.** All five general-purpose libraries default to it. Tier 1 #1 is non-negotiable.
2. **Tanh-sinh is the consensus singularity-handler.** Boost, Mathematica, and the Bailey arbitrary-precision community all converge on this. Tier 1 #4.
3. **Filon and Levin are the consensus oscillatory-handlers.** QUADPACK ships Filon (qawo); Mathematica ships both. Tier 2 entries — they're rarely the *first* thing someone reaches for, but they exist precisely because Gauss-Kronrod fails on `∫₀^∞ sin(1000x)/x dx`-style integrands.

---

## 9. What good looks like (after Tier 1)

```go
// Today (5 functions, no error estimates)
val := calculus.SimpsonsRule(f, 0, 1, 1000)   // hope 1000 was enough

// Post-Tier-1 (12 functions, error estimates)
res := calculus.AdaptiveGaussKronrod(f, 0, 1, calculus.QuadOpts{
    AbsTol: 1e-12, RelTol: 1e-10, MaxEvals: 10_000,
})
if res.Reason != calculus.Converged {
    return fmt.Errorf("quadrature did not converge: %v after %d evals (errEst=%g)",
        res.Reason, res.Evals, res.ErrEst)
}
return res.Value
```

This is the SciPy/Boost ergonomics the topic prompt's named consumers (option pricing, sensitivity analysis) expect by default in 2026.

---

## 10. Summary of recommendations

| Action | Priority | LOC | New goldens |
|---|---|---|---|
| Ship Tier 1 (6 primitives + `QuadResult` type) | **P0** | ~610 | ~120 |
| Fix CLAUDE.md to remove RK4 / root-finding misclaim | **P0** (docs) | ~5 | 0 |
| Ship Tier 2 (9 primitives, esp. arbitrary-n Gauss + tanh-sinh variants) | **P1** | ~900 | ~180 |
| Ship Tier 3 (multi-D, QMC, spectral) — but consider sibling packages for spectral | **P2** | ~1,200 | ~200 |
| Add benchmark file (currently absent) | **P1** | ~150 | n/a |
| Standardize `QuadResult` return for all adaptive integrators | **P0** | included above | n/a |
| Add `Example()` function for each Tier 1 primitive (godoc render) | **P1** | ~200 | n/a |

If only one pull request is shipped from this report, it should be **Tier 1 #1 (`AdaptiveGaussKronrod` with `QuadResult`)** plus the CLAUDE.md fix. That single change moves `calculus` from "Newton-Cotes survey" to "shippable adaptive scalar quadrature library" and sets the API contract for everything that follows.

---

*Sources cited above are by author-year; full references list in any of: Press et al. *Numerical Recipes* 3rd ed. (chs. 4-7), Davis & Rabinowitz *Methods of Numerical Integration* 2nd ed., Trefethen *Approximation Theory and Approximation Practice*, Bailey-Borwein-Borwein-Plouffe arbitrary-precision quadrature literature, QUADPACK book (Piessens et al. 1983).*
