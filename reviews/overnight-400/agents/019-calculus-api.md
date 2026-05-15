# 019 — calculus: API ergonomics — function-handle convention, error reporting, tolerance contracts

**Agent:** 019 of 400
**Date:** 2026-05-07
**Topic:** API ergonomics review of `C:\limitless\foundation\reality\calculus\` (per MASTER_PLAN.md, line 30)
**Files audited:** `calculus/calculus.go`, `calculus/calculus_test.go`. Sibling style sampled: `optim/rootfind.go:22-100`, `optim/gradient.go:30-81`, `chaos/ode.go:36-100`, `chaos/systems.go:17-115`, `prob/conformal/split.go:38-145`, `signal/window.go:15-104`, `signal/filter.go:19-130`, `infogeo/fdiv.go:65-196`. Predecessors 016 (numerics), 017 (missing primitives), 018 (SOTA design axes) inform but are not duplicated — this report is restricted to the 5 functions as they exist on disk today.

## Headline

reality/calculus is the *function-style scalar quadrature/diff toolkit with five hand-rolled signatures and zero unifying types* — every function takes its `func(...) float64` integrand in a different shape (`func(float64) float64` for trapezoid/Simpson/GL, `func([]float64) float64` for gradient/MonteCarlo with no shared `Integrand` alias), every function specifies precision in a different way (`h float64` for diff, `n int` for trapezoid/Simpson, `points int` for GL clamped to {2,3,4,5}, `samples int` + `rng` for MC), every function silently swallows out-of-contract inputs (Simpson rounds odd `n` to even, GL clamps `points` to [2,5], TrapezoidalRule sets `n<1` to 1, MC sets `samples<1` to 1, NumericalGradient assumes `len(out)==len(x)` and panics out-of-bounds otherwise), no function returns an error or an error estimate (every signature returns a bare `float64` — no `(value, error)`, no `QuadResult{Value, ErrEst, NEval}`, no `(value, abserr float64)`), no function accepts a failing integrand (`func(float64) (float64, error)`) so a divide-by-zero or domain error inside the integrand silently produces NaN that propagates to the result, the package exports zero `Default*` constants so consumers must hard-code `1e-5` for `h` and `100`/`1000` for `n` (visible in the test file: nine different `n` values across 1000/10000/100000), naming is bimodal (`NumericalDerivative`/`NumericalGradient` vs `TrapezoidalRule`/`SimpsonsRule`/`GaussLegendre`/`MonteCarloIntegrate` — the integrators do *not* share the prefix that 016/017/018 reference as the "Integrate*" family; that family doesn't exist), and the function-handle order argument is the same for all four scalar integrators (`f, a, b, n`) which is the single saving grace — but `MonteCarloIntegrate(f, dim, lower, upper, samples, rng)` breaks that pattern entirely. The fix-set is small, bounded, and orthogonal to 016/017/018: a `calculus.Integrand`/`Integrand1D`/`IntegrandND` type alias trio, an unexported `clampN`/`clampPoints` helper that returns the clamped value *and* a `bool` (or sentinel `ErrPointsClamped`), a `QuadResult` adopted by `MonteCarloIntegrate` first (where the stderr is free) before being retrofitted into 017's adaptive integrators, exported `DefaultDerivativeStep = 1e-5` / `DefaultIntegrateN = 100` constants, and a docstring contract for what happens when the integrand returns NaN.

---

## 1. Function-handle convention

Five public functions. **Three different integrand types, no shared alias.**

| function | integrand type | dimensionality | error-channel |
|---|---|---|---|
| `NumericalDerivative` | `func(float64) float64` | 1-D scalar | none |
| `NumericalGradient` | `func([]float64) float64` | N-D scalar | none |
| `TrapezoidalRule` | `func(float64) float64` | 1-D scalar | none |
| `SimpsonsRule` | `func(float64) float64` | 1-D scalar | none |
| `GaussLegendre` | `func(float64) float64` | 1-D scalar | none |
| `MonteCarloIntegrate` | `func([]float64) float64` | N-D scalar | none |

Compare with the field:

| system | scalar integrand | vector integrand | failing integrand |
|---|---|---|---|
| reality/calculus | `func(float64) float64` (×4 signatures) | `func([]float64) float64` (×2 signatures) | not supported |
| QUADPACK | `subroutine f(x, fx)` (out-arg) | n/a | flag in caller's common block |
| scipy.integrate | `Callable[[float], float]` | `quad_vec(f: Callable[[float], np.ndarray])` | `f` may raise; caught and re-raised |
| Boost.Math | `F&& f` (template) | `boost::math::quadrature::naive_monte_carlo<F>` | exceptions |
| QuadGK.jl | `f` (any callable) | `quadgk(f::Function, a, b; segbuf, norm)` | exceptions; `DomainError` allowed |
| GSL | `gsl_function` (struct with `function` + `params`) | `gsl_monte_function` | `GSL_ERROR(...)` macro; status int |
| reality/chaos | `func(t float64, y, dydt []float64)` (in-place) | n/a | none |

Three observations:

1. **No `Integrand` type alias.** Every consumer who wants to write a function that *takes an integrator* (e.g., a "double-integral" helper, a "convergence-test runner") must spell `func(float64) float64` literally and re-spell it for the N-D variants. A 4-line addition would close this:
   ```go
   // Integrand is a 1-D real-valued integrand: f: R → R.
   type Integrand func(x float64) float64

   // IntegrandND is an N-D real-valued integrand: f: R^n → R.
   type IntegrandND func(x []float64) float64
   ```
   `prob/copula/archimedean.go` already uses this pattern (`func ClaytonCopulaCDFFn(theta float64) (func(u, v float64) float64, error)` — returns a typed function-handle for downstream composition). `calculus` does not.

2. **No `(value, error)` integrand variant.** A safe-by-default integrand that wants to refuse `1/0` or `log(-1)` has to either pre-validate the abscissa (impossible: caller controls `x`) or return NaN and hope the consumer notices. scipy's `quad` and Julia's `QuadGK.jl` both let the integrand raise; QUADPACK's contract is that the integrand must be defined on the closed `[a,b]`. reality/calculus is silent on this. The 4-function-handle integrators all forward the integrand's NaN into the sum and out the back. **No defensive `math.IsNaN(fx)` check exists in any integrator** (verified by Grep: zero matches in `calculus.go`).

3. **No vector-valued integrand (`func(float64) []float64`)**. Useful for computing N parallel integrals with one set of abscissae (the canonical scipy `quad_vec` use-case). Reality has zero way to express this without re-evaluating the integrand N times. Not a priority for v0.10.0 but worth flagging as the natural Phase-2 signature alongside agent 018's `IntegrandBatch func(xs, ys []float64)` (which is the orthogonal axis: many points × one output, vs one point × many outputs).

## 2. Tolerance contracts

Five functions, **five different precision contracts**:

| function | precision parameter(s) | meaning | what happens out-of-range |
|---|---|---|---|
| `NumericalDerivative` | `h float64` | step size | `h<=0` → division by zero / NaN; doc *recommends* `~1e-5` |
| `NumericalGradient` | `h float64` | step size (per-component) | `h<=0` → NaN per component; `len(out)<len(x)` → panic |
| `TrapezoidalRule` | `n int` | subintervals | `n<1` silently bumped to 1 |
| `SimpsonsRule` | `n int` | subintervals (must be even) | `n<2` silently bumped to 2; odd `n` silently bumped to `n+1` |
| `GaussLegendre` | `points int` | nodes (must be in {2,3,4,5}) | `points<2` clamped to 2; `points>5` clamped to 5 |
| `MonteCarloIntegrate` | `samples int` | sample count | `samples<1` silently bumped to 1 |

There is no `tol`, no `rtol`, no `atol`, no `maxIter`, no `convergenceTest` anywhere in the package. The contract is purely "spend N evaluations; here is the answer." This is a deliberate v0.10.0 design choice (every primitive is non-adaptive — the absence of adaptive/error-controlled quadrature is 017's topic), but it produces three downstream ergonomic problems:

a. **Consumers can't ask "did this converge?"** because the answer is always "you got exactly what you asked for; spend more if you want better." The Monte Carlo case is the worst: `samples=100000` returns a single `float64` whose stderr is `σ/√N` and is *trivially computable* from the same loop (running mean + running M2 = ~3 LOC, see Welford's algorithm). The package exposes zero of this. (Pinned by agent 016 §6 and agent 018 §2.3 — adopting `QuadResult` *just* in `MonteCarloIntegrate` is free even before adaptive integrators land.)

b. **Sibling-package consumers compare-incomparable precision arguments.** Compare:
   - `optim.BisectionMethod(f, a, b, tol float64) float64` — `tol` is *interval width*
   - `optim.NewtonRaphson(f, fPrime, x0, tol, maxIter int) float64` — `tol` is *|f(x)|*
   - `prob/conformal.SplitQuantile(scores, alpha) (float64, error)` — `alpha` is miscoverage rate, returns error
   - `calculus.TrapezoidalRule(f, a, b, n int) float64` — `n` is subintervals, no tol, no error

   A user writing a numerical-methods notebook chains these and gets four different precision-control models in five lines. There is no doc anywhere that says "calculus uses N-eval; optim uses tol; prob/conformal uses alpha — here is why."

c. **The defaults are nowhere.** The `1e-5` recommended `h` lives in *one* godoc comment (`NumericalDerivative` line 28) and is repeated nine times verbatim across `calculus_test.go` (lines 36, 43, 50, 57, 64, 71, 78, 86, 127, 138, 147, 159, 174). There is no exported `DefaultDerivativeStep = 1e-5` constant. Sibling `physics/` exports ~40 named constants; `optim/` exports zero default-tolerance constants either, but at least that package has `(tol float64)` arguments where the user can see they need to pick a value. `calculus.NumericalDerivative` has the same problem with `h`, but the user is expected to know `cbrt(eps_machine) ≈ 6e-6` from the doc-comment recommendation. This is a candidate for `calculus.DefaultStep = 1e-5` (or at least `DefaultDerivativeStep`).

## 3. Error reporting

**Six functions, zero error returns.** This is the single biggest divergence from sibling packages.

Cross-package count of `func ... (..., error)` returners (Grep, output_mode=files_with_matches):
- `infogeo/`: 13 functions return `(value, error)` (KL, ReverseKL, JS, TotalVariation, Hellinger, ChiSquared, Renyi, Bregman, ...)
- `prob/conformal/`: 6 functions return `(value, error)` (SplitQuantile, AdaptiveQuantile, MondrianQuantile, ...)
- `prob/copula/`: 4 functions return `(func, error)` (ClaytonCopulaCDFFn, GumbelCopulaCDFFn, ...)
- `optim/`: 2 functions return `([]float64, float64, error)` (SimplexMethod, InteriorPoint)
- `info/mdl/`: 2 functions return `(int, error)` / `(int, float, error)`
- `calculus/`: **0 functions return error**

The choice to return bare `float64` is internally consistent for the package but inconsistent with the rest of `reality/`. Three failure modes are silently masked:

1. **Out-of-range parameter clamping** (Simpson n→even, GL points→{2,5}, Trapezoidal n→1, MC samples→1). 016 §3 documents these as numerics issues; the API issue is that *no caller can tell whether their requested precision was honored*. A user writing a Richardson-extrapolation harness that calls `SimpsonsRule` at `n = 1, 2, 4, 8, 16` gets `n = 2, 2, 4, 8, 16` actually used, and Richardson silently produces wrong convergence-rate measurements. The fix is a `(value float64, actualN int)` return or a sentinel `var ErrPointsClamped = errors.New(...)`. Either is one line per function.
2. **Integrand returns NaN/Inf**. No defensive check anywhere. NaN propagates straight to the result; the caller has to `math.IsNaN(result)` after every call. A `(value, error)` return where the error is `ErrIntegrandNonFinite` for at least one abscissa would catch this at the boundary.
3. **`a >= b`** for any of the 1-D integrators. `TrapezoidalRule(f, 1, 0, 100)` produces a finite negative number (`h = -0.01`, sum runs the same way) — it's not *wrong* (the integral is genuinely negated), but it's also not flagged. `MonteCarloIntegrate` is worse: if `lower[i] >= upper[i]` for some i, the volume can go negative or zero with no warning.

Compare with QUADPACK's `qag`: returns `(result, abserr, neval, ier)` where `ier` is an integer status code with 6 documented failure modes (`ier=1` max-subdiv reached, `ier=2` roundoff detected, `ier=3` extreme-bad-integrand, etc.). reality/calculus has no equivalent of any of these channels.

The lowest-leverage fix is `MonteCarloIntegrate` first, since the running-stderr is essentially free:

```go
type MCResult struct {
    Value, StdErr float64
    Samples       int
}
func MonteCarloIntegrate(f IntegrandND, dim int, lower, upper []float64, samples int, rng RNG) MCResult
```

This is the smallest possible step toward the `QuadResult` convention agent 018 §2.3 / agent 017 §6 propose for the not-yet-existing adaptive integrators.

## 4. Default parameters

**Zero `Default*` constants exported.** Compare with `physics/` (`StandardGravity`, `StandardAtmosphere`, `RoomTemperatureK`) and `constants/` (the entire package's job).

A consumer writing `calculus.TrapezoidalRule(f, a, b, ???)` has to know that 100 is "okay for smooth functions" and 1000 is "safer." This is hidden lore. A 5-line addition:

```go
const (
    DefaultDerivativeStep = 1e-5  // cbrt(machineEpsilon) for Burden & Faires central difference
    DefaultIntegrateN     = 100   // smooth integrand, O(h^4) Simpson tolerance ~1e-8
    DefaultGLPoints       = 5     // maximum supported by the precomputed table
    DefaultMCSamples      = 10000 // O(1/sqrt(N)) ≈ 1% relative error
)
```

would let consumers write `calculus.TrapezoidalRule(f, a, b, calculus.DefaultIntegrateN)` and self-document the hand-rolled `100` they would otherwise type. The doc-comments already imply the values; promoting them to constants makes them queryable.

Go's idiom would be a `*Options` struct (functional-options pattern), but that is a v1.0 question. Constants are the v0.10.0 win.

## 5. Composability

**Can you chain `Integrate(Integrate(f, ...), ...)` for a double integral?**

In principle: yes, by closure. In practice: no clean way to express it, and the cost is hidden:

```go
// double integral of f(x,y) = x*y over [0,1]^2 via Trapezoidal × Trapezoidal
inner := func(x float64) float64 {
    return calculus.TrapezoidalRule(func(y float64) float64 {
        return x * y
    }, 0, 1, 100)
}
result := calculus.TrapezoidalRule(inner, 0, 1, 100)
// 100 × 101 = 10,100 evaluations of f, plus 101 evaluations of `inner`
```

This works, but:
- The user has to spell the `func(float64) float64` integrand twice.
- The closure captures `x` by value per outer iteration — an allocation per outer evaluation in some cases (Go inlines this for trivial closures, but the moment `inner` does anything non-trivial the escape analysis flips).
- There is no `calculus.DoubleIntegrate(f func(x, y float64) float64, ax, bx, ay, by float64, n int)` helper anywhere. `MonteCarloIntegrate` is the *only* multi-dimensional integrator in the package; it explicitly is not a tensor-product of 1-D rules but a uniform-sample MC.
- The two `n=100` choices are independent, but the user cannot easily see that the inner one runs 100× more often. A `calculus.DoubleTrapezoidal` helper that takes `(nx, ny int)` would expose this asymmetry.

Verdict: composable in the boring-Go sense, not composable in the discoverable sense. No `calculus.Compose`, no `calculus.Tensor2D`, no `Cubature` family.

A second composability gap: `NumericalDerivative` and the integrators do not interoperate. There's no `calculus.Derivative(integral_function, x, h)` example anywhere; nothing tests `d/dx ∫ f(t) dt = f(x)` (the Fundamental Theorem) — which would be a 5-line golden-file test that pins the *interplay* of the two halves of the package. (Out of scope for an API-only review, but worth flagging because the API surface allows the test.)

## 6. Naming

Three naming axes, three different conventions:

a. **Integrators do not share a prefix.** Agent 018 line 5 references "verbose `IntegrateSimpson`/`IntegrateTrapezoidal`/etc." — *those names don't exist*. The actual names are `TrapezoidalRule`, `SimpsonsRule`, `GaussLegendre`, `MonteCarloIntegrate`. The fourth one *does* contain `Integrate`; the first three contain `Rule` or no suffix at all. There is no `IntegrateSimpson` equivalent of the QUADPACK `qag/qags/qng` style or scipy `quad/quadrature/fixed_quad/romberg/simps/trapz`.

   Field convention: scipy uses verb-first (`quad`, `trapz`, `simps`, `romberg`, `fixed_quad`, `quadrature`), Boost.Math uses noun-first (`trapezoidal`, `tanh_sinh`, `gauss_legendre`), QuadGK.jl uses noun-only (`quadgk`, `quadgk!`). reality/calculus is closest to Boost (noun-first) for three of four and verb-first for the MC case. **Pick one.** If "calculus" is the package then `Trapezoidal`, `Simpson`, `GaussLegendre`, `MonteCarlo` (drop the `Rule`/`Integrate` suffixes — the package context disambiguates) reads cleanest at consumer call sites:
   ```go
   calculus.Simpson(f, a, b, n)        // vs calculus.SimpsonsRule
   calculus.Trapezoidal(f, a, b, n)    // vs calculus.TrapezoidalRule
   calculus.GaussLegendre(f, a, b, p)  // unchanged
   calculus.MonteCarlo(f, dim, ...)    // vs calculus.MonteCarloIntegrate
   ```
   This is the v1.0 question (renames are breaking); for v0.10.0 the cheapest move is to add `Simpson` / `Trapezoidal` aliases and deprecate the old names in doc-comments.

b. **`Numerical` prefix is dead weight.** `NumericalDerivative` and `NumericalGradient` — what other kind of derivative would `calculus.Derivative` plausibly mean? The package is *defined* as numerical methods. `calculus.Derivative(f, x, h)` and `calculus.Gradient(f, x, h, out)` would read identically to numpy/scipy convention (`numpy.gradient`, `scipy.misc.derivative`). Sibling `optim/` uses unprefixed names (`BisectionMethod`, `NewtonRaphson` — though those have their own historical-name verbosity).

c. **`SimpsonsRule` apostrophe** — Go convention is `SimpsonRule` or `Simpson` (no possessive 's'). `BachAlgorithm` not `BachsAlgorithm`. The current name is a Go-style wart inherited from the British textbook tradition.

## 7. Identical-looking arguments in different orders

The MASTER_PLAN line says "Look for any API that requires the user to pass identical-looking arguments in different orders."

| function | argument order |
|---|---|
| `NumericalDerivative(f, x, h)` | function, point, step |
| `NumericalGradient(f, x, h, out)` | function, point, step, output |
| `TrapezoidalRule(f, a, b, n)` | function, lo, hi, n |
| `SimpsonsRule(f, a, b, n)` | function, lo, hi, n |
| `GaussLegendre(f, a, b, points)` | function, lo, hi, points |
| `MonteCarloIntegrate(f, dim, lower, upper, samples, rng)` | function, dim, lo, hi, n, rng |

Two real risks:

1. **The 1-D integrators put `(a, b)` adjacent and same-typed (`float64, float64`).** Swapping `a` and `b` produces a finite *negated* answer with no error. This is a `mu, sigma`-style hazard (per agent 014 §3). `reality/optim/rootfind.go:22` has the exact same issue (`BisectionMethod(f, a, b, tol)`). The cross-package convention is to never flag this and to document "a < b" in the doc-comment — both `TrapezoidalRule` line 83 and `SimpsonsRule` line 109 do say "a < b" but neither enforces it.

2. **`MonteCarloIntegrate` puts `lower, upper` adjacent (both `[]float64`).** Same swap hazard, but for vectors. `lower[i] > upper[i]` for any *single* i produces a wrong volume sign; the function's behavior is undefined-by-omission. There's no precondition check.

3. **`NumericalDerivative(f, x, h)` and `NumericalGradient(f, x, h, out)` agree on `(f, x, h)`** but `x` is `float64` in the first and `[]float64` in the second. Same name, same position, different type. Go's type system catches the swap, so this is *safe*; the only cost is mental load when reading both signatures side-by-side. No fix needed.

4. **`MonteCarloIntegrate` is the only function that takes `dim` as a separate argument** *and* takes `lower, upper []float64`. The dim is recoverable from `len(lower)` and is checked in *no* way (the function happily reads `dim` indices into both slices and panics if `len(lower) < dim`). Drop `dim` and use `len(lower)`; this also closes the swap-hazard between `dim` and `samples` (both `int`).

## 8. Truncation-error estimate from CentralDifference

The MASTER_PLAN line specifically asks: "CentralDifference returns just the value — does the API allow returning the truncation error estimate too?"

Answer: **no.** `NumericalDerivative` returns a bare `float64`. The Richardson-extrapolation idiom (compute at `h` and `h/2`, take linear combination) is the textbook way to estimate the truncation error, and it requires *one* extra evaluation pair. The API does not expose a `(value, errEst float64)` variant or a `RichardsonDerivative(f, x, h)` helper.

This is the `MonteCarloIntegrate` pattern again at smaller scale: the error-estimate is *cheap* (one extra eval pair), the algorithm is *standard*, and the consumer cannot get it without rewriting the whole call. A 6-line addition:

```go
// NumericalDerivativeWithError returns d f/dx and an O(h^2) Richardson
// truncation-error estimate. Cost: 4 evaluations (vs 2 for the bare version).
func NumericalDerivativeWithError(f Integrand, x, h float64) (deriv, errEst float64) {
    d1 := NumericalDerivative(f, x, h)
    d2 := NumericalDerivative(f, x, h/2)
    return d2 + (d2-d1)/3, math.Abs(d2-d1) / 3
}
```

Same shape applies to `TrapezoidalRule` (Romberg's first column), `SimpsonsRule` (Richardson on Simpson is degree-6 Newton-Cotes), and `GaussLegendre` (Gauss-Kronrod nesting — agent 018's Tier-1 #1). The API today provides *zero* of these.

## 9. Comparison with sibling packages — naming consistency

Cross-checked four sibling packages on the same conventions:

| convention | calculus | optim | signal | prob/copula |
|---|---|---|---|---|
| function-handle param name | `f` | `f` | n/a (slice in/out) | `Fn` suffix (e.g., `ClaytonCopulaCDFFn`) |
| step / tolerance arg | `h` (size), `n` (count) | `tol`, `maxIter` | `windowSize` | `alpha` |
| output buffer | `out []float64` (gradient only) | n/a | `out []float64` (every signal func) | n/a |
| error return | none | sometimes | none | always (`(val, error)`) |
| precomputed-table lazy init | no (rebuilt every GL call) | n/a | yes (FFT twiddles cached) | n/a |
| panic vs error | "silently clamp" | mixed | panic on shape mismatch | always error |

Three concrete inconsistencies surface:

1. **Output-buffer pattern is in `signal/` everywhere but `calculus/` only for `NumericalGradient`.** The 60-FPS Pistachio consumer cited in the package doc-comment (`calculus.go:9`) would benefit from a `TrapezoidalRuleInto` / `SimpsonsRuleInto` variant... except that integrators return scalars, so there's no buffer to fill. The relevant pre-allocation is `MonteCarloIntegrate`'s `point := make([]float64, dim)` (line 262) which is currently *not* exposed to the caller — a `MonteCarloIntegrateInto(f, dim, lower, upper, samples, rng, workspace)` variant where `workspace` is the per-sample point buffer would let a consumer call MC repeatedly without allocating. Same shape as `signal.HannWindow(n int, out []float64)`.

2. **`prob/conformal` always returns `error`; `calculus` never does.** Same author-time decision (whether failure is in-band or out-of-band) made opposite ways. The repo has no documented standard.

3. **`signal` panics on shape-mismatch; `calculus` silently clamps.** Same risk class (caller passed a bad parameter) handled differently. The `signal` style is louder; the `calculus` style is sneakier. No cross-package convention exists.

## 10. Specific recommended fixes (API-only, no algorithm changes)

Bounded fix-set, ~80 LOC of additions, zero LOC removed, zero behavior changes for valid inputs:

1. **Add `Integrand` and `IntegrandND` type aliases** (4 LOC). Update the 5 function signatures to use them. Pure cosmetic; opens the door for documenting "what is an integrand."

2. **Export 4 default-parameter constants** (5 LOC + doc).

3. **Adopt `MCResult{Value, StdErr float64, Samples int}` in `MonteCarloIntegrate`** (15 LOC, breaking change *but* zero non-test consumers exist per agent 017 §7). This sets the precedent for `QuadResult` everywhere else.

4. **Add `NumericalDerivativeWithError(f, x, h) (deriv, errEst float64)`** (6 LOC). Free Richardson estimate.

5. **Add `(actualN int)` second return to `Trapezoidal/Simpson` or sentinel `var ErrParamsClamped = errors.New(...)`** (~10 LOC). Pick one. Sentinel is cheaper for now; tuple return is more discoverable.

6. **Add precondition checks for `a < b` and `lower[i] < upper[i]`**, returning `ErrInvalidInterval` (10 LOC). Loud failure beats silent sign-flip.

7. **Drop the `dim` parameter from `MonteCarloIntegrate`** (1 LOC); derive from `len(lower)`. Single API removal, zero hazard.

8. **Add aliases `Trapezoidal`/`Simpson`/`MonteCarlo`** with `// Deprecated: use Trapezoidal` on the old names (4 LOC). Sets up the v1.0 rename without breaking anyone.

9. **Add `Derivative` / `Gradient` aliases** for `NumericalDerivative` / `NumericalGradient` (4 LOC). Same deprecation pattern.

10. **Document the integrand-NaN contract** in `doc.go`: "If `f` returns NaN at any abscissa, the integrator returns NaN. Use `MonteCarloIntegrate` with `MCResult.StdErr = NaN` as the convergence signal." (~5 LOC of doc).

Total: ~65 LOC of code + ~5 LOC of doc, no algorithmic change, full backward compatibility (the renames are aliases, not replacements). Closes 0 numerics gaps from 016, 0 missing primitives from 017, 0 SOTA gaps from 018 — purely the API-shape topic the MASTER_PLAN line scopes.

---

## Out-of-scope (handled by predecessors / successors)

- Adaptive Gauss-Kronrod, Romberg, tanh-sinh, finite-difference family — agent 017.
- `QuadResult{Value, ErrEst, NEval, Status}` for adaptive integrators — agent 018 §2.3.
- IEEE-754 NaN/Inf golden vectors — agent 016 §4.
- Performance (precomputed GL tables, allocation-free hot path) — agent 016 §5.
- AD-through-quadrature — agent 018 §6 / agent 183 (synergy-calculus-autodiff).
