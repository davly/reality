# 016 | calculus-numerics

**Scope.** Numerical-correctness audit of `C:\limitless\foundation\reality\calculus\` (one source file, one test file, four golden JSON files). Comparable in scale to the autodiff/audio audits 011-015.

**TL;DR.** `calculus` is a 275-line, 5-function survey package: `NumericalDerivative` (central diff), `NumericalGradient`, `TrapezoidalRule`, `SimpsonsRule`, `GaussLegendre` (2-5 pt), `MonteCarloIntegrate`. Of the eight algorithm families the topic asks about, **the package implements three** (trapezoid, Simpson, central diff) and **does not contain RK4, Romberg, adaptive RK45, root-finding, Richardson, or higher-order derivatives at all** — those live in `chaos/ode.go` (RK4) and `optim/rootfind.go` (bisection / Newton / secant / Brent), each of which is a separate package with its own audit slot. Within the actual calculus surface, the math written is textbook-correct; the gaps are (a) IEEE-754 inputs are unguarded and untested, (b) every error claim in a docstring is asymptotic-only and never numerically pinned, (c) golden coverage is ~8 vectors per function vs. the CLAUDE.md "minimum 20, target 30" rule, (d) Simpson silently mutates `n`, (e) `MonteCarloIntegrate` silently truncates dim mismatches, and (f) the Gauss-Legendre implementation reallocates and re-`math.Sqrt`'s its node table on every call.

---

## 1. What the topic asks about that is not in `calculus/`

| Topic asked about | Where it actually lives in reality | Audit slot |
|---|---|---|
| Simpson 1/3 composite, odd-n handling | `calculus/calculus.go:112-130` | this report |
| Trapezoidal | `calculus/calculus.go:86-96` | this report |
| Central-diff numerical derivative | `calculus/calculus.go:31-33` | this report |
| Richardson extrapolation | **not implemented anywhere** | — |
| Romberg integration | **not implemented anywhere** | — |
| Forward / backward / 5-pt central diff | **only central diff exists** | — |
| RK4 fixed step | `chaos/ode.go` (RK4, RK45, AdaptiveRK45) | a separate `chaos-numerics` slot |
| Adaptive DOPRI5 / step rejection | `chaos/ode.go` AdaptiveRK45 | same |
| Bisection / Newton / Secant / Brent | `optim/rootfind.go` + `testdata/optim/bisection.json` | a separate `optim-numerics` slot |

Reading `calculus/calculus.go` lines 1-13 and `CLAUDE.md` confirms this is intentional: `calculus` is documented as "numerical differentiation and integration primitives" only, and the package table in `CLAUDE.md:31` says "Simpson, trapezoidal, RK4, root finding" — **the table is wrong on RK4 and root finding**, those words appear in the description but the symbols are not in the package. Worth filing as a docs bug.

The remainder of this audit is scoped to what is actually in `calculus/`.

---

## 2. NumericalDerivative — central difference, line 31

```go
func NumericalDerivative(f func(float64) float64, x, h float64) float64 {
    return (f(x+h) - f(x-h)) / (2 * h)
}
```

**Math.** Standard 2-point central difference. Truncation error is `-h²·f‴(ξ)/6`, roundoff is `O(ε/h)`, total minimized at `h ≈ ∛(3ε/|f‴|) ≈ ε^{1/3}` ≈ `6e-6` for unit-scale `f`. The doc-comment recommends `h ≈ ∛(machineEps)·max(1,|x|) ≈ 1e-5` which is correct for unit-scale problems but **wrong by `|x|^{2/3}` for large x** (Press NR §5.7 derives `h_opt ≈ ∛ε · x_c` where `x_c = max(|x|, char-scale)`); the more honest scaling is `h = ∛ε · max(1,|x|)^{2/3}` for scaled problems, but the current rule-of-thumb is the one most textbooks quote and is fine for the ports.

**Concrete numerical issues, none guarded:**

1. **`h = 0`**: returns `NaN` (0/0). No guard.
2. **`h` denormal (≤ 5e-324)**: `2*h` underflows to subnormal then to 0; same `NaN`.
3. **`x ± h == x` due to roundoff** (e.g. `x = 1e308, h = 1`): returns 0/2 = 0 silently. The conventional fix is the Ridders trick `volatile temp = x+h; h_actual = temp - x` to recover the *machine* representable step; not used here. Cited as best practice in Press NR §5.7.
4. **`f` returns NaN at `x±h` only** (e.g. `f = sqrt`, `x = 0`, `h = 1e-5` evaluates `sqrt(-1e-5)` → NaN): result is NaN, propagates silently. The `Log(0) = -Inf` boundary case from agent 011's autodiff audit (Sqrt0/Log0 → ±Inf gradients) is the **same bug surface** here and is also unguarded.
5. **Catastrophic cancellation as `h → 0`**: when `|f(x+h) - f(x-h)| ≈ ε·|f|`, the numerator loses all significant bits. This is *the* classical numerical pitfall the docstring names but does not detect.

**No test exercises any of these.** `TestDerivative_*` (lines 34-90) and the 10 golden vectors all use `h = 1e-5` on smooth functions away from singularities. Zero NaN/Inf vectors. Zero subnormal-`h` vectors. Zero `h = 0` vector. The CLAUDE.md "IEEE 754 edge cases mandatory" rule is **violated for derivative.json**.

**Missing primitives the consumers (Oracle, Causal, Horizon per the package doc) will need:**
- `NumericalDerivativeForward(f,x,h)` — required when `f` is not defined for `x < a` (e.g. log on the left edge of a domain).
- `NumericalDerivative5pt` — `(-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h))/(12h)`, O(h⁴), 8× more accurate at the same h, almost free since the consumers are scalar.
- Second derivative `(f(x+h)-2f(x)+f(x-h))/h²` — Newton's method needs this and currently can't get it from `calculus`.
- `Richardson(D, h, k)` — extrapolate a sequence `D(h), D(h/2), D(h/4)` to kill the leading error; standard 8-line function.

These are all in the topic ("Numerical differentiation: forward/central/Richardson; optimal step size choice") and **none exist**.

---

## 3. NumericalGradient, line 47

```go
for i := 0; i < n; i++ {
    orig := x[i]
    x[i] = orig + h;  fPlus  := f(x)
    x[i] = orig - h;  fMinus := f(x)
    out[i] = (fPlus - fMinus) / (2 * h)
    x[i] = orig
}
```

**Correctness.** Math is right; in-place perturb-and-restore is the right move (no allocations, satisfies CLAUDE.md rule 3). `TestGradient_RestoresX` (line 154) pins this. Cost is `2n` evaluations of `f` for an n-dim gradient — acceptable.

**Issues:**

1. **No length-mismatch check.** If `len(out) < len(x)` the loop panics on out-of-bounds (Go runtime); if `len(out) > len(x)` the trailing entries are stale. Doc says "out must have the same length as x" but does not enforce. A one-line `if len(out) != len(x) { panic("...") }` would be cheap and consistent with the optim package's gradient call sites.
2. **Restore-on-NaN:** if `f` panics or returns NaN, `x[i]` has already been modified. Defer-restore would protect this; the current code restores only on the happy path. (NaN propagates to `out[i]` correctly, but a panicking `f` leaves `x` perturbed — a real-world hazard for the Causal consumer named in the docstring.)
3. **Per-component step size.** `h` is scalar, but well-scaled gradients use `h_i = ∛ε · max(1,|x_i|)`. Currently a single `1e-5` step is applied to a coordinate of magnitude `1e10` (drowned in roundoff) and to one of magnitude `1e-10` (drowned in truncation). Standard fix: accept `h []float64` or `nil` (defaults). Not in the topic but worth noting since `optim/` consumers will pay for this.

`TestGradient_Rosenbrock` at line 165 hits the gradient at the minimum `(1,1)` where the true gradient is `(0,0)` — useful, but **does not exercise an off-minimum point** where Rosenbrock's curvature stresses the central-diff error model. A vector at `(-1.2, 1.0)` (the canonical Rosenbrock starting point) would be the right golden case and is missing.

---

## 4. TrapezoidalRule, line 86

**Math.** Composite trapezoidal, `h/2·(f(a)+f(b)) + h·Σ f(a+ih)`. Correct.

**Edge cases handled:** `n < 1` clamped to 1.

**Edge cases NOT handled:**

1. **`a == b`** → `h = 0`, sum is `0.5·(f(a)+f(b))` then `*0` = 0. Mathematically correct (integral over a point is 0), but accidentally so — the clamp `n=1` then dividing by `n` gives `h=0` and the sum collapses. Worth a comment.
2. **`a > b`**: silently produces a negative-magnitude result (which is the right sign for `∫_a^b` with reversed limits) — but the docstring "Valid range: a < b" says it's not supported. Behaviour and doc disagree. A test pinning the convention either way is missing.
3. **`f` returns Inf at endpoint** (e.g. `1/x` from 0 to 1): `f(0) = +Inf`, sum is +Inf, result is +Inf. **No singular-endpoint variant** (e.g. open trapezoid, midpoint rule) is offered; the topic asks about "singular endpoints" and the answer is *not handled*.
4. **NaN integrand** at any sample: result is NaN. No `IsNaN` guard or documented behaviour.

**Golden coverage: 8 vectors (trapezoidal.json), all smooth, all bounded, all `a < b`, all `n ∈ {10, 100, 1000}`.** No NaN, no Inf, no `n=0`, no `a==b`, no `a > b`, no singular endpoint, no oscillatory function (e.g. sin(50x) on [0,π]) where convergence-order claims actually bite. **Below CLAUDE.md "minimum 20" floor.**

---

## 5. SimpsonsRule, line 112

**Math.** Composite Simpson's 1/3. Correct formula. Error `-(b-a)·h⁴·f⁽⁴⁾(ξ)/180`, O(h⁴), exact for cubics — `TestSimpsons_NEquals2` at line 314 is the right pin (Simpson is exact for `x²` at `n=2`, tolerance 1e-14, passes).

**Behavior on bad `n`:**
- `n < 2` → `n = 2` (silent).
- `n` odd → `n++` (silent).

**This silent rounding is dangerous.** A user passing `n = 99` thinks they got `99` subintervals; they got `100`. This affects:
1. **Reported truncation error**: the user's expected `h^4` constant is computed with the wrong `h`.
2. **Convergence-rate experiments**: a user halving `h` by doubling `n` from `99 → 198 → 396 → ...` sees `100 → 198 → 396 → ...` — a non-uniform refinement that breaks Richardson extrapolation in any consumer that wraps Simpson.

**Recommended fix:** either (a) panic with a clear message — consistent with how `optim` and `linalg` reject malformed inputs — or (b) document the rounding *and* return the `n` that was actually used (would require an API break). Option (a) is in keeping with reality's "fail fast" feel.

**Simpson 3/8** (4-point Newton-Cotes, exact for cubics, used to handle the odd-interval remainder when `n` is odd) is **not implemented**. The proper composite Simpson on odd `n` uses 1/3 on the first `n-3` intervals and 3/8 on the last 3; the current "just bump `n`" is the lazy textbook fallback. This matters when the user has *already evaluated* `f` at fixed nodes (e.g. measured data) and cannot re-sample at different `h`.

**Adaptive Simpson** (Lyness 1969 — recursive bisection with error estimate `|S(a,b) - S(a,m) - S(m,b)| / 15`) is missing. Topic asks about "Simpson… composite vs simple form; truncation error claim" — the **error estimate** is what makes Simpson production-grade and it is not exposed. A `SimpsonsRuleAdaptive(f, a, b, tol)` returning `(value, estimatedError, evaluations)` is ~40 LOC and would be the highest-value addition to the file.

**Golden:** 8 vectors (simpsons.json), 7 smooth + one with `1/x` from 1 to e (well-behaved on that interval). Same gaps as trapezoidal: no NaN/Inf, no `n=1` (which gets bumped to 2), no `n` odd boundary, no oscillatory.

---

## 6. GaussLegendre, line 149

**Math.** Hand-coded nodes and weights for n = 2, 3, 4, 5 from Abramowitz-Stegun. Spot-checking against A-S Table 25.4:
- n=2: ±1/√3, w=1 — correct.
- n=3: 0, ±√(3/5), w = 8/9, 5/9 — correct.
- n=4: nodes `±√((3 ± 2√(6/5))/7)`, weights `(18 ± √30)/36` — correct.
- n=5: nodes `±√(5 ± 2√(10/7))/3`, 0; weights `(322 ± 13√70)/900`, 128/225 — correct.

**Bug-class issues:**

1. **Allocation in hot path (CLAUDE.md rule 3 violation).** Every call to `GaussLegendre` rebuilds the entire `rules` map from scratch — that's 4 map entries, 4 `[]float64` literals (≈18 floats), and 12 `math.Sqrt` invocations including nested ones. For a 60 FPS Pistachio integrator (Consumers tag in calculus.go:8) this is ~720 unnecessary allocations and ~720 sqrt calls per second. Trivial fix: lift `rules` to a `var` initialized in an `init()` or as a package-level constant (Go won't let a map literal be `const` but a package-level `var` initialized once is fine).
2. **`points < 2` clamps to 2; `points > 5` clamps to 5 — silent.** Same family as Simpson's silent `n` adjustment. A user requesting `points = 10` thinks they got n=10 (exact for degree 19 polys) and got n=5 (exact for degree 9). For an `exp(x)` integral the relative error jumps from `~1e-22` to `~1e-10` and the user has no signal.
3. **No higher-order rules.** 5-point caps you at degree-9 exactness. The topic does not ask for adaptive Gauss-Kronrod, but for production a 7-point or 8-point rule (or a 7/15 Gauss-Kronrod pair giving an error estimate for free) would be cheap and useful. Press NR §4.5 has the tables.
4. **`a > b`**: returns negative-magnitude result. Same as Trapezoidal.
5. **`a == b`**: `halfLen = 0`, sum is 0, result is 0. OK.

**Golden:** 8 vectors. Good polynomial-exactness coverage (linear, quadratic, cubic, quartic on [0,1], odd cubic on [-1,1]). `1e-14` tolerances on the polynomial cases are appropriate. No NaN/Inf; no `points = 1` (gets clamped to 2 silently and gives wrong-but-plausible answers); no `points = 6` (clamped to 5).

---

## 7. MonteCarloIntegrate, line 244

**Math.** Standard `V·E[f]` estimator. Variance is `V²·Var(f)/N`, RMS error `O(1/√N)` independent of dimension — docstring is correct.

**Issues:**

1. **No length check on `lower`/`upper` vs `dim`.** If `len(lower) < dim` the volume product loop crashes; if `len(lower) > dim` the extra entries are silently ignored. Both should be a panic.
2. **No `lower[i] < upper[i]` check.** Equal bounds → volume 0 → result 0. Reversed bounds → negative volume → integral with the right sign but the wrong magnitude *of* the right sign — actually correct in expectation (`upper-lower` in volume cancels with the mean over a domain `[upper, lower]` if you accept the convention, but the loop body uses `lower + r·(upper-lower)` which samples *outside* `[upper, lower]` if reversed; it samples `[lower + (upper-lower), lower + 0] = [upper, lower]` which is right by accident).
3. **No variance estimate returned.** A real MC integrator returns `(value, stderr)`. Without stderr the user cannot decide convergence. ~5 lines to add (Welford running variance), and it is the **single most useful addition** to this function.
4. **`rng` interface contract.** `interface{ Float64() float64 }` is fine and matches `math/rand.Rand`'s shape, but the docstring should note that the returned value must be in `[0,1)` and that a buggy RNG returning `1.0` will sample `upper` exactly (boundary case).
5. **No stratified / quasi-MC / Sobol / Halton variants.** Plain MC has the worst convergence of any QMC method. For dim ≥ 3 problems a Halton or Sobol low-discrepancy sequence converges as O((log N)^d / N) instead of O(1/√N), often 100-1000× faster. Out of scope for this audit but a `QMCIntegrate` would be a 50-LOC addition with massive consumer benefit (RubberDuck option pricing in particular).

**Tests:** 7 unit tests, no golden file (would require a deterministic seeded reference; doable). `TestMC_Deterministic` (line 502) pins reproducibility under a fixed seed — good.

---

## 8. Cross-cutting numerical-correctness gaps

| Issue | Severity | Fix size |
|---|---|---|
| No `IsNaN`/`IsInf` guards on any input | HIGH — silent NaN propagation through all 5 functions | ~30 LOC + 5 unit tests + ~20 golden vectors |
| Golden coverage 8 vectors/function vs CLAUDE.md "min 20, target 30" | HIGH — port-correctness compromised | ~80 vectors total (10 per function × 4 funcs × 2 IEEE-754 sweep) |
| Simpson silent `n` rounding | MEDIUM — breaks Richardson/convergence experiments | 2 LOC (panic + doc) |
| GL silent `points` clamping | MEDIUM — silent precision loss | 2 LOC |
| GL rebuilds nodes/weights on every call | MEDIUM — 60 FPS allocation hot path | move `rules` to package `var` |
| No `NumericalSecondDerivative` | MEDIUM — Newton's method consumers can't get one | ~5 LOC + tests |
| No 5-point central diff | MEDIUM — O(h⁴) for almost free | ~5 LOC |
| No Richardson extrapolation | MEDIUM — topic explicitly asks for it | ~10 LOC |
| No forward / backward diff | MEDIUM — domain-edge derivatives unsupported | ~10 LOC |
| No adaptive Simpson (with error estimate) | HIGH — production-grade integrator missing | ~40 LOC |
| No Gauss-Kronrod 7/15 (error estimate for free) | LOW — alternative to adaptive Simpson | ~30 LOC + tables |
| No Romberg | LOW — topic asks; would be ~25 LOC | — |
| MonteCarloIntegrate returns no stderr | HIGH — no convergence signal | ~5 LOC (Welford) |
| No QMC / Sobol / Halton | LOW (out of audit scope, but high value) | ~50 LOC |
| `CLAUDE.md` table claim "RK4, root finding" lives in calculus | DOCS BUG | 1 LOC |

---

## 9. Recommended priority ordering

1. **Fix the docs bug** in `CLAUDE.md` package table (calculus does not contain RK4 or root finding).
2. **Add an IEEE-754 sweep golden file** for each of the 4 deterministic functions: NaN integrand, +Inf integrand, denormal `h`, `h = 0`, `a == b`, `a > b`, `points = 1`, `points = 99`. ~30 vectors.
3. **Lift `GaussLegendre.rules` to package `var`.** Single-line code change, removes 720 allocs/sec at 60 FPS.
4. **Replace silent rounding/clamping in Simpson and GL with panics.** 4 LOC, eliminates a class of silent precision loss.
5. **Add `NumericalSecondDerivative`, `NumericalDerivative5pt`, `Richardson`, `NumericalDerivativeForward`.** ~30 LOC total, covers the entire "Numerical differentiation" topic bullet.
6. **Add `SimpsonsRuleAdaptive(f, a, b, tol) (value, errEst, evals)`.** ~40 LOC, single biggest production-grade upgrade.
7. **Add `MonteCarloIntegrate2(...) (value, stderr)`.** ~5 LOC.
8. **Bring all 4 golden files up to ≥20 vectors each** to satisfy CLAUDE.md.

Total: ~150 LOC of additions, ~10 LOC of bugfixes, ~80 new golden vectors. Closes the entire `calculus-numerics` topic without touching `chaos/` or `optim/` (which have their own slots).

---

## 10. Key paths

- `C:\limitless\foundation\reality\calculus\calculus.go` — 275 lines, 5 public functions
- `C:\limitless\foundation\reality\calculus\calculus_test.go` — 523 lines, 30 unit tests + 4 golden harnesses
- `C:\limitless\foundation\reality\testdata\calculus\derivative.json` — 10 vectors
- `C:\limitless\foundation\reality\testdata\calculus\trapezoidal.json` — 8 vectors
- `C:\limitless\foundation\reality\testdata\calculus\simpsons.json` — 8 vectors
- `C:\limitless\foundation\reality\testdata\calculus\gauss_legendre.json` — 8 vectors
- `C:\limitless\foundation\reality\chaos\ode.go` — RK4 lives here (separate audit)
- `C:\limitless\foundation\reality\optim\rootfind.go` — root-finding lives here (separate audit)
- `C:\limitless\foundation\reality\CLAUDE.md` line 31 — package table claim "RK4, root finding" is wrong for `calculus`
