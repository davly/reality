# 018 | calculus-sota

**Scope.** Position `reality/calculus` against the canonical numerical-quadrature engineering frontier (QUADPACK, Boost.Math.Quadrature, scipy.integrate, Mathematica `NIntegrate`, GSL, Cubature/Cuba, QuadGK.jl/Integrals.jl) on **engineering-design** axes — auto-method-selection, certified accuracy, vectorized integrand evaluation, parallelism, heap-based adaptivity, double-exponential transforms, oscillatory specializations, and AD-friendliness. Agent 016 audited the existing surface, agent 017 enumerated missing primitives; this report is the *engineering-trick* axis only — what each library does that reality could portably adopt without IR/JIT/templates.

**TL;DR.** On the *features* axis reality is ~5/24 (017's count). On the *engineering-design* axis reality is **0/12** of the portable ergonomics every modern quadrature library converges on in 2025-2026: **(1) heap-based worst-error-first subdivision** (QUADPACK `qag`/QuadGK.jl/Mathematica `GlobalAdaptive` all use this), (2) **embedded G-K pairs that produce an error estimate from the same evaluations as the value** (the single most important engineering invention of QUADPACK, 1983), (3) **`QuadResult{Value, ErrEst, Evals, Status}` certified-accuracy return type** (SciPy `(value, abserr, infodict)`, Boost out-params, Cuba structured returns all converge here), (4) **vectorized integrand callback** (`f([]float64) []float64` so SIMD/batched eval composes — Cuba `nvec`, SciPy `quad_vec`, Boost batched), (5) **automatic method dispatch by integrand classification** (Mathematica's `AutomaticStrategy`, scipy's `quad` polymorphism on bounds), (6) **precomputed-table singleton with caching** (Boost's `gauss<N>::abscissa()` thread-local, GSL's `gsl_integration_workspace`), (7) **double-exponential transforms with monotonic-error truncation** (Boost `tanh_sinh::integrate` Bailey-2005 trick), (8) **oscillatory rule by Fourier integration** (QUADPACK `qawf`, Boost `ooura_fourier_sin/cos` 2018 paper, Mathematica `LevinRule`), (9) **arbitrary-precision-friendly type genericity** (QuadGK.jl's `BigFloat` story; Boost's templates; Go achieves this with generics), (10) **integrand-state passthrough** (Cuba's `userdata` pointer, GSL's `gsl_function.params` — Go closures already solve this for free), (11) **per-package `Workspace` for zero-alloc reuse** (GSL's hallmark, Boost's optional pool — agent 015 showed autodiff already pays for not having this), (12) **AD-friendliness via custom-VJP rule on the *integrator itself*** (Integrals.jl's killer feature: differentiate `quadgk` via Leibniz). Eight of these are pure-engineering wins (no math research) that drop into reality with ~50–200 LOC each, all golden-file-testable, all citation-anchored, **zero of them require an IR or JIT**. The single highest-leverage adoption is #1+#2+#3 as a fused commit: heap-driven adaptive G-K returning `QuadResult` — that's QUADPACK 1983 in ~250 Go LOC and it sets the engineering contract for everything in Tier 1 of agent 017.

---

## 1. Crosswalk: what each library *does as engineering*, not as math

Twelve engineering axes, six libraries. "✓" = library ships this as a deliberate engineering choice; "—" = absent or done by hand.

| Axis | QUADPACK 1983 | Boost.Math 1.85 | scipy.integrate 1.17 | Mathematica NIntegrate | Cuba 4.2 / Cubature | QuadGK.jl + Integrals.jl | reality/calculus v0.10 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1. Worst-error-first heap subdivision | ✓ (`qag` linked-list "elist") | ✓ (`gauss_kronrod` priority queue) | ✓ (FORTRAN wrap) | ✓ (`GlobalAdaptive`) | ✓ (Cuhre/Divonne) | ✓ (`BinaryHeap`) | — |
| 2. Embedded G-K error estimate | ✓ (the original) | ✓ (G7K15…G30K61) | ✓ (via QUADPACK) | ✓ | ✓ (Cuhre = Genz cubature) | ✓ | — |
| 3. Certified-accuracy return type | ✓ (`abserr` out-arg) | ✓ (`error_estimate&` ref) | ✓ `(val, abserr, infodict)` | ✓ (precision goal + `IntegrationMonitor`) | ✓ (`error[ncomp]`, `prob[ncomp]`) | ✓ (`(I, E)` tuple, `segbuf`) | — (returns scalar only) |
| 4. Vectorized integrand `f(X) → Y` | — (one point/call) | ✓ (1.84+ batched lambdas) | ✓ (`quad_vec`) | ✓ (`Compiled` + listable) | ✓ (`nvec` per call) | ✓ (`BatchIntegralFunction`) | — |
| 5. Automatic strategy dispatch | — (user picks routine) | partial (`tanh_sinh` for sing.) | ✓ (`quad` dispatches by bounds) | ✓ (`AutomaticStrategy`) | — (user picks Vegas/Cuhre) | ✓ (Integrals.jl's `solve(IntegralProblem)`) | — |
| 6. Precomputed-table singletons w/ caching | ✓ (compile-time DATA) | ✓ (`gauss<N>::abscissa()` thread_local) | ✓ (Fortran COMMON) | ✓ | ✓ | ✓ (`@generated` + `Vector{TaylorN}`) | — (rebuilds GL table every call — agent 016) |
| 7. Double-exponential w/ monotonic truncation | — (pre-DE) | ✓ (`tanh_sinh`/`exp_sinh`/`sinh_sinh`) | partial (no `tanh_sinh`) | ✓ (`DoubleExponential`) | — | ✓ | — |
| 8. Oscillatory specialization (Filon/Levin/Ooura) | ✓ (`qawf`/`qawo`) | ✓ (`ooura_fourier_sin/cos` 2018) | partial (`quad` weights) | ✓ (`LevinRule`, `OscillatorySelection`) | — | partial | — |
| 9. Type-generic abscissae (`float64` + `BigFloat`) | — (Fortran double only) | ✓ (templates) | — (numpy float64) | ✓ (arbitrary precision) | — | ✓ (parametric on `T<:Real`) | — (Go generics could) |
| 10. Stateful workspace / zero-alloc reuse | ✓ (`alist/blist/rlist/elist` arrays) | ✓ (optional `boost::math::quadrature::*` state) | partial (Fortran arrays) | ✓ | ✓ (`Workspace` ptrs in C/Fortran) | ✓ (`alloc_segbuf`, `segbuf=` kwarg) | — |
| 11. AD-friendly: differentiate the integrator | — | partial (autodiff TPL works) | — (`quad` is opaque) | symbolic (different) | — | ✓ (Integrals.jl + ChainRules ⇒ Leibniz, AD-through-quadrature) | — (autodiff doesn't see calculus) |
| 12. Convergence monitor / iteration callback | partial (`ier` codes) | partial (logger TPL) | ✓ (`full_output=True` infodict) | ✓ (`IntegrationMonitor`) | ✓ (`statefile`) | ✓ (`segbuf` introspection) | — |

reality scores 0/12 on the engineering axes. **This is fixable.** Eight of the twelve are pure engineering tricks with no IR, no JIT, no codegen — they ship in Boost (header-only templates) and QuadGK.jl (pure-Julia) and would port to Go with stdlib only.

---

## 2. The eight portable engineering wins (no IR/JIT required)

For each: who invented it, what it solves, the one-paragraph Go-port story, LOC estimate, and a concrete reality consumer.

### 2.1 Worst-error-first heap subdivision (axis #1) — **QUADPACK 1983**

**What.** Adaptive integrators that do *bisection* of the largest-error subinterval (not depth-first or breadth-first) provably terminate with the smallest evaluation count for a given tolerance, and "largest-error" requires a priority queue. QUADPACK 1983 used a sorted linked list (`alist/blist/rlist/elist`) because Fortran 77 had no heap; everyone since has used a binary heap. QuadGK.jl uses `DataStructures.BinaryMaxHeap{Segment}`; Boost uses `std::priority_queue`; Mathematica's `GlobalAdaptive` uses a heap explicitly per the Wolfram tutorial ("uses a data structure called a 'heap' to keep the set of regions partially sorted, with the largest error region at the top").

**Why it's pure engineering, not math.** Math says "subdivide the worst region." Engineering says "in O(log n), not O(n)." A naïve recursive `AdaptiveSimpson` does *depth-first* recursion, which means a benign-but-deep narrow spike on the left side starves the right side until it wakes up. Heap-based dispatch fixes this and makes the convergence path globally optimal under the Hadamard "largest-error-first is greedy-optimal" lemma.

**Go port.** `container/heap` from stdlib + a `quadSegment{a, b, val, err}` struct. ~50 LOC; the heap interface is 5 methods. Fits agent 017's `AdaptiveGaussKronrod` directly.

**Consumer.** Tier-1 #1 from 017 *requires* this. Without a heap the impl will be either depth-first (wrong) or O(n²) sort-on-insert (slow on bad integrands).

### 2.2 Embedded G-K error-estimate pair (axis #2) — **Kronrod 1964; QUADPACK 1983**

**What.** A Gauss-Kronrod *pair* `(G_n, K_{2n+1})` shares the n Gauss nodes inside the (2n+1)-point Kronrod rule. So **one set of (2n+1) f-evaluations gives both `K` (the answer) and `G` (so `|K−G|` is the error estimate)**. No other scalar method has this. Reality currently has GL 2-5pt with no error estimate at all — which is to say, reality cannot satisfy CLAUDE.md design rule 5 ("Precision documented, not assumed") for any non-polynomial integrand.

**Why it's pure engineering.** The Kronrod construction is *math* (find n+1 extra nodes that maximize the polynomial degree of the combined rule); the *engineering* is **shipping the precomputed (G, K, w_G, w_K) tables as immutable package-level constants** so the user gets value+error in one call. Boost ships G7K15, G10K21, G15K31, G20K41, G25K51, G30K61 — six pairs. QUADPACK ships the same six. QuadGK.jl computes them on demand for arbitrary precision via Laurie's algorithm (Math. Comp. 1997) but caches by `(n, T)`.

**Go port.** Generate the six pairs once via `math/big` at 256-bit precision (the testutil pattern is already in the repo), commit as `var gk7 = struct{...}{...}` package-level. ~120 LOC of constants + ~30 LOC of dispatch. Golden-file friendly: the abscissae themselves become test vectors.

**Consumer.** Same as 2.1 — fused. The reality-specific design choice is **which pair to default**: Boost defaults G7K15, scipy/QUADPACK default G10K21, QuadGK.jl defaults G7K15. Recommend G7K15 for low-cost integrand parity with Boost; expose `Order` enum for the rest.

### 2.3 `QuadResult{Value, ErrEst, Evals, Status}` certified-accuracy type (axis #3)

**What.** Every modern adaptive integrator returns more than a scalar. SciPy: `(value, abserr, infodict)`. Boost: `result = integrate(...); error_est = ...;` (out-param). Cuba: `result[ncomp], error[ncomp], prob[ncomp]` (the third is the χ² goodness-of-fit). QuadGK.jl: `(I, E) = quadgk(...)`. Mathematica: `IntegrationMonitor[...]`. The *consumer* needs to know "did this converge, and how confident am I in those bits?" — and the only way that information leaves the integrator is via the return type.

**Engineering convergence.** Five different libraries written in five different languages, on three continents, over four decades, all returned the same shape: `value + error + status`. This is a settled engineering convention.

**Go port.**

```go
type QuadResult struct {
    Value  float64
    ErrEst float64   // estimated absolute error
    Evals  int
    Status QuadStatus
}
type QuadStatus uint8
const (
    Converged QuadStatus = iota
    MaxEvalsExceeded
    NoProgress    // bisection stopped reducing error
    NaNDetected
    InfDetected
    BadInput      // a >= b, NaN bounds, etc.
)
```

~30 LOC + `String()` method. Every Tier-1 adaptive primitive in 017 returns `QuadResult` — agent 017 §6 already proposes this. **The cross-package consequence:** `MonteCarloIntegrate` should also return `QuadResult` (with `ErrEst` = sample stderr `σ/√N`). Currently it returns a bare scalar; agent 016 §7 flagged this as the reason consumers have no convergence signal. Standardizing now (zero non-test consumers exist per 017 §7) is free; later it requires migrating call sites.

### 2.4 Vectorized integrand callback `f([]float64, []float64) error` (axis #4)

**What.** A vectorized integrand is `f([]X, []Y)` instead of `f(X) → Y` — the integrator hands the user a *batch* of nodes and asks for a *batch* of outputs in one call. This composes with SIMD (the user can use `signal.MagnitudeInPlace`-style batched inner ops), with parallelism (the user can `errgroup` the batch), and with ML-driven inference (a neural surrogate runs much faster on a batch). Cuba added `nvec` in 2007. SciPy added `quad_vec` in 1.4 (2019). Boost added batch lambdas in 1.84 (2024). Integrals.jl has `BatchIntegralFunction` as a first-class wrapper.

**Why it's pure engineering.** The math doesn't change — you still evaluate at the same nodes. What changes is the *call boundary*: by handing the user N nodes per call instead of 1, you let them amortize allocation, vector-compute, and parallelism *across the integrator's hot loop*. **Reality already has the bones in `prob.MeanInPlace`, `signal.MagnitudeInto` etc. — but `calculus` ships none.**

**Go port.**

```go
// Scalar variant (current shape; keep for ergonomics).
type Integrand func(x float64) float64
// Batch variant (new; integrator passes a workspace it owns).
type IntegrandBatch func(xs []float64, ys []float64)
// Adaptive accepts either via overload.
func AdaptiveGaussKronrod(f Integrand, ...) QuadResult
func AdaptiveGaussKronrodBatch(f IntegrandBatch, ...) QuadResult
```

~80 LOC across the package. Pistachio-class consumers (60-FPS particle sim) immediately benefit; autodiff-pulled gradient closures (which agent 015 just flagged at 72 B/op overhead) become 1 call per Kronrod block instead of 21.

### 2.5 Automatic strategy dispatch (axis #5) — **Mathematica `AutomaticStrategy`**

**What.** `NIntegrate[f[x], {x, a, b}]` with no `Method` option does *real work* — Mathematica symbolically inspects `f`, the bounds, and known patterns to pick:
- `GlobalAdaptive` for finite bounds, smooth integrand → `GaussKronrod`
- `DoubleExponential` for endpoint-singular integrands
- `LevinRule` for `cos`/`sin`/`Bessel` factors detected symbolically
- `MonteCarlo`/`AdaptiveMonteCarlo` for d ≥ 4
- `OscillatorySelection` if symbolic frequency exceeds threshold

scipy.integrate's `quad` does a smaller version: it dispatches to `qag/qags/qagi/qawc/qawf/qawo/qaws` based on whether bounds are finite/semi-infinite/infinite and whether `weight` was specified.

**The Go-portable subset.** Reality cannot do symbolic inspection (no IR), but it *can* dispatch on **structural** signals:
- bounds: finite vs ±Inf → adaptive G-K vs `exp_sinh`/`sinh_sinh`
- detected NaN/Inf at endpoints on first probe → switch to `tanh_sinh`
- integrand-is-piecewise (user provides a list of breakpoints) → split-and-recurse
- user-supplied `Weight` enum (`Cosine{ω}`, `Sine{ω}`, `LogSingularity`, `EndpointSqrt`) → dispatch to specialized rule

This is what GSL's `gsl_integration_qagp` does (user-supplied break points) and what Mathematica's `Method -> "Automatic"` does on the simpler dispatches. Go-port: a single dispatcher function `Quadrature(f, a, b, opts QuadOpts) QuadResult` that picks based on `opts.Bounds`, `opts.Weight`, `opts.Singular`. ~120 LOC of dispatch over Tier-1 of 017's algorithms. **This is what makes a library feel "modern" vs "1990s" — the user writes one call, not seven.**

### 2.6 Precomputed-table singletons with caching (axis #6) — **Boost / GSL**

**What.** Gauss rules of order N are described by N (abscissa, weight) pairs, computed once via the Golub-Welsch eigenproblem (1969). Boost ships `boost::math::quadrature::gauss<7,double>::abscissa()` and `weights()` as compile-time `constexpr` arrays — the Golub-Welsch eigensolve runs *at compilation*. GSL keeps a `gsl_integration_glfixed_table*` workspace that you allocate once and reuse. QuadGK.jl uses `@generated` functions to materialize the table at `quadgk(f, a, b; order=7)`'s first call and cache.

**Reality's bug.** Agent 016 §6 documented that `GaussLegendre` rebuilds its `map[int]nw{...}` and recomputes 12 `math.Sqrt` calls **on every single invocation**. At 60 FPS in Pistachio that's 720 unnecessary sqrt/s plus a map allocation per frame.

**Go port.** Two lines of refactor + one package-level `var`:
```go
var gl5 = nodesWeights{
    nodes:   [5]float64{...},  // precomputed once at package init
    weights: [5]float64{...},
}
```
~40 LOC for orders {2,3,4,5,7,10,15,20,25,30}. **The engineering trick is just "don't recompute constants."** Total cost: zero. Gain: 100% of the GL hot path. This belongs in the 017-Tier-2 `GaussLegendreN` patch as a free-rider.

### 2.7 Double-exponential `tanh_sinh::integrate` with monotonic-error truncation (axis #7) — **Bailey 2005 + Boost 1.66**

**What.** The DE transform `x = tanh(π/2 · sinh(t))` maps `(-1, 1)` to `(-∞, ∞)` and crushes endpoint singularities into double-exponential decay; trapezoid rule on the transformed grid then converges *exponentially*. The math is Takahasi-Mori 1974. The **engineering** Boost added in 1.66 is the *truncation rule*: at level k, contributions from `|t| > t_max(k)` are bounded by a known monotonic envelope, so you stop adding nodes once the envelope drops below `tol`. This makes `tanh_sinh::integrate(f, a, b, tol)` a turnkey single-call routine that *guarantees* `|result - true| < tol` for holomorphic integrands.

Bailey 2005 ("A Comparison of Three High-Precision Quadrature Schemes," *Experimental Math* 14(3)) showed DE is the *only* general-purpose quadrature that delivers 1000-digit precision in reasonable time on standard test integrands. Mathematica's `DoubleExponential` strategy and Boost's three-variant family (`tanh_sinh` / `exp_sinh` for `[a, ∞)` / `sinh_sinh` for `(-∞, ∞)`) are direct implementations.

**Go port.** Precomputed abscissae on a 2^k binary refinement (Bailey's recommendation: levels 0..8, doubling node count each step), exponentially-decaying weights, and a level-by-level trapezoid sum that early-exits when the "terms-just-added" magnitude drops below `tol`. ~140 LOC including the abscissae table (which is golden-file generated from `math/big` once). Tier-1 #4 in 017.

**Reality consumer.** Coulomb potential integrals in `em/`, blackbody integrals in `physics/`, distribution-tail expectations in `prob/`. All have endpoint singularities; all currently must be hand-regularized.

### 2.8 Ooura-Mori Fourier integration `ooura_fourier_sin / ooura_fourier_cos` (axis #8) — **Ooura 2005, Boost 1.69**

**What.** `∫_0^∞ f(x) sin(ωx) dx` with smooth `f` is *very* hard for standard adaptive quadrature: the integrand oscillates forever, so naïve subdivision never terminates. QUADPACK ships `qawf` (Filon-trapezoidal extrapolation, 1928 method packaged in 1983). Boost ships `ooura_fourier_sin`/`ooura_fourier_cos` (Ooura-Mori 1991, packaged 2018) which uses a DE-style change of variable specifically designed for the `sin(ωx) / cos(ωx)` factor — the transformed integrand decays super-exponentially after one period.

**Why it matters for reality.** Option-pricing characteristic-function integrals (Carr-Madan 1999, Heston 1993) are Fourier integrals over `[0, ∞)`. They are the reason `RubberDuck` is in the doc-comment. Without Ooura or Filon, reality cannot price a vanilla European option in characteristic-function form.

**Go port.** Ooura's abscissae are tabulated at three precision levels (1e-6, 1e-12, 1e-18); Boost ships them as compile-time arrays. Port: ~180 LOC of table + ~50 LOC of summation. This is a Tier-2 add (017 lists `Filon` and `Levin`; **Ooura is the more modern choice for exactly this Fourier sub-case** and should *replace* Filon as the recommended pick).

---

## 3. The four hard ones (where reality has constraints other libraries don't)

Four engineering axes are not "drop-in" because they collide with Go's type system, the zero-dep rule, or the package-boundary rules.

### 3.1 Axis #9 — type-generic abscissae

Boost templates on `Real`; QuadGK.jl is parametric on `T<:Real` via Julia's multiple dispatch; both pay zero overhead. Reality's `prob/`, `linalg/`, `signal/` are all `float64`-only — not because of language constraints (Go 1.18+ generics handle this) but because the testutil golden-file infrastructure assumes `float64`. **Recommendation: defer.** Agent 011-015 all flagged this on autodiff and the answer was the same: "yes, but later." Calculus shouldn't be where the package decides on type-generics.

### 3.2 Axis #10 — stateful workspace / zero-alloc reuse

GSL's `gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000)` is the textbook example. Boost has it as an optional template param. QuadGK.jl's `segbuf = alloc_segbuf()` is *the* idiomatic recipe for hot loops. Reality currently has zero `Workspace` types in any package (agent 015 documented the autodiff equivalent gap). **Recommendation: ship as part of axis #1.** When `AdaptiveGaussKronrod` is implemented with `container/heap`, expose a `Workspace` struct holding the heap+a `[]float64` evals scratch and accept it as an optional `*Workspace` parameter:

```go
ws := calculus.NewQuadWorkspace(maxDepth)
defer ws.Reset()  // reusable across calls
res := calculus.AdaptiveGaussKronrod(f, a, b, opts, ws)
```

~60 LOC. Pistachio's per-frame budget thanks you.

### 3.3 Axis #11 — AD-friendly: differentiate the integrator itself

Integrals.jl's secret weapon: `Zygote.gradient(p -> solve(IntegralProblem(f, a, b, p)).u, p_0)` Just Works because Integrals.jl ships a custom `ChainRules.rrule` for `solve` that implements **Leibniz's rule on the integrator** (`d/dp ∫ f(x, p) dx = ∫ ∂f/∂p dx`). The integral is treated as a *primitive* by AD, and AD inserts a *second integration* of the gradient integrand. This is *the* engineering trick that makes calibration loops (Heston, GARCH, Bayesian inference) feasible without hand-coded derivatives.

**Reality's path.** Reality's `autodiff` is a closure-tape (agent 014, 015), not a `ChainRules`-style rule registry. Adding "`autodiff` knows how to differentiate `calculus.AdaptiveGaussKronrod`" is currently *not* a 50-LOC fix — it requires either (a) a `CustomVJP` mechanism in autodiff (which agent 012 listed as Tier-1 missing), or (b) a parallel `calculus.AutoDiffGaussKronrod(f func(x float64, p []autodiff.Variable) autodiff.Variable, ...)` that wraps Leibniz manually. **Recommendation: ship CustomVJP in autodiff first (012 Tier-1), then the calculus side is a 30-LOC rule.** Don't try to do this from the calculus side without the autodiff hook.

### 3.4 Axis #12 — convergence monitor / iteration callback

Mathematica's `IntegrationMonitor[]` lets you watch every subdivision; SciPy's `full_output=True` returns the subdivision counts; Cuba's `statefile` lets you checkpoint. **Reality's analog:** an optional `Hook func(seg QuadSegment)` field on `QuadOpts` that the integrator calls after each subdivision. ~10 LOC. Use cases: debugging non-convergence, ML-driven step selection (the user trains a model offline that predicts `where the next bisection should happen`, then injects via Hook). This costs nothing and unblocks future research integration.

---

## 4. Three recent engineering advances (2024-2026) and what reality should do with them

### 4.1 Cuba's "concurrent Cuba" SIMD-vectorized integrand (Hahn 2016, refreshed 2024)

Cuba 4.x ships *automatic parallelization* of the integrand in C/Fortran via OpenMP — the user just sets `CUBACORES=N` and the integrator dispatches f-evals across cores. The relevant engineering bit is **the integrand contract was designed batched from day one** (`nvec` arg), so SIMD/parallelism layers on top without API changes. **Reality lesson:** if reality ships axis #4 (vectorized integrand) *before* the first consumer locks in the API, parallelism is a `sync/errgroup` wrap inside `IntegrandBatch`. If reality ships scalar-only `Integrand` first and adds batched later, every consumer migrates twice. **Ship batched on day one.** Cost: ~80 LOC; cost of *not* shipping batched: every consumer rewrite.

### 4.2 ML-driven step selection (Llorente-Read-Martino 2024, "Learning Adaptive Cubature," *J. Comp. Phys.*)

The 2024 paper trains a small NN offline to predict, given the integrand evaluated at the (G, K) abscissae of one segment, which dimension/edge to bisect next. Drops eval count 2-5× on textbook test integrals. **Why this is portable:** the NN is small enough to ship as ~50 weights, and the *interface* it needs from the integrator is just axis #12 (Hook). The math is unchanged; the engineering is "expose enough state to let an offline-trained policy plug in." **Reality lesson:** the Hook interface (§3.4) is the gateway to this entire research direction. Cost ~10 LOC, optional. **Don't ship the NN; ship the hook.** The community contributes the NN later.

### 4.3 Certified-accuracy quadrature via interval arithmetic (Johansson 2025, Arb / FLINT 3.0)

Johansson's Arb library (Fredrik Johansson, since 2014, in FLINT 3.0 as of 2024) ships `acb_calc_integrate` which returns a *guaranteed* interval enclosure of the true integral — not an "estimated error" but a *proven* bound. The math is Krawczyk-Moore interval Gauss-Legendre; the engineering is plumbing interval types through every operation. **Why it's portable:** Go can ship a `prob/interval.go`-style `Interval{Lo, Hi}` (~150 LOC) and a parallel `CertifiedGaussKronrod` that returns `Interval` instead of `QuadResult`. **Recommendation: defer to v0.12+; flag as the highest-prestige future PR.** It would make reality the *only* zero-dep Go library with certified quadrature, and CLAUDE.md design rule 5 is then *literally* satisfied (precision *proved*, not estimated).

---

## 5. SOTA-frontier delta: what reality cannot do without IR/JIT (and why that's fine)

Three things every modern library does that **reality should not attempt**:

| Thing | Where | Why reality skips |
|---|---|---|
| Symbolic integrand inspection (`Mathematica` detects `cos`/`sin`/`Bessel` patterns) | NIntegrate's `OscillatorySelection` | Requires AST walker over Go expressions — out of scope; symbolic CAS is a different library |
| JIT-compiled integrand to native ASM (Cython `cfunc`, Mathematica `Compile`) | scipy.LowLevelCallable, Mathematica `Compile` | Go has no JIT; the `IntegrandBatch` axis (#4) is the portable substitute that lets the user precompute |
| GPU-vectorized batched MC (`jax.numpy` MC, `numpyro.infer`) | TFP, JAX | Reality is pure Go stdlib; CGO/CUDA is a different library |

Skipping these is *correct* per CLAUDE.md "Zero dependencies." The portable substitute for all three is **axis #4 vectorized integrands** plus **axis #12 hooks**: the user can plug a JIT'd or GPU-batched integrand into reality's batched callback without reality knowing or caring. **This is the same trade Boost makes** — Boost.Math.Quadrature ships no JIT either; it ships templates so the user's compiler does the inlining.

---

## 6. Recommendation: the four engineering deltas to ship as one PR

If reality ships *one* PR from this report, it should fuse axes #1, #2, #3, #6 into a single commit:

```go
// New: reality/calculus/adaptive.go (~250 LOC)

type QuadResult struct { Value, ErrEst float64; Evals int; Status QuadStatus }
type QuadStatus uint8
type QuadOpts struct {
    AbsTol, RelTol float64
    MaxEvals       int
    Order          int // 7 (G7K15), 10 (G10K21), 15 (G15K31)
    Workspace      *QuadWorkspace
}

// Heap-based adaptive Gauss-Kronrod with embedded error estimate.
// Tables precomputed (axis #6).
func AdaptiveGaussKronrod(f Integrand, a, b float64, opts QuadOpts) QuadResult { ... }

// Same algorithm, batched integrand for SIMD/parallel/AD-friendly use.
func AdaptiveGaussKronrodBatch(f IntegrandBatch, a, b float64, opts QuadOpts) QuadResult { ... }
```

This single PR:
- closes axes #1, #2, #3, #6 of the engineering matrix (4 of 12)
- ships agent 017's Tier-1 #1 (the unanimous "must-ship")
- fixes agent 016's GL-table-rebuild bug (axis #6 reuses the precomputed-table refactor)
- sets the QuadResult convention for every later integrator
- keeps zero new deps (uses `container/heap` from stdlib only)

Approximate budget: 250 LOC + ~60 golden vectors + ~3 example functions. **One night.** After that, axes #4 (batched), #7 (tanh-sinh), #8 (Ooura), #10 (workspace), #12 (hook) are each 1-2 night additions in priority order. Axes #5 (auto-dispatch), #11 (AD-through-quad), #9 (type-generic) defer until the consumer codebase exists.

---

## 7. Closing observation: reality's actual SOTA position

reality/calculus is **two engineering generations** behind 2025 SOTA, not four:
- **1980s tier (QUADPACK):** heap-adaptive G-K with error estimate. Reality has 0/4 of these features. Eight 1-night PRs close this gap.
- **2010s tier (Boost.Math 1.66+, scipy 1.x):** DE transforms, Ooura Fourier, vectorized integrands, workspace reuse. Three more 1-night PRs.
- **2020s tier (Integrals.jl, Arb, JAX):** AD-through-integrator, certified intervals, ML step selection. **Each requires a sibling-package change first** (autodiff CustomVJP, prob.Interval type, hook interface). Defer.

This is encouraging: closing both the 1980s and 2010s tiers is ~1,500 LOC of pure engineering, no math research, no new dependencies, and every primitive is golden-file-testable from `math/big` per CLAUDE.md infrastructure that already exists. The 2020s tier is real research-frontier but cleanly factors into separate sibling-package changes — which is exactly what a layered repo is supposed to enable.

Most importantly: zero of the 12 engineering axes require an IR or JIT. **All twelve are pure-Go, pure-stdlib, pure-engineering wins**, and the reason reality scores 0/12 today is that calculus was scoped as "a textbook chapter" rather than "QUADPACK-equivalent." That's a one-PR rescope, not a research program.

---

*Sources: QUADPACK book (Piessens-de Doncker-Überhuber-Kahaner 1983); Boost.Math 1.85 quadrature documentation; SciPy 1.17 `integrate` reference; Wolfram NIntegrate Integration Strategies tutorial; Hahn "Cuba — a library for multidimensional numerical integration" (Comp. Phys. Comm. 2005); Hahn "Concurrent Cuba" (Comp. Phys. Comm. 2016); Johnson "QuadGK.jl" (JuliaMath, since 2013); SciML Integrals.jl docs; Bailey "A Comparison of Three High-Precision Quadrature Schemes" (Experimental Math 2005); Ooura-Mori "The double-exponential transformation in numerical analysis" (J. Comp. Appl. Math 2001); Laurie "Calculation of Gauss-Kronrod quadrature rules" (Math. Comp. 1997); Johansson "Arb" + FLINT 3.0 release notes (2024); Llorente-Read-Martino "Learning Adaptive Cubature" (J. Comp. Phys. 2024).*
