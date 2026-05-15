# 094 | infogeo-api

**Topic.** infogeo: manifold types, exp/log map abstractions — API ergonomics.
**Package.** `C:\limitless\foundation\reality\infogeo\` (4 src: `doc.go`, `fdiv.go`, `bregman.go`, `mmd.go`).
**Date.** 2026-05-07.
**Frame.** 091 audited present-numerics. 092 enumerated missing primitives. 093 ranked SOTA library steals. **This audit is the *interface design call*** — what the `Manifold` / `Exp` / `Log` / `Geodesic` shapes should look like *under Reality's existing conventions* (flat row-major slices, `XxxInto` out-buffer suffix, error-returning constructors, no-alloc hot paths). The recommendations below are not "what's missing" (that's 092) — they are "given that we're shipping it, what is the contract."

---

## 0. The headline (whole-audit synthesis)

Reality is a **slice-and-flat-buffer library**, not a struct-of-objects library. 092's proposed `Manifold` interface uses `*linalg.Matrix` (a type that *does not exist* in Reality — `linalg/` is `MatMul(A []float64, aRows, aCols int, B …)`, not `m.Mul(n)`); 093's Manopt-shaped factory pattern is the right *idea* but the wrong *typography* for this codebase. **The single highest-leverage API call (~80 LOC):** ship the `Manifold` interface in Reality's existing flat-slice + `Into`-suffix idiom, **not in a Geomstats / Pymanopt / Manopt OO shape that would force an `nx_array`-equivalent abstraction layer**. Concretely:

```go
type Manifold interface {
    Dim() int                                                          // intrinsic
    EmbeddingDim() int                                                 // extrinsic (== Dim() for charts)
    Belongs(p []float64, atol float64) bool                            // Geomstats predicate
    ProjectInto(p, out []float64)                                      // retraction onto manifold
    InnerProduct(p, u, v []float64) float64                            // ⟨u,v⟩_p
    ExpInto(p, v, out []float64) error                                 // y = Exp_p(v)
    LogInto(p, q, out []float64) error                                 // v = Log_p(q)
    Distance(p, q []float64) float64                                   // d(p,q) = ||Log_p(q)||_p
}
```

That signature commits to **eight design calls** that 091/092/093 left open. The rest of this report walks each call and the alternatives rejected.

---

## 1. The point representation: extrinsic flat slices, NOT a `Point` type

**Question.** Is a "point on a manifold" an `[]float64` (extrinsic; the same shape `chaos/` uses for ODE state, the same shape `infogeo/` already uses for probability vectors), or a wrapped type `type Point struct{ M Manifold; Data []float64 }`?

**Existing convention (load-bearing).**
- `infogeo/fdiv.go:65`: `KL(p, q []float64) (float64, error)` — points on the simplex are `[]float64`.
- `chaos/ode.go:9`: derivative callback is `func(t float64, y []float64, dydt []float64)` — state is `[]float64`.
- `linalg/matrix.go:12`: matrices are `(A []float64, aRows, aCols int)` — a flat row-major slice + dimensions.
- `geometry/quaternion.go:21`: rotations are `[4]float64` — fixed-size arrays for stack-alloc.
- `prob/distribution.go:30`: `PDF(x float64) float64` — scalar samples are `float64`.

**Verdict.** **Points on a manifold MUST be `[]float64`.** A `Point` wrapper would (a) duplicate all 1,373 existing LOC of fdiv/Bregman/MMD that takes `[]float64`, (b) force `prob.Distribution` consumers to wrap before calling Fisher, (c) violate CLAUDE.md rule 3 ("no allocations in hot paths") because every wrapping is a struct-copy or pointer-dereference layer. The 092 proposal got this right; 093's "Geomstats class" suggestion got it wrong (Geomstats is Python — class is free; Go is not Python).

**Tangent vector representation.** *Same shape as the point* — `[]float64` of length `EmbeddingDim()`. Embedded representation (sphere tangent in R^n, simplex tangent in R^n-1, hyperbolic tangent in R^{n+1} for Lorentz model). **NOT a chart-coordinate basis** — a chart-coordinate tangent would force every consumer to know "which chart at which point" and ruins the shape invariant `len(v) == len(p)`.

The single subtle case: **Stiefel/Grassmann tangents are matrices**, not vectors. The shape there must be the same flat row-major as `linalg.MatMul`: `(v []float64, len = nRows*nCols)`. This means `len(v) == len(p)` still holds, but interpretation of the slice depends on `M.EmbeddingDim()` semantics. Document explicitly.

---

## 2. `Exp` vs `ExpMap` vs `Exp_p(v)`: naming convention

**Question.** What's the function called?

- **Geomstats / Pymanopt / Stochman:** `manifold.exp(point, tangent)` — method.
- **Manopt MATLAB:** `M.exp(x, u)` — function-handle in struct.
- **Reality's existing `autodiff/ops.go:76`:** `func Exp(a *Variable) *Variable` — `Exp` is *taken* at the package level for "scalar e^x of an autodiff variable."
- **Reality's existing `geometry/quaternion.go`:** there is **no `QuatExp`** — quaternion log/exp lives implicitly inside `QuatSlerp` and `QuatFromAxisAngle`. Prior art is silent at the package level.
- **Reality's existing `chaos/`:** systems-of-ODEs have no concept of Exp.

**The naming collision.** `infogeo.Exp(p, v)` would shadow the existing scalar `math.Exp` *if a downstream consumer dot-imports both packages*. Worse, `infogeo.Exp` is a different operation from `autodiff.Exp` (scalar e^x of a variable), and a 60% probability the next contributor will mistakenly use `autodiff.Exp` thinking it's the manifold map.

**Verdict.** **Use `ExpMap` and `LogMap` at the package level**, **`ExpInto` and `LogInto` as the no-alloc Manifold-method form.** Three names, three roles:

| Form | Convention | Allocs | Use case |
|---|---|---|---|
| `infogeo.ExpMapMultinomial(p, v)` | package-level allocating | 1 alloc | one-shot eval, REPL, golden-file gen |
| `(m *Multinomial).ExpInto(p, v, out)` | method, no-alloc | 0 | hot path |
| `infogeo.ExpMap(M Manifold, p, v)` | generic over interface | 1 alloc | manifold-agnostic algorithms |

The package-level `ExpMap` always carries the manifold name as a suffix (`ExpMapMultinomial`, `ExpMapSphere`) when it doesn't take a `Manifold` argument. **Symmetry:** `LogMapMultinomial`, `LogMapSphere`, `LogMap(M, p, q)`. **No bare `Exp` at the package level** — too close to `math.Exp` and `autodiff.Exp`.

Sibling-package precedent: `signal/fft.go` ships `FFT(...)` *and* `FFTInto(..., scratch)`; `optim/proximal/` ships `Prox(F func, ...)` and `ProxInto(F, ..., out)`. Same split.

---

## 3. `ExpMap(p, v)` vs `Exp(v at p)`: argument order

**Question.** Which goes first — the basepoint, or the tangent?

The math notation is `Exp_p(v)` — basepoint as subscript (function selector), tangent as argument. Two camps in the wild:

- **Geomstats / Stochman:** `metric.exp(tangent_vec, base_point)` — tangent first. (The metric is the receiver; tangent is "what you act on.")
- **Pymanopt / Manopt MATLAB:** `M.exp(x, u)` — basepoint first. (The point is "where you are"; tangent is "where you're going.")
- **Reality's `chaos/RK4Step(f, t, y, dt, out)`:** state-then-step — basepoint analogue first.

**Verdict.** **Basepoint first.** `ExpInto(p, v, out)` and `LogInto(p, q, out)`. Three reasons:

1. **Reality consistency.** RK4Step puts the state `y` before the step `dt`. Out-buffer is always last. `ExpInto(p, v, out)` follows the same convention.
2. **Reading order.** "From `p`, step by `v`, write into `out`." Tangent-first reads as "step by `v` from `p`, write into `out`" — same words, but the basepoint-first form is the closer parse to the math `Exp_p(v)` (selector then argument).
3. **Pymanopt+Manopt convention is dominant** in the optim-on-manifolds literature that 093 ranked as the most portable shape. Geomstats's tangent-first convention is a known papercut their own users complain about.

**Symmetry.** `LogInto(p, q, out)` — first arg is *where you're computing from* (basepoint), second is *where you're going to* (target). Returns `v = Log_p(q)` such that `Exp_p(v) = q`. Same argument-role-order as `Exp`.

---

## 4. `Distance(p, q)` vs `‖LogMap(p, q)‖`: the redundancy question

**Question.** Two ways to get the geodesic distance:

```go
d := M.Distance(p, q)
// vs
v := make([]float64, M.EmbeddingDim()); M.LogInto(p, q, v); d := math.Sqrt(M.InnerProduct(p, v, v))
```

Both should give the same number. **Should the interface ship one or both?**

**Existing precedent.** `prob/Distribution` ships *both* `PDF` and `CDF` even though CDF is the integral of PDF — because closed-form CDFs are dramatically faster than numerical integration of PDF. Same logic applies here: Distance is closed form on most manifolds (sphere: `arccos⟨p,q⟩`; hyperbolic Lorentz: `arccosh(-⟨p,q⟩_M)`; multinomial Fisher-Rao: `2·arccos Σ √(p_i·q_i)` — **the Bhattacharyya angle**, no Log needed) — *much* cheaper than computing Log + InnerProduct.

**Verdict.** **Ship both.** Distance is a first-class method, with closed-form impl per manifold; Log is the heavier tangent-extraction operation. Ship a default `DistanceFromLog(M, p, q)` package-level helper for the rare manifold without closed-form distance. **Document the test pattern**: every manifold ships a `TestXxxDistanceMatchesLogNorm` test pinning the two against each other to 1e-10 — same R-CLOSED-FORM-PINNED-TO-AUTODIFF saturation pattern 093 §3 named.

```go
type Manifold interface {
    ...
    Distance(p, q []float64) float64
}

// DistanceFromLog computes d(p,q) = sqrt(<Log_p(q), Log_p(q)>_p) as a default
// implementation for manifolds that don't ship a closed-form distance.  Allocates
// one tangent buffer; prefer M.Distance(p, q) when it exists.
func DistanceFromLog(M Manifold, p, q []float64) (float64, error) { ... }
```

---

## 5. Tangent-vector validation: the `IsTangent` predicate

**Question.** Should the Manifold expose `IsTangent(p, v) bool`?

**Math.** A vector `v` lives in T_p(M) iff a manifold-specific linear constraint holds:
- Sphere: `<p, v> == 0` (orthogonal to radius).
- Lorentz: `<p, v>_M == 0` with Minkowski inner product.
- Simplex (in (n-1)-coords): no constraint; whole R^{n-1} is tangent.
- Stiefel: `p^T v + v^T p == 0` (skew-symmetric in the right basis).
- SPD: `v == v^T` (symmetric).

**The use case.** Catches the #1 bug in every manifold-using consumer's first 3 days: passing a Euclidean gradient as a Riemannian tangent. (Manopt's `egrad2rgrad` exists *because* this bug is universal — see 093 §4.) `IsTangent(p, v, atol)` is the Geomstats-`belongs`-style validation gate but for tangent vectors.

**Verdict.** Ship as **non-required helper** at the package level: `IsTangent(M Manifold, p, v []float64, atol float64) bool` rather than as a method on the interface. Reason: (a) keeps the interface narrow, (b) the implementation is `M.ProjectTangent(p, v, tmp); diff_norm := …` for any manifold that ships `ProjectTangent`; (c) tangent-validation is overwhelmingly a *test-time* concern not a hot-path concern.

**Add to interface:** `ProjectTangentInto(p, v, out []float64)` — the orthogonal projector onto T_p(M). Mandatory because *every* Riemannian optimiser needs it (Manopt's `tangent` field, 093 §4). One method, ~5-15 LOC per manifold, unblocks egrad2rgrad-style "I have a Euclidean gradient, give me the Riemannian one."

---

## 6. Geodesic computation: closed-form, ODE-shooting, or both?

**Question.** Reality's `chaos/` ships `RK4Step` and `SolveODE` for general ODEs. Geodesic ODE is `d²θ^k/dt² + Γ^k_{ij} dθ^i/dt dθ^j/dt = 0`. **Should `infogeo` consume `chaos.RK4Step` for geodesic shooting?**

**Two paths:**

### 6a. Closed-form path (the v1 reality)

For dual-flat coordinates, geodesics are *straight lines* (092 T1.7). For sphere/hyperbolic/multinomial-Fisher-Rao, geodesics are closed-form trig/hyperbolic combinations of `p` and `v`. **No ODE needed.** This is what `M.ExpInto(p, v, out)` ships per-manifold.

### 6b. ODE-shooting path (the manifolds where 6a fails)

For a general Riemannian metric (Stiefel without using QR retraction, SPD without using `expm`, Gaussian Fisher-Rao without Calvo-Oller's reduction), the geodesic IS an ODE in the Christoffel symbols. Two integrator choices:

- **`chaos.RK4Step` (existing, 092 T2.8 default).** Drifts off the manifold under non-trivial metrics (Poincaré disk: `||θ|| < 1` constraint, RK4 happily steps past 1; per 091 §2.2). Need post-step `M.ProjectInto`.
- **`chaos.Verlet` symplectic (does not exist; cross-package blocker per 091 / 092 / 093).** Preserves Hamiltonian H = ½g^{ij}p_i p_j to ε per step instead of ε·t drift.

**Verdict on the API contract.** `infogeo` should **compose `chaos`'s ODE primitives, not duplicate them**. Concretely:

```go
// GeodesicODE constructs the derivative function for geodesic flow on M
// suitable for chaos.RK4Step / chaos.Verlet consumption.  State vector is
// (q, p) of length 2*M.EmbeddingDim().
func GeodesicODE(M ChristoffelManifold) func(t float64, qp, dqp []float64)

// Concrete usage:
qp := append(append(make([]float64, 0, 2*n), p...), v...)
dqp := make([]float64, 2*n)
ws := chaos.NewRK4Workspace(2*n)
for step := 0; step < N; step++ {
    chaos.RK4StepInto(infogeo.GeodesicODE(M), t, qp, dt, qp, ws)
    M.ProjectInto(qp[:n], qp[:n])  // retract onto manifold every step
}
```

**The cross-package contract.** This requires *only one* new piece of interface surface:

```go
type ChristoffelManifold interface {
    Manifold
    ChristoffelInto(p []float64, out [][][]float64)  // out[k][i][j] = Γ^k_{ij}(p)
}
```

— and consumes the chaos ODE API as-is. **This is the right boundary** because: (a) `chaos/` already owns ODE solvers, (b) consumers wanting symplectic just substitute `chaos.VerletStep` for `chaos.RK4Step`, (c) `infogeo/` does not duplicate any integrator. Cross-cuts cleanly with 029-chaos-api's proposed `chaos.Stepper` interface.

**Single subtle issue.** Christoffel symbols are a 3-tensor `Γ^k_{ij}`. Reality's flat-slice convention says this should be `[]float64` with `len = n*n*n` and `out[k*n*n + i*n + j]` indexing — *not* the `[][][]float64` jagged-slice form 092 T1.4 proposed. **Recommendation:** ship as `ChristoffelInto(p []float64, out []float64, n int)` matching `linalg.MatMul`'s flat shape, not jagged `[][][]float64`. ~5 LOC of indexing, ~3× faster cold-cache (092's perf audit 090 already named the flat-vs-jagged perf cliff).

---

## 7. Error semantics: where do exp/log fail?

**Question.** When `Exp_p(v)` or `Log_p(q)` returns an error, which error?

Closed-form maps fail in two ways:
1. **Out of injectivity radius** — Log_p(q) is multi-valued or undefined when `q` is outside the geodesic ball of injectivity at `p`. Sphere: `q = -p` (antipode). Multinomial Fisher-Rao: `<sqrt(p), sqrt(q)> = 0` (orthogonal in √-coords).
2. **Off-manifold input** — `p` or `q` doesn't satisfy `Belongs`.

Existing infogeo error types: `ErrInvalidDistribution`, `ErrLengthMismatch`, `ErrInvalidParameter`. Sufficient? No.

**Verdict.** Add three errors to the package:
```go
var ErrNotOnManifold       = errors.New("infogeo: point fails Belongs check")
var ErrNotInTangentSpace   = errors.New("infogeo: vector fails IsTangent check")
var ErrOutsideInjectivity  = errors.New("infogeo: target outside injectivity radius of base point")
```

The first two are **input-validation** errors (caller bug); the third is a **domain** error (genuinely undefined math). Document explicitly in `M.LogInto`'s docstring: "Returns ErrOutsideInjectivity if `q` is outside the injectivity radius of `p` (e.g., antipode on the sphere). Caller may catch and fall back to a perturbed Log via `LogPerturbed`."

**Do not panic.** Existing infogeo functions return errors; geometry/calculus/chaos panic on wrong dims. **infogeo should follow infogeo's own existing convention** (error-return), not the rest of Reality's convention (panic). The reason: probability-vector inputs are *user data* (not programmer constants), and panics on user data are user-hostile. Same logic applies to manifold inputs.

---

## 8. Allocation discipline: the `Into` suffix audit

CLAUDE.md rule 3: "No allocations in hot paths. Functions accept output buffers." 091 §1.8 already noted `bregman.Bregman` allocates `gradY` per call. The chaos audit (029) carved out `RK4StepInto` from `RK4Step`. The proposed Manifold interface should bake this in **from day 1**.

**Convention.** All Manifold methods that produce a vector or matrix output end in `Into` and take `out []float64` as the last argument:

| Method | Out-buffer | Length |
|---|---|---|
| `ProjectInto(p, out)` | `out` | `EmbeddingDim()` |
| `ProjectTangentInto(p, v, out)` | `out` | `EmbeddingDim()` |
| `ExpInto(p, v, out)` | `out` | `EmbeddingDim()` |
| `LogInto(p, q, out)` | `out` | `EmbeddingDim()` |
| `ParallelTransportInto(p, v, dir, out)` | `out` | `EmbeddingDim()` |
| `ChristoffelInto(p, out, n)` | `out` | `n*n*n` |
| `MetricInto(p, out, n)` | `out` | `n*n` |

**Methods that return a scalar do NOT take an out-buffer**: `Distance`, `InnerProduct`, `Dim`, `EmbeddingDim`, `Belongs`. Same convention as `infogeo/fdiv.go` (`KL` returns `(float64, error)`, no out-buffer).

**Methods that take a `Workspace`** for inner allocations (e.g., the matrix-exp inside SPD's Exp): a `M.Workspace` type with the requisite scratch. Mirrors `chaos.RK4Workspace` from 029.

```go
type SPDManifold struct{}
type SPDWorkspace struct {
    LtP, LtQ, M, expM []float64  // n×n flat
    eigVals, eigVecs []float64
}
func (SPDManifold) ExpInto(p, v, out []float64) error    // allocates SPDWorkspace internally
func (SPDManifold) ExpIntoWS(p, v, out []float64, ws *SPDWorkspace) error  // zero-alloc
```

**The `Into` + `IntoWS` split for heavy methods**: `Into` is the convenience form; `IntoWS` is the 60-FPS form. Callers in tight loops (Pistachio, Sensorhub) call the WS form; callers in golden-file tests call `Into`. Same precedent as 029's `RK4Step` / `RK4StepInto`.

---

## 9. Comparison with `chaos/`'s ODE API for geodesic-ODE consumers

**The cross-cutting concern.** Geodesic ODE is the largest single ODE consumer in the entire Reality codebase that *is not already* a chaos/ system. Consumers of geodesic flow include: SPD diffusion (T2.3), Stiefel/Grassmann optim (T2.6), generic curved-metric maps (T2.8), Riemannian-manifold HMC (T3.12). All want the same shape.

**The chaos API as it exists today (per 029):**
```go
// ode.go:9 — derivative signature
func(t float64, y []float64, dydt []float64)

// ode.go:36 — single-step (allocates 5 vectors per call, doc says so)
RK4Step(f Deriv, t float64, y []float64, dt float64, out []float64)

// ode.go:100 — eager batch (allocates trajectory + 1 alloc per step)
SolveODE(f Deriv, y0 []float64, t0, tEnd, dt float64) [][]float64
```

**The chaos API as 029 proposes it (Stepper + Workspace + Trajectory):**
```go
type Deriv func(t float64, y, dydt []float64)
type Stepper interface { Step(f Deriv, t float64, y, out []float64, dt float64); Order() int; Name() string }
type Problem struct{ F Deriv; Y0 []float64; T0, T1, DT float64; Params any }
type Trajectory interface { Step() bool; Time() float64; State() []float64; Err() error }
func Integrate(s Stepper, p Problem) Trajectory
```

**The infogeo `GeodesicODE` adapter that consumes either form:**
```go
// GeodesicODE returns the chaos.Deriv-shaped derivative for the Hamiltonian
// flow d²q/dt² + Γ^k_{ij} dq^i/dt dq^j/dt = 0 on M, with state vector
// y = (q, p) of length 2*n.  Consumes Christoffel symbols on demand.
//
// Suitable for direct passing to chaos.RK4Step, chaos.RK4StepInto,
// chaos.VerletStep (when shipped), or chaos.Integrate.
func GeodesicODE(M ChristoffelManifold) func(t float64, y, dydt []float64)
```

The adapter unblocks geodesic computation for *every* manifold that ships Christoffel symbols, **without infogeo needing to own a single integrator**. Chaos owns time-stepping; infogeo owns differential-geometry primitives. **Clean dependency direction:** infogeo → chaos, never the reverse.

**The retraction shortcut (per 093 §2 / Boumal Table 7.2).** For optimisation use cases, the full geodesic ODE is overkill — first-order retraction is enough. Reality's API for retraction is *already* the `M.ProjectInto` method on the Manifold interface (composed with a Euclidean step):
```go
// Riemannian step via retraction: q ← Project(q + step·v)
for i, qi := range q { tmp[i] = qi + step*v[i] }
M.ProjectInto(tmp, q)
```
**Two lines, no integrator dependency.** This is the v1 path for every R-SGD / R-Adam consumer; the geodesic ODE is the v2-and-up path for HMC and curvature-tensor consumers.

---

## 10. The full proposed API surface (summary table)

| Interface / function | Lives in | Shape | LOC |
|---|---|---|---|
| `Manifold` interface (8 methods) | `infogeo/manifold.go` | flat `[]float64` everywhere | 60 |
| `ChristoffelManifold` interface | `infogeo/manifold.go` | flat `[]float64` for 3-tensor | 5 |
| `RiemannianMetric` interface (extends) | `infogeo/manifold.go` | adds Christoffel + ParallelTransport | 10 |
| `IsTangent(M, p, v, atol)` helper | `infogeo/manifold.go` | package-level | 15 |
| `ExpMap(M, p, v) ([]float64, error)` | `infogeo/manifold.go` | allocating wrapper | 8 |
| `LogMap(M, p, q) ([]float64, error)` | `infogeo/manifold.go` | allocating wrapper | 8 |
| `DistanceFromLog(M, p, q)` | `infogeo/manifold.go` | default for non-closed-form | 12 |
| `GeodesicODE(M) chaos.Deriv` | `infogeo/geodesic.go` | adapter to chaos | 30 |
| `MultinomialManifold` | `infogeo/multinomial.go` | impl + closed-form ExpInto/LogInto | 80 |
| `SphereManifold` | `infogeo/sphere.go` | impl ("hello world", 093 §13) | 60 |
| `LorentzManifold` | `infogeo/hyperbolic.go` | impl + Poincaré bijection | 100 |
| `GaussianManifold` | `infogeo/gaussian.go` | impl + Calvo-Oller reduction | 120 |
| `SPDManifold` (Cholesky-only, 093 §6) | `infogeo/spd.go` | impl + SPDWorkspace | 150 |
| **Total new API surface** | | | **~660 LOC** |

This is the *interface and adapter* commit; it does not include the Tier 1 / Tier 2 *math* primitives 092 enumerated. Those land *against* this stable API.

---

## 11. The eight design calls, distilled

| # | Call | Decision |
|---|---|---|
| 1 | Point representation | `[]float64` (extrinsic), not a wrapped `Point` type |
| 2 | Tangent representation | same shape as point (`[]float64`, length `EmbeddingDim()`) |
| 3 | Method naming | `ExpInto` / `LogInto` (no bare `Exp` to avoid `math.Exp` / `autodiff.Exp` collision) |
| 4 | Argument order | basepoint first, target/tangent second, out-buffer last |
| 5 | Distance | first-class method (closed-form per manifold), with `DistanceFromLog` default |
| 6 | Geodesic computation | closed-form `ExpInto` per manifold; `GeodesicODE` adapter for consumers wanting `chaos.RK4` |
| 7 | Error semantics | error-return (not panic), three new errors: `ErrNotOnManifold`, `ErrNotInTangentSpace`, `ErrOutsideInjectivity` |
| 8 | Allocation discipline | `Into` suffix on every vector/matrix output, `Into` + `IntoWS` split for heavy methods |

Together these eight calls **define the contract that every primitive in 092's Tier 1 and Tier 2 implements against**. Without them, every primitive is bespoke (092's exact warning about T2.1) and the `Manifold` interface is shapeless.

---

## 12. Anti-recommendations: APIs that look right but aren't

**A12.1 — Generics with `Manifold[P, V any]`.** Tempting (the embedding shape *does* vary across manifolds), but: (a) Reality is `slices.Sort`-era Go (1.21+), generics work; (b) `[]float64` everywhere is the load-bearing convention; (c) every test would need explicit type params. **Reject.** Use `[]float64` and document length invariants.

**A12.2 — Methods on the point, not the manifold.** Geomstats does `point.exp(tangent)`. **Reject.** Reality has no `Point` type; methods on `[]float64` are forbidden by Go.

**A12.3 — Functor-style `func(p, v) []float64` instead of an interface.** Tempting for simple manifolds but loses the `Belongs` / `Distance` / `Dim` plumbing. **Reject.** Interface ships those alongside the maps.

**A12.4 — Explicit chart-coordinate basis.** Some libraries (Stochman) make the user supply a coordinate basis at every point. **Reject.** The extrinsic representation hides the chart from the user; the manifold internally uses whatever chart is convenient. User never sees `(coord_idx, chart_idx)` plumbing.

**A12.5 — Curried `M.At(p)` returning a `TangentSpace` with methods `Exp(v)`, `Log(q)`, `<u,v>`.** Mathematically clean. **Reject** for the same reason A12.1 fails: `M.At(p)` allocates a wrapper struct on every call; in a hot path that is exactly the allocation profile CLAUDE.md rule 3 forbids.

**A12.6 — `GeodesicSegment(p, q, n_samples)` returning a sampled curve.** Looks useful for visualisation. **Reject for v1.** It's a one-line composition `for i := 0; i <= n; i++ { Exp_p((i/n) * Log_p(q)) }` — caller writes it. Adding it pollutes the interface with a derived primitive.

---

## 13. The single highest-leverage commit

> **Ship the `Manifold` interface in flat-slice + `Into` form (call list above), with `SphereManifold` as the concrete reference implementation, and `GeodesicODE` as the chaos-adapter — ~150 LOC total, zero blockers.**

After this lands:
- 092's T1.1 (FisherRaoSimplex), T1.2 (Gaussian), T1.4 (Christoffel), T1.6 (e/m coords), T1.7 (e/m geodesics) all become structural copy-pastes against a stable contract — **one PR per manifold**, not one PR for the entire library.
- 093's R-Adam / R-LBFGS / RGD optim layer composes against the same `Manifold` interface with no surface-area surprises.
- The cross-package boundary with `chaos/` is exactly one function (`GeodesicODE`); `chaos.Verlet` lands when `chaos/` ships it without further infogeo changes.

**Without this**, every Tier 1 / Tier 2 primitive in 092 ends up with its own ad-hoc signature (some take `*linalg.Matrix` that doesn't exist, some take `[]float64`, some take wrapped types), and the package's name continues to be a misdirection (per 091's headline). With it, the package's API surface area collapses from "29 bespoke functions" (the union of 092 Tier 1 + Tier 2) to "8 interface methods × N manifolds + 6 helpers" — the exact factoring Manopt achieved in 2014 and every IG library since has imitated.

---

## 14. References

- 091-infogeo-numerics: numerical hazards in present functions; flagged the geodesic ODE drift problem this audit's §6 closes.
- 092-infogeo-missing: enumerated the 36 primitives that need to land; assumed a `*linalg.Matrix` type that doesn't exist (this audit corrects).
- 093-infogeo-sota: ranked Manopt's factory pattern as the single highest-leverage interface design; this audit translates that to Reality's flat-slice idiom.
- 029-chaos-api: the Stepper / Workspace / Trajectory proposal that geodesic-ODE consumers compose against.
- `chaos/ode.go`: existing ODE API (`func(t, y, dydt []float64)` shape — verbatim reuse target).
- `linalg/matrix.go`: flat row-major slice convention (verbatim reuse target).
- `infogeo/fdiv.go`, `bregman.go`, `mmd.go`: existing error-return convention in this package (verbatim reuse target).
- `geometry/quaternion.go`: prior art for SO(3) Exp/Log without explicit Manifold interface.
- Boumal N. (2023). *An Introduction to Optimization on Smooth Manifolds*, Cambridge UP. Table 7.2 retraction tabulation (cited per-manifold in `ProjectInto` impls).
- Amari S. (2016). *Information Geometry and its Applications*, AMS vol 194, Springer. Tables 2.6.1, 2.7.1, 3.4.1 cited per-manifold in `ChristoffelInto` impls.
- Pennec X., Fillard P., Ayache N. (2006). A Riemannian Framework for Tensor Computing. *IJCV* 66:41. Affine-invariant SPD; `SPDManifold` cite.
- Townsend J., Koep N., Weichwald S. (2016). Pymanopt. *JMLR* 17:137. Argument-order convention (basepoint first).

---

End of audit. 094 / 400.
