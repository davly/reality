# 104 — optim: API ergonomics (callbacks, constraints, problem types)

Scope: API surface only. Per-line numerics bugs are owned by 101 (`InteriorPoint`-isn't-IPM, Bland tie-break, Wolfe-vs-Armijo doc lie, NewtonRaphson f'=0 silent return). Algorithm-by-algorithm gap enumeration is owned by 102 (Mehrotra MPC, CG, LM, Brent, Adam-family, Nelder-Mead, DE, PSO, OSQP, ...). Cross-library SOTA-axis matrix and the high-level "scipy `Result` retrofit + Optimization.jl `Problem`/`Solver`/AD-backend three-axis decoupling + NLopt `LD/LN/GD/GN` taxonomy + CMA-ES + TPE" plan is owned by 103. **This report owns the contract / type-shape / signature / ergonomics of the existing functions** — what a caller has to type, what they get back, what they cannot say.

Files surveyed: `optim/gradient.go` (250L), `optim/gradient_validated.go` (244L), `optim/rootfind.go` (119L), `optim/genetic.go` (179L), `optim/metaheuristic.go` (94L), `optim/linear.go` (317L), `optim/interpolate.go` (157L), `optim/proximal/{fbs.go,admm.go,operators.go}`, `optim/transport/sinkhorn.go`. Sibling baselines: `chaos/ode.go`, `control/{pid.go,transfer.go}`.

---

## 1. Inventory of public signatures (what a caller types today)

```go
// Root finders / 1-D
BisectionMethod(f func(float64) float64, a, b, tol float64) float64
NewtonRaphson(f, fPrime func(float64) float64, x0, tol float64, maxIter int) float64
GoldenSectionSearch(f func(float64) float64, a, b, tol float64) float64
LinearInterpolateRoot(x0, y0, x1, y1 float64) float64

// n-D unconstrained gradient
GradientDescent(f, grad, x0, lr, maxIter, tol)        []float64
LBFGS         (f, grad, x0, m, maxIter, tol)          []float64
GradientDescentValidated(...validate)                 (ConvergenceResult, error)
LBFGSValidated          (...validate)                 (ConvergenceResult, error)

// Stochastic / global
SimulatedAnnealing(f, x0, neighbor, temp0, cooling, maxIter, *rand.Rand) ([]float64, float64)
GeneticAlgorithm   (fitness, dim, popSize, gens, mutRate, rng) ([]float64, float64)

// LP
SimplexMethod (c, A, b)                              ([]float64, float64, error)
InteriorPoint (c, A, b)                              ([]float64, float64, error)

// Interpolate
LinearInterpolate(x0, y0, x1, y1, x)                 float64
CubicSplineNatural(xs, ys)                           func(float64) float64

// proximal/
proximal.Fbs (grad GradOp, prox ProxOp, x, work []float64, cfg FbsConfig)  (FbsResult, error)
proximal.Admm(proxF, proxG ProxOp, x, z, u []float64, cfg AdmmConfig)      (AdmmResult, error)
```

Eleven distinct calling conventions across the package. The cost of this survey is that adding a new optimizer is currently an open-ended design exercise instead of "implement `Solve(p Problem, cfg Config) Result`".

---

## 2. Problem-type axis: minimize / minimize-with-constraints / root-find

### 2.1 Today's split

| Problem class | Functions | Constraint expression |
|---|---|---|
| 1-D root-find on bracket | `BisectionMethod`, `LinearInterpolateRoot` | implicit in `[a,b]` |
| 1-D root-find from point | `NewtonRaphson` | none (unconstrained) |
| 1-D minimize on bracket | `GoldenSectionSearch` | implicit in `[a,b]` |
| n-D minimize, smooth, unconstrained | `GradientDescent`, `LBFGS`, `*Validated` | none (caller wraps via penalty) |
| n-D minimize, smooth, **only via custom validator predicate** | `GradientDescentValidated`, `LBFGSValidated` | `validate([]float64) bool` (arbitrary; not used to clip step) |
| n-D minimize, derivative-free, unconstrained | `SimulatedAnnealing`, `GeneticAlgorithm` | implicit `[-5,5]^d` for GA, none for SA |
| n-D minimize, composite f+g | `proximal.Fbs`, `proximal.Admm` | g encodes constraint via prox of indicator |
| LP (linear, polyhedral) | `SimplexMethod`, `InteriorPoint` | only `Ax≤b, x≥0` |

There is no n-D root-find. There is no "minimize-with-bounds" — the GA hard-codes `[-5,5]^d` and BLX-α can leak past them; the GD/LBFGS pair has nothing. There is no equality-constraint primitive. There is no inequality-constraint primitive. There is no n-D Brent-style minimize-on-box. The only constrained smooth solver is the LP pair (and the LP pair is form-rigid: only `Ax≤b, x≥0`, no equality rows, no `x ∈ [lo,hi]` box, no free variables).

### 2.2 Recommendation: a unified `Problem` (NOT a god-struct)

Crucial: do **not** unify all eight problem classes into one struct. The `chaos/ode.go` style — pass closures as positional args — is the right call for 1-D root-find and stays the right call there. But for n-D, the field-count has crossed the threshold (LBFGS already takes 6 positional args; `*Validated` takes 7; SA takes 7) where named-fields beat positionals. Proposed three problem types:

```go
// optim.Problem — n-D unconstrained or bound-constrained smooth minimize
type Problem struct {
    Dim   int
    F     func(x []float64) float64                  // required if no Grad
    Grad  func(x, gOut []float64)                    // optional; numerical FD if nil
    FAndGrad func(x, gOut []float64) float64         // optional; preferred if both needed
    X0    []float64                                  // required
    Lower, Upper []float64                           // optional bounds; nil = ±Inf
    // No equality/inequality at this layer — those go in ConstrainedProblem.
}

// optim.ConstrainedProblem — adds g(x)=0 / h(x)≤0
type ConstrainedProblem struct {
    Problem
    Eq   []func(x []float64) float64    // g_i(x)=0, eqJac optional
    Ineq []func(x []float64) float64    // h_i(x)≤0
    EqJac, IneqJac func(x []float64, J [][]float64) // optional
}

// optim.RootProblem — find x s.t. F(x)=0 in R^n (n≥1)
type RootProblem struct {
    Dim int
    F func(x, fOut []float64)
    J func(x []float64, J [][]float64) // optional
    X0 []float64
    Lower, Upper []float64 // bracket for n=1; box for n>1 (Powell-hybrid clip)
}
```

**Sibling parity.** This matches `chaos`-shape (closure for system function) and `control.TransferFunction`-shape (struct for ratio). Reality already has two precedents for the struct-of-fields pattern (`TransferFunction`, `PIDController`) and one for closures-as-positionals (`SolveODE`). Pick the precedent based on field count: when ≥4 named non-callback parameters are needed, struct.

Why not one unified `Problem` covering all three? Because `RootProblem.F` returns `[]float64` (residual vector) while `Problem.F` returns `float64` (scalar objective) — collapsing them either pessimizes the scalar case (always allocate length-1 slice) or makes the field meaning load-bearing on a flag. Three structs, sharing `Dim`, `X0`, `Lower`, `Upper` via embedding, is cleaner.

### 2.3 Where `optim.Problem` should NOT go

- Bisection / Newton / Golden / `LinearInterpolate*`: positional args are fine. Five-positional-arg `BisectionMethod(f, a, b, tol)` is more ergonomic than `BisectionMethod(RootProblem{...})`. **Cutover threshold: 4+ non-callback parameters → struct.**
- 1-D `GoldenSectionSearch`: keep as-is, but add `BrentMin1D` (102's call-out) using same positional shape.

---

## 3. Result type — the single highest-leverage commit in this report

### 3.1 What callers get today (six different shapes for what is morally one concept)

```
GradientDescent       → []float64                       (just x_final)
LBFGS                 → []float64
*Validated            → ConvergenceResult{X,Iters,Converged,Reason}, error
SimulatedAnnealing    → ([]float64, float64)            (x, f(x))
GeneticAlgorithm      → ([]float64, float64)
SimplexMethod         → ([]float64, float64, error)
InteriorPoint         → ([]float64, float64, error)
proximal.Fbs          → (FbsResult{Iter,Converged,FinalDelta}, error)  — does NOT carry x
proximal.Admm         → (AdmmResult{Iter,Converged,PrimalResid,DualResid}, error)  — does NOT carry x
```

The `proximal` results are the inverse of the smooth ones: x is mutated in place but the returned struct carries diagnostics; the smooth-n-D results are the opposite. The `*Validated` pair is the only one with a `Reason` string. None of the legacy seven carry `Nfev` (number of objective evaluations) — every solver computes this and throws it away.

### 3.2 Proposed unified `Result` (matches scipy `OptimizeResult`, owned by 103)

```go
type Result struct {
    X        []float64   // final iterate (always populated)
    F        float64     // f(X)
    Grad     []float64   // ∇f(X) if computed (else nil)
    Iter     int         // outer iterations
    Nfev     int         // objective evaluations
    Ngev     int         // gradient evaluations
    GradNorm float64     // ‖∇f(X)‖_2 if applicable
    Status   Status      // typed enum, NOT a string
    Message  string      // human-readable diagnostic
}

type Status uint8
const (
    StatusConverged Status = iota
    StatusMaxIter
    StatusLineSearchFailure
    StatusInvalidIterate     // R123 trap
    StatusNumericalFailure   // NaN/Inf in objective or gradient
    StatusUnbounded          // LP-specific
    StatusInfeasible         // LP/constrained
    StatusUserInterrupt      // returned from Callback
)
```

**Critical: `Status` is a typed enum, not the string field that `ConvergenceResult.Reason` uses today.** The current `Reason: "tolerance hit on invalid iterate (R123 trap caught)"` is a free-form string that callers have to substring-match — a downstream R123 audit cannot machine-check the trap branch was hit without `strings.Contains`. Promote to `StatusInvalidIterate`.

The retrofit is mechanical: every existing solver already computes `Iter` (for-loop counter), `Nfev`/`Ngev` are one increment each on the closure call site, `GradNorm` is computed and discarded by GD/LBFGS. ~25-40 LOC per solver, zero behavioural change.

### 3.3 Backwards-compatibility plan (because reality has consumers)

1. Add `Result`-returning sibling: `GradientDescentR(p Problem, cfg GDConfig) (Result, error)`.
2. Old `GradientDescent` becomes a thin wrapper around it returning only `Result.X` — preserves consumer ABI.
3. Mark old signatures `// Deprecated: use GradientDescentR.` (gopls picks this up, no breakage).
4. Same template for LBFGS, SA, GA, Simplex, IP. Six functions total. ~150 LOC of wrappers.

---

## 4. Callback hooks — completely absent

### 4.1 The void

Searched the package: zero per-iteration hooks. Callers cannot:
- log iterates / objective per step (have to instrument the closure they pass)
- trigger early termination on user signal (Ctrl-C, parent context cancel)
- monitor convergence with a custom metric (e.g. validation-set loss in ML)
- visualize the optimization trajectory (Pulse, Pistachio use cases)

The closest existing affordance is the `validate` predicate in `*Validated`, but its semantics is "is this iterate valid?", not "do I want to keep going" or "log this please". It runs only on the convergence-test branch.

### 4.2 Proposed contract

```go
// IterInfo is what a callback sees each outer iteration. Read-only; mutating
// X in the callback corrupts the optimizer.
type IterInfo struct {
    Iter     int
    X        []float64    // current iterate (DO NOT mutate)
    F        float64
    Grad     []float64    // may be nil if solver hasn't computed it
    GradNorm float64      // 0 if Grad is nil
    StepNorm float64      // ‖x_k - x_{k-1}‖_2 (0 on iter 0)
}

// Callback is invoked at the end of each outer iteration. Returning a non-nil
// error halts the optimizer with Status=StatusUserInterrupt and Message=err.
// Returning nil continues. Reserved sentinel: io.EOF means "stop here, treat
// as converged" (Status=StatusConverged, Message="user-requested stop").
type Callback func(IterInfo) error
```

Carry `Callback` on the per-solver `Config`, never on `Problem` (problem is *what*, callback is *how I want to watch it*).

### 4.3 Three high-leverage uses unlocked by this

1. **`context.Context` cancellation** — wrap as `func(IterInfo) error { return ctx.Err() }`. Reality has no ctx-aware solver today. Pulse and Pistachio loops both want to abort optimization on shutdown. Currently impossible.
2. **Convergence on auxiliary metric** — return `io.EOF` when validation loss stops improving, regardless of `tol`.
3. **Trajectory capture for unit tests** — append `IterInfo.X` to a slice; lets golden-file tests assert the *path*, not just the endpoint. (Catches convergence-rate regressions that endpoint-only tests miss.)

### 4.4 Cost

Single `if cb != nil { if err := cb(info); err != nil { ... } }` block at the end of each iteration loop. ~5 LOC per solver × 7 solvers = 35 LOC. Hot-path overhead: zero when nil (one branch).

---

## 5. Constraint specification

### 5.1 Today

Five effective constraint stories, none cohering:

1. **Polyhedral hard-coded form** (LP): `Ax ≤ b, x ≥ 0`. No equality rows, no `x ∈ [lo, hi]` box, no free variables, no x ≤ 0 vars. Caller must convert manually (split free variable into x⁺ - x⁻, or add slack-style equality reformulations).
2. **Implicit bracket** (Bisection / Golden): `[a, b]`.
3. **Implicit init box** (GA): hard-coded `[-5, 5]^dim`. Genuinely surprising: comment at `genetic.go:73` says "for problems with different domains, the fitness function can internally transform coordinates" — i.e. the caller has to remap their actual domain into [-5,5] manually. **This is a footgun.** A user who plugs in a fitness that wants `x ∈ [0, 1000]` will see the GA wander uselessly inside [-5,5].
4. **Validator predicate** (`*Validated`): `validate(x) bool` runs only on the *convergence* check. Iterate may still walk through invalid x; predicate doesn't clip the step. This is a tolerance-trap detector, not a constraint.
5. **Indicator-prox composition** (proximal): the caller supplies `prox_g` for the indicator function of the feasible set. Mathematically clean (and the *right* answer for FBS/ADMM). But moves all the bookkeeping to the caller.

### 5.2 Proposed: bounds at the `Problem` layer, eq/ineq at `ConstrainedProblem` layer

```go
type Problem struct {
    Dim          int
    F            func(x []float64) float64
    Grad         func(x, gOut []float64)
    FAndGrad     func(x, gOut []float64) float64
    X0           []float64
    Lower, Upper []float64  // len(Lower)==Dim or nil; -Inf/+Inf allowed per coord
}
```

Solvers respect `Lower`/`Upper` by:

- **Projected gradient** for GD: `x_{k+1} = clip(x_k - lr·g, Lower, Upper)`. ~5 LOC patch.
- **L-BFGS-B** for LBFGS: this is a real algorithm change (active-set + projected line search per Byrd-Lu-Nocedal 1995, ~300 LOC). 102 calls this out. Until then, the LBFGS solver should `return Result{Status: StatusInfeasible, Message: "LBFGS does not support bounds; use ProjectedGradient"}` rather than silently ignoring `Lower`/`Upper`.
- **GA**: replace hard-coded [-5,5] with `Lower`/`Upper`; default to ±5 only when both nil. **The hard-coded [-5,5] is the highest-leverage 3-LOC fix in this entire report.** Plus a compile-time-style assertion: if `Lower[j] >= Upper[j]`, return error.
- **SA**: clip `neighbor` output into box.

### 5.3 Equality / inequality

Reality has no SQP, no augmented-Lagrangian, no Frank-Wolfe (102 has these in Tier 2). API-wise, the *signatures* should be reserved now so consumers can write the constraint and have the solver return `StatusInfeasible: "no constrained NLP solver implemented yet"` rather than the consumer needing to know.

```go
type ConstrainedProblem struct {
    Problem
    Eq   []func(x []float64) float64        // g_i(x) = 0
    Ineq []func(x []float64) float64        // h_i(x) ≤ 0
    EqJac, IneqJac func(x []float64, J [][]float64)
}
```

Until there is a real SQP, only proximal (via `prox_g` of indicator) and the LP pair can use this — but the field shape is now stable. **Reserving the field shape now is the cheap commit; backfilling solvers is the expensive one.** Don't conflate them.

### 5.4 LP-specific gap: equality rows and free vars

`SimplexMethod`/`InteriorPoint` only accept `Ax ≤ b, x ≥ 0`. Almost every textbook LP has equality rows (`Ax = b`). Add:

```go
type LPProblem struct {
    C  []float64
    A  [][]float64
    B  []float64
    Sense []ConstraintSense    // Leq, Eq, Geq per row; default Leq if nil
    Lower, Upper []float64     // per-variable bounds; nil = [0, +Inf) (current behaviour)
}
```

This is pure form-conversion in the solver (split `Ax = b` into `Ax ≤ b ∧ Ax ≥ b`, free variable as x⁺ - x⁻); ~80 LOC patch but standardises the LP API to scipy.optimize.linprog parity (which is the pure-Python reference).

---

## 6. Tolerance: ftol / xtol / gtol — all collapsed into one `tol`

### 6.1 Today

Every n-D solver has exactly one `tol` parameter. What it means depends:

| Solver | What `tol` means |
|---|---|
| GradientDescent | `‖∇f(x)‖_2 < tol` (gtol) |
| LBFGS | `‖∇f(x)‖_2 < tol` (gtol) |
| BisectionMethod | `b - a < tol` (xtol) |
| GoldenSectionSearch | `b - a < tol` (xtol) |
| NewtonRaphson | `\|f(x)\| < tol` (ftol on residual) |
| proximal.Fbs | `‖x_{k+1} - x_k‖_∞ < AbsTol` (xtol, infinity-norm) |
| proximal.Admm | `‖x-z‖_∞ + ρ‖z-z'‖_∞ < AbsTol` (combined primal/dual residual) |
| SimulatedAnnealing | (no convergence test — runs `maxIter` always) |
| GeneticAlgorithm | (no convergence test — runs `gens` always) |
| SimplexMethod | hard-coded `1e-10` (line 97, line 111) — caller cannot tune |
| InteriorPoint | hard-coded `1e-8` (line 262) and `1e-12` (line 216) and `1e-15` (everywhere) |

Three problems:

1. **`tol` is one variable doing five jobs.** A user who has scaled their objective by 1e6 and wants `f(x) < 1e-3` cannot say so to GradientDescent (whose `tol` is on `‖g‖`).
2. **The two LP solvers ignore the user's tolerance entirely.** `SimplexMethod(c, A, b)` has no tolerance argument at all. `InteriorPoint(c, A, b)` likewise. Hard-coded `1e-10` reduced-cost cutoff is fine for textbook LPs and wrong for ill-conditioned ones (LP relaxations of MIPs routinely need 1e-7).
3. **Stochastic solvers don't check convergence at all.** SA and GA run their full budget. A user with a known target objective value (`fAbsTol`) cannot stop early.

### 6.2 Proposed

```go
type Tolerances struct {
    FAbs   float64  // |f_k - f_{k-1}| < FAbs
    FRel   float64  // |f_k - f_{k-1}| / (|f_k| + 1) < FRel
    XAbs   float64  // ‖x_k - x_{k-1}‖_inf < XAbs
    XRel   float64  // ‖x_k - x_{k-1}‖_inf / (‖x_k‖_inf + 1) < XRel
    GAbs   float64  // ‖∇f‖_inf < GAbs
    // Convergence iff (any non-zero tol) AND (all non-zero tols are met).
    // All-zero Tolerances means "run to MaxIter with no tol-based stop."
}
```

Convergence test: `(any field > 0) && (every field > 0 ⇒ test holds)`. Zero in a field disables that test. Default tolerance for each solver gets sensible defaults via `func DefaultTolerances() Tolerances` named per algorithm class (gradient-based vs derivative-free vs LP).

This matches scipy `xatol`, `xrtol`, `fatol`, `frtol`, `gatol` (103's call-out). The legacy single-`tol` path stays as a wrapper that fills `GAbs=tol` for gradient solvers, `XAbs=tol` for bracket solvers, etc.

### 6.3 Why a struct, not five separate args

Six tolerance parameters as positionals is unusable — caller has to remember argument order. Struct + zero-value-disables idiom is the Go-canonical way. Reality already uses this pattern in `proximal.FbsConfig` (Step, MaxIter, AbsTol, Accelerate as named fields). Generalise.

---

## 7. Random seeding — almost right, slight inconsistency

### 7.1 Today

- `SimulatedAnnealing` takes `*rand.Rand`. Concrete, type-checked, deterministic.
- `GeneticAlgorithm` takes `interface{ Float64() float64 }`. Maximally permissive, can be either `*rand.Rand` or a custom QMC source.

### 7.2 Verdict

GA's interface{Float64()} is the better design (works with QMC or `crypto/rand` adapters), but it's narrower than `*rand.Rand` — no `Intn`, no `NormFloat64`, no `Perm`. GA constructs Gaussians via Box-Muller from `Float64`, which is correct but slower and slightly less accurate (cosine variant has tail issue at u1≈0; `genetic.go:62` clamps u1≥1e-15 to dodge it, but that biases the tail). 

Recommendation: **accept the broader `*rand.Rand` everywhere for stochastic solvers**, since reality has no QMC consumer today. Stdlib `rand.Source` -> `rand.Rand` adapter is one line for callers with a custom source. Tighten typing across the package; expose `interface{...}` only when there is a concrete callsite that needs the looseness.

### 7.3 R-pattern bug bait

There is a latent reproducibility issue: if `cfg.RNG == nil`, what happens? Today, GA panics (nil-pointer on `rng.Float64()`). SA panics. The validator predicate path passes nil silently. **Defensive null-check + a deterministic fallback `rand.New(rand.NewSource(1))` with a `// WARNING: optimizer ran with default-seeded RNG` log is the correct posture.** Or just return `Result{Status: StatusNumericalFailure, Message: "RNG must not be nil for stochastic solver"}`.

---

## 8. Comparison with siblings

### 8.1 chaos/ (ODE Problem analog)

`chaos/ode.go` uses **closures-as-positionals** for `RK4Step`, `EulerStep`, `SolveODE`. No `ODEProblem` struct. The closure shape `func(t, y, dydt)` is canonical and only three call-site parameters (`f, y0, [t0,tEnd], dt`) — no struct needed.

But chaos has the same omissions optim does:
- No callback per step (consumers can't capture trajectory beyond the final return).
- No event detection (callable that returns when `g(t,y)=0`, e.g. Pulse trigger crossings) — *this is the n-D equivalent of bisection-on-the-fly*.
- No tolerance struct — `dt` is the only knob.

**Cross-cutting lesson: every solver-shaped package in reality (chaos, optim, control's filter integrators, signal's iterative filters) should share the `Callback` + `Result` + `Tolerances` design.** A unified `solver` micro-package (or interface declared in `testutil`) would make this enforceable.

### 8.2 control/ (system types analog)

`control.PIDController` is the right precedent for a **stateful** optimizer. It uses:
- struct with public gains (`Kp, Ki, Kd`)
- private state (`integralSum`, `prevError`)
- `Update(setpoint, measured, dt) float64` step method
- `Reset()` to clear state

A streaming optimizer (Adam, RMSprop, online SGD — all in 102's Tier 1) maps perfectly to this shape:

```go
type AdamOptimizer struct {
    LR, Beta1, Beta2, Eps float64  // public, tunable
    // private state: t, m, v
}
func NewAdam(lr float64, dim int) *AdamOptimizer
func (a *AdamOptimizer) Step(grad, x []float64)
func (a *AdamOptimizer) Reset()
```

This is parallel to `PIDController`. Reality has no streaming-optimizer abstraction today — `Adam` etc. would land most naturally in this shape, *not* the batch `Solve(Problem) → Result` shape. Pistachio's per-frame parameter updates want this shape; aicore's mini-batch training loops want this shape.

`TransferFunction` is the right precedent for a **declarative** problem object (poles/zeros/Bode all live as methods on the struct). The same pattern fits `optim.Problem`: `(p *Problem) NumDeriv(x, gOut)` for finite-difference fallback when `Grad` is nil; `(p *Problem) Project(x, xOut)` for box clipping.

---

## 9. Specific R-pattern flags

1. **`*Validated` predicate runs only at convergence check, not per step.** A user expecting "validate is true on every iterate I see" is wrong. Either rename to `validateAtConvergence` (clearer) or also clip the step (real behavioural change). Document either way. (gradient_validated.go:86)
2. **`InteriorPoint` accepts a problem with `b[i] < 0` and silently misbehaves** — `SimplexMethod` corrects (linear.go:58) but `InteriorPoint` does not (linear.go:172). Cross-solver inconsistency: same problem produces different (sometimes wrong) results.
3. **`GA` inits in `[-5, 5]^d` regardless of caller intent.** Already flagged §5.1.3 and §5.2 — this is the single most surprising default in the package.
4. **`NewtonRaphson` returns *current x* on `f'=0` with no error, no flag.** Caller cannot distinguish converged-from-good-start vs failed-due-to-stationary-point. Also flagged in 101; my §3.2 `Status` enum gives the API mechanism to fix it.
5. **`CubicSplineNatural` panics on bad input** (rootfind.go-style would have returned NaN; spline-style panics). Reality has no consistent error-vs-panic policy. Suggest: panic on programmer error (xs unsorted, length mismatch — caller should never), error on data error (no roots in bracket, infeasible LP, line-search failure — domain shape may force this).
6. **`SimulatedAnnealing.neighbor` callback** is the only solver with a "domain-shaped" closure (knows how to perturb x in a problem-specific way). This is the right design and should be lifted: GA could accept an optional `Mutate func(x, out)` to override its hard-coded BLX-α; LBFGS could accept `LineSearch func(x, d, fx float64) float64`. **Compose-by-callback is a missing extension axis across the whole package.**

---

## 10. Top-10 commits, ranked by leverage / LOC

| # | Patch | LOC | What it unblocks |
|---|---|---|---|
| 1 | `Result` struct + `Status` enum, sibling functions | ~250 | Caller introspection; Pulse/Pistachio diagnostics; R123 machine-checkable status |
| 2 | `Callback` hook on every n-D solver | ~35 | ctx cancellation; trajectory capture; aux-metric early stop |
| 3 | Replace GA's hard-coded `[-5,5]^d` with `Lower`/`Upper` | ~10 | GA actually usable; removes silent footgun |
| 4 | `Tolerances` struct (`FAbs/FRel/XAbs/XRel/GAbs`) | ~80 | scipy-parity convergence semantics |
| 5 | `Problem`/`ConstrainedProblem`/`RootProblem` struct types | ~120 | Adding a new solver becomes one PR; constraint shape stable before solvers exist |
| 6 | `LPProblem` with `Sense[]` and `Lower`/`Upper` | ~80 | Equality rows; free variables; scipy.linprog parity |
| 7 | Streaming-optimizer interface (`Step(grad, x)` + `Reset()`) | ~30 | Adam/RMSprop/SGD-Nesterov fit naturally; Pistachio/aicore use case |
| 8 | InteriorPoint b[i]<0 fix to match Simplex | ~5 | Cross-solver consistency |
| 9 | NewtonRaphson `f'=0` returns `Status=StatusNumericalFailure` | ~5 | Audit-trail for divergence |
| 10 | RNG nil-check on stochastic solvers | ~15 | Defensive; no nil-pointer panic |

Total: ~630 LOC. Item #1 alone closes the largest API-quality gap; items #2-#4 the next biggest; the rest are clean-up. None require new mathematics — every commit is interface design or one-liner consistency.

---

## 11. Cross-cutting recommendation

Reality has eleven solver-shaped functions and no solver shape. Three structurally identical concepts (`ConvergenceResult`, `FbsResult`, `AdmmResult` — and the implicit two-tuple from SA/GA) want to be one `Result`. Two structurally identical concepts (`FbsConfig.AbsTol`, `AdmmConfig.AbsTol`, the n-tier `tol` argument, the LP hard-coded `1e-10`) want to be one `Tolerances`. The package will continue to bifurcate every time a new solver lands until these two abstractions exist.

**The lowest-cost moment to introduce them is right now**, before 102's Tier-1 sprint ships ~14 new solvers — every new arrival picks up the right shape automatically, instead of growing the surface and making the eventual cleanup more expensive.

---

End. Report at `agents/104-optim-api.md`.
