# 029 | chaos-api

**Topic:** chaos: ODE callback ergonomics, event detection, dense output
**Date:** 2026-05-07
**Files audited:** `chaos/ode.go` (132 LOC), `chaos/systems.go` (173 LOC), `chaos/analysis.go` (159 LOC), `chaos/chaos_test.go` (888 LOC)
**Sibling refs:** 026 covered numerics (RK4 alloc, symplectic, Lyapunov spectrum). 027 covered missing systems. 028 covered SOTA interface (DynamicalSystem/StateSpaceSet/Trajectory). This report stays in **API-ergonomics** lane: signature shape, callbacks, event detection, dense output, problem-struct vs scattered args, integrator switching, ensemble, streaming-vs-batch, and parity with sibling packages (`signal`, `optim`).

---

## TL;DR

Reality's `chaos/` ODE surface is **three exported integration helpers and zero abstractions on top of them**. Every ergonomic axis the topic asks about — callbacks during integration, event detection, dense output between steps, problem-struct, integrator switching, ensemble integration, streaming vs eager-materialized trajectory — is **absent**. The signature shape itself is the right one (`func(t, y, dydt []float64)` — in-place, CLAUDE.md-compliant) but the *callers* of that signature (`RK4Step`, `EulerStep`, `SolveODE`) all violate the no-allocation rule by `make`-ing 5/1/N+2 slices per call. The `SolveODE` API materializes the entire trajectory into `[][]float64` (one `make` *per step* — see `ode.go:122`), forcing every consumer with bounded memory or a streaming-decision use case (`Pulse` trend modeling, `Oracle` online prediction, `Pistachio` 60 FPS particle sim) to copy-paste an inline RK4 loop. Comparing against sibling packages: `signal/fft.go` ships `FFT(real, imag []float64)` operating in-place on caller buffers; `optim/lbfgs.go` carries a `state` struct; `prob/` exposes online estimators. `chaos/` is the **outlier** — it ships only the eager-batch variant of each operation. The single highest-leverage commit is **A1 (workspace) + A2 (Stepper interface) + A4 (Trajectory iterator)** as one ~250-LOC refactor: it unblocks integrator switching, callbacks, events, dense output, and ensemble — every other topic axis collapses out of those three primitives.

---

## 1. The signature shape (correct) vs the callers (broken)

The exported derivative type is right:

```go
func(t float64, y []float64, dydt []float64)   // ode.go:9, used by all 5 systems
```

In-place, no return-allocation, CLAUDE.md rule 3 honored. `LorenzSystem`, `RosslerSystem`, `LotkaVolterra`, `SIRModel`, `VanDerPol` all close over params and write into `dydt` with zero allocation per call. **Good.**

But every consumer of this signature inside the package allocates:

| Function | Allocs/call | Source |
|---|---|---|
| `RK4Step` | **5** (`k1, k2, k3, k4, tmp`, all `make([]float64, n)`) | `ode.go:38-42` |
| `EulerStep` | **1** (`dydt := make([]float64, n)`) | `ode.go:82` |
| `SolveODE` | **steps+3** (every `row` is its own `make`) | `ode.go:107, 110, 117, 122` |

The doc-comment on `RK4Step` literally says "RK4Step allocates temporary k-vectors internally. For truly allocation-free usage in tight loops, callers should implement the method inline." That is the package telling its consumers the API is wrong. Pistachio at 60 FPS × 1000 particles × n=6 = 28 MB/s of GC pressure (see 026-N1).

**A1 (must-ship, 30 LOC).** Add an `RK4Workspace` struct holding `k1, k2, k3, k4, tmp []float64` and a constructor `NewRK4Workspace(n int)`. Add `RK4StepInto(f, t, y, dt, out, ws *RK4Workspace)` that does zero allocations. Keep `RK4Step` as the one-shot convenience wrapper that allocates a fresh workspace each call. **Mirrors `signal/fft.go`'s `FFT` / `FFTInto` split exactly.** See `signal/fft.go:49` (`FFT(real, imag []float64)`) — the precedent is in the repo.

---

## 2. Time/state separation: autonomous vs non-autonomous

All five named systems in `systems.go` are **autonomous** — they ignore the `t` argument:

```go
return func(t float64, y, dydt []float64) {  // systems.go:18 — t unused
    x, yy, z := y[0], y[1], y[2]
    dydt[0] = sigma * (yy - x)
    ...
}
```

Same for Rössler, Lotka-Volterra, SIR, Van der Pol. `t` is dead-weight in every case the package ships. Yet there's no autonomous-only fast path. Compare DynamicalSystems.jl which has separate `ContinuousDynamicalSystem` (autonomous) vs `ContinuousTimeDynamicalSystem` (non-autonomous with explicit `t`-dependence) so the integrator can skip `t`-related bookkeeping. Forced Van der Pol (`mu*(1-x²)*v - x + A*cos(ω*t)`) is the canonical non-autonomous example and the package can't even express it cleanly because there's no `Forcing` argument idiom — you'd need to bake `A, ω` into the closure, which works but is not parameter-discoverable.

**A2 (ship with A1, 80 LOC).** Define:

```go
type Stepper interface {
    Step(f Deriv, t float64, y, out []float64, dt float64)
    Order() int
    Name() string
}
type Deriv func(t float64, y, dydt []float64)
```

Then `RK4{ws *RK4Workspace}`, `Euler{ws *EulerWorkspace}`, `Leapfrog{ws ...}` (027/028 want symplectic; this is the integration point). Now `SolveODE` becomes `Solve(s Stepper, prob Problem)`, integrator switching is one line, and a non-autonomous flag belongs on `Problem` not on `Stepper`.

---

## 3. The missing `Problem` struct: arguments are scattered

Current `SolveODE` signature:

```go
SolveODE(f func(t, y, dydt []float64), y0 []float64, t0, tEnd, dt float64) [][]float64
```

Five positional float64+slice args in a row is a foot-gun. Real consumers want to **share the problem definition** between (a) one-shot integration, (b) Lyapunov spectrum computation, (c) Poincaré map, (d) bifurcation scan, (e) ensemble of perturbed ICs. Today every one of those re-passes `(f, y0, t0, tEnd, dt)` separately — and `BifurcationDiagram` (`analysis.go:76`) doesn't even use `SolveODE`, it has its own hand-rolled iterate loop because the signature didn't fit.

**A3 (must-ship, 25 LOC).**

```go
type Problem struct {
    F      Deriv         // derivative
    Y0     []float64     // initial state
    T0, T1 float64       // integration interval
    DT     float64       // step size (or hint, if adaptive)
    Params any           // optional, opaque to integrator (echoed in callbacks)
}
```

Then `Solve(s Stepper, p Problem) Trajectory`. Now Lyapunov, Poincaré, basin, bifurcation can all consume `Problem` — the ergonomic key that **unlocks every analysis tool 027/028 enumerate**. DynamicalSystems.jl, scipy `solve_ivp` (which takes `(fun, t_span, y0, ...)`), JiTCODE, OrdinaryDiffEq.jl all converge on a problem-struct. Reality is the outlier.

---

## 4. Callbacks during integration (absent)

There is **no hook** in any of `RK4Step`, `EulerStep`, `SolveODE` for: per-step logging, energy/Hamiltonian monitoring, NaN/Inf early-termination, progress reporting, snapshot-every-N-steps, or maximum-walltime guards. Consumers either (i) re-implement the integration loop inline so they can insert their own check, or (ii) post-process the entire materialized trajectory after the fact (which doesn't help if the integration blew up at step 47 of 10⁶).

Look at `TestLotkaVolterra_HamiltonianConserved` (`chaos_test.go:316`) — it has to run `SolveODE` to completion *then* iterate the full trajectory checking H. If `SolveODE` had a callback the test could short-circuit on first violation; more importantly, real consumers (Pulse trend modeling) need to detect divergence and re-step with smaller `dt` *during* integration.

**A4 (ship with A3, 50 LOC).** Three orthogonal mechanisms:

1. **Iterator pattern** (the streaming primitive, replaces eager `[][]float64`):
    ```go
    type Trajectory interface {
        Step() bool                    // advance one dt; false on done/error
        Time() float64
        State() []float64              // borrowed view, do NOT retain
        Err() error
    }
    func Integrate(s Stepper, p Problem) Trajectory
    ```
    Consumer loops `for tr.Step() { ... }` and decides per-step what to record. **This kills `SolveODE`'s eager `[][]float64` alloc-per-step entirely** (`ode.go:122` makes a fresh `[]float64` per row — for 10⁶ steps that's 10⁶ separate heap allocations).

2. **Callback-list on `Problem`** for non-streaming consumers who still want hooks:
    ```go
    type Hook func(step int, t float64, y []float64) error  // return err to abort
    Problem{ ..., Hooks []Hook }
    ```
    Pre-built hooks: `EveryN(n int, h Hook)`, `EnergyGuard(H func(y) float64, tol float64)`, `NaNGuard()`, `WallClockBudget(d time.Duration)`.

3. **`Solve(s, p) (Trajectory, error)`** convenience wrapper that materializes by draining the iterator — preserves backward-compatibility for consumers who *want* the eager `[][]float64`.

This is the **single highest-leverage refactor in the package** because it composes with everything else: events (§5) become a hook, dense output (§6) becomes an iterator method, ensemble (§8) becomes N parallel iterators.

---

## 5. Event detection / Poincaré (absent)

Zero-crossing detection is the prerequisite for: Poincaré sections (the canonical chaos-visualization tool, named in 028's axis 11), period-doubling-onset detection, attractor-section RQA, basin-of-attraction boundary tracking, and any "stop when X happens" use case in physics simulation (collision, threshold-crossing, escape from region). The standard implementation is:

1. Step from `t_n` to `t_{n+1}`.
2. Evaluate event function `g(t, y)` at both endpoints.
3. If `sign(g_n) ≠ sign(g_{n+1})`: a root lies in the interval.
4. Refine via bisection or Brent on the interpolated state (need dense output, §6).
5. Fire callback with the refined `(t*, y*)` and either continue, restart from `y*`, or terminate.

scipy `solve_ivp` does this via `events=[fn1, fn2, ...]`. JuliaDynamics ships `PoincareMap(ds, plane)` as a first-class object. Reality has nothing — there is no way to compute a Poincaré section for `LorenzSystem` without re-implementing the full integrator loop in your own code.

**A5 (ship after A4, 70 LOC).**

```go
type Event struct {
    G          func(t float64, y []float64) float64  // event when sign-change
    Direction  int                                    // -1, 0, +1 (rising/falling/either)
    Terminal   bool                                   // stop integration on hit
    Tol        float64                                // refinement tolerance
}
Problem{ ..., Events []Event }
type Hit struct{ Idx int; T float64; Y []float64 }
// Trajectory.Hits() []Hit  after termination
```

Poincaré section then becomes `Event{G: func(t,y){return y[2]-z0}, Direction:+1}` and a one-loop reader. Refinement uses Brent on the cubic-Hermite interpolant of dense output.

---

## 6. Dense output (absent)

`SolveODE` exposes only the on-the-grid samples at `t0, t0+dt, t0+2dt, ...`. If a consumer wants `y(t0 + 1.337*dt)` they're stuck — no interpolation between steps, can't query "where was the trajectory exactly when `g(t,y)=0`?", can't render a smooth curve at sub-step resolution, can't snapshot a movie at irregular fps. Every modern integrator (DOPRI5, Verner-7, OrdinaryDiffEq.jl, scipy `solve_ivp` with `dense_output=True`) ships dense output as a tabulated quartic-Hermite polynomial per step.

For RK4 the dense output formula is the cubic-Hermite interpolant
`y(t_n + θ*dt) = (1-θ)·y_n + θ·y_{n+1} + θ(1-θ)·[(1-2θ)(y_{n+1}-y_n) + (θ-1)·dt·k1 + θ·dt·k4]`
which uses `k1, k4` from the step that's already in the workspace. **Free.**

**A6 (ship with A4, 40 LOC).**

```go
type Trajectory interface {
    ...
    Interpolate(t float64, out []float64) error  // dense output between last two steps
}
```

Stash `k1, k4, t_n, y_n, dt` on the iterator after each `Step()`. `Interpolate(t)` does the Hermite blend into caller-supplied `out`. Combined with §5 this gives event-detection-with-refinement at no extra integrator cost.

---

## 7. State reuse: eager `[][]float64` vs streaming

`SolveODE` returns `[][]float64`. Two problems:

1. **Allocates one `[]float64` per step** (`ode.go:122` — `row = make([]float64, n)`) — for a `tEnd=1000, dt=0.001` Lorenz call that's 10⁶ separate heap allocations of 24 bytes each (3 floats), plus a slice header per row, plus the outer `[][]float64` capacity. ~100 MB of allocator churn for a 24 MB result.
2. **No way to stream** — if you want to compute `mean(x)` over the whole trajectory you must hold every step in memory simultaneously. Consumer Pulse (trend modeling) wants online statistics over multi-day simulated horizons; today they cannot use `SolveODE`.

**A7 (subsumed by A4 iterator).** The `Trajectory` iterator from §4 fixes both. For the eager case, `Solve(s, p) [][]float64` stays as a one-shot wrapper but uses **a single contiguous backing slice** of size `(steps+1)*n` and slices into it — single allocation, cache-friendly traversal. (Pattern used by `linalg` for matrices.)

---

## 8. Integrator switching: same problem, different stepper

Today: copy-paste the loop, swap `RK4Step` for `EulerStep`, fix every call site. With the §2 `Stepper` interface and §3 `Problem` struct it becomes:

```go
tr1 := Integrate(chaos.RK4{}, prob)
tr2 := Integrate(chaos.Euler{}, prob)
tr3 := Integrate(chaos.Leapfrog{}, prob)   // 026-N2 wants this
tr4 := Integrate(chaos.DOPRI5{atol: 1e-9, rtol: 1e-6}, prob)  // 026-N1 wants this
```

This is the **payoff** of A2+A3: convergence-order studies, stiff/non-stiff comparisons, symplectic-vs-non-symplectic energy-drift studies (which 026-N2 calls out as currently broken) become one-line A/B tests instead of separate test files.

---

## 9. Ensemble / multi-IC integration

Use cases: Lyapunov-spectrum estimation needs N tangent-vector trajectories integrated jointly; basin-of-attraction needs a grid of ICs each integrated to attractor; uncertainty quantification needs a perturbation cloud; Pistachio's 1000 particles is literally an ensemble of 1000 ICs sharing one derivative. Today every one of these requires N separate `SolveODE` calls (with N separate workspace allocations from §1). There is no batched-IC integrator and no goroutine-parallel Solve.

**A8 (ship after A1+A4, 60 LOC).**

```go
type Ensemble struct{ Problems []Problem }  // shared F is fine; vary Y0/Params
func IntegrateEnsemble(s Stepper, e Ensemble, workers int) []Trajectory
```

Each goroutine owns its own `*RK4Workspace` so there is zero contention. The workers default to `runtime.GOMAXPROCS(0)`. **Embarrassingly parallel; the only real design question is whether iterators can advance in lock-step (needed for tangent-space Lyapunov, where periodic Gram-Schmidt re-orthogonalization couples the N trajectories).** Recommend a `Synchronize() error` method on the ensemble that all iterators block on at chosen step counts.

---

## 10. Comparison with sibling packages

Pattern parity check across the repo:

| Pattern | `signal/` | `optim/` | `prob/` | `linalg/` | `chaos/` |
|---|---|---|---|---|---|
| In-place `out []float64` arg | yes (`FFT`, `Convolve`) | yes (`LBFGS` updates in place) | yes (`Sample`) | yes (`MatMul(out)`) | **partial** (steppers yes, `SolveODE` no) |
| Workspace struct for hot-loop ops | yes (FFT bit-reverse) | yes (`lbfgs.state`) | mixed | yes (`Decomp`) | **no** |
| Online/streaming variant | n/a | yes (`Step()` on optim) | yes (`Welford`) | n/a | **no** |
| Convergence/early-termination callbacks | n/a | yes (`MaxIter`, `Tol`) | n/a | n/a | **no** |
| Problem struct bundling args | n/a | yes (`Problem{F, Grad, X0}`) | yes (`Distribution`) | n/a | **no** |
| Multiple algorithms sharing one problem type | n/a | yes (BFGS, LBFGS, NelderMead share `Problem`) | yes (`Sampler` interface) | n/a | **no** |

`chaos/` is the only computational package in reality without a workspace, without a problem struct, without an interface for swapping algorithms, and without a streaming variant. The patterns are **already in the repo** (`optim/` has every one of them) — `chaos/` just hasn't picked them up.

---

## Ranked action list (API-axis only)

| ID | Item | LOC | Unblocks |
|---|---|---|---|
| **A1** | `RK4Workspace` + `RK4StepInto` (alloc-free) | 30 | 60 FPS Pistachio; 026-N1 |
| **A2** | `Stepper` interface; rename `RK4`/`Euler` to types | 80 | integrator switching, A/B tests, 026-N2 (Leapfrog), 026-N1 (DOPRI5) |
| **A3** | `Problem` struct (`F, Y0, T0, T1, DT, Params`) | 25 | unifies BifurcationDiagram, Lyapunov, Poincaré, basin |
| **A4** | `Trajectory` iterator + `Hooks` + `Solve` wrapper | 50 | streaming, callbacks, kills 10⁶-row alloc storm |
| **A5** | `Event` struct + Brent refinement on dense output | 70 | Poincaré section (028-axis-11), termination, basin boundary |
| **A6** | Dense-output Hermite interpolation on iterator | 40 | sub-step queries, event refinement, smooth rendering |
| **A7** | (subsumed by A4) | — | — |
| **A8** | `Ensemble` + parallel `IntegrateEnsemble` | 60 | tangent Lyapunov, basin grid, particle clouds |
| | **Total** | **~355** | every API axis the topic asked about |

**Single highest-leverage commit:** **A1+A2+A3+A4 together (~185 LOC)**. Everything else (events, dense output, ensemble) plugs into the iterator/stepper/problem trio. Ships with backwards-compatible `RK4Step`/`SolveODE` wrappers so no consumer breaks. Prerequisites for **027 Tier-1 systems registry** and **028 axes 1-8**.

---

## Out-of-scope (covered by sibling reports)

- Numerical accuracy of RK4, symplectic methods, Lyapunov spectrum estimator → **026**
- Missing systems (Hénon, Mackey-Glass, Duffing, Chua, Kuramoto-Sivashinsky) → **027**
- DynamicalSystem first-class type, Systems registry, StateSpaceSet container, `lyapunovspectrum`, `correlationsum`, RQA struct → **028**

This report is interface-shape only.
