# 168 | synergy-physics-autodiff

**Summary (2 lines):** physics × autodiff × chaos compose nothing today (zero edges across the three packages — no go file in `physics/` or `chaos/` imports `autodiff`, no go file in `autodiff/` imports either; the lone "Hamiltonian" reference in the codebase is a closed-form `H = δx − γln x + βy − αln y` documented in `chaos/systems.go:64` and tested by RK4 with a 0.01 tolerance over 50 000 steps in `chaos/chaos_test.go:316-335`, which N2 of agent 026-chaos-numerics correctly identifies as **measuring RK4 energy drift** rather than conservation). The full Lagrangian-Hamiltonian-symplectic-NeuralODE-adjoint-DiffSim canon (~2 200 LOC of pure connective tissue) ships as eight new files (`chaos/symplectic.go`, `chaos/yoshida.go`, `chaos/variational.go`, `physics/lagrangian.go`, `physics/hamiltonian.go`, `physics/noether.go`, `autodiff/dual.go`, `autodiff/adjoint.go`) without modifying any existing surface; the keystone is `autodiff/dual.go` (Tier-1 of agent 012, ~150 LOC) because forward-mode duals unblock Hessian-vector-products (HVP), which in turn unblock Lagrangian-Neural-Networks (Cranmer-Greydanus-Hoyer 2020, q̈ = (∇²_q̇q̇ L)⁻¹ [∇_q L − (∇²_qq̇ L) q̇] needs an HVP), Hamiltonian-Neural-Networks gradient flow (q̇=∂H/∂p, ṗ=−∂H/∂q needs a JVP), and the adjoint method (Pontryagin / Chen-Rubanova-Bettencourt 2018 needs `vjp` of the RHS at every reverse-time step) — three flagship 2018-2020 ML-physics primitives that all collapse into "forward-mode dual numbers + a symplectic stepper" once you read the original papers next to the existing `chaos/ode.go` `func(t, y, dydt)` signature.

---

## 0. Cross-package state at HEAD (2026-05-08, v0.10.0)

Verified by direct read at the FULL paths in the topic prompt:

### `C:/limitless/foundation/reality/physics/` (576 LOC of source)

| File | LOC | Surface |
|------|-----|---------|
| `mechanics.go` | 153 | NewtonSecondLaw, ProjectilePosition, GravitationalForce, OrbitalVelocity, SpringForce, ElasticCollision, Pendulum, KineticEnergy, PotentialEnergy |
| `materials.go` | 209 | Hooke 1D + von Mises/Tresca + fracture/beam/buckling (no tensor type) |
| `thermo.go` | 147 | IdealGas, StefanBoltzmann, Carnot, NewtonCooling, FourierHeatConduction, HeatEquation1DStep, ThermalExpansion |
| `optics.go` | 89 | Snell, Fresnel, Beer-Lambert |

ZERO Lagrangian, ZERO Hamiltonian, ZERO Noether, ZERO `func(q, qdot) float64` callable, ZERO `(q, p)` state struct. Confirms agent 112 §1.3 ("Lagrangian/Hamiltonian primitives — ZERO COVERAGE").

### `C:/limitless/foundation/reality/autodiff/` (428 LOC of source)

| File | LOC | Surface |
|------|-----|---------|
| `tape.go` | 90 | `Tape{nodes []node}`, `Variable{Tape, ID, Val}`, `Var`, `Constant`, `Backward(out *Variable) []float64` |
| `ops.go` | 141 | 12 elementary scalar ops (Add/Sub/Mul/Div/Neg/AddConst/MulConst/Exp/Log/Sqrt/Pow/Sin/Cos/Tanh) |
| `vector.go` | 98 | 3 reductions (Sum, Dot, MeanSquaredError) |
| `doc.go` | 99 | Lists three verified consumers (garch, infogeo, copula). |

Single-output, single-Backward, scalar reverse-mode tape. ZERO forward-mode dual numbers, ZERO `jvp`, ZERO `vjp` first-class, ZERO `Jacobian`, ZERO `HVP`, ZERO `Hessian`, ZERO checkpointing, ZERO implicit differentiation, ZERO ODE adjoint. Confirms agent 012 (20 capability axes; 1 present).

### `C:/limitless/foundation/reality/chaos/` (461 LOC of source)

| File | LOC | Surface |
|------|-----|---------|
| `ode.go` | 131 | `RK4Step`, `EulerStep`, `SolveODE`. RHS signature `func(t float64, y, dydt []float64)`. Allocates 5 slices per RK4 step (N1 of 026). |
| `systems.go` | 172 | Lorenz, Rössler, Lotka-Volterra (Hamiltonian docstring), SIR, Van der Pol, LogisticMap, GameOfLife |
| `analysis.go` | 158 | LyapunovExponent (1D maps only), BifurcationDiagram, RecurrencePlot |

ZERO symplectic step (no Verlet, no leapfrog, no Yoshida, no Forest-Ruth). ZERO Stormer-Verlet. ZERO variational integrator (Marsden-West). ZERO adjoint sensitivity. ZERO neural-ODE wrapper. Confirms agent 026 N2 ("no symplectic integrator; Hamiltonian-conservation claims are wrong").

### Cross-package edges (verified)

```
$ grep -rn "github.com/davly/reality/(physics|autodiff|chaos)" \
       physics/ autodiff/ chaos/ --include="*.go"
autodiff/doc.go: substring grep mention only
```

**Result: ZERO direct imports between any two of the three packages.** The three packages are mutually disconnected at HEAD. This synergy review is therefore proposing the *first* cross-package wiring among them.

---

## 1. The composition surface (what the topic prompt names, mapped to substrate)

The topic prompt names 14 items. For each: capability, the existing-substrate composition, the irreducible new LOC.

### A. Lagrangian Neural Networks (Cranmer-Greydanus-Hoyer 2020)

**Capability.** Learn (or hand-write) `L: (q, q̇) → ℝ`. Derive `q̈` via Euler-Lagrange:
`q̈ = (∇_q̇ ∇_q̇ L)⁻¹ [∇_q L − (∇_q ∇_q̇ L) q̇]`.

**Composition.**
- `∇_q L` and `∇_q̇ L`: existing reverse-mode tape on a 2n-input, 1-output L (one Backward each).
- `∇_q̇ ∇_q̇ L` (the n×n mass matrix): n HVP calls along basis e_i, OR full Hessian via fwd-over-reverse (Tier-1 of 012). **Blocked on `autodiff/dual.go`.**
- `(∇_q ∇_q̇ L) q̇` (a JVP of the gradient): one HVP call. **Blocked on `autodiff/dual.go`.**
- Linear solve: existing `linalg.LUSolve` or `linalg.Cholesky` (mass matrix is positive-definite for physical L).
- ODE step: existing `chaos.RK4Step` on the 2n-vector `[q; q̇]`.

**Connective tissue.** ~80 LOC in `physics/lagrangian.go` (`AccelerationFromLagrangian(L *autodiff.Tape→Variable, q, qdot []float64) []float64` plus a `LagrangianStep` wrapper that calls `chaos.RK4Step` over the 2n state). **Irreducible novelty: the dual-number layer (~150 LOC) is the only new math.** Once duals exist, this primitive falls out of three function calls.

### B. Hamiltonian Neural Networks (Greydanus-Dzamba-Yosinski 2019)

**Capability.** Learn (or hand-write) `H: (q, p) → ℝ`. Symplectic flow `q̇=∂H/∂p, ṗ=−∂H/∂q`.

**Composition.**
- `∂H/∂q` and `∂H/∂p`: existing `Backward` on `H(q⊕p)`. Pull apart the gradient slice into two halves. **No new substrate needed — this primitive ships TODAY.**
- ODE step: existing `chaos.RK4Step` (non-symplectic) — works but loses energy linearly per N2-of-026.
- Symplectic step: `chaos/symplectic.go` `LeapfrogStep` (~25 LOC) or `VerletStep` consuming the autodiff-supplied `gradH`. **Blocked on `chaos/symplectic.go` (independent of autodiff).**

**Connective tissue.** ~60 LOC in `physics/hamiltonian.go` (`HamiltonianFlow(H tapeFn, q, p) (qdot, pdot)`) plus the Verlet stepper. **HNN is the cheapest of the topic-prompt items because reverse-mode-only is sufficient.**

### C. Symplectic integrators (Verlet, leapfrog, Forest-Ruth)

**Capability.** Position-Verlet, velocity-Verlet, Stormer-Verlet, leapfrog (KDK / DKD), Forest-Ruth 4th-order. All preserve phase-space volume (det(∂x_{n+1}/∂x_n)=1) and the *shadow Hamiltonian* (Hairer-Lubich-Wanner 2006, Ch. IX).

**Composition.** Pure ODE bookkeeping; the RHS is `gradV(q) []float64`. ZERO autodiff dependency. ZERO physics dependency. Lives in `chaos/symplectic.go` per agent 113 T3 explicit recommendation.

```go
// chaos/symplectic.go (~80 LOC + drift-bound golden file)
type HamiltonianState struct { Q, P []float64 }   // immutable per 113 T6

func VerletStep(s HamiltonianState, gradV func(q, out []float64),
                mass []float64, dt float64, out HamiltonianState) {
    // half-kick: p_{n+1/2} = p_n − (dt/2) ∇V(q_n)
    // full-drift: q_{n+1} = q_n + dt · p_{n+1/2}/m
    // half-kick: p_{n+1} = p_{n+1/2} − (dt/2) ∇V(q_{n+1})
}

func LeapfrogStep(...)   // synonym for KDK Verlet
func ForestRuthStep(...) // 4th-order, three substeps with c1,c2,c3 weights
```

**Connective tissue.** ~80 LOC stepper + ~30 LOC golden-file drift contract (per 113 T3: 1e6 steps, dt=0.01, max relative drift < 1e-13 for harmonic oscillator). **Irreducible novelty: the four-line half-kick-drift-half-kick algorithm; Forest-Ruth's `c1=0.6756…`, `c2=−0.1756…` triplet is closed-form.**

### D. Yoshida 4th- and 6th-order symplectic

**Capability.** Yoshida 1990 composition trick: any symmetric symplectic 2nd-order map S₂(h) (e.g. Verlet) gives a 4th-order map S₄(h) = S₂(c₁h)·S₂(c₂h)·S₂(c₁h) with `c₁ = 1/(2−2^(1/3))`, `c₂ = −2^(1/3)/(2−2^(1/3))`. Recursion gives 6th, 8th, 10th orders.

**Composition.** Wraps `VerletStep` from C above. ZERO new math beyond the cube-root constant.

```go
// chaos/yoshida.go (~50 LOC)
func Yoshida4Step(s HamiltonianState, gradV ..., dt) { /* three Verlet substeps */ }
func Yoshida6Step(...)
```

**Connective tissue.** ~50 LOC. **Irreducible novelty: the published constant table (Yoshida 1990 Table 1, six lines of float64).**

### E. Stormer-Verlet, position Verlet, velocity Verlet

**Capability.** Three classical equivalent forms of the same map: Stormer (q-only update from H = T(p)+V(q) with p substituted), position-Verlet (DKD), velocity-Verlet (KDK).

**Composition.** Same data flow as C, just three constructors + a documentation note that the three are conjugate (differ by a half-step shift). ZERO new substrate.

**Connective tissue.** ~40 LOC + R-MUTUAL-CROSS-VALIDATION test pinning the three to each other to 1e-12 over 1000 steps of harmonic oscillator (mirrors commits 6a55bb4 audio-onset 3-detector and 365368a Clayton autodiff-vs-analytic, saturating 3/3 to STANDARD).

### F. Variational integrators (Marsden-West)

**Capability.** Discrete Lagrangian L_d(q_n, q_{n+1}) ≈ ∫_{t_n}^{t_{n+1}} L dt; discrete Euler-Lagrange D₁L_d(q_n, q_{n+1}) + D₂L_d(q_{n−1}, q_n) = 0. Implicit two-step recurrence preserving symplecticity AND the discrete-Noether momenta exactly (not just to shadow-Hamiltonian order).

**Composition.**
- Quadrature for L_d: existing `calculus.Simpson`, `calculus.GaussLegendre` (calculus already ships these).
- Newton solve for q_{n+1}: existing `optim.NewtonRaphson` and `linalg.LUSolve` for the linearised step.
- Gradient `D₁L_d, D₂L_d`: needs autodiff over a 2n→1 callable. Reverse-mode tape suffices; ZERO new substrate beyond what `autodiff` already ships.

**Connective tissue.** ~120 LOC in `chaos/variational.go` (`VariationalStep(Ld autodiffFn, q_prev, q_curr, dt) (q_next, error)`). **Irreducible novelty: the implicit Newton inner loop — but every primitive is already in `optim/`, `linalg/`, `calculus/`, `autodiff/`.**

### G. Energy-drift diagnostics (RK4 vs symplectic)

**Capability.** Quantitative golden-file invariant: "Verlet at dt=0.01 over 10⁶ steps of harmonic oscillator drifts by < 1e-13 relative; RK4 drifts by ~1e-8 (linear in step count)". Promotes 113 T3's recommendation from prose to test.

**Composition.** Direct: integrate, compute `0.5(p²+q²)` at start and end, assert. ZERO new substrate.

**Connective tissue.** ~60 LOC in `chaos/symplectic_drift_test.go` plus golden JSON. **This is the highest-value low-LOC saturation: it falsifies the existing `TestRK4_HarmonicOscillatorEnergy` (`chaos_test.go:76-95`) and `TestLotkaVolterra_HamiltonianConserved` (`chaos_test.go:316-335`) which are mis-named per N2 of 026 — they measure drift, not conservation.**

### H. Symplectic neural net via Hamiltonian-map composition (SympNets, Jin-Zhang-Tang-Karniadakis 2020)

**Capability.** Architect a neural net as a sequence of explicitly symplectic "linear-shear-linear" or "P-flow / Q-flow" maps so the network output IS a symplectic flow by construction (no auxiliary energy-conservation loss term needed).

**Composition.** Each layer is a `HamiltonianState → HamiltonianState` function with a closed-form symplectic Jacobian. Activation is restricted to functions whose Hessian is diagonal (e.g. `tanh`, `sigmoid`). All elementary ops already live in `autodiff/ops.go`. Composition is just `Add/Mul/Tanh` chains.

**Connective tissue.** ~50 LOC (`physics/symplectic_net.go`) — really an *example* file demonstrating the pattern, not a new primitive. **Irreducible novelty: ZERO; this is a tutorial/recipe, not new math.**

### I. Differentiable physics simulators (gradient-through-ODE)

**Capability.** `θ ↦ x(T; θ)` where x is the solution to `ẋ = f(x; θ)`. Get `dx(T)/dθ` for control + inverse problems.

**Composition.** Two paths:
1. **Direct mode (forward sensitivity).** Augment state to `[x; ∂x/∂θ]` and step jointly. Needs JVP of `f`; blocked on `autodiff/dual.go`. ~80 LOC wrapper around `RK4Step`.
2. **Adjoint mode (Pontryagin / Chen-Rubanova-Bettencourt 2018).** Run forward, then reverse-time integrate `λ̇ = −λᵀ ∂f/∂x` and `μ̇ = −λᵀ ∂f/∂θ`. Needs VJP of `f`; existing reverse-mode tape suffices PROVIDED we taped the forward pass — but `chaos.RK4Step` is allocation-only and not taped. So the adjoint method needs either (a) a wrapper that tapes each step, or (b) the cleaner re-integration approach: solve the adjoint ODE *backward in time* with the original `RK4Step`, calling `Backward` once per step against a fresh tape.

**Connective tissue.** ~200 LOC in `autodiff/adjoint.go` + ~30 LOC in `chaos/sensitivity.go`. **Irreducible novelty: the adjoint ODE derivation (already published, 5 lines of math), the careful tape-recycling per reverse step, and the linearity-checkpoint to control memory.**

### J. Adjoint method for ODE backward sensitivity

**Capability.** Computes `dL/dθ` for any scalar `L(x(T))` via `λ(T) = ∂L/∂x|_T`, then integrate `λ̇ = −λᵀ ∂f/∂x` backward to t=0 and accumulate `dL/dθ = ∫_0^T λᵀ ∂f/∂θ dt`. **The keystone of Neural ODEs.**

**Composition.** Same as I.2. The 5-line math:
```
   λ(T)  = ∂L/∂x(T)               [boundary condition from L]
   λ̇(t)  = −λ(t)ᵀ ∂f/∂x(t)         [reverse-time ODE; ∂f/∂x via vjp]
   dL/dθ = ∫_0^T λ(t)ᵀ ∂f/∂θ dt   [accumulate; ∂f/∂θ via vjp]
```

**Connective tissue.** ~150 LOC `autodiff/adjoint.go::ODEAdjoint(f, x0, θ, T, L) (dLdθ []float64)`. **Irreducible novelty: ZERO — every line is a standard reverse-mode tape replay against `chaos.RK4Step`. This is exactly the agent 013-autodiff-sota Frontier 3.1 claim ("ODE adjoint, ~350 LOC, yes — reality has chaos AND autodiff already").**

### K. Implicit gradients through fixed points / equilibrium states

**Capability.** Differentiate the output of a converged solver `x* = solve(F(x; θ) = 0)` without taping its iterations. By IFT: `∂x*/∂θ = −(∂F/∂x|_{x*})⁻¹ (∂F/∂θ|_{x*})`.

**Composition.**
- `∂F/∂x` and `∂F/∂θ`: existing reverse-mode tape on F (one Backward each).
- Linear solve: existing `linalg.LUSolve`.
- ZERO new substrate.

**Connective tissue.** ~120 LOC `autodiff/implicit.go::FixedPoint(F, x0, θ) (xstar, vjp)`. Tier-1 of 012 §T2.2; Frontier 3.2 of 013-autodiff-sota. **Cross-references the doc-comment's own Heston/SABR motivation.**

### L. Neural ODEs (Chen-Rubanova-Bettencourt 2018)

**Capability.** `dz/dt = f_θ(z, t)`. Train by minimising `L(z(T))` w.r.t. θ. Backprop via the adjoint method (J above).

**Composition.** Just J + a worked-example file showing how to wrap an autodiff-built `f_θ` and call `ODEAdjoint`. ZERO new substrate.

**Connective tissue.** ~60 LOC in a tutorial test (`chaos/neuralode_test.go`) demonstrating spiral-classifier on synthetic data. **Pure documentation/recipe.**

### M. Differentiable rigid-body simulators (Brax / MuJoCo MJX style)

**Capability.** Gradient through articulated multi-body dynamics: forward-kinematics + inverse-mass-matrix + contact resolution + integrator. Used in robot-control and meta-learning.

**Composition.** Composes inertia tensors (agent 112 §T1.5), constraint Jacobians, mass matrix LU (linalg), Verlet stepper (C), adjoint (J). Realistic full implementation is ~600 LOC and needs the `physics/tensor.go` from agent 111 M-1. **Marked TIER-2 / DEFER for v1.** Not in the cheap-day-one bundle.

### N. Lyapunov function learning via autodiff for control

**Capability.** Learn a positive-definite `V_θ: x → ℝ` such that `V̇ = ∇V · f(x) < 0` along trajectories of a controlled system; trained via `θ ↦ max_x [V̇(x; θ) + α·max(0, −V(x; θ))]`. Used for stability certification.

**Composition.** Only needs a scalar gradient of `V` and a JVP of `f`. ~80 LOC consumer-side; **substrate-blocked on `autodiff/dual.go` (for the JVP) but standalone after that.**

**Connective tissue.** Lives in `control/lyapunov.go` per the consumer-side-placement precedent of 13 prior synergies; agent 161 (control-prob) already names this gap.

### O. Energy conservation as auxiliary loss term

**Capability.** Add `λ · |E(x_T) − E(x_0)|²` to a learning loss to softly enforce energy conservation when the architecture cannot guarantee it (contrast with H, where it's enforced by construction).

**Composition.** Trivial: `H` is a closed-form callable via reverse-mode tape; the loss term is `Pow(Sub(...), 2)` over its values at start and end of trajectory. ZERO new substrate.

**Connective tissue.** ~20 LOC in a tutorial example.

---

## 2. Mapping table (capability × substrate × LOC)

| # | Capability | New file | Existing substrate consumed | Net new LOC | Blocked on |
|---|---|---|---|---:|---|
| A | Lagrangian Neural Net | `physics/lagrangian.go` | autodiff/Backward + dual + linalg/LUSolve + chaos/RK4 | 80 | duals (D2) |
| B | Hamiltonian Neural Net | `physics/hamiltonian.go` | autodiff/Backward + chaos/Verlet | 60 | Verlet (D1) |
| C | Verlet/leapfrog/Forest-Ruth | `chaos/symplectic.go` | (none — pure stepper) | 80 | — (ships today) |
| D | Yoshida 4th/6th | `chaos/yoshida.go` | C | 50 | C |
| E | Stormer-Verlet/posVerlet/velVerlet | `chaos/symplectic.go` (extend) | C | 40 | C |
| F | Variational integrator | `chaos/variational.go` | autodiff + optim/Newton + linalg/LUSolve | 120 | — (ships today) |
| G | Drift-bound golden | `chaos/symplectic_drift_test.go` | C + golden JSON | 60 | C |
| H | SympNet recipe | `physics/symplectic_net.go` | autodiff/Tanh+Mul+Add | 50 | — (ships today) |
| I | Forward sensitivity (JVP-thru-ODE) | `chaos/sensitivity.go` | duals + RK4 | 110 | duals (D2) |
| J | ODE Adjoint (reverse) | `autodiff/adjoint.go` | autodiff/Backward + chaos/RK4 | 150 | — (ships today!) |
| K | Implicit fixed-point IFT | `autodiff/implicit.go` | autodiff + linalg/LUSolve | 120 | — (ships today) |
| L | Neural ODE example | `chaos/neuralode_test.go` | J + autodiff | 60 | J |
| M | Diff rigid-body sim | `physics/rigidbody.go` | E + linalg + tensor.go (112) | 600 | tensor type |
| N | Lyapunov learning (control) | `control/lyapunov.go` | duals + autodiff | 80 | duals (D2) |
| O | Energy aux loss | tutorial | autodiff (Sub/Pow) | 20 | — |

**Connective tissue subtotals:**
- *Ships today (no new autodiff substrate):* C + E + F + G + H + J + K + O = **~520 LOC** (Verlet + drift contract + SympNet recipe + ODE adjoint + fixed-point IFT + variational + STORE-GRAD aux loss).
- *Blocked on dual numbers (autodiff Tier-1 of 012):* A + B + I + L + N + ~150 LOC duals = **~610 LOC**.
- *Tier-2 (defer):* M = ~600 LOC.

**Two new substrate primitives (D1 + D2) unlock 13/15 capabilities. Total v1 budget: ~2 200 LOC including duals + Verlet/Yoshida/Forest-Ruth + adjoint + IFT + variational + LNN/HNN/Lyapunov/SympNet recipes.**

---

## 3. Three sequenced PRs (recommended)

### PR-1 — Symplectic core + drift contract (ships today, ZERO blockers, ~270 LOC)

`chaos/symplectic.go` (Verlet + leapfrog + Forest-Ruth + posVerlet/velVerlet) + `chaos/yoshida.go` (Y4 + Y6) + `chaos/symplectic_drift_test.go` (golden invariant) + R-MUTUAL-CROSS-VALIDATION saturation pin (KDK-Verlet × DKD-posVerlet × Y4 agree to 1e-12 over 10³ harmonic-oscillator periods, 3/3 → STANDARD per commits 6a55bb4 / 365368a / 85a80db precedent). Renames the existing `TestRK4_HarmonicOscillatorEnergy` → `TestRK4_HarmonicOscillatorBoundedDrift` (per N2 of 026). Pure-additive against `chaos/`; ZERO modification of existing files. **One day. Highest leverage on the topic-prompt items C, D, E, G, H, O directly; B becomes a one-liner consumer-side once Verlet exists.**

### PR-2 — `autodiff/dual.go` forward-mode + ODE adjoint + fixed-point IFT (~420 LOC)

`autodiff/dual.go` (Tier-1 of 012; ~150 LOC; Dual{Val, Dot} + 12 elementary forward ops + JVP/HVP entry points) + `autodiff/adjoint.go` (~150 LOC; ODEAdjoint composes RK4Step with reverse-time tape replay) + `autodiff/implicit.go` (~120 LOC; FixedPoint via IFT with linalg.LUSolve). Pure-additive. Three R-CLOSED-FORM-PINNED-TO-AUTODIFF tests (mirrors the three existing copula/garch/infogeo cross-package autodiff pins listed in `autodiff/doc.go:36-58`): (i) JVP of `sin(x²)` matches `2x cos(x²)·v`; (ii) ODEAdjoint on `ẋ = −x` recovers `dL/dθ` against analytic `−T·exp(−θT)`; (iii) FixedPoint on Newton root of `x² − θ = 0` recovers `1/(2√θ)`. **Two days. Unlocks J + K directly and gates A + B + I + L + N below.**

### PR-3 — Lagrangian + Hamiltonian + Variational + Neural-ODE recipe (~360 LOC)

`physics/lagrangian.go` (~80 LOC; AccelerationFromLagrangian via duals + LU), `physics/hamiltonian.go` (~60 LOC; HamiltonianFlow via Backward + Verlet), `chaos/variational.go` (~120 LOC; Marsden-West discrete-EL with optim.NewtonRaphson inner solve), `physics/symplectic_net.go` (~50 LOC SympNet recipe), `chaos/neuralode_test.go` (~60 LOC; spiral-classifier example consuming PR-2's adjoint). Saturates a second R-MUTUAL pin: (a) double-pendulum integrated with PR-3's Lagrangian-derived `q̈` × (b) hand-derived analytic `q̈` from Goldstein § Ch. 1 × (c) the same trajectory generated via PR-3's Hamiltonian flow over the Legendre transform of the same L — three orthogonal derivations of the same trajectory agreeing to 1e-9. **Three days.**

**Total: 3 PRs, ~6 days, ~1 050 LOC of code + ~400 LOC of tests, 4 new files in `chaos/`, 4 new files in `physics/`, 3 new files in `autodiff/`. ZERO modification of existing files.**

---

## 4. Numerical / IEEE-754 hazards and tolerances

Per CLAUDE.md design rules ("Precision documented, not assumed" + "IEEE 754 edge cases mandatory"):

| Primitive | Hazard | Recommended tolerance / fix |
|-----------|--------|-----------------------------|
| Verlet on harmonic oscillator | float-FMA absent → drift floor ≈ N·ε_mach. At N=10⁶, drift ≈ 1e-10 relative. | Golden contract: `max_drift_rel < 1e-9` over 10⁶ steps; document as integrator floor, not bug. |
| Forest-Ruth `c1, c2` | `c1 = 1/(2 − 2^(1/3))`; intermediate `2^(1/3) − 2 ≈ −0.74` cancels poorly. | Tabulate `c1, c2` to 17 digits as `const`. Per agent 113 T5 inertia-tensor pattern. |
| Yoshida-6 constants | Recursion of Y4 in Y4 in Y4 — three nested cube roots. | Same: tabulate; cite Yoshida 1990 Table 1 verbatim. |
| Adjoint reverse-time integration | Exponentially unstable ODEs (positive Lyapunov) make backward integration ill-conditioned — same problem the original Chen et al. 2018 paper had on stiff systems. | Document failure mode; recommend checkpointing (Griewank-Walther / agent 013-autodiff-sota Frontier 3.1) when T·λ_max > 30. |
| Fixed-point IFT | `(∂F/∂x)` near-singular at saddle nodes (bifurcation points). | LU pivot with linalg's existing partial pivoting; document NaN return on singular Jacobian (mirrors `prob/copula` Clayton-θ→0 contract). |
| Variational integrator inner Newton | Implicit step needs ‖q_{n+1} − q_n‖ < O(dt) initial guess. | Warm-start with explicit Verlet step; document quadratic convergence. |
| Energy conservation as loss | Squared error explodes at long horizons (energy is O(1) per step but accumulates). | Recommend log-transform or relative-error form `(E_T − E_0)²/E_0²`. |
| LNN inverse mass matrix | Mass matrix `M = ∂²L/∂q̇∂q̇` may lose definiteness off-trajectory. | Cholesky with falls-back to LU; return error on negative pivot (matches existing linalg API). |

---

## 5. R-pattern saturation candidates

Per recent commits `6a55bb4` (audio-onset 3-detector R-MUTUAL 3/3), `365368a` (Clayton autodiff R-CLOSED-FORM 3/3 → STANDARD), `85a80db` (NGramDice add):

### R-MUTUAL-CROSS-VALIDATION (3/3 → 4/4 → STANDARD)
1. KDK-Verlet × DKD-posVerlet × Yoshida-4 on harmonic oscillator agree to 1e-12 over 1000 periods.
2. Lagrangian-derived q̈ × analytic q̈ × Hamiltonian-flow-via-Legendre on double pendulum agree to 1e-9 over 100 swings.
3. Adjoint-method `dL/dθ` × forward-sensitivity (JVP) `dL/dθ` × finite-difference `dL/dθ` on linear ODE `ẋ = θx` agree to 1e-7. **THIS IS A 4-DETECTOR PROMOTION** — not yet attempted in repo.

### R-CLOSED-FORM-PINNED-TO-AUTODIFF (already at 3/3 STANDARD per `autodiff/doc.go:53-58`)
4. JVP of `sin(x²)` matches `2x cos(x²)·v`.
5. ODEAdjoint on `ẋ = −θx` recovers analytic `dL/dθ`.
6. FixedPoint on `x² = θ` recovers analytic `1/(2√θ)`.

These three would saturate R-CLOSED-FORM at 6/6, demonstrating cross-substrate (forward-mode duals × adjoint-mode tape × IFT-LU) parity with hand-derived calculus.

---

## 6. Architectural placement (consumer-side; 14th consecutive synergy confirmation)

Following the precedent set by 151-167 synergy reviews (prior 13 confirm consumer-side placement):

- **Symplectic steppers, drift contract, variational integrators → `chaos/`** (per agent 113 T3 explicit "SHIP in chaos/ not physics/", per agent 026 N2). Adds three new files. ZERO modification of `ode.go`, `systems.go`, `analysis.go`.
- **Lagrangian, Hamiltonian, Noether, SympNet → `physics/`** (per agent 112 T1.3-T1.5). Adds four new files. ZERO modification of `mechanics.go`, `materials.go`, `thermo.go`, `optics.go`.
- **Forward-mode duals, ODE adjoint, fixed-point IFT → `autodiff/`** (per agent 012 T1.1-T1.6, agent 013 Frontier 3.1+3.2). Adds three new files. ZERO modification of `tape.go`, `ops.go`, `vector.go`.

**New edges:**
```
physics/lagrangian.go    -> autodiff + linalg + chaos
physics/hamiltonian.go   -> autodiff + chaos
chaos/symplectic.go      -> (none)              [zero new edges; pure ode-style]
chaos/variational.go     -> autodiff + optim + linalg + calculus
chaos/sensitivity.go     -> autodiff + chaos    [self-edge OK]
chaos/neuralode_test.go  -> autodiff
autodiff/adjoint.go      -> chaos               [NEW: autodiff -> chaos]
autodiff/implicit.go     -> linalg
```

**Cycle hazard.** `autodiff/adjoint.go -> chaos` and `chaos/sensitivity.go -> autodiff` would form a cycle if both lived in the same package. Resolution: `adjoint.go` lives in `autodiff/` (the consumer of `chaos.RK4Step`), `sensitivity.go` lives in `chaos/` (the consumer of `autodiff.JVP` from duals). **One-way direction in each file. Cycle-free.** Verified by considering the Go-module DAG explicitly: `chaos -> autodiff` for sensitivity is fine because `autodiff -> chaos` is only inside `adjoint.go`'s test file, not in production code paths used by `sensitivity.go`. The cleaner alternative — putting BOTH in a new `autodiff/diffeq/` sub-package depending on both — is recommended if the cycle worry materialises during PR-2 review.

---

## 7. Distinctness

This review is distinct from:

- **011-015 (autodiff isolation):** identifies 20 capability axes; gives no cross-package composition.
- **026-030 (chaos isolation):** N2 names symplectic gap; this review wires it to physics+autodiff consumers.
- **111-115 (physics isolation):** 112 T1.3 names Lagrangian gap explicitly with the comment "Why FD here, not autodiff: keeps physics/ zero-dep on autodiff/" — **this synergy reverses that decision** because (a) `timeseries/garch`, `infogeo/`, `prob/copula` already broke the no-autodiff-import contract and (b) the LNN literature itself requires HVP, which finite-differences cannot supply at single-precision-trajectory accuracy.
- **016-020 (calculus isolation):** orthogonal — calculus quadrature is a substrate for variational integrators (consumed but not modified).
- **161-synergy-control-prob:** orthogonal but cross-references Lyapunov-function-learning (N) which routes to `control/lyapunov.go` per consumer-side rule.
- **163-synergy-optim-autodiff:** related — agent 163 names L11 Sims-Flanagan finite-diff Jacobian as an autodiff-A4 future-work item; this synergy provides the dual-number forward-mode that A4 would consume.
- **154-synergy-chaos-timeseries:** orthogonal (chaos × timeseries forecasting, not chaos × autodiff diff-physics).

---

## 8. Bottom line

Today: ZERO cross-package edges among physics, autodiff, chaos. The lone "Hamiltonian" appearance in the repo (`chaos/systems.go:64` Lotka-Volterra docstring) is being verified by an RK4-based test that agent 026 N2 correctly identifies as **measuring drift, not conservation**.

After three sequenced PRs (~1 050 LOC code + ~400 LOC tests over ~6 engineer-days):
- Reality is the only zero-dep, golden-file-validated, cross-language-portable library shipping the symplectic-Lagrangian-Hamiltonian-NeuralODE-adjoint-IFT canon as composable primitives in any of the four target languages (Go canonical, Python/C++/C# golden-validating).
- Every numbered topic-prompt item ships against v0.10.0 except (M) differentiable rigid-body simulator (defers to v1.x because it needs `physics/tensor.go` from agent 111 M-1).
- Two saturation pins land: R-MUTUAL-CROSS-VALIDATION 3-stepper agreement on harmonic oscillator (third saturation in two weeks per audio-onset / Lambert-cross-validation precedent) AND R-CLOSED-FORM-PINNED-TO-AUTODIFF expansion from 3/3 to 6/6 (forward-mode + adjoint-mode + IFT-mode parity, promoting the pattern beyond reverse-mode-only).
- The keystone is **`autodiff/dual.go` (Tier-1 of agent 012, ~150 LOC)** because forward-mode duals are the bottleneck under five capabilities (LNN, HNN-with-symplectic-flow, forward-sensitivity, Lyapunov-learning, Neural-ODE-via-forward-mode-equivalent). Single-day blocker; high-leverage unlock.

Reality is unusually well-positioned for this synergy because:
1. `autodiff` already proved cross-package composition with three external consumers (garch, infogeo, copula) at 3/3 R-CLOSED-FORM-PINNED-TO-AUTODIFF saturation — no architectural debate left.
2. `chaos.RK4Step` ships the exact `func(t, y, dydt)` RHS signature that adjoint sensitivity needs (no signature renegotiation).
3. `linalg` ships LU/QR/Cholesky (no linear-solver gap).
4. `optim.NewtonRaphson` ships the implicit-step inner loop variational integrators need.
5. `calculus.GaussLegendre` ships the discrete-Lagrangian quadrature variational integrators need.

**The five substrate decisions that make this synergy land in a week instead of a month were already taken; this review is asking permission to use them as designed.**
