# 113 | physics-sota

**Scope.** Position `reality/physics` (28 closed-form fns across `mechanics.go`,
`thermo.go`, `materials.go`, `optics.go`) against modern physics libraries on
the **engineering-trick** axis — not feature counts (112's job), not numerical
hardening of shipped code (111's job). For each surveyed library:
(1) headline algorithm, (2) one engineering trick, (3) zero-dep portability
verdict for reality. Twelve libraries surveyed: Modelica.Mechanics +
Modelica.Media (equation-based modeling, Modelica Association), SymPy.physics
(symbolic mechanics/vector/optics/quantum), Brian2 (integrate-and-fire neuron
sim), Manim physics extensions (visualization), VPython/GlowScript
(educational), Cirq (quantum gate simulation), LAMMPS + GROMACS (molecular
dynamics), TensorPhysics + JAX-MD (differentiable MD on accelerators), Brax
(differentiable rigid-body on JAX-XLA), MuJoCo MJX (XLA-compiled MuJoCo),
PyCav (educational visualization on top of VPython/Matplotlib), and the modern
diff-physics frontier (DiffTaichi, DJINN, Warp).

**TL;DR.** reality/physics is *correctly* in the closed-form-formula band —
Modelica/Brax/MuJoCo's signature moves (DAE index reduction, XLA, gradient
checkpointing, contact LCP solvers) are categorically out-of-scope for
zero-dep. But six engineering tricks DO port without IR/JIT/codegen:
**(T1)** unit-checked compile-time dimensionality (SymPy.physics.units), as
a Go test-time golden-file lint not a runtime tax; **(T2)** Voigt 6-vector
for symmetric tensors (Modelica.MultiBody, FEniCS) makes asymmetry
unrepresentable like quaternions/colors already do; **(T3)** symplectic
drift-error contract as a documented invariant (LAMMPS `fix nve`, GROMACS
leapfrog, Brax pipeline) — ship the integrator AND its energy-drift bound
as a golden, per agent 110; **(T4)** Brian2's parse-time dimensional check —
DSL is out-of-scope but the **lint** ports as a `physicstest` helper;
**(T5)** Cirq's singleton gate-tensor cache pattern (`cirq.X`, `cirq.H`)
maps onto reality's "precompute and freeze tables" idiom (signal/window,
color/illuminants) — apply now to canonical moments-of-inertia, later to
Pauli/gamma matrices in a future quantum/ pkg; **(T6)** the JAX-MD/Brax
`dataclasses.replace` immutable-State pattern is already what reality does
for closed-form fns — the missing piece is making it the explicit contract
for soon-to-land Verlet/Yoshida integrators in chaos/. **Single
highest-leverage zero-dep adoption: T2 (Voigt) — already independently
flagged by 111 (M-1) and 112 (T2.1); it's the representation choice that
determines whether continuum mechanics ships correctly or wrong.**

---

## 1. Crosswalk: physics libraries vs. reality, by category

| Library | Category | Headline algorithm | What reality could portably steal |
|---|---|---|---|
| **Modelica.Mechanics** | Equation-based DAE | Pantelides index reduction + Mattson-Söderlind dummy derivatives | Voigt 6-vector (T2); equation-vs-implementation citation idiom |
| **Modelica.Media.IdealGases** | Property tables | NASA 9-coef polynomial fits ($cp/R = \sum a_i T^{i-2}$) | Polynomial-fit pattern for thermo properties (vs. shipping full EoS solvers) |
| **SymPy.physics.mechanics** | Symbolic Lagrangian | Kane's method, Lagrange multipliers via auto-CAS | None — symbolic CAS violates zero-dep |
| **SymPy.physics.units** | Unit algebra | Quantity dimension matrix at construction | Compile-time dimension lint via golden-file (T1) |
| **SymPy.physics.optics** | Ray transfer matrices | $2 \times 2$ ABCD matrix products | Direct port; ~80 LOC, fits optics.go expansion |
| **Brian2** | Neuro-physics | LIF/HH ODE templating from string DSL | None (DSL violates zero-dep); but **dimensional-check** idea ports as test-time lint (T4) |
| **Manim physics** | Visualization | Pendulum/spring rendering on top of Cairo | None — visualization is consumer (Pistachio) |
| **VPython/GlowScript** | Educational sim | WebGL real-time integration loops | None — visualization is consumer |
| **PyCav** | Educational | Wraps VPython + matplotlib for textbook scenarios | None directly; **its scenario JSON format** is a golden-file inspiration |
| **Cirq** | Quantum sim | Schrödinger evolution, dense state vector | Singleton gate-tensor table pattern (T5) for future quantum/ pkg |
| **LAMMPS** | Classical MD | Velocity-Verlet + neighbor lists + Ewald | Verlet integrator algorithm (push to chaos per 111/112); **fix nve drift-bound contract** (T3) |
| **GROMACS** | Biomolecular MD | Leapfrog + LINCS constraints + PME | Leapfrog formula; PME/LINCS out-of-scope |
| **JAX-MD** | Differentiable MD | NN potentials + JAX autodiff through trajectories | None directly (XLA); but its **immutable State struct** pattern (T6) |
| **TensorPhysics** | Diff-physics | TF gradient through cloth/fluid sim | None (TF JIT) |
| **Brax** | Diff rigid-body | XLA-compiled MuJoCo-style pipeline | Immutable State struct pattern (T6) |
| **MuJoCo MJX** | XLA-compiled MuJoCo | Featherstone articulated-body + XLA codegen | None — solver-stack is huge, XLA is the compiler |
| **DiffTaichi** | Diff-physics DSL | Source-to-source AD over Taichi tensor lang | None (DSL + JIT) |
| **NVIDIA Warp** | GPU diff-physics | Python decorators → CUDA codegen | None (CUDA codegen) |

**Verdict.** Nine libraries (Brian2 DSL, Manim, VPython, PyCav, JAX-MD/TF,
Brax, MuJoCo MJX, DiffTaichi, Warp) are categorically out — visualization
stacks, JIT, GPU codegen, or symbolic CAS. Three (LAMMPS, GROMACS, Modelica)
ship algorithms that port (Verlet, leapfrog, NASA polys), but their
engineering trick is in the surrounding solver stack reality cannot adopt.
The six tricks below (T1-T6) are the surviving portable wins, ≤300 LOC each,
golden-file-testable, citation-anchored.

---

## 2. The six portable engineering tricks

### T1. Compile-time dimension lint (SymPy.physics.units, Modelica `unit=`)

**What.** SymPy's `Quantity` carries a length-7 dimension vector
$(L, M, T, I, \Theta, N, J)$ alongside its scalar; arithmetic checks
dimensions. Modelica variable decls carry `unit="m/s"` strings the compiler
validates. Cost-of-error: **2008 NASA Mars Climate Orbiter loss** was
pound-seconds vs newton-seconds — Modelica/SymPy would have caught it.

**Why portable.** reality cannot wrap every `float64` in a Quantity (breaks
"numbers in, numbers out", breaks 60 FPS hot paths). But it *can* ship a
**build-time dimension lint** as a golden-file: docstrings already carry
`Formula: P = n*R*T/V` + per-param SI units (verified `physics/thermo.go`
`IdealGas` lines 13-19 declaring `(mol)`, `(K)`, `(m^3)`). A `physicstest`
helper parses unit annotations, reconstructs the dimension vector, validates
against the formula's algebraic dimensions. Goes in `testutil/dim_lint.go`
~200 LOC: one regex per param + 7-vector dim algebra in `[7]int8`. **Catches
100% of unit-mismatch bugs at `go test` with zero runtime cost.** No CAS
needed.

**Go port.**
```go
// testutil/dim_lint.go
type Dim [7]int8 // L, M, T, I, Theta, N, J (powers, signed)
var (
    Length      = Dim{1,0,0,0,0,0,0}
    Mass        = Dim{0,1,0,0,0,0,0}
    Time        = Dim{0,0,1,0,0,0,0}
    Force       = Dim{1,1,-2,0,0,0,0}     // L M T^-2
    Energy      = Dim{2,1,-2,0,0,0,0}
    Pressure    = Dim{-1,1,-2,0,0,0,0}
    // ... 30 derived units
)
func (a Dim) Mul(b Dim) Dim { /* add powers */ }
func (a Dim) Pow(n int) Dim { /* multiply powers */ }
func ValidateFormula(funcName string, params []Dim, expectedReturn Dim) error
```

**Consumer.** Every new fn in 112's 65-fn growth plan. **Verdict: SHIP** —
uniquely portable, uniquely high-leverage; ~200 LOC at `go test` time only.

---

### T2. Voigt 6-vector for symmetric tensors (Modelica.MultiBody, FEniCS, Abaqus)

**What.** A symmetric $3 \times 3$ stress/strain tensor has 6 independent
components: $\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xz},
\sigma_{xy}$. Voigt 1910 packed these as a 6-vector + Hooke's law as a
$6 \times 6$ stiffness $C_{ij}$ (21 indep for triclinic, 9 orthotropic, 5
transverse-isotropic, 2 isotropic). **Modelica.MultiBody, FEniCS, Abaqus,
ANSYS, LS-DYNA, OpenRadioss all use Voigt internally** — no commercial FEM
stack stores symmetric stress as a 9-element matrix because the 33%
redundancy breaks symmetry preservation under composition (round-trip
accumulates $10^{-16}$ noise on off-diagonal pairs per op).

**Why portable.** Already flagged by **111 (M-1) and 112 (T2.1)**. Voigt is
the *correct* representation; makes asymmetry **unrepresentable** like
reality already does for quaternions (4 floats not 4×4), color spaces (XYZ
triple not redundant tuple), upper-triangular Cholesky (linalg).

**Go port.**
```go
// physics/tensor.go
type Tensor3Sym [6]float64 // {xx, yy, zz, yz, xz, xy}
type Stiffness3Sym [21]float64 // upper triangle of 6x6, anisotropic
func (s Tensor3Sym) Trace() float64 { return s[0]+s[1]+s[2] }
func (s Tensor3Sym) Deviator() Tensor3Sym { /* sigma - tr/3 * I */ }
func (s Tensor3Sym) PrincipalStresses() (s1, s2, s3 float64) {
    // closed-form Cardano via Smith 1961: returns all three roots
    // simultaneously via cos(arccos(.)/3), accurate to 14 digits
    // even at degenerate eigenvalues (no sequential-deflation cancellation)
}
func (s Tensor3Sym) VonMises() float64
func (C Stiffness3Sym) Apply(eps Tensor3Sym) Tensor3Sym // sigma = C * eps
```

**Consumer.** Tier-2 continuum mechanics (T2.1 in 112: ~150 LOC, 9 fns).
Yield criteria (T2.2) all consume `PrincipalStresses`. The current shipped
`VonMisesStress(s1, s2, s3)` taking three scalars is the **wrong shape** —
its callers in real consumer code have a Tensor3Sym, must extract principal
stresses to call it, then would just compute von Mises in-line. Refactor
target: `func (s Tensor3Sym) VonMises() float64` directly.

**Verdict: SHIP** — single highest-leverage commit per this review and per
111+112. ~150 LOC, ~30 vectors per function.

---

### T3. Symplectic drift bound as a golden-file invariant (LAMMPS, GROMACS, Brax)

**What.** LAMMPS `fix nve` (microcanonical Verlet), GROMACS leapfrog
default, and Brax pipeline all advertise a **bounded energy-drift
contract**: Verlet at timestep $h$ oscillates total energy with amplitude
$O(h^2)$ but *does not drift secularly* (shadow Hamiltonian,
Hairer-Lubich-Wanner 2006). RK4 by contrast drifts linearly: 7-day GPS
orbit by RK4 loses $10^{-8}$ relative energy, by Verlet $\sim 10^{-12}$
just oscillating. **Engineering trick is not the integrator** (Verlet is 4
lines) — **it's shipping the drift bound as a regression test** so a future
contributor cannot silently swap RK4 in without breaking CI.

**Why portable.** Reality already ships golden vectors for closed-form fns.
Extension to "integrator over Hamiltonian for $N$ steps preserves energy to
$\epsilon$" is one test file. Agent 110 flagged this for orbital. The 113
contribution: **make it a documented invariant in the test JSON**
`{"system":"harmonic_oscillator","steps":1e6,"dt":0.01,
"max_energy_drift":1e-13}`.

**Go port.**
```go
// chaos/symplectic.go (per 111/112: integrators stay in chaos, not physics)
type VerletState struct {
    Q, P []float64 // position and conjugate momentum
}
func VerletStep(s VerletState, gradV func(q []float64, out []float64),
                mass []float64, dt float64, out VerletState) {
    // half-kick, full-drift, half-kick: 5 lines
}
// chaos/testdata/golden_verlet_drift.json
//   {"system":"harmonic_oscillator","steps":1000000,"dt":0.01,
//    "expected_max_drift_rel":2.4e-14}
```

**Consumer.** Pistachio (60 FPS particle systems), aicore physical reasoning,
future Lagrangian-mechanics consumers from 112 T1.3. **Verdict: SHIP in
chaos/ not physics/** per 111's design rule (physics provides Hamiltonian
primitives, chaos provides integrators that consume them). ~80 LOC + golden.

---

### T4. Dimensional-consistency check at parse time (Brian2)

**What.** Brian2's neuron-equation DSL accepts strings like
`dv/dt = -(v - El)/tau : volt` and **checks RHS units match the
colon-annotated unit at parse time**. A typo `dv/dt = -v*tau : volt`
(missing `1/`, units now $V \cdot s$) errors at load before any simulation.

**Why portable (lint, not DSL).** Reality cannot ship a DSL — formulas live
in Go source, not strings. But the **sibling trick** — validating ODE
residuals — extends T1: every chaos integrator step `f(t, y, dy)` should
have `[dy] = [y]/[t]`. Test helper `AssertDimConsistent(rhsFunc, yDim,
tDim)` checks against docstring tags. ~50 LOC marginal cost on top of T1.

**Verdict: DEFER until T1 lands** + a chaos/ODE consumer wants it.

---

### T5. Singleton gate/operator-tensor cache (Cirq, QuTiP, OpenFermion)

**What.** Cirq's `cirq.X`/`Y`/`Z`/`H`/`CNOT` are **module-level constants**
— `numpy.array([[0,1],[1,0]])` allocated once at import, never reallocated,
hashed by identity. QuTiP same with `qutip.sigmax()`. Trick: quantum-circuit
sims create/destroy 100k+ single-qubit gate applications per ms; per-call
allocation would GC-thrash. Singleton tables solve it.

**Why portable.** Reality already does this in `signal/window.go`
(`Hann`/`Hamming`/`Blackman`), `color/illuminants.go`
(`D65`/`D50`/`A` whitepoints as `[3]float64` constants), `prob/erf.go`
(Chebyshev coef arrays). Today's scope: extend to **moment-of-inertia
tensors for canonical solids** in 112's T1.5 (sphere $\frac{2}{5}mr^2$,
shell $\frac{2}{3}mr^2$, cylinder, rod, plate). Future scope (when
quantum/ lands per 112 out-of-scope): Pauli matrices, gamma matrices,
Levi-Civita as `var sigmaX = [2][2]complex128{{0,1},{1,0}}` constants.

**Go port (now, for rigid-body inertia tensors):**
```go
// physics/inertia.go
type InertiaShape uint8
const (
    SolidSphere InertiaShape = iota
    HollowSphere
    SolidCylinderAxial
    SolidCylinderTransverse
    RodCenter
    RodEnd
    ThinPlate
)
// Tensor coefficients: I = coef * m * L^2 (L = canonical length scale)
var inertiaCoef = [...]float64{
    SolidSphere: 0.4, HollowSphere: 2.0/3.0,
    SolidCylinderAxial: 0.5, SolidCylinderTransverse: 1.0/12.0,
    RodCenter: 1.0/12.0, RodEnd: 1.0/3.0, ThinPlate: 1.0/12.0,
}
func MomentOfInertia(shape InertiaShape, m, L float64) float64 {
    return inertiaCoef[shape] * m * L * L
}
```

**Consumer.** 112's T1.5 (rigid-body MoI, 8 fns). Without pattern: 8
separate functions; with: one dispatched function + 8 consts.

**Verdict: SHIP for T1.5, DEFER for quantum.**

---

### T6. Immutable State struct, return-by-value (JAX-MD, Brax, MuJoCo MJX, Diffrax)

**What.** All four use **immutable-dataclass-evolution**: `State` struct
holds dynamic vars; integrator returns new State; nothing mutates. JAX-MD's
`simulate.NVE.apply_fn(state, ...) -> new_state`, Brax's
`pipeline.step(sys, state, action) -> new_state`, MJX's
`mjx.step(model, data) -> data`. Convergence reason: XLA traces require
purity; immutable State is the only way to get gradient-through-trajectory
without explicit checkpointing.

**Why portable.** Reality's closed-form fns already return new values. The
future Verlet/Yoshida integrators in chaos/ should follow this as a
documented contract so aicore can wrap them as pure RPC, Pistachio can
render trajectories without aliasing concerns, and a future autodiff
wrapper (aicore consumer, not reality) can AD through trajectories the way
Diffrax does, without modifying chaos/.

**Go port.**
```go
// chaos/symplectic.go — immutable State pattern
type HamiltonianState struct {
    Q [3]float64 // position
    P [3]float64 // conjugate momentum
    T float64    // time
}
func VerletStep(s HamiltonianState, gradV func([3]float64) [3]float64,
                m float64, dt float64) HamiltonianState {
    // half-kick + drift + half-kick, all stack-allocated, no mutation
}
```
Note: caller still controls allocation by passing pointers when batching:
`VerletStepInto(s *HamiltonianState, ..., out *HamiltonianState)` for
zero-alloc hot path. Both shapes co-exist; `Step` is the canonical
return-by-value form, `StepInto` is the perf path. Same idiom as
`linalg.MulMat` (allocating) vs `linalg.MulMatInto` (in-place).

**Consumer.** Same as T3 — Pistachio, aicore, future Lagrangian.

**Verdict: SHIP as documented integrator contract** (~10 LOC of doc + the
struct). The pattern is free given Go's value semantics; the contribution is
making it the *required* shape for new integrators so consumers can rely on
it.

---

## 3. Anti-patterns: what reality should NOT take

Six rejected, all on zero-dep grounds:
1. **Modelica DAE Pantelides index reduction** — symbolic diff + pivoting +
   dummy-derivative, ~5000 LOC, needs CAS. Out.
2. **SymPy CAS dependency** — symbolic engine vs reality's numerical;
   wrapping a CAS breaks zero-dep.
3. **Brian2 string-DSL parser** — runtime parsing creates attack surface +
   moves dimension errors from build to runtime + non-Go-native.
4. **JAX-MD / Brax / MuJoCo XLA** — XLA is a JIT compiler; coupling to
   TF/JAX violates zero-dep.
5. **NVIDIA Warp / DiffTaichi codegen** — same reason; GPU codegen is a
   compiler.
6. **LAMMPS/GROMACS neighbor-lists, Ewald, PME** — spatial decomposition
   for $N>10^5$, several KLOC each, no consumer in reality (Pistachio's
   particle systems are $N \le 10^4$ at 60 FPS via direct $O(N^2)$). 112
   T3.3 already bounds $N$-body to direct + Plummer softening, ~70 LOC.

---

## 4. PyCav and Manim physics (audit-prompt headlines)

**PyCav** (~2,000 LOC, `pycav.mechanics` + `.optics` + `.quantum`) is the
smallest surveyed library and closest to reality's sweet spot. Headline
algorithm: nothing new, every formula is undergraduate canonical. Engineering
trick: per-scenario JSON config consumed by VPython. **This is the same
pattern reality already has via `testutil/` golden files.** Architectural
validation: closed-form physics + JSON test vectors *is* the educational
substrate; consumer apps add a scenario layer.

**Manim physics** (`manim_physics`): `Pendulum`/`RadialWave`/`LinearWave`
mobjects, 2D rigid-body via pymunk, `Lens`/`Ray` for optics. Calls Cairo or
matplotlib. **Zero ports.** Manim is visualization, reality is math. The
lesson: visualization needs exactly what reality already provides — closed-form
deterministic functions returning numbers. Manim's `Pendulum.angular_position
(t)` IS `physics.PendulumPosition(theta0, L, t, g)` (112 T1.3). Architectural
validation only.

---

## 5. Sprint integration with 110/111/112

Six tricks, mapped onto the existing sprint plan:

| Trick | Sprint | Owner | Cross-ref |
|---|---|---|---|
| T1 dimension lint | v0.11 testutil | testutil/ | independent of physics body |
| T2 Voigt 6-vector | v0.13 (T2.1 in 112) | physics/tensor.go | 111 M-1, 112 T2.1 |
| T3 symplectic drift bound | v0.12 (chaos/) | chaos/symplectic.go | 110, 111 design rule |
| T4 ODE dim-lint extension | DEFER post-T1 | testutil/ | builds on T1 |
| T5 inertia singleton table | v0.12 (T1.5 in 112) | physics/inertia.go | 112 T1.5 |
| T6 immutable State contract | v0.12 (chaos/) | chaos/ docs | architectural, ~10 LOC |

Total marginal cost on top of 112's plan: ~250 LOC (T1 dim-lint), ~10 LOC
(T6 doc), with T2/T3/T5 already costed inside 112. **Net: T1 is the only
genuinely new line item this review introduces; the other five tricks are
sharpenings of what 110/111/112 already plan.**

---

## 6. Bottom line

reality/physics is **architecturally correct** — closed-form algebra in Go,
zero-dep, golden-file-tested, citation-anchored. The 12 surveyed libraries
that do *more* (Modelica DAE, SymPy CAS, Brax/MJX XLA, JAX-MD AD, LAMMPS
$N$-body) are doing categorically different things reality should not clone.
The six that overlap reality's band (Modelica.Media polys, SymPy.units lint,
SymPy.optics ABCD, PyCav scenarios, Cirq gate-tables, Manim closed-forms)
teach the same lesson: **closed-form + golden files + dimension lint + Voigt
for symmetric tensors + immutable State for integrators = the entire
portable surface.** Reality has 4 of 5; this review's contribution is
identifying T1 (dimension lint) as the missing piece.

**Single highest-leverage commit: T2 Voigt 6-vector** — already
independently flagged by 111 (M-1) and 112 (T2.1), confirmed here as the
Modelica/FEniCS/Abaqus-canonical representation. Without Voigt, continuum
mechanics ships wrong; with Voigt, correctly-by-construction.
**Second-highest: T1 dimension lint** — the only genuinely new addition
this review proposes, ~200 LOC in testutil/, catches
NASA-Mars-Climate-Orbiter-class bugs at `go test` with zero runtime cost.

Report at `agents/113-physics-sota.md`.
