# 204 | new-symplectic-int

**Summary (2 lines):** reality v0.10.0 ships ZERO symplectic integrators — `chaos/ode.go` exposes `RK4Step` + `EulerStep` + `SolveODE` only, no Verlet/leapfrog/Yoshida/Forest-Ruth/McLachlan/Strang/variational/RATTLE/SHAKE; the lone "Hamiltonian" string in the source tree is a *closed-form* H docstring on `LotkaVolterra` at `chaos/systems.go:64` and a `TestLotkaVolterra_HamiltonianConserved` at `chaos/chaos_test.go:316` whose RK4-with-1e-2-tolerance-over-50000-steps actually *measures RK4 energy drift* and calls it conservation (correctly identified by agent 026-N2 and agent 168). This slot scopes the splitting/composition/variational canon — fourteen primitives I1-I14 totalling ~1900 LOC, with `chaos/symplectic.go` (Verlet/leapfrog/symplectic-Euler ~140 LOC) as the keystone because Yoshida 4/6/8 and Forest-Ruth and McLachlan-Atela are all *coefficient triples wrapped around the same drift-kick-drift core* — once that core ships, six more orders of accuracy follow at ~30 LOC each. Disambiguation versus 168 (synergy-physics-autodiff): 168 lists `chaos/symplectic.go` + `chaos/yoshida.go` + `chaos/variational.go` as connective tissue for HNN/LNN/adjoint-ODE; this slot is the *implementation-detail companion* — the splitting calculus, BCH-derived composition coefficients, modified-Hamiltonian / backward-error analysis, energy-drift bounds, and constrained-system projections (RATTLE/SHAKE) that 168 deferred. Same files, deeper specification.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Verified by direct read.

### `chaos/ode.go` (131 LOC)
- `RK4Step(f, t, y, dt, out)` — classic 4-stage RK4. Allocates 5 slices per call (`k1..k4` + `tmp`). Non-symplectic: phase-space volume contracts at O(dt^5) per step.
- `EulerStep(f, t, y, dt, out)` — explicit forward Euler. Allocates 1 slice per call.
- `SolveODE(f, y0, t0, tEnd, dt) [][]float64` — RK4 trajectory recorder.
- RHS signature: `func(t float64, y []float64, dydt []float64)` — single state vector, no `(q, p)` split.

### `chaos/systems.go` (172 LOC)
- Lorenz, Rössler, Lotka-Volterra, SIR, Van der Pol, LogisticMap, GameOfLife. None ship as a separated `(T(p), V(q))` pair. Lotka-Volterra docstring at line 64 names `H = δx − γln x + βy − αln y` but no Hamiltonian-form RHS exists.

### `chaos/analysis.go` (158 LOC)
- LyapunovExponent (1D maps only), BifurcationDiagram, RecurrencePlot. No symplectic, no canonical map, no Poincaré section, no modified-Hamiltonian estimator.

### `physics/` (598 LOC)
- mechanics + materials + thermo + optics. **ZERO** Lagrangian/Hamiltonian/Noether/`(q, p)` callable surface. Confirms agent 168 §0 and agent 112 §1.3.

### `orbital/`
- Closed-form Kepler/vis-viva/Hohmann/Hill. No long-time-integration N-body code, no Wisdom-Holman, no IAS15, no Encke. Restricted three-body integration deferred (agent 027 T2.S6).

### Repo-wide grep
```
$ grep -rn "symplectic\|Verlet\|leapfrog\|Yoshida\|Forest-Ruth\|McLachlan\|Stormer\|Störmer\|Strang\|Trotter\|RATTLE\|SHAKE\|RKMK\|Munthe-Kaas\|variational integrator" --include="*.go"
chaos/chaos_test.go:316:func TestLotkaVolterra_HamiltonianConserved   # RK4 — wrong test, agent 026-N2
chaos/systems.go:64:                                                    # docstring only
audio/segmentation/min_silence.go                                       # unrelated
optim/proximal/doc.go                                                   # unrelated
```
Zero hits in source code. Confirms agent 026 N2, agent 027 §3.3, agent 168 §0.

---

## 1. The fourteen-primitive scope

Numbered I1-I14. For each: (a) what reality ships, (b) what to add, (c) connective LOC.

### I1 — Symplectic Euler (drift-kick / kick-drift)

(a) **Ships:** nothing. Only forward Euler, which is *non-*symplectic.
(b) **Add:** the two first-order canonical maps, derived from generating functions of type F1 and F2:

```
SymplecticEulerA (drift-kick):                   SymplecticEulerB (kick-drift):
  q_{n+1} = q_n + dt · ∂H/∂p (q_n, p_n)            p_{n+1} = p_n − dt · ∂H/∂q (q_n, p_n)
  p_{n+1} = p_n − dt · ∂H/∂q (q_{n+1}, p_n)        q_{n+1} = q_n + dt · ∂H/∂p (q_n, p_{n+1})
```

Order 1 in `dt`, exactly area-preserving for separable H = T(p) + V(q). Reference: Hairer-Lubich-Wanner 2006 §VI.3.

(c) **LOC:** ~80 in `chaos/symplectic.go` exposing `SymplecticEulerA(gradH, q, p, dt, qOut, pOut)` and B-variant. RHS signature must change from `(t, y, dydt)` to `(q, p, dHdq, dHdp)` (separable case) or `(q, p, force, velocity)` for general H. Add a thin adapter wrapping existing `chaos.LotkaVolterra` etc. at the test boundary.

### I2 — Velocity Verlet / Position Verlet / Leapfrog (Störmer-Verlet 1791/1907)

(a) **Ships:** nothing.
(b) **Add:** the three equivalent second-order schemes for separable H = ½p^T M⁻¹ p + V(q) ≡ T + V:

```
VelocityVerlet (positions and velocities at integer t):
  v_{n+½} = v_n − (dt/2) · M⁻¹ ∇V(q_n)
  q_{n+1} = q_n + dt · v_{n+½}
  v_{n+1} = v_{n+½} − (dt/2) · M⁻¹ ∇V(q_{n+1})
```

```
LeapfrogStep (positions integer, velocities half-integer; minimal force evaluations):
  v_{n+½} = v_{n−½} + dt · a_n
  q_{n+1} = q_n + dt · v_{n+½}
```

Order 2, time-reversible, symplectic, conserves a *modified* Hamiltonian H̃ = H + dt^2·H_2 + dt^4·H_4 + ... (BCH series). Energy drift bounded over exponentially long times t ≤ exp(c/dt) by backward-error analysis (Hairer-Lubich-Wanner 2006 §IX.7 Theorem 7.1). Equivalent to one Strang splitting `e^{(dt/2)L_T} e^{dt L_V} e^{(dt/2)L_T}`.

(c) **LOC:** ~140 in `chaos/symplectic.go`. Exposes `VelocityVerletStep`, `PositionVerletStep`, `LeapfrogStep`. Out-buffer convention; one force evaluation per step (cache the half-step force); zero extra allocations on the hot path. **This is the keystone** — every higher-order scheme (I3-I7) wraps this in a coefficient loop.

### I3 — Forest-Ruth 4th-order symplectic (Forest-Ruth 1990, Candy-Rozmus 1991)

(a) **Ships:** nothing.
(b) **Add:** triple-Verlet composition with the unique symmetric coefficient set canceling 3rd-order error:

```
θ = 1 / (2 − 2^{1/3})
FR4 = Verlet(θ·dt) ∘ Verlet((1−2θ)·dt) ∘ Verlet(θ·dt)
```

Order 4, three force evaluations per step, symmetric. Reference: Forest-Ruth 1990 *Physica D* 43:105-117; Candy-Rozmus 1991 *J. Comp. Phys.* 92:230-256.

(c) **LOC:** ~30 in `chaos/symplectic.go`. Pure wrapper over `VelocityVerletStep`.

### I4 — Yoshida composition: 4th, 6th, 8th-order (Yoshida 1990)

(a) **Ships:** nothing.
(b) **Add:** Yoshida's symmetric-composition theorem — given a 2k-th-order symmetric base method S_2k, build S_{2k+2} from three S_2k applications with weights:

```
w₁ = w₃ = 1 / (2 − 2^{1/(2k+1)})
w₂ = 1 − 2·w₁                                    (ensures w₁ + w₂ + w₃ = 1, sum of squares cancels next order)
S_{2k+2}(dt) = S_2k(w₁·dt) ∘ S_2k(w₂·dt) ∘ S_2k(w₁·dt)
```

Iterate: S₂ (Verlet) → S₄ (≡ Forest-Ruth) → S₆ (7 Verlet calls) → S₈ (15 Verlet calls). Better S₆/S₈ coefficients exist via 7-term and 15-term symmetric compositions (Yoshida 1990 Table 1, Suzuki 1992) — closer-to-optimal phase-space-volume preservation than the Yoshida-triple recursion. Reference: Yoshida 1990 *Phys. Lett. A* 150:262-268.

(c) **LOC:** ~80 in `chaos/yoshida.go`. Ship S₄ (3 base calls), S₆-Yoshida-Table-1 (7 base calls, hand-tabulated coefficients for ~10× smaller error than recursive Yoshida-S₆), S₈-Suzuki-fractal (15 base calls). All three as coefficient slices over a shared `compositionStep(base, coefficients, q, p, dt)` driver.

### I5 — McLachlan-Atela 4th-order (McLachlan-Atela 1992)

(a) **Ships:** nothing.
(b) **Add:** the optimised 4-stage 4th-order *non*-symmetric symplectic integrator:

```
a = [0.5153528374311229, −0.085782019412973646, 0.4415830236164665, 0.1288461583653842]
b = [0.1344961992774310, −0.2248198030794208,    0.7563200005156683, 0.3340036032863214]
KDK pattern: for i in 0..3: drift(a[i]·dt); kick(b[i]·dt)
```

Order 4 with strictly smaller leading-error coefficient than Forest-Ruth (≈ 5× smaller per Blanes-Casas-Murua 2008 benchmarks). Same number of force evaluations (4) as Forest-Ruth's full 3-Verlet (which is 4 evaluations after caching shared half-steps). Reference: McLachlan-Atela 1992 *Nonlinearity* 5:541-562.

(c) **LOC:** ~50 in `chaos/symplectic.go`. Coefficient table + KDK driver shared with I4.

### I6 — General Lie-Trotter and Strang splitting (operator splitting framework)

(a) **Ships:** nothing.
(b) **Add:** the generic exponential splitting machinery underlying I1-I5. For H = A + B with non-commuting flows φ_A, φ_B:

```
Lie-Trotter (order 1):    e^{ε(A+B)} ≈ e^{εA} · e^{εB}
Strang (order 2):         e^{ε(A+B)} ≈ e^{(ε/2)A} · e^{εB} · e^{(ε/2)A}
```

Local error from BCH: e^{εA}·e^{εB} = e^{ε(A+B) + (ε^2/2)[A,B] + O(ε^3)}; Lie-Trotter has ε² leading error, Strang has ε³ (the [A,B] term cancels by symmetry). Reference: Trotter 1959; Strang 1968; McLachlan-Quispel 2002 *Acta Numerica*.

(c) **LOC:** ~60 in `chaos/splitting.go`. API: `LieTrotterStep(flowA, flowB, q, p, dt, qOut, pOut)`, `StrangStep(...)`. `flowA`/`flowB` accept a `func(q, p, dt, qOut, pOut)` signature — Verlet etc. become *applications* of this generic driver with `flowA = drift`, `flowB = kick`.

### I7 — Symmetric composition methods (Suzuki fractal, BABAB...)

(a) **Ships:** nothing.
(b) **Add:** Suzuki's fractal composition (Suzuki 1991 *J. Math. Phys.* 32:400-407):

```
Suzuki4(dt) = Verlet(s·dt)^2 · Verlet((1−4s)·dt) · Verlet(s·dt)^2,    s = 1/(4 − 4^{1/3})
```

5-stage symmetric, order 4, non-trivial improvement over Yoshida-S₄ in phase-space-volume preservation for stiff oscillators. Generalised BABAB...BAB compositions: Blanes-Casas-Ros 1999 *SIAM J. Numer. Anal.* 36 with optimal coefficients for (m, p) = (number-of-stages, order) up to (15, 8). Reference: Suzuki 1991; Blanes-Casas-Ros 1999, 2002.

(c) **LOC:** ~40 in `chaos/yoshida.go`. Append to I4 driver as another coefficient table; Blanes-Casas-Ros optimal sets are pre-tabulated to 16 digits.

### I8 — Implicit symplectic Runge-Kutta (Gauss-Legendre)

(a) **Ships:** nothing. RK4 is non-symplectic.
(b) **Add:** s-stage Gauss-Legendre collocation IRK is symplectic for *any* Hamiltonian (separable or not). Order 2s; A-stable. Standard schemes:

```
GL2 (1-stage, order 2): implicit midpoint                      Y_1 = y_n + (dt/2)·f(t_n + dt/2, Y_1)
                                                                y_{n+1} = y_n + dt·f(t_n + dt/2, Y_1)

GL4 (2-stage, order 4): Butcher tableau with c_i = ½ ± √3/6
GL6 (3-stage, order 6): Butcher tableau with Gauss-Legendre nodes on [0,1]
```

Each step solves a small nonlinear system (Newton or fixed-point on the stages). Implicit cost; required for non-separable H or stiff systems where explicit splitting fails. Reference: Sanz-Serna-Calvo 1994 §4.1; Hairer-Lubich-Wanner 2006 §VI.4.

(c) **LOC:** ~250 in `chaos/sirk.go`. Need a small Newton solver — `optim/rootfind.go` already exists per agent 102. Connective glue ~30 LOC for the s × n stage block.

### I9 — Variational integrators (Marsden-West 2001)

(a) **Ships:** nothing.
(b) **Add:** discrete-Lagrangian-derived integrators. Given L(q, q̇), define a *discrete Lagrangian* L_d(q_n, q_{n+1}) (e.g. midpoint rule L_d = dt·L((q_n + q_{n+1})/2, (q_{n+1} − q_n)/dt)); the discrete Euler-Lagrange equation D₁L_d(q_n, q_{n+1}) + D₂L_d(q_{n−1}, q_n) = 0 gives the integrator. Symplectic by construction *and* preserves the discrete momentum map (so Noether's theorem holds at the discrete level — angular momentum, linear momentum exactly conserved). Reference: Marsden-West 2001 *Acta Numerica*; Lew-Marsden-Ortiz-West 2003.

(c) **LOC:** ~200 in `chaos/variational.go`. Expose `MidpointVariationalStep`, `VerletAsVariational` (Verlet IS the L_d = dt·L((q_n+q_{n+1})/2, (q_{n+1}−q_n)/dt) variational integrator — round-trip identity). Forward-mode autodiff on L_d via dual numbers (agent 168 §A) makes this self-deriving, but a manual finite-difference fallback is fine for v1. Variational integrators *generalise* Verlet/leapfrog/Newmark and unify them with constrained mechanics (I12).

### I10 — Energy drift bound / modified-Hamiltonian / backward-error analysis

(a) **Ships:** nothing — `TestLotkaVolterra_HamiltonianConserved` measures RK4 drift but never quantifies the bound.
(b) **Add:** the cross-language-parity test contract for symplectic methods. Theorem (Hairer-Lubich-Wanner 2006 §IX.7): for an analytic Hamiltonian, the modified Hamiltonian H̃ = H + dt^p H_p + dt^{p+1} H_{p+1} + ... converges asymptotically; truncated H̃_N is conserved by the discrete map up to errors O(exp(−γ/dt)·N), so |H(q_n, p_n) − H(q_0, p_0)| ≤ C·dt^p over t ≤ T·exp(c/dt). Test: log-log slope of energy-drift-vs-dt is −p (the integrator's order) for separable H over one bounded oscillation period.

(c) **LOC:** ~120 in `chaos/symplectic_test.go`. Three golden tests: (i) `TestVerletEnergyDrift_HarmonicOscillator` slope 2.0 ± 0.05; (ii) `TestForestRuth4EnergyDrift_HarmonicOscillator` slope 4.0 ± 0.05; (iii) `TestYoshidaS6EnergyDrift_PendulumNonlinear` slope 6.0 ± 0.1. Cross-language parity: each language re-runs and validates the same slope from a shared 30-sample dt × N_steps grid in `chaos/testdata/symplectic_drift.json`. Replaces the misleading `TestLotkaVolterra_HamiltonianConserved` (file: `chaos/chaos_test.go:316`).

### I11 — Map iteration for Poincaré sections

(a) **Ships:** nothing — `analysis.go` has BifurcationDiagram and RecurrencePlot but no Poincaré section.
(b) **Add:** event-detection layer over a symplectic stepper. Cross of an n-D flow with a hyperplane Σ = {y : g(y) = 0} via Hénon's trick (Hénon 1982): when g changes sign, swap independent variable from t to g and integrate one step in g to reach Σ exactly. Returns iterated 2D area-preserving map suitable for chaos taxonomy. Reference: Hénon 1982 *Physica D* 5:412.

(c) **LOC:** ~150 in `chaos/poincare.go`. API: `PoincareSection(stepper, rhs, sectionFn, gradSectionFn, y0, nCrossings, dtMax) [][]float64`. Pairs with agent 027 T1.S2 (Standard map / Chirikov) and T1.S11 (Hénon-Heiles) and T2.S6 (restricted three-body) for cross-validation against the Hamiltonian-flow form.

### I12 — Symplectic for constrained systems: RATTLE, SHAKE

(a) **Ships:** nothing — no constraint mechanism in the package.
(b) **Add:** SHAKE (Ryckaert-Ciccotti-Berendsen 1977) and RATTLE (Andersen 1983) — Verlet-equivalent integrators that preserve holonomic constraints g(q) = 0 by Lagrange-multiplier projection. RATTLE additionally constrains the velocity to satisfy ∇g·v = 0 and is *symplectic* on the constraint manifold (SHAKE is not, in general). Algorithm:

```
RATTLE step (constraint g(q) = 0, multiplier λ):
  1. v_{n+½} = v_n − (dt/2)·∇V(q_n) + λ_q·∇g(q_n)
  2. q_{n+1} = q_n + dt·v_{n+½}
  3. solve g(q_{n+1}) = 0 for λ_q (Newton or SOR-style cycle over constraints)
  4. v_{n+1} = v_{n+½} − (dt/2)·∇V(q_{n+1}) + λ_v·∇g(q_{n+1})
  5. solve ∇g(q_{n+1})·v_{n+1} = 0 for λ_v (linear solve)
```

Reference: Andersen 1983 *J. Comp. Phys.* 52:24-34; Leimkuhler-Skeel 1994; Hairer-Lubich-Wanner 2006 §VII.1.

(c) **LOC:** ~250 in `chaos/constrained.go`. Newton inner solve via `optim/rootfind.go`. Required for any rigid-body simulation, polymer chain, robotic manipulator (Pistachio-NPC-grade physics), or pendulum-on-circle / spherical-pendulum exact-constraint test.

### I13 — Geometric integrators on Lie groups (Munthe-Kaas, RKMK)

(a) **Ships:** nothing — no Lie-group machinery anywhere in `geometry/` (only quaternions, no exp/log map for SO(3) as a generic Lie-group operation; agent 074 §X confirms).
(b) **Add:** Munthe-Kaas 1998 / Crouch-Grossman 1993 RKMK methods. State y lives on a Lie group G (e.g. SO(3) for rigid-body attitude); dynamics ẏ = A(t, y)·y where A ∈ 𝔤 (Lie algebra). Step in 𝔤 (a vector space) with a classical RK, then exponentiate back: y_{n+1} = exp(σ_n)·y_n where σ_n is a 𝔤-valued RK update of the dexpinv-equation. Preserves G exactly. Reference: Munthe-Kaas 1998 *BIT* 38:92-111; Iserles-Munthe-Kaas-Nørsett-Zanna 2000 *Acta Numerica*.

(c) **LOC:** ~300 in `chaos/lie_integrator.go` + ~80 in `geometry/so3.go` (exp/log on SO(3) via Rodrigues, dexpinv via BCH truncation to 4th order). Lie-Poisson integrators for non-canonical Hamiltonian systems (Marsden-Ratiu 1999) follow at +100 LOC once SO(3) ships.

### I14 — Stochastic / multi-rate / fast-slow extensions

(a) **Ships:** nothing.
(b) **Add (defer-but-design):** (i) stochastic Verlet for Langevin dynamics (BAOAB scheme, Leimkuhler-Matthews 2013) — couples to slot 202 (new-sde); (ii) multi-rate splitting r-RESPA (Tuckerman-Berne-Martyna 1992) for fast-slow Hamiltonians H = T + V_fast + V_slow with one V_slow evaluation per N V_fast evaluations, giving N× speedup at controlled error; (iii) Wisdom-Holman 1991 mapping for Keplerian + perturbation splitting (the orbital-mechanics workhorse used by REBOUND, IAS15-class long-time solar-system integration). Defer all three to v2. Cross-link: 202 owns the Brownian increment substrate; this slot specifies the *symplectic*-side coupling (BAOAB has an unambiguous splitting structure).

(c) **LOC:** scoped, deferred. ~600 LOC if pulled.

---

## 2. Implementation-detail summary table

| ID | Primitive | Order | Force evals/step | LOC | File | Reference |
|----|-----------|-------|------------------|-----|------|-----------|
| I1 | Symplectic Euler A/B | 1 | 1 | 80 | chaos/symplectic.go | Hairer-Lubich-Wanner 2006 §VI.3 |
| I2 | Velocity/Position Verlet, Leapfrog | 2 | 1 (cached half-step) | 140 | chaos/symplectic.go ★ | Störmer 1907; Verlet 1967 |
| I3 | Forest-Ruth 4 | 4 | 3 | 30 | chaos/symplectic.go | Forest-Ruth 1990 |
| I4 | Yoshida S₄/S₆/S₈ | 4/6/8 | 3/7/15 | 80 | chaos/yoshida.go | Yoshida 1990 |
| I5 | McLachlan-Atela 4 | 4 | 4 | 50 | chaos/symplectic.go | McLachlan-Atela 1992 |
| I6 | Lie-Trotter / Strang splitting | 1/2 | flow-dependent | 60 | chaos/splitting.go | Trotter 1959; Strang 1968 |
| I7 | Suzuki fractal / BABAB compositions | 4-8 | 5-15 | 40 | chaos/yoshida.go | Suzuki 1991; Blanes-Casas-Ros 1999 |
| I8 | Implicit symplectic RK (Gauss-Legendre) | 2/4/6 | s implicit stages | 250 | chaos/sirk.go | Sanz-Serna-Calvo 1994 |
| I9 | Variational integrators | varies | varies | 200 | chaos/variational.go | Marsden-West 2001 |
| I10 | Energy-drift / modified-Hamiltonian tests | (test only) | n/a | 120 | chaos/symplectic_test.go | Hairer-Lubich-Wanner 2006 §IX.7 |
| I11 | Poincaré section / Hénon event-detection | (analysis) | 1 stepper call | 150 | chaos/poincare.go | Hénon 1982 |
| I12 | RATTLE / SHAKE constrained Verlet | 2 | 1 + Newton | 250 | chaos/constrained.go | Andersen 1983 |
| I13 | Munthe-Kaas / RKMK Lie-group | 4 | s (RK stages) + exp | 380 | chaos/lie_integrator.go + geometry/so3.go | Munthe-Kaas 1998 |
| I14 | Stochastic Verlet (BAOAB), r-RESPA, Wisdom-Holman | varies | varies | (deferred ~600) | (deferred) | Leimkuhler-Matthews 2013; Tuckerman 1992; Wisdom-Holman 1991 |
|    | **Total core (I1-I12)** | | | **~1450** | | |
|    | **Total core + I13** | | | **~1830** | | |

★ = keystone (everything else wraps the Verlet inner loop).

---

## 3. Tier ordering (ship sequence)

**Tier 1 (350 LOC, ship in 1 sprint, unlocks 168 + 027-T1.S11 + 027-T2.S6):**
1. I2 Velocity Verlet + Leapfrog (140 LOC). Keystone.
2. I1 Symplectic Euler A/B (80 LOC). Pedagogical pair / unit-test seed.
3. I3 Forest-Ruth 4 (30 LOC). One-step upgrade once I2 lands.
4. I10 Energy-drift slope tests (120 LOC). Replaces the misleading existing `TestLotkaVolterra_HamiltonianConserved`. Cross-language-parity contract.

**Tier 2 (370 LOC, ship in 2nd sprint, completes the splitting/composition canon):**
5. I4 Yoshida S₄/S₆/S₈ (80 LOC).
6. I5 McLachlan-Atela 4 (50 LOC).
7. I6 Lie-Trotter / Strang splitting framework (60 LOC). Refactor I1-I5 to call into this.
8. I7 Suzuki fractal + Blanes-Casas-Ros optimal compositions (40 LOC).
9. I11 Poincaré section / Hénon event-detection (150 LOC). Pairs with agent 027 T1.S2 standard map and T1.S11 Hénon-Heiles.

**Tier 3 (730 LOC, ship-when-consumer-pulls):**
10. I8 Gauss-Legendre implicit symplectic RK (250 LOC). Required for non-separable H or stiff Hamiltonians.
11. I9 Variational integrators (200 LOC). Pairs with agent 168 (Lagrangian-Neural-Networks via autodiff dual numbers).
12. I12 RATTLE / SHAKE constrained (250 LOC). Required for rigid-body / robotics consumers.

**Tier 4 (380 LOC, defer-but-design):**
13. I13 Munthe-Kaas / RKMK Lie-group (380 LOC, includes geometry/so3.go).

**Tier 5 (deferred):** I14 stochastic / multi-rate / Wisdom-Holman (600 LOC if pulled — cross-link 202).

---

## 4. Architectural recommendations

**A1. New RHS signature for Hamiltonian flows.** The existing `func(t, y, dydt)` packs (q, p) into one slice; symplectic steppers need them separated to evaluate ∂H/∂p (drift) and ∂H/∂q (kick) independently. Recommended split surface:

```go
// HamiltonianSeparable: H = T(p) + V(q). Most cases.
type HamiltonianSeparable struct {
    GradV func(q []float64, out []float64)            // ∇V(q) → out
    InvM  func(p []float64, out []float64)            // M⁻¹·p → out  (often identity, allow nil)
}

// HamiltonianGeneral: H(q, p) non-separable. For Tier-3 implicit methods.
type HamiltonianGeneral struct {
    DHDQ func(q, p, out []float64)
    DHDP func(q, p, out []float64)
}
```

Adapter `WrapAsRHS(h HamiltonianSeparable) func(t, y, dydt)` lets existing `chaos.SolveODE` consumers fall through unchanged — additive surface, no break.

**A2. Zero-allocation hot path mandate.** Keep CLAUDE.md rule 3. Verlet/leapfrog hot loops must reuse caller-provided buffers — pre-allocate kick/drift scratch in a `SymplecticWorkspace` struct passed to the stepper. Pistachio's 60-FPS particle loop will stress-test this.

**A3. Cross-language parity contract via energy-drift slope.** A symplectic integrator's *defining* property is the long-time energy bound, not single-step accuracy. The golden-file contract MUST be: "log-log slope of |H(t) − H(0)|_max vs dt over [0, T] is exactly the integrator order." This is the cross-language test that catches BCH-coefficient typos which short-time accuracy tests miss. Reference: Hairer-Lubich-Wanner 2006 §IX.7 Theorem 7.1.

**A4. Modified-Hamiltonian estimator as an explicit utility.** Ship `ModifiedHamiltonianEstimate(H, dt, scheme) func(q, p) float64` returning H̃ truncated at the appropriate order — this is what the integrator *actually* conserves and is the right quantity for a "did I detect chaos vs numerical drift" diagnostic.

**A5. Pair I9 (variational) with autodiff dual-numbers from agent 168.** A variational integrator self-derives from a discrete Lagrangian L_d via D₁L_d, D₂L_d. Forward-mode autodiff (autodiff/dual.go in 168 §A) gives both partials in one pass. This is the natural piling order: 168 ships dual numbers → 204 ships variational integrators on top of them, both in `chaos/variational.go`.

---

## 5. Risks / gotchas

**R1. Yoshida coefficients lose digits past order 8.** Recursive Yoshida-S₈ via three-S₆ composition has cancellation-limited accuracy past 12 digits; use Suzuki-1991 / Blanes-Casas-Ros-1999 hand-tabulated coefficients (15-stage S₈) for full double-precision parity. Hard-tabulate to 18 digits in source.

**R2. RATTLE Newton inner solve must converge or fail loud.** SHAKE's iteration can silently stall on degenerate constraint geometry (e.g. parallel constraint gradients); RATTLE's velocity-projection linear solve is exact but the position projection is nonlinear. Default iteration cap = 50, tolerance = 1e-12, return error on non-convergence.

**R3. Forest-Ruth has a NEGATIVE intermediate timestep coefficient (1 − 2θ ≈ −1.7) for separable H = T + V.** This is *correct* — it cancels the BCH commutator — but it surprises users (and breaks consumers that assume monotone time). Document loudly in the docstring; verify the modified-Hamiltonian sign convention.

**R4. Implicit symplectic RK Gauss-Legendre needs Newton tolerance tighter than the step error.** Loose stage-tolerance kills the symplectic property; recommend `stage_tol ≤ machine_eps · |y|` for double precision. Standard guidance: Sanz-Serna-Calvo 1994 §4.2.

**R5. Variational integrator auto-derivation from L_d via finite differences accumulates O(eps^{2/3}) Richardson-extrapolation noise.** Mandate: ship `MidpointVariationalStep` analytically (it's just Verlet), and gate I9 generic auto-derivation on the autodiff dual-numbers landing from agent 168.

**R6. Munthe-Kaas dexpinv truncation order must match the RK base method.** A 4th-order RKMK requires the dexpinv-equation expansion to *at least* 3rd order in the BCH commutator series; truncating at 2nd kills the symplectic property silently. Document the order-pairing constraint explicitly.

**R7. Constrained-system test cases must use exactly-on-manifold initial conditions.** RATTLE / SHAKE projection drift is bounded by the Lagrange-multiplier Newton tolerance; if the initial state is off-manifold by 1e-4, projection at step 0 introduces a non-symplectic kick that pollutes downstream tests. Initial-condition validator: `assert g(q0) < 1e-12 && grad(g, q0)·v0 < 1e-12` before step 0.

---

## 6. Cross-package coupling

| Edge | LOC | Purpose |
|------|-----|---------|
| chaos/symplectic.go → physics/hamiltonian.go (new, agent 168) | 0 (separate-file co-located) | Standard Hamiltonians: HarmonicOscillator, KeplerH, HenonHeiles, DoublePendulum |
| chaos/sirk.go → optim/rootfind.go | 30 | Newton inner solve for Gauss-Legendre IRK stages |
| chaos/constrained.go → optim/rootfind.go | 30 | Newton inner solve for RATTLE position projection |
| chaos/lie_integrator.go → geometry/so3.go (new) | 80 | Lie-group exp/log/dexpinv for SO(3) |
| chaos/variational.go → autodiff/dual.go (agent 168 §A) | 60 | Forward-mode dual numbers for D₁L_d, D₂L_d |
| chaos/symplectic.go → orbital/ | 0 | Wisdom-Holman (deferred I14) consumes Kepler-step substrate |
| chaos/symplectic.go → testdata/symplectic_drift.json | n/a | Cross-language parity grid |
| chaos/poincare.go → chaos/analysis.go | 0 | BifurcationDiagram extension (Poincaré-section in 2D maps) |

Total connective LOC: ~200 across 5 cross-package edges.

---

## 7. Single-highest-leverage 1-day project

**Tier-1 item I2 (Velocity Verlet + Leapfrog, 140 LOC).** Justification:

1. **Keystone.** Every Tier-2 method (I3 Forest-Ruth, I4 Yoshida, I5 McLachlan-Atela, I7 Suzuki) wraps this inner loop with a coefficient table — once I2 lands, six more methods follow at ~30 LOC each.
2. **Closes correctness debt immediately.** `TestLotkaVolterra_HamiltonianConserved` becomes honest the moment Verlet replaces RK4 in the test (the test was wrong per agent 026 N2 and agent 168).
3. **Unblocks agent 027 systems.** T1.S11 (Hénon-Heiles), T2.S6 (restricted three-body), T2.S8 (double pendulum) all *require* a symplectic integrator to test honestly — agent 027 T2.S8 explicitly defers its golden file to "post-026-N2." This slot ships that.
4. **Unblocks agent 168.** Hamiltonian-Neural-Networks integration (the Greydanus-Dzamba-Yosinski 2019 baseline) requires a symplectic stepper for the time-evolution loss — RK4 corrupts the loss landscape.
5. **Zero dependencies, no API churn.** Pure additive `chaos/symplectic.go` file; no break to existing `RK4Step` / `EulerStep` / `SolveODE` surface.

---

## 8. Single-highest-leverage cutting-edge piece

**Tier-2 item I7 (Suzuki fractal + Blanes-Casas-Ros optimal compositions, 40 LOC).** Justification:

1. **Genuine cutting-edge.** Yoshida's recursive composition is widely known; Blanes-Casas-Ros 1999/2002 *optimal* coefficient sets — found by minimising the leading-error-norm under a fixed stage count via constrained nonlinear search — are state-of-the-art and rarely shipped outside specialist code (REBOUND, Newton, GMD).
2. **Order-of-magnitude error reduction.** BCR 8th-order (15-stage) coefficients have ≈10× smaller leading-error-norm than recursive Yoshida-S₈; for solar-system long-time integration this translates to ≈10× longer reliable integration time at fixed dt.
3. **Pure data, ~40 LOC of code.** Coefficient tables (hand-tabulated to 18 digits) wrapped over the I4 driver. No new algorithmic surface.
4. **No standard library ships these.** Go has nothing. SciPy has none of the symplectic compositions. REBOUND ships them but in C and tied to N-body. reality would be the only zero-dependency pure-math library shipping the BCR canon.

---

## 9. Verdict

**SHIP** Tier 1 + Tier 2 (~720 LOC over 2-3 sprints) — the symplectic + composition canon plus the energy-drift parity contract. Closes agent 026 N2 (which is currently a wrong test, not just a missing feature) and unblocks agents 027 (Hénon-Heiles, three-body, double-pendulum systems) and 168 (HNN/LNN/adjoint-ODE).

**DEFER-BUT-DESIGN** Tier 3 (~700 LOC) — Gauss-Legendre IRK + variational integrators + RATTLE. Ship when first non-separable-H or constrained-system consumer pulls (Pistachio rigid-body, Oracle non-canonical Hamiltonian flow).

**DEFER** Tier 4 (Munthe-Kaas Lie-group, ~380 LOC) — wait for an SO(3) consumer in `geometry/`.

**DROP** Tier 5 (stochastic Verlet / r-RESPA / Wisdom-Holman) until specific consumer demand. Cross-link the BAOAB scheme through slot 202 (new-sde) when that lands.

**Cross-slot synergy callouts:** slot 168 (synergy-physics-autodiff) for the LNN/HNN/variational pile-on; slot 026-N2 (chaos-numerics) for the wrong-test fix; slot 027-T1.S11/T2.S6/T2.S8 (chaos-missing) for the Hénon-Heiles / three-body / double-pendulum systems that pair with this; slot 102 (optim-missing) for `optim/rootfind.go` Newton substrate consumed by I8 + I12; slot 074 (graph-missing) is unrelated; slot 202 (new-sde) for the deferred BAOAB cross-link.

---

*204-new-symplectic-int.md — 396 lines.*
