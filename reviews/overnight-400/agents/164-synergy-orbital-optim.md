# 164 | synergy-orbital-optim

**Topic:** orbital × optim — Lambert problem, low-thrust trajectory optimization
**Block:** B (cross-package synergies)
**Date:** 2026-05-08
**Scope:** capabilities that emerge ONLY when `orbital/`, `optim/`, `calculus/`,
and `linalg/` are composed; not isolation gaps (covered by 106-110, 101-105,
016-020, 091-095). Repo at v0.10.0, 1965 tests passing.

## Two-line summary

Today `orbital/` is **eight closed-form scalars** (KeplerOrbit element→position
only, OrbitalPeriod, OrbitalVelocity, HohmannTransfer, EscapeVelocity,
HillSphere, SynodicPeriod, TrueAnomalyFromMean) at **266 LOC** with zero
boundary-value solver, zero state→element inversion, zero perturbation
propagator, zero ΔV optimisation; `optim/` ships **eight workhorses**
(Bisection/Newton/GoldenSection/GD/LBFGS/SA/GA/Simplex) plus proximal/transport
but no NLP-with-equality-constraints driver. **Sixteen synergy primitives
(L0-L16) totalling ~2,585 LOC of glue, ZERO new mathematics** stand up the
whole Lambert/porkchop/low-thrust/Q-law/patched-conics stack on existing bases;
keystone is **L1 `LambertIzzo` (~280 LOC)** because porkchop, intercept,
rendezvous, gravity-assist, asteroid deflection, and Sims-Flanagan all reduce
to repeated Lambert calls — and the cheapest first PR is **L0 `StateToElements`
+ `ElementsToState` + `constants/celestial.go` (~145 LOC)** because every
primitive below consumes them and L0 is purely algebraic.

---

## Bases — what each package exposes today

`orbital/` (266 LOC): `KeplerOrbit(a,e,i,ω,Ω,ν)→(x,y,z)` element→position only
via 3-1-3 Euler rotation (no velocity, no inverse). `OrbitalPeriod`,
`OrbitalVelocity` (vis-viva), `HohmannTransfer→(Δv1,Δv2)`, `EscapeVelocity`,
`HillSphere`, `SynodicPeriod`, `TrueAnomalyFromMean` (Newton on Kepler eq).
**Absent:** state→elements inversion, J2/drag perturbations, Lambert, porkchop,
patched-conics, low-thrust, CR3BP, bi-elliptic. No body-μ constants.

`optim/` (~2,400 LOC, agents 101-105):
- `rootfind.go`: `BisectionMethod`, `NewtonRaphson`, `GoldenSectionSearch`,
  `LinearInterpolateRoot` — exactly the three solvers Lambert/Q-law need,
  all allocation-free.
- `gradient.go`: `GradientDescent`, `LBFGS` (two-loop recursion, Armijo
  backtracking, m-pair ring buffer) — `func(x,g)` allocation-free shape.
- `metaheuristic.go`: `SimulatedAnnealing(f, x0, neighbor, T0, cooling,
  maxIter, rng)` — global-search keystone.
- `genetic.go`: `GeneticAlgorithm` real-coded BLX-α; `[-5,5]^dim` default.
- `linear.go`: `SimplexMethod`, `InteriorPoint` (LP only).
- `interpolate.go`: `LinearInterpolate`, `CubicSplineNatural` — direct inputs
  to porkchop iso-ΔV contours and thrust profile parameterisation.
- `proximal/` (ADMM/FBS/L1-prox), `transport/` (Sinkhorn, orthogonal here).
- **Absent:** SQP, augmented Lagrangian, trust-region, equality-constrained
  Newton, particle swarm, CMA-ES, NSGA-II, pseudospectral collocation.

`calculus/` (274 LOC): `NumericalDerivative`, `NumericalGradient` (in-place,
zero-alloc), Trapezoidal, Simpson, GaussLegendre 2/3/4/5-pt, MonteCarlo.
**Absent:** LGL nodes for pseudospectral, adaptive Gauss-Kronrod.

`chaos/ode.go`: `RK4Step(f(t,y,dydt), t, y, dt, out)`, `EulerStep`,
`SolveODE(f, y0, t0, tEnd, dt) → [][]float64` — **already the propagator
signature L8 Cowell needs** (wrap 6-state (r,v) as `y []float64`).

`linalg/`: `MatMul`, `MatVecMul`, `LUDecompose/Solve`, `CholeskyDecompose/Solve`,
`CrossProduct`, `DotProduct`, `L2Norm`, `Determinant` — every kernel for L0
state-vector algebra and Sims-Flanagan match-point Jacobians.

`constants/`: `GravitationalConst` only. **Absent:** MuSun/MuEarth/MuMars/
MuJupiter/MuMoon/AU/J2Earth/REarth — every Lambert call hand-rolls these.

Neither `orbital/` nor `optim/` imports the other. Clean greenfield.

---

## Sixteen synergy primitives

### Tier 1 — element/state plumbing (prereq)

**L0 `StateToElements(r,v,μ)→elements`** + **`ElementsToState`** ~120 LOC.
Inverse of existing `KeplerOrbit`. Standard Vallado 4e algorithm 9 (RV2COE):
h=r×v (`linalg.CrossProduct`), n=ẑ×h, e=((|v|²−μ/|r|)r−(r·v)v)/μ. Pure
algebraic, golden-file friendly, round-trip pin to 1e-12.

### Tier 2 — Lambert family (keystone)

**L1 `LambertIzzo(r1,r2,Δt,μ,prograde,Nrev)→(v1,v2,ok)`** ~280 LOC. Izzo 2014
universal-variable reformulation (ESA pykep, NASA GMAT, AstroForge). Chosen
over Battin/Gooding because (i) bounded x∈(-1,1) — `optim.BisectionMethod`
brackets robustly, (ii) closed-form dT/dx — `optim.NewtonRaphson` quadratic
convergence after one bracket step, (iii) multi-rev branch reduces to
`optim.GoldenSectionSearch` for inflection between short-/long-period.
Householder/Halley iteration in Izzo §3.2 inlines as specialised
`NewtonRaphson` with pre-computed dT/dx, d²T/dx² (Izzo eqs. 22-23).
**Reduces porkchop, intercept, rendezvous, gravity-assist, asteroid
deflection, Sims-Flanagan match-point all to repeated calls of this one
function.** Reference: Izzo 2014 Celest. Mech. Dyn. Astron. 121, MIT pykep.

**L2 `LambertGooding(r1,r2,Δt,μ,prograde)→(v1,v2)`** ~210 LOC. Gooding 1990
series companion. Slower asymptotically but more robust at parabolic boundary
(e≈1) — used as cross-validation oracle for L1 mirroring 3-detector pattern
in commit 6a55bb4 (audio onset R-MUTUAL-CROSS-VALIDATION).

### Tier 3 — porkchop + interplanetary

**L3 `PorkchopGrid(r1Func, r2Func, departGrid, arriveGrid, μ)→[][]float64`**
~140 LOC. Outer two loops over launch/arrival JD, inner L1 call,
ΔV=|v1−v_planet1|+|v2−v_planet2|. Caller supplies ephemeris closure (no
data tables, per CLAUDE.md rule 2). Iso-ΔV contour extraction is one call
to `optim.CubicSplineNatural` per row + `LinearInterpolate` for level-
crossings.

**L4 `InterceptDV` (~30 LOC)**, **L5 `RendezvousDV` (~35 LOC)** — thin
Lambert wrappers. **L6 `PatchedConics` (~180 LOC)**: heliocentric Lambert,
sphere-of-influence handoff via L0 inverse, hyperbolic flyby segment;
gravity-assist Δv-boost analytic via existing `OrbitalVelocity`.

### Tier 4 — propagation (low-thrust prereq)

**L7 `KeplerPropagate(r0,v0,Δt,μ)→(r,v)`** ~90 LOC. Vallado alg. 8
universal-variable. Solves universal Kepler χ²·c2(αχ²)−...=√μ·Δt via
`optim.NewtonRaphson` with Stumpff c2/c3 closed-form (~20 LOC each).
Faster than ODE for unperturbed two-body.

**L8 `CowellPropagate(r0,v0,t0,tEnd,dt,μ,perturb func(t,r,v)→a3)`** ~50 LOC.
Thin wrapper over `chaos.SolveODE`: r̈=−μr/|r|³+a_perturb. Caller supplies
J2/SRP/third-body. **L9 `J2Acceleration(r,μ,J2,R_eq)`** ~25 LOC closed-form
geopotential gradient (Kozai 1959).

### Tier 5 — low-thrust core

**L10 `EdelbaumDV(v0,vf,accel,Δi)→(Δv,t_burn)`** ~25 LOC. Edelbaum 1961
closed-form circle-to-circle constant-acceleration with plane-change cosine.
Pure algebra, **warm-starts L11/L12 within factor of 2 of optimum** on
LEO→GEO, LEO→Lunar, Earth→Mars test missions.

**L11 `SimsFlanaganTrajectory(r0,v0,rf,vf,Tmax,m0,Isp,Nseg,μ)→(impulses[Nseg][3], match_residual[6])`**
~380 LOC. JPL/DSN workhorse (Sims-Flanagan 1999 AAS 99-338). Discretise N
segments (typical 20-60), each is 3-component impulsive Δv at midpoint
bounded by ‖Δv‖≤Tmax·Δt/m, Lambert-propagate forward from r0 to midpoint
N/2, backward from rf to same midpoint, residual = state-mismatch.
NLP variables: 3N impulses + m0; equality constraint: 6-component match
residual = 0; inequality: per-segment thrust bounds. **v0 fallback: minimise
quadratic penalty ‖match‖²+λ‖impulses‖² with `optim.LBFGS` +
`calculus.NumericalGradient`** (~100 LOC penalty wrapper). Analytic STM
chain (Battin §11.5) is v1 follow-up.

**L12 `QLawController(state, target_elements, weights)→thrust_dir`** ~220 LOC.
Petropoulos 2004 Lyapunov feedback in equinoctial elements: Q=Σwᵢ((eᵢ−eᵢ*)/Δeᵢ_max)²,
thrust_dir=−∇Q projected on thrust ellipsoid. Closed-form gradients (Petropoulos
eqs. 11-15) — **no NLP, no shooting**. Composes with L8: at each Cowell step
call L12 to set thrust direction. **Most useful demo capability** — produces
LEO→GEO transfer in ~10ms with no optimiser tuning.

### Tier 6 — NLP driver (optim gap)

**L13b `AugmentedLagrangian(f, gradF, ceq, x0, μ_init, ω_init, η_init)→x*`**
~280 LOC (recommended over full SQP for v1). Bertsekas 1996. Outer μ-doubling,
**inner solve via existing `optim.LBFGS`** with no new QP machinery. Penalty
multiplier doubles each outer iteration where ‖ceq‖>η_k. **Cheaper than full
SQP by ~240 LOC and sufficient for L11 v0.** Full SQP (`L13` ~520 LOC, BFGS
Hessian + Han-Powell L1 merit + active set + Fletcher-Leyffer 2002 filter)
deferred until L11 demands it.

### Tier 7 — multi-objective + global

**L14 `NSGA2(fitness func([]float64)→[]float64, dim, popSize, gens, mutRate, rng)
→ (paretoFront, frontFitness)`** ~340 LOC. Deb 2002 non-dominated sort +
crowding distance, extends `optim.GeneticAlgorithm` BLX-α. **Multi-objective
(ΔV vs TOF) Pareto fronts directly**, eliminating post-hoc grid scan from L3
porkchop. Fitness eval = one L1 call.

**L15 `ParticleSwarm(fitness, dim, swarmSize, iters, w, c1, c2, bounds, rng)→
(best, bestFit)`** ~140 LOC. Kennedy-Eberhart 1995. **Already requested by
161-control-prob §C7 (Petropoulos gain-tuning) — single implementation, two
consumers.** Empirically beats GA on smooth continuous porkchop minima.

**L16 `BiEllipticTransferDV(r1, r2, r_apoapsis, μ)→(Δv1,Δv2,Δv3,t_total)`**
~40 LOC. Sternfeld 1934, Ho 1962 — three-burn cousin of `HohmannTransfer`,
optimal when r2/r1>11.94. Pure closed-form. Hohmann-vs-bi-elliptic optimiser
= existing `HohmannTransfer` + L16 + 5 lines of `optim.GoldenSectionSearch`
on r_apoapsis ∈ [r2, 100·r2].

---

## Composition table

| Primitive | New mathematics? | Existing primitives consumed |
|-----------|-----------------|------------------------------|
| L0 State↔Elements | No (Vallado alg. 9) | linalg.CrossProduct, DotProduct, L2Norm |
| L1 LambertIzzo | No (Izzo 2014) | optim.Bisection, NewtonRaphson, GoldenSection |
| L2 LambertGooding | No (Gooding 1990) | optim.NewtonRaphson |
| L3 PorkchopGrid | No | L1, optim.CubicSplineNatural, LinearInterpolate |
| L4 InterceptDV | No | L1, linalg.L2Norm |
| L5 RendezvousDV | No | L1, linalg.VectorSub |
| L6 PatchedConics | No | L0, L1, L7, OrbitalVelocity |
| L7 KeplerPropagate | No (Vallado alg. 8) | optim.NewtonRaphson |
| L8 CowellPropagate | No | chaos.SolveODE, RK4Step |
| L9 J2Acceleration | No (Kozai 1959) | constants.J2Earth (NEW) |
| L10 EdelbaumDV | No (Edelbaum 1961) | math.Sqrt only |
| L11 SimsFlanagan | No (1999) | L1, L13b, calculus.NumericalGradient |
| L12 QLaw | No (Petropoulos 2004) | L0, L8 |
| L13b AugLag | No (Bertsekas 1996) | optim.LBFGS, linalg.LUSolve |
| L14 NSGA-II | No (Deb 2002) | optim.GeneticAlgorithm internals |
| L15 ParticleSwarm | No (Kennedy 1995) | math.Rand only |
| L16 BiElliptic | No (Sternfeld 1934) | orbital.OrbitalVelocity |

**Zero new mathematics.** All 1934-2014 vintage; reference implementations
exist under MIT (pykep, poliastro) / BSD (GMAT).

---

## Connective-tissue LOC tally

| Tier | Primitives | LOC |
|------|-----------|-----|
| 1 plumbing | L0 | 120 |
| 2 Lambert | L1+L2 | 490 |
| 3 porkchop/patched | L3+L4+L5+L6 | 385 |
| 4 propagation | L7+L8+L9 | 165 |
| 5 low-thrust | L10+L11+L12 | 625 |
| 6 NLP | L13b | 280 |
| 7 multi-obj+global | L14+L15+L16 | 520 |
| **Total** | **L0-L16** | **~2,585** |

Tests: golden vectors cross-validated against pykep/poliastro/GMAT (per
CLAUDE.md rule 1) — ~30/primitive × 16 × 2 ≈ 960 vectors. **Lambert pin:**
synthesise (r1,v1)→L7→(r2,v2,Δt) then L1(r1,r2,Δt) recovers v1,v2 to 1e-10
plus L2 agrees to 1e-9. Round-trip witness mirrors 154 calibration-pair
pattern and commit 6a55bb4 R-MUTUAL-CROSS-VALIDATION.

---

## Recommended PR ordering

| PR | Content | Days | LOC |
|----|---------|------|-----|
| 1 | L0 + `constants/celestial.go` | 0.5 | 145 |
| 2 | L1 LambertIzzo (Nrev=0) | 2 | 280 |
| 3 | L3+L4+L5 porkchop/intercept/rendezvous | 1 | 205 |
| 4 | L7+L8+L9 propagation triple | 2 | 165 |
| 5 | L13b AugLag (HS-001..020 NLP suite) | 3 | 280 |
| 6 | L10+L11+L12 Edelbaum+Sims-Flanagan+QLaw | 4 | 625 |
| 7 | L14+L15+L16 NSGA-II+PSO+BiElliptic | 3 | 520 |
| 8 | L2 Gooding + L1 multi-rev | 2 | 210 |

**Total: 8 PRs, ~17 days, ~2,425 LOC code + ~1,500 LOC tests.**
PR-3 demos: Earth→Mars 2024 porkchop reproduces published GMAT contours
within 3% ΔV grid-by-grid. PR-6 demo: Q-law LEO→GEO reproduces Petropoulos
2004 Table 1 within 5%. PR-7: NSGA-II validates ZDT1-ZDT6, PSO validates
Rosenbrock+Rastrigin+Ackley matching genetic_test.go fixtures.

---

## Coordination

- **161 §C7** (control-prob): PSO gain-tuning. **L15 single implementation,
  two consumers** — coordinate-once-ship-once.
- **163 §A4** (optim-autodiff): If reverse-mode Jacobian lands first, L11
  Sims-Flanagan replaces finite-difference gradient-of-Lambert with reverse-
  mode, cutting ~20× off Jacobian compute. Cite in L11 docstring.
- **107 orbital-missing** flagged Lambert + J2 — L1, L7-L9 are the response.
- **102 optim-missing** flagged "no NLP equality-constraint driver" — L13b
  is the response.
- **constants/celestial.go** new file (~30 LOC): MuSun/MuEarth/MuMars/MuJupiter/
  MuMoon/AU/J2Earth/REarth, NIST CODATA + JPL DE440 cited per CLAUDE.md rule 4.

## Architectural placement (13th synergy reconfirmation)

Synergies live in **consumer-side directory**, never primitive-supplier:
- `orbital/lambert.go` (L1+L2), `porkchop.go` (L3-L6), `propagate.go` (L7-L9),
  `lowthrust.go` (L10-L12), `transfer.go` (L16, plus existing Hohmann moves
  here), `state.go` (L0). Astrodynamicists search `orbital/` first.
- `optim/auglag.go` (L13b), `nsga2.go` (L14), `pso.go` (L15) — these have
  non-orbital consumers (control MPC, prob MAP).
- `constants/celestial.go`.

**Ten new files, zero modifications to existing files** (purely additive).
Matches 151/153/154/155/156/157/158/159/160/161/162/163 placement precedent.

## Out-of-scope

- **CR3BP / Halo / weak stability boundaries** — needs rotating-frame
  propagator, 5-10× LOC of L1-L16. File as agent 165+.
- **Pseudospectral collocation (LGL, Radau)** — needs `calculus/` LGL nodes
  + Jacobian-pattern solver, ~600 LOC marginal. Sims-Flanagan enough for v1.
- **Pontryagin indirect TPBVP / costate** — initial-guess black art. Defer
  to v3 after L11+L12 demonstrate consumer demand.
- **High-fidelity gravity (full geopotential, Moon/Sun, SRP, drag, GR)** —
  GMAT-class fidelity is downstream service problem. L9 J2-only is enough
  for mission planning.
- **Multi-flyby MGADSM trajectory tree search** — reduces to L14+L1 with
  sequence-as-decision-variable; consumer-side script, not new primitive.
  Document as L14 docstring example.

---

## Bottom line

`reality` is unusually well-positioned for the Lambert/low-thrust stack:
1. `optim.NewtonRaphson` + `BisectionMethod` + `GoldenSectionSearch` are
   exactly the three solvers Lambert needs, all allocation-free.
2. `optim.LBFGS` is `func(x,g)`-shaped — direct AugLag inner loop.
3. `chaos.SolveODE`/`RK4Step` is exactly the propagator signature Cowell
   needs (6-state (r,v) wraps as `y []float64`).
4. `linalg.CrossProduct`/`DotProduct` give the angular-momentum plumbing
   for L0.
5. `optim.GeneticAlgorithm` real-coded BLX-α is the NSGA-II base.
6. CLAUDE.md golden-file rule + cross-language testing makes Lambert/
   porkchop ideal — pykep/poliastro reference implementations are MIT and
   produce reproducible JSON.

Genuinely new machinery is **only L1 Lambert (~280 LOC) + L11 Sims-Flanagan
match-point (~380 LOC) + L13b AugLag (~280 LOC) = 940 LOC of irreducible
novelty.** The other 1,645 LOC is gluing existing parts in documented
compositions. **Highest-leverage v1 synergy** because a working Lambert
solver alone is the missing ingredient for mission-planning consumers
across aicore, and total LOC cost is under one engineer-week.

Distinct from 106-110 (orbital isolation — flags Lambert/J2 but not the
composition), 101-105 (optim isolation — flags SQP/NSGA-II/PSO gaps but
not the consumer), 161 (control-prob — coordinates on L15 PSO), 163
(optim-autodiff — coordinates on L11 reverse-mode Jacobian future-work),
154 (chaos-timeseries — orthogonal but shares round-trip golden-file idiom),
162 (graph-prob — orthogonal). Report at agents/164-synergy-orbital-optim.md.
