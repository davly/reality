# 112 — physics-missing: relativistic, Lagrangian, continuum-mechanics gaps

**Scope:** missing-feature backlog for `C:\limitless\foundation\reality\physics\`
(28 fns across `mechanics.go`, `thermo.go`, `materials.go`, `optics.go`).
Agent 111 confirmed the package is a freshman-physics surface. This report
enumerates canonical primitives NOT yet shipped, sized and tiered for the
v0.11–v1.0 roadmap. Cross-checked against `em/`, `fluids/`, `orbital/`,
`prob/`, `linalg/`, `chaos/`, `constants/` to avoid duplication.

References surveyed: SymPy `physics.{mechanics,units,qft,vector}`;
Modelica `Mechanics.MultiBody`, `Media.IdealGases`; Diffractor.jl;
Feynman *Lectures* I & II; Goldstein *Classical Mechanics* 3e; Jackson
*Classical Electrodynamics* 3e; Misner-Thorne-Wheeler *Gravitation*;
Sadd *Elasticity*; Reichl *Stat Phys*; Peskin & Schroeder (Mandelstam).

---

## 0  Inventory

Already shipped (28 fns): Newton/projectile/spring/pendulum/elastic-1D/
gravitation/KE/PE in `physics/mechanics.go` (F-5 flawed); Hooke 1D + von
Mises/Tresca principal-input + fracture/beam/buckling in `materials.go`
(no tensor type); ideal gas / Stefan-Boltzmann / Carnot / Newton cool /
Fourier / heat-eq / thermal-exp in `thermo.go`; Snell/Fresnel/Beer-Lambert
in `optics.go`. Covered elsewhere: Coulomb/E-field/Ohm/RC/LC in `em/`;
Kepler/escape/Hohmann/Hill in `orbital/`; Reynolds/Bernoulli/drag/Stokes
in `fluids/`.

Six audit-prompt headlines — relativistic kinematics, Lorentz boosts,
Lagrangian/Hamiltonian, Noether currents, Maxwell stress tensor, GR
primitives — have **zero coverage** anywhere in the repo.

---

## 1  Tier 1 — canonical primitives, v0.11–v0.12 (~700 LOC, 32 fns)

Textbook closed-form algebra. Fits "numbers in, numbers out" charter.
Explicit consumers: Pistachio physics visualizer, `aicore` reasoning,
lab-class formula coverage.

### T1.1  Special relativity — Lorentz primitives  *(~120 LOC, 7 fns)*

```go
LorentzGamma(beta float64) float64                        // 1/sqrt(1-β²) via (1-β)(1+β)
LorentzBeta(v, c float64) float64
LorentzBoost1D(t, x, beta float64) (tp, xp float64)
LorentzBoost4(ct, x, y, z, betaX, betaY, betaZ float64) [4]float64
RelativisticMomentum(m, v float64) float64                // p = γmv
RelativisticEnergy(m, v float64) float64                  // E = γmc²
RelativisticKE(m, v float64) float64                      // (γ-1)mc² with Synge-form for β<<1
EnergyMomentumInvariant(E, p float64) float64             // E² - (pc)² = (mc²)²
VelocityAdditionParallel(u, v float64) float64            // (u+v)/(1+uv/c²)
RelativisticDoppler(f0, beta, theta float64) float64      // longitudinal+transverse
```

**Numerical hazards:**
- γ at β→1: naive `1/√(1-β²)` cancels, loses half precision. Use
  `(1-β)(1+β)` directly (Goldberg 1991 §2.4). Recovers ~15 digits to
  β=1-1e-16. +Inf at |β|=1, NaN at |β|>1.
- KE = (γ-1)mc²: cancellation at β<<1 (γ ≈ 1+β²/2). Switch to
  `(γ²β²)/(γ+1)·mc²` Synge form, or Taylor `½mv²(1+¾β²+…)` for β<1e-4.
- Velocity addition saturates at c: confirm 1−(u⊕v)/c at extreme β.

### T1.2  Four-vectors and Mandelstam invariants  *(~80 LOC, 6 fns)*

```go
type FourVector [4]float64                                // [ct, x, y, z] or [E/c, p]
MinkowskiInner(a, b FourVector) float64                   // signature (+,-,-,-)
ProperTime(world []FourVector) float64
MandelstamS(p1, p2 FourVector) float64                    // (p1+p2)²
MandelstamT(p1, p3 FourVector) float64                    // (p1-p3)²
MandelstamU(p1, p4 FourVector) float64                    // (p1-p4)²
ComptonScatteredWavelength(lambda0, theta float64) float64
```

Reference: Peskin & Schroeder §5.4; Jackson §11.5.

### T1.3  Lagrangian / Hamiltonian primitives  *(~100 LOC, 8 fns)*

```go
Lagrangian1D(q, qdot float64, T, V func(float64) float64) float64  // L = T(q̇) - V(q)
Hamiltonian1D(q, p, m float64, V func(float64) float64) float64    // H = p²/(2m) + V(q)
Action(L func(float64,float64) float64, q, qdot []float64, dt float64) float64
                                                          // S = ∫L dt via calculus/Simpson
EulerLagrange1DResidual(L func, q, qdot, qddot, t, eps float64) float64
HamiltonEqDot(H func(q,p float64) float64, q, p, eps float64) (qdot, pdot float64)
PoissonBracket1D(f, g func(q,p float64) float64, q, p, eps float64) float64
LegendreTransform1D(L func(qdot float64) float64, p, eps float64) (q, H float64)
```

**Why FD here, not autodiff:** keeps `physics/` zero-dep on `autodiff/`.
The `func(float64) float64` signature already breaks
numbers-in-numbers-out slightly but is necessary. Symplectic integrators
themselves stay in `chaos/` (per agent 111).

### T1.4  Noether currents  *(~60 LOC, 4 fns)*

`NoetherEnergy` (q̇·∂L/∂q̇ − L, time-translation), `NoetherMomentum`
(∂L/∂q̇, space), `NoetherAngularMomentum1D` (planar), `NoetherCurrent`
(generic infinitesimal symmetry). Pedagogical formula machinery;
integrators in `chaos/`. Goldstein §13.7.

### T1.5  Inertia tensor + rigid-body dynamics  *(~120 LOC, 8 fns)*

```go
InertiaSphere(m, r float64) float64                       // (2/5)mr²
InertiaSphericalShell(m, r float64) float64               // (2/3)mr²
InertiaCylinder(m, r, h float64, axis int) float64
InertiaRod(m, L float64, axis int) float64                // (1/12)mL² center, (1/3)mL² end
InertiaPlate(m, a, b float64) float64                     // (1/12)m(a²+b²)
ParallelAxis(I_cm, m, d float64) float64                  // I = I_cm + md²
PerpendicularAxis(Ix, Iy float64) float64                 // Iz = Ix + Iy
EulerEqRigidBody(I, omega, tau [3]float64) [3]float64     // Iω̇ + ω×(Iω) = τ
```

Off-axis inertia uses the Voigt `[6]float64` from agent 111 M-1
(`physics/tensor.go`); same Cardano solver yields `PrincipalAxesOfInertia`.

### T1.6  Tsiolkovsky rocket  *(~30 LOC, 3 fns)*

`RocketDeltaV(ve, m0, mf) = ve·ln(m0/mf)` — use `log1p((m0-mf)/mf)` for
small Δv near m0≈mf where the ratio cancels; `RocketBurnTime`;
`SpecificImpulseToExhaust(Isp, g0)`. Sutton & Biblarz.

### T1.7  Maxwell-Boltzmann + Boltzmann factor  *(~50 LOC, 5 fns)*

`BoltzmannFactor` (exp(−E/kT)), `MaxwellBoltzmannSpeedPDF`,
`MostProbable` (√(2kT/m)), `Mean` (√(8kT/πm)), `RMS` (√(3kT/m)).
Random-variate sampler points to `prob/distributions.go`.

### T1.8  Heat capacities + entropy of mixing  *(~50 LOC, 5 fns)*

`Cv = (f/2)R`, `Cp = ((f/2)+1)R` (Mayer), `γ = (f+2)/f`,
`EntropyOfMixingIdealGas = −R·Σ nᵢ ln(xᵢ)`, `GibbsFreeEnergyIdealGasMix`.
Hazard: `xᵢ ln xᵢ` at xᵢ→0 — `xlogx` helper (precedent: `prob/`).

### T1.9  Heat-engine cycles  *(~40 LOC, 4 fns)*

Otto `1 − r^(1−γ)`, Diesel `1 − (1/r^(γ−1))·(rc^γ−1)/(γ(rc−1))`,
Brayton `1 − rp^((1−γ)/γ)`, Rankine enthalpy form.
Carnot already shipped (F-9 bug per 111).

### T1.10  Real-gas equations of state (restating 111-M3)  *(~150 LOC, 7 fns)*

```go
VanDerWaalsP(n, T, V, a, b float64) float64
RedlichKwongP(n, T, V, a, b float64) float64
SoaveRedlichKwongP(n, T, V, a, b, omega float64) float64
PengRobinsonP(n, T, V, a, b, omega float64) float64
VirialP(n, T, V float64, B, C, D float64) float64         // B(T)/V² + C(T)/V³ + …
CriticalPointVDW(a, b float64) (Tc, Pc, Vc float64)       // 8a/27Rb, a/27b², 3b
JouleThomsonCoefficient(T, dHdP, Cp float64) float64
```

Hazard: cancellation between repulsive `nRT/(V−nb)` and attractive
`a/V²` on critical isotherm — combined-fraction form when
`|t1+t2| < ε·max(|t1|,|t2|)`.

---

## 2  Tier 2 — solid mechanics, statistical mechanics, classical EM tensors (~600 LOC, v0.13)

Requires the tensor type in agent 111 M-1 (`physics/tensor.go`) to land first.

### T2.1  Continuum stress and strain tensors  *(~150 LOC, 9 fns)*

```go
type Tensor3Sym [6]float64                                // Voigt {xx,yy,zz,yz,xz,xy}
CauchyStress(...)                                         // current configuration
PK1Stress(F [3][3]float64, sigma Tensor3Sym) [3][3]float64    // P = J·σ·F⁻ᵀ
PK2Stress(F [3][3]float64, sigma Tensor3Sym) Tensor3Sym       // S = F⁻¹·P
SmallStrain(uGrad [3][3]float64) Tensor3Sym               // ε = ½(∇u + ∇uᵀ)
GreenLagrangeStrain(F [3][3]float64) Tensor3Sym           // E = ½(FᵀF - I)
AlmansiStrain(F [3][3]float64) Tensor3Sym                 // e = ½(I - F⁻ᵀF⁻¹)
HookesLawIsotropic3D(eps Tensor3Sym, E, nu float64) Tensor3Sym
HookesLawAnisotropic(eps Tensor3Sym, C [6][6]float64) Tensor3Sym
StressInvariants(s Tensor3Sym) (I1, I2, I3 float64)
PrincipalStresses(s Tensor3Sym) (s1, s2, s3 float64)      // Cardano (Smith 1961)
```

Voigt makes asymmetry unrepresentable (same trick as quaternions, color).
Cardano is more accurate than generic eigensolve near degenerate
eigenvalues — see agent 111 §6.

### T2.2  Yield criteria + plasticity  *(~100 LOC, 5 fns)*

`VonMisesFromTensor` (√(3/2 · s'·s')), `TrescaFromTensor` (via
`PrincipalStresses`), `MohrCoulomb` (|τ| − σ·tan φ − c), `DruckerPrager`
(√J2 + α·I1 − k), `J2FlowRule` (dε^p = dλ·∂f/∂σ). Hill, Lubliner.

### T2.3  Elastic-modulus identities (restating 111-M6)  *(~30 LOC, 6 fns)*

`G=E/(2(1+ν))`, `K=E/(3(1−2ν))`, `λ=Eν/((1+ν)(1−2ν))`, `ν=E/(2G)−1`,
`ν=(3K−E)/(6K)`, `E=9KG/(3K+G)`. ν=0.5 (incompressible, K→+∞) and
ν=−1 (auxetic, λ→+∞) documented as +Inf.

### T2.4  Quantum statistics  *(~50 LOC, 4 fns)*

`FermiDiracOccupation = 1/(exp((E−μ)/kT)+1)`,
`BoseEinsteinOccupation = 1/(exp(...)−1)`,
`PartitionFunctionCanonical = Σ exp(−Eᵢ/kT)`,
`ChemicalPotentialIdealGas = kT·ln(n·λ³/V)` (thermal de Broglie).
Hazard: `exp((E−μ)/kT)` overflows for (E−μ)≫kT — use stable
`1/(1+exp(−x))` sigmoid form (softmax-stable trick).

### T2.5  Classical EM stress and energy tensors  *(~120 LOC, 6 fns)*

Lives in `physics/em_classical.go` (NOT `em/em.go`, which is the freshman
circuits-and-fields surface — keeps abstraction layered):

```go
PoyntingVector(E, B [3]float64) [3]float64                // (E×B)/μ₀
EMEnergyDensity(E, B [3]float64) float64                  // ½(ε₀E² + B²/μ₀)
EMMomentumDensity(E, B [3]float64) [3]float64             // S/c²
MaxwellStressTensor(E, B [3]float64) [3][3]float64
                                                           // Tᵢⱼ = ε₀(EᵢEⱼ - ½δᵢⱼE²) + (1/μ₀)(BᵢBⱼ - ½δᵢⱼB²)
EMAngularMomentumDensity(r, E, B [3]float64) [3]float64
EnergyMomentumTensor4(E, B [3]float64) [4][4]float64      // Tᵘᵛ
```

Jackson §6.7, §12.10. Use `constants.VacuumPermittivity` / `VacuumPermeability`.

### T2.6  Vibrations / mode shapes  *(~80 LOC, 5 fns)*

`StringMode = sin(nπx/L)`, `StringFrequency = (n/2L)·√(T/μ)`,
`BeamModeFreqEulerBernoulli` (βₙL roots {4.730, 7.853, …}
cantilever-fixed-fixed), `RayleighDampingMatrix C = αM + βK` (delegates
to `linalg/`), `LogarithmicDecrement = ln(x1/x2)`.

---

## 3  Tier 3 — GR primitives, N-body skeleton, advanced thermo (~400 LOC, on demand)

Niche but real consumers (cosmology playground, GPS clock correction).
Mathematically complete, computationally small, off the v1.0 critical path.

### T3.1  Schwarzschild + GR primitives  *(~150 LOC, 6 fns)*

```go
SchwarzschildMetric(M, r, theta float64) [4][4]float64    // diag(-(1-rs/r), 1/(1-rs/r), r², r²sin²θ)
SchwarzschildRadius(M float64) float64                    // 2GM/c²
ChristoffelSchwarzschild(M, r, theta float64) [4][4][4]float64    // 9 nonzero closed-form
GeodesicEqRHS(g, dg [4][4]float64, u [4]float64) [4]float64
GravitationalRedshift(r1, r2, M float64) float64          // sqrt((1-rs/r2)/(1-rs/r1))
ShapiroDelay(b, M float64) float64                        // (4GM/c³)·ln(4r1·r2/b²)
```

Horizon r=rs documented as divergence — refer caller to MTW for
Kruskal-Szekeres.

### T3.2  Einstein tensor (skeleton)  *(~80 LOC, 3 fns)*

`RicciTensorFromChristoffel(Γ, ∂Γ)`, `RicciScalar = gᵘᵛRᵤᵥ`,
`EinsteinTensor = Rᵤᵥ − ½R·gᵤᵥ`. Formula realizers, not solvers; caller
pre-computes Christoffel. MTW eqs. 8.47, 17.10.

### T3.3  N-body simulation primitives  *(~70 LOC, 4 fns)*

`NBodyAccelerations` (O(N²) direct, fixed output buffer),
`NBodySoftenedForce` (Plummer r/(r²+ε²)^{3/2}), `TotalEnergyNBody`,
`TotalAngularMomentumNBody`. Formula primitives only — Verlet/Yoshida
in `chaos/`, Wisdom-Holman in `orbital/` (agent 110). Tree codes / FMM
out of scope.

### T3.4  Compton + Klein-Nishina  *(~50 LOC, 3 fns)*

`ComptonShift = (h/mₑc)(1−cos θ)`, `ComptonScatteringEnergy =
E0/(1+(E0/mₑc²)(1−cos θ))`, `KleinNishinaCrossSection` (differential).

### T3.5  Joule-Thomson + advanced thermo  *(~50 LOC, 4 fns)*

`JouleThomsonInversionTemperature = 2a/Rb` (van der Waals),
`AdiabaticIndex γ = Cp/Cv`, `SpeedOfSoundIdealGas = √(γRT/M)`,
`ClausiusClapeyron dP/dT = L/(T·ΔV)`.

---

## 4  Cross-cutting

### S-1  File layout post-expansion

`physics/` swells from 4 files → 14, each ≤200 LOC:

```
mechanics.go (existing)   relativity.go [T1.1+T1.2]      lagrangian.go [T1.3+T1.4]
rigidbody.go [T1.5+T1.6]  tensor.go [shared Cardano]      continuum.go [T2.1+T2.2]
elasticity.go [T2.3]       statmech.go [T1.7+T2.4]         cycles.go [T1.9]
realgas.go [T1.10]         em_classical.go [T2.5]          gr.go [T3.1+T3.2]
thermo.go (existing)       materials.go (existing)         optics.go (existing)
```

Surface 28 → ~95 functions.

### S-2  Constants needed in `constants/`

Audit-driven additions (verify before adding — 111 didn't inventory):
SpeedOfLight, ElectronMass, ProtonMass, NeutronMass, PlanckConstant
(+ReducedPlanck), BoltzmannConstant, ElementaryCharge, AvogadroNumber
(GasConstant exists), VacuumPermittivity, VacuumPermeability,
ElectronComptonWavelength, FineStructure. Most likely already present;
`physics/` should `_ = ...` import not redefine.

### S-3  Unit-aware types (defer to agent 114 physics-api)

SymPy.physics.units tracks units. Adding a `Quantity` struct to
`physics/` breaks numbers-in-numbers-out. Right home is a thin `aicore`
wrapper consuming `physics/` outputs and tagging them — same pattern
`aicore` uses for `prob/` distributions. See agent 114.

### S-4  Golden-file coverage gap balloons

Agent 111 C-3: existing coverage is 3/28 = 11%. Adding ~95 functions at
the same rate → 11/123 = 9%. PR gating must require ≥30 vectors per new
function (CLAUDE.md design rule 1). Tier 1 alone is ~960 vectors,
~3000 LOC of JSON. Cross-ref agent 110's `AllocsPerRun(100,fn)==0` gate.

---

## 5  Sprint ordering aligned to v0.11→v1.0

| Version | Tier work                                                               | LOC  | Goldens |
|---------|-------------------------------------------------------------------------|------|---------|
| v0.11   | T1.1 (relativity), T1.6 (rocket), T1.7 (MB), T1.8 (Cv/Cp), T1.9 (cycles), 111-M6 (E↔G↔K↔ν) | ~370 | ~600 |
| v0.12   | T1.2 (4-vec+Mandelstam), T1.3 (Lag/Ham), T1.4 (Noether), T1.5 (inertia), T1.10 (VDW/RK/PR + 111-M3) | ~500 | ~700 |
| v0.13   | T2.1 (continuum), T2.2 (yield), T2.4 (FD/BE), T2.5 (Maxwell stress), T2.6 (vibrations) | ~500 | ~600 |
| v1.0    | T3.1 (Schwarzschild), T3.2 (Einstein), T3.3 (N-body), T3.4 (Compton), T3.5 (J-T)         | ~400 | ~400 |

Total: ~1,800 LOC + ~2,300 vectors → "credible advanced-undergraduate-
through-first-year-grad" coverage. SymPy `physics` is ~30k LOC; reality
at 1.8k LOC delivers 80% of the surface every classical/relativistic
physicist asks for, MIT-clean zero-dep.

---

## 6  Out of scope (deliberately)

- **Quantum mechanics operators** (Hilbert spaces, Pauli, spinors,
  second quantization) — separate `quantum/` package if pursued.
- **Lattice QCD / gauge theory** — out by 3 orders of magnitude.
- **FEM solvers** — mesh-based belongs in `fem/` or `aicore`.
- **Plasma MHD solvers** — defer integrators; closed-form Alfvén /
  plasma-frequency / Debye constants OK.
- **Lattice Ising / Potts** — `prob/` Monte Carlo consumer.
- **Symplectic integrators** (Verlet, Yoshida) — `chaos/` per design
  rule. `physics/` provides only the Hamiltonian + Poisson primitives.
- **Autodiff-coupled Lagrangian** (Diffractor.jl-style) — FD primitives
  here keep zero-dep; `aicore` wrapper layers autodiff.
- **GR numerical relativity** (BSSN, Z4c) — out by 4 orders of
  magnitude. T3.1+T3.2 give the primitives a numerical relativist
  consumes, no more.

---

## 7  Single highest-leverage commit

**T1.1 Lorentz primitives (~120 LOC).** Three reasons:
1. The audit-prompt headline ask — relativistic kinematics is the
   largest gap.
2. Closed-form, zero-iteration, IEEE-754-clean (Goldberg trick).
3. Unblocks T1.2 (four-vectors), T3.1 (Schwarzschild γ-factor), T3.4
   (Compton), and a class of `aicore` physics-reasoning queries that
   currently return "no relativistic support."

Second-highest: T2.1 continuum tensors with Cardano principal stresses
(extends 111-M1, ~150 LOC, unblocks structural-mechanics demos and
T2.2 yield criteria).

---

## 8  Direct answers to audit-prompt headlines

**Relativistic kinematics:** T1.1 + T1.2. ~200 LOC, ~13 fns. Goldberg
trick is the dominant numerical hazard.

**Special-relativity boosts:** T1.1 `LorentzBoost1D` and `LorentzBoost4`
(general 3-velocity). 2x2 / 4x4 matmul, Goldberg-stable γ. Round-trip
identity `B(-β)·B(β) = I` is the natural golden-file invariant.

**Lagrangian/Hamiltonian primitives:** T1.3 ships closed-form L, H,
Action (delegates to `calculus/` Simpson), Euler-Lagrange residual,
Hamilton's equations, Poisson bracket, Legendre transform. Symplectic
integration belongs in `chaos/`.

**Noether currents:** T1.4 ships the four canonical (energy from time,
momentum from space, angular momentum from rotation, generic
infinitesimal-symmetry). ~60 LOC. Pedagogical formula machinery.

**Classical EM stress tensor:** T2.5 Maxwell stress + Poynting + EM
energy density + EM momentum density + EM angular momentum density +
4×4 energy-momentum tensor. ~120 LOC. Lives in `physics/em_classical.go`
not `em/em.go` — keeps the two-tier abstraction.

**GR primitives:** T3.1 (Schwarzschild metric, radius, Christoffel,
geodesic RHS, redshift, Shapiro delay) + T3.2 (Ricci, Einstein tensor).
Formula realizers; integrators in `chaos/`. ~230 LOC, 9 fns.

---

*Reality v0.10.0 — agent 112 backlog, 2026-05-07.*
