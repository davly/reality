# 179 | synergy-em-fluids

**Topic:** em x fluids — magnetohydrodynamics (MHD), plasma physics, Hall-MHD primitives.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `em/`, `fluids/`, and `constants/` compose; not isolation gaps (covered by per-package agents 061-065 for em, 066-070 for fluids, 046-050 for constants). Repo at v0.10.0, 1965 tests passing.

## Two-line summary

`em/` ships **only 213 LOC of pure-scalar electrostatics + DC-circuit primitives** (CoulombForce, ElectricField, OhmsLaw, Power=V*I, ResistorsInSeries/Parallel, CapacitorEnergy=0.5CV^2, InductorEnergy=0.5LI^2, RCTimeConstant=RC, ResonantFrequencyLC=1/(2pi*sqrt(LC))) and `fluids/` ships **235 LOC of incompressible-Newtonian-fluid scalars** (ReynoldsNumber, BernoulliPressure, Colebrook-White PipeFlowFriction, DarcyWeisbach, Drag/Lift, TerminalVelocity, StokesLaw, Mass/VolumetricFlowRate) — there is **zero magnetic field anywhere in em** (no B-field, no Biot-Savart, no Ampere, no Faraday, no curl operator, no vector field anywhere) and **zero compressible/conductive/thermodynamic anything in fluids** (no specific-heat-ratio, no sound-speed, no shock relations, no compressible Euler), and `constants/` is **missing every plasma-physics rest mass** (no ElectronMass, no ProtonMass, no AtomicMassUnit, no BohrRadius, no FineStructure, no IonizationPotential, no any-element atomic data), so the entire MHD/plasma surface (M1-M22 below) is currently **0% implementable** without first adding ~14 new constants and ~6 new primitive scalar/vector helpers. Twenty-two synergy primitives (M1-M22) totalling **~3,180 LOC** of connective tissue stand up MHD/plasma/Hall-MHD on the future bases; the gating prerequisite is a **210-LOC plasma-constants pack + 240-LOC vector3 helper + 180-LOC magnetic-field-of-current-loop/wire** — without those, M1 cannot even compute Larmor radius. Cheapest one-day PR is **the constants+helpers prerequisite + M1 Larmor/gyrofreq/plasma-freq/Debye = 720 LOC** saturating R-MUTUAL-CROSS-VALIDATION 3/3 against NRL Plasma Formulary (Huba 2018) tables; highest-leverage one-day unlock is **M5 AlfvenSpeed v_A=B/sqrt(mu0*rho)** (8 LOC of body, 60 LOC with golden vectors) because it directly couples constants.VacuumPermeability with fluids density mass-flow primitives and is the single most cited MHD scalar; crown jewel is **M14 1D-resistive-MHD-shocktube (Brio-Wu 1988)** at ~480 LOC since it requires composing the not-yet-existing compressible-Euler from fluids agent-067/068 + a not-yet-existing Riemann solver + the not-yet-existing vector B-field — three open prerequisites. **This synergy-axis is the single coldest in the entire 1-179 review-set: capability-readiness is roughly 5%, lower than even synergy-zkmark-crypto (175).**

---

## Bases — what each package exposes today

### `em/` (213 LOC, agents 061-065)

`em.go` is a **single flat file** with 10 exported scalar functions, all pure (no state, no allocation). Provenance: every function cites Griffiths "Introduction to Electrodynamics" 4th ed.

| Function | Formula | LOC | Magnetic? |
|---|---|---|---|
| `CoulombForce(q1,q2,r)` | k*q1*q2/r^2 | 3 | no |
| `ElectricField(q,r)` | k*q/r^2 | 3 | no |
| `OhmsLaw(V,R)` | V/R | 1 | no |
| `PowerElectric(V,I)` | V*I | 1 | no |
| `ResistorsInSeries(rs)` | sum | 5 | no |
| `ResistorsInParallel(rs)` | 1/sum(1/r) | 9 | no |
| `CapacitorEnergy(C,V)` | 0.5CV^2 | 1 | no |
| `InductorEnergy(L,I)` | 0.5LI^2 | 1 | no |
| `RCTimeConstant(R,C)` | RC | 1 | no |
| `ResonantFrequencyLC(L,C)` | 1/(2pi*sqrt(LC)) | 1 | no |

Internal: `coulombConst = 1.0/(4*Pi*constants.VacuumPermittivity)` cached as `var`.

**Absent — every magnetic primitive:** no B-field-of-wire (Biot-Savart `B = mu0*I/(2*pi*r)`), no B-field-of-loop, no B-field-of-solenoid (`B = mu0*n*I`), no Ampere's law `oint B dot dl = mu0 I`, no Faraday EMF `emf = -dPhi/dt`, no Lorentz force `F = q(E + v cross B)`, no magnetic flux `Phi = B*A`, no inductance-from-geometry, no mutual inductance, no Maxwell stress tensor, no Poynting vector `S = (E cross B)/mu0`, no electromagnetic energy density `u = (eps0*E^2 + B^2/mu0)/2`, no skin depth `delta = sqrt(2/(omega*mu*sigma))`, no plasma anything. **No vector primitives** — every value is scalar. No 3-vector cross product, no curl, no divergence, no gradient — those would have to come from `geometry/` or be reinvented. agent-062 (em-missing) flagged 22 of these gaps; they all gate this synergy.

### `fluids/` (235 LOC, agents 066-070)

`fluids.go` is a **single flat file** with 10 exported scalar functions, all pure.

| Function | Formula | LOC | Compressible? Conductive? |
|---|---|---|---|
| `ReynoldsNumber(rho,v,L,mu)` | rho*v*L/mu | 1 | no/no |
| `BernoulliPressure(rho,v1,p1,h1,v2,h2,g)` | classical | 1 | no/no |
| `PipeFlowFriction(Re,eps,D)` | Colebrook-White | ~25 | no/no |
| `DarcyWeisbach(f,L,D,rho,v)` | f*(L/D)*rho*v^2/2 | 1 | no/no |
| `DragForce(Cd,rho,v,A)` | 0.5*Cd*rho*v^2*A | 1 | no/no |
| `LiftForce(Cl,rho,v,A)` | 0.5*Cl*rho*v^2*A | 1 | no/no |
| `TerminalVelocity(m,g,Cd,rho,A)` | sqrt(2mg/(Cd*rho*A)) | 5 | no/no |
| `StokesLaw(mu,r,v)` | 6*pi*mu*r*v | 1 | no/no |
| `MassFlowRate(rho,v,A)` | rho*v*A | 1 | no/no |
| `VolumetricFlowRate(v,A)` | v*A | 1 | no/no |

**Absent — every primitive needed for MHD:** no compressible Euler equations, no specific-heat-ratio gamma, no sound speed `c_s = sqrt(gamma*p/rho)` (agent-067 missing-T1), no Mach number, no Riemann solver (HLL/HLLC/Roe), no shock-jump Rankine-Hugoniot, no isentropic flow relations, no equation of state (ideal gas `p=rho*R*T/M`, polytrope, gamma-law), no entropy/enthalpy primitives, no tensor stress, no Navier-Stokes residual operator, no FVM/FEM scaffold, no grid abstraction, no 1D shock tube, no electrical conductivity, no permeability (would come from constants), no current-density coupling. **No vector velocity** — every velocity is scalar.

### `constants/` (252 LOC, agents 046-050)

Has the 4 SI-2019 plasma-relevant exact constants: `ElementaryCharge` (e=1.602176634e-19 C), `Boltzmann` (k_B=1.380649e-23 J/K), `Avogadro`, `Planck`. Has the 2 EM constants: `VacuumPermittivity` (eps0=8.8541878128e-12 F/m, CODATA 2018), `VacuumPermeability` (mu0=1.25663706212e-6 H/m, CODATA 2018). Has `SpeedOfLight` (c=299792458 m/s exact).

**Absent — every plasma-physics rest mass and atomic constant** (agent-047 missing-T7 flagged 14 of these):

| Constant | Value | Used by |
|---|---|---|
| `ElectronMass` | 9.1093837015e-31 kg | M1 (gyrofreq), M3, M4, M11 |
| `ProtonMass` | 1.67262192369e-27 kg | M1 (ion gyrofreq), M16 (Saha) |
| `NeutronMass` | 1.67492749804e-27 kg | nuclear contexts |
| `AtomicMassUnit` | 1.66053906660e-27 kg | every non-H plasma |
| `BohrRadius` | 5.29177210903e-11 m | Saha, atomic cross-sections |
| `RydbergEnergy` | 13.605693122994 eV | Saha (M16) |
| `FineStructure` | 7.2973525693e-3 | radiation, Bremsstrahlung |
| `ClassicalElectronRadius` | 2.8179403262e-15 m | Thomson scattering |
| `ElectronVolt` | 1.602176634e-19 J | every plasma temperature |
| `ThomsonCrossSection` | 6.6524587321e-29 m^2 | radiation transport |

Also missing the energy-conversion macro `eVToKelvin = ElementaryCharge / Boltzmann` (~11604.518 K/eV) used everywhere in plasma temperature notation.

### `geometry/` and `linalg/` — what they could lend

`linalg/` has dense matrix and LU/QR/Cholesky — would be reused by M21 (Grad-Shafranov fixed-point iteration) and M14 (1D shocktube right-eigenvector decomposition). `geometry/` has quaternions but **no plain-old vec3 cross-product as exported helper** — agent-076 (geometry-numerics) noted Vec3.Cross is internal-only. Adding `Vec3{X,Y,Z}` with `Cross`, `Dot`, `Norm`, `Add`, `Scale` as ~80 LOC public is the cheapest path; **every MHD primitive needs vec3**.

---

## Synergy primitives M1-M22

Each line: capability | composition (C = const, E = em, F = fluids, L = linalg, G = geometry, X = NEW prerequisite) | new connective LOC | gates / pinned by.

### Tier-0 prerequisites (gate everything)

**M0a — `constants` plasma-pack** (~210 LOC, half doc/half values)
ElectronMass, ProtonMass, NeutronMass, AtomicMassUnit, BohrRadius, RydbergEnergy, RydbergConstantInfinity, FineStructure, ClassicalElectronRadius, ElectronVolt, ThomsonCrossSection, JoulesPerEV, KelvinPerEV, MagneticFluxQuantum (h/2e). All SI 2019 / CODATA 2018, all golden-tested with ±1 ulp tolerance per project doctrine.

**M0b — `geometry.Vec3`** (~240 LOC) public Vec3 struct with `Add`, `Sub`, `Scale`, `Dot`, `Cross`, `Norm`, `Normalize`, `Outer` (3x3), `Triple` (a · (b × c)). agent-076 flagged this; six Block-B synergies (177 geometry-optim, 168 physics-autodiff, 159 em-signal, 164 orbital-optim, this one, and 175 zkmark-crypto's pairing pre-image) want it.

**M0c — `em.MagneticFieldWire(I,r)`, `em.MagneticFieldLoop(I,R,z)`, `em.MagneticFieldSolenoid(n,I)`** (~180 LOC). Biot-Savart on-axis closed forms — no integration, all scalar. mu0*I/(2*pi*r), mu0*I*R^2/(2*(R^2+z^2)^1.5), mu0*n*I respectively. Citing Griffiths eq. 5.36, 5.38, 5.59 — same provenance style as em.go.

Total prerequisite LOC: **~630**. Until these land, M1-M22 are unimplementable.

### Tier-1: single-particle plasma scalars (core NRL Plasma Formulary)

**M1 — Gyrofrequency, plasma frequency, Debye length, Larmor radius** | C(M0a)+E | ~120 LOC | NRL Formulary p.28-29

```
omega_ce = e*B/m_e                  // electron cyclotron
omega_ci = Z*e*B/m_i                // ion cyclotron
omega_pe = sqrt(n_e*e^2/(eps0*m_e)) // electron plasma freq
omega_pi = sqrt(n_i*Z^2*e^2/(eps0*m_i))
lambda_D = sqrt(eps0*k_B*T_e/(n_e*e^2))
r_L_e    = m_e*v_perp/(e*B)
r_L_i    = m_i*v_perp/(Z*e*B)
```

Six scalars, ~90 LOC body + 30 LOC golden table from NRL Formulary table I.4. Deserves a new file `em/plasma.go` (em is the right package: B-field is em, density is em-fluid coupling). Saturates R-MUTUAL-CROSS-VALIDATION 3/3 against (a) NRL Formulary, (b) Chen "Introduction to Plasma Physics" 3rd ed eq. 1.34/2.6/4.13, (c) hand-substituted symbolic in CONTEXT.md.

**M2 — Plasma beta** | C+E+F | 8 LOC body | `beta = p/(B^2/(2*mu0))` — pure scalar, but composes fluids static pressure with em magnetic-pressure primitive M5b. Pinned by Bellan "Fundamentals of Plasma Physics" eq. 2.93.

**M3 — Drift velocities ExB, grad-B, curvature** | C+E+G(M0b)+linalg | ~140 LOC | first time this synergy genuinely needs vec3 cross product; without M0b it cannot exist

```
v_ExB     = (E cross B) / |B|^2                      // independent of mass and charge
v_gradB   = (m*v_perp^2 / (2*q*B^3)) * (B cross gradB)
v_curv    = (m*v_par^2 / (q*B^3)) * (R_c cross B / R_c^2)
```

R_c is the radius-of-curvature vector (3-vec). Deserves three Vec3-returning helpers in `em/drifts.go`. Pinned by Chen eq. 2.18, 2.24, 2.26.

**M4 — Magnetic moment mu = m*v_perp^2/(2B), adiabatic invariant** | C+E | 12 LOC | Chen 2.41

### Tier-2: fluid-coupled MHD scalars

**M5 — Alfven speed, magnetosonic speeds** | C+E+F | ~80 LOC | the headline MHD primitive

```
v_A   = B / sqrt(mu0*rho)
c_s   = sqrt(gamma*p/rho)        // requires fluids.SoundSpeed (agent-067 T1, NOT YET LANDED)
v_ms  = sqrt(c_s^2 + v_A^2)      // fast magnetosonic in B-perp limit
v_+/- = sqrt(0.5*((c_s^2+v_A^2) +/- sqrt((c_s^2+v_A^2)^2 - 4*c_s^2*v_A^2*cos^2(theta))))
```

**This is the cheapest one-day PR with maximum leverage:** v_A alone is 8 LOC body + ~50 LOC golden vectors against Chen tables and the NRL Formulary; it is the single most-cited MHD scalar in graduate textbooks (Goedbloed-Poedts 2004, Freidberg 2014). **Currently fluids.SoundSpeed does not exist**, so the fast/slow magnetosonic decoration depends on fluids agent-067 missing-T1 landing first; v_A alone does not.

**M5b — Magnetic pressure p_B = B^2/(2*mu0), magnetic tension T_B = B^2/mu0** | C+E | 6 LOC | gates M2 and M14; pinned by Goedbloed-Poedts eq. 2.92.

**M6 — Magnetic Reynolds number Rm = mu0*sigma*v*L = v*L/eta_m** | C+E+F | 10 LOC | exact analogue of fluids.ReynoldsNumber but with magnetic diffusivity eta_m = 1/(mu0*sigma). Lukens 2003 lecture notes eq. 6.11. **Composes fluids.ReynoldsNumber dimensional form 1:1.**

**M7 — Lundquist number S = mu0*L*v_A/eta_m = Rm at v=v_A** | C+E+F | 5 LOC | reconnection scaling parameter; gates M11 Sweet-Parker.

**M8 — Hartmann number Ha = B*L*sqrt(sigma/mu)** | C+E+F | 7 LOC | electrically-conducting-fluid-in-pipe regime parameter. Davidson "Introduction to MHD" 2017 eq. 1.11.

### Tier-3: ionization and atomic plasma equilibria

**M9 — Saha equation** | C(M0a)+linalg-optional | ~120 LOC | Saha 1920

```
n_(i+1)*n_e/n_i = (2*g_(i+1)/g_i) * (2*pi*m_e*k_B*T)^(3/2)/h^3 * exp(-chi_i/(k_B*T))
```

For hydrogen (Z=1) returns ionization fraction x as solution of x^2/(1-x) = (constants*T^1.5/n) * exp(-13.6 eV / k_B*T). Solves quadratic — ~80 LOC pure-scalar + 40 LOC golden table from Carroll-Ostlie "Modern Astrophysics" 2nd ed table 8.1. Crown jewel for stellar interiors and laboratory plasmas. **Requires RydbergEnergy + ElectronMass from M0a.**

**M10 — Bohm criterion / plasma sheath** | C+E | ~60 LOC | sheath edge ion velocity v_B >= sqrt(k_B*T_e/m_i). Stangeby "Plasma Boundary of Magnetic Fusion Devices" 2000 eq. 2.49. Pure scalar; pinned by R-CROSS-LANGUAGE-DETERMINISM (Chen Q4.1 worked example).

**M11 — Spitzer resistivity, Coulomb log** | C+E | ~90 LOC

```
ln(Lambda) = ln(12*pi*n_e*lambda_D^3)             // Coulomb log
eta_S      = (sqrt(2)*Z*e^2*sqrt(m_e)*ln(Lambda))/(12*pi^1.5*eps0^2*(k_B*T_e)^1.5)
```

NRL Formulary p.34. Composes M0a (electron mass), M1 (Debye length), M5b (magnetic diffusivity unit reconciliation). Gates M6/M7 numerically — without Spitzer eta you cannot evaluate Rm/Lundquist for a real plasma.

### Tier-4: MHD waves & instabilities

**M12 — MHD dispersion relations: Alfven, fast, slow** | C+E+F | ~180 LOC | branch-by-angle decoration of M5; returns 3 angular-frequency curves omega(k,theta) for given (B,rho,p). Goedbloed-Poedts eq. 5.74-5.76.

**M13 — Frozen-in-flux test (ideal MHD limit)** | C+E+F | ~80 LOC | given (E,v,B,eta), returns dimensionless |E + v cross B - eta*J|/|v||B| — should be O(eps_machine) in ideal limit, O(1/Rm) otherwise. Direct test of Alfven 1942 frozen-flux theorem.

### Tier-5: integrated MHD solvers (need 067/068 fluids prerequisites)

**M14 — Brio-Wu 1988 1D resistive MHD shock tube** | C+E+F+L+X(fluids-067) | ~480 LOC | crown jewel. Riemann problem with B-field — requires:
1. Compressible Euler from fluids agent-067 missing-T1 (NOT YET LANDED)
2. HLL or Roe Riemann solver, ~200 LOC, NOT YET LANDED
3. M5/M5b/M6/M7 stack
4. 1D finite-volume sweep with Faraday-law update on transverse B

Brio-Wu test is the canonical MHD code-validation problem; every textbook reproduces fig 4 of Brio-Wu 1988. Saturating against literature requires R-CROSS-LANGUAGE-DETERMINISM 3/3 + R-MUTUAL-CROSS-VALIDATION across PLUTO/Athena/FLASH reference data (Stone 2008 publishes JSON-friendly tables).

**M15 — Sweet-Parker reconnection rate** | C+E+F | ~120 LOC | scaling: V_in/v_A = 1/sqrt(S) where S is M7 Lundquist. Composes M5+M7+M11. Parker 1957 / Sweet 1958. Pinned by R-MUTUAL 2/2 against Priest-Forbes "Magnetic Reconnection" 2000 chap 4.

**M16 — Petschek reconnection rate** | C+E+F | ~140 LOC | ~pi/(8*ln(S)) faster than Sweet-Parker. Petschek 1964.

**M17 — Hall MHD term J cross B / (e*n_e)** | C+E+G(M0b)+M0a | ~110 LOC | the namesake of the third bullet. Modifies generalized Ohm's law:

```
E + v cross B = eta*J + (J cross B)/(e*n_e) - grad(p_e)/(e*n_e)
```

Returns three vec3 contributions (resistive, Hall, electron-pressure). Pinned by Birn-Priest "Reconnection of Magnetic Fields" 2007 eq. 2.18.

**M18 — Two-fluid plasma momentum equations (electrons + ions separately)** | C+E+F+M0b | ~260 LOC | doubles the fluids momentum to two species coupled by Lorentz force; Chen chap 5.

**M19 — E-cross-B drift in crossed-field accelerator** | C+E+M0b | ~70 LOC | applied; Hall thruster benchmark.

**M20 — Field-aligned current scaling J_par = (1/mu0)(B/B^2) cdot (curl B)** | C+E+M0b | ~80 LOC | needs curl operator; cheapest synthetic curl is finite-difference on regular grid (~50 LOC standalone).

**M21 — Grad-Shafranov fixed-point iteration (axisymmetric tokamak equilibrium)** | C+E+L+M0b | ~360 LOC | 2D Poisson-like

```
Delta^*(psi) = -mu0*R^2*dp/dpsi - F*dF/dpsi
```

with Delta^* = R*d/dR(1/R*d/dR) + d^2/dz^2. Composes linalg's Jacobi/Gauss-Seidel iteration (or could land alongside `optim/proximal.Fbs` if reposed as a fixed-point) + linalg.Solve for the Newton step. Freidberg "Ideal Magnetohydrodynamics" 1987 chap 6. Pinned by R-CROSS-LANGUAGE-DETERMINISM 3/3 against Solov'ev analytic solution.

**M22 — Bremsstrahlung emissivity, synchrotron emissivity** | C(M0a)+E | ~140 LOC | radiation losses closing the energy equation.

```
P_brem = (32/3) * sqrt(2*pi/3) * Z^2 * n_e * n_i * e^6 / ((4*pi*eps0)^3 * m_e^2 * c^3) * sqrt(k_B*T_e)
P_sync = (2/3) * (e^4*B^2*v_perp^2*gamma^2)/(4*pi*eps0*m_e^2*c^3)
```

Rybicki-Lightman "Radiative Processes in Astrophysics" eq. 5.14b, 6.7c.

---

## Total LOC tallies

| Tier | Items | LOC body | LOC tests/golden | Total |
|---|---|---|---|---|
| Tier-0 prerequisites | M0a, M0b, M0c | 280 | 350 | 630 |
| Tier-1 single-particle | M1, M2, M3, M4 | 175 | 105 | 280 |
| Tier-2 fluid-coupled | M5, M5b, M6, M7, M8 | 130 | 110 | 240 |
| Tier-3 atomic equilibria | M9, M10, M11 | 200 | 70 | 270 |
| Tier-4 MHD waves | M12, M13 | 180 | 80 | 260 |
| Tier-5 integrated | M14-M22 | ~1040 | ~460 | **~1500** |
| **Total** | **22 + 3 prereq** | **~2005** | **~1175** | **~3180** |

Compare: 178-synergy-control-optim totalled ~3420 LOC; 174-synergy-gametheory-optim ~2900 LOC; this one is ~3180 LOC of which **~630 LOC is unblocking prerequisites that no current code can depend on**. Capability-readiness is the lowest in Block-B.

---

## Cross-package coupling currently in code (`grep -r "fluids" em/` etc.)

- `em/em.go` imports only `math` and `constants`. Zero `fluids` references. Zero `linalg`. Zero `geometry`.
- `fluids/fluids.go` imports only `math`. Zero `em`, zero `constants` (sic — fluids should import constants for StandardGravity but currently parameterizes g as a function arg; agent-066 numerics noted this as design choice not bug).
- `constants/` imports nothing.

Coupling LOC today: **0**. Reverse-direction `grep -r "em\." fluids/` and `grep -r "fluids\." em/`: zero hits (modulo string false-positives on word "fluid" inside docstrings). The two packages are **fully orthogonal in code**, exactly as the design doc intends each to be self-contained.

---

## Pin coverage matrix (R-axes saturated by which Mi)

| R-axis | Pinned by | Saturation |
|---|---|---|
| R-MUTUAL-CROSS-VALIDATION (3/3) | M1 (NRL+Chen+symbolic), M5 (NRL+Goedbloed+Freidberg), M9 (Carroll-Ostlie+Saha original+symbolic) | 3 of 22 |
| R-CROSS-LANGUAGE-DETERMINISM (3/3) | M10 (Bohm), M14 (Brio-Wu), M21 (Solov'ev) | 3 of 22 |
| R-IEEE-754-EDGE | every Mi via project doctrine (B=0, n_e=0, T=0 edge cases) | gated by golden infrastructure |
| R-PROVENANCE-CITED | every Mi has Griffiths/Chen/NRL/Goedbloed/Freidberg/Bellan/Davidson/Stangeby/Priest-Forbes/Birn-Priest/Rybicki-Lightman citation | 22 of 22 |

---

## Recommendations / sequencing for an implementer

1. **Land M0a constants pack first** (~1 day). It gates everything below and is independently reviewable by agent-047 (constants-missing) which already wants 14 of these constants. Zero risk; pure value-table addition.
2. **Land M0b geometry.Vec3 second** (~1 day). Wanted by 6+ other Block-B synergies. Add `Cross`, `Dot`, `Norm`, `Add`, `Sub`, `Scale`, `Outer`, `Triple` as 240 LOC + golden vectors. No risk; pure linalg-adjacent code.
3. **Land M0c em.MagneticFieldWire/Loop/Solenoid third** (~half day). Direct extension of em.go style; Griffiths-cited; 180 LOC.
4. **Land M5 Alfven speed + M5b magnetic pressure/tension fourth** (~half day). 8+6 LOC body; immediate pedagogical and numerical leverage.
5. **Land M1 plasma scalars fifth** (~half day). 90 LOC body; saturates R-MUTUAL 3/3.
6. **Land M2 + M6 + M7 sixth** — single-line scalars; ~30 LOC total.

End of week-1: 740 LOC body + 480 LOC golden = ~1220 LOC of new code, M0+M1+M2+M5-M7 all delivered, R-MUTUAL pinned for the highest-leverage scalar.

Week-2 priorities: M9 Saha, M10 Bohm, M11 Spitzer (Tier-3) — atomic-equilibrium completes the laboratory-plasma surface.

Week-3+: M12-M22 — these are research-grade and require fluids agent-067 compressible Euler to land first; flag this as a hard dependency in MASTER_PLAN.md.

---

## Key risk: this synergy is a **prerequisite-graveyard**

Unlike 178 (control-optim, where `optim/proximal.Admm` already exists and just needs new prox callbacks), or 174 (gametheory-optim, where `SimplexMethod` already exists), em-fluids has **three open prerequisites** before any Mi can be tested: M0a constants (open), M0b vec3 (open), fluids compressible (open via agent-067). The architectural witness of "em x fluids is a real synergy and not two orthogonal libraries" is **not yet expressible in code** — the closest thing is M5's 8-line v_A formula, which currently cannot reference a `B` vector in em or a `rho` in any shared object. Fix the three prerequisites and the entire 3,180-LOC plan becomes implementable; otherwise this remains the coldest Block-B axis surveyed.
