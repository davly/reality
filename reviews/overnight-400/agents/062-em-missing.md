# 062 — em-missing

**Topic:** em: missing — FDTD, MoM, FEM-EM, Smith chart, transmission line, antenna patterns, dipole radiation, waveguide modes.

**Premise from 061:** the `em/` package is 213 LOC of "EM 101 cheat-sheet" — eleven scalar formulas (CoulombForce, ElectricField, OhmsLaw, PowerElectric, ResistorsInSeries/Parallel, Capacitor/InductorEnergy, RCTimeConstant, ResonantFrequencyLC) over `float64`. **Zero** vector fields, **zero** complex impedance, **zero** PDE solvers, **zero** antenna math, **zero** waveguide modes, **zero** material models, **zero** scattering. The package is ~3% of the surface that "electromagnetism" canonically denotes. This audit enumerates what is missing, scoped against `reality`'s "math primitives, no I/O, zero deps, golden-file validated, MIT" constraints (CLAUDE.md §1–§6) and against the F-1..F-6 forward-looking numerical-correctness commitments named in 061.

**Scope filter — what counts as in-scope for `reality/em`:**
- IN: closed-form vector-field formulas (Hertzian dipole, current-loop axial B, finite-wire B, Hallén/Pocklington kernels, multipole expansion).
- IN: `complex128`-based primitives (impedance, S/Y/Z/ABCD, Smith-chart bilinear transforms, propagation γ = α + jβ).
- IN: PDE solver kernels — FDTD Yee step, BEM assembly, MoM mutual impedance, FEM Galerkin matrices — deterministic, allocation-disciplined, golden-testable.
- IN: special-functions needed by EM (elliptic K/E/Π, Bessel J/Y/I/K, Si/Ci, Legendre P/Q, Yₗᵐ, Ei).
- IN: antenna pattern algebra (array factor, Friis, Yagi-Uda mutual coupling, aperture distributions); dispersive-material evaluators (Drude, Lorentz, Debye, Cole-Cole, Sellmeier).
- BORDERLINE: mesh generation — geometry math in-scope, full mesher likely outside `reality`.
- OUT: GPU/CUDA/threading; netlist parsing; Touchstone/STL/GMSH I/O; CAD interop. `complex128` arrays themselves in-scope, serialization isn't.

---

## Headline

`em/` ships 11 of an estimated **170–220 canonical EM primitives** across Griffiths/Jackson/Pozar/Balanis textbook + Meep/openEMS/MoMentum/scikit-rf/FEKO/NEC2/COMSOL frontline. The seven topic-listed headlines (FDTD, MoM, FEM-EM, Smith, transmission line, antenna patterns, dipole, waveguide modes) require **6,500–9,500 LOC across 4 missing layers**, all absent today: (1) **complex-arithmetic surface** (currently `float64` only — RF is irreducibly `complex128`), (2) **vector-field surface** (currently scalar — every Maxwell formula is vector-valued), (3) **special-function library** (every closed-form antenna/waveguide uses one), (4) **PDE solver primitives** (FDTD/MoM/FEM/BEM — the topic's headline asks). **Zero** of the seven headlines are reachable without (1) + (3) at minimum.

**Cheapest first-light:** transmission-line closed-forms (coax/microstrip Hammerstad-Jensen/stripline Cohn/twisted pair) — ~80 LOC, no special-functions, immediate PCB/RF utility. Smith chart geometric mapping is next at ~120 LOC. **Hardest:** FDTD-Yee 3D with PML + dispersive materials (~2,500 LOC, multi-month). **Highest reuse:** Bessel/elliptic family (~600 LOC) unlocks antenna patterns, current-loop fields, waveguide modes, optical fibers simultaneously.

Recommended sprint: T1-COMPLEX → T1-SPECIAL → T1-FIELDS → T1-TLINES → T1-SMITH → T1-DIPOLE → T1-WAVEGUIDE → T1-ARRAY → T2 → T3-FDTD-1D → T3-FDTD-2D/3D → T3-MOM → T3-FEM/BEM. Tier-1 items are independent ≤1-week chunks; tier-3 are multi-month.

---

## Tier 1 — Foundation (blocks every topic primitive)

### T1-COMPLEX — `complex128` impedance / S-parameter / propagation surface (≈250 LOC)
Per 061 §F-4 commits native `complex128`. Functions: `Impedance(R,L,C,ω)`, `AdmittanceParallel`, `SeriesImpedance([]complex128)` and `ParallelImpedance` (Kahan over real and imag per F-4), `ReflectionCoefficient(Z_L,Z_0)`, `VSWR`, `ReturnLoss`, `MismatchLoss`. IEEE-754: `complex128(NaN,NaN)` for `1/0`, document + golden-test.

### T1-SPECIAL — Bessel/elliptic/Legendre/Hankel (≈600 LOC, biggest reuse)
Pure math, every textbook EM formula needs one. None exist anywhere in repo. Bundle: Bessel J₀/J₁/Jₙ/Y₀/Y₁/Yₙ (Hart approx, ~150 LOC); modified I/K (~100 LOC); Hankel H^(1,2) (composition, free); spherical j_n/y_n/h_n (~80 LOC); complete elliptic K(m)/E(m) via Carlson RF/RD per 061 §F-3 (~80 LOC); incomplete F/E/Π (~70 LOC); Legendre Pₗ/Pₗᵐ (~60 LOC); spherical harmonics Yₗᵐ (~40 LOC); Si(x)/Ci(x) (~50 LOC); Ei/E₁ (~30 LOC).

### T1-FIELDS — Vector field types + superposition (≈400 LOC)
Reuse `linalg.Vec3` if exists. `ElectricFieldVec(charges, point)` Σ kqᵢ(r−rᵢ)/|r−rᵢ|³ with pairwise default per F-1; `ElectricFieldVecKahan` explicit Kahan variant; `ElectricPotential` Σ kqᵢ/|r−rᵢ|; `MagneticFieldBiotSavart(currents, point)` line integral over wire segments (~120 LOC); `VectorPotentialA`; `Force_qvB(q,v,B)` Lorentz; `PoyntingVector(E,B)` = (1/μ₀)E×B; `EnergyDensityEM` = ½ε₀|E|² + (1/2μ₀)|B|².

### T1-TLINES — Transmission-line characteristic impedance (≈250 LOC)
Cheapest topic-listed. `CoaxImpedance(D,d,εᵣ)` = (60/√εᵣ)·ln(D/d); `MicrostripImpedance(W,h,εᵣ,t)` Hammerstad-Jensen 1980 + Wheeler ±0.1% for 0.1≤W/h≤20 (~80 LOC); `StriplineImpedance(W,b,t,εᵣ)` Cohn 1955 / IPC-2141 (~50 LOC); `TwistedPairImpedance`; `ParallelWireImpedance` = (120/√εᵣ)·acosh(D/d); `MicrostripEffectivePermittivity`; `LossyLineGamma(R,L,G,C,ω)` = √((R+jωL)(G+jωC)); `LossyLineCharacteristicImpedance` = √((R+jωL)/(G+jωC)).

### T1-SMITH — Smith chart geometric primitives (≈150 LOC)
Pure bilinear `complex128`, no T1-SPECIAL dependency. `Z2Gamma(z)` = (z−1)/(z+1); `Gamma2Z`; `Y2Gamma`; `SmithImpedanceCircle(R_norm)`; `SmithReactanceArc(X_norm)`; `SmithVSWRCircle`; `SmithRotateLambda(γ, ℓ)` rotate Γ by 2β·ℓ for line-length transformation.

### T1-DIPOLE — Hertzian and half-wave dipole (≈300 LOC, needs T1-COMPLEX + T1-SPECIAL)
Topic headline. Balanis Ch. 4-5. `HertzianDipoleField(I_0,dl,r,θ,ω)` full E_r/E_θ/H_φ near+far; `HertzianDipoleFarField` kr≫1; `HalfWaveDipoleField` sin pattern × cos((π/2)cosθ); `DipoleRadiationResistance` Hertzian 80π²(L/λ)², half-wave 73.13Ω via Si/Ci; `DipoleInputImpedance` Carter/King via Si/Ci (~80 LOC); `DipolePatternFunction(θ, L/λ)`.

### T1-ANTENNA-METRICS — Friis, gain, directivity, aperture, link budget (≈200 LOC)
`Directivity(U_max,U_avg)` = 4π U_max/P_rad; `DirectivityFromPattern(F²)` numerical 2-sphere via Lebedev quadrature (call into `calculus/`, ~80 LOC); `Gain` = η·D; `EffectiveAperture` = G·λ²/(4π); `FriisTransmission(P_t,G_t,G_r,λ,R)` = P_t·G_t·G_r·(λ/4πR)²; `FriisLog`; `FreeSpacePathLoss` = 20log₁₀(4πR/λ); `AntennaEfficiency`; `BeamSolidAngle`; `EIRP`/`EIRP_dBm`; `LinkBudget`.

### T1-ARRAY — Array factor, beam pattern, beamforming weights (≈250 LOC)
`LinearArrayFactor(N,d,β,θ,λ)` = sin(Nψ/2)/sin(ψ/2)·e^{j(N−1)ψ/2}, ψ=kd·cosθ+β; `LinearArrayPattern`; `PlanarArrayFactor(M,N,dx,dy,βx,βy,θ,φ,λ)`; `UniformExcitationBeamwidth`; `BroadsideArrayFactor` (β=0); `EndfireArrayFactor` Hansen-Woodyard option; `ChebyshevArrayWeights(N,SLL_dB)` Dolph-Chebyshev (~80 LOC); `TaylorArrayWeights(N,SLL,n̄)` (~70 LOC); `BinomialArrayWeights`.

### T1-WAVEGUIDE — Rectangular/circular/coaxial modes (≈300 LOC, needs T1-SPECIAL Bessel)
Topic headline. Pozar Ch. 3. `RectWaveguideCutoff(m,n,a,b,εᵣ,μᵣ)` = (c/2π)·√((mπ/a)²+(nπ/b)²)/√(εᵣμᵣ); `RectWaveguideTE_E(m,n,a,b,x,y,z,ω)` full E/H components (~80 LOC); `RectWaveguideTM_E` dual; `RectWaveguideImpedanceTE` = ωμ/β; `RectWaveguideImpedanceTM` = β/ωε; `RectWaveguideAttenuationTE10(σ_walls,ω)` Pozar 3.96; `CircWaveguideCutoffTE/TM` via Bessel J' / J zeros; `CoaxialWaveguideCutoffTEM` = 0; `GroupVelocity(ω,ω_c,c)` = c·√(1−(ω_c/ω)²); `PhaseVelocity` = c/√(1−(ω_c/ω)²); `GuideWavelength`.

**T1 subtotal: ~2,700 LOC, ~85 functions** — covers every closed-form RF/microwave/antenna/waveguide/Smith consumer use case.

---

## Tier 2 — Depth (well-defined, after T1)

### T2-MATERIAL — Dispersive material models (≈300 LOC)
Pozar §1.4, Jackson §7.5. `DrudeModel(ω_p,γ,ω)` = 1−ω_p²/(ω²+jωγ) (metals); `LorentzOscillator(ω_0,ω_p,γ,ω)` (bound electrons); `LorentzMultipole([]Pole,ω)`; `DebyeRelaxation(εᵣ_∞,εᵣ_s,τ,ω)` (water/tissue); `ColeColeRelaxation` fractional-power; `ColeDavidsonRelaxation`; `HavriliakNegamiRelaxation` α+β; `SellmeierEquation(B,C,λ)` (optical glass); `KramersKronig_RealFromImag` principal-value (~80 LOC); `SkinDepth(σ,μ,ω)` = √(2/μσω); `SurfaceImpedance` = (1+j)/σδ.

### T2-MULTIPOLE — Multipole expansion (≈250 LOC, needs T1-SPECIAL Yₗᵐ)
Jackson Ch. 4, 9. `MultipoleMoments(charges,l_max)` q_lm = ΣqᵢrᵢˡYₗᵐ*(θᵢ,φᵢ) (~60 LOC); `MultipoleField(q_lm,r,θ,φ)` exterior expansion (~80 LOC); `MultipolePotential` V = Σ q_lm Yₗᵐ/r^(l+1)/(2l+1); `DipoleMoment` p = Σqᵢrᵢ; `QuadrupoleMoment` Q_ij = Σqᵢ(3rᵢrⱼ−δᵢⱼr²); `MagneticMultipole_M`; `MagneticDipoleMoment` m = ½∫r×J dV.

### T2-IMAGE — Image method (≈100 LOC)
Griffiths §3.2. `ImageGroundedPlane(q,h,point)` image at −h with charge −q; `ImageGroundedSphere` Kelvin's inversion q' = −qR/d at d' = R²/d; `ImageDielectricInterface(q,h,εᵣ)` scaled images.

### T2-CONFORMAL — 2D conformal mapping (≈200 LOC)
For 2D Laplace with non-trivial geometry. `SchwarzChristoffel(vertices,angles,w)` (~120 LOC SC parameter problem); `LogarithmicMap`; `JoukowskiMap` airfoil; `BilinearMap` Möbius; `CapacitancePerLength_2D(boundary)` via SC for polygonal cross-sections.

### T2-PLASMA-DISPERSION — Plasma waves (≈200 LOC)
Jackson Ch. 7, Stix. `PlasmaFrequency(n_e)` = √(n_e e²/ε₀ m_e); `CyclotronFrequency` = qB/m; `WhistlerDispersion(ω,ω_p,ω_c,θ)`; `OrdinaryWaveDispersion` n²=1−ω_p²/ω²; `ExtraordinaryWaveDispersion`; `FaradayRotation(B_∥,n_e,L,ω)`; `BirefringenceCalcite(n_o,n_e,θ)`; `HelmholtzWavenumber` k=ω√(εμ); `WaveImpedance` η=√(μ/ε).

### T2-FIBER — Optical fiber modes (≈250 LOC, needs T1-SPECIAL Bessel)
`StepIndexFiber_V(a,n_c,n_cl,λ)` V-number = (2πa/λ)·√(n_c²−n_cl²); `StepIndexFiber_LP01_neff` Bessel root-finding (~80 LOC); `StepIndexFiber_NA` = √(n_c²−n_cl²); `GRIN_ParabolicProfile`; `FiberGroupDelay`; `FiberDispersion` D = −(λ/c)·d²n/dλ².

### T2-SCATTERING — Mie, RCS, Fresnel, geometric optics (≈400 LOC, needs T1-SPECIAL)
`MieScattering(x,m,θ)` Bohren-Huffman series with spherical Bessel + Legendre (~150 LOC); `MieEfficiencyQ_ext/sca/back`; `RayleighScattering` small-particle limit; `RCSSphere_PEC(a,λ)` Mie series; `RCSPlate_Flat` PO/GO; `GeometricOpticsRay(point,dir,surfaces)` Snell at planar/spherical (~100 LOC); `FresnelCoefficients(θ_i,n1,n2)` (r_s,r_p,t_s,t_p complex) TE/TM amplitude (~30 LOC); `BrewsterAngle` = atan(n2/n1); `CriticalAngle` = asin(n2/n1).

### T2-S2P — Network parameter conversions (≈400 LOC)
scikit-rf bread-and-butter. `S2Z(S,Z_0)` = Z₀(I+S)(I−S)⁻¹ via LU-solve per 061 §F-6 (~60 LOC); `Z2S`; `S2Y/Y2S/S2T/T2S/Z2Y/Y2Z`; `ABCD2S/S2ABCD`; `CascadeABCD([]networks)` matrix product; `CascadeT`; `RenormalizeS(S,Z_old,Z_new)`; `MixedModeS` differential/common-mode; `Stability_K(S)` Rollett (K>1 ∧ |Δ|<1 → unconditional); `Stability_Mu(S)` Edwards-Sinsky.

**T2 subtotal: ~2,100 LOC** — adds dispersive/scattering/S-parameter depth matching scikit-rf.

---

## Tier 3 — Research / large engineering

### T3-FDTD — Yee finite-difference time-domain (≈2,500 LOC)
Topic headline, single largest item. **061 §F-2 binding: CFL precondition validated, error returned, no silent clamp.**

- `FDTD1D(grid,dx,dt,steps,source)` E_x/H_y staggered Yee (~150 LOC); `FDTD2D_TE`/`FDTD2D_TM` (~300 LOC each); `FDTD3D` full Yee (~600 LOC).
- `PMLLayer1D/2D/3D` Berenger split-field PML (~400 LOC) — required, first-order Mur ABC unacceptable.
- `CPML`/`UPML` convolutional/uniaxial PML (modern default).
- Material: Drude/Lorentz dispersive update via ADE or recursive convolution (~300 LOC).
- `TFSF_Source` total-field/scattered-field plane-wave injection (~200 LOC).
- `NearField2FarField(E_surf,H_surf,obs)` equivalence-theorem (~250 LOC).
- Stability: every step validates CFL = c·Δt/Δx ≤ 1/√d, returns error. Test commitments per F-2.

**Cross-language port note:** ~250 vectors of 3D field grids, ~MB-scale per primitive. Multi-month even for one author.

### T3-MOM — Method of Moments (≈1,500 LOC, needs T1-SPECIAL + T1-FIELDS)
Foundation of NEC2/FEKO/4NEC2. `EFIE_Galerkin_RWG(triangles,freq)` Rao-Wilton-Glisson (~600 LOC); `MFIE_Galerkin_RWG`; `CFIE_Galerkin_RWG(α)` avoids interior resonances; `WirePocklington(segments,freq)` thin-wire integral equation (~250 LOC); `WireHallen` alternative; `MutualImpedance_Wire(seg_a,seg_b,freq)` 5-pt Gauss (~150 LOC); `MoMSolveCurrents(Z,V)` linalg LU; `MoMFarField(currents,segments,θ,φ,freq)` (~80 LOC); `Yagi_Uda_Analyze(elements,lengths,spacings,freq)` (~200 LOC composition).

### T3-FEM-EM — Vector-element FEM (≈1,800 LOC)
For waveguide eigenmode + driven scattering. `Nedelec_Edge_Element_2D` first-order, 6 DOFs/triangle (~300 LOC); `Nedelec_Edge_Element_3D` tet, 6/12/20 DOFs (~500 LOC); `WaveguideEigenSolve_2D(mesh,εᵣ,μᵣ)` Av=λBv via inverse iteration (~300 LOC, depends 061 §F-5); `Driven_2D_FEM(mesh,source,ABC)` (~400 LOC); first-order ABC or PML (~300 LOC).

### T3-BEM — Boundary element method (≈1,000 LOC, needs T2-MULTIPOLE + T1-SPECIAL)
`BEM_Capacitance_Extract(meshes,εᵣ)` full N-conductor capacitance matrix (~400 LOC); `BEM_Inductance_Extract` partial/loop; `FMM_BEM(...)` fast-multipole acceleration for N>10⁴ (~500 LOC, optional/Tier 3+).

### T3-BPM — Beam propagation (≈600 LOC)
Optical fibers/integrated photonics, paraxial Helmholtz. `BPM_FFT_2D(εᵣ_grid,λ,dz,steps)` split-step Fourier (~250 LOC, depends `signal/fft`); `BPM_FD_2D` finite-diff variant; `WideAnglePadeBPM` Padé(2,2); `EigenmodeExpansion(modes,z,ω)` for piecewise-uniform structures (~200 LOC).

### T3-RAY-TRACE — Geometric optics for high-frequency (≈400 LOC)
For radar/wireless propagation, λ ≪ feature size. `Ray3D` struct + intersections; `RaySphereIntersect`; `RayTriangleIntersect` Möller-Trumbore; `RayPlaneIntersect`; `Reflect(d,n)` = d−2(d·n)n; `Refract(d,n,n1,n2)` Snell with TIR flag; `UTD_DiffractionCoefficient(α_in,α_out,wedge,k)` Keller/Kouyoumjian-Pathak (~150 LOC); `GTD_DiffractionCoefficient`.

### T3-SPECIALIZED-ANTENNAS (≈500 LOC composition)
`LoopAntenna_Field` (uses K, E elliptic); `LoopAntennaImpedance`; `HelicalAntenna_Axial` Kraus; `HornAntenna_Pyramidal(a,b,ρ_e,ρ_h,...)` Fresnel integrals; `ParabolicReflector_FarField(D,F,feed)` aperture integration (~150 LOC); `MicrostripPatch_FarField(W,L,h,εᵣ,...)` cavity model (~120 LOC); `LeakyWaveAntenna`; `FractalAntenna_Sierpinski`.

### T3-HYSTERESIS-MAGNETIC (≈300 LOC)
Nonlinear magnetic. `JilesAthertonModel(M_s,a,k,c,α)` anhysteretic + hysteresis loop, state-bearing/path-dependent (~200 LOC); `PreisachModel(grid)` arbitrary minor loops (~150 LOC); `BHCurveLookup(table,H)` interpolated (B,dB/dH); `MagneticEnergyLoss(loop)` enclosed area.

### T3-POISSON-LAPLACE (≈800 LOC)
`PoissonSolve_2D_Rect(grid,source,bc)` finite-diff Gauss-Seidel + Multigrid (~400 LOC); `LaplaceSolve_2D_Rect`; `PoissonSolve_3D_Rect`; `PoissonSolve_FFT` periodic spectral (~50 LOC over `signal/fft`); `LaplaceSolve_BEM_2D` for non-rectangular domains.

**T3 subtotal: ~9,400 LOC** — multi-month FDTD/MoM/FEM/BEM/BPM lift, takes reality into MEEP/NEC2/openEMS territory on the math axis.

---

## Peer-library coverage map (web research 2026-05-07 baseline)

| Library | Reachable from our scope |
|---|---|
| **MEEP** (MIT, FDTD) | T3-FDTD covers ~70% of math; symmetry exploitation + GPU OOS. |
| **openEMS** (FDTD) | T3-FDTD ~80%; cylindrical coords +~600 LOC. |
| **scikit-rf** (RF networks) | T1-COMPLEX + T2-S2P ~85%; calibration kits TRL/SOLT ~300 LOC additional, in-scope. |
| **NEC2** (wire MoM) | T3-MOM wire path ~90%; Sommerfeld ground integration ~400 LOC bonus. |
| **FEKO** (commercial MoM/FDTD/UTD) | T3-MOM + T3-RAY-TRACE ~50%; characteristic-mode analysis ~300 LOC additional. |
| **COMSOL ACDC** (commercial FEM) | T3-FEM-EM covers math; mesh adaptivity is engineering. |
| **MoMentum** (Keysight, 2.5D MoM PCB) | T3-MOM stratified Green's functions ~600 LOC additional, Tier 3. |

Reachable in 12-18mo: ~80% scikit-rf, ~70% MEEP/openEMS, ~50-60% FEKO/COMSOL on math. IR/GPU/mesh-adaptivity/CAD-interop correctly OOS.

---

## LOC roadmap

| Tier | Bundles | LOC |
|---|---|---|
| **T1** | COMPLEX + SPECIAL + FIELDS + TLINES + SMITH + DIPOLE + ANTENNA-METRICS + ARRAY + WAVEGUIDE | **~2,700** |
| **T2** | MATERIAL + MULTIPOLE + IMAGE + CONFORMAL + PLASMA + FIBER + SCATTERING + S2P | **~2,100** |
| **T3** | FDTD + MOM + FEM-EM + BEM + BPM + RAY-TRACE + SPECIALIZED-ANTENNAS + HYSTERESIS + POISSON-LAPLACE | **~9,400** |
| **TOTAL** | | **~14,200 LOC** |

T1 alone takes `em/` from 11 → ~85 functions and covers every closed-form RF/microwave/antenna/waveguide consumer use case. T2 makes it competitive with scikit-rf. T3 reaches Meep/NEC2/openEMS territory on math while remaining zero-dep, deterministic, golden-tested across 4 languages.

**Most compelling 1-week first ship:** T1-COMPLEX + T1-TLINES + T1-SMITH (~650 LOC, ~150 golden vectors). Solves "design a coax/microstrip/stripline at f and read VSWR/Γ/Smith position." No special-functions dependency.

**Most compelling foundational ship:** T1-SPECIAL (~600 LOC of Bessel/elliptic/Legendre/Yₗᵐ/Si/Ci/Ei) — alone unlocks ~70% of T1+T2 follow-on work.

**Most compelling research-tier ship:** T3-FDTD-1D + PML (~600 LOC) — validates F-2 CFL-precondition contract and establishes golden-vector machinery for the larger 2D/3D FDTD lift.

---

## Cross-package coupling

- **`linalg/`**: T2-S2P needs LU-solve (F-6); T3-FEM-EM needs eigensolvers (F-5); T3-MOM needs LU/QR for dense moment matrices; T3-FDTD CPML needs sparse if implicit time-stepping (defer).
- **`signal/`**: T3-BPM uses FFT directly; T3-FDTD spectral source generation uses windowing. Existing `signal/fft.go` sufficient.
- **`calculus/`**: T1-ANTENNA-METRICS::DirectivityFromPattern needs 2-sphere quadrature → Lebedev grids ~100 LOC in `calculus/` per 017's scope; T3-MOM mutual-impedance needs adaptive Gauss-Kronrod (017 T1-AdaptiveGaussKronrod).
- **`geometry/`**: T3-RAY-TRACE composes with `geometry/sdf.go` ray-marching + `geometry/quaternion.go` rotations; T3-MOM RWG basis composes with triangle-mesh primitives.
- **`constants/`**: μ₀, ε₀, c, e, m_e present. Add: Bohr magneton μ_B, nuclear magneton μ_N, classical electron radius r_e.
- **`autodiff/`**: optional, enables gradient-based RF-circuit optimization for T1-TLINES + T2-S2P. Composition only.
- **`optim/`**: Yagi-Uda design uses L-BFGS / GA via T3-MOM gradient-of-gain. Composition only.

---

## Numerical-correctness commitments inherited from 061

Every primitive above must respect F-1..F-6 before first ship:

- **F-1 (superposition):** all multi-source field functions ship pairwise default + explicit Kahan variant. Naive `+=` rejected.
- **F-2 (FDTD CFL):** every Yee step returns error on CFL violation. Silent-clamp rejected.
- **F-3 (elliptic):** Carlson RF/RD form, not series-with-branches. Required for current-loop axial B and finite-wire fields.
- **F-4 (complex impedance):** native `complex128`. Reject `(real, imag)` float64 pair API.
- **F-5 (LC-ladder eigenmodes):** companion-matrix QR not Durand-Kerner.
- **F-6 (S↔Z conversion):** LU-solve `(I−S)X = (I+S)`, not naive `(I−S)⁻¹`.

These are blocking design decisions; per CLAUDE.md §1 first ship across Go/Python/C++/C# binds the algorithm choice. If T1 lands with naive variants, retrofitting Kahan/Carlson/companion-QR breaks all 4 ports' golden files simultaneously.

---

## Test-coverage commitments

CLAUDE.md §1 mandates ≥20 vectors/function (target 30):

- **T1 (~85 functions):** ~1,700 vectors → ~600 LOC JSON.
- **T2 (~50 functions):** ~1,000 vectors → ~350 LOC JSON.
- **T3 (~70 functions, vector-output):** ~2,100 vectors with full 3D field grids → ~1,800 LOC JSON. T3-FDTD vectors are ≥30 frames × 3D grids, MB-scale per primitive.

Mandatory edges: ω=0 DC limit on every freq-domain function; ω→∞ asymptote on dispersives; |Γ|=1 (open/short/match) on RF; m→1⁻ logarithmic divergence on elliptic; kr≪1 near and kr≫1 far on dipole; ω=ω_c cutoff on every waveguide mode (group velocity → 0); |z|=1 boundary on Smith chart.

---

## Non-overlap statement

- **061 em-numerics** owns numerical hazards in present 213-LOC surface + F-1..F-6 algorithm-choice commitments. 062 references those as binding, does not re-derive.
- **063 em-sota** will own peer-library deep-dive (MEEP grid types, scikit-rf naming, openEMS dispersive material API). 062 names libraries only by capability scope.
- **064 em-api** will own Go-signature shapes (Vec3 vs [3]float64, error-return shape, naming consistency). 062 sketches signatures only enough for plausible LOC estimates.
- **065 em-perf** will own per-call cycle counts, SIMD posture, FDTD inner-loop allocation budget. 062 mentions perf only via F-2 CFL-precondition error-return (correctness, not performance).

---

## Bottom line

`em/` is currently 11 scalar functions in 213 LOC — a freshman EM exam cheat-sheet. Topic prompt names FDTD, MoM, FEM-EM, Smith chart, transmission line, antenna patterns, dipole radiation, waveguide modes; **zero of the eight exist**. Reaching them is ~14,200 LOC across 30+ Tier 1/2/3 bundles, gated on four missing layers (`complex128`, vector fields, special functions, PDE kernels). Tier 1 (~2,700 LOC, ~85 functions) goes from "EM 101" to "RF/antenna/waveguide closed-form toolkit" — single highest-leverage commitment. Tier 2 (~2,100 LOC) adds dispersive/scattering/S-parameter depth matching scikit-rf. Tier 3 (~9,400 LOC) is the multi-month FDTD/MoM/FEM/BEM/BPM lift taking reality into MEEP/NEC2/openEMS territory on math, while remaining zero-dep, deterministic, and golden-validated across 4 languages.
