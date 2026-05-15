# 067 — fluids-missing

**Topic:** fluids: missing — Navier-Stokes, LBM, SPH, vorticity-streamfunction, Spalding, Colebrook iterations, Moody, Karman-Tsien.

**Premise from 066:** the `fluids/` package is 236 LOC of "fluid-mechanics 101 cheat-sheet" — eleven scalar functions (`ReynoldsNumber`, `BernoulliPressure`, `PipeFlowFriction`, `DarcyWeisbach`, `DragForce`, `LiftForce`, `TerminalVelocity`, `StokesLaw`, `MassFlowRate`, `VolumetricFlowRate`) over `float64`. **Zero** PDE solvers, **zero** boundary-layer integrators, **zero** compressible primitives, **zero** turbulence closures, **zero** open-channel hydraulics, **zero** multiphase, **zero** vector fields, **zero** mesh data structures. The package is ~2-3% of the surface that "fluid mechanics" canonically denotes (White, Anderson, Pope, Ferziger-Perić, Versteeg-Malalasekera, Wesseling, Toro). This audit enumerates what is missing, scoped against `reality`'s "math primitives, no I/O, zero deps, golden-file validated, MIT" constraints (CLAUDE.md §1–§6) and against the F-1..F-7 forward-looking commitments named in 066.

**Scope filter — what counts as in-scope for `reality/fluids`:**
- IN: closed-form engineering correlations (Colebrook iterations, Swamee-Jain, Haaland, Churchill, Karman-Tsien, Prandtl-Glauert, Moody fits).
- IN: ODE-class boundary-layer integrals (Blasius, Falkner-Skan, Thwaites, Head, Spalding wall function).
- IN: compressible scalar relations (isentropic, normal shock, oblique shock, Prandtl-Meyer, Rayleigh, Fanno, Mach angle).
- IN: open-channel hydraulics (Manning, Chezy, critical depth, hydraulic jump, specific energy).
- IN: aerodynamic closed forms (thin-airfoil α-pattern, Glauert series, Joukowski transform, lifting-line collocation).
- IN: PDE solver kernels — Navier–Stokes 1D/2D/3D, vorticity-streamfunction Poisson, LBM D2Q9/D3Q19 collide-stream, SPH neighbour kernels, FV Riemann (Godunov, HLL, HLLC, Roe), MUSCL+slope-limiter, WENO5, VOF advection, level-set re-init, immersed-boundary forcing — deterministic, allocation-disciplined, golden-testable.
- IN: turbulence model evaluators (k-ε, k-ω-SST, Spalart-Allmaras source-term math; Smagorinsky `(C_s Δ)²|S|`).
- IN: special-functions needed by fluids (erf for Blasius, Lambert W for Spalding, modified Bessel for Stokes-2nd-problem).
- BORDERLINE: mesh generation — geometry primitives in-scope (already in `geometry/`), full unstructured mesher likely outside `reality`.
- OUT: GPU/CUDA/threading, OpenFOAM polyMesh I/O, VTK/Tecplot/CGNS serializers, particle-tracking visualization, GUI plotting.

---

## Headline

`fluids/` ships 11 of an estimated **220–290 canonical fluid-mechanics primitives** across White-Anderson-Pope-Ferziger-Toro-Versteeg textbook + OpenFOAM/SU2/Nek5000/PyFR/scipy.optimize/lattice-Boltzmann-libraries (Palabos, OpenLB, ESPResSo) frontline. The eight topic-listed headlines (Navier-Stokes, LBM, SPH, vorticity-streamfunction, Spalding, Colebrook, Moody, Karman-Tsien) require **7,500–11,500 LOC across 5 missing layers**, all absent today: (1) **vector-field surface** (currently scalar — every Navier-Stokes operator is vector-/tensor-valued), (2) **structured-grid PDE primitives** (no FV, no Poisson, no advection-diffusion), (3) **compressible surface** (currently incompressible-only — every Mach-aware fluid problem unreachable), (4) **boundary-layer integrators** (every wall-bounded engineering flow uses one), (5) **special functions** specific to fluids (Lambert W, complete erf-derivatives). **Zero** of the eight headlines are reachable without (1) + (2) at minimum.

**Cheapest first-light:** Karman-Tsien / Prandtl-Glauert compressibility correction (~30 LOC, no special-functions, immediate aerodynamics utility). Mach number + isentropic ratios next at ~60 LOC. **Hardest:** 3D incompressible Navier-Stokes with k-ω-SST (~3,000 LOC, multi-month). **Highest reuse:** Mach/isentropic/shock family (~400 LOC) unlocks compressible aerodynamics, supersonic flow, nozzle design, jet/rocket simultaneously.

Recommended sprint: T1-COMPRESSIBLE → T1-COLEBROOK-FAMILY → T1-OPEN-CHANNEL → T1-AIRFOIL → T1-BLAYER-EXPLICIT → T1-MULTIPHASE → T1-WALL-FUNCTIONS → T2-BLAYER-INTEGRAL → T2-SHOCK-PRESSURE → T2-LIFTING-LINE → T3-NS-1D → T3-NS-2D → T3-VORT-STREAM → T3-LBM-D2Q9 → T3-SPH-2D → T3-LBM-D3Q19 → T3-NS-3D → T3-WENO/MUSCL → T3-VOF / level-set / IBM / turbulence closures. Tier-1 items are independent ≤1-week chunks; tier-3 are multi-month.

---

## Tier 1 — Foundation (blocks every topic primitive)

### T1-COMPRESSIBLE — Mach, isentropic ratios, normal/oblique shock, Prandtl-Meyer (≈350 LOC, biggest reuse)
Unblocks every Mach-aware fluid problem. Anderson Ch. 7-9. Functions:
- `MachNumber(v, c) = v/c` and `SpeedOfSoundIdealGas(γ, R, T) = √(γRT)`; `SpeedOfSoundFromBulk(K, ρ) = √(K/ρ)`.
- Isentropic ratios `IsentropicTemperatureRatio(M, γ) = 1/(1+(γ-1)/2 M²)`; `IsentropicPressureRatio` = `T_ratio^(γ/(γ-1))`; `IsentropicDensityRatio` = `T_ratio^(1/(γ-1))`; `AreaRatio(M, γ) = A/A*` for nozzle design.
- Normal-shock relations `NormalShockMachDownstream(M1, γ)` Rankine-Hugoniot; `NormalShockPressureRatio`; `NormalShockDensityRatio`; `NormalShockTemperatureRatio`; `NormalShockStagPressureLoss(M1, γ)`.
- Oblique shock `ObliqueShockBetaThetaM(M1, β, γ) → θ` and inverse `ObliqueShockBetaFromTheta(M1, θ, γ, weak bool)` requires Newton root-find in `[asin(1/M), π/2]`.
- `PrandtlMeyerNu(M, γ) = √((γ+1)/(γ-1)) atan√((γ-1)/(γ+1)(M²-1)) - atan√(M²-1)` for expansion-fan turning angle; inverse `PrandtlMeyerMFromNu` Newton iteration.
- `MachAngle(M) = asin(1/M)` for `M ≥ 1`, NaN otherwise.
- `RayleighFlowT0Ratio(M, γ)`, `RayleighFlowPressureRatio` for heat-addition Rayleigh-line.
- `FannoFlowT_Ratio(M, γ)`, `FannoFlowPressureRatio`, `FannoFlow4fLstarOverD(M, γ)` for adiabatic-friction Fanno-line.
- `KarmanTsienPressureCorrection(C_p_inc, M_inf, γ)` = `C_p_inc / (√(1-M²) + (M²(1+(γ-1)/2 M²)/(2√(1-M²))) C_p_inc)` — explicit topic-prompt headline; supersedes Prandtl-Glauert below `M=0.7`.
- `PrandtlGlauertCorrection(C_p_inc, M_inf) = C_p_inc / √(1-M²)` for `M < 0.7`; document discontinuity vs Karman-Tsien.
- `GöthertCorrection` (3D Prandtl-Glauert generalization); `LaitoneCorrection` extends Karman-Tsien with γ.
- Per-function precision pin: closed-form ratios at exact-bit, Newton-iterated `M(ν)` to `1e-10` relative.

### T1-COLEBROOK-FAMILY — Explicit and high-precision Colebrook variants (≈250 LOC)
Closes 066 N-1, N-2 simultaneously. Topic prompt explicitly headlines Colebrook iterations + Moody.
- `ColebrookFrictionFactor(Re, ε/D)` — high-precision Newton iteration on `1/√f = -2 log₁₀(ε/(3.7D) + 2.51/(Re√f))`, relative tolerance `1e-12`, NaN on cap-exhaust (replaces today's `PipeFlowFriction` interior).
- `SwameeJainExplicit(Re, ε/D)` = `0.25 / log₁₀(ε/(3.7D) + 5.74/Re^0.9)²` valid `5×10³ ≤ Re ≤ 10⁸`, `10⁻⁶ ≤ ε/D ≤ 10⁻²`, ±1% vs Colebrook.
- `HaalandExplicit(Re, ε/D)` = `(-1.8 log₁₀((ε/3.7D)^1.11 + 6.9/Re))^(-2)` ±2% vs Colebrook, 2× faster than Swamee–Jain.
- `ChurchillExplicit(Re, ε/D)` = `8·((8/Re)^12 + 1/(A+B)^1.5)^(1/12)` valid all Re including laminar and transitional — closes 066 N-1 cliff in one call.
- `SerghidesExplicit(Re, ε/D)` 3-step convergent, ±0.0023% vs Colebrook (best explicit).
- `ChenExplicit`, `ZigrangSylvesterExplicit`, `RomeoRoyoMonzonExplicit` for cross-language port matching against scipy/NIST tables.
- `MoodyChartFit(Re, ε/D)` — public alias preferred form (Churchill); document inline that "Moody chart" is not a single equation but a graphical compendium of Colebrook/Hagen-Poiseuille.
- `LaminarFlowFriction(Re) = 64/Re` (split-out from current branched code).
- `RelativeRoughness(material)` named-constant table: drawn copper `1.5e-6 m`, commercial steel `4.6e-5`, galvanized iron `1.5e-4`, cast iron `2.6e-4`, concrete `3e-4 to 3e-3`, riveted steel `9e-4 to 9e-3` — engineering-decision values from Moody 1944, ASHRAE.
- `PipeFlowRegime(Re) → enum {Laminar, Transitional, Turbulent}` thresholds 2300 / 4000.
- All variants share golden-file vectors so cross-language ports and reality's Go agree to spec ULP.

### T1-OPEN-CHANNEL — Manning, Chezy, critical depth, hydraulic jump, specific energy (≈250 LOC)
Currently absent entirely. Chow "Open-Channel Hydraulics" 1959.
- `ManningVelocity(R_h, S, n)` = `(1/n)·R_h^(2/3)·√S` (SI); `ManningFlow(A, R_h, S, n) = A·V`; document English-units `(1.486/n)`.
- `ChezyVelocity(R_h, S, C)` = `C·√(R_h·S)` (predecessor of Manning); `ChezyFromManning(n, R_h)` = `R_h^(1/6)/n`.
- `HydraulicRadius(A, P) = A/P`; `HydraulicDiameter(A, P) = 4A/P` for non-circular pipe Reynolds.
- `RectangularChannelHydraulicRadius(b, y) = by/(b+2y)`; `TrapezoidalChannelHydraulicRadius(b, y, m)`; `CircularChannelHydraulicRadius(D, y)` requires arc-segment math; `ParabolicChannelHydraulicRadius`.
- `CriticalDepthRectangular(q, g)` = `(q²/g)^(1/3)`; `CriticalDepthGeneral(Q, A, T, g)` Newton on `Fr=1` where `Fr = Q²T/(gA³)`.
- `FroudeNumber(v, L, g)` = `v/√(gL)`; `FroudeNumberChannel(v, A, T, g)`.
- `SpecificEnergy(y, v, g) = y + v²/(2g)`; `SpecificEnergyDiagram` returns `(y_min, E_min)` for plotting.
- `SubcriticalFlow(Fr) bool`, `SupercriticalFlow(Fr) bool`.
- Hydraulic jump: `HydraulicJumpDepthRatio(y1, Fr1) = y2/y1 = ½(√(1+8Fr1²) - 1)` Bélanger; `HydraulicJumpEnergyLoss(y1, y2) = (y2-y1)³/(4 y1 y2)`; `HydraulicJumpType(Fr1)` enum {Undular, Weak, Oscillating, Steady, Strong}.
- `BackwaterDirectStep(y0, S0, n, Q, b)` step integration of gradually-varied flow ODE `dy/dx = (S0-Sf)/(1-Fr²)`.
- `GradVariedFlowProfile` enum classification {M1,M2,M3,S1,S2,S3,C1,C3,H2,H3,A2,A3} per Chow.
- `WeirRectangular(L, H, Cd)` = `(2/3)·Cd·L·√(2g)·H^(3/2)`; `WeirVNotch(θ, H, Cd) = (8/15) Cd √(2g) tan(θ/2) H^(5/2)`; `OrificeFlow(Cd, A, ΔP, ρ)`.

### T1-AIRFOIL — Thin-airfoil theory, lift slope, drag polar, Joukowski (≈300 LOC)
Today `LiftForce` takes user-supplied `Cl`. Anderson "Fundamentals of Aerodynamics" Ch. 4-5.
- `ThinAirfoilLiftCoefficient(α)` = `2π·sin(α)` ≈ `2π·α` small-angle; `ThinAirfoilMomentCoefficient(α)` about quarter-chord = 0; `ThinAirfoilZeroLiftAngle(camberFn)` Glauert-series integral.
- `GlauertCoefficients(camberFn, n_max)` = `[A_0, A_1, ...]` Fourier series for general camber; `LiftCoefficientFromGlauert(A_0, A_1, α)` = `π(2A_0 + A_1)`; `MomentCoefficientFromGlauert`.
- `JoukowskiTransform(ζ, a)` = `ζ + a²/ζ` complex map; inverse `JoukowskiInverse`; `JoukowskiAirfoilCirculation(α, R, β)` Kutta condition.
- `KuttaCondition(γ_TE) = 0` enforced in panel solvers.
- `EllipticalLiftDistribution(s, b, Cl_total)` minimum-induced-drag spanwise; `InducedDragCoefficient(Cl, AR, e) = Cl²/(π·AR·e)` Oswald factor.
- `DragPolarParabolic(Cl, Cd0, AR, e) = Cd0 + Cl²/(π·AR·e)`.
- `LiftSlope3D(a_2D, AR)` = `a_2D / (1 + a_2D/(π·AR))` Helmbold; `LiftSlope3D_DATCOM` = `2π·AR / (2 + √(AR²(1-M²)/η² + 4))`.
- `MaxLiftCoefficient_Stall` table entries by airfoil family (NACA 4-digit, 5-digit, 6-series) — pure numerics, no I/O.
- `NACA4DigitCamberLine(m, p, x)` and `NACA4DigitThickness(t, x)` parametric airfoil generation; `NACA5DigitCamberLine`; `NACA6SeriesProfile`.
- `CdSphereSchillerNaumann(Re) = 24/Re·(1+0.15·Re^0.687)` (066 §F-2 explicit); `CdSphereCliftGauvin` extends to drag-crisis `Re≤2×10⁵`; `CdSphereMorrison2013` single all-Re fit; `CdCylinderRoshko` Re-table; `CdFlatPlateLaminar(Re_L) = 1.328/√Re_L` Blasius; `CdFlatPlateTurbulent(Re_L) = 0.074/Re_L^(1/5)` Prandtl 1/7-law.
- Closes 066 N-3 fix path: pluggable `CdFn` consumer for `TerminalVelocityIterative`.

### T1-BLAYER-EXPLICIT — Blasius and Falkner-Skan closed-form features (≈250 LOC)
Schlichting "Boundary Layer Theory" Ch. 7.
- `BlasiusBoundaryLayerThickness(x, Re_x) = 5.0·x/√Re_x`; `BlasiusDisplacementThickness = 1.721·x/√Re_x`; `BlasiusMomentumThickness = 0.664·x/√Re_x`; `BlasiusWallShearStress(ρ, U, x, Re_x) = 0.332·ρU²/√Re_x`; `BlasiusDragCoefficient(Re_L) = 1.328/√Re_L`.
- `BlasiusVelocityProfile(η)` = numerical `f'(η)` from Blasius equation `2f''' + f·f'' = 0` solved via shooting (RK4 from `calculus/`); pre-tabulate η in `[0,10]` step 0.05; `BlasiusF`, `BlasiusFp`, `BlasiusFpp`.
- `FalknerSkanProfile(β, η)` similarity for wedge flow `U = U_∞·x^m`, `β = 2m/(m+1)`; same shooting structure.
- `FalknerSkanSeparationBeta = -0.1988` exact.
- `TurbulentBoundaryLayer_OneSeventh(x, Re_x)`: `δ = 0.37·x·Re_x^(-1/5)`, `Cf = 0.058·Re_x^(-1/5)`, `θ = 0.036·x·Re_x^(-1/5)` Prandtl-7th-power.
- `SchlichtingTransition_Re_x_crit(Tu)` = function of free-stream turbulence intensity, ~5×10⁵ default (smooth).
- `MichelTransitionCriterion(Re_θ, Re_x)` for empirical transition prediction.
- `CrocoBusemann(M, T_w, T_∞)` compressible boundary-layer recovery.

### T1-WALL-FUNCTIONS — Spalding, log-law, Reichardt (≈150 LOC)
Topic-prompt explicit headline. Pope "Turbulent Flows" Ch. 7.
- `SpaldingWallFunction(u_plus)` = `y_plus(u_plus) = u_plus + e^{-κB}·(e^{κu_plus} - 1 - κu_plus - (κu_plus)²/2 - (κu_plus)³/6)` single-equation valid all `y_plus`; inverse via Newton or Lambert-W.
- `LogLawProfile(y_plus, κ, B) = (1/κ)·ln(y_plus) + B` valid `30 < y_plus < 300`, `κ=0.41`, `B=5.0` smooth wall.
- `LinearSublayerProfile(y_plus) = y_plus` valid `y_plus < 5`.
- `BufferLayerProfile(y_plus)` blend `5 < y_plus < 30` (Reichardt single-equation).
- `ReichardtProfile(y_plus)` = `(1/κ)·ln(1+0.4·y_plus) + 7.8·(1 - e^{-y_plus/11} - (y_plus/11)·e^{-y_plus/3})` C∞ blend.
- `VanDriestDamping(y_plus, A=26)` = `1 - e^{-y_plus/A}` for mixing-length closure.
- `RoughnessFunctionΔB(k_s_plus)` Nikuradse for rough-wall log-law shift.
- `WallShearStressFromProfile(u_τ, ρ) = ρ·u_τ²`; `FrictionVelocity(τ_w, ρ) = √(τ_w/ρ)`; `YPlus(y, u_τ, ν) = y·u_τ/ν`; `UPlus(u, u_τ) = u/u_τ`.
- All inverses (`u_plus_from_y_plus`) ship with relative-1e-10 Newton.

### T1-MULTIPHASE-CLOSED — Cavitation, settling, two-phase, surface tension (≈200 LOC)
- `CavitationNumber(p, p_v, ρ, v) = (p - p_v)/(½ρv²)`; `CavitationOnsetPressure(p_v, ρ, v)`; `BernoulliCavitationCheck(p1, v1, v2, ρ, p_v) bool`.
- `WeberNumber(ρ, v, L, σ) = ρv²L/σ`; `BondNumber(ρ_l, ρ_g, g, L, σ) = (ρ_l-ρ_g)g L²/σ`; `EötvösNumber` alias; `CapillaryNumber(μ, v, σ)`; `OhnesorgeNumber(μ, ρ, σ, L)`; `MortonNumber`.
- `SettlingVelocityStokes(d, ρ_p, ρ_f, μ, g) = (ρ_p-ρ_f)g d²/(18μ)` Re<1; `SettlingVelocityIntermediate(d, Re, ...)` Schiller–Naumann iteration; `SettlingVelocityNewton(d, ρ_p, ρ_f, g) = √(3.03·(ρ_p-ρ_f)g d/ρ_f)` Re>1000; `TerminalVelocityFixedPointCdRe(...)` 066 §F-5.
- `LockhartMartinelliMultiplier(X, regime)` two-phase pressure-drop multiplier; `LockhartMartinelliParameter(...)`; `FriedelTwoPhase`; `MullerSteinhagenHeck`.
- `HomogeneousMixtureDensity(α, ρ_l, ρ_g)` = `α·ρ_g + (1-α)·ρ_l`; `VoidFraction(ṁ_g, ṁ_l, ρ_g, ρ_l, S)` slip-ratio.
- `LaplacePressureSphere(σ, R) = 2σ/R` for droplet; `LaplacePressureBubble = 4σ/R` (two interfaces); `YoungLaplaceCapillaryRise(σ, θ, ρ, g, r) = 2σcos(θ)/(ρgr)`.
- `ContactAngle(γ_sg, γ_sl, γ_lg)` Young's equation.

### T1-DIMENSIONLESS — Complete the dimensionless-number library (≈100 LOC)
Currently only `Re`. Add: `Mach`, `Froude`, `Prandtl(μ, c_p, k) = μc_p/k`, `Schmidt(ν, D) = ν/D`, `Lewis(α, D) = α/D`, `Peclet(Re·Pr)`, `Grashof(g, β, ΔT, L, ν) = gβΔTL³/ν²`, `Rayleigh(Gr·Pr)`, `Nusselt`, `Sherwood`, `Stanton(Nu/(Re·Pr))`, `Eckert(v²/(c_p·ΔT))`, `Strouhal(fL/v)`, `Womersley(R√(ω/ν))`, `Knudsen(λ/L)`, `Bejan`, `Brinkman`, `Dean`, `RichardsonGradient`. All exact-arithmetic.

**T1 subtotal: ~1,850 LOC, ~110 functions** — covers every closed-form engineering / aerodynamics / hydraulics / cavitation / boundary-layer / compressibility consumer use case. Closes 066 forward-looking F-1, F-2, F-3, F-4, F-5, F-7. Six topic-prompt explicit headlines (Colebrook, Moody, Karman-Tsien, Spalding) all reachable inside Tier-1.

---

## Tier 2 — Depth (well-defined, after T1)

### T2-BLAYER-INTEGRAL — Thwaites, Head, Stratford (≈250 LOC)
- `ThwaitesIntegral(U_e(x), x_arr) → θ(x), λ(x)` momentum-integral via `θ² = (0.45ν/U_e^6)·∫U_e^5 dx` (numerical via `calculus/`); `ThwaitesShapeFactor(λ)` Cebeci-Bradshaw fit; `ThwaitesSeparationCriterion(λ < -0.09)`.
- `HeadEntrainmentMethod(...)` turbulent boundary-layer integral coupling H₁ entrainment ODE — two coupled ODEs for `θ` and `H` via RK4.
- `WhiteCfFit(Re_θ)` Cf for turbulent BL.
- `StratfordSeparationPrediction(C_p, dC_p/dx)`.
- `HeadShapeFactor(H)` closure `H₁(H) = 3.3 + 0.8234(H-1.1)^-1.287` for H<1.6.
- `ColesWakeProfile(η, Π, κ)` log-law + wake bump.

### T2-SHOCK-AND-NOZZLE — Compressible flow extensions (≈250 LOC)
- `ConicalShock(M_∞, σ)` Taylor-Maccoll ODE numerical (RK4 in `θ`).
- `MOC2DSupersonic(boundary)` Method-of-Characteristics 2D supersonic nozzle/jet.
- `ConvergingDivergingNozzle(M_e_design, p0, T0, A_t, γ, R)` exit conditions, mass-flow `ṁ_choked = (p0·A_t/√T0)·√(γ/R)·(2/(γ+1))^((γ+1)/(2(γ-1)))`.
- `ChokedMassFlow(p0, T0, A_t, γ, R)` standalone.
- `NozzleEfficiency`, `NozzleThrust(ṁ, v_e, p_e, p_∞, A_e)`.
- `ShockTubeRiemann(p4, p1, T4, T1, γ4, γ1)` Sod-problem closed-form (used as golden-file generator for T3 Riemann-solver tests).
- `RankineHugoniotJump(state_left, state_right, γ)` general, returns shock speed.
- `RayleighPitotFormula(M, γ)` for supersonic Pitot probe.

### T2-LIFTING-LINE — Prandtl 3D wing, panel methods (≈300 LOC)
- `PrandtlLiftingLine(chord(y), α(y), b, U_∞, n)` Fourier-series collocation via `linalg.Solve` on dense `n×n` system.
- `EllipticalWingMinimumDrag(b, AR, Cl)` analytic.
- `OswaldEfficiency(AR, Λ, taper)` empirical fit.
- `Glauert3DCorrection(α_2D, AR, e)` to lift slope.
- `VortexPanelMethod2D(panels, α)` source-vortex panels with Kutta condition; `LinearVortexPanel`; `ConstantSourcePanel`.
- `XFOIL-style PanelSolve2D` (without viscous-coupling — that's T3).
- `HessSmithPanel(panels)` purely-source potential flow.

### T2-POROUS-AND-MULTIPHASE — Darcy, Brinkman, Forchheimer (≈150 LOC)
- `DarcyVelocity(K, ΔP/L, μ) = -K/μ·∇P`; `DarcyPermeabilityKozenyCarman(d, ε)`.
- `BrinkmanEquation(K, μ, μ_eff, v, ∇P)`.
- `ForchheimerExtension(α, β, v, ρ, μ)` non-Darcy regime.
- `RichardsEquation` unsaturated flow soil-physics.
- `BuckleyLeverettSaturation(S_w, f_w)` two-phase oil-water reservoir.
- `CapillaryPressureBrooksCorey(S_e, p_d, λ)`.

### T2-TURBULENCE-CLOSED — Algebraic closures (≈200 LOC)
- `MixingLengthPrandtl(y, κ=0.41) = κy`; `Cebeci-Smith` two-layer.
- `BaldwinLomax(y, ω, |ω|_max, y_max)` zero-equation algebraic — full math.
- `Smagorinsky(C_s, Δ, S_ij)` = `(C_s·Δ)²·|S|`, `|S| = √(2 S_ij S_ij)` SGS eddy viscosity.
- `WaleModel(C_w, Δ, S_ij, Ω_ij)` improved near-wall.
- `DynamicSmagorinsky(test_filter)` Germano identity.
- `K_OmegaSourceTerm`, `K_EpsilonSourceTerm`, `K_OmegaSST_BlendingF1`, `SpalartAllmarasNutTilde_Source` — pure source-term math (no PDE solve, that's T3).

### T2-VAR-FLOWS — Stokes 2nd problem, oscillatory flow, Womersley (≈150 LOC)
- `StokesFirstProblem(y, t, U, ν) = U·erfc(y/(2√(νt)))` Rayleigh-impulsive plate.
- `StokesSecondProblem(y, t, U_0, ω, ν)` complex amplitude `U_0·e^{-y/δ}·cos(ωt - y/δ)`, `δ = √(2ν/ω)`.
- `WomersleyFlow(r, t, ω, ρ, μ, ∂P/∂x)` complex `J_0` Bessel solution for arterial flow — needs `T1-SPECIAL J_0/J_1` from `em/` package or local fluids-internal.
- `CouetteFlow(y, U, h)` linear; `CouetteFlowPressureGradient(y, h, dP/dx, μ)` with non-zero pressure gradient.
- `PoiseuilleFlow_Pipe(r, R, dP/dx, μ)` parabolic; `PoiseuilleFlow_Channel(y, h, dP/dx, μ)`.
- `HagenPoiseuilleVolumetricFlow(R, dP/dx, μ) = πR⁴·ΔP/(8μL)` (currently only embedded as `64/Re`; promote standalone).

**T2 subtotal: ~1,300 LOC, ~70 functions.**

---

## Tier 3 — Heavy machinery (multi-month, blocks topic headlines NS/LBM/SPH)

### T3-NS-1D — 1D Navier-Stokes / Burgers / Euler (≈600 LOC)
Precursors before 2D/3D. Toro "Riemann Solvers" §10.
- `Burgers1D_Inviscid(u, dt, dx, scheme)` — Godunov, Lax-Wendroff, MacCormack, MUSCL.
- `Burgers1D_Viscous(u, ν, dt, dx)` — central-diff viscous + upwind convective.
- `Euler1D_Godunov(W, dt, dx)` exact-Riemann solver per cell.
- `Euler1D_HLL(W, dt, dx)` Harten-Lax-van Leer — cheaper than exact, robust.
- `Euler1D_HLLC(W, dt, dx)` HLL+Contact restoration — restores stationary contact discontinuity.
- `Euler1D_Roe(W, dt, dx)` Roe-averaged flux Jacobian.
- `Euler1D_LaxFriedrichs`, `Euler1D_LaxWendroff`, `Euler1D_RusanovLLF`.
- Slope limiters: `MinMod`, `Superbee`, `VanLeer`, `MC`, `Koren`, `VanAlbada` — 6 limiters needed for MUSCL + WENO weights.
- `MUSCL_Reconstruction(left, center, right, limiter)` second-order.
- `WENO5_Reconstruction(stencil[5])` fifth-order non-oscillatory weights `α_k`, `ω_k`.
- `RungeKuttaSSP3(state, RHS, dt)` strong-stability-preserving time integration (Shu-Osher).
- `CFLCondition(u_max, dx) → dt_max`.

### T3-NS-2D — 2D structured-grid Navier-Stokes (≈900 LOC)
Ferziger-Perić Ch. 6-9. Builds on T3-NS-1D building-blocks.
- `IncompressibleNS2D_Projection(u, v, p, dt, dx, dy, ν)` Chorin projection / fractional step.
- `PoissonSolve2D(rhs, bc, method)` SOR / multigrid / FFT — bridges to `signal/` FFT for periodic BC.
- `VorticityStreamFunction2D(ψ, ω, dt, ν, dx, dy)` — topic-prompt explicit headline; pseudocode `∇²ψ = -ω` solve + advection-diffusion of `ω`.
- `StaggeredGridMAC2D` Marker-And-Cell layout (u at i+½/j, v at i/j+½, p at cell center).
- `ConvectionScheme_Upwind1`, `_QUICK`, `_CentralDiff` evaluators.
- `ImmersedBoundaryDirectForcing(u, v, body)` IBM forcing term.
- `SemiLagrangianAdvection(field, u, v, dt, dx)` for VOF/level-set.
- `BoundaryConditions{NoSlip, FreeSlip, Inflow, Outflow, Periodic}` enum + math.
- `RhieChowInterpolation` for collocated-grid checkerboard prevention.
- `SIMPLE_PressureCorrection`, `PISO_PressureCorrection` segregated solvers.

### T3-NS-3D — 3D incompressible NS (≈1200 LOC)
Pope Ch. 13. Direct extension of T3-NS-2D with 3D spectral / staggered MAC. Adds k-ω-SST, Spalart-Allmaras, Smagorinsky LES production. Cost-class similar to T3-FDTD-3D in `em/`.

### T3-LBM — Lattice Boltzmann D2Q9, D3Q19 (≈700 LOC)
Topic-prompt explicit headline. Krüger "The Lattice Boltzmann Method" 2017.
- `D2Q9Weights()` `[4/9, 1/9×4, 1/36×4]`; `D2Q9DirectionVectors()` `(0,0)+(±1,0,±1)`.
- `D3Q19Weights`, `D3Q19DirectionVectors`, plus `D3Q15`, `D3Q27` lattices.
- `BGKCollision(f, f_eq, τ)` `f' = f - (f - f_eq)/τ`.
- `MRTCollision(f, f_eq, M, S)` multi-relaxation-time — better stability than BGK.
- `RegularizedLBM`, `EntropicLBM` for high-Re stability.
- `EquilibriumDistribution_D2Q9(ρ, u, v)` `f_eq = w_i·ρ·(1 + 3·c_i·u + 9/2·(c_i·u)² - 3/2·u²)`.
- `StreamingStep(f, lattice)` neighbour shifting in pre-collision distribution.
- `BounceBackBC(f, wall)` no-slip; `HalfwayBounceBack`; `ZouHePressureBC`, `ZouHeVelocityBC`.
- `LBMMomentumExtraction(f) → (ρ, u, v)`; `LBMStressTensorFromF(f)` non-equilibrium part.
- `KnudsenLayerCorrection(τ, kn)` slip-flow regime.

### T3-SPH — Smoothed Particle Hydrodynamics (≈600 LOC)
Topic-prompt explicit headline. Monaghan 2005 review.
- `SPHKernelGaussian(r, h)`, `SPHKernelCubicSpline(r, h)`, `SPHKernelWendlandC2(r, h)`, `SPHKernelQuintic(r, h)` — four standard kernels with continuous first derivative.
- `SPHKernelGradient(r_vec, h, kernel)`.
- `SPHDensitySummation(particles, h)` Σ m_j W(r_ij, h).
- `SPHContinuityEquation(particles)` `dρ/dt = Σ m_j (v_i - v_j)·∇W`.
- `SPHMomentumEquation(particles, EOS)` `dv/dt = -Σ m_j (P_i/ρ_i² + P_j/ρ_j² + Π_ij) ∇W`.
- `ArtificialViscosityMonaghan(α, β, c, h)` Π_ij stabilization.
- `EquationOfStateTait(ρ, ρ_0, c_0, γ=7)` weakly-compressible `P = (ρ_0·c_0²/γ)((ρ/ρ_0)^γ - 1)`.
- `SPHSurfaceTension(particles)` color-gradient method.
- `SPHBoundaryGhost`, `SPHBoundaryRepulsive`.
- `NeighbourListGridHash(positions, h)` O(N) neighbour finding (depends on `geometry/` spatial hash).
- `KernelCorrectionShepard`, `KernelGradientCorrectionMLS`.

### T3-VOF-LEVELSET-IBM — Interface tracking and immersed boundaries (≈700 LOC)
Topic-prompt headlines.
- `VOFAdvectionDonorAcceptor(F, u, v, dt, dx, dy)` Hirt-Nichols.
- `VOFGeometric_PLIC` piecewise-linear interface reconstruction.
- `VOFGeometric_CICSAM`, `_HRIC`, `_inter-Gamma` modern compressive schemes.
- `LevelSetReinit(φ, dx, n_steps)` Sussman re-initialization PDE.
- `LevelSetCurvature(φ)` `κ = ∇·(∇φ/|∇φ|)`.
- `FastMarchingMethod(speed, source)` Sethian for narrow-band level set.
- `ImmersedBoundary_Peskin(u, body, F)` direct forcing with regularised Dirac delta.
- `GhostFluidMethod(φ, P, ρ)` sharp interface.
- `CoupledLevelSetVOF` — CLSVOF.

### T3-TURBULENCE-PDE — Closure transport equations (≈400 LOC)
- `KEpsilonTransport(k, ε, ν, S, dx, dy, dz, dt)` two-coupled-PDEs.
- `KOmegaSST_Transport(k, ω, F1, F2, ...)`.
- `SpalartAllmaras_Transport(ν̃, S, d, ...)` single-equation.
- `SmagorinskyEddyViscosityField(C_s, Δ, S_ij)` per-cell evaluator.
- `WALE_EddyViscosityField`, `Vreman_EddyViscosityField`, `SigmaModel_EddyViscosityField`.
- `YakhotOrszag_RNG_Coefficients`.

### T3-FV-UNSTRUCTURED — Finite-volume on unstructured cells (≈800 LOC)
- `FVCellTopology` data structures (depends on `geometry/`).
- `FVGradient_GreenGauss`, `_LeastSquares`.
- `FVConvectionTerm`, `FVDiffusionTerm`, `FVSourceTerm` discretizations.
- `RoeSolver_Unstructured`, `HLLC_Unstructured`.
- `MUSCL_Unstructured` reconstruction.
- `BoundaryConditionPatch` ABM.

### T3-OPTIMIZATION-COUPLED — Adjoint shape optimization (≈300 LOC)
Out-of-scope per CLAUDE.md "no I/O" but adjoint math is in-scope:
- `ContinuousAdjoint_NS` adjoint Navier-Stokes RHS.
- `DiscreteAdjoint_FD_Verification`.
- `SensitivityShapeDerivative` Hadamard formula.

**T3 subtotal: ~6,200 LOC, ~85 functions.**

---

## Cross-package dependencies (forward-looking)

| Tier item | Depends on |
|---|---|
| T1-COMPRESSIBLE | `constants/` (γ, R), nothing else |
| T1-COLEBROOK-FAMILY | `optim/` Newton (already exists) |
| T1-OPEN-CHANNEL | `optim/` Newton for critical-depth |
| T1-AIRFOIL | nothing |
| T1-BLAYER-EXPLICIT | `calculus/` RK4 (exists) for Blasius shooting |
| T1-WALL-FUNCTIONS | `optim/` Newton or Lambert-W (new in `combinatorics/`?) |
| T1-MULTIPHASE | nothing |
| T2-BLAYER-INTEGRAL | `calculus/` Simpson + RK4 |
| T2-LIFTING-LINE | `linalg/` Solve (exists) |
| T2-VAR-FLOWS | new Bessel `J_0/J_1` (T1-SPECIAL in 062 em-missing) |
| T3-NS-* | `linalg/` SparseCSR + GMRES (062 §T3 dependency, **does not exist yet**) |
| T3-LBM | nothing (matrix-free) |
| T3-SPH | `geometry/` spatial-hash (check whether exists; likely new) |
| T3-VOF/LSM | `geometry/` SDF primitives (exist in `geometry/`) |
| T3-TURBULENCE-PDE | T3-NS-* solvers |

Critical bottleneck: **T3-NS gates on `linalg.SparseCSR` + iterative solvers**, identical dependency to em-FDFD/FEM in 062. Recommend a single linalg-sparse sprint that unblocks both em/ and fluids/ simultaneously.

---

## What's missing per topic-prompt headline

| Topic-prompt item | Tier | Reachable today? |
|---|---|---|
| Navier-Stokes 1D/2D/3D | T3-NS-1D / -2D / -3D | No — needs vector/grid surface |
| Lattice Boltzmann (D2Q9, D3Q19) | T3-LBM | No — needs `[]float64` grid surface |
| SPH (Smoothed Particle Hydrodynamics) | T3-SPH | No — needs particle data structure + spatial hash |
| Vorticity-streamfunction | T3-NS-2D | No — needs Poisson solver |
| Spalding wall function | T1-WALL-FUNCTIONS | **Yes — single equation, ≤30 LOC** |
| Colebrook iterations (high precision) | T1-COLEBROOK-FAMILY | **Yes — replaces 066 N-1, N-2 in 50 LOC** |
| Moody chart | T1-COLEBROOK-FAMILY | **Yes — alias of Churchill 1977** |
| Karman-Tsien | T1-COMPRESSIBLE | **Yes — single equation, ~20 LOC** |

Four of eight headlines are immediate Tier-1 wins (Spalding, Colebrook precision, Moody, Karman-Tsien). The other four are multi-month T3 efforts.

---

## Recommended order of operations

**Sprint 1 (1 week, ~600 LOC):** T1-COMPRESSIBLE + T1-COLEBROOK-FAMILY + T1-WALL-FUNCTIONS — closes 4 of 8 topic-prompt headlines + 066 §F-1, §F-3, §F-7 + Spalding.

**Sprint 2 (1 week, ~600 LOC):** T1-OPEN-CHANNEL + T1-AIRFOIL + T1-BLAYER-EXPLICIT — adds engineering hydraulics + thin-airfoil + Blasius/Falkner-Skan.

**Sprint 3 (1 week, ~400 LOC):** T1-MULTIPHASE + T1-DIMENSIONLESS + T2-VAR-FLOWS — closes 066 §F-5 (terminal-velocity iterative) and adds Stokes-2nd / Womersley / Hagen-Poiseuille standalone.

**Sprint 4 (2 weeks, ~700 LOC):** T2-BLAYER-INTEGRAL + T2-SHOCK-AND-NOZZLE + T2-LIFTING-LINE — depth on aerodynamics and compressible.

**Sprint 5+ (multi-month):** linalg-sparse sprint (shared with em/), then T3-NS-1D → T3-NS-2D → T3-LBM-D2Q9 → T3-SPH-2D → T3-VOF/LSM/IBM → T3-NS-3D → T3-LBM-D3Q19 → T3-turbulence-PDE.

---

## Web-research notes

- **OpenFOAM** ships ~150 incompressible/compressible solvers (`icoFoam`, `simpleFoam`, `pisoFoam`, `pimpleFoam`, `rhoCentralFoam`, `interFoam` VOF, `dnsFoam`, `compressibleInterFoam`, `chtMultiRegionFoam` conjugate heat transfer); reality only needs the **mathematical kernels** under `src/finiteVolume/`, not the application drivers (which would be I/O / case-setup, out-of-scope).
- **scipy.optimize** Colebrook implementations call `brentq` on the residual; reality should ship Newton (faster + analytic derivative) but golden-validate against scipy's bisection bracket result to ULP for cross-language seam.
- **Palabos / OpenLB** lattice-Boltzmann libraries provide ~30 lattice schemes; reality should ship D2Q9 + D3Q19 + D3Q27 only and explicitly cite-and-skip multi-component / thermal LBM.
- **PySPH** SPH framework provides ~10 particle methods; reality should ship Monaghan's standard SPH + WCSPH (weakly-compressible) + δ-SPH (delta-SPH) and skip incompressible-SPH (ISPH = Poisson solve, expensive).
- **SU2** discrete-adjoint CFD; reality should ship **continuous adjoint math** (closed-form), skip discrete-adjoint AD-based code (depends on `autodiff/`, possible Tier-3 follow-on).
- **scikit-aero / aerocalc** aerospace closed-forms; reality T1-COMPRESSIBLE matches their function list.
- **Anderson "Modern Compressible Flow"** Appendix A-D provides the canonical golden-file source for normal/oblique-shock and Prandtl-Meyer tables — reality should generate to 256-bit `math/big` and validate down-cast.

---

## Numerical-correctness commitments forward (extends 066 §F)

- **F-8 — Compressible-floor.** All compressible primitives must accept `γ` as runtime parameter (not hard-coded 1.4) and document γ-monatomic=5/3, γ-diatomic-air=1.4, γ-water-vapour=1.33, γ-helium=5/3 in package doc. Constants from `constants/` package.
- **F-9 — Boundary-layer ODE shooting.** Blasius `2f''' + ff'' = 0` shooting must use shooting-on-`f''(0)` with bracket `[0.4, 0.5]`, target precision `1e-12` on `f'(η_∞) = 1`, η_∞=10. Pre-tabulate result at `η = 0, 0.05, 0.1, ..., 10` for 200 points; ship as exported lookup with cubic-spline interpolation.
- **F-10 — LBM lattice constants.** D2Q9 weights `4/9, 1/9, 1/36` are exact rationals; ship as `const` literals not float-derived computations to avoid bit-drift across architectures. Lattice speed `c_s² = 1/3` exact.
- **F-11 — Karman-Tsien validity bounds.** Document `M_∞ ≤ 0.85` as physical limit; above which the correction tends toward singular. Compare to Prandtl-Glauert `M_∞ ≤ 0.7` and Göthert.
- **F-12 — SPH kernel-gradient continuity.** Quintic kernel preferred for second-derivative continuity; cubic-spline preferred for compactness. Document continuity class on each kernel (`C^k` for k = 0/1/2/4).
- **F-13 — Manning unit-system.** Ship two named functions `ManningVelocity_SI` and `ManningVelocity_English` with the 1.486 unit factor explicit in the latter — never silent unit conversion.
- **F-14 — Riemann-solver tolerances.** Sod shock-tube golden-file pinned to ULP at output time `t=0.2`, dx=0.01 (Toro standard). Cross-validate at least three solvers (Godunov-exact, HLLC, Roe) against shared analytic answer.

---

## What would change if I had write access (Sprint 1 only)

1. Add `fluids/compressible.go` (~400 LOC) for T1-COMPRESSIBLE incl. Karman-Tsien (closes topic-prompt headline).
2. Add `fluids/colebrook.go` (~250 LOC) splitting today's `PipeFlowFriction` into `Laminar` + `Colebrook` + `SwameeJain` + `Haaland` + `Churchill` + `Serghides` + `Moody` (closes 066 N-1, N-2 + topic-prompt Colebrook/Moody headlines).
3. Add `fluids/wallfunctions.go` (~150 LOC) for Spalding + log-law + Reichardt (closes topic-prompt Spalding headline).
4. Generate `testdata/fluids/{compressible,colebrook_family,wall_functions}.json` golden files at 30-vector minimum each per CLAUDE.md.
5. Update package doc to reflect new surface; add cross-references to T2/T3 forward-looking layers.

Total estimated effort for Sprint 1: 1-2 weeks for full implementation + golden files + cross-language Python/C++/C# port.

---

**Auditor:** agent 067
**Status:** complete
**Next:** progress line appended; eight topic-prompt headlines mapped — four are Tier-1 wins, four are Tier-3 multi-month; recommended sprint plan above.
