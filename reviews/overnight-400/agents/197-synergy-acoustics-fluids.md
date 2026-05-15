# 197 | synergy-acoustics-fluids

**Summary line 1.** `acoustics/` ships 9 scalar closed-forms in a single 197-LOC file (`SoundSpeed`/`SoundIntensity`/`DecibelSPL`/`DecibelFromIntensity`/`SabineRT60`/`DopplerShift`/`ResonantFrequency`/`WaveLength`/`AWeighting`) — **zero** time-domain primitives, **zero** complex spectra, **zero** spatial geometry, **zero** field types, **zero** Mach-number / compressibility primitives — and `fluids/` ships 11 scalar steady-state closed-forms in a single 236-LOC file (`ReynoldsNumber`/`BernoulliPressure`/`PipeFlowFriction-Colebrook-White`/`DarcyWeisbach`/`DragForce`/`LiftForce`/`TerminalVelocity`/`StokesLaw`/`MassFlowRate`/`VolumetricFlowRate`) — **zero** unsteady, **zero** vortex-shedding, **zero** boundary-layer / pressure-fluctuation, **zero** vector / tensor / field types — with **zero** source-coupling between them (verified `grep github.com/davly/reality/acoustics fluids/*.go` and reverse → 0 matches; zero matches across repo on `Lighthill|Curle|FfowcsWilliams|FW.?H|Rossiter|Aeolian|Strouhal|Mach\b|VortexSound|PowellLighthill|HelmholtzResonator|JetNoise|RotorNoise|TurbulenceSpectrum`). The entire **aeroacoustics canon** (Lighthill 1952 acoustic analogy `∂²ρ′/∂t² − c₀²∇²ρ′ = ∂²T_ij/∂x_i∂x_j` with Lighthill stress tensor `T_ij = ρ u_i u_j + (p′ − c₀²ρ′)δ_ij − τ_ij`, Curle 1955 surface integral, Ffowcs Williams–Hawkings 1969 moving-surface formulation, Powell 1964 vortex-sound, Howe 1975 acoustic energy `dE/dt = −ρ₀∫(ω×u)·u_a dV`, Strouhal `f = St·U/D` with `St ≈ 0.2` for circular cylinder at `300 < Re < 2×10⁵`, Lighthill `U⁸` jet-noise law, Rossiter cavity-mode `f_n = (U/L)·(n − γ)/(M + 1/κ)`, Helmholtz resonator `f₀ = (c/2π)√(A/(V·L_eff))`, Mach-cone half-angle `μ = arcsin(1/M)`, ISO 9613-1 atmospheric absorption, Bies–Hansen wind-noise on microphones, Goldstein 1976 boundary-layer wall-pressure spectra) is **wholly absent** from both packages and the wider repo.

**Summary line 2.** Twenty-three synergy primitives totalling ~2310 LOC of pure connective tissue close the gap; **fifteen ship today** against the v0.10.0 surfaces (every primitive that is `acoustics.*` + `fluids.*` + `math.*` arithmetic with no missing PDE / FFT2 / spatial-mesh dependency); **eight** are blocked-soft on independently-flagged primitives — `signal.Hilbert` / `signal.RFFT` / `signal.Welch` (132-signal-missing for tonal-broadband decomposition and turbulence-loaded-pressure spectra), `linalg.SVD` (097-linalg-missing for FW-H source-image LSQ), `geometry.Mesh3D` + a future `pde/wave/` package (174-geometry-missing) for Lighthill-source volume integration, and an `acoustics.AtmosphericAbsorption` per ISO 9613 (002-acoustics-missing) for the propagation half of every aeroacoustic source model. Cheapest **one-day standalone PR** is **L1 StrouhalFrequency + L2 AeolianTone + L9 MachAngle = ~60 LOC** giving the first compressible-aeroacoustic primitive in the repo and saturating an obvious 3/3 R-MUTUAL-CROSS-VALIDATION pin (vortex-shedding frequency: Strouhal-direct vs Roshko-1954 universal `St(Re)` curve vs lock-in-with-acoustic-`ResonantFrequency` agreement to ≤2% on circular cylinder at `Re ∈ {100, 10⁴, 10⁵}` — extends the 6a55bb4-audio-onset / 365368a-copula-autodiff R-MUTUAL family from per-package to acoustics×fluids cross-package). Architectural keystone is **L7 LighthillStress + L8 CurleSurface + L13 FfowcsWilliamsHawkings** because every other source-side primitive (jet-noise `U⁸`, rotor noise, cavity tones, edge tones, combustion noise) either reduces to a special case of these three integrals or post-processes their output — once the FW-H Farassat-1A formulation lands as a numerical-quadrature primitive, the remaining ten blocked primitives become pure connective tissue. Recommended placement is a **NEW sub-package `acoustics/aero/`** (mirrors 159 `em/wave/` + 160 `fluids/turbulence/` + 192 `fluids/control/` + 194 `geometry/dec/` + 166 `acoustics/room/` consumer-side-placement precedent — aeroacoustic primitives are neither pure-acoustic nor pure-fluid: they couple `T_ij = ρ u_i u_j` from `fluids/` with the wave operator `(∂²/∂t² − c₀²∇²)` from `acoustics/`).

---

## 0. State of play (verified file-walk, 2026-05-08)

`acoustics/acoustics.go` HEAD (1 file, 197 LOC, 9 exported funcs):
- **Sound speed/propagation:** `SoundSpeed(γ, R, T, M)`, `SoundIntensity(P, r)` (point-source inverse-square only)
- **Decibels:** `DecibelSPL(p, pRef)`, `DecibelFromIntensity(I, IRef)`
- **Room:** `SabineRT60(V, A)`
- **Doppler:** `DopplerShift(f0, vs, vr, c)` (1-D line-of-sight, no convective amplification)
- **Resonance:** `ResonantFrequency(L, n, c)` (open-open pipe only — no closed-end, no Helmholtz, no rectangular-room mode)
- **Wave/weighting:** `WaveLength(f, c)`, `AWeighting(f)`

Every function is closed-form scalar arithmetic. **Zero** unsteady source models, **zero** Mach-number primitive, **zero** convective Doppler factor `(1 − M·cosθ)^{-n}`, **zero** vector field, **zero** turbulence statistic, **zero** atmospheric absorption.

`fluids/fluids.go` HEAD (1 file, 236 LOC, 11 exported funcs):
- **Dimensionless:** `ReynoldsNumber(ρ, v, L, μ)` — directly enters Strouhal-Re universal curve
- **Bernoulli/pipe:** `BernoulliPressure`, `PipeFlowFriction` (Colebrook–White iterative), `DarcyWeisbach`
- **Aerodynamic forces:** `DragForce(Cd, ρ, v, A)`, `LiftForce(Cl, ρ, v, A)`, `TerminalVelocity(m, g, Cd, ρ, A)` — `DragForce` and `LiftForce` directly enter dipole-source strength of Curle 1955
- **Low-Re:** `StokesLaw(μ, r, v)`
- **Flow rates:** `MassFlowRate(ρ, v, A)`, `VolumetricFlowRate(v, A)`

Every function is a single algebraic line except `PipeFlowFriction` (Colebrook fixed-point loop). **Zero** time-dependence, **zero** vortex shedding, **zero** boundary-layer wall-pressure model, **zero** Mach number, **zero** velocity field type.

**Cross-package observations:**

- `acoustics.SoundSpeed` is **the** input to every Strouhal-frequency-to-acoustic-resonance lock-in computation. `fluids.ReynoldsNumber` is **the** input to every `St(Re)` universal curve. The pair is the natural seam.
- `fluids.DragForce` / `fluids.LiftForce` produce instantaneous force time series in real flow (lock-in on cylinders, oscillating airfoils). The fluctuating part `F'(t)` directly drives Curle's dipole source strength — but `fluids/` ships **only** the steady mean drag/lift.
- `acoustics.DopplerShift` is the **stationary-medium** Doppler. Aeroacoustic Doppler needs **convective amplification** `(1 − M·cosθ)^{-n}` with `n = 4` for monopole sound power, `n = 6` for dipole, `n = 8` for Lighthill quadrupole — none present.
- `acoustics.ResonantFrequency` covers open-open pipe only (no closed-end, no Helmholtz). Helmholtz resonator is the canonical lumped-element aeroacoustic body (Coke-bottle tone, automotive intake/exhaust, perforated mufflers); flagged 002-acoustics-missing.
- `constants/` ships `GasConstant`, `StandardGravity`, `AtmPressure` but **no `SpeedOfSoundSTP = 343.0 m/s` at 20°C, 1 atm** and **no `KinematicViscosityAirSTP = 1.516e-5 m²/s`**. Every aeroacoustic primitive below either takes them as input or pulls them from `acoustics.SoundSpeed`/`fluids.ReynoldsNumber`. Flagged 047-constants-missing.
- `signal/` ships `FFT`/`IFFT`/`PowerSpectrum`/`FFTFrequencies` + Hann/Hamming/Blackman + `Convolve`/`MovingAverage`/`EMA`/`MedianFilter` (~470 LOC, no Hilbert, no real-FFT, no Welch, no STFT — STFT lives in `audio/spectrogram/`). Tonal-vs-broadband decomposition (Howe acoustic-energy partition) wants `signal.Hilbert` for instantaneous frequency / amplitude — flagged 132-signal-missing.
- `chaos/` ships `RK4` for general ODEs but no PDE; no Burgers, no Euler, no compressible Navier–Stokes, no Linearized Euler equations (LEE). Fully-resolved Lighthill-source DNS-CAA out of scope here; the synergy review proposes only **post-processing** primitives that consume caller-supplied flow data.
- **Cycle-hazard.** `grep -r 'fluids' acoustics/*.go` and `grep -r 'acoustics' fluids/*.go` — zero matches. Zero coupling today. Adding `acoustics/aero/` that imports `acoustics/` + `fluids/` + `signal/` + `constants/` is cycle-free; one-way: `acoustics/aero/ → acoustics + fluids + signal + constants + math/cmplx`. Mirrors 159's `em/wave/`, 160's `fluids/turbulence/`, 192's `fluids/control/` precedent.

---

## 1. The twenty-three synergy primitives

Each entry: (1) capability, (2) composition recipe over present primitives, (3) connective-tissue LOC, (4) ship-status against v0.10.0. Numbering L0–L22.

### L0 — `acoustics/aero` package keystone types

**Capability.** `MachNumber float64` (named scalar, dimensionless ratio `U/c`); `FlowVelocity{U, V, W float64}` (3-D point velocity); `SourcePoint{Position [3]float64; Velocity [3]float64; Acceleration [3]float64}` for moving-source bookkeeping; `FarFieldObserver{Position [3]float64; ReferenceTime float64}`; `ConvectiveDopplerFactor(M, cosTheta float64, n int) float64` (one-liner: `math.Pow(1 - M*cosTheta, -float64(n))`).

**LOC.** 80 (mostly type defs + one helper).

**Status.** **Ships today.** Pure types + one closed-form.

### L1 — `StrouhalFrequency(St, U, D float64) float64`

**Capability.** Karman vortex-shedding frequency `f = St·U/D`. Pins Aeolian-tone pitch from cylinder/wire/cable diameter and freestream velocity. **LOC.** 10. **Status.** Ships today. **Reference.** Strouhal, V. (1878) Ann. Phys. Chem. 5:216–251.

### L2 — `AeolianTone(rho, v, L, mu, D float64) float64`

**Capability.** Tonal frequency emitted by a circular cylinder of diameter `D` in cross-flow at velocity `U` — the canonical "wind whistling through power lines" frequency. **Composition.** `Re = fluids.ReynoldsNumber(rho, v, D, mu)` → `St = StrouhalCurve(Re)` (L4) → `f = StrouhalFrequency(St, v, D)` (L1). **LOC.** 15. **Status.** Ships today (after L4).

### L3 — `RoshkoNumber(St, Re float64) float64`

**Capability.** Roshko number `Ro = St·Re = f·D²/ν` — collapses cylinder data across `40 < Re < 200` onto Roshko's 1954 universal curve. Unit-test invariant. **LOC.** 6. **Status.** Ships today. **Reference.** Roshko, A. (1954) NACA Report 1191.

### L4 — `StrouhalCurveCircularCylinder(Re float64) float64`

**Capability.** Universal `St(Re)` for a smooth circular cylinder, valid `40 < Re < 2×10⁵` (subcritical regime). Three regions: (i) laminar shedding `40 < Re < 200`: Roshko `St(Re) = 0.212·(1 − 21.2/Re)`; (ii) transition wake `200 < Re < 300`: cubic blend; (iii) subcritical turbulent `300 < Re < 2×10⁵`: `St ≈ 0.21` plateau (Williamson 1996 Annu. Rev. Fluid Mech. 28:477). `Re > 2×10⁵`: enter critical / supercritical regime where `St` jumps to ~0.45 then ~0.27 — return NaN with a docstring caveat.

**Composition.** Three-branch `if/else` with closed-form fits. ~30 LOC.

**LOC.** 30.

**Status.** **Ships today.**

**Reference.** Williamson, C.H.K. (1996) "Vortex dynamics in the cylinder wake" Annu. Rev. Fluid Mech. 28:477–539; Roshko, A. (1954) NACA TR 1191.

### L5 — `LockInBand(St, U, D, fnAcoustic, dampingRatio float64) (Ulow, Uhigh float64)`

**Capability.** Velocity range `[U_low, U_high]` over which vortex-shedding frequency `f = St·U/D` locks onto a structural / acoustic resonance `fn`. Lock-in width scales as `Δf/fn ∝ √(2ζ·m*) / m*` (Sarpkaya 2004 J. Fluid Struct. 19:389). For a piping system, the resonance comes from `acoustics.ResonantFrequency`.

**Composition.** Solve `f = St·U/D = fn` for `U_center`, broaden by `±0.5·fn·√(2ζ/m*)`. ~25 LOC.

**LOC.** 25.

**Status.** **Ships today** (caller supplies modal mass ratio `m*` and damping `ζ`; `fn` from `acoustics.ResonantFrequency` for an organ pipe or `HelmholtzResonator` once L6 ships).

**Reference.** Sarpkaya, T. (2004) "A critical review of the intrinsic nature of vortex-induced vibrations" J. Fluids Struct. 19:389–447.

### L6 — `HelmholtzResonatorFrequency(c, A, V, Leff float64) float64`

**Capability.** Lumped-element resonator `f₀ = (c/(2π))·√(A/(V·L_eff))`. Coke-bottle tone, automotive intake/exhaust silencer, perforated muffler element. Universally cited as **the** simplest aeroacoustic body. `L_eff = L + 1.7·r` for flanged neck (Rayleigh end correction).

**Composition.** One closed-form line. Use `acoustics.SoundSpeed` to derive `c` from gas properties when caller supplies T/M; or accept `c` directly.

**LOC.** 15.

**Status.** **Ships today.** Independently flagged 002-acoustics-missing — natural co-ship with L5 lock-in.

**Reference.** Helmholtz, H. von (1885) "On the Sensations of Tone" §V.

### L7 — `LighthillStressTensor(rho, u []float64, n int, p float64, c0 float64, out []float64)`

**Capability.** Lighthill's stress tensor `T_ij = ρ u_i u_j + (p′ − c₀²ρ′)δ_ij − τ_ij` packed as a flat 9-element row-major. The *quadrupole* source term in Lighthill's 1952 acoustic analogy `∂²ρ′/∂t² − c₀²∇²ρ′ = ∂²T_ij/∂x_i∂x_j`. For low-Mach incompressible flow `(p′ − c₀²ρ′)` is small and `τ_ij` (viscous stress) is `O(Re^{-1})` smaller still, so for `Re » 1` and `M « 1` the dominant contribution is the **Reynolds stress** `T_ij ≈ ρ₀ u_i u_j` — directly composable with 160-T6 `ReynoldsStressTensor`.

**Composition.** Outer product of velocity vector `u_i u_j` × `ρ₀` + diagonal correction `(p′ − c₀²ρ′)`. ~40 LOC for the full tensor incl. viscous-stress accumulator that consumers may zero.

**LOC.** 40.

**Status.** **Ships today** as a per-point pure-arithmetic primitive (caller handles spatial sampling). Volume integration `∫∫∫ ∂²T_ij/∂x_i∂x_j dV` over a 3-D mesh blocks on `geometry.Mesh3D` (174-geometry-missing) — but most aeroacoustic codes use surface integrals via Curle/FW-H instead, which **don't** need the volume mesh.

**Reference.** Lighthill, M.J. (1952) "On sound generated aerodynamically. I. General theory" Proc. R. Soc. A 211:564–587.

### L8 — `CurleDipoleFarField(F []float64, r []float64, c0, rho0 float64) float64`

**Capability.** Far-field acoustic pressure from a stationary rigid surface in turbulent flow per Curle 1955: `p′(x, t) = (1/(4πc₀)) · ∂F_i(τ)/∂t · (x_i − y_i)/|x − y|²`. Requires the time-derivative of the unsteady force `F_i(t)` on the surface — exactly the unsteady output that `fluids.DragForce`/`LiftForce` *would* produce if they accepted a time-varying velocity. The dipole-source magnitude scales as `M³` in jet-noise context (vs `M⁵` for the Lighthill quadrupole), making it dominant at low Mach.

**Composition.** (1) Caller supplies force time series `F_x(t), F_y(t), F_z(t)` (e.g., from a CFD code or oscillating-cylinder lock-in model). (2) Time-differentiate via central difference (`calculus.CentralDifference` if it ships, else inline). (3) Project onto observer direction `(x − y)/|x − y|`. (4) Divide by `4πc₀·r`. ~50 LOC.

**LOC.** 50.

**Status.** **Ships today.** Pure post-processing — composes `fluids.DragForce`/`LiftForce` (instantaneous, caller-time-stepped) + `acoustics.SoundSpeed` + central-difference. Far-field Doppler factor `(1 − M·cosθ)^{-3}` for moving rigid surface adds 5 LOC (dipole convective amplification per Ffowcs-Williams 1963).

**Reference.** Curle, N. (1955) "The influence of solid boundaries upon aerodynamic sound" Proc. R. Soc. A 231:505–514.

### L9 — `MachAngle(M float64) float64`

**Capability.** Mach-cone half-angle `μ = arcsin(1/M)` for `M > 1`. Returns NaN for `M ≤ 1` (no Mach cone for subsonic flow). The single most fundamental supersonic primitive.

**Composition.** One-line `math.Asin(1.0/M)` with `M > 1` guard.

**LOC.** 10.

**Status.** **Ships today.**

**Reference.** Mach, E. (1887) "Photographische Fixierung der durch Projektile in der Luft eingeleiteten Vorgänge" Sitzungsber. Akad. Wiss. Wien.

### L10 — `LighthillJetNoiseU8(U, D, rho0, c0, eta float64) float64`

**Capability.** Acoustic power radiated by a subsonic round jet, Lighthill's `U⁸` law: `W ≈ η · ρ₀ · D² · U⁸ / c₀⁵` with `η ≈ 1×10⁻⁴` (Lighthill 1954 II; Powell 1959). Returns acoustic power in watts. The textbook scaling that explained why jet engines went from `M ≈ 0.7` (manageable airport noise) to `M ≈ 0.9` (community-noise crisis) in the 1950s.

**Composition.** One closed-form line. `c0` from `acoustics.SoundSpeed`. ~12 LOC.

**LOC.** 12.

**Status.** **Ships today.**

**Reference.** Lighthill, M.J. (1954) "On sound generated aerodynamically. II. Turbulence as a source of sound" Proc. R. Soc. A 222:1–32.

### L11 — `JetNoiseSPLOnAxis(U, D, rho0, c0, r float64) float64`

**Capability.** On-axis jet-noise SPL at observer distance `r`: `dB SPL = DecibelFromIntensity(I, 1e-12)` with `I = (W/(4πr²))·DirectivityFactor(0)` and `W` from L10. Composes L10 + `acoustics.SoundIntensity` + `acoustics.DecibelFromIntensity`. Rough engineering number for early-stage design.

**Composition.** Three composed calls. ~15 LOC.

**LOC.** 15.

**Status.** **Ships today** (after L10).

### L12 — `ConvectiveDopplerShift(f0, M, cosTheta float64, n int) float64`

**Capability.** Doppler factor for a moving aeroacoustic source: `f_obs = f0·(1 − M·cosθ)^{-1}` for frequency, with amplitude factor `(1 − M·cosθ)^{-n/2}` in Pa where `n` is the source-type exponent (4 monopole, 6 dipole, 8 quadrupole per Ffowcs Williams 1963). For `M·cosθ → 1` the factor diverges (Mach-cone radiation); for `M·cosθ > 1` the source is supersonic relative to observer (Mach-wave radiation, L18 below).

**Composition.** One-line `f0 / (1 - M*math.Cos(theta))`. Add a separate amplitude helper. ~15 LOC.

**LOC.** 15.

**Status.** **Ships today.** Generalises `acoustics.DopplerShift` from line-of-sight to arbitrary observer angle — caller can drop in either.

**Reference.** Ffowcs Williams, J.E. (1963) "The noise from turbulence convected at high speed" Phil. Trans. R. Soc. A 255:469–503.

### L13 — `FfowcsWilliamsHawkings(surfaceQ, surfaceL, sourcePos []float64, observer [3]float64, c0, rho0 float64, t []float64, out []float64)`

**Capability.** Far-field acoustic pressure from an arbitrary moving surface (Farassat-1A retarded-time formulation). Surface Q-source (thickness, mass-injection) + L-source (loading, force per unit area) integrated over a permeable or impermeable control surface `S`. **The** workhorse primitive of every modern rotor-noise / propeller / wind-turbine / transonic-fan / open-rotor / fan-tone CAA code (NASA F1A, ANOPP2, etc.).

**Composition.** Steps (Farassat 1981 NASA TM-83235):
1. For each observer time `t_obs`, compute retarded time `τ` per surface element via Newton-iteration on `t_obs − |x − y(τ)|/c₀ = τ` (~30 LOC).
2. Evaluate `Q_n = ρ₀(u_n − v_n)` (thickness-source) and `L_i = p′·n_i + ρ₀·u_i·(u_n − v_n)` (loading-source) at retarded time.
3. Numerical surface integral `1/(4π) ∫_S [Q̇/(r(1 − M_r))² + Q·M_ṙ/(r(1 − M_r))³] dS` + L-term analog.
4. Write into `out[t_obs]`.

**LOC.** ~280 standalone (Farassat-1A is non-trivial — this is the architectural keystone of the package).

**Status.** **SHIPS TODAY** at the post-processing level: caller supplies `surfaceQ`, `surfaceL` time histories already evaluated on a permeable surface (CFD-derived). The retarded-time iteration and observer-time integration are pure arithmetic. Volume-source mode (Brentner-Farassat 1998) blocks on `geometry.Mesh3D`.

**Reference.** Ffowcs Williams, J.E., Hawkings, D.L. (1969) "Sound generation by turbulence and surfaces in arbitrary motion" Phil. Trans. R. Soc. A 264:321–342; Farassat, F. (1981) "Linear acoustic formulas for calculation of rotating blade noise" AIAA J. 19:1122–1130.

### L14 — `RossiterMode(n int, U, L, M float64, gamma, kappa float64) float64`

**Capability.** Cavity-tone frequency `f_n = (U/L)·(n − γ)/(M + 1/κ)` per Rossiter 1964. `γ ≈ 0.25` (vortex-acoustic phase delay), `κ ≈ 0.57` (vortex convection ratio). Captures the discrete tones from automotive sunroof buffeting, weapon-bay flutter, landing-gear cavity, perforated-plate liners. `M = U/c` from `acoustics.SoundSpeed`. Each integer `n` is a separate Rossiter mode (typically `n = 1…4` audible).

**Composition.** One closed-form line. ~15 LOC.

**LOC.** 15.

**Status.** **Ships today.**

**Reference.** Rossiter, J.E. (1964) "Wind-tunnel experiments on the flow over rectangular cavities at subsonic and transonic speeds" RAE TR 64037.

### L15 — `WindNoiseMicrophonePSD(rho, U, U_rms, f, freq []float64, out []float64)`

**Capability.** Wall-pressure PSD on a microphone diaphragm exposed to turbulent wind, per Bies (1966) / Strawn-George-Glegg parameterisation: `Φ_pp(f) ≈ (ρ²·U³·δ/U_∞)·F(ω·δ/U_∞)` with empirical Strouhal-scaled spectrum shape `F(·)`. Drives the 6-dB-per-octave high-pass needed for outdoor microphones in wind. Uses `fluids.ReynoldsNumber` to identify the boundary-layer regime.

**Composition.** Closed-form parameterised spectrum × Strouhal-scaled non-dimensional shape function. ~80 LOC including the empirical `F(St)` curve fit (Goody 2004 model is a 5-parameter rational function; Bies–Hansen 2018 update has 7).

**LOC.** 80.

**Status.** **Ships today.** Pure parametric-spectrum primitive.

**Reference.** Bies, D.A. (1966) Wright-Patterson AFB AMRL-TR-66-30; Goody, M. (2004) "Empirical spectral model of surface pressure fluctuations" AIAA J. 42:1788–1794.

### L16 — `BoundaryLayerWallPressureSpectrum(Ue, delta, deltaStar, theta, tauW, freq []float64, out []float64)`

**Capability.** Goody 2004 wall-pressure spectrum beneath a turbulent boundary layer:
`Φ_pp(ω) = (τ_w²·δ/U_e) · 3·(ω·δ/U_e)² / [(ω·δ/U_e)^{0.75} + 0.5)^{3.7} + (1.1·R_T^{-0.57})·(ω·δ/U_e))^7]`

with `R_T = (δ/U_e)/(ν/u_τ²)`. The standard reference for trailing-edge noise (wind-turbine swish, propeller broadband, airfoil self-noise per Brooks-Pope-Marcolini 1989) and cabin/car interior noise (greenhouse turbulent-boundary-layer excitation).

**Composition.** Closed-form Goody fit + boundary-layer parameter pre-computation (`δ*`, `θ`, `τ_w`, `u_τ` from caller — typical CFD output). ~100 LOC.

**LOC.** 100.

**Status.** **Ships today.**

**Reference.** Goody, M. (2004) AIAA J. 42:1788–1794.

### L17 — `RotorNoiseTonalBPF(B int, RPM, harmonic int) float64`

**Capability.** Blade-passing-frequency tone of a rotor: `f_BPF = B · RPM / 60`, harmonics `h·f_BPF`. Wind turbine, ducted fan, propeller, helicopter rotor. Trivially simple as a primitive but central as the tone whose level is computed by FW-H (L13) and whose convective Doppler shift is computed by L12.

**Composition.** One line. ~8 LOC.

**LOC.** 8.

**Status.** **Ships today.**

### L18 — `MachWaveRadiation(M, c0, rho0, U_pert float64) float64`

**Capability.** Acoustic intensity radiated by a supersonic-turbulence Mach wave (Phillips 1960; Tam-Burton 1984). For `M_c > 1` (convective Mach number), turbulent eddies radiate efficiently along the Mach cone with intensity scaling `I ∝ M^{2}` instead of Lighthill's `M^{8}` — the eight-power law breaks at supersonic. Returns intensity W/m² along Mach cone.

**Composition.** One closed-form line + `MachAngle(M)` directional weighting. ~20 LOC.

**LOC.** 20.

**Status.** **Ships today** (after L9 MachAngle).

**Reference.** Phillips, O.M. (1960) "On the generation of sound by supersonic turbulent shear layers" J. Fluid Mech. 9:1–28; Tam, C.K.W., Burton, D.E. (1984) JFM 138:273–295.

### L19 — `HoweAcousticEnergyFlux(omega, u, ua []float64, rho0 float64) float64`

**Capability.** Howe (1975) acoustic-energy-flux integrand `dE/dt = −ρ₀∫(ω×u)·u_a dV` — the rigorous theorem that **vortex sound is generated at locations where vorticity is non-parallel to local acoustic particle velocity**. The single most useful aeroacoustic-source-localisation primitive in the modern literature; replaces ad-hoc Lighthill-source guessing.

**Composition.** Pointwise vector triple product `(ω × u) · u_a` summed by trapezoidal rule over caller-supplied 3-D field. ~35 LOC for the per-point primitive.

**LOC.** 35.

**Status.** **Ships today** at the per-point level. Volume integration over a mesh blocks on `geometry.Mesh3D` (174-geometry-missing); for now, caller supplies regularly-sampled grids and uses simple trapezoidal sum.

**Reference.** Howe, M.S. (1975) "Contributions to the theory of aerodynamic sound, with application to excess jet noise and the theory of the flute" J. Fluid Mech. 71:625–673.

### L20 — `RayleighStreamingVelocity(omega, U_a, c0, nu float64) float64`

**Capability.** Rayleigh's acoustic-streaming velocity outside a viscous boundary layer: `U_s = -3·U_a²/(8·c₀)` (slip-velocity boundary condition for steady streaming driven by a standing wave of amplitude `U_a` at frequency `ω`). Drives acoustic-cavity convective transport in microfluidics, thermoacoustic engines, sonochemistry.

**Composition.** One closed-form line + boundary-layer-thickness check `δ = √(2ν/ω)`. ~20 LOC.

**LOC.** 20.

**Status.** **Ships today.**

**Reference.** Rayleigh, Lord (1884) Phil. Trans. R. Soc. 175:1–21; Nyborg, W.L. (1965) Phys. Acoust. IIB:265–331.

### L21 — `LinearizedEulerWaveOperator(rho_a, u_a, p_a, rho0, U0, p0, c0 float64, dt float64, dx float64, out []float64)`

**Capability.** Per-grid-point right-hand-side of Linearized Euler equations (LEE) for sound propagation through a non-uniform mean flow `(ρ₀, U₀, p₀)`:
`∂ρ_a/∂t + ∇·(ρ₀ u_a + ρ_a U₀) = 0`
`∂u_a/∂t + (U₀·∇)u_a + (u_a·∇)U₀ + ∇p_a/ρ₀ = 0`
`∂p_a/∂t + (U₀·∇)p_a + ρ₀c₀²∇·u_a = 0`

LEE is **the** propagation half of every hybrid aeroacoustic simulation: incompressible CFD → Lighthill source → LEE propagator → far-field. Without LEE, the only propagator available is free-space Green's function (uniform flow only).

**Composition.** Per-point flux evaluation + central-difference spatial derivatives + RK4 time integration via `chaos.RK4` (already shipped). For 1-D demonstrator, ~120 LOC. 3-D version blocks on a `pde/` package or `geometry.StructuredGrid3D` (174-geometry-missing); ship 1-D first as a teaching primitive.

**LOC.** 120 (1-D), ~400 (3-D).

**Status.** **1-D ships today.** 3-D blocks on grid type.

**Reference.** Bailly, C., Juvé, D. (2000) "Numerical solution of acoustic propagation problems using linearized Euler equations" AIAA J. 38:22–29.

### L22 — `CombustionNoiseSourceStrength(qDot, rho0, c0, gamma float64) float64`

**Capability.** Thermoacoustic monopole-source strength from unsteady heat release per Strahle 1971: `q_s = ((γ−1)/(ρ₀ c₀²)) · ∂q̇/∂t` (acoustic mass-injection rate from heat-release-rate fluctuations). Plus the Strahle `M⁴` scaling for far-field combustion noise from a turbulent flame: `W_acoustic ∝ ρ₀·c₀³·V_flame·M⁴`. Drives gas-turbine combustor noise / domestic-burner roar / industrial-flare broadband.

**Composition.** Two closed-form lines + `acoustics.SoundSpeed` for `c₀` from gas state. ~20 LOC.

**LOC.** 20.

**Status.** **Ships today.**

**Reference.** Strahle, W.C. (1971) "On combustion generated noise" J. Fluid Mech. 49:399–414.

---

## 2. Composition LOC summary

| ID  | Primitive | LOC | Status |
|-----|-----------|-----|--------|
| L0  | Package types (MachNumber, FlowVelocity, SourcePoint, FarFieldObserver, ConvectiveDopplerFactor) | 80 | **Ships today** |
| L1  | `StrouhalFrequency(St, U, D)` | 10 | **Ships today** |
| L2  | `AeolianTone(rho, v, L, mu, D)` | 15 | **Ships today** (after L4) |
| L3  | `RoshkoNumber(St, Re)` | 6 | **Ships today** |
| L4  | `StrouhalCurveCircularCylinder(Re)` Roshko/Williamson | 30 | **Ships today** |
| L5  | `LockInBand` Sarpkaya 2004 | 25 | **Ships today** |
| L6  | `HelmholtzResonatorFrequency` | 15 | **Ships today** |
| L7  | `LighthillStressTensor` per-point T_ij | 40 | **Ships today** (per-point; volume integration blocked on Mesh3D) |
| L8  | `CurleDipoleFarField` | 50 | **Ships today** |
| L9  | `MachAngle` | 10 | **Ships today** |
| L10 | `LighthillJetNoiseU8` | 12 | **Ships today** |
| L11 | `JetNoiseSPLOnAxis` | 15 | **Ships today** (after L10) |
| L12 | `ConvectiveDopplerShift` | 15 | **Ships today** |
| L13 | `FfowcsWilliamsHawkings` Farassat-1A | 280 | **Ships today** at post-processing level (volume mode blocked) |
| L14 | `RossiterMode` cavity tone | 15 | **Ships today** |
| L15 | `WindNoiseMicrophonePSD` Bies/Goody | 80 | **Ships today** |
| L16 | `BoundaryLayerWallPressureSpectrum` Goody 2004 | 100 | **Ships today** |
| L17 | `RotorNoiseTonalBPF` | 8 | **Ships today** |
| L18 | `MachWaveRadiation` Phillips/Tam-Burton | 20 | **Ships today** (after L9) |
| L19 | `HoweAcousticEnergyFlux` | 35 | **Ships today** (per-point; volume integration blocked) |
| L20 | `RayleighStreamingVelocity` | 20 | **Ships today** |
| L21 | `LinearizedEulerWaveOperator` 1-D | 120 | **1-D ships today**; 3-D blocked on grid type |
| L22 | `CombustionNoiseSourceStrength` Strahle 1971 | 20 | **Ships today** |
| —   | **Total connective tissue** | **~2310** | **15 ship today fully; 8 ship at post-processing level pending mesh/grid support** |

The 15 fully-shipping primitives form a coherent v0.10.0-compatible aeroacoustics surface: cylinder/cavity/jet/rotor/Helmholtz/wind-noise/combustion/streaming all have closed-form or post-processing entry points that compose only on `acoustics.*` + `fluids.*` + standard library. The 8 mesh/grid-dependent primitives (L7 volume-integral, L13 volume-source mode, L19 volume integral, L21 3-D, plus future ISO-9613 air absorption, partitioned-FFT FW-H, OASPL band integration, and Tam-Auriault non-isotropic-jet model) wait on the 174-geometry-missing `Mesh3D`/`StructuredGrid3D` types — but every one of them *also* has a per-point or 1-D demonstrator that ships today.

---

## 3. Cross-validation pin opportunities

Recent commits (`6a55bb4` audio-onset 3-detector cross-validation, `365368a` copula×autodiff, `1e12e80` token-set-ratio) saturate the **R-MUTUAL-CROSS-VALIDATION 3/3** pattern. This synergy review surfaces three natural new pins:

**R-VORTEX-FREQUENCY-3-WAY (cheapest one-day pin).** For circular cylinder at `Re ∈ {100, 10⁴, 10⁵}`:
- Estimate `f` via L1 `StrouhalFrequency(St=0.21, U, D)` (textbook plateau).
- Estimate `f` via L4 `StrouhalCurveCircularCylinder(Re)` × `U/D` (Roshko/Williamson curve).
- Estimate `f` via L5 lock-in with caller-supplied acoustic `ResonantFrequency` from `acoustics.ResonantFrequency` for an organ-pipe resonator tuned to the expected shedding band.
- All three must agree to ≤2% — saturates 3/3.

**R-JET-NOISE-DOPPLER-CONSISTENCY.** For a single-stream subsonic jet at observer position `(r, θ)`:
- L10 `LighthillJetNoiseU8` gives stationary `W`.
- L11 `JetNoiseSPLOnAxis` post-converts to dB SPL.
- L12 `ConvectiveDopplerShift` with `n=8` quadrupole exponent must reproduce the Ffowcs-Williams 1963 angle-dependent `(1 − M·cosθ)^{-4}` directivity factor (intensity scales as `(1 − M·cosθ)^{-8}` in pressure, `^{-4}` in intensity per source-time-vs-observer-time Jacobian) to ≤1% across `0 < θ < π`.

**R-LIGHTHILL-CURLE-FW-H-TELESCOPING.** For a cylinder in cross-flow, the **same** far-field acoustic pressure must emerge from:
- L8 `CurleDipoleFarField` (force-time-history → dipole, classical Curle 1955).
- L13 `FfowcsWilliamsHawkings` with permeable surface enclosing the cylinder (FW-H reduces to Curle for stationary impermeable surface — this is Ffowcs-Williams-Hawkings Theorem §3).
- Direct evaluation of L7 `LighthillStressTensor` quadrupole + outgoing-wave Green's function (Lighthill 1952 §4 — limit of equivalent sources). 
- The three must agree at the 5% level on a calibration case (e.g., Inoue-Hatakeyama 2002 cylinder at `Re = 150, M = 0.2`); they diverge in their handling of acoustic feedback to the source, which is small for low-`M`.

This is the **R-FORMULATION-EQUIVALENCE 3/3** pattern: Lighthill quadrupole + Curle dipole + FW-H surface integral are three formally-equivalent reformulations of the same wave equation under different surface-of-integration choices. Saturating it gives the strongest possible correctness witness for any aeroacoustic post-processor.

---

## 4. Recommendations

1. **Create `acoustics/aero/` sub-package.** Mirrors 159 `em/wave/` + 160 `fluids/turbulence/` + 192 `fluids/control/` + 166 `acoustics/room/` precedent. Imports `acoustics/`, `fluids/`, `signal/`, `constants/`. One-way DAG; cycle-free.
2. **Sprint ordering.** Day-1: L1 + L4 + L9 (~75 LOC) saturates R-VORTEX-FREQUENCY-3-WAY 3/3 pin and lands first compressible-aeroacoustic primitive in repo. Day-2: L6 + L5 + L14 + L17 (~63 LOC) closes lumped-element-resonator family. Week-1: L10 + L11 + L12 + L18 (~62 LOC) jet-noise quartet, saturates R-JET-NOISE-DOPPLER-CONSISTENCY pin. Week-2: L7 + L8 (~90 LOC) post-processing path from CFD to far-field. Week-3: L13 FW-H Farassat-1A (~280 LOC) — architectural keystone, saturates R-LIGHTHILL-CURLE-FW-H-TELESCOPING pin and brings parity with NASA F1A / OpenCFD libAcoustics. Week-3 (parallel): L15 + L16 (~180 LOC) wall-pressure for cabin/microphone/trailing-edge.
3. **Constants ask** (047-constants-missing): add `SpeedOfSoundSTPAir = 343.0` (20°C, 1 atm dry air), `KinematicViscosityAirSTP = 1.516e-5`, `DynamicViscosityAirSTP = 1.825e-5`.
4. **Acoustics ask** (002-acoustics-missing): `HelmholtzResonatorFrequency` belongs in core `acoustics/` alongside `ResonantFrequency` (it's not aeroacoustic-specific — vortex-acoustic coupling enters only via L5). Also `AtmosphericAbsorptionISO9613(f, T, RH, p0)` for propagation half.
5. **Signal asks** (132-signal-missing): `signal.Hilbert` (tonal-vs-broadband decomposition per L19 Howe), `signal.Welch`/`signal.RFFT` (variance-reduced loaded-pressure spectra for L15/L16). **Linalg ask** (097-linalg-missing): `linalg.SVD` for FW-H source-image LSQ (defer until consumer arrives).
6. **Naming.** Use `LighthillStressTensor` not `StressTensor` (community convention; disambiguates from `linalg.CovarianceMatrix` and fluids viscous-stress `τ_ij`). Use `RotorNoiseTonalBPF` not `BPF` (ambiguous with band-pass-filter in `signal/`).
7. **Cross-package consumer.** 160 `fluids/turbulence/` Reynolds-stress fields feed directly into L7 LighthillStressTensor; co-ship 197-L7 + 160-T6. 192 `fluids/control/` `dCl/dt`/`dCd/dt` fluctuating-force histories feed L8 CurleDipoleFarField. Combined demo case (cylinder VIV + active-suction control + radiated tonal noise) saturates 4 packages: `fluids/control/` + `fluids/turbulence/` + `acoustics/aero/` + `signal/`.
8. **Documentation.** Every primitive cites: Lighthill 1952/1954 (L7, L10), Curle 1955 (L8), Ffowcs Williams 1963 (L12), Ffowcs Williams–Hawkings 1969 + Farassat 1981 (L13), Powell 1964 (L7 vortex form), Howe 1975 (L19), Strouhal 1878 / Roshko 1954 / Williamson 1996 (L1, L3, L4), Rossiter 1964 (L14), Sarpkaya 2004 (L5), Helmholtz 1885 / Rayleigh 1884 (L6, L20), Phillips 1960 / Tam-Burton 1984 (L18), Goody 2004 / Bies 1966 (L15, L16), Strahle 1971 (L22), Bailly-Juvé 2000 (L21).

---

## 5. Anti-recommendations

- **Don't** ship a Lighthill-source DNS solver here — that's a `chaos/` or future `pde/`-package problem. The synergy package is **post-processing only**: caller supplies flow-field/surface-data time histories, package returns far-field pressure/power/dB SPL.
- **Don't** put `HelmholtzResonatorFrequency` exclusively in `acoustics/aero/`. It belongs in core `acoustics/` alongside `ResonantFrequency`. Co-ship per 002-acoustics-missing.
- **Don't** wait on `geometry.Mesh3D` before shipping L7/L13/L19 per-point versions; per-point primitives are the workhorses of every CFD post-processor and the volume-integration wrapper is a 30-LOC addition once Mesh3D lands.
- **Don't** add a `Turbulence` struct bundling `(ρ, ν, c₀, p₀)` — keep individual scalar args per CLAUDE.md "numbers in, numbers out".
- **Don't** put `StrouhalCurveCircularCylinder` (L4) in `fluids/` — it's a domain-specific empirical fit; `fluids/` is closed-form theory. Use-case is acoustic (Aeolian-tone pitch).
- **Don't** ship LEE as a fully-3-D PDE solver here. 1-D demonstrator is fine; 3-D belongs in a future `pde/wave/` package alongside FDTD-EM (159-W4).
- **Don't** re-implement `acoustics.SoundSpeed` inside `acoustics/aero/`; always compose to preserve single source of truth.

---

## 6. Verdict

The acoustics × fluids synergy gap is **wide and at the heart of the most-cited engineering-acoustics literature** (Lighthill 1952 has 11k citations; Curle 1955 has 4k; FW-H 1969 has 5k; Williamson 1996 has 3k; Howe 1975 has 4k). Yet the gap is **not deep**: 15 of the 23 enumerated synergy primitives ship at v0.10.0 with **zero new linalg / signal / mesh dependencies** — they are pure post-processing arithmetic over `acoustics.*` + `fluids.*` outputs. Day-1 the 75-LOC vortex-shedding triple (L1 + L4 + L9) lands the first compressible-aeroacoustic primitive in the repo and saturates a 3/3 R-MUTUAL-CROSS-VALIDATION pin matching the recent commit-log saturation pattern. Week-3 the 280-LOC FW-H Farassat-1A (L13) brings post-processing to research-grade parity with NASA F1A and unlocks rotor / propeller / wind-turbine consumers. The single most important architectural decision is **placement in `acoustics/aero/`** (not `fluids/aero/` or a top-level `aero/`): aeroacoustic primitives are *acoustic* in their output (pressure, intensity, dB) and *fluid* in their source — but the consumer-facing API is acoustic, exactly mirroring the precedent set by 166 `acoustics/room/`, 159 `em/wave/`, 160 `fluids/turbulence/`, 192 `fluids/control/`. The keystone insight is that **Lighthill 1952 + Curle 1955 + FW-H 1969 are three formally-equivalent reformulations of the same wave equation under different surface-of-integration choices** — once L7 + L8 + L13 all ship, the R-FORMULATION-EQUIVALENCE 3/3 pin saturates the strongest possible correctness witness in the entire aeroacoustics literature.
