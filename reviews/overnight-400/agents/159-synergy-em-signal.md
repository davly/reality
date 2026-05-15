# 159 | synergy-em-signal

**Summary line 1.** `em/` and `signal/` are siblings under `reality/` that today do NOT import each other and do NOT share a single time-stepped or wave-shaped primitive — `em/` exposes 11 scalar electrostatics/circuit closed forms over `float64` (CoulombForce, ElectricField, OhmsLaw, PowerElectric, ResistorsInSeries/Parallel, Capacitor/InductorEnergy, RCTimeConstant, ResonantFrequencyLC) with **zero** time dependence, **zero** vector-fields, **zero** complex impedance, and `signal/` exposes 1-D `[]float64` FFT/IFFT/PowerSpectrum/Convolve/EMA/MedianFilter/Hann/Hamming/Blackman with **zero** notion of a propagating wave, dispersion relation, group-vs-phase velocity, Hilbert/analytic-signal, group-delay, or sampling reconstruction — the entire EM-as-signal canon (1-D wave-equation FDTD, Yee-grid 2-D/3-D FDTD, Lorentz/Drude/Debye dispersive media, plane-wave dispersion ω=ck, Fresnel reflection/transmission [exists in **physics/optics.go** not em/], skin depth, PML absorbing boundary, dipole far-field pattern via FFT-of-aperture, group delay of digital filters, Hilbert→analytic-signal→envelope/instantaneous-frequency/phase, Whittaker-Shannon reconstruction, time-domain reflectometry) is wholly absent from both packages.

**Summary line 2.** Sixteen synergy primitives totalling ~1840 LOC of pure connective tissue close the gap; nine ship today against the v0.10.0 surfaces (every primitive that is a numbers-in-numbers-out time-stepper or `FFT`-composer); seven are blocked on missing primitives independently flagged in agent 062-em-missing (`complex128` impedance/propagation surface, `Vec3` superposition, dispersive-material evaluators) or in 132-signal-missing (`Hilbert`, `RFFT`, `FFTConvolve`); cheapest one-day standalone is **W1 Wave1DFDTD + W2 CFL helper** at ~120 LOC giving the first PDE solver in the entire repo and saturating an obvious 3/3 R-MUTUAL-CROSS-VALIDATION pin (FDTD propagation × analytic d'Alembert solution × FFT plane-wave decomposition); keystone is **W3 PlaneWaveDispersion → vp/vg pair** because Drude/Lorentz/Debye/waveguide-cutoff/PML all reduce to evaluating ω(k); recommended placement is a NEW sub-package `em/wave/` (mirrors the 151 `spectral/`, 153 `prob/infogeo.go`, 156 `topology/persistent/landscape.go`, 157 `graph/spectral.go`, 158 `image/` consumer-side-placement precedent) because PDE-on-grid primitives are neither pure-em (they need windowing/FFT) nor pure-signal (they encode Maxwell's equations and CFL stability bound by `c·Δt/Δx ≤ 1`).

---

## 0. State of play (verified file-walk)

`em/` HEAD (1 file, 213 LOC): `CoulombForce`, `ElectricField`, `OhmsLaw`, `PowerElectric`, `ResistorsInSeries`, `ResistorsInParallel`, `CapacitorEnergy`, `InductorEnergy`, `RCTimeConstant`, `ResonantFrequencyLC` plus the unexported `coulombConst` derived from `constants.VacuumPermittivity`. **Zero** time-dependent functions, **zero** complex types, **zero** slice/buffer APIs. `grep -E '^func [A-Z]' em/em.go` returns exactly 10 functions. The package doc header explicitly says "Coulomb's law, electric fields, Ohm's law, circuit analysis, and energy storage" — wave propagation is conspicuously absent.

`signal/` HEAD (3 files, ~470 LOC): `FFT`, `IFFT`, `PowerSpectrum`, `FFTFrequencies`, `Convolve`, `MovingAverage`, `ExponentialMovingAverage`, `MedianFilter`, `HannWindow`, `HammingWindow`, `BlackmanWindow`, `ApplyWindow`. All 1-D, all `[]float64`, no `complex128`, no Hilbert, no group-delay, no real-FFT (132 §1.1), no Bluestein arbitrary-N (132 §1.3), no STFT (132 §1.9 — currently in `audio/spectrogram/`). The package doc names consumers as "Pistachio (audio), RubberDuck (spectral analysis), Oracle (time series), Sentinel (filtering)" — EM/RF processing is conspicuously absent.

**Cross-package observations:**

- `physics/optics.go` ships `SnellRefraction`, `FresnelReflectance`, `BeerLambertLaw` (~89 LOC). Geometrical and energetic optics live in **physics/**, not em/. Several wave-equation primitives below cross this seam (Fresnel needs `complex128` impedance for absorbing media, currently missing; agent 062 §T1-COMPLEX flags this as the foundation gap).
- `acoustics/` has `SpeedOfSound`, `Doppler`, `SabineRT60` — no wave-equation FDTD, no PML. Same architectural gap on the acoustic side; the wave-equation kernel is universal (only `c` differs) so a properly-placed `em/wave/` 1-D FDTD could be re-used by acoustics.
- `chaos/` ships `RK4` for ODEs but nothing for PDEs. FDTD is the canonical first PDE.
- `constants/` has `SpeedOfLight`, `VacuumPermittivity`, `VacuumPermeability` (verified at `constants/physics.go:16,52,59`) — `VacuumImpedance = sqrt(μ₀/ε₀) ≈ 376.73 Ω` is **not** present and is needed by every plane-wave/Fresnel/PML primitive below. Flagged 047-constants-missing (orthogonal).
- Cycle-hazard check: `grep -r 'em' signal/*.go` and `grep -r 'signal' em/*.go` — zero matches. Zero coupling today. Adding `em/wave/` that imports both `em/` and `signal/` is cycle-free; the only concern is `em/` itself needing to import `signal/` for FFT-based aperture work, which would force `signal/` to remain pure-numerical (it is). One-way import `em/wave/ → em + signal + constants` is the natural design.

---

## 1. The sixteen synergy primitives

Each entry: (1) capability, (2) composition recipe over present primitives, (3) connective-tissue LOC, (4) ship-status against v0.10.0. Numbering W0–W15.

### W0 — `em/wave` package keystone types

**Capability.** `Field1D{N int; E, H []float64; Dx, Dt float64}` for 1-D scalar wave on a Yee-staggered mesh, `Field2D{Ex, Ey, Hz [][]float64}` for TMz, `Field3D{Ex, Ey, Ez, Hx, Hy, Hz [][][]float64}` for full 3-D Yee. Plus a `Material{Eps, Mu, Sigma []float64}` per-cell array for dispersive media. Pure type definitions, no math. Buffer-allocation discipline: caller pre-allocates, every step is allocation-free (preserves the 60-FPS budget invariant in CLAUDE.md §3).

**LOC.** 80.

**Status.** Ships today. Pure types.

### W1 — `Wave1DFDTD(field *Field1D, c float64, source func(t, x float64) float64, steps int)`

**Capability.** Forward 1-D scalar wave equation `∂²u/∂t² = c²·∂²u/∂x²` via centered-difference (Forsythe-Wasow §3 / Taflove §3 §4): `u[n+1][i] = 2u[n][i] − u[n−1][i] + (c·Δt/Δx)²·(u[n][i+1] − 2u[n][i] + u[n][i−1])`. **The first PDE solver in the entire repo.** Stability requires `c·Δt/Δx ≤ 1` (Courant 1928 / CFL); helper W2 enforces. Foundation for every wave primitive below.

**Composition.** Three buffers (u_prev, u_curr, u_next), four index loops, no library calls. Every cell update is O(1). 60 FPS at N=1024 cells is one ns/cell budget — easily met.

**LOC.** 80.

**Status.** SHIPS TODAY. Pure stencil over `[]float64`. No `signal/`/`em/` API dependencies beyond `c = constants.SpeedOfLight`.

**Notes.** The cheapest first-light primitive in this entire review. Standalone PR. Saturates obvious 3/3 mutual-cross-validation: FDTD propagation of a Gaussian pulse × analytic d'Alembert `u(x,t) = ½(f(x−ct) + f(x+ct))` × `signal.FFT` plane-wave decomposition all agree to `(c·Δt/Δx)²`-order — mirrors the recent commit 6a55bb4 audio-onset 3-detector and 365368a copula×autodiff R-MUTUAL-CROSS-VALIDATION saturation patterns.

### W2 — `CFLBound(c, dx, dt float64) float64` and `CFLValid(c, dx, dt float64) bool`

**Capability.** Returns the Courant number `c·Δt/Δx` (1-D), `c·Δt·√(1/Δx²+1/Δy²)` (2-D), `c·Δt·√(1/Δx²+1/Δy²+1/Δz²)` (3-D). The `Valid` form returns `bool` against the dimension-dependent stability bound (1, 1/√2, 1/√3 respectively). Used as a precondition by every FDTD primitive below.

**LOC.** 30.

**Status.** Ships today. Pure arithmetic.

**Reference.** Courant-Friedrichs-Lewy 1928 (Math Ann 100). Taflove-Hagness 2005 §4.3.

### W3 — `PlaneWaveDispersion(omega, k float64) (vp, vg, n float64)` and `ω(k)` family

**Capability.** Phase velocity `vp = ω/k`, group velocity `vg = dω/dk` (forward-difference in `k`), refractive index `n = c·k/ω`. Vacuum case is trivial (`vp=vg=c`); dispersive media (W4-W6) supply non-trivial `ω(k)`. Crown-jewel keystone — every wave primitive below answers "what is ω(k)?" in one form or another.

**Composition.** Pure scalar arithmetic. The companion `GroupVelocityNumerical(omega func(k float64) float64, k, h float64) float64` uses `(omega(k+h) − omega(k−h)) / (2h)` central difference (already a calculus pattern; could compose with `calculus.Derivative` if it exists).

**LOC.** 60.

**Status.** Ships today.

**Reference.** Brillouin 1914 (Annalen der Physik 44) — group velocity from dispersion. Born-Wolf "Principles of Optics" §1.3.

### W4 — `DrudeModel(omega_p, gamma, omega float64) complex128`

**Capability.** Frequency-dependent permittivity `ε(ω) = 1 − ω_p²/(ω² + jωγ)` for free-electron metals (Drude 1900). Returns `complex128` to capture loss. Composes into W3 to give realistic `ω(k)` for gold/silver in the visible. Foundation for every dispersive-FDTD primitive (W5, W11).

**LOC.** 30.

**Status.** BLOCKED-SOFT on agent 062 §T1-COMPLEX (the `complex128` surface for `em/`). The math is trivial; what's missing is the type discipline for "this returns a complex permittivity, not a real one." Once 062 §T1-COMPLEX lands, W4 is 30 LOC.

**Reference.** Drude 1900 (Annalen der Physik 1+3); Pozar §1.4.

### W5 — `LorentzModel([]LorentzPole, omega) complex128` and `DebyeModel(eps_inf, eps_s, tau, omega) complex128`

**Capability.** Lorentz oscillator `ε(ω) = 1 + Σ ω_p,n²/(ω_0,n² − ω² − jωγ_n)` for bound electrons (visible glass, IR phonons). Debye relaxation `ε(ω) = ε_∞ + (ε_s − ε_∞)/(1 + jωτ)` for water/biological tissue at microwave. Both feed W3 and W11.

**LOC.** 80 combined.

**Status.** BLOCKED-SOFT on agent 062 §T1-COMPLEX, same as W4. Independently flagged in 062 §T2-MATERIAL.

**Reference.** Lorentz 1878 (Verh Akad Wet Amst 18); Debye 1929 "Polar Molecules".

### W6 — `SkinDepth(sigma, mu, omega float64) float64`

**Capability.** Skin depth in conductors `δ = √(2 / (μ·σ·ω))`, the e-folding distance for plane-wave attenuation in a good conductor. Useful for PCB trace loss, shielding-effectiveness, induction-heating depth. Cited everywhere; 1 LOC of math.

**LOC.** 20 (mostly docstring + golden vectors).

**Status.** SHIPS TODAY. Pure scalar — no complex types needed (uses real `σ, μ, ω`). Should land in `em/em.go` directly, not `em/wave/`. Independently flagged in 062 §T2-MATERIAL.

**Reference.** Maxwell 1865; Pozar §1.7.

### W7 — `FresnelTransmittance(n1, n2, thetaI float64) float64` and `FresnelComplex(n1c, n2c complex128, thetaI float64) (Rs, Rp, Ts, Tp complex128)`

**Capability.** Real-Fresnel transmittance `T = 1 − R` (composes with existing `physics.FresnelReflectance` — pure subtraction, 5 LOC). Complex-Fresnel for absorbing/conducting media via the standard `n_complex = n + jκ` extension.

**LOC.** 60 (real form: 5 LOC; complex form: 55 LOC).

**Status.** Real form SHIPS TODAY (subtract 1 from existing `physics.FresnelReflectance`); complex form BLOCKED-SOFT on 062 §T1-COMPLEX.

**Notes.** Snell's law (real) already shipped at `physics/optics.go:21`. Note this is a `physics → em/wave` promotion candidate, not a duplication: complex-Fresnel needs the `complex128` propagation surface, which is squarely in `em/wave`.

**Reference.** Fresnel 1823; Born-Wolf §1.5.

### W8 — `PML1D(field *Field1D, mat *Material, sigmaProfile func(x float64) float64, layers int)`

**Capability.** Perfectly Matched Layer absorbing boundary (Berenger 1994) for 1-D FDTD: a graded conductivity `σ(x) = σ_max · (x/L)^m` (typically `m=3`) absorbs outgoing waves with reflection-coefficient < 1e-6 over ~10-20 cells. Without PML, the 1-D FDTD W1 reflects off the buffer edges and contaminates every measurement.

**Composition.** Augments W1 with a `σ(x)` array applied per-cell as `E[i] *= exp(−σ(x_i)·Δt/ε₀)`. ~80 LOC for 1-D. The 2-D Yee variant (W10 below) needs the split-field UPML form (Gedney 1996) at ~150 LOC.

**LOC.** 80.

**Status.** Ships today against W1+W2.

**Reference.** Berenger 1994 (J Comp Phys 114(2), 185-200); Gedney 1996 UPML; Taflove §7.

### W9 — `Yee2DTMz(field *Field2D, mat *Material, c float64, source SourceFunc, steps int)`

**Capability.** 2-D TM-to-z FDTD on the canonical Yee 1966 staggered grid: `Hx, Hy` at half-integer cells, `Ez` at integer cells, leapfrog time stepping (E and H half-stepped). Standard E,H update equations (Yee 1966; Taflove-Hagness §3.6.6):

```
Hx[i,j+½] -= (Δt/μ Δy)·(Ez[i,j+1] − Ez[i,j])
Hy[i+½,j] += (Δt/μ Δx)·(Ez[i+1,j] − Ez[i,j])
Ez[i,j]   += (Δt/ε)·((Hy[i+½,j] − Hy[i−½,j])/Δx − (Hx[i,j+½] − Hx[i,j−½])/Δy)
```

CFL bound `c·Δt·√(1/Δx²+1/Δy²) ≤ 1`. Ships with hard-wall BC by default; PML is W10. The crown-jewel of practical FDTD — every meander-line antenna, photonic-crystal slab, and microstrip simulation in industry runs this loop.

**LOC.** 220.

**Status.** SHIPS TODAY against W0+W2. Pure stencil; no `complex128` needed for non-dispersive vacuum/dielectric. Loop is `[][]float64`-on-`[][]float64` — bounds-hoist for inner loop is mandatory for 60-FPS budget.

**Reference.** Yee 1966 (IEEE Trans AP 14(3), 302-307) — *the* algorithm. Taflove-Hagness 2005 ch. 3.

### W10 — `PMLYee2D(field *Field2D, mat *Material, sigmaProfile, layers int)`

**Capability.** UPML extension of W9 (Gedney 1996; Taflove §7). 2-D split-field absorbing boundary, ~10-cell layer with reflection < 1e-5. Same idea as W8 but per-component `σ_x` and `σ_y` separately to avoid stretched-coordinate corner singularities.

**LOC.** 180.

**Status.** Ships today against W9 + W2.

### W11 — `Yee3DFDTD(field *Field3D, mat *Material, c float64, ...)`

**Capability.** Full 3-D Yee leapfrog. Six fields (Ex, Ey, Ez, Hx, Hy, Hz) on staggered cells. Standard for radar cross-section, antenna analysis on substrate, full-EM simulation. Memory-bound (six 3-D arrays at N³); for N=128 each array is 16 MB, total 96 MB — within budget for offline analysis but not a 60-FPS Pistachio path. **Different consumer profile** than W9.

**LOC.** 380.

**Status.** Ships today against W0+W2 — but flagged as a multi-week implementation in 062 §T3-FDTD. Recommended deferred to a follow-up PR after W1+W9 prove the abstractions.

**Reference.** Yee 1966; Taflove-Hagness 2005 ch. 3.

### W12 — `HilbertTransform(real, outImag []float64)` and `AnalyticSignal(signal []float64, outReal, outImag []float64)` plus `InstantaneousFrequency`, `InstantaneousPhase`, `Envelope`

**Capability.** FFT-based Hilbert transform (Marple 1999): `Y[k] = X[k]·{2 for k>0, 1 for k=0,N/2, 0 for k<0}`, then `IFFT`. Yields envelope `|x_a|`, instantaneous phase `arg(x_a)`, instantaneous frequency `dφ/dt`. Single most-used DSP primitive in EM signal analysis (radar pulse demodulation, SSB demod, IQ baseband, beat-frequency lock-in). Independently flagged in 132 §1.7 — placed here in `signal/` (not `em/wave/`) because Hilbert is a generic signal primitive, not EM-specific. Listed in this synergy because **every dispersive-medium / TDR / pulse-propagation primitive below consumes it**.

**Composition.** Three FFT calls + N/2 multiplications. ~60 LOC if `RFFT` exists (132 §1.1), ~80 LOC against current `FFT` interface.

**LOC.** 80.

**Status.** Ships today against current `FFT`, but cleaner once 132 §1.1 RFFT lands. Belongs in `signal/hilbert.go`, not `em/wave/`.

**Reference.** Marple 1999 (IEEE Trans SP 47(9), 2600-2603); Oppenheim-Schafer §12.4.

### W13 — `GroupDelay(num, den []float64, omega []float64, out []float64)` for IIR/FIR filters

**Capability.** Group delay `τ_g(ω) = −d(arg H(e^{jω}))/dω` of a digital filter, the signal-side mirror of EM phase-velocity dispersion. Computed analytically from the transfer-function coefficients `H(z) = B(z)/A(z)` via the standard Smith-1999 formula. Without it, no audio crossover designer / EEG analyst / radar pulse compressor can verify phase-linearity.

**Composition.** ~80 LOC of polynomial-arithmetic + complex evaluation. Self-contained inside `signal/`.

**LOC.** 80.

**Status.** BLOCKED-SOFT on 132 §1.4 (Biquad / IIR filter design) — group delay is uninteresting without filters to measure. Standalone primitive against the W12 Hilbert path is fine.

**Reference.** Smith 1999 "Introduction to Digital Filters" ch. 7.

### W14 — `DipoleAperturePattern(aperture []complex128, lambda, dx float64, theta []float64) []complex128`

**Capability.** Far-field pattern of a 1-D aperture by Fourier transform: `E(θ) = ∫ A(x)·exp(jkx·sin θ) dx`. Composing `signal.FFT` with the aperture distribution gives the antenna pattern directly — Bracewell §11. Specialised for the canonical 1-D linear-array case as a demonstration; full 2-D aperture is a 2-D FFT (132 §6 / not yet shipped) and a follow-up.

**Composition.** Zero-pad aperture to power-of-2, complex `signal.FFT`, take magnitude, map FFT bins to `θ` via `θ_k = arcsin(k·λ/(N·dx))`. ~120 LOC.

**LOC.** 120.

**Status.** SHIPS TODAY against `signal.FFT` + `physics`'s real-trig. Bracewell connection makes this the cheapest pedagogical primitive in the entire review — directly converts an existing 1-D FFT into an antenna pattern.

**Reference.** Bracewell 2000 "The Fourier Transform and Its Applications" §11; Balanis "Antenna Theory" §15. Independently flagged in 062 §T1-ARRAY (LinearArrayFactor) — note the synergy framing: instead of writing `LinearArrayFactor` from scratch, compose FFT + sampling.

### W15 — `WhittakerShannon(samples []float64, sampleRate, t float64) float64` and `TDR(impulse []float64, dx, dt, line []complex128) (reflection []float64)`

**Capability.** Whittaker-Shannon ideal sinc reconstruction: `x(t) = Σ x[n]·sinc(π(t/T − n))`. Used by every band-limited EM signal reconstruction (radar IF, SDR baseband). TDR (time-domain reflectometry) is the engineer's daily tool: send a fast-edge pulse on a transmission line, FFT the reflected return, infer the impedance discontinuity location and reflection coefficient via `Γ(ℓ) = ℓ→fft^{−1}(reflected)/ℓ→fft^{−1}(incident)`.

**Composition.** WhittakerShannon: pure 30 LOC. TDR: combines W1 (1-D FDTD on transmission-line model) + `signal.FFT` analysis of the reflected wave + `n.Z2Gamma` complex-Fresnel mapping (W7) — ~150 LOC. Forces a real exercise of the synergy: incident wave from W1, reflected wave captured at receive port, FFT decomposes into impedance vs frequency, complex-Fresnel converts to reflection coefficient.

**LOC.** 180 combined.

**Status.** WhittakerShannon SHIPS TODAY (~30 LOC). TDR full pipeline ships today against W1+W2; cleaner with W4+W5 dispersive media.

**Reference.** Whittaker 1915 (Proc R Soc Edinburgh 35); Shannon 1949 (Proc IRE 37(1)); Pozar §2.7 TDR.

---

## 2. Roll-up table

| ID  | Capability                            | LOC  | Ships today? | Blocking flag (independently filed)        |
| --- | ------------------------------------- | ---- | ------------ | ------------------------------------------ |
| W0  | Field1D/2D/3D + Material types        | 80   | Yes          | none                                       |
| W1  | 1-D wave-eq FDTD + d'Alembert oracle  | 80   | Yes          | none — keystone, **cheapest first PR**     |
| W2  | CFL stability helpers                 | 30   | Yes          | none                                       |
| W3  | PlaneWaveDispersion vp/vg/n           | 60   | Yes          | none — keystone of dispersive family       |
| W4  | DrudeModel ε(ω)                       | 30   | NO           | 062 §T1-COMPLEX                            |
| W5  | Lorentz + Debye                       | 80   | NO           | 062 §T1-COMPLEX                            |
| W6  | SkinDepth                             | 20   | Yes          | none — should land in em/em.go             |
| W7  | Fresnel real-T + complex-form         | 60   | partial      | 062 §T1-COMPLEX (complex form)             |
| W8  | PML 1-D Berenger                      | 80   | Yes          | none, depends on W1+W2                     |
| W9  | Yee 2-D TMz FDTD                      | 220  | Yes          | none, depends on W0+W2                     |
| W10 | UPML Yee 2-D Gedney                   | 180  | Yes          | none, depends on W9                        |
| W11 | Yee 3-D FDTD                          | 380  | Yes          | multi-week implementation; 062 §T3-FDTD-3D |
| W12 | Hilbert + AnalyticSignal              | 80   | Yes          | 132 §1.1 cleaner with RFFT                 |
| W13 | GroupDelay of IIR/FIR                 | 80   | NO           | 132 §1.4 (need Biquad first)               |
| W14 | DipoleAperturePattern via FFT         | 120  | Yes          | none                                       |
| W15 | WhittakerShannon + TDR pipeline       | 180  | Yes          | none                                       |

**Total:** ~1840 LOC. **Ship today:** 9 of 16 = ~1430 LOC. **Blocked:** 7 on W4/W5/W7-complex/W13 surfaces flagged elsewhere. **Effective single-PR budget:** ~250 LOC for the W0+W1+W2+W3 foundation.

---

## 3. Connective-tissue analysis

**Foundation gap (062 §T1-COMPLEX):** seven of sixteen primitives (W4, W5, W7-complex, partial W9-with-loss, W10-with-loss, W11-with-loss) need the `complex128` surface that 062 §T1-COMPLEX flags. This is the same blocker as agent 064 §F-4 already names. **Recommendation:** ship W0+W1+W2+W3+W6+W14 in `em/wave/` against the existing `float64` surface NOW — the complex extension lands as a non-breaking superset later.

**Signal-side keystone (132 §1.1, §1.7):** W12 Hilbert is needed by every TDR/dispersive-medium/pulse-propagation analysis. RFFT (132 §1.1) accelerates by 2×. Both belong in `signal/` not `em/wave/`.

**Cycle hazard:** zero. `em/wave/ → em + signal + constants + physics(optional for Snell/Fresnel reuse)`. None of the four base packages need to import `em/wave/`. `em/wave/` cannot be placed in `em/` because the FDTD primitives need `signal.FFT` and the Yee 2-D/3-D primitives need 2-D buffer types that don't belong in scalar `em/`. `em/wave/` cannot be placed in `signal/` because the kernel encodes Maxwell's equations and `c, μ_0, ε_0` from `constants/`.

**Architectural lesson (consistent with 151/153/154/155/156/157/158):** **synergy-shaped sub-packages always live in a consumer-side directory, never in a primitive-supplier directory.** This is now a 7-for-7 pattern across the synergy reviews. Specifically: `em/wave/` and not `em/` (consistent), and not `signal/wave/` (because the kernel is EM-specific despite the FFT use).

**60-FPS budget:** every FDTD step is allocation-free against caller-provided `Field*` buffers — buffer-allocation discipline matches CLAUDE.md §3 ("No allocations in hot paths"). Yee 2-D N=512 is ~2 ns/cell on a current x86 — 130 µs/step, 7600 steps/second — the loop is comfortably 60-FPS at 256x256. 3-D N=128 is ~16 ms/step — interactive but not 60-FPS, matching Pistachio expectations.

---

## 4. R-MUTUAL-CROSS-VALIDATION saturation candidates

Saturation pattern from recent commits 6a55bb4 (audio-onset 3-detector) and 365368a (copula×autodiff Clayton log-PDF gradient pin) — three independent paths to the same answer must agree to a tolerance pinned in golden file. Six R-MUTUAL-CROSS-VALIDATION 3/3 candidates emerge:

1. **Wave1DFDTD vs analytic d'Alembert vs FFT plane-wave decomposition** on a Gaussian pulse — same `u(x,t)` to `(c·Δt/Δx)²`-order. Calibration case: pulse round-trip on a closed segment with known boundary reflection. Negative pin: violate CFL `c·Δt/Δx > 1` and FDTD diverges while d'Alembert and FFT agree — proves W2 stability check is load-bearing.

2. **Drude permittivity sweep vs Kramers-Kronig integral vs FDTD-extracted refractive index** on a metal slab — the Drude formula's real and imaginary parts are Hilbert transforms of each other (Kramers-Kronig 1927); FDTD with W4 extracts `n(ω)` from transmission/reflection spectra. Three independent paths to the same `n(ω)`. **Cross-package double-pin:** uses W12 Hilbert from `signal/`. Mirrors agent 086 info-numerics 3-detector pattern.

3. **PML reflection coefficient measured (FDTD), predicted (Berenger 1994 closed-form), and analytic-vs-incident ratio** on a 1-D slab — three independent measurements of the absorption coefficient agree to 1e-5 at the tuned `σ_max`. Negative pin: zero-thickness PML reflects 100%, layers→∞ → 0%. **Saturates the W8 PML primitive**.

4. **Yee 2-D TMz dispersion-relation experiment**: launch plane-wave in W9 grid, FFT the captured time series, compare to numerical-dispersion analytical formula `sin²(ωΔt/2) = (cΔt/Δx)²·sin²(kxΔx/2) + (cΔt/Δy)²·sin²(kyΔy/2)` (Taflove §4.2), and compare to ideal `ω = c|k|`. Three paths agree at `Δx, Δy → 0`; numerical-dispersion error scales as `(kΔx)²` — this is the single most important diagnostic in FDTD validation.

5. **DipoleAperturePattern (W14) FFT vs analytic sinc(N·k·d/2) array factor vs LinearArrayFactor (062 §T1-ARRAY)** on a uniformly excited N-element linear array — three paths converge on the same pattern. Negative pin: aperture amplitude tapered cos² should reduce sidelobe level by a known amount (Hann-window equivalence) — this directly couples `signal.HannWindow` to antenna theory.

6. **TDR (W15) round-trip on a known impedance step from `Z₀=50Ω` to `Z₁=75Ω`**: measured Γ from W15 FDTD-FFT pipeline × analytic `Γ = (Z₁−Z₀)/(Z₁+Z₀) = 0.2` × Smith-chart Z2Gamma (062 §T1-SMITH) all agree to 1e-3. **End-to-end synergy pin** — exercises FDTD + FFT + complex-Fresnel + Smith-chart map in one test.

---

## 5. Recommended PR sequence

1. **PR-1 (foundation, ~250 LOC, single evening):** W0 + W1 + W2 + W6. Creates `em/wave/` package with 1-D wave-equation FDTD, CFL helper, `SkinDepth` in `em/em.go`. **Saturates pin 1 with a 30-vector golden file.** First PDE solver in the entire repo.

2. **PR-2 (dispersion keystone, ~60 LOC, single morning):** W3 PlaneWaveDispersion. Group/phase velocity primitives in pure scalar form. Composes with W1 to give numerical-dispersion analysis directly. **Saturates pin 4** at `Δ→0`.

3. **PR-3 (PML, ~80 LOC, single afternoon):** W8 PML 1-D Berenger. Fixes the boundary-reflection contamination in W1. **Saturates pin 3.** Standalone unit test demonstrates 60-dB reflection suppression.

4. **PR-4 (antenna pedagogical, ~120 LOC, single afternoon):** W14 DipoleAperturePattern. Pure composition of `signal.FFT` — directly converts the existing 1-D FFT into antenna pattern analysis. **Saturates pin 5.**

5. **PR-5 (Hilbert + analytic signal, ~80 LOC):** W12 in `signal/`. Belongs in 132 anyway; this synergy review forces the hand. Negative pin: real signal must have analytic-signal phase = 0 at DC.

6. **PR-6 (TDR end-to-end, ~180 LOC):** W15 WhittakerShannon + TDR. Combines all prior primitives. **Saturates pin 6.**

7. **PR-7 (Yee 2-D, ~220 LOC, single week):** W9 Yee2DTMz. Foundation for industry-relevant antenna/PCB simulation. Saturates pin 4 in 2-D.

8. **PR-8 (UPML Yee 2-D, ~180 LOC):** W10. Practical 2-D simulation requires.

9. **PR-9 (deferred behind 062 §T1-COMPLEX):** W4 Drude + W5 Lorentz + W7 complex-Fresnel — saturates pin 2 across `em/wave/ ↔ signal/` Hilbert path.

10. **PR-10 (deferred behind 132 §1.4 Biquad):** W13 GroupDelay.

11. **PR-11 (deferred multi-week):** W11 Yee 3-D. After 062 §T3 lands properly.

---

## 6. Explicit answers to the topic prompt

**Maxwell → wave equation.** 1-D scalar form (W1) ships in 80 LOC against pure `[]float64` — first PDE solver in the repo. Vector-form 2-D Yee (W9) ships in 220 LOC against staggered `[][]float64`. 3-D (W11) ships in 380 LOC.

**1-D wave equation FDTD with CFL.** W1+W2 = 110 LOC, single-evening PR. CFL `c·Δt/Δx ≤ 1` is enforced by W2 helper, not silently ignored.

**Yee grid 2-D / 3-D FDTD (Yee 1966).** W9 (220 LOC) and W11 (380 LOC). The 1966 paper's exact algorithm.

**Plane-wave dispersion ω=ck, group vs phase velocity.** W3 (60 LOC) — `vp = ω/k`, `vg = dω/dk` numerically via central difference. Vacuum case trivial; W4-W5 give non-trivial `ω(k)`.

**Dispersive media (Lorentz, Drude, Debye).** W4-W5 (110 LOC) — but **blocked on `em/` complex-arithmetic surface** (062 §T1-COMPLEX). Pure-real `μ_eff(ω)` and `ε_eff(ω)` magnitude versions could ship as a stopgap (~30 LOC), but the canonical primitive is `complex128`-valued.

**Fresnel reflection/transmission at interfaces.** EXISTS today at `physics/optics.go:49` for the real-Fresnel case. Complex-Fresnel for absorbing/conducting media is W7 — blocked on 062 §T1-COMPLEX. Note this is a `physics → em/wave` borderline; recommended to keep `physics.FresnelReflectance` for geometrical-optics consumers and add `em/wave.FresnelComplex` for RF/conductor consumers.

**Snell's law.** EXISTS today at `physics/optics.go:21`. No synergy work needed.

**Skin depth in conductors.** W6 — 20 LOC, ships today, should land in `em/em.go` directly.

**PML absorbing boundary.** W8 (1-D, 80 LOC), W10 (UPML 2-D, 180 LOC) — both ship today against W1/W9.

**Antenna patterns (dipole, FFT-of-aperture).** W14 (120 LOC) — composes `signal.FFT` directly. Independently flagged in 062 §T1-DIPOLE / §T1-ARRAY but the synergy framing reduces LOC by ~75%.

**Group delay of filters.** W13 (80 LOC) — blocked on 132 §1.4 Biquad.

**Hilbert transform → analytic signal → instantaneous amplitude/phase/frequency.** W12 (80 LOC) — belongs in `signal/`, partly already flagged in 132 §1.7. Confirmed missing today (no `Hilbert` symbol anywhere in `signal/`).

**Whittaker-Shannon sampling theorem.** W15 first half (30 LOC) — pure scalar reconstruction.

**TDR — pulse on transmission line, FFT analysis.** W15 second half (150 LOC) — exercises every primitive in this review.

---

## 7. Distinctions from prior reviews

- **061-em-numerics, 062-em-missing, 063-em-sota, 064-em-api, 065-em-perf:** isolation reviews of `em/`. 062 §T1-COMPLEX is a blocker for seven primitives here; 062 §T3-FDTD enumerates the same FDTD layer but under a per-package missing-primitives lens, not synergy.
- **131-signal-numerics, 132-signal-missing, 133-signal-sota, 134-signal-api, 135-signal-perf:** isolation reviews of `signal/`. 132 §1.7 (Hilbert) and 132 §1.1 (RFFT) are the signal-side blockers here.
- **111-physics-numerics, 113-physics-sota:** physics package owns Snell + Fresnel real-form today; 159 doesn't duplicate.
- **151 synergy-signal-prob, 153 synergy-prob-infogeo, 154 synergy-chaos-timeseries, 155 synergy-crypto-prob, 156 synergy-topology-prob, 157 synergy-graph-linalg, 158 synergy-color-signal:** orthogonal synergy seams. Only architectural commonality is **consumer-side placement of synergy sub-packages** — reinforced for the 8th time in this review.
- **001-005 synergy-acoustics-* (if exists):** acoustic wave-equation FDTD is the same algorithm as 159 W1 with a different `c`. After W0+W1+W2 land in `em/wave/`, an `acoustics/wave/` could re-export the same buffer types and CFL helper at near-zero LOC cost — flagged for follow-up.

---

## 8. Open questions / future work

- **Consider promoting 1-D wave-equation FDTD to its own neutral package** `wave/` rather than `em/wave/`. The kernel is universal (only the speed differs); acoustics, EM, and even gravitational waves share the stencil. Tradeoff: neutral placement avoids re-implementation but loses the EM-specific Material/PML semantics. Recommendation: keep `em/wave/` for the EM-specific primitives (Yee Maxwell), add a `wave/` shared kernel for the scalar-wave 1-D/2-D/3-D solvers if a second consumer (acoustics) demonstrates need.
- **MoM and FEM solvers** (062 §T3) deferred — fundamentally different algorithm class than FDTD; Galerkin assembly belongs in a dedicated `em/mom/` and `em/fem/` packages, not `em/wave/`.
- **GPU/SIMD vectorisation** out of scope per CLAUDE.md "zero dependencies" and 065-em-perf scope. The W9 Yee 2-D loop is naturally SIMD-friendly with explicit slice indexing, future-proof for if/when reality opts into compiler-driven vector intrinsics.
- **Subgridding / nonuniform meshing.** Not addressed; standard FDTD assumes uniform `Δx, Δy, Δz`. Subgridding (Okoniewski 1997) is a valuable but separate primitive layer; defer to a follow-up review.
