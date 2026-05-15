# 344 — dive-orbital-perturbations (J2 / SGP4 / Cowell / EGM2008 / atmospheric drag audit)

## Headline
`reality/orbital` is **pure two-body Keplerian** — zero perturbation primitives — so any non-trivial Earth-satellite work (LEO drag decay, SSO design, TLE propagation, GEO station-keeping) is currently impossible without leaving the library; the cheapest day-1 PR is **J2 secular rate closed-forms (Kozai 1959 / Brouwer 1959)** at ~80 LOC, gated only on adding `J2Earth = 1.0826267e-3` and `R⊕ = 6378137.0` to `constants/`.

## Findings

### F1. Current state of perturbation code: nil
`orbital/orbital.go` (266 lines, 8 functions) implements **only** closed-form two-body math — `KeplerOrbit`, `OrbitalPeriod`, vis-viva `OrbitalVelocity`, `HohmannTransfer`, `EscapeVelocity`, `HillSphere`, `SynodicPeriod`, `TrueAnomalyFromMean`. Grep across the whole repo for `J2|SGP4|SDP4|Zonal|EGM|Encke|Cowell|Perturb|Brouwer|Kozai` returns only matches inside `reviews/overnight-400/`, never source. **No `J2`, `J3`, `J4` zonal harmonics. No SGP4. No Cowell numerical integrator. No drag, SRP, third-body, or relativistic correction. No frame transforms, no time scales — so even if a propagator existed, TLE epoch handling would be missing.** 107-orbital-missing already enumerated this gap; this dive extracts the perturbation slice and tiers concrete primitives.

### F2. Constants-package readiness
`constants/physics.go:73` exposes only `StandardGravity`. Searching the constants tree for `Earth|J2|EquatorialRadius` returns *no* dedicated Earth-shape or geopotential constants. **Pre-requisite** for any zonal-harmonic primitive: add a `constants/earth.go` (or extend `physics.go`) with at minimum:
- `EarthRadiusEquatorial = 6378137.0`           (WGS84, IERS 2010)
- `EarthRadiusPolar     = 6356752.3142`         (WGS84)
- `EarthFlattening      = 1.0/298.257223563`
- `EarthMu              = 3.986004418e14`       (m³/s², EGM2008)
- `EarthRotationRate    = 7.2921150e-5`         (rad/s, IERS 2010)
- `J2Earth              = 1.0826267e-3`         (EGM2008/IERS, dimensionless C̄₂₀ already de-normalised)
- `J3Earth              = -2.5327e-6`
- `J4Earth              = -1.6196e-6`
- `J5Earth              = -2.273e-7`
- `MuSun                = 1.32712440018e20`
- `MuMoon               = 4.9028e12`
- `SolarPressure1AU     = 4.56e-6`              (N/m², ≈ 1361 W/m² ÷ c)
Source: NIST CODATA 2018 + IERS 2010 + EGM2008 (Pavlis et al.).

### F3. Standard perturbation models, ranked by physical magnitude (LEO 400 km baseline)
This sets truncation thresholds for any future propagator’s godoc table:

| Effect                          | Acceleration (m/s²) | Precision impact / 1 day | Closed-form? |
|---|---:|---|---|
| Two-body μ/r²                  | 8.9                | (reference)              | yes (Kepler)  |
| J2 (oblateness)                | 1.6 × 10⁻²          | ~1000 km secular drift   | yes (Brouwer/Kozai mean elements) |
| J3 (pear-shape)                 | 5 × 10⁻⁵            | frozen-orbit ω rate      | yes (mean elts) |
| J4                              | 4 × 10⁻⁵            | ω, M secular             | yes |
| Lunar third-body               | 5 × 10⁻⁶            | inclination drift        | analytic ephem ok |
| Solar third-body               | 2 × 10⁻⁶            | RAAN drift               | analytic ephem ok |
| Atmospheric drag (400 km)      | 1 × 10⁻⁷            | semi-major-axis decay    | numerical only |
| Solar radiation pressure       | 5 × 10⁻⁸            | ~m/day at LEO; m at GEO  | analytic shadow model |
| Solid-Earth tides (k₂)         | 1 × 10⁻⁹            | sub-cm                   | analytic |
| GR Schwarzschild               | 1 × 10⁻⁸ (Mercury 1×10⁻⁷) | 43″/cy Mercury    | yes |

(Numbers from Vallado §8.6 Table 8-2 / Montenbruck & Gill §3 Table 3.1.)

### F4. SGP4 sources — canonical, all citable
- **Hoots & Roehrich 1980, "Spacetrack Report #3" (SPACETRACK)** — the original NORAD analytical propagator spec; FORTRAN listing in the report. Defines SGP, SGP4, SDP4, SGP8, SDP8. Every TLE on celestrak.org is propagated by SGP4 (period < 225 min) or SDP4 (deep-space ≥ 225 min: lunar/solar resonance terms added).
- **Vallado, Crawford, Hujsak, Kelso 2006 — "Revisiting Spacetrack Report #3"** (AIAA 2006-6753). The de-facto modern reference C++ implementation (the file `sgp4unit.cpp`, BSD-style license, ~1500 LOC) and the **AIAA-2006-6753 acceptance test suite** (32 reference TLEs × 0–8000 min × 1 min step = ~256 000 reference vectors). Bit-exact equivalence to this suite is the industry compliance bar; Brandon Rhodes' Python `sgp4` package is the cleanest port. Ship a Go re-implementation with these vectors as goldens (gzip well — random-access not needed).
- **Lyddane 1963** — singularity treatment for low-eccentricity / low-inclination TLEs (mean-element formulation Brouwer-Lyddane). Essential inside SGP4.
- **Kozai 1959** ("The motion of a close earth satellite", *AJ* 64) and **Brouwer 1959** ("Solution of the problem of artificial satellite theory without drag", *AJ* 64) — the **closed-form J2 secular and long-period mean-element rates** that SGP4 builds on. These two papers give the regression formulas:
  - `Ω̇_sec = -3/2 · n · J2 · (R⊕/p)² · cos i`            (RAAN regression)
  - `ω̇_sec = +3/4 · n · J2 · (R⊕/p)² · (5cos²i - 1)`    (apsidal advance)
  - `Ṁ_sec = +3/4 · n · J2 · (R⊕/p)² · √(1-e²) · (3cos²i - 1)`
  with `n = √(μ/a³)`, `p = a(1-e²)`. Used to design **sun-synchronous orbits** (set Ω̇ = 360°/yr → i ≈ 98° at 800 km) and **frozen orbits** (e ≈ J3·sin(i)/(2·J2) at ω = 90° or 270°). Trivial closed-forms — every textbook (Vallado §9.6.1, Curtis §12.7, Battin §10.3) gives them.
- **Liu 1969** "Satellite Motion about an Oblate Earth" (AIAA J. 12) — alternative analytic theory (osculating-element form), historical reference.

### F5. EGM gravity-model sources
- **Lemoine et al. 1998, NASA/TP-1998-206861** — EGM-96 release: 360×360 spherical-harmonic coefficients (~130k terms), the prior generation reference. Plain-text coefficient files (~10 MB). Adequate for ground-tracking and weather-orbit applications.
- **Pavlis, Holmes, Kenyon, Factor 2008** — EGM-2008 release: 2160×2160 (~4.7 M terms, ~80 MB), the current geodetic-grade Earth gravity model. Used by IAU 2010 / IERS 2010 conventions. For an in-library N×N truncated gravity model, ship only **N ≤ 8** (45 + tesseral pairs ≈ 80 coefficients, fits in a ~3 KB Go literal); higher orders are diminishing returns vs. drag uncertainty for any real LEO simulation.
- **GRACE / GRACE-FO** subsequent releases (GGM05, JGM-3): incremental refinements; not needed for a pedagogical math library.

### F6. Atmospheric models
- **NRLMSISE-00** (Picone, Hedin, Drob, Aikin 2002, *JGR Space Physics* 107 A12) — the reference upper-atmosphere density model. C reference implementation (~2000 LOC, public domain). Inputs: F10.7 solar flux, Ap geomagnetic index, location, time. Required for **realistic LEO decay forecasting**; without it, exponential ρ(h)·exp(-(h-h₀)/H) (Vallado §8.6.2 Table 8-4, 28 layers, ~50 LOC) is a 10× underestimate during solar max.
- **JB2008 / NRLMSIS 2.0** — newer JB-series replacements; JB2008 (Bowman et al.) is the AFRL operational model. Defer.
- **MET (Marshall Engineering Thermosphere)** — historical NASA model, mostly superseded.

### F7. Numerical-propagator landscape and what `chaos/` already gives us
- `chaos/ode.go` already exposes `RK4Step` and `SolveODE` (verified via test references at `chaos/chaos_test.go:18,46,67,89,110,161`). A Cowell propagator is therefore **a thin wrapper** over `chaos`: `r̈ = -μ·r/r³ + Σ a_perturb_i(r,v,t)` with state `[6]float64 = (r, v)`. ~120 LOC if a `Perturbation func(s State) [3]float64` interface is defined.
- **Encke's method** (Vallado §8.7.2) — integrates the deviation `δr = r - r_kepler` from a Keplerian reference orbit; ~3-5× larger time steps than Cowell when perturbations are small (LEO J2 only). Requires re-osculation when |δr|/|r| > 0.01. Niche; defer to T4.
- **Variation-of-Parameters / Lagrange Planetary Equations** (Vallado §9.6, Battin Ch. 10) — integrates the elements `(a,e,i,Ω,ω,M)` themselves. Long-time-step stability; the foundation of DSST and SGP4 mean-element machinery. Substantial (~250 LOC) but the right home for any future `MeanElementPropagator`.
- **Symplectic integrators** (Wisdom-Holman 1991, Yoshida 1990, IAS15 Rein-Spiegel 2015) — bounded-energy-error multi-Gyr solar-system integration. Slot 204 (new-symplectic-int) covers; mention here only as the right tool for **third-body N-body**, not perturbed-Kepler LEO.

### F8. R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities
The CLAUDE.md "golden files are the proof" rule is satisfied today via Python/C++/C#-against-Go big-rational vectors. Perturbation primitives unlock a stronger cross-validation pattern: **multiple independent algorithms must agree**. Three concrete pins for this slice:
1. **J2 secular ≡ Kozai 1959 closed-form** (regression test, single file). Two routes: (a) call closed-form `J2SecularRates(a,e,i)` → get `(Ω̇, ω̇, Ṁ)`; (b) Cowell-integrate J2 + two-body for one orbit, fit secular slope of `(Ω(t), ω(t), M(t))` via linear regression. Match to 1e-6 relative for 800 km / 0.001 / 98° SSO test orbit. Cites: Kozai 1959, Brouwer 1959, Vallado §9.6.1.
2. **SGP4 ≡ Vallado AIAA 2006-6753 reference vectors** (golden-file). 32 TLEs × 0/1/360/720/1440/2880/4320/5760/7200 min sample times × (rₓ,r_y,r_z,vₓ,v_y,v_z) ≈ 1700 floats per fixture, gz to ~50 KB total. Exact-bit (1e-13) match on Vallado's published `tcppver.out` / `tmatver.out`. Cites: AIAA 2006-6753; Brandon Rhodes' `sgp4` package (Apache-2 / MIT-OK).
3. **Cowell + J2 + drag ≡ SGP4** for ISS-altitude TLE within 1 km after 1 day (regression). Independent algorithm: Cowell numerical integration of the **same** force model SGP4 approximates (J2-J5 + simplified drag). Confirms Cowell wiring + perturbation interface independent of the SGP4 port. Tolerance: 1 km after 86400 s integration of a 7000 km orbit (1.4×10⁻⁴ relative).

### F9. Cross-link to other slots
- **107-orbital-missing** — comprehensive missing-features audit; J2 acceleration / J3-J4 / drag / SRP / 3body / SGP4 / Encke are in T1.7, T2.1, T2.2, T2.3, T2.4, T2.6, T2.8 of that review. This dive deepens those entries with sources and pin opportunities.
- **108-orbital-sota** — likely covers poliastro/Orekit/GMAT/NORAD/Brandon-Rhodes-sgp4 SOTA; cross-cite.
- **106-orbital-numerics** — per-line audit of existing code; F1 there is the parabolic singularity in `TrueAnomalyFromMean`.
- **187-synergy-orbital-control** — combining `control/` PID with orbital propagation for station-keeping; J2 secular rates are the disturbance the PID rejects in the simplest GEO N-S station-keeping demo.
- **164-synergy-orbital-optim** — cousin slot; Lambert + low-thrust optimization needs Cowell to evaluate a candidate trajectory.
- **204-new-symplectic-int** — symplectic integrators are the right substrate for **Sun-Earth-Moon** restricted-three-body, not for perturbed-Kepler LEO (Cowell suffices). Don't conflate.
- **47-constants-missing** — likely tags `J2Earth`, `R_Earth`, `μ_⊕` as missing constants; align constant naming **before** writing perturbation code.

### F10. Mean-elements vs osculating-elements API trap
A common pitfall: TLEs encode **mean Brouwer elements** (specifically: SGP4-flavoured Brouwer-Lyddane mean elements with Kozai-mean-motion), not osculating elements. A user who types `RVfromCOE(tle.a, tle.e, tle.i, ...)` directly into a two-body Kepler routine without the SGP4 mean→osculating transformation will be wrong by ~kilometers. **Any future `orbital/sgp4` sub-package must expose only `SGP4(tle, dt) (r, v State)` — never expose the mean elements** unless wrapped in a distinct `MeanElements` type. The time-scale lattice trap in 107-F (TT/TAI/UTC/UT1) generalises here: **element type as a Go named type prevents the mistake at compile time**.

## Concrete recommendations

1. **Day-1 PR (T0, ~80 LOC).** Add `orbital/perturb_j2.go` with three closed-form secular rates `J2SecularNodalRate(a,e,i)`, `J2SecularApsidalRate(a,e,i)`, `J2SecularMeanAnomalyRate(a,e,i)` returning rad/s, plus the convenience wrapper `SunSynchronousInclination(a, e) float64` solving Ω̇ = 360°/yr → i ≈ acos(...). Pre-requisite: extend `constants/` with `J2Earth`, `EarthRadiusEquatorial`, `EarthMu` per F2. 30 golden vectors generated against Vallado §9.6.1 worked examples + Curtis Example 12.6 + a 800 km / 98° SSO sanity case. **One commit, no chaos/ dependency, validates against published textbook numbers.**

2. **T1 (~120 LOC) — J3 + J4 zonal closed-forms.** Add J3-induced periodic eccentricity & argument-of-perigee perturbation (the **frozen-orbit** equation `e_frozen = -J3·sin(i)/(2·J2)` at ω = 90°/270°) and J4 secular contribution. References: Vallado §9.6.1, Brouwer 1959. Same file. 20 vectors per function.

3. **T1 (~120 LOC) — `J2Acceleration(r [3]float64) [3]float64`.** Vector-form acceleration (Vallado §8.6.1 eq. 8-37). The atom that any numerical propagator composes. Define the `type Perturbation func(s State) [3]float64` contract in this PR; all later perturbations (J3-vector, drag, SRP, third-body) implement the same signature.

4. **T2 (~250 LOC) — Cowell propagator.** New `orbital/propagator/cowell.go`. State = `[6]float64`. Wraps `chaos.SolveODE` with the perturbed two-body RHS `r̈ = -μ·r/r³ + Σ a_p_i`. Variadic `Perturbations ...Perturbation` argument. Pin via R-MUTUAL: J2-only Cowell secular slope ≡ closed-form J2 rates from rec. (1).

5. **T2 (~150 LOC) — exponential atmospheric drag.** `ExponentialDensity(altitude_m float64) float64` from Vallado Table 8-4 (28 layers, 0–1000 km). Then `DragAcceleration(r, v [3]float64, params DragParams) [3]float64` with `DragParams {CdA_over_m, OmegaEarth float64}`. Document **explicitly** that this is 5–10× wrong during solar max; reference NRLMSISE-00 as the production model. Uses exponential pieces only — zero new deps.

6. **T3 (~600–1500 LOC, multi-week) — `orbital/sgp4` sub-package.** Bit-exact port of Vallado 2006 `sgp4unit.cpp`. **Ship the AIAA-2006-6753 acceptance test vectors as goldens** (32 TLEs × 9 sample epochs); compliance bar = match Vallado's `tcppver.out` to 1e-7 m / 1e-10 m/s. Single biggest piece of work in the package's roadmap; biggest unlock (every public TLE on celestrak.org becomes propagatable). MIT-compatible upstream license (Vallado: "use freely"); cite explicitly in source headers. Define `TLE` as a typed struct with line-1/line-2 parsers (column-exact per the NORAD spec — beware the implicit decimal points in `bstar`).

7. **T3 (~150 LOC) — third-body Sun + Moon analytic ephemeris.** Meeus Ch. 25 (Sun) and Ch. 47 (Moon) trigonometric series, ~0.01° accuracy, sufficient for any LEO/MEO/GEO third-body perturbation. Lives in `orbital/ephem` per 107-T2.4. Use **Battin q-formulation** for the third-body acceleration to avoid catastrophic cancellation when r_sat ≈ r_earth (Battin §8.5.7).

8. **T3 (~120 LOC) — solar radiation pressure with cylindrical shadow.** `SRPAcceleration(r_sat, r_sun [3]float64, params SRPParams) [3]float64` with cylindrical Earth-umbra model (Vallado §8.6.4). Required for GEO and lightsail. Trivial once Sun ephemeris exists.

9. **T4 (deferred) — Encke's method.** ~200 LOC, ~3-5× speedup vs Cowell for low-perturbation LEO J2. Niche; ship only after Cowell + SGP4 are in production. Re-osculation logic is the only subtle part.

10. **T5 (deferred) — EGM-2008 N×N truncated.** ~300 LOC for the spherical-harmonic gravity computation (Pines or Cunningham recursion to avoid pole singularity), plus a coefficient table. Truncate at N ≤ 8 (3 KB literal) for in-tree; users wanting full 2160×2160 should pull the file separately. Validate against EGM-2008 `harmonic_synthesis.f` reference output. The frontier item.

11. **API decisions to lock now (zero LOC, prevent future churn).**
    - `type State struct { R, V [3]float64; T float64 }` — canonical state.
    - `type Perturbation func(s State) [3]float64` — return acceleration, no mutation.
    - `func Sum(ps ...Perturbation) Perturbation` — compositional.
    - `type MeanElements struct {...}` distinct from `type COE struct {...}` — prevent the F10 trap.
    - Document perturbation magnitudes (F3 table) in package godoc so users can pick truncations with calibrated intuition.

12. **R-MUTUAL pins to bake in.** Per F8, lock all three pins as `TestMutualXxx_3of3` markers in a `pins_test.go` file at PR time, mirroring the pattern in slot 365 (audio/onset cross-validation) and slot 366 (copula × autodiff Clayton gradient pin). Cheapest pin first: pin #1 (J2-secular ≡ Kozai closed-form) lands with the day-1 PR (rec. 1) and exercises both routes from the same commit.

## Sources

### Repo files
- `C:/limitless/foundation/reality/orbital/orbital.go` — 266 LOC, 8 functions, all pure Kepler two-body, no perturbations.
- `C:/limitless/foundation/reality/orbital/orbital_test.go` — 328 LOC, golden-file driven (8 fixtures).
- `C:/limitless/foundation/reality/orbital/testdata/orbital/` — 8 JSON fixtures (escape_velocity, hill_sphere, hohmann_transfer, kepler_orbit, orbital_period, orbital_velocity, synodic_period, true_anomaly).
- `C:/limitless/foundation/reality/chaos/chaos_test.go:18,46,67,89,110,161` — confirms `RK4Step` + `SolveODE` already exist, ready substrate for Cowell.
- `C:/limitless/foundation/reality/constants/physics.go:73` — only `StandardGravity` present; no Earth-shape, no μ⊕, no J2.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/107-orbital-missing.md` — comprehensive missing-features audit; this dive deepens its T1.7/T2.1/T2.2/T2.4/T2.6/T2.8.
- `C:/limitless/foundation/reality/CLAUDE.md` — golden-file rule §3, zero-dependency rule §2.

### Web / textbook sources
- **Hoots & Roehrich 1980**, "Spacetrack Report #3" — original SGP4/SDP4 spec.
- **Vallado, Crawford, Hujsak, Kelso 2006**, "Revisiting Spacetrack Report #3", AIAA-2006-6753 — modern reference implementation + acceptance test suite.
- **Vallado, "Fundamentals of Astrodynamics and Applications" 4th ed.** — §8 perturbations, §8.6 zonal harmonics, §8.7 Cowell/Encke, §9 SGP4, §9.6 mean-element propagation.
- **Kozai 1959**, "The motion of a close earth satellite", *Astronomical J.* 64 — J2 secular & long-period mean-element rates.
- **Brouwer 1959**, "Solution of the problem of artificial satellite theory without drag", *AJ* 64 — companion mean-element theory.
- **Lyddane 1963** — singularity-free Brouwer mean elements (low e, low i).
- **Liu 1969**, "Satellite motion about an oblate Earth", AIAA J. 12 — alt analytic theory.
- **Lemoine et al. 1998**, NASA/TP-1998-206861 — EGM-96 (360×360).
- **Pavlis, Holmes, Kenyon, Factor 2008** — EGM-2008 (2160×2160), IERS 2010 reference gravity model.
- **Picone, Hedin, Drob, Aikin 2002**, *JGR Space Physics* 107 A12 — NRLMSISE-00 atmospheric density.
- **Battin 1987**, "An Introduction to the Mathematics and Methods of Astrodynamics" — Ch. 10 variation-of-parameters; §8.5.7 q-formulation for third-body cancellation.
- **Curtis 2014**, "Orbital Mechanics for Engineering Students" 3rd ed. — §10.2 Cowell, §12.7 J2 perturbation worked examples.
- **Murray & Dermott 1999**, "Solar System Dynamics" — Ch. 3 perturbation theory.
- **Montenbruck & Gill 2000**, "Satellite Orbits — Models, Methods, Applications" — §3 Table 3.1 magnitude tabulation.
- **Brandon Rhodes' `sgp4` Python package** (MIT) — canonical clean port of Vallado 2006 reference; cross-check target for any Go re-implementation.
- **celestrak.org** — operational TLE feed; SGP4 propagation is implicitly required to consume.
