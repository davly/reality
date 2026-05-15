# 107 — orbital: Missing-Features Audit (canonical primitives gap)

**Scope.** `C:/limitless/foundation/reality/orbital/orbital.go` (267 LOC, **8 exported functions**) + tests + 8 golden vector files. This audit enumerates canonical orbital-mechanics primitives **not yet present** in the package, ranked Tier 1/2/3 by leverage. Companion to 106-orbital-numerics (per-line audit of the existing 8 functions).

**Sibling packages cross-referenced.** `chaos/ode.go` (RK4 / RKF45 generic ODE step — required by every Cowell/Encke/N-body propagator; already in tree, no new dep), `linalg/` (matrix-vector products for state-transition matrices), `calculus/` (root-finding for Lambert / universal-variable solvers — bisection + Newton already present), `constants/` (μ_⊕, R_⊕, J2, c, AU — partial: J2 not yet a constant).

**Reference libraries audited (web research).**
- **poliastro** (Python, ~28k LOC, JPL Spice-grade) — superset of features here, MIT, Astropy-ecosystem; canonical reference for API shapes.
- **Orekit** (Java, CNES/ESA, Apache-2) — production-grade, SGP4/SDP4/Eckstein-Hechler/numerical/DSST propagators, all IERS frame conventions.
- **GMAT** (NASA, Apache-2, ~500k LOC) — ground-truth propagator for spacecraft mission design.
- **Brandon Rhodes' sgp4** (Python/C, MIT, NORAD-grade) — bit-exact port of Vallado's reference SGP4/SDP4 implementation; the standard against which every TLE propagator is validated.
- **AstroPy** (Python, BSD-3) — coordinate frames (ICRS/GCRS/CIRS/ITRS/AltAz), time scales (TT/TAI/UTC/TCB/TDB/UT1/GPS), IERS Earth-orientation lookup; the de facto Python ecosystem for #astrometry.
- **Vallado "Fundamentals of Astrodynamics and Applications"** (4th ed.) — the canonical English-language textbook; Ch. 7 (Lambert), Ch. 8 (perturbations), Ch. 3 (frames), Ch. 9 (SGP4) are the sources for everything below.

**What is currently in `orbital/`.** `KeplerOrbit`, `OrbitalPeriod`, `OrbitalVelocity` (vis-viva), `HohmannTransfer`, `EscapeVelocity`, `HillSphere`, `SynodicPeriod`, `TrueAnomalyFromMean`. Eight textbook two-body closed forms. **No** propagator, **no** Lambert solver, **no** perturbation model (J2/J3/drag/SRP/third-body), **no** SGP4, **no** time-scale conversions, **no** frame transforms, **no** topocentric (az/el/range), **no** anomaly-conversion symmetry (only M→ν is exposed; ν→E, E→M, ν→M, E→ν are not). The package is the smallest in `reality` by feature count; the gap to "credible astrodynamics library" is **~3,500–5,000 LOC of additive, zero-dependency Go**.

---

## Tier 1 — Highest leverage (ship next sprint, ≈ 1,400–1,800 LOC, 18 functions)

These eleven items unlock 80 % of practical use cases (orbit determination, transfer design, Earth-satellite analysis) and are prerequisites for everything in Tier 2/3. Each is a **one-file textbook formula** with established golden-file references in poliastro / Vallado.

### T1.1 — Universal-variable Kepler propagator (Stumpff C(z), S(z)) — ~250 LOC
**The single highest-leverage missing primitive in the package.** Replaces the eccentricity-branch case-split (elliptic/parabolic/hyperbolic) with one formula that handles all three conics. Solves `χ` from universal Kepler equation
`√μ·Δt = r₀·v_r₀/√μ · χ²·C(α·χ²) + (1-α·r₀)·χ³·S(α·χ²) + r₀·χ`
via Newton-Raphson, then Lagrange f, g, ḟ, ġ coefficients map (r₀, v₀, Δt) → (r, v) in one shot. Stumpff functions C(z) = (1-cos√z)/z, S(z) = (√z-sin√z)/z^(3/2) with series fallback for |z|<1 to avoid catastrophic cancellation. Reference: Bate-Mueller-White §4.4–4.5; Vallado §2.2 Algorithm 8 ("Kepler"); poliastro `core/propagation/markley.py`. Closes 106-F1 (parabolic singularity) by construction. Eliminates the e<1 restriction on `TrueAnomalyFromMean`.

### T1.2 — Anomaly-conversion symmetry (4 functions) — ~80 LOC
Currently only `M → ν` exists. Astrodynamics workflows need all six legs of the triangle:
- `MeanFromEccentric(E, e) float64` — `M = E - e·sin E` (elliptic) / `M = e·sinh F - F` (hyperbolic).
- `EccentricFromTrue(nu, e) float64` — `tan(E/2) = √((1-e)/(1+e))·tan(ν/2)`.
- `TrueFromEccentric(E, e) float64` — `tan(ν/2) = √((1+e)/(1-e))·tan(E/2)`.
- `MeanFromTrue(nu, e) float64` — composition of the two above.
Reference: Vallado §2.2.5 "Anomaly Relationships"; poliastro `core/angles.py`. Trivial closed forms; lack of these functions is a pure API completeness bug.

### T1.3 — Hyperbolic Kepler equation `M = e·sinh F − F` — ~70 LOC
Hyperbolic-orbit Newton solver mirroring `TrueAnomalyFromMean`. Initial guess `F_0 = ln(2|M|/e + 1.8)` (Conway-Prussing 1986). Required for **interplanetary trajectories** (any flyby, any escape orbit), absent today. Reference: Vallado §2.2.4 "Hyperbolic Anomaly"; Battin §4.3.

### T1.4 — Parabolic / Barker's equation — ~50 LOC
`M_p = D + D³/3` with closed-form Cardano solution `D = (3M_p/2 + √(1 + 9M_p²/4))^(1/3) - (-3M_p/2 + √(1 + 9M_p²/4))^(1/3) inverse`. No iteration, exact. Required for the e=1 boundary case (interstellar comets, escape trajectories at exactly v_∞=0). Reference: Vallado §2.2.6; Bate-Mueller-White §4.3.

### T1.5 — State-vector ↔ Keplerian-elements conversion (2 functions) — ~180 LOC
The package has `KeplerOrbit (a,e,i,Ω,ω,ν → x,y,z)` (one-way, position-only). Missing the **full state** versions, which are the workhorses of orbit determination:
- `RVfromCOE(a, e, i, raan, argp, nu, mu) (r, v [3]float64)` — adds velocity vector.
- `COEfromRV(r, v [3]float64, mu) (a, e, i, raan, argp, nu, ecc_anomaly float64)` — the inverse (essential for orbit determination from radar/optical observations). Singularity handling for circular (e=0) and equatorial (i=0) orbits via argument-of-latitude / true longitude. Reference: Vallado §2.5 Algorithm 9 ("RV2COE") and Algorithm 10 ("COE2RV"); poliastro `twobody/classical.py`.

### T1.6 — Lambert problem: Izzo's universal solver — ~350 LOC
**The second-highest-leverage missing primitive.** Given r₁, r₂, Δt, solve for the velocity vectors that connect them — i.e., "what orbit goes from A to B in this time?". Required for **every transfer problem** (Hohmann is the special case of two coplanar circular orbits; Lambert is the general case). Izzo (2014) is the modern SOTA: householder iteration on Battin's "x" parameter, single closed-form initial guess, converges in 2–3 iterations to machine precision for any transfer angle, handles multi-revolution solutions. Reference: Izzo, "Revisiting Lambert's problem", *Celestial Mechanics & Dynamical Astronomy* 121 (2014); poliastro `iod/izzo.py` (canonical Python port). Battin (1987) and Gauss (1809) algorithms documented as alternatives but Izzo strictly dominates on robustness + speed.

### T1.7 — J2 zonal-harmonic perturbation acceleration — ~80 LOC
`a_J2 = -3/2 · J2 · μ · R²/r⁵ · [(1 - 5(z/r)²)·x̂ + (1 - 5(z/r)²)·ŷ + (3 - 5(z/r)²)·ẑ]·(z/r)·(R/r)` (factored Vallado form). J2 is the dominant non-spherical Earth perturbation (~1000× larger than J3/J4); without it, no LEO satellite analysis is meaningful. Reference: Vallado §8.6.1 "Zonal Harmonics"; J2_⊕ = 1.0826267×10⁻³ (CODATA / IERS 2010). Add J2 to `constants` package as a new entry. **Plus**: secular rates ΔΩ̇, Δω̇, ΔṀ from J2 (closed-form Brouwer first-order theory) for analytical mean-element propagation — ~50 LOC, enables sun-synchronous-orbit design.

### T1.8 — Cowell's method (direct numerical propagator) — ~120 LOC
Wraps `chaos/ode.go` RKF45 around the perturbed two-body equation `r̈ = -μ·r/r³ + a_perturb(r, v, t)`. Accepts a slice of `Perturbation func(r, v [3]float64, t float64) [3]float64` for compositional accelerations (J2, drag, SRP, third-body). The simplest and most general numerical propagator. Reference: Vallado §8.7 "Cowell's Method"; Curtis "Orbital Mechanics for Engineering Students" §10.2. Requires a thin `State{R, V [3]float64; T float64}` struct.

### T1.9 — Time-scale conversions (TT, TAI, UTC, GPS, JD, MJD) — ~150 LOC
Six pairwise conversions on the time-scale lattice plus Julian-date arithmetic:
- `JDfromGregorian(y, m, d, h, mn, s int, frac float64) float64` — Meeus §7.
- `GregorianFromJD(jd float64) (y,m,d,h,mn,s,frac)`.
- `MJDfromJD(jd) = jd - 2400000.5`.
- `TAIfromUTC(utc, leap_seconds int)` / inverse — leap-second table (37 entries 1972-01-01 through 2017-01-01, hardcode then expose updater).
- `TTfromTAI(tai) = tai + 32.184 s` (constant offset).
- `GPSfromUTC` = TAI − 19s, etc. Reference: Vallado §3.5; SOFA `iauUtctai`, `iauTaitt`. **Note:** TDB requires Fairhead-Bretagnon series — defer to Tier 2 (1100-term polynomial fit).

### T1.10 — Topocentric (azimuth, elevation, range) from ECI/ECEF — ~120 LOC
`AzElRange(observer_lat, observer_lon, observer_alt, target_ecef [3]float64) (az, el, range float64)` — ENU rotation matrix from geodetic lat/lon, then range-vector decomposition. The minimum primitive needed for **ground-based satellite tracking** (any pointing problem, any visibility analysis). Plus `GeodeticFromECEF` and `ECEFfromGeodetic` (Bowring 1976 closed-form, no iteration needed for Earth eccentricity). Reference: Vallado §3.4 "Geodetic Latitude"; §4.4.3 "Topocentric Coordinates".

### T1.11 — Hill / Clohessy-Wiltshire equations (relative motion) — ~150 LOC
Linearised relative-motion equations for a chaser spacecraft about a target in circular orbit:
`ẍ = 3n²x + 2nẏ + a_x`, `ÿ = -2nẋ + a_y`, `z̈ = -n²z + a_z` where n = √(μ/a³).
Closed-form CW state-transition matrix Φ(t) gives the impulse transfer between any two relative states (rendezvous targeting). Required for **ISS/Starship rendezvous, formation-flying, satellite servicing**. Reference: Clohessy-Wiltshire (1960); Vallado §6.7 "Relative Motion"; Curtis §7.4. Bonus: Tschauner-Hempel equations for elliptic-target case (~50 LOC additional, Tier 2).

**Tier 1 totals.** ~1,600 LOC, 21 new exported functions, ~5–7 new golden-file fixtures (one per major routine). Closes the gap from "two-body-only textbook calculator" to "credible Earth-satellite analysis library on par with the basic poliastro feature set". Every item is pure-Go, zero new dependencies, deterministic, golden-file-friendly.

---

## Tier 2 — Substantial leverage (next-quarter sprint, ≈ 2,000–2,500 LOC, 25 functions)

### T2.1 — Higher-order zonal harmonics J3, J4 (and triaxial C₂₂, S₂₂) — ~200 LOC
J3, J4 corrections (each one order of magnitude smaller than J2 but **non-zero secular effects on argument-of-perigee for Molniya / Tundra orbits**). J3 is the largest source of **frozen-orbit** design (e=0.001, ω=270° for sun-synchronous repeat-track satellites). Reference: Vallado §8.6.1; Brouwer (1959). Constants J3=-2.5327×10⁻⁶, J4=-1.6196×10⁻⁶ (EGM2008/IERS).

### T2.2 — Atmospheric drag acceleration (exponential + NRLMSISE-00 stub) — ~350 LOC
`a_drag = -½·ρ(h)·v_rel²·(C_D·A/m)·v̂_rel`. Two density models:
- **Exponential model** (~50 LOC): ρ(h) = ρ₀·exp(-(h-h₀)/H) with Vallado §8.6.2 Table 8-4 scale-height tabulation (28 layers, 0–1000 km).
- **NRLMSISE-00 stub** (~300 LOC): the reference upper-atmosphere model (NASA Goddard, 2002) for high-precision LEO decay prediction. Full implementation requires F10.7 solar-flux and Ap geomagnetic-index inputs (parametric). Either ship the exponential and document NRLMSISE as Tier 3, or port the model directly (Picone et al. 2002; reference C implementation ~2000 LOC under public domain).

### T2.3 — Solar radiation pressure (SRP) — ~120 LOC
`a_SRP = -P_SR·(C_R·A/m)·(AU/r_sun)²·r̂_sun` where P_SR = 4.56×10⁻⁶ N/m² (solar constant 1361 W/m² ÷ c). Cylindrical Earth-shadow model (umbra/penumbra) for eclipse-aware integration. Required for **GEO satellites, interplanetary missions, lightsails**. Reference: Vallado §8.6.4; Montenbruck & Gill §3.4.

### T2.4 — Third-body perturbations (Sun, Moon) — ~150 LOC
`a_3body = μ_3·[(r_sat→3 / |r_sat→3|³) - (r_earth→3 / |r_earth→3|³)]`. The **Battin-formulated** form avoids catastrophic cancellation when r_sat→3 ≈ r_earth→3 (small q-formulation, q = -2·r_sat·r_earth/|r_earth|² + |r_sat|²/|r_earth|²; Battin §8.5.7). Requires lunar/solar position — provide a low-precision analytic ephemeris (Meeus Ch. 47/25, ~0.01° accuracy, ~250 LOC) as a separate `orbital/ephem` sub-package; defer JPL DE-series (~10 MB binary kernel) to Tier 3.

### T2.5 — Restricted three-body problem (CR3BP) — ~250 LOC
- `LagrangePoints(mu_ratio float64) [5][3]float64` — L1, L2, L3 via Newton on the collinear-Lagrange quintic; L4, L5 via 60° equilateral triangle (closed form).
- `JacobiConstant(r, v [3]float64, mu_ratio) float64` — energy invariant for trajectory classification.
- `CR3BPDerivative(state [6]float64, mu_ratio float64) [6]float64` — equations of motion in synodic frame for use with the chaos/ode.go RKF45 integrator.
- `LyapunovStability(L_index int, mu_ratio float64) (lambda complex128, type string)` — characteristic-equation roots at each Lagrange point (L1/L2/L3 saddle-centre-centre, L4/L5 stable iff mu_ratio < 0.0385 = Routh's value). Required for **JWST L2 station-keeping, lunar gateway design, Sun-Earth L1 mission planning**. Reference: Szebehely "Theory of Orbits" (1967); Murray & Dermott Ch. 3.4.

### T2.6 — Encke's method (perturbation w.r.t. Keplerian reference) — ~150 LOC
Integrates only the **deviation** δr = r - r_ref from a Keplerian reference orbit. Larger time steps than Cowell when perturbations are small (LEO J2-only: ×3–5 speedup). Requires careful rectification (re-osculate the reference orbit when |δr|/|r| > 0.01) to maintain validity. Reference: Vallado §8.7.2; Bate-Mueller-White §9.3. Encke's "f-and-g" formulation avoids cancellation in the (1 - (r_ref/r)³) factor via series.

### T2.7 — Variation of Parameters (VOP) — Lagrange / Gauss planetary equations — ~250 LOC
Equations of motion for the **orbital elements themselves** (rather than r, v):
da/dt = (2/n)·∂R/∂σ, de/dt = ((1-e²)/(n·a²·e))·[(1-e²)·∂R/∂σ - ...] — Gauss form for non-conservative perturbations (drag, thrust); Lagrange form for conservative (gravitational). Long time-step integration of secular drift; basis of mean-element propagators (Brouwer-Lyddane, DSST, SGP4). Reference: Vallado §9.6; Battin Ch. 10.

### T2.8 — SGP4 / SDP4 (NORAD propagator from TLE) — ~1,200 LOC
**The most-used satellite propagator in the world** — every TLE on celestrak.org is propagated by SGP4. SDP4 = deep-space variant (period > 225 min: GPS, GEO, Molniya, Moon-resonance). Bit-exact port of Vallado's reference C++ implementation (BSD-style license, ~1500 LOC). Validated against the AIAA-2006-6753 official test suite (32 reference TLEs, 7 days propagation, all components agree to 0.01 m). Reference: Hoots & Roehrich, "Spacetrack Report #3" (1980); Vallado, "Revisiting Spacetrack Report #3" (AIAA 2006-6753). **The hardest item in this entire audit** to ship correctly — recommend a dedicated `orbital/sgp4` sub-package with goldens straight from the AIAA test suite (32 fixtures × 0–8000 min × 1 min step = 256 000 vectors, gzip well).

### T2.9 — Gravity assist (patched-conic flyby) — ~150 LOC
- `FlybyDeltaV(v_inf_in [3]float64, v_planet [3]float64, periapsis_alt float64, mu_planet float64) (v_inf_out [3]float64, deflection_angle float64)` — closed-form turn-angle δ = 2·asin(1/(1 + r_p·v_inf²/μ)).
- Patched-conic SOI transition: hand off state from heliocentric to planetocentric at the sphere-of-influence boundary. Reference: Bate-Mueller-White §8.3; Curtis Ch. 8.

### T2.10 — Bi-elliptic transfer + general two-impulse transfer — ~80 LOC
`BiellipticTransfer(r1, r2, r_b, mu) (dv1, dv2, dv3)` for r2/r1 > 11.94 (the bi-elliptic-beats-Hohmann threshold). Also: phasing maneuver, plane change, combined plane-change + raise. Reference: Vallado §6.3–6.5; Curtis §6.4.

### T2.11 — IAU 2006/2010 precession-nutation, ECI ↔ ECEF (CIO-based) — ~600 LOC
Full IAU 2006 (Capitaine) precession + IAU 2000A nutation + Earth-rotation-angle (ERA) chain to convert GCRF (J2000-equivalent inertial) to ITRF (Earth-fixed). The 2000A nutation series has 1365 luni-solar + 687 planetary terms; full implementation is ~500 LOC of pure trigonometric series + ~50 LOC framework. Earth Orientation Parameters (EOP) — UT1−UTC, polar motion x_p, y_p, ΔX, ΔY corrections — must be **ingested from an IERS Bulletin A file** (CSV, daily, ~500 KB/year, MIT-compatible IERS-EOP-C04 series). Provide a parser + linear interpolator. Reference: SOFA (Standards of Fundamental Astronomy, IAU); Vallado §3.7. **Largest single LOC item in this audit** but the foundation of every modern astrodynamics frame transformation; SGP4 (T2.8) is the only TLE-grade propagator that works without it.

### T2.12 — Tisserand parameter — ~25 LOC
`T_Jupiter = a_J/a + 2·√((a/a_J)·(1-e²))·cos(i)` — invariant for small-body classification (asteroid vs. comet vs. interstellar). Three-line formula but conceptually significant for small-body identification (e.g., ‘Oumuamua T_J = -1.28 → interstellar). Reference: Murray & Dermott §3.4.

**Tier 2 totals.** ~3,500 LOC, 35+ new functions, 8–12 new golden fixtures (SGP4 dominates LOC), 2 new sub-packages (`orbital/sgp4`, `orbital/ephem`). Lifts the package from "Earth-satellite basics" to "interplanetary mission design + operational TLE propagation".

---

## Tier 3 — Lower leverage / specialised (research-grade, ≈ 2,500–4,000 LOC, 20+ functions)

### T3.1 — Battin and Gauss Lambert algorithms — ~400 LOC
Izzo (T1.6) strictly dominates for accuracy and robustness, but Battin's "successive substitutions on x" is the historically referenced method (used by JPL Horizons internally), and Gauss's original Theoria Motus (1809) is of pedagogical / historical value. Add only if a user cites cross-validation requirements; otherwise Izzo alone suffices.

### T3.2 — Kustaanheimo-Stiefel (KS) regularisation — ~300 LOC
Maps the singular Kepler problem (r→0 collision singularity) to a regular four-dimensional harmonic-oscillator problem via spinor coordinates u₁..u₄. Required for **close-encounter trajectories, tether dynamics, very-low-perigee LEO satellites**. The fictitious-time variable s replaces t via `dt = r·ds`. Reference: Stiefel & Scheifele "Linear and Regular Celestial Mechanics" (1971); Bond & Allman "Modern Astrodynamics" Ch. 9. Niche but the canonical answer to "how do I integrate through periapsis without my time step exploding".

### T3.3 — Sundman transformation — ~80 LOC
Time-regularisation `dt = r^α·ds` (α=1 gives KS, α=3/2 gives Levi-Civita). Used as the time-stretching primitive for any regularisation scheme; lighter-weight than full KS. Reference: Sundman (1912) original; Battin §10.7.

### T3.4 — Symplectic N-body integrators — ~400 LOC
- Wisdom-Holman (1991) democratic-heliocentric splitting — the standard for solar-system long-term (10⁹ year) integration.
- Symplectic Yoshida (4th, 6th, 8th-order) compositions of leapfrog.
- HERMES / IAS15 (Rein-Spiegel 2015) adaptive integrator for close encounters.
Required for **planetary-system stability studies, exoplanet dynamics, asteroid hazard analysis**. Reference: Wisdom & Holman *AJ* 102 (1991); Rein & Tamayo *MNRAS* 452 (2015) for IAS15. Each integrator ~80–150 LOC; symplectic property guarantees secular-energy-error boundedness over 10⁹ orbits.

### T3.5 — General relativistic correction (PPN) — ~80 LOC
Schwarzschild post-Newtonian acceleration `a_GR = (μ/(c²·r³))·[(4·μ/r - v²)·r + 4·(r·v)·v]` (parameterised post-Newtonian, β=γ=1 for GR). Required for **Mercury-orbit perihelion precession (43 arcsec/century, the textbook GR test), GPS clock corrections, gravitational-wave timing**. Reference: Will *LRR* 2014 §3.2; Vallado §8.6.5. Tiny but high-prestige for any gravity-test mission.

### T3.6 — Continuous low-thrust trajectories — ~500 LOC
- **Edelbaum's analytic solution** (~50 LOC) — closed-form Δv for circle-to-circle low-thrust with plane change, the textbook starting point.
- **Sims-Flanagan transcription** (~200 LOC) — multiple-shooting NLP for low-thrust mission design.
- **Indirect / Pontryagin** (~250 LOC) — optimal-control formulation with switching function. Required for **electric-propulsion missions (Dawn, BepiColombo, JUICE, NASA Psyche)**. Heavy lift; defer until ion-thruster mission appears in user list.

### T3.7 — DSST (Draper Semi-analytical Satellite Theory) — ~1,200 LOC
The "right" propagator for long-term LEO/MEO/GEO analysis: averages out short-period perturbations analytically, integrates only the slow secular/long-period dynamics. Used by USSPACECOM's "Special Perturbations" catalog. Order of magnitude harder than SGP4 to implement correctly. Reference: Cefola, Long & Holloway *AAS* 1974; Orekit DSST module (~5000 LOC of Java, the most authoritative open implementation).

### T3.8 — TDB (Barycentric Dynamical Time) Fairhead-Bretagnon series — ~400 LOC
1000-term Fourier series for TDB−TT difference (~1.6 ms peak amplitude, dominated by 1.66-ms annual term). Required for **pulsar timing, JPL DE-ephemeris ingestion**. Closed-form replacement for the iterative Moyer formulation. Reference: Fairhead & Bretagnon *A&A* 229 (1990).

### T3.9 — JPL DE-series ephemeris reader — ~500 LOC + binary kernel
Chebyshev-polynomial planetary-position reader for DE440 (the current JPL ephemeris, valid 1550–2650 CE, ~100 MB binary). Provides **arcsecond-accuracy planetary positions** (vs. Meeus's 0.01° analytic series in T2.4). The kernel itself is public domain but ships as a separate data dependency — recommend a thin Go decoder against the canonical SPK binary format, no kernel bundled. Defer until a user actually needs JPL-grade ephemerides; Meeus suffices for 99% of mission-design work.

### T3.10 — HPOP (High-Precision Orbit Propagator) primitives — ~400 LOC
Glue layer combining T1.7 (J2) + T2.1 (J3/J4 + arbitrary-degree spherical-harmonic gravity, requires EGM2008 4°×4° truncated coefficient table ~1 MB) + T2.2 (drag with NRLMSISE-00) + T2.3 (SRP) + T2.4 (third-body) + T3.5 (relativity) + T3.7 (DSST mean-element averaging) + T2.11 (frame transforms). Not a new algorithm — an integration testbench. Add once all dependencies are in place; this is the validation harness against Orekit/GMAT/STK propagator output to AIAA-2006-6753 Group-3 acceptance limits.

**Tier 3 totals.** ~3,400 LOC, 20+ new functions, 4–6 new sub-packages. Each item is justified by a specific community / mission profile; ship only on demand.

---

## Cross-cutting structural recommendations (zero-LOC API decisions)

1. **Sub-package the propagators.** `orbital/propagator` (Cowell, Encke, Kepler-universal), `orbital/lambert` (Izzo, Battin, Gauss), `orbital/sgp4`, `orbital/frames`, `orbital/time`, `orbital/ephem`. Keeps the existing 8-function `orbital` core lean for users who want only textbook two-body math; lets propagator-grade code live in dedicated packages with their own goldens and bench suites. Direct precedent: `signal/dsp` vs. `signal/fft` separation in audio packages.
2. **State struct.** Define `type State struct { R, V [3]float64; T float64 }` as the canonical state representation. Every propagator and Lambert solver consumes/produces `State`. Trivial but prevents the "is it (r,v,t) or (r,v) or [6]float64?" bikeshed war from happening 50 times.
3. **Perturbation interface.** `type Perturbation func(s State) [3]float64` (returns acceleration, no state mutation). All J2/drag/SRP/3body/GR fns implement this. Composable via `func Sum(ps ...Perturbation) Perturbation`. Critical for Cowell/Encke modularity.
4. **Frame enum + transform registry.** `type Frame uint8` (ICRF, GCRF, J2000, MOD, TOD, ITRF, ECEF, TEME, AGI). `type Transform func(s State, t float64) State`. Registry of pairwise transforms; chain via Floyd-Warshall on the registry graph. Same pattern that the `color` package uses for the 8 color-space conversion graph (precedent: `color/space.go`).
5. **Time-scale lattice as named types, not `float64`.** `type TT float64`, `type TAI float64`, etc. Compiler-checked: cannot accidentally pass UTC where TT is expected. The single highest-bug-prevention API decision in the entire orbital package.
6. **Perturbation magnitudes table in godoc.** For every perturbation (J2, J3, drag at 400 km, SRP at GEO, lunar 3body), document the typical acceleration magnitude in m/s² (e.g., J2_LEO ≈ 1.6×10⁻² m/s², drag_400km ≈ 1×10⁻⁷ m/s²). Lets users pick truncation thresholds with calibrated intuition rather than blind faith. Cross-reference: `acoustics/db.go` documents reference levels in the same shape.
7. **Constants additions.** Add to `constants/`: `J2Earth = 1.0826267e-3`, `J3Earth = -2.5327e-6`, `J4Earth = -1.6196e-6`, `MuSun = 1.32712440018e20`, `MuMoon = 4.9028e12`, `EarthRadiusEquatorial = 6378137.0` (WGS84), `EarthRadiusPolar = 6356752.3142`, `EarthFlattening = 1/298.257223563`, `SolarConstant = 1361.0`, `SolarPressure = 4.56e-6`, `LeapSecondsTAIminusUTC = 37` (current as of 2017-01; auto-updates not in scope). Each citation: NIST CODATA 2018, IERS 2010, EGM2008.
8. **`orbital/testdata` golden expansion.** Currently 8 files matching the 8 functions. After Tier 1: expect ~20 files (Lambert with multiple geometries, universal-variable across all three conics, J2 secular rates, CW transition, time-scale conversion at leap-second boundaries 2017-01-01 specifically). After Tier 2: ~40 files including the AIAA-2006-6753 SGP4 test set (32 TLE × ~10 sample times = 320 vectors per gz fixture). Per CLAUDE.md rule §3 (golden files) every new function ships with its 20–30 vectors before merge.

---

## Sprint ordering recommendation

- **Sprint 1 (v0.11, 2 weeks):** T1.1 universal-variable + T1.2 anomaly-symmetry + T1.3 hyperbolic Kepler + T1.5 RV↔COE conversions + T1.7 J2 acceleration + T1.9 time scales (no TDB). ~700 LOC, 14 functions, 6 new goldens. Closes 106-F1 (parabolic singularity) by replacing TrueAnomalyFromMean with universal-variable.
- **Sprint 2 (v0.12, 2 weeks):** T1.6 Izzo Lambert + T1.8 Cowell + T1.10 topocentric + T1.11 Hill-CW + T1.4 Barker + T2.12 Tisserand. ~700 LOC, 8 functions, 6 new goldens. Unlocks transfer design + relative motion + ground tracking.
- **Sprint 3 (v0.13, 4 weeks):** T2.11 IAU frames (subset: ECI↔ECEF only, full IAU 2000A nutation deferred) + T2.3 SRP + T2.4 third-body + T2.6 Encke + T2.10 bi-elliptic + T2.9 flyby. ~1100 LOC, 10 functions.
- **Sprint 4 (v1.0 candidate, 6 weeks):** T2.8 SGP4/SDP4 with full AIAA-2006-6753 golden suite. ~1200 LOC, ships the operational TLE propagator. **Goes to "credible Earth-satellite library" milestone here.**
- **Sprint 5+:** Tier 3 items on demand only.

Total to "credible operational astrodynamics library": ~3,700 LOC across 4 sprints, ≈ 14 weeks single-engineer effort. Total to "research-grade comprehensive" (all of Tier 3 except DE-series and DSST): another ~1,800 LOC across 6 weeks.

## Items intentionally NOT in scope for `reality/orbital`

- **JPL SPICE toolkit** (~600 KLOC C, NASA NAIF) — too large, requires binary kernel ecosystem; users can call SPICE-bindings from a separate package if needed.
- **Liaison / GMAT-grade mission scripting** — application-layer, belongs in a mission-design tool, not a math library.
- **Manoeuvre-targeting NLP solvers** — belongs in `optim/` (already 102-tier-1 work) with a thin orbital wrapper.
- **GUI / 3D visualisation** — out of scope for `reality` per CLAUDE.md (library, not service).
- **Real-time TLE fetching from celestrak / space-track.org** — network I/O, violates zero-dep rule.
- **Cesium / SpiceyPy / Skyfield API mimicry** — those are application-layer; reality should expose primitives, let downstream apps build the user-facing API surface.

---

**Summary.** orbital v0.10.0 ships 8 textbook two-body functions (correct on the happy path per 106 audit) and is **missing the Lambert solver, every propagator, every perturbation model, every frame transform, every time-scale conversion, every TLE-grade primitive, and Hill-Clohessy-Wiltshire relative motion**. Tier-1 sprint plan (~1,600 LOC, 21 functions, ~14 days work) closes the gap to "credible Earth-satellite analysis library"; Tier-2 (~3,500 LOC, 35 functions, ~10 weeks) closes the gap to "operational TLE propagation + interplanetary mission design"; Tier-3 (~3,400 LOC, 20 functions) is research-grade specialised. Every item is pure-Go, zero-dep, golden-file-friendly, with established formulas in Vallado / Battin / Murray-Dermott / Bate-Mueller-White / poliastro / Orekit. **The single highest-leverage commit** is T1.1 universal-variable Stumpff propagator (250 LOC) — replaces the eccentricity case-split, closes the 106-F1 parabolic singularity bug, unlocks every downstream propagator. **The second-highest** is T1.6 Izzo Lambert (350 LOC) — unlocks every transfer-design problem. Priority sequence: T1.1 → T1.5 → T1.6 → T1.8 → T1.7 → T2.8.
