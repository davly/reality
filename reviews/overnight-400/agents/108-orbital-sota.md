# 108 — orbital: SOTA Library Comparison (poliastro, Orekit, GMAT, STK, pykep, sgp4-py, Skyfield, Astropy, ML-augmented)

**Scope.** A head-to-head comparison of the canonical orbital-mechanics libraries that `reality/orbital` will be benchmarked against. Where 106 audits per-line numerical correctness of the existing 8 functions and 107 enumerates *what* is missing as a Tier-1/2/3 backlog, this review answers a different question: *for each SOTA library, what is its headline algorithmic contribution, what is the single engineering trick that makes it fast or robust, and is that trick portable into a zero-dependency Go library?* The output is a portability matrix that the Tier-1 backlog in 107 should be implemented against.

**Subject under review.** `C:/limitless/foundation/reality/orbital/orbital.go` (266 LOC, 8 closed-form functions, 8 golden vector files, 0 propagators, 0 frame transforms, 0 perturbations). The package is the smallest in `reality` by feature count.

**Method.** Web-research of each library's documentation and reference papers (Vallado 2006, Izzo 2014, Curtis 2014, Walker-Owens 2010 review, ESA-ACT publications). Cross-checked against the existing 107 reference list to avoid duplication: 107 catalogs *what* features are missing; this review catalogs *how* the SOTA libraries implement them and which implementations transfer to Go-stdlib-only.

---

## Library-by-library: headline algorithm × engineering trick × zero-dep portability

### 1. poliastro (Python, MIT, ~28k LOC) — `docs.poliastro.space`

**Headline algorithm.** Markley's (1995) bounded-error single-iterate Kepler solver: a fifth-order minimax polynomial in `(M, e)` produces an initial guess accurate to ≤2⁻²² over the entire elliptic domain, after which one Halley iteration converges to ≤2⁻⁴⁹. *Zero* iteration count in the worst case for elliptic orbits, all branches handled. Implemented in `core/propagation/markley.py`.

**Engineering trick.** Numba JIT-compiled core kernels (`@jit(nopython=True)`) called from a high-level Astropy-`Quantity`-typed wrapper. The hot path is pure-Python double-arithmetic compiled to LLVM at first call; the user sees `Orbit.from_vectors(Earth, r, v)` with units, but the propagator is a tight inner loop over arrays. Markley's polynomial is *especially* favourable to Numba because it has no branches on the hot path.

**Zero-dep portability for `reality`.** **HIGH.** Markley 1995 is 28 published polynomial coefficients + one Halley step — zero external dependencies, copy-paste-translatable into Go. Recommend implementing as `MarkleyKepler(M, e float64) (E float64)` to replace the Newton loop in `TrueAnomalyFromMean` (closes 106-F1 silent-non-convergence and 106-F2 high-e cost in one stroke). The Astropy units layer is *not* portable to a zero-dep Go library and is *not* desirable in `reality` — units belong in a higher tier (per CLAUDE.md "zero dependencies").

**What NOT to copy.** The CowellPropagator's reliance on `scipy.integrate.solve_ivp`'s DOP853 — `chaos/ode.go` already has RK4/RKF45, agent 107-T1.8 should compose on those, not reimplement DOP853 (could be a Tier-3 add later).

---

### 2. Orekit (Java, Apache-2, CNES/ESA, ~250k LOC) — `orekit.org`

**Headline algorithm.** *Unified propagator interface* (`o.o.propagation.Propagator`) above analytical (Eckstein-Hechler J2-J6, Brouwer-Lyddane, SDP4/SGP4, Keplerian, GNSS), semi-analytical (DSST — Draper Semianalytic Satellite Theory, mean-element propagation in 1/100th the steps of numerical), and numerical (Cowell on Hipparchus integrators) families. One line `setPropagator(...)` switches fidelity.

**Engineering trick.** **Force-model composition by side-effect**: every `ForceModel` registered with `NumericalPropagator` is called once per integration step with `(state, params) -> contribution`, and contributions are *added* into the time-derivative vector by the propagator. This is a tagged-component pattern — gravity, drag, SRP, third-body, relativity, empirical — each is an independently testable, golden-file-validatable struct. Lets you stack J2+J3+drag+Sun+Moon by `propagator.AddForceModel(j2)` etc. without modifying the integrator.

**Zero-dep portability for `reality`.** **HIGH** for the *pattern*, **MEDIUM** for the *content*. The pattern (slice of `Perturbation func(r, v [3]float64, t float64) [3]float64` reduced into `chaos/ode.go` RHS) is exactly what 107-T1.8 specifies and is one of the cleanest pieces of API in Orekit — 100 % portable. The content (DSST mean-element theory) is ~3000 LOC of deeply analytic algebra and is **not** appropriate for `reality` Tier-1/2; it belongs in a separate `orbital/dsst` sub-package much later. SDP4/SGP4 *content* is portable via Vallado's reference C (see #6 below).

**What NOT to copy.** Orekit's `AbsoluteDate` time class with leap-second tables baked in — `reality/orbital` should split timescale conversions into a small dedicated file (107-T1.9), not a god-class. Orekit's frame tree (FactoryManagedFrame) is over-engineered for a math library; a small ICRF/GCRS/ITRF triple via IAU 2006 nutation series is enough for v1.

---

### 3. GMAT — General Mission Analysis Tool (NASA GSFC, Apache-2, C++, ~500k LOC) — `software.nasa.gov/software/GSC-17177-1`

**Headline algorithm.** **PrinceDormand78** (Prince-Dormand 8(7)13) embedded Runge-Kutta — variable step, FSAL, with the GMAT V&V report explicitly identifying it as "the best all-purpose integrator in GMAT" across 250+ regression scenarios. Tolerance `1e-12` on relative error, error estimate from the embedded 7th-order solution, step-size controller with PI-style smoothing.

**Engineering trick.** **Two propagator *types* in one API** — *numerical* (integrator + force model) and *ephemeris* (SPICE BSP / Code-500 file lookup). Same `Propagate` script command, different backends. Means a mission analyst can swap a high-fidelity numerical run for a tabulated DE440 ephemeris read with no change to the rest of the script. Very clean separation.

**Zero-dep portability for `reality`.** **MEDIUM.** PrinceDormand78 (DOP853 in scientific Python parlance) is well-documented in Hairer-Norsett-Wanner and is a 13-stage Butcher tableau — could be added to `chaos/ode.go` as a sibling of RKF45 (~200 LOC, public-domain coefficients). Worth doing when 107-T1.8 numerical Cowell propagator lands. The ephemeris/SPICE side is **NOT** portable as a zero-dep target — JPL DAF/SPK file format is large and binary; a Tier-3 add at best, and probably better as a separate `orbital/spice` package that takes a `[]byte` mmap.

**What NOT to copy.** GMAT's GUI scripting language (interpreted "Mission Sequence" DSL) is product-engineering, not math. Stay clear.

---

### 4. STK / Ansys (commercial, AGI/Ansys C++, closed source) — `help.agi.com/stk`

**Headline algorithm.** **HPOP (High-Precision Orbit Propagator)**: a numerically-integrated propagator with full spherical-harmonic gravity (EGM2008 to degree 360), Jacchia-Roberts / NRLMSISE-00 atmosphere, third-body Sun/Moon/planets, solar radiation pressure with eclipse modeling, and relativistic corrections. The commercial reference standard against which open-source propagators are validated; a public AGI blog post cites HPOP-vs-Orekit cross-validation showing match to ~1 m over 24 h LEO propagation.

**Engineering trick.** **Astrogator targeting/optimization layer above HPOP**: a differential-corrector that wraps the propagator in a constraint-solving outer loop (Brent / Newton on user-defined targets like apsis altitude, plane angle). This is the trick that turns a propagator into a *trajectory designer* — the propagator is the inner kernel, and a generic differential corrector that knows how to numerically take Jacobians via finite differences is the outer kernel.

**Zero-dep portability for `reality`.** Source is closed; the *algorithms* are textbook (Vallado, Curtis), only the polish is proprietary. The differential-corrector pattern is universally applicable: `optim/` package already has bisection/Newton/L-BFGS — a thin trajectory-targeting helper in `orbital/` that calls into `optim` is a natural fit (Tier-3, ~150 LOC). EGM2008 to degree 360 is **NOT** portable (the harmonic coefficient table alone is 360² × 2 × 8 bytes ≈ 2 MB of static data); start with J2/J3/J4/J6 (degree 6, ~150 floats, fits in source).

**What NOT to copy.** Closed-source obviously; but also: STK's "satellite database" baked in. `reality` is a math library, not a data product.

---

### 5. pykep (ESA Advanced Concepts Team, GPL-3, C++ with Python bindings) — `github.com/esa/pykep`

**Headline algorithm.** **Izzo's Lambert solver (2014)** — universal multi-revolution solver via Householder iteration on Battin's `x = (T-T_min)/(T_par-T_min)` parameter, with a *closed-form initial guess* (logarithmic transformation) that converges in 2–3 iterations to machine precision *for any transfer angle, any number of revolutions*. Cited 20–33 % faster than Gooding (1990) and dramatically more robust at multi-rev solutions than Battin (1987). The ESA reference is the open-access ACT-RPR-MAD-2014 PDF.

**Engineering trick.** **Taylor-series automatic-differentiation propagator** (`pykep.taylor_adaptive_propagator`) for low-thrust trajectory optimization. Where everyone else propagates with RK45 and gets tangent-state via finite differences, pykep propagates *the Taylor expansion itself* — analytic derivatives, machine-precision STM, the same code computes the trajectory and its sensitivities. This is the secret behind their GTOC (Global Trajectory Optimization Competition) wins.

**Zero-dep portability for `reality`.** **MIXED.** Izzo's Lambert solver is the gold standard for 107-T1.6 and is straight-line portable: 350 LOC of Householder iteration on a scalar transcendental, no external dependencies (the ACT report contains pseudocode). **Strongly recommend it as the canonical Lambert implementation.** Taylor-AD propagation is **not** portable in the short term — depends on a pure-Go automatic-differentiation core (the `optim/autodiff.go` audit per agent 102 suggests partial AD primitives exist; integrating with a Taylor propagator is a 6-month project, not a Tier-1 effort).

**What NOT to copy.** GPL-3 — `reality` is MIT, so cannot copy pykep source verbatim. Re-implement from the published papers (Izzo 2014; Biscani-Izzo 2021 "Revisiting high-order Taylor methods for astrodynamics").

---

### 6. Brandon Rhodes' python-sgp4 (Python+C, MIT) — `github.com/brandon-rhodes/python-sgp4`

**Headline algorithm.** **Bit-exact port of Vallado's reference SGP4/SDP4** (Vallado et al. 2006, "Revisiting Spacetrack Report #3"). Validated against the August-2010 reference test corpus; the README explicitly states pure-Python and C-accelerated paths agree to within **0.1 mm** (ten-thousandths of a meter) over a 100-year propagation. NORAD-grade.

**Engineering trick.** **Pure-Python fallback that is bit-exact with the C core**, both validated against the same Vallado golden corpus. This is *exactly* the `reality` golden-file model (CLAUDE.md "Golden files are the proof") applied to SGP4. The Vallado test vectors are public-domain and are the de-facto industry reference — a Go port of SGP4 in `orbital/sgp4.go` validated against those same vectors would be MIT-clean and would slot directly into `reality`'s test infrastructure.

**Zero-dep portability for `reality`.** **VERY HIGH.** This is the single highest-leverage SOTA artifact for `reality/orbital`. Vallado's reference C code (~1500 LOC) is in the public domain (his website + Spacetrack-Report-3 distribution); a Go translation is mechanical, the test vectors carry over verbatim, and the result *is* the reference standard for satellite-tracking accuracy. Recommend it as a single self-contained file `orbital/sgp4.go` per 107-T2.x — no dependency on perturbation framework, no dependency on time-scale conversions beyond JD/MJD (which 107-T1.9 will deliver).

**What NOT to copy.** The OMM (Orbital Mean elements Message) XML/CCSDS parsing — that's data-format engineering, not math. TLE-string parsing should be a separate file (or live in a future `orbital/tle.go`), validated independently. The MIT license of brandon-rhodes/python-sgp4 *does* permit verbatim copying of the C wrapper, but the Go translation is mechanical and `reality` should be a clean reimplementation per CLAUDE.md rule #6.

---

### 7. Skyfield (Brandon Rhodes, MIT) — `rhodesmill.org/skyfield`

**Headline algorithm.** **JPL DE-series Chebyshev-polynomial ephemeris evaluation** via memory-mapped SPK kernels. DE440 covers JD 2287184.50 – 2688976.50 (1549–2650), 14 segments × ~5000 polynomials each, evaluated at the requested instant by Clenshaw's recurrence on the 13-degree Chebyshev coefficients. ~10 μs per body per epoch on cold cache, ~1 μs warm. The de-facto Python ephemeris.

**Engineering trick.** **Memory-mapped DAF format read on demand** — `jplephem` does *not* load the 128 MB DE440 file into RAM; it `mmap`s the file, parses the directory once, then lazily reads only the polynomial coefficients touched by the current query. The OS page-cache handles working-set residency. A LEO orbit-determination loop hitting only the Earth-Moon segments touches ~1 % of the file.

**Zero-dep portability for `reality`.** **LOW for content, HIGH for pattern.** The DE binary format is a SPICE-internal data product; parsing it is a binary-format problem, not a math problem, and `reality`'s "no allocations in hot paths" + zero-dep rules suggest the right home is a separate package or a separate file with `// +build ephemeris` build tag. The *evaluation* (Chebyshev/Clenshaw) is pure math and is portable in 50 LOC — recommend `orbital/chebyshev.go` (or generalize and put it in `calculus/`). The *file parsing* is best deferred.

**What NOT to copy.** Skyfield's heavy reliance on `numpy` arrays as the data interface — the Go-idiomatic equivalent is `[]float64` slices and is much simpler. Skyfield's Pythonic time-class hierarchy is overkill for a math library.

---

### 8. AstroPy (Python, BSD-3, ~500k LOC) — `docs.astropy.org`

**Headline algorithm.** **IAU 2006/2000A precession-nutation series** via the ERFA C library (the open-source release of IAU's SOFA). 1365 nutation terms (luni-solar) + 687 planetary terms; transforms ICRS↔GCRS↔CIRS↔ITRS to <1 mas accuracy. Implements every IAU resolution from 1976 onwards.

**Engineering trick.** **Frame-graph transformation registry**: `SkyCoord` looks up the shortest path through a directed graph of registered frame-pair transformations and composes them on the fly. New frames register a single edge; `SkyCoord` automatically routes any-to-any. Means user code never writes "first to GCRS, then to CIRS, then to ITRS" — just `coord.transform_to('itrs')` and the graph handles it.

**Zero-dep portability for `reality`.** **MEDIUM.** The frame-graph pattern is portable as a small map-of-functions in Go (~50 LOC). The IAU 2006 nutation series is a published table (1365 + 687 floats × 14 columns ≈ 230 KB of constants — borderline for embed in source; comparable in size to existing `signal/` window-function tables). For a *first-pass* `orbital`, a truncated IAU 2006 series (the largest 100 terms gives ~1 arcsecond, fine for textbook satellite work) is ~3 KB and a clean fit. ERFA is BSD-3, license-compatible with `reality`'s MIT — the *algorithms* are public, only the convenience of having ERFA already-tested matters; reality's golden-file model can validate against ERFA outputs offline.

**What NOT to copy.** The full SOFA / ERFA scope (planetary nutation, polar motion, Earth orientation, leap-second tables) is overkill for v1. Start with: ICRS=GCRS (low-precision), GCRS↔ITRS via simple Earth-rotation-angle (ERA) + truncated nutation, and document precision as ~1 arcsec.

---

### 9. ML-augmented propagation (research frontier 2024–2026)

**Headline algorithm.** **Hybrid SGP4 + ML residual learners**: SGP4 produces a baseline state, an LSTM/iTransformer takes (past N states, atmospheric proxy variables) and predicts a *correction vector*. Reported in Curzi et al. 2024 (LEO ML orbit prediction) and 2026 *Astrodynamics* (real-time onboard propagator). 75 % along-track-error reduction over plain SGP4 for LEO/GEO with sub-millisecond inference (iTransformer + FNN at 0.018 s per state).

**Engineering trick.** **Dataset construction from the GNSS log + SGP4 residual.** The "label" for ML training is the difference between a precise GNSS-derived state and the SGP4 prediction at the same epoch. This converts a regression problem into a *correction* problem and makes the network's job tractable (residuals are small and locally smooth in time).

**Zero-dep portability for `reality`.** **OUT OF SCOPE.** ML models require runtime infrastructure (tensors, BLAS, model serialization) that are explicitly outside `reality`'s zero-dep mandate. The right place for ML-augmented propagation is in `aicore` (which imports `reality`), not in `reality` itself. The *baseline* SGP4 implementation is the contribution from `reality`; the ML correction layer belongs upstream.

**What NOT to copy.** Anything network-/tensor-library-dependent. `reality` should ship a pristine SGP4 that ML researchers can use as ground truth, no more.

---

## Cross-library portability matrix (summary)

| Library             | Headline algo                       | Engineering trick                        | Port to `reality`? | Where it lands (per 107)        |
|---------------------|-------------------------------------|------------------------------------------|--------------------|---------------------------------|
| poliastro           | Markley 1995 Kepler solver          | Numba JIT inner loops                    | **HIGH** (algo)    | replace `TrueAnomalyFromMean`   |
| Orekit              | Unified propagator interface        | Force-model composition                  | **HIGH** (pattern) | T1.8 Cowell propagator API      |
| GMAT                | PrinceDormand78 (DOP853)            | Numerical-vs-ephemeris dual backend      | **MEDIUM**         | `chaos/ode.go` add DOP853       |
| STK / Ansys         | HPOP + EGM2008 + Astrogator         | Differential corrector outer loop        | LOW (closed)       | Tier-3 trajectory targeter     |
| pykep               | **Izzo Lambert** + Taylor AD prop.  | Householder on Battin x-parameter        | **HIGH** (Lambert) | T1.6 canonical Lambert solver   |
| python-sgp4         | Vallado 2006 reference SGP4/SDP4    | Bit-exact pure-Python ↔ C cross-validate | **VERY HIGH**      | T2.x `orbital/sgp4.go`          |
| Skyfield            | DE440 Chebyshev SPK lookup          | Lazy mmap'd binary ephemeris             | LOW (file format)  | Tier-3 separate sub-package     |
| Astropy             | IAU 2006/2000A nutation             | Frame-graph transformation registry      | **MEDIUM**         | T2 frames file (truncated)      |
| ML-augmented (2026) | SGP4 + LSTM residual                | GNSS-residual dataset construction       | OUT OF SCOPE       | belongs in `aicore`             |

---

## Specific recommendations for `reality/orbital` (delta to 107)

107 lists *what* to add (Tier 1–3 backlog). This review settles *which implementation* of each:

1. **Kepler solver**: replace Newton in `TrueAnomalyFromMean` with **Markley 1995** (poliastro's choice). Closes 106-F1 *and* 106-F2 in one stroke; the published polynomial coefficients are public-domain. ~80 LOC.

2. **Lambert solver (107-T1.6)**: implement **Izzo 2014** (pykep's choice). The ESA-ACT 2014 paper has full pseudocode; license is GPL-3 in pykep but the algorithm is published in *Celestial Mechanics & Dynamical Astronomy* and is freely re-implementable. **Strongly preferred over Battin / Gooding** for robustness + speed.

3. **Cowell propagator (107-T1.8)**: copy **Orekit's force-model composition pattern** (slice of `Perturbation` functions reduced into an RK45 RHS). Use `chaos/ode.go` RKF45 first; add DOP853 (PrinceDormand78) later when GMAT-grade fidelity is needed.

4. **SGP4 (107-T2.x)**: port **Vallado 2006** reference C (public domain) into a single self-contained `orbital/sgp4.go`, validated against the same August-2010 corpus that brandon-rhodes/python-sgp4 uses. ~1500 LOC, zero dep, NORAD-grade.

5. **Frames (107-T2.x)**: implement **Astropy's frame-graph pattern** with truncated IAU 2006 nutation (top 100 terms, ~3 KB of constants in source, ~1 arcsec precision). Document the precision floor explicitly per CLAUDE.md "precision documented, not assumed".

6. **DSST, EGM2008-360, SPK ephemeris file reading, ML-augmented propagation**: explicitly **defer or out-of-scope**. These would each pull `reality/orbital` over the size threshold where it stops being a math primitive and becomes a domain product.

---

## What this review does *not* cover (boundaries to siblings)

- Per-line audit of the 8 existing functions → **106-orbital-numerics**.
- Tier-1/2/3 backlog of *what* features are missing → **107-orbital-missing**.
- ODE-solver-specific (RK4 / RKF45 / DOP853) audit → handled in `chaos/ode.go` reviews (agent ~26 area).
- Time-scale conversion (TT/TAI/UTC) audit → belongs in a future `time/` review or 107-T1.9 follow-up.
- IAU constants (G·M_⊕, J2, R_⊕) → `constants/` package audit.
- Auto-differentiation underpinning a Taylor propagator → `optim/autodiff.go` (agent 102).

---

## Bottom line

`reality/orbital` is two well-chosen ports away from being a credible orbital-mechanics library at MIT-clean / zero-dep. **(a)** Markley 1995 Kepler solver replacing Newton, **(b)** Izzo 2014 Lambert solver as the canonical transfer-design primitive. Add **(c)** Vallado SGP4 next sprint and `reality/orbital` becomes the only Go library with NORAD-grade satellite tracking. The Orekit force-model-composition pattern + `chaos/ode.go` RKF45 gets perturbation-aware Cowell propagation in ~120 LOC. Astropy frame-graph + truncated IAU 2006 nutation gets ICRS/GCRS/ITRS at ~1 arcsec in ~250 LOC. Everything else in the SOTA landscape (DSST, EGM2008-360, SPK ephemeris, ML residuals, Astrogator targeting, GMAT mission scripting, STK GUI) is correctly *not* in `reality`'s scope and should be left to consumer applications upstream.

Three high-leverage 80/20 ports — Markley + Izzo + Vallado-SGP4 — get `reality/orbital` to the *same algorithmic core* as poliastro, Orekit, and pykep, and a long way past STK at zero closed-source liability.

---

*Sources*

- poliastro: <https://docs.poliastro.space/en/stable/> ; <https://github.com/poliastro/poliastro> ; <https://www.poliastro.space/blog/2019/07/16/2019-07-16-new-propagators/>
- Orekit: <https://www.orekit.org/site-orekit-latest/architecture/propagation.html> ; <https://www.orekit.org/site-orekit-latest/architecture/forces.html>
- GMAT: <https://software.nasa.gov/software/GSC-17177-1> ; <https://documentation.help/gmat/Propagator.html> ; <https://ntrs.nasa.gov/api/citations/20140017798/downloads/20140017798.pdf>
- STK/HPOP: <https://help.agi.com/stk/12.1.0/content/hpop/hpop.htm> ; <https://help.agi.com/stk/Content/stk/vehSat_orbitProp_msgp4.htm>
- pykep: <https://github.com/esa/pykep/> ; <https://www.esa.int/gsp/ACT/doc/MAD/pub/ACT-RPR-MAD-2014-RevisitingLambertProblem.pdf>
- python-sgp4: <https://github.com/brandon-rhodes/python-sgp4> ; <https://pypi.org/project/sgp4/>
- Skyfield: <https://rhodesmill.org/skyfield/planets.html> ; <https://pypi.org/project/jplephem/>
- Astropy: <https://docs.astropy.org/en/stable/coordinates/index.html> ; <https://docs.astropy.org/en/stable/coordinates/transforming.html>
- ML-augmented orbit prediction: <https://arxiv.org/html/2407.11026v1> ; <https://link.springer.com/article/10.1007/s42064-025-0264-6> ; <https://arxiv.org/html/2207.08993v4>
- Lambert-solver comparison: <https://arc.aiaa.org/doi/10.2514/1.G006089> ; <https://arxiv.org/abs/1403.2705>
