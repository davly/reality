# 368 — research-iau-frames (ICRF3 / IAU 2006A / DE441 / SOFA audit)

## Headline
Reality's `orbital` package is frame-agnostic Kepler-only — it has zero ICRF/ICRS, precession-nutation, time-scale, or ephemeris infrastructure; bundling DE441 (~100 MB binary SPK) violates the zero-dependency / no-allocations creed, so reality should ship a thin Chebyshev-evaluator + frame-rotation framework and let consumers (aicore, Pistachio) load DE/INPOP kernels themselves.

## Reality audit (current state)

`orbital/orbital.go` (267 LOC, 7 functions: `KeplerOrbit`, `OrbitalPeriod`, `OrbitalVelocity`, `HohmannTransfer`, `EscapeVelocity`, `HillSphere`, `SynodicPeriod`, `TrueAnomalyFromMean`) is purely two-body, frame-blind:

- No ICRF / ICRS / GCRS / BCRS / ITRS / CIRS / TIRS — no concept of a frame at all. `KeplerOrbit` returns `(x,y,z)` in an unspecified inertial system (effectively the orbit's own perifocal-rotated frame).
- No precession, nutation, polar motion, sidereal time, or CIO/CIP. Repo-wide grep for `ICRF|ICRS|DE44|SOFA|Precess|Nutat|Sidereal|Ephemeris` finds matches only inside `reviews/overnight-400/` (planning text), never in production Go code.
- No time scales: no TT, TAI, UT1, UTC, TDB, TCB, leap-second handling. Time enters only as anomalies in radians.
- No DE/INPOP coefficients, no Chebyshev evaluator, no SPK reader.
- `constants` package has G, c, etc. but no IAU 2015 nominal solar/planetary values (`GM_sun`, `R_earth_eq`, etc.) tagged as nominal IAU constants.

This is fine for the v0.10 scope (Kepler conics, Hohmann, escape) — but anything beyond instantaneous two-body geometry needs frames + ephemerides.

## Survey

### 1. ICRF3 (IAU 2018, effective 2019-01-01)
Third realization of the ICRS, adopted at the IAU GA in Vienna 2018, effective 2019-01-01. Built from ~40 yr of VLBI on 4536 extragalactic radio sources (S/X 8.4/2.3 GHz), augmented with K-band (24 GHz, 824 sources) and X/Ka (32/8.4 GHz, 678 sources). 303 "defining sources" fix the axes. Median positional uncertainty ~0.1 mas RA / 0.2 mas Dec; noise floor 0.03 mas. First ICRF realization to model galactocentric acceleration of the solar-system barycenter (~5.8 μas/yr). Successor to ICRF2 (2009). The ICRF *realizes* the ICRS (the system); reality should reference "ICRS" (the system) when documenting frame conventions and "ICRF3" only when sub-mas accuracy is claimed (which it never will be at float64).

### 2. IAU 2006 precession (P03 / Capitaine et al.)
Adopted Resolution B1, IAU GA Prague 2006. The Capitaine, Wallace, Chapront P03 polynomial replaces the IAU 2000 precession part. Combined with IAU 2000A nutation (1365-term lunisolar + 687-term planetary series), this forms the IAU 2006/2000A precession-nutation, mandatory in IERS Conventions 2010 from 2009-01-01. IAU 2000A and IAU 2006 P03 are not fully dynamically consistent — SOFA applies "P03-adjusted IAU 2000A nutation" with μas-level corrections (Wallace & Capitaine 2006, A&A 459, 981). IAU 2000B is a 77-term truncation (~1 mas accuracy, ~30× faster) for low-precision use. Reality reproducing 2000A from scratch means embedding the full 6 MB Tab.5.3a/b coefficient tables.

### 3. JPL DE441 (Park et al. 2021)
JPL Solar System Dynamics group, created June 2020, posted Feb 2022. Spans **−13,200 to +17,191** (~30 ka). Stored as Chebyshev polynomials over typically 32-day Mercury chunks down to 4-day Moon chunks, in TDB barycentric. DE441 omits lunar core-mantle damping (which would diverge backward in time), so it is *less accurate* than DE440 over modern era but stable archaeoastronomy/paleoeclipse-grade. SPK binary distributed by JPL (~3 GB for full DE441; trimmed `de441_part-1.bsp` ~190 MB). Within the DE440 epoch (1550–2650), DE441 agrees with DE440 to <1 m planetary positions.

### 4. JPL DE440 (Park et al. 2021, AJ 161:105)
Same integration as DE441 but **with** lunar core-mantle damping, more accurate; spans 1550–2650 only. Binary SPK ~120 MB; trimmed `de440s.bsp` ~32 MB (1849–2150, 14 segments). Default planetary ephemeris in NASA HORIZONS as of 2022. Improved Mars by 1–2 orders of magnitude over DE430 thanks to MRO/MEX ranging; Jupiter via Juno; outer planets via stellar occultations.

### 5. JPL DE430 / DE430t / DE432 (Folkner et al. 2014, IPN PR 42-196)
Workhorse 2013–2022, still ubiquitous in deep-space ops. 1550–2650, ~115 MB SPK. DE430t added TT–TDB at integration time (no need for external Fairhead-Bretagnon analytical correction). DE432 is a tweaked version with improved Pluto. Reality consumers running pre-2022 codebases will still see DE430.

### 6. INPOP21a (IMCCE / Paris Observatory)
INPOP = Intégration Numérique Planétaire de l'Observatoire de Paris. INPOP21a (2021, Fienga et al.) is the rival ephemeris fit independently from DE: Sun + 8 planets + Pluto + Moon + lunar libration, ICRF-aligned, distributed as binary or text Chebyshev files (kilometers, TDB). Slightly different fit philosophy (more LLR weight, different asteroid mass set). INPOP19a was used by ESA for BepiColombo and JUICE. Reality should treat DE and INPOP as interchangeable plug-in coefficient sources. As of 2025-01-01 the IMCCE merged with SyRTE into LTE (Laboratoire Temps Espace).

### 7. SPK / DAF / BSP file format (NAIF)
SPICE's SPK kernel = Spacecraft & Planet Kernel, a binary Double Precision Array File (DAF). First 8 bytes: magic `DAF/SPK ` (or `NAIF/DAF`). File holds 1+ segments; each segment provides position+velocity of one target relative to one center over a time interval, with a segment-type code (Type 2 = Chebyshev position-only, Type 3 = position+velocity, Type 21 = extended modified difference array). Reality must NOT vendor SPICE (huge, fortran-flavored C, restrictive disclaimer-only license), but a ~300-LOC pure-Go DAF reader + Type 2/3 evaluator is achievable and has precedent (`pkg.go.dev/github.com/mshafiee/jpleph`, Project Pluto's `jpl_eph`).

### 8. SOFA (IAU Standards Of Fundamental Astronomy)
~190 ANSI C routines maintained by IAU SOFA Board (Wallace et al.). Latest release 2023-10-11 (no 2024 release as of search date). Implements: time scales (TAI/UTC/UT1/TT/TDB/TCB conversions including leap seconds), Earth rotation (GMST, ERA), precession-nutation (`pn00a`, `pn06a`, `c2t06a`), polar motion, frame transforms (BCRS↔ICRS↔GCRS↔CIRS↔TIRS↔ITRS), astrometry (proper motion, parallax, aberration, light deflection). **License caveat**: SOFA original is *not* BSD — its terms forbid redistribution under a different name in modified form, requiring renaming; this is *not* MIT-compatible by default. ERFA (Essential Routines for Fundamental Astronomy, github.com/liberfa/erfa) is a 3-clause-BSD line-by-line clone maintained for astropy and IS reality-license-compatible. Slot 356's claim of "BSD-3 SOFA" was loose — they meant ERFA.

### 9. IERS Conventions 2010 (Tech Note 36, Petit & Luzum)
Authoritative spec for ITRS↔ICRS transformations as of 2026; still current. Mandates IAU 2006/2000A precession-nutation, CIO-based transformation (CIRS→TIRS via Earth Rotation Angle θ rather than GAST), polar-motion model with diurnal/sub-diurnal ocean-tide corrections, EOP ingestion from IERS C04. Bulletin A (rapid) and Bulletin B (final) supply UT1−UTC, polar coords, dX/dY corrections to IAU 2006/2000A.

### 10. IAU 2006 Resolution B3 — TT, TDB, TCB scales
Defines TT–TAI = 32.184 s exactly (TAI adopted 1958); TCB defined relative to BCRS, TT defined relative to GCRS (rate ratio 1−L_G with L_G = 6.969290134×10⁻¹⁰); TDB redefined as a *linear* function of TCB (no longer the older Damour-Soffel-Xu definition): `TDB = TCB − L_B·(JD_TCB − T₀)·86400 + TDB₀` with L_B = 1.550519768×10⁻⁸, T₀ = 2443144.5003725, TDB₀ = −6.55×10⁻⁵ s. Reality currently has none of this; if it ever exposes time-scale conversions, it must follow these defining constants exactly.

## Reality positioning

### Should reality embed DE441? **NO.**

| Option | Size | Build impact | Verdict |
|---|---|---|---|
| Vendor full DE441 SPK | ~3 GB | repo bloat ×30 | NO |
| Vendor DE441 trimmed (1900–2100) | ~30 MB | binary ×3, `go install` slow | NO |
| Vendor DE440s (1849–2150) | ~32 MB | same | NO |
| Vendor DE441 polynomial coeffs as `[]float64` literals | ~25 MB Go source | unparseable by `go vet`, hot OOM | NO |
| Vendor *thin* Chebyshev evaluator + SPK reader, NO data | <50 KB code | clean | **YES** |
| Embed only IAU 2015 nominal constants (GM_sun, GM_earth, AU, R_earth_eq…) | ~200 bytes | trivial | **YES** |

**Reasons to refuse the bundle:**
1. **Zero-dep / lean creed (CLAUDE.md rule 2 & 3):** A 30 MB embedded asset would dwarf the entire codebase. Pistachio's 60 FPS hot-path needs constants, not 30 MB of `embed.FS`.
2. **Coeffs are *data*, not *math.*** Reality's mandate is "universal truth encoded in code" — the Chebyshev coefficients are the *result of* a JPL least-squares fit to LLR/MRO/Cassini/Juno tracking data. They are observational, not first-principles. Embedding them violates rule 6 ("reimplement from first principles").
3. **Versioning hell:** DE430/440/441/INPOP21a all valid; freezing one in source tree picks a fight with downstream consumers who already ship a different one.
4. **Chebyshev evaluator IS first-principles math** — fits reality's charter perfectly, and would be ~80 LOC + ~30 golden vectors.

### Recommendation (concrete, deferable to v0.12+)

**Do now (v0.11) — small, high-value:**
- Add `orbital.IAUFrame` enum stub (string constants `"ICRS"`, `"GCRS"`, `"BCRS"`, `"ITRS"`) and require new functions to *document* which frame their inputs/outputs live in. Backfill `KeplerOrbit` doc: "perifocal rotated by 3-1-3, treat as ICRS-parallel inertial".
- In `constants`, add `IAU2015Nominal` block with `GM_Sun = 1.32712442099e20`, `GM_Earth = 3.986004415e14`, `AU = 1.49597870700e11` (exact since 2012), `R_Earth_eq = 6.3781e6`, `R_Sun_eq = 6.957e8` — ~10 numbers, ~300 bytes, sourced to IAU 2015 Resolution B3 and CODATA 2018.
- Add `orbital.TT_minus_TAI = 32.184` (exact) and `LB`, `LG`, `TDB0`, `T0` constants from IAU 2006 Res B3 — pure numbers, no logic.

**Do soon (v0.12) — medium:**
- New file `orbital/chebyshev.go`: `EvalChebyshev1D(coeffs []float64, t01 float64) float64` and `EvalChebyshev3D(xCoeffs, yCoeffs, zCoeffs []float64, t01 float64) (x,y,z float64)` — pure Clenshaw recurrence, ~40 LOC, golden-fileable against scipy `numpy.polynomial.chebyshev.chebval`. This is the *primitive that makes consuming DE441 trivial*, without shipping coefficients.
- New file `orbital/timescales.go`: `TT_to_TDB(jdTT float64) float64` via Fairhead-Bretagnon series (10-term, ~2 ms accuracy) plus the IAU 2006 linear `TDB↔TCB`. ERFA-compatible signature.
- New file `orbital/precession.go`: `PrecessionAngles_IAU2006(jdTT) (zeta, z, theta)` — the P03 polynomial is just five 5th-degree polynomials in T; ~30 LOC, no tables. (Nutation is a different story — the 1365×7 IAU2000A series IS a 60 KB table; ship IAU 2000B 77-term truncation instead, ~5 KB, ~1 mas; defer 2000A to v1.)

**Don't do (ever):**
- Vendor any SPK file in-tree.
- Wrap SPICE or SOFA C source — both violate zero-dep (and SOFA's redistribution clause is borderline).
- Implement an SPK *reader* in `reality` itself. That belongs one layer up (aicore or a sibling `astro` repo), where I/O is acceptable. Reality should be `reader -> [coeffs]float64 -> reality.EvalChebyshev3D -> (x,y,z)`.

### Cross-links

- **Slot 342 (Kepler)** — `KeplerOrbit` and `TrueAnomalyFromMean` need a frame doc-string upgrade once IAUFrame stub lands; their math is unaffected.
- **Slot 343 (Lambert / Izzo)** — Lambert solver is frame-blind (works on any inertial frame); same doc-only update.
- **Slot 344 (perturbations)** — recommended N=8 EGM gravity model needs ITRS coords, so the proposed precession+ERA scaffolding here directly unblocks slot 344's harmonic-gravity perturbation. Order matters: do this slot's IAU 2006 P03 *before* slot 344's harmonic Earth gravity, otherwise the latter is meaningless.
- **Slot 356 (license audit)** — confirm "BSD-3 SOFA" wording is corrected to **ERFA** (BSD-3); SOFA itself is *not* MIT-compatible verbatim.
- **Slot 367 (CODATA)** — IAU 2015 nominal constants live next to CODATA 2018 values in `constants`; both should carry `Source` and `Year` metadata fields.

## Sources

- [IERS — The 3rd realization of the ICRF (ICRF3)](https://www.iers.org/IERS/EN/DataProducts/ICRF/ICRF3/icrf3.html)
- [Charlot et al. 2020, ICRF3 by VLBI, A&A 644 A159](https://www.aanda.org/articles/aa/full_html/2020/12/aa38368-20/aa38368-20.html)
- [Wallace & Capitaine 2006 — Precession-nutation procedures consistent with IAU 2006, A&A 459 981](https://www.aanda.org/articles/aa/abs/2006/45/aa5897-06/aa5897-06.html)
- [JPL SSD — DE440/DE441 documentation page](https://ssd.jpl.nasa.gov/doc/de440_de441.html)
- [Park et al. 2021 — The JPL Planetary and Lunar Ephemerides DE440/DE441, AJ 161:105 (PDF)](https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf)
- [JPL Planetary and Lunar Ephemerides export](https://ssd.jpl.nasa.gov/planets/eph_export.html)
- [INPOP21a planetary ephemerides (IMCCE PDF)](https://www.imcce.fr/content/medias/recherche/equipes/asd/inpop/inpop21a.pdf)
- [IMCCE INPOP page](https://www.imcce.fr/inpop/)
- [NAIF SPK Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html)
- [IAU SOFA — homepage](https://www.iausofa.org/)
- [ERFA (BSD-3 reimplementation of SOFA)](https://github.com/liberfa/erfa)
- [Astropy wiki — SOFA license for the community](https://github.com/astropy/astropy/wiki/SOFA-license-for-the-community)
- [IAU 2006 Resolution B3 — TT/TDB/TCB definitions (PDF)](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/resolutions/IAU/IAU2006_Resolution_B3.pdf)
- [IERS Conventions 2010 (Tech Note 36, Petit & Luzum, PDF)](https://apps.dtic.mil/sti/tr/pdf/ADA535671.pdf)
- [Wikipedia — JPL Development Ephemeris](https://en.wikipedia.org/wiki/Jet_Propulsion_Laboratory_Development_Ephemeris)
- [Wikipedia — ICRS and its realizations](https://en.wikipedia.org/wiki/International_Celestial_Reference_System_and_its_realizations)
