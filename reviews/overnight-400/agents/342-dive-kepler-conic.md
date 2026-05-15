# 342 — dive-kepler-conic (Newton/Halley/Markley/universal-variable Kepler audit)

## Headline
`orbital.TrueAnomalyFromMean` is a textbook Newton-Raphson on `M = E − e·sin E` with seed `E₀ = M`, `tol = 1e-15`, and silent non-convergence; it has no Halley/Markley starter, no Stumpff/universal-variable path, and no hyperbolic (`e ≥ 1`) branch. The cheapest day-1 PR is a Markley 1995 cubic starter (~80 LOC) gated by `e > 0.6`, which collapses the iteration count from O(20–50) to ≤ 2 across the entire elliptic regime and removes silent failure.

## Findings (existing audit)

### Existing Kepler primitive (single function, single conic type)
- `orbital/orbital.go:232` — `TrueAnomalyFromMean(M, e, maxIter int) float64`. Newton-Raphson with seed `E := M` (line 240), update `dE = (E − e·sinE − M) / (1 − e·cosE)` (line 246), termination `|dE| < 1e-15` (line 248), then half-angle `ν = 2·atan2(√(1+e)·sin(E/2), √(1−e)·cos(E/2))` (line 255). Conversion to true anomaly is correct and quadrant-safe.
- **No Halley** (cubic update with `f''`), **no Markley** (Padé-cubic seed), **no universal variable**, **no hyperbolic** branch anywhere in `orbital/`.
- Loop body lacks any flag for non-convergence — if `maxIter` is exhausted the last `E` is returned silently. Confirmed by grep: only `TrueAnomalyFromMean`, `KeplerOrbit`, `OrbitalPeriod`, `OrbitalVelocity`, `HohmannTransfer`, `EscapeVelocity`, `HillSphere`, `SynodicPeriod` exist (8 functions, all in one 267-LOC file).

### Numerical defects in the current solver
- **F1 (CRIT, already raised by 106-orbital-numerics:15):** denominator `1 − e·cosE` → 0 as `e → 1` and `E → 0`; at e = 0.99, M = 0.01, the seed E = M makes the Newton step blow up by ~1/(1−0.99) = 100×, oscillation possible. The standard mitigation is Markley's seed (cubic Padé approximation; see Sources) which lands inside the basin of quadratic convergence in *one* shot.
- **F2 (HIGH, 106:34):** silent non-convergence — no error/`bool ok` return, no `nan` sentinel, no comparison `iter == maxIter` that would warn. `TestTrueAnomalyFromMean_ConvergesForHighE` (`orbital_test.go:311`) only checks `[0, 2π)` and `!IsNaN`, NOT the residual `|E − e·sinE − M|`, so a stuck-at-seed result passes the test.
- **F3 (MED):** input validation absent — `e ≥ 1` accepted silently; `M = NaN`/`Inf` propagates through `math.Mod` to `NaN` quietly. CLAUDE.md rule 5 requires "valid input range, precision, failure modes" to be documented; godoc says e in [0,1) but code does not enforce.
- **F4 (LOW):** seed `E = M` is the worst possible choice for `e > 0.7`. Better seeds: Danby's `E₀ = M + 0.85·e·sgn(sin M)`; Conway's `E₀ = π` for `M = π`; **Markley's cubic root** for the entire range. Markley empirics (per A&A 2022 study cited below): 1 iter suffices in 59% of cases, ≤ 2 iters in 99.996%.
- **F5 (no-op for low e):** for `e = 0` Newton converges in one step (`E = M` is exact), but the loop still runs `maxIter` body evaluations (no fast-exit on `e == 0`). Not a correctness issue; minor perf.

### Coverage gaps vs. astrodynamics canon
- **Hyperbolic Kepler `M = e·sinh F − F`** absent. Required for any flyby, escape, or interplanetary trajectory (eccentricity > 1). Conway-Prussing 1986 starter `F₀ = ln(2|M|/e + 1.8)` gives ~3-iter Newton. Already flagged in 107-orbital-missing:37.
- **Parabolic Kepler (Barker's equation)** `M_p = D + D³/3` absent. Closed-form cubic root (Cardano). Not on master plan but trivially completes the conic family.
- **Universal-variable Kepler** (Battin §4.3, Vallado §2.2 alg. 8): one solver handles `e < 1`, `e = 1`, `e > 1` by replacing `(t − t₀)` with the universal anomaly χ and Stumpff `C(z) = c₂(z), S(z) = c₃(z)` where `z = α·χ²`. Already specced as 107-T1.1 (~250 LOC). Removes the e<1 restriction *by construction*.
- **Stumpff functions** `c₂(z), c₃(z)` (Stumpff 1947): not present. Series fallback for `|z| < 1` is required to dodge catastrophic cancellation when `cos√z ≈ 1`. 110-orbital-perf:131 already proposed Chebyshev-table evaluation (degree 12, ~10⁻¹⁴ on z ∈ [−50, +50]) — that PR is unblocked once Stumpff lands.
- **Halley iteration** (3rd-order, uses `f''(E) = e·sinE`): trivial extension of existing Newton — `E_{n+1} = E_n − (2·f·f') / (2·(f')² − f·f'')`. Convergence radius larger than Newton's; competitive with Markley for moderate-e but lacks Markley's superior starter.

### Test coverage gaps
- `orbital/testdata/orbital/true_anomaly.json`: golden vectors exist for `TrueAnomalyFromMean` (line 2 grep hit), but the high-e regression (`e ∈ {0.95, 0.99, 0.999}`) is the precise band where Newton + seed `E=M` is most likely to wander. Need `e ∈ {0.99, 0.999}` × `M ∈ {1e-6, π/2, π}` IEEE-vector entries plus an explicit residual assertion.
- No mutual-cross-validation between Newton, Halley, Markley — the R-MUTUAL-CROSS-VALIDATION 3/3 pin pattern (cf. 365368a audio onset commit) is absent for orbital. Three independent paths exist as named methods in the literature; only one is implemented.

### Cross-package consumers
- `aicore` and any 60-FPS visualizer (Pistachio per CLAUDE.md) import `reality/orbital`. Per 110-orbital-perf:147 the universal-variable + Stumpff path is the hot loop for orbital-particle propagation; Markley alone (without Stumpff) suffices for the elliptic-only case at 60 FPS.
- 187-* (orbital propagation) cited in this dive's task as a downstream slot; current `TrueAnomalyFromMean` is the only Kepler primitive available to that work, so its hardening blocks downstream propagators.
- GPS / GNSS broadcast ephemerides use eccentricity `e ≈ 0.001–0.02` (LEO/MEO) — well inside Newton's safe basin; current solver is fine *for that* but not for cometary, Molniya (`e ≈ 0.74`), or Pluto-class (`e ≈ 0.25`) cases at high-precision residual targets.

## Concrete recommendations

### T0 — Harden existing `TrueAnomalyFromMean` (DAY-1 PR, ~50 LOC delta)
1. Add Kepler-equation residual assertion to `TestTrueAnomalyFromMean_ConvergesForHighE` (`orbital_test.go:311`): require `|E − e·sinE − M| < 1e-12` post-call.
2. Add IEEE input validation: `if e < 0 || e >= 1 { return math.NaN() }`; `if math.IsNaN(M) || math.IsInf(M, 0) { return math.NaN() }`. Mirrors signal-package convention.
3. Add `iter == maxIter` warning path. Two design choices: (a) signature-stable: extend godoc to state "if maxIter exhausted, returned ν has unspecified residual"; (b) signature-breaking: introduce `TrueAnomalyFromMeanSafe(M, e, maxIter) (nu float64, ok bool)` alongside. Recommend (b) — preserves existing API + adds correctness lane. **(~50 LOC + 20 test LOC.)**
4. Switch terminator from `|dE| < 1e-15` to `|f(E)| < 1e-13` (residual-based, not step-based). The current `1e-15` is below float64 epsilon for E ~ π and triggers spurious last-step thrashing.

### T1 — Markley 1995 starter + 2-iter modified Newton (~80 LOC) [CHEAPEST HIGH-VALUE PR]
1. New unexported helper `keplerStartMarkley(M, e float64) float64` returning the cubic-root seed E₀ (Markley 1995 eq. 19–20). Single `math.Cbrt` + one `math.Sqrt`.
2. Refactor `TrueAnomalyFromMean` to: branch `e ≤ 0.6` → existing Newton seed `E = M`; `e > 0.6` → Markley seed + 1 iteration of 5th-order modified-Newton (Markley §4 — uses f, f', f'', f''' available analytically). Empirical convergence: ≤ 2 iters in 99.996% of cases.
3. Same signature, same golden file vectors: a regression-pin opportunity.
4. Reference: F. L. Markley, "Kepler equation solver", *Celestial Mechanics & Dynamical Astronomy* 63, 101–111 (1995); reproduced in poliastro `core/iod/markley.py` as a reference Python implementation. **(~80 LOC, immediate 5-10× iteration reduction at high e.)**

### T2 — Universal-variable Kepler with Stumpff `c₂, c₃` (~200 LOC)
1. Add `func StumpffC(z float64) float64`, `func StumpffS(z float64) float64` (Stumpff 1947). Closed form for |z| ≥ ε; Maclaurin series (8 terms) for |z| < 1 to avoid `(1 − cos√z) / z` cancellation. Reference: Battin §4.5; Vallado §2.2.
2. Add `func KeplerUniversal(r0Vec, v0Vec [3]float64, dt, mu float64) (rVec, vVec [3]float64)`. Newton-Raphson on χ; `z = α·χ²` with `α = 2/r₀ − v₀²/μ` (sign of α distinguishes elliptic / parabolic / hyperbolic — branchless). Lagrange f, g, ḟ, ġ coefficients then map to `(r, v)`.
3. Removes `e < 1` restriction by construction; closes 107-T1.1 and 106-F1 simultaneously. **(~200 LOC core + ~50 LOC Stumpff + golden vectors.)**

### T3 — Hyperbolic Kepler `M = e·sinh F − F` (~120 LOC)
1. New `func HyperbolicAnomalyFromMean(M, e float64, maxIter int) float64` mirroring `TrueAnomalyFromMean` API. Conway-Prussing 1986 seed `F₀ = ln(2|M|/e + 1.8)` (Vallado §2.2.4). Newton update `dF = (e·sinh F − F − M) / (e·cosh F − 1)`.
2. Pair with `func TrueAnomalyFromHyperbolic(F, e float64) float64` analogous to the existing half-angle conversion: `ν = 2·atan2(√(e+1)·sinh(F/2), √(e−1)·cosh(F/2))` (Vallado eq. 2-36).
3. Reference: Battin §4.3; Vallado §2.2.4. **(~120 LOC + golden vectors with NIST/ESA Horizons cross-check.)**

### T4 — Halley iteration + R-MUTUAL-CROSS-VALIDATION pin (~40 LOC)
1. Add unexported `keplerHalley(M, e, E0 float64, maxIter int) float64` for cross-validation only.
2. Add `TestKepler_NewtonHalleyMarkley_AgreeAt_e0p7` (e ∈ {0.0, 0.3, 0.5, 0.7}, M ∈ 50 sweeps): assert `|E_newton − E_halley| < 1e-12` and `|E_newton − E_markley| < 1e-12`. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 (third-order Halley + cubic-Padé Markley + textbook Newton).
3. Add `TestKepler_MarkleyConvergesAt_e0p99` (regression): at e = 0.99, M = 1e-3, Newton with seed `E = M` and `maxIter = 5` does **not** reach residual 1e-12; Markley does in 2 iters. Pins the divergence-vs-Markley contrast literally.
4. Add `TestKeplerUniversal_DegradesToNewton_LowE` (after T2): for `e = 0.1` the universal-variable solver and the Newton solver must agree to 1e-11 — the elliptic-low-e degenerate-case pin.

### Recommended day-1 PR
**T0 + T1 stacked (~130 LOC)**: hardens existing API (T0 alone) + ships the Markley starter (T1) behind the same signature. Immediate measurable wins: golden file recomputed (high-e residuals tighten by ~3 orders of magnitude); zero API breakage; sets up T4's three-way pin. T2 (Stumpff/universal) can land in a follow-up because it opens new public surface and needs its own goldens.

## Sources

### Repo files
- `C:\limitless\foundation\reality\orbital\orbital.go:213-266` — `TrueAnomalyFromMean` Newton solver (sole Kepler primitive)
- `C:\limitless\foundation\reality\orbital\orbital_test.go:113-125, 287-315` — golden + unit tests; missing residual assertion
- `C:\limitless\foundation\reality\orbital\testdata\orbital\true_anomaly.json` — existing golden vectors
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\106-orbital-numerics.md:15-66` — F1–F4 numerical defects already raised (this dive corroborates and extends with method-specific recommendations)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\107-orbital-missing.md:23-37` — T1.1 (universal-variable) and hyperbolic-Kepler scope already specced
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\110-orbital-perf.md:131-160` — Stumpff Chebyshev table perf optimization (downstream of T2)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\164-synergy-orbital-optim.md:119` — pairing with `optim.NewtonRaphson`/`BisectionMethod` for robustness
- `C:\limitless\foundation\reality\CLAUDE.md` — rule 5 (precision documented), golden-file mandate, no-allocation policy

### External sources (web)
- F. L. Markley, "Kepler Equation Solver", *Celestial Mechanics & Dynamical Astronomy* **63**, 101–111 (1995). Cubic-Padé starter + 5th-order corrector; ≤ 2 iters in 99.996% of e ∈ [0,1) cases. ResearchGate: <https://www.researchgate.net/publication/227099796_Kepler_Equation_solver>
- Raposo-Pulido & Peláez, "An efficient code to solve the Kepler equation. Elliptic case", *MNRAS* **467**, 1702 (2017): <https://academic.oup.com/mnras/article/467/2/1702/2929272> — current state-of-the-art benchmark; outperforms Markley at very high e.
- Tommasini & Olivieri, "Two fast and accurate routines for solving the elliptic Kepler equation", *A&A* (2022): <https://www.aanda.org/articles/aa/full_html/2022/02/aa41423-21/aa41423-21.html> — modern alternative to Markley; simpler implementation, comparable accuracy.
- Wisdom & Hernandez, "A fast and accurate universal Kepler solver without Stumpff series", *MNRAS* **453**, 3015 (2015): <https://academic.oup.com/mnras/article/453/3/3015/1752673> — alternative universal-variable formulation that avoids Stumpff cancellation entirely.
- Wikipedia "Universal variable formulation": <https://en.wikipedia.org/wiki/Universal_variable_formulation>
- Wikipedia "Stumpff function": <https://en.wikipedia.org/wiki/Stumpff_function>
- Orbital Mechanics & Astrodynamics open textbook (Bayer/Hall): <https://orbital-mechanics.space/time-since-periapsis-and-keplers-equation/universal-variables.html>

### Standard textbooks (cited but not linked)
- Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed. (2013), §2.2 alg. 8 ("Kepler"), §2.2.4 (hyperbolic anomaly), §2.3 (universal variables)
- Battin, *An Introduction to the Mathematics and Methods of Astrodynamics*, rev. ed. (1999), Ch. 3 (hyperbolic), §4.3 (universal-variable formulation), §4.5 (Stumpff functions), §10.7 (Sundman regularization)
- Bate, Mueller, White, *Fundamentals of Astrodynamics* (1971), §4.4–4.5 (universal variables; original textbook treatment)
- Curtis, *Orbital Mechanics for Engineering Students*, 3rd ed. (2014), Ch. 3 (Kepler equation, universal variable, Stumpff numerical implementation with code listings — closest in style to Reality's per-function godoc)
- Danby, *Fundamentals of Celestial Mechanics*, 2nd ed. (1992), §6.6 (Kepler-equation Newton convergence; the "0.85·e" starter)
- Conway & Prussing, *Orbital Mechanics* (1993; 1986 Conway hyperbolic-starter paper) — F₀ = ln(2|M|/e + 1.8)
- Nijenhuis, "Solving Kepler's equation with high efficiency and accuracy", *Cel. Mech. Dyn. Astr.* **51**, 319 (1991) — pre-Markley starter alternative; less robust than Markley but historically referenced
- Mortari & Conway, "Single-stage formation of solutions to Kepler's equation", *J. Astronaut. Sci.* (2017) — non-iterative closed-form path; not suitable for our use because of polynomial-coefficient table size, but cited for completeness
