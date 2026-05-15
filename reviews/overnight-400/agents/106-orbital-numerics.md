# 106 — orbital: Numerical Correctness Audit

**Scope.** Single file: `C:/limitless/foundation/reality/orbital/orbital.go` (267 LOC, 8 exported functions) plus tests (`orbital_test.go`, 329 LOC) and golden vectors (`testdata/orbital/*.json`).

Functions audited: `KeplerOrbit`, `OrbitalPeriod`, `OrbitalVelocity` (vis-viva), `HohmannTransfer`, `EscapeVelocity`, `HillSphere`, `SynodicPeriod`, `TrueAnomalyFromMean` (Kepler-equation solver).

Severity legend: **CRIT** = wrong result / silent divergence / domain violation; **HIGH** = robustness gap, classical-method violation, range failure; **MED** = missing safeguard, contract drift; **LOW** = cosmetic / doc.

**Headline.** The package is the smallest in `reality` (8 functions, no zonal harmonics, no propagator, no time scales, no frame transforms) and on the happy path the closed-form formulas are textbook-correct, but `TrueAnomalyFromMean` has at least three independently observable failure modes near `e→1`: (i) the divisor `1 - e·cos(E)` collapses to `0` at periapsis (M=0, e→1) producing `NaN`/`±Inf`; (ii) the absolute-value Newton-stop `|dE| < 1e-15` is below `eps·E` for E near π and so the loop runs the full `maxIter` budget with no progress on the last bit; (iii) **silent non-convergence** — when `maxIter` is exhausted the routine returns the last iterate with no error, no flag, no counter, the caller cannot distinguish converged-to-machine-eps from "still oscillating". Doc says "0 ≤ e < 1" but no guard rejects e ≥ 1, e < 0, or NaN. Almost the entire MASTER_PLAN audit checklist (universal-variable formulation for any conic, J2/J3 perturbations, ICRF/ITRF/ECI/ECEF transforms, TT/TAI/UTC/TDB/JD time conversions, Kepler propagator over many orbits, parabolic/hyperbolic eccentricity branches) **is not present in the package at all** — those are missing-feature findings owned by the sibling 107 (orbital-missing) audit, but the implications for numerical correctness of the existing 8 functions are noted below where they create silent foot-guns (e.g. `KeplerOrbit` divides by `1+e·cos(ν)` so feeding it ν near π for e≥1 hyperbolic produces wrong-sign `r`).

---

## Findings — numerical correctness of existing 8 functions

### F1 — `TrueAnomalyFromMean`: Newton denominator `1 - e·cos(E)` → 0 at periapsis as e → 1 (CRIT)
File: `orbital.go:246`.

The Newton step is

```go
dE := (E - e*sinE - M) / (1.0 - e*cosE)
```

The denominator `1 - e·cos(E)` is the derivative `dM/dE`. At periapsis (E=0, ν=0) and `e=1` (parabolic limit) the denominator is **exactly 0**, so the very first Newton step from the initial guess `E = M = 0` yields `0/0 = NaN`. For `e = 1 - δ` with small δ and `M ≈ 0`, the denominator is `δ + O(E²)` so the first step blows up to `dE ≈ M/δ`, far overshooting; the iteration recovers but burns most of the iteration budget chasing the overshoot back. There is **no Laguerre fallback**, no bisection backstop, no quadratic-approximation initial guess.

Three classical fixes (Conway 1986; Danby 1992 §6.6):

1. **Better initial guess** for high e: use `E_0 = M + e·sin(M) / (1 - sin(M+e) + sin(M))` (Danby's "best initial guess"), or for `e > 0.8` use Conway-Laguerre starter.
2. **Laguerre-Conway iteration** instead of Newton — order-5 convergence and no zero-denominator pathology because the Laguerre formula `dE = nf / (f' ± √((n-1)²f'² - n(n-1)ff''))` keeps the discriminant bounded away from zero by construction.
3. **Hyperbolic / parabolic branches** — for `e ≥ 1` the equation is `M = e·sinh(F) - F` (hyperbolic) or Barker's equation (parabolic). Currently neither is implemented — feeding `e ≥ 1` to the function silently runs the elliptic Newton on a non-convex residual and converges to garbage, then the `Sqrt(1-e)` factor in the ν conversion becomes `Sqrt(<0) = NaN`.

Severity is CRIT because the package documents `0 ≤ e < 1` but does not guard against violation, and the parabolic limit `e → 1` is the classical numerical flash-point of Kepler-equation solvers.

### F2 — `TrueAnomalyFromMean`: silent non-convergence when `maxIter` exhausted (HIGH)
File: `orbital.go:243-251`.

```go
for iter := 0; iter < maxIter; iter++ {
    ...
    if math.Abs(dE) < 1e-15 {
        break
    }
}
```

If the loop exits via `iter == maxIter` (e.g., e=0.99, M=0.01, maxIter=10), the routine returns the last `E` with no signal that convergence was not achieved. The function signature is `(M, e float64, maxIter int) float64` — single return value, no error, no `Converged bool`, no iteration counter. Compare with `optim.NewtonRaphson` which has the same defect (per agent 101 F-N) — this is a package-wide pattern. Recommend either:
- Add a second return `iters int` (so caller can compare to `maxIter`), or
- Return `(nu float64, err error)` with `errIterationLimit` sentinel, or at minimum
- Document the silent-failure mode in the godoc.

The unit test `TestTrueAnomalyFromMean_ConvergesForHighE` at line 311 checks only that the result is in `[0, 2π)` and not NaN — it does **not** verify Kepler-equation residual `|E - e·sinE - M| < tol`, so a nonconvergent-but-finite result passes the test.

### F3 — Newton stop test `|dE| < 1e-15` is sub-`eps·E` near E ≈ π (HIGH)
File: `orbital.go:248`.

The absolute-value tolerance `1e-15` is `≈ eps/2 = 1.11e-16 × 9` — for `E ≈ π = 3.14...` the relative tolerance is `1e-15 / π ≈ 3.2e-16 ≈ 1.4·eps`, so the stop test is roughly tight for `E ~ 1`. But for `E ~ 6.28` (just under 2π) the relative tolerance becomes `~5.1·eps`, three iterations short of full machine precision; for `E < 1e-15` (very small M near periapsis) the test triggers immediately on the initial residual without doing any work. Recommend:

```go
if math.Abs(dE) < eps*math.Max(1.0, math.Abs(E)) {
    break
}
```

with `const eps = 2.220446049250313e-16` (`math.Nextafter(1, 2) - 1`). Add a redundant residual check `|f| = |E - e·sinE - M| < eps_M` for early exit on flat-residual problems.

### F4 — `TrueAnomalyFromMean`: input validation absent (MED)
File: `orbital.go:232-266`.

No guard against `e < 0`, `e ≥ 1`, `NaN(M)`, `NaN(e)`, `maxIter ≤ 0`. Effects:
- `e < 0`: `Sqrt(1-e) > 1` and `Sqrt(1+e) < 1` swap relative magnitudes — formal output but physically nonsense.
- `e ≥ 1`: as in F1, `Sqrt(1-e)` returns NaN, the result is NaN.
- `NaN(M)` or `NaN(e)`: NaN propagates silently — caller sees NaN with no diagnostic.
- `maxIter ≤ 0`: loop body never executes; function returns ν computed from `E = M` (valid only for circular).

Reality convention (per CLAUDE.md "Precision documented, not assumed") is to either reject out-of-range inputs explicitly or document the IEEE-754 propagation behavior. Currently neither.

### F5 — `KeplerOrbit`: divides by `1 + e·cos(ν)` with no guard for `e ≥ 1`, `ν = π` collision (HIGH)
File: `orbital.go:46`.

```go
r := a * (1 - e*e) / (1 + e*math.Cos(nu))
```

Three problems:
1. For **parabolic** orbits (`e = 1`), the formula `r = a(1-e²)/(1 + e·cos ν)` is `0/0` at ν=0 — the conic equation degenerates to `r = p/(1 + cos ν)` where `p` is the semi-latus rectum, but `a` is infinite for parabolic so the Cartesian-element representation is ill-defined. There is **no Cartesian-via-(p, e, i, Ω, ω, ν) variant** that handles all conics.
2. For **hyperbolic** orbits (`e > 1`), `1 - e²` is **negative**, so `r` comes out negative for `cos(ν) > -1/e` — the function silently returns a wrong-sign Cartesian. Hyperbolic orbits are physically valid (interplanetary flybys, escape trajectories), and the package convention `a > 0` does not communicate "elliptical only" — the doc says `0 ≤ e < 1` but the code does not enforce.
3. For **elliptical** orbits at ν=π (apoapsis), no problem; but at ν approaching the asymptote of a hyperbola (`cos(ν) = -1/e`), the denominator → 0 and `r → +∞`. With `e=1.5`, `cos(ν) = -0.6667` (ν ≈ 2.30 rad), the asymptote is reached and `r` is finite-but-huge until exact match → ±Inf.

Universal-variable formulation (Bate-Mueller-White ch. 4.5, Vallado §2.3) handles all conic types in one branchless routine via Stumpff functions C(z), S(z) — currently **not present** in the package.

### F6 — `OrbitalVelocity` (vis-viva): no guard against negative kinetic-energy regime (MED)
File: `orbital.go:105-107`.

```go
return math.Sqrt(mu * (2.0/r - 1.0/a))
```

For `r > 2a` (which can happen on hyperbolic orbits with `a < 0` by convention, or on bound orbits where caller passes `r` from a different orbit), the argument `2/r - 1/a` becomes **negative** and `Sqrt` returns **NaN**. The doc says `a > 0` but standard astrodynamics convention uses `a < 0` for hyperbolic — vis-viva is `v² = μ(2/r - 1/a)` where `a < 0` for hyperbolic gives `2/r + 1/|a|` (positive, larger than escape), so the formula is **correct** for `a < 0` if you let the sign carry through. The current "valid range a > 0" doc forbids this convention without rationale; either:
- Document explicitly that `a > 0` is required and add `if a <= 0 { return NaN }`, or
- Allow `a < 0` for hyperbolic (more standard), and add a guard for `r ≤ 0` only.

### F7 — `OrbitalPeriod`: silent NaN for hyperbolic / parabolic (MED)
File: `orbital.go:84-86`.

```go
return 2.0 * math.Pi * math.Sqrt(a*a*a/mu)
```

For `a ≤ 0` or `mu ≤ 0`, returns NaN with no diagnostic. The closed-form formula `T = 2π√(a³/μ)` is **only defined for elliptical** orbits (bound state); hyperbolic and parabolic have no period. Currently the function does not enforce this — feeding `a < 0` returns NaN, feeding `a = 0` returns 0, feeding negative `mu` returns NaN. Recommend explicit `if a <= 0 || mu <= 0 { return NaN }` with godoc note "elliptical only; returns NaN for non-elliptical inputs".

Also: `a*a*a` overflows to `+Inf` at `a > 5.6e102`, well below the inputs astronomers use for cosmological-scale computations. Better: `math.Pow(a, 1.5) / math.Sqrt(mu) * 2.0 * math.Pi`, but `Pow(a,1.5)` is slower than three multiplies and overflow at solar-system scale (`a < 1e13` m) is non-issue, so the current form is the right call — just document the `a < 5e34`-ish overflow limit.

### F8 — `HohmannTransfer`: no guard for `r1 == r2`, `r1 > r2`, or non-positive inputs (MED)
File: `orbital.go:135-142`.

For `r1 == r2`, `dv1 = √(μ/r1)·(√1 - 1) = 0` and `dv2 = √(μ/r2)·(1 - √1) = 0` — clean zero (per `TestHohmannTransfer_SameOrbit_ZeroDeltaV`). But:
- For `r1 > r2`, the formula is **still mathematically valid** (transfer from outer to inner orbit) but `dv1` becomes **negative** (retrograde burn) and the doc says "always >= 0" — that contract is violated when the caller passes outer-to-inner. Either swap-and-flag, swap-and-return-as-positive-with-direction-indicator, or document "r2 must be >= r1; otherwise dv1 is negative meaning retrograde".
- For `r1 ≤ 0` or `r2 ≤ 0`, `Sqrt(μ/r)` returns NaN/Inf depending on sign — silent.
- For `r1 + r2 > Float64max/2` (cosmological-scale), `r1 + r2` overflows to +Inf, then `2r2/sum = 0` and `√0 = 0` so `dv1 = -v1` (negative), `dv2 = +v2` (positive) — **clearly wrong** because both burns should be ≥ 0 for inner-to-outer. Real-world risk is zero (no one passes parsecs to Hohmann) but it's a charter-precision issue.

### F9 — `EscapeVelocity` uses `M, r` not `mu, r` — interface inconsistency with rest of package (LOW/MED)
File: `orbital.go:159-161`.

```go
func EscapeVelocity(M, r float64) float64 {
    return math.Sqrt(2.0 * constants.GravitationalConst * M / r)
}
```

Every other function in the package (`OrbitalPeriod`, `OrbitalVelocity`, `HohmannTransfer`) takes `mu = G·M` as a single combined parameter, with the package-level doc explicitly saying "μ is typically known to higher precision for celestial bodies." `EscapeVelocity` breaks this convention — it takes `M` and multiplies by `constants.GravitationalConst` internally, which:
1. Hits the **CODATA 2018 G uncertainty of 2.2e-5 relative** (per `physics.go:41-45`) instead of the much tighter μ values astronomers use (e.g. μ_⊕ = 3.986004418e14 with relative uncertainty ~1e-9).
2. Is **inconsistent** with sibling functions, forcing callers to remember which interface flavor each function uses.

Sibling `physics` package or `em` may have the same dual-interface pattern; recommend either:
- Add `EscapeVelocityMu(mu, r float64) float64 { return math.Sqrt(2*mu/r) }` as the primary form, demote current to convenience helper, or
- Rename to `EscapeVelocityFromMass` to make the lower-precision input explicit.

### F10 — `HillSphere`: `m << M` precondition not enforced (LOW)
File: `orbital.go:181-183`.

The formula `r_H ≈ a · (m/(3M))^(1/3)` is the leading-order approximation; for `m ~ M` the Hill-sphere concept breaks down (binary system, Lagrange-point analysis takes over). Doc says `m << M` but no guard. For `m = M`, returns `a · (1/3)^(1/3) ≈ 0.693·a`, which is dimensionally meaningful but physically wrong (no dominant primary). Recommend either:
- `if m >= M/100 { return NaN }` with strict precondition, or
- Document tolerance: "valid for m/M < 1e-2; accuracy degrades to leading-order approximation for larger ratios."

`Cbrt` (`math.Cbrt`) is the right choice (correctly handles negative inputs and is precise to ~1 ulp); no change needed there.

### F11 — `SynodicPeriod`: `T1 == T2` check uses `== 0` after subtraction, susceptible to cancellation (LOW)
File: `orbital.go:201-207`.

```go
diff := math.Abs(1.0/T1 - 1.0/T2)
if diff == 0 {
    return math.Inf(1)
}
return 1.0 / diff
```

For `T1 = T2` exactly the `==0` check works. For `T1 ≈ T2` the cancellation error is `eps·max(1/T1, 1/T2)` so `1/diff` is finite-but-huge. Edge cases:
- `T1 = T2 = 1e308`: `1/T1 = 1e-308` (denormal-ish), `diff = 0` exactly because both round to same denormal — returns +Inf correctly.
- `T1 = 1.0, T2 = math.Nextafter(1.0, 2.0)`: `diff ≈ 4.44e-16`, returns `1/diff ≈ 2.25e15` — synodic period of ~2.25 quadrillion years for two periods that differ by 1 ULP. Mathematically correct! No issue.
- `T1 < 0` or `T2 < 0`: doc says "T1 > 0, T2 > 0" but no guard. Function does not panic (returns negative or +Inf), but result is meaningless.

LOW because no callers will hit the bad path; but charter-clean would be `if T1 <= 0 || T2 <= 0 { return math.NaN() }`.

### F12 — `KeplerOrbit`: rotation-matrix labelling says "3-1-3 Euler" but is the standard (Ω, i, ω) PQW→ECI transform (LOW, doc only)
File: `orbital.go:60`.

The comment "Rotation matrix elements (3-1-3 Euler: Ω, i, ω)" is technically the correct sequence (rotate about Z by -Ω, then X by -i, then Z by -ω, then we're going from inertial to perifocal — or the inverse for perifocal to inertial). The constructed matrix is correct (verified against Vallado eq. 2-145), so the issue is purely doc clarity. The convention `R(-Ω)·R(-i)·R(-ω)` versus `R_z(Ω)·R_x(i)·R_z(ω)` versus `R_3(-Ω)·R_1(-i)·R_3(-ω)` is the most-confused notation in astrodynamics — recommend the doc explicitly cite Vallado eq. 2-145 or Curtis eq. 4.49 with the column convention used.

### F13 — Reference frame is not specified (MED, doc)
File: `orbital.go` (whole package).

`KeplerOrbit` returns `(x, y, z)` but the doc does not say in which frame:
- Heliocentric ICRF? Geocentric J2000 ECI? Local PQW?

The 3-1-3 rotation from PQW (perifocal) gives the inertial frame whose Z-axis is the reference plane normal and whose X-axis is the ascending-node direction — for Earth orbits this is **ECI J2000** if Ω is measured from the J2000 vernal equinox, but the function does not specify the reference epoch. Caller-supplied (Ω, i, ω) determines the frame, but without doc the user can't reconstruct what frame they're in.

The MASTER_PLAN audit checklist explicitly calls out "Coordinate frame transforms: ICRF / ITRF / ECI / ECEF" — none of these transforms exist in the package. Currently `KeplerOrbit` is the only function that produces vector output and its frame is implicit-from-input. Document that ambiguity.

---

## Findings — checklist items missing entirely (out of scope here, owned by 107)

The MASTER_PLAN bullet list has 10 audit topics; the audited package contains essentially **2.5 of them**:

| MASTER_PLAN bullet | Status in package |
|---|---|
| Kepler equation: M = E - e·sin(E); Newton iteration | Present (with bugs F1-F4) |
| Convergence near e=1 (parabolic limit) | **Broken** (F1, F3) |
| Universal variable formulation for any conic | **Absent** |
| Vis-viva equation: numerical stability | Present (with MED gap F6) |
| Hohmann transfer: closed-form correctness | Present (with MED gap F8) |
| Perturbation: J2, J3 zonal accumulation | **Absent** |
| Coordinate frame transforms: ICRF / ITRF / ECI / ECEF | **Absent** |
| Time conversions: TT, TAI, UTC, TDB, JD precision | **Absent** |
| Kepler propagator: long-horizon energy drift | **Absent** (no propagator at all) |
| Conic eccentricity edge cases (parabolic, hyperbolic) | **Absent** (F1, F5) |

Energy-drift over many orbits (the bullet about "long-horizon energy drift") cannot be audited because there is **no propagator** in the package — `KeplerOrbit` is a pure positional function (given (a, e, i, Ω, ω, ν), output (x, y, z)), with no time-stepping. The user calls `TrueAnomalyFromMean(M, e, maxIter)` once per epoch and feeds the resulting ν back to `KeplerOrbit` — there's no integration loop where energy could drift. (The architectural absence of a propagator is correct per zero-allocation hot-path charter — propagation is application-layer, but Kepler's equation must be solved each step and F1-F4 affect every such call.)

J2/J3 zonal-harmonic perturbation, ICRF/ITRF/ECI/ECEF frame chains, and TT/TAI/UTC/TDB time scales are **completely absent**. The package is "Two-body, point-mass, instantaneous-position" — the entire dynamical-astronomy stack is downstream of what's here. These are missing-feature findings owned by 107-orbital-missing.md, not numerical-correctness findings, but the audit-checklist phrasing forces them into scope here as null findings.

---

## Test-suite gaps related to F1-F4

`orbital_test.go` is 329 LOC, 8 golden-file tests + 13 unit tests. Coverage gaps:

1. **No Kepler-equation residual test**: `TestTrueAnomalyFromMean_*` checks circular identity, E=π fixed point, sign normalization, and "doesn't NaN at e=0.95" — but **does not verify** `|E_returned - e·sin(E_returned) - M| < eps`. A nonconvergent-but-bounded result passes all current tests. Add:
   ```go
   func TestTrueAnomalyFromMean_ResidualSmall(t *testing.T) {
       for _, e := range []float64{0, 0.1, 0.5, 0.9, 0.99, 0.999} {
           for _, M := range []float64{0.01, 0.1, 1.0, 3.0, 6.0} {
               nu := TrueAnomalyFromMean(M, e, 100)
               // Convert back: ν → E → M_check, compare to M
               E := 2 * math.Atan(math.Sqrt((1-e)/(1+e)) * math.Tan(nu/2))
               if E < 0 { E += 2*math.Pi }
               Mcheck := E - e*math.Sin(E)
               if math.Abs(Mcheck - M) > 1e-13 {
                   t.Errorf("e=%g M=%g: residual %g", e, M, Mcheck-M)
               }
           }
       }
   }
   ```
2. **No e ≥ 1 rejection test**: feeding e=1.0 currently returns NaN silently; a test would either pin the NaN behavior or demand a guard.
3. **No periapsis-near-parabolic stress test**: `e=0.999, M=1e-6` is the worst case for Newton-from-E=M starter; no such case in goldens.
4. **`TestKeplerOrbit_*` does not test hyperbolic** (`e > 1`) — would expose F5.
5. **Hohmann reverse-direction not tested** (F8) — `r1 > r2` case has no assertion.

---

## Summary table — patches sized

| F# | Severity | Patch | LOC |
|---|---|---|---|
| F1 | CRIT | Laguerre-Conway iteration + Conway high-e starter, branch on e<1 / e=1 / e>1 | ~80 |
| F2 | HIGH | Add `iters int` second return + `errIterationLimit`, godoc warning | ~20 |
| F3 | HIGH | Replace `1e-15` with `eps·max(1, |E|)` + residual cross-check | ~6 |
| F4 | MED | Input guards (e<0, e≥1, NaN, maxIter≤0) | ~15 |
| F5 | HIGH | Universal-variable Cartesian via Stumpff C(z), S(z) — new function `KeplerOrbitUniversal` | ~120 |
| F6 | MED | Allow `a < 0` for hyperbolic vis-viva, document, guard `r ≤ 0` | ~10 |
| F7 | MED | Guard `a ≤ 0`, `mu ≤ 0` in OrbitalPeriod | ~5 |
| F8 | MED | Document r1 > r2 contract or auto-swap | ~10 |
| F9 | LOW | Add `EscapeVelocityMu(mu, r)` primary form | ~10 |
| F10 | LOW | Document or guard `m/M ≥ 0.01` | ~5 |
| F11 | LOW | Guard `T1 ≤ 0`, `T2 ≤ 0` | ~5 |
| F12 | LOW | Doc: cite Vallado eq. 2-145 explicitly | ~3 |
| F13 | MED | Doc: state output frame conventions | ~10 |

**Sprint-1 (CRIT+HIGH, must-fix for Reality v0.11):** F1+F2+F3+F5 ≈ ~230 LOC. Brings Kepler equation up to Conway-Laguerre standard, handles all conic eccentricities via universal variable, and removes silent non-convergence. **Sprint-2 (MED hygiene):** F4+F6+F7+F8+F13 ≈ ~50 LOC; tightens contract documentation across the package. **Sprint-3 (LOW polish):** F9+F10+F11+F12 ≈ ~25 LOC.

Total v0.11 numerics-correctness sprint: **~305 LOC, 13 patches**, focused on the existing 8 functions only. The 7 missing checklist items (universal variable, J2/J3 perturbation, ICRF/ITRF/ECI/ECEF frame chain, TT/TAI/UTC/TDB time scales, propagator, hyperbolic Kepler, parabolic Barker) are missing-feature findings totalling **~2,500-4,000 LOC** owned by 107-orbital-missing.md.

---

## Cross-cutting

- **Convention drift with siblings**: `EscapeVelocity(M, r)` is the only function in `orbital` that ingests mass instead of μ; sibling `physics.GravitationalAttraction` (if present) likely uses `(m1, m2, r)`, so this might be cross-package convention rather than orbital-internal drift. Worth a sibling check (out of scope here).
- **Float64 reproducibility**: All formulas use only `*`, `/`, `Sqrt`, `Cos`, `Sin`, `Atan2`, `Cbrt`, `Mod`, `Abs`, `Inf`. No `Pow`, no FMA, no transcendental fusion. **Bit-exact across platforms** assumed (matches `testutil` golden-file design). The Newton iteration at line 246 has the only deeply iterative flow — the `1e-15` stop tolerance plus `maxIter` cap make the iteration count deterministic given (M, e, maxIter). Reproducibility is fine.
- **No allocations**: All 8 functions are pure scalar arithmetic, zero `make`, zero slice ops. Pistachio-friendly. **No allocations in hot paths** (CLAUDE.md key rule 3) confirmed.
- **Citations present**: Every function cites Bate-Mueller-White (1971) or Vallado / Murray-Dermott / Meeus / Hill / Hohmann. **Mathematical provenance** (CLAUDE.md key rule 4) present and adequate.
- **Precision documented**: Each godoc states "Precision: limited by float64 sqrt (~15 significant digits)" — adequate for happy-path inputs but **does not document failure modes for edge cases** (e ≥ 1, a ≤ 0, etc.). Per CLAUDE.md rule 5, every function should state failure modes; currently they're implicit.

Reviewer: agent 106.
