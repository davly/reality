# 061 — em-numerics

**Topic:** em: field superposition cancellation errors, FDTD CFL stability
**Scope:** numerical-correctness audit of `C:\limitless\foundation\reality\em\`
**Files audited:**
- `em\em.go` (213 LOC, 11 exported functions)
- `em\em_test.go` (287 LOC)
- `em\testdata\em\*.json` (10 golden files, 45 vectors total)

## Executive summary

`em/` is a 213-LOC "EM 101 cheat sheet" — eleven scalar formulas implemented as one-line arithmetic over `float64`. Numerically the present surface is **correct on its happy paths and IEEE-754-clean** with the sole arithmetic operations being multiply, divide, and `math.Sqrt`. There are no superpositions, no integrals, no field expansions, no FDTD, no current-loop / wire field, no impedance arithmetic, no S-parameters, no eigenvalue analysis. Nothing in the topic-prompt list of cancellation/CFL/elliptic-integral hazards exists in this package because **every primitive that would exhibit those hazards is itself missing**. (Missing-primitive enumeration is package 062; this report flags only the numerical-correctness implications and the four real correctness issues in the present surface.)

Two latent bugs (E-1, E-2) and two specification gaps (E-3, E-4) are the only numerical-correctness concerns in current code. Six **forward-looking** numerical-correctness commitments must be settled before any of the topic-prompt primitives (multi-source field summation, FDTD, complex impedance, magnetic field of current loop) lands, since reality's golden-file contract binds those algorithm choices cross-language at first ship.

## What the package contains today

| Function | Formula | Hot-path ops |
|---|---|---|
| `CoulombForce(q1,q2,r)` | `k·q1·q2/r²` | 3 mul, 1 div |
| `ElectricField(q,r)` | `k·q/r²` | 2 mul, 1 div |
| `OhmsLaw(V,R)` | `V/R` | 1 div |
| `PowerElectric(V,I)` | `V·I` | 1 mul |
| `ResistorsInSeries([]R)` | `Σ Rᵢ` | n add |
| `ResistorsInParallel([]R)` | `1/Σ(1/Rᵢ)` | n div, n add, 1 div |
| `CapacitorEnergy(C,V)` | `½CV²` | 3 mul |
| `InductorEnergy(L,I)` | `½LI²` | 3 mul |
| `RCTimeConstant(R,C)` | `R·C` | 1 mul |
| `ResonantFrequencyLC(L,C)` | `1/(2π√(LC))` | 1 mul, 1 sqrt, 1 mul, 1 div |

Coulomb's constant is precomputed once at package init (`coulombConst = 1/(4π·ε₀)` from `constants.VacuumPermittivity = 8.8541878128e-12 F/m`). No per-call recomputation, no allocations in any function, no internal state.

## Topic-prompt items: what does and doesn't apply

The topic prompt named ten numerical hazard classes. Mapping to current code:

| Hazard class | Applies to current `em/`? | Resolution |
|---|---|---|
| Field superposition cancellation | **No** — no superposition function exists | Forward-looking (§Forward) |
| 1/r² and 1/r singularity at r=0 | **Yes** — present in CoulombForce/ElectricField | Resolved correctly (E-3) |
| Capacitor `½CV²` energy | **Yes** | Numerically clean (mul order order-of-magnitude safe) |
| Inductor `½LI²` energy | **Yes** | Numerically clean |
| RC time constant accuracy | **Yes** | Single mul, exact at IEEE-754 |
| LC eigenvalue analysis | **No** — only ω₀ scalar, no state-space | Forward-looking |
| Series/parallel impedance many-component | **Partial** — purely real R only, no complex Z | E-1 + Forward |
| Ohm's law division by zero | **Yes** | Resolved correctly (E-4) |
| Magnetic field of current loop / wire | **No** | Forward-looking |
| FDTD CFL stability `cΔt/Δx ≤ 1/√3` | **No** — no FDTD | Forward-looking |
| Frequency-domain / S-parameters | **No** — no `complex128` in package | Forward-looking |
| IEEE-754 edge cases | **Yes** — propagate cleanly via direct arithmetic | Resolved correctly (E-3) |

Of ten topic-prompt items, four (1/r singularity, energy storage, RC, Ohm-zero-R) actually exist in code today and three of those are correct. The other six are forward-looking commitments for primitives that don't yet exist.

## Findings (current code)

### E-1 — `ResistorsInParallel` accumulates `1/Rᵢ` without compensation

**Severity:** low (real but only triggers at n ≳ 10⁴ with mixed magnitudes)
**Location:** `em.go:131-143`

The body is a textbook naive accumulation:

```go
sumInv := 0.0
for _, r := range resistances {
    if r == 0 { return 0 }
    sumInv += 1.0 / r
}
return 1.0 / sumInv
```

Each `1.0/r` introduces ~½ ULP rounding; the running sum picks up additional ~½ ULP per term. For `n` terms with mixed magnitudes (e.g., ~10² mixed with 10⁻⁶) the worst-case relative error grows as `O(n·ε)` (`ε ≈ 2.22e-16`). For n=10 mixed-magnitude resistors the error is ~10⁻¹⁵, well inside the 1e-10 tolerance the golden vectors carry. For n=10⁵ (a reasonable VLSI ladder) it climbs to ~10⁻¹¹ — still inside 1e-10 but eroded.

**Fix:** Kahan-Neumaier compensated summation, ~10 LOC, no allocation:

```go
sumInv, c := 0.0, 0.0
for _, r := range resistances {
    if r == 0 { return 0 }
    y := 1.0/r - c
    t := sumInv + y
    c = (t - sumInv) - y
    sumInv = t
}
return 1.0 / sumInv
```

The same Kahan pass should land on `ResistorsInSeries` (today identical naive `total += r`) for consistency.

**Tests today:** zero golden vectors for n>3; no large-n test; no mixed-magnitude test. CLAUDE.md §1 mandates 20+ vectors per function — `resistors_parallel.json` has 6 cases, `resistors.json` has 4. Both gold files are well below mandate.

### E-2 — `ResistorsInParallel(R, ∞)` returns NaN, not R

**Severity:** low (semantically wrong but no caller affected today)
**Location:** `em.go:131-143`

For `ResistorsInParallel([]float64{100.0, math.Inf(1)})` the loop computes `1/100 + 1/Inf = 0.01 + 0.0 = 0.01`, then returns `1/0.01 = 100.0`. **That's correct** — an open-circuit branch (R=∞) in parallel is a no-op, and the function happens to handle it because `1/Inf == 0` in IEEE-754. Good.

But for `ResistorsInParallel([]float64{math.Inf(1), math.Inf(1)})` we get `0 + 0 = 0` → `1/0 = +Inf`. Correct (two open circuits in parallel = open circuit).

For `ResistorsInParallel([]float64{math.NaN(), 100})`: `1/NaN = NaN`, `NaN + ... = NaN`, `1/NaN = NaN`. NaN propagates correctly.

**However:** there is no test verifying any of these. CLAUDE.md §1 mandates IEEE-754 edge cases (+Inf, -Inf, NaN, -0.0, subnormals). Zero of these are tested in `resistors_parallel.json`. Same for `resistors.json` (series). Add ~6 IEEE-754 vectors per function, ~30 LOC JSON. (E-1's Kahan rewrite must preserve all IEEE-754 propagation behavior — verify by re-running enriched golden file before/after.)

### E-3 — `CoulombForce(q,q,0.0)` returns +Inf with sign-of-charges; `r = -0.0` quirk

**Severity:** documentation only
**Location:** `em.go:42-44`, `em.go:60-62`

`r*r` for `r=0.0` yields `+0.0`; `k·q1·q2 / +0.0` yields `+Inf` if `q1·q2 > 0`, `-Inf` if `q1·q2 < 0`, `NaN` if either charge is 0 (because `0·k/0 = 0/0 = NaN`). This is what the docstring claims ("Returns ±Inf if r == 0"), but the docstring **does not mention the `q=0` case where the result is NaN, not 0**. Caller computing the force on an uncharged test particle at the location of a source gets `NaN` from a function whose docstring promises `±Inf`.

For `r = -0.0`: `(-0.0)*(-0.0) = +0.0` so behavior is identical to `r=+0.0`. Fine.

**Fix:** add one sentence to docstring: `// Returns NaN if r==0 and either charge is 0 (indeterminate 0/0).` Or, defensively, special-case `r==0` to return `+Inf` with sign of `q1*q2`. The defensive fix matches physics intuition (force diverges) but adds a branch in a hot path; recommend doc-fix only.

`ElectricField(0, 0)` has the same NaN-not-Inf issue with the same fix.

**Tests today:** zero `r=0` vectors in `coulomb_force.json` or `electric_field.json`. CLAUDE.md mandates these. Add 4 vectors each (+0, -0, +0 with q=0, NaN propagation).

### E-4 — `OhmsLaw(0, 0)` returns NaN, golden file silent on this

**Severity:** documentation only
**Location:** `em.go:78-80`, `ohms_law.json`

`OhmsLaw(10, 0) = +Inf` is tested (`em_test.go:192-197`). `OhmsLaw(0, 0)` returns NaN (0/0). Not tested, not documented in the docstring's "Valid range: R != 0" line which silently elides the V=R=0 indeterminate. Add IEEE-754 vector + one docstring sentence.

### E-5 — `ResonantFrequencyLC` precision

**Severity:** none — implementation is correct
**Location:** `em.go:211-213`

Formula `1/(2π·√(LC))` at IEEE-754 picks up ~4 ULP: 1 from `L*C` mul, ~1 from `math.Sqrt` (Go's stdlib `math.Sqrt` is correctly-rounded on amd64 via `SQRTSD`), 1 from `2π` mul, 1 from final div. Worst-case relative error `~5e-16`, well inside the 1e-12 tolerance the golden file carries. The current alternative `math.Sqrt(1/(L*C)) / (2π)` would not be more accurate — it pushes the same operations through different ordering. **No fix needed.**

For `L=0` or `C=0`: `√(0) = 0`, `1/(2π·0) = +Inf`. Correct (zero LC = infinite resonance, degenerate). For `L<0` or `C<0`: `√(negative) = NaN`. Correct (unphysical, NaN propagates). Neither case is in the golden file.

### E-6 — `coulombConst` precision is bounded by `ε₀`'s reported precision

**Severity:** none — already documented, just noting
**Location:** `em.go:21`, `constants/physics.go:52`

`VacuumPermittivity = 8.8541878128e-12 F/m` (CODATA 2018 value, 9 significant figures). `coulombConst = 1/(4π·ε₀) ≈ 8.987551792262e+9 N·m²/C²`. The Coulomb's constant precomputation introduces ~2 ULP error from `4π` and `1/ε₀`; this is below the precision floor set by `ε₀`'s 9-sig-fig representation (~1e-9 relative). The constant is at least as precise as its input — correct. Docstring at `em.go:39` cites this correctly ("limited by uncertainty in ε₀").

In SI 2019 redefinition, `ε₀` is no longer exact (it's derived from measured values of `α` and `c`), so 9 sig figs is the right honest precision. **No fix.**

## Forward-looking numerical-correctness commitments

The following must be settled **before** the corresponding 062-missing primitive ships — first-ship binds golden-file algorithm choice cross-language.

### F-1 — Multi-source field superposition: pairwise / Kahan-compensated, not naive sum

**Hazard:** catastrophic cancellation when summing fields from N sources at a distant evaluation point. Each source contributes `kq/r²` ~ 10⁻⁹ (e.g., 1nC at 1m → ~9 V/m). At a far point with N=10⁶ sources of mixed sign, naive `Σ Eᵢ` accumulates O(N·ε)·max|Eᵢ| error which can dwarf the actual sum (which is the small difference of large positive and negative contributions).

**Resolution choices for `ElectricFieldSuperposition([]Source, point) Vec3`:**
1. **Naive `+=`** — O(N·ε) error, fails for mixed-sign far fields. Reject.
2. **Pairwise summation** — O(log N · ε) error, no extra storage, ~5% slower than naive. Recommend.
3. **Kahan-Neumaier compensation** — O(ε) error independent of N, ~10% slower, 1 extra accumulator per axis. Recommend for high-N consumer (Pistachio particle systems).
4. **Sort-by-magnitude then sum smallest-first** — O(log N · ε), requires N·8 bytes scratch + O(N log N) sort. Reject for hot path.

Recommended: ship **pairwise as default** with Kahan available as `ElectricFieldSuperpositionKahan` for caller paying ~10% perf to bound error to O(ε). Same shape Numerical Recipes §4.6 recommends, same shape Drake/Bullet physics use for force accumulation.

### F-2 — FDTD CFL: `cΔt ≤ Δx/√d` enforced as precondition not silent-clamp

**Hazard:** FDTD Yee scheme is conditionally stable on `cΔt/Δx ≤ 1/√d` where d is spatial dimension (1D: 1, 2D: 1/√2, 3D: 1/√3 ≈ 0.577). Violation grows exponentially per timestep; one timestep over-CFL → simulation blows up to Inf within ~10-20 steps.

**Resolution:** every `FDTD3DStep`/`FDTD2DStep`/`FDTD1DStep` function MUST validate CFL on entry and either (a) `panic` with explicit message (matches `linalg.LU` policy on singular matrix), (b) return `(stepResult, error)`, or (c) silently substitute Δt = Δx/(c√d·1.01) (silent-clamp — REJECT, hides design bug). Recommend (b) error-return for FDTD because it's an offline-design-time function, not a per-frame Pistachio function.

Test commitment: golden file MUST include CFL=exactly-1 vector (marginal stability, 100 steps, verify bounded), CFL=1.01 vector (verify error returned), CFL=0.99 vector (verify converges to analytical plane-wave solution to ~1e-3 over 100 steps). Pure-math foundation calls for ≥30 vectors per FDTD primitive per CLAUDE.md §1.

### F-3 — Magnetic field of current loop: complete elliptic integrals K(m), E(m), not series approximation

**Hazard:** `B_axial(loop, point)` requires complete elliptic integrals of 1st and 2nd kind K(m), E(m). Naive Taylor series diverges as m→1 (point near loop wire). Series at m=0 converges fast for distant points but loses precision near the loop. Three implementation choices:
1. **Carlson symmetric form RF/RD/RC/RJ** (Numerical Recipes §6.11, Press 1992): O(log(1/ε)) iterations, uniformly convergent over m∈[0,1), branch-free, ~30 LOC. **Recommend.**
2. **AGM (arithmetic-geometric mean)** Brent 1976: O(log log(1/ε)) iterations, fastest, but harder to vectorize; ~25 LOC. Acceptable.
3. **Cephes/Hastings series with branch at m=0.5**: 4 LOC trivial, ~1e-11 precision, but algorithm-divergent across the branch and **fails CLAUDE.md cross-language golden test** because Python/C#/C++ ports might choose different m₀.

Recommend Carlson RF/RJ — Boost.Math, scipy.special, mpmath, GSL all use Carlson; choosing it makes Python/C++ ports trivially golden-equivalent.

Test commitment: ≥30 vectors spanning m ∈ {0, 1e-15, 1e-6, 0.5, 0.999, 0.9999} including the limit m→1 where K diverges as ½·ln(16/(1-m)) (analytical asymptote testable to ~1e-12).

### F-4 — Complex impedance arithmetic: `complex128` not `(real, imag) float64` pairs

**Hazard:** Future `Impedance(R, L, C, ω)` and `SeriesImpedance([]complex128) complex128` and `ParallelImpedance([]complex128) complex128`. Two API choices:
1. **Native `complex128`** — Go's complex arithmetic is bit-exact across `math/cmplx`, golden-file contract binds via `cmplx.Abs/Phase`. Recommend — matches `signal/`, `control/` (in 053 recommendation).
2. **Pair of float64** — caller composes manually. Reject — fragments cross-package consistency, doubles golden-file argument count.

Same numerical hazard as E-1 for parallel-impedance: `1/Σ(1/Zᵢ)` over `complex128` accumulates O(N·ε) error; recommend complex Kahan summation (separate compensators for real and imag parts), ~20 LOC.

Edge case: `1/Z` for `Z = 0+0i` returns `complex128(NaN, NaN)` per Go spec. Document and golden-test.

### F-5 — Eigenvalue analysis of LC ladder: characteristic-polynomial vs companion-matrix

**Hazard:** N-stage LC ladder has 2N modes; finding them via characteristic-polynomial roots (e.g., Durand-Kerner per `control/transfer.go`) suffers the multiple-root degradation 051 documented (control-numerics: Durand-Kerner degrades to linear convergence on multiple roots, multi-root error ~ε^(1/m)). For an LC ladder with degenerate L=L=L=L (uniform line), the characteristic polynomial has clustered roots → DK exposes ~1e-5 error.

**Recommended:** when 062-missing's `LCLadderModes(L, C, n)` lands, use **companion-matrix QR** via `linalg/eigen.go` (when that ships) rather than Durand-Kerner. This is parallel to 051's recommended fix C-TF-ROOTS-1 for `control/`.

### F-6 — S-parameter conversion: numerical conditioning of S↔Z↔Y↔ABCD

**Hazard:** S → Z conversion `Z = Z₀(I+S)(I-S)⁻¹` has condition number unbounded as S → I (passive lossless network at resonance). Naive matrix inversion `linalg.MatrixInverse(I-S)` blows up.

**Recommended:** use `linalg.SolveLU(I-S, Z₀(I+S))` (when available) for stability; document that `S → Z` conversion fails (returns error) when `det(I-S)` < 1e-14, which physically corresponds to a network with unity reflection coefficient (perfectly reflecting at port). Same posture for Y → ABCD, T-parameters, etc.

## Test coverage audit

| Function | Vectors | CLAUDE.md mandate (20+) | IEEE-754 edges? |
|---|---|---|---|
| CoulombForce | 5 | **fails** (need 15 more) | none — no r=0, no q=0, no NaN/Inf |
| ElectricField | 4 | **fails** (need 16 more) | none |
| OhmsLaw | 4 | **fails** (need 16 more) | one (R=0 in unit test, not golden) |
| PowerElectric | 4 | **fails** (need 16 more) | none |
| ResistorsInSeries | 4 | **fails** (need 16 more) | none — no [], no Inf, no NaN |
| ResistorsInParallel | 6 | **fails** (need 14 more) | one (R=0 short) |
| CapacitorEnergy | 4 | **fails** (need 16 more) | none |
| InductorEnergy | 4 | **fails** (need 16 more) | none |
| RCTimeConstant | 4 | **fails** (need 16 more) | none |
| ResonantFrequencyLC | 4 | **fails** (need 16 more) | none — no L=0, C=0, negative |

**Totals:** 45 vectors across 11 functions = 4.1 avg vs 20 mandate = **~20% of mandate**. Same shortfall pattern flagged in 056 crypto-numerics (5% of mandate). Recommend bundle EM-GOLDEN-1 ~250 LOC JSON to bring all 11 functions to mandate, ~25 vectors each.

IEEE-754 mandatory edges (CLAUDE.md §1: "+Inf, -Inf, NaN, -0.0, subnormals"): zero functions cover the full mandate, two cover one edge each. Bundle EM-GOLDEN-2 ~80 LOC to add 6 IEEE-754 vectors per function = 66 cases.

## Recommended bundle

Ship-now non-breaking, total ~395 LOC:

| ID | LOC | Description |
|---|---|---|
| EM-NUM-1 | 10 | Kahan-Neumaier in `ResistorsInSeries` + `ResistorsInParallel` (E-1) |
| EM-DOC-1 | 12 | Docstring fixes for `q=0,r=0` NaN cases on Coulomb/Field/Ohm (E-3, E-4) |
| EM-GOLDEN-1 | 250 | Bring all 11 functions to 20+ vectors each |
| EM-GOLDEN-2 | 80 | Add 6 IEEE-754 mandatory edges per function (66 cases) |
| EM-DOC-2 | 5 | Document floating-point precision floor (~ε₀ 9-sig-fig limit) at package level |
| EM-DOC-3 | 38 | Forward-looking F-1..F-6 numerical-correctness specification block in `em.go` package doc, settled before 062 ships |

Forward-looking architectural commitments (zero LOC today, 062 prerequisite):
- F-1 pairwise + Kahan superposition pattern
- F-2 FDTD CFL precondition error-return contract
- F-3 Carlson elliptic-integral choice
- F-4 native `complex128` for impedance
- F-5 companion-matrix QR for ladder modes (depends on `linalg/eigen`)
- F-6 LU-solve for S↔Z conversion (depends on `linalg/lu`)

## Non-overlap statement

This report covers **numerical correctness of present `em/` surface + numerical-correctness specification commitments for forward primitives**. Specifically:

- **062 em-missing** owns the primitive enumeration: which superposition / FDTD / loop-field / impedance / S-parameter functions to add, in what tier order, with LOC estimates. 061 names them only as forward-looking *numerical*-correctness commitments (F-1..F-6 algorithm-choice questions whose answers must be settled when each primitive ships).
- **063 em-sota** owns architectural patterns: how peer EM libraries (Meep, Lumerical, COMSOL RF, scikit-rf, MEEP-FDTD) type their fields/sources/grids. 061 makes no peer-library statements.
- **064 em-api** owns Go signature shapes once primitives land. 061 names ergonomic shapes (e.g., `ElectricFieldSuperposition vs Kahan` variant) only because they are inseparable from the algorithm-choice numerical question.
- **065 em-perf** owns per-call cycle counts and SIMD/parallel posture. 061 mentions perf only via "5%/10% slower" magnitudes that bear on the algorithm-choice tradeoff.

## Bottom line

Eleven scalar functions, 213 LOC, all arithmetically correct on happy paths, all IEEE-754 propagation-clean, no allocations, no internal state. **Two real bugs** (E-1 missing Kahan, E-3/E-4 missing q=0/V=R=0 docstring) totaling ~22 LOC fix. **Test coverage at ~20% of CLAUDE.md mandate** — single largest near-term action is EM-GOLDEN-1+2 ~330 LOC of JSON to lock present surface cross-language. **Six forward-looking numerical-correctness commitments** (F-1..F-6) must be answered before 062's missing primitives ship — first-ship binds golden-file algorithm choice cross-language.

The catastrophic-cancellation, FDTD CFL, elliptic-integral, and S-parameter hazards named in the topic prompt are forward-looking: zero such code exists in `em/` today. Recommended posture is to settle the F-1..F-6 algorithm choices in the 062 design phase and golden-test them at first ship rather than retrofit corrected algorithms post-release once Python/C++/C# ports have already crystallized around naive variants.
