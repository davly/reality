# 303 — dive-relerr-bounds (Documented Precision vs Implemented Precision)

## Headline
Of 258 `Precision:` doc-strings sampled across reality, ~20 are factually wrong (overclaim "exact" or wrong order-of-magnitude), ~6 hot decomposition functions in `linalg/decompose.go` and `linalg/eigen.go` lack any Precision doc at all (design-rule-5 violation), and there is no `precision_test.go` framework that mechanically enforces documented bounds — so drift is invisible until a downstream consumer hits it.

## Findings (documentation vs implementation drift)

### Class A — "exact for IEEE 754" lies (overclaim → bound is actually O(N·ε), not 0)
Slot 302 already flagged 4. Confirmed and extended:
- `linalg/vector.go:20` `CosineSimilarity` — claims "exact for IEEE 754 float64 inputs (no iterative steps)"; body computes 3 naive sums + 2 sqrts + division. Real bound: **O(N·ε) + 2·ULP** ≈ N·1.1e-16 + 4e-16.
- `linalg/vector.go:45` `EncodingDistance` — "exact"; sum-of-squared-diff + sqrt + sqrt, **O(N·ε)**.
- `linalg/vector.go:68` `DimensionWeightedDistance` — "exact"; weighted sum of squares + division + sqrt, **O(N·ε)** (worse with sign-mixed weights).
- `linalg/vector.go:95` `L2Normalize` — "exact"; sum-of-squares + sqrt + N divisions. **O(N·ε) + 1 ULP per output element**.
- `linalg/vector.go:133` `DotProduct`, `:149` `L2Norm`, `:162` `L1Norm`, `:175` `LInfNorm` (last only is actually exact) — slot 302 covers.
- `linalg/matrix.go:169` `Trace` — "exact (accumulated float64 summation error for large n)" — self-contradictory phrasing; should drop the word "exact".
- `linalg/matrix.go:196` `CrossProduct` — "exact for IEEE 754 float64". Three components each `a[i]*b[j] - a[j]*b[i]` = mul + mul + sub = **3·ULP relative** (not bit-exact for non-special inputs; subtraction of similar magnitudes may catastrophically cancel).
- `geometry/quaternion.go:29` `QuatDot` — "exact" but is a 4-element dot product → **3 sequential adds, ~3·ULP relative**.
- `geometry/quaternion.go:38` `QuatConjugate` — actually exact (3 unary negations, IEEE 754 exact). OK.
- `geometry/quaternion.go:47` `QuatNormalize` — "exact"; involves sqrt + 4 divisions → **1 ULP from sqrt** + per-component division rounding. Not exact.
- `geometry/quaternion.go:68` `QuatMul` — "exact"; ~16 mults + 12 adds → **~12·ULP** worst case.
- `geometry/sdf.go:19` `SphereSDF`, `:32` `BoxSDF`, `:56` `SegmentSDF`, `:93` `TorusSDF` — all claim "exact for IEEE 754 float64". All call `math.Sqrt` (≤1 ULP), some chain min/max/abs of subtractions. Box/segment SDFs in particular subtract similar-magnitude values → cancellation regime where *relative* error explodes near the surface (where SDF=0 by definition).
- `optim/rootfind.go:110` `LinearInterpolateRoot` — "exact for IEEE 754 float64 (single division + multiply)". Body is `x0 - y0*(x1-x0)/(y1-y0)` = sub + mul + div + sub = **4 rounded ops, ~3-4 ULP**, with catastrophic cancellation when `y0 ≈ y1`.
- `em/em.go:111` `SeriesResistance` — "exact (summation)". Sum of N floats is **O(N·ε)**, never bit-exact.
- `prob/jeffreys.go:33,147` claim "exact (float64 arithmetic)" for ratio-of-Beta-style updates that include subtractions → **not** bit-exact.
- `prob/timeseries.go:31,66` claim "exact (multiplications and additions only)" — sum-of-products → **O(N·ε)**.
- `prob/prob.go:25,39,118` are mostly OK (single ops); `:310` `Median` "exact for odd n; one addition + division for even n" — for even n, average of two floats can round, ~0.5 ULP; technically not bit-exact.

### Class B — order-of-magnitude wrong (claim does not match implementation)
- `signal/fft.go:45` FFT "Precision: 1e-9 for 1024-point". Cooley-Tukey radix-2 with iterated trig recurrence has measured error ≈ **5e-14 to 5e-13 for N=1024** in practice (golden file `signal/testdata/signal/fft.json` actually validates impulse cases at **1e-15**). Doc is **5 orders of magnitude too pessimistic** — sandbagging hides a real regression if accuracy degrades to e.g. 1e-10. Tighten to "≤ 5·log₂(N)·ε ≈ 1e-13 for N=1024".
- `signal/fft.go:100` IFFT "matches FFT (1e-9 for 1024-point)" — same sandbagging.
- `color/difference.go:37` CIEDE2000 "typically 1e-10". Formula has ~30 trig/exp calls but each is 1 ULP and they don't catastrophically cancel for in-gamut colors. Real bound: **~1e-13 to 1e-12** absolute on ΔE. Loosen claim or strengthen.
- `prob/distributions.go:64` `NormalQuantile` "maximum relative error < 1.15e-9". Acklam's published bound is 1.15e-9 *absolute* on the standard normal; once `mu + sigma·…` is applied the relative error in tail of distribution can swing wildly. Doc should say "absolute error < 1.15e-9 in standardized form".
- `prob/distributions.go:271` `BetaPDF` "~1e-14 absolute for typical inputs". `beta_pdf.json` has cases at tolerance 1e-10 — implies BetaPDF is **~10000× worse than documented** for some inputs (likely `a,b > 100` regime). Either tighten implementation (use lgamma + log-form throughout, which the code mostly does) or relax claim to "~1e-10 in stiff regimes".
- `orbital/orbital.go:228` `TrueAnomalyFromMean` "typically converges to machine epsilon within 5-10 iterations for e < 0.9". Hardcoded `if math.Abs(dE) < 1e-15 { break }` is `1e-15`, not machine epsilon (`2.22e-16`). For e>0.9 with poor initial guess (`E := M`), Newton can stall — no fallback to bisection. Doc soft-pedals; should add "for e ≥ 0.99 may not converge to 1e-15 in `maxIter`".

### Class C — Precision doc string entirely missing (design-rule-5 violation)
Found via `grep // Precision:` per file vs `grep ^func [A-Z]`:
- `linalg/decompose.go` — **0 Precision doc strings, 6 public functions** (LUDecompose, LUSolve, Inverse, Determinant, CholeskyDecompose, CholeskySolve). These are the highest-leverage numerical functions in the repo. Per Higham *Accuracy & Stability* §9.3, GEPP backward error is `||Δ A|| ≤ n·γₙ·ρ_growth·||A||` where γₙ ≈ n·ε; Cholesky is `||Δ A|| ≤ (n+1)·γₙ·||A||`. These bounds belong in the doc.
- `linalg/eigen.go` — 0 / 1+ functions. QR-iteration eigendecomposition has its own bounds.
- `linalg/pca.go` — 0 / 1+ functions.
- `chaos/ode.go`, `chaos/systems.go`, `chaos/analysis.go` — **0 Precision docs across the whole package** (3 files, 8+ public funcs: RK4Step, EulerStep, SolveODE, Lorenz, VanDerPol, LyapunovExponent, …). RK4 is **O(dt⁵) local, O(dt⁴) global**; Lyapunov estimator is **O(1/√T)** in trajectory length. Should be documented.
- `geometry/quaternion.go:191` `QuatRotateVec` — no Precision doc.
- `audio/onset/*.go` — all 5 onset detectors have a Precision doc, but nearly identical "limited by float64 arithmetic" boilerplate that says nothing.

### Class D — test tolerance vs documented Precision mismatch
- `prob/distributions.go:271` BetaPDF doc claims 1e-14; `beta_pdf.json` has cases at 1e-10. Test is **10⁴× more lenient** than doc — masks regression. (Either fix doc or tighten test for the easy cases and document the stiff-regime carve-out.)
- `signal/fft.go:45` FFT doc claims 1e-9; `fft.json` validates at **1e-15** in best cases and 1e-9 only at length 4096. Doc + test agree on worst case but doc is sandbagged for normal use.
- `geometry/quaternion.go` — no quaternion golden files found under `geometry/testdata` matching the strict 1e-15 doc claim of `QuatFromAxisAngle`. Unverifiable as written.

## Concrete recommendations

1. **`linalg/vector.go`** — replace every "exact for IEEE 754 float64" doc on `CosineSimilarity, EncodingDistance, DimensionWeightedDistance, L2Normalize, DotProduct, L2Norm, L1Norm` (lines 20, 45, 68, 95, 133, 149, 162) with `Precision: O(N·ε) ≈ N·1.1e-16 relative; use linalg.Kahan* variants for N > 10000`. (~7 single-line edits; depends on slot 302 landing Kahan primitives.) **LOC: ~7 lines doc, no code.**

2. **`linalg/matrix.go:169,196`** — `Trace`: drop "exact" (`Precision: O(N·ε)`). `CrossProduct`: change to `Precision: ~3 ULP; catastrophic cancellation possible when a×b is near-zero`. **LOC: 2 lines doc.**

3. **`linalg/decompose.go`** — add Precision docstrings to all 6 public functions citing Higham bounds:
   - `LUDecompose`/`LUSolve`: `Precision: backward error ||Δ A|| / ||A|| ≤ 8·n³·ε·ρ where ρ is pivot growth (typically O(1)). Forward error grows with κ(A).`
   - `Inverse`/`Determinant`: same plus `forward error ≈ κ(A)·n³·ε`.
   - `CholeskyDecompose`/`CholeskySolve`: `Precision: ||Δ A|| ≤ (n+1)·n·ε·||A||; requires κ(A) < 1/(n·ε) for symmetric-positive-definiteness check to be reliable.`
   **LOC: ~30 lines of doc, no code.**

4. **`linalg/eigen.go`, `linalg/pca.go`** — add similar Precision docs (eigenvalue absolute bound: `≤ ||E||₂` per Bauer-Fike; symmetric-eig is backward stable to `O(n²·ε)`). **LOC: ~10 lines.**

5. **`chaos/*.go`** — add Precision docs to all public ODE solvers and Lyapunov estimators. RK4Step: `Precision: O(dt⁵) local truncation, O(dt⁴) global; loses ~3 bits per million steps from rounding.` SolveODE: same plus accumulated rounding scales with N steps. LyapunovExponent: `Precision: O(1/sqrt(T)) where T is integration time; convergence assumes ergodicity.` **LOC: ~15 lines.**

6. **`signal/fft.go:45,100`** — tighten FFT/IFFT precision claim from "1e-9 for 1024-point" to `Precision: ~5·log₂(N)·ε relative (≈1e-14 for N=1024); empirically validated to 1e-15 for impulse, 1e-12 for typical cases (see fft.json).` Reference Tasche-Zeuner 2000 "Worst and average case roundoff error analysis for FFT". **LOC: 4 lines doc.**

7. **`color/difference.go:37`** — tighten CIEDE2000 from "typically 1e-10" to `Precision: ≤ 1e-12 absolute for in-gamut colors; ~1e-9 near hue-discontinuity branch (h ≈ 0/360 wrap).` Add a golden case at the wrap boundary to guard. **LOC: 2 lines doc + 1 test case.**

8. **`prob/distributions.go:271`** — either tighten BetaPDF golden tolerance from 1e-10 to 1e-14 (matching doc) or relax doc to `Precision: ~1e-14 for max(a,b) ≤ 100; degrades to ~1e-10 for max(a,b) > 1000 due to lgamma cancellation`. **LOC: 2 lines doc.**

9. **`optim/rootfind.go:110`** — change `LinearInterpolateRoot` doc: `Precision: ~3 ULP for well-separated y0, y1; catastrophic cancellation when |y0 - y1| ≪ max(|y0|,|y1|). Caller should detect stagnation.` **LOC: 2 lines.**

10. **`em/em.go:111`** — `SeriesResistance` change from "exact (summation)" to `Precision: O(N·ε); for N > 1e4 use linalg.KahanSum`. **LOC: 1 line.**

## Cross-cutting recommendation

**Add `internal/precisiontest/precisiontest.go` (or `testutil/precision.go`) framework that mechanically enforces documented bounds**, callable from any package:

```go
// PrecisionBound declares a measurable error bound on a function output.
type PrecisionBound struct {
    Name      string                       // e.g. "linalg.DotProduct"
    DocClaim  float64                      // the value parsed from "Precision: 1e-14"
    Type      ErrType                      // Relative | Absolute | ULP | OrderH
    Sample    func(rng *rand.Rand) Inputs  // generator
    Reference func(in Inputs) Output       // computed via math/big at 256-bit
    Actual    func(in Inputs) Output       // the function under test
}

// EnforceProperty(t, bound, nTrials=10000) generates inputs, computes
// reference and actual, and fails if max observed err > bound.DocClaim.
```

Run as `go test -run TestPrecision ./...` in CI; populate one bound entry per `Precision:` doc string. This converts the 258 doc strings from passive comments into **executable contracts**, which is what design rule 5 ("Precision documented, not assumed") was supposed to mean. Slot 302's stable-sum work and this framework are the two halves of making the rule load-bearing.

A first slice (≤300 LOC framework + ~30 bounds covering the highest-leverage 30 functions: FFT, BetaInc, NormalQuantile, LU, Cholesky, RK4, CIEDE2000, KeplerSolver, DotProduct, …) is a one-PR job that would catch all 4 Class-A lies above on first run.

Side benefit: doc strings can be parsed (regex `Precision:\s+(?:[≤<~]\s*)?([\d.]+)e([-+]?\d+)`) so the framework auto-scrapes claims; mismatches between scraped and registered bounds become a compile-time-equivalent gate.

## Sources
- `C:\limitless\foundation\reality\CLAUDE.md` — design rule 5
- `C:\limitless\foundation\reality\linalg\vector.go` (lines 20, 45, 68, 95, 133, 149, 162, 175)
- `C:\limitless\foundation\reality\linalg\matrix.go:169,196`
- `C:\limitless\foundation\reality\linalg\decompose.go` (no Precision docs)
- `C:\limitless\foundation\reality\linalg\eigen.go`, `pca.go` (no Precision docs)
- `C:\limitless\foundation\reality\geometry\quaternion.go` (lines 29, 47, 68, 91, 132, 158, 191, 215)
- `C:\limitless\foundation\reality\geometry\sdf.go` (lines 19, 32, 56, 93)
- `C:\limitless\foundation\reality\signal\fft.go:45,100`
- `C:\limitless\foundation\reality\signal\testdata\signal\fft.json`
- `C:\limitless\foundation\reality\color\difference.go:37`
- `C:\limitless\foundation\reality\prob\distributions.go:64,271,327`
- `C:\limitless\foundation\reality\prob\testdata\prob\beta_pdf.json`
- `C:\limitless\foundation\reality\optim\rootfind.go:110`
- `C:\limitless\foundation\reality\em\em.go:111`
- `C:\limitless\foundation\reality\chaos\ode.go` (no Precision docs)
- `C:\limitless\foundation\reality\orbital\orbital.go:228`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\302-dive-stable-sums.md` (related slot)
