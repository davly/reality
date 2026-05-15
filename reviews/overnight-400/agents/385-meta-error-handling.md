# 385 — meta-error-handling (panic vs error vs NaN policy audit)

## Headline
Reality uses three error-signaling patterns inconsistently: panic dominates linalg/signal/queue (programmer-error guards), NaN dominates prob/fluids/optics (math infelicity), error returns are confined to crypto, graph, copula, and a few optim subpackages — but the policy is informal and the same error class is handled differently across packages (e.g. negative-arg → NaN in `prob.NormalPDF` vs panic in `queue.OfferedLoad`).

## Method
Static count of three patterns in production `*.go` (test files excluded by `_test.go` filtering where possible). "Panic count" = literal `panic(` occurrences in non-test files. "Error returns" = `errors.New|fmt.Errorf` constructions (excludes `var Err...`). "NaN returns" = `return math.NaN()` plus a few `return math.Inf(...)` for "math infelicity" treated as the same class. Counts are coarse but directionally reliable.

## Per-package error-handling style

| Package | Panic count | Error returns | NaN returns | Notes |
|---|---:|---:|---:|---|
| acoustics | 0 | 0 | 0 | Pure float→float; no failure paths. |
| calculus | 0 | 0 | 0 | Pure numerical methods; convergence failure is silent (returns whatever it has). |
| chaos | 0 | 0 | 0 | ODE integrators; can never fail per task spec. |
| color | 0 | 0 | 0 | Pure color space conversions. |
| combinatorics | 0 | 0 | 0 | Pure counting; overflow is silent (uint64 wrap). |
| compression | 0 | 0 | 2 | `entropy.go` returns NaN/Inf for log(0). |
| constants | 0 | 0 | 0 | Pure values. |
| control | 5 | 0 | 0 | `transfer.go` and `pid.go` panic on bad construction. |
| crypto | 0 | 4 | 0 | `modular.ChineseRemainder` returns errors only. `IsPrime(<2)` returns `false` silently. |
| em | 0 | 0 | 0 | Pure formulas; `Ohm's law` etc. let IEEE 754 propagate. |
| fluids | 0 | 0 | 2 | `fluids.go:25` documents "Returns +Inf/-Inf if mu==0 (per IEEE 754)" — the most disciplined doc-block in the repo. |
| gametheory | 0 | 0 | 2 | `kelly.go` returns NaN for invalid prob/odds. |
| geometry | 0 | 0 | 0 | Quaternion, SDF, polygon — all pure. |
| graph | 0 | 1 | 0 | `flow.go` exposes `var ErrCycleDetected`. |
| linalg | 52 | 0 | 0 | Heavy panic-style — every shape mismatch panics. `Inverse` returns `bool` (singularity is not panic). |
| optim | 3 | 19 | 0 | Mixed: `interpolate.go` panics on shape; `transport`, `proximal` return errors; `rootfind.go` uses NaN once. |
| orbital | 0 | 0 | 0 | Pure two-body; `math.Inf` returned once. |
| physics | 1 | 0 | 1 | `optics.go:24` returns NaN for total internal reflection. |
| prob | 2 | 35 | 35 | The most heterogeneous package: distributions return NaN; copula uses errors; conformal uses errors; `conformal/nonconformity.go` panics. |
| queue | 32 | 0 | 0 | Heaviest panic-style. Validates everything: empty network, non-square routing, neg probabilities. |
| signal | 27 | 0 | 0 | `fft.go` panics on non-power-of-2 N (slot 301 confirmed). |
| testutil | — | — | — | Test infra; not user-facing. |

Whole-repo totals (non-test): panics ≈ 122 (production code), error returns ≈ 60, NaN returns ≈ 42.

## Concrete inconsistency examples

1. **Same class, different handling: invalid distribution params.**
   - `prob/distributions.go:33` `NormalPDF(sigma<=0)` → returns NaN.
   - `queue/metrics.go:71` `OfferedLoad(lambda<=0)` → panics.
   - Both are "user error: nonpositive parameter that mathematically should be positive." Different responses.

2. **Same class, different handling: shape mismatch.**
   - `linalg/matrix.go:14` `MatMul` len mismatch → panic.
   - `optim/interpolate.go:47` `CubicSplineNatural` xs/ys len mismatch → panic.
   - `prob/copula/errors.go:43` `ErrLengthMismatch` paired inputs unequal → returned as error.
   - All three are textbook "programmer error" (slice contract violation), yet copula chose the user-error pattern. Inconsistent.

3. **Same class, different handling: singular / degenerate matrix.**
   - `linalg/decompose.go:153` `Inverse(singular)` → returns `bool` (false). Custom signal channel.
   - `prob/copula/errors.go:24` `ErrSigmaNotPSD` → error return.
   - Both are "input is the wrong kind of matrix." Three different mechanisms in two packages (panic for shape, bool for singularity, error for non-PSD).

4. **Same class, different handling: empty input.**
   - `linalg/correlation.go:136` `CovarianceMatrix(empty)` → panic.
   - `crypto/modular.go:98` `ChineseRemainder(empty)` → error return.
   - `prob/copula/errors.go:8` `ErrEmptyU` → error.
   - `queue/metrics.go:29` `BurstinessIndex(<2 samples)` → panic.

5. **Convergence failure handling.**
   - `queue/network.go:114` non-converging traffic equations → `panic`. Non-recoverable.
   - `optim/rootfind.go` Newton-Raphson divergence → silent (returns last iterate; one path returns NaN at line 115).
   - `calculus` integrators → silent (no convergence detection at all).
   - These are all the same situation; three responses.

6. **`IsPrime(0)` returns `false`** (`crypto/prime.go:27`). Reasonable, but not documented as a deliberate convention; contrast with `prob.NormalPDF(sigma=0)` which returns NaN by explicit guard.

## Recommendation: a written policy reality should adopt

Add a top-level `DOC_ERROR_POLICY.md` (or section in the main README/CLAUDE.md) with this rubric:

| Error class | Mechanism | Examples |
|---|---|---|
| **Programmer error** (slice length contract, nil receiver, negative `n` for matrix dim, output buffer too small) | `panic` with `package.Func: <reason>` | `linalg.MatMul` len, `signal.FFT` non-power-of-2, `queue.JacksonNetwork` non-square |
| **User error** (legal-typed but semantically invalid input that a user could plausibly supply: empty data slice, non-PSD matrix, non-coprime moduli, FFT N that *could* be padded) | `(T, error)` return | `crypto.ChineseRemainder`, `prob/copula` validators, `optim/transport` |
| **Math infelicity** (input where the result is mathematically undefined or improper, but IEEE 754 has a sentinel: log(0), 0/0, sqrt(-1), sigma=0 in pdf, total internal reflection) | `return math.NaN()` or `math.Inf(...)`; **must** appear in the `// Valid range:` and `// Returns:` doc lines | `prob.NormalPDF`, `fluids.ReynoldsNumber`, `physics/optics`, `gametheory.KellyGrowthRate` |
| **No failure path** (mathematically total) | nothing — no error, no NaN, no panic | `chaos.Lorenz` step, `acoustics.SpeedOfSound`, `color.RGBToXYZ`, `constants.*` |

Boundary calls and required fixes:

- **`signal.FFT(N not power of 2)`**: currently panics. This is a programmer error (length contract violation) — keep as panic. Document explicitly. Provide a separate `FFTAny` if/when bluestein/CZT lands (slot 301).
- **`linalg.Inverse` returning `bool`**: drop the bespoke bool and either (a) return `error` (singular = user-error class) or (b) write `out[i] = NaN` and return — but a `bool` is the worst option. Pick one.
- **`prob` distributions**: keep NaN-style (math-infelicity), but add a `// Returns NaN if:` line to **every** distribution function. Currently ~70% have it, ~30% don't (sample: PDFs document it; some quantile/sample helpers don't).
- **`prob/conformal/nonconformity.go` panics**: align with the rest of `prob/conformal` which uses error returns.
- **`queue/*`**: most current panics could be reclassified as user errors (a user might pass bad routing data without a "shape" violation per se). Recommend: keep `panic` only for true Go-level contract (length mismatches between slices the function declares paired); convert "routing row sum > 1", "all service rates positive" etc. to errors. ~20 of 32 panics in `queue` are doing user-input validation, not contract enforcement.
- **`linalg.CovarianceMatrix(empty)` panic**: should be an error (user gave an empty dataset — that's a real-world condition, not a buggy caller).

Convergence/iteration failures deserve their own subrule:

- Iterative solvers (`optim`, `chaos`, `calculus.Newton...`) should return **either** the iterate plus a `bool converged` **or** `(T, error)` with a typed `ErrNotConverged`. Silent non-convergence (the `calculus` and most of `optim`) is the worst case — it gives wrong results without a signal.

## Estimated cost of standardization

- `linalg`: ~10 panic→error reclassifications (only `Inverse` and `CovarianceMatrix`-class).
- `queue`: ~20 panic→error reclassifications.
- `prob`: ~15 docstrings to add a "Returns NaN if:" line. No code changes.
- `optim`: replace `bool` returns with `error`; standardize convergence reporting.
- `signal`: documentation-only (panic policy is correct; just say so).

Total: ~45 LOC behavior change, ~50 LOC docs. Mostly mechanical.

## Cross-references

- **slot 301** (`dive-fft-correctness`) confirmed `signal.FFT` panics on non-power-of-2 N. This audit reaffirms keeping that as panic (programmer-error class) and documenting it.
- **slot 303** (`dive-relerr-bounds`) is the natural companion: every function that returns NaN at boundary inputs should also document its precision and relative-error bound away from the boundary. The "Returns NaN if:" convention proposed here lines up exactly with the "Precision:" doc convention slot 303 audits.
- **slot 382** (`meta-test-coverage`) tests should include for *every* function: at least one input that triggers each declared NaN/error/panic path. A grep for `assert.Panics` and `assert.True(math.IsNaN(...))` would tell us how well coverage tracks the policy. (Out of scope for this slot; flagged for 382.)

## Sources

- `C:/limitless/foundation/reality/linalg/matrix.go` (panic-style examples)
- `C:/limitless/foundation/reality/linalg/decompose.go:153` (`Inverse` bool return)
- `C:/limitless/foundation/reality/signal/fft.go:52,55,107` (panic on bad N)
- `C:/limitless/foundation/reality/queue/network.go:52-128` (heavy panic validation)
- `C:/limitless/foundation/reality/queue/metrics.go:29-74` (panic vs the doc precedent)
- `C:/limitless/foundation/reality/prob/distributions.go:32-69` (NaN-style PDFs/CDFs)
- `C:/limitless/foundation/reality/prob/copula/errors.go` (error-return cluster, the gold-standard pattern)
- `C:/limitless/foundation/reality/crypto/modular.go:98-126` (error return for ChineseRemainder)
- `C:/limitless/foundation/reality/crypto/prime.go:26` (silent `false` for invalid input)
- `C:/limitless/foundation/reality/fluids/fluids.go:25` (best-in-class doc convention)
- `C:/limitless/foundation/reality/physics/optics.go:24` (NaN for total internal reflection)
- `C:/limitless/foundation/reality/gametheory/kelly.go:122-132` (NaN for invalid prob/odds)
- `C:/limitless/foundation/reality/optim/interpolate.go:47-58` (panic on shape mismatch)
- `C:/limitless/foundation/reality/graph/flow.go:108` (lone exported error)
- `C:/limitless/foundation/reality/control/transfer.go:38-88` (panic on empty polynomials)
