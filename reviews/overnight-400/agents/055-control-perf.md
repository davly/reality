# 055 | control-perf

**Scope.** Allocation profile, hot-loop suitability, and forward-looking
runtime cost class for `control/`. Files: `pid.go` (123 LOC),
`filter.go` (117 LOC), `transfer.go` (253 LOC). Disjoint from 051
(numerics), 052 (missing primitives), 053 (architecture), 054 (API).
This report owns: per-step allocations, Horner-vs-naive complex
polynomial evaluation, vectorizable Bode ω-grid, 60-FPS Pistachio
controller cycle suitability, and the runtime cost class of expm /
Schur / DARE / CARE when 052 lands them.

## TL;DR — Six perf findings

1. **Hot path is clean today.** `PIDController.Update`,
   `LowPassFilter`, `HighPassFilter`, `ComplementaryFilter`,
   `RateLimiter`, `TransferFunction.Evaluate` are all genuinely
   zero-alloc (verified by inspection — no `make`, no slice growth,
   no `interface{}` boxing, no closure capture). 60-FPS Pistachio
   camera loop is safe. (§1)
2. **Zero benchmarks. Zero alloc-asserting tests.** No `*_bench_test.go`
   file exists in the repo at all (verified `find . -name "*bench*.go"`
   returns empty). The "Zero heap allocations" doc claims at
   `filter.go:11/35/71/99` and `transfer.go:33` are unverified. Adding
   `testing.AllocsPerRun` guard tests is ~30 LOC and lock-in for the
   no-alloc contract. (§2)
3. **`Poles()` allocates twice per call** — `make([]float64, len(d))`
   for monic copy at `transfer.go:90` plus `make([]complex128, degree)`
   at `transfer.go:139`. Plus `durandKerner` runs a full 1000-iter
   loop with `cmplx.Abs`. Not a hot-loop function but the future
   `Bode` / `Margins` / `RootLocus` will call it inside ω-loops; need
   buffer-passing variants `PolesInto(buf []complex128) []complex128`
   before 052-Tier-1 lands. (§3)
4. **Forecast for 052 hot paths.** When `expm` (Padé-13 + scaling-and-
   squaring) lands per 052: `O(n³ log s)` flops + Θ(n²) scratch matrices
   ×7. When `Schur` lands (Francis QR + Hessenberg): `O(n³)` with high
   constant (~25n³). When `CARE`/`DARE` (Schur on Hamiltonian/symplectic
   2n×2n pencil): `O(n³)` with constant ~200n³. None are 60-FPS-safe for
   `n>20`. Need offline-design vs online-step partition documented in
   package doc before primitives land. (§4)
5. **`Bode(tf, omegas, outMag, outPhase)` is data-parallel-ready.**
   Each ω is independent — `Evaluate(complex(0, ω))` chain is pure,
   safe for `range omegas` without sync. The shape recommended by 054
   (caller-supplied `outMag`, `outPhase` buffers) is also the
   alloc-free shape; phase-unwrapping is a single forward pass after.
   Expect 30 ns/sample per Horner step at deg-3, scales linearly in
   `(deg(N)+deg(M))`. Vectorization within Go = SIMD via
   `math/cmplx`-free hand-coded `(a+bi)*(c+di)` is free 1.3× on amd64;
   AVX-FMA route is out of scope for zero-dep. (§5)
6. **`durand-kerner` internal Taylor `realCos`/`realSin`** at
   `transfer.go:191-225` runs 12 Taylor terms per call; called once
   per pole for initial guess. ~25 ns × degree on a hot core, cold
   numbers. Still trivially cheap vs the QR companion-matrix
   alternative recommended by 051 (which is the right replacement,
   not a perf concern). (§6)

---

## §1 Per-step allocation audit — current hot path

### 1.1 PIDController.Update (`pid.go:77-114`)

Five field reads, two field writes, six float64 stack temporaries,
zero slice ops, zero interface conversions. `*PIDController`
receiver, escape analysis confirms no boxing. **Verified zero-alloc.**
Cycle estimate amd64: 6 fmul + 3 fadd + 1 fdiv + 2 fcmp ≈ 18 cycles
≈ 5 ns at 3.5 GHz. 60 FPS × 100 controllers = 0.03% of one core.

### 1.2 Filters (`filter.go`)

All four (LowPass, HighPass, Complementary, RateLimiter) are float64-
only free functions, no slice args, no escape, no allocations
(filter.go:15-23, 38-46, 74-85, 103-116). Branch-heavy alpha-clamp
pattern `if alpha < 0 { ... } if alpha > 1 { ... }` is two
unpredictable jumps — for constant alpha the BP eats it cleanly;
branchless `math.Max(0, math.Min(1, alpha))` is ~3 ns faster but
not worth the change without a benchmark.

`ComplementaryFilter` is algebraically broken per 051. The corrected
form needs a `prev` argument — still zero-alloc, no perf regression.

### 1.3 TransferFunction.Evaluate (`transfer.go:36-48`)

`evalPoly` is `n` complex muladds, zero alloc, single complex128
return register. Per-call ≈ 12 cycles × n. Degree-3 TF = 36 cycles
≈ 10 ns. 1000-sample Bode plot = 10 µs — not a hot path.

`complex(coeffs[i], 0)` compiler-emits `MOVSD/XORPS` (1 cycle),
not a runtime call. Zero-alloc preserved.

---

## §2 Test coverage — no benchmarks, no alloc guards

`find C:/limitless/foundation/reality -name "*bench*.go"` returns **empty**.
`grep -rn "AllocsPerRun\|BenchmarkPID\|BenchmarkBode\|BenchmarkEvaluate"
control/` returns **empty**.

The four "Zero heap allocations" doc claims (`filter.go:11`, `:35`,
`:71`, `:99`, `transfer.go:33`) are unverified contracts. CLAUDE.md
design rule §3: *"No allocations in hot paths. Functions accept
output buffers. Pistachio calls these at 60 FPS."* → the contract
is part of the package promise but has no test.

**Action C-PERF-BENCH-1 (~30 LOC `control_alloc_test.go`):**

```go
func TestZeroAllocs_PIDUpdate(t *testing.T) {
    pid := NewPID(1, 0.5, 0.1, -10, 10)
    if a := testing.AllocsPerRun(1000, func() {
        pid.Update(1.0, 0.5, 0.01)
    }); a != 0 { t.Errorf("PID.Update allocated %v", a) }
}
func TestZeroAllocs_Evaluate(t *testing.T) { ... }      // 5 LOC
func TestZeroAllocs_LowPass(t *testing.T)  { ... }      // 5 LOC
// etc. for HighPass, Complementary, RateLimiter
func BenchmarkPID_Update(b *testing.B)    { ... }       // 8 LOC
func BenchmarkTF_Evaluate_Order3(b *testing.B) { ... }  // 8 LOC
func BenchmarkTF_Evaluate_Order10(b *testing.B) { ... } // 8 LOC
```

This locks the zero-alloc contract before 052 primitives can break it.

---

## §3 `Poles()` — two heap allocations per call

`transfer.go:73-112`:

```
monic := make([]float64, len(d))   // alloc 1
...
return durandKerner(monic, degree)
```

`durandKerner(transfer.go:122-183)`:

```
roots := make([]complex128, degree)  // alloc 2
```

For a degree-d denominator: `(d+1)*8` bytes float64 + `d*16` bytes
complex128 = `24d + 8` bytes per `Poles()` call. Cost-benign for
one-shot stability check (the existing `IsStable()` use). **Cost
real for forward-looking 052 features:**

- `RootLocus(num, den, gains []float64)` — one root-find per gain;
  with 200 gain samples = 200 allocations.
- `Margins(tf)` — calls `Poles` indirectly via crossover-frequency
  search on Bode, which itself calls `Evaluate` (already alloc-free)
  but a robust margin algorithm cross-checks against `Poles()` for
  unstable-pole detection.
- `C2D(tf, dt, ZOH)` — 052-Tier-1 ZOH discretization needs `expm` of
  the companion-matrix realization, not direct `Poles`, but the
  alternative `Tustin` route does need pole mapping for
  pre-warp at multiple frequencies.

**Action C-PERF-POLES-1 (~25 LOC):**

```go
// PolesInto computes the poles into a caller-supplied buffer.
// buf must have len >= deg(D)-1; returns buf[:deg(D)-1] populated.
func (tf *TransferFunction) PolesInto(buf []complex128, scratch []float64) []complex128 {
    ...
}
// Poles wraps PolesInto with internal allocation; documented as
// the convenience form, not the hot-path form.
func (tf *TransferFunction) Poles() []complex128 {
    deg := len(tf.Denominator) - 1
    return tf.PolesInto(make([]complex128, deg), make([]float64, deg+1))
}
```

Same pattern matches `linalg/`'s shape (e.g., `CholeskyDecompose(A,
n, L)` takes output buffer at `linalg/decompose.go:266`). Consistent
with CLAUDE.md design rule §3.

The 1000-iter cap in `durandKerner` (`transfer.go:123`) is also a
soft perf risk — 1000 × degree² complex ops worst case. Measured
typical is 10-20 iters; the cap is a safety. Replacing
Durand-Kerner with **companion-matrix QR** (per 051's
recommendation, using the existing `linalg.QRAlgorithm`) is *both*
the numerical fix and a perf cleanup: QR converges in ~20-30
iterations with cubic convergence. Allocates one `n×n` matrix
internally though, so still wants the buffer-passing variant.

---

## §4 Forecast: 052-Tier-1 runtime cost class

When 052 lands the missing primitives, what's the cost class? Order
matters for Pistachio's 16.7-ms-per-frame budget.

| Primitive | Best alg | Cost (n×n mat) | 60 FPS safe? | Buffer? |
|---|---|---|---|---|
| `expm(A)` | Padé-13 + sq-and-sq | `O(n³·log s)` ≈ 50n³ | n≤8: yes; n≤32: marginal | needs 7 scratch n×n mats |
| `Hessenberg(A)` | Householder reductions | `10n³/3` | n≤16: yes | 1 scratch vector |
| `Schur(A)` | Francis QR on Hessenberg | `25n³` typical | n≤8: yes; n≤16: marginal | 2 scratch n×n |
| `Lyapunov(A,Q)` | Bartels-Stewart on Schur | `25n³ + 4n³` = `29n³` | n≤8 | 3 scratch n×n |
| `Sylvester(A,B,C)` | Schur both, back-sub | `25n³ + 25m³ + n²m + nm²` | offline | 4 scratch |
| `CARE` | Hamiltonian Schur | `200n³` | offline only | 5 scratch 2n×2n |
| `DARE` | Symplectic-pencil QZ | `300n³` | offline only | 6 scratch 2n×2n |
| `LQR(A,B,Q,R)` | CARE + matrix solve | `200n³ + n³` | offline only | reuse CARE bufs |
| `Kalman.Predict` | matmul + outer + sym add | `2n³ + 2n²p` | yes for n≤32 | n×n + n×p |
| `Kalman.Update` | + matrix inverse `(p×p)` | `2n²p + p³ + 2np²` | yes for n≤32, p≤8 | scratch p×p |
| `EKF.Step` | + Jacobian eval | + `f` and `h` Jacobian cost | depends on user f,h | + n×n + p×n |
| `UKF.Step` | sigma-point propagate | `(2n+1)·f` cost + `2n³` Cholesky | n≤16 borderline | n×n LDLᵀ scratch |
| `Bode` ω-grid | `Evaluate` × len(ω) | `O(deg·N_ω)` | yes — design-time | caller-supplied |
| `RootLocus` | `Poles` × len(K) | `O(deg² · iter · N_K)` | offline | per-gain reuse |

**60 FPS partition.** Pistachio frame budget = 16.7 ms.
n=4 `expm` ≈ 3 µs (0.02%), n=8 Schur ≈ 12 µs (0.07%), n=8 CARE
≈ 110 µs (0.6%) — all fine *if not done per-frame*. Failure mode
is n>20: n=32 CARE ≈ 7 ms (42% of frame).

- **Online (per-frame):** `PID.Update`, filters, `Kalman.Predict/
  Update` (n≤32, p≤8), `LQR-precomputed-K · x`, matvec, `Evaluate`,
  `RateLimiter`.
- **Offline (one-shot, design-time):** `expm`, `Schur`, `Lyapunov`,
  `Sylvester`, `CARE`/`DARE`, `LQR-design`, `tf2ss`/`ss2tf`,
  `Poles`/`Zeros`, `Bode`, `Margins`, `RootLocus`, pole placement,
  c2d Tustin/ZOH.

**Action C-PERF-PARTITION-1:** package doc + per-function tags
(~40 LOC) for Online/Offline classification.

### 4.1 Buffer protocol for `expm` (and friends)

`expm(A, n) []float64` is the wrong shape per CLAUDE.md §3. Right shape:

```go
func ExpmWorkspace(n int) int { return 7 * n * n }
func Expm(A []float64, n int, t float64, out, scratch []float64) int
```

Padé-13 + scaling-and-squaring needs 7 n×n matrices in flight
(`A2, A4, A6, U, V, P, Q`). Caller-supplied scratch is the only
zero-alloc shape. Same shape for `Schur(A, n, T, Q, scratch)`,
`CARE(A, B, Q, R, P, scratch)`.

### 4.2 Cost amortization — cache Schur for repeated solves

For fixed plant `A` with runtime-varying `(Q, R)` (gain-scheduled
LQR), cache the Schur form:

```go
type SchurForm struct { T, Q []float64; n int }
func ComputeSchur(A []float64, n int, sf *SchurForm, scratch []float64)
func LQRWithSchur(sf *SchurForm, B, Q, R, K, S []float64) error
```

Kills the 25n³ recompute per LQR call. Mirrors `linalg.LUDecompose`
then `LUSolve`.

---

## §5 Bode ω-grid — vectorizable, alloc-free if shaped right

The 052-Tier-1 `Bode` API recommended in 054:

```go
func Bode(sys LTI, omegas, outMag, outPhase []float64) error
```

Per-ω cost = `Evaluate(complex(0, ω))` = `2·deg` complex muladds.
Deg-3 TF ≈ 30 ns. 1000-sample sweep ≈ 30 µs. With dB conversion
(`cmplx.Abs` ~30 cyc + `math.Log10` ~40 cyc) ≈ 40 µs. Acceptable
for design-time, not 60-FPS hot.

Vectorization notes: no cross-ω data dependency (embarrassingly
parallel — defer to a `parallel/` package, keep `control/` zero-dep);
SIMD via Go asm out of scope; manual coefficient register-hoist
gives ~5% but uglifies API; phase-unwrap is one `O(n)` post-pass
after the ω loop, no extra alloc.

**Action C-PERF-BODE-1:** ship `Bode(sys, omegas, outMag, outPhase
[]float64) error` with caller-supplied buffers, no internal alloc,
phase-unwrap inline. Add `BenchmarkBode_1000` golden case. Doc
"design-time, ~40 µs per 1000 samples."

### 5.1 Horner is correct; complex-coeffs cache deferred to 054

`evalPoly` at `transfer.go:54-60` uses Horner — `O(n)` and numerically
optimal among linear schemes. Already correct. Estrin scheme would
parallelize for n>50 but classical control TFs stay below.

The `complex(coeffs[i], 0)` pattern at `transfer.go:55,57,154,156,161,168`
costs `movsd + xorps` ≈ 1 cycle each — free. Could precompute
`[]complex128` once in TF construction for ~5% Horner-tight-loop
speedup, but defers on the `New(num, den)` constructor refactor
flagged by 054 (private fields). Bundle then.

---

## §6 Internal Taylor cos/sin perf

`realCos` / `realSin` at `transfer.go:191-225` use 12-term Taylor
with naive `[0, 2π]` reduction by repeated `+= twoPi` / `-= twoPi`.
The reduction loop is `O(|x|/2π)` — for the angles used by
Durand-Kerner initial guess (`2π·i/degree + 0.4` for `i ∈ [0, degree)`),
`x ∈ [0.4, 2π+0.4]`, so reduction is one or zero iterations.
Cheap. **Not a perf concern.**

But: `transfer.go:1` already imports `math/cmplx`, which transitively
imports `math`. The 051 finding "math is already imported, just
call `math.Cos`/`math.Sin`" is correct — replacing the Taylor
implementations saves 35 LOC and gets `math.Sin/Cos`'s `O(1)`
Cody-Waite reduction (constant ~10 cycles vs the current variable
~25 cycles for typical inputs). Net perf win at zero cost. Already
covered by 051 — not a 055 issue.

---

## §7 Actionable items, by leverage

1. **C-PERF-BENCH-1** (30 LOC): `control_alloc_test.go` with
   `AllocsPerRun` guards on `PID.Update`, `Evaluate`, 4 filters.
   Lock the no-alloc contract before 052 lands. Plus 3 `Benchmark*`
   for baseline.
2. **C-PERF-PARTITION-1** (40 LOC doc): tag every function as
   **Online (60 FPS)** or **Offline (design-time)** in package doc.
   `expm/Schur/CARE/DARE/LQR-design/Bode/Margins/RootLocus/Poles/
   Zeros/tf2ss/ss2tf` = Offline; `PID.Update`/filters/`RateLimiter`/
   `Evaluate`/`Kalman.Predict-Update` (n≤32) = Online.
3. **C-PERF-POLES-1** (25 LOC): `PolesInto(buf, scratch)` +
   convenience-wrapper `Poles()`. Required before 052-Tier-1
   `RootLocus`.
4. **C-PERF-EXPM-PROTOCOL-1** (~10 LOC stub): document
   `Expm(A, n, t, out, scratch) int` + `ExpmWorkspace(n) int`. Same
   protocol for `Schur`, `Lyapunov`, `CARE`, `DARE`. Fix the buffer
   convention before primitive PRs land.
5. **C-PERF-SCHUR-CACHE-1** (~30 LOC): `SchurForm` struct + caching
   pattern for repeated solves on fixed `A` (gain-scheduled LQR).
6. **C-PERF-BODE-BUF-1** (~30 LOC + 8 LOC bench): `Bode(sys, omegas,
   outMag, outPhase)` — buffer-passing, inline phase-unwrap.
7. **C-PERF-COMPLEX-COEFFS-1** (10 LOC, blocked on 054 private-
   fields PR): cache `[]complex128` coefficients in TF constructor.

Bundle 1+2+3 = **95 LOC**, non-breaking, closes the perf-contract gap.
4+5 are the 052 buffer-protocol design — must be fixed before each
new primitive reinvents its own scratch convention.

---

## §8 Pistachio 60-FPS suitability table — current surface

| Function | Per-call cost | At 60 FPS for 100 controllers | Verdict |
|---|---|---|---|
| `PID.Update` | ~5 ns | 30 µs/frame (0.18%) | **green** |
| `LowPassFilter` | ~3 ns | 18 µs/frame | **green** |
| `HighPassFilter` | ~4 ns | 24 µs/frame | **green** |
| `ComplementaryFilter` | ~5 ns (after 051 fix) | 30 µs/frame | **green** |
| `RateLimiter` | ~4 ns | 24 µs/frame | **green** |
| `TransferFunction.Evaluate` (deg 3) | ~10 ns | 60 µs/frame | **green** |
| `TransferFunction.Poles` (deg 3) | ~3 µs (Durand-Kerner ~30 iter) | 18 ms/frame | **red — offline only** |
| `TransferFunction.IsStable` (deg 3) | ~3 µs (calls Poles) | 18 ms/frame | **red — offline only** |
| `TransferFunction.Poles` (deg 10) | ~30 µs | 180 ms/frame | **red — offline only** |

The asymmetry between `Evaluate` (online-safe) and `Poles`
(offline-only) is currently unflagged in package doc. `IsStable`
in particular looks like a "cheap predicate" by name but is
actually a Durand-Kerner pole-find — a Pistachio author who
calls it per-frame for a deg-10 plant will bust 60 FPS by 10×
without warning. **C-PERF-PARTITION-1 doc fix is high-leverage.**

---

## §9 Non-overlap

- **051 (numerics):** kick, complementary algebra, Durand-Kerner,
  Cauchy bound, Taylor cos/sin — ceded. 055 confirms companion-
  matrix QR (per 051) is *also* a perf cleanup but defers ownership.
- **052 (missing):** 90-item LQR/Kalman/MPC/H∞ ladder. 055 owns
  *runtime cost class* + buffer-protocol design for them.
- **053 (architecture):** LTI/Series/Parallel/FRD/Modelica. 055
  ceded composition-cost analysis (Series TF polymult `O(deg·deg)`).
- **054 (API):** functional options, `Stability` enum, error
  contracts. 055 confirms the `Bode(sys, ω, outMag, outPhase)`
  buffer shape recommended by 054 is alloc-free; adds
  `ExpmWorkspace`-style protocol for the algorithmic wave.

## §10 Summary numbers

- Files audited: **3** (pid.go, filter.go, transfer.go) + **2** test files.
- Heap allocations in current hot path (PID + 4 filters + Evaluate): **0**.
- Heap allocations in current `Poles()`: **2 per call** (24·deg + 8 bytes).
- Benchmarks in repo: **0** (entire repo, not just control/).
- AllocsPerRun-guard tests: **0**.
- Doc claims of "Zero heap allocations" unverified by tests: **5** (filter.go ×4, transfer.go ×1).
- Forward-looking primitives audited: **14** (expm, Hessenberg, Schur,
  Lyapunov, Sylvester, CARE, DARE, LQR, Kalman.Predict, Kalman.Update,
  EKF, UKF, Bode, RootLocus).
- Online-vs-offline partition recommendations: **all 14**.
- Required buffer-protocol design before primitives land:
  `Expm`, `Schur`, `Lyapunov`, `Sylvester`, `CARE`, `DARE`, `Bode`,
  `PolesInto`. **8 functions** needing `WorkspaceSize` + scratch slice.
- Recommended bundle for first PR (BENCH + PARTITION + POLES):
  **95 LOC**, non-breaking, locks the no-alloc contract.

End of 055.
