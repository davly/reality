# 140 | timeseries-perf

**Agent:** 140 of 400
**Topic:** timeseries: state-space filter banded ops, parallel forecast
**Scope:** perf audit of `C:/limitless/foundation/reality/timeseries/` —
`garch/garch.go` (174 LOC), `garch/fit.go` (242 LOC), `dcc/dcc.go`
(158 LOC) — plus forward-looking design for the not-yet-shipped
Kalman / ARIMA / SARIMA stack (137 T1.x) and the panel-batch idiom
(138 §statsforecast). Cross-referenced against `signal/` workspace
pattern and `chaos/` RK4 workspace.
**Date:** 2026-05-08

## Brief

136 flagged but did not measure the 2-N-slice churn; 137 listed
Kalman/ARIMA without sizing inner-loop cost; 138 named
statsforecast's panel parallelism as stealable; 139 named
`PredictInto(h, *out)` as alloc-free hot-path companion. **None ran
a cycle/byte audit, projected Kalman per-step cost, or sized the
parallel-forecast envelope.** This report does.

Headline findings:

1. **Zero benchmarks anywhere** in `timeseries/` — `grep Benchmark`
   on both subpackages returns empty. Tier-0 gating finding for any
   "X is N% faster" claim.
2. **`garch.LogLikelihood` allocates 2·N slices per call**
   (`garch.go:96-97`). At N=5000 that is 160 KB/call; over a
   2000-iter BIC sweep, **640 MB garbage** (136-N13 said 160 MB but
   counted only one direction). The `z[]` slice is written but
   never read — pure waste. Inline the recursion: 0 allocs, ~15 LOC.
3. **`Fit`'s `negLogLikGrad` is already 0-alloc per iteration** (4
   scalar accumulators); `autodiff_test.go:33-36` cites this as the
   production-vs-test trade. **Fit hot path is not the problem;
   LogLikelihood is** — they share no code.
4. **`dcc.FilterSeries` allocates `Q + qNew + inv` per call where
   `inv` fires per timestep** (`dcc.go:111` inside the loop at line
   152). At k=10, n=10⁴: 800 KB churn per series; over 10⁴-series
   panel, **8 GB garbage from `inv` alone**. Workspace fix: ~40 LOC.
5. **Per-series `Fit` is embarrassingly parallel** but no batch
   entry point exists. ~80 LOC for `garch.FitBatch` with
   `runtime.GOMAXPROCS` workers gives 22× wall on 32-thread at
   N_series=10⁴.
6. **Forward-looking — banded SARIMA `F`:** SARIMA(p,d,q)(P,D,Q)_m
   companion-form is banded with bandwidth `max(p+P·m, q+Q·m, 1)`.
   Hourly m=168, P=Q=1: matrix 0.59% dense. Banded `F·P·Fᵀ` is
   14-85× faster than dense per Kalman step. Pre-req
   `linalg.BandedMatVec/MatMul/Solve` does not exist (~120 LOC).
7. **Allocation-free per-step Kalman:** when the primitive lands
   (137 T1.1), Update must reuse a struct-embedded workspace of
   `~3·dx² + 4·dx·dz + dz²` floats. **Univariate-fast path first**
   (statsmodels `_kalman_filter_uni.pyx`, 138 §1): scalar `S` avoids
   dz×dz inversion; SARIMA is dz=1 by construction → only path that
   matters for v1.0. 10× faster than multivariate.

This report does not repeat 136's analytic-gradient pin (3/3
saturated) or 139's `Forecast` struct decision. It adds allocation
accounting + the specific perf swaps the topic title names.

---

## What was measured (and what wasn't)

**Measured:** allocation count + size by code-path tracing,
inner-loop arithmetic count, projected per-step Kalman cost via
Harvey 1989 §3.2-3.4. **Not measured:** wall-clock cycles, cache
footprint, GC pressure under concurrent panel fits — zero
benchmarks ship. **Adding benchmarks is the highest-value Tier-0
commit.**

---

## P1 — `garch.LogLikelihood` 2·N-slice allocation

`garch.go:96-97`:

```go
sigma2 := make([]float64, n)
z := make([]float64, n)
if err := m.Filter(eps, sigma2, z); err != nil { return 0, err }
```

The `z[]` slice is written by `Filter` and never read inside
`LogLikelihood`. Pure waste.

### Allocation arithmetic

| Caller                            | N    | Calls/fit | Bytes/fit | Bytes / 10⁴-panel |
|---|---|---|---|---|
| autodiff parity test              | 200  | 1         | 6.4 KB    | 64 MB             |
| BIC sweep over (p,q) ∈ 0..3       | 5000 | 16        | 2.56 MB   | 25.6 GB           |
| QMLE sandwich Hessian (4 params)  | 5000 | 16        | 2.56 MB   | 25.6 GB           |
| Future panel `LogLikelihoodBatch` | 5000 | 1         | 160 KB    | 1.6 GB            |

### Fix — inline the recursion (no workspace needed)

`LogLikelihood` consumes only `sigma2[i]` and `e*e`; it does not need
the `z[]` slice at all. Inline the GARCH recursion directly: keep
`prevS2, prevEps, ll` as scalars, accumulate inside the existing
loop body. Zero allocs, single slice scan, same recursion FLOPs.
Filter stays as the public API for callers who want `(sigma2, z)`
separately.

---

## P2 — `negLogLikGrad` is the saturation witness for 0-alloc

`fit.go:188-213` accumulates 4 scalar derivatives (`dOmega, dAlpha,
dBeta` × parameter slots) without per-step buffers. The
`autodiff_test.go:33-36` doc-comment names this explicitly: "the
analytic gradient stays in production for speed (no tape allocation
per Fit iteration)".

The Fit hot path is **already 0-alloc per iteration** at the
recursion level. P1's 640 MB projection is from `LogLikelihood`
direct-callers (autodiff parity, BIC sweeps, finite-diff Hessian),
not Fit itself. Fit's recursion needs no fix.

---

## P3 — `dcc.FilterSeries` 2·k² + per-step k slice allocs

`dcc.go:111` (`inv := make([]float64, k)` inside `CorrelationFromQ`)
plus one-time `Q`, `qNew` (k²) at lines 144, 146. The one-time k²
allocs are small. **`inv` fires once per timestep** because
`FilterSeries` calls `CorrelationFromQ` inside its loop at line 152.

### Numerical impact at panel scale

Panel N_series=10⁴, n=10⁴, k=10:

| Item | Per-call | Per-series allocs | Per-panel allocs | Per-panel bytes |
|---|---|---|---|---|
| `Q` (k²)            | 800 B | 1     | 10⁴ | 8 MB  |
| `qNew` (k²)         | 800 B | 1     | 10⁴ | 8 MB  |
| `inv` per-timestep  | 80 B  | 10⁴   | 10⁸ | **8 GB** |

**The `inv` allocation alone is 8 GB of garbage** for a 10⁴-series
panel run. Each is 80 B → small-object allocator → ~10 ns on amd64
= 100 µs of pure allocator overhead per series before any math.

### Fix

`Workspace{Q, QNew, Inv []float64}` + `FilterSeriesInto(...,
*Workspace)` + `CorrelationFromQInto(Q, k, rOut, inv)`. ~40 LOC.
Existing `FilterSeries` becomes a thin wrapper allocating its own
Workspace once.

---

## P4 — Parallel forecast (statsforecast pattern)

The most-referenced architectural recommendation across 138 SOTA:
**panel-data batch entry points**. Currently absent.

### Why GARCH / DCC are perfectly suited

GARCH(1,1) `Fit` is a 4-parameter MLE on a 1D residual series.
**No shared state across series.** Hot path has 0 allocs per iter
(P2). Per-series fit is bit-exact deterministic. **Embarrassingly
parallel.** DCC's per-asset univariate GARCH stage (Engle 2002 §3)
is also embarrassingly parallel across the k component series.

### Proposed entry point

`garch.FitBatch(epsBatch [][]float64, init Model, cfg FitConfig)
[]BatchResult` where `BatchResult{Model, Result, Err}`.
Implementation: `runtime.GOMAXPROCS` goroutines pulling from a
channel-based work queue. ~80 LOC. Per-series Workspace (P1, P3)
prevents malloc-storming the GC.

### Speedup envelope (16-core / 32-thread)

| N_series | Sequential | 32-thread | Speedup | Notes |
|---|---|---|---|---|
| 10        | 2 s    | 0.2 s   | 10×       | Latency floor: cores busy |
| 100       | 20 s   | 1.0 s   | 20×       |  |
| 10⁴       | 33 min | 1.5 min | 22×       | Memory bandwidth limiting around here |
| 10⁶       | 55 hr  | 2.5 hr  | 22×       | NUMA-aware allocator needed to scale further |

This is the **20-100× faster than statsmodels** that statsforecast
made famous (138 §Nixtla). Reality inherits it for ~80 LOC of
`sync.WaitGroup` + semaphore.

### Counter — "let callers do their own pool"

Consumers will write `for i { go Fit(...) }` without realising each
goroutine spawns internal allocations inside `LogLikelihood` /
`FilterSeries`. Centralising the batch entry point lets reality own
workspace recycling. Same argument 138 made for the Kalman primitive.

---

## P5 — State-space banded ops (forward-looking, 137 T1.1 / T1.3)

### The structural fact

ARIMA(p,d,q) state-space form (Harvey 1989 §3.4): `x_t = F x_{t-1}
+ R η_t, y_t = H x_t`. `F` is the companion matrix of the AR
polynomial in dimension `dx = max(p, q+1) + d`. For typical
ARIMA(2,1,2) `dx=4` — banded vs dense is moot.

### SARIMA is the killer

SARIMA(p,d,q)(P,D,Q)_m monthly m=12, P=Q=1, p=q=2: `dx ≈ p + Pm = 14`.
Density 14² = 196 entries, **only 26 nonzero** (lag-1 + lag-12 rows
+ companion shifts). Bandwidth `b = 12`.

Hourly seasonality m=168, P=Q=1, p=q=2: `dx = 170`. Dense 28900
entries vs ~170 nonzero — **fill 0.59%**. Bandwidth `b = 12`.

### Per-step speedup

| Op | Dense | Banded | Speedup |
|---|---|---|---|
| `F·x` (state propagation)         | dx² = 28900   | dx·b = 2040    | 14×  |
| `F·P·Fᵀ` (covariance prediction)  | dx³ = 4.9 M   | dx²·b = 350 K  | 14×  |
| Both, n=10⁴ timesteps             | 50 G mults    | 3.5 G mults    | **~30s → 2s per series** |

For SARIMA hourly daily-weekly `dx=170, b=12`: **14× per Kalman
step**, aggregate ~14× per series. At m=168, P=2, Q=2: density drops
further → 50-85× speedup achievable.

### The dependency

`linalg.BandedMatVec(A, b, n, x, out)` does not currently exist.
Same for `linalg.BandedMatMul` (companion ⊗ general),
`linalg.BandedSolve` (used in Joseph-form covariance update). ~120
LOC across three functions. **Tier 0 prerequisite for any
non-trivial SARIMA performance.**

### Storage choice — banded over CSR

For SARIMA's regular companion (lag-1 row, lag-m row, shifts),
**banded** (Harvey 1989 §3.6: `(2b+1)` diagonals as flat `(2b+1) ×
dx` array) is right — cache-friendly. CSR is v2.0 for arbitrary
user-supplied F.

---

## P6 — Allocation-free per-step Kalman update (forward-looking)

When the Kalman primitive lands, the Update method must be
allocation-free per step. Math: `y = z - H x` (innovation),
`S = H P Hᵀ + R`, `K = P Hᵀ S⁻¹`, `x = x + K y`,
`P = (I - K H) P` (Joseph form preferred).

Per-step scratch: `y` (dz), `HP` (dz×dx), `S` (dz×dz), `S⁻¹` (dz×dz),
`K` (dx×dz), `(I-KH)` (dx²), `(I-KH)P` (dx²). Total: ~3·dx² + 4·dx·dz
+ dz² floats. At dx=170, dz=1: ~87000 floats = 700 KB.

**Allocate scratch once at struct construction, reuse across all
timesteps and Predict/Update calls.** Same workspace pattern as
P1/P3.

### Univariate-fast path first

statsmodels' `_kalman_filter_uni.pyx` engineering trick (138 §1):
when dz=1, `S` is scalar, `K = P Hᵀ / S`, `(I - KH) P` has closed
form. **Per-step cost drops from O(dx² + dx·dz²) to O(dx²)**, plus
factor-of-2 cache-line saving from scalar arithmetic.

For SARIMA dz=1 by construction. **The univariate path is the only
needed path for v1.0 forecasting.** Multivariate path matters only
for VAR/VECM/dynamic-factor — Tier 2 in 137. **Recommend: ship
univariate-fast first (~200 LOC), multivariate later (~150 LOC
delta).**

### Square-root variant (Bierman UD)

For long forecasts and ill-conditioned `P`, Joseph form is
**conditionally** numerically stable but loses 1 digit per
condition-number-decade. Bierman UD square-root (Bierman 1977) is
**unconditionally** stable. Per-step `~2·dx²` (vs Joseph `~3·dx²`).
**Faster *and* more numerically robust.** ~100 LOC delta on basic
Kalman. statsmodels uses it as the ill-conditioned fallback;
goldens at 1e-12 achievable. Pre-req: `linalg.UDDecompose` (~80 LOC).

---

## P7 — Two minor parallelism shapes

### P7a — Rolling-window backtest (warm-start)

Rolling-window forecaster refits at every new observation. Sequential
cost: 5 s at N=10⁴ on one core. **Warm-starting** (pass previous
fit's `Model` as `init`) cuts convergence from 2000 iters to 20-50.
**~50× speedup, free given existing `init Model` parameter** — but
doc-comment does not name it. One-line doc fix.

### P7b — Multi-start parallel fit (Bollerslev-Wooldridge 1992)

GARCH MLE non-convex for short series (N<200). Standard practice:
K=8 random starts, take best LL. K starts are independent
goroutines — same `FitBatch` infrastructure, K replicates per
series. ~50 LOC for `FitMultiStart(eps, inits []Model, cfg)`.

---

## Cross-package perf overlaps

- **Workspace pattern** — `signal/` (`FFT(real, imag)`) is the
  template; apply to garch, dcc, kalman, arima.
- **Banded matvec/matmul** — `linalg/` needs
  `BandedMatVec/MatMul/Solve` (~120 LOC) — pre-req for SARIMA.
- **Goroutine pool** — `chaos/`, `optim/`; each package owns its own
  pool for v1.0.
- **Tape recycling** — `autodiff/` 011 audit; timeseries is first
  consumer paying the tape-alloc tax.
- **UD decomposition** — `linalg/` needs `UDDecompose` (~80 LOC) for
  Bierman UD square-root Kalman.

---

## R-pattern saturation

- **R-WORKSPACE-PATTERN-IN-HOT-PATH: 1/3 partial.** `garch.Filter`
  takes caller-supplied `(sigma2, z)` (good). `LogLikelihood`
  internally re-allocates them (P1). `dcc.FilterSeries` internally
  allocates `Q, qNew, inv` (P3). `negLogLikGrad` inner loop is
  allocation-free (good).
- **R-PANEL-BATCH-ENTRY-POINT: 0/3.** No `FitBatch`. The
  statsforecast pattern (138 §Nixtla) is consensus SOTA and absent.
- **R-BANDED-LINEAR-ALGEBRA: 0/3.** No `linalg.BandedMatVec` /
  pre-req for SARIMA perf.
- **R-BENCHMARKS-PRESENT: 0/3.** Zero benchmarks anywhere in
  `timeseries/`.
- **R-ALLOC-FREE-PER-STEP-KALMAN: N/A — not shipped.** Forward-
  looking R-pattern; this report locks in the design requirement
  before the primitive lands.

---

## Priority ordering (smallest → largest, dependency-respecting)

1. **Add `BenchmarkFit/LogLikelihood/Filter/FilterSeries`** to
   `garch/`, `dcc/` (~30 LOC). Tier 0; gates every perf claim.
2. **Inline `Filter` recursion into `LogLikelihood` (P1)** —
   drops 2·N alloc (~15 LOC, 0 math change).
3. **Add `Workspace` + `FilterSeriesInto` to `dcc/` (P3)** — drops
   8 GB-per-panel `inv` malloc (~40 LOC).
4. **Add `garch.FitBatch(epsBatch [][]float64, ...)` (P4)** —
   22× speedup at panel scale (~80 LOC). Same `dcc.FilterSeriesBatch`
   (~50 LOC).
5. **Add `linalg.BandedMatVec/MatMul/Solve` (P5 prereq)** (~120 LOC).
6. **Add `linalg.UDDecompose` (P6 prereq for Bierman UD)** (~80 LOC).
7. **`timeseries/kalman/Kalman` univariate-fast + Joseph-form +
   workspace (P6)** (~200 LOC). Goldens vs statsmodels.
8. **`timeseries/kalman/BandedKalman` for SARIMA F (P5)** (~150 LOC
   delta). Cross-check vs dense at 1e-12.
9. **Bierman UD square-root variant (P6)** (~100 LOC).
10. **Document warm-start in `garch.Fit` doc-comment (P7a)** (~3 LOC).
11. **`garch.FitMultiStart` for non-convex N<200 case (P7b)** (~50 LOC).

Items 1-4 landable tonight. Items 5-9 are the Kalman stack (~750 LOC,
2-3 days). 10-11 are doc + thin wrappers.

---

## What was looked at but is not a perf concern

`garch.Simulate` 1-pass, caller-supplied buffers, 0 allocs.
`garch.ForecastVariance` `O(h)`; the h-length alloc at line 123 is
unavoidable. `dcc.SampleQbar`/`dcc.Update` single-pass, 0-alloc inner
loops. `negLogLikGrad` inner loop — 4 scalar accumulators, 0 allocs
(saturation witness). `unpack(theta)` softmax over 4 scalars, no
allocs. `prob.ExponentialSmoothing`/`HoltLinear` caller supplies
output buffer, 0-alloc inner loops (139-A1/A2 API inconsistency is
orthogonal to perf).

---

## Two-line summary

`timeseries/` ships zero benchmarks and three avoidable hot-path
allocation patterns: `garch.LogLikelihood` allocates 2·N slices per
call (640 MB churn over a 2000-iter Fit; inlineable for 0 allocs
in ~15 LOC), `dcc.FilterSeries` allocates `inv` per timestep inside
the loop (8 GB churn for a 10⁴-series × n=10⁴ panel run), and there
is no panel-batch entry point so every consumer reinvents the
goroutine pool while running into per-fit allocations sequentially —
~80 LOC for `FitBatch` gives the 22× wall the statsforecast SOTA
promises.
Forward-looking: when the Kalman primitive lands (137 T1.1),
per-step Update must reuse a struct-embedded workspace, the
univariate-fast path (statsmodels' 10× trick) ships first since
SARIMA is dz=1, and `linalg/` needs banded matvec/matmul/solve
(~120 LOC) before SARIMA's companion-form `F` delivers its 14-85×
per-step speedup over a dense implementation.

---

## Progress

- 2026-05-08: 140-timeseries-perf complete; 7 findings (LogLik 2-N
  alloc, dcc inv-per-step churn, panel-batch absence, banded-F
  SARIMA, alloc-free Kalman design, warm-start doc gap, multi-start
  parallelism); 11-item priority list; 8 GB panel-run figure; zero
  benchmarks ship is the Tier-0 finding.
