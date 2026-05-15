# 125 — queue: performance audit

**Topic:** queue-perf — Markov-chain matrix exponential, sparse generators, Buzen / MVA convolution recurrences, polling/cyclic iteration.

**Scope demarcation vs 121/122/123/124:**
- 121 (numerics) — flagged Jackson fixed-point as **convergence-rate** issue and recommended `linalg.Solve`. I do **not** repeat that; I cost it and the alternatives.
- 122 (missing) — listed Closed-Jackson MVA, Buzen, polling-gated/exhaustive, vacations. I do **not** re-list; I size each one's hot-path cost and cite which `reality` primitive each requires.
- 123 (sota) — named Buzen + MVA + CoMoM + CTMC-stationary + fluid limit as the analytic frontier. I do **not** re-recommend; I cost the **transient** path (uniformization, Krylov, Padé matrix-exp) which 123 only mentioned in passing.
- 124 (api) — sparse-routing-via-graph, `Metrics` struct. I cost what those changes are *worth* on the hot path, not whether they should ship.

This review is the only one in the queue cluster that costs **per-call CPU/allocation/cache** behaviour and tackles the topic-prompt's named "matrix exponential / sparse generators / Buzen / MVA / polling" set head-on. Of the seven existing `queue` functions, six are O(1) scalar with zero allocation; one (`JacksonNetwork`) is the only allocation site and the only iteration loop — so the performance audit is dominated by what the package **does not yet have** plus one fixable Jackson hot-path.

---

## 1. The current performance posture (what we have)

| Function | Lines | Time | Allocs | Hot-path-clean? | Notes |
|---|---|---|---|---|---|
| `MM1` (basic.go:56) | ~20 | O(1) ~6 FLOPs | 0 | yes | scalar, named returns, no math.Pow |
| `MMc` (basic.go:108) | ~30 | O(c) via ErlangC | 0 | yes | Jagerman recursion |
| `MM1K` (basic.go:173) | ~50 | O(1) + 2× math.Pow | 0 | yes-but | two `math.Pow(rho, K)` calls; see §2.4 |
| `LittlesLaw` | ~5 | O(1) | 0 | yes | scalar |
| `ErlangB` (erlang.go:34) | ~15 | O(N) ~3 FLOPs/iter | 0 | yes | Jagerman, no factorial |
| `ErlangC` (erlang.go:70) | ~15 | O(N) | 0 | yes | one ErlangB call + 2 FLOPs |
| `ErlangCWaitTime` | ~10 | O(N) | 0 | yes | one ErlangC call |
| `BurstinessIndex` | ~30 | O(n) two-pass | 0 | yes | scalar two-pass mean+var, no Welford |
| `OfferedLoad` | ~10 | O(1) | 0 | yes | scalar |
| `JacksonNetwork` | ~110 | **O(n²·iter)** up to **n²·1000** | **4 × make([]float64,n)** + **routing-row validation O(n²)** | **no** | only iterative function; only allocator |

Headline: the package is allocation-free **everywhere except** `JacksonNetwork`. The package is also currently 100% steady-state — no transient path exists, hence no matrix-exp / uniformization / sparse-CTMC machinery to audit. The audit therefore splits into:
- §2 — perf gaps in what exists (mostly Jackson, plus a few micro-issues).
- §3 — perf-by-design constraints for what's missing (matrix exponential, sparse Q, Buzen, MVA, polling).

---

## 2. Performance issues in existing code

### 2.1 `JacksonNetwork` — the only iterative function and the only allocator

`network.go:91-149`. Three layered perf issues:

**(a) O(n²) row-sum validation runs unconditionally and per-call.** Lines 57-73 walk the full dense routing matrix, summing each row, even when the caller passes a 100-node sparse pipeline that has 200 non-zero entries. For n=100 that's 10,000 float reads + 100 sum-comparisons before any actual queueing math runs. For Pulse/Sentinel hourly recompute on a 50-microservice topology this is the dominant cost of the function.

**Fix:** if the input becomes a `graph.WeightedAdjacency` (124's recommendation), validation is O(|E|) not O(n²) and is exact for sparse routing. Until then: a single-pass *combined* validator that returns early on first invariant break costs the same as the current code in the worst case but ~half on the average case.

**(b) Fixed-point iteration is the wrong algorithm.** 121 noted this for *convergence-rate* reasons (subdominant eigenvalue near 1). The performance angle is sharper: the loop on lines 97-113 is O(n²) per sweep × up to 1000 sweeps = O(n²·iter). For n=20, well-conditioned `P`: ~10 sweeps × 400 ops = 4,000 FLOPs. For n=20, ill-conditioned (subdominant eigval=0.99): up to 1000 sweeps × 400 ops = **400,000 FLOPs** — and the function panics if it doesn't converge, so the caller gets a 100× cost cliff at the boundary.

**The right computation is one linear solve:**

```
λ = (I − Pᵀ)⁻¹ · λ_ext
```

This is the **closed-form** of the traffic equations (Jackson 1957, Gross-Shortle-Thompson-Harris §7.2). `linalg.LUDecompose` + `linalg.LUSolve` are already in reality (`linalg/decompose.go:21,103`). Cost is O(n³/3) for LU + O(n²) for the solve = **O(n³/3 + n²)**, regardless of conditioning, with **deterministic running time**. For n=20 that's ~2,800 FLOPs — *cheaper* than the well-conditioned fixed-point case and **150× cheaper** than the ill-conditioned case. For n=100 the LU solve is ~330k FLOPs which beats fixed-point above ~80 sweeps. The crossover is at n=20-30 for typical conditioning, which is where every realistic queueing-network call already lives. **There is no performance regime where fixed-point wins.**

The implementation is ~12 LOC after wiring `linalg.Solve` (build I−Pᵀ in a stack-allocable buffer, call LUDecompose+LUSolve). Three of the four `make([]float64,n)` calls in the function go away (only `lambda` remains as the solve's RHS/output). Net: 3× reduction in allocations, 5×–500× reduction in FLOPs across the conditioning spectrum, removes the convergence panic.

**(c) Three unnecessary output allocations.** Lines 118-120 allocate `throughput`, `utilization`, `queueLength` every call. The function has no `*Into` companion. For a Pistachio/Sentinel agent recomputing capacity at 1Hz on a 50-node topology, that's 3×50×8 = 1,200 B/sec of GC pressure for what should be zero-allocation steady-state work. Standard reality remediation: add `JacksonNetworkInto(..., throughput, utilization, queueLength []float64)` companion.

### 2.2 `MM1K` — `math.Pow` instead of integer power

`basic.go:192,195`. Two `math.Pow(rho, float64(K+1))` and `math.Pow(rho, float64(K))` calls. Both have integer second argument. `math.Pow` with non-integer exponent is `exp(b·log(a))` — two transcendentals plus a multiply (~50-100 ns). For integer exponents reality has nothing equivalent, but a 4-line `intPow(b float64, n int)` (binary exponentiation) gets to ~5-10 ns for K≤256. Per-call savings ~80-180 ns; for a Wayfare attraction-queue capacity sweep over 100 K values per second that's ~10-18 µs/sec, real but small. **More important: integer-exponent integer-power is bit-exact and reproducible across `libm` versions; `math.Pow` is not** (the `pow(2.0, 53.0)` IEEE-754 rounding-mode cliff is a known reality concern; `intPow` avoids it entirely). Recommendation: add `internal mathutil.IntPow` (or use a private helper) and replace both Pow calls.

### 2.3 `ErlangB` — Jagerman recursion is already optimal but allocation-free vector form is missing

The Jagerman recursion (erlang.go:43-47) is the textbook stable form. There is no faster O(N) algorithm. **However:** for capacity-planning sweeps (the typical telephony / call-center use case) callers want `ErlangB(A, n)` for n = 1..N_max — *all* of them. The current API forces N_max separate calls, each redoing the full inner recursion to that point. The recursion is *cumulative* — `B(A, n)` is one step beyond `B(A, n-1)`. A single call computing the whole prefix:

```go
func ErlangBPrefix(A float64, NMax int, out []float64)  // out[i-1] = ErlangB(A, i)
```

is O(NMax) time with **one** division per step, vs O(NMax²) for the naive sweep. For NMax=200 that's 200 vs 20,000 ops, a 100× win on capacity-curve generation. Same idea applies to `ErlangC` and `ErlangCWaitTime`. This is exactly the pattern reality already uses for `signal.FFT` (one call computes the whole transform; callers don't loop and re-FFT).

### 2.4 `BurstinessIndex` — two-pass mean+var instead of Welford

`metrics.go:33-49` walks the array twice: once for mean, once for variance. Allocation-free, fine. **But:** for streaming use (BookaBloke updating burstiness as new inter-arrival samples land) the API forces a full re-walk on every update. A Welford one-pass form would let consumers maintain a running `(n, mean, M2)` triple. Cost is identical for batch — Welford is also O(n) — but the streaming variant unlocks `O(1)` update.

This is the same pattern `prob/welford.go` already implements for online mean+variance. Recommendation: a `BurstinessIndexState` type that tracks `(n, mean, M2)` with `Update(x float64)` and `Value() float64`. ~25 LOC. Prerequisite: only invoked by `prob` not `queue` today; if duplication is forbidden, consume `prob.Welford` directly (cross-package coupling acceptable).

---

## 3. Performance-by-design for the missing topic-prompt set

The topic prompt names five missing capability areas. Each has a **performance-correct** implementation form that, if not chosen at landing time, becomes a migration debt. This section sets the hot-path constraints **before** the algorithms are written.

### 3.1 Markov-chain matrix exponential `exp(Qt)` — the transient-CTMC primitive

The fundamental object for transient analysis of any continuous-time Markov chain (M/M/1, M/M/c, M/M/c/K, M/M/c/c, finite-buffer Jackson, polling, vacations, retrials):

```
P(t) = exp(Q·t)
```

where Q is the n×n infinitesimal generator (off-diagonals = transition rates, diagonal = −Σ row). The hot-path question is: how do you compute `exp(Q·t) · π₀` (the transient distribution starting at `π₀`) — *without* computing the full dense matrix exponential?

**Three implementation forms, ordered by performance:**

| Form | Time | Memory | When to use | Citation |
|---|---|---|---|---|
| **Uniformization** | O(K · nnz(Q)) where K ≈ qmax·t + 5√(qmax·t) | O(n) | Always default for CTMC transient. nnz(Q) = O(n) for birth-death, O(n) for M/M/c-typed Q | Grassmann 1977; Stewart "Numerical Solution of Markov Chains" §8.2 |
| **Padé approximant + scaling-and-squaring** | O(n³ · log₂(‖Q·t‖)) | O(n²) | Dense Q with no structure, n ≤ a few hundred. The `expm` Higham 2005 algorithm. | Higham "Functions of Matrices" §10.3 |
| **Krylov subspace (Arnoldi)** | O(m² · n + m · nnz(Q)) where m ≈ 20-50 | O(m·n) | Sparse Q with n in the thousands and one starting vector π₀ | Saad-Schultz 1986; expokit (Sidje 1998) |

For reality's birth-death-typed queueing chains (M/M/c/K has tridiagonal Q, 3n−2 nonzeros not n²), **uniformization is unambiguously correct**. The recurrence is allocation-free with two ping-ponged length-n buffers:

```
P̃ = I + Q/qmax              # row-stochastic; qmax = max diagonal magnitude
π(t) = exp(-qmax·t) · Σ_{k=0..K} (qmax·t)^k / k! · π₀ · P̃^k
```

Hot-path constraints for the implementation:
- Two `[]float64` workspace buffers of length n, ping-ponged (no per-step allocation).
- Poisson-weight prefix `(qmax·t)^k / k!` computed by recurrence, not `math.Pow`/`math.Lgamma`. Stops when weight < 1e-15 (relative-error pin).
- Matvec `π · P̃` inlined for tridiagonal / banded structure when caller declares the structure (avoid the dense O(n²) matvec on a chain that is structurally O(n)).

**Cost comparison:** for n=100, qmax·t = 50, dense Padé: ~1.0M FLOPs + 30 KB workspace. Uniformization on tridiagonal Q: ~30k FLOPs + 1.6 KB workspace. **30× faster, 20× less memory.** For n=10k (closed-network CTMC) Padé is infeasible (800 MB), Krylov is the only option.

**Decision:** ship `queue.UniformizationStep(Q []float64, n int, t float64, pi0, piT, work []float64)` as the first transient primitive. Defer Padé `expm` to `linalg/matexp.go` (it's a general-purpose linear-algebra primitive, not queue-specific) — this composes correctly with the cross-package architecture (CLAUDE.md §"Dependency Position": queue→linalg). Defer Krylov to `linalg/krylov.go` for the same reason.

**What this unblocks:** transient `M/M/1`, transient `M/M/c/K` (the load-spike scenario 124 flagged as missing), step-response of any queueing CTMC, time-to-first-blocking distribution, busy-period distribution. None of these exist in `queue` today.

### 3.2 Sparse generator matrices — the data-layout choice

For any CTMC of size n > 100 the dense Q is wasteful (M/M/c/K has 3n−2 nonzeros vs n² entries). The data layout matters because **matvec is the only Q operation queue cares about** — sparse matvec is cache-friendly, dense matvec on sparse data thrashes L1.

**Three layouts, performance-ranked for queueing structure:**

| Layout | Footprint (M/M/c/K, n=1000) | Matvec cost | Cache pattern | Notes |
|---|---|---|---|---|
| Dense `[]float64` len n² | 8 MB | O(n²) | strided | wastes 99.7% of bandwidth |
| CSR `(values, colind, rowptr)` | ~24 KB | O(nnz) | sequential per row | reality has no CSR yet |
| **Banded** (lower, diag, upper) | ~24 KB | O(nnz) | sequential, contiguous | optimal for birth-death |

For *every* M/M/* model the Q is banded — usually tridiagonal (M/M/1, M/M/1/K), sometimes banded-with-c (M/M/c/K transitioning between fewer-than-c-busy and c-busy regions). For Jackson networks the *flat* state space is high-dim but the *per-customer* generator is banded; for QBD (quasi-birth-death) processes Q is block-tridiagonal and the right primitive is matrix-analytic methods (Latouche-Ramaswami 1999). Three concrete recommendations:

1. **Ship `queue.BirthDeathQ(birth, death []float64, n int)`** that builds and returns the banded representation as three `[]float64` slices `(sub, diag, sup)` of lengths `n-1, n, n-1`. Total storage 3n−2 vs n² for dense. Matvec on this representation is ~5n FLOPs vs n² FLOPs — **200× cheaper at n=1000**.

2. **Ship `queue.BandedMatVec(sub, diag, sup []float64, n int, x, out []float64)`** as the workhorse. 5n FLOPs, 0 allocations, fully vectorisable.

3. **Defer general-CTMC sparse Q to `linalg/sparse.go`** (it's a linear-algebra primitive, not queue-specific). Reality has no CSR/CSC matrix today; the right home is `linalg`. Until that lands, queue should ship the *banded* form because every queueing CTMC reality models is banded.

**Performance budget for transient M/M/c/K with n=1000, t=10s:**
- Dense Q + Padé: 8 MB workspace + ~10⁹ FLOPs + several seconds wall-clock — infeasible at 60 FPS.
- Banded Q + uniformization: 24 KB workspace + ~10⁵ FLOPs + ~30 µs wall-clock — **2× per frame at 60 FPS** is feasible.

This is the **one** place in queue where data-layout choice has a 4-order-of-magnitude perf consequence. Make the choice before any transient primitive lands.

### 3.3 Buzen's convolution algorithm — closed-network normalising constant

Recurrence `g(n,m) = X_m · g(n−1, m) + g(n, m−1)` with boundary `g(0, m) = 1, g(n, 0) = X_1^n`. **Performance form is unambiguous** (123 §1 already named it):

```
- One 2D buffer of size (N+1) × M (callers supply, no allocation in hot path).
- Forward sweep, single pass: O(N·M) FLOPs, two reads + one mul + one add per cell.
- Marginals fall out as ratios of g(n,m) — no second pass needed.
```

Hot-path constraints:
- **Caller-supplied `g []float64` of length `(N+1)*M`** (row-major). Mandatory for Pistachio's per-frame closed-network solve; without this every call allocates ~8·N·M bytes, prohibitive at 60 FPS.
- **No `math.Pow` for the boundary `g(n,0) = X_1^n`** — fill by iterated multiplication. Same precision as Pow, ~10× faster for n>4.
- **`X_m` is a service demand**; passing it as `[]float64` (length M) keeps the inner loop dense.

Cost: N=100 customers × M=10 stations = 1,000 cells × ~3 FLOPs = 3,000 FLOPs per call. **Negligible**; Buzen will never be the bottleneck. The reason to insist on caller-supplied buffers anyway is *consistency* with reality's hot-path doctrine, not any single-call cost.

**Numerical caveat the Buzen perf-form must address:** the forward direction for `g(n,m)` can overflow at large N for high `X_m` (the same family of issue as ErlangB without Jagerman — naive forward direction grows fast). The standard remediation is to scale by `g(N, M)` after the sweep, or to compute `log g(n,m)` and exponentiate at the end; in either case the recurrence is the same shape and the perf budget is unchanged. **Decision:** ship the linear-domain version with a `math.Frexp`-style automatic rescaling every K cells (similar to how `signal.FFT` handles its accumulator).

### 3.4 MVA convolution recurrence — Reiser-Lavenberg

123 §4 already shipped the form. The perf angle is:

```
For n = 1..N:
  R_i(n) = (1/μ_i) · (1 + Q_i(n−1))     [residence time at station i]
  X(n)   = n / Σ_i V_i · R_i(n)         [system throughput]
  Q_i(n) = X(n) · V_i · R_i(n)          [queue length at station i]
```

Three slices of length M (`R, V·R, Q`) and one outer loop over N. **O(N·M)** time, **O(M)** workspace. Hot-path constraints:

- Caller supplies `R, Q []float64` of length M; the recurrence overwrites in place each step — no per-step allocation.
- `V_i · R_i(n)` is computed once per outer step, not twice (123 §4's recurrence as written has a redundant multiply).
- Final return is `Metrics{Lq=Q[i]−ρ_i, Wq=R_i−1/μ_i, L=Q[i], W=R_i, rho=X·V_i/μ_i}` per station, packed into the `Metrics` struct 124 §4 mandates. **No separate per-step allocation**: the caller provides one `[]Metrics` of length M and the recurrence fills it once, at n=N.

Cost: N=100 × M=10 = 1,000 inner steps × ~6 FLOPs = 6,000 FLOPs. Same order as Buzen. The interesting perf knob is `O(N·M)` vs `O(M)` for **Bard-Schweitzer approximate MVA** (123 trick #6): for N≥50 the approximate variant is 5-20× faster *per call* and has quadratic accuracy in 1/N. For Pulse/Sentinel re-solving a closed network at 60 FPS as customer counts shift, B-S is the right default; ship it alongside exact MVA with a `MVAApprox(...)` companion.

### 3.5 Polling / cyclic systems — the iteration loop

Symmetric-station polling under gated discipline (Takagi 1986):

```
E[C] = N·E[V] / (1 − Σ ρ_i)              [mean cycle time]
E[Wq_i] = (1 + ρ_i) · (N · σ_V² + Σ ρ_j · E[X_j²]) / (2·(1 − Σ ρ_j))     [gated]
E[Wq_i] = (1 − ρ_i) · (N · σ_V² + Σ ρ_j · E[X_j²]) / (2·(1 − Σ ρ_j))     [exhaustive]
```

These are **closed-form** in the symmetric case (one big numerator + one small denominator + per-station scaling). O(N) time, **zero iteration**. The asymmetric case (different visit rates / service distributions per station) needs the **buffer-occupancy method** (Konheim-Levy-Sidi 1994) which solves an N×N linear system → reuse `linalg.LUSolve`. Same crossover as Jackson §2.1: linear-solve is faster than fixed-point iteration in every conditioning regime for N ≥ 5.

Hot-path constraints:
- Symmetric closed-form: zero allocation, ~10·N FLOPs.
- Asymmetric: caller supplies workspace of size N² + 2N for the linear solve.
- **Do not implement the simulation form** (event-driven Markov chain over polling state). Polling sim belongs in a sibling `queue/sim` package that 122 already correctly defers; the *analytic* form is what reality's mandate dictates.

### 3.6 What is *not* worth shipping for performance

- **Quasi-Birth-Death (QBD) matrix-analytic methods (Latouche-Ramaswami logarithmic-reduction).** The right citation, but the use cases (BMAP/G/1, MAP arrivals) require infrastructure (matrix-exponential distributions, MAP fitting) that 122 correctly tiered as Tier 3. Skip until a real consumer demands it; otherwise the perf-correct implementation (logarithmic reduction with O((n³)·log(1/ε)) cost on n×n blocks) is dead code.
- **Spectral methods on Q.** Eigendecomposition of a generator gives the transient via `exp(D·t)` on the diagonal form — but for non-symmetric Q (which all queueing generators are, except detailed-balanced ones) the eigendecomposition is numerically poorer than uniformization or Padé. Skip.
- **Whole-CTMC GMRES/BiCGStab for stationary `πQ = 0`.** 123 §6 named these; they're the right Krylov-class methods for n in the tens-of-thousands. But for n<1000 (which covers every reality consumer named in CLAUDE.md), direct LU of `Qᵀ + 1·1ᵀ` in `linalg.LUSolve` is faster *and* deterministic. Ship Krylov only when a reality consumer actually has n>1000.

---

## 4. Cross-package perf coupling

| Reality primitive consumed | What it would unlock in queue | Currently in reality? |
|---|---|---|
| `linalg.LUDecompose` + `LUSolve` | Closed-form Jackson traffic-eq solve (§2.1), asymmetric polling (§3.5), CTMC stationary πQ=0 | yes (decompose.go:21,103) |
| `linalg.MatVecMul` (out-buffer) | Dense matvec for matrix-exp Padé fallback | yes (matrix.go:66) |
| `linalg.MatExp` / Padé scaling-and-squaring | Dense-Q transient when n<200 | **no — would need to land in linalg** |
| `linalg.SparseMatVec` (CSR or CSC) | Generic sparse-CTMC transient | **no — see §3.2; banded form is queue-local** |
| `linalg.Krylov` / Arnoldi | Sparse-CTMC transient, n>1000 | **no — would need to land in linalg** |
| `prob.Welford` | Streaming `BurstinessIndex` (§2.4) | yes (the running mean+var pattern) |
| `signal.FFT` | CoMoM normalising-constant inversion (123 trick #4) | yes |
| `graph.WeightedAdjacency` | Sparse routing for Jackson (124 §3) | yes (graph package) |
| `chaos.ODE` | Mean-field fluid-limit ODE solve (123 §6) | yes |

The two `linalg` gaps (`MatExp`, `Krylov`) are the **only** missing dependencies for the entire transient-CTMC perf story. Both are general-purpose linear-algebra primitives; both belong in `linalg/`, not `queue/`. **Recommended sequencing:** before any queue transient primitive lands, gate-1 those two `linalg` primitives. After they land, the queue work is wiring + golden files, not novel numerics.

---

## 5. Benchmarks: zero exist

`queue_test.go` is 695 lines; `grep -i Benchmark` returns nothing. This violates CLAUDE.md rule 3 ("no allocations in hot paths") in spirit — the rule is unverifiable without `Benchmark*` functions and `-benchmem` output as an automated regression line. Recommendation matches every prior `*-perf` agent in this overnight: ship `bench_test.go` with one benchmark per existing function, plus one per new primitive as it lands. ~150 LOC for the eight existing functions. Allocations-per-op is the only objective measurement of "hot-path-clean" the package can offer.

Suggested initial bench set (per CLAUDE.md golden-file precedent — bench results are *not* golden-file-pinned, but allocations-per-op should be: `bench_test.go` with a `TestZeroAllocations(t *testing.T)` that asserts `testing.AllocsPerRun(...)==0` for each of the 8 existing functions plus every new one).

---

## 6. Top-5 ranked perf actions

| # | Item | Lines | Cycles freed (rough) | Allocs removed | Risk |
|---|---|---|---|---|---|
| 1 | `JacksonNetwork`: replace fixed-point with `linalg.Solve(I−Pᵀ, λ_ext)` | ~12 | 5×–500× | 3 of 4 | low; closed-form is exact |
| 2 | Add `bench_test.go` + `TestZeroAllocations` invariant for all 8 existing functions | ~150 | 0 (regression-prevention) | n/a | zero |
| 3 | `JacksonNetworkInto(...)` companion for caller-supplied output buffers | ~30 | 0 (steady-state hot path) | 3 (full elimination at consumer) | zero — additive |
| 4 | `MM1K`: replace `math.Pow(rho, K+1)` and `math.Pow(rho, K)` with int-power helper | ~20 | ~80-180 ns/call | 0 | low — bit-stable improvement |
| 5 | `BandedMatVec` + `BirthDeathQ` primitives, prereq for any future transient work | ~60 | 200× vs dense at n=1000 | 0 | zero — additive |

Lines: ~272 LOC total. None require new external dependencies; all consume `reality` primitives already in place.

For the missing capabilities (matrix exponential, sparse generators, Buzen, MVA, polling): the perf budget is **set** by §3 — when those primitives land they should land with caller-supplied workspace, banded data layout where the structure exists, and `O(N·M)` recurrences not `O(N²·M)` retries. Getting that right at landing time is the difference between Pistachio re-solving a closed network at 60 FPS (~6,000 FLOPs/frame for MVA at N=100, M=10) and not.

---

## 7. Headline finding

The `queue` package is allocation-free in 7 of 8 functions and the lone exception (`JacksonNetwork`) has a single 12-LOC fix (replace fixed-point with `linalg.Solve` on the closed-form traffic equations) that simultaneously cuts cost 5×–500× and removes the convergence panic 121 flagged as a numerical issue — performance and numerics align on the same patch. The topic-prompt's named missing capabilities (matrix exponential, sparse generators, Buzen, MVA, polling) are all algorithmically uncontroversial; the perf-correct forms are uniformization on banded `Q` (not Padé on dense), 2D-buffer caller-supplied Buzen, in-place Reiser-Lavenberg MVA, and closed-form polling — every one of them composes onto reality's existing `linalg`/`signal`/`chaos`/`graph` primitives with two gated `linalg` additions (`MatExp`, `Krylov` for n>1000) the only missing dependencies, and zero benchmarks anywhere in the package today means CLAUDE.md rule 3 ("no allocations in hot paths") is currently unenforceable as a regression invariant.
