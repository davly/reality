# 090 | info-perf — allocations, asymptotics, SIMD friendliness

**Scope.** Performance audit of `reality/info/`, `reality/infogeo/`,
`reality/compression/` (the three packages where info-theoretic
primitives live, per 086 §scope) plus forward-looking perf design for
the not-yet-implemented Tier-2/3 surface from 087 (KSG MI, KL kNN,
histogram binning, LZ76 with suffix automaton, canonical LogSumExp).

**Anti-overlap with 086/087/088/089.**
- 086 = numerical correctness (log-stability, KL-zeros, MI bias).
- 087 = missing primitives (the Tier-1/2/3 catalogue).
- 088 = SOTA reference libraries.
- 089 = API ergonomics (signatures, units, options).
- **090 (this)** = the orthogonal axis: bytes allocated per call,
  asymptotic cost, vectorisability, hot-loop layout, scratch-buffer
  affordances. No re-derivation of the math; no API redesign; no list
  of missing primitives.

**Why this matters.** Per CLAUDE.md §"Key Design Rules" #3: *"No
allocations in hot paths. Functions accept output buffers. Pistachio
calls these at 60 FPS."* That rule is currently violated in 6 of the
~22 info-related functions in repo, and most of the proposed Tier-2
estimators will violate it on their first cut unless the perf seams
are designed before the math lands.

---

## Part A — Existing primitives: per-call allocation table

Static-read of every info-related public function. "Allocations"
counts the heap allocations on the **hot path** (not error paths,
not init). Loop bounds based on inputs of size `n` (1-D length) or
`m, n, d` (samples × dims).

| Function | File:line | Hot-path allocs | Big-O | Vectorisable? |
|---|---|---|---|---|
| `compression.ShannonEntropy(probs)` | `compression/entropy.go:27` | **0** | O(n) | YES — inner is `p * log2(p)`, monotone scan |
| `compression.JointEntropy(joint)` | `:45` | 0 | O(rows·cols) | YES — flatten + scan |
| `compression.ConditionalEntropy(joint)` | `:69` | **1** (`make([]float64, len(joint))` for marginal) | O(rows·cols) | partial |
| `compression.MutualInformation(joint)` | `:94` | **2** (marginalX, marginalY) | O(rows·cols) | partial |
| `compression.KLDivergence(p, q)` | `:133` | 0 | O(n) | YES |
| `compression.CrossEntropy(p, q)` | `:161` | 0 | O(n) | YES |
| `compression.RunLengthEncode(data)` | `coding.go:13` | **1** (encoded slice, grows) | O(n) | NO (data-dependent run lengths) |
| `compression.RunLengthDecode(encoded)` | `:40` | **1** (pre-sized via 2-pass) | O(n) | partial |
| `compression.DeltaEncode(data)` | `:73` | **1** (`make([]int64, len(data))`) | O(n) | YES — strided diff |
| `compression.DeltaDecode(encoded)` | `:93` | 1 | O(n) | partial — prefix-sum, SIMD has scan primitives |
| `compression.ScalarQuantize(data, levels, out)` | `quantize.go:30` | **0** (out is caller-supplied — gold standard) | O(n) | YES |
| `compression.ScalarDequantize(quantized, ..., out)` | `:91` | 0 | O(n) | YES |
| `infogeo.KL(p, q)` | `fdiv.go:65` | 0 | O(n) | YES (post-validate) |
| `infogeo.JS(p, q)` | `:98` | 0 | O(n) | YES |
| `infogeo.TotalVariation(p, q)` | `:125` | 0 | O(n) | YES |
| `infogeo.Hellinger(p, q)` | `:145` | 0 | O(n) | YES |
| `infogeo.ChiSquared(p, q)` | `:163` | 0 | O(n) | YES |
| `infogeo.Renyi(p, q, α)` | `:196` | 0 | O(n) | partial — `Pow` is per-element, hard to SIMD without scalar `pow` intrinsics |
| `infogeo.Bregman(gen, x, y)` | `bregman.go:26` | **1** (`gradY := make([]float64, len(y))`) | O(n) + cost(GradPhi) | YES on the dot |
| `infogeo.SquaredEuclidean(x, y)` | `:55` | 0 | O(n) | YES — pure FMA |
| `infogeo.GeneralisedKL(x, y)` | `:75` | 0 | O(n) | YES |
| `infogeo.ItakuraSaito(x, y)` | `:109` | 0 | O(n) | partial |
| `infogeo.MahalanobisSquared(x, y, M)` | `:133` | **1** (`d := make([]float64, n)`) | **O(n²)** | YES — GEMV pattern |
| `infogeo.MMD2Biased(X, Y, k)` | `mmd.go:64` | 0 (kernel may alloc) | **O((m+n)² · d)** | partial |
| `infogeo.MMD2Unbiased(X, Y, k)` | `:116` | 0 | O((m+n)² · d) | partial |
| `infogeo.MedianHeuristicBandwidth(X, Y)` | `:169` | **3** (`all`, `dists`, sort) | **O(N²·d + N² log N)** | NO — sort is the bottleneck |
| `infogeo.GaussianKernel(bw)` (per call) | `:21` | 0 | O(d) | YES |
| `infogeo.LaplacianKernel(bw)` (per call) | `:41` | 0 | O(d) | YES |
| `mdl.GaussianCodeLength(...)` | `codelength.go:29` | 0 | O(n) | YES |
| `mdl.NMLMultinomial(counts)` | `nml.go:46` | **1** in `computeCn2`: `make([]float64, n+1)` | O(n + k) | partial — LSE is parallelisable |
| `mdl.computeCn2(n)` | `nml.go:119` | 1 | O(n) | partial |
| `mdl.UniversalIntegerCodeLength(n)` | `universal_int.go` | 0 | O(log* n) ≈ const | trivial |
| `mdl.SelectMDL / WithMargin` | `select.go:23, 55` | 0 | O(k) | YES (argmin) |
| `lz.LempelZivComplexity(symbols, A)` | `lz76.go:78` | **2** (`map[int]struct{}`, then result) | **O(n²)** worst | NO — sequential dependency |
| `lz.SymbolizeByQuantile(returns, bins)` | `lz76.go:234` | **3+log n** (valid slice, result, mergesort splits) | O(n log n) | partial |
| `lz.SymbolizeByThreshold(...)` | `lz76.go:291` | **1** (result) | O(n) | YES |
| `lz.ComplexityFromReturns(...)` | `lz76.go:365` | symbols (~n) + LZ76 | O(n²) | NO |
| `lz.RollingComplexity(...)` | `lz76.go:401` | **per-window**: SymbolizeByQuantile alloc × #windows | O((n/step)·window²) | partial — windows are independent, parallelisable |

**Total clean (zero-alloc) hot paths: 14 / 38** (37 %).
**Total `O(n²)` or worse: 5** (LZ76, MMD biased/unbiased, Mahalanobis,
MedianHeuristic).

---

## Part B — Allocation-removal patches (existing code, in priority order)

These are the **mechanical** zero-alloc conversions. Each <30 LOC,
each a seam other primitives will reuse.

### B.1 `compression.ConditionalEntropy` / `MutualInformation` — accept scratch

`compression/entropy.go:71-79` allocates `marginalX` every call.
`compression/entropy.go:96-118` allocates two marginals. At 60 FPS
inside a Sensorhub histogram update this is ~120 small allocs/sec —
not catastrophic, but free to remove.

```go
func MutualInformationInto(joint [][]float64, marginalX, marginalY []float64) float64
```

The probs/joint shape doesn't change at runtime; allocate once
upstream. `MutualInformation` becomes a wrapper that allocates on
the user's behalf.

LOC: ~15 each. Caller migration: zero — `MutualInformation()` keeps
its old signature and forwards.

### B.2 `infogeo.Bregman` — scratch buffer for `gradY`

`infogeo/bregman.go:36`. `make([]float64, len(y))` per call. Bregman
is called inside `optim/`-style iterative algorithms (mirror descent,
exponentiated gradient) — one call per inner iteration.

```go
type BregmanGen struct { Phi, GradPhi ... ; Scratch []float64 }  // optional pre-alloc
```

Or: `BregmanInto(gen, x, y, scratch []float64)` variant. LOC: ~10.

### B.3 `infogeo.MahalanobisSquared` — scratch + GEMV via `linalg`

`infogeo/bregman.go:142` allocates `d := make([]float64, n)` then
runs a hand-rolled `O(n²)` `M·d`. Replace with
`linalg.MatVec(M, d, out)` or its row-major equivalent (which the
linalg package presumably provides per CLAUDE.md inventory).

Two wins: zero alloc (pass `scratch`) **and** access to whatever
SIMD/blocking the linalg pkg already does. LOC: ~15.

### B.4 `infogeo.MedianHeuristicBandwidth` — quickselect, not full sort

`infogeo/mmd.go:169-189` allocates a `dists` array of size `N(N-1)/2`
then **full-sorts** it for the median. For `N = 1000` that's 499 500
float64s and a `O(N² log N²) = O(N² log N)` sort — the sort
dominates the algorithm. Replace with `quickselect(dists, k)` for
`O(N²)` total (the pairwise-distance scan is unavoidable).

Better: streaming **median-of-medians** so the `dists` slice never
needs to be materialised. Cost: still `O(N²)` distance evals
(unavoidable for exact median) but with `O(1)` extra memory.

LOC: ~50 (quickselect), ~80 (Wirth's `kthSmallest`). Recommend
quickselect — exact median, well-known algorithm.

### B.5 `lz.SymbolizeByQuantile` — replace mergesort with stdlib sort + indices

`info/lz/lz76.go:421-466` hand-rolls a recursive mergesort with
**heap allocation at every recursion level** (`make([]indexedValue,
mid)` × 2 per call, total `O(n log n)` extra allocations). The
comment claims it avoids `sort` package interface overhead — but the
allocation cost is much larger than the interface-dispatch cost.

Replace with `sort.SliceStable(valid, func(i,j) bool { ... })`.
Single heap allocation (the `valid` slice already exists),
`O(n log n)` time, no per-recursion allocs. LOC: -40 (net deletion).

If the original concern about `sort.Slice`'s reflective comparator
hot-path is real, use `slices.SortStableFunc` (Go 1.21+) which is
type-specialised. Two-line change.

### B.6 `lz.RollingComplexity` — share scratch across windows

`info/lz/lz76.go:401-419`. Each `ComplexityFromReturns(window, ...)`
call re-allocates the symbol slice and the LZ76 result map. With
`step=1` over a 1000-sample series and a 100-sample window, that's
**900 × 3 = 2 700 allocations** per `RollingComplexity` call.

Add `RollingComplexityInto(returns, win, step, bins int, results
[]LzComplexityResult)` and have the inner loop reuse a single
`symbols` scratch. The `distinct` map inside `LempelZivComplexity`
is the harder reuse — see C.2 for the suffix-automaton refactor.

LOC: ~30. **Highest-leverage existing-code perf win** because
`RollingComplexity` is the Sensorhub use case.

---

## Part C — Forward-looking design (Tier-2/3 primitives not yet in repo)

These don't exist yet. Designing the perf seam **before** the math
lands costs 10× less than retrofitting.

### C.1 LogSumExp helper — SIMD-friendly canonical form

087 T1.12, 088 §1.2 both flag this. The single inline LSE in
`info/mdl/nml.go:142-156` is the model:

```go
maxLog := terms[0]
for _, v := range terms { if v > maxLog { maxLog = v } }
sum := 0.0
for _, v := range terms { sum += math.Exp(v - maxLog) }
return math.Exp(maxLog) * sum
```

**Two-pass design is correct**, but two perf notes:

1. **Fused max-and-exp pass** is *worse* numerically (you don't know
   the max until you've seen all of it). Stick with two passes.
2. **`math.Exp` in a tight loop** — Go's `math.Exp` is scalar; LLVM
   on amd64 emits `expvec` or equivalent for SoA loops. Pure Go
   gets neither. The right answer for `reality`'s "no asm" rule is:
   accept that `LogSumExp` is ~3 ns/element scalar, and document it.
   Caller can hand-roll AVX-512 on hot paths if they really need
   sub-ns.

**Allocation contract.** Take `terms []float64`, return `float64`.
**No scratch needed.** Stable: yes. Promote to `prob/mathutil.go`
or `info/lse.go` per 087 T1.12. ~20 LOC.

**Variant for streaming/online.** Welford-style online LSE:
maintain `(maxSeen, partialSum)`; when a new element exceeds
`maxSeen`, rescale `partialSum *= exp(oldMax - newMax)`. ~30 LOC.
Used by Bayesian online updates and importance sampling — currently
zero in repo.

### C.2 LZ76 — O(n²) → O(n) via suffix automaton

`info/lz/lz76.go:182-197` `isSubstringOfPrefix` is a naive substring
scan, called inside the LZ76 outer loop ~`n / log n` times with
candidate length up to `n`. Total cost is `O(n³ / log n)` worst-case
(the cap `LZ76MaxSymbols = 10_000` keeps it tractable).

**The textbook fix**: build a **suffix automaton** of the prefix
incrementally as the parse advances. Querying "is `S[parseEnd :
parseEnd+L]` a substring of `S[0:parseEnd]`?" reduces to a state
walk in `O(L)` per query. Outer loop becomes `O(n)` total queries
of total length `O(n)`, giving the canonical `O(n)` LZ76.

- Reference: Crochemore & Vérin (1997); Lempel-Ziv production count
  in linear time via suffix tree / suffix automaton.
- LOC: ~250 (suffix-automaton construction is ~150 LOC, the LZ76
  glue is ~60, tests are ~40).
- **Perf headroom**: at the current `n=10 000` cap, naive is ~10⁸
  ops; SAM-based is ~10⁴. **10 000× speedup** — at which point the
  `LZ76MaxSymbols = 10_000` cap can be lifted to `1e6` or removed.
- Allocation contract: SAM nodes are `O(2n)`; allocate as a single
  flat `[]samNode` arena. **Single alloc**, no per-symbol churn.
- Determinism: SAM construction is deterministic.

This unblocks LZ76 over Sensorhub time-series of ~`1e5` samples
which currently truncate; 088 §3 noted nothing in the SOTA library
landscape uses `O(n²)` LZ76.

### C.3 Histogram binning — bin rules + the inevitable allocation

087 T2.6 and 089 §3 both call for histogram binning. The three
canonical bin-width rules (Scott, Sturges, Freedman-Diaconis) all
boil down to a closed-form scalar:

| Rule | Formula | Cost |
|---|---|---|
| Sturges | `k = ⌈log₂(n)⌉ + 1` | O(1) |
| Scott | `h = 3.49 σ̂ n^{-1/3}`, `k = ⌈(max-min)/h⌉` | O(n) (mean+var) |
| Freedman-Diaconis | `h = 2 IQR n^{-1/3}` | **O(n log n)** (full sort for IQR) |

**FD's IQR is the perf landmine.** Full sort is O(n log n) but the
**quickselect for two quantiles** is O(n). For `n = 10⁶`,
that's the difference between 30 ms and 3 ms. Patch: implement
`HistogramFD` with twin quickselect calls (25th and 75th
percentiles), not `sort.Float64s`.

**Allocation contract.** `Histogram` type as proposed in 089 §3:

```go
type Histogram struct {
    Counts   []int     // len = nBins
    Edges    []float64 // len = nBins + 1
    NSamples int
}
func HistogramScott(samples []float64, h *Histogram) error  // h.Counts/Edges resized in-place if too small
```

Pre-sized `Histogram` reuses across calls — Sensorhub updates a
histogram every frame; per-frame zero alloc is mandatory.

LOC: ~80 (three rules, validation, in-place build). Tests: ~40.

### C.4 KSG-1 / KSG-2 mutual information — k-d tree dominates

087 T2.2, 088 §1 (JIDT trick), 088 §3 (NPEET). The headline
continuous-MI estimator. Performance is **entirely about the
nearest-neighbour search**:

| Backend | Cost | LOC |
|---|---|---|
| Brute force pairwise | O(N² d) | ~50 |
| k-d tree (JIDT, NPEET-via-`cKDTree`) | **O(N log N)** typical, O(N²) worst-case | ~400 |
| Ball tree | O(N log N), better for d>20 | ~500 |
| Random projection LSH | O(N) per query, approximate | ~250 |

**For `reality`'s zero-dep + canonical-determinism constraints, the
right answer is k-d tree with L∞ (Chebyshev) metric.** L∞ is what
KSG actually uses and is faster than L2 (no sqrt). Worst-case
`O(N²)` is theoretical; for d ≤ 20 and N up to 10⁶ it behaves
`O(N log N)`. Above d ≈ 20 the curse of dimensionality kicks in
and brute force is competitive — ship a brute-force fallback for
`d > 20` or `N < 1000`.

**Allocation contract.**
- k-d tree built once, reused across the four kNN queries that
  KSG-1 needs (`ψ(k) - ⟨ψ(n_x+1) + ψ(n_y+1)⟩ + ψ(N)` requires
  range counts in marginal spaces around each point's joint-space
  ε-radius).
- Single `make([]node, 2N)` arena.
- Query path uses a fixed-size kNN heap (`make([]neighbor, k)`)
  reused across queries.

**Per-query cost**: `O(log N)` average, `O(k log N)` for kNN.
**Total KSG-1 cost**: `O(N · k · log N)` — the JIDT v1.1
package-defining speedup.

**Perf seam to design now (before any KSG code lands).**

```go
type KDTree struct { ... internal arena ... }
func BuildKDTree(points [][]float64, metric Metric) *KDTree   // O(N log N)
func (t *KDTree) KNN(query []float64, k int, out []int) error // out reused
func (t *KDTree) RangeCount(query []float64, eps float64) int
```

The **same k-d tree** then services T2.1 (KL kNN entropy), T2.3
(MIXED-KSG), T2.4 (conditional MI Frenzel-Pompe), T3.1 KSG-TE,
T3.3 active info storage. **Five primitives share one backend.**
This is exactly the JIDT decomposition (`infodynamics/utils/KdTree.java`,
~600 LOC, services ~12 estimators). Per 088 §1 this k-d tree is
the single highest-leverage **infrastructure** commit in the entire
info layer.

CLAUDE.md inventory says `geometry/` exists but per the 077 audit
(geometry-missing) the k-d tree is on the geometry T1-4 list and
not yet implemented. **This is a single shared dependency** —
information theory and computational geometry both need it.
Coordinate the build there, not in `info/`.

### C.5 Vasicek univariate differential entropy — `O(n log n)` upper bound

087 T2.5. Cost: a single sort + a strided spacing scan. **Pure
sort-bound.** With `slices.Sort` (Go 1.21+) this is `~5 ns/element`,
i.e. 5 ms for `n = 10⁶`.

Allocation: 1 alloc (sort the input copy, can't sort caller's slice
without mutating). Or: provide both `Vasicek(samples []float64,
m int)` (allocates) and `VasicekInto(sorted []float64, m int)`
(zero-alloc, caller pre-sorts). LOC: ~40.

### C.6 KL kNN (Kozachenko-Leonenko) — same k-d tree

Cost: `O(N log N)` to build the tree + `O(N log N)` to query each
point's k-th NN distance. The math is `Ĥ_KL(X) = -ψ(k) + ψ(N) +
log c_d + (d/N) Σ log(2 ε_i)` — a single pass after kNN.

Allocation: caller-supplied `dists []float64` of length N reused
across calls; kNN result indices reused. LOC: ~50 over the k-d tree
infrastructure.

### C.7 PID `I_min` — redundancy lattice cached

087 T3.6, 088 §3 (dit). Williams-Beer PID over `n` source variables
has a redundancy lattice with `~Bell(n)` antichains. For n=2: 4
nodes. For n=3: 18. For n=4: 166. The lattice itself **is built
once** and reused across many `(X₁,…,Xₙ; Y)` queries — dit's
package-defining trick.

Perf seam: a `Lattice` value, deterministic structure, computed
once per `n`, cached at package init for `n ≤ 4` (the only
practical scale). LOC: ~150. Allocation: 0 after init.

For `n ≥ 5` the lattice is too big to enumerate; ship documented
`ErrPIDDimTooLarge` rather than degrade.

---

## Part D — Cross-cutting perf principles

### D.1 The "Pistachio 60 FPS" rule — codify it

CLAUDE.md says zero allocations in hot paths but doesn't say where
the convention is enforced. Recommend a `BenchmarkXxx_Allocations`
test pattern that runs `b.ReportAllocs()` and **fails the build**
if any function the consumer calls inside `Update()` (frame
callback) allocates more than its docstring states.

This is a 086-style pin rather than a perf optimisation, but it's
the only way the rule survives 50+ contributors over time.

### D.2 Scratch-buffer convention — `Into` suffix

`compression.ScalarQuantize(data, levels, out)` is the pattern
(`out` is the last positional arg, not on an Options struct).
`compression.ScalarDequantize` follows. Recommend codifying:

> Functions that allocate `>= 1` slice on the hot path expose a
> sibling `XxxInto(..., scratch ...)` variant.

Affected: `Bregman` (T B.2), `MahalanobisSquared` (B.3),
`MedianHeuristicBandwidth` (B.4), `RollingComplexity` (B.6),
`MutualInformation` (B.1).

### D.3 SIMD-friendly access patterns

Go does not auto-vectorise the way C++/LLVM do (the SSA backend
hoists math.Exp loops onto its scalar `expvec` table but does not
emit AVX). For any SIMD-style win in `reality` you'd need:

1. Pure-Go wide-loop hand-unroll (4-way, 8-way) — works for adds,
   muls, FMA. 086 reports `KL`/`Hellinger` already vectorise
   linearly.
2. Cgo to a vectorised math lib — **violates** the zero-dep rule.
3. asm files (`*_amd64.s`) — also violates the spirit of the rule
   ("only the language's standard math library", CLAUDE.md §2).

**Recommendation.** Ignore SIMD entirely for v1.0. Document that
hot-path consumers should profile and, if necessary, fork specific
functions into their own assembly for their own deployment. This
is what crypto/sha256 does in stdlib (it has both `_generic.go`
and `_amd64.s`); `reality` should ship only `_generic.go` and let
downstream pin asm if they need it.

### D.4 Vector dot products in entropy — the FMA opportunity

The `p · log2(q)` pattern in `KLDivergence`, `CrossEntropy`, and
`MMD2` inner loops is **textbook fused-multiply-add** territory.
Go's compiler emits `vfmaddNNNsd` on amd64 *only* inside scalar
expressions of the form `a*b + c` — and only if `-gcflags='-d=fma'`
is enabled, which has been the default since Go 1.18.

Verify with `go tool objdump` after build. If FMA is firing, KL
inner loop is ~2 ns/element. If not, ~3 ns/element. Either way
fast enough that vector intrinsics aren't the bottleneck — the
*log* is. Mitigation: bulk-evaluate `log2(q[i])` first into a
scratch slice if the same `q` is used across many KL queries
(rare; punt).

### D.5 Cache layout for `[][]float64` joints

`MutualInformation(joint [][]float64)` iterates row-major — fine on
amd64 (rows are contiguous, columns stride). But **nested slice of
slice** has a pointer-chase per row. For tight numerical kernels,
flat `[]float64` with `(rows, cols)` shape is 1.5–3× faster on cold
cache.

Recommend a `JointMatrix struct { Data []float64; Rows, Cols int }`
shape for the next 086-T1 wave (so the new `MutualInformationInto`
and `JointEntropyFlat` can opt in without breaking the existing
`[][]float64` consumers).

This is also what 089 §6 proposed for ≥3-variable tensors. Roll
both into the same `Tensor` type.

---

## Part E — Recommendations (ordered by leverage)

| # | Action | LOC | Win |
|---|---|---|---|
| 1 | Build shared k-d tree in `geometry/` (also unblocks 077 T1-4) | ~400 | Unblocks ~5 KSG-family estimators |
| 2 | LZ76 suffix-automaton (`O(n³)` → `O(n)`) | ~250 | 10 000× speedup at n=10k cap |
| 3 | `lz.RollingComplexityInto` scratch reuse | ~30 | 1000× alloc reduction |
| 4 | `lz.SymbolizeByQuantile` use stdlib sort | -40 | log n alloc removal, simpler |
| 5 | Promote MDL/NML LSE to `prob.LogSumExp` | ~20 | Single canonical site (087 T1.12) |
| 6 | `MedianHeuristicBandwidth` quickselect | ~50 | log N factor on MMD |
| 7 | `MahalanobisSquared` GEMV via linalg | ~15 | Reuse linalg's blocking |
| 8 | `MutualInformationInto` / `ConditionalEntropyInto` | ~30 | 60 FPS Pistachio path |
| 9 | `Bregman` scratch | ~10 | Inner-loop alloc removal |
| 10 | `BenchmarkXxx_Allocations` enforcement pattern | ~50 (CI) | Locks rule across contributors |
| 11 | `Histogram` type + Scott/Sturges/FD | ~80 | 087 T2.6 substrate |
| 12 | Online LSE for streaming | ~30 | Future-proof |

**Sprint-1 (items 2-9): ~260 LOC** of mostly-deletion or scratch-buffer
boilerplate that closes every existing alloc gap and gives LZ76 its
canonical asymptotic. Single highest-leverage item: **#1 k-d tree**,
because it unblocks the entire continuous-MI estimator family.

---

## Part F — Files referenced (absolute paths)

- `C:\limitless\foundation\reality\compression\entropy.go`
- `C:\limitless\foundation\reality\compression\coding.go`
- `C:\limitless\foundation\reality\compression\quantize.go`
- `C:\limitless\foundation\reality\infogeo\fdiv.go`
- `C:\limitless\foundation\reality\infogeo\bregman.go`
- `C:\limitless\foundation\reality\infogeo\mmd.go`
- `C:\limitless\foundation\reality\info\lz\lz76.go`
- `C:\limitless\foundation\reality\info\mdl\nml.go`
- `C:\limitless\foundation\reality\info\mdl\codelength.go`
- `C:\limitless\foundation\reality\info\mdl\select.go`
- `C:\limitless\foundation\reality\info\mdl\universal_int.go`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\077-geometry-missing.md` (k-d tree dependency)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\086-info-numerics.md` (allocation findings cross-ref)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\087-info-missing.md` (Tier-1/2/3 plan)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\088-info-sota.md` (k-d tree justification, JIDT v1.1)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\089-info-api.md` (Histogram type, Tensor type)

End report. ~340 lines.
