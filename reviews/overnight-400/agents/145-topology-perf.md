# 145 | topology-perf — column algorithms, clearing, twist, allocation, sparsity

**Scope.** Perf audit of the matrix-reduction core in
`topology/persistent/`. Distinct from 141 (numerics), 142 (missing),
143 (SOTA libs), 144 (API). This report measures the column-add
hot loop, allocation amortisation, sparse representation, wedge-vs-
cohomology orientation, and the TAOCP combinatorial-number-system
indexing 143 flagged as the highest-leverage refactor.

**Files inspected.** `topology/persistent/barcode.go` (280 LOC, the
entire reduction pipeline lives here); `vr.go` (216 LOC, filtration
build); `bottleneck.go` (271 LOC, perf-relevant in candidate-pool
loop and adjacency-list build). **No benchmarks ship anywhere in
the repo** (`grep Benchmark ./topology` = zero hits) — call this F.0.

**Yardsticks.** Ripser 1.2.1 column-add inner loop (~60 ns/step,
hand-tuned C++). PHAT 1.5 heap-column variant (2-5× over flat-vector
XOR per BKR 2014 Tab 2). Bauer 2021 apparent pairs (~80% pairs
eliminated before any reduction). Reality v0.10.0 has none.

---

## 0. Headline scorecard

| # | Concern | LOC | Speedup | Alloc reduction |
|---|---|---:|---:|---:|
| F.0 | Add `BenchmarkComputeBarcode` to repo | 30 | (measure) | (measure) |
| F.1 | Replace `map[string]int` with TAOCP combinatorial-number-system index | ~150 | 5–10× | -100% |
| F.2 | Rewrite `symDiff` to write-in-place into preallocated buffer | ~25 | 1.5–2× | -50% per add |
| F.3 | Twist (BKR 2014): process dim 2 first | ~50 | 5–30× | (small) |
| F.4 | Clearing (Chen-Kerber 2011): skip columns whose pivot is known | ~40 | 2–5× stacked | (small) |
| F.5 | Switch to **persistent cohomology** (Morozov 2011, Ripser) | ~250 | 10–100× | (small) |
| F.6 | Heap-column XOR (PHAT BKR 2014) | ~80 | 2–5× per add | (small) |
| F.7 | Apparent-pairs predicate (Bauer 2021) | ~150 | 5–20× stacked | (small) |
| F.8 | Emergent-pairs short-circuit (Bauer 2021) | ~80 | 2–5× stacked | (small) |
| F.9 | Flatten distance matrix `[][]float64` → `[]float64` | ~10 | 1.1–1.3× | -n header allocs |
| F.10 | Triangle loop: hoist `dij` + early exit at edge clip | ~10 | 1.2× | none |
| F.11 | `boundaryColumn` per-call scratch buffers | ~15 | 1.3× | -3 allocs/edge |
| F.12 | Bottleneck adjacency rebuild per binary-search step | ~60 | 5-30× | -O(log C × m) |
| F.13 | `epsBottleneck` hoist out of inner edge-add loop | ~3 | 1.05× | none |

**Cumulative ceiling once F.1 + F.2 + F.5 + F.7 + F.8 land:**
~250-3000× over the textbook `barcode.go:60-164`, ~1100 LOC total,
math-stdlib only. Lifts Phase-A practical n from 50 → ~500 at
maxDim=1, ~200 at maxDim=2.

**Single most under-appreciated trick** for this codebase: **F.1,
the TAOCP index.** It single-handedly drops every `make([]byte, ...)`
in the hot path, removes the `map[string]int` lookup tax, and
**replaces the post-reduction sort with a comparison on uint64
keys** — the determinism dance dissolves.

---

## 1. The column-add hot loop today

`barcode.go:90-102` is the reduction core. `symDiff`
(`barcode.go:254-280`) **unconditionally allocates a fresh `[]int`
of size `len(a)+len(b)` per call**. On a Klein-bottle filtration
with m ~10^4 simplices and ~5 column-adds per killer, that is
~5·10^4 fresh heap allocations during reduction; each is GC-traced
(`cols[j] = col` reassigns; previous slice unreferenced).

**F.2 patch — scratch buffer hand-off:** allocate
`scratch := make([]int, 0, 2*m)` once, replace `col = symDiff(col,
cols[jp])` with `scratch = symDiffInto(scratch[:0], col, cols[jp]);
col, scratch = scratch, col`. Single buffer reused for the entire
reduction; at most one realloc when column size grows past the
high-water mark. **Allocations ~5·10^4 → O(log m).**

**F.6 alternative (heap column).** PHAT BKR 2014 Tab 2: heap-of-
row-indices is 2-5× over flat XOR because identical rows from
different columns merge in O(log) instead of being walked twice.
For sparse VR right; for dense maxDim=2 flat XOR wins on cache.
**Recommendation:** flat XOR with F.2 scratch for v0.11; heap
column optional after Ripser-port.

---

## 2. F.1 — TAOCP combinatorial-number-system index

**Where it bites today.** `barcode.go:74-77` builds a
`map[string]int` keyed by `simplexKey(s)` (`barcode.go:170-195`),
which **allocates a `[]byte` and materialises a `string` per call**.
Called once per simplex during index build (m calls) plus (k+1)
times per column inside `boundaryColumn` (`barcode.go:237`). With
m ~10^4 and k=1,2: **~3·10^4 string allocations during boundary
build alone**. Each key avg ~6 bytes for a 2-simplex on n=50 →
~24 B header × 3·10^4 = **~720 KB GC'd garbage per build.**

**The fix.** Knuth TAOCP 4A §7.2.1.3: a k-simplex on n vertices
identified by sorted vertex tuple `(v_0 < … < v_k)` maps to:
`idx(σ) = Σ_{i=0}^{k} C(v_i, i+1)`. Properties:
- **Bijection** between sorted k-simplices on n and integers in
  `[0, C(n, k+1))`. For n=50, maxDim=2: C(50,3)=19600; uint16
  suffices.
- **Faces are O(k) integer arithmetic from idx,** no enumeration.
- **Coface enum is O(n) with no allocation.**
- **Sort key is the integer itself.** Post-reduction sort at
  `barcode.go:153-161` becomes a no-op.

**Patch (~120 LOC).** New `simplex_index.go` with
`combinatorialIndex{n, maxDim, binomial [][]uint64}` and methods
`Encode(s) uint64`, `Decode(idx, k, out)`, `FaceIndex(idx, k, skip)
uint64`. Table memory n=500, maxDim=2 ≈ 16 KB, amortised once.

`barcode.go:74-77` becomes a dense `[]int` instead of a map.
`boundaryColumn` becomes allocation-free: take parent `idx`,
call `ci.FaceIndex(idx, k, skip)` (k+1) times, look up
`indexBy[faceIdx]`.

**Net: the entire `simplexKey` / `appendInt` / `make([]byte, ...)`
apparatus (`barcode.go:170-216`, ~50 LOC) deletes.** This is
the single highest-leverage perf refactor. 141 (numerics issue 2)
and 143 (PR #1) point at this; perf audit confirms with allocation
math.

---

## 3. F.3 — Twist (Bauer-Kerber-Reininghaus 2014)

**Why ELZ left-to-right is wasteful.** ELZ reduces every column.
**Twist processes columns in decreasing dimension order:** after a
dim-2 column j reduces to non-empty with pivot row r (a dim-1 edge
index), edge r is already known to be "negative" (it dies a class).
When the dim-1 pass hits column r, **skip it** — pair already
known. On a 12-pt Gaussian cloud maxDim=1 (~30 edges + ~20
triangles): dim-2 marks ~15 edges as negative; dim-1 pass skips
them. **30-50% reduction in column-adds.** For maxDim=2 (142
§T1.2), twist is mandatory: without it, n=50 maxDim=2 goes from
"fast" to "multi-second".

**Patch (~50 LOC).** Wrap reduction in two passes, high-dim first;
maintain `killed[m]int` (init `-1`), where `dimStart[]` is
precomputable once at filtration-build (filtration is already
sorted by dim within tied times). Inside dim-d loop:
`if killed[j] >= 0 { continue }`; reduction uses `killed[low]`
instead of the separate `pivotCol` map. Stacked with F.1+F.2:
1.5-3× on dim ≤ 1; 5-30× on dim ≤ 2.

---

## 4. F.4 — Clearing (Chen-Kerber 2011)

**Dual of twist.** A column whose simplex was already paired (as
the *pivot* of a higher-dim killer) does not need to be reduced.
Inside the twist loop:
```go
if killed[j] >= 0 { continue }   // bar already emitted by higher-dim
```
Skips both **column read** and **column reduction**. 2-5× on dim ≤ 2
stacked over twist; negligible at dim ≤ 1.

---

## 5. F.5 — Cohomology vs homology (wedge-vs-Morozov)

**Reality today is wedge-style** (ELZ 2000): boundary D, reduce
L→R. **Ripser is cohomology** (Morozov 2011, dSMV-J 2011):
coboundary D^T, reduce R→L. For a k-simplex in m simplices,
boundary nnz/col is k+1; coboundary nnz/col is O(n-k) cofaces —
but **most don't intersect in reduction**, giving naturally sparse
columns at every step. Ripser → 10-100× on Klein-bottle 2k.

**Mathematical guarantee.** dSMV-J 2011 Thm 4.1: persistence
diagram of homology of D = persistence diagram of cohomology of
D. **Bars match exactly**; consumer-visible behaviour unchanged.

**Why F.5 above F.6 (heap col).** Heap col is 2-5× constant
factor on a slow algorithm. Cohomology is 10-100× algorithmic
factor that **also makes columns sparser**, so heap matters less —
PHAT's heap-on-cohomology is ~2-3× on top.

**Patch (~250 LOC).** New `cobarcode.go`: `coboundaryColumn(s, ci,
indexBy, out)` enumerates cofaces (O(n) per simplex), encodes via
TAOCP. Reduction is mirror-symmetric: process simplices in
**reverse** filtration order, pivot is the **largest** row,
column-add when higher-index column's pivot collides. Pair
extraction same. 142 §T1.3 verbatim; 143 §2 quantifies the win.

**Recommendation: F.5 + F.1 + F.2 ship together as one PR.**
~525 LOC, brings reality from ELZ-textbook to cohomology+TAOCP —
algorithmic Ripser-parity modulo F.7/F.8.

---

## 6. F.7 + F.8 — apparent and emergent pairs

**Apparent pairs (Bauer 2021).** A pair (σ, τ) is *apparent* iff:
- τ is the smallest coface of σ in filtration order, **and**
- σ is the largest face of τ in filtration order.

Detectable in **O(d) per simplex** with no matrix work. Apparent
pairs contribute zero-persistence bars; **skip both simplices in
the reduction**. On dense VR ~80% of pairs are apparent (Bauer
2021 Tab 1) — what makes Ripser feel "magic" vs GUDHI on the same
algorithm.

**Patch (~150 LOC).** Single predicate over `(σ, ci, indexBy,
times)`: enumerate σ's cofaces, find smallest by filtration time;
enumerate that coface's faces, find largest by filtration time;
return `(τ, sigmaEqual(sigma, largestFace))`. Filtration-time
lookup is `times[indexBy[ci.Encode(s)]]` — O(1).

**Emergent pairs (Bauer 2021).** Finer: a pair is emergent if its
pivot row appears as a pivot the **first** time it is checked.
Subsumes apparent but slightly cheaper to detect (one walk vs
walk+face enum). Stacked: 5-20× over cohomology + TAOCP.

---

## 7. F.9 — Distance-matrix layout

`vr.go:197-216` allocates `[][]float64`: **n+1 heap allocs + an
extra slice header per row.** At n=500: 501 allocs + 12 KB
fragmentation.

**F.9 patch (~10 LOC).** Flat `make([]float64, n*n)`. Inner-loop
access `dist[i*n+j]` is branchless and cache-friendly. **1.1-1.3×
cache speedup, -n allocations.**

**F.10 prune.** `vr.go:127` computes `math.Max(...)` for every
(i,j,k) triple even when (i,j) was already clipped. Hoist `dij`
to the j-loop with `if dij > maxRadius { continue }`: ~30% of
triangle iterations skipped on typical clouds. **~10 LOC, 1.2×
on triangle build.**

---

## 8. F.11 — `boundaryColumn` per-call alloc

`barcode.go:221-248` allocates two slices per call: `rows` and
`face` scratch. At n=50: m_edges ≤ 1225 + m_triangles ≤ 19600 →
**~42k tiny allocations during boundary build**, each ~16-32 B.
GC tax for the build alone.

**F.11 patch (~15 LOC).** Pass `boundaryScratch{rows, face}` into
the function; reset between calls. **~42k allocs → O(log m).**

Compose with F.1: `face` becomes irrelevant because face indices
come from `ci.FaceIndex(parent, k, skip)` without materialising
the vertex set. **F.11 subsumed by F.1; document but don't ship
separately.**

---

## 9. F.12 — Bottleneck adjacency rebuild every binary-search step

`bottleneck.go:184-244` is called **once per binary-search step**.
Each call: `adj := make([][]int, left)` (≈O(m) rows); per (i,j)
`adj[i] = append(...)` (amortised but allocates on first); `matchL,
matchR, visited` allocs (each O(m)); per outer-vertex u, `visited
:= make([]bool, right)`. **At m=30: ~900 cands × ~30 visited × ~10
iters = 270 visited allocs × 60 B = 16 KB garbage per call.**

**F.12 patch (~60 LOC).** (a) **Build adjacency once, parameterise
on delta.** Edge `(a[i], b[j])` is in graph for all `delta ≥
linfDistance(a[i], b[j])`. Sort edges by threshold once; binary
search becomes "include all edges with threshold ≤ cands[mid]".
(b) **Reuse `matchL/matchR/visited` across iterations** via
in-place reset. KMN 2017 Algorithm 2 prescribes this; current Go
ignores it. **5-30× per `BottleneckDistance`, most GC removed.**

---

## 10. F.13 — `epsBottleneck` hoist

`bottleneck.go:200`:
```go
if linfDistance(a[i], b[j]) <= delta+epsBottleneck { ... }
```

`delta` is a function arg; Go can't prove it loop-invariant from
inside the inner loop. Hoist:
```go
deltaPlus := delta + epsBottleneck
for j := 0; j < lb; j++ {
    if linfDistance(a[i], b[j]) <= deltaPlus { ... }
}
```
**1.05× on `hasPerfectMatching`.** Free win. (141 §6 flagged the
*correctness* angle — abs eps should be relative; this hoist
applies to whichever value chosen.)

---

## 11. F.0 — benchmark scaffolding

The package ships **zero `*Benchmark*` functions** anywhere in
`topology/`. Every speedup claim above is back-of-envelope from
algorithmic analysis + cited Ripser/PHAT/Bauer numbers. **Reality
cannot self-validate any of them.**

**F.0 patch (~30 LOC).** Add `topology/persistent/benchmark_test.go`
with `BenchmarkComputeBarcode_Hexagon`, `_Random12`, `_Random50`,
`BenchmarkVietorisRipsComplex_50`, `BenchmarkBottleneckDistance_30Bars`.
Establishes baseline before F.1–F.12. **First PR on the perf
roadmap, before F.1.**

---

## 12. Wedge-style vs cohomology summary

| Aspect | Wedge (today) | Cohomology (Ripser) |
|---|---|---|
| Boundary nnz/col on VR | k+1 | O(n-k) coboundary |
| Reduction column-add count | O(m^2) worst | ~O(m) typical |
| Apparent pairs detectable | No | Yes |
| Emergent pairs detectable | No | Yes |
| Diagram correctness | dSMV-J 2011 identical | dSMV-J 2011 identical |
| LOC to ship from current state | (baseline) | ~250 + F.1 |

**The sequence.** F.0 (benchmark) → F.1 (TAOCP) → F.2 (scratch) →
F.5 (cohomology) → F.7 (apparent) → F.8 (emergent) → F.3+F.4
(twist+clearing for completeness on maxDim ≥ 2) → F.6 (heap col,
optional) → F.9+F.10+F.12 (filtration build + bottleneck cleanups).
**~1100 LOC total, single roadmap, math-stdlib only, no API break.**

This audit's opinion on **what to ship first**: F.0 + F.1 + F.5
together, ~430 LOC, single PR. F.0 lets us measure; F.1 is
prerequisite for F.5; F.5 buys the algorithmic 10-100×.

---

## 13. Closing scorecard

| Pillar | Today | After perf roadmap |
|---|---|---|
| Reduction algorithm | wedge ELZ-textbook | cohomology + apparent + emergent + twist + clearing |
| Simplex indexing | `map[string]int` (allocates per face) | TAOCP combinatorial-number uint64 |
| Column-add allocation | fresh slice per `symDiff` (5·10^4/build) | scratch hand-off (~log m) |
| Coface enum | string-key + map lookup | O(1) integer arithmetic |
| Distance matrix | `[][]float64` (n+1 allocs) | flat `[]float64` (1 alloc) |
| Bottleneck binary search | rebuild adj per step | sorted edge list, reuse buffers |
| Benchmarks | none | `BenchmarkComputeBarcode_*` + bottleneck |
| Phase-A practical n | ~50 (timeout > 50) | ~500 at maxDim=1, ~200 at maxDim=2 |

**Highest-leverage finding for ≤ 200 LOC of work:** F.0 + F.1 +
F.2. F.1 alone removes ~42k+ small heap allocs per build; F.2
removes ~5·10^4 per reduction; F.0 lets us measure.

**Highest-leverage finding without LOC ceiling:** F.5 (cohomology),
gating on F.1. The algorithmic order-of-magnitude. Everything else
is constant factors.

**Stacked ceiling:** ~1000-3000× over current ELZ textbook on
Phase-A fixtures, ~1100 LOC, single PR sequence, math-stdlib only.

---

## Summary (2 lines)

The persistent-homology reduction core is wedge-style ELZ textbook
with `map[string]int` simplex indexing that allocates per face
lookup and per column-add; the path to Ripser-parity is ~1100 LOC
dominated by F.0 benchmark scaffolding (prerequisite, ~30 LOC), F.1
TAOCP combinatorial-number-system indexing (~150 LOC, removes
~42k+ allocations per build, replaces post-reduction sort), F.2
scratch-buffer column-add (~25 LOC, ~5·10^4 alloc → log m), F.5
cohomology orientation (~250 LOC, 10-100× algorithmic), and F.7+F.8
apparent + emergent pairs (~230 LOC, additional 10-100×), with
F.3+F.4 twist+clearing as a wedge-side fallback if F.5 deferred
and F.9-F.13 as filtration-build + bottleneck cleanups.

---

Progress: 145-topology-perf complete — perf audit of `topology/persistent` reduction core covering F_2 column-add hot loop, allocation patterns (`map[string]int` simplex key, per-`symDiff` slice alloc, per-call boundary scratch, n+1 distance-matrix headers), wedge-vs-cohomology orientation choice, and 13 quantified findings (F.0 benchmark scaffolding → F.13 inner-loop hoisting); single highest-leverage refactor is F.1 TAOCP combinatorial-number-system simplex index (~150 LOC, removes ~42k+ tiny allocations per build) gating F.5 cohomology orientation (~250 LOC, 10-100× algorithmic) gating F.7+F.8 apparent+emergent pairs (~230 LOC, additional 10-100×); stacked ceiling ~1000-3000× on Phase-A fixtures, ~1100 LOC, math-stdlib only, no API break; F.0 benchmark scaffolding flagged as prerequisite-zero since repo currently ships no `Benchmark*` for topology and every speedup claim above is back-of-envelope analysis pending machine measurement.
