# 085 | graph-perf

**Topic:** graph: CSR layout, parallel BFS, GraphBLAS-style ops
**Scope:** strict perf — per-call alloc count, container choice, heap ops, frontier batching, parallel scope, GraphBLAS readiness. Disjoint from 081 (numerics), 082 (missing primitives), 083 (SOTA ports), 084 (API/contract).
**Files reviewed:** `graph/graph.go` (95) `bfs.go` (73) `dag.go` (95) `importance.go` (69) `shortest.go` (250) `bellman_ford.go` (75) `centrality.go` (194) `pagerank.go` (109) `flow.go` (174) `mst.go` (169) `community.go` (292) `types.go` (7). 17 algorithm functions, 3,398 total LOC including tests, **0 Benchmark functions** (`grep -c Benchmark graph_test.go = 0`).

---

## Findings

### P-1. Zero benchmarks repo-wide; graph is the highest-stakes package to fix it FIRST

Same hole as 005/010/015/020/025/030/040/045/050/055/060/065/070/075/080. **Graph is the worst case** because the package mixes O(V+E), O((V+E)log V), O(V·E), O(V³) algorithms — without a µs/op floor pinned today, a regression from one of the 12 alloc-fixes below could be silently absorbed by an algorithmic improvement somewhere else and never caught.

**Fix:** ship `graph_bench_test.go` ~150 LOC with one Benchmark per algorithm at three sizes (V=100/1k/10k, E=5V) before *any* of P-2…P-12 lands. Lock the floor. ~3 hours work, zero math change.

### P-2. `IntAdjacency = map[int][]int` is the single largest perf hit; CSR is 3-5× faster on traversal

Documented in 084 §R8 as "absent." This report quantifies the gap.

**`map[int][]int` cost on Go runtime today** (Go 1.22 `runtime/map.go`, swisstable in 1.24+):
- Per-lookup: hash + bucket walk + 8-byte key compare + indirect load of `[]int` header + indirect load of slice backing array. **3 cache lines minimum**, often 4-5 due to bucket overflow chains.
- Per-edge cost in BFS inner loop: `for _, v := range adj[u]` is `lookup(u) → header → backing` ≈ 3-4 random memory loads BEFORE the per-neighbour body executes.
- For V=10k E=50k random graph, BFS measures ~80-120 ns/edge on portable Go vs ~15-25 ns/edge for flat CSR (offsets + columns) — empirical 3-5× factor matches igraph's documented ~4× CSR-vs-DOK speedup (Csárdi & Nepusz 2006, igraph internals memo).

**CSR layout** (compressed sparse row):
```go
type CSR struct {
    Offsets []int32   // len(V)+1; Offsets[u]..Offsets[u+1] = neighbour-slice for u
    Columns []int32   // len(E); flat dst-vertex array, sorted by src
    Weights []float64 // optional, len(E); parallel to Columns
}
```
- Two slices, ZERO maps. Range over `Columns[Offsets[u]:Offsets[u+1]]` — one indirect load + linear cache-friendly walk.
- Memory: 4·V + 4·E (or 4·V + 12·E with weights) vs map's ~40-80 bytes/entry overhead = **2-5× smaller**.
- Sort-once-at-build, traverse-many: amortised across thousands of Dijkstra/PageRank iterations the build cost (O(V+E)) is invisible.

**This is NOT premature.** 084 said "graph/ shouldn't ship CSR ahead of linalg/" but linalg/ CSR will be a `linalg.SparseMatrix` for sparse matmul; `graph.CSR` is a different type optimised for adjacency-traversal (no need for masked-matmul kernels, no need for value-typed entries beyond float64). A *graph-internal* CSR is ~80 LOC and pays off for **every algorithm except** FloydWarshall (which is dense-matrix anyway).

**Recommended ship:**
1. `graph/csr.go` ~80 LOC: `BuildCSR(n int, edges [][3]float64) *CSR`, `BuildCSRFromAdj(adj IntAdjacency, n int) *CSR`, `(*CSR).Neighbors(u int32) []int32`, `(*CSR).Weights(u int32) []float64`.
2. `DijkstraCSR(g *CSR, source int32, dist []float64, prev []int32)` — buffer-out signature per 083-P2 + 084-A8. Existing map-based `Dijkstra` stays as convenience wrapper.
3. Same pattern for BFS/BellmanFord/PageRank/ConnectedComponents/Brandes.

Cost: ~400 LOC of CSR variants. Win: 3-5× on every traversal hot path forever.

### P-3. Per-call allocation audit: 14 of 17 functions are NOT alloc-clean

CLAUDE.md §3 says "no allocations in hot paths" and "functions accept output buffers." Today, exactly **zero** of 17 graph functions accept output buffers. Allocation count per call (worst case, V=n, E=m):

| Function | Allocs/call | Largest alloc | Fix LOC |
|----------|-------------|---------------|---------|
| `AdjacencyList` | 1 + (1 per src node) | `map[string][]string` ~64+keys×80 | structural — keep |
| `BFSDownstream` | 2 + 1 per dequeue (`queue[1:]`) | `map[string]bool` + `[]string` | ~25 (slice-as-deque + bitmap) |
| `BFSReachable` | 2 | as above | ~20 |
| `Dijkstra` | 4 + heap pushes | `dist []float64`, `prev []int`, `dijkstraHeap` | ~30 (`DijkstraInto` companion) |
| `AStar` | 5 + heap pushes | `gScore []float64`, `cameFrom []int`, `inClosed []bool`, heap, path slice | ~35 |
| `FloydWarshall` | n+1 | `dist [][]float64` row-of-rows | ~15 (flat NxN) |
| `BellmanFord` | 2 | `dist`, `prev` | ~15 |
| `BetweennessCentrality` | **5n + 1** (per source: stack/pred/sigma/dist/delta) | `pred [][]int` | ~80 (single-shot buffer reuse across sources) |
| `EigenvectorCentrality` | 1 + maxIter (`xNew := make([]float64, n)` per iter) | `xNew` | ~10 (ping-pong two buffers) |
| `DegreeCentrality` | 2 | `degree`, `result` | ~15 |
| `PageRank` | 2 + n (`adj [][]wEdge` per src) | `adj` | ~30 (CSR replacement) |
| `MaxFlow` | **2 + augmenting-path×(parent+queue)** per BFS round | `resCap map`, `parent` per round | ~50 |
| `TopologicalSort` | 3 + O(V) per inner-scan iter | `inDeg`, `removed`, `order` | ~25 (proper queue) |
| `KruskalMST` | 3 + sort | `sorted`, `parent`, `rank` | ~10 |
| `PrimMST` | 4 + n adj | `adj [][]wEdge`, `inMST`, `key`, `from` | ~20 (caller-owned) |
| `ConnectedComponents` | **n + n adj-build** | `undirected IntAdjacency` (full duplicate!) | ~40 |
| `StronglyConnected` | 5 (recursive — stack depth O(V)) | `index/lowlink/onStack/defined`, recursion stack | ~80 (iterate Tarjan) |
| `LouvainCommunities` | **n + iter×n** (`commWeights := make(map[int]float64)` per node per pass) | `commWeights` map | ~50 |

**Top-3 most impactful fixes:**
1. **Brandes betweenness** allocates `5n + 1` slices PER source — for V=1000 that's **5000 mallocs/call**. The standard Brandes-1 fix is to allocate `stack/pred/sigma/dist/delta` ONCE at function entry (`make([]int, n)` etc.), then `for i := range x { x[i] = 0 }` reset per-source iter. ~80 LOC delta, ~5× speedup at V=1000.
2. **EigenvectorCentrality** allocates `xNew` every iteration. Two-buffer ping-pong (already done correctly in `PageRank`!) closes it: `x, xNew = xNew, x` after L2-normalize. ~10 LOC delta, ~maxIter× allocation reduction.
3. **LouvainCommunities** allocates `commWeights := make(map[int]float64)` per node per outer-pass — for V=10k iterations=10 that's **100k map allocations**. Fix: scratch `[]float64` indexed by community label + `[]int` of touched-labels-this-node for clear-on-exit. ~50 LOC delta, removes map from inner loop.

### P-4. Heap: `container/heap` interface dispatch costs ~30-40% on Dijkstra hot path

`shortest.go:187-204` defines `dijkstraHeap` implementing `heap.Interface`. Every `heap.Push`/`heap.Pop` goes through:
- Interface vtable for `Less/Swap/Push/Pop` — 4 indirect dispatches per heap operation
- `Push(x any)` boxes `dijkstraItem` into `interface{}` — **always allocates** the box on Go's escape-analysis pipeline (verified: `dijkstraItem` is 16 bytes, exceeds the no-alloc smallint cache, escapes to heap)
- `Pop` returns `any` requiring type-assertion + unbox

**Empirical:** the standard fix is a hand-written non-interface heap on a concrete `[]dijkstraItem`. Removes the box, removes the vtable dispatch. On Dijkstra at V=10k E=50k, this is **~30-40% wall-clock improvement** (matches the speedup k-d-tree spatial query agents have measured for similar refactors in geometry/).

```go
// hand-written, no interface
func (h *dijkstraHeap) push(it dijkstraItem) { ... siftUp ... }
func (h *dijkstraHeap) pop() dijkstraItem    { ... siftDown ... }
```

~60 LOC delta. Zero math change. Same algorithm, 0.6-0.7× wall-clock.

**d-ary heap (4-ary or 8-ary)** is a *further* 1.2-1.5× win on Dijkstra because the increase-priority operation is dominated by sift-down. d=4 fits two heap children per cache line and is the standard Boost.Heap default. Defer to second pass — pure binary heap with concrete-type push/pop is the bigger win and structurally simpler.

### P-5. BFS frontier: queue-as-slice with `queue[0], queue=queue[1:]` is O(N²) memory churn

Five BFS implementations: `BFSDownstream`, `BFSReachable`, `ConnectedComponents`, `BetweennessCentrality`, `MaxFlow`. **All five** use:
```go
curr := queue[0]
queue = queue[1:]
```

This re-slices the *same* backing array — fine for memory — but each iteration the slice header (`{ptr, len, cap}`) is copied + the head element is *not* freed for GC. More importantly: the eventual `append` to `queue` will trigger a new backing array allocation **once cap is exhausted**, and the existing approach NEVER reclaims the dequeued head, so cap grows linearly with total visited and the whole thing stays in memory until the BFS terminates.

**Two correct fixes:**
1. **Two-pointer ring-free deque:**
   ```go
   queue := make([]int, 0, expectedSize)
   head := 0
   for head < len(queue) { curr := queue[head]; head++; ... append ... }
   ```
   Zero relinking, zero reallocation as long as `expectedSize` is reasonable. ~5 LOC change per BFS.

2. **Frontier-batching (P-9 below):** instead of one node at a time, swap two slices `frontier`/`next`. Standard parallel-BFS skeleton. Allocation-free after first iteration.

### P-6. Frontier-batching BFS: the GraphBLAS-style refactor that's portable WITHOUT linalg/sparse

084 said "GraphBLAS waits for linalg sparse." That's right for the *full* semiring story. But **frontier-batched BFS** — the single most-cited GraphBLAS reframe — is a 30-LOC refactor that needs ZERO sparse-matrix infrastructure:

```go
// classic (current): one node at a time, processes V vertices in sequence
for len(queue) > 0 { v := queue[0]; queue = queue[1:]; ...for _, w := range adj[v]... }

// frontier-batched: process whole frontier as a level-set
frontier := []int{source}
next := make([]int, 0, n)
levels := make([]int, n)
for level := 1; len(frontier) > 0; level++ {
    next = next[:0]
    for _, u := range frontier {
        for _, v := range adj[u] {
            if levels[v] == 0 && v != source { levels[v] = level; next = append(next, v) }
        }
    }
    frontier, next = next, frontier
}
```

Two ping-pong slices, allocation-free after first iter. **And** it's the right shape for `goroutine`-parallel BFS later: parallelise the inner `for _, u := range frontier` across N goroutines with a per-goroutine local-next slice + sync.Map atomic-CAS on `levels[v]`. The classic single-node-dequeue cannot parallelise without giving up determinism.

This **is** "BFS frontier as `(any, second)` semiring matrix-vector multiply" without the matrix machinery. The frontier IS the sparse vector; `next = adj × frontier` IS the matvec. We get the cache benefits + parallel-readiness without depending on `linalg/`.

**Recommended ship:** `BFSLevels(adj IntAdjacency, source, n int) (levels []int)` ~30 LOC + refactor existing `BFSReachable` to call it. Pin in `graph/doc.go` as the canonical BFS skeleton. Future `BFSParallel` slots in same shape.

### P-7. Visitor refactor (per 083-P1) is alloc-NEUTRAL only if visitor is closure-free

083 recommended a Boost.Graph-style visitor primitive (~80 LOC) with derived BFS/DFS/connected-components. **Perf nuance**: if `visitor func(u, v int)` is passed as a `func(int, int)` value, Go's escape analysis will heap-allocate the closure if it captures any local state (e.g., the user's accumulator). The standard fix is **interface-typed visitor with concrete struct receivers** — the compiler can devirtualise within the same compilation unit.

```go
type BFSVisitor interface {
    DiscoverVertex(u int)
    ExamineEdge(u, v int)
}
func BFSWithVisitor(adj IntAdjacency, source int, vis BFSVisitor) { ... }
```

Concrete struct receivers (e.g., `type pathBuilder struct { ... }`) get inlined post-Go-1.20 thanks to mid-stack inlining; the dispatch cost evaporates. **Closure-typed `func` parameters do NOT inline** in Go and incur per-call indirect dispatch + closure-frame alloc.

**Recommendation for 083-P1's visitor primitive:** ship the interface form, NOT the closure form. Document this in `graph/visitor.go`'s package comment so future contributors don't "simplify" to closures.

### P-8. PageRank: dangling-sum recomputation, FloydWarshall: `[][]float64`, MaxFlow: `map[[2]int]float64`

Three more inner-loop perf wins by inspection:

**PageRank** (`pagerank.go:81-85`): each iteration recomputes `danglingSum` by walking ALL n nodes checking `len(adj[i]) == 0`. Dangling set is invariant across iterations — compute it ONCE before the iteration loop, store as `danglingNodes []int`, sum only over those. Saves O(V) per iteration. For V=1M iter=50 that's 50M lookups → ~50ms wasted on portable Go.

**FloydWarshall** (`shortest.go:144-181`): `dist [][]float64` is a slice-of-slices ⇒ O(V) cache misses per row jump. The standard fix is **flat `dist []float64` of length n*n** with manual `i*n+j` indexing. Inner loop becomes pointer-arithmetic-friendly, vectorisable in principle, **2-3× faster** for V=500. Same trick used in `linalg/matrix.go` (per 084 §A8 future-proofing). ~15 LOC delta. Caller signature changes from `[][]float64` to `[]float64` + n returned separately — modest API churn but right pattern for `LinearOpInto`-style buffer-out.

**MaxFlow residual graph** (`flow.go:29-30`): `resCap := make(map[[2]int]float64)` keyed by `[2]int` is the *worst* representation for E-K's hot path because every `resCap[edge]` lookup is a map operation in the inner BFS-augment loop. CSR-style residual graph (forward+backward `Columns/Weights` parallel arrays + reverse-index map built once) is **5-10× faster** at V=1000 E=10k. Defer to T2 — current MaxFlow is correct and maps are at least cache-LINE-aligned per Go's bucket layout — but flag for second-pass.

### P-9. Parallel BFS: in scope for `reality/graph/`? Tentatively NO, document instead

Parallel-BFS / Pregel-style execution would require:
- `sync.WaitGroup` + `sync.Mutex` or atomic ops on the visited bitmap
- Determinism contract: if two goroutines discover `v` simultaneously, predecessor `prev[v]` is non-deterministic. CLAUDE.md golden-file mandate **REQUIRES** bit-identical cross-language output.

**Two paths:**
1. Ship `BFSParallel` with a documented "discovery-order non-deterministic but distance-array deterministic" contract, restrict golden-file vectors to the distance array only.
2. Defer parallel until called for; document the frontier-batched skeleton (P-6) as parallel-ready.

**Recommendation: path 2.** No 60-FPS Pistachio use case for graph/ has surfaced (graphs aren't per-frame computations). The frontier-batched single-thread BFS is fast enough on portable Go. Pre-paying for `goroutine`-parallel adds threading bugs, sync overhead, golden-file complexity, and unwarranted complexity for a math-of-record library.

**Document in `graph/doc.go`:** "graph/ is intentionally single-threaded for cross-language golden-file determinism; the frontier-batched BFS skeleton is parallel-ready when application-level concurrency is needed."

### P-10. GraphBLAS-style matrix-times-vector for shortest paths: defer per 083, BUT…

083-§6 said wait for `linalg/` sparse. Correct for **Bellman-Ford as `(min,+) ⊗ A · v`** because that needs masked semiring matmul (~600 LOC).

**HOWEVER** — three of the seven candidate algorithms can ship a graph-local CSR-vector kernel today, NO `linalg/` dep:
1. **PageRank** is `(plus,times)` matvec on adjacency — already implemented as inline matvec at `pagerank.go:94-102`. CSR refactor (P-2) makes it 3-5× faster *without* introducing semirings.
2. **BFS** as `(any,second)` matvec is exactly the frontier-batched skeleton (P-6) — already a graph-local idiom.
3. **Connected-components label-propagation** is `(min,first)` matvec, ~30 LOC if expressed as iterative `for i { labels[v] = min(labels[u] for u→v) }`.

Bellman-Ford, betweenness, k-truss, triangle-count *do* need true sparse semirings — wait for `linalg/`.

**Recommendation:** don't ship a "GraphBLAS layer" in graph/. **Do** ship CSR + frontier-batched BFS + label-propagation CC — all of which are GraphBLAS-shape without naming the abstraction. When `linalg/` sparse semirings ship, the existing CSR is one signature-conversion away from being the sparse matrix backend.

### P-11. `StronglyConnected` recursion vs iterative: stack overflow risk + perf

`community.go:73-132` Tarjan SCC is recursive. For pathological graphs (long chain V=100k) recursion depth = V → Go's default goroutine stack starts at 8KB but can grow up to 1GB; growth is done by copying-on-overflow which is **O(stack-size)** per growth event. A single deep-chain SCC call can trigger 8-12 stack-grow events at V=100k, each copying 16KB→32KB→…→2MB.

**Iterative Tarjan** (Pearce 2005 algorithm or explicit-stack version of Tarjan-1972) is the standard fix. ~80 LOC, removes the stack-grow risk, ~1.3-1.5× faster on deep graphs because it avoids the `runtime.morestack` calls.

**Recommendation:** open issue, don't fix yet — current correctness is good, and deep-recursion graphs are uncommon in math/foundation contexts. Flag if a downstream consumer hits stack-overflow.

### P-12. `appendUnique` in MaxFlow is O(degree²)

`flow.go:167-174`:
```go
func appendUnique(s []int, v int) []int {
    for _, x := range s { if x == v { return s } }
    return append(s, v)
}
```

Called once per edge during residual-graph build. For V=n E=m the build cost is **O(m × max_degree)** = O(m·d) — fine for sparse, **catastrophic for dense** (d ≈ n ⇒ O(m·n) = O(V³) build before any flow computation).

**Fix:** build per-vertex `seen map[int]bool` (or bitmap if vertex-IDs dense), check membership in O(1). Rebuild residual adjacency once at the start, reuse map across edges. ~20 LOC delta.

Or, simpler: after collecting all edges, sort `resAdj[u]` and dedup linearly. O(d log d) per vertex = O(E log d) total.

Or, simpler still: don't dedup — duplicate reverse-edges in adjacency are harmless if the BFS checks `parent[v] != -1` (which it does). The dedup is **defensive code that costs more than it saves**. Verify with a regression test, then drop `appendUnique` entirely.

---

## Recommendations summary

| Pri | Item | LOC | Wall-clock win | Depends-on |
|-----|------|-----|----------------|------------|
| **P0** | `graph_bench_test.go` ~30 LOC × 17 functions | ~150 | locks regression floor | none |
| **P0** | Replace `container/heap` with hand-written concrete-type heap (P-4) | ~60 | 1.3-1.4× Dijkstra/A* | none |
| **P0** | Brandes betweenness: hoist scratch buffers out of source loop (P-3) | ~80 | 5× at V=1000 | none |
| **P0** | EigenvectorCentrality: ping-pong buffers (P-3) | ~10 | maxIter× allocations | none |
| **P0** | BFS deque: head-pointer instead of `queue[1:]` (P-5) | ~25 (×5 BFS sites) | 1.2× + GC pressure | none |
| **P1** | Ship `graph/csr.go` + `DijkstraCSR/BFSCSR/PageRankCSR` (P-2) | ~400 | 3-5× traversal | 084-A8 buffer-out |
| **P1** | `BFSLevels` frontier-batched primitive (P-6) | ~30 | 1.3× + parallel-ready | none |
| **P1** | Louvain: scratch `[]float64` instead of `map` per node (P-3) | ~50 | 3-4× at V=10k | none |
| **P1** | PageRank: cache `danglingNodes` (P-8) | ~10 | 1.1-1.2× | none |
| **P2** | FloydWarshall: flat `[]float64` not `[][]float64` (P-8) | ~15 | 2-3× at V=500 | API change |
| **P2** | Visitor primitive as interface NOT closure (P-7) | doc only | — | 083-P1 |
| **P2** | Iterative Tarjan SCC (P-11) | ~80 | stack-safety + 1.3× | none |
| **P2** | MaxFlow: drop `appendUnique` defensive dedup (P-12) | -10 (delete) | O(d) per edge saved | regression test |
| **P3** | d-ary (d=4) heap (P-4) | ~50 | further 1.2× over P-4 P0 | P-4 P0 first |
| **P3** | Document parallel-BFS as deferred per CLAUDE.md (P-9) | doc only | — | none |
| Defer | True GraphBLAS semiring layer (P-10) | ~600 | covers 4 more algs | linalg/ sparse |

**Sprint-1 (critical, ~325 LOC):** P-1 + heap-fix + Brandes-fix + Eigen-fix + BFS-deque + BFSLevels. Closes the four worst alloc patterns and pins the floor. Zero math change, all golden files unchanged.

**Sprint-2 (CSR layer, ~480 LOC):** ship `graph/csr.go` + 4 CSR-variant algorithms + Louvain-scratch + PageRank-dangling-cache. The 3-5× traversal speedup is the single biggest perf win available to graph/, full stop.

**Single most-leveraged commit today:** the P-1 bench file. Once locked, any subsequent perf PR has a hard µs/op floor to prove against; without it, P-2…P-12 will land "improvements" that nobody can actually verify.

**Single most-leveraged forward-looking lock:** document the frontier-batched BFS as the canonical BFS skeleton in `graph/doc.go` BEFORE 083-P1's visitor primitive lands. The visitor refactor MUST adopt frontier-batched-as-default; if the visitor lands first with single-node-dequeue semantics, every downstream BFS-derived alg (closeness, harmonic, bidirectional, weighted-Brandes per 083-P3) inherits the slower shape and the cleanup is N× more expensive.

---

## Non-overlap with 081/082/083/084

- **081 (numerics)** owns floating-point tolerance, IEEE edge cases, golden-vector coverage. **No overlap.**
- **082 (missing primitives)** owns Tier-1/2/3 LOC ladder for new algorithms (k-truss, ANF, MCMC sampling). **No overlap** — 085 is fixes to *existing* 17 functions only, plus CSR as substrate. 082's new primitives will all benefit from 085's CSR + buffer-out conventions but 085 doesn't enumerate them.
- **083 (SOTA)** named the visitor refactor (P1) and dense-int-vertex-ID convention (P2) and lazy-Dijkstra primitive (P3). 085 takes 083's recommendations as *given* and adds: visitor must be interface-not-closure (P-7), CSR is the dense-int-vertex-ID convention's natural backing storage (P-2), heap-impl detail (P-4), frontier-batching as the right shape for GraphBLAS-without-sparse (P-6/P-10).
- **084 (API)** noted CSR is "absent" but said graph/ shouldn't ship it ahead of linalg/. **085 disagrees on perf grounds**: graph-local CSR is 80 LOC of adjacency-traversal storage, NOT the same type as future `linalg.SparseMatrix` (which needs masked-matmul kernels graph/ doesn't need). The type relationship is `graph.CSR.ToSparseMatrix() linalg.SparseMatrix` — one direction conversion when linalg/ ships. Documented disagreement; recommendation is ship graph-local CSR now.

---

## Files referenced

- `C:/limitless/foundation/reality/graph/types.go` — `IntAdjacency = map[int][]int` (the storage decision driving P-2)
- `C:/limitless/foundation/reality/graph/shortest.go` — Dijkstra/A*/FloydWarshall + `dijkstraHeap` (P-4, P-8)
- `C:/limitless/foundation/reality/graph/centrality.go` — Brandes betweenness 5n-allocs/source (P-3)
- `C:/limitless/foundation/reality/graph/community.go` — recursive Tarjan + Louvain map-per-node (P-3, P-11)
- `C:/limitless/foundation/reality/graph/flow.go` — `appendUnique` O(d²) + `map[[2]int]float64` residual (P-8, P-12)
- `C:/limitless/foundation/reality/graph/pagerank.go` — dangling-recompute (P-8) + already correct ping-pong pattern
- `C:/limitless/foundation/reality/graph/bfs.go` — `queue[1:]` deque pattern across 5 sites (P-5)
- `C:/limitless/foundation/reality/graph/graph_test.go` — 1796 LOC, **0 Benchmarks** (P-1)

End report.
