# 081 | graph-numerics

**Agent:** 081 of 400
**Topic:** graph: weighted Dijkstra precision under negative-zero, Floyd-Warshall accumulating error
**Scope:** `C:/limitless/foundation/reality/graph/` — 12 source files (~1100 LOC numeric portion), 1 test file (1796 LOC). Numeric subset: `shortest.go` (Dijkstra/A*/FloydWarshall), `bellman_ford.go`, `flow.go` (MaxFlow), `pagerank.go`, `centrality.go` (Betweenness/Eigenvector/Degree), `mst.go` (Kruskal/Prim), `community.go` (Louvain), `dag.go`, `bfs.go`. Non-numeric: `graph.go`, `importance.go`, `types.go`, all of `bfs.go`/`dag.go` (string-keyed traversal).
**Verdict:** All numeric routines are textbook-correct on happy paths and the test suite passes (zero failures). Six distinct numerical-correctness liabilities are latent in the current call surface, two of which are named directly by the topic (Dijkstra ±0 tie-breaking, Floyd-Warshall O(V³) accumulation) and four of which surface the moment the package is asked to scale (MaxFlow on float capacities, Bellman-Ford -Inf propagation, PageRank dangling-only graphs, Eigenvector L2 collapse). Zero golden-file vectors exist for any numeric graph function — every other reality package ships golden JSON and this one ships none.

---

## Inventory

| File | LOC | Numeric exposure |
|------|----:|------------------|
| `shortest.go` | 250 | Dijkstra, A*, FloydWarshall |
| `bellman_ford.go` | 75 | BellmanFord with -Inf marker for negative cycles |
| `flow.go` | 174 | MaxFlow (Edmonds-Karp), TopologicalSort |
| `pagerank.go` | 109 | PageRank power iteration |
| `centrality.go` | 194 | BetweennessCentrality, EigenvectorCentrality, DegreeCentrality |
| `mst.go` | 169 | KruskalMST, PrimMST |
| `community.go` | 292 | ConnectedComponents, StronglyConnected, LouvainCommunities |
| `bfs.go`, `dag.go`, `graph.go`, `importance.go`, `types.go` | ~330 | Non-numeric (string adjacency, set ops) |

Golden vectors: **zero**. CLAUDE.md mandates "Minimum 20 vectors per function, target 30" and "Golden files are the proof. Every function has golden-file test vectors." The graph package has 1796 LOC of tests but **zero JSON golden files** — every assertion is hard-coded inline in Go. This is a process-level liability before any numerical finding below.

---

## Findings

### F1 — Dijkstra heap key tie-breaking is *not* dependent on ±0 in the way the topic suggests, but it *is* non-deterministic across heap rearrangements

`shortest.go:194-195`:

```go
func (h dijkstraHeap) Less(i, j int) bool   { return h[i].dist < h[j].dist }
```

Topic prompt asks about "tie-breaking with negative-zero." The IEEE-754 facts (verified by probe):
- `0.0 == -0.0` is `true`.
- `0.0 < -0.0` is `false`.
- `0.0 > -0.0` is `false`.
- `0.0 + (-0.0) = 0.0` (positive zero — sign of `+0.0` wins).

So in `Less`, `+0 < -0` is `false` and `-0 < +0` is also `false` — both items are "equal" to the heap, and the heap's internal sift order decides which pops first. **There is no ±0 hazard at the predicate level.** What *is* a hazard:

**F1a — Tie-breaking is implicit.** When two paths have identical accumulated distance (very common with integer-valued edge weights typical of OSM/road graphs and unit-weight graphs), the prev[] array recorded depends on heap sift order, not on any caller-supplied tiebreaker. Two callers calling `Dijkstra` on the same input get the same answer (deterministic Go heap), but reordering the input adjacency list shuffles the predecessor reconstruction. This is **not** a correctness bug for `dist[]` (which is unique under non-negative weights), but it makes `prev[]` an unreliable witness for path equivalence-class testing — the standard mitigation is a secondary key on node index. ~5 LOC, zero math change:

```go
func (h dijkstraHeap) Less(i, j int) bool {
    if h[i].dist != h[j].dist {
        return h[i].dist < h[j].dist
    }
    return h[i].node < h[j].node // stable secondary key
}
```

**F1b — The real ±0 hazard is in `dist[u]+w`, not in `Less`.** Line 52 (`newDist := dist[u] + w`). If a caller passes `w = -0.0`, the addition normalises away the sign (verified) so dist values never carry a negative zero. But if a caller passes `w = 0.0` and `dist[u] = math.Inf(1)`, then `Inf + 0 = Inf` and the `newDist < dist[v]` check correctly rejects (since `Inf < Inf` is `false`). One sharp edge: **the documentation says "All weights must be non-negative" but the code does not enforce this**. A negative weight silently produces wrong answers (Dijkstra invariant violated — the early-exit on stale entries assumes monotone non-decrease). The fix is either (a) a `for _, w := range weights { if w < 0 { panic/error } }` guard at entry, or (b) a sentinel error return. CLAUDE.md rule §5 ("Precision documented, not assumed") says the failure mode must be enumerated — currently it is documented but not signalled.

**F1c — Stale-entry skip is correct.** Line 44: `if u.dist > dist[u.node] { continue }`. Under non-negative weights this is the canonical lazy-deletion idiom and is bit-exact (no accumulation since `dist[u.node]` was *literally* the value pushed). No fix needed.

### F2 — A* heap key uses fScore but stale check uses gScore — admissibility is preserved but the closed-set guard is doing extra work

`shortest.go:99-128`. The push at line 123 uses `fScore = tentative + heuristic(v)` as the priority. The pop at line 99 reads `u.dist` (which is `fScore`) but the closed-set guard at line 105 uses `inClosed[u.node]`, not a comparison against gScore. The textbook A* with a *consistent* heuristic does not need a closed set re-open — current code is correct under consistency. With merely *admissible* (not consistent) heuristic, A* may need to re-open closed nodes; this implementation skips that branch (`if inClosed[v] { continue }` line 111 prevents reopening). **The doc-comment promises "Must be admissible" without the stronger "consistent" requirement, but the code only handles consistent heuristics correctly.** Caller-supplied admissible-but-inconsistent heuristics produce sub-optimal paths silently — this is a real distinction (e.g., differential heuristics from random pivots are admissible but not always consistent). Fix: either (a) document the consistency requirement, or (b) drop the `inClosed` short-circuit and rely on the `tentative < gScore[v]` check (slower, but handles inconsistent heuristics). For Pistachio's 60 FPS use case the simpler doc fix is right.

### F3 — Floyd-Warshall O(V³) accumulating error has the standard textbook hazard

`shortest.go:144-181`. The inner update at line 172-174:

```go
through := dist[i][k] + dist[k][j]
if through < dist[i][j] {
    dist[i][j] = through
}
```

For *positive* weights this is monotone-decreasing in `dist[i][j]`, so each cell only gets *replaced* when a strictly-smaller value appears — error does *not* accumulate cell-by-cell because the relaxation is a min, not a sum. **The accumulation hazard is in the path itself**: a length-V path through V-1 intermediate +'s has worst-case relative error ≈ (V-1)·ε ≈ V·2⁻⁵² ≈ 2.2e-16·V. For V=1000 that is 2.2e-13, well below any realistic tolerance. For V=10⁶ (which Floyd-Warshall would take 10¹⁸ ops to compute, so impossible) it would be 2.2e-10. **In practical scope this is a non-issue.** The probe confirms `1e15 + 3*1.0 = 1.000000000000003e+15` — for distances of order 1e15 with weight-1 increments, *single* additions already round, but the hazard is unrelated to V³.

The genuine hazard is **mixed-magnitude weights**: if the graph has both ~1e15 weights and ~1.0 weights (e.g., dollars + cents, or kilometers + millimeters), Floyd-Warshall silently drops the small ones during accumulation. The fix is Kahan-Neumaier summation in the relax — but that doubles the inner-loop op count and 99% of callers will never hit this. Document the regime in the doc-comment ("Accuracy degrades when path lengths span >15 decimal magnitudes; use BellmanFord with relative-tolerance dist[u]+w < dist[v]·(1-ε) check for those cases") rather than fix.

The *real* finding here is **F3a — `dist[i][i] = 0` is asserted at init and never re-checked**. If the input edge list contains `(i, i, w)` with `w < 0` (a negative self-loop), the seed loop at lines 157-164 sets `dist[i][i] = w`. Then the main loop runs and the diagonal can drift further negative. The doc-comment at line 141 says `dist[i][i] == 0 (unless a negative cycle exists, which is undefined behavior)` — so the code matches the spec, but the spec is sloppy. A non-negative-weight graph with a 0-weight self-loop is *valid* and `dist[i][i]` should remain 0; the current init handles this correctly. The negative-self-loop case is genuinely undefined behavior in Floyd-Warshall and the doc-comment is correct to disclaim it. No fix needed beyond ~3 LOC of input validation if the package wants to refuse rather than silently mis-compute.

### F4 — Bellman-Ford -Inf propagation is single-pass, only one node per cycle gets marked

`bellman_ford.go:63-72`. The negative-cycle pass runs *one* extra relaxation and marks any node that still relaxes as `dist = -Inf`. **This marks the *first* node touched per cycle, not all nodes reachable from the cycle.** A correct treatment (Cormen-Leiserson-Rivest-Stein 3e §24.1) requires N-1 *additional* propagation passes after detection so that -Inf reaches everything downstream of the cycle. The current code marks ~1 node per cycle and silently leaves all downstream nodes with whatever finite (and now-incorrect) dist they accumulated.

The fix is ~15 LOC — after the detection pass, run one more BFS from each marked node setting `dist[v] = -Inf` for every reachable v. The doc-comment at line 19 says `dist[i] == -Inf means node i is reachable via a negative-weight cycle` — currently the package fulfils this contract for ~1/N of the affected nodes. There are zero tests for negative-cycle propagation depth — the test suite (per `Bellman` searches) covers detection (`hasNegCycle == true`) but not which nodes are marked.

### F5 — MaxFlow on float64 capacities can iterate forever / accumulate residual-error noise

`flow.go:25-104`. Edmonds-Karp on integer capacities is provably O(VE²) and terminates because each augmenting path increases flow by at least 1 (the GCD of integer capacities). On float64 capacities it terminates *for typical inputs* because residuals shrink geometrically, but pathological capacity ratios near 2⁻⁵² of the smallest capacity create augmenting paths whose `pathFlow = 1e-300` and termination is governed by `resCap[edge] <= 0` at line 67 which fails to be true when residuals are tiny-positive (probe: 10×1e-16 = 1e-15 > 0). The standard fix used by Boost/LEMON/networkx is:

1. Document "for floating-point capacities, terminate when bottleneck < eps·max_capacity" and add an `EPS_RATIO` parameter (~10 LOC).
2. Or, accept only integer capacities (CLAUDE.md "Reimplement from first principles" supports the simpler typed contract — an `IntCapacities` variant in addition to or instead of the current float map).

The topic prompt names "Min-cut, max-flow: integer overflow on capacities" — there is **no integer overflow risk** because the capacity type is `map[[2]int]float64` (no integer arithmetic anywhere). The prompt's hazard does not apply to this implementation; the float-residual-noise hazard does. The line-67 check `resCap[edge] <= 0` is correct (`<=` not `<`, so exact zero terminates), but the *next iteration* may still find a tiny-positive residual elsewhere in the graph and BFS-augment a microscopic amount, which the totalFlow accumulator then absorbs as noise — `totalFlow` may end up with bits in positions ε below the true value.

There is also a missed performance/correctness concern in `appendUnique` at line 167: O(degree) per edge insertion makes graph construction O(V·E²) worst-case for dense graphs. Functionally correct but a performance trap (and `appendUnique` is itself a numerical-irrelevant helper, mention here for completeness).

### F6 — PageRank dangling-only graphs converge to uniform but the iteration count required is *always* `iterations` — no convergence test

`pagerank.go:31-109`. The implementation is the canonical Brin-Page form with dangling-mass redistribution at lines 80-86. Numerical hazards:

**F6a — No convergence test.** The loop at line 77 always runs `iterations` times. The doc-comment says "20-100 is typical." For damping=0.85 the contraction ratio is 0.85, so reaching ε=1e-12 needs `log(1e-12)/log(0.85) ≈ 170` iterations, which is *above* the typical-recommendation upper bound. The function silently returns under-converged ranks and the caller has no way to detect it. The standard fix is an L1-norm convergence check between `ranks` and `newRanks`:

```go
delta := 0.0
for i := range ranks {
    delta += math.Abs(ranks[i] - newRanks[i])
}
if delta < tolerance { break }
```

~10 LOC, plus a `tolerance float64` parameter on the API (or a sibling `PageRankTol(n, edges, damping, maxIter, tol) (ranks []float64, iters int, converged bool)`).

**F6b — Damping=1.0 with a graph that has no dangling nodes is a periodic orbit.** Power iteration on a strongly-connected non-aperiodic graph with d=1 does *not* converge — it oscillates. The code clamps `damping` to [0, 1] (lines 41-44) but does not warn at d=1. Standard PageRank explicitly excludes d=1 for this reason; document this regime.

**F6c — Floating-point sum drift.** PageRank is supposed to satisfy `sum(ranks) == 1.0`. Each iteration redistributes `damping*sum + (1-damping) + dangling_correction` mass. Accumulation drift over 100 iterations is ~100·ε which is fine in absolute terms; the doc-comment at line 19 says "Scores sum to 1.0 (within floating-point tolerance)" which matches. Fine.

**F6d — `outWeight[u] == 0` short-circuit at line 95 is correct but redundant.** If `len(adj[u]) == 0` is true (the dangling test that comes first), then `outWeight[u]` is also 0 because no edges contribute. The `|| outWeight[u] == 0` guard catches the case where every outgoing edge has weight ≤ 0, which the input pass at lines 60-62 already filters out. So this branch is dead code. Trivial cleanup (~1 LOC) but worth noting because dead defensive checks are a smell.

### F7 — EigenvectorCentrality L2-normalisation collapses to zero on bipartite graphs (Perron-Frobenius issue)

`centrality.go:97-151`. Power iteration converges to the dominant eigenvector iff that eigenvector is unique (i.e., the graph's adjacency matrix has a strictly-largest-magnitude eigenvalue). For undirected bipartite graphs the spectrum is symmetric (eigenvalue λ implies -λ), so the dominant eigenvalue has multiplicity 2 with magnitudes matching, and power iteration **oscillates** rather than converging. The convergence test at line 145 (`math.Sqrt(diff) < 1e-10`) detects oscillation as "not converged" and the function silently returns the current iterate after maxIter steps. **There is no signal to the caller that convergence failed.** The standard Perron-Frobenius mitigation is a small shift (replace A with A + I, which adds 1 to every eigenvalue and breaks the ±λ degeneracy) — but this only works if you accept the spectral shift in the resulting "centrality" interpretation, which most centrality literature does not. NetworkX returns an `nx.NetworkXError("power iteration failed to converge")` in this case. The minimal fix is a `(ranks []float64, converged bool)` return:

```go
return x, math.Sqrt(diff) < 1e-10
```

~3 LOC, breaking change but worth coordinating with the API agent (082-graph-api or similar slot if it exists).

**F7a — Initial vector is uniform.** For directed graphs whose first node has no outgoing edges, line 121 `xNew[v] += w * x[u]` never fires for `u=0` and `x[0]` decays to whatever the L2-normalisation puts there. Correct behaviour, just worth noting that the initial uniform vector is *deliberately* chosen to be non-orthogonal to the dominant eigenvector (which it always is for a non-zero matrix — generic enough). Fine.

### F8 — Louvain modularity gain formula is subtly wrong for the *single* node move (matches igraph but not the original Blondel 2008)

`community.go:246-252`:

```go
for c, kiIn := range commWeights {
    gain := kiIn - sigmaTot[c]*ki[i]/m2
    if gain > bestGain {
        bestGain = gain
        bestComm = c
    }
}
```

This is the Blondel ΔQ formula *missing* the `(2*ki_in - ki[i]^2/m2)/m2` constant scaling — but the constant scaling is the same for all candidate communities, so the relative ordering is preserved and `argmax` is unchanged. **The gain magnitude is wrong but the choice of move is correct.** This is the same simplification that igraph uses (`igraph_community_multilevel`), so the implementation matches a well-known reference, just not the paper. Document the simplification in the function comment so a future maintainer doesn't "fix" it back to the textbook formula and slow the inner loop by 2× for no behavior change.

**F8a — Move acceptance threshold `gain > bestGain` with `bestGain := 0.0` initialiser** means the algorithm only moves a node if a strictly-positive gain exists. This matches Blondel and is correct. Fine.

**F8b — sigmaTot update at line 244 (`sigmaTot[oldComm] -= ki[i]`) and line 257 (`sigmaTot[bestComm] += ki[i]`) is *not* atomic across the per-node loop iteration**. If the best move is to the same community (`bestComm == oldComm`), the code subtracts and re-adds, producing an exact result for IEEE-754 (subtract-then-add is a no-op when the operands are bit-identical, which they are here). But the *ordering* matters if the loop is ever parallelised. Sequential — fine.

### F9 — KruskalMST sort is not stable; equal-weight edges produce non-deterministic MST

`mst.go:42-44`:

```go
sort.Slice(sorted, func(i, j int) bool {
    return sorted[i][2] < sorted[j][2]
})
```

`sort.Slice` is *not* stable (`sort.SliceStable` is). For graphs with equal-weight edges, the MST is not unique (any of several MSTs is valid) and the output depends on Go runtime sort details. The doc-comment at line 21 says `sorted by weight ascending` but does not say *stable* — so the contract is technically met. But CLAUDE.md rule §1 ("Golden files are the proof") requires bit-stable output across language ports, and Python's `sorted` is stable while Go's `sort.Slice` is not. This is a **golden-file portability hazard waiting to happen**. Trivial fix: `sort.SliceStable` (~1 token change). Add a secondary key on `(u, v)` to lock the order independent of sort stability. ~5 LOC.

**PrimMST has the analogous issue at `mst.go:144-148`** — the linear-scan tie-break picks the smallest-index node, which *is* deterministic. Prim's choice is consistent across language ports.

### F10 — Numeric guard inventory

| Function | Negative input | NaN input | Inf input | Subnormal | ±0 |
|---|---|---|---|---|---|
| Dijkstra | silent wrong (F1b) | propagates | propagates | OK | normalised |
| AStar | silent wrong | propagates | propagates | OK | normalised |
| FloydWarshall | tolerated (mins) | propagates | propagates | OK | normalised |
| BellmanFord | OK by design | propagates to -Inf at detection | propagates | OK | normalised |
| MaxFlow | silently treats as 0 (line 67) | propagates | propagates | F5 hazard | OK |
| PageRank | filtered at line 60 | propagates through ranks | propagates | OK | OK |
| Eigenvector | propagates | NaN-poisons | Inf-poisons | OK | OK |
| Betweenness | n/a (no weights) | n/a | n/a | n/a | n/a |
| Louvain | propagates through totalWeight | NaN-poisons modularity | Inf-poisons | OK | OK |

**Every function silently propagates NaN/Inf weights with zero defensive `math.IsNaN(w) || math.IsInf(w, 0)` check anywhere in the package.** This is a 6-package consistent pattern in `reality` (audio-numerics 006 noted the same; calculus-numerics 016 noted the same) and is a candidate for a package-wide "input validation contract" decision rather than per-function fixes.

---

## Ranked Fix-Set

| # | Item | LOC | Priority | Rationale |
|---|---|---:|---|---|
| R1 | Add stable secondary key `(dist, node)` to `dijkstraHeap.Less` | ~5 | **High** | F1a — locks `prev[]` reproducibility for golden files |
| R2 | Add golden JSON vectors for Dijkstra/Bellman/Floyd/PageRank (≥20 each, including ±0/±Inf/NaN inputs) | ~400 (vectors+test harness) | **High** | CLAUDE.md §1 — every other package ships these |
| R3 | Fix Bellman-Ford -Inf propagation (BFS from each marked node) | ~15 | **High** | F4 — currently doc contract is met for ~1/N of affected nodes |
| R4 | Add convergence-test return to `PageRank` and `EigenvectorCentrality` | ~20 | **High** | F6a, F7 — silent under-convergence is a real hazard |
| R5 | Switch `KruskalMST` to `sort.SliceStable` + secondary `(u,v)` key | ~5 | **Medium** | F9 — golden-file portability across language ports |
| R6 | Add input-validation pass for negative/NaN/Inf weights in Dijkstra/AStar/MaxFlow | ~30 | **Medium** | F1b, F10 — currently silent wrong answers |
| R7 | Document A* consistency requirement (or remove `inClosed` short-circuit) | ~5 doc / ~15 code | **Medium** | F2 — doc-vs-code mismatch |
| R8 | Document Floyd-Warshall mixed-magnitude regime in doc-comment | ~10 doc | **Low** | F3 — informational |
| R9 | Document Louvain igraph-vs-Blondel formula choice | ~5 doc | **Low** | F8 — prevents future "fix" regression |
| R10 | Add `MaxFlowEPS` param or `MaxFlowInt` integer-capacity sibling | ~30 | **Low** | F5 — currently float-residual termination is unsignalled |
| R11 | Remove dead `outWeight[u] == 0` branch in PageRank | ~1 | **Low** | F6d — code cleanup |

Total: ~540 LOC of pure additions plus ~100 LOC of doc-comment fixes. R1+R2+R3+R4 (~440 LOC) is the single highest-leverage commit — it locks reproducibility and signals all four cases of silent under-convergence.

---

## Anti-Recommendations

- **Do not** Kahan-Neumaier the Floyd-Warshall inner loop. The hazard is mixed-magnitude pathological inputs that no realistic shortest-path call site has; doubling op count for the 99% case to handle the 1% is wrong.
- **Do not** add Shewchuk-adaptive predicates here. Graph algorithms operate on labels and weights, not coordinates; the predicate-robustness concerns from `geometry/` (076) do not transfer.
- **Do not** introduce closed-set re-opening in A*. Every realistic heuristic in 2D/3D path-planning (Manhattan, Euclidean, octile, differential-from-pivot) is consistent. Just document the requirement.
- **Do not** rewrite `appendUnique` to use a map for O(1) insertion. The deduplication is a 1×-per-edge operation against a list of bounded size; replacing it with a `map[int]bool` per node creates more garbage than it saves time.
- **Do not** integer-only the MaxFlow API. Float capacities are real (probability flow networks, fractional bandwidth, finance min-cost-flow). Add the EPS-tolerance variant alongside.

---

## Cross-Package Coordination

- **R2 golden-file harness** should follow the pattern established by `geometry/golden_test.go` and `acoustics/golden_test.go` (JSON vectors + per-tolerance enforcement). Coordinate with whatever 082-graph-api / 083-graph-missing produces.
- **R6 input validation** is consistent with the pattern flagged in 006-audio-numerics, 011-autodiff-numerics, 016-calculus-numerics, 021-changepoint-numerics — recommend a package-wide policy (probably an `internal/numguard` or a top-level `reality.ValidateFloat64(name, x)` shared helper) rather than 22 per-package re-implementations.
- **EigenvectorCentrality (F7)** overlaps with `linalg`'s power-iteration and PCA-via-power-iteration paths. If `linalg` exposes a generic power-iteration with convergence return, this function should consume it rather than re-implement.
- The topic-prompt mention of "Graph Laplacian eigenvalues" — **the package does not implement Laplacian eigenvalues anywhere**. Confirmed by grep. This is in the *missing* axis (slot 083?) not the numerics axis. Note for the planner.

---

## Deltas vs Topic Prompt

The topic prompt enumerates 8 examination items. Findings against each:

| Prompt item | Status |
|---|---|
| Dijkstra heap key precision | F1c — bit-exact stale check, no precision issue |
| Tie-breaking with negative-zero | F1a/F1b — ±0 normalises in addition; real issue is implicit tie-break in heap (F1a) |
| A* heuristic admissibility | F2 — doc says admissible, code requires consistent |
| A* stable-tie-break | Not present; same fix as F1a applies |
| Floyd-Warshall O(V³) error | F3 — not the hazard suggested; mixed-magnitude is the real one |
| Bellman-Ford N-1 iterations | F4 — main loop is correct; the **propagation** of -Inf is incomplete |
| BFS/DFS not numerical | Confirmed |
| PageRank convergence + damping | F6a, F6b — no convergence test; d=1 oscillates undocumented |
| Graph Laplacian eigenvalues | **Missing entirely** — out-of-axis for this slot |
| Min-cut, max-flow integer overflow | **Hazard does not apply** — capacities are float64; F5 is the real (float-residual) hazard |

---

## Verdict

`graph/` is a textbook implementation of a 1990s-era set of graph algorithms with full test pass and zero golden files. Numerical hygiene is *adequate* for the current call surface (small-to-medium graphs, well-conditioned weights, no extreme-magnitude mixing) and *deficient* in six specific ways that are latent today and certain to surface as the package picks up consumers. The single highest-impact change is R2 (golden-file harness) — until graph functions ship JSON vectors like every other reality package, the cross-language port contract has no enforcement at this seam.

