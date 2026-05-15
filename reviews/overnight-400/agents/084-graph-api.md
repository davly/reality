# 084 | graph-api

**Agent:** 084 of 400
**Topic:** graph: edge-list vs adjacency, directed/undirected types, weight semantics, vertex IDs, self-loops, multi-edges, mutation, iteration
**Scope:** API ergonomics. Strictly orthogonal to 081 (numerical hygiene), 082 (algorithm gap-list), 083 (SOTA design philosophies). 083 already named the *direction* — porting igraph's dense-int convention (P2), Boost visitor refactor (P1), graph-tool lazy-Dijkstra primitive (P3), NetworKit naming (P4). This file does **not** repeat those four ports. It catalogues, function-by-function, the *current* API surface in `reality/graph/` (17 algorithms, ~900 LOC), names every distinct representation in use, every distinct vertex-ID convention in use, every distinct weight-passing convention in use, and produces a single normalisation table for what each algorithm should accept post-cleanup.

**Headline:** `reality/graph/` ships 17 algorithms across 5 distinct graph representations (`[]Edge` string-pair, `[][3]float64` weighted-edge-triples, `IntAdjacency = map[int][]int`, `[][]wEdge` private slice-of-slice, dense matrix `[][]float64`), 2 distinct vertex-ID conventions (string node names, dense int 0..n-1), and 4 distinct weight-passing conventions (`map[[2]int]float64`, `[][3]float64` triples, `[][]wEdge` private types, no weights at all). No mutation API exists; no graph type exists; no neighbour iteration helper exists. Every algorithm reinvents bounds-checks (`if u < 0 || u >= n` appears 19 times), graph-size inference (`graphSize` / `graphSize3` helpers, called from 2 sites), and undirected-symmetrisation (open-coded in `KruskalMST`, `PrimMST`, `ConnectedComponents`, `LouvainCommunities`). Self-loops and multi-edges are *implicitly* allowed everywhere but documented nowhere — Dijkstra eats them silently, MaxFlow's `appendUnique` deliberately suppresses them in the residual graph, and PageRank's weight aggregation produces nonsense on parallel edges. The two-namespace split (string-keyed `bfs.go`/`graph.go`/`importance.go` vs int-keyed everything-else) is a *historical* artefact of the aicore extraction documented in `graph.go:11`, not a design — and it leaks into every consumer that wants to feed a string-named graph through Dijkstra.

---

## 1. Inventory: every representation in use today

| # | Representation | Type signature | Used by | Lines |
|--|--|--|--|--|
| R1 | **String edge-list** | `[]Edge = [][2]string` | `AdjacencyList`, `Nodes`, `InDegree`, `Roots`, `Leaves`, `BFSDownstream`, `BFSReachable`, `DAGDepth`, `ReachableLeaves`, `NodeImportance`, `EdgeFraction` | graph.go, bfs.go, dag.go, importance.go |
| R2 | **String adjacency map** | `map[string][]string` | Internal — built by `AdjacencyList(edges)` and consumed by every R1 algorithm | graph.go:20-28 |
| R3 | **Int adjacency map** | `IntAdjacency = map[int][]int` | `Dijkstra`, `AStar`, `MaxFlow`, `TopologicalSort`, `BetweennessCentrality`, `EigenvectorCentrality`, `DegreeCentrality`, `ConnectedComponents`, `StronglyConnected`, `LouvainCommunities` | shortest.go, flow.go, centrality.go, community.go |
| R4 | **Int weighted-edge-triples** | `[][3]float64` (each `{from, to, weight}`) | `BellmanFord`, `FloydWarshall`, `KruskalMST`, `PrimMST`, `PageRank` | bellman_ford.go, shortest.go:144, mst.go, pagerank.go |
| R5 | **Sparse weight side-table** | `map[[2]int]float64` | `Dijkstra`, `AStar`, `MaxFlow` (capacity), `EigenvectorCentrality`, `LouvainCommunities` | shortest.go, flow.go, centrality.go, community.go |
| R6 | **Private slice-of-slice weighted adjacency** | `[][]wEdge` (private struct `{to int; weight float64}`) | `KruskalMST` (rebuilds from R4), `PrimMST` (rebuilds from R4), `PageRank` (rebuilds from R4) | mst.go:118, mst.go:122, pagerank.go:48 |
| R7 | **Dense matrix** | `[][]float64` of size n×n | `FloydWarshall` *return value only* | shortest.go:144 |
| R8 | **CSR / CSC** | not present | — | — |

**R6 is the smell.** Three algorithms (`KruskalMST`, `PrimMST`, `PageRank`) each define their own private `wEdge` struct, build a `[][]wEdge` from R4, never expose it, and throw it away. They are doing the same conversion in three places because R4 (`[][3]float64`) is the wrong shape for the inner loop and there is no canonical weighted-adjacency type in the package.

**R8 absent.** No CSR (compressed-sparse-row) representation exists. CLAUDE.md §3 calls for "no allocations in hot paths"; CSR is the canonical zero-alloc representation and is what `linalg/` will need for sparse matmul (per 083's GraphBLAS deferred-port). Flagging the gap; not a recommendation for `graph/` to ship CSR ahead of `linalg/`.

---

## 2. The two-namespace split

`reality/graph/` has **two disjoint algorithm families** with no bridge between them:

**Family A — string-keyed (R1+R2):** `AdjacencyList`, `Nodes`, `InDegree`, `Roots`, `Leaves`, `BFSDownstream`, `BFSReachable`, `DAGDepth`, `ReachableLeaves`, `NodeImportance`, `EdgeFraction`. **11 functions.** Provenance: `graph.go:11` says "Extracted from: github.com/davly/aicore/causalmath" — these were lifted from a causal-DAG library where node IDs were human-readable strings ("symptom_A", "evidence_B"). The string keys are load-bearing in their original use case.

**Family B — int-keyed (R3+R4+R5):** `Dijkstra`, `AStar`, `FloydWarshall`, `BellmanFord`, `KruskalMST`, `PrimMST`, `MaxFlow`, `TopologicalSort`, `BetweennessCentrality`, `EigenvectorCentrality`, `DegreeCentrality`, `PageRank`, `ConnectedComponents`, `StronglyConnected`, `LouvainCommunities`. **15 functions.** Provenance: written for `reality/`; deliberately use dense int IDs for cache locality and the 60-FPS allocation-free constraint.

**The gap:** there is no bridge function. A caller with a string-named DAG (the natural shape for causal inference, dependency graphs, knowledge graphs) cannot run Dijkstra without reinventing the intern table by hand. There is no `func Intern(edges []Edge) (intEdges [][2]int, names []string)` helper, no `func StringAdjToInt(adj map[string][]string) (IntAdjacency, names []string)` helper. Per 083's P2 recommendation this should be a ~50-LOC adapter; flagging the *consumer impact* here:

- **`aicore/causalmath` itself** (the original source of Family A) cannot use Family B against the same graph it builds. If `aicore` wants to compute SCCs or PageRank on a causal DAG it has to re-encode every node name. This is the real cost of the split.
- **`reality/graph/importance.go:NodeImportance`** runs `O(|V|·(|V|+|E|))` and is documented as such (importance.go:15). It would be `O(V+E)` if it ran on Family B — Tarjan SCC + a single condensation-DAG pass — but that requires the bridge. The slow O(V·(V+E)) implementation is the *direct* cost of having no string→int adapter.

**Recommendation 084-A1 (~50 LOC):** add the bridge. Two functions only:

```go
// Intern assigns dense int IDs to string node names.
// Returns the int edge list, the int->name table, and the name->int map.
func Intern(edges []Edge) (intEdges [][2]int, names []string, lookup map[string]int)

// IntAdjacencyFrom builds an IntAdjacency from interned int edges.
func IntAdjacencyFrom(intEdges [][2]int) IntAdjacency
```

That's it. No new representation, no new graph type. Just the missing 50 LOC that lets Family A callers reach Family B algorithms.

---

## 3. Vertex-ID conventions

| Convention | Used in | Implicit assumptions |
|--|--|--|
| `string` (any Unicode) | Family A | Empty string `""` is a sentinel meaning "skipped" (graph.go:23, graph.go:38, dag.go:62) |
| `int ≥ 0`, dense `0..n-1` | Family B | Negative IDs are silently skipped (`if u < 0 || u >= n { continue }` — appears 19 times) |
| `int ≥ -1` for "no predecessor" | `prev` array returns from Dijkstra/Bellman-Ford/A*/KruskalMST | `-1` is the sentinel (shortest.go:32, bellman_ford.go:33, mst.go:48, mst.go:78) |

**Three conventions, three sentinels.** `""` for "skip", `-1` for "no predecessor", and the implicit `n` upper bound for "out of range — silently skip." None of these are documented at the package level (graph.go:1-11 mentions only "directed string-labeled graphs," predating Family B's existence).

**Per-algorithm `n` inference is inconsistent:**
- `Dijkstra` *infers* `n` from `graphSize(adj, source)` which scans `adj` — convenient but O(|V|+|E|) per call (shortest.go:212-225).
- `AStar` *infers* `n` from `graphSize3(adj, source, target)` — same pattern.
- `BellmanFord`, `FloydWarshall`, `KruskalMST`, `PrimMST`, `PageRank`, `BetweennessCentrality`, `EigenvectorCentrality`, `DegreeCentrality`, `ConnectedComponents`, `StronglyConnected`, `LouvainCommunities`, `TopologicalSort`, `MaxFlow` — caller passes `n` explicitly.
- This is a **2-out-of-15 vs 13-out-of-15 split** with no documented rationale. `MaxFlow` is the worst offender: takes both `adj` *and* infers via `graphSize3(adj, source, sink)` — no explicit `n` (flow.go:25-26).

**Recommendation 084-A2 (~30 LOC, doc + 2 fn signatures):** standardise `n` as an explicit caller-passed parameter for all int-keyed algorithms. Keep `graphSize`/`graphSize3` as exported helpers (`graph.NodeCount(adj IntAdjacency) int`) for callers who don't know `n` up-front. Update `Dijkstra` and `AStar` signatures to require explicit `n`; provide `Dijkstra` (explicit `n`) and `DijkstraAuto` (back-compat alias that calls `NodeCount` first). This is purely API hygiene; no algorithm changes.

---

## 4. Weight-passing conventions: 4 in use

| Convention | Used by | Allocates | Random-access | Iteration-friendly |
|--|--|--|--|--|
| **W1: `map[[2]int]float64` side-table, paired with `IntAdjacency`** | `Dijkstra`, `AStar`, `MaxFlow`, `EigenvectorCentrality`, `LouvainCommunities` | yes (map) | O(1) avg | poor (must walk adj, lookup map per edge) |
| **W2: `[][3]float64` triples** | `BellmanFord`, `FloydWarshall`, `KruskalMST`, `PrimMST`, `PageRank` | yes (slice) | none | excellent (linear walk) |
| **W3: Private `[][]wEdge` slice-of-slice** | Internal in `KruskalMST`, `PrimMST`, `PageRank` (built from W2) | yes | O(1) per neighbour | excellent (tight inner loop) |
| **W4: No weights** | `BFSDownstream`, `BFSReachable`, `DAGDepth`, `ReachableLeaves`, `NodeImportance`, `EdgeFraction`, `BetweennessCentrality`, `DegreeCentrality`, `ConnectedComponents`, `StronglyConnected`, `TopologicalSort` | n/a | n/a | n/a |

**Three observations:**

1. **W1 vs W2 are interchangeable but algorithms pick one with no documented criterion.** `Dijkstra` could equally accept `[][3]float64` (it does an `O(deg(u))` lookup per pop, so the map gives no benefit over a slice-of-slice); `BellmanFord` could equally accept `(IntAdjacency, map[[2]int]float64)`. The choice tracks author preference, not algorithm structure. Internal consistency would say: pick one canonical weighted-adjacency type and use it everywhere.

2. **W3 is private and re-derived three times.** `KruskalMST:118`, `PrimMST:122`, `PageRank:48` each define `type wEdge struct{ to int; weight float64 }` locally. Promoting `wEdge` to an exported `WeightedAdjacency = [][]struct{ To int; Weight float64 }` would eliminate ~30 LOC of duplication and give the package a canonical iteration-friendly weighted representation.

3. **W1's "missing edge means absent" semantics is bug-prone.** From `Dijkstra` (shortest.go:48-50): "`if !ok { continue }`." A weight of `0.0` is *valid* (shortest distance unchanged when traversing); but a *missing* entry means edge does not exist. If a caller forgets to register a weight, the edge is silently dropped — and the `IntAdjacency` *separately* says the edge exists. **The `adj` and `weights` arguments can disagree** and the algorithm produces wrong results without any error. This is the same failure mode as W4-vs-W1 confusion: an edge is in `adj` but not `weights` and gets silently skipped.

**Recommendation 084-A3 (~40 LOC + ~30 LOC of consumer migration):** introduce a single canonical weighted-adjacency type and rewrite the three internal `wEdge` definitions to use it.

```go
// WeightedNeighbor is an out-edge with a numeric weight.
type WeightedNeighbor struct {
    To     int
    Weight float64
}

// WeightedAdjacency is a directed adjacency list with edge weights.
// adj[u] is the slice of out-edges from vertex u.
type WeightedAdjacency = [][]WeightedNeighbor
```

Add `WeightedAdjacencyFromTriples(n int, edges [][3]float64) WeightedAdjacency` and `WeightedAdjacencyFromMaps(adj IntAdjacency, weights map[[2]int]float64) WeightedAdjacency` constructors. Algorithms keep accepting their existing shapes for back-compat; the new type is *additive*. Net: one canonical iteration-friendly type, no breaking changes, ~40 LOC added.

---

## 5. Directed vs undirected: per-call semantics, no type distinction

`reality/graph/` correctly punts the directed/undirected distinction to *per-algorithm semantics* rather than baking it into a `Graph` vs `DiGraph` type (per 083 §1's NetworkX critique — this is the right call for Go). But the per-call documentation is **inconsistent across algorithms**:

| Algorithm | Doc-comment says | Treats edges as |
|--|--|--|
| `Dijkstra` | "weighted directed graph" | directed |
| `AStar` | "weighted directed graph" | directed |
| `FloydWarshall` | "weighted directed graph" | directed |
| `BellmanFord` | "weighted directed graph" | directed |
| `MaxFlow` | (no explicit statement) | directed (with implicit reverse residual) |
| `TopologicalSort` | "directed acyclic graph" | directed |
| `BetweennessCentrality` | "directed graph" | directed |
| `EigenvectorCentrality` | "directed graph" | directed |
| `DegreeCentrality` | "directed graph" | directed; counts in+out |
| `KruskalMST` | "undirected graph" | **undirected** |
| `PrimMST` | "undirected graph" | **undirected** (symmetrises in mst.go:128-129) |
| `ConnectedComponents` | "undirected graph" / "treated as undirected" | **undirected** (symmetrises in community.go:21-26) |
| `LouvainCommunities` | "weighted undirected graph" / "treated as undirected" | **undirected** (symmetrises in community.go:181-197) |
| `StronglyConnected` | "directed graph" | directed |
| `PageRank` | "directed graph" | directed |
| BFS family (string-keyed) | none of the doc-comments mention directedness | directed (implicit from `AdjacencyList` builder) |

**Symmetrisation is open-coded in 4 places.** `KruskalMST` (mst.go: builds undirected `adj` inline), `PrimMST` (mst.go:128-129: appends both directions), `ConnectedComponents` (community.go:21-26: builds `undirected` map), `LouvainCommunities` (community.go:181-197: deduplicates with `seen` map). Same boilerplate, four implementations, three of them slightly different (e.g., `LouvainCommunities` deduplicates parallel edges; `ConnectedComponents` and `PrimMST` do not).

**Recommendation 084-A4 (~20 LOC):** add `Symmetrize(adj IntAdjacency) IntAdjacency` as a public helper and replace the four open-coded sites. Document the convention package-wide: each algorithm's doc-comment includes one of "**Treats edges as directed**" or "**Treats edges as undirected (symmetrises internally)**". This is a doc-only change for 13 of 15 functions; 4 algorithms get a 2-line internal cleanup.

**Anti-recommendation:** do *not* introduce a `Directed bool` flag or `DiGraph`/`UnGraph` types. Per 083 §1, NetworkX's four-type hierarchy is the wrong port. The correct convention is what `reality/graph/` already does — per-algorithm semantics — but it needs better naming and a shared helper.

---

## 6. Weight semantics — what does a weight *mean*?

Cross-algorithm semantic table:

| Algorithm | Weight interpretation | Sign constraint | Zero meaning | Negative meaning |
|--|--|--|--|--|
| `Dijkstra` | edge cost (additive) | **must be ≥ 0** (undocumented assumption — incorrect output if violated) | zero-cost traversal | undefined behaviour |
| `AStar` | edge cost | **must be ≥ 0** (undocumented) | zero-cost | undefined |
| `BellmanFord` | edge cost | any sign allowed (documented bellman_ford.go:14) | zero-cost | shorter path |
| `FloydWarshall` | edge cost | any (negative cycles → undefined) | zero-cost | shorter |
| `MaxFlow` | edge **capacity** (not cost) | must be ≥ 0 (otherwise residual graph breaks) | edge effectively absent | undefined |
| `KruskalMST` | edge cost | any (sort-by-weight) | zero in MST | sorts first |
| `PrimMST` | edge cost | any | zero in MST | sorts first |
| `PageRank` | edge **weight share** (multiplicative) | must be > 0 (zero/neg ignored: pagerank.go:60-62) | edge ignored | edge ignored |
| `EigenvectorCentrality` | edge influence | typically ≥ 0; algorithm runs on any sign but Perron-Frobenius assumes nonneg | normal | runs but interpretation breaks |
| `LouvainCommunities` | edge similarity | should be ≥ 0 for valid modularity | normal | unhandled |

**Three different weight semantics in the package** (cost / capacity / share/similarity) and **no algorithm documents which one it expects.** A user who passes a "similarity score in [0,1]" matrix into `Dijkstra` gets the *minimum-similarity* path, which is almost certainly not the intent.

**Three undocumented sign constraints:** Dijkstra/AStar/MaxFlow all require non-negative weights but don't say so. (081 covers the *numerical* consequences of negative-weight Dijkstra; 084 flags the *API documentation* gap — the function should refuse or warn, or at least say so in its doc-comment.)

**Recommendation 084-A5 (~30 LOC across 10 doc-comments + 0 algorithm changes):** every weighted algorithm's doc-comment gets a `// Weight semantics:` line stating one of three values (`cost` / `capacity` / `weight-share`) and a `// Sign constraint:` line stating one of (`any` / `nonneg` / `positive`). No code changes; pure documentation hygiene. This is the lowest-LOC, highest-clarity item in the whole 084 list.

---

## 7. Self-loops and multi-edges: implicitly allowed, never tested

**Self-loops (`u == v`):**
- `AdjacencyList` (graph.go:20): self-loops added with no special handling. `nodes[v]` will appear once in `adj[v]`.
- `Dijkstra` (shortest.go:47): self-loop `u → u` with weight `w` is examined; `newDist = dist[u] + w >= dist[u]` (assuming nonneg) so no update. Silently discarded. Correct.
- `BellmanFord` (bellman_ford.go:50): same — self-loop with positive weight does nothing; with negative weight triggers neg-cycle detection on the next pass. Correct.
- `KruskalMST` (mst.go:82-84): self-loop `u→u` is processed; `find(u) == find(u)`, `union` returns `false`, edge is rejected. Correct but allocates the union-find call.
- `PageRank` (pagerank.go:55-65): self-loop counted in `outWeight[u]`; in the iteration `newRanks[u] += contrib * w` is a self-feedback loop. **Inflates the node's own rank** — possibly desired, possibly not, but not documented.
- `LouvainCommunities` (community.go:181-197): self-loop `[u,u]` keyed twice (`seen[key]` and `seen[rev]` are both `[u,u]`). The dedup logic accepts the *first* occurrence and adds it to both `neighbors[u]` *and* `neighbors[u]` again — double-counting. **Silent bug** for self-loops in Louvain.
- `StronglyConnected` (community.go:73): self-loops are correctly handled — node is in its own SCC trivially.

**Multi-edges (parallel `u → v` with different weights):**
- `AdjacencyList` (graph.go:20): both edges appear as separate entries in `adj[u]`, both pointing at `v`. BFS visits `v` only once (first wins).
- `Dijkstra` (shortest.go:47-49): each parallel edge is examined; the lookup `weights[[2]int{u, v}]` returns *one* value (the last one written by the caller into the map). **The W1 weight-side-table convention loses parallel-edge information.** The two parallel edges in `adj[u]` get the same weight from the map.
- `KruskalMST` / `PrimMST`: parallel edges in `[][3]float64` are processed in sort order; lower weight wins (Kruskal) or earliest in scan wins (Prim). Documented in mst.go:18 ("Duplicate edges are handled correctly (lowest weight wins)").
- `PageRank`: parallel edges are summed into `outWeight[u]`; the iteration distributes correctly weighted by share — multi-edges are well-defined and aggregate.
- `MaxFlow` (flow.go:39-41): `appendUnique` deliberately deduplicates (silent collapse of parallel edges; capacity loss).
- `LouvainCommunities`: dedup with `seen` map keyed on `[u,v]`/`[v,u]` rejects parallel edges.
- `BetweennessCentrality`: walks `adj[v]` directly so parallel edges are double-counted in the BFS frontier expansion, which inflates `sigma[w]` (number of shortest paths) — wrong by an integer factor.

**Summary: 6 distinct self-loop/multi-edge behaviours, 0 documented.**

| Algorithm | Self-loop behaviour | Multi-edge behaviour |
|--|--|--|
| Dijkstra/AStar/BellmanFord/FloydWarshall | Correct (no update / triggers neg-cycle if neg) | **W1 loses info; W2 picks first relaxation** |
| KruskalMST | Correctly rejected via union-find | **Documented:** lowest weight wins |
| PrimMST | Adds twice to adjacency, doesn't break | First-in-scan wins |
| PageRank | **Inflates self-rank; undocumented** | **Aggregates correctly** |
| MaxFlow | Self-loop in residual graph doesn't break | **Silently collapses parallel edges (capacity loss)** |
| LouvainCommunities | **Double-counts in adjacency build (bug)** | **Silently dedups (capacity loss)** |
| BetweennessCentrality | Trivially OK | **Inflates sigma counts (wrong)** |

**Recommendation 084-A6 (~50 LOC across docs + 2 small fixes):**
1. Document each algorithm's self-loop and multi-edge handling in its doc-comment (per algorithm: 2 lines).
2. Fix the `LouvainCommunities` self-loop double-count (1-line guard `if u == v { continue }` in the symmetrisation pass).
3. Document `MaxFlow`'s `appendUnique` capacity-collapse on parallel edges as a known limitation; alternatively, sum capacities (~3 LOC; the more useful semantics).
4. Document `BetweennessCentrality`'s parallel-edge sigma-inflation; the textbook fix (Brandes 2001 §2.2) is to pre-deduplicate the adjacency or use a multiset-aware variant.

This is hygiene; not a behaviour change for any single-edge-per-pair caller (the common case).

---

## 8. Mutation: there is no mutation API

`reality/graph/` is **fully immutable from the public API.** No function adds an edge, removes an edge, adds a node, removes a node. Every algorithm takes the graph as an input value and returns a new value.

This is *probably* correct for a pure-math foundation library. CLAUDE.md §2 ("zero dependencies") and §3 ("no allocations in hot paths") favour value-in / value-out. The current convention is consistent: build edge-list once, run algorithms.

**But the gap shows up in three places:**

1. **`NodeImportance`** (importance.go:36-40) wants to compute "what happens if I remove node X." Today it uses a per-node `exclude` parameter threaded through `ReachableLeaves` (dag.go:79). This is a *partial* mutation API: "give me the algorithm result *as if* this node were not there." Five string-keyed functions take this `exclude` parameter:
   - `BFSReachable(edges, starts, exclude string)`
   - `ReachableLeaves(edges, roots, exclude string)`
   - `NodeImportance(edges)` — calls `ReachableLeaves` with each node as `exclude`
   - (Implicitly) `DAGDepth` does *not* support exclude even though it would be useful for the same use case.
   - The int-keyed family has *no* equivalent — there is no `Dijkstra(adj, weights, source, exclude int)` for "shortest path avoiding node X."

2. **The `exclude string` convention is opt-in by author preference, not systematic.** Three of the eleven Family-A functions support it (`BFSReachable`, `ReachableLeaves`, transitively `NodeImportance`); zero of the fifteen Family-B functions do.

3. **Edge removal is impossible.** Even via `exclude`. Want to compute Dijkstra ignoring edge `(u,v)`? You must rebuild the entire `adj` and `weights` maps with that edge filtered out. This is not catastrophic but it's the *natural* failure mode for fault-tolerance / link-failure analysis — a use case that lives one level up in `reality/`'s consumer stack.

**Recommendation 084-A7 (decision-only; ~0 LOC):** keep `reality/graph/` immutable. Do not add a `Graph` struct with `AddEdge`/`RemoveEdge` methods. *Do* document the convention package-wide. *Do* consider adding an `excludeNodes []int` and `excludeEdges []` parameter slot to the int-keyed algorithms in 082-Tier-2 territory (shortest-path-avoiding-nodes is a real consumer need). For now, immutable + per-call rebuild is the right discipline.

**Anti-recommendation:** do *not* add a `*Graph` type with mutator methods. CLAUDE.md §3 forbids it; igraph's value-in/value-out (per 083 §2) is the correct port; NetworkX's mutable `G.add_edge(...)` would be a regression for Pistachio's 60-FPS constraint.

---

## 9. Iteration patterns over neighbours

Today's idiom (across all 15 int-keyed algorithms) is identical:

```go
for _, v := range adj[u] {
    if v < 0 || v >= n {
        continue
    }
    // ... do work on edge (u, v) ...
}
```

**The bounds-check appears 19 times** (across `shortest.go`, `flow.go`, `centrality.go`, `community.go`, `mst.go`, `bellman_ford.go`, `pagerank.go`). Each one is a defensive guard against malformed adjacency input — a node ID outside `[0, n)`. In the common case (well-formed graph) this is dead code.

The **right Go-idiomatic factoring** would be a `Neighbors(adj IntAdjacency, u, n int) []int` function that does the bounds-check once at allocation time, or — better — a `func ForEachNeighbor(adj, u, n int, f func(v int))` callback. But this is the visitor refactor that 083-P1 already names; not new in 084.

**Specific 084 finding on iteration:** there is **no iteration helper for weighted graphs.** `LouvainCommunities` walks `neighbors[u]` (a private `[]neighbor`); `KruskalMST` walks `[][]wEdge`; `PageRank` walks `[][]wEdge`; `Dijkstra` walks `adj[u]` (unweighted) and looks up the weight in a map per iteration. There are 4 distinct weighted-iteration idioms in the package.

**Recommendation 084-A8 (~30 LOC, follows from A3):** once `WeightedAdjacency` is the canonical weighted type (per A3), provide:

```go
// ForEachOutEdge calls f(v, w) for each out-edge (u → v) with weight w in adj.
func ForEachOutEdge(adj WeightedAdjacency, u int, f func(v int, w float64))

// Neighbors returns the out-neighbours of u, with no weight info.
func Neighbors(adj IntAdjacency, u int) []int
```

`Neighbors(adj, u)` is just `adj[u]` — but the *named* function gives the package a documented neighbour-iteration contract that callers can rely on. Future-proof for a CSR backend (`Neighbors` becomes a slice into a CSR offset array; consumer code doesn't change).

---

## 10. Normalisation table — the canonical post-cleanup API

The table below is what each algorithm *should* accept after applying 084-A1 through A8. It is **purely additive** to today's API — every existing signature stays valid (back-compat); the new types are recommended for consumers building new code.

| Algorithm | Input graph type | Input weight type | Output |
|--|--|--|--|
| BFS / DFS family | `IntAdjacency` (Family B) **or** `[]Edge` (Family A) | none | `[]int` of visit order or `map[int]bool` of visited |
| Dijkstra / AStar / BellmanFord | `WeightedAdjacency` (canonical) **or** `IntAdjacency + map[[2]int]float64` (back-compat) | sign-constrained per-fn | `dist []float64, prev []int` |
| FloydWarshall | `[][3]float64` triples (back-compat) **or** `WeightedAdjacency` | any | `[][]float64` matrix |
| KruskalMST / PrimMST | `[][3]float64` triples (back-compat) **or** `WeightedAdjacency` | any | `[][3]float64` MST edges, `float64` total |
| PageRank | `[][3]float64` triples (back-compat) **or** `WeightedAdjacency` | positive | `[]float64` ranks |
| MaxFlow | `IntAdjacency + map[[2]int]float64` (back-compat) **or** `WeightedAdjacency` | nonneg (capacity) | `float64` flow |
| Centrality family | `IntAdjacency` | none / optional `WeightedAdjacency` | `[]float64` |
| Community family | `IntAdjacency` | optional `WeightedAdjacency` | `[]int` labels |
| TopologicalSort | `IntAdjacency` | none | `[]int` order, `error` |

The point is: **two canonical types** (`IntAdjacency`, `WeightedAdjacency`) plus one canonical builder pair (`Intern`, `WeightedAdjacencyFromTriples`), and every algorithm's first-best signature uses one of them. The `[][3]float64` triples and `map[[2]int]float64` side-tables remain accepted *for back-compat* but the new code path is one-of-two canonical types.

---

## 11. Ranked recommendations

| # | Recommendation | LOC | Cost | Benefit |
|--|--|--:|--|--|
| **A5** | **Document weight semantics + sign constraints in 10 doc-comments** | ~30 | doc-only | Closes the highest-impact correctness footgun (Dijkstra-with-negative-weights = silent wrong answer); zero risk |
| **A6** | **Document self-loop + multi-edge handling per algorithm; fix Louvain self-loop bug; fix Betweenness sigma-inflation** | ~50 | doc + 2 small fixes | Closes 2 latent bugs; documents 7 implicit conventions |
| **A1** | **String↔int bridge (`Intern`, `IntAdjacencyFrom`)** | ~50 | additive | Bridges Family A and Family B; unblocks aicore consumers; aligns with 083-P2 |
| **A4** | **Public `Symmetrize` helper; replace 4 open-coded sites; document directed/undirected per-algorithm** | ~20 | additive + 4 internal cleanups | Removes ~30 LOC of duplication; one-line doc convention package-wide |
| **A3** | **Canonical `WeightedAdjacency` type + 2 constructors; replace 3 internal `wEdge`** | ~40 + 30 migration | additive | One canonical iteration-friendly weighted type; eliminates R6 internal duplication |
| **A8** | **Public `Neighbors`, `ForEachOutEdge` helpers** | ~30 | additive | Names the iteration contract; future-proof for CSR; depends on A3 |
| **A2** | **Standardise explicit `n` parameter; expose `NodeCount` helper** | ~30 | minor signature change | API consistency; 2 of 15 fns currently differ |
| **A7** | **Decision: keep immutable; document the convention; defer `exclude*` parameters to 082-Tier-2** | 0 | doc-only | Confirms the right architectural call; no premature mutation API |

**Total LOC for all 8 recommendations: ~250 additive + ~30 internal migration + 0 breaking changes.**

The whole 084 package fits in <300 LOC, all back-compatible. Priority order: A5/A6 (correctness docs) before A1 (bridge) before A3/A4/A8 (canonical types and helpers) before A2 (signature consistency) before A7 (architectural confirmation).

---

## 12. Anti-recommendations

| # | Anti-port | Why not |
|--|--|--|
| AA1 | **Add a `*Graph` mutable type with `AddEdge`/`RemoveEdge`/`AddNode` methods** | Wrong for Go + CLAUDE.md §3; per 083 §1 the NetworkX hierarchy is the wrong port; current value-in/value-out is correct |
| AA2 | **Add `Directed` / `Weighted` flags to algorithms** | Per-algorithm semantics is the right call; flags multiply the test surface and obscure the contract |
| AA3 | **Generic vertex IDs via type parameters (`type Graph[V comparable]`)** | Adds Go 1.18+ generics complexity for no measurable benefit; per 083 §2 dense-int is the canonical choice; string-keyed Family A handles the human-readable case via the A1 bridge |
| AA4 | **Multiple graph types (`Graph`, `DiGraph`, `MultiGraph`, `MultiDiGraph`)** | Per 083 §1 anti-port; one `IntAdjacency` + per-algorithm semantics is correct |
| AA5 | **Mutator methods on adjacency (`adj.AddEdge(u, v)`)** | Maps in Go are reference types; consumers can mutate `adj` directly if they need to. No method indirection needed |
| AA6 | **Builder pattern (`graph.New().Directed().Weighted().Build()`)** | Java/Kotlin idiom; not Go-idiomatic; slot-coordinate with the overall reality/ API style which is package-level functions on plain types |
| AA7 | **A `Graph` interface with `Neighbors`/`Edges`/`Weight` methods** | Tempting (per 083 §3 Boost.Graph concepts) but Go interfaces would force allocation in hot paths via the iface vtable; current direct slice access is faster and CLAUDE.md §3-compliant |

---

## 13. Cross-package coordination notes

- **084-A1 (`Intern` bridge)** unblocks `aicore/causalmath` consumers that today re-encode their string DAGs by hand to use Family B.
- **084-A3 (`WeightedAdjacency`)** is a *type* that future `linalg/` sparse-matrix work might need to interop with. Coordinate naming: if `linalg/` ships `linalg.Sparse` or `linalg.CSR`, a future `graph.WeightedAdjacencyToCSR(adj WeightedAdjacency) linalg.CSR` adapter is one-line obvious.
- **084-A8 (`Neighbors`, `ForEachOutEdge`)** is the iteration contract that 083-P1's visitor refactor builds on. Slot-coordinate: do A3 + A8 first, then 083-P1 visitor refactor, in that order.
- **084-A6 (Louvain self-loop fix)** is a 1-line bug fix in the Louvain modularity build; 081-F8 covers a *different* Louvain numerical bug (the gain-formula sign). Both fixes, neither overlapping.
- **084-A5 (weight-semantics docs)** is purely doc; coordinate with 081's broader documentation pass on numerical contracts (Dijkstra-on-negative-weights documented as undefined behaviour, not silent wrong answer).

---

## 14. Sources

- `reality/graph/graph.go`, `bfs.go`, `dag.go`, `importance.go` — Family A (string-keyed, 11 fns).
- `reality/graph/types.go`, `shortest.go`, `bellman_ford.go`, `mst.go`, `flow.go`, `centrality.go`, `community.go`, `pagerank.go` — Family B (int-keyed, 15 fns).
- `reality/CLAUDE.md` §2 (zero dep), §3 (no allocs in hot paths), §5 (precision documented).
- `reality/reviews/overnight-400/agents/081-graph-numerics.md` — F8 Louvain modularity gain (different bug from A6 Louvain self-loop).
- `reality/reviews/overnight-400/agents/082-graph-missing.md` — Tier 1+2 sequencing for `excludeNodes` parameter slot.
- `reality/reviews/overnight-400/agents/083-graph-sota.md` — P1 visitor refactor (depends on 084-A8); P2 dense-int port (depends on 084-A1); §1 NetworkX hierarchy anti-port (motivates 084-AA4); §2 igraph value-in/value-out (motivates 084-AA1, AA5).
- Csárdi & Nepusz 2006 — igraph's int-vector convention (the model for Family B).
- Hagberg, Schult, Swart 2008 — NetworkX's mutable G + `add_edge` (the anti-model for AA1, AA5).
- Siek-Lee-Lumsdaine 2002 — Boost.Graph property-map / vertex-descriptor (the model for A8 + 083-P1 visitor refactor).

---

**Headline restated:** Five distinct graph representations in use, three vertex-ID conventions with three sentinel values, four weight-passing conventions, four open-coded undirected-symmetrisation sites, three private re-derivations of `wEdge`, two disjoint algorithm namespaces (Family A string-keyed, Family B int-keyed) with no bridge, and zero documentation of weight semantics / sign constraints / self-loop / multi-edge handling on any of the 15 weighted algorithms. None of this is wrong code per se — every algorithm individually does the right thing for its inputs — it is *unspecified contract* code that lets the two latent bugs found here (Louvain self-loop double-count, Betweenness sigma-inflation on multi-edges) slip past tests that never exercise self-loops or parallel edges. The 8 recommendations (~250 LOC additive + 30 LOC internal migration, fully back-compatible) name the canonical post-cleanup API: two canonical types (`IntAdjacency`, `WeightedAdjacency`), one bridge function pair (`Intern`, `IntAdjacencyFrom`), one symmetrisation helper, one neighbour-iteration helper pair, and a 30-LOC documentation pass that says — for every weighted algorithm — what the weight *means*, what sign it must have, what happens to self-loops, and what happens to parallel edges. The architectural call to keep `reality/graph/` immutable is correct; the seven anti-ports (AA1-AA7) protect that call against well-meaning future PRs that would regress it.
