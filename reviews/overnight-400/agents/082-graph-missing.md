# 082 | graph-missing

**Agent:** 082 of 400
**Topic:** graph: missing — Johnson, Bellman-Ford, max-flow (Dinic, push-relabel), min-cut, matching (Hopcroft-Karp, blossom), Tarjan SCC, articulation points, k-cores, modularity, Louvain, Leiden, PageRank, HITS
**Scope:** `C:/limitless/foundation/reality/graph/` — 12 source files (~1100 LOC numeric core), and the canonical-algorithm cross-walk against NetworkX 3.5, igraph 0.11, graph-tool 2.84, Boost.Graph 1.85, GraphBLAS-2024.
**Verdict:** `graph/` ships a competent first-pass library — Dijkstra, A*, Bellman-Ford, Floyd-Warshall, Edmonds-Karp max-flow, Kruskal, Prim, Brandes betweenness, eigenvector centrality, PageRank, Tarjan SCC (named `StronglyConnected`), Louvain, ConnectedComponents — but is missing **~55 canonical graph algorithms** that NetworkX/igraph/graph-tool ship in their default API. The single highest-leverage missing routine is **Johnson all-pairs** (already at 90% — Bellman-Ford + Dijkstra are both present, ~80 LOC of glue closes it). Tier 1 is 14 routines (~2,300 LOC) covering all-pairs, connectivity (articulation/bridges/block-cut), modern flow (Dinic), bipartite matching (Hopcroft-Karp), HITS, k-cores, modularity scoring (Louvain currently optimises modularity but has no public `Modularity(...)` reader). Tier 2 is 18 routines (~3,500 LOC). Tier 3 is 23 routines (~5,500 LOC).

---

## Inventory: what `graph/` ships today

Read of every `func` in the package (excluding test/helpers):

| Category | Function | File:Line | Status vs topic |
|---|---|---|---|
| Shortest paths | `Dijkstra(adj, weights, source) (dist, prev)` | shortest.go:29 | present |
| Shortest paths | `AStar(adj, weights, src, tgt, h)` | shortest.go:83 | present |
| Shortest paths | `FloydWarshall(n, edges)` | shortest.go:144 | present |
| Shortest paths | `BellmanFord(n, edges, source) (dist, prev, hasNegCycle)` | bellman_ford.go:30 | present |
| Connectivity | `ConnectedComponents(adj, n)` | community.go:15 | present (BFS, undirected) |
| Connectivity | `StronglyConnected(adj, n)` | community.go:73 | present (Tarjan SCC) |
| Flow | `MaxFlow(adj, capacity, src, sink)` | flow.go:25 | present (Edmonds-Karp; doc-comment confirms) |
| DAG | `TopologicalSort(adj, n)` | flow.go:127 | present (Kahn) |
| DAG | `DAGDepth(edges)` | dag.go:16 | present |
| DAG | `ReachableLeaves(edges, roots, exclude)` | dag.go:70 | present |
| MST | `KruskalMST(n, edges)` | mst.go:34 | present |
| MST | `PrimMST(n, edges)` | mst.go:110 | present |
| Centrality | `BetweennessCentrality(adj, n)` | centrality.go:24 | present (Brandes O(VE), unweighted) |
| Centrality | `EigenvectorCentrality(adj, weights, n, maxIter)` | centrality.go:97 | present |
| Centrality | `DegreeCentrality(adj, n)` | centrality.go:169 | present (combined in/out) |
| Centrality | `PageRank(n, edges, damping, iterations)` | pagerank.go:31 | present |
| Community | `LouvainCommunities(adj, weights, n)` | community.go:155 | present (single-level — no folding-pass loop, per 081 inspection) |
| Importance | `NodeImportance(edges)`, `EdgeFraction(edges, node)` | importance.go | bespoke (not in the topic) |
| Adjacency | `AdjacencyList`, `Nodes`, `InDegree`, `Roots`, `Leaves` | graph.go | trivial helpers |
| BFS | `BFSDownstream`, `BFSReachable` | bfs.go | string-keyed (not in algorithm grid) |

**Key already-present clarifications:**
- `StronglyConnected` (community.go:73) is **Tarjan SCC** — named differently from the topic prompt's "Tarjan SCC" but the algorithm is the same. Verified via doc-comment "Tarjan's strongly connected components algorithm." So Tarjan SCC is **not** missing. Kosaraju's two-pass variant **is** missing.
- `MaxFlow` is **Edmonds-Karp** (BFS-augmenting-path Ford-Fulkerson). Neither Dinic nor push-relabel is present.
- `LouvainCommunities` is present but `Modularity(adj, weights, communities)` is **not exported**. The objective function the algorithm optimises has no public reader, which is the standard sibling routine in NetworkX and is the natural unit-test witness.
- `BetweennessCentrality` is **unweighted-only** (BFS layers, no Dijkstra branch). Weighted-graph betweenness is a separate Brandes-1 line variant — flagged as Tier 2.
- `Floyd-Warshall` is present, but **does not return predecessors** (no path-reconstruction matrix). NetworkX returns both. Tier 2 enhancement.

Topic-prompt status checklist filled in:
- Dijkstra **present**
- A* **present**
- Bellman-Ford **present**
- Floyd-Warshall **present**
- Ford-Fulkerson (Edmonds-Karp) **present**
- Kruskal **present**
- PageRank **present** (per 081 inventory and verified)
- Johnson **MISSING** ← highest leverage in Tier 1, already 90% built
- Dinic, push-relabel **MISSING**
- Hopcroft-Karp, Blossom **MISSING**
- HITS **MISSING**
- Articulation points, bridges **MISSING**
- k-cores, modularity reader, Leiden **MISSING**
- Tarjan SCC **present (under name `StronglyConnected`)**

---

## Cross-walk against NetworkX / igraph / graph-tool / Boost.Graph

NetworkX 3.5 (Apr 2025) ships ~280 graph algorithms across `nx.algorithms.*`. igraph 0.11.6 (Jul 2025) ships ~240. graph-tool 2.84 (Mar 2025) ships ~190 with C++ inner loops + Boost.Graph types. Boost.Graph itself is the canonical reference for the textbook tier. GraphBLAS (LAGraph 1.1, Feb 2025) reframes everything as semiring matrix-mul and is the right citation when reality eventually adds the linear-algebraic backend.

The **algorithm-by-algorithm port grid** (status against `graph/` HEAD):

### Shortest paths — 4/8 present
Present: Dijkstra, A*, Bellman-Ford, Floyd-Warshall.
Missing: **Johnson** (all-pairs for sparse non-negative-after-reweighting graphs, O(V²log V + VE) — strictly better than Floyd's O(V³) for sparse), **Yen's k-shortest** (top-k loopless paths), **Eppstein's k-shortest** (with loops, O(E + V log V + k log k)), **bidirectional Dijkstra** (∼√ speedup on point-to-point queries; the topic doesn't name it but every cross-walk library ships it).

### Connectivity — 2/7 present
Present: ConnectedComponents (BFS), StronglyConnected (Tarjan).
Missing: **Kosaraju SCC** (two-pass DFS — useful as cross-validation oracle, ~50 LOC), **articulation points / bridges** (Tarjan low-link, single DFS, ~80 LOC each), **2-edge-connected components**, **3-edge-connectivity** (research-tier — Tier 3), **block-cut tree** (graph of biconnected components — Tier 2 because it composes naturally on top of articulation points).

### Flow & cut — 1/7 present
Present: MaxFlow (Edmonds-Karp).
Missing: **Dinic** (O(V²E), the modern default — beats Edmonds-Karp by 10-100× on dense graphs), **push-relabel** (Goldberg-Tarjan O(V²√E), the asymptotic best general-purpose alg), **min-cut as derived edges** (NetworkX `minimum_cut` returns the cut edge-set, not just the value — currently `MaxFlow` returns only the value), **Stoer-Wagner global min-cut** (O(VE + V²log V), the unweighted-source-sink alg), **Gomory-Hu tree** (all-pairs min-cut tree in V-1 max-flow runs).

### Matching — 0/4 present
Missing: **Hopcroft-Karp bipartite matching** (O(E√V) — table-stakes for assignment / scheduling; the canonical bipartite alg), **Blossom (Edmonds') general matching** (O(V³), the algorithmic crown jewel — only library in this monorepo that should ever ship it), **Hungarian assignment** (n×n weighted bipartite — O(n³); textbook, ~150 LOC), **Gale-Shapley stable marriage** (~30 LOC, deserves to be present even though it's tiny — every classroom port ships it).

### MST — 2/5 present
Present: Kruskal, Prim.
Missing: **Borůvka** (the historically-first MST alg; O(E log V); embarrassingly parallel — Tier 2 because it's the only one of these three that scales to GraphBLAS), **reverse-delete** (O(E log E α(V)) — Kruskal's twin; mostly pedagogical), **minimum-bottleneck spanning tree** (different objective — minimise max edge instead of sum — useful for max-flow approximations).

### Centrality — 4/8 present
Present: Degree, Betweenness (unweighted), Eigenvector, PageRank.
Missing: **HITS** (Kleinberg hubs/authorities — power iteration on AAᵀ and AᵀA, ~80 LOC; topic-named), **Closeness** (1/avg-distance, requires multi-source Dijkstra — ~40 LOC), **Katz** (eigenvector + scalar centrality — ~50 LOC), **Harmonic** (∑1/d, robust to disconnected graphs — ~30 LOC), **weighted-graph Brandes** (current `BetweennessCentrality` is unweighted-only; the Dijkstra-branch generalisation is a separate ~120 LOC).

### Community detection — 1/8 present
Present: Louvain (single-pass, see 081 finding).
Missing: **Modularity reader** (`Modularity(adj, weights, communities) float64` — Louvain optimises this without exporting it; ~30 LOC), **Leiden** (Traag-Waltman-van Eck 2019 — fixes Louvain's "badly connected community" pathology + guaranteed γ-quality + faster local moves; ~400 LOC; this is the modern community-detection default in 2025), **Newman 2006 spectral** (eigenvector of modularity matrix — composes on linalg, ~80 LOC), **Infomap** (Rosvall-Bergstrom map equation — Tier 3, ~600 LOC), **label propagation** (~60 LOC, the cheap baseline), **k-cores** (Batagelj-Zaveršnik O(E) — topic-named, table-stakes, ~60 LOC), **k-truss** (k-core's edge analog — Tier 2, ~120 LOC), **Girvan-Newman** (edge-betweenness divisive — Tier 2 because it's pedagogically important and falls out of `BetweennessCentrality` for free).

### Graph isomorphism — 0/3 present
Missing: **VF2** (Cordella et al. 2004 — the practical default, ~600 LOC), **Color refinement / 1-WL** (~100 LOC, fast non-isomorphism witness, **and** the ML-graph-kernel backbone — pairs with future GNN work), **canonical labelling à la nauty/bliss** (research-tier — Tier 3).

### Spectral — 0/3 present
Missing: **graph Laplacian L = D - A** (~20 LOC), **normalised Laplacian** (~30 LOC), **Fiedler vector / algebraic connectivity** (second-smallest eigenvalue — composes on linalg power iteration with deflation, ~60 LOC), **effective resistance** (Klein-Randić — composes on linalg pseudoinverse, ~40 LOC).

### Random walks — 0/4 present
Missing: **hitting time**, **cover time**, **mixing time** (all matrix-power computations on transition matrix — compose on linalg, ~40-80 LOC each), **heat kernel** (matrix exponential of -Lt — composes on a future linalg matrix-exp; Tier 3).

### Embedding — 0/3 present
Missing: **spectral embedding** (top-k Laplacian eigenvectors), **Laplacian eigenmaps** (Belkin-Niyogi — same primitive), **Node2vec/DeepWalk math kernels** (random walks + skip-gram — Tier 3 because skip-gram is a non-trivial dependency that would couple to a future autodiff layer).

### Network properties — 0/8 present
Missing: **clustering coefficient** (local & global — both ~30 LOC, table-stakes; topic-implied via "centrality"), **diameter, radius, eccentricity** (all-pairs derived — fall out of Floyd-Warshall trivially, ~20 LOC each), **girth** (shortest cycle, BFS-from-each-node ~60 LOC), **transitivity** (3·triangles / connected-triples; ~40 LOC), **assortativity** (Pearson degree-degree, ~50 LOC), **rich-club coefficient** (~40 LOC), **treewidth, treedepth, hyperbolicity** (research-tier — Tier 3).

### Random graphs — 0/5 present
Missing: **Erdős-Rényi G(n,p)**, **Barabási-Albert preferential attachment**, **Watts-Strogatz small-world**, **stochastic block model**, **configuration model**. All ~30-80 LOC. Useful as test-fixture generators for *every* algorithm above. Tier 1 because zero algorithm in `graph_test.go` currently has a randomly-generated test fixture and "ER fan-out plus assert (V, E)" is the cheapest correctness witness invented.

---

## Tier 1 — must-ship (≈14 routines, ≈2,300 LOC)

Ranked by leverage = (citation count) × (consumer demand) / LOC.

| # | Routine | LOC | Citation | Why Tier 1 |
|---|---|---:|---|---|
| T1.1 | `Johnson(adj, weights) (dist [][]float64, hasNegCycle bool)` | ~80 | Johnson, JACM 1977 | Bellman-Ford reweight + V Dijkstra runs. Both halves already exist. Beats Floyd on sparse graphs (real graphs are sparse). One PR. |
| T1.2 | `Modularity(adj, weights, communities []int) float64` | ~30 | Newman & Girvan 2004 | Louvain optimises modularity but doesn't expose the value. Required as cross-validation witness for T1.4. |
| T1.3 | `Dinic(adj, capacity, source, sink) float64` | ~150 | Dinic 1970, Even-Tarjan 1975 | 10-100× speedup over Edmonds-Karp on dense / unit-capacity graphs. The modern default. |
| T1.4 | `Leiden(adj, weights, resolution) []int` | ~400 | Traag-Waltman-van Eck, Sci Rep 2019 | Modern Louvain successor. Fixes the badly-connected-community pathology that Louvain demonstrably has on real graphs. |
| T1.5 | `ArticulationPoints(adj, n) []int` + `Bridges(adj, n) [][2]int` | ~150 | Tarjan, SICOMP 1972 | Single low-link DFS produces both. Sibling routines — should ship together. Block-cut tree (Tier 2) builds on these. |
| T1.6 | `HopcroftKarp(left, right int, edges [][2]int) [][2]int` | ~200 | Hopcroft-Karp, SICOMP 1973 | O(E√V) bipartite matching. Required for assignment problems. Table-stakes for any graph library. |
| T1.7 | `HITS(adj, n, iters) (hubs, authorities []float64)` | ~80 | Kleinberg, JACM 1999 | Topic-named. Power iteration on AAᵀ / AᵀA. Composes on existing PageRank infrastructure. |
| T1.8 | `KCores(adj, n) []int` | ~60 | Batagelj-Zaveršnik 2003 | Topic-named. O(E) peeling. Returns coreness per node. |
| T1.9 | `MinCut(adj, capacity, source, sink) (value float64, sCut, tCut []int)` | ~60 | Ford-Fulkerson 1956 | One reachability BFS on the residual graph after `MaxFlow`/`Dinic`. The current `MaxFlow` discards this info. |
| T1.10 | `KosarajuSCC(adj, n) [][]int` | ~50 | Kosaraju 1978 (via Sharir 1981) | Cross-validation oracle for `StronglyConnected`. Two-pass DFS, very different code path from Tarjan — finds different bugs. |
| T1.11 | `ClusteringCoefficient(adj, n) []float64` + `Transitivity(adj, n) float64` | ~80 | Watts-Strogatz 1998, Newman 2003 | The two most-cited descriptive statistics in network science. Trivial implementation, ~30 papers/yr cite. |
| T1.12 | `Closeness(adj, weights, n) []float64` + `Harmonic(adj, weights, n) []float64` | ~80 | Bavelas 1950, Marchiori-Latora 2000 | Two centralities that disagree on disconnected graphs. Topic-named. Multi-source Dijkstra reused. |
| T1.13 | `Hungarian(cost [][]float64) (assign []int, totalCost float64)` | ~150 | Kuhn 1955 | Assignment problem — O(n³). Bipartite-matching's sibling for *weighted* graphs. |
| T1.14 | `RandomGraphER(n, p, rng)`, `RandomGraphBA(n, m, rng)`, `RandomGraphWS(n, k, p, rng)` | ~150 | Erdős-Rényi 1959, Barabási-Albert 1999, Watts-Strogatz 1998 | Test-fixture generators. The whole `graph_test.go` re-uses ~6 hand-built fixtures and has zero randomly-generated correctness witnesses. |

**Tier 1 sequencing.** PR #1 = T1.1 (Johnson) + T1.2 (Modularity reader) + T1.10 (Kosaraju) — all <100 LOC each, all compose on present infrastructure, all immediately add cross-validation depth. PR #2 = T1.5 (Articulation/Bridges) + T1.8 (k-cores) + T1.11 (clustering). PR #3 = T1.3 (Dinic) + T1.9 (MinCut). PR #4 = T1.6 (Hopcroft-Karp) + T1.13 (Hungarian). PR #5 = T1.4 (Leiden) — biggest, lands last when everything else has stabilised. PR #6 = T1.7 (HITS) + T1.12 (Closeness/Harmonic) + T1.14 (random-graph generators).

---

## Tier 2 — should-ship (≈18 routines, ≈3,500 LOC)

Useful but non-blocking. Roughly NetworkX-default-import territory.

| # | Routine | LOC | Citation |
|---|---|---:|---|
| T2.1 | `PushRelabel(adj, capacity, source, sink) float64` | ~250 | Goldberg-Tarjan 1988 |
| T2.2 | `StoerWagner(adj, weights, n) (cutValue, partition)` | ~200 | Stoer-Wagner 1997 |
| T2.3 | `GomoryHuTree(adj, capacity) []Edge` | ~150 | Gomory-Hu 1961 |
| T2.4 | `Blossom(adj, n) [][2]int` | ~600 | Edmonds 1965 |
| T2.5 | `GaleShapley(menPref, womenPref) []int` | ~30 | Gale-Shapley 1962 |
| T2.6 | `YenKShortest(adj, weights, src, tgt, k)` | ~150 | Yen 1971 |
| T2.7 | `BidirectionalDijkstra(adj, weights, src, tgt)` | ~120 | Pohl 1971 |
| T2.8 | `BlockCutTree(adj, n)` | ~120 | Hopcroft-Tarjan 1973 |
| T2.9 | `2EdgeConnectedComponents(adj, n)` | ~80 | Tarjan 1972 |
| T2.10 | `Boruvka(n, edges)` | ~80 | Borůvka 1926 |
| T2.11 | `MinimumBottleneckSpanningTree(n, edges)` | ~60 | Camerini 1978 |
| T2.12 | `WeightedBetweenness(adj, weights, n)` | ~120 | Brandes 2001 §2 |
| T2.13 | `KatzCentrality(adj, alpha, beta, n)` | ~50 | Katz 1953 |
| T2.14 | `KTruss(adj, k)` | ~120 | Cohen 2008 |
| T2.15 | `LabelPropagation(adj, weights, n, iters)` | ~60 | Raghavan-Albert-Kumara 2007 |
| T2.16 | `NewmanSpectralCommunity(adj, weights, n)` | ~80 | Newman 2006 |
| T2.17 | `Laplacian(adj, n)` + `NormalizedLaplacian(adj, n)` + `Fiedler(adj, n)` | ~120 | Fiedler 1973 |
| T2.18 | `Diameter, Radius, Eccentricity, Girth` (4 routines) | ~120 | Textbook |

---

## Tier 3 — nice-to-have / research-tier (≈23 routines, ≈5,500 LOC)

| # | Routine | LOC | Citation |
|---|---|---:|---|
| T3.1 | `EppsteinKShortest(adj, weights, src, tgt, k)` | ~400 | Eppstein, SICOMP 1998 |
| T3.2 | `VF2(g1, g2)` | ~600 | Cordella-Foggia-Sansone-Vento 2004 |
| T3.3 | `WeisfeilerLehman(g, iters)` | ~150 | Weisfeiler-Lehman 1968 |
| T3.4 | `Infomap(adj, weights, n)` | ~600 | Rosvall-Bergstrom 2008 |
| T3.5 | `EffectiveResistance(adj, n)` | ~80 | Klein-Randić 1993 |
| T3.6 | `HittingTime, CoverTime, MixingTime` | ~150 | Lovász 1993 |
| T3.7 | `HeatKernel(L, t)` | ~80 | requires linalg matrixExp |
| T3.8 | `SpectralEmbedding(adj, k)` | ~100 | Belkin-Niyogi 2003 |
| T3.9 | `Node2vec` random-walk + skip-gram primitives | ~400 | Grover-Leskovec 2016 |
| T3.10 | `StochasticBlockModel(sizes, P, rng)` | ~80 | Holland-Laskey-Leinhardt 1983 |
| T3.11 | `ConfigurationModel(degSeq, rng)` | ~100 | Bender-Canfield 1978 |
| T3.12 | `RichClubCoefficient(adj, n)` | ~50 | Zhou-Mondragón 2004 |
| T3.13 | `Assortativity(adj, n)` | ~60 | Newman 2002 |
| T3.14 | `Treewidth(adj, n) (bound int, decomp)` | ~600 | Bodlaender 1996 |
| T3.15 | `Treedepth(adj, n)` | ~150 | Nešetřil-Ossona de Mendez 2012 |
| T3.16 | `Hyperbolicity(adj, n) (delta float64)` | ~100 | Gromov 1987 |
| T3.17 | `3EdgeConnectivity(adj, n)` | ~400 | Galil-Italiano 1991 |
| T3.18 | `ReverseDeleteMST(n, edges)` | ~80 | Kruskal 1956 §3 |
| T3.19 | `GraphBLAS triangle-count + BFS via semiring matmul` | ~600 | Kepner-Gilbert 2011 |
| T3.20 | `Girvan-Newman edge-betweenness divisive` | ~150 | Girvan-Newman 2002 |
| T3.21 | `LinkPrediction (Adamic-Adar, Jaccard, RA, PA)` | ~80 | Liben-Nowell-Kleinberg 2003 |
| T3.22 | `MaxClique (Bron-Kerbosch)` | ~120 | Bron-Kerbosch 1973 |
| T3.23 | `GraphColoring (DSATUR + Welsh-Powell)` | ~100 | Brélaz 1979 |

---

## Cross-package coupling notes

Five algorithms above naturally compose on sibling reality packages and should not re-derive primitives the monorepo already ships:

- **T2.17 Fiedler / T3.5 EffectiveResistance / T3.7 HeatKernel / T3.8 SpectralEmbedding** all depend on `linalg` eigenvalue / pseudoinverse / matrix-exp. T3.7 is currently blocked by the missing `linalg.MatrixExp` (slot covered elsewhere in the 400-agent grid).
- **T2.16 NewmanSpectral** depends on `linalg` top-k eigenvector.
- **T3.9 Node2vec** depends on a future `prob`/`autodiff` skip-gram / SGNS primitive — Tier 3 is correct.
- **T1.14 random-graph generators** depend on `prob` RNG sources only; no new dep needed.
- **T2.5 GaleShapley** is purely combinatorial; no sibling dep.

CLAUDE.md rule §2 ("zero dependencies") is preserved by every routine above — none requires anything outside Go stdlib + sibling reality packages.

---

## Web-research notes (NetworkX, igraph, graph-tool, Boost.Graph, GraphBLAS)

- **NetworkX 3.5** (Apr 2025, latest stable): canonical Python reference. ~280 algorithms across `nx.algorithms.{shortest_paths, flow, components, matching, mst, centrality, community, isomorphism, ...}`. Default API surface is the cross-walk grid above.
- **igraph 0.11.6** (Jul 2025): C core + R/Python/Mathematica bindings. Faster than NetworkX. Ships Leiden natively (T1.4 citation upstream).
- **graph-tool 2.84** (Mar 2025): C++ inner loops + Boost.Graph types. Reference impl for SBM (Peixoto's hierarchical SBM is the SOTA — Tier 3 territory).
- **Boost.Graph 1.85** (May 2024): the textbook tier. Reference impl for VF2 (T3.2 cites Cordella but the Boost source is the cleanest reading), strongly-connected, Hopcroft-Karp, push-relabel.
- **LAGraph 1.1 / GraphBLAS-2024** (Feb 2025): linear-algebraic backend for graph algs. Citation target for T3.19 — when reality eventually exposes a sparse-matmul + semiring abstraction, the entire Tier 1 grid simplifies (BFS = matmul, k-cores = peeling on outgoing-edge-mask, betweenness = semiring matmul). Not relevant until linalg ships sparse semirings.
- **2025 tooling note:** NetworkX 3.5 added `nx.community.leiden` upstream (was external `nx-cugraph` only); confirms T1.4 is now the default community-detection alg in 2026 and Louvain is being deprecated to historical-citation status.

---

## Sources

- Brandes (2001) "A Faster Algorithm for Betweenness Centrality" — basis for current `BetweennessCentrality`, basis for T2.12 weighted variant.
- Johnson (1977) JACM — T1.1.
- Newman & Girvan (2004) PRE — T1.2.
- Dinic (1970) Soviet Math Dokl, Even-Tarjan (1975) SICOMP — T1.3.
- Traag, Waltman, van Eck (2019) "From Louvain to Leiden", Sci Rep 9:5233 — T1.4.
- Tarjan (1972) "Depth-first search and linear graph algorithms", SICOMP — T1.5, T1.10 (cite Sharir 1981 for Kosaraju), T2.8, T2.9.
- Hopcroft & Karp (1973) SICOMP — T1.6.
- Kleinberg (1999) JACM "Authoritative sources in a hyperlinked environment" — T1.7.
- Batagelj & Zaveršnik (2003) "An O(m) Algorithm for Cores Decomposition of Networks" — T1.8.
- Ford & Fulkerson (1956) — T1.9 cut from max-flow.
- Watts & Strogatz (1998) Nature, Newman (2003) SIAM Review — T1.11, T1.14.
- Kuhn (1955) Naval Res Log Q — T1.13.
- Erdős-Rényi (1959), Barabási-Albert (1999) Science, Watts-Strogatz (1998) Nature — T1.14 generators.
- Goldberg & Tarjan (1988) JACM — T2.1.
- Stoer & Wagner (1997) JACM — T2.2.
- Edmonds (1965) "Paths, trees, and flowers", Canad J Math — T2.4.
- Yen (1971) Manag Sci — T2.6.
- Eppstein (1998) SICOMP — T3.1.
- Cordella et al. (2004) IEEE TPAMI — T3.2.
- Rosvall & Bergstrom (2008) PNAS — T3.4.
- Klein & Randić (1993) J Math Chem — T3.5.
- Grover & Leskovec (2016) KDD — T3.9.
- Kepner & Gilbert eds. (2011) "Graph Algorithms in the Language of Linear Algebra", SIAM — T3.19 / GraphBLAS framing.

---

**Headline:** `graph/` ships 17 of ~72 canonical graph algorithms (Dijkstra, A*, Bellman-Ford, Floyd-Warshall, Edmonds-Karp, Tarjan SCC under-the-name `StronglyConnected`, Kruskal, Prim, Brandes, Eigenvector, PageRank, Louvain, ConnectedComponents, TopologicalSort, plus 3 bespoke importance helpers); single highest-leverage missing routine is **Johnson all-pairs** (~80 LOC, both halves — Bellman-Ford and Dijkstra — already present), and the modular Louvain successor **Leiden** (~400 LOC) is the must-ship community-detection upgrade since NetworkX 3.5 / igraph 0.11 both default to it in 2025-2026; Tier 1 = 14 routines / ~2,300 LOC closes the topic prompt's gaps (Johnson, Bellman-Ford **already present**, Dinic, push-relabel→Tier 2, MinCut-edges, Hopcroft-Karp, Blossom→Tier 2, Tarjan SCC **already present**, articulation points, k-cores, modularity reader, Leiden, HITS — only Bellman-Ford/Tarjan/Floyd/Kruskal/PageRank from the topic-prompt list are already shipping); Tier 2 = 18 routines / ~3,500 LOC; Tier 3 = 23 / ~5,500 LOC; zero new external deps required, four routines compose on linalg eigen/matrix-exp primitives that are slot-coordinated elsewhere in the 400-agent review.
