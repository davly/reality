# 083 | graph-sota

**Agent:** 083 of 400
**Topic:** graph: compare with NetworkX, igraph, graph-tool, Gephi, Boost.Graph (+ GraphBLAS/LAGraph, NetworKit, SNAP, nauty/bliss, Pregel/GraphX)
**Scope:** survey what each SOTA reference library bets on, the *single* engineering trick that defines it, and which of those tricks are realistically portable to `reality/graph/` (Go, zero-dep, ~1100 LOC numeric core, 17 algorithms shipping). Companion to 081 (numerical hygiene, six liabilities) and 082 (algorithm gap-list, 14/18/23 Tier 1/2/3 routines). This file deliberately stays orthogonal: not "what is missing" (that's 082) and not "is the math right" (that's 081), but **"what design idea each library is famous for, and which ones port to Go-zero-dep."**

---

## TL;DR

Ten reference points — NetworkX 3.5, igraph 0.11, graph-tool 2.84, Gephi 0.10, Boost.Graph 1.85, GraphBLAS/LAGraph, NetworKit 11.0, SNAP/snap.py, nauty/bliss, Pregel/GraphX/Giraph — split into **four design philosophies** that `reality/graph/` cannot blend without breaking either CLAUDE.md §2 (zero-dep) or §3 (no allocs hot path):

1. **"Algorithm encyclopedia, ergonomics first"** (NetworkX, Gephi). Cover every algorithm in the textbook; pay any constant-factor for clarity. Headline trick: **dict-of-dict adjacency** (NetworkX) — every node is a Python dict key, every edge a nested dict — so the API reads like math (`G[u][v]['weight']`) at the cost of 5–50× slowdown vs C-backed peers. Portable lesson: API legibility wins adoption; the perf cost is irrelevant to a *correctness-witness* library, which is what `reality/` is. **Don't port** the dict-of-dict; **do port** the convention `G.has_edge(u, v)` / `G.degree(u)` / `G.neighbors(u)` as the named operations the library guarantees.

2. **"C kernel + thin language binding"** (igraph, NetworKit, SNAP). Tight C/C++ inner loops, swap-in language wrappers, ship one binary that's 50–500× faster than NetworkX. Headline trick: **integer-vector representation** (igraph's `igraph_vector_int_t` everywhere — node IDs are dense `int` arrays, never strings/objects). Portable lesson: dense `int` vertex IDs are the only representation that works for Pistachio's 60-FPS allocation-free constraint. The current `reality/graph/` API mixes string-keyed BFS (`bfs.go`) with int-indexed shortest-paths (`shortest.go`); the int-indexed half is the one to keep, and the string half should be a *thin convenience layer* that calls into it after a one-shot intern.

3. **"Type-system as algorithm correctness proof"** (Boost.Graph, graph-tool). Every algorithm is templated on a concept (`VertexListGraph`, `IncidenceGraph`, `BidirectionalGraph`); the compiler refuses to instantiate Dijkstra against a graph that doesn't model `IncidenceGraph`. Headline trick: **visitor pattern** — the algorithm is a skeleton; the user supplies a `BFSVisitor` with hooks (`discover_vertex`, `tree_edge`, `examine_edge`) that get called at the right moment. Portable lesson: in Go, replace concepts-as-templates with **interfaces** (already idiomatic) and visitors-as-classes with **callbacks-as-function-values**. The trick is *not* the C++-template machinery (doesn't port) but the *factoring*: distinguish "graph traversal" (skeleton) from "what to do at each node" (callback). `reality/graph/` currently bakes the work into each algorithm; refactoring BFS/DFS to expose visitor callbacks is ~30 LOC and unlocks a dozen downstream algorithms (articulation points, bridges, biconnected components, all topological-sort variants) as 10-LOC visitors.

4. **"Linear algebra is the abstraction"** (GraphBLAS / LAGraph, partly graph-tool). Every graph algorithm rewritten as sparse-matrix multiplication on a semiring. Headline trick: **adjacency matrix + semiring choice = algorithm**. `(min, +)` semiring on adjacency = Bellman-Ford. `(or, and)` = reachability. `(any, second)` = BFS frontier. Portable lesson: **wait**. GraphBLAS is the right long-term abstraction but requires a sparse-matrix kernel in `linalg/` first (CSR + masked matmul, ~600 LOC) and isn't worth pre-paying. Note for the planner: when `linalg/` ships sparse semirings (slot-coordinated elsewhere), seven algorithms in `graph/` simplify to ~20 LOC each and gain a free parallel implementation.

The **four trick-by-trick ports worth doing** in priority order:

- **igraph dense-int vertex IDs as the canonical representation** (Csárdi-Nepusz 2006): `[]int` adjacency lists with vertex IDs `0..n-1`. Already the convention in `shortest.go`/`mst.go`/`flow.go`; just needs to be promoted to *the* convention with a one-time `Intern(strings []string) []int` adapter for callers who have string IDs. ~50 LOC.
- **Boost.Graph visitor refactor of BFS/DFS** (Siek-Lee-Lumsdaine 2002): expose `BFSVisit(adj, src, visitor func(VisitorEvent))` as the primitive; build `ConnectedComponents`, `BipartiteCheck`, `ArticulationPoints`, `Bridges`, `TopologicalSort` as 10–30 LOC visitors. ~80 LOC primitive + ~150 LOC of derived algs = whole set 082-T1.5/T1.10/T1.11 closed at the cost of one refactor.
- **Brandes-style "lazy" Dijkstra primitive** (graph-tool's pattern): the inner Dijkstra loop is exposed as a generator-like callback `(node, dist) -> bool` so the same primitive backs single-source shortest paths, weighted Brandes betweenness (082-T2.12), Closeness (082-T1.12), Harmonic (082-T1.12), and bidirectional Dijkstra (082-T2.7). ~50 LOC of refactor on `shortest.go`'s existing 50 LOC of Dijkstra body.
- **NetworKit "label-correcting vs label-setting" naming convention** (Staudt-Sazonovs-Meyerhenke 2016): name `Dijkstra` as label-setting and `BellmanFord` as label-correcting in doc-comments; document the equivalence under non-negative weights; expose a unified `ShortestPaths(adj, weights, src, opts)` constructor that picks based on weight signs. ~10 LOC, ergonomic only.

**Three tricks not worth porting:**

- **NetworkX's dict-of-dict-of-dict adjacency.** Idiomatic in Python, anti-idiomatic in Go (would mean `map[int]map[int]map[string]float64` and 4× the allocs of the current `[][]int` + `map[[2]int]float64`). The *API* legibility port (named methods) is good; the *representation* port is wrong.
- **Boost.Graph's full template-concept machinery.** Go has interfaces, not templates; the equivalent is `interface { OutNeighbors(v int) []int }` and that's already implicit in `reality/graph/`. The `graph_traits<G>::out_edge_iterator` style is a C++ idiom that doesn't translate.
- **GraphBLAS semiring-everything reframe.** The right answer eventually but premature now. Wait for `linalg/` sparse + semiring, then revisit. Building an in-package toy semiring layer just for `graph/` would duplicate ~400 LOC of what `linalg/` will need to ship.

---

## Library-by-library: headline algorithm, engineering trick, portability

### 1. NetworkX 3.5 (Hagberg, Schult, Swart — LANL → community)

| | |
|---|---|
| **Headline algorithm** | None *specifically*: ~280 algorithms across `nx.algorithms.{shortest_paths, flow, components, matching, mst, centrality, community, isomorphism, bipartite, ...}` is the headline. Recent (2024–25) addition: `nx.community.leiden` upstream after years as external `nx-cugraph` only. |
| **Architectural trick** | **Dict-of-dict-of-dict adjacency** (`G._adj[u][v][key] = data`) plus a `Graph` / `DiGraph` / `MultiGraph` / `MultiDiGraph` four-type hierarchy. Every node can be any hashable Python object (string, int, frozenset, custom class). Every edge can carry an arbitrary `data` dict. Algorithms are written against the abstract `G.neighbors(u)` / `G[u][v]` / `G.degree(u)` API, never the storage. The 5–50× slowdown vs igraph/graph-tool is the explicit price paid for "any hashable as node." |
| **What works because of it** | Onboarding is the lowest in any graph library. Code reads like the math: `for u, v in G.edges()`, `nx.shortest_path(G, source, target)`, `nx.betweenness_centrality(G)`. The encyclopedia coverage works because every contributor can prototype a new algorithm against the abstract API in ~30 LOC without worrying about storage. |
| **Zero-dep port for reality** | **Convention port only, no code port.** Adopt the *named-operation* API surface — `Neighbors(adj, u)`, `Degree(adj, u)`, `HasEdge(adj, u, v)` — so a NetworkX user can read `reality/graph/` source and recognize every operation. Already partly there: `graph.go` exposes `AdjacencyList`, `Nodes`, `InDegree`, `Roots`, `Leaves`. **Don't port** the dict-of-dict storage — Go's `[][]int` is correct for the perf budget. **Don't port** the four-type hierarchy (`Graph` / `DiGraph` / `MultiGraph` / `MultiDiGraph`) — for `reality/graph/` the directed-vs-undirected distinction is a per-algorithm input contract, not a type. The MultiGraph case (parallel edges) is real but rare; document that current adjacency representations *do* allow parallel edges and that algorithms must handle them per-call (Dijkstra: take the min; Kruskal: keep both; PageRank: aggregate weights). |
| **What we should not copy** | The `nx.algorithms.X.Y.Z` deeply-nested module path (NetworkX has `nx.algorithms.flow.maxflow`, `nx.algorithms.flow.mincost`, `nx.algorithms.flow.boykovkolmogorov` — over-engineered for `reality/`). Flat package, one file per algorithm-family (current layout: `shortest.go`, `mst.go`, `flow.go`, `centrality.go`) is correct. Also don't copy NetworkX's habit of shipping multiple algorithms behind one function name with an `algorithm=` kwarg dispatcher — Go's no-default-arg style favors `Dinic(...)` and `EdmondsKarp(...)` as separate functions, which is also what 082-T1.3 calls for. |
| **Citation provenance** | Hagberg, Schult, Swart 2008 SciPy Proc; latest stable v3.5 Apr 2025; Leiden upstream merge late 2024. ~280 algorithms enumerated at https://networkx.org/documentation/stable/reference/algorithms/index.html. |

### 2. igraph 0.11 (Csárdi, Nepusz — KTH → community)

| | |
|---|---|
| **Headline algorithm** | **Leiden** (Traag-Waltman-van Eck 2019) — igraph was the first major lib to ship it, before NetworkX merged it upstream. Plus a faithful, fast Reingold-Tilford layout, multilevel community detection (Louvain) since 2008, and one of the few libs with usable graph-isomorphism backends (calls bliss internally). |
| **Architectural trick** | **Integer-vector representation everywhere** (`igraph_vector_int_t`, `igraph_vector_t`, `igraph_matrix_t`). Vertex IDs are dense `int` `0..n-1`; edges are pairs of ints; weights are a parallel `double` vector. The whole library is a layer over ~30 vector/matrix/heap primitives. The C-core is then bound to R, Python, Mathematica via thin FFI shims. Zero allocation per algorithm-call beyond the result vector — the API is `igraph_dijkstra(graph, &result, ...)` where `result` is caller-owned. |
| **What works because of it** | 50–500× faster than NetworkX on the same graphs because the inner loops are dense-int-indexed array walks with no Python dict overhead. The thin-binding model means the C core is the *single source of truth* and bindings stay in lockstep — same algorithms, same names, same return shapes across Python and R. Bliss-backed isomorphism actually works for graphs of 10⁵ nodes (NetworkX's pure-Python VF2 is unusable past 10² nodes). |
| **Zero-dep port for reality** | **Direct port, top priority.** The dense-int-vertex-IDs convention is *the* design that makes Pistachio's 60-FPS allocation-free constraint achievable. Already the convention in 13 of 17 algorithms in `reality/graph/`; promote it to *the* documented convention. The `result-buffer-as-out-param` API (caller-owned slices) is the right Go-equivalent of CLAUDE.md §3 "no allocations in hot paths" — refactor over time so that hot-path callers pass `dist []float64, prev []int` buffers that get reused. ~80 LOC of API additions: `DijkstraInto(adj, weights, src int, dist []float64, prev []int)` alongside the existing allocating `Dijkstra(...)`. **Specifically don't port** igraph's `igraph_t` opaque-struct-with-attributes design — Go's open structs + `[][]int` adjacency is more honest and lets users write their own iteration when needed. |
| **What we should not copy** | The error-code return convention (`igraph_integer_t igraph_dijkstra(...)` returning `IGRAPH_SUCCESS` / `IGRAPH_ENOMEM` / etc.). C convention; Go has multi-return + error type and that's already idiomatic. Also don't copy igraph's habit of shipping every algorithm in *both* directed and undirected variants as separate functions (`igraph_betweenness` vs `igraph_betweenness_directed`) — for `reality/`, the input adjacency tells you which it is. |
| **Citation provenance** | Csárdi & Nepusz 2006 InterJournal CX.18; v0.11.6 Jul 2025; Leiden via Traag-Waltman-van Eck 2019 Sci Rep 9:5233. https://igraph.org/. |

### 3. graph-tool 2.84 (Tiago Peixoto — Univ. Bremen)

| | |
|---|---|
| **Headline algorithm** | **Hierarchical stochastic block models** (Peixoto 2014, 2017) — the only library shipping principled Bayesian community detection with model-selection (no resolution-parameter tuning). Plus C++-template-speed inner loops on every algorithm, plus a maintained Python binding. |
| **Architectural trick** | **C++ Boost.Graph types under the hood** with Python `PropertyMap`s as the user-facing data. Every vertex / edge property (weight, label, color) is a `PropertyMap` — a typed array indexed by vertex/edge. Graph mutations and property updates are O(1); the property-map-as-array idiom is exactly the dense-int-vertex convention from igraph but generalized to per-attribute storage. |
| **What works because of it** | Often the fastest of the four (NetworkX / igraph / NetworKit / graph-tool) for serial algorithms because the C++ template-instantiation collapses property accesses to direct array reads. SBM and Bayesian inference are realistically usable on graphs of 10⁵–10⁶ nodes — research papers actually use it on real datasets. |
| **Zero-dep port for reality** | **Convention port only.** The PropertyMap idea translates to Go as **separate `[]float64` / `[]int` / `[]string` parallel arrays indexed by vertex ID**: `degree[u]`, `weight[u]`, `community[u]`. Already the de-facto pattern in `reality/graph/centrality.go` and `pagerank.go`. Document it as a convention. **Don't port** the C++-template instantiation tricks — Go has no equivalent and the dense-int-array model already gets ~80% of the cache-locality benefit. The SBM algorithm itself is 082-Tier-3 territory; the *property-map convention* is the portable lesson. |
| **What we should not copy** | The Python-via-Boost-via-C++ build chain (`graph-tool` is famously hard to install because of this). Stay pure-Go. Also don't copy graph-tool's `Graph()` constructor that takes a `directed=True` kwarg and stores it; for `reality/`, directedness is per-algorithm input semantics. |
| **Citation provenance** | Peixoto 2014 PRX 4:011047 (degree-corrected SBM); Peixoto 2017 ARXIV 1705.10225 (nonparametric SBM); v2.84 Mar 2025. https://graph-tool.skewed.de/. |

### 4. Gephi 0.10 (Bastian, Heymann, Jacomy — Univ. Paris)

| | |
|---|---|
| **Headline algorithm** | **ForceAtlas2 layout** (Jacomy et al. 2014 PLoS ONE) — a force-directed layout that scales to 10⁵ nodes interactively because of the Barnes-Hut spatial index for repulsive-force approximation. Plus built-in modularity/Louvain, PageRank, betweenness — all wrapped around an interactive viz. |
| **Architectural trick** | **Visualization-driven streaming graph model**: the graph is a live mutable object that emits change-events to a viewport. Algorithms are wrapped as `Statistics` plugins that compute attributes (PageRank, betweenness, modularity) and push them back as vertex colors / sizes. The Barnes-Hut layout uses a quadtree to approximate `O(n²)` repulsive forces in `O(n log n)`. |
| **What works because of it** | Interactive analysis on graphs that visualize at 10⁴–10⁵ nodes. Force-Atlas2's "linlog" mode (logarithmic attraction, linear repulsion) is the only force-directed algorithm that produces *legible* layouts of community-rich graphs without manual tuning. Nobody else replicates this without copying the FA2 paper. |
| **Zero-dep port for reality** | **Single-algorithm port: ForceAtlas2 layout** (~250 LOC including Barnes-Hut quadtree). Belongs in 082-Tier-2 territory as `ForceAtlas2(adj, weights, n, iters, opts) [][2]float64` returning x/y coordinates. The Barnes-Hut tree composes on the existing `geometry/` k-d / quadtree infrastructure once 077-Tier-1 lands; until then, brute-force `O(n²)` FA2 is fine for n < 1000. **Don't port** the interactive event-streaming model — that's an application-layer concern (Pistachio handles its own viz state); `reality/graph/` should ship the layout-coordinates-as-pure-function. |
| **What we should not copy** | The Java + NetBeans Platform UI architecture (Gephi is shipped as a desktop app, not a library). The plugin-as-class design for algorithms (Go favors functions). The change-event / observer pattern for graph mutation (caller-side concern). |
| **Citation provenance** | Bastian, Heymann, Jacomy 2009 ICWSM; Jacomy, Venturini, Heymann, Bastian 2014 PLoS ONE 9:e98679 (FA2); v0.10 stable since Sep 2022. https://gephi.org/. |

### 5. Boost.Graph 1.85 (Siek, Lee, Lumsdaine — Indiana Univ. → community)

| | |
|---|---|
| **Headline algorithm** | The full classical-algorithms tier (Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson, Tarjan SCC, Kosaraju SCC, biconnected components, articulation points, edge connectivity, max-flow Edmonds-Karp/Boykov-Kolmogorov/push-relabel, Hopcroft-Karp, max-cardinality matching, betweenness, eigenvector, MST Kruskal/Prim) — the textbook reference, often *the* C++ source other libraries vendor. |
| **Architectural trick** | **Concept-based generic programming + visitor pattern.** Every algorithm is templated on a `Graph` concept (`VertexListGraph`, `IncidenceGraph`, `BidirectionalGraph`, `EdgeListGraph`) plus a `PropertyMap` concept for weights/colors/etc. Algorithm bodies don't know whether the graph is `adjacency_list<vecS, vecS, undirectedS>` or `adjacency_matrix<>` or `compressed_sparse_row_graph<>`. The user passes a `BFSVisitor` / `DijkstraVisitor` with hooks (`discover_vertex(u, g)`, `examine_edge(e, g)`, `tree_edge(e, g)`, `back_edge(e, g)`) that fire at the right point in the traversal. |
| **What works because of it** | One BFS implementation backs ~15 derived algorithms (connected components, bipartite check, articulation points, single-source unweighted shortest paths, topological sort via DFS visitor, biconnected components, etc.). One Dijkstra implementation backs ~6 derived algorithms. The visitor pattern is the *single highest leverage* design idea in the entire graph-library design space — it cuts canonical-algorithm count by ~3× without losing any algorithm. |
| **Zero-dep port for reality** | **Direct port, top priority.** The visitor pattern translates to Go with **callbacks-as-function-values**: `BFSVisit(adj [][]int, src int, visitor BFSVisitor)` where `BFSVisitor` is a struct of optional `func(u int)`, `func(u, v int)`, etc. fields, or a single `func(event BFSEvent)` with a sum type. Refactor `bfs.go` to expose `BFSVisit(...)` and `DFSVisit(...)` as primitives; rebuild `ConnectedComponents` (currently in `community.go:15`) and `TopologicalSort` (currently in `flow.go:127`) as 10-LOC visitors atop them; add 082-Tier-1's `ArticulationPoints` and `Bridges` as ~30 LOC visitors each. **Cost:** ~80 LOC primitive + ~150 LOC of derived algs. **Benefit:** closes 082-T1.5 and T1.10, plus pre-stages T2.8 (block-cut tree) and T2.9 (2-edge-connected components) to ~40 LOC each. The single highest-leverage refactor in the whole `reality/graph/` plan. |
| **What we should not copy** | The C++-template machinery itself (no Go equivalent). The `graph_traits<G>::vertex_descriptor` type indirection (Go uses `int` directly). The `property_map<G, vertex_index_t>::type` style — replace with parallel `[]int` / `[]float64` arrays. The `boost::tie()` / `boost::tuples::tuple` call style (Go's multi-return is the equivalent and much cleaner). |
| **Citation provenance** | Siek, Lee, Lumsdaine 2002 *The Boost Graph Library* (Addison-Wesley); v1.85 May 2024. https://www.boost.org/doc/libs/1_85_0/libs/graph/doc/. |

### 6. GraphBLAS / LAGraph 1.1 (Davis, Mattson, Buluç — TAMU/MIT/LBNL)

| | |
|---|---|
| **Headline algorithm** | **Every graph algorithm as sparse matrix multiplication** on a user-chosen semiring. Triangle count: `C = (A * A) ∘ A; sum(C)/6`. BFS: matrix-vector multiply on `(any, second)` semiring. Bellman-Ford: matrix-vector on `(min, plus)`. PageRank: matrix-vector on `(plus, times)`. The GraphBLAS C API standardized this in 2017; LAGraph is the reference algorithm library on top of it. |
| **Architectural trick** | **Semiring as algorithm parameter.** A semiring is a pair `(monoid, binary_op)` — e.g., `(min, +)` for shortest-path-style accumulation. Sparse matrix-multiply is parameterized on the semiring. Replacing the semiring switches the algorithm without changing the matmul code. Add an `accum` operator (how new values combine with existing) and a `mask` (which positions to update) and you have ~80% of graph traversal as one primitive. |
| **What works because of it** | One highly-optimized sparse matmul kernel (~600 LOC for SuiteSparse:GraphBLAS's CSR + masked-matmul) becomes the *whole* algorithm library. Parallel and GPU implementations port for free — GraphBLAS gets distributed implementations (Combinatorial BLAS, NVIDIA cuGraph) without rewriting algorithms. The 2024 LAGraph 1.1 release ships triangle count, k-truss, BFS, SSSP, betweenness, PageRank, connected components — all in <50 LOC each on top of the GraphBLAS API. |
| **Zero-dep port for reality** | **Wait — 2-PR coupling.** The right answer for `reality/graph/` long-term but blocked on `linalg/` shipping (a) sparse matrix CSR/CSC format and (b) masked semiring matmul. Both are slot-coordinated elsewhere in the 400-agent grid. When that lands, ~7 algorithms in `graph/` collapse to ~20 LOC each: BFS, SSSP, triangle count, PageRank, connected components, betweenness, k-truss. Pre-paying by building a graph-internal toy semiring layer would duplicate ~400 LOC of what `linalg/` will need. **Recommendation:** flag the dependency to the planner; do not ship the GraphBLAS-style refactor in `graph/` ahead of `linalg/`'s sparse layer. |
| **What we should not copy** | Pre-paying. Building a semiring layer just inside `graph/` for one user. Stay aligned with the future `linalg/` sparse API even if it means current implementations look "duplicated." |
| **Citation provenance** | Kepner & Gilbert eds. 2011 *Graph Algorithms in the Language of Linear Algebra* (SIAM); GraphBLAS C API spec v2.0 2021; LAGraph v1.1 Feb 2025. http://graphblas.org/. |

### 7. NetworKit 11.0 (Staudt, Sazonovs, Meyerhenke — KIT/HU Berlin)

| | |
|---|---|
| **Headline algorithm** | **Parallel-by-default classical algorithms** — OpenMP parallelization on every algorithm where the inner loop is embarrassingly parallel (BFS, connected components, all-pairs shortest paths, betweenness via Brandes' parallel variant, modularity, k-cores). Plus their own approximate-betweenness (KADABRA) for graphs of 10⁷ nodes. |
| **Architectural trick** | **C++ + OpenMP + Python binding via Cython.** Same as NetworKit's family (igraph, graph-tool) but specifically chooses *parallel by default* — if the algorithm scales, it runs on all cores without an explicit knob. The naming convention is also notable: explicit "label-correcting" (Bellman-Ford) vs "label-setting" (Dijkstra) terminology in doc-comments, which is the textbook framing and the cleanest way to teach the equivalence. |
| **What works because of it** | 5–10× faster than igraph on 16-core machines for parallelizable algorithms. Approximate betweenness (KADABRA, Borassi-Natale 2016) is the only realistic option for graphs of 10⁶+ nodes. |
| **Zero-dep port for reality** | **Two convention ports, no algorithm port.** First, **adopt the label-correcting / label-setting naming** in doc-comments — name `Dijkstra` as label-setting in its doc, name `BellmanFord` as label-correcting, document the equivalence under non-negative weights, document why the latter is required when negative weights exist. Second, **factor parallelism as opt-in via a separate `Parallel*` API surface** rather than baked-in: `Dijkstra(...)` is sequential and deterministic; `ParallelBFSLevels(...)` is non-deterministic across runs but produces the same level assignment per node. This is 085-graph-perf territory; flag for that slot. **Don't port** the C++/OpenMP parallelization itself — Go's goroutines + `sync.WaitGroup` is the equivalent and idiomatic. |
| **What we should not copy** | The "everything parallel by default" stance. Pistachio's 60-FPS use case is per-frame call-once with small graphs (n < 100); spinning up goroutines per call is more overhead than the work. Parallel variants belong in a separate API surface for callers with n > 10⁴. |
| **Citation provenance** | Staudt, Sazonovs, Meyerhenke 2016 Network Sci 4:508; KADABRA: Borassi & Natale 2016 ESA. v11.0 Apr 2025. https://networkit.github.io/. |

### 8. SNAP / snap.py 6.0 (Leskovec et al. — Stanford)

| | |
|---|---|
| **Headline algorithm** | **Big-graph descriptive analytics** — diameter approximation via ANF (Palmer-Gibbons-Faloutsos 2002), effective diameter, in/out-degree-distribution fitting, community detection (BigClam, CESNA — overlapping community models), graph generators (Kronecker, R-MAT) for benchmarking, plus all the classical tier. |
| **Architectural trick** | **Hash-of-int-vector adjacency** with explicit "small-graph" vs "big-graph" type splits (`TUNGraph` vs `TBigGraph`). The big-graph variant assumes the graph fits on disk, not in RAM; algorithms stream over edges in pages. ANF (Approximate Neighborhood Function) replaces all-pairs shortest paths with probabilistic counting (Flajolet-Martin sketches per node), giving an O(E·log V) approximation of the diameter. |
| **What works because of it** | Diameter / effective-diameter on graphs of 10⁹ edges in minutes. Kronecker-graph generators reproduce real-graph degree distributions for benchmarking. The R-MAT generator (Chakrabarti-Zhan-Faloutsos 2004) is the de-facto benchmark generator in graph-analytics research. |
| **Zero-dep port for reality** | **Single-algorithm port: ANF for approximate diameter** (Palmer-Gibbons-Faloutsos 2002) — ~120 LOC including Flajolet-Martin sketches. Belongs in 082-Tier-2 (currently lists `Diameter, Radius, Eccentricity, Girth` as ~120 LOC of exact computation; add `ApproxDiameter` alongside). Also worth porting: **R-MAT graph generator** (~60 LOC) for the test-fixture generators in 082-T1.14. **Don't port** the SNAP big-graph paging architecture — Pistachio's working set fits in RAM by definition; out-of-core is wrong layer. |
| **What we should not copy** | The C++ + small-Python wrapper architecture (SNAP is C++ with a thin snap.py veneer; same lesson as graph-tool — wait for the right C/Go binding need). The TBigGraph paging system. The hash-of-int-vector storage (slower than dense `[][]int` for in-RAM). |
| **Citation provenance** | Leskovec, Sosič 2016 ACM TIST 8:1 (snap.py); Palmer, Gibbons, Faloutsos 2002 KDD (ANF); Chakrabarti, Zhan, Faloutsos 2004 SDM (R-MAT). v6.0 2024. https://snap.stanford.edu/snap/. |

### 9. nauty / bliss 2.8 (McKay & Piperno; Junttila & Kaski)

| | |
|---|---|
| **Headline algorithm** | **Canonical labelling of graphs** for isomorphism testing. Nauty (McKay 1981; rewrite McKay-Piperno 2014) is the textbook reference; bliss (Junttila-Kaski 2007) is the open-source competitor that sometimes wins on sparse graphs. Both produce a canonical permutation of vertex labels such that two isomorphic graphs map to the same output, and detect graph automorphisms as a side effect. |
| **Architectural trick** | **Equitable partition refinement via individualization-and-refinement (I/R) tree search.** Start with vertices partitioned by degree; refine the partition by neighbor-color counts until stable; pick a vertex to "individualize" (split its color); recurse; backtrack with automorphism-pruning. The pruning by previously-discovered automorphisms is what makes the algorithm tractable on real graphs — the search tree is exponential in pathological cases (random regular graphs) but linear-ish in most real graphs. |
| **What works because of it** | The only realistic graph-isomorphism algorithm for general graphs at scale (10³–10⁴ nodes routinely; up to 10⁵ for sparse). VF2 (Cordella et al. 2004, currently 082-T3.2) is competitive for *subgraph* isomorphism but not full canonicalization. Used in chemoinformatics (canonical SMILES), combinatorial enumeration (orderly generation), and graph databases (canonical graph keys). |
| **Zero-dep port for reality** | **Possible but Tier 3.** The full nauty algorithm is ~3000 LOC of careful equitable-partition-refinement bookkeeping. The simpler **1-WL / color-refinement** half (Weisfeiler-Lehman 1968) is ~100 LOC and gives a *non-isomorphism witness* (if two graphs disagree under WL, they are non-isomorphic — but the converse fails for ~3% of graph pairs). 082-T3.3 already lists WL as Tier 3. **Recommendation:** ship 1-WL color refinement (~100 LOC, useful as ML-graph-kernel primitive *and* as fast-path non-isomorphism check), defer full canonical labelling to a future research-tier slot. The pure Go port of nauty is feasible but would dominate the package's LOC budget. |
| **What we should not copy** | The packed-bitset vertex-set representation (nauty's `setword*` arrays). Idiomatic in C, anti-idiomatic in Go; use `[]bool` or `map[int]bool` for now and revisit only if 1-WL turns out to bottleneck. |
| **Citation provenance** | McKay 1981 Cong Numer 30:45; McKay & Piperno 2014 J Symb Comput 60:94 (nauty 2.8); Junttila & Kaski 2007 ALENEX (bliss); Weisfeiler-Lehman 1968 Nauchno-Tekh Inf 2:9. https://pallini.di.uniroma1.it/, http://www.tcs.hut.fi/Software/bliss/. |

### 10. Pregel / Apache Giraph / GraphX (Google → Apache → UC Berkeley AMP)

| | |
|---|---|
| **Headline algorithm** | **Vertex-centric "think like a vertex" computation model** — the user writes a `compute(vertex, messages) → newState, outgoingMessages` function; the framework runs it in synchronous supersteps across thousands of machines. Pregel (Google 2010) was the original; Giraph (Apache 2012) and GraphX (Spark 2014) are open-source descendants. |
| **Architectural trick** | **Bulk-synchronous parallel (BSP) execution + message-passing.** Each superstep: every active vertex runs `compute()`, sends messages along outgoing edges, vertex deactivates. Superstep ends when no messages are in flight. The framework partitions vertices across machines, replicates messages along edge-cuts. PageRank, SSSP, connected components, community detection all become 20-LOC vertex programs. |
| **What works because of it** | Algorithms scale to 10¹⁰+ edges across compute clusters. The vertex-centric API is teachable to non-graph-experts — write the local update rule, get a global algorithm. GraphX added a `Pregel` operator on top of Spark RDDs to bring the same model to the Hadoop ecosystem. |
| **Zero-dep port for reality** | **Convention port only.** The vertex-centric API is the right framing for *one* future API surface in `graph/`, even single-machine: a `Pregel(adj, init, compute, maxSupersteps) []State` that lets users write graph algorithms as local updates. ~100 LOC. Useful as an extension point for *user-defined* algorithms (label-propagation variants, custom centrality measures, message-passing GNN layers). **Don't port** the distributed/BSP runtime — out of scope for `reality/`; that belongs in an application layer. **Don't port** the message-passing-as-network-IPC story — single-machine `[]message` queues per vertex are the equivalent. |
| **What we should not copy** | The framework-as-runtime model (Giraph is a Hadoop job; GraphX is a Spark library; Pregel was a Google internal service). For `reality/`, ship the *programming model* as a Go function, not a runtime. Distributed-graph computing is application infra, not foundational truth. |
| **Citation provenance** | Malewicz et al. 2010 SIGMOD (Pregel); Avery 2011 (Giraph); Xin et al. 2013 GRADES (GraphX). |

---

## Synthesis: the four ports, ranked

| # | Port | LOC | Source library | Why this trick |
|---|---|---:|---|---|
| P1 | **Visitor-pattern refactor of BFS/DFS** | ~80 primitive + ~150 derived | Boost.Graph | Single highest-leverage idea in graph-library design. Closes 082-T1.5 (articulation/bridges), T1.10 (Kosaraju), T2.8 (block-cut tree), T2.9 (2-edge-connected) at ~30 LOC each instead of ~80 each. Net LOC saved across Tier 1+2: ~300. |
| P2 | **Dense-int vertex IDs as the documented convention** | ~50 (intern adapter + doc) | igraph | Already de-facto in 13/17 algorithms; promote to *the* convention. Unlocks `reality/graph/` for Pistachio's 60-FPS allocation-free constraint. The string-keyed `bfs.go` becomes a thin convenience layer atop the int-keyed primitive. |
| P3 | **Lazy-Dijkstra primitive with callback** | ~50 refactor | graph-tool's pattern | One Dijkstra inner loop backs SSSP, weighted Brandes (082-T2.12), Closeness (T1.12), Harmonic (T1.12), bidirectional (T2.7). Net LOC saved across the 4 derived algs: ~150. |
| P4 | **Label-setting / label-correcting naming convention** | ~10 doc | NetworKit | Pure ergonomics; teaches the Dijkstra/Bellman-Ford equivalence in one line of doc-comment. Caller can pick the right algorithm by reading the contract. |

**Anti-ports (do not do):**

- A1. NetworkX's dict-of-dict adjacency. Wrong storage idiom for Go.
- A2. Boost.Graph's full template-concept machinery. No Go equivalent; interfaces already cover the use cases.
- A3. GraphBLAS-internal-to-graph semiring layer. Premature; wait for `linalg/` sparse + semiring.
- A4. Distributed/BSP runtime from Pregel/Giraph/GraphX. Wrong architectural layer.
- A5. Full nauty canonical labelling. ~3000 LOC; defer; 1-WL color refinement (~100 LOC) covers the realistic non-isomorphism witness use case.

---

## Cross-package coordination notes

- **P1 (visitor refactor)** is local to `graph/` — no other package coordination needed.
- **P3 (lazy-Dijkstra primitive)** is local to `graph/` but should coordinate with 082-graph-missing's Tier 1 sequencing: refactor *first*, then build T1.12 (Closeness/Harmonic) and T2.12 (Weighted Brandes) on top.
- **P2 (dense-int convention)** has no cross-package implications; the string-keyed `bfs.go` becomes a 20-LOC adapter.
- The **GraphBLAS port (deferred)** depends on `linalg/` shipping sparse CSR matrices and masked semiring matmul. Coordinate with the `linalg-missing` and `linalg-perf` slots in the 400-agent grid.
- **ForceAtlas2 layout (Gephi)** depends on `geometry/` shipping a 2D quadtree (Barnes-Hut). Currently 077-Tier-1 territory. Ship FA2 only after that lands; brute-force fallback for small `n` is fine in the interim.
- **1-WL color refinement (nauty/bliss)** is a useful primitive shared with future ML-graph-kernel work in `aicore`. Note for the planner: if `aicore` ships graph kernels for downstream GNN work, the 1-WL primitive lives in `reality/graph/` and `aicore` consumes it.

---

## Sources

- Hagberg, Schult, Swart 2008 SciPy Proc — NetworkX.
- Csárdi & Nepusz 2006 InterJournal CX.18 — igraph.
- Peixoto 2014 PRX 4:011047, Peixoto 2017 ARXIV 1705.10225 — graph-tool / SBM.
- Bastian, Heymann, Jacomy 2009 ICWSM — Gephi.
- Jacomy, Venturini, Heymann, Bastian 2014 PLoS ONE 9:e98679 — ForceAtlas2.
- Siek, Lee, Lumsdaine 2002 *The Boost Graph Library* (Addison-Wesley) — Boost.Graph + visitor pattern.
- Kepner & Gilbert eds. 2011 *Graph Algorithms in the Language of Linear Algebra* (SIAM) — GraphBLAS framing.
- GraphBLAS C API spec v2.0 (Davis, Mattson, Buluç eds.) 2021; LAGraph v1.1 Feb 2025.
- Staudt, Sazonovs, Meyerhenke 2016 Network Sci 4:508 — NetworKit.
- Borassi & Natale 2016 ESA — KADABRA approximate betweenness.
- Leskovec & Sosič 2016 ACM TIST 8:1 — snap.py.
- Palmer, Gibbons, Faloutsos 2002 KDD — ANF approximate diameter.
- Chakrabarti, Zhan, Faloutsos 2004 SDM — R-MAT.
- McKay 1981 Cong Numer 30:45; McKay & Piperno 2014 J Symb Comput 60:94 — nauty.
- Junttila & Kaski 2007 ALENEX — bliss.
- Weisfeiler & Lehman 1968 Nauchno-Tekh Inf 2:9 — 1-WL color refinement.
- Malewicz et al. 2010 SIGMOD — Pregel.
- Avery 2011, ApacheCon — Giraph.
- Xin, Crankshaw, Dave, Gonzalez, Franklin, Stoica 2013 GRADES — GraphX.
- Brönnimann, Burnikel, Pion 2001 — static filter (cited via 076 / for context on adaptive predicates as a related "filter cheaply, fall back when needed" pattern in the geometry sibling).

---

**Headline:** The four design philosophies — encyclopedia-ergonomics (NetworkX/Gephi), C-kernel-int-vectors (igraph/NetworKit/SNAP), type-system-as-correctness (Boost.Graph/graph-tool), linear-algebra-as-abstraction (GraphBLAS/LAGraph) — pick out four portable tricks for `reality/graph/`: **(P1)** the Boost.Graph **visitor refactor of BFS/DFS** (~80 LOC primitive + ~150 LOC of derived algs that close 082-T1.5/T1.10/T2.8/T2.9 at ~30 LOC each, the highest-leverage refactor in the graph-package plan); **(P2)** igraph's **dense-int vertex-IDs convention** documented as *the* contract (~50 LOC adapter); **(P3)** graph-tool's **lazy-Dijkstra primitive with callback** that backs SSSP / weighted Brandes / Closeness / Harmonic / bidirectional (~50 LOC refactor saving ~150 LOC across 4 derived algs); **(P4)** NetworKit's **label-setting / label-correcting naming convention** (~10 LOC doc only). Five anti-ports flagged: NetworkX dict-of-dict storage, Boost.Graph templates, GraphBLAS-internal semirings (wait for `linalg/`), Pregel distributed runtime, full nauty canonicalization. Single ForceAtlas2-layout port from Gephi (~250 LOC including Barnes-Hut) blocked on 077-Tier-1 quadtree. Single ANF-approximate-diameter port from SNAP (~120 LOC) belongs in 082-Tier-2. Net algorithmic-LOC saved by P1+P3 across Tier 1+2 of 082's plan: ~450 LOC. Net LOC added for the four ports themselves: ~190. The visitor refactor is the single recommendation with the highest leverage-per-LOC in the entire graph-package design space.
