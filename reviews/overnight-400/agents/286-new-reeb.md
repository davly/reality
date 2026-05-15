# 286 — new-reeb (Reeb Graphs / Contour Trees / Merge Trees)

## Headline
reality v0.10.0 ships ZERO Reeb / contour-tree / merge-tree / join-tree / split-tree / branch-decomposition / level-set / Reeb-space / Jacobi-set surface (`Reeb|ContourTree|MergeTree|JoinTree|SplitTree|BranchDecomposition|LevelSet|Sublevel|Superlevel|MorseSmaleReeb|JacobiSet|InterleavingDistance|MergeTreeEditDistance` repo-wide grep on `*.go` outside reviews/ returns ZERO callable matches); the only sibling pieces present are `topology/persistent/{vr,barcode}.go` (VR + ELZ-2000 F_2 column-reduction barcode capped at maxDim ∈ {0,1}) and a *re-derived ad-hoc union-find inside `graph/mst.go::KruskalMST`* (lines 47-76, recursive path-compression `find` + union-by-rank, **not exported, not reusable**) — a reusable `graph/dsu.UnionFind` is the **first 50-LOC enabler** for the entire Reeb stack. Slot 286 is the **CHEAPEST tier in the entire Block C TDA stack (slots 283/284/285/286)**: Reeb-graph computation is *purely combinatorial* (no Eigvec, no Smith form, no LP, no Delaunay) — it is a single union-find sweep over a vertex order; the day-1 PR `topology/reeb/` (T0+T1+T2+T3+T4) ships ~480 LOC, gates only on a graph 1-skeleton (existing `[]Edge` from `graph/types.go` suffices for the PL-curve case), and is **fully decoupled** from slot 283 SimplexTree / slot 284 CellComplex / slot 285 DMT — those slots gate Reeb on a *full d-complex* (T5 onwards) but the contour-tree itself ships standalone on graph 1-skeletons. Pure-Go MIT zero-dep ABSENT in any language ecosystem (libtourtre BSD C++ ~3k LOC, Topology ToolKit BSD-3 C++ ~50k, Tierny et al.'s code BSD; no maintained Rust / JS / Julia clean-room MIT impl).

## Findings

### State at HEAD (verified by direct grep on `*.go`)

| Surface | Path | Reeb-relevance |
|---|---|---|
| ad-hoc union-find inside `KruskalMST` | `graph/mst.go:47-76` | Recursive `find` w/ path-compression + union-by-rank. **PRIVATE, NOT REUSABLE.** Same algorithm needed by Carr-Snoeyink-Axen-2003 sweep ×N. |
| `Edge = [2]string`, `IntAdjacency = map[int][]int` | `graph/types.go:7,14` | 1-skeleton substrate. Sufficient for contour-tree on PL-graph; insufficient for Reeb-on-2-manifold (need full triangle list → gates on slot 283 `SimplicialComplex`). |
| `KruskalMST`, `PrimMST`, `BFS/DFS`, `Dijkstra`, `centrality` | `graph/*.go` | All read graph 1-skeleton directly. None compute level-sets / merge structure. |
| `topology/persistent/vr.go::Filtration` | `topology/persistent/vr.go:50` | Sorted-by-filtration simplex list. **For 0-dim persistence on a graph (β_0 only) the merge-tree IS the join-tree dendrogram**; ELZ pairs ≡ merge-tree pairing of saddles to extrema (Edelsbrunner-Letscher-Zomorodian-2002 §3). |
| `topology/persistent/barcode.go::ComputeBarcode` | barcode.go:60 | Naive ELZ; **0-dim bars on a sublevel filtration of a graph ≡ merge-tree of the scalar field on that graph** (regression target — pin #1 below). |
| `optim/`, `signal/`, `geometry/sdf.go` | various | None Reeb-relevant. |
| repo-wide grep `Reeb\|ContourTree\|MergeTree\|JoinTree\|SplitTree\|BranchDecomposition\|LevelSet\|Sublevel\|Superlevel\|JacobiSet\|MorseSmaleReeb\|InterleavingDistance\|MergeTreeEditDistance` | `*.go` outside reviews/ | **ZERO hits.** |

### Slot boundaries (no overlap with 283/284/285)

- **Slot 283 (simplicial complexes) ←** ChainComplex + SimplexTree + simplicial homology + Hodge L_k. **Disjoint axis from 286.** Reeb consumes *one scalar function f*; 283 consumes *the complex K* and computes ranks of homology *of the complex itself* (not of level sets of a function on K). **Overlap point:** Reeb on a 2-manifold (T5 below) consumes 283-T0 SimplicialComplex (need triangle list to detect saddle splits).
- **Slot 284 (cell complexes / cubical) ←** CubicalComplex + cubical persistence on raster scalar fields. **Cross-validation pin opportunity:** cubical 0-dim persistence (T-construction or V-construction lower-star) on raster f ≡ merge-tree of f on the cubical 1-skeleton (R-MUTUAL-CROSS-VALIDATION pin #2 below). Tiny n=8×8 raster fixture saturates.
- **Slot 285 (discrete Morse) ←** DMT compresses a cell complex while preserving homology. **Distinct from Reeb:** DMT is a topology-of-the-complex compressor; Reeb summarizes topology of *level sets of a function*. **Composition:** DMT-collapse first → run merge-tree on critical-cell complex with f restricted = same merge-tree at ~99% fewer cells. Mischaikow-Nanda-2013 §4 establishes filtration-respecting collapse preserves persistence pairs ≡ preserves merge-tree pairing.
- **Slot 097-T1 (linalg-missing, Eigvec) ←** **NO BLOCKER.** Reeb is purely combinatorial. **Cheapest tier in Block C TDA.**
- **Slot 077/078 (geometry-{missing,sota}) ←** **NO BLOCKER for graph case.** For Reeb on a triangulated surface (T5), no Delaunay needed — the input *is* a triangle list (typically MRI mesh, terrain TIN, molecular surface).
- **Slot 142 (topology-missing).** 142 punts merge-tree / contour-tree to a future slot — **this slot owns it.** Re-read 142 after 286 lands.
- **Slot 156 (synergy-topology-prob).** Cross-link: merge-trees on stochastic scalar fields → Cohen-Steiner-Edelsbrunner-Harer-2007 stability theorem (bottleneck distance bounded by sup-norm of f-perturbation) → Bayesian inference of merge-tree from noisy samples (Curry-Mukherjee-Turner 2018).
- **Slot 281 (temporal graphs).** Time-varying Reeb (Edelsbrunner-Harer-Mascarenhas-Pascucci 2008): track Reeb critical points across timesteps. Composes 286-T8 streaming Reeb with 281's temporal-graph substrate.
- **Slot 277 (combinatorial-optimization, BnB).** Optimal merge-tree edit distance (Beketayev-Yeliussizov-Morozov-Phillips-Weber-Hamann-2014) is NP-hard in general → BnB consumer of slot-277-I3 just like slot-285-T3 optimal DMT.

### Web context (canonical reference set, MIT pure-Go zero-dep ABSENT)

- **Reeb-1946** *Comptes Rendus* 222:847 "Sur les points singuliers d'une forme de Pfaff" — original definition. The *Reeb graph* of a smooth function f: M → R on a manifold M is the quotient space M/~ where x ~ y iff f(x) = f(y) and x, y lie in the same connected component of f^{-1}(f(x)). **Identifies points lying in the same level-set component** → 1-complex graph (vertices = critical-level-set components, edges = isotopic level-set families).
- **Edelsbrunner-Harer-2010** *Computational Topology: An Introduction* AMS Chapter VI + VII — canonical textbook treatment. **Merge tree** of f = Reeb graph of f restricted to sublevel sets {x : f(x) ≤ t} (the "join tree" tracks merging components as t increases). **Split tree** = merge tree of -f (tracks splitting components as t decreases / equivalently merging in superlevel sets). **Contour tree** = Reeb graph for *simply-connected* domain (no loops). Theorem (Carr-Snoeyink-Axen 2003): on a simply-connected domain, Reeb graph = contour tree = merge of join tree + split tree.
- **Carr-Snoeyink-Axen 2003** *Computational Geometry: Theory & Apps* 24:75 "Computing contour trees in all dimensions" — **the** canonical contour-tree algorithm. Three phases: (1) compute join tree by sweeping vertices in *decreasing* f-order using union-find — each new vertex merges its open higher-f-components; (2) compute split tree dually (sweep increasing); (3) merge join + split tree by iteratively pruning leaves whose other tree has matching upper/lower link. **Time complexity O((n + t) α(n))** where n = vertices, t = simplices, α = inverse-Ackermann (effectively linear). Reference: libtourtre BSD-3 C++ ~3k LOC. **Operates on a graph 1-skeleton with f-values at vertices** — does NOT need full d-complex for the contour-tree itself. Higher-d simplices only matter if the underlying space has nontrivial β_1 (loops) — then contour-tree is replaced by Reeb graph (T5).
- **Pascucci-Cole-McLaughlin 2002** *IEEE Visualization* "Parallel computation of the topology of level sets" — **streaming/parallel** contour-tree + Reeb-graph on out-of-core data. Sort vertices once globally; process in chunks; maintain active-arc structure; merge across chunks via interface-edge processing. **Time O(n log n) sort + O((n+t) α(n)) sweep parallelizable.** Standard for terabyte-scale scientific data.
- **Edelsbrunner-Harer-Patel 2008** *SoCG* "Reeb spaces of piecewise linear mappings" — **Reeb space** generalizes Reeb graph from f: M → R (1D codomain) to f: M → R^d (multi-dimensional codomain). Quotient M/~ where x ~ y iff f(x) = f(y) and same component of fiber f^{-1}(f(x)). For d=2, Reeb space is a 2-complex; structure encoded by **Jacobi set** = {x ∈ M : df_1, df_2 linearly dependent} (Edelsbrunner-Harer-2002). **Foundation for multi-field visualization** (climate data with temperature × pressure × humidity, computational chemistry with electrostatic × hydrophobic potential).
- **Edelsbrunner-Harer-2002** *Foundations of Computational Math* "Jacobi sets of multiple Morse functions" — Jacobi set algorithm: critical points of *one function restricted to level sets of another*, identified combinatorially on a triangulated domain.
- **Doraiswamy-Natarajan 2009** *IEEE-TVCG* 15:1697 "Efficient algorithms for computing Reeb graphs" — Reeb graph of f on a *non-simply-connected* domain via segmenting Reeb arcs at critical points + merging across loops. **Time O(n log n + t α(n)).** Standard modern Reeb-graph algorithm. Earlier alternatives (Cole-McLaughlin-Edelsbrunner-Hart-Pascucci 2003 *SoCG*) require expensive surface-tracing.
- **Tierny-Gyulassy-Simon-Pascucci 2009** *IEEE-TVCG* 15:1177 "Loop surgery for volumetric meshes: a topological approach to robust contour tree computation" — handles non-simply-connected domains by cutting handles → makes domain simply-connected → run Carr-Snoeyink-Axen → reattach loops. **Practical workhorse** for terrain / medical-imaging meshes; foundation for Topology ToolKit (TTK) **BSD-3 C++ ~50k LOC**.
- **Pascucci-Scorzelli-Bremer-Mascarenhas 2007** *ACM-SIGGRAPH* "Robust on-line computation of Reeb graphs: simplicity and speed" — incremental update of Reeb graph as new triangles added (streaming triangle stream). Foundation for time-varying / out-of-core analysis.
- **Edelsbrunner-Harer-Mascarenhas-Pascucci 2004** *SoCG* "Time-varying Reeb graphs for continuous space-time data" — track Reeb-graph topology evolution over time. Used for fluid-flow visualization, weather data, simulated time-series.
- **Pascucci-Tricoche-Hagen-Tierny 2011** *Topological Methods in Data Analysis Vol II* Springer — comprehensive survey including streaming + time-varying.
- **Branch decomposition** (Pascucci-Cole-McLaughlin-Scorzelli 2004 *Vis* "The branch decomposition hierarchy and its applications") — recursive decomposition of contour tree into "branches" (root-to-leaf paths) ranked by persistence. **Multi-scale topological feature extraction:** prune low-persistence branches → simplified Reeb graph. O(n log n).
- **Bauer-Munch-Wang 2015** *SoCG* "Strong equivalence of the interleaving and functional distortion metrics for Reeb graphs" — Reeb graphs form a metric space under **interleaving distance** d_I(R_1, R_2) (smallest ε such that ε-perturbations of f_1 and f_2 produce isomorphic Reeb graphs); equivalent up to constants to functional-distortion (Bauer-Geiß-Wang 2014) and edit distance.
- **Morozov-Beketayev-Weber 2013** *TopoInVis* "Interleaving distance between merge trees" — first practical algorithm for interleaving distance on merge trees. Equivalent to **bottleneck distance** on the persistence diagrams induced by the merge tree (Cohen-Steiner-Edelsbrunner-Harer 2007). **Cross-validation pin opportunity #3 below.**
- **Beketayev-Yeliussizov-Morozov-Phillips-Weber-Hamann 2014** *TopoInVis* "Measuring the distance between merge trees" — **edit distance** on merge trees: minimum cost of (relabel vertex / contract edge / delete subtree) operations to transform M_1 into M_2. NP-hard in general; ILP / branch-and-bound (Beketayev §4). **Consumes slot-277 BnB.**
- **Carr-Weber-Sewell-Snoeyink 2017** *IEEE-TVCG* 23:921 "Multiscale contour tree" — Carr-Snoeyink-Axen + branch decomposition + multi-scale simplification by persistence. THE modern unified treatment.
- **Sridharamurthy-Masood-Kamakshidasan-Natarajan 2020** *IEEE-TVCG* 26:1518 "Edit distance between merge trees" — polynomial-time algorithm for **constrained** merge-tree edit distance (preserve depth-monotonicity); O(n^4) in general, O(n^2 log n) on path-trees. Practical for shape-matching applications.
- **Wetzels-Leitte-Garth 2022** *IEEE-TVCG* 28:1197 "Branch decomposition-independent edit distances for merge trees" — branch-decomposition-distance: extract canonical branch decomp from each merge tree first → distance is hierarchical-clustering distance on branch-trees. **Cross-validation pin #4: interleaving ≡ edit ≡ branch-decomposition distance up to constants on small fixtures.**
- **Cohen-Steiner-Edelsbrunner-Harer 2007** *Discrete-Comput-Geom* 37:103 "Stability of persistence diagrams" — **stability theorem**: bottleneck distance between persistence diagrams of f and g is bounded by ‖f − g‖_∞. Specializes to merge trees: small f-perturbation → small interleaving distance.
- **Lukasczyk-Maack-Edelsbrunner 2020** *Comput-Graph-Forum* "Dynamic Reeb graphs" — track Reeb graph changes when underlying scalar field f(t) varies continuously; identify critical events (saddle-saddle merge, swallowtail). Time-varying generalization.
- **Open-source landscape:**
  - **Topology ToolKit (TTK)** Tierny et al. — BSD-3 C++/VTK ~50k LOC; ParaView plugin. SOTA for visualization.
  - **libtourtre** BSD-3 C++ ~3k LOC; Carr-Snoeyink-Axen reference impl; archived.
  - **vtkReebGraphFilter / vtkContourTreeFilter** in VTK BSD-3.
  - **GUDHI** ships *no* Reeb / contour-tree (their TDA stack is simplicial / persistence).
  - **scikit-tda / giotto-tda** Python — no Reeb / contour-tree.
  - **Julia Eirene** — no Reeb.
  - **Pure-Go MIT zero-dep ABSENT for ALL of:** Reeb graph, contour tree, merge tree, join tree, split tree, branch decomposition, multi-scale contour tree, interleaving distance, edit distance, Reeb space, Jacobi set, time-varying Reeb, streaming Reeb. Reality has the **single largest open MIT-licensed zero-dep gap in scientific-visualization topology** in any language ecosystem.

## Concrete recommendations

### T0 — UnionFind extracted to `graph/dsu.go` (cheapest, ~50 LOC, ZERO blocker, unblocks 4+ slots)

1. **`graph/dsu.go::UnionFind` (~50 LOC).** Extract the ad-hoc impl from `graph/mst.go:47-76` into a public reusable type:
   ```go
   type UnionFind struct { parent, rank []int; n, components int }
   func NewUnionFind(n int) *UnionFind
   func (uf *UnionFind) Find(x int) int           // path-compression (iterative; not recursive — avoids stack overflow on n > 1e6 chains)
   func (uf *UnionFind) Union(x, y int) bool      // union-by-rank, returns true if merged
   func (uf *UnionFind) Connected(x, y int) bool
   func (uf *UnionFind) NumComponents() int
   func (uf *UnionFind) Reset()                   // reuse buffer across multiple sweeps (T2/T3)
   ```
   `KruskalMST` retrofits to use it (drop ~30 LOC). **Composes existing** rather than introduces.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#0):** for every fixture in `graph/graph_test.go` MST tests, retrofit Kruskal ≡ original Kruskal output bit-identical.
   - **Cross-link:** unblocks slot-274 network-flow (Boykov-Kolmogorov DSU), slot-275 matroid (rank oracle DSU), slot-280 SBM (component-counting), slot-282 hypergraph connected-components.

### T1 — ScalarField interface + lower-star edge ordering (~80 LOC, ZERO blocker)

2. **`topology/reeb/types.go::ScalarField` interface (~50 LOC).**
   ```go
   // ScalarField: assign one real value per vertex of a graph 1-skeleton or simplicial complex.
   // Function values must be totally ordered; ties are broken by lex order on vertex index
   // (Edelsbrunner-Harer §VI.3 simulation-of-simplicity).
   type ScalarField interface {
       NumVertices() int
       Value(v int) float64
       Edges() [][2]int           // 1-skeleton; for d-complex inputs Edges() is induced subgraph
       Tiebreak(u, v int) bool    // f(u) < f(v) under SoS; ties → u < v
   }
   type GraphField struct { Vals []float64; Adj [][]int }   // ships standalone, ZERO complex dep
   type SimplicialField struct { sc *simplicial.SimplicialComplex; Vals []float64 } // gates on slot 283 T0
   func (f *GraphField) UpperLink(v int) []int   // {u ∈ N(v) : Tiebreak(v, u)}
   func (f *GraphField) LowerLink(v int) []int   // {u ∈ N(v) : Tiebreak(u, v)}
   ```
   **Lower-star / upper-star** = the simplices σ ∋ v with v as their min/max-f-vertex (Edelsbrunner-Harer §VI.3). **Single load-bearing combinatorial primitive** for join-tree sweep.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#1, sublevel ↔ superlevel duality):** `LowerLink_f(v) ≡ UpperLink_{−f}(v)` for all v on every fixture. **Catches sign bugs at boundary.**

### T2 — JoinTree via Carr-Snoeyink-Axen sweep-by-decreasing-f (~120 LOC)

3. **`topology/reeb/jointree.go::JoinTree(f ScalarField) *MergeTree` (~120 LOC).** Carr-Snoeyink-Axen-2003 §3:
   ```
   sort vertices in decreasing f-order (with SoS tiebreak)
   uf := NewUnionFind(n); arcs := []Arc{}
   for each v in decreasing order:
       L := upper-link components of v in current uf state
       if L is empty:           v starts a new component (regular max)
           uf adds singleton; arcs.append(start v)
       elif |L| == 1:           v extends one component (regular interior point on arc)
           uf.union(v, L[0]); arcs[L[0]].extend(v)
       else (|L| ≥ 2):          v is a saddle (join — multiple components merge here)
           merge all L into uf; arcs.append(saddle v with parents=L); record merge node in tree
   final tree: root = global min vertex; leaves = local maxima; internal = saddles
   ```
   Returns `MergeTree{Nodes []*Node; Root *Node; Verts []int}` where each node carries `(VertexIdx int, F float64, Children []*Node, Type {Max,Saddle,Min})`. **Time O((n + e) α(n))** = effectively linear.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#2, sublevel-persistence ≡ merge-tree-pairing):** for every fixture, 0-dim `topology/persistent.ComputeBarcode` on the lower-star sublevel filtration of `f` produces the SAME birth-death pairs as JoinTree's saddle-to-leaf pairing under the **elder rule** (Edelsbrunner-Letscher-Zomorodian 2002 §3): when two components merge at a saddle, the *younger* (later birth) dies, paired with the saddle. Three-way saturation: ELZ persistence ≡ merge-tree elder pairing ≡ cubical 0-dim persistence (slot-284 cross-validation when 284-T1 lands).

### T3 — SplitTree via dual sweep on −f (~40 LOC)

4. **`topology/reeb/splittree.go::SplitTree(f ScalarField) *MergeTree` (~40 LOC).** One-liner: `return JoinTree(NegateField(f))`. Validates by symmetry.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#3, join↔split duality):** `SplitTree(f) ≡ JoinTree(-f)` structurally (verbatim equality after f-value sign-flip on every node). **Single load-bearing regression on the entire sweep machinery.** If join sweep has any sign bug, split sweep on −(−f) catches it.

### T4 — ContourTree from Carr-Snoeyink-Axen merge of join + split tree (~140 LOC)

5. **`topology/reeb/contourtree.go::ContourTree(f ScalarField) *ContourTreeT` (~140 LOC).** Carr-Snoeyink-Axen-2003 §4: iteratively prune leaves whose corresponding leaf in the other tree has the matching valence; record arcs. Result is the contour tree (a tree if M is simply-connected). **Time O((n + e) α(n)).**
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#4, contour-tree-from-merge ≡ direct-Reeb-graph-when-simply-connected):** when slot-283 SimplexTree T0 lands, run Reeb graph (T5 below) on a triangulated *disk* (simply-connected) directly; result must be isomorphic to ContourTree. **Cross-link to 283.**
   - **Pin (#5, merge-tree-pairing ≡ contour-tree-pairing):** the persistence pairing extracted from the contour tree equals the pairing from JoinTree on a simply-connected domain. Three-way: ELZ ≡ JoinTree ≡ ContourTree.

### T5 — Reeb graph on triangulated 2-manifold (gates on slot 283 T0) (~220 LOC)

6. **`topology/reeb/reebgraph.go::ReebGraph(sc *simplicial.SimplicialComplex, f []float64) *ReebGraphT` (~220 LOC, **gates on 283-T0**).** Doraiswamy-Natarajan-2009 algorithm: like contour tree, but at each saddle the link valence determines whether the saddle is a "loop saddle" (creates a 1-cycle in Reeb graph — closure of two arcs into a loop) or a "join/split saddle" (tree merge). Loops detected via β_1 of the link in the simplex tree; **direct consumer of slot-283 simplex-tree cofaces** (link of v = link in simplex tree).
   - **Pin (#6, Reeb-on-2-sphere ≡ contour-tree-on-2-sphere-minus-pole):** Reeb graph of f on S^2 = contour tree of f on S^2 (since S^2 is simply-connected after removing one pole). Verify on standard fixture (fxnal sphere, four critical points → bipartite Reeb graph, Morse inequalities saturated).
   - **Pin (#7, β_1(Reeb graph) ≤ β_1(M)):** for any compact M, β_1 of the Reeb graph is upper-bounded by β_1(M). Cross-validate against slot-283 BettiF2 on the input complex.

### T6 — Branch decomposition (~120 LOC)

7. **`topology/reeb/branchdecomp.go::BranchDecomposition(ct *ContourTreeT) *BranchTree` (~120 LOC).** Pascucci-Cole-McLaughlin-Scorzelli-2004: greedy hierarchical pairing — repeatedly pair the leaf-saddle with smallest persistence first; record paired arcs as a "branch"; the parent of a branch is the saddle's branch. Result: **branch tree** with persistence-ordered nodes. Time O(n log n).
   - **Pin (#8, branch-decomposition ≡ persistence diagram):** the multiset of branch-persistences ≡ the multiset of bar lengths in slot-283 0-dim ELZ barcode. Three-way: ELZ ≡ JoinTree-elder-pairing ≡ BranchDecomposition.

### T7 — Persistence simplification (cancel low-persistence pairs) (~80 LOC)

8. **`topology/reeb/simplify.go::SimplifyByPersistence(mt *MergeTree, eps float64) *MergeTree` (~80 LOC).** For each saddle-extremum pair with persistence < eps, **cancel the pair** (Forman cancellation on the merge tree = remove the saddle vertex and its paired leaf, reconnect arc through). Multi-scale topological feature extraction; **direct consumer-pull from slot-285 DMT cancellation theory** (Bauer-Edelsbrunner-2014 — merge-tree cancellation IS Forman cancellation on the 0-dim Hasse diagram).
   - **Pin (#9, simplification stability):** Cohen-Steiner-Edelsbrunner-Harer 2007 — `bottleneck(D(f), D(g)) ≤ ‖f − g‖_∞`. After ε-simplification, bars of length ≤ ε disappear; remaining bars within ε in bottleneck sense.

### T8 — Streaming / incremental Reeb graph (Pascucci-Cole-McLaughlin 2002) (~280 LOC)

9. **`topology/reeb/streaming.go::StreamingReeb(stream <-chan VertexOrTriangle) *ReebStream` (~280 LOC).** Pascucci-2002 §3 streaming algorithm: maintain *active arcs* + *active union-find* keyed by current sweep value; process vertices/triangles in arrival order; output Reeb-graph nodes once their entire link has been seen. **Out-of-core: O(active set) RAM**, not O(total). Standard for terabyte-scale terrain / weather / climate / fluid-simulation data.
   - Composes `signal/streaming.go` (if present) or a new `chan` interface.

### T9 — Reeb space for multivariate f: M → R^d (~360 LOC, frontier; gates on Jacobi-set computation in slot-283 SimplexTree)

10. **`topology/reeb/jacobi.go::JacobiSet(sc, f1, f2 []float64) []Edge` (~120 LOC, gates on 283-T0).** Edelsbrunner-Harer-2002 algorithm: for each edge (u, v) in the link of vertex w in sc, check if `(f1, f2)` restricted to the link is non-monotone in lower-star ordering → w is a Jacobi point. **Combinatorial detection** of where df_1, df_2 become linearly dependent.
11. **`topology/reeb/reebspace.go::ReebSpace(sc, f []float64 [][]float64) *ReebSpaceT` (~240 LOC, frontier).** Edelsbrunner-Harer-Patel-2008: **multivariate generalization** — for each fiber f^{-1}(p), compute connected components, build quotient 2-complex. Used by climate-data multi-field visualization, computational chemistry electrostatic-×-hydrophobic Reeb-space.

### T10 — Distances between merge trees (~520 LOC, frontier; T10c gates on slot-277 BnB)

12. **`topology/reeb/distance/interleaving.go::InterleavingDistance(m1, m2 *MergeTree) float64` (~140 LOC).** Morozov-Beketayev-Weber-2013 `_2013` algorithm. Polynomial in tree size for non-crossing case; identical to bottleneck on persistence diagrams of merge trees (Cohen-Steiner-2007 corollary). **Cross-validation pin opportunity to slot-156 stability theorem.**
13. **`topology/reeb/distance/branchdecomp_dist.go::BranchDecompositionDistance(b1, b2 *BranchTree) float64` (~120 LOC).** Wetzels-Leitte-Garth-2022. Polynomial-time hierarchical-clustering edit distance on canonical branch decompositions.
14. **`topology/reeb/distance/edit.go::MergeTreeEditDistance(m1, m2 *MergeTree) float64` (~260 LOC, **gates on slot-277-I3 BnB**).** Beketayev-2014 ILP / BnB algorithm. **Second concrete consumer of slot-277 (after slot-285-T3 optimal DMT)** — formulates as 0-1 ILP over edge-mapping pairs with sub-tree-consistency constraints + lazy constraint generation.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#10, three-way distance equivalence):** on small fixtures (n ≤ 8 leaves) where exact computation is tractable: `InterleavingDistance ≡ BranchDecompositionDistance ≡ MergeTreeEditDistance` up to multiplicative constants (Bauer-Munch-Wang-2015 strong-equivalence theorem). **Three-way saturation R-MUTUAL-CROSS-VALIDATION 3/3 pin** — three independent algorithmic pipelines producing equivalent metrics.

### T11 — Time-varying Reeb (Edelsbrunner-Harer-Mascarenhas-Pascucci 2004) (~360 LOC, defer; gates on slot-281 temporal-graph axis)

15. **`topology/reeb/timevarying.go::TimeVaryingReeb(sc, f_t func(t float64) []float64) *TimeVaryingReebT` (~360 LOC).** Track Reeb-graph topological events (saddle-saddle merge, swallowtail) as f varies continuously. Defer until slot-281 temporal-graph substrate ships.

## Single cheapest day-1 PR

**`topology/reeb/` package T0+T1+T2+T3+T4 = ~430 LOC, ZERO blockers, ships standalone on graph 1-skeleton with f-values at vertices.**

- `graph/dsu.go` — UnionFind extracted (T0, ~50 LOC); KruskalMST retrofit (~−30 LOC net).
- `topology/reeb/types.go` — ScalarField interface + GraphField (T1, ~80 LOC).
- `topology/reeb/jointree.go` — Carr-Snoeyink-Axen JoinTree (T2, ~120 LOC).
- `topology/reeb/splittree.go` — SplitTree-via-(-f) (T3, ~40 LOC).
- `topology/reeb/contourtree.go` — Carr-Snoeyink-Axen merge of join+split (T4, ~140 LOC).

**Tests** (R-MUTUAL-CROSS-VALIDATION 3/3 pins saturate on day 1):
- `TestUnionFind_KruskalRegression` — pin #0; retrofit Kruskal ≡ pre-extraction Kruskal bit-identical on all `graph/graph_test.go` MST fixtures.
- `TestScalarField_LinkDuality` — pin #1; LowerLink_f ≡ UpperLink_{−f} on every fixture.
- `TestJoinTree_ELZBarcode` — pin #2; on n ≤ 50-vertex graph fixture from `topology/persistent/persistent_test.go`, JoinTree elder-pairing ≡ ELZ 0-dim barcode pairs on lower-star sublevel filtration.
- `TestSplitTree_NegationDuality` — pin #3; SplitTree(f) ≡ JoinTree(-f) verbatim.
- `TestContourTree_TreeOnDisk` — Carr-Snoeyink-Axen contour tree on a 2D-grid scalar field is a tree (β_1 = 0); count nodes; verify Morse inequalities.
- `TestContourTree_TerrainFixture` — small synthetic 8x8 elevation map; manually-derived contour tree compared bit-identical.
- `TestPersistencePairs_ThreeWay` — three-way pin: 0-dim ELZ barcode pairs ≡ JoinTree elder pairs ≡ ContourTree leaf-pairing on the same fixture. **R-MUTUAL-CROSS-VALIDATION 3/3 saturated on day 1.**

This PR ships:
1. The first reusable `UnionFind` in reality (4+ slots benefit immediately: 274 network-flow, 275 matroid, 280 SBM, 282 hypergraph CC).
2. The first scalar-function-based topology summary in reality (currently `topology/persistent/` only summarizes simplex-filtration topology, never function-on-complex topology).
3. The most heavily-cited primitive in scientific visualization (Pistachio terrain analysis is the obvious internal consumer; molecular surfaces, MRI, weather data are all standard applications).
4. A combinatorial-only subpackage with **no algebraic prerequisites** (no Eigvec, no Smith form, no LP) — **ships TODAY without waiting on slot 097-T1 / slot 077 Delaunay / slot 277 BnB**.

T5 Reeb-on-manifold ships next-PR after slot-283 T0 SimplexTree. T6/T7/T8 ship orthogonally on T2-T4 substrate. T9 Reeb-space + T10c edit-distance ship after slot-283 (Jacobi set) + slot-277 (BnB) respectively.

## Cross-cutting

- **Pistachio terrain analysis ←** T4 ContourTree on terrain elevation function f(x,y) reveals ridge/valley network. T6 BranchDecomposition + T7 SimplifyByPersistence ranks features by topological persistence — **direct consumer pull**, the canonical use-case for contour trees in visualization.
- **Pistachio scene-topology / loop-closure ←** T5 ReebGraph on a scene-depth function f reveals scene structure (number of "objects" = β_0 of Reeb graph; loops in scene = β_1). Cross-link to slot-283 β_1 loop-closure pin.
- **Aicore molecular dynamics / electrostatic potential surfaces ←** T5 Reeb graph of electrostatic potential on molecular surface identifies binding pockets. Standard tool in computational chemistry (Lazaridis-2006).
- **MRI brain segmentation ←** T4 ContourTree + T7 SimplifyByPersistence on MRI intensity → topological hierarchical segmentation (Carr-Snoeyink-vandePanne 2004 *J-Real-Time-Image* 11:127).
- **Slot 097-T1 (linalg-missing) ←** **NO blocker.** All Reeb T0-T8 ships unblocked by Eigvec.
- **Slot 077/078 (geometry) ←** **NO blocker for graph case.** T5 ReebGraph on triangulated mesh consumes triangle list directly (no Delaunay).
- **Slot 142 (topology-missing) ←** explicit punt absorbed; **single-source ownership**: Reeb / contour-tree / merge-tree under `topology/reeb/`.
- **Slot 156 (synergy-topology-prob) ←** T7 SimplifyByPersistence + Cohen-Steiner-Edelsbrunner-Harer-2007 stability theorem → Bayesian merge-tree inference from noisy samples (Curry-Mukherjee-Turner 2018).
- **Slot 171 (synergy-graph-topology) ←** Reeb graph as a graph object — consumes graph package primitives (BFS for connectivity in upper/lower link).
- **Slot 274 (network-flow), 275 (matroid), 280 (SBM), 282 (hypergraphs) ←** T0 UnionFind unblocks all four. **Highest-leverage day-1 primitive.**
- **Slot 277 (combinatorial-optimization, BnB) ←** T10c MergeTreeEditDistance is **second concrete consumer** (after slot-285-T3 optimal DMT). Pin: 277 → 286-T10c.
- **Slot 281 (temporal-graphs) ←** T11 time-varying Reeb composes 281's temporal-graph substrate with 286-T8 streaming Reeb. Defer till 281 lands.
- **Slot 283 (simplicial-complexes) ←** T5 ReebGraph + T9 ReebSpace + T10 Jacobi-set gate on 283-T0 SimplexTree (need link / cofaces / triangle list).
- **Slot 284 (cell complexes / cubical) ←** **R-MUTUAL-CROSS-VALIDATION pin #2 cross-package**: cubical 0-dim persistence (lower-star T-construction) ≡ JoinTree on cubical 1-skeleton restricted to f. Saturates on raster fixture.
- **Slot 285 (discrete Morse) ←** Bauer-Edelsbrunner-2014 — merge-tree cancellation = Forman cancellation on 0-Hasse. T7 SimplifyByPersistence ≡ slot-285 cancellation pairs at 0-dim. Composition: slot-285 DMT collapse → slot-286 merge-tree on critical-cell-only complex with f restricted = same merge tree at 99% fewer cells.
- **Workshop ecosystem topology dashboard ←** named at `topology/persistent/doc.go:28-30`. Reeb graph is the *visualization-friendly* topology summary (1-complex graph, easily rendered) versus persistence diagrams (point cloud, less interpretable).
- **Insights blast-radius topology ←** T5 ReebGraph of dependency-depth function on service graph reveals layered structure of cascading-failure regions.
- **Structure-from-motion (sfm) ←** Reeb graph of camera-trajectory time-function over 3D reconstruction reveals topology of acquisition path (Carlsson-2009 §5).
- **Shape matching ←** T10 InterleavingDistance / EditDistance on merge trees of two shapes' geodesic-distance functions = topological shape-matching distance (Hilaga-Shinagawa-Kohmura-Kunii 2001 SIGGRAPH precursor).
- **Feature tracking in time-varying simulations ←** T8 streaming Reeb + T11 time-varying Reeb over fluid-flow / weather data tracks vortex / front evolution.

## Sources

- `graph/mst.go:34-90` (ad-hoc UnionFind to extract), `graph/types.go:7,14`, `graph/graph.go`
- `topology/persistent/vr.go:14,50,91`, `topology/persistent/barcode.go:60,221,254`, `topology/persistent/doc.go:28-30,118-124`
- `linalg/eigen.go:20` (Eigvec — NOT a blocker for this slot)
- Reviews: `agents/097-linalg-missing.md` (NO blocker for this slot — note explicitly), `agents/142-topology-missing.md` (explicit punt of contour-tree absorbed here), `agents/156-synergy-topology-prob.md`, `agents/171-synergy-graph-topology.md`, `agents/277-new-copo.md` (BnB consumer pin), `agents/281-new-temporal-graphs.md` (T11 future composition), `agents/283-new-simplicial-complexes.md` (T0 substrate for T5+), `agents/284-new-cw-complexes.md` (cubical-persistence cross-validation pin), `agents/285-new-discrete-morse.md` (Bauer-Edelsbrunner cancellation theory composition)
- Reeb-1946 *Comptes Rendus* 222:847 "Sur les points singuliers d'une forme de Pfaff"
- Edelsbrunner-Harer-2010 *Computational Topology: An Introduction* AMS, Chapters VI-VII (canonical merge/contour-tree textbook reference)
- Carr-Snoeyink-Axen-2003 *Computational Geometry* 24:75 "Computing contour trees in all dimensions" (THE algorithm)
- Pascucci-Cole-McLaughlin-2002 *IEEE Vis* "Parallel computation of the topology of level sets"
- Edelsbrunner-Harer-Patel-2008 *SoCG* "Reeb spaces of piecewise linear mappings"
- Edelsbrunner-Harer-2002 *FoCM* "Jacobi sets of multiple Morse functions"
- Doraiswamy-Natarajan-2009 *IEEE-TVCG* 15:1697 "Efficient algorithms for computing Reeb graphs"
- Tierny-Gyulassy-Simon-Pascucci-2009 *IEEE-TVCG* 15:1177 "Loop surgery for volumetric meshes"
- Edelsbrunner-Harer-Mascarenhas-Pascucci-2004 *SoCG* "Time-varying Reeb graphs for continuous space-time data"
- Pascucci-Scorzelli-Bremer-Mascarenhas-2007 *ACM-SIGGRAPH* "Robust on-line computation of Reeb graphs"
- Pascucci-Cole-McLaughlin-Scorzelli-2004 *IEEE-Vis* "The branch decomposition hierarchy"
- Cohen-Steiner-Edelsbrunner-Harer-2007 *Discrete-Comput-Geom* 37:103 "Stability of persistence diagrams" (stability cross-validation pin)
- Bauer-Munch-Wang-2015 *SoCG* "Strong equivalence of interleaving and functional distortion metrics for Reeb graphs"
- Morozov-Beketayev-Weber-2013 *TopoInVis* "Interleaving distance between merge trees"
- Beketayev-Yeliussizov-Morozov-Phillips-Weber-Hamann-2014 *TopoInVis* "Measuring the distance between merge trees"
- Sridharamurthy-Masood-Kamakshidasan-Natarajan-2020 *IEEE-TVCG* 26:1518 "Edit distance between merge trees"
- Wetzels-Leitte-Garth-2022 *IEEE-TVCG* 28:1197 "Branch decomposition-independent edit distances for merge trees"
- Carr-Weber-Sewell-Snoeyink-2017 *IEEE-TVCG* 23:921 "Multiscale contour tree"
- Lukasczyk-Maack-Edelsbrunner-2020 *Comput-Graph-Forum* "Dynamic Reeb graphs"
- Edelsbrunner-Letscher-Zomorodian-2002 *Discrete-Comput-Geom* 28:511 "Topological persistence and simplification" (elder rule reference)
- Bauer-Edelsbrunner-2014 *J-Topol-Anal* 6:531 "Morse theory of Čech and Delaunay" (cancellation = Forman, slot-285 composition)
- libtourtre BSD-3 C++ ~3k LOC (Carr-Snoeyink-Axen reference impl, archived)
- Topology ToolKit (TTK) BSD-3 ~50k LOC (Tierny et al., SOTA visualization)
- VTK vtkReebGraphFilter / vtkContourTreeFilter BSD-3
