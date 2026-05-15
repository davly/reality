# 171 | synergy-graph-topology

**Topic:** graph × topology — clique complex, flag complex, network homology.
**Block:** B (cross-package synergies). **Date:** 2026-05-08. **Scope:**
capabilities that emerge ONLY when `graph/` and `topology/persistent/` are
composed; not in-package gaps (096-100 own graph, 141-145 own topology, 142-T1.1
names "flag complex from a graph" topology-side, 156 topology×prob, 157
graph×linalg, 162 graph×prob).

## Two-line summary

`graph/` ships 12 deterministic algorithm files (~1,400 LOC: BFS / Dijkstra /
Bellman-Ford / Brandes-betweenness / Tarjan-SCC / Louvain / Kruskal+Prim MST /
PageRank / EigenvectorCentrality / Edmonds-Karp / Kahn-topo / NodeImportance)
keyed on `IntAdjacency = map[int][]int` and `[]Edge` string-pairs, while
`topology/persistent/` ships exactly three primitives (`VietorisRipsComplex` on
a point cloud, `ComputeBarcode` F_2 column-reduction, `BottleneckDistance`
Hopcroft-Karp+binary-search) hard-wired to `points [][]float64` Euclidean input
and `maxDim ∈ {0,1}` (~1,378 LOC); the entire graph-as-input simplicial canon —
clique/flag complex from a weighted graph, independence complex, neighborhood
complex (Lovász 1978), Reeb graph, Mapper (Singh-Memoli-Carlsson 2007),
magnitude/magnitude-homology (Leinster 2010, Hepworth-Willerton 2017), Ihara
zeta + Bass identity, Z_2 cycle-space basis, PH of a weighted graph filtered by
edge weight, heat-kernel signature, chromatic-number topological lower bound —
is wholly absent (zero matches across both packages on
`Clique|Flag|Independence|Neighborhood|Reeb|Mapper|Magnitude|Ihara|Bass|
HeatKernel|CycleSpace|Chromatic`; only `flag` hit is Bellman-Ford `hasNegCycle`).

Sixteen synergy primitives (T1-T16) totalling ~2,420 LOC of pure connective
tissue close the gap; **cheapest one-day PR is T1 `FlagComplex` + T3
`WeightedGraphPersistence` + T9 `BettiNumbers` (~210 LOC)** because the
`topology/persistent/vr.go:111-134` triangle-enumeration loop is the morally
identical inner kernel — lifting `points + maxRadius` to
`distMatrix + threshold` is a pure refactor — saturating the Tether import-graph
+ Insights blast-radius graph-PH consumers `topology/persistent/doc.go:36-41`
explicitly anticipates; **highest-leverage architectural lift is T10
NeighborhoodComplex + T11 LovaszChromaticBound (~320 LOC, T11 gated on
142-T1.2 maxDim≥2)** since Lovász 1978's chromatic lower bound is the
textbook hello-world for combinatorial topology that graph/ has zero coverage
of; **crown jewel is T6 `MapperGraph` (~380 LOC)** — the single most-pulled
non-PH primitive in scikit-tda/giotto-tda/KeplerMapper, composes
`graph.ConnectedComponents` + caller-supplied lens with zero new mathematics.

---

## Bases (verified file-walk)

### graph/ (12 files, ~1,400 LOC)

`graph.go` AdjacencyList/Nodes/InDegree/Roots/Leaves; `types.go`
`IntAdjacency = map[int][]int`; `bfs.go` BFSDownstream/BFSReachable;
`bellman_ford.go` BellmanFord; `centrality.go` BetweennessCentrality (Brandes)
/ EigenvectorCentrality (power-iter on adjacency) / DegreeCentrality;
`community.go` ConnectedComponents / StronglyConnected (Tarjan) /
LouvainCommunities; `dag.go` DAGDepth/ReachableLeaves; `flow.go` MaxFlow
(Edmonds-Karp) / TopologicalSort (Kahn); `importance.go` NodeImportance /
EdgeFraction; `mst.go` KruskalMST / PrimMST (load-bearing for T8);
`pagerank.go` PageRank; `shortest.go` Dijkstra / AStar / FloydWarshall.
**Zero math/rand calls; zero linalg import; zero topology import.**

### topology/persistent/ (5 files, ~1,378 LOC)

`vr.go`: `Simplex []int`, `Filtration{Simplices, Times}`,
`VietorisRipsComplex(points, maxRadius, maxDim)` — Euclidean point cloud only,
maxDim ∈ {0,1}. `barcode.go`: `Bar{Dim,Birth,Death}`, `ComputeBarcode`
(Edelsbrunner-Letscher-Zomorodian 2000 F_2 column reduction).
`bottleneck.go`: Hopcroft-Karp + binary search. **Zero graph import.**

The `vr.go:111-134` triangle loop walks all `(i,j,k)` triples and emits a
simplex with filtration time = max pairwise distance. The only point-cloud
coupling is `pairwiseDistanceMatrix(points)` at vr.go:110 and the
`maxRadius`-vs-distance comparison at vr.go:114,128. **Lift those two lines
to a caller-supplied `dist [][]float64` matrix and the loop becomes a
flag-complex builder on any weighted graph.** This is the entire content of T1.

---

## What does NOT exist anywhere in the repo

Verified by grep across all 22 packages: no FlagComplex / CliqueComplex /
IndependenceComplex / NeighborhoodComplex / weighted-graph PH / MapperGraph /
ReebGraph / Magnitude / IharaZeta / BassIdentity / CycleSpaceBasis /
HeatKernelSignature / Hochster formula / chromatic topological bound. **The
synergy is 100% greenfield** — no existing connective code to refactor.

---

## The sixteen synergy primitives

Each: (1) capability, (2) composition, (3) LOC, (4) blocking flag.

### T1 — `FlagComplex(adj, weights, maxThreshold, maxDim) Filtration` (~120 LOC)

(1) Build the flag (clique) complex of a weighted graph filtered by edge
weight: k-simplex {v_0..v_k} included with birth = max pairwise edge weight iff
all (k+1 choose 2) edges exist with weight ≤ maxThreshold. The single bridge
between the two packages. Generalises VR (point cloud → distance graph → flag
complex of proximity graph) to correlation graphs, service graphs, social
networks. (2) Refactor `topology/persistent/vr.go:91-166` to extract inner
triangle-enumeration into `flagFromDistMatrix(dist, maxRadius, maxDim)`; expose
`FlagComplex(adj graph.IntAdjacency, weights map[[2]int]float64, threshold,
maxDim) Filtration` materialising adjacency-derived dist (∞ off-edge,
edge-weight on-edge). VR becomes the special case
`FlagComplex(completeGraph(points), euclideanDist, maxRadius, maxDim)`.
(3) 120 (40 refactor + 60 graph-shaped wrapper + 20 dim≥2 lift). Placement
`topology/persistent/flag.go`. Refactor non-breaking. Ref: Zomorodian 2010.

### T2 — `IndependenceComplex(adj, n)` (~80 LOC)

(1) Simplicial complex whose simplices are the independent sets of G — the
clique complex of the **complement graph**. Fundamental in matching theory,
Lovász local lemma, chromatic bounds (T11). (2) Build complement (~25 LOC),
delegate to T1 with unit weights at threshold 1.0. (3) 80.

### T3 — `WeightedGraphPersistence(adj, weights, maxDim) []Bar` (~60 LOC)

(1) PH of a weighted graph filtered by edge weight: H_0 bars (components knit
as edges of decreasing weight enter), H_1 bars (independent cycles). The
keystone consumer for **Tether import-graph topology** and **Insights
blast-radius topology** that doc.go:36-41 explicitly anticipates — both are
graph-shaped, neither today calls VR (would have to lie that their graph is a
Euclidean point cloud). (2) T1 ⟹ ComputeBarcode. (3) 60.

### T4 — `EulerCharacteristic(filtration) []float64` (~40 LOC)

(1) χ(t) = Σ_k (-1)^k f_k(t), f_k(t) = #{k-simplices born ≤ t}. Cheapest TDA
feature for time-series classification (giotto-tda baseline). (2) Walk
Filtration.Simplices in order, signed-sum dim-keyed counter at each
filtration time. (3) 40.

### T5 — `BettiCurve(bars, dim, grid) []int` (~30 LOC)

(1) Step function "how many H_k bars alive at t." Single-dim companion to T4.
(2) For each grid t, count `bar.Birth ≤ t < bar.Death`. (3) 30.

### T6 — `MapperGraph(points, lensFn, cover, clusterFn) IntAdjacency` (~380 LOC) — CROWN JEWEL

(1) Singh-Memoli-Carlsson 2007: project point cloud through scalar lens (e.g.
density, eccentricity, PC-1), partition lens range into overlapping intervals,
cluster each preimage, emit nerve graph (vertices = clusters; edges = nonempty
intersection across consecutive intervals). The single most-pulled non-PH
primitive in scikit-tda / giotto-tda / KeplerMapper. (2) Rolling cover over
`[min(lens), max(lens)]` with caller `nIntervals` + `overlapFraction` (~80
LOC). For each interval, single-linkage cluster preimage via
`graph.ConnectedComponents` of proximity graph at scale ε with default
"first-jump-in-dendrogram" heuristic (~120 + 60 LOC). Emit nerve as
`IntAdjacency` with edges between clusters from consecutive intervals sharing
≥ 1 point (~40 LOC). (3) 380. Placement `topology/persistent/mapper.go`. Ref:
Singh-Memoli-Carlsson SPBG '07.

### T7 — `ReebGraph(adj, weights, scalar) IntAdjacency` (~280 LOC)

(1) Reeb graph of f: V → R on G — contract each connected component of each
level set f^{-1}(c) to a point. Captures topology of level-set foliation
(births/deaths of components as threshold sweeps). Standard mesh-processing
primitive (libigl, MeshLab, GUDHI). (2) Sweep f-sorted vertices left-to-right;
incremental union-find of currently-active components inside sub-level set
f ≤ f(v) using ConnectedComponents semantics adapted to streaming insertion
(~160 LOC, simplified Edelsbrunner-Harer 2008). Emit Reeb-adjacency + node
attrs (f-value at birth/merge). (3) 280. Ref: Reeb 1946; Edelsbrunner-Harer
2008.

### T8 — `CycleSpaceBasis(adj, n) [][]int` (~100 LOC)

(1) Basis for the Z_2 cycle space of an undirected graph as fundamental cycles
(edge-index lists). dim Z_2 cycle-space = |E| − |V| + c (the first Betti
number β_1). The most fundamental "network homology" object and the cheapest
H_1 generator computable from existing primitives. (2) `graph.KruskalMST`
already partitions edges into tree + cotree. For each cotree edge (u,v), the
unique path u → v in the MST plus (u,v) is a fundamental cycle (BFS in tree,
~50 LOC). β_1 = |cotree|. (3) 100. Placement `graph/cycle.go`.

### T9 — `BettiNumbers(adj, n) (b0, b1)` (~30 LOC)

(1) β_0 = #components, β_1 = #independent cycles. Two most basic topological
invariants. Today the user calls `graph.ConnectedComponents` + does the
arithmetic by hand. (2) ConnectedComponents ⟹ b0; b1 := |E| − n + b0.
(3) 30.

### T10 — `NeighborhoodComplex(adj, n) Filtration` (~80 LOC)

(1) Lovász 1978 N(G): simplices are sets of vertices sharing a common
neighbor. (2) Iterate N(v) over each v, collect each as maximal simplex,
close under face inclusion (~40 LOC). Birth = 0 (unfiltered; consumers can
promote with weights). (3) 80.

### T11 — `LovaszChromaticBound(adj, n) int` (~240 LOC, BLOCKED on 142-T1.2)

(1) Lovász 1978: χ(G) ≥ conn(N(G)) + 3 — topological lower bound on chromatic
number via connectivity of the neighborhood complex. Famously settled
Kneser's conjecture. Zero-mathematics extension of T10 + higher-dim
ComputeBarcode. (2) T10 ⟹ ComputeBarcode (lifted) ⟹ search smallest k with
H̃_k ≠ 0 ⟹ +Lovász offset = lower bound on χ. (3) 240 (60 logic + 180
maxDim≥2 lift; the lift is also called by T7 beyond H_1, T15 magnitude
homology). (4) **Blocked on 142-T1.2** ComputeBarcode lifted to maxDim ≥ 2
(twist + clearing + sparse columns; ~150 LOC standalone). Ref: Lovász
JCT-A 25 1978.

### T12 — `IharaZeta(adj, n) func(complex128) complex128` (~180 LOC)

(1) ζ_G(u) = ∏_{[C]} (1 − u^|C|)^{-1} over equivalence classes of primitive
backtrack-free closed walks. Graph analogue of the Riemann zeta. **Bass 1992
identity** ζ_G(u)^{-1} = (1 − u²)^{|E|−|V|} det(I − u·A + u²·(D − I)) reduces
this to a determinant. (2) Build A (157-G1) and D, parameterise M(u) = I −
u·A + u²·(D − I), evaluate det(M(u)) at any u ∈ ℂ. Returns closure
evaluating ζ at any complex u; consumers chain to their own root-finder.
(3) 180 (40 wrapper + 60 complex-det via Bareiss + 80 complex-arith helpers).
(4) Soft-blocked on linalg complex-matrix support (today linalg.Determinant
real-only); pure-real workaround unblocks day-one PR (loses half spectral
data). Ref: Ihara 1966; Bass 1992.

### T13 — `BassIdentityWitness(adj, n, u) (lhs, rhs)` (~50 LOC)

(1) Cross-validation pin: compute ζ from infinite-product side (truncated
to walks of length ≤ N) AND from Bass determinant side; assert
|lhs − rhs| < tol. **R-MUTUAL-CROSS-VALIDATION** idiom matching commits
6a55bb4 (audio/onset 3-detector) and 365368a (Clayton autodiff vs analytic).
(2) Walk-enumeration via DFS for lhs (~30 LOC); T12 for rhs. (3) 50.

### T14 — `MagnitudeFunction(adj, weights, n, t) float64` (~120 LOC)

(1) Leinster 2010 magnitude |G|(t): Z_{ij} = exp(−t·d(i,j)), |G|(t) = sum of
all entries of Z^{-1}. Captures effective number of points at scale 1/t:
|G|(0) = n, |G|(∞) = #isolated. (2) graph.FloydWarshall ⟹ Z = exp(−t·D) ⟹
linalg.LUSolve(Z, 1) ⟹ sum entries. (3) 120. Ref: Leinster Doc. Math. 18 2010.

### T15 — `MagnitudeHomology(adj, n, maxDim, maxLen) []Bar` (~320 LOC, BLOCKED)

(1) Hepworth-Willerton 2017 categorification of T14: bigraded chain complex
MC_{k,ℓ}(G), magnitude = alternating Betti sum across ℓ. (2) Enumerate
length-ℓ paths (BFS, ~120 LOC); build (k,ℓ)-bigraded complex with horizontal
differential (~150 LOC); reduce per-ℓ slice via lifted ComputeBarcode (~50
LOC). (3) 320. (4) Blocked on 142-T1.2 (maxDim ≥ 2) **and** ComputeBarcode
generalisation to arbitrary chain complexes (not just VR-derived filtrations,
~80 LOC abstraction lift). Ref: Hepworth-Willerton HHA 19(2) 2017.

### T16 — `HeatKernelSignature(adj, n, ts) [][]float64` (~150 LOC, BLOCKED)

(1) Sun-Ovsjanikov-Guibas 2009: HKS_v(t) = (e^{−tL})_{vv} — canonical shape
signature for graph-classification (giotto-tda + GUDHI). Spectral form
Σ_i exp(−t·λ_i) · φ_i(v)². (2) 157-G2 Laplacian ⟹ 157-G6 Fiedler extended
to top-k pairs ⟹ assemble HKS. (3) 150. (4) Blocked on 157-G2 (~25 LOC) +
157-G6 (~80 LOC) + 097-T1 linalg Eigvec (full eigenvectors). Same blockers
as 157-G7 / 157-G9. Ref: Sun-Ovsjanikov-Guibas SGP '09.

---

## Roll-up

| # | Primitive | LOC | Day-1? |
|---|-----------|-----|--------|
| T1 | FlagComplex | 120 | yes |
| T2 | IndependenceComplex | 80 | yes (depends T1) |
| T3 | WeightedGraphPersistence | 60 | yes (depends T1) |
| T4 | EulerCharacteristic | 40 | yes |
| T5 | BettiCurve | 30 | yes |
| T6 | **MapperGraph** | 380 | yes |
| T7 | ReebGraph | 280 | yes |
| T8 | CycleSpaceBasis | 100 | yes |
| T9 | BettiNumbers | 30 | yes |
| T10 | NeighborhoodComplex | 80 | yes (depends T1) |
| T11 | LovaszChromaticBound | 240 | blocked: 142-T1.2 |
| T12 | IharaZeta | 180 | yes (real) / blocked complex |
| T13 | BassIdentityWitness | 50 | yes (depends T12) |
| T14 | MagnitudeFunction | 120 | yes |
| T15 | MagnitudeHomology | 320 | blocked: 142-T1.2 + bigraded |
| T16 | HeatKernelSignature | 150 | blocked: 157-G2/G6 + 097-T1 |
| | **Total** | **2,420** | **2,150 today; 270 gated** |

---

## R-MUTUAL-CROSS-VALIDATION pin candidates

Three pin opportunities (matching 6a55bb4 audio-onset / 365368a
Clayton-autodiff idiom):

- **β_1 cross-validation**: T8 CycleSpaceBasis count (`|E|−|V|+β_0`)
  cross-validates T3 WeightedGraphPersistence count of H_1 bars at threshold
  +∞; both must agree on β_1. ~40 LOC pin.
- **Bass identity**: T13 frames this directly. ~50 LOC pin.
- **Magnitude vs Euler at t→∞**: T14 at large t equals #components which
  equals β_0 from T9 which equals ConnectedComponents count. Three-way pin
  ~30 LOC.

---

## Recommended placement

Mirroring 158-160 / 165-170 consumer-side-placement precedent — **synergy
lives in the package that shapes the API surface, not the package providing
primitives**:

- `topology/persistent/flag.go` (T1, T2, T3)
- `topology/persistent/curves.go` (T4, T5)
- `topology/persistent/mapper.go` (T6)
- `topology/persistent/reeb.go` (T7)
- `topology/persistent/neighborhood.go` (T10, T11)
- `topology/persistent/magnitude.go` (T14, T15)
- `graph/cycle.go` (T8, T9) — pure-graph output
- `graph/zeta.go` (T12, T13) — graph-spectral object
- `graph/spectral.go` (T16, alongside 157-G10)

Cycle-free edges: graph/ → topology/persistent/ never goes back;
topology/persistent/ optionally imports graph/ (today does not — T1 / T2 / T3
/ T6 / T7 / T10 / T11 introduce the import).

---

## Distinctness from prior reviews

- **141-145 topology isolation**: 142-T1.1 names "flag complex from a graph"
  topology-side at ~80 LOC but does NOT connect to graph/ data structures or
  Mapper/Reeb/Ihara — this review's T1 lifts that to a graph-shaped public
  API and traces how T3, T8, T11 collapse out of it.
- **096-100 graph isolation**: enumerates spectral graph theory absence
  (Laplacian, Fiedler, GFT) but not TDA-on-graphs synergy.
- **156 topology × prob**: PD statistics, FLRSW bootstrap, persistence
  landscapes; orthogonal to graph-input direction.
- **157 graph × linalg**: spectral graph theory (Laplacian, Fiedler, Cheeger,
  GFT, heat kernel as linalg object); 157-G10 HeatKernelSignature overlaps
  with T16 here — confirmed shared placement at `graph/spectral.go`.
- **162 graph × prob**: random-graph generators (ER/SBM/BA/percolation) and
  ERGM/latent-space inference; orthogonal to topology.
- **170 info × prob**: T11 Lovász has Shannon-shaped fractional-chromatic
  variant — out of scope here.

The graph-input direction of TDA is genuinely under-reviewed — first agent
in 400-sequence to compose `topology/persistent` with `graph/`-shaped inputs.
142-T1.1 named the gap but did not synergize.

---

## Precision and stability hazards

- **F_2 boundary on graph-derived complexes**: identical to F_2 boundary on
  VR-derived complexes; `topology/persistent/barcode.go:79-101` is dim-agnostic
  and works unchanged.
- **Disconnected graphs**: β_0 ≥ 2; T9 / T8 must accept |E|−|V|+β_0 ≥ 0
  invariant; T3 emits multiple essential H_0 bars (one per component) with
  Death = +Inf each.
- **Self-loops and multi-edges**: flag complex normally on simple graphs; T1
  rejects self-loops and dedupes parallel edges (or document min-weight per
  parallel-edge convention).
- **Mapper cluster-count instability**: T6's ConnectedComponents-at-scale-ε
  inherits ε's filtration-instability; pin Singh-Memoli-Carlsson 2007 default
  (first-jump-in-dendrogram).
- **Magnitude at small t**: T14 ill-conditioned as t → 0 (Z → all-ones).
  Document validity range t > 0 strict; emit ErrIllConditioned at
  cond(Z) > 1e12.
- **Ihara zeta at poles**: ζ_G poles ↔ graph-spectrum eigenvalues; T12
  returns ±Inf or NaN per IEEE-754 convention, not error.
- **Cycle-space basis non-uniqueness**: cycle space unique, bases are not;
  T8 returns one MST-cotree-derived basis with deterministic tie-break by
  edge-index lex order for golden-file parity.

---

## Cheapest day-1 PR

T1 + T3 + T9 + T8 + T13 = 360 LOC code + ~120 LOC tests over ~2
engineer-days. Lands the Tether/Insights graph-PH consumer gap, pins
Bass identity (R-MUTUAL pin saturation), and the β_1 three-way pin
across T8/T9/T3. This is the recommended landing.
