# 287 — new-mapper (Mapper Algorithm: pullback / cover / nerve, multinerve, ball, persistent)

## Headline
reality v0.10.0 ships ZERO Mapper / Ball-Mapper / Multinerve-Mapper / Cover / Lens / Eccentricity / Nerve / Pullback / DBSCAN / SingleLinkage surface (`Mapper|BallMapper|Multinerve|Cover|Lens|Pullback|Nerve|Eccentricity|DBSCAN|SingleLinkage|AverageLinkage` repo-wide grep on `*.go` outside reviews/ returns ZERO callable hits, with the only filter-function precursor being `linalg/pca.go::PCA` at line 33 which composes verbatim as the most popular Mapper lens); slot 283 explicitly **punts Mapper to this slot at T10.23** (line 143 of 283 review) and slot 286 plans an extracted `graph/dsu.go::UnionFind` + `topology/reeb/types.go::ScalarField` interface that **Mapper composes verbatim** (Mapper of f: X→R with intervals = Reeb graph in the limit of overlap → 1 — Singh-Memoli-Carlsson-2007 §2.3); the cheapest day-1 PR is `topology/mapper/` package (T0+T1+T2+T3+T4 ≈ 480 LOC) shipping FilterFunction interface + Cover interface + 1D uniform-overlap cover + Mapper algorithm + single-linkage-on-preimage clustering, gating only on slot-286-T0 UnionFind extraction (50 LOC, 4-slot leverage independently established) — and **R-MUTUAL-CROSS-VALIDATION 3/3 saturates day 1** via the three-way pin Mapper(f, R, intervals) ≡ Reeb graph (slot-286-T4 ContourTree) ≡ Multinerve-Mapper (T8) on synthetic-known-Reeb fixtures (S^1 = circle f=height → Mapper = circle graph; standard Munch-Wang-2016 convergence test); pure-Go MIT zero-dep ABSENT in any language (kepler-mapper Apache-2 Python, GUDHI mapper BSD-3 C++, Ayasdi proprietary, Mapper-Interactive MIT Python at github.com/MathieuCarriere/sklearn-tda).

## Findings

### State at HEAD (verified by direct grep on `*.go`)

| Surface | Path | Mapper-relevance |
|---|---|---|
| `linalg.PCA` | `linalg/pca.go:33` | **Composes verbatim as the most-cited Mapper filter function (lens).** Singh-Carlsson-2007 §3.1: PCA-1 / PCA-2 lens projects high-dim data to R or R^2 then covers image. |
| `topology/persistent/{vr,barcode}.go` | — | VR + ELZ barcode. **Disjoint axis from Mapper** — VR consumes pointcloud + distance and emits barcode; Mapper consumes pointcloud + filter + cover and emits *graph* (1-skeleton of nerve). Mapper is the "TDA for data scientists" alternative — easier to interpret, cheaper to compute, more popular in industry (Ayasdi cancer-subtype, Lum-2013). |
| `graph/types.go::Edge` | `graph/types.go:7` | 1-skeleton substrate; Mapper graph IS a graph object. Reuse verbatim. |
| `audio/separation/ica.go::ICA` | `audio/separation/ica.go` | Independent-components — secondary lens choice (less common than PCA). |
| `linalg/decompose.go` | — | LU/QR/Cholesky/SVD building blocks for PCA-lens. SVD already powers `linalg.PCA`. |
| repo-wide grep: `Mapper\|BallMapper\|Multinerve\|Cover\|Lens\|Pullback\|Nerve\|Eccentricity\|DBSCAN\|SingleLinkage\|AverageLinkage\|HierarchicalCluster\|HDBSCAN\|MinClusterSize` | `*.go` outside reviews/ | **ZERO callable hits.** No clustering primitives at all (no k-means, no DBSCAN, no hierarchical / single-linkage, no HDBSCAN). |

### Slot boundaries (no overlap with 283/284/285/286, explicit pull from 277)

- **Slot 283 (simplicial complexes) ←** Slot-283 T10.23 line 143 explicitly punts `topology/mapper/mapper.go` to **this slot**. Mapper *output* is a SimplicialComplex (the **nerve** of the cover-pullback), so the Mapper graph is `topology/simplicial.SimplicialComplex.Skeleton(1)`. Composition: Mapper-graph = 1-skeleton of Mapper-simplicial-complex; Mapper-2-complex (T7 below) keeps triangles where 3 clusters all share ≥1 point. **Direct upstream gate:** Mapper-simplicial-complex (T7) on slot-283 SimplexTree once 283-T0 lands. **Mapper-graph (T3) ships earlier on slot-286 graph 1-skeleton.**
- **Slot 286 (Reeb graphs) ←** **Mapper IS a generalisation of Reeb graph.** Singh-Memoli-Carlsson-2007 §2.3 — when filter f: X→R, cover = overlapping intervals on R, and clustering = connected-components within preimage, **Mapper-graph converges to Reeb graph as cover-resolution → ∞ and overlap → 0+**. So 287 directly consumes 286-T0 ScalarField interface (extending it from R to R^d codomain) and 286-T4 ContourTree as the *correct-answer regression target* on simply-connected domains. **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#1) below saturates this immediately.**
- **Slot 285 (discrete Morse) ←** No direct pin. Mapper does not consume DMT.
- **Slot 284 (cubical complexes) ←** Mapper of a raster scalar field with intervals on R = cubical-merge-tree, which is itself = cubical 0-dim persistence (slot 284). Indirect 3-way pin via slot-286.
- **Slot 271 (spectral-clustering) ←** Optional clustering inside-preimage. Spectral clustering is an **alternative to single-linkage / DBSCAN** for the per-preimage clusterer in T3-Mapper. Reality must ship at least one preimage-clusterer; single-linkage (T4) is cheapest (composes UnionFind from slot-286). Spectral / DBSCAN ship as alternative back-ends behind the `Clusterer` interface.
- **Slot 273 (spectral-embedding) ←** Alternative lens: Laplacian-eigenmaps lens (Belkin-Niyogi-2003) = use first-d Laplacian eigenvectors as filter. Composes verbatim as a `FilterFunction` once 273 lands.
- **Slot 272 (manifold-learning) ←** UMAP / t-SNE / ISOMAP can also be lenses (Carrière-Michel-2018 use UMAP-2 in their experiments). Composes verbatim once 272 lands.
- **Slot 277 (combinatorial-optimisation, ILP / B&B) ←** **NEW pin established by this slot:** *optimal Mapper cover* is an ILP (Carrière-Oudot-2018 §4.3) — given pointcloud P + filter f, find minimum-cardinality cover with fixed overlap that produces a stable Mapper graph. Formalise: variables = 0-1 indicator per candidate-cover-element (intervals from finite candidate set); constraints = each pointcloud point covered by at least one element with overlap ≥ ε; objective = minimise cardinality + persistence-stability term. **Second concrete consumer of slot-277 BnB after slot-286-T10c MergeTreeEditDistance.** Pin: 277 → 287 advanced cover optimisation.
- **Slot 097-T1 (linalg-missing) ←** **NO blocker.** Mapper is purely combinatorial after clustering. PCA already ships at `linalg/pca.go:33`. No Eigvec needed.
- **Slot 077/078 (geometry) ←** **NO blocker** for 1D / hexagonal / Voronoi covers in low-dim. T6 generic Voronoi-cover gates on `geometry.Voronoi` if Mapper is to use Voronoi cover — but ships standalone with 1D + hexagonal covers.
- **Slot 142 (topology-missing) ←** 142 punts Mapper to a future slot — **this slot owns it**. Same situation as slot-286 owning contour-tree from 142. Re-read 142 after 287 lands.
- **Slot 156 (synergy-topology-prob) ←** Statistical Mapper (Carrière-Michel-2018) — bootstrap confidence on Mapper graph; consumes prob-package bootstrap.
- **Slot 281 (temporal-graphs) ←** Time-varying Mapper — track Mapper-graph topology evolution; defer to 281 substrate.

### Web context (canonical reference set, MIT pure-Go zero-dep ABSENT)

- **Singh-Memoli-Carlsson-2007** *Eurographics SPBG* "Topological methods for the analysis of high dimensional data sets and 3D object recognition" — **the** Mapper paper. Algorithm: (1) pick filter `f: X → R^d` (the *lens*); (2) cover image `f(X) ⊆ R^d` by overlapping sets `{U_i}`; (3) for each `U_i`, cluster the preimage `f^{-1}(U_i) ⊆ X` into connected pieces `{V_{i,j}}`; (4) form **nerve** simplicial complex with vertices = `{V_{i,j}}` and `k`-simplex `{V_{i_0,j_0}, ..., V_{i_k,j_k}}` whenever `∩ V_{i_l,j_l} ≠ ∅`. The 1-skeleton (vertices + edges) is the **Mapper graph**. Generalises Reeb graph (cover = intervals on R → Mapper = Reeb in limit).
- **Lum-Singh-Lehman-Ishkanov-Vejdemo-Johansson-Alagappan-Carlsson-Carlsson-2013** *Sci-Rep* 3:1236 "Extracting insights from the shape of complex data using topology" — **THE Mapper application paper.** Cancer-subtype discovery (rediscovers known subtypes + finds a previously-uncharacterised survival-correlated subgroup), basketball-player position re-categorisation, voter-preference clustering. Established Mapper as the practical TDA technique for industry. **Ayasdi** (founded by Carlsson + Singh + Memoli) commercialised it.
- **Carrière-Michel-2018** *JMLR* 19:1 "Statistical analysis and parameter selection for Mapper" — **stochastic-stability / cover-parameter selection** for Mapper. Bootstrap subsamples → estimate confidence on each Mapper-graph edge → prune unstable edges. Provides theoretical bounds on number of intervals + overlap for asymptotic Reeb-graph convergence. **Foundation for T9 stochastic Mapper.**
- **Carrière-Oudot-2018** *Found-Comput-Math* 18:1333 "Structure and stability of the one-dimensional Mapper" — **stability theorem** for Mapper graphs. Bottleneck distance between Mapper-graphs of `f` and `g` bounded by `‖f − g‖_∞` plus cover-resolution term. **Cross-validation pin: Mapper stability ≡ Cohen-Steiner-2007 persistence stability ≡ Bauer-Munch-Wang-2015 Reeb interleaving stability.**
- **Munch-Wang-2016** *SoCG* "Convergence between categorical representations of Reeb space and Mapper" — **Multinerve Mapper** generalisation. Modifies the nerve construction to record full intersection-pattern of cover-elements (not just pairwise) → provides provable convergence of Multinerve-Mapper to true Reeb-space (Edelsbrunner-Harer-Patel-2008) as cover-resolution → ∞. Standard Mapper does NOT have this convergence (Carrière-Oudot show counterexamples). **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#1):** Multinerve-Mapper(f, fine cover) ≡ Reeb graph (slot-286) ≡ standard Mapper-graph(f, fine cover with overlap → 0).
- **Dey-Memoli-Wang-2017** *J-Comput-Geom* 8:128 "Topological analysis of nerves, Reeb spaces, mappers, and multiscale Mapper" — **multiscale persistent Mapper.** Vary cover-resolution as a filtration parameter; track how Mapper-graph topology changes. Produces a *persistence diagram of Mapper graphs* (analogous to Cohen-Steiner persistence). **Foundation for T10 multiscale Mapper.**
- **Dłotko-2019** *arXiv:1901.07410* "Ball Mapper: a shape summary for topological data analysis" — **Ball-Mapper.** Single-parameter alternative: cover the *pointcloud itself* with metric balls of radius ε (no filter function needed); vertices = balls; edge if balls intersect. **No filter function** — captures geometric shape directly. Used in finance for market-structure topology (Dłotko-Quian-2020 *Topology and its Applications*). Trivial to implement (~150 LOC) once metric is supplied.
- **Mapper-Interactive / Zhou-2021** *IEEE-VIS* "Mapper Interactive: A scalable, extendable, and interactive toolbox for the visual exploration of high-dimensional data" — **parallel + interactive Mapper.** GPU-accelerated cover-element processing; user can drag cover parameters and see Mapper update in real time. MIT Python (github.com/MathieuCarriere/sklearn-tda derivative). Foundation for **T11 streaming Mapper.**
- **Chazal-Cohen-Steiner-Glisse-Guibas-Oudot-2009** *SoCG* "Proximity of persistence modules and their diagrams" — interleaving distance between persistence modules; theoretical foundation that Carrière-Oudot-2018 builds on for Mapper stability.
- **Singh-2007** + **Carlsson-2009** *Bull-AMS* 46:255 "Topology and Data" — TDA survey containing Mapper §6.
- **Standard Mapper filter functions** (Lum-2013 §2 inventory):
  - **PCA-1 / PCA-2 projection** — first 1 or 2 principal components. Composes `linalg/pca.go::PCA`.
  - **Density estimate** `f(x) = 1/n ∑_y K_h(x − y)` with Gaussian kernel `K_h`. Standard for "denser regions deeper in Mapper-graph."
  - **Eccentricity** `e_p(x) = (1/n ∑_y d(x, y)^p)^{1/p}` — average distance to all other points; large at periphery, small at centre. Lum-2013 use eccentricity-1 (= mean distance) as primary lens for cancer-subtype Mapper.
  - **L-infinity centrality** `c(x) = − max_y d(x, y)` — eccentricity in sup-norm; identifies extremes.
  - **Graph centrality** (PageRank / closeness) when underlying X has a graph structure. Composes `graph/centrality.go`.
- **Standard cover types:**
  - **1D uniform cover** — split `[min, max]` of `f(X)` into `n` overlapping intervals of equal length `L = (max − min)/n` and overlap `p ∈ [0, 1)`. **Cheapest cover; ships day 1.**
  - **2D hexagonal cover** — for `f: X → R^2`, hexagonal lattice with overlap. Standard in Mapper-Interactive 2D-lens visualisations.
  - **Voronoi cover** — adaptive cover from a sample of cover-centres; Voronoi cells with overlap-buffer. Adaptive to data density. Gates on `geometry.Voronoi`.
- **Standard preimage clusterers** (per Singh-2007 §2.4):
  - **Single-linkage** with elbow-cutoff at largest hierarchical-clustering gap. **The Mapper-default clustering choice.** Composes UnionFind from slot-286.
  - **DBSCAN** — density-based; handles noise. Extra dependency: rangeCount per point.
  - **k-means** — requires `k` choice; usually rejected for Mapper (Mapper's *job* is to find `k`).
  - **Average-linkage / complete-linkage** — alternatives to single-linkage.
- **Open-source landscape:**
  - **kepler-mapper** Apache-2 Python (github.com/scikit-tda/kepler-mapper) — most-used Mapper impl. ~3k LOC. Sklearn-style API.
  - **GUDHI Mapper** BSD-3 C++/Python (gudhi.inria.fr) — newer Mapper module by Carrière. Ships Statistical Mapper bootstrap.
  - **Mapper-Interactive** MIT Python — Zhou-2021 (github.com/MathieuCarriere/sklearn-tda) — parallel + interactive UI.
  - **TDA-tools / GIOTTO-TDA** Apache-2 Python — Mapper as part of larger TDA stack.
  - **Ayasdi** proprietary commercial — original Mapper deployment.
  - **R TDA** GPL R — basic Mapper.
  - **Pure-Go MIT zero-dep ABSENT for ALL of:** Mapper, Ball-Mapper, Multinerve-Mapper, Multiscale-Mapper, Statistical Mapper, Stochastic Mapper, parallel Mapper, lens functions (PCA-Mapper-lens / density / eccentricity / L-inf-centrality), 1D/2D uniform covers, hexagonal covers, Voronoi covers, Mapper-graph stability bounds, Mapper-graph distance, Mapper-graph 1-skeleton extraction, nerve construction, single-linkage with elbow cutoff, DBSCAN. Reality has the **single largest open MIT zero-dep gap in TDA-for-data-scientists** in any language ecosystem — and Mapper is the canonical "TDA technique that crossed over to industry" (Ayasdi cancer subtypes, finance market-structure).

## Concrete recommendations

Recommended placement: **NEW sub-package `topology/mapper/`** with sibling sub-packages `topology/mapper/cover/`, `topology/mapper/lens/`, `topology/mapper/cluster/`. Builds on slot-286-T0 UnionFind, slot-286-T0 ScalarField (extended to R^d), slot-283-T0 SimplicialComplex (for T7 nerve-as-complex). Optional ILP-cover gate on slot-277.

### T0 — FilterFunction interface + Cover interface (~120 LOC, gates on slot-286 ScalarField R^d extension)

1. **`topology/mapper/types.go::FilterFunction` (~40 LOC).** Generalisation of slot-286 `ScalarField` from `R` to `R^d`:
   ```go
   // FilterFunction: lens / filter mapping each input point to R^d.
   type FilterFunction interface {
       NumPoints() int
       Codim() int                       // d in R^d; typically 1 or 2
       Value(i int) []float64            // len == Codim()
       BoundingBox() (lo, hi []float64)  // axis-aligned bbox of f(X) for cover construction
   }
   type GenericFilter struct { Vals [][]float64 }   // pre-computed lens; ships standalone
   ```
   Extends slot-286 `ScalarField` (which had `Value(v) float64`) to vector-valued. Pin: `ScalarField` ≡ `FilterFunction` with `Codim() == 1`.

2. **`topology/mapper/types.go::Cover` (~50 LOC).** Cover of `f(X) ⊆ R^d` by overlapping sets:
   ```go
   // Cover: family of overlapping subsets of R^d that together contain f(X).
   type Cover interface {
       NumElements() int
       Contains(elem int, p []float64) bool      // is p ∈ U_elem?
       PullbackElements(p []float64) []int       // {elem : p ∈ U_elem} — typically ≤ 2 in 1D w/overlap < 1
   }
   ```

3. **`topology/mapper/types.go::Clusterer` (~30 LOC).** Pluggable preimage clustering:
   ```go
   // Clusterer: partition the preimage f^{-1}(U_elem) ⊆ X into connected pieces.
   type Clusterer interface {
       Cluster(pointIdx []int, ambient FilterFunction) [][]int
       // returns partition of pointIdx into clusters (each cluster a list of original indices)
   }
   ```

### T1 — 1D uniform interval cover (~80 LOC, simplest, ships day 1)

4. **`topology/mapper/cover/interval.go::IntervalCover(numIntervals int, overlap float64) Cover` (~80 LOC).** 1D uniform cover. Given filter range `[a, b]`, split into `n` intervals of length `L = (b − a) / n`, then *expand each interval by `overlap × L`* (overlap ∈ [0, 1) is the *fractional overlap* between adjacent intervals; standard default 0.5). Result: each `R`-point falls in ≤ 2 intervals (provided `overlap < 0.5`). Singh-Carlsson-2007 §2.3 default. **Cheapest cover; ZERO blocker.**
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#0, cover-coverage):** every point in `f(X)` is contained in `≥1` cover element (`PullbackElements` non-empty); for `overlap = 0`, exactly 1; for `overlap > 0`, in `[1, 2]`.

### T2 — Single-linkage clustering with elbow cutoff (~140 LOC, composes slot-286 UnionFind verbatim)

5. **`topology/mapper/cluster/singlelinkage.go::SingleLinkage(points [][]float64, metric Metric) *Dendrogram` (~80 LOC).** Single-linkage hierarchical clustering via Kruskal-MST on the complete distance graph (Gower-Ross-1969 *Appl-Stat* 18:54). **Direct re-use of slot-286 `graph/dsu.go::UnionFind`** + `graph/mst.go::KruskalMST` (already present). Returns dendrogram = sequence of merges.
6. **`topology/mapper/cluster/elbow.go::ElbowCutoff(d *Dendrogram) float64` (~30 LOC).** Largest-gap heuristic: cut dendrogram at the merge where the inter-merge-distance jump is largest (Singh-Carlsson-2007 §2.4, kepler-mapper default). Returns the cluster partition.
7. **`topology/mapper/cluster/wrapper.go::SingleLinkageClusterer{Metric}` (~30 LOC).** Implements `Clusterer` interface.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#0b, MST regression):** single-linkage merges = Kruskal-MST edges in increasing weight order. Three-way pin: SingleLinkage ≡ Kruskal-MST ≡ UnionFind sweep.

### T3 — Generic Mapper algorithm: pullback → cluster → graph (~180 LOC, ships day 1 once T0/T1/T2 land)

8. **`topology/mapper/mapper.go::MapperGraph(f FilterFunction, cov Cover, cl Clusterer) *MapperGraphT` (~180 LOC).** Singh-Memoli-Carlsson-2007 algorithm:
   ```
   for elem in 0..cov.NumElements():
       preimagePoints := { i : cov.Contains(elem, f.Value(i)) }
       clusters[elem] := cl.Cluster(preimagePoints, f)
   nodes := flatten clusters into [(elem, j, points)] tuples
   edges := for each pair of nodes (a, b) with a.elem ≠ b.elem,
            add edge if a.points ∩ b.points ≠ ∅
   return MapperGraphT{Nodes, Edges}
   ```
   Returns `MapperGraphT{Nodes []*Node; Edges [][2]int; PointToNodes map[int][]int}` where each Node carries `(CoverElem int, ClusterID int, Points []int, Centroid []float64)`. Edges are computed by intersecting Points sets between cover-adjacent clusters (overlap > 0 implies cover-adjacent clusters share preimage points). **Time:** O(K · cluster-cost) where K = number of cover elements; for single-linkage with n_i preimage points, O(K · n_i^2 log n_i).
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#1, Mapper ≡ Reeb in limit):** for `f: X → R` with cover = fine intervals + overlap → 0+, MapperGraph(f, intervals) IS isomorphic to slot-286-T4 ContourTree(f) on simply-connected X. Three-way: MapperGraph(f, fine) ≡ ContourTree (slot 286) ≡ JoinTree (slot 286 T2, since simply connected ⇒ contour-tree = merge-tree). **Saturates R-MUTUAL-CROSS-VALIDATION 3/3 day 1** on synthetic-known-Reeb fixtures (height function on disk, height function on annulus → annulus shows non-trivial loop in Mapper).

### T4 — Standard filter functions (~220 LOC)

9. **`topology/mapper/lens/pca.go::PCALens(data [][]float64, ncomp int) FilterFunction` (~30 LOC).** Composes `linalg/pca.go::PCA` directly. **Cheapest standard lens; SOTA Lum-2013 default.**
10. **`topology/mapper/lens/density.go::DensityLens(data [][]float64, h float64) FilterFunction` (~80 LOC).** Gaussian-kernel density estimator: `f(x) = (n h^d)^{-1} ∑_y exp(−‖x − y‖^2 / (2 h^2))`. Bandwidth `h` defaults to Silverman's rule. **Identifies dense vs sparse regions** — central use-case in Mapper (cancer Mapper: dense = canonical subtype, sparse = tail / outlier).
11. **`topology/mapper/lens/eccentricity.go::EccentricityLens(data [][]float64, p float64) FilterFunction` (~50 LOC).** `e_p(x) = (1/n ∑_y d(x, y)^p)^{1/p}`. p=1 = mean-distance lens (Lum-2013 default for cancer Mapper); p=∞ = `max d(x, y)` = L∞-centrality.
12. **`topology/mapper/lens/centrality.go::LInfCentralityLens(data [][]float64) FilterFunction` (~30 LOC).** `c(x) = -max_y d(x, y)`. Identifies extremes.
13. **`topology/mapper/lens/graph.go::PageRankLens(g graph.Graph) FilterFunction` (~30 LOC).** Composes `graph/centrality.go::PageRank` if present. Lens for graph-data Mapper.

### T5 — DBSCAN clustering (~180 LOC, alternative to single-linkage; SHIPS STANDALONE for prob/cluster too)

14. **`topology/mapper/cluster/dbscan.go::DBSCAN(points [][]float64, eps float64, minPts int) [][]int` (~140 LOC).** Ester-Kriegel-Sander-Xu-1996 *KDD* "A density-based algorithm for discovering clusters." Each point classified as core (≥ minPts neighbours within eps), border, or noise; clusters formed by transitively connecting core points. **Time O(n log n) with a kd-tree, O(n²) naive.** Reality has no kd-tree → ship the O(n²) version; kd-tree gates on slot 077 / 225.
15. **`topology/mapper/cluster/wrapper.go::DBSCANClusterer{Eps, MinPts}` (~40 LOC).** Implements `Clusterer` interface.
   - **Cross-link:** DBSCAN is the canonical SOTA density-based clusterer (cited > 50,000 times). Ships standalone for the prob / cluster ecosystem in addition to powering Mapper. **Reality has zero clustering primitives today** — DBSCAN + single-linkage alone fill a major gap; the day-1 PR establishes the `topology/mapper/cluster/` (or `cluster/` top-level) package as reality's clustering home.

### T6 — 2D hexagonal + Voronoi covers (~260 LOC, second-PR)

16. **`topology/mapper/cover/hex2d.go::HexCover(numU, numV int, overlap float64) Cover` (~140 LOC).** 2D hexagonal lattice cover for `f: X → R^2`. Each hex cell expanded by overlap × cell-radius. Standard for Mapper-Interactive 2D lens visualisations (PCA-2 lens + hex cover).
17. **`topology/mapper/cover/voronoi.go::VoronoiCover(centers [][]float64, overlap float64) Cover` (~120 LOC, gates on `geometry.Voronoi`).** Adaptive Voronoi cover; gates on slot-077 `geometry.Voronoi2D` / `Voronoi3D`. Defer.

### T7 — Mapper as full simplicial complex (1-skeleton + higher) (~140 LOC, gates on slot-283-T0 SimplicialComplex)

18. **`topology/mapper/nerve.go::MapperComplex(f, cov, cl) *simplicial.SimplicialComplex` (~140 LOC, gates on 283-T0).** Generalise T3-MapperGraph to full nerve-simplicial-complex: `k`-simplex `{V_{i_0,j_0}, ..., V_{i_k,j_k}}` for every set of `k+1` clusters with non-empty common-points intersection. **MapperGraph = 1-skeleton of MapperComplex.** Used for higher-order topology (β_2 of Mapper-complex of 2D-lens data reveals voids in data).
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#2, 1-skeleton):** `MapperComplex(...).Skeleton(1) ≡ MapperGraph(...)` bit-identical edge-set on every fixture.

### T8 — Multinerve Mapper with Reeb-convergence guarantee (~220 LOC, gates on T7 + slot-286-T4 ContourTree)

19. **`topology/mapper/multinerve.go::MultinerveMapper(f, cov) *MultinerveT` (~220 LOC).** Munch-Wang-2016 algorithm: refine the nerve construction to track *full intersection-component patterns* (not just non-empty intersection). Provides provable convergence to true Reeb space as cover-resolution → ∞.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#3, Reeb-convergence saturation):** on synthetic-known-Reeb fixtures (S^1 height function, S^2 height function, torus height function, annulus height function), Multinerve-Mapper(f, fine) converges to true Reeb-graph (slot-286 ContourTree on simply connected, ReebGraph T5 on torus) within fixed bottleneck distance ≤ cover-resolution. **Three-way pin saturates: Multinerve-Mapper ≡ Reeb-graph ≡ standard-Mapper-graph(small-overlap, fine-cover).**

### T9 — Stochastic / Statistical Mapper (Carrière-Michel-2018) (~200 LOC, gates on slot-156 bootstrap)

20. **`topology/mapper/stochastic.go::StatisticalMapper(f, cov, cl, nBoot int) *MapperConfidence` (~200 LOC).** Bootstrap: subsample `f(X)` `nBoot` times → compute Mapper graph for each subsample → estimate edge-confidence (fraction of bootstraps in which edge exists). Carrière-Michel §4. Output: `MapperConfidence{Edges []*Edge; PvalUpperBound []float64}`. **Foundation for cancer-Mapper p-value reporting** (Lum-2013 manually-justified subtype-significance; this primitive automates it).

### T10 — Multiscale persistent Mapper (Dey-Memoli-Wang-2017) (~280 LOC, frontier)

21. **`topology/mapper/multiscale.go::MultiscaleMapper(f, coverFamily func(scale float64) Cover) *MultiscaleMapperT` (~280 LOC).** Vary cover-resolution as filtration parameter; emit *persistence diagram of Mapper-graphs* tracking when edges/nodes appear and disappear with cover-resolution. **Cross-validation pin:** in the limit, multiscale-Mapper-persistence ≡ persistent homology of the Reeb-space (Dey-2017 main theorem). **Composes slot-283 PersistenceTwist** when 283 lands.

### T11 — Ball-Mapper (Dłotko-2019) (~150 LOC, single-parameter alternative; ZERO blocker)

22. **`topology/mapper/ballmapper.go::BallMapper(points [][]float64, eps float64, metric Metric) *MapperGraphT` (~150 LOC).** Cover the *pointcloud itself* with metric balls of radius ε (no filter function). Greedy ball-packing: while uncovered points remain, pick one as a new ball-centre, mark all within ε as covered. Vertices = balls; edge `(b_i, b_j)` if `points(b_i) ∩ points(b_j) ≠ ∅`. **No filter function — captures geometric shape directly.** Used in finance market-structure (Dłotko-Quian-2020). **Trivial implementation; ships standalone day 2.**
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#4, Ball-Mapper ≡ ε-Cech 1-skeleton):** for fixed ε, Ball-Mapper graph IS the 1-skeleton of the ε-Cech complex on points (or 1-skeleton of ε-VR complex if metric is Euclidean and ε small). Cross-validates against `topology/persistent.VietorisRipsComplex` 1-skeleton.

### T12 — Mapper-graph stability bounds (Carrière-Oudot-2018) (~180 LOC, frontier)

23. **`topology/mapper/stability.go::MapperStabilityBound(f1, f2 FilterFunction, cov Cover) float64` (~180 LOC).** Carrière-Oudot-2018 Theorem 4.5: bottleneck distance between Mapper-graphs of f1 and f2 ≤ `2 ‖f1 − f2‖_∞ + cover-resolution`. Returns the upper bound. Used for confidence intervals in Statistical Mapper.

### T13 — Optimal Mapper cover via ILP (gates on slot-277 BnB) (~220 LOC)

24. **`topology/mapper/cover/optimal.go::OptimalCover(f, candidates []Interval, eps float64) Cover` (~220 LOC, gates on slot-277-T1 BnB).** Carrière-Oudot-2018 §4.3 cover-optimisation. ILP: variables = 0-1 indicator per candidate cover-element; constraint = each point covered by ≥1 element with overlap ≥ eps; objective = minimise cardinality + persistence-stability term. **Second concrete consumer of slot-277 BnB after slot-286-T10c MergeTreeEditDistance.** Defer until 277 ships.

### T14 — Streaming / parallel Mapper (Zhou-2021) (~340 LOC, defer)

25. **`topology/mapper/streaming.go::StreamingMapper(stream <-chan Point, cov, cl) *StreamingMapperT` (~340 LOC).** Process points in streaming fashion: maintain per-cover-element preimage point-set; recompute affected clusters incrementally on point arrival. Defer to second-PR; depends on T3 substrate.

## Single cheapest day-1 PR

**`topology/mapper/` package, T0 (1–3) + T1.4 + T2 (5–7) + T3.8 ≈ 480 LOC, gates on slot-286-T0 UnionFind extraction landing first.**

- `topology/mapper/types.go` — FilterFunction + Cover + Clusterer interfaces (T0.1+T0.2+T0.3, ~120 LOC)
- `topology/mapper/cover/interval.go` — 1D uniform-overlap cover (T1.4, ~80 LOC)
- `topology/mapper/cluster/singlelinkage.go` — Kruskal-MST single-linkage + elbow cutoff + wrapper (T2.5+T2.6+T2.7, ~140 LOC)
- `topology/mapper/mapper.go` — MapperGraph algorithm (T3.8, ~140 LOC)

**Tests (R-MUTUAL-CROSS-VALIDATION 3/3 pins saturate on day 1):**
- `TestIntervalCover_Coverage` — pin #0; every point of `f(X)` in ≥1 cover element; for overlap < 0.5, in [1, 2] elements.
- `TestSingleLinkage_KruskalRegression` — pin #0b; single-linkage merges = Kruskal-MST edges in increasing weight order on every fixture.
- `TestMapperGraph_HeightOnCircle` — synthetic `S^1` (n=64 points sampled on unit circle), filter f = y-coordinate, intervals=8, overlap=0.5: Mapper graph ≡ cycle graph C_2 (the canonical Mapper-of-S^1 textbook example, Singh-2007 Fig. 4). Pin #1 partial.
- `TestMapperGraph_ReebRegression_DiskHeight` — synthetic disk + height function: MapperGraph(f, fine intervals, overlap → 0+) edge-set converges to slot-286-T4 ContourTree of f on the disk's 1-skeleton. **Pin #1 saturated.**
- `TestMapperGraph_ReebRegression_AnnulusHeight` — synthetic annulus + height function: ContourTree has β_1=1 (one loop); MapperGraph at fine resolution ALSO has β_1=1 — three-way: MapperGraph β_1 = ContourTree β_1 = topological β_1 of annulus = 1. **Pin #1 fully three-way saturated.**
- `TestMapperGraph_LumCancerSubtypeFixture` — small-n Lum-2013 synthetic-cancer fixture (e.g. 4 known subtype clusters in PCA-2 space): MapperGraph with PCA-1 lens + 8 intervals + overlap 0.5 + single-linkage produces 4 connected components after deletion of low-confidence edges. Regression-test against Lum-2013 figure 2.

This PR ships:
1. The first clustering primitives in reality (single-linkage + Kruskal-MST elbow cutoff) — base for further DBSCAN / k-means / spectral-clustering.
2. The first **Mapper algorithm** in reality — direct cancer-subtype / market-structure / scene-topology consumer pull.
3. A combinatorial-only sub-package with **no algebraic prerequisites** beyond the slot-286 UnionFind extraction.
4. Three-way R-MUTUAL-CROSS-VALIDATION 3/3 pin saturation: Mapper(fine) ≡ Reeb (slot 286) ≡ canonical β_1 on synthetic annulus / S^1 / disk fixtures — **the Carrière-Oudot-2018 stability theorem in regression-test form.**

T4 (lens functions) ships independently second-PR, requires nothing new. T5 DBSCAN ships independently. T7 MapperComplex gates on slot-283-T0 SimplexTree. T8 MultinerveMapper + T10 MultiscaleMapper compose slot-283 + slot-286 once both land. T13 OptimalCover gates on slot-277 BnB.

## Cross-cutting

- **Slot 286 (Reeb graphs) ←** R-MUTUAL-CROSS-VALIDATION 3/3 pin #1 saturates day 1 via MapperGraph(f, fine intervals) ≡ ContourTree(f) on simply-connected fixtures + ReebGraph(f) on annulus / torus fixtures. **Mapper IS Reeb in the cover-resolution → ∞ limit, overlap → 0+ limit (Singh-Memoli-Carlsson-2007 §2.3, Munch-Wang-2016 main theorem).**
- **Slot 286 — UnionFind extraction blocker.** T2.5 single-linkage composes `graph/dsu.go::UnionFind` verbatim. **287 cannot ship until 286-T0 lands** (50 LOC). Trivial 4-slot-leverage primitive; should ship independently regardless.
- **Slot 283 (simplicial complexes) ←** Mapper-output is a simplicial complex (the nerve). T7 MapperComplex gates on 283-T0 SimplexTree. **Slot-283 line 143 explicitly punts Mapper to this slot — single-source ownership confirmed.**
- **Slot 277 (combinatorial-optimisation, BnB) ←** **NEW pin established by this slot:** T13 OptimalCover via ILP. **Second consumer of slot-277 BnB after slot-286-T10c MergeTreeEditDistance.** Pin: 277 → 287 advanced cover optimisation. Strengthens 277's downstream-pull case.
- **Slot 156 (synergy-topology-prob) ←** T9 Statistical-Mapper bootstrap composes prob bootstrap; Cohen-Steiner-2007 + Carrière-Oudot-2018 stability bounds ≡ slot-156 stability theorem.
- **Slot 271 (spectral-clustering), 273 (spectral-embedding), 272 (manifold-learning) ←** Alternative lenses (UMAP, t-SNE, ISOMAP, Laplacian-eigenmaps) and alternative preimage-clusterers (spectral clustering). Compose verbatim once 271/272/273 land. Mapper is the *aggregator* of these as filter functions.
- **Slot 142 (topology-missing) ←** explicit Mapper punt absorbed; **single-source ownership**: Mapper / Ball-Mapper / Multinerve / Multiscale-Mapper under `topology/mapper/`.
- **Slot 097-T1 (linalg-missing) ←** **NO blocker.** Mapper is purely combinatorial after clustering. PCA already exists at `linalg/pca.go:33`.
- **Slot 077/078 (geometry) ←** **NO blocker for 1D + hexagonal + Ball-Mapper.** T6 Voronoi cover gates on geometry.Voronoi.
- **Slot 225 (ANN), 077 kd-tree ←** T5 DBSCAN ships O(n²); kd-tree-accelerated O(n log n) DBSCAN gates on 225-T1 / 077 kd-tree. Defer.
- **Slot 281 (temporal-graphs) ←** Time-varying Mapper consumes 281's temporal-graph substrate. Defer.
- **Pistachio scene-topology / loop-closure ←** Mapper of feature-track lens (e.g. PCA of feature-descriptor vectors) reveals scene-graph topology — direct alternative to slot-283 β_1. Cross-link to slot-286 ReebGraph but Mapper is the **interpretable visualisation primitive** (1-complex graph that humans can read off).
- **Aicore high-dim data exploration ←** Mapper of gene-expression matrix (cells × genes) with PCA-1 + density lens = canonical cancer-subtype Mapper (Lum-2013). **Direct consumer pull**; Mapper is THE TDA technique that crossed over to industry biotech.
- **Workshop ecosystem topology dashboard ←** named at `topology/persistent/doc.go:28-30` and slot-283 line 189. Mapper is the *human-interpretable* topology summary (vs persistence diagrams).
- **Insights blast-radius topology ←** Mapper of service-graph with PageRank lens reveals cascading-failure region structure as connected components of Mapper-graph (Lum-2013-style "centrality clusters in Mapper of social network").
- **Drug discovery ecosystem (chemical-space exploration) ←** Mapper of chemical-fingerprint-distance matrix with eccentricity lens reveals chemical-class topology (Cang-Wei-2017-style applications generalised to Mapper).
- **Financial market structure topology ←** Ball-Mapper (T11, Dłotko-Quian-2020) on stock-correlation distance matrix reveals market-regime topology. **Direct consumer-pull for Ball-Mapper as standalone day-2 PR.**
- **Neural-net loss-landscape topology ←** Mapper of weight-space samples with loss-value lens reveals loss-landscape topology (recent line of work: Horoi-2022 *NeurIPS* "Exploring the geometry and topology of neural network loss landscapes via persistent homology" — Mapper variant in §4).

## Sources

- `linalg/pca.go:33` (PCA — composes verbatim as cheapest Mapper lens), `audio/separation/ica.go` (ICA — secondary lens)
- `graph/types.go:7` (Edge — Mapper-graph 1-skeleton substrate), `graph/mst.go:34-90` (Kruskal-MST — single-linkage substrate), `graph/centrality.go` (PageRank lens for graph data)
- `topology/persistent/vr.go:14,91`, `topology/persistent/barcode.go:60`, `topology/persistent/doc.go:28-30,118-124`
- Reviews: `agents/283-new-simplicial-complexes.md` line 143 (T10.23 Mapper punt to this slot — single-source ownership), `agents/286-new-reeb.md` (UnionFind extraction blocker, ScalarField → FilterFunction extension, Reeb ≡ Mapper-fine-cover convergence pin), `agents/277-new-copo.md` (T13 OptimalCover ILP downstream pin), `agents/142-topology-missing.md` (Mapper explicit punt absorbed), `agents/156-synergy-topology-prob.md` (Statistical-Mapper bootstrap composition), `agents/171-synergy-graph-topology.md`, `agents/271-new-spectral-clustering.md` (spectral-clusterer back-end alternative), `agents/272-new-manifold-learning.md` (UMAP / t-SNE / ISOMAP lens alternative), `agents/273-new-spectral-embedding.md` (Laplacian-eigenmaps lens alternative), `agents/284-new-cw-complexes.md`, `agents/285-new-discrete-morse.md`
- Singh-Memoli-Carlsson-2007 *Eurographics SPBG* "Topological methods for the analysis of high dimensional data sets and 3D object recognition" (the Mapper paper)
- Lum-Singh-Lehman-Ishkanov-Vejdemo-Johansson-Alagappan-Carlsson-Carlsson-2013 *Sci-Rep* 3:1236 "Extracting insights from the shape of complex data using topology" (the Mapper-application paper, cancer subtypes)
- Carrière-Michel-2018 *JMLR* 19:1 "Statistical analysis and parameter selection for Mapper"
- Carrière-Oudot-2018 *Found-Comput-Math* 18:1333 "Structure and stability of the one-dimensional Mapper" (stability theorem)
- Munch-Wang-2016 *SoCG* "Convergence between categorical representations of Reeb space and Mapper" (Multinerve-Mapper, Reeb-convergence guarantee)
- Dey-Memoli-Wang-2017 *J-Comput-Geom* 8:128 "Topological analysis of nerves, Reeb spaces, mappers, and multiscale Mapper"
- Dłotko-2019 *arXiv:1901.07410* "Ball Mapper: a shape summary for topological data analysis"
- Dłotko-Quian-2020 *Topology and its Applications* "Topological data analysis on the financial markets" (Ball-Mapper finance application)
- Zhou-Chen-Wang-Liu-2021 *IEEE-VIS* "Mapper Interactive: A scalable, extendable, and interactive toolbox for the visual exploration of high-dimensional data"
- Chazal-Cohen-Steiner-Glisse-Guibas-Oudot-2009 *SoCG* "Proximity of persistence modules and their diagrams" (Carrière-Oudot-2018 theoretical foundation)
- Cohen-Steiner-Edelsbrunner-Harer-2007 *Discrete-Comput-Geom* 37:103 "Stability of persistence diagrams" (cross-validation pin)
- Bauer-Munch-Wang-2015 *SoCG* "Strong equivalence of interleaving and functional distortion metrics for Reeb graphs" (Reeb stability ≡ Mapper stability cross-package pin)
- Ester-Kriegel-Sander-Xu-1996 *KDD* "A density-based algorithm for discovering clusters" (DBSCAN)
- Gower-Ross-1969 *Appl-Stat* 18:54 "Minimum spanning trees and single linkage cluster analysis" (single-linkage = Kruskal-MST)
- Belkin-Niyogi-2003 *Neural-Comput* 15:1373 "Laplacian eigenmaps for dimensionality reduction" (Laplacian-eigenmaps lens)
- Carlsson-2009 *Bull-AMS* 46:255 "Topology and Data" (TDA + Mapper survey)
- Edelsbrunner-Harer-Patel-2008 *SoCG* "Reeb spaces of piecewise linear mappings" (Multinerve-Mapper convergence target)
- Horoi-Huang-Rieck-Wolf-Hamprecht-Krishnaswamy-2022 *NeurIPS* "Exploring the geometry and topology of neural network loss landscapes via persistent homology" (Mapper-of-loss-landscape application)
- kepler-mapper Apache-2 Python (github.com/scikit-tda/kepler-mapper) — most-used Mapper impl, ~3k LOC
- GUDHI Mapper BSD-3 C++/Python (gudhi.inria.fr) — Carrière's Mapper module incl. Statistical-Mapper
- Mapper-Interactive MIT Python (Zhou-2021)
- giotto-tda Apache-2 Python — Mapper as part of larger TDA stack
- Ayasdi proprietary commercial — original Mapper deployment (Carlsson + Singh + Memoli)
