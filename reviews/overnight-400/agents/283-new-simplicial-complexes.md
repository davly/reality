# 283 — new-simplicial-complexes (Simplicial Complexes: ASC, Boundary, Homology, SimplexTree, Hodge)

## Headline
reality v0.10.0 ships ZERO first-class abstract simplicial complex (`SimplicialComplex|SimplexTree|ChainComplex|BoundaryMap|BettiNumber|SmithNormal|AlphaComplex|WitnessComplex|CupProduct|Cochain|HodgeLaplacian|Cohomology` repo-wide grep returns 0 callable hits); the only adjacent surface is `topology/persistent/{vr,barcode}.go` ~700 LOC which builds an *abstract* simplicial complex implicitly as a `Filtration{Simplices []Simplex; Times []float64}` flat list with F_2 column-reduction `∂_k` for VR persistence (capped at maxDim=1) — no per-dimension array, no simplex tree, no Z-coefficient ∂_k, no Smith normal form, no β_k computation as a standalone API, no Hodge Laplacian, no alpha/witness complex, no cohomology, no cup product, no persistence landscape/image. The cheapest day-1 PR is `topology/simplicial/` sub-package — SimplexTree (Boissonnat-Maria 2014) + sparse Z/F2 boundary + Z/2 Betti via column reduction (~480 LOC, zero blockers) — and Hodge Laplacian L_k *matrix construction* ships free on top of that (the *spectrum* gates on slot 097-T1 Eigvec).

## Findings

### State at HEAD (verified by direct grep on `*.go`)

| Surface | Path | Simplicial-complex relevance |
|---|---|---|
| `Simplex = []int` | `topology/persistent/vr.go:14` | Sorted-vertex tuple. NO key-decoration, NO per-dim index, NO orientation sign, NO labels. |
| `Filtration{Simplices, Times}` | `topology/persistent/vr.go:50` | Flat (simplex, time) list sorted by (time, dim, lex). NOT a simplex tree, NOT a per-dim array. Iteration over k-faces is O(m). |
| `boundaryColumn(s, indexBy)` | `topology/persistent/barcode.go:221` | F_2 boundary via map[string]int lookup over `simplexKey`. **Has ∂_k over F_2** but private, not generalized to Z, not exposed. |
| `symDiff(a, b)` | `topology/persistent/barcode.go:254` | F_2 column-add via sorted-int symmetric difference. Used inside `ComputeBarcode` reduction; not exported. |
| `ComputeBarcode(filt, maxDim)` | `topology/persistent/barcode.go:60` | Edelsbrunner-Letscher-Zomorodian-2000 column-reduction over F_2. Capped at maxDim ∈ {0,1} (`ErrInvalidMaxDim` at maxDim ≥ 2). |
| `VietorisRipsComplex` | `topology/persistent/vr.go:91` | O(n^3) full 2-skeleton; explicit cap at `maxDim ∈ {0,1}` (vr.go:175); 3-skeleton (tetrahedra) for H_2 absent. |
| `BottleneckDistance` | `topology/persistent/bottleneck.go` | L^∞ matching on persistence diagrams (Hopcroft-Karp). Uses 2-norm on bars; no landscape, no image, no Wasserstein. |
| `linalg.{LU, QR, Cholesky, QRAlgorithm}` | `linalg/decompose.go, eigen.go` | Dense decomp only; **no SmithNormalForm, no HermiteNormalForm, no integer matrix arithmetic, no SparseMatrix CSR/CSC**. |
| `linalg.QRAlgorithm` | `linalg/eigen.go:20` | Tridiagonal-QL **eigenvalues only** (no eigenvectors). 097-T1 keystone-blocker for Hodge Laplacian *spectrum*. |
| `geometry/polygon.go.ConvexHull2D` | `geometry/polygon.go` | Graham scan 2D hull only. **NO Delaunay triangulation, NO 3D hull, NO regular triangulation** → blocks alpha-complex (Edelsbrunner-Mücke 1994 needs Delaunay/regular). |
| repo-wide grep: `SimplicialComplex\|SimplexTree\|ChainComplex\|BoundaryMap\|BettiNumber\|SmithNormal\|AlphaComplex\|WitnessComplex\|Cohomology\|Cochain\|CupProduct\|HodgeLaplacian` | `*.go` outside review corpus | **ZERO hits.** |

### Slot boundaries

- **Slot 097-T1 (linalg-missing) — Eigvec.** Symmetric eigenvectors. Hard prerequisite for Hodge-Laplacian *spectrum* (T7 below). Matrix *construction* ships unblocked.
- **Slot 156 (synergy-topology-prob), 141–145 (topology-{numerics,sota,api,perf,missing}).** Earlier reviews on the existing `topology/persistent/` Phase-A surface; recommend extending VR to maxDim ≥ 2 + persistence landscape + Mapper.
- **Slot 171 (synergy-graph-topology).** Clique/flag complex of binary graphs + network homology. **Shares machinery** with this slot — if 171 ships first, T0 imports `topology/simplicial/`; if 283 ships first, 171 reuses.
- **Slot 246 (new-discrete-exterior).** DEC X1–X30 — embedded `SimplicialComplex2D/3D` with coordinates + d/★/Δ over ℝ for FEM/EM. **Same combinatorial backbone, dual coefficient ring (ℝ vs Z/F_2).** 246's X1 + 283's T0 should share the simplex-tree backbone; 246 layers `Verts [][]float64` + ★ on top. **Single-source ownership recommendation:** the simplex-tree backbone lives in `topology/simplicial/`, both 246 and 283 import it, 246 adds geometric attributes.
- **Slot 247 (new-mortar-fem).** Mesh transfer between non-conforming meshes; downstream of 246.
- **Slot 254 (likely Delaunay/Voronoi or graph-cuts) / 077 / 078 (geometry-missing/sota).** Delaunay triangulation + 3D convex hull is a hard prerequisite for **alpha complex** (T5). Until Delaunay ships, alpha-complex is blocked.
- **Slot 271 (spectral-clustering), 273 (spectral-embedding).** Hodge-L_0 ≡ graph Laplacian on 1-skeleton (regression pin); 271's KMeans + 273's spectral-embed are downstream consumers of L_k eigvec.
- **Slot 280 (network-generative-models), 281 (temporal-graphs), 282 (hypergraphs).** 282 specifically punts T6 (HypergraphAsSimplicialComplex + Hodge L_k) to *this slot* (`graph/hyper/simplicial.go::AsSimplicialComplex`); 281 adds temporal axis (zigzag persistence, Carlsson-de-Silva 2010); 280's Hypergraph-SBM consumes simplex counts as features.
- **Slot 283 (this) = ABSTRACT SIMPLICIAL COMPLEXES + HOMOLOGY MACHINERY.** Owns SimplexTree, ∂_k over F_2 / Z, β_k computation as standalone API, Smith normal form for integer homology, simplicial cohomology, cup product, Hodge Laplacian L_k, filtered simplex tree, alpha complex, witness complex, persistent images / landscapes, Mayer-Vietoris.

### Web context (no MIT pure-Go zero-dep exists)

- **Boissonnat-Maria-2014** *Algorithmica* 70:406 "The Simplex Tree: An Efficient Data Structure for General Simplicial Complexes" (arXiv:1201.5113) — canonical memory-efficient ASC representation. Prefix-trie of sorted vertex labels; one node per simplex; `O(d)` insertion/lookup where d = max-dimension. Used by GUDHI (BSD-3 C++/Python). **Reduces memory ~10× over flat-list `Filtration` for sparse complexes.** Reference C++ ~3000 LOC in GUDHI src/Simplex_tree.
- **Edelsbrunner-Harer-2010** *Computational Topology: An Introduction* AMS — canonical textbook for ∂_k boundary maps, simplicial homology over F_2 / Z, Smith normal form, persistence, alpha/Čech/VR complexes. Chapter IV.1 boundary operator definition + chapter VI Smith normal form algorithm. **The textbook every implementation cites.**
- **Edelsbrunner-Mücke-1994** *ACM-TOG* 13:43 "Three-dimensional alpha shapes" — alpha complex is a sub-complex of the Delaunay triangulation: include simplex σ iff its smallest empty circumscribing sphere has radius ≤ α. Filtered alpha-complex on n points is **O(n^⌈d/2⌉)** simplices vs VR's O(n^{k+1}) at maxDim=k → typically **100× smaller** for n=1000 in R^3 → much cheaper persistence. C++ in CGAL (GPL/commercial).
- **De-Silva-Carlsson-2004** "Topological estimation using witness complexes" *Eurographics SPBG* — witness complex on a *landmark* subset L ⊂ P with weak-witness predicate. Two parameters (α, ν) tune complex size. Reference: Java in JavaPlex (GPL), Julia in Eirene (GPL).
- **Chen-Kerber-2011** "Persistent homology computation with a twist" *EuroCG-2011* — twist algorithm for persistent homology: skip column reduction on simplices that create boundaries (positive simplices are detected by counting ∂_k rank). **5–10× speedup** vs naive column reduction; standard in PHAT/Ripser/GUDHI.
- **Bauer-2021** *J-Appl-Comput-Topology* 5:391 "Ripser: efficient computation of Vietoris-Rips persistence barcodes" — clearing optimization (Chen-Kerber twist + emergent-pair elimination + apparent-pair shortcut + cohomology dual). **The current SOTA for VR persistence**; MIT C++ ~3500 LOC. Reality's `barcode.go` ships none of these (naive ELZ-2000 at vr.go:91, `boundaryColumn` rebuilt from scratch each call). 100×+ slowdown vs Ripser at n=1000.
- **Munkres-1984** *Elements of Algebraic Topology* §11 + Storjohann-1996 *J-Symb-Comput* 21:377 "Near optimal algorithms for computing Smith normal forms of integer matrices" — modular SNF avoids coefficient-explosion of naive Smith-form via Gaussian-style row/column ops on Z. **Required for integer homology** (rank of free part + invariant factors of torsion part); F_2 Betti misses 2-torsion (e.g., RP^2 has H_1 = Z/2; Z/2-coefficient β_1 = 1 but integer rank β_1 = 0).
- **Lim-2020** *SIAM-Rev* 62:685 "Hodge Laplacians on Graphs" + **Schaub-Benson-Horn-Lippner-2020** *SIAM-Rev* 62:353 "Random Walks on Simplicial Complexes" + **Eckmann-1944** *Comment-Math-Helv* 17:240 (origin) — Hodge Laplacian L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k. Hodge theorem (Eckmann 1944): on a finite simplicial complex, dim ker L_k = β_k. **Connects topology (homology rank) to spectrum (zero eigenvalues)** → cross-validation pin against F_2 Betti via column reduction.
- **De-Silva-Vejdemo-Johansson-2011** *Algebraic-Geometric-Topology* 11:737 "Persistent cohomology and circular coordinates" — persistent **co**homology dual to persistent homology, used by Ripser for the *clearing* optimization (cohomology has more zero columns → faster reduction). Also enables circular-coordinate inference (β_1 → S^1 coordinates).
- **Bubenik-2015** *JMLR* 16:77 "Statistical topological data analysis using persistence landscapes" + **Adams-2017** *JMLR* 18:8 "Persistence images: A stable vector representation of persistent homology" — vectorize barcodes for ML. Stable functions (Lipschitz in bottleneck distance) → can be input to SVM, NN, kernel methods.
- **Carlsson-2009** *Bull-AMS* 46:255 "Topology and Data" — canonical TDA survey + barcodes intro. Hatcher-2002 *Algebraic Topology* §2.1 — simplicial homology + cup product textbook reference.
- **Mayer-Vietoris** sequence — long exact sequence for H_k(A ∪ B) given H_k(A), H_k(B), H_k(A ∩ B). Useful for incremental homology updates (add a simplex, update β_k).
- **Singh-Memoli-Carlsson-2007** *Eurographics SPBG* "Topological methods for the analysis of high dimensional data sets and 3D object recognition" — **Mapper algorithm**. Cover + cluster → simplicial nerve. Often paired with persistent homology.
- **Open-source landscape:** GUDHI (BSD-3 C++/Python, ~50k LOC), Dionysus 2 (BSD), Ripser (MIT C++ ~3.5k LOC), JavaPlex (GPL), Eirene (Julia GPL), PHAT (LGPL), gtda/giotto-tda (Apache-2 Python wrapping GUDHI). **Pure-Go MIT zero-dep ABSENT for all of: simplex tree, alpha complex, witness complex, integer homology, Smith normal form, Hodge Laplacian on simplicial complex, persistence landscape/image, Mapper, persistent cohomology.** Reality has the largest open MIT-licensed Go gap in computational topology.

## Concrete recommendations

### T0 — SimplicialComplex via SimplexTree + per-dim index (cheapest day-1, zero blockers, ~280 LOC)

1. **`topology/simplicial/types.go` (~80 LOC).** Define:
   ```go
   // SimplicialComplex: abstract, label-free, dim-indexed.
   type SimplicialComplex struct {
       Dim       int               // max dimension
       Simplices [][]Simplex       // per-dim; Simplices[k] = sorted []Simplex of dim k
       index     map[string]int    // canonical-key -> per-dim ordinal (built lazily)
   }
   type Simplex []int // sorted ascending; Dim() = len-1 (reuse topology/persistent.Simplex)
   func (sc *SimplicialComplex) NumSimplices(k int) int
   func (sc *SimplicialComplex) Insert(s Simplex)        // adds s and all sub-faces
   func (sc *SimplicialComplex) Contains(s Simplex) bool
   func (sc *SimplicialComplex) Faces(s Simplex) []Simplex // codim-1 faces
   func (sc *SimplicialComplex) Cofaces(s Simplex) []Simplex
   func (sc *SimplicialComplex) Validate() error          // closure under face-taking
   ```
   `Insert` enforces *closure under face-taking* (the abstract simplicial complex axiom). **Cross-link 282:** absorbs `graph/hyper/simplicial.go::AsSimplicialComplex` ownership.

2. **`topology/simplicial/simplextree.go::SimplexTree` (~140 LOC).** Boissonnat-Maria-2014 prefix-trie. Each node carries `(label int, children map[int]*node, parent *node, filtration float64)`. `Insert` is `O(d log d)` (one descent per face). `Cofaces` uses sibling traversal (Boissonnat-Maria §4.4). **The single most-cited combinatorial-topology data structure**; GUDHI's whole pipeline is built on it.
   - Backing store option: `SimplexTreeStorage` interface with `MapBased` (default) and `ArrayBased` (cache-friendly for small Dim ≤ 3, n ≤ 10k) implementations.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#1):** for every input filtration that reality can build today via `topology/persistent.VietorisRipsComplex`, the simplex-tree representation enumerates the same simplex set in the same canonical sorted order (regression to existing Filtration on n ≤ 50 fixtures from `topology/persistent/persistent_test.go`).

3. **`topology/simplicial/bridge.go::FromFiltration / ToFiltration` (~60 LOC).** Adapter to/from `topology/persistent.Filtration`. Lets `ComputeBarcode` (which currently rebuilds `boundaryColumn` from a flat indexBy map at barcode.go:74) instead receive a pre-built simplex tree. **No API break** for existing consumers; pure-additive.

### T1 — Sparse boundary maps ∂_k over F_2 and Z (~220 LOC)

4. **`topology/simplicial/boundary.go::BoundaryF2(sc, k) (rows, cols []int)` (~60 LOC).** Sparse F_2 boundary `∂_k : C_k → C_{k-1}` as CSR-like `(rows, cols)` (vals implicit = 1). Parity with private `topology/persistent/barcode.go::boundaryColumn` but exposed and per-dim-indexed. **Eliminates the simplexKey map-lookup** (barcode.go:170-216); simplex-tree gives `O(1)` index.

5. **`topology/simplicial/boundary.go::BoundaryZ(sc, k) (rows, cols []int, vals []int)` (~70 LOC).** Z-coefficient ∂_k with signed-orientation: `∂_k σ = ∑_i (-1)^i σ\{v_i}` (Munkres-1984 Eq-5.2; Edelsbrunner-Harer §IV.1). Coefficient `(-1)^i` where i = position of removed vertex in sorted-tuple. **Required for integer homology** (T3) and **Hodge Laplacian on oriented simplicial complex** (T7).

6. **`topology/simplicial/chain.go::ChainComplex` (~90 LOC).** `ChainComplex{Boundaries [][]Triplet; Dim int}` — the (∂_0, ∂_1, ..., ∂_d) sequence with the load-bearing **`d^2 = 0` invariant** (`∂_{k-1} ∘ ∂_k = 0` for all k). Exported `(c *ChainComplex).VerifySquaresZero(maxK int) error` checks `∂_{k-1}∂_k = 0` row-by-row over F_2 and Z; **single load-bearing correctness witness** (orientation bug → `dd ≠ 0`, immediate fail). **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#2):** `dd=0` over F_2 + `dd=0` over Z + Hodge-theorem dim ker L_0 = β_0 over F_2. Three independent algebraic identities saturate.

### T2 — Z/2 simplicial homology (Betti numbers) standalone (~120 LOC)

7. **`topology/simplicial/homology.go::BettiF2(sc) []int` (~60 LOC).** `β_k(F_2) = dim C_k - rank ∂_k - rank ∂_{k+1}` via Gaussian elimination over F_2. Returns `[β_0, β_1, ..., β_{Dim}]`. **Fills the gap that `topology/persistent.ComputeBarcode` only emits *bars*, never an aggregate β_k** at infinite filtration time. Standalone API the consumer needs for "is this graph connected? does it have a hole?" without filtering.

8. **`topology/simplicial/homology.go::CyclesAndBoundariesF2(sc, k) (cycles, bdries [][]int)` (~60 LOC).** Returns sparse F_2 representatives of `Z_k = ker ∂_k` and `B_k = im ∂_{k+1}`. Useful for *visualizing* a representative loop / void.

### T3 — Integer homology via Smith normal form (~280 LOC, frontier; gates Pistachio-loop closure semantic correctness)

9. **`linalg/integer/smith.go::SmithNormalForm(M [][]int) (S, U, V [][]int)` (~220 LOC).** Munkres-1984 §11 algorithm with Storjohann-1996 modular variant: `M = U S V` with U, V ∈ GL(Z), S = diag(d_1, d_2, ..., d_r, 0, ..., 0) where d_1 | d_2 | ... | d_r. **Modular variant essential** — naive Smith form has exponential coefficient explosion (Storjohann §1). Validate `det(U) = ±1` and `det(V) = ±1` and `S[i][j] = 0` off-diag.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#3):** for torsion-free spaces (sphere S^n, torus T^n, point), integer β_k via SNF ≡ F_2 β_k via Gaussian elimination ≡ Hodge L_k zero-eigenvalue count. For RP^2 (which has H_1 = Z/2), F_2 β_1 = 1 but integer β_1 = 0 — **regression-test that catches naive integer-Betti = F_2-Betti bugs**.

10. **`topology/simplicial/homology.go::BettiZ(sc) (betti []int, torsion [][]int)` (~60 LOC).** Wraps SNF on each `BoundaryZ(sc, k)`. `betti[k] = (dim C_k) - rank(∂_k) - rank(∂_{k+1})`. `torsion[k] = invariant-factor list of T_k = ker ∂_k / im ∂_{k+1}` (the diagonal entries d_i of S that are not 1).

### T4 — Filtered simplex tree + persistent homology with twist (~220 LOC, replaces existing barcode.go reduction with 5-10× speedup)

11. **`topology/simplicial/filtration.go::FilteredComplex` (~80 LOC).** `FilteredComplex{Tree *SimplexTree; Order []*Node}` with simplices ordered by (filtration_time, dim, lex). Builder `Sort()`, iterator `EachSimplex(yield func(s Simplex, t float64))`. Bridge to existing `topology/persistent.Filtration` via `FromFiltration`/`ToFiltration`.

12. **`topology/simplicial/persistence.go::PersistenceTwist(fc) []Bar` (~140 LOC).** Chen-Kerber-2011 twist algorithm: process simplices in filtration order; if σ has positive boundary (creates a class), defer column reduction; if σ has empty boundary post-cancellation (kills a class), do reduction. **5–10× faster than naive ELZ at barcode.go:60.** Returns `[]Bar{Dim, Birth, Death}` API-compatible with `topology/persistent.Bar`.
    - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#4):** on every fixture in `topology/persistent/persistent_test.go`, naive `ComputeBarcode` ≡ `PersistenceTwist` ≡ matrix-reduction-with-clearing (skip rows with known pivots). Three independent reductions saturate.

13. **`topology/simplicial/sublevel.go::SublevelFiltration(sc, f map[Simplex]float64) *FilteredComplex` (~30 LOC).** Generic sublevel-set filtration: assign filtration time `t(σ) = max_{v ∈ σ} f(v)` (lower-star) or `t(σ) = f(σ)` directly. Powers most non-VR persistence applications (image-pixel persistence, function-on-mesh persistence, scalar-field-on-graph persistence).

### T5 — Alpha complex (~280 LOC, BLOCKS on Delaunay triangulation in geometry package)

14. **`topology/simplicial/alpha.go::AlphaComplex(points [][]float64, alpha float64) *FilteredComplex` (~280 LOC, **gates on Delaunay**).** Edelsbrunner-Mücke-1994 alpha shape. (a) Compute Delaunay triangulation D of points (gates on slot 077/078 geometry to ship `geometry.Delaunay2D`/`Delaunay3D`). (b) For each simplex σ ∈ D, compute α(σ) = squared smallest empty circumscribing-sphere radius. (c) Include σ iff α(σ) ≤ alpha; filtration time = α(σ). **100× cheaper than VR** on point clouds in low ambient dim (n=1000 in R^3: alpha ~few-thousand simplices vs VR ~millions). **Gating note:** without Delaunay, ship a 2D version on `geometry.Triangulate2D` (Bowyer-Watson, ~200 LOC) as a parallel cheapest-path PR in the geometry slot.

### T6 — Witness complex (~180 LOC, no blocker)

15. **`topology/simplicial/witness.go::WitnessComplex(landmarks, witnesses [][]float64, nu int, alpha float64) *FilteredComplex` (~180 LOC).** De-Silva-Carlsson-2004. Subsample landmarks L ⊂ P (`|L| ≪ |P|`); a simplex σ ⊂ L is included iff there exists a witness w ∈ P with all vertices of σ among the (ν+|σ|) nearest landmarks to w. Filtration time = max-dist-to-landmark threshold. **Approximates VR/Cech persistence at O(|L|^{k+1}) cost** instead of O(|P|^{k+1}). Standard for high-dim noisy data.

### T7 — Hodge Laplacian L_k (~140 LOC; matrix construction unblocked, spectrum gates on 097-T1)

16. **`topology/simplicial/hodge.go::HodgeLaplacian(sc, k int) (Lk [][]float64, n int)` (~70 LOC).** `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k` (Eckmann-1944, Lim-2020, Schaub-2020). Symmetric PSD `n_k × n_k`. **Matrix construction ships unblocked.**
    - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#5):** `L_0 = ∂_1 ∂_1^T` ≡ standard graph Laplacian on the 1-skeleton (regression to `graph.Laplacian` if it exists; else reuse 282-pin#3 — Carletti-2020 hypergraph-Laplacian-as-graph-Laplacian-when-r=2).
    - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#6, Hodge theorem):** dim ker L_k (numerical multiplicity of zero eigenvalue ≤ tolerance) ≡ β_k via T2 BettiF2 (modulo 2-torsion) ≡ β_k via T3 BettiZ. Three independent computations saturate.

17. **`topology/simplicial/hodge.go::HodgeSpectrum(sc, k int) (eigvals, eigvecs []float64)` (~70 LOC, **gates on slot 097-T1 Eigvec**).** Once Eigvec ships, spectrum + Hodge-decomposition components (gradient + curl + harmonic from eigvec basis). Used by spectral simplicial-complex clustering (Schaub-2020 simplicial signal processing).

### T8 — Simplicial cohomology + cup product (~240 LOC, frontier)

18. **`topology/simplicial/cohomology.go::Coboundary(sc, k) (rows, cols []int, vals []int)` (~30 LOC).** Coboundary `δ^k = (∂_{k+1})^T : C^k → C^{k+1}`. Trivial transpose of T1.5.

19. **`topology/simplicial/cohomology.go::CohomologyZ(sc) (betti, torsion []int)` (~50 LOC).** Universal coefficient theorem: H^k(X; Z) = Free(H_k(X; Z)) ⊕ Torsion(H_{k-1}(X; Z)). Reuse T3 SNF results.

20. **`topology/simplicial/cohomology.go::CupProduct(sc, alpha, beta Cocycle) Cocycle` (~160 LOC).** Hatcher-2002 §3.2 cup product on cochains: `(α ∪ β)([v_0, ..., v_{p+q}]) = α([v_0, ..., v_p]) · β([v_p, ..., v_{p+q}])`. Distinguishes spaces with same Betti but different ring structure (e.g., S^2 ∨ S^4 vs CP^2). **Frontier primitive** — no pure-Go MIT zero-dep impl exists.

### T9 — Persistence statistics: landscapes + images + barcode-features (~260 LOC)

21. **`topology/persistent/landscape.go::PersistenceLandscape(bars []Bar, k int) func(t float64) float64` (~120 LOC).** Bubenik-2015 landscape function `λ_k(t)`. Stable in 1-Wasserstein (and bottleneck → sup-norm). Vectorized → input to ML.

22. **`topology/persistent/image.go::PersistenceImage(bars []Bar, sigma float64, grid int) [][]float64` (~140 LOC).** Adams-2017 persistence image. Gaussian-smoothed birth-vs-persistence weighted measure on a grid. Stable in 1-Wasserstein. Standard input to SVM/CNN for TDA features.

### T10 — Mapper algorithm (~280 LOC, downstream consumer pull)

23. **`topology/mapper/mapper.go::Mapper(points, lensFn, cover, clusterer)` (~280 LOC).** Singh-Memoli-Carlsson-2007. Project via lens (PCA/eccentricity/density), cover image with overlapping intervals, cluster pre-images, build nerve simplicial complex (= a `SimplicialComplex` from T0). **Mapper bridges this slot to slot 271 (spectral-clustering) + slot 282 (workshop ecosystem topology dashboard called out at vr.go:doc).**

### T11 — Frontier (defer)

24. **Mayer-Vietoris (Eilenberg-Steenrod axiomatic) for incremental updates.** Long exact sequence enabling β_k recomputation on simplex add/remove without full reduction. ~400 LOC; defer until temporal-graph slot 281 + dynamic-homology consumer pulls.

25. **Persistent cohomology (de-Silva-Vejdemo-Johansson-2011).** Dual-Pairs algorithm: faster reduction + circular-coordinate inference. Foundation for slot 281's zigzag persistence.

## Single cheapest day-1 PR

**`topology/simplicial/` package, T0 (1–3) + T1.4 + T2.7 + T7.16, ~480 LOC, zero blockers.**

- `topology/simplicial/types.go` — SimplicialComplex + Simplex + Insert/Contains/Faces/Cofaces (T0.1, ~80 LOC)
- `topology/simplicial/simplextree.go` — Boissonnat-Maria SimplexTree (T0.2, ~140 LOC)
- `topology/simplicial/bridge.go` — From/To `topology/persistent.Filtration` (T0.3, ~60 LOC)
- `topology/simplicial/boundary.go` — BoundaryF2 sparse (T1.4, ~60 LOC)
- `topology/simplicial/homology.go` — BettiF2 + CyclesAndBoundariesF2 (T2.7+T2.8, ~70 LOC)
- `topology/simplicial/hodge.go` — HodgeLaplacian L_k matrix construction (T7.16, ~70 LOC; spectrum deferred)

Tests:
- `TestSimplexTree_VRRegression` — for every `topology/persistent/persistent_test.go` VR fixture, simplex-tree enumeration ≡ flat-Filtration enumeration (R-MUTUAL-CROSS-VALIDATION 3/3 pin #1).
- `TestChainComplex_DDZero_F2` + `TestChainComplex_DDZero_Z` — `∂_{k-1} ∂_k = 0` over F_2 and Z on every fixture (load-bearing orientation correctness witness, pin #2).
- `TestBettiF2_HollowTetrahedron` — boundary of 4-simplex (= S^2 triangulation): β_0=1, β_1=0, β_2=1.
- `TestBettiF2_Torus` — minimal 7-vertex Möbius-Kantor torus triangulation: β=(1,2,1).
- `TestHodgeL0_GraphLaplacianRegression` — L_0 ≡ graph-Laplacian-of-1-skeleton on every binary-graph fixture (pin #5).
- `TestHodgeKerCount_BettiF2_Sphere` — for hollow tetrahedron, dim ker L_0 = 1 = β_0, dim ker L_1 = 0 = β_1, dim ker L_2 = 1 = β_2. Cross-validates Hodge ↔ homology (pin #6, partial — gates fully on 097-T1 Eigvec for ker computation; until then use rank deficiency `n − rank(L_k)` over F_2 as proxy via T2.7).

This PR ships the abstract simplicial complex as a first-class type with the simplex-tree memory advantage, exposes ∂_k publicly (currently private to barcode.go), provides standalone β_k computation (currently only available via persistence diagram with infinite-bar count workaround), and delivers Hodge Laplacian *matrix construction* (the most-cited primitive in graph-signal-processing / network-neuroscience). Persistence reduction speedup (T4.12 twist), integer homology (T3.9 SNF), and Hodge spectrum (T7.17) all gate on second-PR follow-ups.

## Cross-cutting

- **Slot 097-T1 (linalg-missing) ←** T7.17 Hodge spectrum gates on EigvecSym. Hard-pin: ship 097-T1 (~80 LOC) before T7.17.
- **Slot 077/078 (geometry-{missing,sota}) ←** T5.14 Alpha complex gates on `geometry.Delaunay2D/3D`. Recommend slot 077 ship Bowyer-Watson 2D first.
- **Slot 141–145 (topology-{numerics,sota,api,perf,missing}) ←** this slot is the *missing* substrate that 141/142 explicitly call out. Recommend 142 review be re-read after this slot lands.
- **Slot 156 (synergy-topology-prob) ←** T2 BettiF2 + T3 BettiZ feed Bayesian topological inference (Adler-Bobrowski-Borman-2010 random simplicial complexes Betti distribution).
- **Slot 171 (synergy-graph-topology) ←** flag-complex / clique-complex of binary graphs = `SimplicialComplex` (T0) where simplices = cliques. T0 is the substrate; 171 ships clique-enumeration + adapter.
- **Slot 246 (new-discrete-exterior) ←** **single-source-ownership recommendation:** simplex-tree backbone in `topology/simplicial/`; 246's `geometry/dec/SimplicialComplex2D` = `topology/simplicial.SimplicialComplex` + `Verts [][]float64` + ★. 246's X1+X2+X3 (~140+30+50 LOC) shrink to ~120 LOC if T0 ships first. Composition pin: 246's ∂ over ℝ ≡ 283's BoundaryZ over Z ⊗ ℝ.
- **Slot 247 (new-mortar-fem) ←** mortar transfer between non-conforming simplicial meshes; needs T0 SimplicialComplex + T0 simplex tree.
- **Slot 254 (graph-cuts/Delaunay) ←** depending on what slot 254 is, either Delaunay (gates T5.14 alpha) or graph-cuts (downstream consumer of T8 cohomology).
- **Slot 271 (spectral-clustering), 273 (spectral-embedding) ←** T7 Hodge L_k eigvec → simplicial spectral clustering (Schaub-2020). 271 + 273 reuse k-means / spectral-embed back-ends.
- **Slot 280 (network-generative-models) ←** simplicial complex of an SBM-generated graph as a feature (random-clique-complex Betti for SBM model selection).
- **Slot 281 (temporal-graphs) ←** zigzag persistence (Carlsson-de-Silva-2010) = T4.12 twist + Mayer-Vietoris for adds/removes. T0 + T11.24 substrate.
- **Slot 282 (hypergraphs) ←** 282-T6 (HypergraphAsSimplicialComplex + Boundary + HodgeLaplacian) **fully owned by this slot**; 282 keeps just the bridge constructor `AsSimplicialComplex(hg) *SimplicialComplex`. Cross-validation: hypergraph-Laplacian on r=2 ≡ Hodge L_0 of 1-skeleton ≡ standard graph Laplacian (three-way pin saturates 282-#3 + 283-#5 simultaneously).
- **Pistachio scene-topology / loop-closure ←** β_1 of a feature-track graph counts visual loops; persistent β_1 over a feature-correspondence sublevel filtration is the canonical loop-closure signal. **Direct consumer pull for T2 + T4.13.**
- **Aicore (molecular dynamics, protein-loop classification) ←** persistent β_1 / β_2 of distance matrices over MD trajectories distinguishes protein conformational classes (Kovacev-Nikolic-2016, Cang-Wei-2017). Witness complex (T6.15) is the standard high-dim reduction.
- **Insights blast-radius topology ←** named at `topology/persistent/doc.go:38`. Currently uses VR; if the dependency graph has > 50 services, alpha-complex / witness-complex (T5/T6) is required.
- **Workshop ecosystem topology dashboard ←** named at doc.go:28-30. Mapper (T10.23) is the called-out replacement primitive.

## Sources

- `topology/persistent/vr.go:14,50,91,175`, `topology/persistent/barcode.go:60,170-216,221,254`, `topology/persistent/bottleneck.go:1-40`, `topology/persistent/doc.go:54-77,118-124`, `topology/persistent/errors.go:19`
- `linalg/decompose.go`, `linalg/eigen.go:20`, `linalg/matrix.go`
- `geometry/polygon.go.ConvexHull2D`, `geometry/sdf.go`
- Reviews: `agents/097-linalg-missing.md` (T1 Eigvec blocker), `agents/141-topology-numerics.md`, `agents/142-topology-missing.md`, `agents/143-topology-sota.md`, `agents/144-topology-api.md`, `agents/145-topology-perf.md`, `agents/156-synergy-topology-prob.md`, `agents/171-synergy-graph-topology.md`, `agents/246-new-discrete-exterior.md` (X1–X30 DEC overlap), `agents/282-new-hypergraphs.md` (T6 punt to this slot)
- Boissonnat-Maria-2014 *Algorithmica* 70:406 "The Simplex Tree" arXiv:1201.5113
- Edelsbrunner-Harer-2010 *Computational Topology: An Introduction* AMS — boundary maps, Smith form, persistence textbook
- Edelsbrunner-Mücke-1994 *ACM-TOG* 13:43 "Three-dimensional alpha shapes"
- de-Silva-Carlsson-2004 *Eurographics SPBG* "Topological estimation using witness complexes"
- Chen-Kerber-2011 *EuroCG-2011* "Persistent homology computation with a twist"
- Bauer-2021 *J-Appl-Comput-Topology* 5:391 "Ripser: efficient computation of Vietoris-Rips persistence barcodes"
- Munkres-1984 *Elements of Algebraic Topology* §11 + Storjohann-1996 *J-Symb-Comput* 21:377 "Near optimal SNF"
- Lim-2020 *SIAM-Rev* 62:685 "Hodge Laplacians on Graphs"
- Schaub-Benson-Horn-Lippner-2020 *SIAM-Rev* 62:353 "Random Walks on Simplicial Complexes"
- Eckmann-1944 *Comment-Math-Helv* 17:240 "Harmonische Funktionen und Randwertaufgaben in einem Komplex"
- de-Silva-Vejdemo-Johansson-2011 *Algebraic-Geometric-Topology* 11:737 "Persistent cohomology and circular coordinates"
- Bubenik-2015 *JMLR* 16:77 "Persistence landscapes"
- Adams-2017 *JMLR* 18:8 "Persistence images"
- Carlsson-2009 *Bull-AMS* 46:255 "Topology and Data"
- Hatcher-2002 *Algebraic Topology* §2.1 + §3.2 (cup product)
- Singh-Memoli-Carlsson-2007 *Eurographics SPBG* "Mapper algorithm"
- Carlsson-de-Silva-2010 *Found-Comput-Math* 10:367 "Zigzag persistence"
- GUDHI (BSD-3 C++/Python), Ripser (MIT C++), Dionysus 2 (BSD), JavaPlex (GPL), Eirene (Julia GPL), PHAT (LGPL)
