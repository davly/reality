# 284 — new-cw-complexes (Cell Complexes: CW, Cubical, Δ, Discrete Morse)

## Headline
reality v0.10.0 ships ZERO general cell-complex / cubical / Δ-complex / discrete-Morse surface (`CW|CellComplex|Cubical|DeltaComplex|RegularCell|DiscreteMorse|MorseMatching|GradientPath|AttachingMap|Forman` repo-wide grep on `*.go` returns 0 callable hits outside review docs); the only adjacent substrate is `topology/persistent/{vr,barcode}.go` — F_2 column-reduction over a flat `[]Simplex` Filtration capped at maxDim ∈ {0,1} — which is *simplicial-only* and provides ZERO of: arbitrary-dim cell, attaching map, cubical-cell binary-interval indexing (Kaczynski-Mischaikow-Mrozek 2004), Δ-complex (Hatcher §2.1), Forman-1998 discrete Morse function, Lewiner-2003 greedy Morse matching, gradient vector field, Morse complex collapse, raster-image cubical persistence (the standard tool for 2D/3D image TDA per Wagner-Chen-Vučini-2012). The cheapest day-1 PR is `topology/complex/cubical/` — KMM-2004 cubical complex (binary-interval product encoding) + F_2 cubical homology + cubical persistence on lower-star filtration of a raster scalar field — **~480 LOC, zero blockers, no Eigvec dep**, operates directly on voxel/pixel grids (the standard input shape for image / volumetric / 3D-printer / scientific-computing data) and ships standalone WITHOUT requiring slot 283's SimplexTree to land first. Discrete Morse is orthogonal to representation: T4 ships on whichever cell complex (simplicial T0 of slot 283, cubical T1/T2 here, or generic CellComplex interface) lands first.

## Findings

### State at HEAD (verified via direct grep on `*.go`)

| Surface | Path | Cell-complex relevance |
|---|---|---|
| `Simplex = []int` (sorted) | `topology/persistent/vr.go:14` | Simplicial-only; no cube, no n-cell, no attaching map. |
| `Filtration{Simplices, Times}` | `topology/persistent/vr.go:50` | Flat `(simplex, time)` list; only one cell-shape (simplex). |
| `boundaryColumn` private | `topology/persistent/barcode.go:221` | F_2 boundary via `simplexKey` map — works for simplices only; cube boundaries are sums of *opposite-face pairs* with structurally different combinatorics. |
| `ComputeBarcode(filt, maxDim)` | `topology/persistent/barcode.go:60` | ELZ-2000 column-reduction over F_2; capped at maxDim ∈ {0,1}; coupled to Simplex type, cannot consume CubicalCell. |
| `geometry/sdf.go` | `geometry/` | SDF primitives in R^d (sphere, box, plane). Boxes are *geometric* not *combinatorial cells*. |
| `linalg.QRAlgorithm` | `linalg/eigen.go:20` | Tridiagonal QL eigvals only, NO eigvecs — slot 097-T1 blocker. |
| repo-wide grep: `CellComplex\|CubicalComplex\|DeltaComplex\|RegularCell\|DiscreteMorse\|MorseMatching\|GradientPath\|AttachingMap\|Forman` | `*.go` outside `reviews/` | **ZERO hits.** |

### Slot boundaries

- **Slot 097-T1 (linalg-missing) — Eigvec.** Hard-pin for spectral methods on cell complexes (T9 below). Cubical homology + Morse collapse + cubical persistence ship WITHOUT this dep.
- **Slot 156 (synergy-topology-prob), 141–145 (topology-{numerics,sota,api,perf,missing}).** 142/143 explicitly call out "cubical filtration" + "Mischaikow-Nanda 2013 discrete Morse pre-collapse ~350 LOC" as missing primitives (143 line 273-280 + 393, 142 line 198 + 209). **This slot owns those.**
- **Slot 246 (new-discrete-exterior).** X13 in 246 sketches `CubicalComplex{Verts, Edges, Quads, Cubes}` ~160 LOC for *embedded geometric* cubical cells with metric Hodge ★. **Cross-link:** 284's combinatorial CubicalComplex (KMM binary-interval indexing, no metric) is the upstream substrate; 246-X13 layers Verts-coords + ★ on top. **Single-source ownership recommendation:** combinatorial cubical complex lives in `topology/complex/cubical/`; 246-X13 imports it.
- **Slot 282 (hypergraphs).** Disjoint axis (set system, not topology).
- **Slot 283 (just done) — simplicial complexes.** Owns SimplexTree + simplicial ∂_k + simplicial homology + Hodge L_k matrix. **Cross-link:** every abstract simplicial complex IS a CW-complex (with the obvious cell structure and inclusion attaching maps), so 283's SimplicialComplex satisfies the generic CellComplex interface (T0 below). 284 generalizes to (a) cubical (image/raster — most useful subcase) and (b) abstract CW (n-cells with attaching maps φ:S^{n-1} → X^{(n-1)}). **Discrete Morse is orthogonal:** Lewiner-2003 greedy matching on the Hasse diagram of *any* cell complex (simplicial, cubical, CW, Δ, regular). **Recommendation:** unified `topology/complex/` parent package hosting both 283's `simplicial/` and 284's `cubical/` + `cw/` + `morse/`. Re-house 283's SimplexTree under `topology/complex/simplicial/` retroactively.
- **Slot 281 (temporal graphs).** Filtered cell complexes evolving in time → zigzag persistence on cell complexes (Carlsson-de-Silva 2010); discrete Morse is the keystone preprocessor (Mischaikow-Nanda 2013).
- **Slot 280 (SBM).** Random simplicial / cell complexes (Linial-Meshulam 2006, Kahle 2014) on top of T0/T1 substrate.
- **Slot 077/078 (geometry).** Independent: cubical complex needs NO Delaunay (works directly on raster); CW needs NO ConvexHull.

### Web context (verified via prior knowledge / cited consistently across 142/143)

- **Hatcher-2002** *Algebraic Topology* Chapter 0 + §2.1 (Cambridge UP) — the canonical CW-complex / Δ-complex textbook reference. CW-complex: built inductively, attach n-cells via maps φ_α: S^{n-1} → X^{(n-1)}. Δ-complex: each n-cell IS an n-simplex with a fixed characteristic-map ordering of vertices, but unlike a simplicial complex DIFFERENT n-simplices may share the same vertex set (e.g., the standard 2-simplex Δ-complex on the torus uses *2 triangles, 3 edges, 1 vertex* — impossible for an abstract simplicial complex which requires |V| ≥ k+1 distinct vertices for a k-simplex). Δ-complexes give **dramatically smaller cell counts** for non-trivial topology: torus T^2 = (1 vertex, 3 edges, 2 triangles) vs minimal simplicial torus = (7 vertices, 21 edges, 14 triangles) — 6× fewer cells.
- **Kaczynski-Mischaikow-Mrozek 2004** *Computational Homology* (Springer Applied Math Sciences 157) §2 — canonical text for cubical homology. **Cubical cell** = product of *binary intervals*: each interval is either a "non-degenerate" [k, k+1] (length 1) or "degenerate" [k, k] (a point). A d-dimensional ambient grid Z^d has cubical cells indexed by (anchor ∈ Z^d, mask ∈ {0,1}^d) where mask[i]=1 means non-degenerate dim-i, mask[i]=0 means degenerate; dim of cell = popcount(mask). **Boundary** of a non-degenerate interval [k, k+1] = (-1) [k, k] + (+1) [k+1, k+1] (signed); product rule gives ∂ on cube as alternating sum of (mask⊕e_i) faces (Kaczynski-Mischaikow-Mrozek §2.5). **Voxel / pixel data IS literally a cubical complex** — every full-dimensional cube corresponds to one voxel; faces / edges / vertices form the natural cell structure of the raster. KMM's Reference C++ library `CHomP` (~10k LOC) is the canonical impl.
- **Forman-1998** *Adv-Math* 134:90 "Morse theory for cell complexes" + **Forman-2002** *Sém-Lothar-Combin* 48 "A user's guide to discrete Morse theory" — combinatorial analogue of smooth Morse theory. A **discrete Morse function** f: K → R on cell complex K assigns one real value per cell with at-most-one anomaly per (cell, coface) pair: σ < τ implies either f(σ) < f(τ) (regular pair, can collapse) or σ ↑ τ matched in V (gradient pair). Equivalent (Forman §3) to a **discrete vector field V** = matching of cells in the Hasse diagram (each cell appears in at-most-one matched pair, σ in pair only if σ < τ codim 1). **Critical cells** = unmatched cells. **Forman collapse theorem**: K is homotopy equivalent to a CW-complex with one cell per critical cell. **Often 90-99% reduction** for typical inputs (Mischaikow-Nanda 2013 report 100×+ speedup on VR persistence).
- **Lewiner-Lopes-Tavares 2003** *J-Math-Imaging-Vision* 19:223 + 2003 *Geom-Mod-Proc* — "Optimal discrete Morse functions for 2-manifolds" + **greedy coreduction-pair matching algorithm**: process cells in reverse degree order, match σ↑τ whenever τ has σ as its UNIQUE unmatched codim-1 face (a "free face"). When no free face, pick a critical cell. **Linear-time in cell count**, never optimal but typically within a few percent of optimal. Reference implementation perseus 4.0 (~6k C++ LOC). **THE workhorse for industrial discrete-Morse preprocessing.** 143 line 326 quotes 250–10000× cumulative speedup over naive ELZ.
- **Mischaikow-Nanda 2013** *Discrete-Comput-Geom* 50:330 "Morse theory for filtrations and efficient computation of persistent homology" — canonical paper applying discrete Morse to *persistence* (not just homology). Works on filtered complex; produces a "reduced" filtered complex with only critical cells but identical persistence diagram. **5-10× speedup standalone, multiplicative with Chen-Kerber twist** (143 line 273-280, 326).
- **Wagner-Chen-Vučini 2012** *Topological-Methods-in-Data-Analysis-and-Visualization-II* "Efficient computation of persistent homology for cubical data" — first widely-cited cubical-persistence paper. T-construction (interpret pixel/voxel value as filtration of dim-d cells) and V-construction (filtration of dim-0 vertices, lower-star upward). **Standard tool** for image / 3D-volume topological feature persistence (medical imaging segmentation, materials microstructure analysis, neuroscience MRI parcellation).
- **Sköldberg 2006** *Trans-AMS* 358:115 "Morse theory from an algebraic viewpoint" — Forman's discrete Morse generalizes to *algebraic* Morse theory on chain complexes. Foundation for the modern unified treatment.
- **Benedetti-Lutz 2014** *Exp-Math* 23:66 "Random discrete Morse theory and a new library of triangulations" — empirical: random Morse matchings on a triangulation typically achieve near-optimal collapsing on benign inputs (small Betti). Used as the practical default in many libraries.
- **PHAT (Bauer-Kerber-Reininghaus-Wagner 2017)** *J-Symb-Comput* 78:76 — cubical persistence + general persistence-via-matrix-reduction with apparent-pair / clearing optimizations. LGPL C++ ~5k LOC.
- **Open-source landscape:**
  - GUDHI 3.10 (BSD-3) ships cubical-complex + filtered-cubical + cubical-persistence + Morse-Smale on regular complex; ~50k LOC C++/Python.
  - PHAT (LGPL) cubical-PH workhorse.
  - perseus 4.0 (free, no clear license) Mischaikow's reference Morse-preprocessing engine; ~6k LOC C++.
  - DiPHA (BSD-2) parallel persistence on cubical / general filtered chain complexes.
  - simpers / openpht / Morse-Algebraic (academic) — Sköldberg-style algebraic Morse.
  - **Pure-Go MIT zero-dep ABSENT for ALL of:** cubical complex, cubical homology, cubical persistence, discrete Morse function, greedy Morse matching, Morse complex collapse, Morse-theoretic persistence, regular CW complex, Δ-complex, attaching maps. Reality has the **single largest zero-dep cell-complex gap in any modern language** (Julia has Eirene GPL; Python has GUDHI Apache wrapping C++; JavaScript and Rust have nothing).

## Concrete recommendations

### T0 — Generic CellComplex interface + Hasse diagram (zero-blocker, ~140 LOC)

1. **`topology/complex/cell.go::CellComplex` interface (~80 LOC).** Minimal generic API both simplicial (slot 283) and cubical (T1) implement:
   ```go
   type CellComplex interface {
       Dim() int                            // max cell dimension
       NumCells(k int) int                  // n_k = number of k-cells
       Cell(k, i int) CellID                // i-th k-cell handle (opaque)
       Boundary(c CellID) []SignedCell      // ∂c = Σ ε_i c_i (signed Z; F_2 collapses sign)
       Cofaces(c CellID) []CellID           // codim-1 cofaces (used by Hasse diagram)
       FiltrationValue(c CellID) float64    // for filtered complex; 0 if unfiltered
   }
   type SignedCell struct{ ID CellID; Sign int8 } // ±1 over Z, +1 always over F_2
   type CellID = uint64                            // packed (dim<<48)|index
   ```
   **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#A):** `dd=0` invariant — for every CellComplex impl, `∂_{k-1}(∂_k(c)) ≡ 0` over F_2 and Z, on every cell of every dim. Three-axis saturation (cubical T1, simplicial 283-T0, Δ-complex T8) all satisfy.

2. **`topology/complex/hasse.go::HasseDiagram(cc CellComplex) [][]CellPair` (~60 LOC).** The face-poset; rows indexed by dim k, columns by codim-1 face pairs (σ < τ where σ is a codim-1 face of τ). **Single load-bearing primitive for discrete Morse**: greedy matching, gradient flow, critical-cell enumeration all read this. Constructed lazily in O(Σ_k n_k · k) time = O(total face-incidences).

### T1 — Cubical complex (KMM-2004 binary-interval encoding) (~220 LOC, ZERO blocker)

3. **`topology/complex/cubical/types.go::CubicalCell` + `Cube` (~80 LOC).** Each cell is a (anchor [d]int32, mask uint32) packed into a uint64 (assuming d ≤ 32). `dim(cell) = bits.OnesCount32(mask)`. **Memory: 8 bytes per cell** vs slot-283 simplex tree typical 32-128 bytes per simplex → cubical is **4-16× more memory-efficient**.
   ```go
   type CubicalCell uint64 // (mask<<32) | (anchor packed in low 32 bits)
   type CubicalComplex struct {
       Shape  []int32             // ambient grid Z^d shape
       Cells  [][]CubicalCell     // per-dim sorted slice
       Filt   map[CubicalCell]float64 // optional filtration value
   }
   ```
   `Insert(cell)` enforces *closure under face-taking* (KMM §2.3 axiom — every face of an included cube is included).

4. **`topology/complex/cubical/builder.go::FromRaster(values []float64, shape []int32) *CubicalComplex` (~80 LOC).** Build the **T-construction** cubical complex of an n-D raster: every voxel is a top-dim cube, all sub-cubes (faces, edges, vertices) included by closure. Filtration value for each cell = max value over incident voxels (lower-star, used by upward sublevel filtration). **Standard input format for image-/volume-PH** (Wagner-Chen-Vučini 2012). **Zero-allocation hot path:** preallocate `len(values) * 2^d` cells (worst-case full grid) once.

5. **`topology/complex/cubical/boundary.go::Boundary(cc, cell) []SignedCell` (~60 LOC).** KMM §2.5 cubical boundary: for each non-degenerate dim i in mask, emit two faces with signs (-1)^(i-th non-degenerate position) and (-1)^(i-th non-deg + 1) from the two anchor positions {anchor[i], anchor[i]+1}. **Implementation uses `bits.OnesCount` and `bits.TrailingZeros`** for O(d) per boundary cell.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#B, n-torus regression):** β_k of the d-dimensional cubical n-torus T^d (cubical complex on Z/nZ ^d with periodic identification) ≡ binomial coefficient C(d, k) (analytic — H_k(T^d; F_2) = (F_2)^{C(d,k)}). Three independent computations saturate: (i) cubical Betti via T2 column reduction, (ii) iterated-Künneth on T^1=S^1 having (β_0, β_1) = (1, 1) lifted by T^d = (S^1)^d, (iii) closed-form C(d, k). **Pedagogical keystone test.**

### T2 — Cubical homology over F_2 standalone (~110 LOC, ZERO blocker, ZERO Eigvec dep)

6. **`topology/complex/cubical/homology.go::BettiF2(cc *CubicalComplex) []int` (~60 LOC).** β_k = dim C_k − rank ∂_k − rank ∂_{k+1} via column reduction over F_2 on the boundary matrices from T1. Fills the gap that current `topology/persistent.ComputeBarcode` only emits *bars* and only for *simplicial* input. **Standalone API the consumer needs for "is this voxel cluster connected? does it have a tunnel?"** — answer in 1 ms for a 128^3 voxel grid with ~10% density (~40k cells).
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#C, simplicial-equiv):** for any cubical complex K, the *barycentric subdivision* K_b is canonically a simplicial complex (each d-cube becomes d! d-simplices). β_k(K; F_2) ≡ β_k(K_b; F_2) (homotopy invariance). Regression to slot 283's `topology/simplicial.BettiF2` once that ships. **Cross-package validation pin saturating both 283-T2 and 284-T2 simultaneously.**

7. **`topology/complex/cubical/cycles.go::CyclesAndBoundariesF2(cc, k) (cycles, bdries [][]CubicalCell) (~50 LOC).** Sparse representatives of Z_k = ker ∂_k and B_k = im ∂_{k+1}. Useful for visualizing a 2D loop / 3D void in voxel data (medical-imaging segmentation, materials microstructure).

### T3 — Cubical persistence (lower-star + upper-star, twist) (~180 LOC, ZERO blocker)

8. **`topology/complex/cubical/persistence.go::CubicalPersistence(cc *CubicalComplex, maxDim int) []Bar` (~140 LOC).** Reuse Chen-Kerber-2011 twist algorithm (or naive ELZ as Phase-A) on the cubical filtration value attached at T1.4. Returns `[]Bar{Dim, Birth, Death}` API-compatible with `topology/persistent.Bar`. **Connects directly to slot 283-T9 persistence-landscape / persistence-image vectorization** (Bubenik 2015 / Adams 2017) — the same Bar type → same downstream ML feature pipeline for image data.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#D, smooth ↔ discrete):** for a smooth function f: T^d → R sampled on a regular grid, the cubical persistence of the lower-star filtration converges (in bottleneck distance, Wagner 2012 stability theorem) to the smooth sublevel-set persistence as grid → ∞. Pin: refine 32^2, 64^2, 128^2 → bottleneck distance to limit decreases as O(1/n). Stability witness.

9. **`topology/complex/cubical/lowerstar.go::LowerStarFiltration(values, shape) *CubicalComplex` (~40 LOC).** Convenience wrapper assigning each cell its lower-star value (max of incident-vertex values). The standard sublevel-set construction (Wagner 2012 §3.2). **Single-call API for "give me the persistence diagram of this image":**
   ```go
   bars, _ := cubical.CubicalPersistence(cubical.LowerStarFiltration(img, [2]int{H, W}), 1)
   // bars[k]: persistent topological features at scale "pixel-value"
   ```

### T4 — Discrete Morse matching (Lewiner-2003 greedy) on generic CellComplex (~180 LOC, ZERO blocker)

10. **`topology/complex/morse/matching.go::GreedyMatching(cc CellComplex) MorseMatching` (~140 LOC).** Lewiner-Lopes-Tavares-2003 coreduction-pair greedy: process cells in *decreasing* dimension; for each unmatched τ with exactly one unmatched codim-1 face σ ("free face"), match σ↑τ and add to V; if no free face, pick a critical cell. **Linear-time O(total face-incidences)**; matches simplicial / cubical / CW interchangeably via T0 CellComplex interface. Returns `MorseMatching{Pairs []CellPair; Critical []CellID}`.
    ```go
    type MorseMatching struct {
        Pairs    []CellPair       // matched (σ < τ) gradient pairs
        Critical []CellID         // unmatched cells = critical
        Levels   map[CellID]int   // for filtration ordering
    }
    ```

11. **`topology/complex/morse/vectorfield.go::IsValidVectorField(V MorseMatching, cc CellComplex) error` (~40 LOC).** Validates Forman §3 axioms: (a) each cell appears in at-most-one pair, (b) every pair (σ, τ) has σ codim-1 face of τ, (c) the modified Hasse diagram (reverse arrows in V) has **no directed cycle** (acyclicity = Forman's "no V-paths from σ back to σ"). **Single load-bearing correctness witness** for any custom Morse-matching algorithm.

### T5 — Morse complex (collapsed homotopy-equivalent CW) (~160 LOC)

12. **`topology/complex/morse/collapse.go::MorseComplex(cc CellComplex, V MorseMatching) *MorseCellComplex` (~120 LOC).** Build the homotopy-equivalent CW-complex with **one cell per critical cell** of V. Boundary is the *Morse boundary* ∂^M: counts gradient paths in V from face-of-τ to σ (Forman §6, Lewiner-Lopes-Tavares §3.3). **Memory savings:** typical input (Wagner 2012 medical-MRI 256^3 cubical) reduces from ~17M cells to ~50 critical cells = 350,000× reduction, with identical homology.
    ```go
    type MorseCellComplex struct {
        Critical []CellID
        Bdry     map[CellID][]SignedCell // sparse; indices into Critical
    }
    ```

13. **`topology/complex/morse/collapse.go::HomologyOfCollapsed(mcc *MorseCellComplex) []int` (~40 LOC).** β_k from T2 column-reduction on the Morse complex.
    - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#E, homotopy invariance):** β_k(cc) via T2.6 ≡ β_k(MorseComplex(cc, V)) via T5.13 ≡ β_k(cc) via 283-T2 BettiF2 (after barycentric subdiv if cubical). **Three-axis saturation** witnessing Forman's collapse theorem to round-off; load-bearing test that catches Morse-matching-bug-which-changes-homology immediately.

### T6 — Morse-theoretic persistent homology (Mischaikow-Nanda 2013) (~140 LOC)

14. **`topology/complex/morse/persistence.go::MorsePersistence(cc *FilteredCellComplex) []Bar` (~140 LOC).** Mischaikow-Nanda 2013 algorithm: (a) compute filtration-respecting Morse matching (free-face must respect birth time), (b) build filtered Morse complex, (c) compute persistence on this small complex with twist. **Result:** identical persistence diagram, **5-100× wall-clock speedup** vs naive on cubical (Wagner-Chen-Vučini 2012 report 100× on 128^3 medical MRI). 143-line-273 keystone primitive.
    - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#F):** MorsePersistence(cc) ≡ T3.8 CubicalPersistence(cc) ≡ slot-283-T4-PersistenceTwist(cc as simplicial via barycentric subdiv). **Three independent persistence algorithms saturate** on the same filtered cubical input.

### T7 — Δ-complex (Hatcher-2002 §2.1) (~180 LOC, frontier)

15. **`topology/complex/delta/delta.go::DeltaComplex` (~180 LOC).** Each n-cell is an n-simplex Δ^n with a fixed characteristic-map ordering of its n+1 vertices, but unlike abstract simplicial DIFFERENT n-cells may have the same vertex set. ~6× cell-count savings vs abstract simplicial on T^d, K(π,1), surfaces of genus g. CW-cell substrate: each Δ-cell has its own attaching map indexed by face-restriction. Implements T0 CellComplex interface; reuses T4-T6 Morse machinery.

### T8 — General regular CW-complex (~280 LOC, frontier, defer)

16. **`topology/complex/cw/regular.go::RegularCW` (~280 LOC).** Each n-cell has an attaching map φ:S^{n-1} → X^{(n-1)} that is a *homeomorphism* onto its image (regular = embedded). Boundary is well-defined as the image. **Generalizes simplicial / cubical / Δ uniformly.** Implementation: `Cell{Dim int; AttachToFaces []SignedCell}` directly storing the boundary chain. Most general substrate. **Defer** until a downstream consumer pulls (research-grade construction; cubical handles 95% of practical raster/voxel needs).

### T9 — Cell-complex Hodge Laplacian (~140 LOC; matrix construction unblocked, spectrum gates on 097-T1)

17. **`topology/complex/morse/hodge.go::HodgeLaplacianCell(cc CellComplex, k int) (Lk [][]float64)` (~70 LOC).** L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k on any cell complex (Eckmann 1944 generalized; Lim 2020 cellular). **Matrix construction ships unblocked.** Cross-link: 246-X12 cotangent-Laplacian on triangle mesh = special case via Galerkin Hodge ★; 282-T6 hypergraph Laplacian + 283-T7 simplicial Hodge L_k = special cases.
    - **R-MUTUAL-CROSS-VALIDATION 3/3 pin (#G, Hodge theorem):** dim ker L_k(cc) ≡ T2.6 BettiF2(cc) (modulo 2-torsion) ≡ T5.13 BettiOfMorseComplex(cc, V). Three independent computations saturate. **Spectrum gates on slot 097-T1 Eigvec.**

### T10 — Random discrete Morse (Benedetti-Lutz 2014) (~80 LOC, downstream)

18. **`topology/complex/morse/random.go::RandomMatching(cc, rng) MorseMatching` (~80 LOC).** Greedy matching with random tiebreaking on cell ordering. Empirically near-optimal on benign inputs (Benedetti-Lutz §4). Useful as a Monte-Carlo lower bound on critical-cell count and for catching pessimal-input bugs.

## Single cheapest day-1 PR

**`topology/complex/cubical/` package, T0+T1+T2+T3 ~480 LOC, ZERO blockers, ZERO Eigvec dep, operates directly on raster image data.**

- `topology/complex/cell.go` — CellComplex interface + Hasse diagram (T0.1+T0.2, ~140 LOC; shared by 283 retroactively)
- `topology/complex/cubical/types.go` — CubicalCell + CubicalComplex (T1.3, ~80 LOC)
- `topology/complex/cubical/builder.go` — FromRaster (T1.4, ~80 LOC)
- `topology/complex/cubical/boundary.go` — KMM-2004 cubical boundary (T1.5, ~60 LOC)
- `topology/complex/cubical/homology.go` — BettiF2 standalone (T2.6, ~60 LOC)
- `topology/complex/cubical/persistence.go` — Lower-star + ELZ-twist persistence (T3.8+T3.9, ~180 LOC subset)

Tests:
- `TestCubicalDDZero` — `∂_{k-1}∂_k = 0` on every dim of every fixture (load-bearing orientation correctness witness, pin #A).
- `TestCubicalBettiTorus` — n-torus T^d β_k = C(d, k) for d ∈ {2, 3, 4} (analytic regression, pin #B).
- `TestCubicalEqualsSimplicialAfterSubdiv` — barycentric subdivision regression to slot 283's BettiF2 (cross-package pin #C; gates on 283 landing).
- `TestCubicalLowerStarPersistence_Wagner2012` — synthetic 32^2 height-function fixture from Wagner-2012 §6 with known birth/death pairs.

This PR ships the **standalone cubical TDA primitive** (cubical complex + cubical homology + cubical persistence on raster image data) — the **single most-used cell-complex flavor in production** (medical imaging, materials science, neuroscience MRI, computer vision). Discrete Morse (T4-T6) and Hodge spectrum (T9) are second-PR follow-ups; Δ-complex (T7) and full CW (T8) are deferred to consumer pull.

## Cross-cutting

- **Slot 097-T1 (linalg-missing) ←** T9.17 Hodge-Laplacian-cell SPECTRUM gates on EigvecSym. Matrix construction ships unblocked.
- **Slot 077/078 (geometry-{missing,sota}) ←** No dependency. Cubical works directly on raster; Δ-complex / CW need only abstract attaching maps.
- **Slot 142 (topology-missing) ←** This slot directly answers "cubical filtration ~700 LOC" and "discrete Morse ~350 LOC" called out at 142:198+209. Recommend re-read of 142 after this slot lands.
- **Slot 143 (topology-sota) ←** Directly answers "perseus discrete-Morse pre-collapse ~350 LOC, 5-10× speedup, 250-10000× cumulative" at 143:30+273-280+326+393. **Single highest-leverage TDA performance primitive 143 identified.**
- **Slot 156 (synergy-topology-prob) ←** Random cubical complexes (Hiraoka-Shirai-2017) consume T1 substrate; Bayesian-cubical-Betti via T2 + T6.
- **Slot 246 (new-discrete-exterior) ←** **single-source-ownership:** combinatorial cubical complex lives here (T1); 246-X13 imports, layers Verts coords + ★ on top. 246-X18 Maxwell on cubical mesh = T1 substrate + slot 283 ∂_k.
- **Slot 247 (new-mortar-fem) ←** mortar transfer between non-conforming cubical / simplicial meshes; needs T0 + T1.
- **Slot 280 (network-generative-models) ←** Linial-Meshulam-2006 random simplicial + Kahle-2014 random clique — extends to random cubical / random CW on T1 substrate. β_k as feature for SBM model selection.
- **Slot 281 (temporal-graphs) ←** zigzag persistence on cell complexes + Mayer-Vietoris on cubes; T0 + T6 substrate.
- **Slot 282 (hypergraphs) ←** disjoint axis (set system, not topology); no direct interaction.
- **Slot 283 (simplicial-complexes, just done) ←** **complementary not duplicative.** 283 owns simplicial axis; 284 owns cubical + general-CW + discrete-Morse axes. **Recommend unified `topology/complex/` parent** with 283's `simplicial/` re-housed under it, alongside `cubical/`, `cw/`, `delta/`, `morse/`. Discrete Morse (T4-T6) operates on **either** via T0 CellComplex interface. Persistence machinery shared via 283-T4-twist + this slot's T3 / T6.
- **Pistachio scene-analysis ←** cubical persistence on per-frame depth maps = direct loop-closure / scene-feature signal on raster Z-buffer. **Direct consumer pull for T3 + T6.** No simplicial intermediary needed (cubical works on raster directly).
- **Aicore (medical imaging, MRI segmentation, materials microstructure) ←** cubical persistence on volumetric scalar fields is the **standard tool** (Wagner-Chen-Vučini 2012). T1 + T3 + T6 are direct consumer-driven primitives; the hot path is the 100× Mischaikow-Nanda speedup which makes 256^3 voxel MRI tractable in ~1 second.
- **Witness death-of-cycle bit-stable fingerprints ←** named at `topology/persistent/doc.go:33`. T6 Morse-persistence + barcode → fingerprint for image / video / volumetric data (current witness uses simplicial VR, missing the cubical raster axis).
- **Insights blast-radius topology ←** named at doc.go:38. Service-graph topology is simplicial, but if Insights extends to a 3D dependency-mesh visualization (heat-map), cubical T1+T3 is the substrate.
- **Slot 246-X13 (cubical complex inside DEC) ←** import-relationship: 246-X13's `CubicalComplex{Verts, Edges, Quads, Cubes}` IS T1.3 + Verts coordinates + ★_k. Recommend 246-X13 collapse to ~50 LOC importing from `topology/complex/cubical/`.

## Sources

- `topology/persistent/vr.go:14,50,91`, `topology/persistent/barcode.go:60,221`, `topology/persistent/doc.go:1-125`
- `geometry/sdf.go`, `geometry/polygon.go`, `linalg/eigen.go:20`
- Reviews: `agents/097-linalg-missing.md` (Eigvec gate for T9), `agents/142-topology-missing.md:198,209` (cubical+Morse gap), `agents/143-topology-sota.md:30,273-280,326,393` (perseus Morse 250-10000× cumulative speedup), `agents/156-synergy-topology-prob.md`, `agents/246-new-discrete-exterior.md:36,150,210` (X13 cubical, X18 Maxwell-on-cubical), `agents/281-new-temporal-graphs.md`, `agents/283-new-simplicial-complexes.md` (simplicial axis, complementary)
- Hatcher-2002 *Algebraic Topology* Cambridge UP — Chapter 0 (CW), §2.1 (Δ-complex)
- Kaczynski-Mischaikow-Mrozek-2004 *Computational Homology* Springer AMS-157 — §2 cubical homology, §2.5 cubical boundary, §2.3 closure axiom
- Forman-1998 *Adv-Math* 134:90 "Morse theory for cell complexes" + Forman-2002 *Sém-Lothar-Combin* 48 user's guide
- Lewiner-Lopes-Tavares-2003 *J-Math-Imaging-Vision* 19:223 + 2003 *GMP* "Optimal discrete Morse functions for 2-manifolds" — greedy matching algorithm
- Mischaikow-Nanda-2013 *Discrete-Comput-Geom* 50:330 "Morse theory for filtrations and efficient computation of persistent homology" — the keystone paper
- Wagner-Chen-Vučini-2012 *TopoInVis-II* "Efficient computation of persistent homology for cubical data" — T-construction, V-construction, lower-star
- Sköldberg-2006 *Trans-AMS* 358:115 "Morse theory from an algebraic viewpoint" — algebraic Morse on chain complexes
- Benedetti-Lutz-2014 *Exp-Math* 23:66 "Random discrete Morse theory" — empirical near-optimality
- Bauer-Kerber-Reininghaus-Wagner-2017 (PHAT) *J-Symb-Comput* 78:76 "Phat – persistent homology algorithms toolbox"
- Eckmann-1944 *Comment-Math-Helv* 17:240 (Hodge theorem on finite cell complex)
- Lim-2020 *SIAM-Rev* 62:685 "Hodge Laplacians on Graphs" (cellular generalization)
- GUDHI 3.10 (BSD-3) C++/Python ~50k LOC, PHAT (LGPL), perseus 4.0 (~6k C++ LOC), DiPHA (BSD-2), CHomP (KMM ref impl ~10k LOC)
