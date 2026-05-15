# 142 | topology-missing — canonical TDA primitives absent from `topology/`

Scope: enumerate computational-topology / TDA primitives absent from `C:\limitless\foundation\reality\topology\`. Agent 141 confirmed the package today is **one sub-package, `topology/persistent/`, ~1378 LOC**, shipping exactly: Vietoris–Rips filtration on a point cloud (maxDim ∈ {0,1}); F_2 column-reduction barcode (Edelsbrunner–Letscher–Zomorodian 2000); bottleneck distance on barcodes (Cohen-Steiner–Edelsbrunner–Harer 2007 stability theorem). Web crosswalk: GUDHI 3.10, Ripser 1.2.1, Dionysus2, giotto-tda 0.6, scikit-tda (KeplerMapper / Persim 0.4 / Ripser 0.6 / UMAP-flavoured Mapper), Eirene.jl, RIVET 1.1, JavaPlex, PHAT 1.5, OpenPH, perseus.

Files inspected: `topology/persistent/{doc,vr,barcode,bottleneck,errors,persistent_test}.go`. Adjacent reuse considered: `optim/transport` (1-D Wasserstein), `linalg` (sparse matrices), `signal` (FFT for sliding-window embedding), `graph` (BFS/DFS/Dijkstra for Reeb graph backbone), `geometry` (convex hull → α-complex Delaunay).

---

## Roster present today

**`topology/persistent/`:** `Simplex`, `Filtration`, `VietorisRipsComplex(points, maxRadius, maxDim)`, `ComputeBarcode(filtration, maxDim)`, `Bar { Dim, Birth, Death }` + `Persistence()` + `IsEssential()`, `BottleneckDistance(d1, d2, dim)`. Sentinel errors. **Total public surface: 6 types/functions.**

**Hard limits today:** maxDim ∈ {0,1}; F_2 coefficients only; one filtration kind (VR); one diagram metric (bottleneck); one input shape (R^d point cloud).

**Adjacent in repo (not topology, but reusable as deps for the additions below):**
- `optim/transport`: `Wasserstein1D`, `WassersteinND` (Sinkhorn). Reusable as the metric-computation backbone for **p-Wasserstein on diagrams** (T1.4) once the lift to "+ unbounded diagonal" is added.
- `signal`: `FFT`, `Hilbert`, window functions. Reusable for **sliding-window embedding (Takens)** (T1.7).
- `graph`: BFS / Dijkstra / connected components. Reusable for **Mapper graph extraction**, **Reeb-graph traversal**, **merge-tree linkage**.
- `geometry`: convex-hull / Delaunay primitives — load-bearing for the **α-complex** (T1.1); without Delaunay there is no α-complex, so this is a hard cross-package coupling.
- `linalg`: sparse-matrix path is required for any non-toy reduction (twist + clearing + Z_p coefficients).

**Roster gap delta vs 141:** 141 documented six numerics gaps in what ships. This report enumerates **everything the package does not ship at all**, ranked by canonical-library prevalence (GUDHI / Ripser / giotto-tda / scikit-tda / RIVET / Dionysus2).

---

## Tier 1 — Must-ship (table-stakes; every TDA library has these)

### T1.1 Filtration types beyond Vietoris–Rips (~700 LOC)

VR is the single filtration today. Every canonical library ships **at least four**:

- **Čech complex** (PDF: σ ⊆ Čech_r iff ∩B(v_i, r) ≠ ∅). Strictly tighter than VR (Čech_r ⊆ VR_r ⊆ Čech_{2r/√3} on R^d via Jung's theorem). Provides exact homology of the union of balls (the Nerve Theorem). GUDHI `CechComplex`, Dionysus `fill_cech`. The hot path is the **miniball / smallest-enclosing-ball** subroutine — Welzl 1991 expected O(n) for fixed d, ~150 LOC. Plus the simplex enumeration (~80 LOC). The bottleneck of Čech is *miniball-of-d+2-points exact arithmetic*; for Phase-A point clouds in R^2 / R^3 this is a closed-form root of a quartic and 100 % stable.
- **α-complex** (Edelsbrunner–Mücke 1994). Subcomplex of Delaunay; for n points in R^d gives the smallest filtration with the same homotopy type as the union of balls. **By far the most efficient point-cloud filtration in low d** — n simplices instead of 2^n. GUDHI `AlphaComplex`. Hard requirement: 2-D / 3-D Delaunay triangulation in `geometry/`. If `geometry/` ships convex hull but not Delaunay, this T1 gates on a `geometry` PR. ~250 LOC topology-side once Delaunay is available.
- **Witness complex** (de Silva–Carlsson 2004). **Sub-sample-based:** L landmarks (chosen by max-min or random) + n witnesses (the rest). Builds a complex of size O(L^d) instead of O(n^d). Critical for n > 1000 where VR is intractable. GUDHI `WitnessComplex` + `StrongWitnessComplex`. ~180 LOC.
- **Sparse Rips** (Sheehy 2013, Cavanna–Jahanseir–Sheehy 2015 "geometric net"). ε-approximation of full VR with size O(n) instead of O(n^d). Within (1+ε) on barcode in d_B. Ripser `--ratio`, GUDHI `SparseRipsComplex`. ~200 LOC. **Prerequisite for any n > 100 use-case** — the audit-141 phase-A bound n ≤ 50 dissolves the moment Witness wants intra-day prices at minute resolution.
- **Cubical complex** (top-down from a uniform grid). Image / volume input. GUDHI `CubicalComplex`, persim. Sub-level-set filtration on a scalar function `f: V → R`. ~150 LOC. Required for **persistence on images / spectrograms / scalar fields** (Pistachio frame-buffers, Witness's NN attention maps).
- **Flag / clique complex from a graph** (general). Today the VR builder is morally a clique-complex on the proximity graph; lift the API so any graph (e.g., a correlation graph from `prob/`, a service graph from `graph/`) can be filtered. Ripser-Live, GUDHI `FlagComplex`. ~80 LOC once the simplex enumerator is decoupled.
- **Lower-star / upper-star filtration** on a simplicial complex with vertex values. Standard reduction for scalar-field persistence on a triangulated mesh. ~50 LOC.

### T1.2 Higher-dimensional homology (H_2, H_3, …) (~150 LOC + sparse-reduction rewrite)

Today: maxDim ∈ {0,1}. The defence (141 audit point §1) is sound at n ≤ 50 with a dense reduction, but the **arithmetic** of higher-dimension reduction is a one-liner — the package's `boundaryColumn` is dim-agnostic. The blockers are:
- **Twist optimization** (Bauer–Kerber–Reininghaus 2014): process highest dimension first, mark pairs, skip cleared columns. ~50 LOC, **5–50× speedup on real inputs**, mandatory for maxDim ≥ 2.
- **Clearing optimization** (Chen–Kerber 2011): dual of twist; in the up-pass, columns whose boundary contains a known pivot are cleared. ~40 LOC.
- **Sparse boundary representation**. Today's `intSet` is fine at n ≤ 50; for maxDim=2 with n=50 the boundary matrix has 230k columns × ~3 nnz each — needs the integer-index refactor 141 already proposed.

Without these, lifting maxDim to 2 is correct but unusable (multi-second on n=50). With them, n=200 at maxDim=2 is sub-second.

### T1.3 Persistent cohomology (Morozov 2011) (~250 LOC)

The **dual** computation: same chain complex, transposed boundary, processed left-to-right. Mathematically equivalent diagrams for finite filtrations (de Silva–Morozov–Vejdemo-Johansson 2011), but **far faster in practice on the maxDim ≥ 1 case** because the cohomology boundary matrix is sparser column-wise. **This is what Ripser 1.2.1 does and is the reason Ripser is 10–100× faster than the C# / VR-from-scratch implementations.** Required if the package wants to be competitive at all with Ripser/GUDHI/Dionysus on benchmarks. Returns identical bars to the homology path — passes through the same tests. ~250 LOC of careful indexing.

### T1.4 p-Wasserstein distance on persistence diagrams (~200 LOC)

Today: only **bottleneck** (p = ∞). Every canonical library also ships **p = 1, 2, ∞**:
- d_W^p(D, D') = (inf_M Σ ||p − M(p)||_∞^p)^{1/p}, with the diagonal as an unbounded reservoir.
- Reduces to a **balanced-assignment problem** — Hungarian algorithm O(n^3) or auction algorithm O(n^2 log n max-cost). The Kerber–Morozov–Nigmetov 2017 paper gives the canonical encoding (already correctly used in `bottleneck.go` for p=∞).
- p=2 is Pistachio's preferred metric for **smooth gradient-based diagram comparisons** (giotto-tda's `WassersteinDistance(p=2)`).
- Stability: 1-Wasserstein has a different stability constant (Cohen-Steiner–Edelsbrunner–Harer–Mileyko 2010); the docstring should cite both.
- Ships alongside the existing `BottleneckDistance` at parity API: `WassersteinDistance(d1, d2, dim, p float64)`. ~200 LOC including a sparse Hungarian ~150 LOC.
- Reuse: `optim/transport` already has 1-D Wasserstein and Sinkhorn — but persistence-diagram Wasserstein is the **diagonal-augmented** variant which `optim/transport` does not implement. Either add a `transport.PersistenceWasserstein` helper or keep it in `topology/`.

### T1.5 Vectorized features: persistence images / landscapes / silhouettes (~400 LOC)

The **#1 reason TDA is used in machine-learning pipelines today** — diagrams are not Euclidean, so they're not directly fed to classifiers. The vectorizations:
- **Persistence landscapes** (Bubenik 2015). λ_k(t) = k-th largest of {Λ_p(t) : p ∈ D}, where Λ_p(t) is the tent function on bar p. **Stable in 1-Wasserstein, p-integrable, vectorizable to a fixed grid → 1-D vectors usable in any classifier.** ~120 LOC. persim `PersLandscapeApprox`, GUDHI `Landscape`.
- **Persistence images** (Adams–Emerson–Kirby–Neville–Peterson–Shipman–Chepushtanova–Hanson–Motta–Ziegelmeier 2017). Map each (b, d) to (b, d − b), Gaussian-kernel onto a 2-D grid, weighted by a non-decreasing function of persistence. **Stable in 1-Wasserstein.** ~150 LOC. persim `PersImage`, giotto-tda `PersistenceImage`.
- **Silhouettes** (Chazal–Fasy–Lecci–Rinaldo–Wasserman 2014). Power-weighted average of landscape tents → single function instead of a sequence. ~80 LOC.
- **Persistence betti curves / Euler characteristic curves** — count of bars alive at each filtration time. Trivial step-function from the barcode (~40 LOC) but **the simplest TDA feature for time-series classification** — used in giotto-tda's TS-fresh-style baselines.
- **Persistence entropy** (Atienza–González-Díaz–Soriano-Trigueros 2019). H(D) = − Σ p_i log p_i where p_i = persistence(b_i)/total_persistence. Single scalar; very cheap; used as a lightweight feature. ~30 LOC.

These five are **table-stakes for any TDA-feeds-ML use-case** (and Witness's morning-report and RubberDuck's risk-analyzer both want a vectorized form for downstream regressors).

### T1.6 Stability witness + diagram-arithmetic helpers (~150 LOC)

Beyond the metric, every library ships a small zoo of diagram operations:
- `Diagram` first-class type (today bars are `[]Bar`; a typed `Diagram` with attached metadata `(filtrationName, maxDim, builderHash)` lets consumers compare apples-to-apples). ~50 LOC.
- `Bar.IsTrivial()` (b == d) and `Diagram.PruneTrivial()` — 141 audit gap §3.
- `Diagram.PruneShortBars(eps)` — drop bars with persistence < eps. ~10 LOC.
- `Diagram.MaxPersistence()`, `Diagram.TotalPersistence()`. ~10 LOC.
- `Diagram.MeanBirth()`, `Diagram.MeanDeath()`, `Diagram.Centroid()`. ~20 LOC.
- `Diagram.Subdiagram(dim)` — already done implicitly inside bottleneck. Make it public. ~10 LOC.
- **Stability test fixture** as documented in 141 (perturb input by ε, assert d_B ≤ ε). ~30 LOC test.

### T1.7 Sliding-window embedding for time-series (Takens) (~120 LOC)

The **canonical bridge from a 1-D time-series to TDA**. Given x[t], embed as ψ(t) = (x[t], x[t+τ], …, x[t+(m−1)τ]) ∈ R^m, then run VR on the embedded cloud. Perea–Harer 2015 prove the resulting H_1 captures **periodicity** (a true period gives an essential H_1 of magnitude ~amplitude). giotto-tda `SingleTakensEmbedding`. ~120 LOC plus the τ / m auto-selection (mutual-information minimum and false-nearest-neighbours); these auto-selectors live more naturally in `signal/` or `chaos/` (which already has Lyapunov exponents and would benefit from a shared embedding helper). **Pistachio sentiment-time-series and Witness daily price series both want this.**

---

## Tier 2 — High-value (most canonical libraries ship; second-consumer-pull worthy)

### T2.1 Mapper algorithm (Singh–Memoli–Carlsson 2007) (~400 LOC)

The other half of TDA in industry. Given (X, f, cover, cluster):
1. Filter `f: X → R^k` (height function — typically a 1-D or 2-D coordinate, or PCA-1 / eccentricity).
2. Cover the image of `f` by overlapping intervals/hypercubes.
3. For each cover element U_i, cluster the preimage `f^{-1}(U_i)` (single-linkage / DBSCAN).
4. Make a graph: one node per cluster, edges between clusters whose preimages share a point.

Ships in: KeplerMapper, GUDHI `MapperComplex`, giotto-tda `MapperPipeline`, Ayasdi (commercial). **The Workshop ecosystem-topology dashboard explicitly names Mapper** (`doc.go:28-30`). The composable parts:
- Filter functions: PCA-1, eccentricity, density estimator, L^∞-centrality, per-vertex value (~80 LOC).
- Cover: uniform-interval (1-D) and uniform-grid (2-D) with overlap percentage (~60 LOC).
- Clustering: single-linkage with a **gap heuristic** (skip clusters separated by < gap). ~100 LOC.
- Nerve graph construction (~80 LOC; reuse `graph/` for adjacency).
- Output: a `MapperGraph { Nodes []MapperNode; Edges [][2]int }` with each node carrying the indices of its constituent points (so a UI can drill down). ~50 LOC.

Plus the **multi-resolution Mapper** (Carrière–Michel–Oudot 2018) which gives Mapper a **persistent** flavour and connects it back to the rest of this package.

### T2.2 Wasserstein-stable diagram kernels (~200 LOC)

For **TDA-meets-kernel-methods**:
- **Persistence Scale-Space Kernel** (Reininghaus–Huber–Bauer–Kwitt 2015). Heat-equation evolution of a Dirac at each diagram point + Dirac at its mirror. Closed-form ⟨D, D'⟩_PSS = 1/(8πσ) Σ_{p∈D} Σ_{q∈D'} (exp(−||p−q||²/8σ) − exp(−||p−q̄||²/8σ)). ~80 LOC.
- **Persistence Weighted Gaussian Kernel** (Kusano–Hiraoka–Fukumizu 2016). Like PSS but with a learnable persistence-dependent weight. ~80 LOC.
- **Sliced Wasserstein Kernel** (Carrière–Cuturi–Oudot 2017). Average 1-D Wasserstein over random projections. **Fast, positive-definite, library-friendly.** ~120 LOC. Uses `optim/transport.Wasserstein1D`.

These kernels are the standard input to SVMs / GPs over diagrams, and underpin both the giotto-tda `Kernel` namespace and the `Persim` `kernels` module.

### T2.3 Reeb graph / merge tree / contour tree (~600 LOC)

The other "shape descriptor" family — **graph-valued summaries of a scalar function on a domain**:
- **Merge tree** of `f: K → R` on a simplicial complex K — the tree of evolving sub-level-set components. Builds in O(α(n)·m) via Tarjan-style union-find. ~150 LOC. The H_0 part of persistence is precisely the persistence of this tree.
- **Contour tree** (Carr–Snoeyink–Axen 2003) — the join-tree merged with the split-tree. Captures the topology of every level set of `f`. **Standard input to scientific-visualisation pipelines.** ~250 LOC.
- **Reeb graph** (Reeb 1946) — quotient of K by "same connected component of level set". Generalises contour tree to non-simply-connected domains. Edelsbrunner–Harer–Patel 2008 algorithm. ~250 LOC.
- **Mapper as a Reeb-graph approximation** (Carrière–Oudot 2017) — ties T2.1 and T2.3 together cleanly.

These need a **simplicial-complex-with-vertex-function** input type (today's `Filtration` is the related-but-different "indexed simplices"). Lift the input model to support both.

### T2.4 Wasserstein-amortized persistent-pair tracking (vineyard) (~250 LOC)

**Cohen-Steiner–Edelsbrunner–Morozov 2006 vineyards.** Given a one-parameter family of filtrations f_t(σ), track each (birth, death) pair as t varies. Output: a "vineyard" — a curve in R³ (birth, death, t) for each persistent class. Standard for **dynamic point clouds / time-evolving filtrations** — Witness's day-over-day, Pistachio's frame-over-frame. Implementation: a single transposition swap costs O(m) on the reduced matrix. ~250 LOC including the swap-update primitive.

### T2.5 Discrete Morse theory (~350 LOC)

Forman 1998. A **gradient vector field** on a simplicial complex K — a partial matching of faces and cofaces such that the unmatched simplices ("critical") generate the same homology as K but at much lower cost. perseus, GUDHI's `simplex_tree::collapse`. The cofaceless-pair greedy algorithm (Mischaikow–Nanda 2013) reduces a typical n=1000 VR complex by 90 % before reduction. ~350 LOC. **Multiplicative speed-up on T1.2 + T1.3.**

### T2.6 Z_p coefficients (typically Z_2, Z_3, Z_5) (~150 LOC)

Today: F_2 only. Z_p coefficients catch **torsion** that F_2 misses (e.g., the Klein bottle has H_1(K, Z_2) = Z_2 ⊕ Z_2 but H_1(K, Z) = Z ⊕ Z_2 — only Z_p for p odd separates the torsion). For p > 2 the boundary signs matter (alternating-sign formula on faces). ~150 LOC: signed boundary, modular-arithmetic XOR replacement, parametric `prime int`. Ripser supports p=2,3,5; GUDHI any prime.

### T2.7 Approximate/sparse algorithms for n > 1000 (~600 LOC)

- **Apparent-pair shortcut** (Bauer 2021 / Ripser). Detect persistence pairs that are obvious from the filtration order itself (every "apparent pair" is a born-and-dies-immediately bar) and exclude them from reduction. **Ripser's primary speed-up after twist + cohomology.** ~150 LOC.
- **Emergent pairs** (Bauer 2021). Same idea, finer detection. ~80 LOC.
- **Edge-collapse preprocessing** (Boissonnat–Pritam 2020). Reduce a flag complex by collapsing dominated edges before homology. **Frequently 10–100× size reduction with zero info loss.** ~250 LOC.
- **Strong collapse** (Barmak–Minian 2012). Vertex-level collapse; cheaper than edge-collapse, similar gains on dense inputs. ~120 LOC.

---

## Tier 3 — Frontier / specialized (not yet table-stakes; document as "out of scope unless N consumers pull")

### T3.1 Multiparameter persistence (~1500+ LOC)

**The active research frontier.** When the filtration depends on > 1 real parameter (e.g., scale × density), the persistence module is no longer decomposable by Crawley-Boevey 2015. RIVET (Lesnick–Wright 2015) computes the **fibered barcode** + **Hilbert function** + **bigraded Betti numbers** via line-search through the parameter plane. ~1500 LOC. **Cite RIVET as the canonical reference, defer.** The Pistachio "scale × density" view of attention maps could pull on this; until a flagship asks, defer.

### T3.2 Zigzag persistence (Carlsson–de Silva 2010) (~700 LOC)

When the sequence of complexes has both insertions **and** deletions: K_0 ⊂ K_1 ⊃ K_2 ⊂ K_3 ⊃ …. Standard for **levelset persistence on scalar fields** (sub-level + super-level interleaved) and **dynamic / time-varying** inputs. Decomposition of the zigzag module into intervals (right-filtration algorithm of Maria–Oudot 2014). ~700 LOC. Dionysus2 `ZigzagPersistence`. **Defer pending consumer.**

### T3.3 Persistent local homology (Bendich–Wang–Mukherjee 2012) (~400 LOC)

H_*(X, X − x) at each point x → "stratification descriptor". Used for **detecting singularities, branch-points, dimensionality changes** in stratified spaces. Applications are research-only as of 2025. ~400 LOC. **Defer pending consumer.**

### T3.4 Morse–Smale complex (~600 LOC)

Edelsbrunner–Harer–Zomorodian 2003. Given a Morse function on a manifold, decompose into ascending+descending manifolds of critical points. **The structural backbone of scalar-field topology** (terrain analysis, scientific-viz, segmentation). Discrete version (Robins–Wood–Sheppard 2011) via discrete Morse theory (T2.5). ~600 LOC. **Defer until cubical/triangulated-mesh input is requested.**

### T3.5 Persistent homology transform (PHT) (Turner–Mukherjee–Boyer 2014) (~250 LOC)

A shape descriptor: scan a height function over all directions in S^{d-1}, take the persistence diagram of each, get a function S^{d-1} → Diagrams. **Statistically inverts the shape (Turner et al. 2014, Curry–Mukherjee–Turner 2018).** ~250 LOC + S^{d-1} sampler. Used in computational anatomy, cosmology. **Defer.**

### T3.6 Topological autoencoders / loss functions (~300 LOC)

Moor–Horn–Rieck–Borgwardt 2020 differentiable persistence. Hofer–Kwitt–Niethammer–Uhl 2017 connectivity-preserving losses. **Trains models with topological constraints** (preserving cycles, separating clusters). Differentiable through the column reduction (Carriere–Chazal–Ike–Lacombe–Royer–Umeda 2021). ~300 LOC. **Defer; aicore-side experiment first.**

### T3.7 Quiver / TDA-MAPPER cluster analysis (~200 LOC)

The cluster-analysis side of T2.1. **DBSCAN** + **HDBSCAN** would also benefit `prob/` and `optim/`. ~200 LOC if shared with `prob`. **Defer; cross-cuts to a clustering package that doesn't exist yet.**

### T3.8 Quotient / equivariant persistence (~400 LOC)

Persistence on a group-action quotient (e.g., rotation-invariant shape descriptors). 2024 active research. **Defer.**

### T3.9 Tree barcodes / dendrograms (~150 LOC)

Hierarchical-clustering output as a barcode (this is just H_0 persistence on the single-linkage filtration). **Already implicitly covered by T1.1 + the existing H_0 path** if a `SingleLinkageFiltration` constructor is added. ~150 LOC; **borderline T1.5 / T2.6 — list here as "trivial wrapper, ship with T1.1".**

### T3.10 Spectral simplicial methods (~400 LOC)

Hodge Laplacian L_k on a complex; spectral decomposition gives **harmonic representatives of cohomology** + a **Cheeger-style cluster signal**. Lim 2020 surveys. Used in graph-signal processing on simplicial complexes. ~400 LOC. **Defer.**

---

## Tier-priority summary

| Tier | Group | Approx LOC | Priority anchor |
|---|---|---:|---|
| T1.1 | Filtration types (Čech, α, witness, sparse-Rips, cubical, flag, lower-star) | 700 | GUDHI ships all 7 |
| T1.2 | H_2+ + twist + clearing | 150 + sparse-rewrite | Ripser ships standard |
| T1.3 | Persistent cohomology | 250 | Ripser's central optimisation |
| T1.4 | p-Wasserstein on diagrams | 200 | giotto-tda / persim ship |
| T1.5 | Vectorisations (landscapes, images, silhouettes, betti, entropy) | 400 | persim ships all 5 |
| T1.6 | Diagram-arithmetic helpers + stability test | 150 | Hygiene |
| T1.7 | Sliding-window (Takens) embedding | 120 | giotto-tda ships |
| T2.1 | Mapper | 400 | Workshop dashboard explicit pull |
| T2.2 | Stable diagram kernels (PSS, PWG, Sliced-Wasserstein) | 200 | Persim / giotto-tda ship |
| T2.3 | Reeb / contour / merge tree | 600 | TopologyToolKit ships |
| T2.4 | Vineyards | 250 | Dionysus2 ships |
| T2.5 | Discrete Morse theory | 350 | perseus / GUDHI ship |
| T2.6 | Z_p coefficients | 150 | Ripser p ∈ {2,3,5} |
| T2.7 | Apparent-pairs / edge-collapse | 600 | Ripser / GUDHI ship |
| T3 | Multiparameter, zigzag, PLH, Morse-Smale, PHT, topological loss, quiver, equivariant, tree-barcodes, spectral | 4400+ | Defer per Pre-Mortem 007 |

**Tier-1 total: ~1970 LOC.** Brings `topology/persistent` from "ELZ-2000 textbook column reduction at maxDim ≤ 1" to "feature-parity with Persim/Ripser-Lite/Mapper". Tier-2 adds another ~2550 LOC for parity with GUDHI's full surface.

---

## Cross-package coupling (load-bearing dependencies for the additions)

- **T1.1 α-complex requires `geometry/Delaunay`** in 2-D / 3-D. If `geometry/` doesn't ship Delaunay, this T1 entry gates on a `geometry` PR. Audit `geometry/` before scheduling T1.1.
- **T1.4 p-Wasserstein** can either live in `topology/` (specialised diagonal-augmented variant) or as `optim/transport.PersistenceWasserstein` (general-balanced-assignment with a diagonal absorber). Prefer `topology/` because the diagonal augmentation is TDA-specific.
- **T1.7 Takens embedding** wants `signal.MutualInformation` (for τ-selection) and **belongs more naturally in `signal/` or `chaos/`**. The `topology/` API can be a thin wrapper that calls into the canonical impl.
- **T2.1 Mapper** wants `graph/` connected-components + a clustering primitive. The clustering primitive is *not* in the repo today (no DBSCAN, no single-linkage, no k-means). **Highest-value cross-cut for the next overnight planning batch.**
- **T2.3 Reeb / merge / contour** wants a richer simplicial-complex type than today's `Filtration` — lift the input model to support `(K, f: V → R)` directly.

---

## Headline gap vs canonical libraries

| Library | Filtrations | Coefficients | Vectorisations | Mapper | Multi-param | Notes |
|---|---|---|---|---|---|---|
| GUDHI 3.10 | VR, Čech, α, witness, sparse-Rips, cubical, flag, lower-star | F_2, Z_p any p | landscapes, images, silhouettes, kernels, entropy | yes | partial (RIVET-style) | C++ + Python, broadest surface |
| Ripser 1.2.1 | VR (sparse) | Z_p p ∈ {2,3,5} | none | no | no | Fastest VR by 10–100× |
| giotto-tda 0.6 | VR, Čech, cubical, sparse-Rips, weak-α | F_2 | landscapes, images, silhouettes, betti, entropy | yes | no | Best ML pipeline |
| scikit-tda (persim 0.4) | (relies on Ripser) | F_2 | landscapes, images, kernels (PSS, PWG, sliced-W) | KeplerMapper | no | Most ML-friendly |
| Dionysus2 | VR, α, custom | F_2 | none | no | no | Vineyards, zigzag |
| RIVET 1.1 | VR, generic bi-filtrations | F_2 | fibered barcode, Hilbert function | no | **yes (canonical)** | The multi-param tool |
| Eirene.jl | VR, custom | F_2 | none | no | no | Julia, Erdős-radius α |
| **reality v0.10.0** | **VR only** | **F_2 only** | **none** | **no** | **no** | maxDim ≤ 1, n ≤ ~50 |

The cleanest single-PR delta from "absent" to "minimum-viable competitive": **T1.1 (Čech + sparse-Rips) + T1.3 (cohomology) + T1.4 (p-Wasserstein) + T1.5 (landscapes + images + entropy)** — ~1550 LOC, takes the package from "ELZ textbook implementation" to "Persim + Ripser-Lite parity at maxDim ≤ 1". The α-complex slot is parked behind a `geometry/Delaunay` audit; T2.1 Mapper is parked behind a clustering-primitive decision.

---

## Summary (2 lines)

Today's `topology/` ships exactly **VR + ELZ + bottleneck at maxDim ∈ {0,1} over F_2** — six public symbols total, omitting every other canonical TDA primitive (Čech / α / witness / sparse-Rips / cubical / flag / lower-star filtrations; persistent cohomology; p-Wasserstein; landscapes / images / silhouettes / betti / entropy vectorisations; Mapper; Reeb / merge / contour trees; vineyards; discrete Morse theory; Z_p ≠ 2 coefficients; apparent-pairs / edge-collapse acceleration; multiparameter; zigzag; persistent local homology; Morse-Smale; PHT; topological autoencoders). Tier-1 (~1970 LOC) brings parity with Persim + Ripser-Lite + KeplerMapper; Tier-2 (~2550 LOC) brings parity with GUDHI's full surface; Tier-3 is research frontier deferred per Pre-Mortem 007.

---

Progress: 142-topology-missing complete — enumerated TDA primitives absent from `topology/` (today: VR + ELZ + bottleneck at maxDim ∈ {0,1}, 6 public symbols); Tier-1 ~1970 LOC for Persim/Ripser-Lite parity (Čech/α/witness/sparse-Rips/cubical/flag filtrations, H_2+, persistent cohomology, p-Wasserstein, landscapes/images/silhouettes/betti/entropy, Takens), Tier-2 ~2550 LOC for GUDHI parity (Mapper, Reeb/merge/contour, vineyards, discrete Morse, Z_p, apparent-pairs/edge-collapse, stable kernels), Tier-3 (multiparameter / zigzag / PLH / Morse-Smale / PHT / topological-loss / spectral) deferred.
