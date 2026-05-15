# 282 — new-hypergraphs (Hypergraph Spectral, Motifs, Laplacians, Simplicial Extension)

## Headline
reality v0.10.0 ships ZERO hypergraph surface (`Hyper|Hyperedge|Hypergraph|Tensor|Simplicial|ChainComplex|Incidence` repo-wide grep returns 0 hits outside review corpus); the entire Zhou-2006 / Hein-Setzer-2013 / Louis-2015 / Chan-2018 / Cooper-Dutle-2012 / Lotito-2022 / Carletti-2020 / Karypis-1999-hMETIS canon is greenfield; cheapest-day-1 PR is `graph/hyper/` sub-package (Hypergraph type + sparse incidence + Zhou-2006 normalized Laplacian via clique expansion, ~380 LOC, zero blockers — gates only on slot 097-T1 Eigvec for the *spectral* methods).

## Findings

### State at HEAD (verified by direct grep on `*.go`)

| Surface | Path | Hypergraph relevance |
|---|---|---|
| `Edge = [2]string` | `graph/graph.go:14` | Static **arity-2** edge tuple. NO hyperedge. Keystone-blocker. |
| `IntAdjacency = map[int][]int` | `graph/types.go:7` | Adjacency only. NO incidence-matrix-style representation. |
| `LouvainCommunities` | `graph/community.go:155` | Modularity on binary-edge graphs. Kumar-Vaidyanathan-2020 hypergraph-modularity ABSENT. |
| `EigenvectorCentrality / PageRank` | `graph/centrality.go, pagerank.go` | Power-iteration on binary adjacency. NO tensor-power-iteration (Cooper-Dutle adjacency-tensor eigvec) for k-uniform hypergraphs. |
| `MaxFlow` | `graph/flow.go` | Static binary s-t cut. Hypergraph-cut / NP-hard hypergraph-min-cut (Lawler-1973) ABSENT. |
| `linalg.QRAlgorithm` | `linalg/eigen.go:20` | Tridiagonal-QL eigenvalues only — **no eigenvectors**. Slot 097-T1 keystone-blocker for every *spectral* hypergraph method (Zhou-2006 Lap-eigvec → spectral-cut, Louis-2015 second-eigvec → Cheeger). |
| `linalg/pca.go::ipi` (private) | `linalg/pca.go:101-174` | Private inverse-iteration; slot 097-PR-4 / 157-PR-4 extracts as public `linalg.InverseIteration` and unblocks Zhou-2006 spectral-cut. |
| `topology/persistent/{vr,barcode}.go` | `topology/persistent/` | Vietoris-Rips simplicial-complex over points + persistence diagrams — **NOT** abstract simplicial complex, NO chain-complex / boundary maps `∂_k` / Hodge Laplacian `L_k = ∂_k^T ∂_k + ∂_{k+1} ∂_{k+1}^T`. |
| `linalg.{Sparse,Tensor,3DArray,MultiArray}` | `linalg/*.go` | **ALL ABSENT.** No CSR/CSC sparse, no order-3+ tensor type, no MTTKRP/CP/Tucker (slot 257 owns tensor-decomp but is downstream). |
| `Hyper\|Hypergraph\|Hyperedge\|Simplicial\|ChainComplex\|MultilinearMap\|Incidence` | repo-wide grep | **ALL ABSENT** outside review corpus. |

### Slot boundaries (delineates 282 from neighbours)

- **Slot 097-T1 (linalg-missing)** — dense symmetric **eigenvectors** (accumulate Z in QL, ~80 LOC). Hard prerequisite for Zhou-2006 / Louis-2015 / Chan-2018 spectral-methods.
- **Slot 162** (synergy-graph-prob) — random-graph **generators**; Ghoshdastidar-2017 Hypergraph-SBM generator is borderline 162/280, fits cleanly under slot 280's "inference axis = fitting" boundary as a 282 subordinate (sample is 162/280, fit is 282).
- **Slot 171** (synergy-graph-topology) — clique complex / flag complex / network homology of *binary* graphs. Slot 171 ships the **machinery** (`SimplicialComplex`, boundary maps `∂_k`, Hodge Laplacian); slot 282 ships hypergraph-as-simplicial-complex bridge + **non-binary** Laplacians.
- **Slot 241 / 246** (discrete-exterior-calculus) — discrete `d`/`δ`/`Δ` on simplicial meshes for PDEs (Hirani-2003). Slot 282's Hodge Laplacian is the same operator from the ML/network angle, not the FEM angle — *share an eventual `simplicial/` package*.
- **Slot 254** (graph-cuts / max-flow) — α-expansion + binary s-t cut. Slot 282 needs hypergraph s-t cut (Lawler-1973 reduces *star-expansion* hypergraph s-t cut to bipartite max-flow on (V ∪ E*, |V|+|E*|+1) with ∞ caps); 254 ships the back-end, 282 ships the bridge.
- **Slot 257** (tensor-decomposition) — CP / Tucker / HOSVD. Slot 282's *Cooper-Dutle adjacency tensor* is the natural application; 257 ships `Tensor` type + decomposers, 282 ships `HypergraphAdjacencyTensor()` constructor.
- **Slot 271** (spectral-clustering) — k-means on Lap-eigvec **of binary graphs**. Slot 282 ships hypergraph-Lap-eigvec → 271 reuses k-means back-end.
- **Slot 273** (spectral-embedding) — ASE/LSE for *binary*-graph latent-space. Hypergraph-spectral-embedding (Ghoshdastidar-Dukkipati 2017 *Ann-Stat* 45:289) is downstream of 273 + 282.
- **Slot 280** (network-generative-models) — SBM-fit on binary graphs. Hypergraph-SBM-fit (Ghoshdastidar-Dukkipati-2017, Lin-Chien-Ying-2025 *AISTATS* projected-tensor-power-method) is downstream of 280 + 282.
- **Slot 281** (temporal-graphs) — `(u,v,t)` link-stream. Composition with 282 → temporal hypergraphs (Cencetti-Battiston-2023 *PRX-Life* temporal-higher-order); 281 ships timestamp axis, 282 ships arity axis, joint package is 281+282 future-work.
- **Slot 282 (this) = HYPERGRAPHS — set-edges (arity ≥ 3) as first-class.** Owns Hypergraph type, normalized/unnormalized hypergraph Laplacian, hypergraph cut/conductance, hypergraph random walk + PageRank, p-Laplacian, tensor-spectral, hypergraph motifs, hypergraph s-t cut bridge to 254, simplicial-complex / chain-complex (composition with 171/241/246).

### Web context (no MIT pure-Go exists)

- **Zhou-Huang-Schölkopf-2006** *NeurIPS* 19:1601 "Learning with Hypergraphs: Clustering, Classification, and Embedding" — **canonical** normalized hypergraph Laplacian `L = I − D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}` (H = |V|×|E| incidence matrix, D_v / D_e diag of vertex / edge degrees, W diag of hyperedge weights). Equivalent (up to non-zero spectrum) to clique-expansion Laplacian and to Zhou's *star-expansion* Laplacian. NeurIPS-2006 ref is matlab ~150 LOC.
- **Hein-Setzer-Jost-Rangapuram-2013** *NeurIPS* 26 (arXiv:1312.5179) "The Total Variation on Hypergraphs — Learning on Hypergraphs Revisited" — hypergraph **total variation** `TV_H(f) = ∑_e w_e max_{u,v∈e} |f_u − f_v|` and corresponding **p-Laplacian** as Fréchet derivative; non-quadratic, non-linear, but *convex* for p≥1 and admits PDHG / FISTA solvers. Ref Matlab on CISPA ~600 LOC.
- **Louis-2015** *STOC* 47:713 "Hypergraph Markov Operators, Eigenvalues and Approximation Algorithms" (arXiv:1408.2425) — non-linear hypergraph Markov-operator `M(f)` whose principal eigvec → Cheeger-cut on hypergraphs; **proves no linear operator captures hypergraph-expansion in Cheeger-sense**. Establishes O(√(γ_2 log r)) Cheeger-inequality for hypergraphs (r = max hyperedge size). Reference: theory paper, no canonical implementation.
- **Chan-Louis-Tang-Zhang-2018** *J-ACM* 65:15 (arXiv:1605.01483) "Spectral Properties of Hypergraph Laplacian and Approximation Algorithms" — explicit construction of Louis's nonlinear Laplacian as a diffusion process; gives polynomial-time procedural-minimizer algorithm for k-th eigenvalue with O(log r) approximation; small-set-expansion + sparsest-cut applications.
- **Chan-Liang-2018** *IPCO* "Generalizing the Hypergraph Laplacian via a Diffusion Process with Mediators" — sub-modular generalization with edge-internal "mediator" weights.
- **Cooper-Dutle-2012** *Linear-Alg-Appl* 436:3268 "Spectra of Uniform Hypergraphs" — k-uniform hypergraph adjacency **tensor** of order k; H-eigvalue / Z-eigvalue (Qi-2005). Tensor-power-iteration (shifted symmetric higher-order, Kolda-Mayo-2011) computes principal H-eigpair.
- **Carletti-Battiston-Cencetti-Fanelli-2020** *PRE* 101:022308 "Random Walks on Hypergraphs" — hypergraph random-walk Laplacian `L_{rw} = I − D_v^{-1} H W D_e^{-1} H^T` reduces to standard random-walk Laplacian when r=2; closed-form stationary distribution `π_v ∝ ∑_{e∋v} w_e (|e|−1)`.
- **Karypis-Aggarwal-Kumar-Shekhar-1999** *IEEE-TVLSI* 7:69 + Karypis-Kumar-1998 hMETIS user-guide — **multilevel hypergraph partitioning**: coarsening (heavy-edge / hyperedge-coarsening), recursive bisection on coarsest, FM-refinement on uncoarsening. C reference ~10k LOC GPL. KaHyPar (Schlag-2021 *J-Exp-Algorithmics* 25:1.8) is current SOTA C++ GPL ~30k LOC. **Pure-Go MIT ABSENT.**
- **Lotito-Musciotto-Montresor-Battiston-2022** *Commun-Phys* 5:79 (arXiv:2108.03192) "Higher-order motif analysis in hypergraphs" — defines higher-order motifs (small connected sub-hypergraphs canonicalized over hyperedge-arity), exact + sampling enumeration. Ref Python `hypergraph-motif` MIT ~1500 LOC.
- **Ghoshdastidar-Dukkipati-2017** *Ann-Stat* 45:289 "Consistency of spectral hypergraph partitioning under planted partition model" + *JMLR* 18:50 "Uniform Hypergraph Partitioning: Provable Tensor Methods and Sampling Techniques" — Hypergraph-SBM weak-consistency proof + tensor-power-iteration for community recovery.
- **Bick-Gross-Harrington-Schaub-2023** *SIAM-Rev* 65:686 "What Are Higher-Order Networks?" — canonical 2023 survey unifying hypergraphs ↔ simplicial complexes ↔ cell complexes; reference for naming conventions and Hodge-Laplacian definition `L_k = ∂_k^T ∂_k + ∂_{k+1} ∂_{k+1}^T`.
- **Benson-Gleich-Leskovec-2016** *Science* 353:163 "Higher-order organization of complex networks" — motif-conductance on **binary** graphs via tensor-of-motif-counts → Cheeger-eigvec. Bridge to hypergraph Laplacian: motif-adjacency-matrix is hypergraph-clique-expansion when motif = k-clique.
- **Aksoy-Joslyn-Marrero-Praggastis-Purvine-2020** *J-Compl-Net* 8:cnaa018 "Hypernetwork science via high-order hypergraph walks" — hyperedge-overlap (s-walk) and dual-hypergraph foundations.

## Concrete recommendations

### T0 — Hypergraph type + sparse incidence (cheapest-day-1, zero blockers)

1. **`graph/hyper/types.go` (~110 LOC, zero deps).** New sub-package. Define:
   ```go
   // Hypergraph: undirected, hyperedge-weighted, vertex-weighted.
   // Edges are ordered slices (sorted ascending) of vertex indices ∈ [0, NumVertices).
   type Hypergraph struct {
       NumVertices int
       Edges       [][]int   // Edges[e] = sorted vertex indices in hyperedge e
       EdgeWeights []float64 // len == len(Edges); default 1.0
       VertWeights []float64 // len == NumVertices; default 1.0
   }
   func (h *Hypergraph) NumEdges() int
   func (h *Hypergraph) VertexDegree(v int) float64                // ∑_{e∋v} w_e
   func (h *Hypergraph) EdgeDegree(e int) int                       // |edge|
   func (h *Hypergraph) Incidence() (rows, cols []int, vals []float64) // CSR-like H ∈ ℝ^{V×E}
   func (h *Hypergraph) Validate() error                            // sorted, no dup, no out-of-range
   ```
   Unblocks: every primitive in this slot.

2. **`graph/hyper/expand.go::CliqueExpand(*Hypergraph) (adj, weights)` (~70 LOC).** For each hyperedge `e` of size `r`, emit `C(r,2)` pairwise edges with weight `w_e / (r−1)` (Zhou-2006 reduction; `1/(r−1)` is the unique weighting making clique-expansion-Laplacian spectrum match Zhou Laplacian's non-zero spectrum). Output is `(graph.IntAdjacency, map[[2]int]float64)` — directly reusable by Dijkstra/PageRank/Louvain. **Bridge primitive:** hypergraph problems → existing binary-graph back-end.

3. **`graph/hyper/expand.go::StarExpand(*Hypergraph) (bipartiteAdj, leftN, rightN int)` (~50 LOC).** Bipartite expansion: vertices on one side, hyperedges on the other; edge `(v, e)` for each `v ∈ e`. Used by Lawler-1973 hypergraph-min-cut reduction (T7 below) and Zhou's star-Laplacian-equivalence proof.

### T1 — Zhou-2006 normalized Laplacian + cut/conductance (singular cheapest-day-1 PR)

4. **`graph/hyper/laplacian.go::NormalizedLaplacian(*Hypergraph) (denseLDense []float64, n int)` (~100 LOC).** Construct `L = I − D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}` as dense `n×n` row-major. Symmetric, PSD, eigvalues in `[0,2]`. **R-MUTUAL-CROSS-VALIDATION 3/3 pin (Zhou-2006 Thm-1):** when all hyperedges are size-2, `NormalizedLaplacian(hypergraph)` equals `D^{-1/2}(D−A)D^{-1/2}` of the underlying graph **bit-exact** within `1e-12` (regression to graph case).
5. **`graph/hyper/laplacian.go::UnnormalizedLaplacian(*Hypergraph) ([]float64, int)` (~50 LOC).** `L = D_v − H W D_e^{-1} H^T`. Same regression to graph-Laplacian when r=2.
6. **`graph/hyper/cut.go::HypergraphCut(*Hypergraph, S map[int]bool) float64` (~60 LOC).** Cut weight `cut(S) = ∑_{e: e∩S≠∅, e∩(V\S)≠∅} w_e` (Zhou-2006). Plus `Vol(S) = ∑_{v∈S} d_v`, `Conductance(h, S) = cut(S) / min(Vol(S), Vol(V\S))`.
7. **`graph/hyper/cut.go::SpectralBisection(*Hypergraph) (S map[int]bool, conductance float64) (~140 LOC, BLOCKS on 097-T1 Eigvec).** Compute Fiedler eigvec of normalized hypergraph Laplacian via slot-097-T1's `linalg.EigvecSym` once it lands; sweep over sorted vertex order, return min-conductance bisection (Cheeger-style).

**T0+T1 PR shape:** ~380 LOC. Steps 1–6 are **zero-blocker, ship today**; step 7 gates on 097-T1.

### T2 — Hypergraph random walks + PageRank

8. **`graph/hyper/randomwalk.go::RandomWalkLaplacian(*Hypergraph) ([]float64, int)` (~50 LOC).** `L_{rw} = I − D_v^{-1} H W D_e^{-1} H^T` (Carletti-Battiston-Cencetti-Fanelli-2020 *PRE* 101:022308). **R-MUTUAL-CROSS-VALIDATION 3/3 pin:** `RandomWalkLaplacian(hypergraph)` equals `I − D^{-1}A` of the binary graph when r=2.
9. **`graph/hyper/randomwalk.go::Stationary(*Hypergraph) []float64` (~30 LOC).** Closed-form `π_v ∝ ∑_{e∋v} w_e (|e|−1)` (Carletti-2020 Eq-12). **R-MUTUAL-CROSS-VALIDATION 3/3 pin:** equals `d_v / 2|E|` for binary undirected graph (asymptotic equivalence to graph PageRank as α→0 in damped variant).
10. **`graph/hyper/pagerank.go::PageRank(*Hypergraph, alpha, tol float64, maxIter int) []float64` (~90 LOC).** Power iteration on damped random-walk operator `(1−α) P + α (1/n) 11^T` where `P = D_v^{-1} H W D_e^{-1} H^T`. Reuses `graph/pagerank.go` numerical pattern. Zero blockers.

### T3 — Tensor / nonlinear / p-Laplacian

11. **`graph/hyper/tensor.go::AdjacencyTensor(*Hypergraph, k int) (vals []float64, idx [][]int)` (~80 LOC, k-uniform only).** Cooper-Dutle-2012 order-k symmetric adjacency tensor `A_{i_1...i_k} = 1/(k−1)!` if `{i_1,...,i_k} ∈ E`. Sparse `(idx, vals)` storage — `k!|E|` non-zeros after symmetrization. **Composes with slot 257 Tensor type when that lands.**
12. **`graph/hyper/tensor.go::TensorPowerIter(vals, idx, k, n int, maxIter int) (lambda float64, x []float64)` (~140 LOC).** Kolda-Mayo-2011 *SISC* shifted symmetric higher-order power-iteration for principal H-eigpair of symmetric tensor `T x^{k−1} = λ x` with `||x||_2 = 1`. Foundation for projected-tensor-power Hypergraph-SBM (Lin-Chien-Ying-2025) feeding slot 280.
13. **`graph/hyper/p_laplacian.go::TotalVariation(*Hypergraph, f []float64, p float64) float64` (~50 LOC).** Hein-Setzer-Jost-Rangapuram-2013 `TV_H(f) = ∑_e w_e (max_{u,v∈e} |f_u − f_v|)^p`. Convex for p≥1; non-smooth.
14. **`graph/hyper/p_laplacian.go::PDHGSemiSupervised(...)` (~250 LOC, BLOCKS on optim FISTA / convex-prox).** Hein-2013 primal-dual splitting for `argmin TV_H(f) s.t. f_v = y_v ∀v ∈ labelled`. Defer to T4.

### T4 — Spectral clustering on hypergraphs

15. **`graph/hyper/spectral.go::SpectralKWay(*Hypergraph, k int) (labels []int) (~140 LOC, BLOCKS on 097-T1 + slot-271 KMeans).** k bottom Lap-eigvec of `NormalizedLaplacian` → row-normalized `n×k` embedding → `kmeans` (slot 271). Ng-Jordan-Weiss-2002 reduction. **R-MUTUAL-CROSS-VALIDATION 3/3 pin:** k=2 + r=2 ≡ standard spectral-bisection ≡ Fiedler-vector sign-pattern.

### T5 — Hypergraph partitioning (multilevel)

16. **`graph/hyper/partition/coarsen.go::HeavyEdgeCoarsen(*Hypergraph) (*Hypergraph, contractionMap)` (~180 LOC).** Karypis-1999-hMETIS coarsening: greedy heavy-hyperedge matching, vertex-pair-merge into super-vertex. Iterate until `|V| < threshold`.
17. **`graph/hyper/partition/recursive_bisect.go::RecursiveBisect(h *Hypergraph, k int, eps float64) (parts []int)` (~250 LOC).** Coarsen → spectral-bisect on coarsest (T1.7) → uncoarsen with FM-style move-gain refinement. Standard multilevel paradigm.

### T6 — Simplicial-complex chain-complex + Hodge Laplacian (composition with slot 171 / 241 / 246)

18. **`graph/hyper/simplicial.go::AsSimplicialComplex(*Hypergraph) *SimplicialComplex` (~120 LOC).** Bridge: a hypergraph + downward-closure → abstract simplicial complex (every subset of every hyperedge is a face). Define `SimplicialComplex` minimally here (or in `topology/simplicial/` if 171 has shipped).
19. **`graph/hyper/simplicial.go::Boundary(sc *SimplicialComplex, k int) (rows, cols []int, vals []float64)` (~80 LOC).** Sparse `∂_k : C_k → C_{k−1}` with `∂_k σ = ∑_i (−1)^i σ\{v_i}` over k-faces in lex-canonical orientation.
20. **`graph/hyper/simplicial.go::HodgeLaplacian(sc *SimplicialComplex, k int) (denseL []float64, n int)` (~70 LOC).** `L_k = ∂_k^T ∂_k + ∂_{k+1} ∂_{k+1}^T`. **R-MUTUAL-CROSS-VALIDATION 3/3 pin:** `L_0 = ∂_1 ∂_1^T` ≡ standard graph Laplacian of the 1-skeleton (regression to graph case).

### T7 — Hypergraph s-t min-cut (composition with slot 254)

21. **`graph/hyper/cut.go::HypergraphMinSTCut(h *Hypergraph, s, t int) (cutValue float64, S map[int]bool) (~150 LOC, BLOCKS on slot 254 max-flow).** Lawler-1973 reduction: build star-expansion bipartite graph `(V ∪ E*, edges)` with `v→e*` capacity `+∞`, `e*→v` capacity `+∞`, plus a dummy `e**` per hyperedge with `e**→all-vertices-in-e` capacity `w_e`; run binary max-flow → min-cut on this auxiliary graph. Reduces hypergraph s-t cut to slot 254's bipartite max-flow back-end.

### T8 — Hypergraph motifs (Lotito-2022)

22. **`graph/hyper/motifs.go::ExactMotifs3(h *Hypergraph) map[string]int (~250 LOC).** Lotito-Musciotto-Montresor-Battiston-2022 *Commun-Phys* 5:79 exact 3-node-higher-order motif enumeration (12 canonical motifs for size-3 sub-hypergraphs). Canonicalize by (vertex-orbits + hyperedge-orbits) signature.
23. **`graph/hyper/motifs.go::SampledMotifs(h *Hypergraph, k, samples int) map[string]int (~180 LOC).** Lotito-Musciotto-2023 *Computing* 105 sampling estimator for motifs of size k > 3 (exact intractable).

### T9 — Hypergraph SBM inference (composition with slot 280)

24. **`graph/hyper/hsbm.go::FitHSBM(h *Hypergraph, k int) (labels []int, B [][]float64) (~280 LOC, BLOCKS on T3.12 TensorPowerIter).** Ghoshdastidar-Dukkipati-2017 *JMLR* 18:50 + Lin-Chien-Ying-2025 *AISTATS* projected-tensor-power-method for Hypergraph-SBM community recovery on r-uniform hypergraphs.

## Cross-cutting

- **Slot 097-T1 (linalg-missing) ←** every spectral hypergraph primitive (T1.7, T4.15) blocks here. **Hard-pin:** ship 097-T1 (~80 LOC) before T1.7.
- **Slot 171 (synergy-graph-topology) ←** T6 (simplicial-complex / Hodge-Laplacian) **shares machinery**. If 171 ships first, T6 imports `topology/simplicial/`; if 282 ships first, T6 defines minimal complex and 171 reuses.
- **Slot 241 / 246 (discrete-exterior-calculus) ←** T6.20 Hodge-Laplacian is the same operator from the network angle vs the FEM angle; future-work consolidation.
- **Slot 254 (graph-cuts / max-flow) ←** T7.21 hypergraph s-t cut reduces to slot 254's bipartite max-flow back-end (Lawler-1973 star-expansion).
- **Slot 257 (tensor-decomposition) ←** T3.11 AdjacencyTensor + T3.12 TensorPowerIter feed slot 257's tensor primitives; reciprocally, slot 257's CP/Tucker can decompose hypergraph tensor for community recovery (Ghoshdastidar-Dukkipati-2017).
- **Slot 271 (spectral-clustering) ←** T4.15 SpectralKWay reuses 271's k-means back-end.
- **Slot 273 (spectral-embedding) ←** hypergraph-ASE downstream of 273 + 282 T1 normalized-Laplacian.
- **Slot 280 (SBM-inference) ←** T9.24 Hypergraph-SBM extends slot 280's binary-SBM-inference axis to hyperedges.
- **Slot 281 (temporal-graphs) ←** future-work composition: temporal hypergraphs (Cencetti-Battiston-2023). 281 = timestamp axis, 282 = arity axis.

## Single cheapest day-1 PR (one-shot recommendation)

**`graph/hyper/` package, T0 + T1.4–6 only, ~380 LOC, zero blockers.**

Files:
- `graph/hyper/types.go` — Hypergraph + Validate (T0.1, ~110 LOC)
- `graph/hyper/expand.go` — CliqueExpand + StarExpand (T0.2 + T0.3, ~120 LOC)
- `graph/hyper/laplacian.go` — Normalized + Unnormalized Laplacian (T1.4 + T1.5, ~150 LOC)
- `graph/hyper/cut.go` — HypergraphCut + Conductance (T1.6, ~60 LOC)

Tests (regression-to-graph-case):
- `TestNormalizedLaplacian_BinaryEquivRegression` — for r=2 hypergraph, `NormalizedLaplacian` ≡ graph normalized Laplacian (1e-12).
- `TestRandomWalkStationary_BinaryEquivRegression` (T2.9 if shipping with day-1) — `Stationary` ≡ `d_v / 2|E|` (1e-12).
- `TestCliqueExpand_LaplacianSpectralEquiv` — non-zero spectrum of clique-expansion graph-Laplacian ≡ non-zero spectrum of Zhou hypergraph-Laplacian (proves Zhou-2006 equivalence; **R-MUTUAL-CROSS-VALIDATION 3/3 pin**).
- Golden-file vectors (≥20 per fn): toy 4-vertex 2-hyperedge example + Zachary-karate-as-2-uniform-hypergraph.

This PR ships hypergraphs as a first-class data type, two equivalent bridge expansions to existing binary-graph algorithms (free reuse of Dijkstra / PageRank / Louvain via clique expansion), and the canonical Zhou-2006 normalized Laplacian without taking on the 097-T1 Eigvec dependency. Spectral methods (T1.7, T4) gate on 097-T1 and ship in a follow-up PR.

## Sources

- `graph/graph.go:14`, `graph/types.go:7`, `graph/community.go:155`, `graph/centrality.go`, `graph/pagerank.go`, `graph/flow.go`, `linalg/eigen.go:20`, `linalg/pca.go:101-174`, `topology/persistent/{vr,barcode}.go`
- Reviews: `reviews/overnight-400/agents/097-linalg-missing.md` (T1 Eigvec blocker), `agents/280-new-sbm.md`, `agents/281-new-temporal-graphs.md`, `agents/254` slot, `agents/271-new-spectral-clustering.md`, `agents/257-new-tensor-decomp.md`
- Zhou-Huang-Schölkopf-2006 *NeurIPS* 19:1601 "Learning with Hypergraphs" https://proceedings.neurips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf
- Hein-Setzer-Jost-Rangapuram-2013 *NeurIPS* 26 "The Total Variation on Hypergraphs" arXiv:1312.5179
- Louis-2015 *STOC* 47:713 "Hypergraph Markov Operators, Eigenvalues" arXiv:1408.2425
- Chan-Louis-Tang-Zhang-2018 *J-ACM* 65:15 "Spectral Properties of Hypergraph Laplacian" arXiv:1605.01483
- Chan-Liang-2018 *IPCO* "Generalizing the Hypergraph Laplacian via a Diffusion Process with Mediators"
- Cooper-Dutle-2012 *Linear-Alg-Appl* 436:3268 "Spectra of Uniform Hypergraphs"
- Carletti-Battiston-Cencetti-Fanelli-2020 *PRE* 101:022308 "Random Walks on Hypergraphs" arXiv:1911.06523
- Karypis-Aggarwal-Kumar-Shekhar-1999 *IEEE-TVLSI* 7:69 hMETIS multilevel hypergraph partitioning
- Lotito-Musciotto-Montresor-Battiston-2022 *Commun-Phys* 5:79 "Higher-order motif analysis in hypergraphs" arXiv:2108.03192
- Ghoshdastidar-Dukkipati-2017 *Ann-Stat* 45:289 + *JMLR* 18:50 Hypergraph-SBM consistency / tensor methods
- Bick-Gross-Harrington-Schaub-2023 *SIAM-Rev* 65:686 "What Are Higher-Order Networks?"
- Benson-Gleich-Leskovec-2016 *Science* 353:163 "Higher-order organization of complex networks"
- Aksoy-Joslyn-Marrero-Praggastis-Purvine-2020 *J-Compl-Net* 8:cnaa018 hypernetwork s-walks
- Lawler-1973 *Networks* 3:265 hypergraph min-cut via star-expansion
- Kolda-Mayo-2011 *SISC* 32:1095 shifted symmetric higher-order power method
