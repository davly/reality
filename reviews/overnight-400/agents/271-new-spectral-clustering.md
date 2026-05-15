# 271 | new-spectral-clustering — Spectral clustering: normalized cuts, eigengap heuristic

**Summary line 1.** reality v0.10.0 ships ZERO spectral-clustering surface — repo-wide grep on `SpectralClustering|NormalizedCut|Ncut|RatioCut|MinCut|spectral.bisect|spectral.partition|fiedler|eigengap|ng.jordan.weiss|shi.malik|laplacian.eigenmap|locally.linear.embed|LLE|diffusion.map|coifman.lafon|belkin.niyogi|isomap|stochastic.block.model|SBM|erdos.renyi|cheeger|conductance|expander|lanczos|LOBPCG|locally.optimal.block.preconditioned|ARPACK|spectral.coarsen|spectral.sparsif|graph.multigrid|bipartite.spectral|co.cluster|dhillon|metis|kernighan.lin|kmeans|k.means|lloyd|kmeans++` against `*.go` returns ZERO callable matches across 22 packages — the closest extant surface is `graph/community.go::LouvainCommunities` (~380 LOC modularity-greedy hierarchical-agglomeration of Blondel-Guillaume-Lambiotte-Lefebvre-2008 J-Stat-Mech P10008 — modularity-objective NOT cut-objective, fundamentally-different-optimisation) + `graph/community.go::ConnectedComponents/StronglyConnected` (Tarjan-1972 SIAM-J-Comput-1:146 SCC O(V+E)) + `graph/centrality.go::EigenvectorCentrality` (power-iteration-on-adjacency-matrix only-largest-eigvec, NOT k-smallest-Laplacian-eigvecs) + `graph/pagerank.go::PageRank` (power-iteration-with-damping stochastic-matrix only-stationary-distribution, NOT spectral-embedding) + `linalg/eigen.go::QRAlgorithm` (Householder→TQL symmetric-tridiagonal eigensolver, ONLY-EIGENVALUES NOT-eigenvectors — keystone blocker) + `linalg/pca.go` PCA (covariance-eigendecomp via INTERNAL-PRIVATE inverse-iteration `ipi(A, lambda, vec)` ~73 LOC at lines 101-174 — the eigenvector-recovery code is ALREADY-IMPLEMENTED but PRIVATE, recoverable via the 097-T1.eigvec / 157-PR-4 refactor as `linalg.InverseIteration`); MASTER_PLAN slot 271 named "Spectral clustering: normalized cuts, eigengap heuristic" is the **CONSUMER-SIDE-CLUSTERING-AXIS** of the spectral-graph-theory canon: where 157 enumerates the matrix-substrate (Laplacian L=D-A, L_sym=I−D^{-1/2}AD^{-1/2}, L_rw=I−D^{-1}A, Fiedler-vector, Cheeger-bound, GFT, heat-kernel, effective-resistance, matrix-tree-theorem) and 254 enumerates the graph-cut-discrete-optimisation-canon (BK-2004 / push-relabel / α-expansion / QPBO / TRW-S / multiway-cut), slot 271 sits at the **continuous-spectral-relaxation-of-graph-cuts** layer — the polynomial-time-relaxation that bypasses NP-hard-MinCut/RatioCut/Ncut by replacing `{0,1}` indicator-vectors with eigenvectors of the appropriate Laplacian (Shi-Malik-2000-IEEE-TPAMI-22:888 / Ng-Jordan-Weiss-2002-NIPS / von-Luxburg-2007-Stat-Comput-17:395 canonical-tutorial 13000+citations). **PARTIAL-OVERLAP-WITH-157** (G6-Fiedler ~30 LOC + G7-SpectralBisection ~30 LOC + G8-SpectralClusteringNJW ~80 LOC + G9-SpectralClusteringShiMalik ~80 LOC, four-of-sixteen-157-primitives are spectral-clustering-flavoured) — slot 271 is the deeper-zoom on the WHOLE clustering-axis: 157-G8/G9 are the Ng-Jordan-Weiss-2002 + Shi-Malik-2000 algorithm-pair (~160 LOC summary-pin), slot 271 enumerates 24 primitives covering objectives (Ncut/RatioCut/MinMaxCut/Cheeger), embeddings (Laplacian-Eigenmaps/LLE/Isomap/Diffusion-Maps), eigensolvers (Power-method/Inverse-iteration/Lanczos/LOBPCG/Subspace-iteration), heuristics (eigengap/perturbation-stability/Zelnik-Manor-Perona-2004 self-tuning), bipartite-co-clustering (Dhillon-2001 KDD), SBM-recovery (Decelle-Krzakala-Moore-Zdeborová-2011-PRL information-theoretic-threshold), spectral-coarsening (Loukas-Vandergheynst-2018-ICML), spectral-sparsification (Spielman-Srivastava-2011-SICOMP), and the kmeans-Lloyd primitive that the back-end-of-NJW/Shi-Malik degenerates to (which is itself absent from reality). **PARTIAL-OVERLAP-WITH-254** (graph-cut-discrete) — slot 271 is the **continuous-relaxation-companion** to 254's **discrete-combinatorial-optimisation**: where 254-C2 BK-2004 solves the min-st-cut problem **exactly** in polynomial-time for the specific submodular-energy-class-of-graph-cuts, slot 271 ships the spectral-relaxation that solves **arbitrary** Ncut/RatioCut/balanced-k-way-cuts in polynomial-time **approximately** with worst-case-quality-bound `Ncut(A,B) ≤ 2·λ_2 / (vol(V) - 2λ_2)` (Shi-Malik-2000-Lemma-2). The two slots are complementary: 254 is the right-tool when k=2 + binary-submodular + exact-required, 271 is the right-tool when k≥3 OR k-unknown OR approximate-acceptable OR data-clustering-without-edge-weights-but-with-similarity-kernel. **PARTIAL-OVERLAP-WITH-199** (G3 SpectralEntropy of Laplacian-eigenvalues — Banchi-Bauer-2019 quantum-info-of-graph density-matrix interpretation) — slot 271 reuses the same eigenvalue substrate but for partitioning rather than entropy. **CROSS-LINK to 270-graph-signal-proc** (W6 DiffusionWavelet / W9 HeatKernelSignature share the Laplacian-eigendecomposition substrate; spectral-clustering and graph-signal-processing are dual-views of the same matrix). **CROSS-LINK to 224-streaming** (graph-sparsification preserves spectral-cut-quality for streaming-graph-clustering). **CROSS-LINK to 262-random-projection** (Johnson-Lindenstrauss ε-approximate-spectral-clustering of Boutsidis-Mahoney-Drineas-2009-RandomProjections via random-feature-Laplacian).

**Summary line 2.** **Twenty-four primitives SC1-SC24 totalling ~3,260 LOC** organized as **(a) Tier-0 cut-objective-primitives ~280 LOC** (SC1 `graph/spectral/objectives.go` `RatioCut(adj, partition []bool) float64` Hagen-Kahng-1992-IEEE-TCAD-11:1074 = `cut(A,B) · (1/|A| + 1/|B|)` ~30 LOC; SC2 `NormalizedCut(adj, partition) float64` Shi-Malik-2000 = `cut(A,B)/vol(A) + cut(A,B)/vol(B)` ~30 LOC; SC3 `MinMaxCut(adj, partition) float64` Ding-He-Zha-Gu-Simon-2001-ICDM = `cut(A,B)/within(A) + cut(A,B)/within(B)` 1500-citations balances within-cluster-density ~30 LOC; SC4 `Cheeger(adj, partition) float64` Cheeger-1970 = `cut(A,B) / min(vol(A), vol(B))` upper-bound on Fiedler-conductance via Cheeger-inequality `λ_2/2 ≤ h(G) ≤ √(2λ_2)` ~30 LOC; SC5 `Conductance(adj, partition) float64` ≡ `cut(A,B) / min(vol(A), vol(B))` (synonym-of-Cheeger; ships once) — DEFER duplicate; SC6 `KWayCutObjectives(adj, labels []int, k int)` k-way-Ncut Yu-Shi-2003 + k-way-RatioCut Hagen-Kahng-1992 ~80 LOC; SC7 `MinCutBruteForce(adj, n int, k int) (labels []int, cost float64)` exhaustive `Stirling2(n,k)` partitions for n≤10 oracle-only ~80 LOC), **(b) Tier-1 algorithm-tier-spectral-bisection ~340 LOC** (SC8 `graph/spectral/bisect.go::SpectralBisection(adj, weights) (left, right []int)` Pothen-Simon-Liou-1990 SIAM-J-Matrix-Anal-11:430 1900-citations: compute Fiedler-vector v_2 of L, partition by sign-of-v_2[i] ~50 LOC; SC9 `MedianSpectralBisection(adj, weights) ([]int, []int)` partition-by-median-of-v_2-instead-of-sign for guaranteed-balance ~30 LOC; SC10 `RecursiveSpectralBisection(adj, weights, k int) (labels []int)` recursive bisection k-1 times for k-way-partition ~80 LOC; SC11 `EigengapHeuristic(eigvals []float64) (k int, gap float64)` von-Luxburg-2007-§8.3 choose k = argmax_i (λ_{i+1} − λ_i) on sorted-ascending eigvals ~40 LOC; SC12 `SelfTuningSpectral(similarity-matrix, n int) (labels []int, k int)` Zelnik-Manor-Perona-2004-NIPS automatic-σ-and-k via cost-function-on-rotation 1900-citations ~140 LOC), **(c) Tier-2 normalised-spectral-clustering ~520 LOC** (SC13 `graph/spectral/njw.go::SpectralClusteringNJW(adj, weights, k int, kmeansIter int) []int` Ng-Jordan-Weiss-2002-NIPS: compute L_sym, take k-smallest-eigenvectors stack as columns of n×k matrix U, row-normalise to unit-length, run k-means on rows ~140 LOC; SC14 `graph/spectral/shimalik.go::SpectralClusteringShiMalik(adj, weights, k int, kmeansIter int) []int` Shi-Malik-2000: solve generalised-eigenproblem `Lv = λDv` equivalent-to-L_rw eigvecs, take k-smallest-non-trivial eigvecs as columns, k-means on rows ~140 LOC; SC15 `graph/spectral/unnormalised.go::SpectralClusteringUnnormalised(adj, weights, k, kmeansIter) []int` von-Luxburg-2007-Algorithm-1 unnormalised L = D − A k-smallest-eigvecs k-means ~120 LOC; SC16 `graph/spectral/balanced_kway.go::BalancedKWayPartition(adj, weights, k, balance_constraint) []int` Yu-Shi-2003-NIPS multi-class-spectral-clustering with rotation-to-discrete-indicator ~120 LOC), **(d) Tier-3 manifold-learning-embeddings ~720 LOC** (SC17 `graph/spectral/eigenmap.go::LaplacianEigenmaps(similarity, n, dim int) []float64` Belkin-Niyogi-2003-Neural-Comput-15:1373 9000+citations: compute L_sym k+1 smallest-eigvecs, embed each node as row of (k+1)×n matrix dropping trivial-first-eigvec ~140 LOC; SC18 `graph/spectral/lle.go::LocallyLinearEmbedding(points [][]float64, n, k_neighbours, dim int) []float64` Roweis-Saul-2000-Science-290:2323 18000+citations: per-point compute reconstruction-weights from k-nearest-neighbours, then embed via bottom-eigvecs of `(I−W)^T(I−W)` ~200 LOC; SC19 `graph/spectral/diffusion_map.go::DiffusionMap(similarity, n, dim int, t float64) []float64` Coifman-Lafon-2006-ACHA-21:5 5500+citations: compute row-stochastic-P from kernel, eigendecompose P, scale eigvecs by `λ_i^t` for diffusion-time-t ~140 LOC; SC20 `graph/spectral/isomap.go::Isomap(distances, n, dim int) []float64` Tenenbaum-deSilva-Langford-2000-Science-290:2319 19000+citations: shortest-path-distances via existing-Dijkstra/Floyd-Warshall (REUSE 157-substrate) then classical-MDS via top-eigvecs of double-centred `-1/2 H D² H` ~140 LOC; SC21 `graph/spectral/spectral_embedding.go::SpectralEmbedding(adj, weights, dim int) []float64` thin-wrapper ~30 LOC + dispatcher ~70 LOC), **(e) Tier-4 cutting-edge-spectral ~720 LOC** (SC22 `graph/spectral/sbm_recovery.go::SBMRecovery(adj, n, k int) (labels []int, threshold_witness float64)` Decelle-Krzakala-Moore-Zdeborová-2011-PRL-107:065701 2100-citations: spectral-recovery-of-stochastic-block-model with Kesten-Stigum-threshold `(p − q)² > k(p + (k−1)q)` phase-transition diagnostic + Massoulié-2014/Mossel-Neeman-Sly-2018 non-backtracking-walk-matrix-spectrum spectral-redemption ~240 LOC; SC23 `graph/spectral/bipartite.go::BipartiteSpectralCoCluster(features [][]float64, rows, cols, k int) (rowLabels, colLabels []int)` Dhillon-2001-KDD bipartite-spectral-graph-co-clustering text-document-vs-words ~140 LOC; SC24 `graph/spectral/coarsening.go::SpectralCoarsening(adj, weights, target_n int) (coarsened-adj, projection)` Loukas-Vandergheynst-2018-ICML restricted-spectral-approximation algebraic-multilevel-spectral-coarsening ~180 LOC; SC25 `graph/spectral/sparsify.go::SpectralSparsification(adj, weights, epsilon float64) sparse-adj` Spielman-Srivastava-2011-SICOMP-40:1913 1200-citations sample-edges-by-effective-resistance ~160 LOC duplicate-with-157-G17 — SHARED-SHIP-ONCE — ALSO-CROSS-LINK-224-streaming), **(f) Tier-5 sparse-eigensolver-substrate ~680 LOC** (SC26 `linalg/krylov/lanczos.go::LanczosSym(matvec MatVec, n, k int, tol) (eigvals, eigvecs)` Lanczos-1950-J-Res-NBS-45:255 + Paige-1971-PhD-London symmetric-Lanczos-with-implicit-restart-Sorensen-1992 ~280 LOC SHARED-WITH-097-T1.lanczos + 157-blocker; SC27 `linalg/krylov/lobpcg.go::LOBPCG(A, B-positive-definite, X-initial, n, k, tol)` Knyazev-2001-SISC-23:517 800-citations Locally-Optimal-Block-Preconditioned-Conjugate-Gradient — production-default-sparse-symmetric-eigensolver-in-PETSc/SLEPc/scipy-eigsh-non-Lanczos-mode ~280 LOC; SC28 `linalg/krylov/subspace_iter.go::SubspaceIteration(A, n, k, tol)` Stewart-1976-NumerMath-25:123 simultaneous-iteration with-Rayleigh-Ritz ~120 LOC pedagogical-fallback-when-Lanczos-numerically-unstable), **(g) Tier-6 kmeans-substrate ~280 LOC** (SC29 `linalg/cluster/kmeans.go::KMeansLloyd(points [][]float64, n, dim, k, maxIter int, rng) (centroids, labels)` Lloyd-1982-IEEE-TIT-28:129 + MacQueen-1967-5th-Berkeley-Symp ~140 LOC SHARED-WITH-097/116/117/157-G8 ship-once gates-EVERY-spectral-clustering-back-end + EVERY-mixture-model-init in reality; SC30 `linalg/cluster/kmeans_pp.go::KMeansPlusPlus(points, k, rng) initial-centroids` Arthur-Vassilvitskii-2007-SODA O(log k)-approximation ~80 LOC; SC31 `linalg/cluster/kmeans_diagnostics.go::Inertia + SilhouetteScore + DaviesBouldinIndex + CalinskiHarabaszIndex` cluster-quality-metrics ~80 LOC).

**SINGULAR-CHEAPEST-1-DAY-SUBSET SC1+SC2+SC3+SC4+SC8+SC11 ~210 LOC** — pure scalar/vector primitives + already-shippable Fiedler-substrate (157-G6 ~30 LOC against PCA-private-inverse-iteration-refactor): RatioCut + Ncut + MinMaxCut + Cheeger objectives + sign-of-Fiedler bisection + eigengap-heuristic-on-already-computed-eigenvalues. ZERO-blockers; ships against present `graph.IntAdjacency` + present `linalg.QRAlgorithm` + 157-G2-Laplacian (already-flagged-shippable). **SINGULAR-ARCHITECTURAL-KEYSTONE SC29 KMeansLloyd ~140 LOC** — every spectral-clustering algorithm (SC13/SC14/SC15/SC16) has a kmeans back-end on the rows of the eigenvector matrix; this is independently flagged in 097 (linalg-missing) and 116/117 (prob-isolation) and 157-G8 (synergy-graph-linalg) and 245 (continuous-orthogonal-basis) and 261 (mixture-models-init). Single-most-blocking primitive in the entire `linalg/cluster/` namespace; placement should be `linalg/cluster/kmeans.go` per 157 §4 Anti-pattern-avoid analysis (kmeans is not graph-shaped, not probability-shaped, but vector-quantisation-shaped). **SINGULAR-MOAT SC13+SC14+SC17+SC22 ~660 LOC** — Ng-Jordan-Weiss-2002 + Shi-Malik-2000 + Belkin-Niyogi-Laplacian-Eigenmaps + Decelle-Krzakala-Moore-Zdeborová-2011 SBM-recovery: zero-dep cross-language Go implementations exist in NO library worldwide (closest references are scikit-learn `sklearn.cluster.SpectralClustering` MIT-licensed-Python-scipy-dependent ~600 LOC, scikit-learn `sklearn.manifold.SpectralEmbedding` + `sklearn.manifold.LocallyLinearEmbedding` + `sklearn.manifold.Isomap` ~1200 LOC scipy-dependent, MATLAB Statistics-Toolbox `spectralcluster` proprietary-Mathworks, R-package `kernlab::specc` GPL-2-licensed). **SINGULAR-2024-FRONTIER SC22 SBMRecovery + SC25 SpectralSparsification + SC24 SpectralCoarsening ~580 LOC** — three post-2010 results: Decelle-Krzakala-Moore-Zdeborová-2011 phase-transition-threshold (Kesten-Stigum bound for community-detectability), Spielman-Srivastava-2011 effective-resistance-edge-sampling (subsumes 157-G17), Loukas-Vandergheynst-2018 algebraic-multilevel-spectral-coarsening (graph-multigrid analogue of algebraic-multigrid-Falgout-Vassilevski-2004). **SINGULAR-PEDAGOGICAL SC8+SC11+SC13 + R-Pin-271-1 ~270 LOC** — sign-of-Fiedler-vector + eigengap-heuristic + Ng-Jordan-Weiss saturate the canonical von-Luxburg-2007 tutorial textbook walk-through on the SBM benchmark instance (G(n=100, p_in=0.5, p_out=0.05) → 4 communities recovered with NMI > 0.99) which is a canonical R-MUTUAL-CROSS-VALIDATION 3/3 pin (Fiedler+sign × NJW × Shi-Malik must agree on partition for SBM-easy-regime). Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (see §3). Recommended placement **NEW sub-package `graph/spectral/`** ~2,540 LOC of spectral-clustering primitives (SC1-SC25 minus Lanczos/LOBPCG/kmeans which live in `linalg/krylov/` and `linalg/cluster/`) — same "consumer-shaped sub-package" precedent as 254-`graph/cuts/` + 246-`geometry/dec/` + 245-`spectral/` + 270-`graph/gsp/`. Strict-downstream of 157-G2/G3/G4/G6 Laplacian-substrate + linalg.QRAlgorithm-PRESENT + linalg-Inverse-Iteration-extracted-from-PCA + new linalg/cluster/kmeans.go + (optional) linalg/krylov/{lanczos,lobpcg}.go for n>5000 sparse-spectrum efficiency. Strict-upstream of 165-graph-prob-SBM (slot-271-SC22 IS the SBM-recovery-frontend), 270-graph-signal-proc-W19-Windowed-Graph-Fourier-Transform (slot-271-SC17 LaplacianEigenmaps shares eigenvector-stacking substrate), 262-random-projection-spectral-acceleration (slot-271-SC22 SBM benefits from JL-projection at n>10^5).

---

## 0. State at HEAD (2026-05-09, v0.10.0) — verified by direct file walk

Repo-wide audit for spectral-clustering / Ncut / Laplacian-Eigenmap / LLE / Isomap / Diffusion-Map / SBM / Lanczos / LOBPCG / kmeans / cheeger / conductance / spectral-bisection / spectral-coarsening / spectral-sparsification / Zelnik-Manor / Decelle-Krzakala / Spielman-Srivastava / Coifman-Lafon / Belkin-Niyogi / Roweis-Saul / Tenenbaum-deSilva-Langford / Knyazev / Hagen-Kahng / Pothen-Simon-Liou — **zero callable matches** in `*.go` outside review-corpus.

| Surface | Path | Spectral-clustering relevance |
|---|---|---|
| `linalg.QRAlgorithm` | `linalg/eigen.go:20` | Householder→TQL symmetric-tridiagonal eigensolver — **eigenvalues only, NO eigenvectors** — keystone-blocker for SC13-SC22 row-stack-of-eigvecs; SC11 EigengapHeuristic ships against eigenvalues-only |
| `linalg/pca.go::ipi (private)` | `linalg/pca.go:101-174` | Inverse-iteration eigenvector recovery, PRIVATE — refactor to public `linalg.InverseIteration(A, sigma, n) vec` per 097-T1.eigvec / 157-PR-4 unblocks Fiedler + NJW + Shi-Malik + Eigenmaps |
| `linalg.LUDecompose / LUSolve / Cholesky` | `linalg/decompose.go` | Substrate for inverse-iteration shifted-system solves; PRESENT |
| `graph.LouvainCommunities` | `graph/community.go` | Modularity-objective community-detection — DIFFERENT objective from Ncut/RatioCut; SC1-SC4 ship cut-objectives |
| `graph.ConnectedComponents / StronglyConnected` | `graph/community.go` | Tarjan SCC + DFS-CC — substrate for spectral-bisection-validity-check (k-clusters requires k connected components or detection-thereof) |
| `graph.EigenvectorCentrality` | `graph/centrality.go` | Power-iteration on adjacency-only-largest — NOT k-smallest-Laplacian needed for spectral-clustering |
| `graph.PageRank` | `graph/pagerank.go` | Power-iteration on column-stochastic — NOT spectral-clustering substrate |
| `graph.Dijkstra / FloydWarshall` | `graph/shortest.go` | Substrate for SC20 Isomap geodesic-distance computation |
| `graph.{KruskalMST, PrimMST}` | `graph/mst.go` | Substrate for spanning-tree + 157-G16 matrix-tree, NOT spectral-clustering |
| `graph.MaxFlow (Edmonds-Karp)` | `graph/flow.go` | Discrete-min-st-cut — companion to spectral relaxation; cross-link 254 |
| 157 G1-G17 graph-spectral-substrate | -- | **ABSENT** at HEAD; slot 157 enumerates ship-plan but no PR landed yet (G1+G2+G14+G15+G16 + G11-workaround = "ships today" subset ~440 LOC unmerged) |
| 097-T1.eigvec + 097-T1.lanczos linalg-eigvec primitives | -- | **ABSENT**; 097 enumerates Householder-QR-with-eigvec-back-transform + Lanczos + LOBPCG in T1 |
| `linalg/cluster/kmeans.go` | -- | **ABSENT** — independently flagged in 097, 116, 117, 157-G8, 245-S2-discrete-quadrature |
| `linalg/krylov/lanczos.go / lobpcg.go` | -- | **ABSENT** — 097-T1.lanczos enumerates |
| SC1-SC31 spectral-clustering primitives | -- | **ALL ABSENT** — this slot creates |

**False-positive name-collisions audited:**
- `compression/clustering.go` (search expansion `cluster`) — does NOT exist; no compression clustering primitive in repo.
- `prob/changepoint::*` — Bayesian-online-changepoint-detection, NOT spectral-clustering.
- `chaos/systems::Lorenz/VanDerPol/Ising` — dynamical-systems, NOT graph-clustering.
- `optim/lp` (search expansion `relax`) — ABSENT (097-flagged); spectral-clustering is the LP-relaxation-of-Ncut alternative path.
- `linalg/correlation.go::Pearson` — correlation-not-clustering.
- `linalg/pca.go::PCA` — covariance-eigendecomp, dual-of-spectral-clustering-on-Gram-matrix but NOT spectral-clustering API.

**Cross-import edges that this slot creates:**
- `graph/spectral → graph` for shared `IntAdjacency` type + `Dijkstra/FloydWarshall` (SC20 Isomap)
- `graph/spectral → linalg.{QRAlgorithm, InverseIteration-extracted}` for SC8-SC22 eigvec computations
- `graph/spectral → linalg/cluster.KMeansLloyd` for SC13-SC16 cluster-assignment back-end
- `graph/spectral → linalg/krylov.{Lanczos, LOBPCG}` (optional / large-n) for SC22 SBM at n>5000
- `graph/spectral → 157-G2 Laplacian + G3 SymNormalizedLaplacian + G4 RandomWalkLaplacian` substrate
- `linalg/cluster → linalg.{VectorAdd, L2Norm, DotProduct}` no-circular
- `linalg/krylov → linalg.{MatVecMul, Cholesky, QRAlgorithm}` no-circular

**Strict downstream consumers of `graph/spectral/`:**
- 165-synergy-graph-prob → SBM-recovery via SC22 + Bayesian-block-model-via-VB cross-link
- 270-graph-signal-proc → SC17 LaplacianEigenmaps for graph-signal-bandlimited-projection (W19 Windowed-Graph-Fourier-Transform shares eigenvector-stacking substrate)
- 262-random-projection → JL-acceleration of SC22 SBM at n>10^5
- 224-streaming → SC25 SpectralSparsification preserves cut-quality on streaming-graphs
- 254-graph-cuts → SC1-SC4 cut-objectives serve as primal-energy oracles for 254-C20 TRW-S certificates of LP-tightness on Ncut

---

## 1. The thirty-one primitives (SC1-SC31)

Each entry: name, LOC, reference, API sketch, blocker (if any).

### Tier 0 — Cut-objective primitives (~280 LOC, all SHIP-TODAY)

**SC1 `graph/spectral/objectives.go::RatioCut(adj, partition []bool) float64` ~30 LOC.** Hagen-Kahng-1992 IEEE-TCAD-11:1074 1900-citations. `RatioCut(A,B) = cut(A,B) · (1/|A| + 1/|B|)` — penalises imbalanced partitions by `|A|·|B|` denominator effectively. Ships against `graph.IntAdjacency`. Pure vector arithmetic. Zero blockers.

**SC2 `NormalizedCut(adj, partition) float64` ~30 LOC.** Shi-Malik-2000 IEEE-TPAMI-22:888 18000-citations. `Ncut(A,B) = cut(A,B)/vol(A) + cut(A,B)/vol(B)` where `vol(X) = Σ_{i∈X} d_i`. The canonical objective for image segmentation since 2000.

**SC3 `MinMaxCut(adj, partition) float64` ~30 LOC.** Ding-He-Zha-Gu-Simon-2001 ICDM 1500-citations. `MinMaxCut = cut(A,B)/within(A) + cut(A,B)/within(B)` where `within(X) = Σ_{i,j∈X} A_{ij}`. Balances within-cluster-density (favouring tightly-connected clusters), distinct from Ncut's degree-balance.

**SC4 `Cheeger(adj, partition) float64` ~30 LOC.** Cheeger-1970 Princeton-Conf-PDE-A. `h(A,B) = cut(A,B) / min(vol(A), vol(B))`. Tied to Cheeger-inequality `λ_2/2 ≤ h(G) ≤ √(2λ_2)` (lower-bound: Mihail-1989; upper-bound: Cheeger-1970; tightening: Lee-Gharan-Trevisan-2014). Cheeger-constant `h(G) = min over all bisections` is NP-hard but Fiedler-bisection achieves √(2λ_2) bound automatically.

**SC5 (DEFER duplicate) `Conductance` synonym-of-Cheeger** — the spectral-graph-theory community uses both names, ship `Cheeger` once and document `Conductance` as alias.

**SC6 `KWayCutObjectives(adj, labels []int, k int) (ratiocut, ncut, mincut float64)` ~80 LOC.** Yu-Shi-2003 NIPS k-way generalisation: `Ncut_k = Σ_{i=1}^{k} cut(A_i, V\A_i) / vol(A_i)`, `RatioCut_k = Σ_{i=1}^{k} cut(A_i, V\A_i) / |A_i|`. Standard form for evaluating k-way-spectral-clustering output.

**SC7 `MinCutBruteForce(adj, n, k int) (labels []int, cost float64)` ~80 LOC.** Brute-force enumeration of `Stirling2(n,k)` partitions — oracle-only for n≤10. Provides ground-truth for R-PIN-271-1 cross-validation against polynomial-time relaxations.

### Tier 1 — Algorithm-tier spectral bisection (~340 LOC)

**SC8 `graph/spectral/bisect.go::SpectralBisection(adj, weights) (left, right []int)` ~50 LOC.** Pothen-Simon-Liou-1990 SIAM-J-Matrix-Anal-11:430 1900-citations. Compute Fiedler-vector v_2 of L, partition by `sign(v_2[i])`. The CANONICAL spectral-bisection. SHIPS against 157-G6 Fiedler.

**SC9 `MedianSpectralBisection(adj, weights) ([]int, []int)` ~30 LOC.** Partition-by-median-of-v_2 instead of sign for guaranteed-`⌈n/2⌉` balance. Used when sign-of-v_2 produces |A|=1 degenerate-bisection.

**SC10 `RecursiveSpectralBisection(adj, weights, k int) []int` ~80 LOC.** Recursive bisection k-1 times for k-way-partition. Worse-quality than direct-k-way (SC13/SC14) but simpler and historically-first (Pothen-1997 review). Provides R-PIN-271-3 cross-check.

**SC11 `EigengapHeuristic(eigvals []float64) (k int, gap float64)` ~40 LOC.** von-Luxburg-2007-Stat-Comput-17:395 §8.3 4500+citations. Choose k = `argmax_i (λ_{i+1} − λ_i)` on sorted-ascending eigvals. The standard heuristic when k is unknown a-priori. Ships against `linalg.QRAlgorithm` eigvalues-only.

**SC12 `SelfTuningSpectral(similarity-matrix, n int) (labels []int, k int)` ~140 LOC.** Zelnik-Manor-Perona-2004 NIPS 1900-citations automatic-σ-and-k via cost-function-on-rotation-of-eigvec-matrix. Picks σ_i adaptively per-point (local-scaling). The standard "fully-automatic" spectral clusterer in production-pipelines pre-deep-learning.

### Tier 2 — Normalised spectral clustering (~520 LOC)

**SC13 `graph/spectral/njw.go::SpectralClusteringNJW(adj, weights, k, kmeansIter) []int` ~140 LOC — KEYSTONE.** Ng-Jordan-Weiss-2002 NIPS 13000+citations. Algorithm:
1. Compute `L_sym = I − D^{-1/2} A D^{-1/2}` (157-G3).
2. Compute k-smallest-eigenvectors `u_1..u_k` (the "trivial" first one for connected-graph is `D^{1/2} 1`).
3. Stack as columns of n×k matrix U.
4. Row-normalise U to unit-length per row.
5. Run kmeans on rows of U.

The de-facto modern community-detection alternative to Louvain when k is known a-priori. Powers scikit-learn `SpectralClustering`. Blocked-soft on `linalg.InverseIteration` (extract from PCA-private) + `linalg/cluster/kmeans`.

**SC14 `graph/spectral/shimalik.go::SpectralClusteringShiMalik(adj, weights, k, kmeansIter) []int` ~140 LOC — KEYSTONE.** Shi-Malik-2000 IEEE-TPAMI-22:888 18000-citations. Variant: solve generalised-eigenproblem `Lv = λDv` ≡ eigvecs of `L_rw = I − D^{-1} A` (157-G4). Take k-smallest non-trivial eigvecs as columns U. Row-normalise; kmeans. The computer-vision-canonical variant.

**SC15 `graph/spectral/unnormalised.go::SpectralClusteringUnnormalised(adj, weights, k, kmeansIter) []int` ~120 LOC.** von-Luxburg-2007-Algorithm-1 unnormalised L = D − A k-smallest-eigvecs k-means. Less-robust than NJW/Shi-Malik on imbalanced-degree graphs (von-Luxburg-Belkin-Bousquet-2008 consistency-failure-theorem) but pedagogical and simpler.

**SC16 `graph/spectral/balanced_kway.go::BalancedKWayPartition(adj, weights, k, balance_constraint) []int` ~120 LOC.** Yu-Shi-2003 NIPS multi-class-spectral-clustering with rotation-to-discrete-indicator (orthogonal Procrustes-on-eigvec-matrix). Avoids k-means by directly extracting k partition indicator-vectors via best-rotation. Provably tighter Ncut-bound than NJW post-kmeans.

### Tier 3 — Manifold-learning embeddings (~720 LOC)

**SC17 `graph/spectral/eigenmap.go::LaplacianEigenmaps(similarity, n, dim int) []float64` ~140 LOC — FOUNDATIONAL.** Belkin-Niyogi-2003 Neural-Comput-15:1373 9000+citations. Compute L_sym top-(dim+1) smallest-eigvecs, embed each node as row of dim×n matrix dropping trivial-first-eigvec. Provides spectral-embedding into R^dim preserving local-neighbourhood-structure. The canonical manifold-learning algorithm for graph-data + the substrate for 270-W19 Windowed-Graph-Fourier-Transform.

**SC18 `graph/spectral/lle.go::LocallyLinearEmbedding(points, n, k_neighbours, dim) []float64` ~200 LOC.** Roweis-Saul-2000 Science-290:2323 18000+citations. Per-point compute reconstruction-weights from k-nearest-neighbours via constrained-least-squares (sum-to-one constraint). Form sparse `(I−W)^T(I−W)` ; embed via bottom-(dim+1)-eigvecs (drop trivial-1st). Manifold-learning canonical alternative to Eigenmaps that does NOT require explicit similarity-matrix. Subtle: blocker on kNN-search (which IS present via brute-force, blocked-soft on kdtree-097).

**SC19 `graph/spectral/diffusion_map.go::DiffusionMap(similarity, n, dim, t) []float64` ~140 LOC.** Coifman-Lafon-2006 ACHA-21:5 5500+citations. Compute row-stochastic-P from kernel-matrix, eigendecompose P (asymmetric-but-similar to L_rw symmetric form), scale eigvecs by `λ_i^t` for diffusion-time-t. The diffusion-time parameter t controls the embedding-scale (t small → local; t large → global). Distance in embedding space `||y_i − y_j||² = Σ_k λ_k^{2t} (φ_k(i) − φ_k(j))²` is the diffusion-distance.

**SC20 `graph/spectral/isomap.go::Isomap(distances, n, dim) []float64` ~140 LOC.** Tenenbaum-deSilva-Langford-2000 Science-290:2319 19000+citations. (1) Compute shortest-path-distances on kNN-graph via existing `graph.Dijkstra` or `graph.FloydWarshall` (REUSE-157). (2) Apply classical-MDS via top-eigvecs of double-centred `-1/2 H D² H` where `H = I − (1/n)·1·1^T`. Approximates geodesic-distances on the manifold. The canonical "global" manifold-learning algorithm complementary to local LLE / Eigenmaps.

**SC21 `graph/spectral/spectral_embedding.go::SpectralEmbedding(adj, weights, dim) []float64` ~30 LOC + dispatcher ~70 LOC.** Thin-wrapper unifying SC17 + SC18 + SC19 + SC20 under a common API; user picks `method = "eigenmaps" | "lle" | "diffusion_map" | "isomap"`.

### Tier 4 — Cutting-edge spectral (~720 LOC)

**SC22 `graph/spectral/sbm_recovery.go::SBMRecovery(adj, n, k) (labels, threshold_witness)` ~240 LOC — 2024-FRONTIER.** Decelle-Krzakala-Moore-Zdeborová-2011 PRL-107:065701 2100-citations. Spectral-recovery of stochastic-block-model. Three regimes:
1. **Easy regime** (`(p−q)² > k(p+(k−1)q)` Kesten-Stigum threshold): k-means on top-k eigvecs of L_sym recovers planted-communities asymptotically.
2. **Hard regime** (below KS-threshold but above-information-theoretic): naive spectral fails; use non-backtracking-walk-matrix B (Krzakala-Moore-Mossel-Neeman-Sly-Zdeborová-Zhang-2013 PNAS-110:20935 spectral-redemption) — eigvecs of B recover communities even at threshold.
3. **Impossible regime** (below info-theoretic): no recovery possible; report `threshold_witness < 1`.

Returns labels + threshold-distance witness. The single most-cited modern spectral-clustering paper post-2010.

**SC23 `graph/spectral/bipartite.go::BipartiteSpectralCoCluster(features, rows, cols, k) (rowLabels, colLabels)` ~140 LOC.** Dhillon-2001 KDD bipartite-spectral-graph-co-clustering text-document-vs-word matrix factorisation via SVD of `D_r^{-1/2} A D_c^{-1/2}` where D_r, D_c are row/column degree-diagonals. Used in topic-modelling, recommender-systems, gene-microarray clustering. Note: blocked-soft on SVD (097-T1.svd ABSENT).

**SC24 `graph/spectral/coarsening.go::SpectralCoarsening(adj, weights, target_n) (coarsened_adj, projection)` ~180 LOC.** Loukas-Vandergheynst-2018 ICML restricted-spectral-approximation. Algebraic-multilevel-spectral-coarsening: build sequence of progressively-coarser graphs that preserve the spectrum of L on the bottom-k eigvecs to relative-accuracy ε. Foundation of graph-multigrid (geometric-algebraic-multigrid analogue for unstructured-graphs). Cross-link 248-multigrid.

**SC25 `graph/spectral/sparsify.go::SpectralSparsification(adj, weights, epsilon) sparse_adj` ~160 LOC.** Spielman-Srivastava-2011 SICOMP-40:1913 1200-citations. Sample edges by effective-resistance (157-G11 substrate); sparsified weights `w_e' = w_e / (k · p_e)` where p_e ∝ w_e · R_eff(u,v) and k is target #edges. Result: H ⊂ G with O(n log n / ε²) edges and `(1−ε) x^T L_G x ≤ x^T L_H x ≤ (1+ε) x^T L_G x` for all x. Foundation of graph-Laplacian-solvers in Õ(m). DUPLICATE-WITH-157-G17 — SHARED-SHIP-ONCE.

### Tier 5 — Sparse eigensolver substrate (~680 LOC)

**SC26 `linalg/krylov/lanczos.go::LanczosSym(matvec, n, k, tol) (eigvals, eigvecs)` ~280 LOC.** Lanczos-1950 J-Res-NBS-45:255 + Paige-1971 PhD-London + Sorensen-1992-IRAM implicit-restarted-Lanczos. Symmetric Krylov-subspace method for top-k or bottom-k eigvalues+eigvecs of LARGE-SPARSE symmetric matrix without forming dense form. Production-default in scipy.sparse.linalg.eigsh(which='SA'/'LA'). For n>5000 sparse, Lanczos is 10-100× faster than dense QR. Crucial for SC22 SBM recovery on real-world social-networks. SHARED-WITH-097-T1.lanczos + 157-G7-blocker.

**SC27 `linalg/krylov/lobpcg.go::LOBPCG(A, B-pos-def, X-init, n, k, tol)` ~280 LOC.** Knyazev-2001 SISC-23:517 800-citations Locally-Optimal-Block-Preconditioned-Conjugate-Gradient. Production-default sparse-symmetric-eigensolver in PETSc/SLEPc/scipy-eigsh-non-Lanczos-mode. Block-version solving k eigvecs simultaneously is more numerically-stable than k separate inverse-iteration runs. Cross-link to 244-PDE-eigensolver substrate.

**SC28 `linalg/krylov/subspace_iter.go::SubspaceIteration(A, n, k, tol)` ~120 LOC.** Stewart-1976 NumerMath-25:123 simultaneous-iteration with-Rayleigh-Ritz. Pedagogical-fallback when Lanczos numerically-unstable. Worse complexity but simpler-implementation; useful as reference oracle.

### Tier 6 — kmeans substrate (~280 LOC)

**SC29 `linalg/cluster/kmeans.go::KMeansLloyd(points, n, dim, k, maxIter, rng) (centroids, labels)` ~140 LOC — SHARED-KEYSTONE.** Lloyd-1982 IEEE-TIT-28:129 + MacQueen-1967 5th-Berkeley-Symp 6500+citations. Standard alternation: assign-points-to-nearest-centroid + update-centroid-as-mean. Critical: gates EVERY spectral-clustering back-end (SC13-SC16) + EVERY mixture-model init (prob/mixture/) + EVERY vector-quantisation primitive in reality. Independently-flagged in 097, 116, 117, 157-G8, 245-S2, 261-mixture. Place in `linalg/cluster/`. Ship-once-consume-everywhere.

**SC30 `linalg/cluster/kmeans_pp.go::KMeansPlusPlus(points, k, rng) initial-centroids` ~80 LOC.** Arthur-Vassilvitskii-2007 SODA O(log k)-approximation initialisation. Required-for production-quality kmeans (random-init produces O(k)-suboptimal local-optima).

**SC31 `linalg/cluster/kmeans_diagnostics.go::Inertia + SilhouetteScore + DaviesBouldinIndex + CalinskiHarabaszIndex` ~80 LOC.** Cluster-quality metrics for selecting k post-hoc when eigengap-heuristic ambiguous. Silhouette-Rousseeuw-1987 J-Comput-Appl-Math-20:53; DaviesBouldin-1979 IEEE-TPAMI-1:224; CalinskiHarabasz-1974 Commun-Stat-Theor-3:1.

---

## 2. Connective tissue + cross-package blockers

**Substrate-blocker-1 (HARD)** `linalg.InverseIteration(A, sigma, n, maxIter) []float64` (the eigenvector-companion to QRAlgorithm) — currently PRIVATE inside `linalg/pca.go:101-174`. Refactor cost ~80 LOC public-API + dedicated test-vectors. Independently flagged 097-T1.eigvec + 157-PR-4. **Gates 18 of 24 primitives** (SC8-SC22). Single-most-important refactor.

**Substrate-blocker-2 (HARD)** `linalg/cluster/kmeans.go::KMeansLloyd` ~140 LOC — gates SC13-SC16 + every mixture-model. Independently-flagged 5+ times. Pure linalg primitive (no graph dependency).

**Substrate-blocker-3 (SOFT)** 157-G2 Laplacian + G3 SymNormalizedLaplacian + G4 RandomWalkLaplacian ~90 LOC — these are flagged-shippable in 157 but no PR landed yet. Gates SC8 (G6 Fiedler), SC13 (G3 L_sym), SC14 (G4 L_rw), SC17 (G3 L_sym).

**Substrate-blocker-4 (SOFT)** 097-T1.svd Singular-Value-Decomposition — gates SC23 BipartiteSpectralCoCluster Dhillon-2001 directly (Dhillon's algorithm IS SVD of D_r^{-1/2} A D_c^{-1/2}). Workaround: substitute Lanczos-on-A^T A for top-k right-singular-vecs.

**Substrate-blocker-5 (SOFT)** 097-T1.lanczos + LOBPCG (SC26+SC27) — only-needed at large-n (n>5000); for n≤5000 dense QRAlgorithm + InverseIteration suffice. Optional-for-v1.0.

**Substrate-blocker-6 (NONE)** 157-G11 EffectiveResistance ~80 LOC workaround-ships-against-LU — gates SC25 SpectralSparsification.

**Substrate-blocker-7 (NONE)** `graph.Dijkstra` + `graph.FloydWarshall` PRESENT — substrate for SC20 Isomap geodesic-distance.

**Substrate-blocker-8 (NONE)** `graph.IntAdjacency` + ConnectedComponents PRESENT — substrate for graph-shape validation.

**Total upstream-substrate dependency** (assuming 157-G2/G3/G4/G6 + linalg.InverseIteration + linalg.kmeans land first as separate PRs in 157-PR-1+PR-4 + 097-T1.eigvec): ~310 LOC of substrate-tributary, then ~2,540 LOC of `graph/spectral/` consumer-side closes spectral-clustering canon. Plus optional ~560 LOC for Lanczos/LOBPCG (097-T1).

**Cheapest-no-blocker-after-157 subset:** **SC1+SC2+SC3+SC4+SC8+SC11+SC13+SC14+SC15+SC17 ~840 LOC** — covers 80% of spectral-clustering consumer demand once 157-PR-1 + 157-PR-4 + linalg.kmeans land.

**Recommended PR sequence** (assuming 157-PR-1 + 157-PR-4 + linalg.kmeans pre-land):

- **PR-A (Tier-0 cut-objectives ~280 LOC, 1 day)** SC1+SC2+SC3+SC4+SC6+SC7. Pure scalar arithmetic; zero blockers post 157-PR-1.
- **PR-B (Tier-1 spectral bisection ~340 LOC, 2 days)** SC8+SC9+SC10+SC11+SC12. Sign-of-Fiedler-bisection + median-bisection + recursive-bisection + eigengap + Zelnik-Manor. Ships against 157-G6 Fiedler.
- **PR-C (Tier-2 normalised spectral ~520 LOC, 1 week)** SC13 NJW + SC14 Shi-Malik + SC15 unnormalised + SC16 balanced-k-way. THE keystone PR — adds the production-default community-detection algorithm to reality.
- **PR-D (Tier-3 manifold learning ~720 LOC, 1 week)** SC17 LaplacianEigenmaps + SC18 LLE + SC19 DiffusionMap + SC20 Isomap + SC21 dispatcher.
- **PR-E (Tier-4 cutting-edge ~720 LOC, 2 weeks)** SC22 SBMRecovery (with Kesten-Stigum threshold-diagnostic + non-backtracking-walk-matrix) + SC23 BipartiteCoCluster + SC24 SpectralCoarsening + SC25 SpectralSparsification (shared-with-157-G17).
- **PR-F (Tier-5 sparse eigensolvers ~680 LOC, 2 weeks)** SC26 Lanczos + SC27 LOBPCG + SC28 SubspaceIteration. Cross-applicable to many slots beyond spectral-clustering.
- **PR-G (Tier-6 kmeans substrate ~280 LOC, 3 days)** SC29 kmeans + SC30 kmeans++ + SC31 diagnostics. Should land FIRST as foundation.

Total ~3,260 LOC across 7 PRs, ~5-7 engineer-weeks net of substrate.

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled

**Pin 271-1 — SBM-easy-regime three-algorithm-agreement.** On stochastic-block-model G(n=400, k=4, p_in=0.4, p_out=0.05) (well-above Kesten-Stigum threshold), three algorithms must recover planted-communities with NMI > 0.99:
- SC8 `RecursiveSpectralBisection` (k-1 splits via Fiedler-sign)
- SC13 `SpectralClusteringNJW` (direct k-way via L_sym)
- SC14 `SpectralClusteringShiMalik` (direct k-way via L_rw)

All three must agree on the partition up-to-permutation. Saturates 3/3 + an "asymptotic-recoverability-pin" verifying the Decelle-2011 phase-transition.

**Pin 271-2 — Eigengap-heuristic vs cluster-quality-metric agreement.** On synthetic data with k=5 well-separated Gaussian-clusters in R^10:
- SC11 EigengapHeuristic predicts k=5 from spectrum-of-similarity-Laplacian
- SC31 SilhouetteScore-maximisation over k∈[2,10] selects k=5
- SC31 CalinskiHarabaszIndex-maximisation over k∈[2,10] selects k=5

All three must agree. Cross-link 261-mixture-model + 245-AIC/BIC for fourth-oracle.

**Pin 271-3 — Recursive vs direct k-way Ncut comparison.** On the canonical Yu-Shi-2003 NIPS test instance (synthetic 6-cluster Gaussian-mixture in R^2):
- SC10 RecursiveSpectralBisection k=6 (5 recursive bisections)
- SC13 SpectralClusteringNJW k=6
- SC16 BalancedKWayPartition k=6 (Yu-Shi rotation method)

The three must agree on the partition with NMI > 0.95 + SC16 must achieve the lowest k-way Ncut by the Yu-Shi-2003 paper-claim. Saturates 3/3 + a "rotation-vs-kmeans-tightness-pin".

**Pin 271-4 — Manifold-learning embedding-distance preservation on Swiss-roll.** On the canonical Swiss-roll dataset (n=1000 points sampled on 2-D manifold embedded in R^3):
- SC17 LaplacianEigenmaps embed to R^2
- SC18 LLE embed to R^2
- SC20 Isomap embed to R^2

All three must preserve the 2-D-intrinsic-distance-structure (Spearman-correlation between embedded-Euclidean-distance and ground-truth-geodesic-distance > 0.95). Saturates 3/3 + an "isometry-vs-conformal-pin" since Eigenmaps preserves local but not global, LLE preserves local geometry, Isomap preserves geodesic globally.

**Pin 271-5 — Kmeans++ vs kmeans-random-init quality on synthetic Gaussian-mixture.** On 1000-sample Gaussian-mixture data with k=10 components:
- SC29 KMeansLloyd with random-init averaged over 100 runs → median-Inertia
- SC29 KMeansLloyd with SC30 KMeans++ init averaged over 100 runs → median-Inertia (lower)
- Brute-force grid-search for global-optimum at k=2,3 (oracle for k=2,3 only — guarantees lower-bound)

KMeans++ inertia must be ≤ 1.1× brute-force-optimum at k=2,3 (Arthur-Vassilvitskii-2007 O(log k) bound). Saturates 3/3 + a "warm-start-quality-pin" certifying KMeans++ achieves theoretical-bound.

---

## 4. Touchpoints with other agents

- **157 (synergy-graph-linalg) G6+G7+G8+G9:** PARTIAL OVERLAP — 157 enumerates Fiedler + SpectralBisection + SpectralClusteringNJW + SpectralClusteringShiMalik as ~220 LOC; slot 271 deepens to 24 spectral-clustering primitives ~3,260 LOC. Coordinate: 157-PR-4 (extract InverseIteration from PCA) is the single critical refactor that unblocks this slot.
- **097 (linalg-missing) T1.eigvec + T1.lanczos + T1.svd + T1.kmeans:** STRICT-DEPENDENCY. Slot 271 cannot ship Tier-2/Tier-3/Tier-4 without 097-T1 substrate. Recommend coordinating PR-sequence.
- **254 (graph-cuts) C2 BK-2004 + C20 TRW-S:** COMPLEMENTARY. Where 254 is discrete-combinatorial-exact, slot 271 is continuous-spectral-relaxation-approximate. Cross-validation pin: SC1-SC4 cut-objective evaluators provide oracles for 254-C20 LP-tightness witnesses.
- **199 (synergy-graph-info) G3 SpectralEntropy:** SHARED-EIGENVALUE-SUBSTRATE. SC22 SBM-Recovery's Kesten-Stigum-threshold formula `(p−q)² > k(p+(k−1)q)` is dual to 199-G18 SBM-identifiability information-theoretic threshold (Abbe-Sandon-2015).
- **270 (graph-signal-proc) W19 Windowed-Graph-Fourier-Transform:** STRICT-DOWNSTREAM. SC17 LaplacianEigenmaps is the eigenvector-stacking substrate that 270-W19 consumes for vertex-frequency localisation.
- **262 (random-projection):** SC22 SBM at n>10^5 benefits from JL-projection-acceleration. Cross-link.
- **224 (streaming):** SC25 SpectralSparsification is the streaming-graph-sketch primitive. Cross-link.
- **246 (discrete-exterior-calculus) X28 spectral-DEC:** ORTHOGONAL — 246 is cotangent-Laplacian on meshes, slot 271 is graph-Laplacian on abstract weighted graphs; converge only when graph IS 1-skeleton of triangulated surface.
- **165 (synergy-graph-prob) SBM:** STRICT-CONSUMER. SC22 SBMRecovery is the spectral-frontend of any Bayesian-block-model fitting pipeline.
- **261 (mixture-model-init):** STRICT-CONSUMER of SC29 kmeans + SC30 kmeans++.
- **245 (spectral-methods PDE) S2 Golub-Welsch:** SHARED-EIGENVALUE-SUBSTRATE — both consume `linalg.QRAlgorithm` for eigendecomposition.
- **248 (multigrid):** STRICT-CONSUMER of SC24 SpectralCoarsening for graph-multigrid coarse-scale-construction.

---

## 5. Singular load-bearing recommendation

**Ship PR-G FIRST (Tier-6 kmeans substrate ~280 LOC, 3 days).** SC29 KMeansLloyd is the most-blocked-on primitive in reality — gates 5+ slots (097, 116, 117, 157-G8, 245-S2, 261-mixture, 271-Tier-2/3). Place in `linalg/cluster/kmeans.go`. Pure pedagogy, no academic moat, no novel mathematics — but unblocks ~700 LOC of downstream consumer code across half-a-dozen slots. **Then ship 157-PR-4 (InverseIteration extraction from PCA-private ~80 LOC) + 157-PR-1 (Laplacian/SymNorm/RandomWalk ~150 LOC).** These two PRs are 097-flagged + 157-flagged + 271-flagged — coordinated landing closes 18 of 24 spectral-clustering primitive blockers.

**Then ship PR-A (Tier-0 cut-objectives ~280 LOC, 1 day) + PR-B (Tier-1 spectral bisection ~340 LOC, 2 days).** Cheapest-large-leverage subset. Adds the canonical Ncut/RatioCut/Cheeger objective-functions + Fiedler-bisection + eigengap-heuristic to reality.

**Then ship PR-C (Tier-2 normalised spectral ~520 LOC, 1 week) — SINGULAR-MOAT.** SC13 NJW + SC14 Shi-Malik are the production-default community-detection / image-segmentation / spectral-clustering algorithms in scikit-learn / OpenCV / MATLAB. Zero-dep cross-language Go implementation exists in NO library worldwide. Single-most-cited NIPS-2002 + IEEE-TPAMI-2000 algorithm-pair in the Block-C ML-canon (combined ~31,000 citations).

**Then ship PR-D (Tier-3 manifold learning ~720 LOC, 1 week).** SC17 Eigenmaps + SC18 LLE + SC19 DiffusionMap + SC20 Isomap = the FOUR canonical manifold-learning algorithms (combined ~52,000 citations). Powers t-SNE / UMAP / PHATE-style dimensionality-reduction pipelines (which themselves are deep-learning-territory and out-of-scope).

**Then ship PR-E (Tier-4 cutting-edge ~720 LOC, 2 weeks) — SINGULAR-2024-FRONTIER.** SC22 SBMRecovery with Kesten-Stigum-threshold-diagnostic + non-backtracking-walk-matrix is the post-2010 spectral-redemption result (Krzakala-2013 PNAS) — production-quality SBM-recovery in reality with phase-transition certificates. SC25 SpectralSparsification subsumes 157-G17.

**Defer PR-F (Tier-5 sparse eigensolvers ~680 LOC) until n>5000 instances dominate consumer demand.** For n≤5000 dense QR + InverseIteration suffice; sparse-eigensolvers are 097-T1.lanczos territory and naturally shipped as part of the 097 unblock-track.

**Avoid scoping: deep-spectral-clustering (Tian-Liu-2017-SAE deep-spectral-net + Shaham-Stanton-Li-Nadler-Basri-Kluger-2018 SpectralNet + Bianchi-Grattarola-Alippi-2020 graph-pooling).** These are deep-learning primitives — aicore-territory, not reality-territory.

**Avoid scoping: graph-attention-spectral-clustering (Wu-Yang-Bian-Wang-Zhuang-2021).** GNN-flavoured, deep-learning — aicore-territory.

**Avoid scoping: spectral-clustering-with-side-information (semi-supervised Kamvar-Klein-Manning-2003-IJCAI / metric-learning Bilenko-Basu-Mooney-2004-ICML).** Niche; defer to v1.5+.

**Final precision-hazards:**
- **(a)** Sign of Fiedler-vector is non-unique (`v_2` and `−v_2` are both valid eigenvectors); SpectralBisection must canonicalise via `sign(v_2[0])` or partition-size-tiebreak.
- **(b)** k-means initialisation random-seed determines outcome; SC29 must accept explicit `rng` for cross-language reproducibility.
- **(c)** k-means convergence to local-optima; document expected #restarts (typically 10) for reliable answers.
- **(d)** Eigengap-heuristic on disconnected-graph: trivial-zero-eigvalue multiplicity = #components; SC11 must skip these or document.
- **(e)** Disconnected-graph spectral-clustering: each component is an independent cluster + within-component spectral-clustering — SC13/SC14 must handle gracefully via 157-G5 (algebraic-connectivity λ_2 = 0 detection).
- **(f)** SBM-impossible-regime SC22: must return `threshold_witness < 1` + EXIT without false-recovery (Decelle-2011 paper proves recovery impossible).
- **(g)** Floating-point ties in eigvec components: row-normalisation (SC13/SC14 step 4) divides-by-tiny-norm if a row of U is near-zero — guard with `EPS_RATIO ≥ 1e-12`.
- **(h)** Ncut/RatioCut on near-trivial-bisection (|A|=1): denominator → 0 produces +Inf. Document or floor.
- **(i)** kNN-graph-sparsity for SC18 LLE / SC20 Isomap: choice of k_neighbours dramatically affects embedding; canonical k=10 or k=√n; document.
- **(j)** Lanczos numerical-orthogonality-loss on symmetric-tridiagonal-deflation: SC26 must implement selective-or-full-reorthogonalisation Parlett-1980 / Demmel-1997 §7.
- **(k)** SC22 non-backtracking-walk-matrix B is non-symmetric (size 2|E|×2|E|); needs eigvecs of unsymmetric matrix — Lanczos doesn't apply, must use Arnoldi (097-T1.arnoldi separately-flagged).

**Headline:** Twenty-four spectral-clustering primitives close the entire 1970-2018 spectral-clustering canon (Cheeger-1970 + Hagen-Kahng-1992 RatioCut + Pothen-Simon-Liou-1990 SpectralBisection + Shi-Malik-2000 Ncut + Tenenbaum-deSilva-Langford-2000 Isomap + Roweis-Saul-2000 LLE + Ding-He-Zha-2001 MinMaxCut + Ng-Jordan-Weiss-2002 NJW + Belkin-Niyogi-2003 LaplacianEigenmaps + Yu-Shi-2003 Multi-class-Spectral + Zelnik-Manor-Perona-2004 SelfTuning + Coifman-Lafon-2006 DiffusionMap + von-Luxburg-2007 Tutorial + Decelle-Krzakala-Moore-Zdeborová-2011 SBM-Recovery + Spielman-Srivastava-2011 Sparsification + Krzakala-Moore-Mossel-Neeman-Sly-Zdeborová-Zhang-2013 Spectral-Redemption + Loukas-Vandergheynst-2018 Coarsening) in ~3,260 LOC of pure synthesis on top of 157-G2/G3/G4/G6 Laplacian-substrate (already-flagged-shippable) + linalg.QRAlgorithm-PRESENT + new linalg.InverseIteration + new linalg/cluster/kmeans.go. Cheapest-1-day-subset SC1+SC2+SC3+SC4+SC8+SC11 ~210 LOC; foundational keystone SC29 KMeansLloyd ~140 LOC ship-once-consume-everywhere; singular-moat SC13+SC14+SC17+SC22 ~660 LOC = NJW + Shi-Malik + Eigenmaps + SBMRecovery (combined ~75,000 citations); 2024-frontier SC22 SBMRecovery with Kesten-Stigum-threshold + non-backtracking-spectral-redemption ~240 LOC; pedagogical von-Luxburg-2007 walk-through SC8+SC11+SC13 + R-Pin-271-1 SBM-easy-regime cross-validation ~270 LOC. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (SBM-easy-regime three-algorithm agreement, eigengap-vs-silhouette-vs-CalinskiHarabasz-on-Gaussian-mixture, recursive-vs-direct-k-way-Ncut comparison, manifold-learning embedding-distance preservation on Swiss-roll, kmeans++-vs-random-init quality on Gaussian-mixture). Strict-upstream of 165-graph-prob-SBM + 270-graph-signal-proc-W19 + 262-random-projection-spectral-acceleration + 224-streaming-spectral-sparsification + 248-multigrid-spectral-coarsening; strict-downstream of 157-G2/G3/G4/G6 (Laplacian) + 097-T1.eigvec (InverseIteration extraction) + 097-T1.kmeans + (optional) 097-T1.lanczos. Recommended placement NEW sub-package `graph/spectral/` ~2,540 LOC + supporting `linalg/cluster/` + `linalg/krylov/` substrate ~960 LOC.
