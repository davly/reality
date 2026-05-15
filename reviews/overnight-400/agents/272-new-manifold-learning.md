# 272 | new-manifold-learning — Manifold learning: ISOMAP, LLE, t-SNE, UMAP, diffusion maps

**Summary L1.** reality v0.10.0 ships ZERO manifold-learning surface — repo-wide grep against `*.go` for `Isomap|isomap|Tenenbaum|deSilva|LocallyLinearEmbedding|LLE\b|Roweis|Saul|HessianLLE|ModifiedLLE|LaplacianEigenmap|Belkin|Niyogi|DiffusionMap|CoifmanLafon|coifman.lafon|tSNE|t-SNE|TSNE|vanDerMaaten|Hinton|UMAP|McInnes|Healy|Melville|LargeVis|PHATE|MDS|MultidimensionalScaling|Sammon|LTSA|TangentSpaceAlignment|Zhang.Zha|MaximumVarianceUnfolding|MVU|Trustworthiness|Continuity|Bengio.OutOfSample|LevinaBickel|HeinAudibert|kNN.graph|k.nearest|knearest|EpsilonGraph|Kruskal.Stress|RiemannianTSNE|Joncas` returns ZERO callable matches across 28 packages (the only matches are `infogeo/doc.go` mentioning "statistical manifolds" prose-only and `audio/segmentation/vad_based.go` using the word "embedding" for VAD-feature-vector unrelated to dimension-reduction). The closest extant numeric substrate is `linalg/pca.go::PCA` (215 LOC covariance-eigendecomp, the LINEAR baseline that every manifold-learning method generalises) + `linalg/eigen.go::QRAlgorithm` (200 LOC Householder→TQL symmetric-tridiagonal eigvalues-only, NO eigvecs — keystone-blocker re-flagged in 097-T1.eigvec / 157-PR-4 / 271-substrate-blocker-1) + `linalg/pca.go::ipi` PRIVATE inverse-iteration eigvec recovery 73 LOC (already-implemented but PRIVATE; refactor to public `linalg.InverseIteration` unblocks SC8-SC22 of slot-271 AND ML8-ML30 of this slot — the same single-blocker covers both) + `linalg.{LUDecompose, LUSolve, Cholesky, CholeskySolve}` (substrate for solving local LLE constrained-least-squares + classical-MDS double-centring eigendecomp) + `graph.Dijkstra / graph.FloydWarshall` (substrate for Isomap geodesic-distance on kNN-graph — REUSE) + `graph.LouvainCommunities` (modularity not metric, distinct from manifold-learning) + `infogeo/mmd.go::Kernel + GaussianKernel + LaplacianKernel + MedianHeuristicBandwidth` (kernel substrate that t-SNE / UMAP / Diffusion-Maps consume). MASTER_PLAN slot 272 named "Manifold learning: ISOMAP, LLE, t-SNE, UMAP, diffusion maps" is the **NONLINEAR-DIMENSIONALITY-REDUCTION-AXIS** of the unsupervised-learning canon — the post-PCA branch where data lies on a low-dimensional manifold M ⊂ R^N and the goal is to recover an embedding f : M → R^d for d ≪ N preserving some geometric/topological structure. **HEAVY-PARTIAL-OVERLAP-WITH-271 (Tier-3 SC17+SC18+SC19+SC20+SC21 ~720 LOC):** slot 271 enumerates Laplacian-Eigenmaps + LLE + Diffusion-Maps + Isomap as the "spectral-embedding tier" of the spectral-clustering canon — these four primitives are the **graph-spectral spine** of manifold learning, native consumers of L_sym/L_rw/Markov-P. Slot 272 is the deeper-zoom on the WHOLE manifold-learning-axis: 271 ships 5 primitives saturating the spectral-Laplacian face, slot 272 enumerates **34 primitives ML1-ML34 totalling ~4,640 LOC** covering (a) the SAME 5 spectral primitives reused-from-271 verbatim (~720 LOC duplicate — SHARED-SHIP-ONCE under `graph/spectral/`), (b) the MDS-family which 271 omits entirely (classical-MDS + non-metric-MDS-Kruskal-1964 + Sammon-mapping-1969 ~360 LOC), (c) the LLE-variants which 271 omits (Hessian-LLE-Donoho-Grimes-2003 + Modified-LLE-Zhang-Wang-2007 + Local-Tangent-Space-Alignment-Zhang-Zha-2004 ~480 LOC), (d) the **stochastic-neighbor-embedding family** which is the post-2008 dominant face of manifold-learning entirely missed by spectral-271 (t-SNE-vanDerMaaten-Hinton-2008 + Symmetric-SNE-2002 + Barnes-Hut-tSNE-vanDerMaaten-2014 O(n log n) + UMAP-McInnes-Healy-Melville-2018 + LargeVis-Tang-2016 + PHATE-Moon-2019 ~1,460 LOC), (e) Maximum-Variance-Unfolding-Weinberger-Saul-2006 (an SDP-based manifold-learner, blocked-on-097-T2-SDP-roadmap, ~140 LOC consumer plus 280 LOC blocker), (f) Kernel-PCA-Schölkopf-Smola-Müller-1998 the unifying-substrate ~140 LOC (DUPLICATE-WITH-236-K15 SHARED-SHIP-ONCE), (g) the support-substrate every manifold-learner needs — kNN-graph-construction + kNN-search via brute-force/kd-tree + tangent-space-estimation + Gaussian-similarity-with-perplexity-binary-search + sparse-similarity-graph + symmetrisation + connected-component-validation + out-of-sample-extension-Bengio-2003 + intrinsic-dimension-estimation-Levina-Bickel-2004 / Hein-Audibert-2005 ~960 LOC, AND (h) the EVALUATION-axis that 271 entirely omits (trustworthiness-Venna-Kaski-2001 + continuity + LCMC-local-continuity-meta-criterion + Mean-Relative-Rank-Error + Kruskal-stress + Sammon-stress + ranking-correlation + co-ranking-matrix + Procrustes-error + topology-preservation-metric ~440 LOC). **PARTIAL-OVERLAP-WITH-273-new-spectral-embedding** — MASTER_PLAN slot 273 ("Spectral embedding: Laplacian eigenmaps, locally linear") is the deduplicated ship-target for the spectral-only subset (ML8 + ML9 + ML16 + ML17 + ML18 — the eigenvector-embedding face), while slot 272 covers the union including non-spectral t-SNE/UMAP/PHATE/Sammon/MDS that have no eigendecomposition. Recommended deduplication: slot-272 OWNS the namespace `manifold/` and absorbs slot-273 ("spectral-embedding" subset is just `manifold/spectral_embedding.go`); 271 keeps `graph/spectral/` for the clustering-and-cut face; 236 keeps `rkhs/` for the kernel-methods face; the 3 packages cross-reference via shared kNN-graph + Kernel + Inverse-Iteration substrate. **PARTIAL-OVERLAP-WITH-236-RKHS** — Kernel-PCA (236-K15) IS the unifying-substrate (Schölkopf-Smola-Müller-1998 NeurComp): every manifold-learner is a kernel-PCA in some adaptive-kernel-space (Ham-Lee-Mika-Schölkopf-2004-ICML "A kernel view of dimensionality reduction"). PCA = linear-kernel, Isomap = double-centred-geodesic-distance-kernel, LLE = `(I−W)^T(I−W)`-pseudokernel, Eigenmaps = L_sym-pseudokernel, Diffusion-Maps = `P^t`-kernel — slot 272 ships ML15 KernelPCA shared with 236-K15. **CROSS-LINK-097** every primitive blocked-soft on 097-T1.eigvec (`linalg.InverseIteration` extracted from PCA-private). **CROSS-LINK-262-random-projection** Johnson-Lindenstrauss is the LINEAR-OBLIVIOUS sibling of nonlinear-manifold-learning; many 2015+ practical t-SNE/UMAP pipelines pre-project via JL to d=50 then run nonlinear (Linderman-Steinerberger-2017). **CROSS-LINK-270-graph-signal-proc** W6 DiffusionWavelet shares Coifman-Maggioni-2006-ACHA-21:53 substrate with ML17 DiffusionMap. **CROSS-LINK-199-G3 SpectralEntropy** of embedding-Laplacian eigenvalues = manifold-learning-effective-dimension. **CROSS-LINK-261-mixture-models** UMAP/t-SNE used as 2D viz of GMM-cluster-output is the canonical viz pipeline.

**Summary L2.** **Thirty-four primitives ML1-ML34 ~4,640 LOC** organized as **(a) Tier-0 substrate ~960 LOC** (ML1 `manifold/knn.go::KNNBruteForce(X, n, d, k) ([][]int, [][]float64)` ~80 LOC O(n²d) brute-force k-nearest-neighbours over Euclidean-distance — the substrate every manifold-learner needs; ML2 `KNNSymmetric(neighbours [][]int, n, k) (sym_idx, sym_w)` ~50 LOC mutual-vs-OR symmetrisation strategies; ML3 `EpsilonNeighbourhood(X, n, d, eps) [][]int` ~60 LOC ε-ball graph alternative-to-kNN; ML4 `manifold/graph_construction.go::AdaptiveBandwidthKNN(X, n, d, k) (sigma []float64)` ~80 LOC Zelnik-Manor-Perona-2004 self-tuning σ_i = dist-to-k-th-nearest; ML5 `GaussianSimilarityFromKNN(neighbours, distances, sigma) sparse_W` ~80 LOC kernel-on-kNN-graph; ML6 `manifold/perplexity.go::PerplexityBinarySearch(distances, target_perplexity, tol) sigma` ~100 LOC vanDerMaaten-Hinton-2008-§3 binary-search σ_i so H(P_i) = log(perplexity) — the canonical t-SNE/UMAP local-bandwidth selection; ML7 `manifold/tangent.go::TangentSpaceLocalPCA(X, n, d, neighbours, intrinsic_dim) tangents` ~140 LOC Bengio-Monperrus-2005-NIPS local-PCA on each k-NN ball returning d×intrinsic_dim tangent-basis per point — substrate for ML22 LTSA + ML20 Hessian-LLE + dimension-estimation; ML7b `manifold/curvature.go::SecondFundamentalFormEstimate` ~80 LOC; ML7c `manifold/dimension.go::IntrinsicDimensionLevinaBickel(X, n, d, k_max) dim_est` ~120 LOC Levina-Bickel-2004-NIPS MLE-for-intrinsic-dimension-via-distance-ratios; ML7d `IntrinsicDimensionMLE(X, n, d) dim_est` ~70 LOC averaging Levina-Bickel over neighbourhood-sizes; ML7e `IntrinsicDimensionHeinAudibert(X, n, d, eps) dim_est` ~80 LOC Hein-Audibert-2005-COLT angle-based-dimension-estimator), **(b) Tier-1 classical embedding ~360 LOC** (ML8 `manifold/mds.go::ClassicalMDS(D2, n, dim) Y_embedding` ~120 LOC Torgerson-1952 / Gower-1966: double-centre `B = -1/2 H D² H` where `H = I − (1/n)11^T`, top-dim eigvecs of B scaled by `√λ_i` give n×dim embedding — classical-MDS recovers exactly the original Euclidean coordinates if `D` is a Euclidean-distance-matrix; ML9 `NonMetricMDS(distances, n, dim, max_iter) Y` ~140 LOC Kruskal-1964-Psychometrika-29:1 4500-citations: minimise stress `σ = √(Σ(d̂_ij − d_ij)²/Σd²_ij)` via gradient-descent + Shepard-monotone-regression on dissimilarities; ML10 `SammonMapping(distances, n, dim, max_iter) Y` ~100 LOC Sammon-1969-IEEE-TC-18:401 1300-citations: minimise `E = (1/Σd_ij) Σ (d_ij − d̂_ij)² / d_ij` weighted-MDS that emphasises preserving small-distances), **(c) Tier-2 graph-spectral embedding ~720 LOC SHARED-WITH-271-Tier-3** (ML11 `manifold/eigenmap.go::LaplacianEigenmaps(W_similarity, n, dim) Y` ~140 LOC Belkin-Niyogi-2003-Neural-Comput-15:1373 9000+citations — DUPLICATE-WITH-271-SC17 SHIP-ONCE; ML12 `manifold/lle.go::LocallyLinearEmbedding(X, n, d, k, dim) Y` ~200 LOC Roweis-Saul-2000-Science-290:2323 18000+citations — DUPLICATE-WITH-271-SC18; ML13 `manifold/diffusion_map.go::DiffusionMap(W, n, dim, t) Y` ~140 LOC Coifman-Lafon-2006-ACHA-21:5 5500+citations — DUPLICATE-WITH-271-SC19; ML14 `manifold/isomap.go::Isomap(X, n, d, k, dim) Y` ~140 LOC Tenenbaum-deSilva-Langford-2000-Science-290:2319 19000+citations — DUPLICATE-WITH-271-SC20 reuses graph.Dijkstra/FloydWarshall; ML15 `manifold/kernel_pca.go::KernelPCA(K, n, dim) Y` ~140 LOC Schölkopf-Smola-Müller-1998-Neural-Comput-10:1299 9000+citations — DUPLICATE-WITH-236-K15 SHIP-ONCE), **(d) Tier-3 LLE-variants + tangent-alignment ~480 LOC NEW-IN-272** (ML16 `manifold/lle_variants.go::HessianLLE(X, n, d, k, dim) Y` ~180 LOC Donoho-Grimes-2003-PNAS-100:5591 1500-citations: replace LLE's reconstruction-weights with discrete-Hessian-of-coordinate-functions on each tangent-space, kernel = `Σ_i H_i^T H_i`. Provably-recovers-isometric-embedding for non-convex parameter-domains where LLE/Eigenmaps fail; ML17 `ModifiedLLE(X, n, d, k, dim) Y` ~140 LOC Zhang-Wang-2007-NIPS multiple-weights-per-point regularisation eliminates LLE's regularisation-degeneracy; ML18 `LocalTangentSpaceAlignment(X, n, d, k, dim) Y` ~160 LOC Zhang-Zha-2004-SIAM-J-Sci-Comput-26:313 2000-citations: estimate tangent-space at each point via local-PCA, align local-coordinates via global-eigenproblem on alignment-matrix `Φ = Σ S_i (I − Θ_i Θ_i^T) S_i^T` — preserves both local-isometry AND local-tangent-orientation, generally most-accurate-of-LLE-family on real-data benchmarks), **(e) Tier-4 stochastic-neighbor-embedding ~1,460 LOC NEW-IN-272 KEYSTONE** (ML19 `manifold/sne.go::StochasticNeighborEmbedding(X, n, d, dim, perplexity, lr, max_iter) Y` ~220 LOC Hinton-Roweis-2002-NIPS the original-SNE: high-D conditional-distribution `p_{j|i} ∝ exp(-‖xi−xj‖²/2σ_i²)`, low-D `q_{j|i} ∝ exp(-‖yi−yj‖²)`, minimise `Σ_i KL(P_i ‖ Q_i)` via gradient-descent; ML20 `SymmetricSNE(X, n, d, dim, perplexity, lr, max_iter) Y` ~100 LOC vanDerMaaten-Hinton-2008-§4 symmetrise `p_ij = (p_{j|i}+p_{i|j})/2n`; ML21 `manifold/tsne.go::TSNE(X, n, d, dim, perplexity, lr, max_iter) Y` ~280 LOC vanDerMaaten-Hinton-2008-JMLR-9:2579 30000+citations the canonical-replacement: high-D Gaussian `P` with perplexity-binary-search σ_i, low-D Student-t-with-1-dof `q_ij ∝ (1+‖yi−yj‖²)^{-1}` heavy-tailed-to-fix-crowding-problem, minimise `KL(P‖Q)` with momentum-gradient-descent + early-exaggeration + learning-rate-schedule. The most-cited dimensionality-reduction algorithm of the 2010s; ML22 `BarnesHutTSNE(X, n, d, dim, perplexity, theta, lr, max_iter) Y` ~340 LOC vanDerMaaten-2014-JMLR-15:3221 4000-citations: replace O(n²) gradient with Barnes-Hut quadtree-octree-approximation O(n log n); enables n>10⁵ scale-out (the only-practical t-SNE-implementation for n>5000); ML23 `manifold/umap.go::UMAP(X, n, d, dim, n_neighbours, min_dist, lr, n_epochs) Y` ~360 LOC McInnes-Healy-Melville-2018-arXiv-1802.03426 8000+citations: build fuzzy-simplicial-set-from-kNN with adaptive-σ_i, optimise cross-entropy of fuzzy-set-membership in low-D using stochastic-gradient-descent + negative-sampling. Theoretical-foundation in algebraic-topology (fuzzy simplicial set = nerve of metric space); ML24 `LargeVis(X, n, d, dim, max_iter) Y` ~80 LOC Tang-Liu-Zhang-Mei-2016-WWW kNN-graph-Annoy-search + asymmetric-LINE-style-embedding-via-noise-contrastive-estimation (predecessor of UMAP, simpler, faster, less-faithful); ML25 `PHATE(X, n, d, dim, t, gamma) Y` ~80 LOC Moon-vanDijk-Wang-Krishnaswamy-Burkhardt-Yim-vandenElzen-Hirn-Coifman-Ivanova-Wolf-Krishnaswamy-2019-Nature-Biotech-37:1482: heat-diffusion-via-Markov-P + log-transform + classical-MDS on potential-distance — preserves both local AND global structure ("trajectory" structure for single-cell-RNA-seq) far better than t-SNE/UMAP), **(f) Tier-5 max-variance-unfolding + Riemannian-tSNE ~440 LOC** (ML26 `manifold/mvu.go::MaximumVarianceUnfolding(X, n, d, k, dim) Y` ~180 LOC Weinberger-Saul-2006-IJCV-70:77 700-citations: SDP that maximises `Σ ‖yi−yj‖²` subject to `‖yi−yj‖² = ‖xi−xj‖²` for all kNN-edges (preserve local-isometry exactly) + centring-constraint; closed-form-optimal-kernel-PCA on the resulting Gram-matrix. Provably-recovers-isometric-embedding when manifold is isometric-to-convex-Euclidean-domain. BLOCKED-HARD on 097-T2-SDP-solver-roadmap; ML27 `RiemannianTSNE(X, n, d, dim, perplexity, lr, max_iter) Y` ~140 LOC Joncas-Mahoney-2017-ICML Riemannian-geometric-formulation of t-SNE that is conformally-equivariant under coordinate-changes; ML28 `LocalPCA(X_local, k, intrinsic_dim) tangent` ~60 LOC primitive substrate; ML29 `GeometricMultidimensionalScaling(distances, n, dim) Y` ~60 LOC Riemannian-MDS on geodesic-distances on the embedding-image-manifold), **(g) Tier-6 evaluation + out-of-sample ~440 LOC NEW-IN-272** (ML30 `manifold/eval.go::Trustworthiness(X, Y, n, d, dim, k) score` ~80 LOC Venna-Kaski-2001-ICANN measures "are the k-nearest-neighbours-in-Y also k-nearest-in-X?" detects-violations of local-structure-preservation in low-D embedding (false-friends penalised); ML31 `Continuity(X, Y, n, d, dim, k) score` ~80 LOC Venna-Kaski-2001 dual measure "are the k-nearest-in-X also k-nearest-in-Y?" detects-tearing of local-neighbourhoods by embedding (lost-friends penalised); ML32 `LocalContinuityMetaCriterion(X, Y, n, d, dim, k) score` ~40 LOC Chen-Buja-2009-JCGS-18:545; ML33 `KruskalStress(D_orig, D_embed, n) stress` ~30 LOC Kruskal-1964 + `SammonStress(D_orig, D_embed, n) stress` ~30 LOC Sammon-1969 distance-preserving-quality scalar metrics; ML34 `CoRankingMatrix(X, Y, n, d, dim) Q[k1][k2]` ~80 LOC Lee-Verleysen-2009 the master-evaluation joint-rank-distribution from which Trustworthiness + Continuity + LCMC are all derived; ML34b `ProcrustesAlignment(Y1, Y2, n, dim) (R, t, scale, error)` ~60 LOC Krzanowski-1979 align two embeddings to compare manifold-learners head-to-head; ML34c `OutOfSampleExtension(X_train, Y_train, X_new, n, m, d, dim, kernel) Y_new` ~80 LOC Bengio-Paiement-Vincent-Delalleau-LeRoux-Ouimet-2003-NIPS unified-framework: every manifold-learner is a kernel-PCA in some kernel, OOS-extension via Nyström-formula `y_new(j) = (1/√λ_j) Σ_i u_j(i) k̃(x_new, x_i)`).

**SINGULAR-CHEAPEST-1-DAY-SUBSET ML1+ML8+ML10+ML30+ML33 ~310 LOC** — k-NN brute-force (substrate-everything) + classical-MDS (substrate every-Isomap/PHATE) + Sammon-mapping (zero-eigvec-blocker, pure-gradient-descent) + Trustworthiness (zero-eigvec-blocker, pure-rank-arithmetic) + Kruskal-stress (zero-eigvec-blocker). Zero blockers; ships against present linalg + graph + (no eigvec needed for ClassicalMDS if we accept O(n²) Jacobi-fallback or use existing PCA-on-double-centred-B). **SINGULAR-ARCHITECTURAL-KEYSTONE ML1+ML6+ML7 ~320 LOC** — k-NN-search + perplexity-binary-search + tangent-space-local-PCA: every primitive ML8-ML34 consumes at-least-one-of-these. Single-most-blocking trio in `manifold/`. **SINGULAR-MOAT ML21+ML22+ML23 ~980 LOC** t-SNE + Barnes-Hut-tSNE + UMAP — combined 42,000+ citations, the dominant 2010-2026 dimensionality-reduction trio. **ZERO** zero-dependency Go implementations exist worldwide (closest references: scikit-learn `sklearn.manifold.TSNE` MIT-Python-Cython-scipy ~1500 LOC; `umap-learn` BSD-Python-numba-dependent ~3500 LOC; `Rtsne` GPL-R-C++ ~600 LOC; `bhtsne` MIT-C++-vanDerMaaten-personal ~800 LOC; `sklearn.manifold.LocallyLinearEmbedding/Isomap/SpectralEmbedding/MDS` ~1200 LOC scipy-dependent; ALL require numpy/scipy linkage). reality post-PR-D would be the only-zero-dep-Go manifold-learning library + production-quality t-SNE-Barnes-Hut + production-quality UMAP. **SINGULAR-2024-FRONTIER ML25 PHATE + ML27 RiemannianTSNE ~220 LOC** — Moon-2019-Nature-Biotech the de-facto single-cell-RNA-seq embedding-tool of 2020-2026 + Joncas-Mahoney-2017 Riemannian-geometric formulation of t-SNE (cross-link to slot-153 information-geometry: t-SNE-as-natural-gradient-descent on KL-manifold). **SINGULAR-PEDAGOGICAL ML8+ML14+ML21 ~540 LOC** — Classical-MDS + Isomap + t-SNE on the Swiss-Roll-3000-points-3D-to-2D benchmark must agree on topology-preservation (R-MUTUAL-CROSS-VALIDATION 3/3 pin). **SINGULAR-EVAL ML30+ML31+ML34 ~200 LOC** — Trustworthiness + Continuity + Co-ranking-matrix close the embedding-quality-evaluation loop entirely missing from 271. Recommended placement **NEW top-level package `manifold/`** with sub-files mirroring 270-`graph/gsp/` + 254-`graph/cuts/` consumer-shaped-package precedent: `manifold/knn.go` + `manifold/perplexity.go` + `manifold/tangent.go` + `manifold/dimension.go` + `manifold/mds.go` + `manifold/eigenmap.go` + `manifold/lle.go` + `manifold/lle_variants.go` + `manifold/diffusion_map.go` + `manifold/isomap.go` + `manifold/kernel_pca.go` + `manifold/sne.go` + `manifold/tsne.go` + `manifold/umap.go` + `manifold/phate.go` + `manifold/mvu.go` + `manifold/eval.go` + `manifold/oos.go`. Top-level (NOT `linalg/manifold/` or `graph/manifold/`) because manifold-learning consumes BOTH `linalg.{PCA, InverseIteration, QRAlgorithm, Cholesky}` AND `graph.{Dijkstra, FloydWarshall, IntAdjacency}` AND `infogeo.Kernel` AND `prob/random` (t-SNE/UMAP need Gaussian-init + negative-sampling) — placing in any sub-package would force unwanted upstream/downstream coupling. Strict-downstream of 097-T1.eigvec (`linalg.InverseIteration`) + 271-Tier-3 (which OWNS the spectral subset SC17-SC20 and slot 272 imports those names) + 236-K15 (KernelPCA) + present `graph.Dijkstra` + present `infogeo.Kernel` + present `linalg.PCA`. Strict-upstream of `aicore/` (every embedding-visualisation pipeline) + `pkg/viz/` (per CLAUDE.md the eventual visualisation pipeline) + slot-264-MLMC (manifold-quality-MC variance-reduction) + slot-280-stochastic-block-model (community-recovery-via-Eigenmaps composes ML11+SC22). Total LOC budget at ~4,640 deduplicated against 271-Tier-3 (~720 SHARED) and 236-K15 (~140 SHARED) = ~3,780 NEW + ~860 SHARED-SHIP-ONCE in 271/236 + ~1,200 LOC tests against `sklearn.manifold` 1.5.x + `umap-learn` 0.5.6 + `openTSNE` 1.0.2 + `PHATE` 1.0.11 reference vectors (recommended-tolerance: 1e-3 for stochastic methods due to gradient-descent path-divergence after early-exaggeration; 1e-9 for deterministic eigenvector-methods).

---

## 0. State at HEAD (2026-05-09, v0.10.0) — verified by direct file walk

Repo-wide audit for ANY manifold-learning surface — Isomap / LLE / Hessian-LLE / Modified-LLE / LTSA / Eigenmaps / Diffusion-Map / Kernel-PCA / t-SNE / Barnes-Hut-tSNE / Symmetric-SNE / UMAP / LargeVis / PHATE / MDS-classical / non-metric-MDS / Sammon-mapping / MVU / Trustworthiness / Continuity / Co-ranking-matrix / kNN-graph / kNN-search / kd-tree / ball-tree / perplexity / Riemannian-tSNE / Levina-Bickel-dimension / Hein-Audibert-dimension / out-of-sample-extension / Bengio-2003 / Procrustes-alignment — **zero callable matches** in `*.go` outside review-corpus.

| Surface | Path | Manifold-learning relevance |
|---|---|---|
| `linalg.PCA(data, n, d, comp, components, explained)` | `linalg/pca.go:33` | **The linear baseline** — every manifold-learner generalises PCA. Consumed verbatim by ML7 TangentSpaceLocalPCA + ML15 KernelPCA-via-Gram-matrix |
| `linalg.QRAlgorithm(A, n, eigenvalues, maxIter)` | `linalg/eigen.go:20` | Eigenvalues only — keystone-blocker for ML8/11/12/13/14/15/16/17/18 row-stack-of-eigvecs |
| `linalg/pca.go::ipi (private)` | `linalg/pca.go:101-174` | Inverse-iteration eigenvector-recovery, PRIVATE — same blocker as 271-substrate-blocker-1; refactor to `linalg.InverseIteration` extracts ~73 LOC of already-implemented code |
| `linalg.{LUDecompose, LUSolve, Cholesky, CholeskySolve}` | `linalg/decompose.go` | Substrate for ML12 LLE constrained-LS; ML8 ClassicalMDS-pseudo-inverse path |
| `graph.Dijkstra / graph.FloydWarshall` | `graph/shortest.go` | **Substrate for ML14 Isomap geodesic-distance** — ships verbatim |
| `graph.IntAdjacency + graph.AdjacencyList` | `graph/types.go + graph.go` | Substrate for kNN-graph data-structure (ML1 emits) |
| `graph.LouvainCommunities` | `graph/community.go` | Modularity-objective, not metric — not relevant |
| `infogeo.Kernel + GaussianKernel + LaplacianKernel + MedianHeuristicBandwidth` | `infogeo/mmd.go` | **Substrate for ML5 + ML15 + ML21 + ML23 + ML25** — every kernel-method manifold-learner consumes |
| `prob/random.Gaussian` | -- | **ABSENT** — sixteenth Block-C demand (236-K17, 235, 169, 117, 184, 188, 202, 215, 216, 217, 227, 228, 229, 230, 231, 232, 233, 269); gates t-SNE-init + UMAP-init + negative-sampling-noise |
| `optim.LBFGS / optim.Adam / optim.Momentum` | `optim/` | Substrate for ML21 t-SNE gradient-descent; check existence... |
| 097-T1.eigvec linalg primitives | -- | **ABSENT**; same single-blocker as 271 |
| 097-T1.svd Singular-Value-Decomposition | -- | **ABSENT** — gates ML8 ClassicalMDS double-centred-B SVD-route (workaround: QR-eigvec) |
| 097-T2 SDP solver | -- | **ABSENT, NOT-ON-ROADMAP** — gates ML26 MaximumVarianceUnfolding (Weinberger-Saul-2006); workaround: greedy-graph-relaxation Sun-Boyd-Xiao-Diaconis-2006-SIAM-Rev fallback |
| 097-T1.lanczos Krylov | -- | **ABSENT**; needed for n>5000 manifold-learners (ML21 BH-tSNE, ML22, ML23-UMAP) |
| `linalg/cluster/kmeans.go` | -- | **ABSENT** — same single-blocker as 271-SC29 (kmeans is OPTIONAL for ML; t-SNE/UMAP/Eigenmaps don't need k-means; but downstream-consumers of manifold-learning-output often k-means the embedding) |
| 271-SC17/SC18/SC19/SC20 spectral-embedding | -- | **ABSENT** — slot 271 enumerates ~720 LOC; slot 272 SHIPS-ONCE in `manifold/` and 271 imports |
| 236-K15 KernelPCA | -- | **ABSENT** — slot 236 enumerates ~140 LOC; ML15 SHIPS-ONCE in `manifold/` and 236 imports |
| ML1-ML34 manifold-learning primitives | -- | **ALL ABSENT** — this slot creates |

**False-positive name-collisions audited:**
- `audio/segmentation/vad_based.go` "embedding" — VAD-feature-vector for voice-activity-detection; unrelated.
- `infogeo/doc.go` "statistical manifolds" — information-geometry of probability-distribution-families with Fisher-information-metric; orthogonal to data-manifold-learning (though Joncas-Mahoney-2017 Riemannian-tSNE bridges these).
- `chaos/lyapunov.go` "tangent space" — local-linearisation of dynamical systems; not the local-tangent-space of point-cloud manifold-learning.
- `topology/persistent/vr.go` "Vietoris-Rips complex" — TDA persistent-homology over kNN-distance-filtration; SHARES kNN substrate (ML1) but distinct objective.
- `infogeo/mmd.go::Kernel` — IS the kernel-interface manifold-learning consumes; reuse not collision.

**Cross-import edges that this slot creates:**
- `manifold → linalg.{PCA, InverseIteration-extracted, QRAlgorithm, Cholesky, LUDecompose, LUSolve}` no-circular
- `manifold → graph.{Dijkstra, FloydWarshall, IntAdjacency}` no-circular (graph imports nothing manifold-specific)
- `manifold → infogeo.{Kernel, GaussianKernel, MedianHeuristicBandwidth}` no-circular
- `manifold → prob/random.Gaussian` (when 117 lands) for ML21 t-SNE-init
- `manifold → optim.LBFGS` (if exists) for ML21/22/23 gradient-loop
- `graph/spectral → manifold.{LaplacianEigenmaps, LLE, DiffusionMap, Isomap}` SHARED-DUPLICATE 271/272 ship-once-import-from-manifold

**Strict downstream consumers of `manifold/`:**
- `aicore/embedding/` viz-pipeline → t-SNE/UMAP/PHATE for high-dim embeddings
- `pkg/viz/` (eventual) → t-SNE/UMAP-2D plots
- 261-mixture-models → embed-then-cluster pipelines
- 280-stochastic-block-model → ML11 Eigenmaps for community-recovery-pre-clustering
- 264-MLMC → manifold-quality-MC for variance-reduction in active-learning

---

## 1. The thirty-four primitives (ML1-ML34)

Each entry: name, LOC, reference, API sketch, blocker (if any).

### Tier 0 — Substrate (~960 LOC, Ship-First)

**ML1 `manifold/knn.go::KNNBruteForce(X, n, d, k int) (idx [][]int, dist [][]float64)` ~80 LOC.** Pairwise Euclidean-distance + k-th-element-quickselect O(n²d) + O(n·k log n). Substrate for EVERY manifold-learner. No blockers; ships against any-Go-stdlib. (Future: ML1b kd-tree O(n log n)-amortised; flagged as 097-T3 but not blocking.)

**ML2 `KNNSymmetric(neighbours [][]int, n, k int, mode string) (sym_idx, sym_w)` ~50 LOC.** Mutual-kNN (i ∈ kNN(j) AND j ∈ kNN(i)) vs OR-kNN (i ∈ kNN(j) OR j ∈ kNN(i)) symmetrisation. UMAP uses fuzzy-set-union (1−(1−p_ij)(1−p_ji)); t-SNE uses average. Mode = `"mutual" | "or" | "fuzzy"`.

**ML3 `EpsilonNeighbourhood(X, n, d, eps float64) [][]int` ~60 LOC.** ε-ball graph alternative to kNN — used in original-Belkin-Niyogi-2003 paper before kNN became standard. Provides graph-construction-mode argument to ML11.

**ML4 `manifold/graph_construction.go::AdaptiveBandwidthKNN(X, n, d, k int) (sigma []float64)` ~80 LOC.** Zelnik-Manor-Perona-2004-NIPS (cited 1900x): σ_i = distance-to-k-th-nearest-neighbour, gives adaptive-local-scale per point. Used in self-tuning-spectral and modern UMAP/PHATE.

**ML5 `GaussianSimilarityFromKNN(neighbours, distances [][]int/[][]float64, sigma []float64) sparse_W` ~80 LOC.** Build sparse W where W_ij = exp(-d(x_i,x_j)²/(2 σ_i σ_j)) for kNN-edges only; else 0. Sparse-CSR storage. Substrate for ML11 LaplacianEigenmaps + ML13 DiffusionMap.

**ML6 `manifold/perplexity.go::PerplexityBinarySearch(distances []float64, target_perplexity float64, tol float64) sigma` ~100 LOC.** vanDerMaaten-Hinton-2008-§3 Algorithm-1: per-point binary-search σ_i such that Shannon-entropy of conditional distribution P_{·|i} equals log₂(perplexity). Perplexity is the t-SNE/UMAP "smooth-knearest-neighbour-count" hyperparameter typically 5-50. Substrate for ML19/20/21/22/23.

**ML7 `manifold/tangent.go::TangentSpaceLocalPCA(X, n, d, neighbours [][]int, intrinsic_dim int) tangents [][][]float64` ~140 LOC.** Bengio-Monperrus-2005-NIPS / Zhang-Zha-2004 §3.1: for each point i, run linalg.PCA on its k-NN ball, return the first intrinsic_dim principal components as the local-tangent-basis Θ_i ∈ R^{d × intrinsic_dim}. Substrate for ML16 HessianLLE + ML18 LTSA + dimension estimators.

**ML7b `manifold/curvature.go::SecondFundamentalFormEstimate(X, n, d, neighbours, tangents) curvature` ~80 LOC.** Cazals-Pouget-2003 estimate Riemannian-curvature via local-quadric-fit; useful for adaptive-perplexity and dimension-validation.

**ML7c `manifold/dimension.go::IntrinsicDimensionLevinaBickel(X, n, d, k_max int) (dim_est []float64)` ~120 LOC.** Levina-Bickel-2004-NIPS MLE-of-intrinsic-dimension: under Poisson-process-on-manifold assumption, `m̂_k(x) = [(1/(k-1)) Σ log(T_k(x)/T_j(x))]^{-1}` where T_j(x) = distance-to-j-th-NN. Returns dim-estimate per-point + per-k; aggregate over k_min ≤ k ≤ k_max recommended.

**ML7d `IntrinsicDimensionMLE(X, n, d int) dim_est_global` ~70 LOC.** Average Levina-Bickel over k-window for global-dimension scalar.

**ML7e `IntrinsicDimensionHeinAudibert(X, n, d, eps float64) dim_est` ~80 LOC.** Hein-Audibert-2005-COLT: angles-between-edges-in-ε-ball converge to spherical-cap-distribution whose moment yields d_intrinsic. Less-bias than Levina-Bickel at high-curvature regions.

### Tier 1 — Classical embedding (~360 LOC)

**ML8 `manifold/mds.go::ClassicalMDS(D2 []float64, n, dim int) (Y []float64)` ~120 LOC.** Torgerson-1952-Psychometrika-17:401 + Gower-1966-Biometrika-53:325. Algorithm:
1. Square the dissimilarity matrix → D²
2. Double-centre: B = -1/2 H D² H where H = I - (1/n)11^T
3. Eigendecompose B (symmetric, PSD if D Euclidean): λ_1 ≥ ... ≥ λ_n
4. Y_i = (√λ_1 u_1[i], ..., √λ_dim u_dim[i]) for top-dim eigenvectors
Recovers ORIGINAL Euclidean coordinates if D is a Euclidean-distance-matrix. Shipping-blocker-soft on `linalg.InverseIteration`. Substrate for ML14 Isomap + ML25 PHATE.

**ML9 `NonMetricMDS(distances [][]float64, n, dim, max_iter int) (Y, stress []float64)` ~140 LOC.** Kruskal-1964-Psychometrika-29:1 + Shepard-1962-Psychometrika-27:125. Iterative-monotone-regression-then-gradient-descent: minimise STRESS-1 σ = √(Σ_{i<j} (d_{ij} − f(δ_{ij}))² / Σ d_{ij}²) where f is monotone-non-decreasing transformation of dissimilarities δ. Pool-Adjacent-Violators algorithm for monotone-regression step.

**ML10 `SammonMapping(distances [][]float64, n, dim, max_iter int) (Y, stress)` ~100 LOC.** Sammon-1969-IEEE-TC-18:401. Loss: E = (1/Σ d_{ij}) Σ (d_{ij} − d̂_{ij})² / d_{ij} weighted-MDS with 1/d_{ij} weights emphasising small-distance preservation. Newton-method gradient-descent (closed-form Hessian-diagonal).

### Tier 2 — Graph-spectral embedding (~720 LOC, SHARED-WITH-271 SHIP-ONCE)

**ML11 `manifold/eigenmap.go::LaplacianEigenmaps(W []float64, n, dim int) Y` ~140 LOC — DUPLICATE-WITH-271-SC17.** Belkin-Niyogi-2003 9000+citations. Compute L_sym = I − D^{-1/2} A D^{-1/2}, take dim+1 smallest eigvecs (drop trivial first), embed as rows. Cross-link to 271/SC17 + 270/W19.

**ML12 `manifold/lle.go::LocallyLinearEmbedding(X, n, d, k, dim int) Y` ~200 LOC — DUPLICATE-WITH-271-SC18.** Roweis-Saul-2000 18000+citations. Per-point: solve constrained-LS for reconstruction-weights from k-NN (Σ_j W_ij = 1). Form sparse M = (I−W)^T(I−W). Embed via dim+1 smallest eigvecs of M (drop trivial first).

**ML13 `manifold/diffusion_map.go::DiffusionMap(W, n, dim int, t float64) Y` ~140 LOC — DUPLICATE-WITH-271-SC19.** Coifman-Lafon-2006 5500+citations. P = D^{-1} A row-stochastic; eigendecompose; Y_i(j) = λ_j^t φ_j(i) for non-trivial eigvecs. The diffusion-time-t parameter is the unique multiscale-knob.

**ML14 `manifold/isomap.go::Isomap(X, n, d, k, dim int) Y` ~140 LOC — DUPLICATE-WITH-271-SC20.** Tenenbaum-deSilva-Langford-2000 19000+citations. (1) Build kNN-graph. (2) Compute geodesic-distance via graph.Dijkstra all-pairs (or graph.FloydWarshall for n<500). (3) ClassicalMDS on geodesic-distance². Reuses ML1 + ML8 + graph.Dijkstra verbatim.

**ML15 `manifold/kernel_pca.go::KernelPCA(K, n, dim int) Y` ~140 LOC — DUPLICATE-WITH-236-K15.** Schölkopf-Smola-Müller-1998 9000+citations. Centre Gram-matrix K_c = H K H. Eigendecompose; project. The unifying-substrate (Ham-Lee-Mika-Schölkopf-2004-ICML "kernel view of dimensionality reduction") under which Isomap = KernelPCA-on-double-centred-geodesic, LLE = KernelPCA-on-(I−W)^T(I−W)-pseudokernel, Eigenmaps = KernelPCA-on-L^†, Diffusion-Map = KernelPCA-on-P^t.

### Tier 3 — LLE-variants + tangent-alignment (~480 LOC, NEW-IN-272)

**ML16 `manifold/lle_variants.go::HessianLLE(X, n, d, k, dim int) Y` ~180 LOC.** Donoho-Grimes-2003-PNAS-100:5591 1500-citations. Replace LLE's reconstruction-weights with discrete-Hessian-of-coordinate-functions estimated on each tangent-space (ML7). The matrix M_HLLE = Σ_i H_i^T H_i; embed via bottom-(dim+1)-eigvecs. **Provably-recovers-isometric-embedding** for non-convex parameter-domains where LLE/Eigenmaps fail (theoretical-advantage; numerically-fragile-on-noisy-data-in-practice). Blocked-soft on linalg.InverseIteration.

**ML17 `ModifiedLLE(X, n, d, k, dim int) Y` ~140 LOC.** Zhang-Wang-2007-NIPS. Standard-LLE has degeneracy when k > d (rank-deficient local-Gram); MLLE uses multiple-weight-vectors per-point regularisation. More-robust than LLE in practice, less-theoretically-strong than HLLE.

**ML18 `LocalTangentSpaceAlignment(X, n, d, k, dim int) Y` ~160 LOC.** Zhang-Zha-2004-SIAM-J-Sci-Comput-26:313 2000-citations. Algorithm: (1) Tangent-basis Θ_i via local-PCA on k-NN (ML7). (2) Local-coordinates Z_i = projection-onto-Θ_i. (3) Alignment-matrix Φ = Σ_i S_i (I − Θ_i Θ_i^T) S_i^T where S_i is selection-matrix mapping global-indices to local-window. (4) Bottom-(dim+1)-eigvecs of Φ give global-embedding. Most-accurate-of-LLE-family on real-data benchmarks (Saul-Roweis-2003 spiral + swissroll + faces).

### Tier 4 — Stochastic-neighbor-embedding (~1,460 LOC, KEYSTONE-MOAT)

**ML19 `manifold/sne.go::StochasticNeighborEmbedding(X, n, d, dim int, perplexity, lr float64, max_iter int) Y` ~220 LOC.** Hinton-Roweis-2002-NIPS the original-SNE pre-t-SNE. Conditional probabilities `p_{j|i} = exp(-‖xi−xj‖²/2σ_i²) / Σ_{k≠i} exp(-‖xi−xk‖²/2σ_i²)`. Same Gaussian-form in low-D. Loss: `Σ_i KL(P_i ‖ Q_i)`. Gradient-descent. Asymmetric `KL(P‖Q) ≠ KL(Q‖P)` is the source of the crowding-problem that t-SNE later fixes.

**ML20 `SymmetricSNE(X, n, d, dim int, perplexity, lr, max_iter) Y` ~100 LOC.** vanDerMaaten-Hinton-2008-§4. Symmetrise to joint distribution `p_ij = (p_{j|i}+p_{i|j}) / (2n)`. Cleaner gradient; still Gaussian-low-D so retains crowding-problem.

**ML21 `manifold/tsne.go::TSNE(X, n, d, dim int, perplexity, lr float64, max_iter int) Y` ~280 LOC — KEYSTONE.** vanDerMaaten-Hinton-2008-JMLR-9:2579 30000+citations. The single-most-cited dimensionality-reduction algorithm of the 2010s. Algorithm:
1. Compute pairwise-distance² in input-space.
2. Per-point Gaussian-kernel with σ_i found by perplexity-binary-search (ML6) → P symmetrised.
3. Initialise Y ~ N(0, 10^{-4} I).
4. Gradient-descent on KL(P‖Q) where Q is **Student-t-with-1-degree-of-freedom**: `q_ij ∝ (1+‖yi−yj‖²)^{-1}`. The heavy-tailed Student-t fixes the crowding-problem (heavy-tail allows moderately-distant points in high-D to be moderately-distant in low-D rather than collapsing).
5. Tricks: early-exaggeration (multiply P by α=12 first 100 iters to encourage cluster-separation), momentum-gradient (β_t = 0.5 ramp to 0.8), adaptive-learning-rate-Jacobs-1988.
Output: 2D-or-3D embedding optimised for **visual cluster-separation**. Distortion of large-distances expected and accepted.

**ML22 `BarnesHutTSNE(X, n, d, dim int, perplexity, theta, lr float64, max_iter int) Y` ~340 LOC.** vanDerMaaten-2014-JMLR-15:3221 4000-citations. The O(n²) gradient of t-SNE has two terms: F_attr = Σ_j p_ij(yi−yj) q_ij Z (sparse over kNN, O(nk)) + F_rep = -Σ_j q_ij²(yi−yj) Z (dense, O(n²)). Replace F_rep computation by **Barnes-Hut quadtree-octree-approximation**: cells with diagonal < θ·distance treated as single-particle-at-centre-of-mass. O(n log n). Enables n=10⁶ in practice. Theta-parameter trades accuracy-vs-speed (default 0.5). Cross-link to slot-153 N-body / slot-263-quasi-MC quadtree-sampling.

**ML23 `manifold/umap.go::UMAP(X, n, d, dim, n_neighbours int, min_dist, lr float64, n_epochs int) Y` ~360 LOC.** McInnes-Healy-Melville-2018-arXiv-1802.03426 8000+citations. Algorithm:
1. Build kNN-graph with n_neighbours (typically 15).
2. For each point, find ρ_i = distance-to-1st-NN, σ_i = adaptive-bandwidth such that Σ_j exp(-(d_{ij}-ρ_i)/σ_i) = log₂(n_neighbours). (UMAP's variant of perplexity.)
3. High-D fuzzy-simplicial-set: `μ_ij = exp(-(d_{ij}-ρ_i)/σ_i)` directed; symmetrise via fuzzy-set-union `μ̂_ij = μ_ij + μ_ji − μ_ij·μ_ji`.
4. Low-D fuzzy-set: `ν_ij = (1 + a‖yi−yj‖^(2b))^{-1}` where (a,b) fit min_dist hyperparameter.
5. Optimise cross-entropy `Σ μ̂_ij log(μ̂_ij/ν_ij) + (1−μ̂_ij) log((1−μ̂_ij)/(1−ν_ij))` via SGD with negative-sampling (sample non-edges to push apart).
6. Output dim-D embedding (typically dim=2, also routinely 3-5-10 for downstream-clustering).
Theoretical-foundation in algebraic-topology (fuzzy simplicial set = Cech/Vietoris-Rips nerve of metric space with adaptive radius; embedding minimises cross-entropy between high-D and low-D nerves). Faster + better-global-structure-preservation than t-SNE in practice; supplanting t-SNE in single-cell-genomics 2018-2026.

**ML24 `LargeVis(X, n, d, dim int, max_iter int) Y` ~80 LOC.** Tang-Liu-Zhang-Mei-2016-WWW. kNN-graph via Annoy random-projection-tree-search (or brute-force for v1) + LINE-style noise-contrastive-embedding (Tang-Wang-Zhang-Yan-Mei-2015-WWW). Predecessor of UMAP, simpler, less-faithful; we ship a minimal version for cross-validation.

**ML25 `PHATE(X, n, d, dim int, t int, gamma float64) Y` ~80 LOC.** Moon-vanDijk-Wang-Krishnaswamy-Burkhardt-Yim-vandenElzen-Hirn-Coifman-Ivanova-Wolf-Krishnaswamy-2019-Nature-Biotech-37:1482 1500-citations. Algorithm:
1. kNN-distance² → α-decay-kernel `K_ij = exp(-(d_ij/σ_i)^α)` α=adaptive.
2. P = D^{-1} K diffusion-operator.
3. Diffusion-time-t (auto-selected via von-Neumann-entropy-elbow).
4. Potential-distance: `U_t = -log(P^t)`, M_ij = ‖U_t[i,:] − U_t[j,:]‖₂ (or KL-divergence variant).
5. ClassicalMDS on M (or non-metric MDS).
Preserves both **local AND global** structure (trajectory-structure for single-cell-RNA-seq) far better than t-SNE/UMAP at the cost of being slower. The de-facto single-cell-RNA-seq trajectory-tool 2020-2026 (krishnaswamy-lab).

### Tier 5 — Maximum-Variance-Unfolding + Riemannian-tSNE (~440 LOC)

**ML26 `manifold/mvu.go::MaximumVarianceUnfolding(X, n, d, k, dim int) Y` ~180 LOC.** Weinberger-Saul-2006-IJCV-70:77 700-citations. SDP:
```
max Tr(K)  s.t.  K ≽ 0, Σ_ij K_ij = 0, K_ii − 2 K_ij + K_jj = ‖x_i−x_j‖² ∀ (i,j) ∈ kNN_edges
```
Maximises variance subject to local-isometry preservation. Closed-form-optimal-kernel-PCA on the resulting Gram-matrix K. **Provably-recovers-isometric-embedding** when manifold is isometric-to-convex-Euclidean-domain. **BLOCKED-HARD on 097-T2 SDP-solver** — defer until SDP infrastructure lands. Workaround: greedy-graph-relaxation Sun-Boyd-Xiao-Diaconis-2006-SIAM-Rev fast-MVU 90% of the way; ships-against-Cholesky-only.

**ML27 `RiemannianTSNE(X, n, d, dim int, perplexity, lr, max_iter) Y` ~140 LOC.** Joncas-Mahoney-2017-ICML "Improved Spectral Convergence Rates for Graph Laplacians on ε-Graphs and k-NN Graphs". Riemannian-geometric formulation of t-SNE — gradient-descent on KL-divergence with respect to Fisher-information-metric on the embedding-manifold (cross-link slot-153 information-geometry). Conformally-equivariant under coordinate-changes; reduces to vanilla t-SNE under unit-metric.

**ML28 `LocalPCA(X_local []float64, k_local, intrinsic_dim int) tangent_basis` ~60 LOC.** Bare primitive used by ML7 + ML16 + ML18. Pure linalg.PCA on a window of points.

**ML29 `GeometricMultidimensionalScaling(distances, n, dim) Y` ~60 LOC.** Bronstein-Bronstein-Kimmel-2006 Riemannian-MDS — preserves geodesic distances under the Riemannian metric of the embedding manifold rather than Euclidean.

### Tier 6 — Evaluation + out-of-sample (~440 LOC, NEW-IN-272)

**ML30 `manifold/eval.go::Trustworthiness(X, Y, n, d, dim, k int) score float64` ~80 LOC.** Venna-Kaski-2001-ICANN. Define "false-friends in Y" = points that ARE in kNN(Y_i) but NOT in kNN(X_i):
```
T(k) = 1 − (2 / (n·k·(2n−3k−1))) · Σ_i Σ_{j ∈ U_k(i)} (r(i,j) − k)
```
where `r(i,j)` = rank of j among ascending-distance from i in input-space, U_k(i) = false-friends-set. T(k) ∈ [0,1], 1 = perfect-trust. The canonical-quality-metric for manifold-learning post-2001.

**ML31 `Continuity(X, Y, n, d, dim, k int) score` ~80 LOC.** Dual of Trustworthiness. "Lost-friends in Y" = points that ARE in kNN(X_i) but NOT in kNN(Y_i). Trust-and-Continuity together characterise local-structure-preservation completely.

**ML32 `LocalContinuityMetaCriterion(X, Y, n, d, dim, k int) score` ~40 LOC.** Chen-Buja-2009-JCGS-18:545. `LCMC(k) = (k/(n−1)) − (k/(n−1))² + (1/(n·k)) Σ_i |U_k^X(i) ∩ U_k^Y(i)|`. Combines T+C into single-scalar; chance-corrected.

**ML33 `KruskalStress(D_orig, D_embed, n) stress` ~30 LOC + `SammonStress(D_orig, D_embed, n) stress` ~30 LOC.** Stress = √(Σ(d_orig − d_embed)²/Σd_orig²) with Sammon-weighted variant.

**ML34 `CoRankingMatrix(X, Y, n, d, dim) Q [][]int` ~80 LOC.** Lee-Verleysen-2009-Neurocomputing-72:1431. Q[k1][k2] = #{(i,j) : rank_X(i,j) = k1 AND rank_Y(i,j) = k2}. The **master joint-rank-distribution** from which Trust + Continuity + LCMC + ranking-correlation are all derived. Diagonal-mass = perfect rank-preservation; off-diagonal-upper = false-friends; off-diagonal-lower = lost-friends.

**ML34b `ProcrustesAlignment(Y1, Y2, n, dim) (R, t, scale, error)` ~60 LOC.** Krzanowski-1979 Procrustes-orthogonal-superimposition: find optimal rotation R + translation t + (optionally) uniform-scale s to minimise Σ‖s·R·Y1_i + t − Y2_i‖². Used to compare two manifold-learners head-to-head (e.g. embedding from t-SNE vs UMAP).

**ML34c `OutOfSampleExtension(X_train, Y_train, X_new, n, m, d, dim, kernel) Y_new` ~80 LOC.** Bengio-Paiement-Vincent-Delalleau-LeRoux-Ouimet-2003-NIPS unified-Nyström-framework. Every manifold-learning algorithm is a kernel-PCA on some adaptive-kernel `k̃(x_i, x_j)`:
- PCA: linear kernel
- MDS: -1/2 d²(x,y) double-centred
- Isomap: -1/2 d_geodesic² double-centred
- LLE: pseudo-kernel from `(I−W)^T(I−W)`
- Eigenmaps: Heat kernel / Laplacian pseudoinverse
- Diffusion: P^t

OOS-extension: `y_new(j) = (1/√λ_j) Σ_{i=1}^{n} u_j(i) k̃(x_new, x_i)`. Gives a way to embed new points without retraining — historically-missing from t-SNE/UMAP (resolved by parametric-tSNE / parametric-UMAP separately).

---

## 2. Connective tissue + cross-package blockers

**Substrate-blocker-1 (HARD)** `linalg.InverseIteration(A, sigma, n, maxIter) []float64` — same blocker as 271-substrate-blocker-1. Currently PRIVATE inside `linalg/pca.go:101-174`. Refactor cost ~80 LOC public-API + dedicated test-vectors. **Gates 9 of 34 primitives** (ML8/11/12/13/14/15/16/17/18). Single-most-important refactor; SHARED-WITH-271/SC8-SC22 + 097-T1.eigvec + 157-PR-4. Ship-once-amortised across 18 + 9 = 27 primitives.

**Substrate-blocker-2 (HARD)** `prob/random.Gaussian` (~80 LOC Box-Muller or Marsaglia-polar). Sixteenth Block-C demand cumulating from 117/184/188/202/215/216/217/227/228/229/230/231/232/233/235/236/269. Gates ML19/20/21/22/23/26 (every t-SNE/UMAP/MVU needs Gaussian-init + perturbation).

**Substrate-blocker-3 (SOFT)** `optim` gradient-descent with momentum + adaptive-learning-rate. Check existence — if `optim.{Momentum, Adam, NesterovAccel}` exist, ML19-23 ship in O(50 LOC) extra; else add ~120 LOC SGD-with-momentum to optim.

**Substrate-blocker-4 (SOFT)** Quadtree / octree primitive for ML22 Barnes-Hut-tSNE. Defer to slot-153 N-body if exists; else ship ~180 LOC quadtree-only inside `manifold/barnes_hut.go` (this is genuinely-isolated-purpose, no other package needs it).

**Substrate-blocker-5 (HARD)** `linalg/sparse` sparse-CSR-matrix + sparse-eigenvector. Most manifold-learners scale to n=10⁵+ only via sparse-Laplacian-Lanczos (097-T1.lanczos). For v1.0 with n≤5000, dense-QR + InverseIteration suffice. Optional-for-v1.0; mandatory-for-v2.0-at-n>10⁵.

**Substrate-blocker-6 (NONE)** `graph.Dijkstra` + `graph.FloydWarshall` PRESENT — substrate for ML14 Isomap geodesic. Complete.

**Substrate-blocker-7 (NONE)** `infogeo.{Kernel, GaussianKernel, MedianHeuristicBandwidth}` PRESENT — substrate for ML5/15/21/23.

**Substrate-blocker-8 (HARD-DEFER)** 097-T2-SDP-solver — gates ML26 MVU exact. Workaround: greedy-graph-relaxation ships against present Cholesky only.

**Total upstream-substrate dependency** (assuming `linalg.InverseIteration` extract + `prob/random.Gaussian` + optional optim-momentum land first as separate PRs): ~280 LOC of substrate-tributary, then ~3,780 LOC of `manifold/` consumer-side closes manifold-learning canon. Plus ~860 LOC SHARED-SHIP-ONCE with 271-Tier-3 + 236-K15 absorbed-into-`manifold/`. Plus optional ~280 LOC for SDP (097-T2 / ML26).

**Cheapest-no-blocker subset:** **ML1+ML8+ML10+ML30+ML31+ML33 ~360 LOC** — k-NN brute-force + ClassicalMDS-via-PCA-route + Sammon (gradient-descent-no-eigvec-needed) + Trustworthiness + Continuity + Kruskal-stress. Pure scalar/vector arithmetic.

**Recommended PR sequence:**

- **PR-A (Tier-0 substrate ~960 LOC, 1 week)** ML1 kNN + ML2 symmetrise + ML3 ε-ball + ML4 adaptive-bandwidth + ML5 Gaussian-similarity + ML6 perplexity + ML7 tangent-local-PCA + ML7c/d/e dimension-estimation. Foundation everything else builds on; zero blockers.
- **PR-B (Tier-1 classical ~360 LOC, 3 days)** ML8 ClassicalMDS + ML9 NonMetricMDS + ML10 Sammon. Ships against present linalg.PCA; no blockers if InverseIteration extracted.
- **PR-C (Tier-2 spectral ~720 LOC, 1 week, SHARED-WITH-271/236)** ML11/12/13/14/15. Ships co-owned with 271-PR-D + 236-PR-rkhs. Single source of truth in `manifold/`; 271 imports.
- **PR-D (Tier-3 LLE-variants ~480 LOC, 1 week)** ML16 HLLE + ML17 MLLE + ML18 LTSA. Strictly downstream of PR-A (ML7 tangent) + PR-C (ML12 LLE).
- **PR-E (Tier-4 SNE family ~1,460 LOC, 3 weeks) — KEYSTONE-MOAT** ML19 SNE + ML20 SymSNE + ML21 t-SNE + ML22 BH-tSNE + ML23 UMAP + ML24 LargeVis + ML25 PHATE. Each algorithm is 1-2 papers and 100-360 LOC. Test against openTSNE 1.0.2 + umap-learn 0.5.6 + PHATE 1.0.11 reference vectors.
- **PR-F (Tier-5 MVU + Riemannian ~440 LOC, 2 weeks)** ML26 MVU-greedy-fallback + ML27 RiemannianTSNE + ML28 LocalPCA + ML29 GeometricMDS. Defer ML26-exact-SDP to 097-T2 PR.
- **PR-G (Tier-6 evaluation ~440 LOC, 1 week)** ML30 Trust + ML31 Continuity + ML32 LCMC + ML33 stress + ML34 co-ranking + ML34b Procrustes + ML34c OOS-extension. Strictly orthogonal infrastructure, ships independent of other tiers. Should land EARLY to validate PR-B/C/D/E.

Total ~4,640 LOC across 7 PRs, ~5-7 engineer-weeks net of substrate.

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled

**Pin 272-1 — Swiss-Roll three-method-agreement (local-structure).** On Swiss-Roll dataset (n=2000 samples, 3D ambient, 2D intrinsic), three local-manifold-learners must produce embeddings that achieve Trustworthiness(k=12) > 0.95 AND Continuity(k=12) > 0.95:
- ML12 LLE
- ML16 HessianLLE
- ML18 LTSA

All three preserve local-neighbourhoods up to permutation. R-3/3-cross-validation pin. Plus reference-validate against `sklearn.manifold.{LocallyLinearEmbedding(method='standard'/'hessian'/'ltsa')}` 1e-3 tolerance (gradient-path-divergence).

**Pin 272-2 — Swiss-Roll global-structure three-method-agreement.** Same dataset, three global-manifold-learners must achieve Procrustes-error vs ground-truth-2D < 0.10:
- ML14 Isomap
- ML11 LaplacianEigenmaps
- ML8 ClassicalMDS-on-Isomap-geodesics

Global-structure-preservation R-3/3 pin.

**Pin 272-3 — t-SNE / UMAP / PHATE clustering-agreement on Mixture-of-Gaussians.** On 5-cluster mixture-of-Gaussians (n=1000, d=20-D, 5 well-separated 20-D Gaussians with small overlap), three modern stochastic-neighbour embedders must recover cluster-structure with Adjusted-Rand-Index > 0.90 when followed by k-means on the 2D embedding:
- ML21 t-SNE
- ML23 UMAP
- ML25 PHATE

R-3/3-cross-validation that all three correctly visualise cluster-structure. Plus reference-validate against `openTSNE.TSNE` + `umap.UMAP` + `phate.PHATE` (1e-3 tolerance for stochastic-init).

**Pin 272-4 — Linear-kernel Kernel-PCA equivalence to PCA.** ML15 KernelPCA with linear-kernel must match `linalg.PCA` to 1e-12 on N=300 samples 20-D Gaussian. Pure linear-substitution sanity check; saturates 3/3 (PCA-via-cov + PCA-via-Gram-matrix + KernelPCA-with-linear-kernel).

**Pin 272-5 — Intrinsic-dimension recovery on known-manifolds.** On Swiss-roll (intrinsic d=2, ambient 3), S-curve (intrinsic d=2, ambient 3), and unit-sphere (intrinsic d=2, ambient 3), THREE dimension-estimators must agree on intrinsic-d=2 ± 0.1:
- ML7c IntrinsicDimensionLevinaBickel
- ML7d IntrinsicDimensionMLE-aggregated
- ML7e IntrinsicDimensionHeinAudibert

R-3/3 cross-validation across estimators + manifolds.

---

## 4. Conceptual identity + competitive-moat statement

**Conceptual identity:** Manifold learning IS the **non-linear generalisation of PCA via adaptive-kernel-PCA**. Every ML algorithm corresponds to a choice of kernel:
- PCA = Linear kernel `k(x,y) = x·y`
- MDS = Distance-pseudo-kernel `-1/2 d²(x,y)` double-centred
- Isomap = Geodesic-distance-pseudo-kernel `-1/2 d_geo²(x,y)` double-centred
- LLE = Local-reconstruction pseudokernel `[(I−W)^T(I−W)]^†`
- Eigenmaps = Laplacian-pseudoinverse `L_sym^†` (heat-kernel as t→∞)
- Diffusion-Map = Markov-power kernel `P^t`
- t-SNE = Gaussian-attraction + Student-t-repulsion (no kernel form; gradient-descent on KL)
- UMAP = Fuzzy-set kernel + adaptive-bandwidth
- PHATE = Diffusion-potential kernel `-log(P^t)`
- MVU = SDP-optimal positive-semidefinite kernel max-Tr(K)

This is the **Ham-Lee-Mika-Schölkopf-2004-ICML-§1 unifying framework**. Every embedding-method is a kernel-PCA in some kernel-space, and out-of-sample-extension via Nyström-formula (ML34c) follows automatically.

**Competitive-moat statement:** Production-quality zero-dep Go implementations of manifold-learning canon exist in **NO library worldwide today**:
- scikit-learn `sklearn.manifold` 1.5.x is MIT-Python-Cython-scipy-dependent
- `umap-learn` 0.5.6 BSD-Python-numba-dependent  
- `openTSNE` 1.0.2 BSD-Python-Cython
- `Rtsne` GPL-R-C++
- `bhtsne` MIT-C++-vanDerMaaten-personal (Barnes-Hut only, no UMAP/PHATE)
- `PHATE` BSD-Python-scipy
- gonum.org/v1 — has linear-PCA only, NO manifold-learning

reality post-PR-D would be the only-zero-dep-Go manifold-learning library worldwide, plus cross-language-validated golden-files (Go ↔ Python ↔ C++ ↔ C#) — UNIQUELY positioned for embedded / kiosk / mobile / WASM deployment where scipy-numba-numpy-stack is unviable.

**2024-2026 frontier:** PHATE (Moon-2019) + Riemannian-tSNE (Joncas-Mahoney-2017) + parametric-UMAP (Sainburg-McInnes-Gentner-2021) + diffusion-VAE-priors (Falorsi-2018) — slot 272 covers PHATE + RiemannianTSNE; parametric-UMAP defers to a future neural-network-coupled-slot.

**Pedagogical entry-point:** Tenenbaum-deSilva-Langford-2000 + Roweis-Saul-2000 simultaneous-Science-papers (December 22 2000 issue) opened the manifold-learning era; vanDerMaaten-Hinton-2008 t-SNE + McInnes-Healy-Melville-2018 UMAP + Moon-2019 PHATE define the 2008-2026 dominant-tools arc. Belkin-Niyogi-2003 Eigenmaps + Coifman-Lafon-2006 Diffusion-Maps + Donoho-Grimes-2003 HLLE + Zhang-Zha-2004 LTSA define the spectral-graph-Laplacian middle-period (2003-2006) before stochastic-neighbour-embedding rose to dominate. Reading order for new contributors: vanDerMaaten-2009 PhD-thesis-survey → Bengio-2003-NIPS unified-out-of-sample → Ham-2004-ICML unified-kernel-view → Lee-Verleysen-2007-book Nonlinear-Dimensionality-Reduction.

**Singular hard-stop blocker:** `linalg.InverseIteration` extraction from PCA-private (097-T1.eigvec). 18 of 34 primitives strict-blocked. Same single-blocker as 271-substrate-blocker-1; double-amortised-fix.
