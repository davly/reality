# 280 — new-sbm (Network Generative Models: SBM, DCSBM, MMSB, Latent-Space, ERGM, Graphon)

## Headline
reality v0.10.0 ships ZERO probabilistic-network-model surface (SBM/DCSBM/MMSB/ERGM/latent-space/graphon all absent); slot 162 enumerated the *generators*, slot 280 owns the *inference / fitting* axis (modularity-Louvain present at `graph/community.go:155` is the only adjacent surface) — entire 1997-2017 statistical-network-models canon is greenfield.

## Findings

### State at HEAD (verified by direct grep on `*.go`)

Repo-wide grep for `SBM | StochasticBlock | BlockModel | DegreeCorrected | MixedMembership | ERGM | ExponentialRandomGraph | LatentSpace | LatentPosition | Graphon | Modularity | Leiden | Infomap | MMSB | DCSBM | Karrer | Newman | Airoldi | Hoff | Raftery | Handcock | Decelle | Massoulié | Peixoto | Bickel | Daudin | Snijders | Nowicki | Matias | Miele` → **only** matches outside review-corpus:

| Surface | Path | SBM-inference relevance |
|---|---|---|
| `LouvainCommunities(adj, weights, n)` | `graph/community.go:155` (~140 LOC) | **Modularity-greedy local-move only** (Blondel-Guillaume-Lambiotte-Lefebvre-2008 *J-Stat-Mech* P10008 first phase only — NO multi-level aggregation, NO refinement, NO resolution parameter). Modularity Q is one inference method for SBM (Newman-2006 *PNAS* 103:8577 spectral-modularity ≡ MAP estimator under uniform-degree planted SBM under Bickel-Chen-2009 *PNAS* 106:21068 profile-likelihood). Closest extant surface to slot 280; gates the cheapest-day-1 PR. |
| `ConnectedComponents / StronglyConnected` | `graph/community.go:15,73` | Tarjan-1972 SCC; substrate for SBM-recovery validity (recovered partition → connected blocks). |
| `EigenvectorCentrality / PageRank` | `graph/centrality.go, graph/pagerank.go` | Power-iteration on adjacency / column-stochastic — largest-eigenvalue only, NOT k-leading-eigvecs needed for spectral-SBM (Lei-Rinaldo-2015 *Ann-Stat* 43:215). |
| `linalg.QRAlgorithm` | `linalg/eigen.go:20` | Householder→TQL symmetric-tridiagonal eigensolver — **eigenvalues only, NO eigenvectors** — keystone-blocker for Lei-Rinaldo spectral-SBM and Newman-2006 modularity-eigenvector. |
| `linalg/pca.go::ipi` (private) | `linalg/pca.go:101-174` | Inverse-iteration eigenvector recovery PRIVATE — slot 097-T1.eigvec / 157-PR-4 refactor extracts as public `linalg.InverseIteration` and unblocks every spectral inference primitive in this slot. |
| `prob.{BetaPDF, BinomialPMF, GammaPDF, PoissonPMF}` | `prob/distributions.go` | Substrate-priors PRESENT for SBM Bayesian inference (Beta-conjugate for block edge-probabilities, Dirichlet ABSENT, Multinomial ABSENT, Categorical ABSENT). |
| `prob.MarkovSimulate` | `prob/markov.go` | **Only LCG sampler in entire prob/ surface** (private + tied to Markov-chain), NOT a public RNG. ERGM-MCMC + latent-space-MCMC + graphon-MCMC all blocked here. |
| Public seeded sampling surface | `prob/sample.go` | **ABSENT** — slot 162-G0 enumerates the bridge file; slot 280 STRICTLY DOWNSTREAM of that ship. |
| Random graph generators (Erdős-Rényi, SBM, MMSB, DCSBM, latent-space, Watts-Strogatz, Barabási-Albert, configuration-model) | `graph/random/*.go` | **ALL ABSENT** — slot 162-G1..G9 enumerated. **Generators are slot-162 territory; slot 280 owns the inference / fitting axis.** |
| Leiden, Infomap, Stochastic-Blockmodel-EM, DCSBM-EM, MMSB-VEM, ERGM-MCMC-MLE, latent-space-MCMC, graphon-empirical-estimator | -- | **ALL ABSENT.** |

**Slot boundaries for clarity:**
- Slot 254 (`new-network-flow` / `graph-cuts`) = max-flow + min-st-cut + α-expansion (DISCRETE-OPTIMISATION, exact).
- Slot 255 (`new-random-graphs`, partner of 162) + slot 162-synergy = **GENERATORS** (forward sampling — Erdős-Rényi, SBM(P, blocks), MMSB(α, B), DCSBM(θ, ω, blocks), Watts-Strogatz, Barabási-Albert).
- Slot 271 (`new-spectral-clustering`) = SC22 SBMRecovery is the **spectral-eigvec back-end** for two-block SBM detection at the Kesten-Stigum threshold.
- Slot 273 (`new-spectral-embedding`) = Adjacency-Spectral-Embedding (Sussman-Tang-Fishkind-Priebe-2012-JASA) ≡ point-estimate of latent-space-model positions.
- **Slot 280 (this) = INFERENCE — fit SBM/DCSBM/MMSB/latent-space/ERGM/graphon to OBSERVED graph.** Mostly orthogonal to 162/254/255/271/273 with strict-downstream-of-162 (uses generators as `assert: gen→fit→recover`).

### What slot 280 OWNS (and 162 / 271 / 273 do not)

162 enumerates the **forward** model: `gen := SBM(blocks, P, seed); A := gen.Sample()`. 280 enumerates the **inverse** model: given observed `A`, recover `(blocks_hat, P_hat)`. Inference algorithms — variational EM (Daudin-Picard-Robin-2008 *Stat-Comput* 18:173), MCMC-Gibbs (Snijders-Nowicki-1997 *J-Class* 14:75), profile-likelihood (Bickel-Chen-2009), spectral (Lei-Rinaldo-2015), nested SBM (Peixoto-2014 *PRX* 4:011047), MCMC-MLE for ERGM (Geyer-Thompson-1992 *JRSS-B* 54:657), Hoff-Raftery-Handcock-2002 *JASA* 97:1090 latent-space MCMC.

### Web context (no MIT pure-Go exists)

- Karrer-Newman-2011 *Phys-Rev-E* 83:016107 — DCSBM + EM-fit; reference C++ `mod` 600 LOC GPL-2.
- Airoldi-Blei-Fienberg-Xing-2008 *JMLR* 9:1981 — MMSB + variational-EM; reference R `lda` GPL-2.
- Hoff-Raftery-Handcock-2002 *JASA* 97:1090 + Handcock-Raftery-Tantrum-2007 *JRSS-A* 170:301 latent-position-cluster-model — `latentnet` R-package GPL-2 ~3,500 LOC.
- ERGM canon (Holland-Leinhardt-1981 *JASA* 76:33 → Frank-Strauss-1986 *JASA* 81:832 → Snijders-Pattison-Robins-Handcock-2006 *Sociol-Methodol* 36:99 curved-ERGM → Hunter-Handcock-2006 *JCGS* 15:565 GLM-ERGM with MCMC-MLE Geyer-Thompson-1992 / Hummel-Hunter-Handcock-2012 step-length stepping algorithm) — `statnet/ergm` R-package GPL-3 ~25,000 LOC.
- Graphon estimation: Bickel-Chen-Levina-2011 *Ann-Stat* 39:2280 method-of-moments; Wolfe-Olhede-2013 (NIPS) sorted-graph-empirical-estimator; Klopp-Tsybakov-Verzelen-2017 *Ann-Stat* 45:316 minimax-rate. Pure-Go = NONE.
- Spectral-SBM: Lei-Rinaldo-2015 *Ann-Stat* 43:215 consistency proof; Massoulié-2014 *STOC* non-backtracking-walk-spectral; Mossel-Neeman-Sly-2018 *Probab-Theory-Relat-Fields* 162:431 belief-propagation-redemption; Decelle-Krzakala-Moore-Zdeborová-2011 *PRL* 107:065701 Kesten-Stigum-threshold `(p−q)² > k(p+(k−1)q)`.
- Dynamic SBM: Yang-Chi-Zhu-Gong-Jin-2011 *KDD* online-Gibbs; Matias-Miele-2017 *JRSS-B* 79:1119 dynamic-SBM-EM with HMM-on-block-membership.
- Reference impls in pure-Go = **none on GitHub** (verified by query "stochastic block model golang"; closest is `gonum/graph` which has no probabilistic models).

## Concrete recommendations

Placement: **NEW sub-package `graph/netmodels/`** (precedent: 254 `graph/cuts/`, 270 `graph/gsp/`, 271 `graph/spectral/`, 273 `graph/embedding/`). Total target ~3,000 LOC across 14 primitives + golden vectors.

### Tier 0 — Modularity completeness (gates Day-1; ZERO blockers)

1. **N1** `graph/netmodels/modularity.go::Modularity(adj, weights, labels []int, n int) float64` — Newman-Girvan-2004 *PRE* 69:026113 modularity Q = 1/(2m) Σ_{ij} (A_ij − k_i k_j / 2m) δ(c_i, c_j). ~40 LOC. **Unblocks**: cross-validation pin against Louvain; fitness metric for every clustering. Zero blockers — present `IntAdjacency` + `weights map[[2]int]float64` (same shape as `LouvainCommunities`).
2. **N2** `graph/netmodels/modularity.go::ModularityResolution(adj, weights, labels, gamma) float64` — Reichardt-Bornholdt-2006 *PRE* 74:016110 resolution-limit-aware modularity Q_γ = 1/(2m) Σ_{ij} (A_ij − γ k_i k_j / 2m) δ. ~25 LOC. **Unblocks**: Fortunato-Barthélemy-2007 resolution-limit experimentation.
3. **N3** `graph/netmodels/louvain_full.go::LouvainHierarchical(adj, weights, n, gamma, maxLevels int) ([][]int, []float64)` — full Blondel-2008 with **multi-level aggregation** (current `LouvainCommunities` is phase-1-only). Returns dendrogram + Q-per-level. ~180 LOC. **Unblocks**: hierarchical community detection; consumes N1+N2.
4. **N4** `graph/netmodels/leiden.go::LeidenCommunities(adj, weights, n, gamma, rng Source) []int` — Traag-Waltman-vanEck-2019 *Sci-Rep* 9:5233 refinement-step + well-connected-guarantee (provably-fixes-Louvain disconnected-communities). ~280 LOC. **Unblocks**: production-grade community detection (default in `igraph`/`networkx` as of 2022); R-MUTUAL pin against Louvain on benchmark.
5. **N5** `graph/netmodels/spectral_modularity.go::NewmanSpectralSplit(adj, n int) ([]int, float64)` — Newman-2006 *PNAS* 103:8577 leading-eigenvector of modularity matrix B = A − kk^T/2m, recursive bisection by sign. ~120 LOC. **Blocks-on**: `linalg.InverseIteration` (097-T1 / 157-PR-4 extraction from `linalg/pca.go::ipi`). **Unblocks**: cross-validation against Louvain/Leiden.

### Tier 1 — Vanilla SBM inference (Daudin VEM + Bickel-Chen profile)

6. **N6** `graph/netmodels/sbm_em.go::FitSBM(adj, n, k int, maxIter int, tol float64) (labels []int, P [][]float64, logLik float64)` — Daudin-Picard-Robin-2008 *Stat-Comput* 18:173 variational-EM for vanilla planted-partition SBM with Bernoulli-edge model. E-step variational mean-field; M-step closed-form τ̂_a = Σ z_ia / n, P̂_ab = Σ_{i≠j} z_ia z_jb A_ij / Σ z_ia z_jb. ~280 LOC. **Blocks-on**: `prob.LogBernoulli` + Dirichlet-prior helper (slot 162-G0 bridge). **Unblocks**: every downstream consumer.
7. **N7** `graph/netmodels/sbm_profile.go::FitSBMProfileLikelihood(adj, n, k int, maxIter int) (labels []int, logLik float64)` — Bickel-Chen-2009 *PNAS* 106:21068 profile-likelihood with greedy-swap optimisation (Zhao-Levina-Zhu-2012 *Ann-Stat* 40:2266 consistency). ~180 LOC. Pairs with N6 as cross-validation.

### Tier 2 — Spectral SBM (BLOCKED on slot 097 Eigvec)

8. **N8** `graph/netmodels/sbm_spectral.go::SBMSpectralRecovery(adj, n, k int) (labels []int, eigvals []float64)` — Lei-Rinaldo-2015 *Ann-Stat* 43:215 (k-leading-adjacency-eigvecs → k-means on rows; consistency proven for K-block-SBM at sparse-regime). ~180 LOC. **Blocks-on**: `linalg.InverseIteration` + `linalg/cluster.KMeansLloyd` (097-T1 + 157-G8 + 271-SC29). Identical algorithmic structure to 271-SC22 — **ship-once, both slots cite**.
9. **N9** `graph/netmodels/sbm_threshold.go::KestenStigumThreshold(p, q float64, k int) (detectable bool, gap float64)` — Decelle-Krzakala-Moore-Zdeborová-2011 *PRL* 107:065701 phase-transition diagnostic: detectable iff `(p − q)² > k(p + (k−1)q)`. ~30 LOC scalar test, **ZERO blockers**. **Unblocks**: regression test for N6/N8 (above-threshold ⇒ recovery > 0.7; below ⇒ ≈ 0.5).

### Tier 3 — DCSBM (Karrer-Newman 2011)

10. **N10** `graph/netmodels/dcsbm_em.go::FitDCSBM(adj, n, k int, maxIter int) (labels []int, theta []float64, omega [][]float64, logLik float64)` — Karrer-Newman-2011 *PRE* 83:016107 degree-corrected SBM with Poisson-edge model + EM. E-step variational; M-step closed-form θ̂_i = k_i / Σ_{j∈c(i)} k_j and ω̂_ab = m_ab. ~320 LOC. **Blocks-on**: N6 (shared E-step structure) + Poisson-PMF (PRESENT). Critical for biological networks where degree-heterogeneity dominates (gene-regulatory networks, protein-interaction networks). **R-PIN**: under uniform-degree planted SBM → DCSBM ≡ SBM (regression test against N6).
11. **N11** `graph/netmodels/dcsbm_spectral_init.go::DCSBMSpectralInit(adj, n, k int) []int` — Qin-Rohe-2013 *NIPS* normalized-Laplacian-regularised-spectral init for DCSBM (degree-corrected variant of N8). ~120 LOC. **Blocks-on**: `linalg.InverseIteration` + KMeansLloyd. **Unblocks**: warm-start for N10 EM (Karrer-Newman-§IV recommends spectral init).

### Tier 4 — Mixed-membership (Airoldi 2008)

12. **N12** `graph/netmodels/mmsb_vem.go::FitMMSB(adj, n, k int, alpha []float64, maxIter int) (theta [][]float64, B [][]float64, logLik float64)` — Airoldi-Blei-Fienberg-Xing-2008 *JMLR* 9:1981 mixed-membership SBM with variational-EM (per-edge mixed-membership θ_i ~ Dir(α), z_{i→j} ~ Cat(θ_i), z_{j→i} ~ Cat(θ_j), A_ij ~ Bern(B[z_{i→j}, z_{j→i}])). ~360 LOC. **Blocks-on**: `prob.DirichletLogPDF` + `prob.LogSumExp` (147-prob-numerics review flag); E-step uses `digamma` (PRESENT in `prob/mathutil.go` — verify). Captures soft-membership for nodes belonging to multiple communities.

### Tier 5 — Latent-space (Hoff-Raftery-Handcock 2002)

13. **N13** `graph/netmodels/latent_space.go::FitLatentSpace(adj, n, dim int, mcmcIters int, rng Source) (positions [][]float64, alpha float64, accept float64)` — Hoff-Raftery-Handcock-2002 *JASA* 97:1090 distance model `logit P(A_ij=1) = α − ‖z_i − z_j‖`; MCMC-Metropolis-Hastings on `(z_i, α)`. ~340 LOC. **Blocks-on**: `prob.NormalSample` (Box-Muller — slot 173 / 161 pin), `prob.UniformSample`. Cross-link: 273 (Adjacency-Spectral-Embedding gives consistent point-estimate of `z_i` — use as MCMC init).
14. **N14** `graph/netmodels/latent_position_cluster.go::FitLPCM(adj, n, dim, k int, mcmcIters int, rng) (positions, clusters, mu, sigma2)` — Handcock-Raftery-Tantrum-2007 *JRSS-A* 170:301 latent-position-cluster-model with Gaussian-mixture on positions. ~280 LOC. **Blocks-on**: N13 + Gaussian-mixture inference.

### Tier 6 — Dynamic SBM (Matias-Miele 2017)

15. **N15** `graph/netmodels/dynamic_sbm.go::FitDynamicSBM(adjT [][][]int, n, k, T int, maxIter int) (labels [][]int, P [][]float64, transition [][]float64, logLik float64)` — Matias-Miele-2017 *JRSS-B* 79:1119 dynamic-SBM with HMM-on-block-membership: π_i^t = transition · π_i^{t−1}; A^t_ij | z^t ~ Bern(P[z^t_i, z^t_j]). EM with forward-backward on block-membership. ~360 LOC. **Blocks-on**: N6 + present `prob/markov.go` HMM substrate (verify ForwardBackward). Cross-link: slot 252 / 268 HMM-extensions.

### Tier 7 — ERGM (Holland-Leinhardt 1981 → Hunter-Handcock 2006)

16. **N16** `graph/netmodels/ergm.go::FitERGM(adj, n int, stats []StatFn, mcmcIters int, rng Source) (theta []float64, vcov [][]float64, accept float64)` — Hunter-Handcock-2006 *JCGS* 15:565 GLM-ERGM with MCMC-MLE Geyer-Thompson-1992 *JRSS-B* 54:657. Sufficient stats include edges, triangles, k-stars, geometrically-weighted-degree. ~480 LOC. **Blocks-on**: full RNG surface + Cholesky for sandwich-estimator (PRESENT) + Hummel-Hunter-Handcock-2012 step-length stepping. Single most-complex primitive in slot.
17. **N17** `graph/netmodels/ergm_stats.go::{EdgeCount, TriangleCount, KStar, GWDegree, GWESP}` — sufficient-statistic computers. ~160 LOC. **Blocks-on**: triangle counting (substrate ABSENT — verify; if absent flag as separate primitive).

### Tier 8 — Graphon estimation (Bickel-Chen-Levina 2011, Wolfe-Olhede 2013)

18. **N18** `graph/netmodels/graphon_sba.go::EstimateGraphonSBA(adj, n, k int) (W [][]float64, blockBoundaries []int)` — Wolfe-Olhede-2013 stochastic-block-approximation graphon estimator (degree-sort + block-average). ~180 LOC. **Blocks-on**: nothing exotic; sort-by-empirical-degree then block-average. **R-PIN**: under SBM ground-truth, recovered W converges to step-function P-matrix.
19. **N19** `graph/netmodels/graphon_usvt.go::EstimateGraphonUSVT(adj, n int, threshold float64) [][]float64` — Chatterjee-2015 *Ann-Stat* 43:177 universal-singular-value-thresholding. ~120 LOC. **Blocks-on**: full SVD (slot 097-T1 + 261 online-SVD share substrate).

### Tier 9 — Cross-language golden-vector test surface

20. **N20** `graph/netmodels/testdata/golden/*.json` — minimum 30 vectors per primitive over **planted-partition n=200, K=2/3/4** with known `(blocks, P)`; **karate-club n=34** classic-benchmark; Lazega-lawyers n=71 ERGM-canon; Sampson-monks n=18 dynamic-SBM-canon; **Newman-Girvan benchmark** (n=128, k=4, mu∈[0.0, 0.5] mixing parameter). Tolerance: NMI ≥ 0.95 above-threshold, NMI ≥ 0.5 below; loglik tol 1e-6 (EM-converged).

## Cross-cutting

- **Slot 162 (synergy-graph-prob)** ← strict-upstream: 162-G0 `prob/sample.go` + 162-G1..G9 generators are PREREQUISITE for slot-280 fit→recover R-PINs (cannot test SBMRecovery without SBMGenerator). Coordinate-once: ship 162's `graph/random/` first, then 280 builds `graph/netmodels/` on top using same `IntAdjacency` + same RNG surface.
- **Slot 097-T1 (linalg-missing)** ← BLOCKER for N5 (NewmanSpectralSplit), N8 (SBMSpectralRecovery), N11 (DCSBMSpectralInit). The `linalg.InverseIteration` extraction from `linalg/pca.go::ipi` private code path is named-blocker; ~30 LOC refactor unblocks 3 slot-280 primitives + every spectral-clustering primitive in 271 + every spectral-embedding primitive in 273.
- **Slot 271 (new-spectral-clustering)** ← N8 SBMSpectralRecovery is the **same algorithm** as 271-SC22; ship-once-cite-twice. Same precedent for N11 ↔ 273-spectral-embedding.
- **Slot 173 (Box-Muller / Normal sampler)** + **Slot 161 (synergy-control-prob)** + **Slot 037-perf (Cholesky)** ← BLOCKERS for N13/N14 latent-space MCMC.
- **Slot 254 (graph-cuts)** is COMPLEMENTARY: ratio-cut/Ncut continuous-relaxation ↔ slot-280 generative-model is the **probabilistic-model-counterpart** to 254's combinatorial-cut. Both can recover the same partition on planted-SBM; cross-validation pin opportunity.
- **Slot 273 (new-spectral-embedding)** ← Adjacency-Spectral-Embedding (Sussman-Tang-Fishkind-Priebe-2012-JASA) gives consistent point-estimate of `z_i` in latent-space-model; warm-start for N13 MCMC.
- **Slot 268 (new-hmm-extensions)** ← N15 dynamic-SBM uses HMM on block-membership; share forward-backward substrate.
- **Slot 199 (synergy-graph-info)** ← Modularity ≡ map-equation under specific limits (Rosvall-Bergstrom-2008 *PNAS* 105:1118 Infomap = compression interpretation). Cross-link.
- **Pistachio relationship-graphs / fraud-detection / gene-regulatory networks / trade-flow / co-authorship**: DCSBM (N10) is the workhorse model for any network with degree-heterogeneity (which is essentially every real-world network).

### Singular cheapest day-1 PR (~470 LOC, ZERO blockers)

**N1 + N2 + N3 + N9 + N4** = full Modularity surface (Newman-Girvan + Reichardt-Bornholdt + hierarchical Louvain + Kesten-Stigum threshold + Leiden). All against present `IntAdjacency`, `weights map[[2]int]float64`, no eigvec, no RNG (Leiden uses `Source` from 162-G0 if landed, otherwise sequential-deterministic order). Modularity is the foundational fitness metric — every other primitive in this slot validates against it. Ship-once unlocks Tier-0 + cross-validation tests.

### Singular architectural keystone

**N6 FitSBM (Daudin VEM)** ~280 LOC — the canonical SBM-inference primitive that every Tier-3+ extension generalises (DCSBM = SBM + degree-correction; MMSB = SBM + Dirichlet-prior; dynamic-SBM = SBM + HMM-on-membership). Single most-templating primitive. Blocks on `prob.LogSumExp` + Dirichlet-PDF (147-flag).

### Singular 2024-frontier

**N18 Wolfe-Olhede graphon SBA + N19 Chatterjee USVT** ~300 LOC — graphon estimation is the **non-parametric limit** of SBM (`k → ∞`); 2013-2017 vintage; no MIT pure-Go implementation worldwide.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

1. **Pin-280-1 (Modularity ≡ Louvain ≡ Leiden ≡ Newman-spectral on karate-club n=34)** — Q=0.4198 (Newman-2006 result); all 4 implementations must agree to 1e-3.
2. **Pin-280-2 (SBM-EM ≡ SBM-spectral ≡ SBM-modularity-Louvain on planted n=200, K=3, p=0.3, q=0.05)** — recovery NMI ≥ 0.95 for all three; agreement on labels within 2 swaps.
3. **Pin-280-3 (DCSBM ≡ SBM under uniform-degree planted SBM)** — when θ_i ≡ 1 ∀i, DCSBM-EM = SBM-EM logLik to 1e-6.
4. **Pin-280-4 (Kesten-Stigum threshold sharpness)** — for `(p−q)² > k(p+(k−1)q)` ⇒ N6/N8 recovery > 0.7; below ⇒ ≈ 0.5 (within 0.05).
5. **Pin-280-5 (Latent-space ≡ Adjacency-Spectral-Embedding for Hoff-2002 with d=2)** — N13 MCMC posterior mean ≈ 273-ASE point estimate (Procrustes-aligned, RMSE < 0.05).
6. **Pin-280-6 (Graphon-SBA ≡ Graphon-USVT under SBM ground-truth)** — both recover step-function P-matrix at MISE < 1e-3.
7. **Pin-280-7 (ERGM Bernoulli-only ≡ Erdős-Rényi MLE)** — N16 with `stats=[EdgeCount]` should recover `θ̂ = logit(p̂)` matching closed-form ER MLE to 1e-4.

### Cross-link consumers (downstream)

- **Pistachio relationship graphs** ← N4 Leiden + N10 DCSBM for hierarchical user-community detection.
- **Fraud detection** ← N16 ERGM + N15 dynamic-SBM (anomalous block-transitions).
- **Gene-regulatory networks** ← N10 DCSBM (degree-heterogeneity dominant).
- **Trade-flow / co-authorship** ← N12 MMSB (countries / authors belong to multiple communities).
- **Recommendation systems** ← N13/N14 latent-space + LPCM (matrix-factorisation interpretation).
- **Communication networks** ← N15 dynamic-SBM + N16 ERGM-with-temporal-stats.
- **Cluster-validation slot (~ if exists)** ← N1 Modularity + NMI/ARI metrics from prob/cluster (cross-link).

## Sources

### Repo files cited
- `graph/community.go:15` ConnectedComponents, `:73` StronglyConnected, `:155` LouvainCommunities (phase-1-only)
- `graph/centrality.go` EigenvectorCentrality
- `graph/pagerank.go` PageRank
- `graph/types.go` IntAdjacency
- `linalg/eigen.go:20` QRAlgorithm (eigvals only)
- `linalg/pca.go:101-174` ipi (private inverse-iteration — extraction blocker for N5/N8/N11)
- `prob/distributions.go` Beta/Binomial/Poisson/Gamma PMFs (substrate)
- `prob/markov.go` MarkovSimulate (only LCG sampler, private)
- `reviews/overnight-400/agents/162-synergy-graph-prob.md` (G0..G9 generators — strict upstream)
- `reviews/overnight-400/agents/271-new-spectral-clustering.md` (SC22 SBMRecovery — ship-once shared)
- `reviews/overnight-400/agents/273-new-spectral-embedding.md` (ASE for latent-space init)
- `reviews/overnight-400/agents/083-graph-sota.md:66` (graph-tool / Peixoto SBM canon)
- `reviews/overnight-400/agents/097-linalg-missing.md` (T1.eigvec extraction blocker)
- `reviews/overnight-400/agents/199-synergy-graph-info.md` (modularity ≡ map-equation cross-link)

### Web / literature (no MIT pure-Go implementations exist)
- Newman-Girvan-2004 *PRE* 69:026113 — modularity Q.
- Reichardt-Bornholdt-2006 *PRE* 74:016110 — resolution-aware modularity.
- Newman-2006 *PNAS* 103:8577 — spectral-modularity-eigenvector.
- Blondel-Guillaume-Lambiotte-Lefebvre-2008 *J-Stat-Mech* P10008 — Louvain (multi-level).
- Traag-Waltman-vanEck-2019 *Sci-Rep* 9:5233 — Leiden refinement.
- Snijders-Nowicki-1997 *J-Class* 14:75 — MCMC-Gibbs SBM.
- Daudin-Picard-Robin-2008 *Stat-Comput* 18:173 — variational-EM SBM.
- Bickel-Chen-2009 *PNAS* 106:21068 — profile-likelihood SBM consistency.
- Karrer-Newman-2011 *PRE* 83:016107 — degree-corrected SBM + EM.
- Decelle-Krzakala-Moore-Zdeborová-2011 *PRL* 107:065701 — Kesten-Stigum threshold.
- Lei-Rinaldo-2015 *Ann-Stat* 43:215 — spectral-SBM consistency.
- Massoulié-2014 *STOC* — non-backtracking-walk-spectral.
- Mossel-Neeman-Sly-2018 *Probab-Theory-Relat-Fields* 162:431 — belief-propagation-redemption.
- Qin-Rohe-2013 *NIPS* — regularised-spectral DCSBM.
- Peixoto-2014 *PRX* 4:011047 — nested-SBM + MDL.
- Peixoto-2017 *arXiv* 1705.10225 — non-parametric SBM.
- Airoldi-Blei-Fienberg-Xing-2008 *JMLR* 9:1981 — MMSB variational-EM.
- Hoff-Raftery-Handcock-2002 *JASA* 97:1090 — latent-space MCMC.
- Handcock-Raftery-Tantrum-2007 *JRSS-A* 170:301 — latent-position-cluster-model.
- Sussman-Tang-Fishkind-Priebe-2012 *JASA* 107:1119 — adjacency-spectral-embedding.
- Frank-Strauss-1986 *JASA* 81:832 — Markov graphs (early ERGM).
- Geyer-Thompson-1992 *JRSS-B* 54:657 — MCMC-MLE.
- Snijders-Pattison-Robins-Handcock-2006 *Sociol-Methodol* 36:99 — curved-ERGM.
- Hunter-Handcock-2006 *JCGS* 15:565 — GLM-ERGM.
- Hummel-Hunter-Handcock-2012 *JCGS* 21:920 — step-length-stepping.
- Bickel-Chen-Levina-2011 *Ann-Stat* 39:2280 — graphon method-of-moments.
- Wolfe-Olhede-2013 *NIPS* — sorted-graph-empirical-graphon.
- Chatterjee-2015 *Ann-Stat* 43:177 — universal-singular-value-thresholding.
- Klopp-Tsybakov-Verzelen-2017 *Ann-Stat* 45:316 — graphon minimax.
- Yang-Chi-Zhu-Gong-Jin-2011 *KDD* — online-Gibbs dynamic-SBM.
- Matias-Miele-2017 *JRSS-B* 79:1119 — dynamic-SBM-EM (HMM-membership).
- Rosvall-Bergstrom-2008 *PNAS* 105:1118 — Infomap (cross-link 199).
- Fortunato-Barthélemy-2007 *PNAS* 104:36 — modularity resolution-limit.
- Zhao-Levina-Zhu-2012 *Ann-Stat* 40:2266 — profile-likelihood consistency.

### Reference implementations (none are MIT pure-Go zero-dep)
- `graph-tool` (Peixoto, GPL-3, C++/Python, ~80k LOC) — gold-standard nested-SBM.
- `statnet/ergm` (R, GPL-3, ~25k LOC) — gold-standard ERGM-MCMC-MLE.
- `latentnet` (R, GPL-3, ~3.5k LOC) — Hoff-Raftery-Handcock latent-space.
- `igraph` (C/Python/R, GPL-2) — Leiden, Louvain, Infomap.
- `leidenalg` (Python/C++, GPL-3) — Leiden reference.
- `sklearn` — has no SBM/DCSBM/ERGM (graphs not first-class in scikit-learn).
- Pure-Go: `gonum/graph` has community detection (modularity-only) but ZERO probabilistic models. **No MIT pure-Go zero-dep implementation of any primitive in this slot exists worldwide.**
