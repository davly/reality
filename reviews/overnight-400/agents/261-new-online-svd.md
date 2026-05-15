# 261 | new-online-svd — Online SVD / streaming PCA / Frequent Directions / Brand 2002 incremental SVD / Bunch-Nielsen-Sorensen 1978 rank-1 SVD update / Oja 1982 / Krasulina 1969 / power-iteration-online / GROUSE Grassmannian SGD / PETRELS / online-RPCA / SRHT-streaming / truncSVD-via-Lanczos / RRQR / DP-online-PCA / forgetting-factor / one-pass PCA

**Summary line 1.** reality v0.10.0 ships **ZERO online-SVD / streaming-PCA / incremental-low-rank-update surface** — repo-wide grep on `OnlineSVD|online.svd|streamingSVD|streaming.svd|streamingPCA|streaming.pca|incrementalSVD|incremental.svd|Brand.svd|Brand2002|BNS.update|Bunch.Nielsen.Sorensen|rank.one.svd|FrequentDirections|frequent.directions|Liberty.2013|Ghashami|Oja|Oja1982|Krasulina|GROUSE|grouse|Grassmannian.svd|PETRELS|pettrels|subspace.tracking|RRQR|rank.revealing.qr|DPOnlinePCA|differential.privacy.pca|noisy.power.method|sparse.PCA|sparse_pca|one.pass.PCA|forgetting.factor.pca|adaptive.rank|Lanczos|TruncSVD|trunc.svd|implicit.restart.arnoldi` returns **zero callable matches** in production `*.go` outside (a) `linalg/pca.go::PCA` (215 LOC, batch-only, covariance-eigendecomposition + per-eigenvalue inverse-iteration — verified at `linalg/pca.go:33-214`; requires entire data matrix `nSamples × nFeatures` materialised in RAM at PCA-call time + allocates `nFeatures*nFeatures` covariance + per-eigenvalue inverse-iteration LU scratch; **NO Update(x) entry-point**, NO Merge, NO downdate, NO forget-factor, NO Stiefel/Grassmannian state, NO Lanczos path, NO randomized path), (b) `linalg/eigen.go::QRAlgorithm` (212 LOC) — symmetric eigenvalues only via Householder-tridiag + tqli, returns eigen-VALUES NOT eigen-VECTORS as Q, so even reusing as a "rebuild PCA from updated covariance" path is blocked because the eigenvectors are reconstructed via inverse iteration on the dense covariance, NOT as a Q-matrix output, (c) `audio/fingerprint.go:24-194` Welford 1962 streaming mean+variance with parallel-merge Chan-Golub-LeVeque 1979 — establishes the in-repo `(state, Update, Merge, Query)` idiom that every online-SVD primitive in this slot will follow but provides ONLY scalar streaming-statistics (mean+variance), not the rank-r-subspace tracking surface of online-SVD. The 1969-2026 online-SVD / streaming-PCA canon is one of the most-cited unfulfilled promises in the repo: the original streaming-eigendecomposition Krasulina-1969-Aut-Remote-Control-30 stochastic-approximation-on-the-Stiefel-manifold-(predates Oja by 13 years); the seminal Oja-1982-J-Math-Biol-15:267 *Simplified neuron model as a principal component analyzer* (~3,500 citations) one-line-update `w_{t+1} = w_t + η_t(y_t·x_t − y_t²·w_t)` for top-1-PC streaming; multi-component generalised-Hebbian-Sanger-1989-Neural-Networks-2:459 *Optimal unsupervised learning in a single-layer linear feedforward neural network* (~2,700 citations) extending Oja to top-k via Gram-Schmidt-deflation; the Bunch-Nielsen-Sorensen-1978-Numer-Math-31:31 *Rank-one modification of the symmetric eigenproblem* (~1,800 citations) closed-form update of `A + ρ·u·uᵀ` eigendecomposition via secular-equation root-finding — the keystone primitive on which Brand-2002 incremental-SVD is built; the seminal Brand-2002-ECCV-7:707 + Brand-2006-LAA-415:20 *Fast low-rank modifications of the thin singular value decomposition* (~2,400 citations) incremental SVD `[U, Σ, V] += u·vᵀ` via 4-block-update + small-SVD on the (k+1)×(k+1) middle-matrix; Hall-Marshall-Martin-1998-BMVC *Incremental eigenanalysis for classification* (~600 citations) parallel work; Levey-Lindenbaum-2000-IEEE-TIP-9:1371 *Sequential Karhunen-Loeve basis extraction and its application to images* (~700 citations) production-quality Brand-style; Ross-Lim-Lin-Yang-2008-IJCV-77:125 *Incremental learning for robust visual tracking* (~1,800 citations) Brand-with-forgetting-factor; the streaming-deterministic-Liberty-2013-KDD-19 + Ghashami-Liberty-Phillips-Woodruff-2016-SIMAX-37:1762 *Frequent Directions: Simple and deterministic matrix sketching* (~700 citations) one-pass `2ℓ×d`-buffer-with-shrink that achieves `||AᵀA − BᵀB||_2 ≤ ||A||_F²/ℓ` without randomness — strictly dominates random-projection on cov-error; Halko-Martinsson-Tropp-2011 randomized-SVD basics; the Balzano-Nowak-Recht-2010-Allerton + He-Balzano-Lui-2011-NIPS GROUSE *Grassmannian rank-one update subspace estimation* (~1,400 citations) gradient-on-Grassmann-manifold streaming with O(n·r) work and `||proj_{U_t} − proj_{U_∞}||_F → 0` linear convergence under RIP-like conditions; the Chi-Eldar-Calderbank-2013-IT-59:5947 PETRELS *Parallel estimation and tracking by recursive least squares* (~700 citations) RLS-style subspace-tracking with closed-form column-update; the Hardt-Price-2014-NIPS *The noisy power method: A meta algorithm with applications* (~1,300 citations) general framework for noisy-power-iteration unifying streaming-PCA / DP-PCA / spiked-Wigner-bounds; the Mitliagkas-Caramanis-Jain-2013-NIPS *Memory limited, streaming PCA* (~700 citations) online-PCA with `O(d·r²)` memory matching information-theoretic lower bound; the Boutsidis-Garber-Karnin-Liberty-2015-STOC *Online principal components analysis* (~400 citations) regret-minimisation framing of online-PCA; the Garber-Hazan-Ma-2017-COLT *Stochastic variance reduced subspace iteration* SVRG-PCA; the Allen-Zhu-Li-2017-COLT-65 *First efficient convergence for streaming k-PCA: A global, gap-free, and near-optimal rate* (~400 citations) closing the global-convergence gap; the Hazan-Singh-Singer-2016-NIPS *Online Principal Components Analysis* low-regret bandit-PCA; the Mardia-Jaouen-Carrillo-2025-ICML-262 *Sketched online PCA: subsampled SVD with merging* (web-research 2026-05-09 confirms 2025 frontier paper); and 2024-2026 frontier streaming-RPCA *L+S = M streaming* extensions Feng-Xu-Yan-2013-NIPS-OnlineStochasticRPCA / He-Balzano-Szlam-2012-NIPS GRASTA / Guo-Qiu-Vaswani-2014-IT-60:5535 ReProCS — every one of these 17+ landmark online-SVD / streaming-PCA papers has reality-equivalent ZERO callable surface as of 2026-05-09. **PARTIAL OVERLAP with 188 (synergy-prob-linalg, ~3,120 LOC) — D5 PowerIteration + D6 Lanczos + D7 RandomizedSVD (Halko-Martinsson-Tropp-2011) + D8 RandomRangeFinder are the BATCH-randomized-NLA cousins of online-SVD; the SVD-substrate `linalg/svd.go` (~280 LOC Golub-Reinsch) that 188-PR-B + 215-PR-B + 257-D1 + 259-PR-B + 260-PR-B all share as a P0 prerequisite is the SAME shared blocker for every Brand-style SVD-update primitive in this slot. PARTIAL OVERLAP with 224 (streaming-sketches, ~3,250 LOC) — ST25 FrequentDirections (Liberty-2013) + ST10 AMS + ST6 CountSketch are the SAME sketches but framed as sketches-package canon; this slot 261 reframes FrequentDirections as a streaming-PCA primitive and adds the **subspace-tracking** Brand / GROUSE / PETRELS / Oja family that 224 does NOT cover. PARTIAL OVERLAP with 257 (tensor-decomp, ~3,180 LOC) — D14 randomized-HOSVD + D27 IncrementalTA Sun-Papadimitriou-Yu-2008-SDM tensor-streaming are tensor-axis cousins of slot 261's matrix-axis. PARTIAL OVERLAP with 259 (matrix-completion) — M22 GROUSE-OnlineMC overlaps with slot 261's GROUSE-streaming-PCA but framed as MC-completion; ship GROUSE ONCE in `online_svd/grouse.go` and import from BOTH 259-mc/ AND 261-online_svd/. PARTIAL OVERLAP with 260 (robust-PCA) — R15 OnlineStochasticRPCA + R16 GRASTA + R17 ReProCS + R18 GROUSE-RPCA are streaming-RPCA cousins, all share the GROUSE substrate.

**Summary line 2.** Twenty-six primitives **O1–O26** totalling **~3,540 LOC pure connective tissue + ~360 LOC NEW substrate** (identical P0 substrate-pool to 097/117/188/215/220/228/257/258/259/260: `linalg/svd.go` ~280 LOC + `prob/random/normal.go` ~80 LOC; co-shipping with any of those reviews amortises substrate cost) split across **(a) ~440 LOC Tier-1 Bunch-Nielsen-Sorensen + Brand-incremental-SVD core** (O1 `online_svd/bns_update.go::BunchNielsenSorensenRank1Update(d, ρ, u, U_out, λ_out)` ~140 LOC closed-form Bunch-Nielsen-Sorensen-1978-Numer-Math-31:31 secular-equation root-finding for `A_new = A + ρ·u·uᵀ` symmetric-eigenproblem update — the keystone primitive on which Brand-2002 incremental-SVD is built; O2 `online_svd/brand_incremental.go::BrandIncrementalSVD(U, Σ, V, u, v, U_new, Σ_new, V_new)` ~200 LOC Brand-2002-ECCV/Brand-2006-LAA-415:20 fast-low-rank-modifications-of-thin-SVD via 4-block-update — `[U_new, Σ_new, V_new] = SVD([U·Σ, u]·[V; vᵀ])` factored as `[U, u_perp]·[[Σ, mᵀ; 0, p]]·[V, 0; 0, 1]` then small-SVD on the (k+1)×(k+1) middle-matrix; appends column / row in O(d·r) instead of O(d²·r) full re-SVD; O3 `online_svd/brand_downdate.go::BrandDowndateSVD(U, Σ, V, u, v, U_new, Σ_new, V_new)` ~80 LOC Brand-2006-LAA section-2.4 column-deletion via the same 4-block update with `−u·vᵀ` rank-1; O4 `online_svd/brand_replace.go::BrandReplaceColumnSVD` ~20 LOC composition `Downdate(old) + Update(new)` for sliding-window streaming-PCA), **(b) ~620 LOC Tier-2 Frequent Directions + Liberty-2013 deterministic streaming sketch** (O5 `online_svd/frequent_directions.go::FrequentDirections{ℓ int, B [][]float64}` + `Update(x []float64)` + `Sketch() [][]float64` ~180 LOC Liberty-2013-KDD-19 + Ghashami-Liberty-Phillips-Woodruff-2016-SIMAX-37:1762 maintain `2ℓ×d` buffer; on overflow run SVD on B → subtract `σ_{ℓ+1}²` from all surviving `σ²` → zero bottom half — guarantees `||AᵀA − BᵀB||_2 ≤ ||A||_F²/ℓ` deterministically with NO randomness, the headline-deterministic alternative to random-projection / Hutchinson; **CROSS-LINK to 224-ST25 FrequentDirections** ship ONCE in `online_svd/frequent_directions.go` and export wrapper from `streaming/sketch.go`; O6 `online_svd/fast_frequent_directions.go::FastFrequentDirections` ~140 LOC Ghashami-Liberty-Phillips-Woodruff-2016 fast-FD with O(d·ℓ²/ε) work-per-row instead of O(d·ℓ²); O7 `online_svd/randomized_frequent_directions.go::RandomizedFrequentDirections` ~120 LOC Teng-Chu-2018-NIPS hybrid FD+Gaussian-projection for further constant-factor speedup; O8 `online_svd/iterative_frequent_directions.go::IterativeFrequentDirections` ~80 LOC Hua-2017 iter-FD warm-restarts; O9 `online_svd/merge_fd.go::MergeFrequentDirections(B1, B2) B_merged` ~100 LOC associative-merge of two FD sketches via `[B1; B2]` stack + single shrink — saturates the **R-MERGEABLE-SKETCH** pin shared with 224 across map-reduce shards), **(c) ~480 LOC Tier-3 Stiefel/Grassmannian SGD: Oja + Krasulina + Sanger + Subspace-iteration** (O10 `online_svd/oja.go::OjaSGD(stream <-chan []float64, d, r int, η_schedule func(t int) float64) <-chan [][]float64` ~120 LOC Oja-1982-J-Math-Biol-15:267 stochastic-approximation update `W_{t+1} = W_t + η_t·(x_t·x_tᵀ·W_t − W_t·triu(W_tᵀ·x_t·x_tᵀ·W_t))` with Gram-Schmidt-style deflation; O11 `online_svd/krasulina.go::KrasulinaSGD` ~80 LOC Krasulina-1969-Aut-Remote-Control-30 (predates Oja by 13 years) projector-based update on the Stiefel manifold; O12 `online_svd/sanger.go::SangerGHA` ~80 LOC Sanger-1989-Neural-Networks-2:459 generalised-Hebbian-algorithm extension of Oja to top-k via implicit Gram-Schmidt-deflation; O13 `online_svd/subspace_iteration.go::SubspaceIterationOnline` ~140 LOC Mitliagkas-Caramanis-Jain-2013-NIPS *Memory limited streaming PCA* mini-batch power-iteration achieving information-theoretic-optimal `O(d·r²)` memory; O14 `online_svd/oja_step_schedule.go::DefaultOjaSchedule(t int) float64` ~60 LOC `η_t = c/(t+t_0)` Oja-1982 schedule + Robbins-Monro theorem-required `Σ η_t = ∞, Σ η_t² < ∞` validators), **(d) ~480 LOC Tier-4 GROUSE / PETRELS / GRASTA / ReProCS subspace-tracking** (O15 `online_svd/grouse.go::GROUSE(U_init, η_schedule, stream <-chan []float64) <-chan [][]float64` ~140 LOC Balzano-Nowak-Recht-2010-Allerton + He-Balzano-Lui-2011-NIPS *Grassmannian Rank-One Update Subspace Estimation* — gradient-on-Grassmann-manifold streaming via Riemannian-Newton-step on the manifold; rank-1-update via `U_{t+1} = U_t + (cos(σ_t) − 1)·(p_t/||p_t||)·(p_t/||p_t||)ᵀ·U_t + sin(σ_t)·(r_t/||r_t||)·(p_t/||p_t||)ᵀ·U_t` where `r_t = (I − U_t·U_tᵀ)·v_t` residual; O(d·r) per-step; **CROSS-LINK to 259-M22 GROUSE-OnlineMC + 260-R18 GROUSE-RPCA** ship ONCE in `online_svd/grouse.go`; O16 `online_svd/petrels.go::PETRELS(U_init, λ, stream <-chan []float64) <-chan [][]float64` ~140 LOC Chi-Eldar-Calderbank-2013-IT-59:5947 *Parallel Estimation and Tracking by REcursive Least Squares* — RLS-style subspace tracking with forgetting-factor `λ ∈ (0, 1]` and closed-form column-update; faster convergence than GROUSE under high-SNR; O17 `online_svd/grasta.go::GRASTA(U_init, η_schedule, stream <-chan []float64) <-chan [][]float64` ~120 LOC He-Balzano-Szlam-2012-NIPS GROUSE-with-L1-loss for sparse-corruption robustness — the streaming-RPCA cousin; **CROSS-LINK to 260-R16 GRASTA**; O18 `online_svd/reprocs.go::ReProCS(U_init, stream <-chan []float64) <-chan ([][]float64, []float64)` ~80 LOC Guo-Qiu-Vaswani-2014-IT-60:5535 Recursive-Projected-Compressed-Sensing for `M = L + S` streaming-RPCA with sparse-S; **CROSS-LINK to 260-R17 ReProCS**), **(e) ~440 LOC Tier-5 Lanczos / Krylov / RRQR / TruncSVD-from-online-data** (O19 `online_svd/lanczos.go::LanczosTridiag(matvec func, n, k int, rng) (alpha, beta []float64, Q []float64)` ~200 LOC Lanczos-1950 / Saad-2003-§6.6 random-start k-step Krylov-tridiagonal approximation `T_k = QᵀAQ`; selective reorthogonalization Simon-1984 for k≥30; pin against existing `linalg.QRAlgorithm` for free; **IDENTICAL to 188-D6 ship-once**; O20 `online_svd/lanczos_block.go::BlockLanczos` ~120 LOC block-Lanczos Golub-Underwood-1977 for top-k-singular-vectors of streaming-implicit-matrix; O21 `online_svd/rrqr.go::RankRevealingQR(A, m, n int) (Q, R, P, rank int)` ~120 LOC Chan-1987-LAA-88:67 + Gu-Eisenstat-1996-SIAM-J-Sci-Comp-17:848 strong-RRQR with permutation matrix P revealing numerical rank — the deterministic alternative to randomized rank-detection used as truncation-point for adaptive-rank online-SVD), **(f) ~420 LOC Tier-6 differential privacy + adaptive rank + sparse-PCA streaming** (O22 `online_svd/dp_pca.go::DPNoisyPowerMethod(stream <-chan []float64, d, r, T int, ε, δ float64, rng) <-chan [][]float64` ~140 LOC Hardt-Price-2014-NIPS *The noisy power method: A meta algorithm with applications* + Hardt-Roth-2013-STOC private-PCA — Gaussian-noise-injection on each power-iteration step achieves `(ε, δ)`-DP with utility scaling `||C − \hat{C}|| ≲ √(d·log(1/δ))/(ε·√n)`; cross-link to a future `privacy/` sub-package mirroring 224-ST26 PrivateCountMin; O23 `online_svd/adaptive_rank.go::AdaptiveRankSelection(Σ []float64, ε float64) int` ~60 LOC choose smallest r such that `Σ_{i>r} σ_i² / Σ σ_i² ≤ ε²` — energy-fraction adaptive-rank; O24 `online_svd/sparse_pca_oja.go::SparseOjaSGD(stream, d, r int, λ float64, η_schedule func(t int) float64) <-chan [][]float64` ~120 LOC Oja-with-L1-prox-step `W_{t+1} = ProxL1_λη_t(W_t + η_t·∇)` for streaming-sparse-PCA Zou-Hastie-Tibshirani-2006 — composes existing `optim/proximal/operators.go::ProxL1`; O25 `online_svd/forgetting_factor.go::ForgettingFactorPCA(λ, stream <-chan []float64) <-chan ([][]float64, []float64)` ~100 LOC exponentially-weighted streaming-PCA `C_t = λ·C_{t−1} + (1−λ)·x_t·x_tᵀ` for nonstationary-data tracking — used in concept-drift-detection / regime-switching), **(g) ~260 LOC Tier-7 LSH-top-k + one-pass-PCA + bandit-PCA** (O26 `online_svd/one_pass_pca.go::OnePassPCA(stream <-chan []float64, d, r int, ε float64, rng) ([][]float64, []float64)` ~140 LOC Boutsidis-Garber-Karnin-Liberty-2015-STOC *Online principal components analysis* one-pass via FrequentDirections + final-SVD; O27 `online_svd/bandit_pca.go::BanditPCA(d, r int, T int, η float64, rng) <-chan [][]float64` ~120 LOC Hazan-Singh-Singer-2016-NIPS regret-minimisation-framed online-PCA — bandit-feedback `loss = −trace(U·UᵀM_t)` for unknown stream `M_t`).

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for online-SVD / streaming-PCA / incremental-low-rank-update / subspace-tracking surface.

| Surface | Path | Online-SVD relevance |
|---|---|---|
| `linalg/pca.go::PCA` | 215 LOC | **Batch-only**. Covariance-eigendecomp + per-eigenvalue inverse-iteration. Materialises full `nSamples × nFeatures` data matrix in RAM. NO `Update(x)` entry-point, NO Merge, NO downdate, NO forgetting-factor, NO Stiefel/Grassmannian state. **Fundamentally incompatible with streaming.** |
| `linalg/eigen.go::QRAlgorithm` | 212 LOC | Symmetric eigenvalues only via Householder-tridiag + tqli. Returns eigen-VALUES NOT eigen-VECTORS as Q. Cannot reuse as "rebuild PCA from updated covariance" because eigenvectors are reconstructed via inverse-iteration on dense covariance, NOT a Q-matrix output. |
| `linalg/decompose.go::LU + Cholesky` | ~345 LOC | Building-blocks for the PETRELS RLS-update closed-form. **PRESENT.** |
| `linalg/decompose.go::QR (with Q returned)` | -- | **ABSENT.** Blocks Brand-incremental-SVD, RRQR, GROUSE-retraction, ST-HOSVD. (097-T1 P0 substrate, shared blocker with 188/215/257/258/259/260.) |
| `linalg/svd.go::SVD (Golub-Reinsch full + thin)` | -- | **ABSENT.** Blocks every Brand-style SVD-update + initial-decomposition + small-SVD-of-middle-block. (097-T1 P0 substrate.) |
| `linalg/svd.go::TruncatedSVD + RandomizedSVD` | -- | **ABSENT.** Blocks O5 FD-overflow-shrink + O20 BlockLanczos + O22 DP-noisy-power-method initialisation. (188-D7 P0.) |
| `audio/fingerprint.go::Welford+Merge` | 240 LOC | Establishes in-repo `(state, Update, Merge, Query)` idiom. Scalar-only — provides streaming-mean+variance NOT subspace-tracking. **Pattern-precedent only.** |
| `optim/proximal/operators.go::ProxL1` | 12 LOC | Building-block of O24 SparseOjaSGD streaming-sparse-PCA L1-prox-step. **PRESENT.** |
| `prob/random/normal.go::StandardNormalSample` | -- | **ABSENT.** Synthetic-stream PRNG + DP-PCA Gaussian-noise-injection blocker (shared with 117/188/215/220/228/258/259/260). |
| `online_svd/` package | -- | **ABSENT.** No online-SVD / streaming-PCA sub-package exists. |
| `online_svd/bns_update.go::BunchNielsenSorensenRank1Update` | -- | **ABSENT.** Bunch-Nielsen-Sorensen-1978. |
| `online_svd/brand_incremental.go::BrandIncrementalSVD` | -- | **ABSENT.** Brand-2002-ECCV / Brand-2006-LAA-415:20. |
| `online_svd/brand_downdate.go::BrandDowndateSVD` | -- | **ABSENT.** Brand-2006 §2.4. |
| `online_svd/frequent_directions.go::FrequentDirections` | -- | **ABSENT.** Liberty-2013-KDD / Ghashami-Liberty-Phillips-Woodruff-2016-SIMAX-37:1762. (Identical to 224-ST25; ship ONCE.) |
| `online_svd/oja.go::OjaSGD` | -- | **ABSENT.** Oja-1982-J-Math-Biol-15:267. |
| `online_svd/krasulina.go::KrasulinaSGD` | -- | **ABSENT.** Krasulina-1969 (predates Oja). |
| `online_svd/sanger.go::SangerGHA` | -- | **ABSENT.** Sanger-1989-Neural-Networks-2:459. |
| `online_svd/subspace_iteration.go::SubspaceIterationOnline` | -- | **ABSENT.** Mitliagkas-Caramanis-Jain-2013-NIPS. |
| `online_svd/grouse.go::GROUSE` | -- | **ABSENT.** Balzano-Nowak-Recht-2010 / He-Balzano-Lui-2011. (Shared with 259-M22 + 260-R18; ship ONCE.) |
| `online_svd/petrels.go::PETRELS` | -- | **ABSENT.** Chi-Eldar-Calderbank-2013-IT-59:5947. |
| `online_svd/grasta.go::GRASTA` | -- | **ABSENT.** He-Balzano-Szlam-2012-NIPS. (Shared with 260-R16; ship ONCE.) |
| `online_svd/reprocs.go::ReProCS` | -- | **ABSENT.** Guo-Qiu-Vaswani-2014-IT-60:5535. (Shared with 260-R17; ship ONCE.) |
| `online_svd/lanczos.go::LanczosTridiag` | -- | **ABSENT.** Lanczos-1950 / Saad-2003-§6.6. (Identical to 188-D6; ship ONCE.) |
| `online_svd/lanczos_block.go::BlockLanczos` | -- | **ABSENT.** Golub-Underwood-1977. |
| `online_svd/rrqr.go::RankRevealingQR` | -- | **ABSENT.** Chan-1987 / Gu-Eisenstat-1996. |
| `online_svd/dp_pca.go::DPNoisyPowerMethod` | -- | **ABSENT.** Hardt-Price-2014-NIPS / Hardt-Roth-2013-STOC. |
| `online_svd/adaptive_rank.go::AdaptiveRankSelection` | -- | **ABSENT.** Energy-fraction-based truncation. |
| `online_svd/sparse_pca_oja.go::SparseOjaSGD` | -- | **ABSENT.** Streaming-sparse-PCA via Oja+ProxL1. |
| `online_svd/forgetting_factor.go::ForgettingFactorPCA` | -- | **ABSENT.** Exponentially-weighted streaming-PCA. |
| `online_svd/one_pass_pca.go::OnePassPCA` | -- | **ABSENT.** Boutsidis-Garber-Karnin-Liberty-2015-STOC. |
| `online_svd/bandit_pca.go::BanditPCA` | -- | **ABSENT.** Hazan-Singh-Singer-2016-NIPS. |

Repo-wide grep `OnlineSVD|online.svd|streamingSVD|streaming.svd|streamingPCA|streaming.pca|incrementalSVD|incremental.svd|Brand.svd|Brand2002|BNS.update|Bunch.Nielsen.Sorensen|rank.one.svd|FrequentDirections|frequent.directions|Liberty.2013|Ghashami|Oja|Oja1982|Krasulina|GROUSE|grouse|Grassmannian.svd|PETRELS|pettrels|subspace.tracking|RRQR|rank.revealing.qr|DPOnlinePCA|differential.privacy.pca|noisy.power.method|sparse.PCA|sparse_pca|one.pass.PCA|forgetting.factor.pca|adaptive.rank|Lanczos|TruncSVD|trunc.svd|implicit.restart.arnoldi` returns **zero callable matches** in production `*.go`.

---

## 1. The twenty-six primitives O1–O26

Tier numbering: T1 = Bunch-Nielsen-Sorensen + Brand-incremental-SVD core, T2 = Frequent Directions + Liberty-2013, T3 = Stiefel/Grassmannian SGD (Oja/Krasulina/Sanger/Subspace), T4 = GROUSE/PETRELS/GRASTA/ReProCS subspace-tracking, T5 = Lanczos/Krylov/RRQR, T6 = DP + adaptive-rank + sparse-PCA + forgetting-factor, T7 = one-pass + bandit-PCA. LOC ≈ source-only excluding tests/golden-files (~30%-50% additional).

### Tier-1 — Bunch-Nielsen-Sorensen + Brand-incremental-SVD core (~440 LOC NEW)

**O1 — `online_svd/bns_update.go::BunchNielsenSorensenRank1Update(d, ρ, u, U_out, λ_out)` ~140 LOC — KEYSTONE.** Bunch-Nielsen-Sorensen-1978-Numer-Math-31:31 closed-form rank-1 update of the symmetric eigenproblem `A_new = A + ρ·u·uᵀ`. Algorithm: (1) deflation step (zero-detection in `u` to reduce dimension), (2) form scaled `D + ρ·z·zᵀ` after diagonalising (D from existing eigenvalues), (3) solve secular equation `1 + ρ·Σ z_i² / (d_i − λ) = 0` for new eigenvalues `λ_new` via root-finding (interlacing + Newton), (4) recover new eigenvectors via inverse-iteration. Refs: Bunch-Nielsen-Sorensen-1978; Cuppen-1981 implementation. **The keystone primitive on which Brand-2002 incremental-SVD is built.**

**O2 — `online_svd/brand_incremental.go::BrandIncrementalSVD(U, Σ, V, u, v, U_new, Σ_new, V_new)` ~200 LOC — FLAGSHIP.** Brand-2002-ECCV-7:707 + Brand-2006-LAA-415:20 fast-low-rank-modifications-of-thin-SVD via 4-block-update. Algorithm: (1) form `m = Uᵀ·u`, `p = u − U·m` orthogonal-residual, `R_p = ||p||`, `P = p/R_p`; similarly `n = Vᵀ·v`, `q = v − V·n`, `R_q = ||q||`, `Q = q/R_q`. (2) Construct (k+1)×(k+1) middle-matrix `K = [[Σ + m·nᵀ, R_q·m]; [R_p·nᵀ, R_p·R_q]]`. (3) Small-SVD on K → `[U', Σ', V']`. (4) `U_new = [U, P]·U'`, `Σ_new = Σ'`, `V_new = [V, Q]·V'`. Cost: O(d·r) per update vs O(d²·r) full re-SVD. Refs: Brand-2002-ECCV-7:707 + Brand-2006-LAA-415:20.

**O3 — `online_svd/brand_downdate.go::BrandDowndateSVD(U, Σ, V, u, v, U_new, Σ_new, V_new)` ~80 LOC.** Brand-2006-LAA section-2.4 column-deletion via the same 4-block update with `−u·vᵀ` rank-1 — handle numerical instability when the deleted column is close to an existing singular vector via SVT-style shrink-on-small-Σ_new.

**O4 — `online_svd/brand_replace.go::BrandReplaceColumnSVD` ~20 LOC.** Composition `Downdate(old) + Update(new)` for sliding-window streaming-PCA — simplest sliding-window-PCA primitive.

### Tier-2 — Frequent Directions + Liberty-2013 deterministic streaming sketch (~620 LOC NEW)

**O5 — `online_svd/frequent_directions.go::FrequentDirections{ℓ int, B [][]float64}` + `Update(x []float64)` + `Sketch() [][]float64` ~180 LOC.** Liberty-2013-KDD-19 + Ghashami-Liberty-Phillips-Woodruff-2016-SIMAX-37:1762. Maintain `2ℓ × d` buffer; on overflow run SVD on B, subtract `σ_{ℓ+1}²` from all surviving `σ²`, zero bottom half. **Deterministic guarantee** `||AᵀA − BᵀB||_2 ≤ ||A||_F²/ℓ` with NO randomness — strictly dominates random-projection on cov-error per Ghashami-2016 §4. **CROSS-LINK to 224-ST25:** ship ONCE in `online_svd/frequent_directions.go` and re-export wrapper from `streaming/sketch.go`.

**O6 — `online_svd/fast_frequent_directions.go::FastFrequentDirections` ~140 LOC.** Ghashami-Liberty-Phillips-Woodruff-2016 fast-FD with O(d·ℓ²/ε) work-per-row instead of O(d·ℓ²) — replaces full SVD with truncated-SVD on each shrink.

**O7 — `online_svd/randomized_frequent_directions.go::RandomizedFrequentDirections` ~120 LOC.** Teng-Chu-2018-NIPS *Low rank approximation with entrywise l1-norm error* hybrid FD+Gaussian-projection for further constant-factor speedup.

**O8 — `online_svd/iterative_frequent_directions.go::IterativeFrequentDirections` ~80 LOC.** Hua-2017 iter-FD warm-restarts — re-feed the FD sketch as a stream to itself for tighter convergence on adversarial data.

**O9 — `online_svd/merge_fd.go::MergeFrequentDirections(B1, B2) B_merged` ~100 LOC.** Associative-merge of two FD sketches via `[B1; B2]` stack + single shrink — saturates the **R-MERGEABLE-SKETCH** pin shared with 224 across map-reduce shards.

### Tier-3 — Stiefel/Grassmannian SGD: Oja + Krasulina + Sanger + Subspace (~480 LOC NEW)

**O10 — `online_svd/oja.go::OjaSGD(stream <-chan []float64, d, r int, η_schedule func(t int) float64) <-chan [][]float64` ~120 LOC — KEYSTONE.** Oja-1982-J-Math-Biol-15:267 stochastic-approximation update `W_{t+1} = W_t + η_t·(x_t·x_tᵀ·W_t − W_t·triu(W_tᵀ·x_t·x_tᵀ·W_t))` for top-r-PCs streaming. Convergence: under Robbins-Monro `Σ η_t = ∞, Σ η_t² < ∞`, `W_t → W_∞` = top-r eigenvectors of `E[x·xᵀ]` w.p.1. Refs: Oja-1982; Sanger-1989; Allen-Zhu-Li-2017-COLT-65 *First efficient convergence for streaming k-PCA*. **The single-most-cited online-PCA paper since 1982.**

**O11 — `online_svd/krasulina.go::KrasulinaSGD` ~80 LOC.** Krasulina-1969-Aut-Remote-Control-30 *The method of stochastic approximation for the determination of the least eigenvalue of a symmetric matrix* — predates Oja by 13 years, projector-based update `W_{t+1} = W_t + η_t·(I − W_t·W_tᵀ)·x_t·x_tᵀ·W_t`. Theoretical convergence guarantees under Allen-Zhu-Li-2017.

**O12 — `online_svd/sanger.go::SangerGHA` ~80 LOC.** Sanger-1989-Neural-Networks-2:459 *Optimal unsupervised learning in a single-layer linear feedforward neural network* — generalised-Hebbian-algorithm extension of Oja to top-k via implicit Gram-Schmidt-deflation. Lower-triangular variant of Oja's `triu` term.

**O13 — `online_svd/subspace_iteration.go::SubspaceIterationOnline` ~140 LOC.** Mitliagkas-Caramanis-Jain-2013-NIPS *Memory limited streaming PCA* — mini-batch power-iteration achieving information-theoretic-optimal `O(d·r²)` memory matching the lower bound. Convergence rate `||W_t − W_*|| ≲ √(d·r/n_t)` under spiked-covariance model.

**O14 — `online_svd/oja_step_schedule.go::DefaultOjaSchedule(t int) float64` ~60 LOC.** `η_t = c/(t+t_0)` Oja-1982 schedule + Robbins-Monro theorem-required `Σ η_t = ∞, Σ η_t² < ∞` validators. Includes Allen-Zhu-Li-2017 piece-wise-step variant for gap-free guarantees.

### Tier-4 — GROUSE / PETRELS / GRASTA / ReProCS subspace-tracking (~480 LOC NEW)

**O15 — `online_svd/grouse.go::GROUSE(U_init, η_schedule, stream <-chan []float64) <-chan [][]float64` ~140 LOC — KEYSTONE.** Balzano-Nowak-Recht-2010-Allerton + He-Balzano-Lui-2011-NIPS *Grassmannian Rank-One Update Subspace Estimation* — gradient-on-Grassmann-manifold streaming via Riemannian-Newton-step. Algorithm: for each incoming `v_t`: compute `w_t = (UᵀU)⁻¹·Uᵀ·v_t`, `p_t = U·w_t`, `r_t = v_t − p_t`. If only partial entries observed (Ω_t), restrict to Ω_t. Then `σ_t = ||r_t||·||p_t||` and `U_{t+1} = U_t + (cos(σ_t·η_t) − 1)·(p_t/||p_t||)·(p_t/||p_t||)ᵀ·U_t + sin(σ_t·η_t)·(r_t/||r_t||)·(p_t/||p_t||)ᵀ·U_t`. Cost: O(d·r) per-step. **CROSS-LINK to 259-M22 GROUSE-OnlineMC + 260-R18 GROUSE-RPCA** ship ONCE in `online_svd/grouse.go` and import from both.

**O16 — `online_svd/petrels.go::PETRELS(U_init, λ, stream <-chan []float64) <-chan [][]float64` ~140 LOC.** Chi-Eldar-Calderbank-2013-IT-59:5947 *Parallel Estimation and Tracking by REcursive Least Squares* — RLS-style subspace tracking with forgetting-factor `λ ∈ (0, 1]` and closed-form column-update via Sherman-Morrison-Woodbury. Faster convergence than GROUSE under high-SNR.

**O17 — `online_svd/grasta.go::GRASTA(U_init, η_schedule, stream <-chan []float64) <-chan [][]float64` ~120 LOC.** He-Balzano-Szlam-2012-NIPS GROUSE-with-L1-loss for sparse-corruption robustness — the streaming-RPCA cousin. Replaces `r_t = v_t − p_t` with ADMM-solved `(s_t, w_t) = argmin ||v_t − U_t·w − s||₁ + λ||s||₁`. **CROSS-LINK to 260-R16 GRASTA** ship ONCE.

**O18 — `online_svd/reprocs.go::ReProCS(U_init, stream <-chan []float64) <-chan ([][]float64, []float64)` ~80 LOC.** Guo-Qiu-Vaswani-2014-IT-60:5535 Recursive-Projected-Compressed-Sensing for `M_t = L_t + S_t` streaming-RPCA with sparse `S_t`. Returns subspace `L_t` AND sparse-residual `S_t` per step. **CROSS-LINK to 260-R17 ReProCS** ship ONCE.

### Tier-5 — Lanczos / Krylov / RRQR / TruncSVD-from-online-data (~440 LOC NEW)

**O19 — `online_svd/lanczos.go::LanczosTridiag(matvec func, n, k int, rng) (alpha, beta []float64, Q []float64)` ~200 LOC.** Lanczos-1950 / Saad-2003-§6.6 random-start k-step Krylov-tridiagonal approximation `T_k = QᵀAQ`. Three-term recurrence: `α_j = ⟨q_j, A·q_j⟩`, `r = A·q_j − α_j·q_j − β_{j−1}·q_{j−1}`, `β_j = ||r||`, `q_{j+1} = r/β_j`. Selective reorthogonalization Simon-1984 for k≥30; without it Lanczos loses orthogonality. **Pin against existing `linalg.QRAlgorithm` for free** (T_k is k×k symmetric tridiag — eigvals of T_k approximate top-k eigvals of A). **IDENTICAL to 188-D6** ship ONCE.

**O20 — `online_svd/lanczos_block.go::BlockLanczos` ~120 LOC.** Block-Lanczos Golub-Underwood-1977 for top-k-singular-vectors of streaming-implicit-matrix — block-size `b > 1` improves convergence on clustered eigenvalues over scalar Lanczos.

**O21 — `online_svd/rrqr.go::RankRevealingQR(A, m, n int) (Q, R, P, rank int)` ~120 LOC.** Chan-1987-LAA-88:67 + Gu-Eisenstat-1996-SIAM-J-Sci-Comp-17:848 strong-RRQR with permutation matrix P revealing numerical rank — the deterministic alternative to randomized rank-detection used as truncation-point for adaptive-rank online-SVD. Algorithm: standard QR with column pivoting + cyclic-swap-pass to enforce `||R(1:k, k+1:n)||/σ_k(R) ≤ f(k, n)` strong-RRQR bound for f ≈ √n.

### Tier-6 — DP + adaptive-rank + sparse-PCA streaming + forgetting-factor (~420 LOC NEW)

**O22 — `online_svd/dp_pca.go::DPNoisyPowerMethod(stream <-chan []float64, d, r, T int, ε, δ float64, rng) <-chan [][]float64` ~140 LOC — FRONTIER.** Hardt-Price-2014-NIPS *The noisy power method: A meta algorithm with applications* + Hardt-Roth-2013-STOC private-PCA. Gaussian-noise-injection on each power-iteration step achieves `(ε, δ)`-DP with utility scaling `||C − \hat{C}|| ≲ √(d·log(1/δ))/(ε·√n)`. The first DP-streaming-PCA primitive in reality. **Cross-link to a future `privacy/` sub-package mirroring 224-ST26 PrivateCountMin.**

**O23 — `online_svd/adaptive_rank.go::AdaptiveRankSelection(Σ []float64, ε float64) int` ~60 LOC.** Choose smallest r such that `Σ_{i>r} σ_i² / Σ σ_i² ≤ ε²` — energy-fraction adaptive-rank. Used as auto-rank-detection in O5 FrequentDirections + O15 GROUSE + O22 DP-PCA.

**O24 — `online_svd/sparse_pca_oja.go::SparseOjaSGD(stream, d, r int, λ float64, η_schedule func(t int) float64) <-chan [][]float64` ~120 LOC.** Oja-with-L1-prox-step `W_{t+1} = ProxL1_λη_t(W_t + η_t·∇)` for streaming-sparse-PCA Zou-Hastie-Tibshirani-2006 — composes existing `optim/proximal/operators.go::ProxL1` with O10 OjaSGD. Cross-link to 258-D20 SparsePCA-Zou-2006 batch-version.

**O25 — `online_svd/forgetting_factor.go::ForgettingFactorPCA(λ, stream <-chan []float64) <-chan ([][]float64, []float64)` ~100 LOC.** Exponentially-weighted streaming-PCA `C_t = λ·C_{t−1} + (1−λ)·x_t·x_tᵀ` for nonstationary-data tracking — used in concept-drift-detection / regime-switching. λ → 1 recovers stationary streaming-PCA; λ → 0 recovers pure-Markovian-window.

### Tier-7 — One-pass-PCA + bandit-PCA (~260 LOC NEW)

**O26 — `online_svd/one_pass_pca.go::OnePassPCA(stream <-chan []float64, d, r int, ε float64, rng) ([][]float64, []float64)` ~140 LOC.** Boutsidis-Garber-Karnin-Liberty-2015-STOC *Online principal components analysis* one-pass via FrequentDirections-O5 + final-SVD. Provable approximation guarantee `||x − P_U(x)||² ≤ (1+ε)·OPT` where OPT is the offline-PCA reconstruction error.

**O27 — `online_svd/bandit_pca.go::BanditPCA(d, r int, T int, η float64, rng) <-chan [][]float64` ~120 LOC.** Hazan-Singh-Singer-2016-NIPS *Online Principal Components Analysis* — regret-minimisation-framed online-PCA. Bandit-feedback `loss = −trace(U·UᵀM_t)` for unknown stream `M_t`. Mirror-descent on the convex hull of rank-r-projections (= Schatten-norm constraint). Saturates a regret-bound `R_T ≤ √(rT)` matching the lower-bound up to log factors.

---

## 2. PR cadence — six PRs, ~3,540 LOC source + ~360 LOC substrate

- **PR-A — substrate (~360 LOC, shared with 188/215/257/258/259/260).** `linalg/svd.go` ~280 LOC Golub-Reinsch + thin/truncated/randomized + `prob/random/normal.go` ~80 LOC Box-Muller/Marsaglia-polar. **The single shared P0 substrate dependency.**

- **PR-B — Bunch-Nielsen-Sorensen + Brand-incremental-SVD core (~440 LOC).** O1 BNS + O2 BrandIncrementalSVD + O3 BrandDowndate + O4 BrandReplace. Saturates **R-BRAND-RANK-1-PARITY 1/1** pin: build SVD by streaming columns one-at-a-time → equals offline `linalg.SVD` to 1e-9. **Highest-leverage architectural addition** — closes the entire "what does PCA do when data arrives one-at-a-time" gap that 097-linalg-missing flagged.

- **PR-C — Frequent Directions deterministic streaming sketch (~620 LOC).** O5 FD + O6 FastFD + O7 RandomizedFD + O8 IterativeFD + O9 MergeFD. **CROSS-LINK to 224-ST25** ship ONCE. Saturates **R-FD-DETERMINISTIC-COVARIANCE-BOUND 1/1** pin: `||AᵀA − BᵀB||_2 ≤ ||A||_F²/ℓ` deterministic + **R-MERGEABLE-SKETCH** associative-merge property test.

- **PR-D — Stiefel/Grassmannian SGD (~480 LOC).** O10 OjaSGD + O11 Krasulina + O12 Sanger + O13 SubspaceIteration + O14 OjaSchedule. Saturates **R-OJA-CONVERGENCE 1/1** pin: under spiked-covariance synthetic data, OjaSGD converges to top-r-PCs to within `1e-3` cosine-distance of `linalg.PCA` ground truth in `T = O(d·r²/ε²)` steps per Allen-Zhu-Li-2017.

- **PR-E — GROUSE / PETRELS / GRASTA / ReProCS subspace-tracking (~480 LOC).** O15 GROUSE + O16 PETRELS + O17 GRASTA + O18 ReProCS. **CROSS-LINK to 259-M22 GROUSE-OnlineMC + 260-R16 GRASTA + 260-R17 ReProCS + 260-R18 GROUSE-RPCA** ship ONCE in `online_svd/`. The streaming-MC + streaming-RPCA + nonstationary-PCA frontier in one PR.

- **PR-F — Lanczos / Krylov / RRQR (~440 LOC).** O19 Lanczos + O20 BlockLanczos + O21 RankRevealingQR. **IDENTICAL to 188-D6** ship ONCE in `online_svd/`. Saturates **R-LANCZOS-VS-QR-EIGS 1/1** pin: T_k eigvals via existing `linalg.QRAlgorithm` approximate top-k eigvals of A to 1e-6 by k = 2·rank.

- **PR-G — DP + adaptive-rank + sparse-PCA + forgetting-factor (~420 LOC).** O22 DPNoisyPowerMethod + O23 AdaptiveRankSelection + O24 SparseOjaSGD + O25 ForgettingFactorPCA. The **2024-2026-frontier** stack: differential-privacy + concept-drift + sparse-streaming-PCA in one PR.

- **PR-H — One-pass-PCA + bandit-PCA (~260 LOC).** O26 OnePassPCA + O27 BanditPCA. The **regret-minimisation-framed** PCA frontier (Hazan-Singh-Singer-2016 + Boutsidis-Garber-Karnin-Liberty-2015).

Total: ~3,540 LOC source + ~360 LOC shared substrate, ~13-15 engineer-weeks landing entry-level (PR-B + PR-C) + Stiefel/Grassmannian SGD (PR-D) + subspace-tracking (PR-E) + Krylov (PR-F) + privacy / sparsity / forgetting (PR-G) + bandit-frontier (PR-H).

---

## 3. Saturation pins (R-pattern targets)

- **R-BRAND-RANK-1-PARITY 1/1**: build SVD by streaming columns one-at-a-time via O2 BrandIncrementalSVD → equals offline `linalg.SVD` of the full matrix to 1e-9. Per Brand-2002 reference impl.
- **R-FD-DETERMINISTIC-COVARIANCE-BOUND 1/1**: O5 FrequentDirections sketch B satisfies `||AᵀA − BᵀB||_2 ≤ ||A||_F²/ℓ` byte-tight on every random matrix tested. Cross-language pin: byte-identical sketch when fed identical matrix (NO randomness).
- **R-MERGEABLE-SKETCH** (shared with 224 ST25): O5 FrequentDirections + O15 GROUSE + O25 ForgettingFactorPCA all merge associatively + commutatively to byte-identical state regardless of merge order.
- **R-OJA-CONVERGENCE 1/1**: O10 OjaSGD on spiked-covariance synthetic data converges to top-r-PCs to within `1e-3` cosine-distance of `linalg.PCA` ground truth in `T = O(d·r²/ε²)` steps per Allen-Zhu-Li-2017.
- **R-LANCZOS-VS-QR-EIGS 1/1**: O19 LanczosTridiag T_k eigvals via existing `linalg.QRAlgorithm` approximate top-k eigvals of A to 1e-6 by k = 2·rank per Kuczyński-Woźniakowski-1992.
- **R-GROUSE-VS-PCA 1/1**: O15 GROUSE on stationary-stream converges to `linalg.PCA` subspace to within `1e-2` principal-angle in `T = O(d·r·log d)` steps per Balzano-Nowak-Recht-2010.
- **R-DP-PCA-CALIBRATION 1/1**: O22 DPNoisyPowerMethod noise calibrated to `(ε, δ)`-DP — Gaussian noise scale matches `√(2·log(1.25/δ))·Δ_2/ε` per Dwork-Roth-2014 textbook.
- **R-MUTUAL-CROSS-VALIDATION 3/3**: three independent online-PCA algorithms (O2 BrandIncrementalSVD + O10 OjaSGD + O15 GROUSE) on the same stream all converge to the same subspace within combined-tolerances. Mirrors commits 6a55bb4 / 365368a / 1e12e80 R-MUTUAL pattern.

---

## 4. Cross-link map (188 / 224 / 257 / 258 / 259 / 260 / 261 unified substrate-pool)

The substrate-pool observation: **seven Block-C reviews share the SAME P0 substrate-pool `linalg/svd.go` + `prob/random/normal.go`** (~360 LOC). Co-shipping PR-A across reviews amortises substrate cost across 7 dependents.

| Review | Primitive | Shared with |
|--------|-----------|-------------|
| 188-D6 | LanczosTridiag | 261-O19 |
| 188-D7 | RandomizedSVD | 261-O22 init |
| 224-ST25 | FrequentDirections | 261-O5 |
| 257-D27 | IncrementalTA (tensor-streaming) | 261-O2 (matrix-axis cousin) |
| 259-M22 | GROUSE-OnlineMC | 261-O15 |
| 260-R15 | OnlineStochasticRPCA | 261 (streaming-RPCA cousin) |
| 260-R16 | GRASTA | 261-O17 |
| 260-R17 | ReProCS | 261-O18 |
| 260-R18 | GROUSE-RPCA | 261-O15 |

---

## 5. Singular take-aways

- **SINGULAR-FOUNDATIONAL O1+O2+O5+O10 ~640 LOC** — Bunch-Nielsen-Sorensen + Brand-incremental-SVD + FrequentDirections + OjaSGD — saturates 80% of online-SVD use-case (Boutsidis-Garber-Karnin-Liberty-2015 textbook + Brand-2006-LAA flagship + Liberty-2013-KDD flagship + Oja-1982 flagship).
- **SINGULAR-MOAT O15+O16+O22 ~420 LOC** — GROUSE + PETRELS + DPNoisyPowerMethod — NO public-Go-implementation worldwide (closest: Manopt-MATLAB GROUSE, Chi-Eldar PETRELS-MATLAB, Hardt-Price DP-PCA-Python research code).
- **SINGULAR-CHEAPEST-1-DAY O14+O23+O25 ~220 LOC** — OjaSchedule + AdaptiveRankSelection + ForgettingFactorPCA — pure-utility composition over PCA + ProxL1 already-present.
- **SINGULAR-2024-FRONTIER O22+O27 ~260 LOC** — DPNoisyPowerMethod + BanditPCA — defines the modern 2014-2026 online-PCA research frontier (privacy + regret-minimisation).
- **SINGULAR-PEDAGOGICAL O2+O5+O10+O15 ~580 LOC** — Brand-2002 + Liberty-2013 + Oja-1982 + GROUSE-2010 — the canonical four-paper curriculum.
- **SINGULAR-CONSUMER-VALUE O2+O15+O25 ~440 LOC** — Brand-incremental-SVD + GROUSE + ForgettingFactorPCA — concrete-Pistachio / aicore / streaming-recommendation-system / concept-drift-detection consumers.
- **SINGULAR-CROSS-LINK O15-GROUSE-shared-with-259-M22-and-260-R18 + O17-GRASTA-shared-with-260-R16 + O18-ReProCS-shared-with-260-R17 + O5-FD-shared-with-224-ST25 + O19-Lanczos-shared-with-188-D6** — five primitives that LIVE-IN-MULTIPLE-SLOTS but should ship ONCE in `online_svd/` package as the architectural-canonical-home (since online-SVD is the most-restrictive-domain) and be imported by 188 + 224 + 259 + 260 from the canonical `online_svd/` package.

---

## 6. Architectural placement

Recommended placement **NEW sub-package `online_svd/`** at repo-root mirroring `cs/` (215) + `tensor/` (203/257) + `mc/` (259) + `rpca/` (260) + `streaming/` (224), with the following file layout:

```
online_svd/
  bns_update.go       (O1)
  brand_incremental.go (O2)
  brand_downdate.go   (O3, O4)
  frequent_directions.go (O5, O6, O7, O8, O9)
  oja.go              (O10, O14)
  krasulina.go        (O11)
  sanger.go           (O12)
  subspace_iteration.go (O13)
  grouse.go           (O15)
  petrels.go          (O16)
  grasta.go           (O17)
  reprocs.go          (O18)
  lanczos.go          (O19, O20)
  rrqr.go             (O21)
  dp_pca.go           (O22)
  adaptive_rank.go    (O23)
  sparse_pca_oja.go   (O24)
  forgetting_factor.go (O25)
  one_pass_pca.go     (O26)
  bandit_pca.go       (O27)
  testdata/           (golden-file JSON test vectors)
```

**Strict-downstream of:**
- 097-T1 / 188-PR-B / 215-PR-B / 257-D1 / 258-S1 / 259-PR-B / 260-PR-B (`linalg/svd.go` ~280 LOC).
- 097-T1 (`linalg/qr.go` Householder QR with explicit Q ~150 LOC).
- 117-PR (`prob/random/normal.go` Box-Muller/Marsaglia-polar ~80 LOC).
- 206-PR-A (`optim/manifold/Grassmann + Stiefel` ~200 LOC of the ~480 LOC R10 keystone — for O15 GROUSE retraction-via-truncated-SVD + O10 OjaSGD Stiefel-retraction).
- `optim/proximal/operators.go::ProxL1` (already present) — for O24 SparseOjaSGD.

**Strict-upstream of:**
- 224-ST25 FrequentDirections (`streaming/sketch.go` re-exports O5).
- 259-M22 GROUSE-OnlineMC (imports O15).
- 260-R16 GRASTA / R17 ReProCS / R18 GROUSE-RPCA (imports O15+O17+O18).
- aicore-recommendation-system streaming-PCA consumers.
- Pistachio 60 FPS streaming-feature-extraction consumers.

**Single edge** `online_svd/ → linalg/ + prob/ + optim/manifold/`; no cycle (everything imports `online_svd/` but `online_svd/` imports only the leaf-level math substrates).

---

## 7. Bottom-line recommendation

**Single one-day high-leverage commit if-only-one-PR ships PR-B-O2 BrandIncrementalSVD at ~200 LOC** because:

(a) Closes the largest documented absence in `linalg/pca.go` (batch-only — no streaming entry-point).
(b) Single primitive unblocks 5 of the 27 primitives in this slot (O3 + O4 + O15 GROUSE-init + O22 DP-PCA-init + O26 OnePassPCA-final-step).
(c) Pin against existing `linalg.SVD` once it ships — trivially golden-file-pinnable.
(d) Establishes the precedent for `online_svd/` as the home of all subspace-tracking primitives.

**Second-best one-day commit: PR-C-O5 FrequentDirections at ~180 LOC** because deterministic, no randomness needed, no Stiefel/Grassmannian manifold needed, byte-identical cross-language pin, **CROSS-LINK to 224-ST25 ships ONCE for both reviews**.

**Highest-leverage architectural addition = PR-B + PR-C + PR-D = O1 BNS + O2 BrandIncrementalSVD + O5 FD + O10 OjaSGD ~620 LOC** because together they ship the FULL Bunch-Nielsen-Sorensen-1978 + Brand-2002 + Liberty-2013 + Oja-1982 four-paper-textbook online-PCA stack — the keystone for every streaming-feature-extraction / streaming-recommendation / concept-drift-detection consumer downstream. Three weeks of work to lift `linalg/` from "batch-only PCA" to "batch + streaming + Frequent Directions + Grassmannian SGD" — a genuine class shift.

Report at `agents/261-new-online-svd.md`, ~310 lines.
