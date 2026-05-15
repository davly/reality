# 273 | new-spectral-embedding — Spectral embedding: Laplacian eigenmaps, locally linear

**Summary L1.** **CANDOR-FIRST: slot 273 is 95% redundant with the union of 271-Tier-3 (~720 LOC SC17-SC21 LaplacianEigenmaps+LLE+DiffusionMap+Isomap+dispatcher) ∪ 272-Tier-2 (~720 LOC ML11-ML15 same five primitives + KernelPCA SHARED-SHIP-ONCE) ∪ 272-Tier-3 (~480 LOC ML16-ML18 HessianLLE+ModifiedLLE+LTSA) ∪ 272-Tier-6-ML34c OutOfSampleExtension-Bengio-2003.** Every named-algorithm in the slot title ("Laplacian eigenmaps, locally linear") and the seven listed-areas (Eigenmaps, LLE, HLLE, LTSA, Diffusion-Maps, Isomap) appears verbatim in 271-Tier-3 OR 272-Tier-2/3 with full reference-citations + LOC-budget + API-sketch + golden-vector-plan + R-MUTUAL-CROSS-VALIDATION-pin. The 272 review-doc explicitly self-declares "absorbs-273-as-manifold/spectral_embedding.go" (272 line 3, the cross-link section) — i.e. the engineer who wrote 272 ALREADY recommended folding 273 into 272 as a single file. **The honest first-order answer to "what new belongs in 273?" is: NOTHING NEW IN ALGORITHMS — 271+272 saturate the algorithm-axis. The only NON-redundant axis available to 273 is the CONVERGENCE-THEORY + SAMPLE-COMPLEXITY + SPECTRAL-CONSISTENCY layer that BOTH 271 AND 272 OMIT BY ARCHITECTURE (both are algorithm-soup ship-plans focused on producing-an-embedding, NEITHER ships the theoretical bounds + asymptotic-consistency-witnesses + discrete-to-continuous limit-Laplacian operators that the original Belkin-Niyogi-2008 / von-Luxburg-Belkin-Bousquet-2008 / Singer-2006 / Coifman-Lafon-2006 / Hein-Audibert-vonLuxburg-2007 / Trillos-Slepčev-2018 / García-Trillos-Slepčev-2020 / Calder-García-Trillos-2022 / Cheng-Wu-2022 papers proved).** **Slot 273's NON-redundant niche is therefore narrow but real**: a `manifold/consistency/` (or `graph/spectral/consistency/`) sub-package providing (a) the **continuous limit-operator** Laplace-Beltrami discretisation oracle the discrete Laplacians must converge to, (b) the **Davis-Kahan / Weyl perturbation bounds** quantifying how well discrete-eigenvectors approximate continuous-eigenfunctions, (c) the **Pinsker-style sample-complexity bounds** giving worst-case `n` required for a target embedding-error, (d) the **Nyström-out-of-sample-extension** unified theoretical-derivation (Bengio-2003 ALREADY in 272-ML34c, but 273 deepens via Williams-Seeger-2000 statistical-error-bounds + RKHS-cross-link-236 the FORMAL-WHY-it-works), (e) the **anisotropic-diffusion-maps-α-family** Coifman-Lafon-2006 §3 that 271-SC19 / 272-ML13 enumerate but mention only the α=0 default (the α=1 isotropic-Laplace-Beltrami + α=1/2 Fokker-Planck variants are ABSENT), (f) the **non-backtracking-walk-spectral-redemption** Krzakala-Moore-Mossel-Neeman-Sly-Zdeborová-Zhang-2013 PNAS-110:20935 below-Kesten-Stigum-threshold spectral-consistency (271-SC22 mentions this but as algorithm-only NOT as consistency-theorem). **Slot 273's value-add is the "WHY does spectral-embedding work?" layer, not the "HOW to compute it" layer.** **REPO STATE:** v0.10.0 ships ZERO consistency-theory primitives (verified repo-wide grep on `LaplaceBeltrami|laplace.beltrami|limit.Laplacian|spectral.consistency|Davis.Kahan|Weyl.perturbation|Belkin.Niyogi.convergence|Coifman.anisotropic|alpha.diffusion|nystrom.extension|Bengio.outofsample|sample.complexity.embedding|Trillos.Slepčev|Garcia.Trillos|Hein.Audibert.vonLuxburg|spectral.consistency|kernel.bandwidth.consistency|graph.Laplacian.convergence|continuous.limit.operator|Calder.GarciaTrillos|Cheng.Wu` returns ZERO callable matches in `*.go` outside review-corpus). Closest substrate: `infogeo/mmd.go::Kernel + GaussianKernel + LaplacianKernel + MedianHeuristicBandwidth` (the kernel-substrate every consistency-theorem assumes; bandwidth-selection is the central tuning-parameter the convergence-rates depend on) + `linalg/pca.go::PCA + ipi-private-inverse-iteration` (linear-baseline; the consistency proofs reduce to PCA-on-feature-space-of-kernel) + `linalg/eigen.go::QRAlgorithm` (eigenvalues-only, the keystone-blocker shared with 271+272).

**Summary L2.** **Twenty-two primitives CT1-CT22 ~2,140 LOC, all NEW vs 271+272 (zero overlap with already-enumerated algorithms — these are CONSISTENCY-WITNESSES + LIMIT-OPERATORS + PERTURBATION-BOUNDS + SAMPLE-COMPLEXITY-CALCULATORS that no other slot covers).** Organized as **(a) Tier-0 limit-operator primitives ~480 LOC** (CT1 `manifold/consistency/limit_operator.go::LaplaceBeltramiOnSphere(degree int) eigenvalues []float64` ~80 LOC: closed-form spectrum `λ_k = k(k+d-1)` of −Δ on S^d Sphere — the gold-standard oracle for verifying that discrete-Laplacian-on-sphere-sampled-points converges to it; CT2 `LaplaceBeltramiOnTorus(modes []int)` ~60 LOC closed-form `λ = 4π² Σ k_i²` for flat-torus T^d; CT3 `LaplaceBeltramiOnFlatRectangle(modes [][2]int) []float64` ~60 LOC Dirichlet-Neumann-eigenfunctions analytical sin/cos product `λ_{m,n} = π²(m²/L_x²+n²/L_y²)`; CT4 `HeatKernelOnSphere(t float64, x, y unit-vec) float64` ~80 LOC zonal-spherical-harmonic-expansion `K_t(x,y) = Σ_k (2k+d-1)/((d-1)·ω_d) · e^{-λ_k t} · C_k^((d-1)/2)(x·y)/C_k^((d-1)/2)(1)` Gegenbauer-polynomial sum to numerical-tolerance — substrate for verifying Coifman-Lafon-2006 t-step Markov-kernel converges to heat-kernel on manifold; CT5 `LimitGraphLaplacianBandwidth(n int, eps float64, d_intrinsic int) (rate float64, regime string)` ~60 LOC Hein-Audibert-vonLuxburg-2007-Annals-Stat-35:1 §3 the rate `eps_n → 0` with `n·eps_n^{d+2}/log(n) → ∞` is necessary AND sufficient for pointwise-consistency of unnormalised L_n on iid-sample from compact-d-manifold; returns regime "consistent | inconsistent | borderline" + rate-witness; CT6 `BiasVarianceTradeoffBandwidth(n, d_intrinsic, target_error) eps_optimal` ~80 LOC Singer-2006-ACHA-21:128 optimal-eps minimising bias `O(eps²)` + variance `O(1/(n·eps^{d+2}))` yields `eps_opt ∼ n^{-1/(d+4)}` and convergence-rate `n^{-2/(d+4)}` — the **optimal-bandwidth oracle** every kNN-bandwidth-selection should converge to; CT7 `KernelPolynomialBoundConsistency(kernel_type, manifold_curvature, n int) (bias, variance, total_error float64)` ~60 LOC Coifman-Lafon-2006 §4 anisotropic α-family bias-variance decomposition for α∈{0, 1/2, 1}), **(b) Tier-1 perturbation-bounds + Davis-Kahan ~440 LOC** (CT8 `manifold/consistency/davis_kahan.go::DavisKahanSinTheta(eigvals_pop, eigvals_emp, k_target int, eps_perturbation float64) (sin_theta, sample_size_lower_bound int)` ~140 LOC Davis-Kahan-1970-SIAM-J-Numer-Anal-7:1 + Yu-Wang-Samworth-2015-Biometrika-102:315 gives `‖sin Θ(V_pop, V_emp)‖_F ≤ √2·‖A−Â‖_op / Δ_k` where `Δ_k = λ_k − λ_{k+1}` is the spectral-gap. Returns the worst-case sin(angle)-between-empirical-and-population-eigenspaces; **the canonical perturbation-bound for spectral-embedding stability**. CT9 `WeylPerturbationBound(A_dim int, eigvals []float64, perturbation_norm float64) []float64` ~80 LOC Weyl-1912 `|λ_k(A+E) − λ_k(A)| ≤ ‖E‖_op` per-eigenvalue-additive bound — substrate for SC-bound + spectral-gap-stability witnesses. CT10 `WeylSchurInterlacing(eigvals_full, mask []bool) interlaced_eigvals` ~60 LOC Cauchy-1829-interlacing-Schur-1923 inclusion-principle for sub-matrix-eigenvalues; useful for sub-graph-spectral-stability. CT11 `MattiraSpectralGapStability(eigvals []float64, k_clusters int) (gap, perturbation_max_safe float64)` ~80 LOC computes `Δ_k = λ_{k+1}−λ_k` AND the max perturbation that preserves k-cluster-structure (Mattira-2006 / Yu-Wang-Samworth-2015 reverse-direction). CT12 `WedinTheoremSVD(sv_pop, sv_emp []float64, k int, perturbation_norm float64) sin_theta` ~80 LOC Wedin-1972-BIT-12:99 SVD-version of Davis-Kahan for non-symmetric matrices; substrate for non-backtracking-walk-matrix B perturbation in 271-SC22 SBM-recovery), **(c) Tier-2 sample-complexity calculators ~360 LOC** (CT13 `manifold/consistency/sample_complexity.go::EigenmapsSampleComplexity(d_intrinsic, k_eigenvecs int, target_sin_theta float64, manifold_volume, curvature_max float64) n_required int` ~100 LOC García-Trillos-Slepčev-2020-Found-Comput-Math-20:827 + Calder-García-Trillos-2022-Found-Comput-Math-22:1037 explicit-constants `n ≥ C(d) · log(n) · target^{-2-d/2}` for k-th-eigenfunction-ε-recovery on compact-d-manifold; **answers "how many samples for embedding to be ε-correct?"**; CT14 `LLESampleComplexity(d_intrinsic, dim_target, k_neighbours int, noise_level float64) n_required` ~60 LOC Belkin-Niyogi-2008-J-Comput-Syst-Sci-74:1289 LLE-specific sample-complexity using the local-tangent-PCA error-bound `O(eps^{d+2}/n)` per-point-bias; CT15 `IsomapSampleComplexity(d_intrinsic, manifold_diameter, kNN_k int) (n_required, kNN_required int)` ~60 LOC Bernstein-deSilva-Langford-Tenenbaum-2000-Stanford-Tech-Rep first-results geodesic-recovery `n^{-2/d}` rate; CT16 `DiffusionMapSampleComplexity(d_intrinsic, t_diffusion, k_eigenvecs int) n_required` ~60 LOC Singer-2006-ACHA-21:128 + Singer-Wu-2017-CommACM-60:54 anisotropic-diffusion-rate; CT17 `tSNEStatisticalConsistency(n, d_intrinsic, perplexity, dim_target int) (cluster_recovery_rate, distortion_upper_bound float64)` ~80 LOC Linderman-Steinerberger-2019-AOS-47:2398 + Arora-Hu-Kothari-2018-COLT t-SNE-consistency-theorem-proof for cluster-recovery in well-separated regime), **(d) Tier-3 anisotropic-diffusion-α-family ~280 LOC NEW-VS-271/272** (CT18 `manifold/consistency/anisotropic_diffusion.go::AnisotropicDiffusionMap(W, n, dim int, alpha float64, t float64) Y` ~140 LOC Coifman-Lafon-2006-ACHA-21:5 §3 the FULL α-family normalisation `K^(α) = D^{-α} K D^{-α}` then `P^(α) = (D^(α))^{-1} K^(α)` — α=0 graph-Laplacian-default-271/272; α=1/2 Fokker-Planck-on-data-density-removed; α=1 Laplace-Beltrami-on-pure-geometry-density-removed — three different limit-operators. The α∈{0,1/2,1} TRIO is the SINGULAR-NEW-ALGORITHM that 273 contributes that 271+272 do not (271-SC19 + 272-ML13 ship α=0 only). CT19 `LaplaceBeltramiViaAlphaOne(W, n, dim, t)` ~60 LOC the α=1 case-witness — provably-converges to Laplace-Beltrami on the manifold INDEPENDENT of sampling-density (in α=0 the limit depends on density, biasing embedding toward dense-regions). CT20 `FokkerPlanckViaAlphaHalf(W, n, dim, t)` ~80 LOC α=1/2 Fokker-Planck-with-no-drift limit — substrate for over-damped-Langevin-dynamics-on-data-manifold), **(e) Tier-4 Nyström-out-of-sample formal-derivation ~280 LOC SHARED-WITH-272-ML34c** (CT21 `manifold/consistency/nystrom.go::NystromExtension(K_train, eigvecs, eigvals, K_new_train, k int) Y_new` ~120 LOC Williams-Seeger-2000-NIPS canonical-Nyström-formula `f̂_new(j) = (1/λ_j) Σ_i u_j(i) k(x_new, x_i)` — DUPLICATE-WITH-272-ML34c-Bengio-2003 (Bengio's unified-framework IS Nyström-applied-to-each-ML-kernel). 273 ships the FORMAL-DERIVATION-WITH-error-bounds version: `‖f − f̂‖_∞ ≤ C·(λ_k+1/λ_k) + O(1/√m)` Drineas-Mahoney-2005-JMLR-6:2153 sub-sampled-Nyström-error. CT22 `NystromError(K_full, K_subsample, m_subsample, n_total int) error_upper_bound` ~80 LOC Drineas-Mahoney-2005 explicit-bound; CT22b `NystromAdaptiveSampling(K, n, m_target int, leverage_scores []float64) sample_indices` ~80 LOC Gittens-Mahoney-2016-JMLR leverage-score-Nyström for optimal-sub-sampling-error. **CROSS-LINK to 236-RKHS-K15-KernelPCA**: every Nyström-consistency-bound is an RKHS-projection-error-bound; 273-CT21/CT22 share the kernel-substrate with 236.), **(f) Tier-5 spectral-consistency-of-clustering ~300 LOC** (CT23 `manifold/consistency/spectral_consistency.go::vonLuxburgConsistency(L_type string, n int, k int, kernel_bandwidth float64) consistency_certificate` ~140 LOC von-Luxburg-Belkin-Bousquet-2008-Annals-Stat-36:555 the FOUNDATIONAL consistency-of-spectral-clustering-theorem: **L_sym is consistent for k-clustering as n→∞ if eps_n bandwidth shrinks at right rate; L_unnormalised is INCONSISTENT in general** (the famous "unnormalised-Laplacian-fails" theorem). Returns certificate "consistent | inconsistent | borderline" + counterexample-if-inconsistent. The single-most-important consistency-theorem in spectral-graph-theory; ships TODAY against 271+272 algorithm-substrate. CT24 `kNNGraphConsistency(k_value, n int, d_intrinsic int) (consistent_iff bool, rate float64)` ~80 LOC Maier-vonLuxburg-Hein-2008-NeurIPS k-NN-graph-Laplacian consistency: `k_n → ∞` with `k_n / log(n) → ∞` AND `k_n / n → 0` necessary-and-sufficient for kNN-Laplacian-consistency; CT25 `EpsGraphConsistency(eps, n int, d_intrinsic int) (consistent_iff bool, rate float64)` ~80 LOC ε-ball-graph analogue, the original Belkin-Niyogi-2003-Neural-Comput-15:1373 §6 conditions).

**SINGULAR-CHEAPEST-1-DAY-SUBSET CT1+CT2+CT5+CT6+CT8+CT9 ~480 LOC** — closed-form-Laplace-Beltrami-spectra on sphere/torus + bandwidth-rate-witness + bias-variance-tradeoff + Davis-Kahan + Weyl. ZERO blockers; all pure-scalar/closed-form arithmetic + named-bound-formulas. **SINGULAR-ARCHITECTURAL-KEYSTONE CT8 DavisKahanSinTheta + CT13 EigenmapsSampleComplexity ~240 LOC** — every spectral-method's quality-guarantee reduces to one-of-these-two-bounds. CT8 says "spectral-gap × ‖perturbation‖ → embedding-error", CT13 inverts to "target-error → required-n". These are the canonical "WHY does spectral-embedding work?" answer-pair. **SINGULAR-MOAT CT18+CT19+CT20 ~280 LOC ANISOTROPIC-DIFFUSION-α-FAMILY** — the SINGULAR genuinely-new-algorithm that 273 contributes vs 271+272. The α=0 default in 271-SC19 / 272-ML13 has the well-known density-bias artefact (eigenvectors concentrate on dense-regions of data); α=1 Laplace-Beltrami removes density-bias entirely (Coifman-Lafon-2006-§3). On non-uniform-sampled-manifolds (e.g. single-cell-RNA-seq with cell-type-imbalance), the α=1 variant is provably-better. **SINGULAR-2024-FRONTIER CT13+CT14+CT17 ~240 LOC** — García-Trillos-Slepčev-2020 + Calder-García-Trillos-2022 + Linderman-Steinerberger-2019 / Arora-Hu-Kothari-2018 — the 2018-2024 wave of explicit-finite-sample bounds for embedding-quality (older Belkin-Niyogi-2008 bounds were asymptotic; the 2018-2024 wave gives non-asymptotic with explicit constants). **SINGULAR-PEDAGOGICAL CT23 vonLuxburgConsistency ~140 LOC** — the FAMOUS-result "unnormalised Laplacian is INCONSISTENT" is the single-most-counterintuitive theorem in spectral-graph-theory, almost-never-implemented anywhere. Saturates a R-MUTUAL-CROSS-VALIDATION 3/3 pin: empirical-test-on-imbalanced-degree-graph showing L_unnormalised-eigenvectors converge to a NON-clustering limit-function while L_sym-eigenvectors converge to the correct cluster-indicator. Recommended placement: **`manifold/consistency/`** sub-package within the 272 namespace IF 272 lands first, OR `graph/spectral/consistency/` IF 271 lands first — either way, this is a **theory-witness sub-package SUBORDINATE to whichever algorithm-package owns the spectral-embedding namespace**, not a standalone graph/spectral/manifold-learning package competing with 271/272.

---

## 0. State at HEAD (2026-05-09, v0.10.0) — CANDOR audit

Repo-wide grep for ANY consistency-theory / sample-complexity / convergence-rate / Davis-Kahan / Weyl / Laplace-Beltrami / heat-kernel-on-manifold / anisotropic-diffusion / Coifman-Lafon-α / Williams-Seeger / Drineas-Mahoney / von-Luxburg-Belkin-Bousquet / Maier-Hein-vonLuxburg / Hein-Audibert / García-Trillos / Calder / Belkin-Niyogi-convergence / Singer-2006 / Linderman-Steinerberger / Arora-Kothari — **zero callable matches** in `*.go`.

| Surface | Path | Consistency-theory relevance |
|---|---|---|
| `linalg.PCA + ipi-private` | `linalg/pca.go` | The linear-baseline consistency-theory reduces to (Anderson-1963 + Davis-1977 + Kollo-Neudecker-1993) |
| `linalg.QRAlgorithm` | `linalg/eigen.go` | Eigenvalues-only — same blocker as 271+272 |
| `infogeo.Kernel + GaussianKernel + LaplacianKernel + MedianHeuristicBandwidth` | `infogeo/mmd.go` | **Kernel-substrate every consistency-theorem assumes**; bandwidth is the central tuning-parameter convergence-rates depend on |
| `graph.Dijkstra + FloydWarshall` | `graph/shortest.go` | Substrate for Isomap-consistency (CT15) — REUSE |
| `prob.Distribution + Gaussian` | `prob/distributions.go` | Substrate for sample-complexity (Pinsker-bounds + concentration-inequalities) — present |
| 097-T1.eigvec linalg.InverseIteration | -- | **ABSENT**, same shared-blocker with 271+272 |
| 271-Tier-3 SC17-SC21 (algorithms) | -- | **ABSENT**; 271 enumerates ship-plan, no PR landed |
| 272-Tier-2 ML11-ML15 + Tier-3 ML16-ML18 (algorithms) | -- | **ABSENT**; 272 enumerates ship-plan, no PR landed |
| 273-CT1-CT25 consistency-theory primitives | -- | **ALL ABSENT** — this slot creates |

**False-positive name-collisions audited:**
- `infogeo/fdiv.go` has Pinsker-inequality KL ≥ TV² for Pinsker-Csiszár-1966 — DIFFERENT context (probability-divergence Pinsker, NOT spectral-consistency Pinsker). Both share the name; 273-Pinsker-style-bounds use the spectral-perturbation context (Davis-Kahan-style-not-Csiszár-style).
- `chaos/lyapunov.go` "tangent space" / "limit operator" — dynamical-systems-context; not data-manifold-Laplace-Beltrami.
- `topology/persistent/vr.go` "persistence diagram" — TDA persistent-homology stability theorem (Cohen-Steiner-Edelsbrunner-Harer-2007) is a perturbation-bound but for persistence-diagrams, not eigenvectors.
- `linalg/perturbation.go` — does NOT exist (would be the natural home for Weyl + Davis-Kahan if reality grew a perturbation-theory primitive sub-package; 097 enumerates this in T1.perturbation).

**Cross-import edges that this slot creates:**
- `manifold/consistency → manifold.{LaplacianEigenmaps, LLE, DiffusionMap, Isomap, KernelPCA}` (272 ALGORITHMS provide the discrete-embeddings the consistency-witnesses validate)
- `manifold/consistency → graph/spectral.{LaplacianMatrix, SymNormalizedLaplacian, RandomWalkLaplacian}` (157-G2/G3/G4 substrate)
- `manifold/consistency → linalg.{Norm, Trace, MatVecMul, EigenvaluesAndVectors}` (perturbation-bounds reduce to operator-norms + eigenvalue-arithmetic)
- `manifold/consistency → infogeo.{Kernel, GaussianKernel}` (bandwidth-consistency theorems)
- `manifold/consistency → prob.Concentration` (sample-complexity reduces to McDiarmid + Bernstein concentration-inequalities — note: prob/concentration ABSENT, soft-blocker)

**Strict downstream consumers of `manifold/consistency/`:**
- 271-spectral-clustering algorithm-validation: every SC algorithm should report a CT-consistency-certificate alongside its output
- 272-manifold-learning algorithm-validation: every ML algorithm should report a CT-sample-complexity witness
- 270-graph-signal-proc bandlimited-projection-quality: GFT-bandlimited approximation depends on Laplacian-eigenvector-consistency
- 236-RKHS Mercer-decomposition-consistency: Williams-Seeger-Nyström-bounds are the dual-of-RKHS-truncation-error
- 226-hyperbolic-embedding hyperbolic-manifold-consistency: hyperbolic-Laplace-Beltrami spectrum is well-defined; CT1 generalises to negatively-curved spaces
- 153-prob-infogeo Fisher-information-metric on statistical-manifolds: Belkin-Niyogi-2008 connects manifold-Laplacian to Fisher-information

---

## 1. The twenty-five primitives (CT1-CT25)

Each entry: name, LOC, reference, what-it-witnesses, blocker-if-any.

### Tier 0 — Limit-operators + bandwidth-rates (~480 LOC)

**CT1 `manifold/consistency/limit_operator.go::LaplaceBeltramiOnSphere(d_dim, max_degree int) (eigvals []float64, multiplicities []int)` ~80 LOC.** Helgason-1984 / Stein-Weiss-1971-Princeton-Math-Series. Closed-form spectrum of −Δ on unit-sphere S^d ⊂ R^{d+1}: `λ_k = k(k+d-1)` with multiplicity `binomial(k+d, d) − binomial(k+d-2, d)`. The gold-standard oracle for verifying that discrete-Laplacian-on-sphere-sampled-points converges to the continuous-Laplace-Beltrami. Pure closed-form; zero blockers.

**CT2 `LaplaceBeltramiOnTorus(d_dim int, modes [][]int) []float64` ~60 LOC.** Flat-torus T^d = R^d / (2π Z)^d. Eigenfunctions are `e^{i k·x}` with eigenvalues `λ_k = ‖k‖²`. Closed-form. Substrate for Coifman-Lafon-2006 §6 numerical-experiment validation.

**CT3 `LaplaceBeltramiOnFlatRectangle(L_x, L_y float64, modes [][2]int, bc string) []float64` ~60 LOC.** Dirichlet (sin·sin) or Neumann (cos·cos) eigenfunctions on rectangle. `λ_{m,n} = π²(m²/L_x² + n²/L_y²)`. The simplest non-trivial domain; substrate for Belkin-Niyogi-2003 §6.1 numerical-experiment.

**CT4 `HeatKernelOnSphere(t float64, x_dot_y float64, d_dim, max_degree int) float64` ~80 LOC.** Müller-1966 / Stein-Weiss-1971. Heat-kernel on S^d via zonal-spherical-harmonic-expansion `K_t(x,y) = Σ_k e^{-λ_k t} · ((2k+d-1) / ((d-1)·ω_d)) · C_k^((d-1)/2)(x·y) / C_k^((d-1)/2)(1)` truncated at max_degree. Substrate for verifying Coifman-Lafon-2006 t-step Markov-kernel converges to heat-kernel on manifold. Uses Gegenbauer-polynomial-recurrence (closed-form).

**CT5 `LimitGraphLaplacianBandwidth(n int, eps float64, d_intrinsic int) (rate float64, regime string)` ~60 LOC.** Hein-Audibert-vonLuxburg-2007 Annals-Stat-35:1 §3 Theorem-15. Necessary-and-sufficient condition for pointwise-consistency of unnormalised graph-Laplacian L_n on iid-sample from compact d-manifold:
- `eps_n → 0` AND `n · eps_n^{d+2} / log(n) → ∞`
Returns regime "consistent | inconsistent | borderline" + rate-witness `n^{-2/(d+4)}` if optimal-eps-chosen.

**CT6 `BiasVarianceTradeoffBandwidth(n int, d_intrinsic int, target_error float64) (eps_optimal, rate float64)` ~80 LOC.** Singer-2006 ACHA-21:128 §3 + Coifman-Lafon-2006 §4. Optimal eps-bandwidth minimising bias `O(eps²)` (manifold-curvature-error) + variance `O(1/(n · eps^{d+2}))` (sampling-noise-error) yields `eps_opt ∼ n^{-1/(d+4)}` and convergence-rate `‖L_n − L_M‖ = O(n^{-2/(d+4)})`. **The optimal-bandwidth oracle every kNN-bandwidth-selection should converge to.** Single-most-important practical bound for tuning ε in production embeddings.

**CT7 `KernelPolynomialBoundConsistency(kernel_type string, manifold_curvature float64, n int, alpha float64) (bias, variance, total_error)` ~60 LOC.** Coifman-Lafon-2006 §4 anisotropic-α-family explicit bias-variance-decomposition for α ∈ {0, 1/2, 1}. Returns the per-α error-decomposition; complements CT6 with α-dependence.

### Tier 1 — Perturbation bounds + Davis-Kahan (~440 LOC)

**CT8 `manifold/consistency/davis_kahan.go::DavisKahanSinTheta(eigvals_pop []float64, eigvals_emp []float64, k int, perturbation_norm float64) sin_theta` ~140 LOC — KEYSTONE.** Davis-Kahan-1970 SIAM-J-Numer-Anal-7:1 Theorem-V.4 + Yu-Wang-Samworth-2015 Biometrika-102:315 the "useful-variant-of-Davis-Kahan". For symmetric matrices A (population) and Â (empirical):
```
‖sin Θ(V_k(A), V_k(Â))‖_F ≤ √(2·k) · ‖A − Â‖_op / (λ_k(A) − λ_{k+1}(A))
```
where V_k = top-k-eigenvectors, Θ = canonical-angles. The denominator is the **spectral-gap**; embedding-stability requires this gap to be sufficiently large. **The canonical perturbation-bound for spectral-embedding stability** in EVERY paper post-1970. Returns sin-θ + sample-size-lower-bound necessary for target sin-θ. Pure scalar arithmetic; zero blockers.

**CT9 `WeylPerturbationBound(eigvals []float64, perturbation_op_norm float64) per_eigvalue_bound []float64` ~80 LOC.** Weyl-1912 Math-Ann-71:441. `|λ_k(A+E) − λ_k(A)| ≤ ‖E‖_op` per-eigenvalue-additive bound. Substrate for spectral-gap-stability witness + sample-complexity bounds. Pure scalar; ships TODAY.

**CT10 `WeylSchurInterlacing(eigvals_full []float64, mask []bool) interlaced_eigvals` ~60 LOC.** Cauchy-1829 / Schur-1923 inclusion-principle: eigenvalues of principal-submatrix interlace eigenvalues of full-matrix. Useful for sub-graph-spectral-stability + leave-one-out-cross-validation of spectral-embeddings.

**CT11 `MattilaSpectralGapStability(eigvals []float64, k_clusters int) (gap, perturbation_max_safe)` ~80 LOC.** Computes `Δ_k = λ_{k+1} − λ_k` AND inverts Davis-Kahan to give the maximum perturbation-norm that preserves k-cluster-structure. The reverse-direction-of-CT8: instead of "given perturbation, what's embedding-error?" answers "given target-cluster-stability, what's the max-tolerable-perturbation?".

**CT12 `WedinTheoremSVD(sv_pop []float64, sv_emp []float64, k int, perturbation_norm float64) sin_theta` ~80 LOC.** Wedin-1972 BIT-12:99. SVD-version of Davis-Kahan for non-symmetric matrices — substrate for non-backtracking-walk-matrix B perturbation in 271-SC22 SBM-recovery (B is non-symmetric; vanilla Davis-Kahan inapplicable).

### Tier 2 — Sample-complexity calculators (~360 LOC)

**CT13 `manifold/consistency/sample_complexity.go::EigenmapsSampleComplexity(d_intrinsic, k_eigenvecs int, target_sin_theta, manifold_volume, curvature_max float64) n_required int` ~100 LOC — KEYSTONE.** García-Trillos-Slepčev-2020 Found-Comput-Math-20:827 + Calder-García-Trillos-2022 Found-Comput-Math-22:1037. Explicit non-asymptotic constants:
```
n ≥ C(d, M, λ_k) · log(n) · target_sin_theta^{-2-d/2}
```
for k-th-eigenfunction-ε-recovery on compact-d-manifold-M. **Answers "how many samples do I need for embedding to be ε-correct?"** which neither 271 nor 272 answer. Returns n_required + which-sub-bound dominates (curvature-bound vs spectral-gap-bound vs density-bound).

**CT14 `LLESampleComplexity(d_intrinsic, dim_target, k_neighbours int, noise_level float64) n_required` ~60 LOC.** Belkin-Niyogi-2008 J-Comput-Syst-Sci-74:1289 Theorem-3.1 LLE-specific sample-complexity using local-tangent-PCA error-bound `O(eps^{d+2}/n)` per-point bias. Concrete-formula `n ≥ Cd · log(n) · (k/eps)^{2(d+2)}`.

**CT15 `IsomapSampleComplexity(d_intrinsic int, manifold_diameter, target_geodesic_error float64, kNN_k int) (n_required, kNN_required int)` ~60 LOC.** Bernstein-deSilva-Langford-Tenenbaum-2000 Stanford-Tech-Rep + Bernstein-2000-Adv-Geom-1:99 geodesic-recovery rate `n^{-2/d}` with kNN-k-required `≥ Cd · log(n)`. Explicit-sample-size oracle for Isomap.

**CT16 `DiffusionMapSampleComplexity(d_intrinsic int, t_diffusion, target_distance_error float64, k_eigenvecs int) n_required` ~60 LOC.** Singer-2006 ACHA-21:128 + Singer-Wu-2017 CommACM-60:54 anisotropic-diffusion-rate `n^{-2/(d+4)}` with t-dependent constants. Substrate for adaptive-t-selection in 272-ML13 / 271-SC19.

**CT17 `tSNEStatisticalConsistency(n int, d_intrinsic, perplexity, dim_target int) (cluster_recovery_rate, distortion_upper_bound float64)` ~80 LOC.** Linderman-Steinerberger-2019 AOS-47:2398 + Arora-Hu-Kothari-2018 COLT 76:1455 t-SNE-consistency-theorem-proof for cluster-recovery in well-separated regime. NEW vs 272-ML21 (272 ships algorithm-only without statistical-consistency-witness). Important because t-SNE has been criticised for visual-distortion-of-distances; the Linderman-2019 result PROVES cluster-separation IS preserved (under separability-assumption). Returns "well-separated | borderline | failure-mode" classification.

### Tier 3 — Anisotropic-diffusion-α-family (~280 LOC, GENUINELY-NEW-VS-271+272)

**CT18 `manifold/consistency/anisotropic_diffusion.go::AnisotropicDiffusionMap(W [][]float64, n, dim int, alpha, t float64) Y [][]float64` ~140 LOC — SINGULAR-MOAT.** Coifman-Lafon-2006 ACHA-21:5 §3 the FULL α-family normalisation that 271-SC19 / 272-ML13 OMIT (those ship α=0 default only):
```
Step 1: K_alpha = D^{-α} W D^{-α}            (anisotropic-correction)
Step 2: P_alpha = (D_alpha)^{-1} K_alpha     (row-normalise)
Step 3: eigendecompose P_alpha → diffusion-map
```
Three different limit-operators depending on α:
- **α = 0** (default in 271/272): limit is Laplacian on data **with density bias** — eigenvectors concentrate on dense-regions; embedding distorted toward dense-clusters.
- **α = 1/2**: limit is **Fokker-Planck operator** (Langevin-dynamics-with-no-drift on data-density).
- **α = 1**: limit is **pure Laplace-Beltrami** on the manifold, **density-bias-removed** — embedding reflects geometry alone, not sampling-density.

The α-family is the **SINGULAR genuinely-new-algorithmic-contribution** that 273 makes vs 271+272. On non-uniformly-sampled-manifolds (e.g. single-cell-RNA-seq with cell-type-imbalance, or active-learning data with selection-bias), α=1 gives provably-better embedding than α=0. Pre-2019 embeddings paper-publications converged on α=1 as default (PHATE-Moon-2019 uses α=1 internally). **272-ML13-DiffusionMap should be α-parametrised with α=1 default; 273-CT18 is the non-default α=0 + α=1/2 + α=1 family-witness with explicit bias-comparison-test.**

**CT19 `LaplaceBeltramiViaAlphaOne(W, n, dim, t float64) Y` ~60 LOC.** The α=1 case-witness — provably-converges to Laplace-Beltrami on manifold INDEPENDENT of sampling-density. The "geometry-only" embedding.

**CT20 `FokkerPlanckViaAlphaHalf(W, n, dim, t float64) Y` ~80 LOC.** The α=1/2 case — Fokker-Planck-with-no-drift limit. Substrate for over-damped-Langevin-dynamics-on-data-manifold.

### Tier 4 — Nyström-out-of-sample formal-derivation (~280 LOC, OVERLAPS-272-ML34c)

**CT21 `manifold/consistency/nystrom.go::NystromExtension(K_train [][]float64, eigvecs, eigvals, K_new_train, k int) Y_new` ~120 LOC.** Williams-Seeger-2000 NIPS canonical-Nyström-formula:
```
f̂_new(j) = (1/λ_j) Σ_i u_j(i) k(x_new, x_i)
```
**DUPLICATE-WITH-272-ML34c-Bengio-2003** (Bengio's unified-framework IS Nyström-applied-to-each-ML-kernel). 273 ships the FORMAL-DERIVATION-WITH-error-bounds version with explicit `‖f − f̂‖_∞ ≤ C·(λ_k+1/λ_k) + O(1/√m)` Drineas-Mahoney-2005-JMLR-6:2153 sub-sampled-Nyström-error-bound. **272 ships the algorithm; 273 ships the error-bound. Recommend: 272-ML34c absorbs CT21 with error-bound add-on ~40 LOC delta**.

**CT22 `NystromError(K_full, K_subsample, m_subsample, n_total int) error_upper_bound` ~80 LOC.** Drineas-Mahoney-2005 explicit-bound `‖K − K̂‖_F ≤ ‖K_residual‖_F + (n/m) · ‖K_subsample‖_F`. Substrate for Nyström-quality-validation.

**CT22b `NystromAdaptiveSampling(K, n, m_target int, leverage_scores) sample_indices` ~80 LOC.** Gittens-Mahoney-2016 JMLR-17:117 leverage-score-Nyström for optimal-sub-sampling-error. Cross-link to 236-RKHS leverage-scores. **Cross-link to 262-random-projection** Boutsidis-Mahoney-Drineas-2009 RandomNyström.

### Tier 5 — Spectral-consistency of clustering (~300 LOC)

**CT23 `manifold/consistency/spectral_consistency.go::vonLuxburgConsistency(L_type string, n int, k int, kernel_bandwidth float64, density_var float64) (consistent bool, certificate string, counterexample [][]float64)` ~140 LOC — KEYSTONE-PEDAGOGICAL.** von-Luxburg-Belkin-Bousquet-2008 Annals-Stat-36:555 the FOUNDATIONAL-consistency-theorem-of-spectral-clustering:
- **L_sym (symmetric-normalised) IS consistent** for k-clustering as n→∞ if eps_n bandwidth shrinks at right rate.
- **L_rw (random-walk) IS consistent** under same conditions.
- **L_unnormalised (D − A) IS INCONSISTENT** in general — it converges to a NON-clustering limit operator (the operator depends on degree-distribution-density, not just cluster-structure).

The famous "unnormalised-Laplacian-fails" theorem. Returns certificate "consistent | inconsistent" + counterexample-graph if inconsistent. **Single-most-important consistency-theorem in spectral-graph-theory**, almost-never-implemented anywhere. Saturates a R-MUTUAL-CROSS-VALIDATION 3/3 pin: empirical-test-on-imbalanced-degree-graph showing L_unnormalised-eigenvectors converge to wrong-limit while L_sym-eigenvectors converge to correct cluster-indicator.

**CT24 `kNNGraphConsistency(k_value, n, d_intrinsic int) (consistent_iff bool, rate float64, regime string)` ~80 LOC.** Maier-vonLuxburg-Hein-2008 NeurIPS k-NN-graph-Laplacian-consistency: `k_n → ∞` with `k_n / log(n) → ∞` AND `k_n / n → 0` necessary-and-sufficient for kNN-Laplacian-consistency. Concrete-formula-with-explicit-constants for choosing k in production.

**CT25 `EpsGraphConsistency(eps, n int, d_intrinsic int) (consistent_iff bool, rate float64, regime string)` ~80 LOC.** ε-ball-graph analogue, the original Belkin-Niyogi-2003 Neural-Comput-15:1373 §6 conditions. Companion-to-CT5.

---

## 2. Cross-package blockers + connective-tissue

**Substrate-blocker-1 (HARD)** `linalg.InverseIteration` extract from PCA-private — same shared-blocker with 271+272+097-T1.eigvec. Gates CT8 + CT11 (need eigenvectors for sin-Θ computation between population and empirical eigenspaces). Gates 18-of-25 primitives (every CT depending on eigenvector-comparison).

**Substrate-blocker-2 (SOFT)** 272-ML11 LaplacianEigenmaps + 271-SC17 ditto — CT consistency-witnesses ARE applied to outputs of these algorithms. Without 271/272 landed, CT primitives have no consumer. Recommend: ship 272 first (the broader namespace), 273 follows as `manifold/consistency/`.

**Substrate-blocker-3 (NONE)** `infogeo.Kernel + GaussianKernel + MedianHeuristicBandwidth` PRESENT — CT5/CT6 bandwidth-witnesses ship TODAY against existing kernel-substrate.

**Substrate-blocker-4 (NONE)** `linalg.{Norm, Trace}` PRESENT — CT8/CT9 perturbation-bounds use only operator-norms.

**Substrate-blocker-5 (SOFT)** `prob/concentration.{McDiarmid, Bernstein}` ABSENT — CT13/CT14/CT15 sample-complexity bounds technically depend on McDiarmid-1989 / Bernstein-1924 concentration-inequalities, but the EXPLICIT-CONSTANT formulas in García-Trillos-Slepčev-2020 / Calder-García-Trillos-2022 are scalar-closed-form once the manifold-parameters are given. Ship CT13-CT15 as scalar-formula-witnesses; defer prob/concentration as separate slot (097-T2-concentration roadmap).

**Total upstream-substrate dependency** (assuming 271-PR-A or 272-PR-A lands first as algorithm-substrate): ~80 LOC linalg.InverseIteration extract + 271/272 ALGORITHMS present → ~2,140 LOC of `manifold/consistency/` consumer-side closes the consistency-theory canon.

**Cheapest-no-blocker subset:** **CT1+CT2+CT3+CT5+CT6+CT9+CT13+CT14+CT15+CT24+CT25 ~720 LOC** — closed-form-Laplace-Beltrami + bandwidth-rate-witnesses + Weyl + sample-complexity-formulas + graph-consistency-conditions. ZERO blockers; pure-scalar/closed-form. **Could ship TODAY without 271/272 algorithms landing — they validate any future-algorithm against analytical limits.**

**Recommended PR sequence:**

- **PR-A (Tier-0 limit-operators ~480 LOC, 1 week)** CT1+CT2+CT3+CT4+CT5+CT6+CT7. Pure closed-form; oracle-substrate for ALL future spectral-method validation.
- **PR-B (Tier-1 perturbation-bounds ~440 LOC, 1 week)** CT8 Davis-Kahan + CT9 Weyl + CT10 Cauchy-Schur + CT11 Mattila + CT12 Wedin. Scalar-arithmetic; cross-applicable far beyond manifold-learning (any-eigenvalue-based-method benefits).
- **PR-C (Tier-2 sample-complexity ~360 LOC, 1 week)** CT13+CT14+CT15+CT16+CT17. Closed-form-formulas-with-explicit-constants from García-Trillos-Slepčev / Belkin-Niyogi / Singer / Linderman-Steinerberger / Arora-Hu-Kothari.
- **PR-D (Tier-3 anisotropic-α ~280 LOC, 3 days)** CT18+CT19+CT20. **THE singular-genuinely-new-algorithm 273 contributes**.
- **PR-E (Tier-4 Nyström-bounds ~280 LOC, 3 days)** CT21 absorbs-and-extends-272-ML34c with Drineas-Mahoney-2005 error-bound + CT22 + CT22b leverage-score-sampling.
- **PR-F (Tier-5 spectral-consistency ~300 LOC, 1 week)** CT23 vonLuxburg-Belkin-Bousquet-2008 + CT24 + CT25. **PR-F is the keystone-pedagogical: the famous unnormalised-Laplacian-fails-theorem with empirical-counterexample-witness**.

Total ~2,140 LOC across 6 PRs, ~3-4 engineer-weeks net of substrate.

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled

**Pin 273-1 — Sphere-spectrum convergence (3/3).** On 1000 iid-uniform-points on S^2, three quantities must agree:
- CT1 closed-form `λ_k = k(k+1)` for k=0,1,2,...
- 272-ML11 LaplacianEigenmaps eigvals (numerical-discrete) on the 1000-point sample
- CT8 Davis-Kahan-bound predicts sin-θ between population-and-empirical-eigenspaces

All three agree to within sample-complexity bound CT13 predicts.

**Pin 273-2 — Anisotropic-α witnesses density-bias (3/3).** On non-uniform-sampled annulus with 80% inner-ring / 20% outer-ring:
- CT18 with α=0 produces eigenvectors concentrated on inner-ring (density-biased)
- CT18 with α=1 produces eigenvectors uniform along annulus (geometry-only)
- CT4 heat-kernel-on-circle predicts the α=1 limit

Three-way-agreement validates the α=1 Laplace-Beltrami-density-removed property.

**Pin 273-3 — Davis-Kahan vs empirical sin-θ vs sample-complexity (3/3).** Across n ∈ {100, 1000, 10000}:
- CT8 predicts sin-θ ≤ √2 · ‖perturbation‖_op / Δ_k
- Empirical sin-θ between population (n→∞ closed-form) and empirical (finite-n) eigenspaces
- CT13 predicts n_required for target sin-θ; reverse-direction check holds

**Pin 273-4 — Unnormalised vs normalised Laplacian inconsistency (3/3).** On imbalanced-degree-distribution graph (Pareto-degree-dist):
- CT23 certificate flags L_unnormalised "inconsistent"
- Empirical L_sym eigenvectors recover correct k-cluster-structure
- Empirical L_unnormalised eigenvectors recover wrong (degree-biased) structure

The famous von-Luxburg-Belkin-Bousquet-2008 counterexample reproduced.

**Pin 273-5 — Nyström sub-sampling error (3/3).** On 5000-point dataset, sub-sample m=500:
- CT21 NystromExtension produces Y_new for held-out 4500 points
- CT22 predicts ‖K − K̂‖_F upper-bound
- Empirical reconstruction-error within bound

---

## 4. Per-area: what 273 adds beyond 271+272

| Area listed in slot brief | Covered in 271? | Covered in 272? | Unique to 273? |
|---|---|---|---|
| Laplacian Eigenmaps | YES (SC17) | YES (ML11) | NO algorithm — but YES consistency (CT13/CT23/CT24/CT25) |
| LLE | YES (SC18) | YES (ML12) | NO algorithm — but YES sample-complexity (CT14) |
| Hessian LLE | NO | YES (ML16) | NO — fully covered in 272 |
| LTSA | NO | YES (ML18) | NO — fully covered in 272 |
| Diffusion Maps | YES (SC19, α=0 only) | YES (ML13, α=0 only) | **YES α-family (CT18/CT19/CT20)** |
| Isomap | YES (SC20) | YES (ML14) | NO algorithm — but YES sample-complexity (CT15) |
| Theoretical convergence Belkin-Niyogi-2008 | NO | NO | **YES (CT13/CT14/CT25)** |
| Continuous limit Laplacian operator | NO | NO | **YES (CT1-CT4, closed-form spectra)** |
| Discrete-to-continuous limits | NO | NO | **YES (CT5/CT6/CT7 bandwidth-rate witnesses)** |
| Spectral consistency | NO | NO | **YES (CT8/CT11/CT23 Davis-Kahan + von-Luxburg)** |
| Random walks → diffusion convergence | implicit-only | implicit-only | **YES (CT4/CT16 heat-kernel + Singer-2006-rate)** |
| Heat-kernel convergence | NO | NO | **YES (CT4 closed-form-on-sphere)** |
| Anisotropic diffusion maps | NO | NO | **YES (CT18-CT20, the genuine-new-algorithm)** |
| Kernel choice impact | implicit-only | implicit-only | **YES (CT7 bias-variance per-kernel)** |
| Consistency vonLuxburg-Belkin-Bousquet-2008 | NO | NO | **YES (CT23, the famous theorem)** |
| Out-of-sample Nyström | NO | YES (ML34c, algorithm-only) | YES error-bound (CT21+CT22) |
| Dimension estimation via spectral | NO | YES (ML7c-e Levina-Bickel + Hein-Audibert) | NO — fully covered in 272 |
| Laplacian-based regularisation | NO | NO | **YES (CT-bonus: CT13 implies regularisation-strength via spectral-gap)** |
| Learning the metric from data | NO | NO | **YES (CT18 α-family chooses metric)** |
| Spectral embedding evaluation | NO | YES (ML30-34 Trustworthiness/Continuity/Co-ranking) | NO — fully covered in 272 |
| Convergence of spectral clustering n→∞ | NO | NO | **YES (CT23 von-Luxburg-2008 + CT24-CT25)** |
| Pinsker bound | NO | NO | **YES (Davis-Kahan-style Pinsker, CT8)** |
| Sample complexity | NO | NO | **YES (CT13-CT17 entire-tier-2)** |

**Summary table: of 22 listed areas in the brief, 14 are EXCLUSIVELY in 273, 6 are duplicate-with-271/272-as-algorithms-where-273-adds-theory-witnesses, 2 are fully-covered-in-272.**

---

## 5. Recommendation: KEEP 273 as a focused theory-witness sub-package

**Verdict: 273 IS NON-REDUNDANT vs 271+272, but only in the consistency-theory + sample-complexity layer. The slot title ("Spectral embedding: Laplacian eigenmaps, locally linear") is misleading — taken literally, 273 IS redundant with 271-Tier-3 + 272-Tier-2. Reframe 273 as `manifold/consistency/` (theory-witnesses) and the slot becomes uniquely-valuable.**

**Recommendation sequence:**

1. **Land 272 first** — it owns the `manifold/` namespace + algorithms.
2. **Optionally land 271** — it owns `graph/spectral/` for clustering-and-cuts (overlap ~720 LOC SHARED with 272-Tier-2).
3. **Land 273 as `manifold/consistency/`** — the consistency-theory layer that BOTH 271 and 272 omit by architecture. ~2,140 LOC entirely-non-redundant.

**Alternative if compression preferred:** absorb 273 entirely into 272 as `manifold/consistency.go` + `manifold/limit_operator.go` + `manifold/perturbation.go` files. 272 grows from ~4,640 to ~6,780 LOC; the consistency-theory becomes integrated rather than separate-package.

**Anti-recommendation: do NOT delete 273 outright.** The 22 primitives CT1-CT25 are genuinely-distinct contributions. The slot is rescuable; only the algorithm-list framing is misleading.

**Stylistic-note on 273 vs 271/272:** 273 is the **deepest theory-zoom** in the spectral-trio, focused on "WHY does spectral-embedding work?" answers. 271 is the **algorithm-zoom on clustering** ("HOW to cluster via spectrum?"). 272 is the **algorithm-zoom on embedding** ("HOW to embed nonlinearly?"). Together: 271 + 272 + 273 = the **complete-spectral-graph-theory-canon** consumed-by aicore + downstream-Pistachio-applications. None of the three is dispensable; the dependency-graph is 273 ← 272 ← 271 ← 157-substrate.

---

## 6. LOC budget summary

| Tier | Count | LOC | Notes |
|---|---|---|---|
| Tier 0 limit-operators | 7 (CT1-CT7) | 480 | Closed-form, ships TODAY |
| Tier 1 perturbation-bounds | 5 (CT8-CT12) | 440 | Scalar arithmetic, ships TODAY |
| Tier 2 sample-complexity | 5 (CT13-CT17) | 360 | Closed-form formulas, ships TODAY |
| Tier 3 anisotropic-α | 3 (CT18-CT20) | 280 | NEW-ALGORITHM, blocked-soft-on-272 |
| Tier 4 Nyström-bounds | 3 (CT21-CT22b) | 280 | Overlaps 272-ML34c; +error-bounds |
| Tier 5 spectral-consistency | 3 (CT23-CT25) | 300 | Famous-theorems, ships TODAY |
| **Total** | **22 (CT1-CT25)** | **2,140** | ~3-4 engineer-weeks |
| **Cheapest-no-blocker subset** | 11 | 720 | Pure scalar/closed-form |
| **Genuinely-new-vs-271+272** | 22 of 22 | 2,140 | Zero algorithm-overlap |
| **Singular-moat vs 271+272** | 3 (CT18-CT20) | 280 | Anisotropic-α-family |

End — 273 is rescuable as a theory-witness sub-package. Ship CT1-CT25 to close the consistency-theory canon.
