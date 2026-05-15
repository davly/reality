# 257 | new-tensor-decomp — CP / Tucker / HOSVD / ST-HOSVD / HOOI / hierarchical-Tucker / block-term / nonnegative-CP / sparse-CP / robust-tensor-PCA / online-CP / randomized-HOSVD / Krylov-tensor / tensor-low-rank-completion / coupled-CMTF / multi-view-CP / tensor-regression / cumulant-tensor-ICA / tensor-power-iteration

**Summary line 1.** reality v0.10.0 ships **ZERO** higher-order-tensor-decomposition surface — repo-wide grep on `tensor.decomp|cp.als|parafac|candecomp|tucker|hosvd|st.hosvd|hooi|hierarchical.tucker|hackbusch|kuhn.2009|block.term|btd|nonnegative.tensor|ntf|robust.tensor|tensor.completion|coupled.tensor|cmtf|multi.view.cp|tensor.regression|cumulant.tensor|tensor.power.iteration|khatri.rao|mode.n.product|mode.n.unfold|matricisation|matricization|kronecker|kruskal|n.way|outer.product.sum|alternating.least.squares|als|de.lathauwer|lathauwer.de.moor.vandewalle|kolda.bader|harshman.1970|carroll.chang.1970|hitchcock.1927|tucker.1966|kruskal.1977|battaglino.ballard.kolda|sketch.cp|randomized.cp|kossaifi.tensorly` returns **zero callable matches** in production `*.go` outside (a) `chaos/analysis.go::recurrencePlot3D` `[][][]float64` accumulator-scratch (NOT a tensor-decomposition surface — just a 3-D histogram), (b) `acoustics/aero/lighthill.go` per-point `[3][3]float64` Lighthill stress-tensor (rank-2 array, NOT higher-order), (c) `linalg/pca.go` covariance-matrix-eigendecomposition (rank-2 only, recovers eigenvectors via inverse-iteration NOT SVD), (d) `linalg/eigen.go::QRAlgorithm` returns eigen-VALUES of symmetric matrices via Householder tridiagonalisation + tqli but does NOT return eigen-VECTORS as Q (verified at lines 5-50 of eigen.go), (e) `linalg/decompose.go::LUDecompose` + Cholesky only — NO SVD, NO QR-with-Q-returned, NO bidiagonalisation, NO rank-revealing decomposition. The entire 1927-2025 tensor-decomposition canon (Hitchcock-1927-J-Math-Phys-6:164 ORIGINAL polyadic-rank-decomposition; Tucker-1966-Psychometrika-31:279 ORIGINAL three-mode-PCA; Carroll-Chang-1970-Psychometrika-35:283 CANDECOMP; Harshman-1970-UCLA-WP-PARAFAC ORIGINAL ALS-based-CP-fitting; Kruskal-1977-LAA-18:95 unique-CP-decomposition-condition; De-Lathauwer-De-Moor-Vandewalle-2000-SIMAX-21:1253 HOSVD-original; De-Lathauwer-De-Moor-Vandewalle-2000-SIMAX-21:1324 HOOI-best-multilinear-rank-approximation; Bro-1997-Chemo-38:149 ALS-CP-for-chemometrics; Andersson-Bro-2000 SLOWED-CP-ALS for swamps; Acar-Dunlavy-Kolda-Morup-2011-Chemo-106:41 SCAR / NCP / CMTF; Hackbusch-Kühn-2009-J-Fourier-15:706 hierarchical-Tucker; Grasedyck-2010-SIMAX-31:2029 H-Tucker-and-TT-equivalence; De-Lathauwer-2008-SIMAX-30:1033 block-term-decomposition; Cichocki-Mandic-Phan-Caiafa-Zhou-Zhao-De-Lathauwer-2015-IEEE-SP-Mag-32:145 tensor-decomposition-survey; Kolda-Bader-2009-SIAM-Rev-51:455 CANONICAL TENSOR-DECOMPOSITION REVIEW with 380 citations; Kolda-2001-SIMAX-23:243 orthogonal-CP-uniqueness; Vannieuwenhoven-Vandebril-Meerbergen-2012-SIMAX-33:1027 ST-HOSVD sequentially-truncated; Battaglino-Ballard-Kolda-2018-SIMAX-39:876 randomized-CP-via-tensor-sketching; Halko-Martinsson-Tropp-2011-SIAM-Rev-53:217 randomized-SVD generalising to randomized-HOSVD; Mu-Huang-Wright-Goldfarb-2014-ICML tensor-completion-via-square-deal; Goldfarb-Qin-2014-MP-152:1 robust-tensor-PCA; Liu-Musialski-Wonka-Ye-2013-PAMI-35:208 tensor-completion-via-trace-norms; Karatzoglou-Amatriain-Baltrunas-Oliver-2010-RecSys multiverse-recommendation tensor-CP-completion; Anandkumar-Ge-Hsu-Kakade-Telgarsky-2014-JMLR-15:2773 tensor-decomposition-via-power-method-for-LATENT-VARIABLE-MODELS; Comon-2014-ICASSP cumulant-tensor-decomposition-via-CP for ICA; Cardoso-1989-ICASSP higher-order-cumulants-for-blind-separation; De-Lathauwer-Castaing-2008-SIPN cumulant-tensor-decomposition-via-CP for ICA; Sun-Papadimitriou-Yu-2008-SDM incremental-tensor-analysis ITA; Mardani-Mateos-Giannakis-2015-IT-61:5374 streaming-tensor-decomposition; Smith-Karypis-2015-SC SPLATT-sparse-CP; Ma-Solomonik-2018-SIMAX accelerated-CP-ALS; Lim-Comon-2009 nonnegative-tensor-rank; Phan-Tichavsky-Cichocki-2013-IT-59:5588 fast-CP-ALS; Sidiropoulos-De-Lathauwer-Fu-Huang-Papalexakis-Faloutsos-2017-IT-65:3551 tensor-decomposition-for-signal-processing-and-machine-learning, 320-citation modern survey; Tomioka-Suzuki-Hayashi-Kashima-2011-NIPS overlapped-Schatten-tensor-completion; Sedighin-Cichocki-Phan-2021-IEEE-SPL randomized-NTF) is wholly **ABSENT**. **PARTIAL OVERLAP with 203 (tensor-networks, slot-precursor) — twenty-four primitives T0a-T20 ~5,400 LOC** including 203-T0a `linalg/svd.go` (~400 LOC SUBSTRATE-PRINCIPAL Golub-Reinsch + thin/truncated/randomized SUBSTRATE that gates EVERYTHING in this slot), 203-T0b `tensor/tensor.go` (~150 LOC canonical multi-dim array + Reshape + mode-n unfold/fold + Permute SUBSTRATE), 203-T0c `tensor/contract.go` (~150 LOC mode-n product + Khatri-Rao + Kronecker + general Contract SUBSTRATE), 203-T1 CP-via-ALS (~300 LOC), 203-T2 Tucker via HOSVD-init+HOOI-refine (~300 LOC), 203-T3 HOSVD-Adaptive ε-bounded (~150 LOC), 203-T4 TT representation+arithmetic (~250 LOC), 203-T5 TT-SVD-Oseledets-2011-flagship (~250 LOC), 203-T6 TT-rounding (~200 LOC), 203-T7 TT-cross-Oseledets-Tyrtyshnikov-2010-DMRG-cross (~250 LOC), 203-T8 contraction-order-optimisation (~250 LOC), 203-T9 Einsum-parser (~150 LOC), 203-T10 DMRG-White-1992 (~400 LOC), 203-T11 TEBD-Vidal-2003 (~300 LOC), 203-T12 hierarchical-Tucker-Hackbusch-Kühn-2009 (~250 LOC), 203-T13 block-term-decomposition-De-Lathauwer-2008 (~200 LOC), 203-T14 PEPS DEFER, 203-T15 MERA DEFER, 203-T16 randomized-tensor-decompositions Halko-2011+Battaglino-Ballard-Kolda-2018 (~250 LOC), 203-T17 sparse-tensor-COO+CSF+MTTKRP-Smith-Karypis-2015 (~150 LOC), 203-T18 quantum-circuit-as-MPS (~200 LOC), 203-T19 AMEn-Dolgov-Savostyanov-2014 DEFER, 203-T20 high-d-PDE-pipeline DEFER. **Slot 257's value is the CP/Tucker/HOSVD/HT-DEEP-DIVE that lives ORTHOGONAL to 203's TT/MPS/DMRG-axis** — slot 257 enumerates the *psychometric-chemometric-machine-learning-side* of the tensor-decomposition canon (Hitchcock-1927/Tucker-1966/Carroll-Chang-1970/Harshman-1970/Kolda-Bader-2009/Sidiropoulos-2017 lineage) which 203 covered only AT-A-GLANCE in T1+T2+T3+T12+T13+T16+T17 (~1,500 of the ~5,400 LOC). Slot 257's *additive* surface is **(a) the DEEP-DIVE on CP-ALS edge-cases (swamps, degeneracy, border-rank, Kruskal-uniqueness, deterministic-multistart, non-negativity, sparsity, line-search-CP-ALS Phan-Tichavsky-Cichocki-2013, fast-Khatri-Rao MTTKRP)**, **(b) the DEEP-DIVE on Tucker variants (ST-HOSVD Vannieuwenhoven-2012, randomized-HOSVD, Tucker-low-multilinear-rank-bound, HOOI-convergence-monotonicity)**, **(c) tensor-completion (low-rank-tensor-recovery from partial entries — the recommendation-system / chemometric-imputation workhorse, NOT enumerated in 203)**, **(d) tensor-regression (Zhou-Li-Zhu-2013 generalised-low-rank-tensor-regression — NOT enumerated in 203)**, **(e) coupled-tensor-matrix factorisation (CMTF, Acar-Dunlavy-Kolda-Morup-2011 — multi-relational data fusion, NOT enumerated in 203)**, **(f) cumulant-tensor-decomposition for ICA (Comon-1994 / Cardoso-1989 / De-Lathauwer-2008 — the *original* application of higher-order tensors in signal processing, NOT enumerated in 203)**, **(g) tensor-power-iteration for latent-variable model learning (Anandkumar-Ge-Hsu-Kakade-Telgarsky-2014 — solves topic models / mixture models / HMMs via single 3rd-order moment tensor decomposition with provable polynomial-time identifiability — NOT enumerated in 203)**, **(h) online / streaming tensor decomposition (Sun-Papadimitriou-Yu-2008-ITA, Mardani-Mateos-Giannakis-2015 — for time-evolving data — NOT enumerated in 203)**.

**Summary line 2.** Twenty-four primitives D1-D24 totalling ~3,180 LOC organised as **(a) Tier-0 SUBSTRATE 203-shared ~700 LOC** (D1-D4 are EXPLICIT CROSS-REFERENCES to 203-T0a `linalg/svd.go` + 203-T0b `tensor/tensor.go` + 203-T0c `tensor/contract.go` + 203-T0d `linalg/qr.go`; this slot does NOT re-enumerate them — they ship in 203-PR-A and slot 257 imports them via `tensor` package), **(b) Tier-1 CP-decomposition deep-dive ~620 LOC NEW additive on 203-T1** (D5 `tensor/cp/als.go` baseline-CP-ALS-Harshman-1970-Bro-1997 with Khatri-Rao MTTKRP+normalised-factor-extraction+per-iteration-fit-improvement-monitor+swamp-detection-via-fit-stagnation-counter ~140 LOC NEW additive on 203-T1's plain ALS, 203-T1 enumerates only the algorithm shell, slot 257-D5 ships swamp-detection + Bro-1997-line-search-acceleration that 203-T1 omitted; D6 `tensor/cp/multistart.go` deterministic-multistart with seed-sequence Halton(2,3,5) for cross-language-byte-identical reproducibility ~60 LOC; D7 `tensor/cp/uniqueness.go` Kruskal-1977 sufficient-condition checker `k_A + k_B + k_C >= 2R+2` where k_X is the k-rank of factor X — the EXACT-uniqueness diagnostic that distinguishes a proper CP solution from a degenerate-rank-1-blow-up swamp-iterate ~80 LOC; D8 `tensor/cp/border_rank.go` border-rank vs rank diagnostic via `MaxFactorNorm` early-stop guard against De-Silva-Lim-2008 ill-posedness ~60 LOC; D9 `tensor/cp/nonnegative.go` NCP via projected-ALS / multiplicative-update Lee-Seung-1999-style nonnegative-CP for chemometric-spectroscopy-data and topic-modeling ~140 LOC; D10 `tensor/cp/sparse_cp.go` ℓ1-regularised CP-ALS via proximal-soft-threshold-on-factors for sparse-pattern-discovery ~80 LOC; D11 `tensor/cp/fast_als.go` Phan-Tichavsky-Cichocki-2013-IT-59:5588 fast-CP-ALS with all-at-once factor update via Hadamard-product-of-Gramians trick saving O(d) inner-products per sweep ~60 LOC), **(c) Tier-2 Tucker / HOSVD deep-dive ~480 LOC NEW additive on 203-T2/T3/T16** (D12 `tensor/tucker/st_hosvd.go` Vannieuwenhoven-Vandebril-Meerbergen-2012-SIMAX-33:1027 Sequentially-Truncated HOSVD: instead of d independent SVDs of mode-unfoldings, sequentially-truncate each mode in turn, propagating the truncation error to subsequent modes — gives ~50% memory and time savings vs HOSVD with same quasi-optimality bound ~140 LOC; D13 `tensor/tucker/hooi.go` HOOI-Higher-Order-Orthogonal-Iteration-De-Lathauwer-De-Moor-Vandewalle-2000-SIMAX-21:1324 ALS-refinement with monotone-Frobenius-fit-improvement guarantee — extends 203-T2's HOOI shell with restart-on-stagnation + adaptive-rank-truncation ~120 LOC; D14 `tensor/tucker/randomized_hosvd.go` Halko-Martinsson-Tropp-2011 randomized-HOSVD: each mode-n SVD is replaced by RandomizedSVD with `oversample=10` — gives O(d·n^d·r) → O(d·n·r²·log(n)) flop reduction for large tensors ~120 LOC; D15 `tensor/tucker/multilinear_rank.go` adaptive-multilinear-rank-selection: choose `R_n` such that mode-n discarded singular values sum-to-square ≤ ε² · total — De-Lathauwer-2000 mode-by-mode-quasi-optimality-bound + Vannieuwenhoven-2012 ST-bound `√d ε^global` ~100 LOC), **(d) Tier-3 hierarchical Tucker + block-term ~360 LOC NEW additive on 203-T12/T13** (D16 `tensor/htucker/htucker.go` Hackbusch-Kühn-2009-J-Fourier-15:706 hierarchical-Tucker: dimension-tree binary-partition of mode set + per-leaf factor matrix + per-internal-node transfer-tensor; storage O(d·n·r + d·r³) — same as TT but tree-structured and more efficient when mode interactions are hierarchical (image-frequency-channel-time tensors, multivariate-PDE-discretisations) ~180 LOC NEW additive on 203-T12 which only enumerated the data structure; D17 `tensor/htucker/htucker_truncate.go` HT-truncation via root-to-leaf SVD-truncation analogous to TT-rounding (203-T6) but on the dimension-tree, NOT a chain ~120 LOC; D18 `tensor/btd/block_term.go` Block-Term-Decomposition-De-Lathauwer-2008-SIMAX-30:1033 generalised CP+Tucker: T = Σ_r G_r ×_1 U_r ×_2 V_r ×_3 W_r — sum of *partial* Tuckers, useful for blind-source-separation with structured-signals BTD-(L,L,1) for telecommunication-channel-identification ~60 LOC NEW additive on 203-T13 which only enumerated the algorithm shell, slot 257-D18 ships the BTD-(L,L,1)-channel-identification specialisation), **(e) Tier-4 tensor-completion + tensor-regression ~440 LOC NEW** (D19 `tensor/completion/lr_completion.go` Liu-Musialski-Wonka-Ye-2013-PAMI-35:208 low-rank-tensor-completion via overlapped-trace-norm minimisation `min Σ_n ||T_(n)||_*  s.t.  Ω(T) = Ω(M)` where Ω is the observation set; ADMM solver alternating between SVT-on-each-mode-unfolding and observation-projection ~180 LOC; D20 `tensor/completion/square_deal.go` Mu-Huang-Wright-Goldfarb-2014-ICML square-deal: more-balanced matricisation that achieves better recovery-bound than overlapped-trace-norm; cross-link to compressed-sensing-215 ~100 LOC; D21 `tensor/regression/glm_tensor.go` Zhou-Li-Zhu-2013-JASA-108:540 generalised-low-rank-tensor-regression: y_i = ⟨X_i, B⟩ + ε_i where B has CP-rank R structure; alternating-update over factors of B ~80 LOC; D22 `tensor/robust/rtpca.go` Goldfarb-Qin-2014-MP-152:1 robust-tensor-PCA: T = L + S where L is low-multilinear-rank and S is sparse; tensor-extension of Candes-Li-Ma-Wright-2011 robust-PCA via overlapped-nuclear-norm + ℓ1 ~80 LOC), **(f) Tier-5 coupled+multi-view+ICA+latent-variable-models ~360 LOC NEW** (D23 `tensor/coupled/cmtf.go` Acar-Dunlavy-Kolda-Morup-2011 coupled-matrix-tensor-factorisation: jointly factor tensor T and matrix M sharing one mode (e.g., user-item-time tensor + user-attribute matrix) — fundamental for multi-view recommendation, multi-relational data integration, and chemometric-spectroscopy-coupled-data-fusion ~120 LOC; D24 `tensor/multiview/multi_cp.go` multi-view-CP: jointly factor multiple coupled tensors with shared and view-specific factors — extends D23 to >2 views ~80 LOC; D25 `tensor/cumulant/ica_cumulant.go` Comon-1994 / Cardoso-1989 / De-Lathauwer-Castaing-2008-SIPN ICA via 4th-order-cumulant-tensor CP-decomposition: build C_4 the 4-th-order cumulant tensor from samples, decompose via CP — separation matrix recovered from CP-factors of C_4; the *original* application of higher-order tensors in signal processing ~100 LOC; D26 `tensor/lvm/power_iteration.go` Anandkumar-Ge-Hsu-Kakade-Telgarsky-2014-JMLR-15:2773 tensor-decomposition-via-tensor-power-method for latent-variable models: topic models / mixture models / HMMs identified via single 3rd-order moment tensor M_3 decomposed via deflation-power-iteration — the FIRST polynomial-time provably-correct algorithm for these LVMs ~60 LOC), **(g) Tier-6 streaming/online tensor decomposition ~220 LOC NEW** (D27 `tensor/online/incremental_ta.go` Sun-Papadimitriou-Yu-2008-SDM incremental-tensor-analysis ITA: when a new slice arrives along the time mode, update the Tucker decomposition incrementally without recomputing from scratch — the streaming-Tucker-update workhorse for monitoring ~120 LOC; D28 `tensor/online/streaming_cp.go` Mardani-Mateos-Giannakis-2015-IT-61:5374 streaming-CP-decomposition with PARAFAC-in-the-data-stream framework — bounded-memory online-CP for sensor-network-data ~100 LOC). **PR-G-257 SINGULAR-FOUNDATIONAL D5+D7+D12+D13+D14 ~580 LOC** — CP-ALS-with-swamp-detection + Kruskal-uniqueness-checker + ST-HOSVD + HOOI + randomized-HOSVD — the FIVE primitives that are present-in-every-tensor-decomposition-textbook (Kolda-Bader-2009 + Cichocki-2015 + Sidiropoulos-2017) and that ship as a SINGLE coherent PR once 203-T0a `linalg/svd.go` substrate lands. **SINGULAR-CHEAPEST-1-DAY D6 Multistart + D7 KruskalUniqueness + D8 BorderRank ~200 LOC** — three pure-API utility primitives wrapping 203-T1 CP-ALS with no NEW math; ship in one engineer-day against the 203-PR-A substrate. **SINGULAR-MOAT D14 RandomizedHOSVD + D19 LRCompletion + D26 TensorPowerIteration ~360 LOC** — three primitives that have NO public-Go-implementation worldwide and that 203 enumerated only AT-A-GLANCE: randomized-HOSVD beats deterministic-HOSVD by 10x for n>1000 large tensors; tensor-completion is the recommendation-system workhorse with no zero-dep Go library; tensor-power-iteration is the FIRST polynomial-time provably-correct LVM-learner. **SINGULAR-2024-FRONTIER D14+D19+D22+D26 ~440 LOC** — Halko-2011 randomized-HOSVD + Liu-2013 tensor-completion + Goldfarb-Qin-2014 robust-tensor-PCA + Anandkumar-2014 tensor-power-iteration are the four primitives that define the 2010-2024 modern tensor-decomposition canon. **SINGULAR-PEDAGOGICAL D5+D7+D12+D13 ~480 LOC** — the canonical Kolda-Bader-2009-SIAM-Rev-51:455 four-algorithm core (CP-ALS + Kruskal-uniqueness + Tucker-via-HOSVD + HOOI) that defines the entry-level tensor-decomposition curriculum. **SINGULAR-CROSS-LINK D25 ICA-Cumulant-CP ~100 LOC** — the *original* Comon-1994 application of higher-order tensors that bridges to signal-processing-ICA; cross-link to slot 215 (compressed sensing) and to a future `audio/separation/` blind-source-separation module. **SINGULAR-CONSUMER-VALUE D19 LRCompletion + D23 CMTF + D27 IncrementalTA ~420 LOC** — three primitives with concrete CONCRETE-aicore-recommendation-engine consumers (multi-relational user-item-time tensor completion + coupled-matrix-tensor for cross-feature integration + streaming-Tucker for real-time monitoring). Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (see §3). Recommended placement **NEW sub-package `tensor/cp/`, `tensor/tucker/`, `tensor/htucker/`, `tensor/btd/`, `tensor/completion/`, `tensor/regression/`, `tensor/robust/`, `tensor/coupled/`, `tensor/multiview/`, `tensor/cumulant/`, `tensor/lvm/`, `tensor/online/`** all under the proposed `tensor/` package whose root + `tensor/contract.go` ships from 203-PR-A. Strict-downstream of 203-PR-A (`linalg/svd.go` + `tensor/tensor.go` + `tensor/contract.go`); strict-upstream of compressed-sensing 215-T-tensor-completion-extension and recommendation-system aicore consumers and audio-separation BSS via cumulant-CP-ICA.

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for tensor-decomposition / CP / Tucker / HOSVD / HT / completion / regression / cumulant / online surface.

| Surface | Path | Tensor-decomp relevance |
|---|---|---|
| `chaos/analysis.go::recurrencePlot3D` | `chaos/analysis.go` | `[][][]float64` accumulator-scratch only — NOT a tensor-decomposition surface; private 3-D histogram for recurrence-plot density estimation. |
| `acoustics/aero/lighthill.go` | `acoustics/aero/lighthill.go` (planned slot 197) | Per-point `[3][3]float64` Lighthill stress tensor — rank-2 fixed-shape array, NOT higher-order tensor abstraction. |
| `linalg/decompose.go` | LU + Cholesky | NO SVD, NO QR (with Q returned), NO bidiagonalisation, NO rank-revealing decomposition. **Critical missing prerequisite for every tensor decomposition.** |
| `linalg/eigen.go::QRAlgorithm` | Symmetric matrix eigenvalues via Householder + tqli | Returns eigenvalues NOT eigenvectors. Tucker / HOSVD / HOOI need top-k singular VECTORS. |
| `linalg/pca.go::PCA` | Covariance-eigendecomposition + inverse-iteration | Recovers eigenvectors via inverse iteration on covariance matrix (rank-2 only). Cannot truncate-SVD a rectangular matricisation. |
| `linalg/matrix.go` | MatMul, MatTranspose, Identity, Trace, MatAdd, MatScale | NO Reshape, NO Fold/Unfold, NO Kronecker, NO Khatri-Rao. |
| `signal/fft.go` | Cooley-Tukey FFT | Tangentially useful for FFT-based-CP-acceleration (Phan-Tichavsky-Cichocki-2013). |
| `autodiff/` | Scalar reverse-mode AD over Variables | NO batched / matrix / tensor ops; Townsend-2016 differentiable-SVD would unlock differentiable-tensor-decomposition. |
| `optim/` | Newton / L-BFGS / simulated annealing / proximal | Can drive ALS inner loops once the linear-LS substrate (least-squares = `linalg.MatMul(A^T, A)` solve via LU/Cholesky) is wrapped. |

Verified with grep on `tensor.decomp|cp.als|parafac|candecomp|tucker|hosvd|st.hosvd|hooi|hierarchical.tucker|btd|nonnegative.tensor|robust.tensor|tensor.completion|coupled.tensor|cmtf|tensor.regression|cumulant.tensor|tensor.power.iteration|khatri.rao|kronecker|kruskal|n.way|de.lathauwer|kolda.bader|harshman|hitchcock|tucker.1966|battaglino.ballard.kolda|sketch.cp|randomized.cp|kossaifi.tensorly` — zero callable matches in production `*.go`.

---

## 1. Twenty-eight primitives D5-D28 ~3,180 LOC NEW additive on 203-PR-A

Demand ranking weights: (a) explicit consumer in CONTEXT.md / aicore / Pistachio downstream, (b) frequency in Kolda-Bader-2009 / Cichocki-2015 / Sidiropoulos-2017 review-article corpora, (c) connective-tissue readiness on top of 203-PR-A, (d) "no-zero-dep-library-ships-this" cutting-edge score.

### Tier-0 — substrate (203-shared, 0 LOC additive)

D1 `linalg/svd.go` (~400 LOC, ships in 203-PR-A) — Golub-Reinsch full + thin/truncated/randomized SVD. **Blocks every primitive in this slot.**
D2 `linalg/qr.go` (~150 LOC, ships in 203-PR-A) — Householder QR with explicit Q returned. Used by HT-rounding and HOOI-orthogonalisation.
D3 `tensor/tensor.go` (~150 LOC, ships in 203-PR-A) — canonical Tensor type with flat data + shape + strides + Reshape + Unfold/Fold + Permute.
D4 `tensor/contract.go` (~150 LOC, ships in 203-PR-A) — ModeNProduct + Contract + Kronecker + KhatriRao primitives.

### Tier-1 — CP decomposition deep-dive (~620 LOC NEW additive on 203-T1)

D5 `tensor/cp/als.go` ~140 LOC — baseline-CP-ALS-Harshman-1970-Bro-1997 with Khatri-Rao MTTKRP + normalised-factor-extraction + per-iteration-fit-improvement-monitor + swamp-detection-via-fit-stagnation-counter. **Adds swamp detection that 203-T1 omitted.** API: `CPALS(T *Tensor, rank int, opts CPOpts) (CPResult, error)` where `CPOpts.SwampThreshold` triggers restart when `(fit_t − fit_{t-K}) / fit_t < threshold` for K consecutive sweeps.

D6 `tensor/cp/multistart.go` ~60 LOC — deterministic multistart with Halton(2,3,5)-seed-sequence for cross-language byte-identical reproducibility.

D7 `tensor/cp/uniqueness.go` ~80 LOC — Kruskal-1977-LAA-18:95 sufficient-condition checker `k_A + k_B + k_C >= 2R+2` where k_X is the k-rank of factor X. The EXACT-uniqueness diagnostic that distinguishes a proper CP solution from a degenerate-rank-1-blow-up swamp.

D8 `tensor/cp/border_rank.go` ~60 LOC — border-rank-vs-rank diagnostic via `MaxFactorNorm` early-stop guard against De-Silva-Lim-2008 ill-posedness (border-rank-(R-1) approximation may not exist).

D9 `tensor/cp/nonnegative.go` ~140 LOC — NCP via projected-ALS / multiplicative-update Lee-Seung-1999-style nonnegative-CP. Used in chemometric-spectroscopy and nonneg-topic-modeling.

D10 `tensor/cp/sparse_cp.go` ~80 LOC — ℓ1-regularised CP-ALS via proximal-soft-threshold-on-factors for sparse-pattern discovery.

D11 `tensor/cp/fast_als.go` ~60 LOC — Phan-Tichavsky-Cichocki-2013-IT-59:5588 fast-CP-ALS with all-at-once factor update via Hadamard-product-of-Gramians trick saving O(d) inner-products per sweep.

### Tier-2 — Tucker / HOSVD deep-dive (~480 LOC NEW additive on 203-T2/T3/T16)

D12 `tensor/tucker/st_hosvd.go` ~140 LOC — Vannieuwenhoven-Vandebril-Meerbergen-2012-SIMAX-33:1027 Sequentially-Truncated HOSVD. ~50% memory and time savings vs vanilla HOSVD with the same `√d ε^global` quasi-optimality bound. Algorithm: instead of d independent SVDs of mode-unfoldings, sequentially truncate each mode in turn, propagating the truncation error to subsequent mode-unfoldings. **Default-recommended HOSVD variant in production.**

D13 `tensor/tucker/hooi.go` ~120 LOC — HOOI-Higher-Order-Orthogonal-Iteration-De-Lathauwer-De-Moor-Vandewalle-2000-SIMAX-21:1324 ALS-refinement on top of HOSVD-init. Provides monotone-Frobenius-fit improvement guarantee + restart-on-stagnation + adaptive-rank-truncation.

D14 `tensor/tucker/randomized_hosvd.go` ~120 LOC — Halko-Martinsson-Tropp-2011 randomized-HOSVD: each mode-n SVD replaced by RandomizedSVD with `oversample=10`. O(d·n^d·r) → O(d·n·r²·log(n)) flop reduction for large tensors (n > 1000).

D15 `tensor/tucker/multilinear_rank.go` ~100 LOC — adaptive-multilinear-rank-selection: choose `R_n` such that mode-n discarded singular values sum-to-square ≤ ε² · total. De-Lathauwer-2000 mode-by-mode quasi-optimality bound + Vannieuwenhoven-2012 ST-bound `√d ε^global`.

### Tier-3 — hierarchical Tucker + block-term (~360 LOC NEW additive on 203-T12/T13)

D16 `tensor/htucker/htucker.go` ~180 LOC — Hackbusch-Kühn-2009-J-Fourier-15:706 hierarchical-Tucker. Dimension-tree binary-partition of mode set + per-leaf factor matrix + per-internal-node transfer-tensor. Storage O(d·n·r + d·r³). **Adds full HT data structure + traversal + element evaluation that 203-T12 only enumerated.**

D17 `tensor/htucker/htucker_truncate.go` ~120 LOC — HT-truncation via root-to-leaf SVD-truncation, analogous to TT-rounding but on the dimension-tree (NOT a chain).

D18 `tensor/btd/block_term.go` ~60 LOC — Block-Term-Decomposition-De-Lathauwer-2008-SIMAX-30:1033. T = Σ_r G_r ×_1 U_r ×_2 V_r ×_3 W_r — sum of *partial* Tuckers. Specialisation BTD-(L,L,1) for telecommunication-channel-identification.

### Tier-4 — tensor-completion + tensor-regression (~440 LOC NEW)

D19 `tensor/completion/lr_completion.go` ~180 LOC — Liu-Musialski-Wonka-Ye-2013-PAMI-35:208 low-rank-tensor-completion via overlapped-trace-norm minimisation. ADMM solver alternating between SVT-on-each-mode-unfolding and observation-projection.

D20 `tensor/completion/square_deal.go` ~100 LOC — Mu-Huang-Wright-Goldfarb-2014-ICML square-deal: more-balanced matricisation for better recovery-bound than overlapped-trace-norm. **Cross-link to compressed-sensing 215.**

D21 `tensor/regression/glm_tensor.go` ~80 LOC — Zhou-Li-Zhu-2013-JASA-108:540 generalised-low-rank-tensor-regression. y_i = ⟨X_i, B⟩ + ε_i where B has CP-rank R structure; alternating-update over factors of B.

D22 `tensor/robust/rtpca.go` ~80 LOC — Goldfarb-Qin-2014-MP-152:1 robust-tensor-PCA: T = L + S where L is low-multilinear-rank and S is sparse. Tensor-extension of Candes-Li-Ma-Wright-2011 robust-PCA via overlapped-nuclear-norm + ℓ1.

### Tier-5 — coupled + multi-view + ICA + LVM (~360 LOC NEW)

D23 `tensor/coupled/cmtf.go` ~120 LOC — Acar-Dunlavy-Kolda-Morup-2011 coupled-matrix-tensor-factorisation. Jointly factor tensor T and matrix M sharing one mode (user-item-time tensor + user-attribute matrix). Foundation of multi-view recommendation, multi-relational data integration, chemometric-spectroscopy-coupled-data-fusion.

D24 `tensor/multiview/multi_cp.go` ~80 LOC — multi-view-CP: jointly factor multiple coupled tensors with shared and view-specific factors.

D25 `tensor/cumulant/ica_cumulant.go` ~100 LOC — Comon-1994 / Cardoso-1989 / De-Lathauwer-Castaing-2008-SIPN ICA via 4th-order-cumulant-tensor CP-decomposition. Build C_4 the 4-th-order cumulant tensor from samples, decompose via CP — separation matrix recovered from CP factors of C_4. **The original application of higher-order tensors in signal processing.**

D26 `tensor/lvm/power_iteration.go` ~60 LOC — Anandkumar-Ge-Hsu-Kakade-Telgarsky-2014-JMLR-15:2773 tensor-decomposition-via-tensor-power-method for latent-variable models (topic models / mixture models / HMMs identified via single 3rd-order moment tensor M_3 decomposed via deflation-power-iteration). **First polynomial-time provably-correct LVM learner.**

### Tier-6 — streaming / online tensor decomposition (~220 LOC NEW)

D27 `tensor/online/incremental_ta.go` ~120 LOC — Sun-Papadimitriou-Yu-2008-SDM incremental-tensor-analysis ITA. When a new slice arrives along the time mode, update the Tucker decomposition incrementally without recomputing from scratch.

D28 `tensor/online/streaming_cp.go` ~100 LOC — Mardani-Mateos-Giannakis-2015-IT-61:5374 streaming-CP-decomposition with PARAFAC-in-the-data-stream framework. Bounded-memory online-CP for sensor-network data.

---

## 2. Connective tissue — what each new edge buys

Eight cross-package edges activate once `tensor/cp/`, `tensor/tucker/`, `tensor/htucker/`, `tensor/completion/` land on top of 203-PR-A:

| Edge | LOC of glue | What it unlocks |
|---|---|---|
| `tensor/cp/ → autodiff/` | 80 | Differentiable CP via Townsend-2016 SVD-gradient (already gated on 203-T0a SVD); enables backprop through tensor regression. |
| `tensor/completion/ → optim/proximal/` | 30 | Reuses existing or 102-flagged proximal nuclear-norm operator on each mode-unfolding. |
| `tensor/coupled/ → linalg/pca.go` | -30 (refactor) | Multi-view CP subsumes joint-PCA and joint-NMF; PCA refactors to call coupled-NMF as special case. |
| `tensor/cumulant/ → audio/separation/` | 50 | Cumulant-CP-ICA powers blind-source-separation in audio when consumer pulls. |
| `tensor/lvm/ → prob/markov.go` | 80 | Anandkumar-2014 tensor-power-iteration identifies HMM parameters via 3rd-order-moment tensor. **Cross-link to 165 (HMM).** |
| `tensor/online/ → changepoint/` | 50 | Streaming-tensor-decomposition + BOCPD = online tensor-segmentation. |
| `tensor/regression/ → optim/lbfgs.go` | 30 | GLM-tensor-regression inner-loop uses L-BFGS. |
| `tensor/htucker/ → graph/` | 50 | HT dimension-tree is a graph; reuses graph traversal + tree-decomposition. |

**No existing package needs an API break.** New sub-packages enumerated above, no existing-package mutation.

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled

| Pin | Three independent oracles | Tolerance | Test |
|---|---|---|---|
| **Pin-1 CP-recover-synthetic-rank-3** | (i) D5 CPALS on T = Σ_{r=1}^3 a_r ⊗ b_r ⊗ c_r with d=3, n=10, R=3, deterministic-multistart=5; (ii) tensorly.decomposition.parafac on the same tensor; (iii) brute-force gradient-descent-on-CP-loss via 203-autodiff-extension | reconstruction error ≤ 1e-6 | `TestCP_RecoverSyntheticR3` |
| **Pin-2 HOSVD↔ST-HOSVD↔randomized-HOSVD bound consistency** | (i) D12 ST-HOSVD; (ii) 203-T2 vanilla HOSVD; (iii) D14 randomized-HOSVD with oversample=10 | quasi-optimality bound `||T − T_HOSVD||_F ≤ √d · ||T − T_best||_F` holds for all three | `TestHOSVDQuasiOptimalityCrossValidation` |
| **Pin-3 Kruskal-uniqueness-condition** | (i) D7 Kruskal-condition checker on a synthetic CP-rank-3 tensor with k_A=k_B=k_C=3 → predicted-unique; (ii) D5 CPALS with 5 multistarts → all converge to the same factors up to permutation/scaling; (iii) brute-force pseudo-random multistart with 100 seeds → same factor recovery | factor-recovery agreement modulo (perm, scaling) at 1e-8 | `TestKruskalUniqueness_Pin3` |
| **Pin-4 Tensor-completion-recovery-bound** | (i) D19 LRCompletion via overlapped-trace-norm-ADMM on T = low-CP-rank-tensor with 30% entries observed; (ii) D20 SquareDeal recovery; (iii) brute-force Tucker-fit-on-completed-tensor | reconstruction error on missing entries ≤ 1e-4 (Liu-2013-PAMI-35:208 Theorem 3 bound) | `TestTensorCompletion_Pin4` |
| **Pin-5 Tensor-power-iteration↔matrix-power-iteration↔SVD on rank-1-symmetric-tensor** | (i) D26 deflation-power-iteration on T = u ⊗ u ⊗ u; (ii) matrix-power-iteration on T_(1) (mode-1-unfolding); (iii) D1 SVD of T_(1) → top singular vector should match u up to sign | factor recovery agreement at 1e-10 | `TestTensorPowerIteration_Pin5` |

---

## 4. Risks and gotchas

- **G1. CP rank-truncation is ill-posed.** Border-rank issue (De-Silva-Lim-2008). Best-rank-(R-1) approximation may not exist; CP-ALS may diverge. Ship `MaxFactorNorm` early-stop guard with fallback to multistart.
- **G2. CP swamps.** ALS-iterates plateau in fit-improvement for hundreds of sweeps before either escaping or diverging. Default to swamp-detection (Bro-1997) + multistart; document loud.
- **G3. CP non-uniqueness.** Without Kruskal-1977 condition, two distinct factor sets may give equally-good reconstruction. Ship D7 Kruskal-checker as default-on diagnostic.
- **G4. HOSVD ranks per-mode.** Each mode has its own `R_n`; `ranks []int` argument with len(ranks) == d, not a single int. Common API confusion; surface in error messages.
- **G5. Tucker is NOT the best rank-(R_1, R_2, R_3) approximation by HOSVD alone.** De-Lathauwer-2000 quasi-optimality bound `√d · ||T − T_best||_F` is loose; HOOI refinement (D13) can cut error by half. Document as default-recommendation: HOSVD-init then HOOI-refine.
- **G6. Mode-n unfolding convention.** Kolda-Bader-2009 §2 ordering vs LAPACK / numpy / tensorly column-ordering. Pick *one* and pin it byte-for-byte against numpy.tensorly. Document loud-fail when consumer mixes conventions.
- **G7. NCP non-uniqueness without sparsity.** Lee-Seung-1999 multiplicative updates converge to KKT point but not to global minimum; multi-start required. Document.
- **G8. Tensor-completion observation pattern.** Random-uniform sampling has different recovery-bound than coherent-block sampling (Mu-2014 square-deal). Document that LRCompletion-ADMM assumes random-uniform; coherent-block requires square-deal.
- **G9. Streaming-tensor-decomposition rank drift.** Online-CP rank may need to grow as new modes arrive; D27/D28 ship rank-adaptation triggers but document as a heuristic, NOT a theoretical guarantee.
- **G10. Cumulant-tensor-ICA scaling/permutation indeterminacy.** Standard ICA ambiguity; document that recovered sources match originals up to (sign, scale, permutation).

---

## 5. Cross-language parity targets

Eight pinned tests covering the foundational decomposition correctness and error bounds:

| Test | Pin | Tolerance | Reference |
|---|---|---|---|
| `TestCPALS_SyntheticRank3` | reconstruction `||T − Σ λ_r ⊗ a_r ⊗ b_r ⊗ c_r||_F` | 1e-6 | Kolda-Bader-2009 Algorithm 3.1 |
| `TestSTHOSVD_QuasiOptimality_d4n8r3` | `||T − T_STHOSVD||_F ≤ √d · ||T − T_best||_F` | bound holds | Vannieuwenhoven-2012 SIMAX-33:1027 |
| `TestHOOIMonotone_d3n10r2` | per-iteration Frobenius-fit decreases monotonically | non-increasing | De-Lathauwer-2000 SIMAX-21:1324 Thm 4.1 |
| `TestRandomizedHOSVD_OversampleK10` | `||T − T_RHOSVD||_F ≤ (1+ε) · ||T − T_HOSVD||_F` for ε=0.1 | probability ≥ 0.99 | Halko-Martinsson-Tropp-2011 |
| `TestKruskalUniqueness_R3_d3` | k-rank diagnostic agrees with multistart-converged-factors | 1e-8 (factor matching) | Kruskal-1977 LAA-18:95 |
| `TestNCP_Reconstruction` | NCP-multiplicative-update-fit on nonneg synthetic tensor | 1e-6 | Lee-Seung-1999 + Cichocki-2009 |
| `TestLRCompletion_30PctObserved` | LRCompletion recovery on missing-entries when underlying tensor has CP-rank ≤ 5 | 1e-4 | Liu-Musialski-Wonka-Ye-2013 PAMI-35:208 Thm 3 |
| `TestCumulantICA_3Source_Mixture` | recovered sources match originals up to (sign, perm) | 1e-8 (after alignment) | Comon-1994 + De-Lathauwer-Castaing-2008 |

---

## 6. Verdict

**Ship Tier-1 + Tier-2 (~1,100 LOC over 4-6 sprints) once 203-PR-A `linalg/svd.go` + `tensor/tensor.go` + `tensor/contract.go` substrate lands:**

- Sprint 1: D5 CPALS-with-swamp-detection (140) + D6 Multistart (60) + D7 KruskalUniqueness (80) — CP foundations
- Sprint 2: D8 BorderRank (60) + D9 NCP (140) + D10 SparseCP (80) + D11 FastALS (60) — CP variants
- Sprint 3: D12 STHOSVD (140) + D13 HOOI (120) — Tucker production-defaults
- Sprint 4: D14 RandomizedHOSVD (120) + D15 MultilinearRank (100) — Tucker scaling
- Sprint 5: D16 HTucker (180) + D17 HTuckerTruncate (120) + D18 BTD (60) — hierarchical
- Sprint 6: cross-substrate parity tests + documentation polish

**Ship Tier-4 + Tier-5 (~800 LOC over 3-4 sprints) when consumer pulls:**

- Sprint 7: D19 LRCompletion (180) + D20 SquareDeal (100) — completion
- Sprint 8: D21 GLMTensorRegression (80) + D22 RobustTensorPCA (80) — supervised + robust
- Sprint 9: D23 CMTF (120) + D24 MultiCP (80) — coupled / multi-view
- Sprint 10: D25 CumulantICA (100) + D26 TensorPowerIteration (60) — signal-processing + LVM

**Defer Tier-6 (~220 LOC) until streaming-data consumer pulls:** D27 IncrementalTA, D28 StreamingCP.

**Single-highest-leverage 1-day project (assuming 203-PR-A landed):** D5 CPALS-with-swamp-detection (~140 LOC). Ships the FIRST production-Go deterministic CP-ALS with swamp-detection and Bro-1997-line-search-acceleration; saturates Pin-1 against tensorly cross-validation.

**Single-highest-leverage cutting-edge piece:** D26 TensorPowerIteration (~60 LOC) for Anandkumar-2014 latent-variable-model learning. The FIRST polynomial-time provably-correct algorithm for topic-model / mixture-model / HMM identification via single 3rd-order-moment tensor decomposition. **Tiny LOC, huge theoretical value.** Cross-link to slot 165 (HMM).

**Single-highest-leverage moat:** D14 RandomizedHOSVD (~120 LOC) + D19 LRCompletion (~180 LOC) + D26 TensorPowerIteration (~60 LOC) = ~360 LOC of capability with ZERO public-Go-implementation worldwide. Three primitives that 203 enumerated only AT-A-GLANCE, that this slot ships in production-grade.

**Cross-slot synergy callouts:**
- Slot 203 tensor-networks: shares 203-T0a SVD + 203-T0b Tensor + 203-T0c contractions substrate. Slot 257 is the *psychometric-chemometric-machine-learning-side* deep-dive that 203 enumerated only as T1+T2+T3+T12+T13+T16+T17 at-a-glance. Combined surface 203 + 257 = full Kolda-Bader-2009 + Cichocki-2015 + Sidiropoulos-2017 review-article corpus.
- Slot 165 HMM: D26 tensor-power-iteration provides polynomial-time HMM identification via 3rd-order moment tensor.
- Slot 215 compressed-sensing: D19 LRCompletion + D20 SquareDeal share the recovery-bound theory with matrix-completion/CS.
- Slot 081 linalg-missing / 084 linalg-sota: SVD/QR substrate is the 203-T0a flagship; this slot is downstream consumer.
- Slot 102 optim-missing: nuclear-norm proximal operator (already flagged "needs SVD") is reused by D19 LRCompletion ADMM inner loop.
- Slot 168 physics-autodiff: differentiable-SVD (Townsend-2016) is the entry point for differentiable tensor decompositions, unlocking gradient-based learning of CP / Tucker structure.

**One-line verdict.** Slot 257 ships the CP/Tucker/HOSVD/HT-ML-side deep-dive (~3,180 LOC across 28 primitives) that lives ORTHOGONAL to 203's TT/MPS-physics-side axis; recommended placement NEW sub-packages under `tensor/` (203-shared root), strict-downstream of 203-PR-A SVD+Tensor+Contract substrate, with the SINGULAR-FOUNDATIONAL Tier-1+Tier-2 ~1,100 LOC PR shipping the production-default CP-ALS-with-swamp-detection + ST-HOSVD + HOOI + randomized-HOSVD + Kruskal-uniqueness-checker as the canonical Kolda-Bader-2009 four-algorithm core, and the SINGULAR-MOAT D14+D19+D26 ~360 LOC delivering three primitives that have NO public-Go-implementation worldwide.
