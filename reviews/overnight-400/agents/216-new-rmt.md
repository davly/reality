# 216 | new-rmt

**Summary line 1.** SIXTEENTH Block-C cutting-edge-math review and FIRST random-matrix-theory (RMT) scoping in the 400-sequence covering Wigner-1955-semicircle-law `ρ_sc(λ) = (1/2π σ²)·√(4σ² − λ²)` for symmetric n×n matrices with iid sub-Gaussian off-diagonal entries (eigenvalue density on bulk `[−2σ√n, +2σ√n]` after scaling) / Gaussian Orthogonal/Unitary/Symplectic Ensembles GOE/GUE/GSE Dyson-1962 with Dyson-index `β ∈ {1, 2, 4}` (real-symmetric / complex-Hermitian / quaternion-self-dual) and joint eigenvalue density `p(λ) ∝ Π_{i<j}|λ_i − λ_j|^β · Π_i e^{−β·n·V(λ_i)/4}` / Marchenko-Pastur-1967 Math.USSR-Sb-1(4) sample-covariance-eigenvalue-density `f_MP(x) = √((λ_+ − x)(x − λ_−))/(2π·c·σ²·x)` with edges `λ_± = σ²(1 ± √c)²` for ratio `c = p/n` (with point mass `(1 − 1/c)·δ_0` for `c > 1`) — null distribution against which spike detection calibrates / Tracy-Widom-1994-CMP-159 Tracy-Widom-1996-CMP-177 distribution `F_β(s) = exp(−∫_s^∞ (x − s)·q²(x) dx)` of largest-eigenvalue centered fluctuation via Painlevé-II solution `q''(s) = s·q(s) + 2·q³(s)` with Hastings-McLeod-1980 boundary `q(s) ~ Ai(s)` as `s → ∞` (β=1 GOE Tracy-Widom-1996, β=2 GUE Tracy-Widom-1994, β=4 GSE) / Painlevé-II ODE numerical integration Prähofer-Spohn-2004 reference tables / Forrester-Rains-2001 Pfaffian bridge between β=1 and β=4 / Baik-Ben-Arous-Péché-2005-Ann.Probab-33(5) BBP phase transition for spiked-covariance `Σ = I + Σ_i θ_i u_i u_i^T` rank-r perturbation: spike `θ > σ²·√(p/n)` separates from MP edge with Gaussian fluctuations, spike `θ ≤ σ²·√(p/n)` buried in bulk with Tracy-Widom fluctuations / Johnstone-2001-Ann.Stat-29(2) GOE-Tracy-Widom for sample covariance largest eigenvalue under H_0 with centering `μ_np = (√(n−1) + √p)²` and scaling `σ_np = (√(n−1) + √p)·(1/√(n−1) + 1/√p)^{1/3}` / spiked covariance model Johnstone-2001 detection threshold `θ_c = σ²·√(p/n)` / Voiculescu-1985-1991 free probability and free convolution `μ ⊞ ν` (additive) and `μ ⊠ ν` (multiplicative) for non-commutative random variables / R-transform Voiculescu-1986 `R_μ(z) = G_μ^{-1}(z) − 1/z` linearizes free additive convolution `R_{μ⊞ν} = R_μ + R_ν` analogous-to-classical-cumulants / S-transform Voiculescu-1987 multiplicative analog `S_{μ⊠ν} = S_μ · S_ν` / Cauchy-Stieltjes transform `G_μ(z) = ∫ dμ(λ)/(z − λ)` characterizes empirical spectral distribution / inversion via Stieltjes-Perron `f_μ(λ) = −(1/π)·lim_{ε↓0} Im G_μ(λ + iε)` / resolvent `R(z) = (zI − A)^{-1}` and self-consistent equation `G(z) = 1/(z − Σ_self(G))` for free deformed-GUE / local laws Erdős-Yau-Yin-2012 bulk-rigidity `|λ_i − γ_i| ≤ N^{-1+ε}` and edge-rigidity at scale `N^{-2/3}` with Tracy-Widom universality / universality theorem Tao-Vu-2010 Erdős-Yau-Yin-2012-Schlein bulk and edge universality for sub-Gaussian-iid Wigner / Hermite/Laguerre/Jacobi tridiagonal models Dumitriu-Edelman-2002-J.Math.Phys.-43(11) — β-ensembles for arbitrary `β ∈ R_+` not just `β ∈ {1,2,4}` reducing GOE/GUE/GSE eigenvalue computation from O(n³) full diagonalization to O(n²) tridiagonal eigensolve / determinantal point processes (DPP) Macchi-1975 with kernel `K(x,y)` and likelihood `P(S) ∝ det(K_S)` / k-DPP Kulesza-Taskar-2012 conditioning on subset cardinality / DPP sampling Hough-Krishnapur-Peres-Virág-2006 spectral-decomposition algorithm / DPP-MAP via greedy submodular maximization / Fyodorov-Bouchaud-2008-J.Phys.A-41(32) random landscape complexity counting saddle points via Kac-Rice formula / topological trivialization transition / Wigner-1955 nuclear-spectra-energy-level-spacings `P_β(s) = (Π_β/2)·s^β·exp(−c_β·s²)` Wigner-surmise Wigner-Dyson statistics for chaotic quantum systems / Bohigas-Giannoni-Schmit-1984 conjecture chaos→GOE / Mehta-1967-2004 Random-Matrices canonical reference / RMT for stress-test of financial portfolios Laloux-Cizeau-Bouchaud-Potters-1999 PRL-83(7) eigenvalue cleaning above MP edge / High-dimensional MANOVA Wilks-1932 lambda statistic `Λ = det(E)/det(E+H)` joint eigenvalue distribution Jacobi ensemble / James-1964 zonal polynomials / Roy's-largest-root canonical-correlation TW-Type-I asymptotic Johnstone-2008 / random matrix concentration inequalities Tropp-2012-Found.Comp.Math-12(4) "User-friendly tail bounds for sums of random matrices" — Matrix-Bernstein, Matrix-Chernoff, Matrix-Hoeffding, Matrix-Khintchine, Matrix-Azuma / Ahlswede-Winter-2002 predecessor / dimension-free Bernstein Minsker-2017 / linear concentration of measure Talagrand-1995 Ledoux-2001 / sub-Gaussian and sub-exponential matrix tail bounds Vershynin-2018 / BLUE / GLS estimators with random design — reality v0.10.0 ships **ZERO** RMT-specific surface verified by repo-wide grep on `Wigner|MarchenkoPastur|TracyWidom|Wishart|Stieltjes|Voiculescu|free.*prob|RTransform|STransform|Painleve|BBP|spiked|GOE|GUE|GSE|determinantal.*point|Fyodorov|Bouchaud|tridiagonal.*beta|Dumitriu|Edelman|local.*law|Mehta|Bohigas|Wigner.*surmise|spectral.*density|MatrixBernstein|MatrixChernoff|MatrixConcentration|matrix.*Tropp` returning ZERO matches across all 22 packages and `linalg/eigen.go` ships symmetric-eigenvalues only via QR-algorithm with NO eigenvectors-returned NO tridiagonal-public-API NO β-ensemble-sampler NO resolvent NO local-law-test NO Stieltjes-transform-machinery; partial overlap with **184-synergy-linalg-prob C9 MarchenkoPasturDensity + C10 TracyWidomQuantile/CDF + C11 BBPSpikeTest + C13 WishartSample/LogPDF** (proposed-not-shipped at ~520 LOC across `prob/rmt.go` + `prob/multivariate.go`) and **188-synergy-prob-linalg D2 HutchinsonTraceEstimator + D6 LanczosTridiag + D14 LogDetEstimator + D19 WishartSample + D20 RandomOrthogonalMatrix + D25 RandomMatrixSpectralDensity** (proposed-not-shipped at ~1170 LOC); both prior reports converge on `prob.StandardNormalSample` Box-Muller-keystone (184-C12, 188-D1) as sole-keystone-blocking-everything; 184 placed RMT primitives in `prob/rmt.go` while 188 placed them in `linalg/`, and **the third independent vote here from a pure-RMT lens places them in `prob/rmt.go` matching 184** (rationale §3 below).

**Summary line 2.** Twenty-two RMT primitives **R1–R22** totalling ~3,940 LOC of pure connective tissue / new-mathematical-content split across **(a) ~1,150 LOC already-proposed-by-184/188** (R1=184-C9 MP, R2=184-C10 TW, R3=184-C11 BBP, R8=184-C13 + 188-D19 Wishart, R10=188-D6 Lanczos, R11=188-D20 random-orthogonal — same primitives, same pins; SHIP ONCE) and **(b) ~2,790 LOC NET-NEW absent from 184/188** including R4-Wigner-direct-ensemble-sampler GOE/GUE/GSE for Hermitian-iid generation (184/188 covered Wishart-via-MVN but not Hermitian-Wigner-via-symmetrize-iid-Gaussian — distinct distribution, distinct edge), R5-Painlevé-II-ODE-integration-Hastings-McLeod-boundary giving Tracy-Widom CDF independent-of-tabulation, R6-Stieltjes-transform-and-inversion the foundational-tool-of-RMT-analysis NEITHER 184/188 surfaced (Cauchy transform, Stieltjes-Perron-inversion, self-consistent equations) — without Stieltjes machinery the entire local-law / free-probability / resolvent literature is inaccessible, R7-R-transform-and-S-transform Voiculescu free-cumulants enabling free additive/multiplicative convolution `μ ⊞ ν` and `μ ⊠ ν` (free probability is the entire theoretical framework underlying RMT-as-non-commutative-probability and zero of it exists in `prob/`), R9-Dumitriu-Edelman-2002 β-ensemble tridiagonal models reducing GOE/GUE/GSE eigenvalue draws from O(n³) to O(n²) via χ-distribution diagonal entries, R12-Wigner-surmise spacing-distributions `P_β(s)` for β=1,2,4 the canonical-empirical-test-of-quantum-chaos in physics, R13-determinantal-point-processes (DPP) sampling via Hough-Krishnapur-Peres-Virág-2006 spectral algorithm — single most-important RMT consumer in machine learning since Kulesza-Taskar-2012 NeurIPS for diversity-aware subset selection, R14-Fyodorov-Bouchaud-2008 random landscape complexity quantifying number-of-saddles via Kac-Rice formula bridging RMT to optimization landscape theory, R15-Wigner-semicircle-density-and-CDF the historical-foundation Wigner-1955, R16-Matrix-Bernstein/Chernoff/Hoeffding/Khintchine four-pin family Tropp-2012 (vs 184-C19 Bernstein-only), R17-Bohigas-Giannoni-Schmit chaos-vs-GOE statistical test, R18-spiked-covariance-generative-model R19-Wilks-Lambda-MANOVA Jacobi-ensemble R20-Roy's-largest-root R21-empirical-spectral-distribution R22-edge-local-law / bulk-local-law Erdős-Yau-Yin-2012 finite-N rigidity tests; Tier-1 keystone **R1+R2+R3+R4+R5+R6 = `prob/rmt.go` MP-density + TW-distribution-via-Painlevé + BBP-test + Wigner-ensemble-sampler + Stieltjes-transform ~1,180 LOC** is the irreducible foundation; **SINGULAR competitive moat: R5 Painlevé-II + R7 R/S-transform free-convolution + R13 DPP sampler ~840 LOC** because no zero-dependency Go library ships any of them — Julia-RandomMatrices.jl is the only canonical reference; reality would be FIRST production Go RMT library AND FIRST cross-language byte-identical Painlevé-II-via-Hastings-McLeod numerical-integration giving Tracy-Widom CDF independent of Prähofer-Spohn lookup; **SINGULAR Block-C-2026 frontier: R5 + R6 + R22 local-laws** since Erdős-Schlein-Yau "universality conjecture" trilogy 2010-2024 is the most-cited 2010s RMT breakthrough (~5000 citations across) and reality's golden-file pinning at IEEE-754 boundaries against Bornemann-2010 Painlevé-II reference numerical-integration would be unique; cross-package blockers `prob.StandardNormalSample` Box-Muller absent (gates R4/R8/R9/R11/R13/R18), `linalg.SVD` absent gates R20-Roy's-root, `linalg.SymmetricEigenvectors` absent (`linalg/eigen.go:212` ships eigenvalues-only via QR-algorithm — no eigvecs returned even though `pca.go` privately uses inverse-iteration trick) gates R13 DPP-spectral-sampling and R22 local-law-tests; recommended placement single new file `prob/rmt.go` ~3,790 LOC plus R10 LanczosTridiag in `linalg/krylov.go` (188-D6 owner) ~150 LOC — NO new packages; landing order PR-1=R1+R2+R3+R15+R16 ~520-LOC ship-now-against-existing-linalg.QRAlgorithm-eigvals saturating canonical-MP-vs-Wigner three-way R-MUTUAL-3/3 pin, PR-2=R4+R8+R9+R11 ~640-LOC blocked-on-Box-Muller, PR-3=R5+R6 ~480-LOC stand-alone Painlevé-II + Stieltjes (the moat), PR-4=R7+R12 ~440-LOC stand-alone R/S-transform + Wigner-surmise, PR-5=R13+R14+R17 ~460-LOC blocked-on-symmetric-eigenvectors, PR-6=R10+R22 ~520-LOC blocked-on-Lanczos. Cross-link to 184: R1=C9 R2=C10 R3=C11 R8=C13 R16⊃C19 — six RMT primitives shared, ship once. Cross-link to 188: R10=D6 R11=D20 R8=D19 R6⊃D14-machinery — four shared, ship once. Cross-link to 215-CS R18-spiked-covariance dual-to-CS-sensing-matrix-conditioning. Differentiation §6: this report is RMT-theory-pure (Wigner-Mehta-Tracy-Widom-Voiculescu-Dyson canon) where 184 was high-dim-covariance-application-focused and 188 was randomized-NLA-application-focused; zero overlap on R5/R6/R7/R9/R12/R13/R14/R17/R20/R22 (10 of 22 primitives unique to this slot).

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Repo-wide audit for RMT surface — verified via Grep on full primitive lexicon:

| Surface | Path | RMT relevance |
|---|---|---|
| `linalg.QRAlgorithm` (sym eigvals) | `linalg/eigen.go:20` | Eigenvalues only via Householder-tridiag + tqli — no eigvecs returned; PRESENT-but-partial |
| `linalg.PCA` | `linalg/pca.go:215` | Has eigvecs internally for top-k via inverse-iteration; NOT exported |
| `linalg.CholeskyDecompose` / `CholeskySolve` | `linalg/decompose.go` | Wishart-via-Bartlett needs Cholesky; PRESENT |
| `linalg.MatMul` / `MatVecMul` / `MatTranspose` | `linalg/matrix.go` | RMT operations build on these; PRESENT |
| `linalg.CovarianceMatrix(data, out)` | `linalg/correlation.go` | Sample-covariance; PRESENT |
| `prob.NormalPDF/CDF/Quantile` | `prob/distributions.go` | Closed-form; PRESENT but no SAMPLER (Box-Muller absent) |
| `prob.WishartSample`, `MarchenkoPasturPDF/CDF`, `TracyWidomCDF/Quantile`, `WignerSemicirclePDF/CDF`, `WignerEnsembleSample`, `DumitriuEdelmanBetaEnsemble`, `PainleveII`, `StieltjesTransform`, `RTransform`/`STransform`, `DPPSample`/`DPPLikelihood`, `WignerSurmise`, `FyodorovBouchaudComplexity`, `MatrixBernsteinBound`, `LocalLawTest`, `EmpiricalSpectralDistribution`, `RoysLargestRoot`, `BohigasGiannoniSchmitTest` | -- | **ALL ABSENT** (22 distinct primitives) |

**Cross-import edges.** `prob/` → `linalg/`: ZERO today. `linalg/` → `prob/`: ZERO today. Adding R-primitives in `prob/rmt.go` creates the FIRST `prob/` → `linalg/` edge (matching 184). Single direction; no cycles; `linalg/` remains leaf-of-reality per CLAUDE.md "reality imports nothing".

**Cross-package blockers.**

| Blocker | Owner | Blocks | LOC est |
|---|---|---|---:|
| `prob.StandardNormalSample` (Box-Muller) | 117-T2 / 184-C12 / 188-D1 | R4, R8, R9, R11, R13, R18 | 50 |
| `linalg.SymmetricEigenvectors` (export inverse-iter) | 097-T1 | R13 (DPP spectral), R22 (bulk-eigvec test) | 80 |
| `linalg.SVD` (Golub-Reinsch) | 097-T1 | R20 (Roy's root via canonical corr SVD) | 280 |
| `linalg.LanczosTridiag` (188-D6) | 188-D6 / 097-T1 | R10 (spectral density), R22 (matrix-free local-law) | 150 |

Total substrate gating PR-2/PR-5/PR-6 ~560 LOC; PR-1/PR-3/PR-4 ship today against existing surface.

---

## 1. The twenty-two RMT primitives

Each entry: capability + reference / composition / LOC / cross-link / blocking-flag.

**R1 — `MarchenkoPasturPDF/CDF(x, ratio, sigma2) float64`.** Marchenko-Pastur-1967 closed-form `f_MP(x) = √((λ_+ − x)(x − λ_−))/(2π·c·σ²·x)` for `x ∈ [λ_−, λ_+]` with edges `λ_± = σ²·(1 ± √c)²`; for `c > 1` add point mass `(1 − 1/c)·δ_0`. Pure scalar formula; CDF via piecewise arcsin/arctan closed-form. ~150 LOC. **Identical to 184-C9; ship once.** Pin via Monte-Carlo Wishart (R8) KS < 0.01.

**R2 — `TracyWidomCDF/Quantile/PDF(s, beta int) float64`.** Tracy-Widom-1994 / 1996 distribution `F_β(s) = exp(−∫_s^∞ (x − s)·q²(x) dx)` of largest-eigenvalue centered fluctuation via Painlevé-II solution. β=1 GOE, β=2 GUE, β=4 GSE. Tabulated CDF via Prähofer-Spohn-2004 200-grid public-domain tables with monotone interpolation; bisection for quantile; asymptotic Type-2-Gumbel-tail above s=6. ~250 LOC. **Identical to 184-C10; ship once.** Pin scipy.stats.tracy_widom 1e-6 over `s ∈ [-10, 6]`.

**R3 — `BBPSpikeTest(eigvals, n, p, sigma2) (spikes []int, pvalues []float64)`.** Baik-Ben-Arous-Péché-2005 phase-transition spike detection. For each eigenvalue: scaled deviation `s_i = (λ_i − μ_np)/σ_np` per Johnstone-2001 GOE-TW centering; `p-value = 1 − F_β(s_i)`. Composes R1 + R2. ~100 LOC. **Identical to 184-C11; ship once.**

**R4 — `WignerEnsembleSample(n, beta, sigma2, rng, out)`.** Generate one n×n Wigner-iid ensemble matrix. β=1 GOE: real-symmetric `A_ii ~ N(0, 2σ²)`, `A_ij ~ N(0, σ²)`. β=2 GUE: Hermitian `A_ii` real `~ N(0, σ²)`, off-diag `a_ij + i·b_ij` with `a, b ~ N(0, σ²/2)`. β=4 GSE: quaternion-self-dual 2n×2n. Walk lower triangle, draw via Box-Muller (BLOCKED), variance-scale, mirror. ~120 LOC. **NOT proposed by 184/188** — they covered Wishart (X X^T / n) but not Hermitian-Wigner-via-symmetrize-iid-Gaussian (DISTINCT density, semicircle vs MP). Pin n=200 GOE eigenvalues vs Wigner-semicircle (R15) KS < 0.02.

**R5 — `PainleveII(s, q_inf) (q, q_prime)` and `TracyWidomCDFFromPainleve(s, beta)`.** Numerical integration of Painlevé-II ODE `q''(s) = s·q + 2·q³` with Hastings-McLeod-1980 Airy boundary `q(s) ~ Ai(s)` as `s → +∞`. Backward-integrate from `s_max=10` (where `q(s_max) ≈ Ai(s_max)`) to s_min=−10 via adaptive RK4 (existing `chaos.RK4`); reconstruct `F_2(s) = exp(−∫_s^∞ (x − s)·q²(x) dx)` via auxiliary trapezoidal quadrature; `F_1(s)² = F_2(s)·exp(−∫_s^∞ q(x) dx)`; F_4 via Forrester-Rains-2001 Pfaffian (defer). ~280 LOC. **ENRICHES 184-C10 (table-only); both should ship.** Pin Bornemann-2010 Math.Comp.-79(270) reference Painlevé solver; agreement to 1e-10 over `s ∈ [-10, 6]` for β=2. **SINGULAR moat — no other Go library ships this.**

**R6 — `StieltjesTransform(measure, weights, z complex128) complex128` and `StieltjesPerronInvert(G, lambda, eps) float64`.** Cauchy-Stieltjes transform `G_μ(z) = ∫ dμ(λ)/(z − λ)` and inversion `f_μ(λ) = −(1/π)·lim_{ε↓0} Im G_μ(λ + iε)`. Foundational tool of RMT analysis (Bai-Silverstein-2010, Anderson-Guionnet-Zeitouni-2010). Forward: discrete-measure `Σ w_i / (z − x_i)`. Self-consistent equation solver `G(z) = 1/(z − Σ_self(G))` via fixed-point or Newton's-method-on-complex. Inversion: evaluate `G(λ + iε)` for small `ε`. ~180 LOC. **NOT proposed by 184/188** — abstract-foundation neither prior report surfaced. Pin forward against `G_MP(z)` closed-form to 1e-9 for `z = 5 + 0.1i`; inverse to 1e-3 (limited by ε regularization).

**R7 — `RTransform/STransform/FreeAdditiveConvolution/FreeMultiplicativeConvolution`.** Voiculescu-1985-1991 free-probability framework. R-transform `R_μ(z) = G_μ^{-1}(z) − 1/z` linearizes free additive `R_{μ⊞ν} = R_μ + R_ν`. S-transform `S_μ(z) = (1+z)/z · ψ_μ^{-1}(z)` linearizes free multiplicative `S_{μ⊠ν} = S_μ · S_ν`. Free convolution gives spectral-distribution-of-sum-of-free-random-matrices. Numerical inversion of `G_μ` via complex Newton's-method; discrete-measure approximation throughout (1000-point grids). ~280 LOC. **Wholly absent from 184/188.** Free probability is the entire theoretical framework underlying RMT-as-non-commutative-probability — zero of it currently in `prob/`. Pin `μ_GOE ⊞ μ_GOE = scaled-semicircle-with-σ²·2` per Voiculescu to 1e-4.

**R8 — `WishartSample(L_chol_Sigma, p, df, rng, out)` / `WishartLogPDF(W, Sigma, p, df) float64`.** Bartlett-1933 / Smith-Hocking-1972 decomposition: `W = L · A · A^T · L^T`, `A_ii = √χ²(df − i + 1)`, `A_ij ~ N(0,1)` for `i > j`. Log-density via standard formula with multivariate-Gamma `Γ_p` from existing `prob.LogGamma` (`prob/mathutil.go:37` ships Lanczos for log-gamma — verified). ~180 LOC. **Identical to 184-C13 / 188-D19; ship once.** BLOCKED on Box-Muller. Panic on `df ≤ p − 1`.

**R9 — `DumitriuEdelmanBetaEnsemble(n, beta, model string, rng) (alpha, beta_arr)`.** Dumitriu-Edelman-2002 J.Math.Phys.-43(11) β-ensemble tridiagonal. For arbitrary β ∈ R_+: Hermite tridiagonal `α_i ~ N(0, σ²)`, `β_i ~ χ_{β·(n−i)}` — eigenvalues exactly β-Hermite distributed. Reduces O(n³) full-matrix-eigendecomposition to O(n²) tridiagonal-eigendecomposition. Equivalent Laguerre tridiagonal model for β-Wishart and Jacobi tridiagonal for β-Wilks-MANOVA. ~200 LOC. **NOT proposed by 184/188** — SOTA generative model; particularly valuable for goldenfile generation at large n. Pin β=2 (GUE) Hermite tridiag at n=500 vs Wigner-semicircle KS < 0.02.

**R10 — `LanczosTridiag(matvec, n, k, rng) (alpha, beta, Q)`.** Identical to **188-D6**. Random-start Lanczos building k-step Krylov tridiag of symmetric A; selective reorthogonalization (Simon-1984) for k ≥ 30. Used in R22 (matrix-free local-law) and R10-spectral-density. ~150 LOC. Ship in `linalg/krylov.go` per 188-D6 placement.

**R11 — `RandomOrthogonalMatrix(n, rng, Q)` / `RandomUnitaryMatrix(n, rng, Q)` / `HaarRandomQuaternion(n, rng)`.** Identical to **188-D20**. Stewart-1980 + Mezzadri-2007-sign-correction Haar-uniform on O(n) / U(n) / Sp(n). Mezzadri sign correction (`Q ← Q · diag(sign(diag(R)))`) CRITICAL — without it Q is biased toward +1 on diagonal. Used in R4 GUE/GSE conjugation, R13 DPP spectral. ~120 LOC. **BLOCKED on 097-T1 Householder QR.**

**R12 — `WignerSurmise(s, beta) float64` / `WignerSurmiseCDF(s, beta) float64`.** Wigner-1956 / Mehta-1967-2004 surmise `P_β(s) = (Π_β/2)·s^β·exp(−c_β·s²)` with `(Π_1, c_1) = (π/2, π/4)`, `(Π_2, c_2) = (32/π², 4/π)`, `(Π_4, c_4) = (262144/(729·π³), 64/(9·π))`. Pure scalar; CDF via incomplete-Gamma. ~80 LOC. **NOT proposed by 184/188** — quantum-chaos consumer non-statistical / non-covariance, the canonical-physics-application of RMT. Pin Bohigas-Giannoni-Schmit-1984 PRL reference; sodium-Rydberg-atom-spacings empirical match.

**R13 — `DPPSample(L_kernel, n, rng, out)` / `DPPLikelihood(S, L_kernel, n) float64` / `KDPP(L, k, n, rng, out)`.** Macchi-1975 determinantal point process with kernel L (n×n, SPD). Likelihood `P(S) ∝ det(L_S)`. Sampling via Hough-Krishnapur-Peres-Virág-2006 spectral algorithm: eigendecompose L; per eigenvector flip coin `p_k = λ_k/(1+λ_k)`; iteratively project + sample subset given selected eigenvectors. k-DPP Kulesza-Taskar-2012 elementary-symmetric-polynomial dynamic-programming O(nk). ~280 LOC. **NOT proposed by 184/188** — single most-important ML consumer of RMT (diversity-aware-subset-selection in summarization, recommendation, deep-learning batch-construction; Kulesza-Taskar-2012 NeurIPS, Bıyık-Anari-Sadigh-2019). Pin Python `dppy` reference KS < 0.01. **BLOCKED on linalg.SymmetricEigenvectors export.**

**R14 — `FyodorovBouchaudComplexity(n, mu) float64` / `KacRiceFormula(...)`.** Fyodorov-Bouchaud-2008 J.Phys.A-41(32) annealed complexity counting saddle-points-of-random-Gaussian-fields via Kac-Rice-1944 formula. For spherical p-spin: `Σ_n(μ) = (1/n)·log E[# critical points with energy < μ]` undergoes topological-trivialization-transition at critical `μ_c`. Hessian distribution is GOE-shifted (Auffinger-Ben-Arous-Černý-2013 CPAM); integrate via R1 + R15. ~180 LOC. **NOT proposed by 184/188.** Bridges RMT to optimization-landscape theory (deep-learning-loss-landscapes Choromanska-Henaff-Mathieu-Arous-LeCun-2015 derives loss-landscape from spherical p-spin, predicting GOE Hessian and BBP-transition for spurious local minima). Pin Auffinger-Ben-Arous-Černý-2013 reference at `p=3, n=100` to 1e-3.

**R15 — `WignerSemicirclePDF/CDF/Support(x, sigma2)`.** Wigner-1955 Ann.Math.-62(3) semicircle law `ρ_sc(λ) = (1/2π σ²)·√(4σ² − λ²)` for `|λ| ≤ 2σ`; CDF `F_sc(λ) = (1/2) + λ·√(4σ² − λ²)/(4π σ²) + (1/π)·arcsin(λ/(2σ))`. Pure scalar formula. ~80 LOC. **NOT proposed by 184/188** — historical-foundation Wigner-1955 nuclear-physics; 184/188 jumped to MP without addressing Hermitian-iid case. DISTINCT from MP at edge: semicircle symmetric `[-2σ, 2σ]`, MP asymmetric `[(1−√c)², (1+√c)²]`. Pin n=500 GOE empirical histogram vs `ρ_sc` chi² p-value > 0.05.

**R16 — `MatrixBernsteinBound / MatrixChernoffBound / MatrixHoeffdingBound / MatrixKhintchineBound`.** Tropp-2012 Found.Comp.Math-12(4) four-pin family: Bernstein `P(λ_max(Σ Z_k) ≥ t) ≤ d·exp(−t²/(2(σ² + Lt/3)))`, Chernoff (Thm 5.1.1 SPD `Z_k`), Hoeffding (Thm 4.1.1 Rademacher `Z_k = ±A_k`), Khintchine `E‖Σ ε_k A_k‖ ≤ √(2 log d) · ‖(Σ A_k²)^{1/2}‖`. Pure scalar formulas. ~120 LOC. **Strict super-set of 184-C19** (Bernstein-only). Pin Tropp-2015 Found.Trends.Mach.Learn-8(1-2) numerical examples.

**R17 — `BohigasGiannoniSchmitTest(spacings, beta_expected) (chi2, pvalue)`.** Bohigas-Giannoni-Schmit-1984 PRL-52(1) chaos↔GOE test: take consecutive eigenvalue-differences (unfolded by local-mean spacing), histogram, χ² goodness-of-fit against Wigner-surmise R12. ~120 LOC. **NOT proposed by 184/188.** Pin sodium-atom Rydberg-spectrum (NIST table) chaos hypothesis: rejects Poisson at p < 0.001 vs accepts GOE at p > 0.05.

**R18 — `SpikedCovarianceModel(p, theta_population, sigma2, n, rng) (sample_eigvals, sample_topvecs)`.** Generative model for spiked-covariance `Σ = σ²·I_p + Σ_i θ_i u_i u_i^T`. Random orthonormal U via R11 (BLOCKED); Cholesky `Σ = LL^T`; n samples `x_k = L·z_k` for `z_k ~ N(0, I)` (BLOCKED on Box-Muller); sample covariance + eigendecomp via existing `linalg.QRAlgorithm`. ~150 LOC. **NOT proposed by 184/188 explicitly** though implicit in 184-C11 BBP-test infrastructure. Standalone primitive enables Monte-Carlo BBP power-curves. Pin: `θ_1 = 2·σ²·√(p/n)` slightly above BBP threshold → R3 detection probability > 0.9.

**R19 — `WilksLambdaMANOVA(E, H, p, n_e, n_h) (lambda, pvalue)` / `JacobiEnsemblePDF`.** Wilks-1932 lambda statistic `Λ = det(E)/det(E + H)` for high-dim multivariate ANOVA. Joint-eigenvalue distribution of `E^{-1}H` is Jacobi-ensemble (Forrester-2010). Finite-n via Bartlett-correction; large-p / large-n joint asymptotics via Tracy-Widom-Type-I (Johnstone-2008 Ann.Stat.-36(6)). Use existing `linalg.Determinant`. ~200 LOC. **NOT proposed by 184/188.**

**R20 — `RoysLargestRoot(R_xx, R_xy, R_yy, p, q) float64`.** Largest eigenvalue of `R_xx^{-1} R_xy R_yy^{-1} R_yx` — canonical-correlation statistic. H_0 distribution Tracy-Widom-Type-I in large-p/q-asymptotics (Johnstone-2008). Largest singular-value squared via SVD of cross-correlation `R_xy`. ~120 LOC. **BLOCKED on linalg.SVD (097-T1).**

**R21 — `EmpiricalSpectralDistribution(eigvals) func(lambda) float64` / `KSTestVsTheoretical`.** `F_n(λ) = (1/n)·#{i : λ_i ≤ λ}` step CDF; KS-test vs theoretical (Wigner-semicircle, MP, etc.). Sort eigenvalues (stdlib `sort.Float64s`); KS via existing `prob.nonparametric` Kolmogorov-Smirnov-test. ~80 LOC.

**R22 — `LocalLawBulkTest(empirical_eigvals, theoretical_density, n) (chi2, pvalue)` / `LocalLawEdgeTest(largest_k_eigvals, n, beta) (chi2, pvalue)`.** Erdős-Yau-Yin-2012 Adv.Math.-229(3-4) local-laws. Bulk-rigidity `|λ_i − γ_i| ≤ N^{-1+ε}` where `γ_i = F^{-1}(i/n)` classical-location quantile; edge-rigidity at scale `N^{-2/3}` with TW-fluctuation. Top-k eigenvalues should be `μ_n + σ_n · TW_β` distributed via R2. ~200 LOC. **NOT proposed by 184/188.** Most-cited 2010s RMT-breakthrough.

---

## 2. Three cross-cutting connective-tissue patterns

**P1 — Three-way Wigner-MP-Wishart spectral R-MUTUAL pin (R4 + R8 + R15 + R1).** Sample n×n GOE via R4 (n=500); spectrum via `linalg.QRAlgorithm`; histogram vs Wigner-semicircle R15. Independently sample Wishart(I, df=n) via R8; spectrum; histogram vs Marchenko-Pastur (c=1) R1. The two distributions agree on rescaling-equivalence (Bai-Silverstein-2010 §3.4 free-additive-convolution). **Three witnesses on one identity.** Saturates R-MUTUAL-CROSS-VALIDATION 3/3.

**P2 — Two-method Tracy-Widom R-MUTUAL pin (R2-table + R5-Painlevé).** Lookup-table TW (Prähofer-Spohn) vs Painlevé-II-direct-integration (Bornemann-2010); agreement to 1e-9 over `s ∈ [-10, 6]`. Two independent algorithms agree on classical CDF.

**P3 — Three-ensemble Dyson-index pin (R4-β=1 + R4-β=2 + R4-β=4 + R9-tridiagonal).** GOE/GUE/GSE direct R4 vs Hermite-tridiagonal Dumitriu-Edelman R9 at all four βs. Eigenvalue level-spacings R12 match Wigner-surmise `P_β(s)` per ensemble. Cross-check tridiagonal-O(n²) vs full-O(n³) gives same spectrum to floating-point precision.

---

## 3. Architectural placement — `prob/rmt.go` (matching 184)

184 places RMT in `prob/rmt.go`; 188 places randomized-NLA in `linalg/`. **This report agrees with 184** for ALL R-primitives except R10 LanczosTridiag (in `linalg/krylov.go` per 188-D6 placement).

Justification: (1) RMT primitives are PROBABILISTIC OBJECTS — eigenvalue distributions, fluctuation laws, free-convolution algebras. Caller intent: "what's the null distribution of the largest sample-covariance eigenvalue?"; reaches for `prob/`. (2) `prob/copula/` already lives in `prob/` not `linalg/` even though copulas use multivariate-distribution algebra; same logic. (3) Discoverability — statistician asks "where is Marchenko-Pastur?"; scrolling `prob/` finds it next to other distributions. (4) Single-file containment — 22 R-primitives at ~3,790 LOC fit in one file; matches `prob/distributions.go` 700-LOC file size.

| File | Primitives | LOC | Edges |
|---|---|---:|---|
| `prob/rmt.go` (NEW) | R1-R9, R11-R22 (21 of 22) | ~3,790 | prob → linalg (Cholesky, MatMul, QRAlgorithm) |
| `linalg/krylov.go` (188-D6 owner) | R10 LanczosTridiag | ~150 | linalg → prob (StandardNormalSample) |

**Total ~3,940 LOC.** First `prob/` → `linalg/` edge in repo. No cycles.

---

## 4. Landing order

- **PR-1 (4 days, 520 LOC).** R1 + R2 + R3 + R15 + R16 = MP + TW-table + BBP + Wigner-semicircle + Matrix-concentration four-pin. **Stand-alone**, ships against existing `linalg.QRAlgorithm`. Saturates 184-P1 + adds Wigner-semicircle goodness-of-fit pin. Largest immediate-consumer-leverage; closes 184-PR-5 first three primitives.
- **PR-2 (4 days, 640 LOC).** R4 + R8 + R9 + R11 = Wigner-ensemble-sampler + Wishart + β-ensemble + Mezzadri orthogonal. **BLOCKED on Box-Muller AND linalg.QR.** Once both ship, keystone "I can sample any RMT ensemble at any β" PR.
- **PR-3 (3 days, 480 LOC).** R5 + R6 = Painlevé-II + Stieltjes-transform. **Stand-alone moat.** Lookup-table-free TW + foundational analytical tool. Highest-quality cross-language pin available (Bornemann-2010 reference).
- **PR-4 (3 days, 440 LOC).** R7 + R12 = R/S-transform + Wigner-surmise. **Stand-alone.** Free probability + spacing-statistics. Quantum-chaos test bridge.
- **PR-5 (2 days, 460 LOC).** R13 + R14 + R17 = DPP + Fyodorov-Bouchaud + Bohigas-Giannoni-Schmit. **BLOCKED on linalg.SymmetricEigenvectors export.** ML-diversity + landscape-complexity + chaos-test consumers.
- **PR-6 (3 days, 520 LOC).** R10 + R22 = Lanczos-spectral-density + local-laws. **BLOCKED on Lanczos (188-D6).** Matrix-free spectral density + most-cited 2010s RMT breakthrough.
- **PR-7 (2 days, 320 LOC).** R18 + R19 + R21 = Spiked-covariance + Wilks-Λ + EmpiricalSpectralDistribution. **Stand-alone consumer-utility wrappers.**
- **PR-8 (BLOCKED).** R20 = Roy's-largest-root. **BLOCKED on linalg.SVD (097-T1).**

Total ~3,930 LOC source + ~1,400 LOC tests/golden over ~21 engineer-days, three R-MUTUAL pins (P1, P2, P3) saturated.

---

## 5. Precision hazards

R1 MP density singular at λ_− when c=1 (square data) — guard with `x ≤ λ_− + ε` branch. R2 TW lookup undefined outside `s ∈ [-10, 6]` — clamp + asymptotic-Type-2-Gumbel-tail. R3 BBP power vanishes near θ = σ²·√(p/n) — document detectability boundary. R4 Wigner GUE off-diagonal `a + ib` with `a, b ~ N(0, σ²/2)` — common error using `~ N(0, σ²)` gives factor √2 wrong spectrum. R5 Painlevé-II backward-integration from `s_max` only stable when `s_max ≥ 13` (Bornemann-2010 §3) for double-precision. R6 Stieltjes-Perron inversion `ε ≈ 10^{-3}` introduces O(ε) bias. R7 R-transform numerical inversion has multiple branches — guard with branch-cut handling on positive-imaginary half-plane. R8 Wishart singular almost surely if `df ≤ p − 1` — panic. R9 Dumitriu-Edelman non-integer β needs gamma-distribution sampling (Marsaglia-Tsang-2000), not just sum-of-normals. R10 Lanczos loses orthogonality after ~30 steps — selective reorth required for k ≥ 30. R11 Mezzadri sign-correction CRITICAL. R12 2×2 surmise approximate — true GOE level-spacing differs ~5% in tails. R13 DPP kernel L must be SPD with eigenvalues in [0, 1]. R14 Fyodorov-Bouchaud closed-form only for spherical-p-spin. R16 Bernstein assumes `‖Z_k‖ ≤ L` deterministically — for unbounded `Z_k` use sub-exponential variant (defer). R17 local-mean-unfolding sensitive to spectrum-density-variation. R20 Roy's-root TW-Type-I valid only for `min(p, q, n−q) → ∞` bounded ratios. R22 bulk-rigidity scale `N^{-1+ε}` requires N ≥ 10⁴ for finite-N agreement.

---

## 6. Cross-language pinning targets

| Primitive | Reference impl | Tolerance |
|---|---|---|
| R1 | scipy doesn't ship MP directly; pin via Monte-Carlo Wishart KS-test | 0.01 |
| R2 | scipy.stats.tracy_widom (1.13+) | 1e-6 |
| R3 | Onatski-2009 reference table | 1e-3 |
| R4 | Julia RandomMatrices.jl `GaussianHermite(n)` | distribution match |
| R5 | Bornemann-2010 reference Mathematica solver tables | 1e-10 |
| R6 | discrete-sum forward eval pinned analytically; inverse via dense-grid R1 | 1e-9 |
| R7 | Julia FreeProb.jl free additive convolution | 1e-4 |
| R8 | scipy.stats.wishart.rvs / logpdf | 1% mean over 10⁴ draws |
| R9 | Dumitriu-Edelman-2002 J.Math.Phys reference | KS < 0.02 |
| R10 | scipy.sparse.linalg.eigsh (Lanczos under hood) | 1e-8 |
| R11 | scipy.stats.ortho_group.rvs (Mezzadri-corrected) | uniform over O(n) |
| R12 | Mehta-2004 Ch.3 closed-form table | 1e-12 |
| R13 | Python `dppy` library | KS < 0.01 |
| R14 | Auffinger-Ben-Arous-Černý-2013 reference | 1e-3 |
| R15 | classical formula | 1e-15 |
| R16 | Tropp-2015 worked examples | 1e-15 |
| R17 | Bohigas-Giannoni-Schmit-1984 PRL reference values | 1e-3 |
| R19 | scipy.stats.f for small-p Bartlett; Johnstone-2008 large-p | 1e-6 |
| R22 | Erdős-Yau-Yin-2012 finite-N reference simulations | 0.05 chi² pvalue |

---

## 7. Differentiation from prior agents

- **184-synergy-linalg-prob** (high-dim covariance application: LW / OAS / GLasso / PCA factor / GP / BLR). Six-primitive overlap (R1=C9, R2=C10, R3=C11, R8=C13, R16⊃C19). 16 net-new HERE.
- **188-synergy-prob-linalg** (randomized-NLA application: rSVD / Hutchinson / sketched-LS / Krylov-with-random-start). Four-primitive overlap (R8=D19, R10=D6, R11=D20, R6⊃D14-machinery). 18 net-new HERE.
- **097-linalg-missing** identifies QR/SVD/Lanczos/SymEigvec as T1 absences. THIS adds 5 RMT consumers blocked on those (R10/R11/R13/R20/R22); sharpens 097-T1 priority.
- **117-prob-missing** flags Box-Muller / Wishart / MVN. THIS shows downstream RMT canon they unblock — 6 of 22 primitives directly downstream.
- **120-prob-perf** orthogonal — perf comes after primitives ship.
- **161-synergy-control-prob** Kalman is special case; orthogonal generative process.
- **180-synergy-physics-prob** Boltzmann/Gibbs is statistical-physics counterpart; R12/R17 quantum-chaos is RMT-physics adjacent. Cross-link: nuclear-physics-spectra historical-motivation Wigner-1955 = R4 + R12 + R17 trio.
- **183-synergy-calculus-autodiff** cross-link at R5 Painlevé-II — autodiff over Painlevé-II ODE-integration would enable derivative-of-TW-CDF for likelihood-gradient methods.
- **201-new-optimal-transport** dual at free-probability level — Wasserstein-distance and free-convolution both define metric structures on probability measure spaces. Different layer; orthogonal.
- **214-new-pairings** orthogonal (cryptography vs RMT).
- **215-new-compressed-sensing** cross-link at R18 spiked-covariance vs CS-sensing-matrix-conditioning. Co-shipping prob.StandardNormalSample unblocks both 215-CS-sensing AND R4/R8/R9 simultaneously.
- **184-188 placement disagreement**: 184 places RMT in `prob/`, 188 places randomized-NLA in `linalg/`. THIS is third independent vote confirming `prob/rmt.go` correct for RMT-specific objects (Wigner, MP, TW, Stieltjes, free-convolution, DPP, local-laws). 188 placement of randomized-NLA in `linalg/` remains correct for that distinct class.

---

## 8. Bottom-line recommendation

**Single-day high-leverage commit if-only-one-PR ships PR-1 = R1 + R2 + R3 + R15 + R16 = 520 LOC source + 200 LOC tests** because (a) closes largest documented absence in `prob/` (no RMT support today) at single-PR cost, (b) saturates canonical Wigner-MP-Wishart three-way R-MUTUAL pin (P1) against `linalg.QRAlgorithm` ground-truth, (c) stands alone — zero new linalg primitives, zero new prob keystones, (d) establishes FIRST `prob/` → `linalg/` import edge, architecturally pivotal, unlocks next 20 R-primitives over weeks rather than months, (e) closes 184-PR-5 (three of its four primitives are R1/R2/R3 here; R16 supersedes 184-C19), (f) positions reality as FIRST zero-dep Go RMT library (no existing Go library ships any of MP / TW / BBP / Wigner-semicircle / matrix-Bernstein).

**Second-best one-day commit: PR-3 = R5 + R6 = 480 LOC = Painlevé-II + Stieltjes-transform** as singular-competitive-moat. Bornemann-2010-quality numerical Painlevé-II in pure Go gives lookup-table-free TW CDF AND foundational analytical tool of RMT theory — neither exists in any Go library today. Architectural-leverage commit moving reality from "RMT consumer of lookup tables" to "RMT first-principle implementer".

**Highest-leverage architectural addition = (PR-1 + PR-3 + PR-7 = R1+R2+R3+R15+R16+R5+R6+R18+R19+R21) at ~1,320 LOC** because together they ship: (i) canonical-distribution-family (MP, TW, Wigner-semicircle); (ii) analytical-numerical machinery (Painlevé-II, Stieltjes-transform); (iii) consumer-utility wrappers (spiked-covariance, Wilks-Λ, EmpiricalSpectralDistribution). Three R-MUTUAL pins saturated. Five engineer-days. After this, every RMT-paper-since-1955 is composable from reality primitives.

**Cross-package critical-path: prob.StandardNormalSample (Box-Muller) is the SINGLE SHARED keystone with 117-T2 / 184-C12 / 188-D1.** All four reports converge on this 50-LOC primitive as the most-leveraged single commit reality could ship. Once shipped, R4/R8/R9/R11/R13/R18 (six RMT primitives), C12a-C20 (six covariance primitives in 184), D2-D18 (seventeen randomized-NLA primitives in 188) all unblock. Combined unblock factor ~30×. **THIS is the load-bearing recommendation across all four reports.**
