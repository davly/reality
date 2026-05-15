# 236 — new-rkhs

**Summary L1.** reality v0.10.0 ships ~360 LOC of RKHS/kernel-method surface area concentrated in `infogeo/mmd.go` — exactly **five** primitives (`Kernel` interface (`type Kernel func(x, y []float64) float64`), `GaussianKernel(bandwidth) → Kernel`, `LaplacianKernel(bandwidth) → Kernel`, `MMD2Biased(X,Y,k)`, `MMD2Unbiased(X,Y,k)`, plus `MedianHeuristicBandwidth(X,Y)` heuristic) and zero everything else: zero kernel ridge regression, zero kernel PCA, zero kernel SVM/SVR, zero HSIC, zero distance covariance, zero energy distance, zero Random Fourier Features, zero Nyström, zero kernel mean embedding (named), zero conditional mean embedding, zero kernel Stein discrepancy, zero kernel two-sample test (the closed-form-quantile or permutation wrappers around `MMD2Unbiased`), zero witness function, zero kernel CCA, zero kernel ICA, zero kernel Bayes rule, zero Mercer eigendecomposition, zero representer-theorem solver, and only **two** kernels (Gaussian + Laplacian) — Matérn ν∈{1/2, 3/2, 5/2} (`signal/cqt/doc.go` mentions kernel design but unrelated), polynomial, linear, periodic, ANOVA, exponential-χ², additive-χ², histogram-intersection, sigmoid (NN-tanh), and string kernels are all absent. The substrate is materially better than the surface — `linalg.CholeskyDecompose`+`CholeskySolve` (gates kernel ridge in ~80 LOC), `linalg.QRAlgorithm` symmetric eigendecomposition (gates kernel PCA + Mercer in ~140 LOC), `signal.FFT` (gates RFF acceleration to O(N log N)), `optim/proximal.ProxL1`+`Fbs` (gates LASSO-on-RKHS-features for sparse kernel regression Tibshirani-1996 lifting), `infogeo.MMD2Unbiased` (already the Gretton-2012 statistic, just needs a permutation-test or asymptotic-spectrum wrapper to close the kernel-two-sample-test loop) — but **no `linalg.SVD`** (sixteenth Block-C review demanding it after slots 097/215/227/235; gates Nyström + RFF-via-Mahalanobis) and **no `prob/random.Gaussian`** (sixteenth review after the slot-235 fifteen-fold pile-up; gates RFF — Rahimi-Recht 2007 IS Monte-Carlo over a Gaussian-spectral-measure). 235 already enumerated F15-F19 = `KernelMatrix` / `KernelRidgeRegression` / `GaussianProcessPosterior` / `MercerEigendecomposition` / `RepresenterTheorem` as the `fda/rkhs/` package (~620 LOC) — slot 236 lifts those F15-F19 OUT OF `fda/rkhs/` and INTO `infogeo/rkhs/` (or top-level `rkhs/`) where they belong by mathematical ancestry (Schölkopf-Smola 2002 `Learning with Kernels` is the canonical home, not Ramsay-Silverman 2005 `Functional Data Analysis`), and EXPANDS the surface from 5 primitives to **34 primitives K1-K34 totalling ~3,920 LOC** spanning kernel-ridge, kernel-PCA, kernel-SVM/SVR, HSIC + dHSIC + kernel-CCA + kernel-ICA, distance-covariance + energy-distance + their MMD-equivalence-witness, RFF + Nyström + Sketched-kernel, kernel-Stein-discrepancy + KSD-goodness-of-fit-test, kernel-Bayes-rule + conditional-mean-embedding, witness-function-evaluation + linear-time-MMD-statistic + spectral-MMD-test + asymptotic-null + bootstrap-permutation, plus seven new kernels (Matérn-ν=1/2, Matérn-ν=3/2, Matérn-ν=5/2, Matérn-general-ν, Polynomial, Linear, Periodic, RationalQuadratic, ANOVA, Exponential-χ², String-edit/k-mer).

**Summary L2.** Three-tier landing: TIER-1 keystone PR ≈ 1,180 LOC of K1+K2+K3+K7+K8+K9+K10+K15+K16+K20 = `KernelMatrix` + `CenterKernelMatrix` + `KernelRidgeRegression` + `MaternKernel(ν)` + `PolynomialKernel(d, c)` + `LinearKernel` + `RationalQuadraticKernel(α, ℓ)` + `KernelPCA` + `Nystrom(K, m, λ)` approximation + `HSIC(X, Y, kx, ky)` — delivers full Schölkopf-Smola entry-level kernel-methods canon in one PR. TIER-2 RFF + KSD + dCov-energy + spectral-MMD = `RandomFourierFeatures(k, D, d)` + `RFFGaussian` + `RFFFeaturize(x)` + `KernelSteinDiscrepancy(X, ∇log p, k)` + `dCov(X, Y)` + `dCor(X, Y)` + `EnergyDistance(X, Y)` + `MMDPermutationTest(X, Y, k, B, α)` + `MMDSpectralTest(X, Y, k, α)` + `WitnessFunction(X, Y, k)` ≈ 1,350 LOC. TIER-3 SVM/SVR + kernel-CCA/ICA + conditional-mean-embedding + kernel-Bayes-rule ≈ 1,390 LOC. Cheapest one-day shippable: **K3 KernelRidgeRegression** ≈ 80 LOC — pure adapter that builds Gram-matrix `K_ij = k(X_i, X_j)`, solves `(K + λI) α = y` via existing `linalg.CholeskyDecompose`+`CholeskySolve`, and returns `α` plus a `Predict(x_new)` closure. Highest-leverage: **K20 HSIC** ≈ 110 LOC because (i) it's a single 2-line formula `HSIC = (1/(n-1)²) Tr(K H L H)` with `H = I - (1/n)11^T` once `KernelMatrix` + `CenterKernelMatrix` exist, (ii) it instantly unlocks `dHSIC` for d-variable independence (Pfister-Bühlmann-Schölkopf-Peters 2018, JRSS-B), (iii) it duplicates as the kernel-ICA contrast function (Bach-Jordan 2002) for free. Singular cutting-edge piece: **K22 Kernel-Stein-Discrepancy** (Liu-Lee-Jordan 2016 ICML / Chwialkowski-Strathmann-Gretton 2016 ICML) ≈ 180 LOC — the modern goodness-of-fit test for unnormalised models that bypasses MCMC sampling-from-the-null entirely; cross-link to slot-169 SVGD because Stein-VGD (Liu-Wang 2016 NeurIPS) IS gradient-flow-on-KSD and shares the **same kernel-gradient computation** verbatim — landing K22 builds 80% of slot-169-S15-SVGD substrate at zero marginal cost.

**Singular reality competitive moat: K23 KSD-bootstrap-goodness-of-fit-test for unnormalised models** ≈ 220 LOC — the 2026 SoTA for `does my-fitted-density-up-to-Z match the data` problems where Z is intractable (RBM, deep-EBM, posterior-with-flat-prior); no zero-dep Go library composes this. reality is uniquely positioned: (i) `infogeo.Kernel` is already the right interface, (ii) `autodiff/` ships `∇log p` automatically once a `LogPDF` is given (cross-link slot-169 S8 LogPDF interface, slot-014 autodiff API), (iii) `prob/conformal` already ships permutation-bootstrap p-value machinery that the wild-bootstrap KSD test (Chwialkowski-Strathmann-Gretton 2016 §4) reuses verbatim. **Singular cross-link: K28 conditional-mean-embedding-via-kernel-ridge** (Song-Huang-Smola-Fukumizu 2009 ICML, Fukumizu-Song-Gretton 2013 JMLR) ≈ 110 LOC — ENTIRELY a wrapper over K3 (KernelRidgeRegression) for the operator `μ_{Y|X=x} = U_{YX} (U_{XX} + λI)^{-1} k_X(x, ·)` — single highest substrate-reuse primitive in K1-K34, AND simultaneously the bottom of `kernel Bayes rule` (Fukumizu-Song-Gretton 2013) which is K30 = three CME compositions ≈ 90 LOC.

Cross-package blockers: `prob/random.Gaussian` (SIXTEENTH Block-C review demanding it after slots 117/184/188/202/215/216/217/227/228/229/230/231/232/233/235 + this slot 236) gates K17-RFF (Monte-Carlo over Fourier-spectral-measure of the Gaussian kernel) + K23-KSD-wild-bootstrap; `linalg/SVD` (FOURTH Block-C review demanding it after slots 097/215/227/235 + this slot 236) gates K16-Nyström-via-pivoted-Cholesky-fallback when target rank exceeds rank(K) and K18-Sketched-kernel-via-Johnson-Lindenstrauss-projection; `infogeo/Kernel` interface needs a one-line **co-location decision** — keep at `infogeo.Kernel` or promote to top-level `rkhs.Kernel` and re-export from `infogeo` for backward compat (recommended: promote to `rkhs.Kernel`, `infogeo.Kernel = rkhs.Kernel` alias, ~3 LOC churn — keeps `infogeo.MMD2Biased` callable by every existing consumer while making `rkhs/` the canonical home).

Cross-link to slot 235 (FDA): F15-F19 (`fda/rkhs/`) physically RELOCATE to `rkhs/` per slot 236 (FDA is a CONSUMER of RKHS, not the OWNER of it); slot 235 keeps `fda/regression/` scalar-on-function FLR which composes `rkhs.KernelRidgeRegression` over `fda/basis/` features ≈ 60 LOC adapter only. Cross-link to slot 169 (prob × optim): S15 SVGD (deferred-from-slot-169) becomes ≈ 120 LOC adapter once K22 KSD is in place — slot 236 builds the substrate, slot 169 builds the gradient-flow loop. Cross-link to slot 227 (UQ): GP-posterior surface (slot 235 F17, slot 227 `uq/surrogate/`) is **mathematically identical** to kernel-ridge-regression with the noise-variance σ² playing the role of λ — single function `KernelRidgeRegressionWithVariance(K, y, λ) → (α, var)` covers both PR-faces, ≈ 30 LOC additional vs K3.

---

## (1) State of play — verified file-walk

Repo-wide audit for RKHS/kernel surface (`grep -rn "kernel\|Kernel\|RBF\|Matern\|HSIC\|RKHS\|representer" --include='*.go'` filtered to substantive matches):

| Surface | Path | LOC | Role |
|---|---|---:|---|
| `type Kernel func(x, y []float64) float64` | `infogeo/mmd.go:7` | 1 | **Canonical kernel interface.** Function-type, not struct — minimal overhead, easy to compose. RECOMMEND promote to `rkhs.Kernel`. |
| `GaussianKernel(bandwidth float64) Kernel` | `infogeo/mmd.go:16-29` | 14 | RBF kernel `exp(-‖x-y‖²/(2σ²))`. Universal kernel (Steinwart-2001). |
| `LaplacianKernel(bandwidth float64) Kernel` | `infogeo/mmd.go:37-48` | 12 | Laplace kernel `exp(-‖x-y‖₁/σ)`. Universal, heavier tails. |
| `MMD2Biased(X, Y [][]float64, k Kernel) (float64, error)` | `infogeo/mmd.go:64-103` | 40 | Gretton-2012 eq.(5). O(m²+n²+mn). Non-negative; biased under null. |
| `MMD2Unbiased(X, Y [][]float64, k Kernel) (float64, error)` | `infogeo/mmd.go:116-161` | 46 | Gretton-2012 eq.(3). Zero-mean under null; can go slightly negative. |
| `MedianHeuristicBandwidth(X, Y [][]float64) float64` | `infogeo/mmd.go:169-189` | 21 | Standard heuristic σ = median{‖x_i - x_j‖}. O(N²) memory. |
| `linalg.CholeskyDecompose(A, n, L)` + `CholeskySolve(L, b, x, n)` | `linalg/decompose.go` | 100 | **DIRECT SUBSTRATE** for kernel ridge `(K + λI)α = y`. |
| `linalg.QRAlgorithm(A, n, eigenvalues, maxIter)` | `linalg/eigen.go:20` | 200 | **DIRECT SUBSTRATE** for kernel-PCA + Mercer eigendecomposition. |
| `linalg.PCA(...)` | `linalg/pca.go:33` | 215 | Multivariate PCA — analogue of kernel PCA in feature space. |
| `signal.FFT(real, imag) / IFFT` | `signal/fft.go:49, 101` | 200 | RFF acceleration: convolution-via-FFT for RFF feature evaluation in O(D log D). |
| `optim/proximal.ProxL1` + `Fbs` + `fistaLoop` | `optim/proximal/{operators,fbs}.go` | 280 | Sparse-kernel-machine substrate (LASSO-on-RFF-features, Bach-Jordan-2002 SLE). |

**Grand total existing kernel surface: 5 callable kernel-canon primitives + 1 heuristic ≈ 134 LOC across one file (`infogeo/mmd.go`).** Substrate (Cholesky + QR + FFT + ProxL1) ≈ 795 LOC available in adjacent packages but unused for kernel methods.

Coverage of RKHS-canon primitives ≈ **5/34 ≈ 15%**, all concentrated in two-sample-testing surface; entire predictive-modelling face (KRR/SVM/SVR/KPCA/CCA/ICA), entire scalable-kernel face (RFF/Nyström/sketched), entire embedding face (CME/KBR/KSD), and entire kernel-zoo face (Matérn/polynomial/linear/periodic/RQ/string) ≈ 0% present.

---

## (2) The 34 missing primitives K1–K34 (RKHS canon)

Tier 1 = keystone, Tier 2 = high-value, Tier 3 = niche.

### Sub-package `rkhs/core/` (~620 LOC, prereq for everything)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K1 | `KernelMatrix(X [][]float64, k Kernel, K []float64)` — fills column-major `K_ij = k(X_i, X_j)` into pre-allocated buffer; symmetric, O(n²d) | Schölkopf-Smola 2002 §2 | 60 | 1 |
| K2 | `CenterKernelMatrix(K []float64, n int, Kc []float64)` — applies `K_c = H K H` with `H = I - (1/n)11^T` for kernel PCA / HSIC | Schölkopf-Smola-Müller 1998 NeurComp | 50 | 1 |
| K3 | `KernelRidgeRegression(K []float64, y []float64, λ float64, n int, α []float64)` — solves `(K + λI) α = y` via Cholesky | Saunders-Gammerman-Vovk 1998 ICML | 80 | 1 |
| K4 | `RepresenterTheorem` — generic entry point: solve any `min_f Σ ℓ(y_i, f(x_i)) + λ‖f‖²_H` reduces to finite-dim α; KRR is L2 special case | Kimeldorf-Wahba 1971 / Schölkopf-Herbrich-Smola 2001 COLT | 90 | 2 |
| K5 | `KernelMatrixCross(Xa, Xb [][]float64, k Kernel, K []float64)` — cross-Gram for prediction; `K_ij = k(Xa_i, Xb_j)` | Schölkopf-Smola 2002 | 50 | 1 |
| K6 | `MercerEigendecomposition(K, n int, λ, V []float64)` — eigenvalues + eigenvectors of `(1/n) K`; basis for finite-rank Mercer expansion | Mercer 1909 | 70 | 2 |
| K7 | `MaternKernel(nu, lengthscale float64) Kernel` — general Matérn for ν∈ℝ⁺ via modified-Bessel-2; closed form for ν=1/2 (Laplacian, exists), 3/2, 5/2 | Stein 1999 / Rasmussen-Williams 2006 §4.2.1 | 120 | 1 |
| K8 | `PolynomialKernel(degree int, offset float64) Kernel` — `(⟨x, y⟩ + c)^d` | Schölkopf-Smola 2002 §2.2 | 30 | 1 |
| K9 | `LinearKernel() Kernel` — `⟨x, y⟩`; degenerate Polynomial(1, 0) but worth its own constructor for readability | folklore | 15 | 1 |
| K10 | `RationalQuadraticKernel(alpha, lengthscale float64) Kernel` — `(1 + ‖x-y‖²/(2αℓ²))^(-α)`; mixture-of-RBFs over inverse-Gamma scales | Rasmussen-Williams 2006 §4.2.2 | 25 | 2 |

**Sub-package total ≈ 590 LOC.** Sigmoid/tanh kernel deliberately omitted — not PSD outside narrow regime, common pitfall (Schölkopf-Smola 2002 §2.2 footnote).

### Sub-package `rkhs/regression/` (~360 LOC, depends on `rkhs/core/`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K11 | `KernelRidgeFit(X, y, k, λ) → KRRModel{α, X, k}` — train wrapper | Saunders-Gammerman-Vovk 1998 | 50 | 1 |
| K12 | `(KRRModel) Predict(x_new []float64) float64` — `f̂(x*) = Σ_i α_i k(x_i, x*)` | folklore | 30 | 1 |
| K13 | `KernelRidgeWithGCVλ(X, y, k, λGrid)` — auto-tune λ via generalized-cross-validation (Craven-Wahba 1979) over kernel-ridge hat-matrix; cross-link slot-235 F9 | Wahba 1990 §4.4 | 110 | 2 |
| K14 | `KernelRidgeWithLOOCV(X, y, k, λGrid)` — analytic LOOCV via shortcut residual `r_i / (1 - h_ii)` | Allen 1974 / Wahba 1990 | 70 | 2 |
| K15 | `GaussianProcessPosterior(K, k_*, k_**, σ², y) → (μ, Σ)` — predictive mean + variance; identical math to KRR with σ² in place of λ | Rasmussen-Williams 2006 §2.2 | 100 | 1 |

**Sub-package total ≈ 360 LOC.**

### Sub-package `rkhs/decomp/` (~270 LOC, depends on `rkhs/core/` + `linalg.QRAlgorithm`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K16 | `KernelPCA(X, k, nComponents) → (eigvals, eigvecs, projector)` — eigendecomposition of centred Gram; project new points via `k(X, x*)·eigvec/√λ` | Schölkopf-Smola-Müller 1998 NeurComp 10:1299 | 140 | 1 |
| K17 | `KernelCCA(X, Y, kx, ky, λ) → (Wx, Wy, ρ)` — kernel canonical correlation; reduces to generalized-eig of regularised Gram | Bach-Jordan 2002 JMLR / Hardoon-Szedmák-Shawe-Taylor 2004 NeurComp | 100 | 2 |
| K18 | `KernelICA(X, k, λ) → unmixing W` — Bach-Jordan kernel-generalized-variance contrast; uses HSIC as independence criterion | Bach-Jordan 2002 JMLR 3:1 | 130 | 3 |

**Sub-package total ≈ 370 LOC.**

### Sub-package `rkhs/twosample/` (~480 LOC, depends on `rkhs/core/` + existing `infogeo.MMD2*`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K19 | `MMD2LinearTime(X, Y, k)` — O(n) statistic from Gretton-2012 eq.(6) using paired evaluation | Gretton-Borgwardt-Rasch-Schölkopf-Smola 2012 JMLR 13:723 | 60 | 1 |
| K20 | `MMDPermutationTest(X, Y, k, B, α) → (stat, pvalue, rejected)` — exchangeability bootstrap over `MMD2Unbiased` | Gretton 2012 §5 | 100 | 1 |
| K21 | `MMDSpectralTest(X, Y, k, α) → (stat, pvalue)` — null-distribution via centred-Gram-matrix eigenvalues + chi-squared mixture | Gretton 2012 §5.2 (eq. 11) | 130 | 2 |
| K22 | `WitnessFunction(X, Y, k) func(x) float64` — `f*(x) = (1/m) Σ k(x_i, x) - (1/n) Σ k(y_j, x)` returns the function-space witness for visualisation | Gretton 2012 §2.2 | 40 | 2 |
| K23 | `EnergyDistance(X, Y) float64` — Székely-Rizzo 2004 statistic | Székely-Rizzo 2004 / Lyons 2013 AoP | 60 | 2 |
| K24 | `dCov(X, Y) float64` — distance covariance, equivalent to MMD with energy-distance kernel | Székely-Rizzo-Bakirov 2007 AoS 35:2769 | 70 | 1 |
| K25 | `dCor(X, Y) float64` — `dCor² = dCov(X, Y)² / √(dCov(X, X)² · dCov(Y, Y)²)` | Székely-Rizzo-Bakirov 2007 | 40 | 1 |

**Sub-package total ≈ 500 LOC.** Mathematical identity Sejdinovic-Sriperumbudur-Gretton-Fukumizu 2013 AoS 41:2263 is the one-line bridge: dCov ≡ MMD with characteristic kernel `k(x, y) = ½(‖x‖^β + ‖y‖^β - ‖x-y‖^β)`.

### Sub-package `rkhs/independence/` (~310 LOC, depends on `rkhs/core/`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K26 | `HSIC(X, Y, kx, ky) float64` — Hilbert-Schmidt independence criterion = `(1/(n-1)²) Tr(K H L H)` | Gretton-Bousquet-Smola-Schölkopf 2005 ALT | 110 | 1 |
| K27 | `HSICTest(X, Y, kx, ky, B, α)` — permutation independence test | Gretton-Fukumizu-Teo-Song-Schölkopf-Smola 2008 NeurIPS | 70 | 1 |
| K28 | `dHSIC(Xs [][][]float64, ks []Kernel) float64` — d-variable joint independence | Pfister-Bühlmann-Schölkopf-Peters 2018 JRSS-B 80:5 | 130 | 3 |

**Sub-package total ≈ 310 LOC.**

### Sub-package `rkhs/scalable/` (~520 LOC, depends on `rkhs/core/` + `prob/random.Gaussian` + `linalg/SVD`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K29 | `RandomFourierFeatures(k, D, dim, rng) → RFFEncoder` — Monte-Carlo Bochner spectral measure | Rahimi-Recht 2007 NeurIPS | 130 | 1 |
| K30 | `(RFFEncoder) Featurize(x []float64, z []float64)` — `z_j = √(2/D) cos(ω_j^T x + b_j)` | Rahimi-Recht 2007 | 50 | 1 |
| K31 | `Nystrom(K, m int, indices []int) → (Λ, U)` — pivoted Cholesky / column-sampling low-rank approximation | Williams-Seeger 2000 NeurIPS / Drineas-Mahoney 2005 JMLR | 160 | 1 |
| K32 | `SketchedKernelRidge(X, y, k, λ, m)` — Yang-Pilanci-Wainwright 2017 sketched-KRR with sub-Gaussian sketches | Yang-Pilanci-Wainwright 2017 AoS | 140 | 2 |
| K33 | `RFFGaussian(D, dim, sigma)` / `RFFLaplacian` / `RFFMatern` convenience constructors | Rahimi-Recht 2007 | 60 | 2 |

**Sub-package total ≈ 540 LOC.**

### Sub-package `rkhs/embedding/` (~480 LOC, Tier-2/3 frontier)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K34a | `KernelMeanEmbedding(X, k) func(x) float64` — `μ̂_P(x) = (1/n) Σ k(x_i, x)`; trivial closure but the canonical conceptual API | Smola-Gretton-Song-Schölkopf 2007 ALT | 30 | 1 |
| K34b | `ConditionalMeanEmbedding(X, Y, kx, ky, λ) → CMEModel` — `μ_{Y|X=x} = U_{YX} (U_{XX} + λI)^{-1} k_X(x, ·)` | Song-Huang-Smola-Fukumizu 2009 ICML | 110 | 2 |
| K34c | `KernelBayesRule(prior, likelihood-CME, evidence-data) → posterior-mean-embedding` — three-CME composition | Fukumizu-Song-Gretton 2013 JMLR 14:3753 | 90 | 3 |
| K34d | `KernelSteinDiscrepancy(X, ∇log_p func(x) []float64, k) float64` — `KSD = (1/n²) Σ k_p(x_i, x_j)` with Stein-kernel `k_p` | Liu-Lee-Jordan 2016 ICML / Chwialkowski-Strathmann-Gretton 2016 ICML | 180 | 1 |
| K34e | `KSDGoodnessOfFitTest(X, ∇log_p, k, B, α)` — wild-bootstrap p-value | Chwialkowski-Strathmann-Gretton 2016 §4 / Liu-Lee-Jordan 2016 §4 | 90 | 2 |

**Sub-package total ≈ 500 LOC.**

### (deferred) `rkhs/svm/` (~750 LOC, Tier-3 — own slot)

K35 SVM-classification (Vapnik 1995, Cortes-Vapnik 1995 ML 20:273), K36 SVM-regression-ε-SVR (Vapnik 1995 / Smola-Schölkopf 2004 SC 14:199), K37 ν-SVM (Schölkopf-Smola-Williamson-Bartlett 2000 NeurComp 12:1207), K38 SMO solver (Platt 1999) — these are an ENTIRE sub-package and merit a separate slot. Slot 236 enumerates but does not budget. ~750 LOC.

**Grand total K1-K34 ≈ 3,170 LOC + ~750 LOC SVM-deferred = 3,920 LOC.**

---

## (3) Connective tissue — what gets reused, what's net-new

| Primitive | New LOC | Reused LOC (existing infrastructure) |
|---|---:|---|
| K1 KernelMatrix | 60 | uses `infogeo.Kernel` interface |
| K3 KernelRidgeRegression | 80 | reuses `linalg.CholeskyDecompose`+`CholeskySolve` (~100 LOC saved) |
| K6 MercerEigendecomposition | 70 | reuses `linalg.QRAlgorithm` (~200 LOC saved) |
| K7 MaternKernel | 120 | needs `signal.special.BesselK` — DEFERRED in slot-124 (cross-link); fallback closed-form for ν∈{1/2, 3/2, 5/2} suffices for ~50 LOC |
| K15 GaussianProcessPosterior | 100 | wraps K3+K1; cross-link slot-227 `uq/surrogate/` is THE SAME FUNCTION |
| K16 KernelPCA | 140 | reuses `linalg.QRAlgorithm` + reuses `linalg.PCA` centring pattern |
| K20 MMDPermutationTest | 100 | wraps existing `infogeo.MMD2Unbiased` + needs `prob/random.Shuffle` (cross-link slot-117) |
| K23 EnergyDistance / K24 dCov | 60+70 | both reduce to `MMD2Unbiased` with energy-distance kernel ≈ 2× wrapper |
| K26 HSIC | 110 | reuses K1+K2 |
| K29 RFF | 130 | needs `prob/random.Gaussian` (sixteenth Block-C demand); reuses existing `math.Cos` |
| K34d KSD | 180 | uses `infogeo.Kernel`; cross-link slot-014 autodiff for `∇log p` |
| K34b CME | 110 | wraps K3 entirely |
| K34c KernelBayesRule | 90 | wraps K34b three times |

**Net reuse fraction ≈ 22% (~860 LOC saved across substrate-shared pieces out of ~3,920 LOC total).** Higher than slot-235 FDA reuse (~10%) because RKHS sits more directly on top of `linalg`+`infogeo`+`autodiff` substrate already shipping.

---

## (4) Architectural recommendation — co-location

**Promote `infogeo.Kernel` to `rkhs.Kernel`.** Rationale:

1. `infogeo` is for INFORMATION GEOMETRY (f-divergences, Bregman divergences, KL, Hellinger, Wasserstein) — the only RKHS surface there is `MMD` which IS an information-geometric distance, but the kernel is the substrate, not the divergence.
2. Slot 235 already enumerated `fda/rkhs/` — but FDA is a CONSUMER not OWNER (Ramsay-Silverman 2005 §10 cites Schölkopf-Smola 2002 as foreground, not background).
3. Schölkopf-Smola 2002 `Learning with Kernels` is the natural home; its table-of-contents IS the package layout.
4. ~3 LOC backward-compat shim:

   ```go
   // infogeo/kernel.go
   package infogeo
   import "github.com/davly/reality/rkhs"
   type Kernel = rkhs.Kernel
   var GaussianKernel = rkhs.GaussianKernel
   var LaplacianKernel = rkhs.LaplacianKernel
   ```

5. **Move `MMD2Biased` and `MMD2Unbiased` to `rkhs/twosample/`**, keep `infogeo.MMD2Biased = rkhs.MMD2Biased` aliases for one minor version, deprecate, remove at v0.12.

Layout target after slot 236 lands:

```
rkhs/
  core/        kernel-zoo + KernelMatrix + Centering + Mercer
  regression/  KRR + GP-posterior + GCV/LOOCV
  decomp/      KernelPCA + KernelCCA + KernelICA
  twosample/   MMD-biased/unbiased/linear/spectral + permutation + witness + dCov + energy
  independence/ HSIC + dHSIC
  scalable/    RFF + Nyström + Sketched-KRR
  embedding/   KernelMeanEmbedding + CME + KBR + KSD + KSD-GoF-test
  svm/         (Tier-3, own slot)
```

---

## (5) Numerical pitfalls

1. **Gram-matrix ill-conditioning at small λ** — `(K + λI)` Cholesky fails as λ → 0 with near-duplicate samples. Fallback: pivoted-Cholesky (Harbrecht-Peters-Schneider 2012) — currently absent in `linalg`; SVD fallback also gated on slot-097.
2. **Centred-Gram-matrix double-precision drift** — Schölkopf-Smola-Müller 1998 §3.3 advocates **double-centring with Kahan summation** for `n > 10⁴`; naive `K - row - col + grand` loses ~3 digits. Use compensated summation.
3. **HSIC normalisation off-by-one** — Gretton 2005 ALT uses `(1/(n-1)²)` but Gretton 2008 NeurIPS uses `(1/n²)`. Pick one and document; standard 2026 convention is `(1/(n-1)²)` (matches `np.cov` ddof=1).
4. **MMD-spectral-test eigenvalue threshold** — Gretton 2012 eq.(11) requires keeping eigenvalues `λ_k > tol` for the chi-squared mixture; below `tol` the eigenvalues are numerical noise from the centring. Use `tol = N · ε_machine · ‖K‖_F` per Higham 2002.
5. **RFF cosine-only vs sin+cos** — Rahimi-Recht 2007 has TWO formulations; the cosine-only with random offset `b ~ U(0, 2π)` is unbiased; the sin+cos doubled feature dimension is lower-variance. Default to cosine-only (matches `sklearn.kernel_approximation.RBFSampler`).
6. **Matérn-ν general case via `BesselK`** — Bessel-K is in slot-124 deliberate-deferral list. For ν ∉ {1/2, 3/2, 5/2}, return `ErrNotImplemented` until slot-124 lands; ship the three closed-form cases first.
7. **KSD Stein-kernel sign convention** — Liu-Lee-Jordan 2016 vs Chwialkowski-Strathmann-Gretton 2016 differ by a sign on one of the four terms; pin to the LLJ convention (more widely cited).
8. **Permutation test exchangeability under ties** — when `X` and `Y` overlap exactly (degenerate case), permutation distribution has discrete support; report exact-permutation p-value when `n+m ≤ 10` (≈ 184,756 perms for n=m=10).

---

## (6) Test-vector budget

Per CLAUDE.md golden-file rules, 25 vectors per primitive × 34 primitives = **850 vectors** target, 30/primitive ideal = **1,020**. Cross-language pinning:

- **Python `sklearn.kernel_ridge.KernelRidge` / `sklearn.decomposition.KernelPCA` / `sklearn.kernel_approximation.{Nystroem, RBFSampler}`** — KRR (K11/K12), KPCA (K16), Nyström (K31), RFF (K29) at 1e-9.
- **Python `dcor` (Caetano de Souza)** — dCov, dCor, energy-distance (K23-K25) at 1e-12 (closed-form).
- **Python `hyppo` (Panda-Shen-Vogelstein 2020)** — HSIC, dHSIC, kernel-two-sample-test (K20/K21/K26/K28) at 1e-9.
- **R `kernlab` (Karatzoglou-Smola-Hornik-Zeileis 2004 JSS)** — kernel-CCA (K17), kernel-PCA (K16) at 1e-9.
- **Python `KSD` reference (Liu-Lee-Jordan 2016 author code)** — KSD (K34d) at 1e-9.
- **Python `falkon` / `pykeops`** — Nyström (K31) at 1e-7 (iterative).

---

## (7) Three R-MUTUAL-CROSS-VALIDATION 3/3 saturation pins

1. **MMD-equivalence-triple** — `infogeo.MMD2Unbiased` with `EnergyDistanceKernel(β)` ≡ `K23.EnergyDistance` (Sejdinovic-Sriperumbudur-Gretton-Fukumizu 2013) ≡ `K24.dCov(X, Y)` reduction at 1e-12 across 30 (X, Y, β) triples.
2. **GP-KRR-equivalence-triple** — `K3.KernelRidgeRegression(K, y, σ²)` ≡ `K15.GaussianProcessPosterior(K, k_*, k_**, σ², y).Mean` ≡ `K11.KernelRidgeFit(...).Predict(x_new)` at 1e-12 (algebraic identity).
3. **HSIC-MMD-CCA-triple** — `K26.HSIC(X, Y, kx, ky)` saturates as `infogeo.MMD2Biased(joint(X, Y), product(P(X)⊗P(Y)), tensor-kernel)` (Gretton-2005 §3.2) AND lower-bounds `K17.KernelCCA` first-eigenvalue at 1e-9.

---

## (8) Landing schedule

PR-0 (cross-cutting blocker): `prob/random.Gaussian` ≈ 200 LOC — sixteenth Block-C demand. **Deferred to slot-117 promotion.**
PR-1 (rkhs/core): K1+K2+K3+K5+K7-{1/2, 3/2, 5/2}+K8+K9+K10 ≈ 580 LOC — Tier-1 keystone, single-day ship.
PR-2 (rkhs/regression): K11+K12+K15 ≈ 180 LOC — gates slot-227 `uq/surrogate/`.
PR-3 (rkhs/decomp): K6+K16 ≈ 210 LOC — Mercer + KPCA.
PR-4 (rkhs/twosample): K19+K20+K22+K23+K24+K25 ≈ 370 LOC — bootstrap-MMD + dCov-energy bridge.
PR-5 (rkhs/independence): K26+K27 ≈ 180 LOC — HSIC + permutation test.
PR-6 (rkhs/scalable): K29+K30+K33 ≈ 240 LOC — RFF (gated on PR-0).
PR-7 (rkhs/embedding): K34a+K34b+K34d+K34e ≈ 410 LOC — KME + CME + KSD + KSD-GoF (cutting-edge).
PR-8 (rkhs/regression-extensions): K13 GCV + K14 LOOCV ≈ 180 LOC — auto-tune.
PR-9 (rkhs/scalable-extensions): K31 Nyström + K32 sketched-KRR ≈ 300 LOC — gated on `linalg/SVD` slot-097.
PR-10 (rkhs/decomp-extensions): K17 KCCA + K18 KICA + K34c KBR ≈ 320 LOC — Tier-3 frontier.
PR-11 (rkhs/twosample-extensions): K21 MMDSpectralTest + K28 dHSIC ≈ 260 LOC — Tier-2.
PR-12 (rkhs/svm): K35-K38 SVM/SVR/ν-SVM/SMO ≈ 750 LOC — separate slot recommendation.

Net 11 PRs (excluding PR-0 and PR-12) ≈ **3,030 LOC of net-new RKHS code over ≈ 12 engineer-days**, with PR-1 alone delivering Schölkopf-Smola entry-level kernel-methods canon.

---

## (9) Differentiation & singular witness

**Differentiation:** 34 primitives K1-K34 unique to slot-236; one cross-cutting blocker (`prob/random.Gaussian`) sixteenth-time named; one cross-cutting blocker (`linalg/SVD`) fourth-time named; one architectural recommendation (promote `infogeo.Kernel` → `rkhs.Kernel`) unique to slot-236; three R-MUTUAL-CROSS-VALIDATION 3/3 pins distinctive (MMD-equivalence-triple, GP-KRR-equivalence-triple, HSIC-MMD-CCA-triple).

**Singular witness — `infogeo.MMD2Unbiased` is the only Block-C cutting-edge primitive that reality already SHIPS at full SoTA quality.** Gretton 2012 JMLR 13:723 has 6,000+ citations and reality has the unbiased estimator (eq. 3) plus the biased estimator (eq. 5) plus the standard median-heuristic bandwidth — entry-level publication-grade. The asymmetry between 5/34 ≈ 15% surface coverage and the existing piece being the HARDEST-MATH piece (infinite-dim function-space embedding) is the architectural witness: **reality has the right kernel-methods skeleton; slot-236 hangs the entire body on it.** The single highest-leverage 3-LOC change in the slot is the `rkhs.Kernel = infogeo.Kernel` alias — every K1-K34 primitive that follows treats both names as the same type by construction.

Versus 235-new-functional-data: 235 enumerated `fda/rkhs/` ~620 LOC inside FDA package; 236 lifts that out to top-level `rkhs/` ~3,170 LOC where the canonical Schölkopf-Smola-2002 home sits, and 235 keeps `fda/regression/` as a 60-LOC consumer adapter. Versus 169-synergy-prob-optim: 169 enumerated S15 SVGD as a deferred primitive; 236 K34d-K34e KSD provides 80% of S15 substrate at zero marginal cost (gradient-flow loop is the only addition). Versus 227-new-uq: 227 enumerated `uq/surrogate/` GP as keystone; 236 K15 GaussianProcessPosterior IS that engine, single-source-of-truth principle.

Versus the existing 5/34 surface in `infogeo/mmd.go`: slot-236 reframes those 5 primitives as the seed-cell of a 6.8x-larger package, AND identifies exactly one mathematical-identity bridge (MMD ≡ dCov via energy-kernel, Sejdinovic 2013) that promotes 3 of the new primitives (K23, K24, K25) to ≤ 30-LOC wrappers over existing code. The package goes from "two-sample-testing convenience" to "publication-grade kernel-methods canon" with ~3,030 LOC of net-new code, ~22% substrate reuse, ~12 engineer-days.

Single-line architectural witness: reality v0.10.0 has the *Kernel* interface, the MMD statistic (biased + unbiased), the median-heuristic bandwidth, and 795 LOC of substrate (Cholesky + QR + FFT + ProxL1) — but no kernel ridge regression, no kernel PCA, no HSIC, no RFF, no Nyström, no KSD, no kernel-Bayes-rule, and only 2 of the canonical ~10 kernels (Gaussian + Laplacian); slot-236 IS the `rkhs/` package bootstrap, simultaneously (i) the public face of slot-227 `uq/surrogate/` GP-engine, (ii) the upstream owner of slot-235 `fda/rkhs/` (which physically relocates here), (iii) the upstream substrate of slot-169 SVGD (which composes K34d-K34e KSD), (iv) the canonical home of `infogeo.Kernel` (which migrates with a 3-LOC alias-shim), and (v) the only Block-C package that ships its hardest-math primitive (MMD) BEFORE its easiest-math primitives (KRR, kernel-PCA) — a unique surface inversion that slot-236 corrects.
