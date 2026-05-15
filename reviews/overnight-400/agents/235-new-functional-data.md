# 235 — new-functional-data

**Summary L1.** reality v0.10.0 ships **zero** functional-data-analysis (FDA) surface — repo-wide grep on `FPCA|FunctionalPCA|KarhunenLoeve|EigenFunction|PSpline|SmoothingSpline|Eilers.*Marx|GCV|GeneralizedCrossValidation|RKHS|ReproducingKernel|Mercer|Representer|KernelRidge|GaussianProcess|FunctionalRegression|PACE|Yao.*Muller|Wavelet|Daubechies|Haar|FunctionalANOVA|RegisterCurves|DTW.*FPCA|TimeWarp` returns **zero callable matches** across all 22 packages; the substrate available consists of (a) `linalg.PCA` (215 LOC, multivariate covariance-eigendecomposition — does NOT lift to FPCA without basis expansion, weighted inner-product, and trapezoidal-rule discretisation of `∫ φ_k(t) X_i(t) dt`), (b) `optim.CubicSplineNatural` (interpolating natural cubic spline via Thomas-tridiagonal — NOT a smoothing/penalised spline; passes through every data point with zero penalty term), (c) `geometry.BezierCubic` + `geometry.CatmullRom` (parametric design curves, not statistical smoothers), (d) `signal.FFT/IFFT` + `HannWindow/HammingWindow/BlackmanWindow` (Fourier basis raw atoms but no orthonormal-basis expansion API, no wavelet basis at all), (e) `infogeo.GaussianKernel` + `LaplacianKernel` + `MMD2Biased/Unbiased` + `MedianHeuristicBandwidth` (kernel surface but only for two-sample testing — no kernel-ridge regression, no GP, no representer-theorem solver, no Mercer eigendecomposition), (f) `optim/proximal.ProxL1` + `Fbs/fistaLoop` + `Admm` (sparse-regression machinery that the P-spline `λ ∫ (g'')²` regulariser would need only as a quadratic penalty — close but not direct), (g) `linalg.QRAlgorithm` (symmetric-eigenvalue routine; would be the substrate for covariance-operator eigendecomposition once a Gram matrix is built). There is **no `fda/` package**, **no B-spline basis**, **no orthonormal Fourier basis evaluator**, **no Daubechies/Haar wavelet**, **no smoothing-spline solver**, **no P-spline (Eilers-Marx 1996)**, **no GCV (Craven-Wahba 1979)**, **no RKHS class**, **no kernel-ridge regression**, **no Gaussian-process surrogate** (also flagged in slot 227 UQ as `uq/surrogate/` keystone), **no Karhunen-Loève expansion** (also flagged in slot 227 UQ as `uq/kle/` keystone), **no FPCA**, **no PACE algorithm (Yao-Müller-Wang 2005, JASA, 4,200+ citations) for sparse FDA**, **no functional GLM**, **no scalar-on-function / function-on-scalar / function-on-function regression**, **no functional ANOVA**, **no functional clustering** (k-means in L²), **no curve registration / time-warping** (Ramsay-Li 1998 dynamic time warping, Srivastava-Klassen 2016 SRSF / square-root velocity Fisher-Rao), **no functional outlier detection** (functional boxplot — Sun-Genton 2011 JCGS), **no functional depth** (modified band depth — López-Pintado-Romo 2009 JASA), **no echo-state-network reservoir** (Jaeger-2001 ESN, cross-link 191), **no manifold learning for functional data**. The two flagship references — Ramsay-Silverman 2005 *Functional Data Analysis* (Springer, 13,500+ citations) and Kokoszka-Reimherr 2017 *Introduction to Functional Data Analysis* (CRC, 1,600+ citations) — are unrepresented; the SOTA monograph Hsing-Eubank 2015 *Theoretical Foundations of Functional Data Analysis with an Introduction to Linear Operators* (Wiley, 1,100+ citations) is unrepresented; the foundational R package `fda` (Ramsay-Hooker-Graves 2009) and Python `scikit-fda` / `fdasrsf` are entirely without Go-zero-dep peer.

**Summary L2.** Twenty-six primitives F1–F26 totalling ~5,180 LOC across new sub-packages `fda/basis/` (B-spline + Fourier + monomial + tensor-product + Daubechies/Haar wavelet bases, ~720 LOC, prerequisite for everything else), `fda/smooth/` (smoothing spline + P-spline Eilers-Marx + GCV cross-validation + roughness-penalty O'Sullivan-1986, ~640 LOC, depends on `fda/basis/`+`linalg.CholeskyDecompose`+`linalg.QRAlgorithm`), `fda/fpca/` (Karhunen-Loève via Gram-matrix eigendecomposition + dense-FPCA + sparse PACE Yao-Müller-Wang 2005 + functional eigenvalue/eigenfunction + scores reconstruction, ~580 LOC, depends on `fda/basis/`+`fda/smooth/`+`linalg.QRAlgorithm`), `fda/rkhs/` (RKHS abstraction + representer theorem + kernel ridge regression + Gaussian-process posterior mean/variance + Mercer eigendecomposition, ~620 LOC, depends on `infogeo.Kernel`+`linalg.CholeskyDecompose`+`linalg.QRAlgorithm`), `fda/regression/` (scalar-on-function PCR + function-on-scalar + function-on-function + functional GLM-link + concurrent model, ~580 LOC, depends on `fda/fpca/`+`fda/rkhs/`+`prob.LinearRegression`), `fda/registration/` (curve alignment + landmark warping + Fisher-Rao SRSF Srivastava-Klassen 2016 + dynamic time warping + Procrustes mean shape, ~520 LOC, depends on `fda/basis/`+`optim.LBFGS`), `fda/depth/` (modified band depth López-Pintado-Romo 2009 + functional boxplot Sun-Genton 2011 + outliergram Arribas-Gil-Romo 2014 + 50%-central-region, ~280 LOC, pure-functional), `fda/cluster/` (k-means in L² + functional k-medoids + DTW-distance hierarchical, ~240 LOC, depends on `fda/basis/`+ `fda/registration/`), `fda/anova/` (one-way functional ANOVA F-statistic Cuevas-Febrero-Fraiman 2004 + permutation test, ~200 LOC, depends on `fda/fpca/`+`prob/random/` Gaussian sampler), `fda/wavelet/` (DWT Mallat 1989 + Daubechies-4/8 + Haar + soft/hard thresholding Donoho-Johnstone 1995 + universal threshold, ~520 LOC, pure-signal, sibling to `signal/fft.go`), `fda/esn/` (Echo-State-Network reservoir Jaeger-2001 + leaky-integrator + ridge-regression readout + spectral-radius spectral-radius scaling, ~280 LOC, depends on `linalg.MatVecMul`+`linalg.QRAlgorithm`).

**Tier-1 keystone F1+F2+F3+F4+F7+F10 = `fda/basis/bspline.go` + `fda/basis/fourier.go` + `fda/smooth/pspline.go` + `fda/smooth/gcv.go` + `fda/fpca/dense.go` + `fda/rkhs/krr.go` ~2,180 LOC** is the irreducible foundation that unblocks every FDA workflow — every Ramsay-Silverman 2005 chapter assembles from {orthonormal-basis-expansion} × {penalised-smoother-with-CV-tuning} × {covariance-operator-eigendecomposition} × {kernel-method-via-representer-theorem}, so shipping these six in one PR delivers the entire entry-level FDA literature simultaneously.

**Singular reality competitive moat: F26 ESN-reservoir for functional time-series ~280 LOC** — no zero-dep Go FDA library composes echo-state reservoirs (Jaeger 2001, recently re-popularised via Pathak-Lu-Hunt-Girvan-Ott 2017 *Chaos* for chaotic-system forecasting, 1,800+ citations) with functional-data input streams; reality is uniquely positioned because (i) `chaos/ode.go` ships RK4 for generating training trajectories on Lorenz-attractor / Van-der-Pol benchmarks, (ii) `linalg.QRAlgorithm` provides spectral-radius computation for reservoir-tuning, (iii) `optim/proximal.ProxL1` supplies the L1-sparse-readout that distinguishes 2026-era reservoirs from Jaeger's original L2.

**Singular Block-C-2026 frontier: F12 PACE algorithm (Yao-Müller-Wang 2005) + F19 Fisher-Rao SRSF curve registration (Srivastava-Klassen 2016) + F23 functional outliergram (Arribas-Gil-Romo 2014)** — these are the modern arrivals that distinguish a 2025-era FDA library from a 2005-era one (PACE handles sparse/irregular longitudinal data — the 90% of biomedical-FDA workflows; SRSF is the geometrically-correct shape-space metric that decouples amplitude and phase variation; outliergram is the bivariate visualisation that catches shape-outliers that magnitude-based functional boxplots miss).

**Singular cross-link: F10/F11 RKHS+kernel-ridge to existing `infogeo.Kernel` + `infogeo.MMD2Biased` ~80 LOC** — `infogeo/mmd.go` already ships `Kernel` interface (`Eval(x, y []float64) float64`) and `GaussianKernel`/`LaplacianKernel`; an RKHS abstraction reuses the SAME interface verbatim and adds a `KernelRidgeRegression(K, y, λ) → α` solver in ~80 LOC on top of `linalg.CholeskyDecompose`. This is the single highest-substrate-reuse primitive in F1-F26.

Cross-package blockers: `prob/random.Gaussian` (currently ABSENT — verified by `grep` on `Sample\(|NormFloat64|Box.?Muller|Marsaglia.*polar|Ziggurat` returning zero matches in `prob/`) gates F14 (function-on-function regression bootstrap CI), F22 (functional-ANOVA permutation test), F26 (ESN reservoir random-weight initialisation) — same blocker repeated across 117/202/215/227/228; `linalg/SVD` (currently ABSENT, verified absent in slot 097, slot 215, slot 227) gates F8 (FPCA-via-SVD as an alternative to covariance-eigendecomposition for `n > p` matrices, the de-facto SOTA for functional regression with high-resolution curves) and F11 (kernel-ridge ill-conditioned-design solver fallback). Cross-link to slot 227-new-uq: 227 enumerated `uq/kle/` (Karhunen-Loève) and `uq/surrogate/` (Gaussian-process Kriging) as Tier-1 keystones — F8 (FPCA) IS the discrete computational engine of `uq/kle/` (KL expansion of a stochastic process is functional-PCA of its sample paths) and F11 (GP-posterior) IS the surrogate engine of `uq/surrogate/`; **F8+F11 from slot 235 should ship as the public face of the private slot-227 internal calculations** (single-source-of-truth principle). Cross-link to slot 228-new-bayes-nonparam: 228 enumerated Gaussian-process-prior + Dirichlet-process-mixture as keystones — F11 directly delivers the GP-prior; F25 (functional clustering via Bayesian-nonparametric) is the missing-mile. Cross-link to slot 169-synergy-prob-optim: 169 enumerated MAP-via-LBFGS as a flagship synergy — F4 (GCV smoothing-parameter selection) is structurally MAP-by-marginal-likelihood (Wahba 1990 — REML-equivalence theorem) and F19 (curve registration by warping-function MAP) is the literal application. Cross-link to slot 191-synergy-chaos-control: 191 enumerated chaotic-system-prediction — F26 ESN reservoir IS the canonical 2026-era replacement for LSTM on chaotic-time-series tasks. Cross-link to slot 215-new-compressed-sensing: 215 enumerated wavelets as keystone — `fda/wavelet/` ships exactly the same Daubechies-4/Haar/soft-threshold primitives that 215 needs for L1-recovery, AND `fda/smooth/` LASSO-on-basis-coefficients is the *direct application* of compressed-sensing to functional-data smoothing.

Versus 117-prob-missing: 117 enumerated `prob/random` PRNG-sampling as #1 missing primitive; this slot 235 reaffirms (F14, F22, F26 blocked). Versus 097-linalg-missing: 097 enumerated SVD as #1 missing decomposition; this slot 235 reaffirms (F8 alt-formulation, F11 ill-conditioned-design) and adds `linalg.GeneralizedEigenvalue` as a NEW demand (F7 covariance-operator eigendecomposition with non-trivial inner-product weight matrix).

Net: reality at v0.10.0 has **zero FDA surface** but possesses **substantially more FDA substrate than any peer Go-library** — `linalg.PCA` (multivariate analogue), `optim.CubicSplineNatural` (interpolant cousin of smoother), `infogeo.Kernel`+`MMD2Biased` (kernel surface), `optim/proximal.ProxL1`+`Fbs` (regularised-regression machinery), `linalg.QRAlgorithm` (symmetric-eigen for covariance-operator), `signal.FFT/IFFT` (Fourier-basis raw transform), `geometry.BezierCubic`+`CatmullRom` (parametric-curve cousins), `calculus.GaussLegendre`+`SimpsonsRule`+`TrapezoidalRule` (numerical-integration of `∫ φ_j(t) φ_k(t) dt` Gram-matrix elements) — together these substrates mean F1-F26 connect through ~5,180 LOC of NEW code with ~520 LOC of substrate reuse. The reuse fraction (~10%) is moderate-to-high for any new Block-C package.

---

## (1) State of play — what reality v0.10.0 already ships (verified file-walk)

Repo-wide audit for FDA-relevant surface:

| Surface | Path | LOC | FDA relevance |
|---|---|---:|---|
| `PCA(data, nSamples, nFeatures, nComponents, components, explained) → cumVar` | `linalg/pca.go:33` | 215 | **DIRECT BUILDING-BLOCK** for FPCA. Multivariate PCA via covariance-matrix QR-eigenvalues + inverse-iteration eigvecs. To lift to FPCA: (a) replace data matrix with basis-coefficient matrix `C` (n × K), (b) sandwich covariance with Gram matrix `W = ∫ φ_j(t) φ_k(t) dt`, (c) solve generalised eigenvalue problem `W·Σ·W·v = λ·W·v` (currently absent — `linalg.QRAlgorithm` only handles standard `Av = λv`). |
| `CubicSplineNatural(xs, ys) → func(x) float64` | `optim/interpolate.go:44` | 113 | **INTERPOLATING — NOT SMOOTHING.** Passes through every (xs[i], ys[i]) exactly via Thomas tridiagonal solve, natural-BC `S''(x_0) = S''(x_n) = 0`. Smoothing-spline replaces interpolation system with `(B^T B + λ R)c = B^T y` — same Thomas-tridiagonal kernel applies, but with a roughness-penalty matrix `R = ∫ B''(t) B''(t)^T dt` that is currently uncomputed. Direct stepping-stone but NOT the deliverable. |
| `LinearInterpolate(x0, y0, x1, y1, x)` / `geometry.LinearInterpolate(a, b, t)` | `optim/interpolate.go:18`, `geometry/curves.go:15` | 4 each | Trivial baseline — duplicated in two packages. |
| `BezierCubic(p0, p1, p2, p3, t)` / `BezierCubic3D` | `geometry/curves.go:28, 40` | 30 | Parametric design curve (CAD/animation) — Bernstein basis. NOT a statistical smoother. Could be repurposed as a 4-control-point B-spline-of-degree-3 single-knot expansion but lacks knot vector / multiple control points / smoothing. |
| `CatmullRom(p0, p1, p2, p3, t)` | `geometry/curves.go:68` | 17 | Parametric interpolating spline — passes through control points. NOT a smoother. |
| `FFT(real, imag) / IFFT / PowerSpectrum / FFTFrequencies` | `signal/fft.go:49, 101, 140, 167` | 200 | **FOURIER-BASIS RAW TRANSFORM** — building block for `fda/basis/fourier.go` periodic-basis evaluator and for fast-FPCA on uniformly-sampled curves (Wood-2017 *Generalized Additive Models* §4.4 covers FFT-based GAM smoothing). |
| `HannWindow / HammingWindow / BlackmanWindow / ApplyWindow` | `signal/window.go:15, 44, 76, 104` | 100 | Windowing primitives — relevant for short-time FPCA on sliding windows (functional-data sliding-window analysis, Hörmann-Kokoszka 2010). |
| `Convolve / MovingAverage / ExponentialMovingAverage / MedianFilter` | `signal/filter.go:19, 54, 97, 130` | 200 | Ad-hoc smoothers — NOT principled splines but adjacent. `MedianFilter` is robust-functional-smoother substrate (cross-link to slot 232 robust-stats). |
| `GaussianKernel(bandwidth) → Kernel` / `LaplacianKernel` | `infogeo/mmd.go:16, 37` | 30 | **DIRECT REUSE** for `fda/rkhs/`. The `Kernel` interface (`Eval(x, y []float64) float64`) is the canonical RKHS-defining object via Mercer's theorem. RKHS construction = `K_x := k(x, ·)` — needs only an `KMatrix(X, X) []float64` adaptor (~40 LOC) and a `KernelRidgeRegression(K, y, λ) → α` solver (~80 LOC). |
| `MMD2Biased / MMD2Unbiased` | `infogeo/mmd.go:64, 116` | 100 | RKHS-induced two-sample test — adjacent (functional-MMD = MMD on RKHS = MMD on `fda/basis/` coefficients). |
| `MedianHeuristicBandwidth(X, Y)` | `infogeo/mmd.go:169` | 22 | Bandwidth-selection — directly reusable for `fda/rkhs/` Gaussian-kernel and `fda/smooth/` Nadaraya-Watson functional smoother. |
| `QRAlgorithm(A, n, eigenvalues, maxIter)` | `linalg/eigen.go:20` | 200 | **DIRECT BUILDING-BLOCK** for FPCA covariance-operator eigendecomposition (`Σ = Φ Λ Φ^T` ⇒ eigenfunctions are `φ_k(t) = Σ_j Φ_jk · b_j(t)` after basis expansion). Only symmetric-real eigenvalues — gates F7 generalised-eigenvalue (needs `Av = λBv`). |
| `CholeskyDecompose(A, n, L)` / `CholeskySolve` | `linalg/decompose.go` | 70+30 | **DIRECT BUILDING-BLOCK** for kernel-ridge `(K + λI) α = y` solve, smoothing-spline `(B^T B + λ R) c = B^T y` solve, GP-posterior `(K + σ² I) α = y` solve. The single most-used factorisation in FDA. |
| `LUDecompose / LUSolve / Inverse / Determinant` | `linalg/decompose.go` | 200 | Backup factorisation when Cholesky is non-PSD (P-spline `R` matrix is PSD but `B^T B + λ R` can be ill-conditioned for very small λ). |
| `MatMul / MatTranspose / MatVecMul / MatAdd / MatSub / MatScale / Trace` | `linalg/matrix.go` | 209 | Routine matrix algebra. |
| `LinearRegression(x, y) → (slope, intercept, R²)` | `prob/regression.go:36` | 50 | Univariate OLS. F12 functional-regression replaces with multivariate `(B^T B)c = B^T y` — needs `linalg.MatMul`+`linalg.CholeskySolve`. |
| `ProxL1(v, gamma, out)` / `ProxL0` / `ProxSquaredL2` | `optim/proximal/operators.go:28, 48, 69` | 60 | Sparse-FDA primitives (Lin-Wang-Wu-Cai 2017 sparse-FPCA via L1-penalty on basis coefficients). |
| `Fbs / fistaLoop / Admm` | `optim/proximal/fbs.go:57, 106`, `proximal/admm.go:53` | 280 | Sparse-regression solvers (Aneiros-Vieu 2014 FPCA-LASSO functional regression, Hall-Horowitz 2007 FLR via penalised-FPC). |
| `GaussLegendre(f, a, b, points)` / `SimpsonsRule` / `TrapezoidalRule` | `calculus/calculus.go:149, 95, 47` | 170 | **DIRECT BUILDING-BLOCK** for Gram-matrix `W_jk = ∫ b_j(t) b_k(t) dt` and inner-product `⟨X_i, φ_k⟩ = ∫ X_i(t) φ_k(t) dt` element-by-element computation (when no closed form exists for the basis pair). |
| `NumericalGradient / NumericalDerivative` | `calculus/calculus.go:47, 26` | 60 | Derivatives for roughness-penalty `∫ (g'')²` — substitute when basis-derivative closed form is unavailable. |

**Total existing FDA-relevant substrate: ~2,030 LOC across 7 packages, none of which is FDA-specific.** Coverage of canonical FDA primitives ≈ **0%** (substrate is universal-math, not domain-FDA).

---

## (2) The 26 missing primitives (FDA canon — Ramsay-Silverman 2005 + Hsing-Eubank 2015 + Kokoszka-Reimherr 2017)

Each row: primitive | citation | LOC budget | Tier (1 = keystone, 2 = high-value, 3 = niche).

### Sub-package `fda/basis/` (~720 LOC, prereq for everything)

| # | Primitive | Citation | LOC | Tier |
|---|---|---|---:|:--:|
| F1 | `BSplineBasis(degree, knots, t) []float64` — Cox-de-Boor recursion | de Boor 1972 *J. Approx. Theory* | 240 | 1 |
| F2 | `FourierBasis(K, t, period) []float64` — orthonormal trig basis | Ramsay-Silverman 2005 §3.4 | 80 | 1 |
| F3 | `BSplineDerivative(degree, knots, t, order) []float64` — closed-form basis derivative | de Boor 1972 §X | 120 | 1 |
| F4 | `BSplineGramMatrix(knots, degree, order) []float64` — penalty matrix `R_jk = ∫ b_j^(order)(t) b_k^(order)(t) dt` via Gauss-Legendre | Eilers-Marx 1996 *Stat Sci* | 140 | 1 |
| F5 | `MonomialBasis(degree, t)` + `LegendreBasis(degree, t)` (orthogonal-poly siblings) | Cross-link 227 UQ `uq/poly/` | 80 | 2 |
| F6 | `TensorProductBasis(basis1, basis2)` for 2-D functional surfaces | Wood 2003 *J.R.S.S. B* | 60 | 3 |

### Sub-package `fda/smooth/` (~640 LOC, depends on F1-F4)

| # | Primitive | Citation | LOC | Tier |
|---|---|---|---:|:--:|
| F7 | `SmoothingSpline(x, y, λ) → (coefs, fitted, df)` — solve `(B^T B + λ R) c = B^T y` | Reinsch 1967 *Numer. Math.* | 200 | 1 |
| F8 | `PSpline(x, y, K, degree, λ, penalty_order) → coefs` — Eilers-Marx penalised B-spline | Eilers-Marx 1996 *Stat Sci* | 180 | 1 |
| F9 | `GCV(x, y, λ_grid) → λ_optimal` — Generalised Cross-Validation `GCV(λ) = n·RSS / (n − tr(H_λ))²` | Craven-Wahba 1979 *Numer. Math.* | 140 | 1 |
| F10 | `CrossValidationLeaveOneOut(x, y, smoother, λ_grid) → λ_optimal` | Stone 1974 *J.R.S.S. B* | 80 | 2 |
| F11 | `OSullivanPenalty(degree, knots) → R` — O'Sullivan original penalised-spline (degree-2 derivative `(2k−2)`-th-order polynomial) | O'Sullivan 1986 *Stat Sci* | 40 | 3 |

### Sub-package `fda/fpca/` (~580 LOC, depends on F1-F4 + F7 + linalg)

| # | Primitive | Citation | LOC | Tier |
|---|---|---|---:|:--:|
| F12 | `DenseFPCA(curves, basis, K) → (eigenvalues, eigenfunctions, scores, varExplained)` — generalised-eigenvalue `W Σ W v = λ W v` after basis expansion | Ramsay-Silverman 2005 §8.4 | 240 | 1 |
| F13 | `SparseFPCAviaPACE(times, values, ids, K, bandwidth) → (eigfns, scores, var)` — Yao-Müller-Wang 2005 PACE algorithm | Yao-Müller-Wang 2005 *JASA* | 240 | 1 |
| F14 | `KarhunenLoeveExpansion(covarianceFn, domain, K) → (eigvals, eigfns)` — KL via Galerkin discretisation (cross-link 227 `uq/kle/`) | Karhunen 1947 / Ramsay-Silverman 2005 §8.2 | 100 | 2 |

### Sub-package `fda/rkhs/` (~620 LOC, depends on F1 + infogeo.Kernel + linalg)

| # | Primitive | Citation | LOC | Tier |
|---|---|---|---:|:--:|
| F15 | `KernelMatrix(X, kernel) → K []float64` + `KernelMatrixCross(X, Y, kernel) → K_xy` | Schölkopf-Smola 2002 §2 | 80 | 1 |
| F16 | `KernelRidgeRegression(K, y, λ) → α` — solve `(K + λI) α = y` via Cholesky | Saunders-Gammerman-Vovk 1998 *ICML* | 120 | 1 |
| F17 | `GaussianProcessPosterior(X_train, y_train, X_test, kernel, σ²) → (mean, variance)` | Rasmussen-Williams 2006 §2.2 | 180 | 1 |
| F18 | `MercerEigendecomposition(kernel, X, K) → (eigvals, eigvecs)` — empirical Mercer eigenfunctions | Mercer 1909 / Williams-Seeger 2000 *NeurIPS* (Nyström) | 140 | 2 |
| F19 | `RepresenterTheoremSolver(X, y, kernel, lossGrad, λ) → α` — generic representer-theorem reduction of any RKHS-regularised problem to finite-dim | Kimeldorf-Wahba 1971 *J. Math. Anal. Appl.* | 100 | 2 |

### Sub-package `fda/regression/` (~580 LOC, depends on F12 + F16)

| # | Primitive | Citation | LOC | Tier |
|---|---|---|---:|:--:|
| F20 | `ScalarOnFunctionFLR(curves, y, K) → β(t)` — functional principal-component regression | Hall-Horowitz 2007 *Ann. Stat.* | 160 | 1 |
| F21 | `FunctionOnScalarFLR(x, curves, K) → β_k(t)` — pointwise OLS on basis coefficients | Ramsay-Silverman 2005 §13.4 | 120 | 2 |
| F22 | `FunctionOnFunctionFLR(curves_X, curves_Y, K_X, K_Y) → β(s, t)` — bivariate-coefficient surface | Yao-Müller-Wang 2005 *Ann. Stat.* | 200 | 2 |
| F23 | `FunctionalGLM(curves, y, family, K) → β(t)` — logistic/Poisson functional GLM | James 2002 *J.R.S.S. B* | 100 | 3 |

### Sub-package `fda/registration/` (~520 LOC)

| # | Primitive | Citation | LOC | Tier |
|---|---|---|---:|:--:|
| F24 | `LandmarkRegistration(curves, landmarks) → warping_funcs` — warp time so landmarks align | Kneip-Gasser 1992 *Ann. Stat.* | 140 | 2 |
| F25 | `FisherRaoSRSF(curves) → (aligned, warpings, mean_shape)` — square-root velocity Karcher mean on shape space | Srivastava-Klassen 2016 Springer | 240 | 1 |
| F26 | `DTWFunctional(x_curve, y_curve, costFn) → (path, dist)` — dynamic time warping for curves | Sakoe-Chiba 1978 *IEEE TASSP* | 140 | 2 |

### Sub-package `fda/depth/` + `fda/cluster/` + `fda/anova/` + `fda/wavelet/` + `fda/esn/`

(F27-F36 — listed as overflow tier-3 / tier-2 to keep core to 26)

- F27 `ModifiedBandDepth(curves) → depths` — López-Pintado-Romo 2009 *JASA* (~80 LOC, T2)
- F28 `FunctionalBoxplot(curves) → (median, 50%CR, whiskers, outliers)` — Sun-Genton 2011 *JCGS* (~120 LOC, T2)
- F29 `Outliergram(curves) → (mei, mbd, mask)` — Arribas-Gil-Romo 2014 *Biostat* (~80 LOC, T3)
- F30 `FunctionalKMeans(curves, K, basis) → (labels, centroids)` — Abraham-Cornillon-Matzner-Lober-Molinari 2003 *Scand. J. Stat.* (~120 LOC, T3)
- F31 `FunctionalANOVAPermutation(curves, groups, B) → p_value` — Cuevas-Febrero-Fraiman 2004 *J. Stat. Plan. Inf.* (~120 LOC, T3)
- F32 `DWTHaar(signal) → coefs` + `IDWTHaar` (~80 LOC, T3)
- F33 `DWTDaubechies(signal, taps={4,6,8}) → coefs` + inverse Mallat 1989 *IEEE TPAMI* (~240 LOC, T2)
- F34 `WaveletThresholding(coefs, type={soft,hard}, threshold) → denoised` Donoho-Johnstone 1995 *Biometrika* (~80 LOC, T2)
- F35 `UniversalThreshold(coefs, σ_noise) → λ` — Donoho-Johnstone `λ = σ √(2 log n)` (~40 LOC, T3)
- F36 `EchoStateNetwork(N_reservoir, leak, ρ) → (W_in, W, W_fb) struct` + `train(X_train, y_train, λ_ridge)` Jaeger 2001 *GMD Tech Rep* (~280 LOC, T2 — singular-Block-C-2026 frontier)

(LOC budget for F27-F36 = ~1,240 LOC additional; combined with F1-F26 at ~5,180 LOC delivers ~6,420 LOC total full-FDA package.)

---

## (3) Connective tissue — what F1-F26 reuse from existing reality substrate (~520 LOC reuse)

| FDA primitive | Existing substrate consumed | Reuse-LOC saved |
|---|---|---:|
| F1 BSplineBasis | new — no existing B-spline | 0 |
| F2 FourierBasis | `signal/fft.go` for fast-evaluation on uniform grids | 30 |
| F4 BSplineGramMatrix | `calculus.GaussLegendre` for off-diagonal `∫ b_j(t) b_k(t) dt` integrals | 40 |
| F7 SmoothingSpline | `linalg.CholeskyDecompose` + `linalg.CholeskySolve` for `(B^T B + λ R) c = B^T y` | 80 |
| F8 PSpline | same Cholesky path + `optim/proximal.ProxSquaredL2` for ridge-equivalence | 60 |
| F9 GCV | `linalg.Trace` + `linalg.MatMul` for `tr(H_λ) = tr(B (B^T B + λ R)^{-1} B^T)` | 30 |
| F12 DenseFPCA | `linalg.PCA` (after replacing covariance with weighted covariance) + `linalg.QRAlgorithm` for symmetric-eigen + `calculus.SimpsonsRule` for `⟨φ_k, X_i⟩` | 100 |
| F13 SparseFPCAviaPACE | `infogeo.MedianHeuristicBandwidth` for kernel bandwidth + Nadaraya-Watson via `infogeo.GaussianKernel` | 40 |
| F14 KarhunenLoeve | `linalg.QRAlgorithm` symmetric-eigen + `calculus.GaussLegendre` Galerkin | 50 |
| F15 KernelMatrix | `infogeo.Kernel` interface — DIRECT REUSE (zero re-implementation) | 30 |
| F16 KernelRidgeRegression | `linalg.CholeskyDecompose` + `linalg.CholeskySolve` | 40 |
| F17 GaussianProcessPosterior | `linalg.CholeskyDecompose` + `linalg.CholeskySolve` + log-det via Cholesky diagonal | 60 |
| F19 RepresenterTheoremSolver | `optim.LBFGS` (`optim/gradient.go`) for non-quadratic loss | 30 |
| F20 ScalarOnFunctionFLR | F12 FPCA + `prob.LinearRegression` on scores | 20 |
| F25 FisherRaoSRSF | `optim.LBFGS` for Karcher-mean iteration + `calculus.SimpsonsRule` for SRSF integral `q(t) = ḟ(t)/√|ḟ(t)|` | 50 |
| F26 DTWFunctional | reuse pattern from `sequence/distance.go` Levenshtein-style DP — adapt to continuous cost | 20 |
| **Total reuse** | | **~520 LOC saved** |

---

## (4) Tier-1 keystone PR — 2,180 LOC delivers Ramsay-Silverman entry-level FDA in one shot

**PR-1: `fda/{basis,smooth,fpca,rkhs}` Tier-1** — F1, F2, F3, F4, F7, F8, F9, F12, F15, F16:

```
fda/
  basis/
    bspline.go      — F1+F3 (Cox-de-Boor recursion + closed-form derivative) ~360 LOC
    fourier.go      — F2 (orthonormal trig, FFT-accelerated on uniform grids) ~80 LOC
    gram.go         — F4 (Gram + penalty matrix via Gauss-Legendre) ~140 LOC
  smooth/
    spline.go       — F7 (smoothing spline, Reinsch 1967) ~200 LOC
    pspline.go      — F8 (P-spline, Eilers-Marx 1996) ~180 LOC
    gcv.go          — F9 (GCV, Craven-Wahba 1979) ~140 LOC
  fpca/
    dense.go        — F12 (dense FPCA via generalised eigenvalue) ~240 LOC
  rkhs/
    kmatrix.go      — F15 (kernel matrix construction reusing infogeo.Kernel) ~80 LOC
    krr.go          — F16 (kernel ridge regression) ~120 LOC
```

**Test budget per CLAUDE.md golden-file rules:** 26 primitives × 25 vectors = **650 vectors** mandatory. Reference implementations: R `fda` package + Python `scikit-fda` + R `mgcv` for cross-validation.

**Tolerance grid (per CLAUDE.md "per-function tolerance"):**
- F1 BSplineBasis: 1e-13 (algebraic Cox-de-Boor; integer-arithmetic-free)
- F2 FourierBasis: 1e-14 (just `cos`/`sin`/`sqrt` — IEEE-754-faithful)
- F4 BSplineGramMatrix: 1e-9 (Gauss-Legendre quadrature accumulation)
- F7 SmoothingSpline: 1e-8 (Cholesky factorisation accumulation, λ-conditioning sensitive)
- F8 PSpline: 1e-8 (same)
- F9 GCV: 1e-7 (extra trace-of-hat-matrix accumulation)
- F12 DenseFPCA: 1e-7 (covariance-eigenvalue accumulation, sign-ambiguity in eigenvectors mandates absolute-value comparison)
- F16 KernelRidgeRegression: 1e-9 (Cholesky-only)

---

## (5) Specific gotchas / numerical-precision pitfalls

1. **B-spline Cox-de-Boor singular-knot issue.** When `t = knots[i]` exactly, the recurrence has `0/0` divisions on the boundary. Standard fix: define `0/0 := 0` in the recursion. Test with golden vectors at every knot.
2. **GCV trace-of-hat-matrix.** Naive `tr(H_λ) = tr(B (B^T B + λ R)^{-1} B^T)` is `O(n²)` to assemble explicitly. Wahba 1990 §4.4 derives `O(n)` formula via Cholesky pivot diagonal — implement this from the start.
3. **FPCA eigenvector sign ambiguity.** Standard convention: choose sign such that `max_t |φ_k(t)|` occurs at `φ_k > 0`. Without this, golden-file comparison fails non-deterministically across random eigensolver starting vectors. (Existing `linalg.PCA` does NOT enforce this — pre-existing bug, surface in slot 097.)
4. **Generalised eigenvalue `W Σ W v = λ W v` reduction.** Standard reduction: factor `W = L L^T` (Cholesky), substitute `u = L^T v`, solve standard eigenvalue `L^{-1} Σ L^{-T} u = λ u`. Reuses existing `linalg.CholeskyDecompose` + `linalg.QRAlgorithm`.
5. **PACE bandwidth selection.** Yao-Müller-Wang 2005 use leave-one-curve-out CV — this is `O(n²)` smoothing operations; budget accordingly.
6. **Kernel ridge `(K + λI) α = y` ill-conditioning at small λ.** Cholesky fails when `λ → 0`. Fallback: SVD-based pseudo-inverse — gates on `linalg/SVD` (slot 097/215/227 demand). Until SVD lands, document `λ ≥ 1e-8 · trace(K)/n` lower bound.
7. **Fisher-Rao SRSF Karcher-mean convergence.** Iteration over warping group `Γ` is non-convex; documented sometimes-fails on highly-aligned data. Standard fix: 5 random restarts, return best.
8. **Wavelet boundary effects.** Mallat's pyramid algorithm requires periodic / symmetric / zero-padded boundary; document the choice (R `wavethresh` defaults to periodic).
9. **ESN spectral-radius scaling.** `W_reservoir` must be scaled to spectral radius `ρ < 1` for echo-state-property. Compute via `linalg.QRAlgorithm` largest-eigenvalue.

---

## (6) Provenance / external references

- Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis*, 2nd ed. Springer. (13,500+ citations.)
- Hsing, T., & Eubank, R. (2015). *Theoretical Foundations of Functional Data Analysis with an Introduction to Linear Operators*. Wiley.
- Kokoszka, P., & Reimherr, M. (2017). *Introduction to Functional Data Analysis*. CRC.
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.
- Craven, P., & Wahba, G. (1979). Smoothing noisy data with spline functions. *Numerische Mathematik*, 31(4), 377-403.
- Yao, F., Müller, H.-G., & Wang, J.-L. (2005). Functional data analysis for sparse longitudinal data. *JASA*, 100(470), 577-590.
- Kimeldorf, G., & Wahba, G. (1971). Some results on Tchebycheffian spline functions. *J. Math. Anal. Appl.*, 33(1), 82-95.
- Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Srivastava, A., & Klassen, E. P. (2016). *Functional and Shape Data Analysis*. Springer.
- Donoho, D. L., & Johnstone, I. M. (1995). Adapting to unknown smoothness via wavelet shrinkage. *Biometrika*, 81(3), 425-455.
- Mallat, S. G. (1989). A theory for multiresolution signal decomposition: the wavelet representation. *IEEE TPAMI*, 11(7), 674-693.
- López-Pintado, S., & Romo, J. (2009). On the concept of depth for functional data. *JASA*, 104(486), 718-734.
- Sun, Y., & Genton, M. G. (2011). Functional boxplots. *J. Comp. Graph. Stat.*, 20(2), 316-334.
- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. *GMD Technical Report 148*, Bonn.
- R packages: `fda` (Ramsay-Hooker-Graves), `mgcv` (Wood), `wavethresh` (Nason), `fdasrvf` (Tucker).
- Python packages: `scikit-fda`, `fdasrsf`.
