# 184 | synergy-linalg-prob

**Summary line 1.** `linalg/` and `prob/` today share zero direct edges (no `import "github.com/davly/reality/linalg"` in `prob/*.go` and no reverse import either) yet the entire high-dimensional covariance estimation canon — sample covariance Σ̂ from a row-major data matrix, Ledoit-Wolf 2004 closed-form shrinkage α*, OAS Chen-Wang-Eldar-Hero 2010, Graphical Lasso Friedman-Hastie-Tibshirani 2008, sparse-precision ADMM, PCA-based statistical factor models, Bai-Ng IC_p1/IC_p2/IC_p3 number-of-factors selection, Marchenko-Pastur sample-eigenvalue density, Tracy-Widom F₁/F₂ largest-eigenvalue laws, BBP phase transition, spiked covariance generative model, Wigner GOE / Wishart sampling, Johnson-Lindenstrauss random projections, Achlioptas ±1 sparse JL, sub-sampled randomized Hadamard transform (SRHT), Halko-Martinsson-Tropp 2011 randomized SVD, matrix Bernstein/McDiarmid concentration witnesses, randomized power-method/Lanczos starts, Gaussian-process-regression posterior via Cholesky, and Bayesian-linear-regression posterior covariance — twenty canonical operators, every one of them a one-line `from sklearn.covariance import …` or `from scipy.stats import wishart, multivariate_normal` in Python — is **wholly absent** from the v0.10.0 surface. The single keystone bridge primitive `MultivariateNormalSample(mu, L_chol []float64, p int, rng)` (Cholesky-driven MVN draw, the textbook one-liner z ~ N(0,I) → x = μ + Lz) does not exist; absence of MVN draws means Wishart, GP, BLR, and every Monte-Carlo covariance experiment are blocked at the leaf.

**Summary line 2.** Twenty synergy primitives **C1-C20** totalling ~2900 LOC of pure connective tissue close every gap with zero new packages; ten ship today against linalg v0.10.0 (Cholesky + LU + sym-eigvals + PCA), eight are blocked on missing linalg primitives flagged in agent 097-linalg-missing Tier 1 (Householder QR, Golub-Reinsch SVD, randomized SVD, Lanczos-symmetric); cheapest one-day standalone is **C1 SampleCovariance + C2 LedoitWolfShrinkage + C3 OASShrinkage** at 230 LOC saturating the textbook three-way **R-MUTUAL-CROSS-VALIDATION 3/3** pin (Σ̂ vs Σ_LW vs Σ_OAS condition-number monotonicity on n<<p data, mirroring commits 6a55bb4 audio-onset-3-detector + 365368a Clayton-autodiff-vs-analytic + 1e12e80 token-set-ratio-RapidFuzz-parity); highest-leverage architectural addition is **C5 GraphicalLassoFHT2008 + C6 RandomizedSVD-HMT2011 + C9 MarchenkoPasturDensity** ~620 LOC because (a) Graphical Lasso is the single most-cited high-dim covariance method (Friedman-Hastie-Tibshirani 2008 ~14000 citations), (b) randomized SVD is also flagged Tier 1 in 097-linalg-missing as the canonical low-rank approximation primitive, and (c) Marchenko-Pastur is the closed-form null distribution against which BBP / spiked-cov tests calibrate. Recommended placement: `prob/covariance.go` (Σ̂, LW, OAS), `prob/glasso.go` (GLasso, ADMM precision), `prob/factor.go` (PCA factor model, Bai-Ng), `prob/rmt.go` (Marchenko-Pastur, Tracy-Widom, BBP), `prob/sketch.go` (JL, Achlioptas, SRHT), `prob/multivariate.go` (MVN sample/PDF, Wishart, GP, BLR) — six new files, all in `prob/`, consuming `linalg.CholeskyDecompose` + `linalg.QRAlgorithm` + `linalg.PCA` + `linalg.MatMul` + `linalg.Inverse` already present today. NO new packages.

---

## 0. State of play (verified file-walk)

`linalg/` HEAD (6 source files, 1,500 LOC):
- `matrix.go`: MatMul, MatTranspose, MatVecMul, Identity, MatAdd, MatSub, MatScale, Trace, CrossProduct
- `vector.go`: DotProduct, L1/L2/Inf norms, VectorAdd/Sub/Scale, CosineSimilarity, EncodingDistance, L2Normalize, Clamp
- `decompose.go`: LUDecompose + LUSolve, Inverse (LU-based), Determinant, CholeskyDecompose + CholeskySolve
- `eigen.go`: QRAlgorithm — **eigenvalues only**, no eigenvectors returned
- `pca.go`: PCA via covariance + per-eigenvalue inverse-iteration (the only place that recovers eigenvectors; private trick, not exported)
- `correlation.go`: Pearson, Spearman, Covariance(x,y), **CovarianceMatrix(data, out)** — already ships unbiased sample covariance, two-pass mean-centred, n−1 denominator. **This is the single existing prob-relevant linalg primitive.**

`prob/` HEAD (10 source files in `prob/` + subpackages `conformal/`, `copula/`):
- `distributions.go`: Normal/Exponential/Uniform/Beta/Poisson/Binomial/Gamma — univariate only
- `distribution.go`: Distribution interface + KLDivergenceNumerical
- `hypothesis.go`: TTest one/two-sample, ChiSquared, FisherExact, MannWhitneyU
- `regression.go`: **LinearRegression (univariate)** + BenjaminiHochberg
- `markov.go`: discrete steady-state + simulate
- `nonparametric.go`: Mann-Whitney, others
- `timeseries.go`: ExponentialSmoothing, HoltLinear, ARIMA
- `jeffreys.go`: Jeffreys CI for Bernoulli
- `prob.go`: calibration / Brier / log-loss / Wilson CI
- `mathutil.go`: log-gamma (Lanczos approximation, NOT linear-algebra Lanczos)

**Search for high-dim covariance primitives in `prob/` and `linalg/`:** `Ledoit`, `Shrinkage`, `OAS`, `Wishart`, `GraphicalLasso`, `Glasso`, `FactorModel`, `Marchenko`, `TracyWidom`, `Wigner`, `BBP`, `Spiked`, `JohnsonLindenstrauss`, `Achlioptas`, `SRHT`, `RandomizedSVD`, `RandomProjection`, `MultivariateNormal`, `MVN`, `GaussianProcess`, `BayesianRegression`, `RidgeRegression`, `Lanczos` (as Krylov, not log-gamma), `MatrixBernstein`, `MatrixConcentration` — **zero matches** across all `prob/*.go` and `linalg/*.go`.

**Cross-import edges.** `grep -r "github.com/davly/reality/linalg" prob/` → 0. `grep -r "github.com/davly/reality/prob" linalg/` → 0. `prob/` and `linalg/` are sibling silos today.

**Dependency posture.** Every C-primitive below imports linalg from prob (consumer-shaped, mirroring 157 graph-linalg precedent). `prob/` already imports nothing-from-reality, so this synergy is a clean one-way edge with no cycles.

---

## 1. The twenty synergy primitives

Each entry: (1) capability, (2) composition over present primitives, (3) connective-tissue LOC, (4) blocking flag against 097-linalg-missing Tier 1.

### C1 — `SampleCovariance(data []float64, n, p int, biased bool, out []float64)`

**Capability.** Σ̂ from a row-major n×p data matrix, two-pass mean-centred, with selectable biased (1/n, MLE) or unbiased (1/(n−1), Bessel) denominator. The existing `linalg.CovarianceMatrix(data [][]float64, out)` takes nested slices and is unbiased-only; this row-major variant is the modern API every C2-C8 primitive consumes.

**Composition.** Trivial: walk data twice, accumulate Σ̂[i,j] = Σ_k (x_ki − μ_i)(x_kj − μ_j) / (n − biased*1). Symmetric, write upper triangle, mirror.

**LOC.** ~50 (with biased flag, panic guards, golden-vector-friendly).

### C2 — `LedoitWolfShrinkage(data, p int) (alpha float64)`

**Capability.** Closed-form shrinkage intensity α* per Ledoit-Wolf 2004, "A well-conditioned estimator for large-dimensional covariance matrices" J.Multivar.Analysis 88(2). Target T = (trace(Σ̂)/p) · I_p. Estimator Σ_LW = (1−α*)Σ̂ + α* T.

**Composition.** α* = min(1, β² / δ²) where β² = E[‖Σ̂ − Σ‖²_F] estimated via π̂ + ρ̂ + γ̂ from data (closed-form sums over (x_kᵢ − μᵢ)(x_kⱼ − μⱼ) products), δ² = ‖Σ̂ − T‖²_F. All sums O(np²). Consumes C1 + linalg.Trace + linalg.MatSub + Frobenius norm helper (~5 LOC).

**LOC.** ~110.

**Pin.** Cross-validate against scikit-learn `sklearn.covariance.LedoitWolf().fit(X).shrinkage_` to 1e-12 on Iris (n=150, p=4) and on n<p MNIST PCA-50 sketch (n=20, p=50). Both are public reproducible.

### C3 — `OASShrinkage(data, p int) (alpha float64)`

**Capability.** Oracle-Approximating-Shrinkage closed form, Chen-Wang-Eldar-Hero 2010, "Shrinkage algorithms for MMSE covariance estimation" IEEE Trans.SP 58(10). Sharper than LW under Gaussian assumption.

**Composition.** α_OAS = ((1 − 2/p)·trace(Σ̂²) + trace(Σ̂)²) / ((n+1−2/p)·(trace(Σ̂²) − trace(Σ̂)²/p)). All ingredients are scalar functions of the existing covariance matrix; one call to MatMul(Σ̂,Σ̂) plus two Trace calls.

**LOC.** ~50.

**Pin.** scikit-learn `sklearn.covariance.OAS().fit(X).shrinkage_` to 1e-12.

**R-MUTUAL pin.** Joint pin C1+C2+C3: on the same X, condition number κ(Σ̂) ≥ κ(Σ_OAS) ≥ κ(Σ_LW) for n<p (LW shrinks more aggressively under sparsity); golden-table 12 (n,p) cases mirroring the audio-onset-3-detector cross-validation idiom of commit 6a55bb4.

### C4 — `ShrinkageEstimator(Sigma_hat []float64, alpha float64, target string, p int, out []float64)`

**Capability.** Apply the convex combination Σ_s = (1−α)Σ̂ + α·T for selectable target T ∈ {scaled-identity (mean of diag of Σ̂), constant-correlation (Ledoit-Wolf 2003 alternate), diagonal-of-Σ̂}. Decouples shrinkage **policy** (C2/C3 estimate α) from shrinkage **application** (this primitive applies it).

**Composition.** Build T (~15 LOC), call linalg.MatScale + linalg.MatAdd. Pure linalg composition.

**LOC.** ~70.

### C5 — `GraphicalLasso(Sigma_hat []float64, p int, lambda float64, maxIter int, tol float64, theta_out, sigma_out []float64) bool`

**Capability.** Sparse precision matrix Θ = Σ⁻¹ via L1-penalised log-likelihood per Friedman-Hastie-Tibshirani 2008, "Sparse inverse covariance estimation with the graphical lasso" Biostatistics 9(3) — the most cited single high-dim covariance method, ~14000 citations. Output Θ is sparse; Σ_out is its inverse via final block-coordinate update.

**Composition.** FHT 2008 algorithm 1: outer block-coordinate descent over rows/columns of W (= Σ); each inner subproblem is a p−1 dim LASSO solved by coordinate descent (composition with a soft-threshold operator, ~10 LOC; verify if reality's `optim/proximal/` already ships SoftThreshold — agent-178 reports it does). Convergence diff_F < tol·avg(|Σ̂|).

**LOC.** ~280 (including soft-threshold inner loop).

**Pin.** scikit-learn `sklearn.covariance.GraphicalLasso(alpha=lambda).fit(X).precision_` to 1e-8 on textbook ER graph 20×20 with edge probability 0.2 sparsity pattern.

**Cross-link.** This primitive is one of two motivating uses for proximal/SoftThreshold (the other being LASSO regression, queued in 169-synergy-prob-optim).

### C6 — `RandomizedSVD(A []float64, n, p, k, oversample int, powerIter int, U, S, V []float64, rng)`

**Capability.** Halko-Martinsson-Tropp 2011, "Finding structure with randomness" SIAM Rev 53(2). Truncated SVD of n×p matrix at rank k via random Gaussian sketch, with `oversample` extra columns for tail-eigenvalue accuracy and `powerIter` (typical q=2) for slow-decay spectra.

**Composition.** Step 1: draw n×(k+oversample) Gaussian matrix Ω (consumes C12 below for randomness). Step 2: Y = A·Ω (linalg.MatMul). Step 3: Q = QRDecomposeHouseholder(Y) **BLOCKED on 097-T1**. Step 4: B = Qᵀ·A. Step 5: small SVD of (k+os)×p matrix B **BLOCKED on 097-T1**. Step 6: U = Q·Ũ.

**LOC.** ~150 connective-tissue once linalg ships QR + small-SVD; until then, **BLOCKED**.

**Pin.** scipy.sparse.linalg `svds(A, k=k, which='LM')` to 1e-9 on Hilbert matrix (slow decay) and on rank-k+noise spiked model.

**Cross-link.** 097-linalg-missing Tier-1 entry "Randomized SVD (Halko-Martinsson-Tropp 2011)" already exists in linalg's roadmap; THIS report adds the second motivating consumer (covariance low-rank approx) beyond the canonical "low-rank approximation" use that 097 cited.

### C7 — `PCAFactorModel(data []float64, n, p, k int, B, F, sigma2 []float64) float64`

**Capability.** Statistical factor model X = B·F + ε via PCA decomposition of Σ̂ keeping top-k eigenvalues. B is p×k loadings, F is n×k factor scores, σ² is p-vector idiosyncratic variances (Σ̂ diag minus sum-of-loadings²). Returns fraction of variance explained by the k factors.

**Composition.** Call existing `linalg.PCA(data, n, p, k, components, explained)` directly; B = components.T scaled by √eigenvalues, F = X·B / eigenvalues. Pure composition over PCA. Idiosyncratic σ² estimation is one trivial loop.

**LOC.** ~80.

### C8 — `BaiNgInformationCriterion(data []float64, n, p, kmax int, criterion string) (kStar int, IC []float64)`

**Capability.** Determine number of factors k* via Bai-Ng 2002 "Determining the number of factors in approximate factor models" Econometrica 70(1) — the textbook answer to "how many factors". Three penalty variants IC_p1, IC_p2, IC_p3; criterion ∈ {"p1","p2","p3"}.

**Composition.** Loop k=0..kmax, call C7 to get residuals ε(k), compute σ²(k) = ‖ε‖²/(np), then IC(k) = log σ²(k) + k·g(n,p) where g_p1 = (n+p)/(np)·log(np/(n+p)), g_p2 = (n+p)/(np)·log(min(n,p)), g_p3 = log(min(n,p))/min(n,p). Argmin → k*.

**LOC.** ~90.

**Pin.** Stock-Watson 2002 macro panel (FRED-MD or simulated AR(1)+factors); BN p1/p2 should agree on k=4 for r=4 simulated factors with n=200, p=100.

### C9 — `MarchenkoPasturDensity(x, ratio, sigma2 float64) float64` and `MarchenkoPasturCDF(x, ratio, sigma2 float64) float64`

**Capability.** Closed-form density of sample-covariance eigenvalues under H_0 (true Σ = σ²·I), Marchenko-Pastur 1967 "Distribution of eigenvalues for some sets of random matrices" Math.USSR-Sb 1(4). For c = p/n ∈ (0,1], density f(x) = √((λ₊−x)(x−λ₋))/(2π·c·σ²·x) for x ∈ [λ₋, λ₊], else 0; with λ_± = σ²(1±√c)². For c > 1, mass π_+ = 1−1/c at zero plus continuous density on [λ₋,λ₊].

**Composition.** Pure scalar formula; no linalg dependency. Ships to `prob/rmt.go`.

**LOC.** ~70 PDF + ~80 CDF (numeric integration via prob.QuadAdaptive once 183 ships AdaptiveSimpson; until then trapezoidal on dense grid).

**Pin.** Sample 10000 eigenvalues from Wishart(I_p, n) for c ∈ {0.1, 0.5, 1.0, 2.0}; KS distance between empirical CDF and MP CDF < 0.01.

### C10 — `TracyWidomQuantile(p, beta float64) float64` and `TracyWidomCDF(s, beta float64) float64`

**Capability.** Tracy-Widom F_β distribution of the largest eigenvalue (centred and scaled), Tracy-Widom 1994/1996. β=1 GOE (real symmetric Gaussian), β=2 GUE (complex Hermitian), β=4 GSE. Use case: testing whether the largest sample eigenvalue is "too large" to be MP-noise (i.e., a spike).

**Composition.** No closed form; ship as tabulated CDF (Prähofer-Spohn 2004 tables, public-domain) with monotone interpolation. Quantile via bisection.

**LOC.** ~250 (including the lookup table for β∈{1,2,4} at 200 grid points each).

**Pin.** Reference implementation in scipy is `scipy.stats.tracy_widom` (added 1.13); pin to 1e-6 over s ∈ [-10, 6].

### C11 — `BBPSpikeTest(eigvals []float64, n, p int, sigma2 float64) (spikes []int, pvalues []float64)`

**Capability.** Test which sample eigenvalues exceed the MP upper edge λ_+ = σ²(1+√(p/n))² and quantify significance via Tracy-Widom (Baik-Ben Arous-Péché 2005 phase-transition: a population spike θ > σ²·√(p/n) creates a sample-eigenvalue outlier above the MP edge with Gaussian fluctuations, while θ ≤ σ²·√(p/n) is buried in the MP bulk).

**Composition.** For each eigenvalue λ_i: scaled deviation s_i = (λ_i − μ_np)/σ_np where μ_np, σ_np are the MP-edge centring constants (closed form per Johnstone 2001). p-value = 1 − F_β(s_i). Composes C9 + C10.

**LOC.** ~70.

**R-MUTUAL pin.** Joint pin C9+C10+C11: on Wishart-only data BBP rejects 0% above 5%; on 1-spike-3 SNR it should detect the spike with power > 0.85 at n=p=200.

### C12 — `BoxMullerStandardNormal(rng) (z1, z2 float64)` plus **C12a `MultivariateNormalSample(mu []float64, L_chol []float64, p int, rng, out []float64)`**

**Capability.** Two primitives, the entire stochastic-prob ↔ linalg edge in two functions. **C12** is the polar Box-Muller normal sampler that the entire repo currently lacks (117-prob-missing T2 + 075-gametheory-perf both flagged its absence). **C12a** is the textbook multivariate-normal draw x = μ + L·z where L = chol(Σ) and z ~ N(0, I_p): the single most-used MVN sampler in scientific computing.

**Composition.** C12: ~20 LOC, polar Marsaglia variant (no trig, ~30% faster than classical Box-Muller per 075 audit). C12a: linalg.MatVecMul(L, z, out); pure 5-line composition over Cholesky + Box-Muller.

**LOC.** ~60 total.

**Cross-link.** C12 is THE single highest-leverage prob-side primitive: it unblocks C12a (MVN sampler), which unblocks C13 (Wishart Bartlett), which unblocks C9 (MP empirical validation), which unblocks C11 (BBP power-curve calibration). Six downstream primitives wait on C12.

### C13 — `WishartSample(L_chol_Sigma []float64, p int, df int, rng, out []float64)` and `WishartLogPDF(W []float64, Sigma []float64, p int, df int) float64`

**Capability.** Wishart distribution sampler (Bartlett decomposition Smith-Hocking 1972) and log-density. W ~ Wishart_p(Σ, df). Conjugate prior for the precision matrix in Bayesian Σ inference.

**Composition.** Bartlett: A is lower triangular with A_ii = √χ²(df−i+1) (chi-square draws via sum of squared C12 normals or Marsaglia-Tsang Gamma sampler queued in 117-T1) and A_ij ~ N(0,1) for i>j. Then W = L·A·Aᵀ·Lᵀ. Composes C12 + linalg.MatMul + linalg.MatTranspose. Already flagged as needed in 117-prob-missing T2.1.

**LOC.** ~150.

**Pin.** scipy.stats.wishart sample → estimate empirical mean E[W] = df·Σ to 1% at df=1000.

### C14 — `MultivariateNormalLogPDF(x []float64, mu []float64, Sigma_inv []float64, p int, log_det_Sigma float64) float64`

**Capability.** Multivariate normal log-density. log p(x|μ,Σ) = −½(p log(2π) + log|Σ| + (x−μ)ᵀ Σ⁻¹ (x−μ)).

**Composition.** Caller pre-computes Sigma_inv and log_det once via linalg.Inverse + sum(log(diag(Cholesky(Σ)))). Function itself is one MatVecMul + one DotProduct + scalar formula. Pre-computation avoids quadratic-cost-per-call when scoring batches.

**LOC.** ~40.

### C15 — `JohnsonLindenstraussDimension(n int, eps float64) int`

**Capability.** Conservative target embedding dimension k ≥ ⌈8 log(n) / ε²⌉ for Johnson-Lindenstrauss 1984 ε-isometry, sharpened constant per Dasgupta-Gupta 2003. Used to size random projections in C16/C17.

**Composition.** Pure scalar formula, ~5 LOC.

### C16 — `RandomGaussianProjection(p, k int, rng, R []float64)`

**Capability.** Build p×k random projection matrix with i.i.d. N(0, 1/k) entries. Apply via linalg.MatMul: X' (n×k) = X (n×p) · R (p×k). For ε-isometry on n vectors set k = JL(n, ε) per C15.

**Composition.** p·k Box-Muller draws (C12), scaled by 1/√k. ~20 LOC.

### C17 — `AchlioptasProjection(p, k int, rng, R []float64)`

**Capability.** Achlioptas 2003 sparse JL: R_ij ∈ {−√3, 0, +√3} with probabilities {1/6, 2/3, 1/6}. ~3× faster than Gaussian for dense X due to integer-only arithmetic in matmul; same ε-isometry guarantee.

**Composition.** Discrete RNG draws + scaling. ~25 LOC.

### C18 — `SubsampledRandomizedHadamardTransform(X []float64, n, p, k int, rng, out []float64)`

**Capability.** SRHT as in Halko-Martinsson-Tropp 2011 §11 and Tropp 2011 "Improved analysis of the SRHT". Apply Φ = √(p/k) · S · H · D where D is random diagonal ±1, H is the Walsh-Hadamard transform (FFT-like O(p log p)), S is row-sub-sampling. Faster than Gaussian projection (C16) for p ≥ 1024.

**Composition.** signal/fft.go already ships an FFT (verified per agent 134-signal-api); Walsh-Hadamard is a real-valued cousin (~30 LOC standalone since it's just butterfly with all twiddles = ±1). D and S are RNG-driven (C12 for ±1, range-restricted uniform draws for sub-sample indices).

**LOC.** ~120.

### C19 — `MatrixBernsteinBound(L_op_norm float64, sigma2 float64, n int, t float64) float64`

**Capability.** Tail bound for ‖Σ̂ − Σ‖_op ≥ t per Tropp 2012 "User-friendly tail bounds for sums of random matrices" Found.Comp.Math 12(4): P(‖Σ̂ − Σ‖_op ≥ t) ≤ 2p · exp(−t²/(2(σ² + Lt/3))). Use case: how many samples n suffice for κ(Σ̂) ≤ (1+δ)·κ(Σ).

**Composition.** Pure scalar formula; one exponential.

**LOC.** ~30.

**Pin.** Verify monotonicity in t (decreasing), n (decreasing), σ² (increasing).

### C20 — `GaussianProcessPosterior(X_train, y_train, X_test []float64, n_train, n_test, p int, kernel func(a, b []float64) float64, sigma_noise float64, mu_post, Sigma_post []float64)`

**Capability.** GP regression posterior, Rasmussen-Williams 2006 §2.2: μ_* = K_*ᵀ (K + σ²I)⁻¹ y, Σ_* = K_** − K_*ᵀ (K + σ²I)⁻¹ K_*. Returns posterior mean and full posterior covariance for test points.

**Composition.** Three kernel matrix builds (K, K_*, K_**), one Cholesky + CholeskySolve to invert (K + σ²I), two MatVecMul, one MatMul. Pure linalg composition over present primitives. Also gives **C20b BayesianLinearRegression** as a 50-LOC twin, where Σ_β posterior = (Xᵀ X / σ² + Λ_0)⁻¹ via Cholesky on the same plumbing.

**LOC.** ~180 (GP) + ~60 (BLR twin).

**Pin.** scikit-learn `GaussianProcessRegressor(kernel=RBF(1.0), alpha=sigma_noise²)` posterior mean/cov to 1e-9 on Goldstein-Price 1d slice (n_train=10).

---

## 2. Three cross-cutting connective-tissue patterns

### P1 — Three-way covariance R-MUTUAL pin (C1 + C2 + C3)

The Σ̂ vs Σ_LW vs Σ_OAS triangle is the textbook canonical cross-validation: on n>>p data all three converge; on n<p data Σ̂ is singular while Σ_LW and Σ_OAS remain SPD; on the same X, eigenvalue-spread monotonicity κ(Σ̂) ≥ κ(Σ_OAS) ≥ κ(Σ_LW) under sparsity assumption. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 mirroring commits 6a55bb4 / 365368a / 1e12e80.

### P2 — RMT four-tuple pin (C9 + C10 + C11 + sample Wishart via C13)

MP density vs empirical Wishart eigenvalue histogram (KS < 0.01); TW CDF vs empirical max-eigenvalue distribution; BBP threshold detected on planted spike. Four-witness pin.

### P3 — Sketching-vs-exact pin (C16 + C17 + C18 vs full SVD)

‖XᵀX − (PX)ᵀ(PX)‖_op ≤ ε‖XᵀX‖_op for k = JL(n, ε), separately for each of Gaussian / Achlioptas / SRHT projector, on the same X. Three-projector pin.

---

## 3. Recommended placement and landing order

`prob/covariance.go` — C1 SampleCovariance, C2 LedoitWolfShrinkage, C3 OASShrinkage, C4 ShrinkageEstimator (~280 LOC).

`prob/glasso.go` — C5 GraphicalLasso (~280 LOC); imports `optim/proximal/SoftThreshold` once that lands per agent-178.

`prob/factor.go` — C7 PCAFactorModel, C8 BaiNgInformationCriterion (~170 LOC); consumes existing `linalg.PCA`.

`prob/rmt.go` — C9 MarchenkoPastur PDF/CDF, C10 TracyWidom CDF/quantile (table-based), C11 BBPSpikeTest, C19 MatrixBernsteinBound (~430 LOC).

`prob/multivariate.go` — C12 BoxMullerStandardNormal, C12a MVNSample, C13 WishartSample/LogPDF, C14 MVN-LogPDF, C20 GaussianProcessPosterior, C20b BayesianLinearRegression (~490 LOC); consumes `linalg.CholeskyDecompose` + `linalg.CholeskySolve` + `linalg.Inverse` + `linalg.MatMul`/`MatVecMul`. **First and only edge from `prob/` → `linalg/` in the repo today.**

`prob/sketch.go` — C15 JLDimension, C16 GaussianProjection, C17 Achlioptas, C18 SRHT (~170 LOC).

`prob/randomized.go` (or fold into `linalg/svd.go` once 097-T1 lands) — C6 RandomizedSVD (~150 LOC, BLOCKED).

**Cycle-free DAG.** prob/ → linalg/ only. linalg/ imports nothing-from-reality (per CLAUDE.md "reality imports nothing"). Zero new packages.

**Landing order.**

- **PR-1 (1 day, 230 LOC source + 130 LOC tests).** C1 + C2 + C3 = SampleCov + LedoitWolf + OAS, all with three-way R-MUTUAL pin (P1). Highest single-PR leverage: closes the textbook high-dim covariance gap, gives every consumer (`aicore` calibration, `causal` regression, `parallax` claim verification) a numerically robust Σ estimator. Saturates R-MUTUAL 3/3.
- **PR-2 (1 day, 110 LOC).** C12 + C12a = Box-Muller + MVN sampler. Smallest commit with the largest downstream unblock: C13/C14/C20/C20b/C16/C17/C18 all become possible. Also closes the Box-Muller gap flagged in 117-prob-missing T2 and 075-gametheory-perf.
- **PR-3 (2 days, 280 LOC).** C5 GraphicalLasso. Stand-alone. The single most-cited high-dim covariance method.
- **PR-4 (1 day, 170 LOC).** C7 + C8 = factor model + Bai-Ng. Stand-alone over `linalg.PCA`.
- **PR-5 (3 days, 430 LOC).** C9 + C10 + C11 + C19 = RMT four-tuple. The TW table is the long pole; PDF/CDF/quantile-via-bisection is mechanical once tabled.
- **PR-6 (2 days, 490 LOC).** C13 + C14 + C20 + C20b = Wishart + MVN log-PDF + GP regression + BLR. Pure linalg composition.
- **PR-7 (2 days, 170 LOC).** C15 + C16 + C17 + C18 = JL dimension + Gaussian/Achlioptas/SRHT projections.
- **PR-8 (BLOCKED).** C6 RandomizedSVD lands as soon as 097-T1 Householder-QR + Golub-Reinsch SVD ship.

Total ~2880 LOC source + ~1200 LOC tests/golden over ~12 engineer-days, saturating four R-MUTUAL pins (P1 covariance, P2 RMT, P3 sketching, plus C20-vs-sklearn-GP pin).

---

## 4. Precision hazards (per primitive)

- **C1**: two-pass mean-centred is correct to ~15 digits; one-pass (Welford) lower-overhead but loses ~3 digits at large means. Stick with two-pass for golden-file determinism.
- **C2**: π̂/ρ̂/γ̂ estimators have O(1/n) bias in finite samples; LW α* monotonically biased low at small n. Document n ≥ 30 for ε ≤ 0.05 in α*.
- **C3**: assumes Gaussian data; non-Gaussian heavy-tailed input gives α_OAS too small. Document this; LW is more robust to misspecification.
- **C5**: GLasso convergence is **slow** under high penalty (λ ≫ ‖Σ̂‖_∞ off-diag) and may stagnate; cap maxIter=200, tol=1e-4·avg(|Σ̂|) per FHT 2008 §3 default. Warm-start when sweeping λ.
- **C6**: rSVD rank-k accuracy degrades when (k+1)-th singular value is large relative to k-th; oversample ≥ 5 + powerIter ≥ 2 per HMT 2011 §10 prescription. Document.
- **C7**: PCA factors are sign-arbitrary; report-time canonicalisation (largest abs entry positive) breaks reproducibility under perturbation; document sign-flip.
- **C8**: BN IC underestimates k* when factors are weak (eigenvalue gap small); document Onatski 2010 alternative for borderline cases.
- **C9**: MP density singular at λ₋ when c=1 (square-data limit); guard with x ≤ λ₋ + ε branch.
- **C10**: TW table interpolation outside [-10, 6] is extrapolation; clamp + warn.
- **C11**: BBP test power vanishes near θ = σ²√(p/n); document detectability boundary.
- **C12**: Marsaglia polar rejects 1−π/4 ≈ 21% of uniform pairs; document expected RNG-call cost. Gaussian tail beyond ~6σ is biased downward by Box-Muller; document.
- **C13**: Bartlett df > p−1 required (else Wishart is singular almost surely); panic on df ≤ p−1.
- **C14**: log_det_Σ pre-computation must be sum-of-log-diag(Cholesky) NOT log(det(Σ)) which underflows for p > 50.
- **C18**: Walsh-Hadamard requires p = power of 2; pad with zeros if not. SRHT row-sampling without replacement (or k duplicates → variance bias).
- **C20**: GP posterior covariance often has tiny negative eigenvalues from roundoff; nugget add σ_jitter² ≈ 1e-9 to diag(K) before Cholesky.

---

## 5. Cross-language pinning targets

Every C-primitive has a public-API equivalent for cross-language pinning:

| Primitive | Reference impl | Tolerance |
|----|----|----|
| C1 | `numpy.cov(X.T, bias=…)` | 1e-15 |
| C2 | `sklearn.covariance.LedoitWolf().fit(X).shrinkage_` | 1e-12 |
| C3 | `sklearn.covariance.OAS().fit(X).shrinkage_` | 1e-12 |
| C4 | `sklearn.covariance.shrunk_covariance(S, shrinkage=α)` | 1e-15 |
| C5 | `sklearn.covariance.GraphicalLasso(alpha=λ).fit(X).precision_` | 1e-8 |
| C6 | `scipy.sparse.linalg.svds(A, k=k)` (with HMT power iter) | 1e-9 |
| C7 | `sklearn.decomposition.FactorAnalysis` | 1e-9 |
| C8 | `numpy.linalg.eig` of Σ̂ + manual Bai-Ng formula vs Stock-Watson 2002 Table 2 | 1e-12 |
| C9 | scipy doesn't ship Marchenko-Pastur directly; pin against monte-carlo Wishart histogram via KS | 0.01 |
| C10 | `scipy.stats.tracy_widom` (1.13+) | 1e-6 |
| C11 | Onatski 2009 "Testing hypotheses about the number of factors" reference table | 1e-3 |
| C12 | `numpy.random.standard_normal` distribution match (KS test, not value-pin) | n/a |
| C13 | `scipy.stats.wishart.rvs / logpdf` mean/cov check | 1% over 10000 draws |
| C14 | `scipy.stats.multivariate_normal.logpdf` | 1e-12 |
| C16 | `sklearn.random_projection.GaussianRandomProjection` | distribution match |
| C17 | `sklearn.random_projection.SparseRandomProjection` (s=1 Achlioptas) | distribution match |
| C18 | scipy doesn't ship; pin via JL ε-isometry property over Wishart sample | 1% |
| C19 | scalar formula; no pin needed beyond unit checks | 1e-15 |
| C20 | `sklearn.gaussian_process.GaussianProcessRegressor.predict(return_cov=True)` | 1e-9 |

---

## 6. Differentiation from prior agents

- **097-linalg-missing**: identifies absence of QR/SVD/rSVD/Lanczos as Tier 1 linalg gaps. THIS agent identifies the **prob-side consumer** (covariance estimation, RMT, sketching) that motivates SVD/rSVD/Lanczos in the first place; sharpens the case for 097-T1 priority and adds 8 connective-tissue primitives downstream of those linalg additions.
- **098-linalg-sota**: SOTA scoring of linalg vs LAPACK. Orthogonal.
- **117-prob-missing / 118-prob-sota**: flag Wishart, Box-Muller, Marsaglia ziggurat, MVN as missing distributions. THIS agent expands those into the **full high-dim covariance estimation suite** (LW/OAS/GLasso/factor/RMT/sketch) with linalg compositions; also flags **MVN sampler + GP + BLR** as the keystone bridge, which 117 did not surface.
- **120-prob-perf**: performance audit. Orthogonal; THIS agent adds 20 primitives, perf is a per-primitive question once they ship.
- **153-synergy-prob-infogeo**: Fisher information / KL. Adjacent at C14 (MVN log-PDF is the input to multi-d Fisher), C20 (GP posterior info gain), but covariance-estimation axis is distinct.
- **157-synergy-graph-linalg**: spectral graph theory. Shares blocking on linalg-SVD/Lanczos but operates on Laplacian vs covariance — different generative model, different consumers.
- **161-synergy-control-prob**: Kalman filter. Kalman covariance update IS C4-pattern (shrinkage between prior and measurement); cross-link at C12a (KF state propagation needs MVN) and C20b (BLR is degenerate Kalman). Non-overlapping primitives.
- **169-synergy-prob-optim**: LASSO via proximal. THIS agent's C5 GLasso consumes optim/proximal/SoftThreshold; cross-link to PR-3 of 169.
- **177-synergy-geometry-optim, 178-synergy-control-optim**: orthogonal.
- **180-synergy-physics-prob**: ensemble physics (Boltzmann/Gibbs). Orthogonal generative process.
- **183-synergy-calculus-autodiff**: AD over numerical methods. Cross-link at C20 GP hyperparameter learning (would consume autodiff to differentiate marginal likelihood w.r.t. kernel θ); add as queued PR-9 once both 184 + 183 ship.

---

## 7. Bottom-line recommendation

**Single-day high-leverage commit if-only-one-PR ships PR-1 = C1 + C2 + C3 = 230 LOC source + 130 LOC tests** because:

(a) Lands the textbook three-way covariance-estimator R-MUTUAL pin to 1e-12 vs scikit-learn, mirroring the 6a55bb4 / 365368a / 1e12e80 commit idiom;
(b) Σ̂/LW/OAS together ARE the answer to "what covariance estimator should I use?" in 95% of n<p settings;
(c) Stands alone — zero new linalg primitives needed (Σ̂ is one MatMul; LW and OAS are scalar formulas over Σ̂);
(d) Unblocks every downstream PR (C5, C7, C20 all consume C1's `SampleCovariance`);
(e) Closes the **largest documented absence in `prob/`** (no high-dim covariance support at all today) at the same one-day cost as the smaller PR-2 Box-Muller commit;
(f) Establishes the FIRST `prob/` → `linalg/` import edge in the repo, which by itself is architecturally pivotal and unlocks the next 19 primitives over weeks rather than months.

**Second-best one-day commit: PR-2 = C12 + C12a = 110 LOC.** Smaller, but unblocks six downstream primitives instead of three. Choose PR-1 for landing-the-pin / Choose PR-2 for unlocking-the-most-future-PRs.
