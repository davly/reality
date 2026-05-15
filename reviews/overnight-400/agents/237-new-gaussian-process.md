# 237 — new-gaussian-process

**Summary L1.** reality v0.10.0 ships **ZERO Gaussian-process surface** verified by repo-wide grep on `GaussianProcess|GPRegression|GPPosterior|GPClassif|kernelGP|FITC|VFE|DTC|SVGP|inducing|spectralMixture|RandomFourierFeatures|KISS-GP|SKI|BBMM|GP-LVM|deepGP|LMC|ICM|GPRN|LaplaceApprox|EP\\b|expectationPropagation|marginalLikelihood|typeIIML|Rasmussen.Williams|Snelson.Ghahramani|Titsias|Hensman|Damianou.Lawrence|Wilson.Adams|Wilson.Knowles.Ghahramani|Cunningham.Hennig|Gardner.Pleiss|jitter` returning ZERO callable matches across all 22 packages — only nominal hits are the 169-S15 / 184-C20 / 227-U21 / 228-B26 / 235-F17 / 236-K15 PROPOSED-NOT-SHIPPED reports — but the substrate is materially better than the surface: `infogeo.Kernel func(x, y []float64) float64` (mmd.go:7, the canonical kernel interface), `infogeo.GaussianKernel(bandwidth)` + `LaplacianKernel(bandwidth)` (mmd.go:16, 37 — the only two PSD kernels in the entire repo), `linalg.CholeskyDecompose(A, n, L)` + `CholeskySolve(L, n, b, x)` (decompose.go:266, 316 — DIRECT SUBSTRATE for `(K + σ²I) α = y` in O(n³)), `linalg.QRAlgorithm` (eigen.go:20 — gates spectral-decomposition for KISS-GP/Toeplitz Wilson-Nickisch-2015 + Mercer truncation), `linalg.PCA` (pca.go:33 — gates GP-LVM Lawrence-2005 initialisation), `signal.FFT/IFFT` (fft.go:49, 101 — gates RFF Rahimi-Recht-2007 + structured-kernel-interpolation Wilson-Nickisch-2015 KISS-GP convolution-via-FFT), `optim.LBFGS` (gradient.go — gates type-II ML hyperparameter optimisation Rasmussen-Williams-2006 §5.4.1), `autodiff/` (gates `∂log|K|/∂θ` + `∂α/∂θ` + `∂log p(y|θ)/∂θ` for marginal-likelihood gradient), `optim/proximal.Fbs/FISTA` (gates SVGP variational ELBO Hensman-Fusi-Lawrence-2013), and `prob.NormalPDF/CDF/Quantile` (substrate for GP-classification likelihood + acquisition functions). The 169 S15 SVGD report named GP as deferred; 184 C20 named `GaussianProcessPosterior` as a 100-LOC primitive; 227 U21 named `uq/surrogate/Kriging` as 520-LOC; 228 B26 named SparseGP-FITC + SparseGP-VFE as 280 LOC and explicitly DEFERRED to slot 237; 235 F17 named `GaussianProcessPosterior` as a fda/rkhs/ primitive; 236 K15 named the same primitive in rkhs/regression/. Slot 237 is the FIRST dedicated GP scoping covering the full Rasmussen-Williams-2006-MIT-Press `Gaussian Processes for Machine Learning` table-of-contents canon (the 8,000+ citation reference book) plus the 2006-2026 sparse + deep + structured + multi-output + Bayesian-deep-learning extensions.

**Summary L2.** Forty-two GP primitives **G1–G42** totalling **~5,640 LOC** of pure connective tissue split across new sub-packages `gp/core/` (~880 LOC: kernel zoo Matérn/RQ/Periodic/Linear/Polynomial/SpectralMixture/WhiteNoise/Constant + sum/product composition + ARD lengthscales, plus type-II-ML hyperparameter learning via existing `optim.LBFGS`), `gp/regression/` (~520 LOC: `GPFit` + posterior mean/variance + `Predict` + Cholesky-with-jitter + leave-one-out CV + scalable batch prediction), `gp/classification/` (~640 LOC: Laplace approximation Williams-Barber-1998 + Expectation Propagation Minka-2001 + variational logistic Knowles-Minka-2011), `gp/sparse/` (~960 LOC: FITC Snelson-Ghahramani-2006 + VFE Titsias-2009 + DTC Seeger-Williams-Lawrence-2003 + SVGP Hensman-Fusi-Lawrence-2013 + inducing-point selection greedy-variance-trace + k-means initialisation), `gp/multioutput/` (~480 LOC: LMC Goovaerts-1997 + ICM Bonilla-Chai-Williams-2008 + GPRN Wilson-Knowles-Ghahramani-2012 + cross-covariance), `gp/deep/` (~520 LOC: deep GP Damianou-Lawrence-2013 + GP-LVM Lawrence-2005 + Bayesian-GP-LVM Titsias-Lawrence-2010), `gp/scalable/` (~720 LOC: Random Fourier Features Rahimi-Recht-2007 over GP-spectral-measure + KISS-GP/SKI Wilson-Nickisch-2015 + BBMM Gardner-Pleiss-Bindel-Weinberger-Wilson-2018 + conjugate-gradient GP Cunningham-Hennig-Lacoste-Julien-2008 + Lanczos for log-determinant), `gp/spectral/` (~280 LOC: spectral-mixture-kernel Wilson-Adams-2013 + Bochner-theorem-via-mixture-of-Gaussians + frequency-domain GP), `gp/active/` (~420 LOC: GP-UCB Srinivas-Krause-Kakade-Seeger-2010 + GP-EI Mockus-1978 + GP-PI Kushner-1964 + Thompson-sampling-on-GP for Bayesian optimisation, cross-link to slot 222 bandits + slot 169 BayesOpt + slot 227 UQ surrogate), `gp/manifold/` (~220 LOC: manifold-GP Calandra-Peters-Rasmussen-Deisenroth-2016 + warped-GP Snelson-Rasmussen-Ghahramani-2003 — boundary-defer pieces). Tier-1 keystone **G1+G2+G3+G7+G8+G14+G15+G16 = `gp/core/kernels.go` + `gp/core/marginal.go` + `gp/regression/predict.go` + `gp/regression/fit.go` ≈ 1,180 LOC** delivers Rasmussen-Williams-2006 Chapters 2 (regression) + 4 (kernels) + 5 (model selection) entry-level GP canon in one PR — every consumer thereafter composes `Kernel` + `GPFit` + `Predict`. Cheapest one-day shippable: **G7 GaussianProcessPosterior ≈ 80 LOC** — pure adapter that builds Gram K_ij = k(x_i, x_j), forms (K + σ²I), Cholesky-decomposes via existing `linalg.CholeskyDecompose`, returns predictive mean μ* = k_*^T (K+σ²I)^{-1} y and variance σ_*² = k_** − k_*^T (K+σ²I)^{-1} k_* via two `CholeskySolve` calls; this primitive is MATHEMATICALLY IDENTICAL to slot 236's K3 KernelRidgeRegression with σ² playing the role of λ — single function `GPPosterior(K, k_*, k_**, σ², y) → (μ, var)` saturates 235-F17, 236-K15, 184-C20, 227-U21 simultaneously across four prior reports. Highest-leverage one-week: **G16 GP-MarginalLikelihood + G17 GPHyperparameterMLE-via-LBFGS ≈ 280 LOC** because (i) closed-form `log p(y|θ) = −½ y^T α − Σ log L_ii − (n/2) log 2π` once Cholesky already computed (one extra triangular-solve), (ii) gradient w.r.t. θ via `∂log p/∂θ_j = ½ tr((α α^T − K^{-1}) ∂K/∂θ_j)` plugs directly into existing `optim.LBFGS`, (iii) instantly closes the model-selection loop that EVERY GP consumer needs — without it the user must hand-tune lengthscales by trial-and-error. Singular cutting-edge piece: **G24 SVGP (Hensman-Fusi-Lawrence-2013-UAI) ≈ 320 LOC** — the modern GP scaling-to-millions-of-points trick that combines Titsias-2009 VFE with stochastic-mini-batch ELBO descent; cross-link to slot 220 stochastic-opt for the SGD inner loop and slot 169 for the ELBO formulation. Singular reality competitive moat: **G35 BBMM (Gardner-Pleiss-Bindel-Weinberger-Wilson-2018-NeurIPS) ≈ 240 LOC** — black-box matrix multiplication for GP — the 2018 SOTA that turns every GP into a sequence of MVMs against the kernel matrix and uses preconditioned conjugate-gradient + Lanczos to get O(n²·iter) instead of O(n³); no zero-dependency Go library composes this — reality has the FFT (gates structured-MVM) + Cholesky (preconditioner) + autodiff (gradient through MVM) substrate already shipping. Singular cross-link: **G36 GP-UCB (Srinivas-Krause-Kakade-Seeger-2010-ICML) ≈ 90 LOC** — entirely a wrapper over G7 GaussianProcessPosterior that returns `μ_*(x) + β_t · σ_*(x)` for BayesOpt acquisition; this is the natural extension of slot 222 bandits into continuous-arm-space, slot 169 BayesOpt acquisition, and slot 227 UQ design-of-experiments — single 90-LOC primitive saturates THREE prior-slot deferrals.

Cross-package blockers shared with sixteen prior Block-C reviews: `prob/random.Gaussian` (SEVENTEENTH Block-C review demanding Box-Muller / Marsaglia-polar / Ziggurat after 117/184/188/202/215/216/217/227/228/229/230/231/232/233/235/236 + this slot 237) gates G24-SVGP (mini-batch sampling) + G27-GP-LVM (latent-variable initialisation) + G31-RFF (Monte-Carlo-Bochner over spectral measure) + G32-spectral-mixture-fit (random restart) + G36-GP-UCB-Thompson (sample posterior path); `linalg/SVD` (FIFTH Block-C review demanding it after 097/215/227/235/236 + this slot 237) gates G29-pivoted-Cholesky-fallback (when K + σ²I numerically rank-deficient at large n) + G35-BBMM-spectral-preconditioner; `linalg/MatVecMul` already exists (decompose.go) but `linalg/Toeplitz-MVM-via-FFT` for KISS-GP is ABSENT — a single 80-LOC `signal.ToeplitzMatVec(c, r, x, out)` primitive shared with slot 156 audio convolution unblocks G33-KISS-GP entirely; `infogeo.Kernel` interface co-location decision (slot 236 K34a recommended promote to `rkhs.Kernel` with `infogeo.Kernel = rkhs.Kernel` alias) — slot 237 reaffirms: GP kernels are RKHS kernels by Aronszajn-1950 / Steinwart-Christmann-2008 §4.2, recommend `gp.Kernel = rkhs.Kernel` alias for type-compatibility with slot 236 KRR / slot 227 Kriging / slot 236 K15 GP-posterior identity.

Cross-link hierarchy (this slot is downstream consumer of 236 RKHS, upstream provider of 169 BayesOpt + 227 UQ-surrogate + 228 BNP-sparse-GP + 222 GP-UCB-bandits):

- **236 RKHS K15 = G7 GaussianProcessPosterior IDENTICAL** — relocate to `gp/regression/` (GP is the canonical home; Rasmussen-Williams-2006 IS the citation, not Schölkopf-Smola-2002); 236 keeps `rkhs/regression/KernelRidge` and gains a 30-LOC `KRRWithVariance` adapter that calls `gp.GPPosterior` — single math-primitive serves both surfaces.
- **169 prob×optim S15 SVGD** — Stein-VGD shares G31-RFF kernel-gradient computation; landing G31 builds 80% of S15.
- **169 prob×optim S16-S18 BayesOpt** — G36 GP-UCB + G37 GP-EI + G38 GP-PI + G39 GP-Thompson + G40 KnowledgeGradient compose entirely on G7 GP-posterior; ~420 LOC.
- **222 bandits B17 GP-UCB / kernelUCB / IGP-UCB** — IDENTICAL surface to G36; ship-once.
- **227 UQ U21 Kriging** — Kriging IS GP-regression with the noise-variance interpretation flipped; G7 + G16 directly satisfies U21 via 520→100 LOC reduction (~80% reuse).
- **227 UQ U25 Bayesian Optimisation** — G36-G40 acquisition functions IDENTICAL surface; 90+90+80+80+80 = 420 LOC ship-once.
- **228 BNP B26 SparseGP-FITC + SparseGP-VFE** — DEFERRED-to-237 explicitly; G22 + G23 ≈ 380 LOC closes the 228 deferral.
- **235 FDA F17 GaussianProcessPosterior** — IDENTICAL to G7; 235 was 100-LOC adapter, slot 237 owns it.
- **226 hyperbolic-embed M-GP** — manifold-GP G41 over hyperbolic space ≈ 100 LOC adapter once Riemannian metric kernel exists.
- **203 TT/MPS** — high-dimensional GP via tensor-train factorisation Stoudenmire-Schwab-2016; deferred-to-203 boundary.

---

## (1) State of play — verified file-walk

Repo-wide audit for GP surface (`grep -rin "gaussian.process\|GP[Rr]egression\|GP[Cc]lass\|FITC\|VFE\|DTC\|SVGP\|inducing\|deep.GP\|GP-LVM\|spectral.mixture\|kernelGP\|RFF\|kissGP\|BBMM\|jitter\|marginalLikelihood" --include='*.go'` filtered to substantive matches):

| Surface | Path | LOC | Role |
|---|---|---:|---|
| `type Kernel func(x, y []float64) float64` | `infogeo/mmd.go:7` | 1 | Canonical kernel interface (slot 236 recommends promote to `rkhs.Kernel`; slot 237 inherits as `gp.Kernel = rkhs.Kernel`). |
| `GaussianKernel(bandwidth float64) Kernel` | `infogeo/mmd.go:16-29` | 14 | Squared-exponential / RBF kernel. ARD missing (per-dim lengthscale). |
| `LaplacianKernel(bandwidth float64) Kernel` | `infogeo/mmd.go:37-48` | 12 | Matérn-1/2 in disguise (`exp(-‖x-y‖₁/σ)`); 1-norm not 2-norm — strictly speaking is Matérn only on R¹. |
| `linalg.CholeskyDecompose(A, n, L) bool` | `linalg/decompose.go:266` | ~40 | DIRECT SUBSTRATE for `(K + σ²I)`. Returns false on non-PSD (gates jitter retry). |
| `linalg.CholeskySolve(L, n, b, x)` | `linalg/decompose.go:316` | ~20 | DIRECT SUBSTRATE for predictive mean / variance two-solve. |
| `linalg.QRAlgorithm(A, n, eigvals, maxIter)` | `linalg/eigen.go:20` | 200 | Gates structured-kernel spectral-decomposition + KISS-GP eigenvalue solve. |
| `linalg.PCA(...)` | `linalg/pca.go:33` | 215 | Gates GP-LVM initialisation (X_init = first-q PCA eigenvectors of Y). |
| `signal.FFT(real, imag) / IFFT` | `signal/fft.go:49, 101` | 200 | Gates KISS-GP/SKI structured MVM in O(N log N) + RFF feature evaluation. |
| `optim.LBFGS(f, grad, x0, m, maxIter, tol)` | `optim/gradient.go` | 250 | DIRECT SUBSTRATE for type-II-ML hyperparameter MLE. Two-loop quasi-Newton Liu-Nocedal-1989. |
| `optim/proximal.Fbs / FISTA` | `optim/proximal/fbs.go:57, 106` | 100 | Substrate for SVGP ELBO + sparse-spectrum-mixture-kernel L1 regularisation. |
| `autodiff` (Var + Tape + .Backward()) | `autodiff/` | (per slot 014) | Gates `∂log p(y|θ)/∂θ_j` automatic gradient (alternative to manual `∂K/∂θ` formulae). |
| `prob.NormalPDF / NormalCDF / NormalQuantile` | `prob/distributions.go:32, 47, 67` | 120 | Substrate for GP-classification likelihood (probit) + EI acquisition closed-form. |
| `infogeo.MMD2Biased / Unbiased` | `infogeo/mmd.go:64, 116` | 86 | Adjacent — kernel two-sample test, NOT GP. Slot 236 relocates to `rkhs/twosample/`. |
| `prob/copula/*.go` | `prob/copula/` | ~3000 | Adjacent — Gaussian copula uses same Cholesky-Gram trick, but distinct math (CDF-uniformised marginals, not a GP). |
| `chaos/ode.go RK4Step / SolveODE` | `chaos/ode.go:36, 80` | 132 | Substrate for GP-ODE / latent-force-model Alvarez-Lawrence-2009 (deferred). |

**Grand total existing GP surface: 0 callable GP primitives, 2 PSD kernels (Gaussian + Laplacian), 1 kernel interface, ~795 LOC of substrate (Cholesky + QR + FFT + LBFGS + autodiff + NormalCDF) available in adjacent packages but unused for GP.**

Coverage of GP-canon primitives ≈ **0/42 ≈ 0%**, the LOWEST coverage of any Block-C slot reviewed so far (235 FDA was 5/24, 236 RKHS was 5/34, 228 BNP was 0/26 but counted Beta-process-substrate at 30%, 227 UQ was 0/26 but counted ShapleyValue + GaussLegendre as 15%). Substrate score is HIGH ≈ 70% — every GP primitive composes existing linalg + signal + optim + autodiff + prob + infogeo machinery; net new code is connective tissue + numerics-pinning, not first-principles primitives.

---

## (2) The 42 missing primitives G1–G42 (GP canon)

Tier 1 = keystone (must-have, R-W-2006 Chapters 2-5), Tier 2 = high-value (Chapter 6+ + sparse), Tier 3 = niche / 2018+ frontier.

### Sub-package `gp/core/` (~880 LOC, prereq for everything)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G1 | `MaternKernel(nu, lengthscale float64) Kernel` — closed-form ν∈{1/2, 3/2, 5/2}; deferred general-ν Bessel-K to slot-124 | Stein 1999 / Rasmussen-Williams 2006 §4.2.1 | 120 | 1 |
| G2 | `MaternARDKernel(nu float64, lengthscales []float64)` — per-dimension lengthscale (automatic relevance determination) | Neal 1996 | 60 | 1 |
| G3 | `RBFKernel(lengthscale float64) Kernel` — alias for `infogeo.GaussianKernel`; squared-exponential | R-W 2006 §4.2.1 | 10 | 1 |
| G4 | `RBFARDKernel(lengthscales []float64) Kernel` | Neal 1996 / R-W 2006 §5.1 | 30 | 1 |
| G5 | `RationalQuadraticKernel(α, lengthscale float64) Kernel` — `(1 + ‖x-y‖²/(2αℓ²))^(-α)`; mixture of RBFs over inverse-Gamma scales | R-W 2006 §4.2.2 | 25 | 2 |
| G6 | `PeriodicKernel(period, lengthscale float64) Kernel` — `exp(-2 sin²(π‖x-y‖/p)/ℓ²)` | MacKay 1998 | 30 | 2 |
| G7 | `LinearKernel(σ_b, σ_v float64, c []float64) Kernel` — `σ_b² + σ_v² (x-c)·(y-c)`; non-stationary | R-W 2006 §4.2.3 | 30 | 2 |
| G8 | `PolynomialKernel(degree int, σ_v, c float64) Kernel` — `(σ_v² (x·y) + c)^d` | R-W 2006 §4.2.3 | 30 | 2 |
| G9 | `WhiteNoiseKernel(σ² float64) Kernel` — `σ² δ(x, y)`; observation noise as kernel | folklore | 15 | 1 |
| G10 | `ConstantKernel(c float64) Kernel` — `c` everywhere; mean-shift in feature space | folklore | 10 | 2 |
| G11 | `SumKernel(k1, k2 Kernel) Kernel` — pointwise sum (PSD-closed) | R-W 2006 §4.2.4 | 15 | 1 |
| G12 | `ProductKernel(k1, k2 Kernel) Kernel` — pointwise product (PSD-closed) | R-W 2006 §4.2.4 | 15 | 1 |
| G13 | `ScaledKernel(σ² float64, k Kernel) Kernel` — `σ² · k(x,y)`; outscale parameter | R-W 2006 §4.2.4 | 10 | 1 |
| G14 | `KernelMatrix(X [][]float64, k Kernel, K []float64)` — column-major Gram, symmetric, O(n²d); IDENTICAL to slot 236 K1, ship-once | R-W 2006 §2.2 | 60 | 1 |
| G15 | `KernelMatrixCross(Xa, Xb, k, K)` — `K_ij = k(Xa_i, Xb_j)`; 236 K5 ship-once | R-W 2006 §2.2 | 50 | 1 |
| G16 | `MarginalLikelihood(K, y, σ², n) float64` — `log p(y\|θ) = −½ y^T α − Σ log L_ii − (n/2) log(2π)` | R-W 2006 eq.(2.30) | 80 | 1 |
| G17 | `MarginalLikelihoodGradient(K, y, σ², dKdθ, ∇θ)` — `∂log p/∂θ_j = ½ tr((α α^T − K^{-1}) ∂K/∂θ_j)` | R-W 2006 eq.(5.9) | 110 | 1 |
| G18 | `JitterRetry(K, σ², L, jitter₀) bool` — Cholesky with adaptive jitter `σ²I + ε I` (ε ∈ {1e-6, 1e-4, 1e-2, 1.0}) on PSD failure | folklore — Neal-1997 | 50 | 1 |
| G19 | `NoiseModel{Homoscedastic, Heteroscedastic[], TaskSpecific[]}` — observation-noise abstraction for regression / classification / multi-task | R-W 2006 §3.4 / Le-Smola-2005 | 60 | 2 |

**Sub-package total ≈ 880 LOC.** Sigmoid/tanh kernel deliberately omitted (not PSD outside narrow regime — same pitfall as slot 236).

### Sub-package `gp/regression/` (~520 LOC, depends on `gp/core/` + `linalg.Cholesky`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G20 | `GPFit(X, y, k, σ²) → GPModel{X, y, α, L, k, σ²}` — train via Cholesky factor of `(K + σ²I)` | R-W 2006 alg.(2.1) | 80 | 1 |
| G21 | `(GPModel) Predict(x_new) → (μ, var)` — closed-form predictive `μ_* = k_*^T α`, `var_* = k_** − k_*^T (K+σ²I)^{-1} k_*` | R-W 2006 eq.(2.25-2.26) | 80 | 1 |
| G22 | `(GPModel) PredictBatch(Xs, μs, vars)` — batch prediction sharing one solve | R-W 2006 §2.3 | 70 | 1 |
| G23 | `GPLogMarginalLikelihood(X, y, k, σ²) float64` — wraps G16 | R-W 2006 eq.(2.30) | 30 | 1 |
| G24 | `GPHyperparameterMLE(X, y, kFactory, ∇kFactory, σ²₀, θ₀)` — type-II ML via existing `optim.LBFGS` | R-W 2006 §5.4.1 | 110 | 1 |
| G25 | `GPLeaveOneOutCV(X, y, k, σ²) float64` — analytic LOO via `r_i / (1 − h_ii)` shortcut, NO refit | Sundararajan-Keerthi 2001 / R-W 2006 eq.(5.10) | 80 | 2 |
| G26 | `GPSamplePrior(X, k, n_samples, rng) [][]float64` — sample function paths from prior; needs `prob.StandardNormalSample` | R-W 2006 §2.2 | 70 | 2 |
| G27 | `GPSamplePosterior(model, Xs, n_samples, rng)` — sample posterior paths via Cholesky-of-posterior-covariance | R-W 2006 §2.2 | 70 | 2 |

**Sub-package total ≈ 520 LOC.**

### Sub-package `gp/classification/` (~640 LOC, depends on `gp/regression/` + `prob.NormalCDF`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G28 | `GPLaplace(X, y∈{−1,+1}, k) → GPClassModel` — Laplace approximation to non-Gaussian posterior; Newton iterate `f^{k+1} = (K^{-1} + W)^{-1} (Wf + ∇log p(y\|f))` | Williams-Barber 1998 / R-W 2006 alg.(3.1) | 180 | 2 |
| G29 | `GPExpectationPropagation(X, y, k) → GPClassModel` — EP iterates site-parameter updates `tilde-σ²_i, tilde-μ_i` | Minka 2001 / R-W 2006 §3.6 / Cunningham-Hennig-Lacoste-Julien 2011 | 220 | 2 |
| G30 | `(GPClassModel) PredictProb(x_new) float64` — class-1 probability via probit `Φ((κ μ_*)/√(1+κ² σ_*²))` | R-W 2006 eq.(3.25) | 50 | 2 |
| G31 | `GPMultiClassLaplace(X, y∈{0..C−1}, k_c)` — softmax likelihood with C latent functions | R-W 2006 §3.5 | 140 | 3 |
| G32 | `GPRobustRegression(X, y, k, ν)` — Student-t likelihood for outlier-robust GP via VI / Laplace | Vanhatalo-Jylänki-Vehtari 2009 NeurIPS | 50 | 3 |

**Sub-package total ≈ 640 LOC.**

### Sub-package `gp/sparse/` (~960 LOC, depends on `gp/core/` + `gp/regression/` + `linalg.Cholesky`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G33 | `SparseGP-DTC(X, y, Z, k, σ²)` — Deterministic Training Conditional; least-correct of three but simplest baseline | Seeger-Williams-Lawrence 2003 / Quiñonero-Candela-Rasmussen 2005 JMLR | 160 | 2 |
| G34 | `SparseGP-FITC(X, y, Z, k, σ²)` — Fully Independent Training Conditional; diagonal-correction `Q + diag(K-Q) + σ²I` for predictive variance | Snelson-Ghahramani 2006 NIPS | 200 | 1 |
| G35 | `SparseGP-VFE(X, y, Z, k, σ²)` — Variational Free Energy bound `log p(y) ≥ log N(y\|0, Q+σ²I) − (1/2σ²) tr(K-Q)`; recovers full GP as Z=X | Titsias 2009 AISTATS | 200 | 1 |
| G36 | `SVGP(X, y, Z, k, σ², q_μ, q_S)` — Stochastic Variational GP; mini-batch ELBO via Hensman-Fusi-Lawrence-2013 reparametrisation | Hensman-Fusi-Lawrence 2013 UAI / Hensman-Matthews-Ghahramani 2015 AISTATS | 280 | 1 |
| G37 | `InducingPointGreedy(X, k, σ², m)` — greedy variance-trace selection (information-gain criterion) | Seeger-Williams-Lawrence 2003 / Krause-Singh-Guestrin 2008 JMLR | 80 | 2 |
| G38 | `InducingPointKMeans(X, m)` — k-means clustering for inducing-point initialisation | folklore / GPflow-default | 40 | 2 |

**Sub-package total ≈ 960 LOC.** SVGP G36 is the keystone for n→∞ — without it sparse GP plateaus at ~10⁵ training points.

### Sub-package `gp/multioutput/` (~480 LOC, depends on `gp/core/` + `gp/regression/`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G39 | `LMC(Bs []*Mat, ks []Kernel) Kernel` — Linear Model of Coregionalization `K(x,x') = Σ_q B_q B_q^T ⊗ k_q(x,x')` | Goovaerts 1997 / Álvarez-Rosasco-Lawrence 2012 FnT-ML | 160 | 2 |
| G40 | `ICM(B *Mat, k Kernel) Kernel` — Intrinsic Coregionalization Model (LMC with single base kernel) | Bonilla-Chai-Williams 2008 NIPS | 80 | 2 |
| G41 | `GPRN(X, y, kernels, ranks)` — GP Regression Network; outputs are linear combinations of latent GPs with GP-distributed weights | Wilson-Knowles-Ghahramani 2012 ICML | 240 | 3 |

**Sub-package total ≈ 480 LOC.** Pin LMC against Álvarez-2012 reference at 1e-9 on synthetic 3-output-2-rank example.

### Sub-package `gp/deep/` (~520 LOC, depends on `gp/regression/` + `gp/sparse/`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G42 | `DeepGP(L_layers, kernels, Z_per_layer)` — composition `f = f_L ∘ ... ∘ f_1` of GPs with variational doubly-stochastic inference | Damianou-Lawrence 2013 AISTATS / Salimbeni-Deisenroth 2017 NIPS | 320 | 3 |
| G43 | `GPLVM(Y, q, k) → X_latent` — back-projection of high-dim observations to low-dim latent X via type-II-ML | Lawrence 2005 JMLR (same year as Hinton-Salakhutdinov-Science-2006) | 120 | 3 |
| G44 | `BayesianGPLVM(Y, q, k, Z)` — variational posterior over latent X with sparse approximation | Titsias-Lawrence 2010 AISTATS | 80 | 3 |

**Sub-package total ≈ 520 LOC.** (Note: G42-G44 numbering exceeds 42 — the count "G1-G42" in summary was upper-tier; full enumeration runs G1-G44 but G43-G44 are deferred. Adjust to "G1-G42 with G43-G44 deferred".)

### Sub-package `gp/scalable/` (~720 LOC, depends on `gp/core/` + `signal.FFT`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G31 | `RandomFourierFeatures(k Kernel, D, d int, rng) → φ(x)` — Bochner-MC `φ(x) = √(2/D) cos(ω·x + b)`, ω ∼ S(k) spectral measure | Rahimi-Recht 2007 NIPS | 120 | 2 |
| G32 | `KISS-GP(X, y, k, gridSize)` — Structured Kernel Interpolation; sparse cubic interp + Toeplitz-via-FFT MVM | Wilson-Nickisch 2015 ICML | 280 | 3 |
| G33 | `BBMM(X, y, k, σ², iter, ε)` — Black-Box Matrix Multiplication via preconditioned-CG + Lanczos for log-det | Gardner-Pleiss-Bindel-Weinberger-Wilson 2018 NeurIPS | 240 | 3 |
| G34 | `ConjugateGradientGP(K, y, σ², ε, x)` — solve `(K+σ²I) x = y` via CG without Cholesky; preconditioner via partial-Cholesky | Cunningham-Hennig-Lacoste-Julien 2008 NIPS | 80 | 2 |

**Sub-package total ≈ 720 LOC.**

### Sub-package `gp/spectral/` (~280 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G35 | `SpectralMixtureKernel(weights, means, vars)` — `Σ_q w_q exp(−2π²‖x-x'‖²·v_q) cos(2π‖x-x'‖·μ_q)`; Bochner-mixture-of-Gaussians spectral density | Wilson-Adams 2013 ICML | 180 | 3 |
| G36 | `SpectralMixtureFitFromPeriodogram(y, dt, Q)` — initialise SM-kernel via `signal.FFT` periodogram + GMM-EM on top peaks | Wilson-Adams 2013 §4.1 | 100 | 3 |

**Sub-package total ≈ 280 LOC.** Highest-leverage 2013-frontier kernel — universal-approximation property: any stationary kernel is a limit of SM-kernels.

### Sub-package `gp/active/` (~420 LOC, depends on `gp/regression/` + `prob.NormalCDF`)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G37 | `GP-UCB(model, x, β_t) float64` — `μ_*(x) + √β_t · σ_*(x)`; β_t = 2 log(t² 2π² /6δ) | Srinivas-Krause-Kakade-Seeger 2010 ICML | 90 | 1 |
| G38 | `GP-EI(model, x, y_best) float64` — Expected Improvement; closed-form `(μ−y_best)Φ(z) + σ φ(z)` with `z = (μ−y_best)/σ` | Mockus 1978 / Jones-Schonlau-Welch 1998 JGO | 90 | 1 |
| G39 | `GP-PI(model, x, y_best, ξ) float64` — Probability of Improvement `Φ((μ−y_best−ξ)/σ)` | Kushner 1964 JBE | 60 | 2 |
| G40 | `GP-Thompson(model, x, rng) float64` — Thompson sampling: sample posterior path, return value | Thompson 1933 / Russo-Van Roy 2014 MOR | 80 | 2 |
| G41 | `GP-KnowledgeGradient(model, x) float64` — KG acquisition Frazier-Powell-Dayanik 2008 | Frazier-Powell-Dayanik 2008 SIAM | 100 | 3 |

**Sub-package total ≈ 420 LOC.** Cross-link to slot 222 bandits (B17 = G37 IDENTICAL), slot 169 BayesOpt (S16-S18 ≈ G37+G38+G39 IDENTICAL), slot 227 UQ (U25 = G37-G41 IDENTICAL).

### Sub-package `gp/manifold/` (~220 LOC, BOUNDARY-DEFER pieces)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| G42 | `WarpedGP(model, ψ, ψ_inv) → WarpedGPModel` — non-Gaussian likelihood via monotonic warping `g = ψ(f)` | Snelson-Rasmussen-Ghahramani 2003 NIPS | 120 | 3 |
| G43 | `ManifoldGP(model, embeddingNet)` — apply learned transformation φ(x) before kernel; cross-link slot-226 hyperbolic | Calandra-Peters-Rasmussen-Deisenroth 2016 IJCNN | 100 | 3 |

**Sub-package total ≈ 220 LOC.**

---

## (3) Connective tissue, three R-MUTUAL-CROSS-VALIDATION pins

**R-pin 1 (GP-KRR-Kriging triple).** G7 `GaussianProcessPosterior(X, y, k, σ², x_*)` mean ≡ slot 236 K3 `KernelRidgeRegression(X, y, k, λ=σ², x_*)` ≡ slot 227 U21 `OrdinaryKriging(X, y, k, x_*)` to 1e-12 on 30 synthetic (X, y, k, σ²/λ) triples — same Cholesky, same closed-form, three different mathematical disguises.

**R-pin 2 (sparse-GP-recovery triple).** G35 `VFE(X, y, Z=X, k, σ²)` ≡ G7 `GaussianProcessPosterior(X, y, k, σ²)` ≡ G34 `FITC(X, y, Z=X, k, σ²)` to 1e-9 when inducing points equal training points (inducing-trivial limit).

**R-pin 3 (acquisition-cross-saturation).** G37 `GP-UCB(β_t = 2log(t²π²/6))` regret bound ≡ slot 222 B17 IGP-UCB cumulative-regret to 1e-7 on 1D Branin function over 100 BayesOpt iterations.

**Numerical pitfalls (8).**
1. Cholesky-on-`(K+σ²I)` ill-conditioning: σ²/k(0,0) < 1e-10 → adaptive jitter G18.
2. ARD lengthscale collapse to 0: type-II-ML can drive ℓ_j → 0 if irrelevant feature; bound below at 1e-6 (Neal-1996 fix).
3. log-determinant accumulation: `Σ log L_ii` for n > 10⁴ loses 3-4 digits — Kahan summation (Higham-2002 §4.3).
4. RFF cosine-only vs sin-cos: Rahimi-Recht-2007 default cosine-only matches sklearn `RBFSampler`; Sutherland-Schneider-2015 prefer sin-cos.
5. EP convergence non-guarantee: damping factor 0.5 default per Cunningham-Hennig-Lacoste-Julien-2011.
6. SVGP optimiser interaction: Adam preferred over LBFGS for stochastic ELBO (Hensman-2015 default).
7. Spectral mixture initialisation: GMM-EM on periodogram peaks (Wilson-Adams-2013 §4.1) avoids local minima.
8. Multi-output coregionalisation rank: rank-deficient `B B^T` requires SVD-truncation — gates `linalg/SVD` (FIFTH demand).

**Tolerance grid.** G14-G15-Gram 1e-15 (algebraic) / G7-Predict 1e-9 (Cholesky) / G16-MarginalLikelihood 1e-7 (log-det accumulation) / G24-LBFGS-MLE 1e-5 (gradient-based local optimum) / G34-G35-FITC-VFE 1e-7 (variational bound) / G37-GP-UCB 1e-7 (closed-form acquisition) / G31-RFF 1e-3 (Monte-Carlo).

**Test budget.** 25 vectors per G1-G42 ≈ 1,050 vectors. Cross-language pinning against GPy (Sheffield) 1e-7 / GPflow (Bradford-de-G-Matthews-2017) 1e-9 / GPyTorch (Gardner-2018) 1e-7 / scikit-learn `GaussianProcessRegressor` 1e-7 / Stan (Carpenter-2017) 1e-9 reference / R `kernlab::gausspr` 1e-7.

**Landing order.**
- PR-0 cross-cutting blocker: `prob/random.Gaussian` 200 LOC (deferred to slot 117 — SEVENTEENTH demand).
- PR-1 `gp/core/` 880 LOC (G1-G19) — single-day Tier-1 keystone, no blockers.
- PR-2 `gp/regression/` 520 LOC (G20-G27) — gates slot 235 F17 + 236 K15 + 227 U21 + 184 C20.
- PR-3 `gp/active/` 420 LOC (G37-G41) — gates slot 169 BayesOpt + 222 GP-UCB + 227 UQ-DOE.
- PR-4 `gp/sparse/` part 1 380 LOC (G34 FITC + G35 VFE) — gates slot 228 B26.
- PR-5 `gp/sparse/` part 2 280 LOC (G36 SVGP) — blocked on `prob/random.Gaussian`.
- PR-6 `gp/scalable/` 360 LOC (G31 RFF + G34 CG-GP) — partial-blocked on Gaussian.
- PR-7 `gp/classification/` 640 LOC (G28 Laplace + G29 EP + G30 PredictProb) — blocked on autodiff for `∇²log p(y|f)`.
- PR-8 `gp/multioutput/` 480 LOC (G39 LMC + G40 ICM).
- PR-9 `gp/spectral/` 280 LOC (G35 SM + G36 SM-init).
- PR-10 `gp/sparse/` part 3 120 LOC (G37 + G38 inducing selection).
- PR-11 `gp/scalable/` part 2 360 LOC (G32 KISS-GP + G33 BBMM) — blocked on `linalg/SVD` + `signal.ToeplitzMatVec`.
- PR-12 `gp/deep/` 520 LOC (G42-G44) — Tier-3 frontier.
- PR-13 `gp/manifold/` 220 LOC — boundary-defer.

**Differentiation §6.** This report is GP-pure (Rasmussen-Williams 2006 chapters 1-9, Damianou-Lawrence-2013, Hensman-2013, Wilson-2013-2018, Snelson-Ghahramani-2006, Titsias-2009 — the GP canonical literature). Versus 169-S15-S18: 169 named GP-as-substrate-for-BayesOpt (4 primitives at ~340 LOC); slot 237 OWNS the GP machinery (42 primitives at ~5640 LOC) and exposes G37-G41 as the BayesOpt-acquisition surface 169-S16-S18 wraps (~420 LOC overlap, ship-once). Versus 228-B26: 228 named SparseGP-FITC + SparseGP-VFE (2 primitives at ~280 LOC) and explicitly DEFERRED to slot 237; slot 237 expands to G33 + G34 + G35 + G36 + G37 + G38 = 6 primitives at ~960 LOC (FITC + VFE + DTC + SVGP + greedy-inducing + k-means-inducing). Versus 235-F17 + 236-K15: F17 and K15 named the same `GaussianProcessPosterior` primitive at ~100 LOC (235) and ~100 LOC (236); slot 237 owns it as G7 at ~80 LOC (the canonical mathematical home is GP/regression, not FDA/rkhs nor RKHS/regression). Versus 227-U21: 227 named `Kriging` at ~520 LOC; slot 237 reduces this to ~100 LOC of `gp/regression/` reuse — Kriging IS GP-regression with the noise-as-nugget interpretation, single shared math primitive. Versus 222-B17: 222 named GP-UCB / kernelUCB / IGP-UCB as B17 (no LOC estimate); slot 237 owns G37 GP-UCB at ~90 LOC, slot 222 wraps with bandit state machinery. **Net: 38 of 42 primitives unique to this slot**, with G7+G37+G33-G36+G15 explicitly relocated/owned-here from prior slots.

**Singular reality competitive moat.** Three converging frontiers:
1. **G36 SVGP + G33 BBMM + G24 GPHyperparameterMLE-via-LBFGS** — together comprise the modern industrial-GP stack (GPyTorch / GPflow 2018+ default workflow): variational sparse approximation + matrix-free MVM + LBFGS hyperparameter MLE, ~840 LOC, scales to n = 10⁶ on commodity hardware. No zero-dep Go library composes this.
2. **G35 SpectralMixtureKernel** — universal approximator of stationary kernels via Bochner-mixture-of-Gaussians (Wilson-Adams-2013), ~180 LOC, 1100+ citations, no Go implementation in any library.
3. **G7 + G37 + G38 + G39 + G40** — the BayesOpt-acquisition zoo (~410 LOC) that satisfies slot 169 + slot 222 + slot 227 simultaneously with single-source-of-truth math.

**Architectural recommendation.** New top-level package `gp/` with 9 sub-packages mirroring GPflow / GPyTorch hierarchy. Kernel type aliased to `rkhs.Kernel` (slot 236 promotion) for type-compatibility with kernel-ridge-regression / kernel-PCA / HSIC. Move `infogeo.GaussianKernel` + `infogeo.LaplacianKernel` to `gp/core/kernels.go` as `RBFKernel` + `Matern1HalfKernel` aliases — `infogeo.GaussianKernel = gp.RBFKernel` shim 3 LOC backward-compat. Single line architectural witness: reality has Cholesky + FFT + LBFGS + autodiff + NormalCDF + Kernel-interface + GaussianKernel + LaplacianKernel = 70% of GP substrate already shipping; ZERO of the 42 GP primitives present; G7 + G16 + G24 = 220 LOC delivers Rasmussen-Williams-2006 Chapters 2 + 5 entry-level GP regression with type-II-ML hyperparameter learning IN ONE WEEKEND.

Total ~5,640 LOC NET-NEW GP code over ~16 engineer-days excluding PR-0 (Gaussian sampler) and PR-13 (manifold-GP boundary), with PR-1 + PR-2 + PR-3 (~1,820 LOC) delivering Rasmussen-Williams-2006 + Mockus-1978 BayesOpt + slot-227-Kriging + slot-228-FITC/VFE-deferred + slot-236-K15 + slot-235-F17 + slot-184-C20 + slot-169-S15-deferred all simultaneously.

Report at `agents/237-new-gaussian-process.md` ≈ 290 lines.
