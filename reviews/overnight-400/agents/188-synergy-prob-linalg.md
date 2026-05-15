# 188 | synergy-prob-linalg

**Summary line 1.** This is the INVERSE direction of agent 184 (linalg→prob): NOT "what does linalg enable in prob?" but "what does **prob enable in linalg**?" — i.e., what classical linear-algebra problems become **faster, cheaper, or feasible-for-the-first-time** when randomness is introduced. The textbook answer is the entire Halko-Martinsson-Tropp 2011 / Drineas-Mahoney / Tropp 2015 / Martinsson-Tropp 2020 / Cortinovis-Kressner / Meyer-Musco 2021 RandNLA canon — randomized SVD, randomized range-finder, sketched least-squares (Sarlós 2006), CUR decomposition, leverage-score column sampling (Drineas-Mahoney-Muthukrishnan 2008), Hutchinson trace estimator (Hutchinson 1990) and its modern Hutch++/XTrace variants (Meyer-Musco-Musco-Woodruff 2021), randomized log-determinant via Lanczos quadrature (Ubaru-Chen-Saad 2017), randomized matrix multiplication (Drineas-Kannan-Mahoney 2006), randomized Kaczmarz (Strohmer-Vershynin 2009), randomized coordinate descent (Nesterov 2012), sub-sampled randomized Hadamard transform (Tropp 2011), Achlioptas ±1 sparse JL (2003), Bayesian conjugate gradient (Cockayne-Oates-Sullivan-Girolami 2019) and probabilistic numerics generally (Hennig-Osborne-Girolami 2015 Proc.Roy.Soc.A) — twenty-five canonical operators that turn O(np²) into O(np·log·k) and unlock matrices that classical dense LAPACK literally cannot fit in memory. v0.10.0 ships **zero** of them: no rSVD, no Hutchinson, no Hutch++, no Krylov-Lanczos quadrature, no randomized matmul, no rKaczmarz, no rCD, no leverage-score sampling, no CUR, no sketched-LS, no probabilistic-numerics machinery — every randomized-NLA primitive is a one-line `numpy.random.randn @ A` or `scipy.sparse.linalg.svds` in Python, **wholly absent** from the v0.10.0 surface. The single missing keystone — `prob.StandardNormalSample(rng)` (the Box-Muller / Marsaglia polar drawing the standard-normal scalar that EVERY randomized-NLA algorithm starts from) — does not exist; absence of standard-normal draws blocks all 18 of D1-D18 below at the leaf.

**Summary line 2.** Twenty-five synergy primitives **D1-D25** totalling ~3120 LOC of pure connective tissue close every gap with zero new packages; 12 ship today against `linalg v0.10.0` (LU + Cholesky + sym-eigvals + MatMul + MatVec + DotProduct + Identity + correlation), 9 are blocked on the same Tier-1 linalg gaps that 097-linalg-missing flagged (Householder QR, Golub-Reinsch SVD, symmetric eigenvectors, Lanczos-symmetric-Krylov), 4 are blocked on the same prob-side keystone (D1 standard-normal sampler) that 184 flagged as the bridge primitive. Cheapest one-day standalone ships **D2 HutchinsonTraceEstimator + D3 HutchPlusPlus + D4 GirardHutchinsonAGirard variant** at 230 LOC saturating the textbook three-way **R-MUTUAL-CROSS-VALIDATION 3/3** pin (three trace estimators on the same matrix, monotone variance reduction, mirroring commits 6a55bb4 / 365368a / 1e12e80); highest-leverage architectural addition is **D7 RandomizedSVD-HMT2011 + D9 SketchedLeastSquares-Sarlos + D11 LeverageScoreSampling** ~580 LOC because (a) rSVD is THE single most-cited randomized-NLA primitive (HMT 2011 SIAM Rev, ~5000 citations; the same primitive 184 flagged as their C6), (b) sketched-LS turns linalg.LinearRegression into something that scales to n=10⁶ rows, (c) leverage scores are the principled-importance-sampling probability that all of CUR / column-subset / coreset construction reduce to. Recommended placement: `linalg/randomized.go` (D7 rSVD, D8 randRangeFinder, D9 sketchedLS, D10 randMatMul), `linalg/trace.go` (D2 Hutchinson, D3 Hutch++, D4 XTrace, D14 logdet-Lanczos), `linalg/krylov.go` (D5 power-iteration, D6 Lanczos, D17 randKaczmarz, D18 randCD), `linalg/sample.go` (D11 leverage scores, D12 CUR, D13 column-subset), `prob/sample.go` (D1 standard-normal, D19 Wishart, D20 random-orthogonal). NO new packages. Sole architectural decision: D2–D14 belong in `linalg/` (consumer of the prob-keystone D1) NOT in `prob/` (where 184 placed C6 rSVD); rationale below in §3. Five randomized-NLA primitives are **already pinned** for free against the existing `linalg.QRAlgorithm` exact-eigenvalue scaffolding — Hutchinson against `Trace`, rSVD against `QRAlgorithm`-eigvals-of-AAᵀ, log-det via Lanczos quadrature against `sum(log(diag(LU.U)))` — three R-MUTUAL pins for free.

---

## 0. State of play (verified file-walk)

`linalg/` HEAD (v0.10.0, 6 source files, ~1500 LOC) — verified file inventory:
- `matrix.go` (208 LOC): MatMul, MatTranspose, MatVecMul, Identity, MatAdd, MatSub, MatScale, Trace, CrossProduct
- `vector.go` (241 LOC): DotProduct, L1/L2/Inf norms, VectorAdd/Sub/Scale, CosineSimilarity, EncodingDistance, L2Normalize, Clamp
- `decompose.go` (345 LOC): LUDecompose+LUSolve, Inverse, Determinant, CholeskyDecompose+CholeskySolve
- `eigen.go` (212 LOC): QRAlgorithm — symmetric eigenvalues only via Householder-tridiag + tqli (no eigvecs returned)
- `pca.go` (215 LOC): PCA via covariance + per-eigenvalue inverse-iteration
- `correlation.go` (183 LOC): Pearson, Spearman, Covariance, CovarianceMatrix

`prob/` HEAD (10 files in `prob/` + `conformal/` + `copula/`):
- `distributions.go`: NormalPDF/CDF/Quantile, Exponential, Uniform, Beta, Poisson, Binomial, Gamma — ALL CDF/PDF, **no samplers**
- `prob.go`, `hypothesis.go`, `regression.go`, `markov.go`, `nonparametric.go`, `timeseries.go`, `jeffreys.go`, `mathutil.go` — none contains an RNG-driven sampler

**Search for randomized-NLA primitives:** `Randomized`, `rSVD`, `Sketch`, `JLDimension`, `Hadamard` (as transform, not Walsh-coding), `Achlioptas`, `Hutchinson`, `Hutch`, `XTrace`, `Lanczos` (as Krylov, not log-gamma — `prob/mathutil.go:37` is the ONLY hit and it's Lanczos coefficients for log-gamma, NOT the Krylov method), `Kaczmarz`, `LeverageScore`, `CURDecomp`, `LogDetEstimator`, `SketchedLS`, `Krylov`, `Arnoldi`, `RandomMatMul`, `RandomCol`, `BayesCG`, `ProbNumerics` — **zero matches** across `linalg/*.go` and `prob/*.go`.

**The single missing keystone — `StandardNormalSample(rng) float64`.** `prob/distributions.go` ships closed-form PDF/CDF/Quantile for Normal / Exponential / Uniform / Beta / Poisson / Binomial / Gamma but **zero samplers**. Inverse-CDF sampling z = NormalQuantile(U(0,1)) **technically works** at the 1e-9 precision of the Beasley-Springer-Moro tail in `standardNormalQuantile`, but is rejection-acceptance-sampling-style "wrong" because (a) NormalQuantile via standardNormalQuantile is a 12-coefficient rational approx with 1e-9 tail precision, NOT IEEE-deterministic, and (b) inverse-CDF requires `rand.Float64()`-uniform which itself is not in `prob/`. The textbook fast path is Box-Muller (or Marsaglia polar, ~30% faster with rejection) directly from two uniform draws. **THIS IS the keystone identical to agent 184's C12 — both reports converge on the same primitive.**

**Cross-import edges.** `grep -r "github.com/davly/reality/prob" linalg/ → 0`; `grep -r "github.com/davly/reality/linalg" prob/ → 0`. Same as 184. The recommendation here is **the opposite import direction** from 184's: D1 ships in `prob/` (already 184's recommendation) but D2-D18 ship in `linalg/` (not `prob/` as 184 inferred for randomized-NLA primitives), creating the FIRST `linalg/ → prob/` edge in the repo. The architectural distinction is justified in §3.

---

## 1. The twenty-five synergy primitives (prob → linalg direction)

Each entry: (1) capability, (2) composition over existing primitives, (3) connective-tissue LOC, (4) prob-side dependency, (5) blocking flag against 097-linalg-missing T1 if any.

### D1 — `StandardNormalSample(rng *rand.Rand) float64` and `StandardNormalBatch(n int, rng *rand.Rand, out []float64)`

**Capability.** Marsaglia polar variant of Box-Muller producing one (or `n`) i.i.d. standard normal scalars per call. The single primitive every randomized-NLA algorithm in §1 reduces to via "draw a Gaussian sketch matrix Ω" or "draw a Rademacher probe vector v".

**Composition.** Marsaglia polar: rejection-sample (u₁, u₂) ∈ [-1,1]² until s = u₁²+u₂² ∈ (0,1), then z₁ = u₁·√(−2·ln(s)/s), z₂ = u₂·√(−2·ln(s)/s). Two draws per pair, ~21% rejection rate (1 − π/4), no trig vs classical Box-Muller. **In `prob/sample.go`.**

**LOC.** ~50 (with batched variant + golden-mean/var pin).

**Pin.** KS test against `numpy.random.standard_normal` distribution match (not value-pin); empirical mean ≤ 1e-2, variance within 1e-2 over 10⁶ draws.

**Cross-link.** Identical to 184-C12. Both reports converge on this as THE single highest-leverage prob-side primitive. Once D1 ships, D5-D18 unblock. Already flagged absent in 117-prob-missing T2 + 075-gametheory-perf.

### D2 — `HutchinsonTraceEstimator(matvec func(x, y []float64), n, m int, rng) (trEst, stdErr float64)`

**Capability.** Estimate tr(A) for an n×n implicit matrix A via the Hutchinson 1990 formula tr(A) = E[vᵀAv] where v is a random ±1 Rademacher vector (variance-minimal among isotropic distributions per Avron-Toledo 2011). Returns mean over m probes plus standard error.

**Composition.** Loop m times: draw v ∈ {±1}ⁿ via D1 (Rademacher = sign(D1)) or via Bernoulli(0.5) from `prob/distributions.go:BinomialPMF`-adjacent draw, call matvec(v, Av), accumulate ⟨v, Av⟩. Pure scalar accumulation. Caller passes A as a closure (matvec func) so this works on **implicit** matrices — sparse, tensor-product, low-rank — not just dense `[]float64`. **In `linalg/trace.go`.**

**LOC.** ~80 (Rademacher variant + Gaussian variant + std-err).

**Pin.** Three-way R-MUTUAL: on a dense 100×100 SPD A, exact `linalg.Trace(A, n)` vs Hutchinson(A, m=10000) within 3·stdErr 99% of the time.

**Cross-link to D1.** Hutchinson with v ∈ {±1}ⁿ has variance 2·(‖A‖²_F − ‖diag(A)‖²); v ∈ N(0,Iₙ) has variance 2·‖A‖²_F. Rademacher is strictly lower variance; document this.

### D3 — `HutchPlusPlus(matvec func, n, m int, rng) (trEst, stdErr float64)`

**Capability.** Hutch++ from Meyer-Musco-Musco-Woodruff 2021 "Hutch++: Optimal Stochastic Trace Estimation" SODA. Combines a low-rank deflation (m/3 random projections + range-finder) with vanilla Hutchinson on the residual. Achieves O(m^{−1}) RMSE vs Hutchinson's O(m^{−1/2}) — a quadratic improvement.

**Composition.** Step 1: draw n×(m/3) Gaussian sketch S via D1. Step 2: Y = AS via 2·m/3 matvec calls. Step 3: Q = QR(Y) **BLOCKED on 097-T1 Householder QR**. Step 4: tr_low = tr(QᵀAQ) via m/3 explicit matvec + dot. Step 5: Hutchinson on (I − QQᵀ)A(I − QQᵀ) with remaining m/3 probes. Sum tr_low + tr_residual.

**LOC.** ~150 once QR ships; until then **BLOCKED**.

**Pin.** On a synthetic A = Σᵢσᵢuᵢuᵢᵀ with σᵢ = i^{−2} (fast-decay), Hutch++ achieves ε = 1e-3 with m = 100 vs Hutchinson m ≈ 10⁶ for same accuracy. Reference: Meyer et al. 2021 Fig 1.

**R-MUTUAL pin.** Joint pin D2+D3+D14: on the same A, three independent estimators of tr(log(A)) via {D2 Hutchinson on log via Padé, D3 Hutch++ on Padé, D14 Lanczos quadrature directly} — all should agree within combined std-errs. **Three witnesses on one quantity.**

### D4 — `XTrace(matvec func, n, m int, rng) (trEst, stdErr float64)`

**Capability.** XTrace from Epperly-Tropp-Webber 2024 "XTrace: Making the Most of Every Sample in Stochastic Trace Estimation". Variance-reduced Hutch++ via leave-one-out exchangeability — saves a factor of 4-10× over Hutch++ at the same m.

**Composition.** Maintain m sketches (Yᵢ = Asᵢ where sᵢ ~ N(0,Iₙ)); for each i, build leave-one-out range Q_{−i} from {Yⱼ}ⱼ≠ᵢ via QR; estimator combines diagonal of QᵀAQ with leave-one-out residuals. Composes D1 + Householder QR + matvec.

**LOC.** ~200 once QR ships; **BLOCKED on 097-T1**.

**Pin.** Reference Julia/Python `XTrace` implementation by the authors; pin RMSE ratio vs Hutch++ ≥ 4× on Hilbert(200) at m=20.

### D5 — `PowerIterationDominantEigvec(matvec func, n, maxIter int, tol float64, rng, eigvec []float64) (eigval float64, iters int)`

**Capability.** Top-eigenvalue / eigenvector of an implicit symmetric matrix via the von Mises power method, starting from a **random** unit vector x₀ ~ N(0, Iₙ)/‖·‖₂. The randomness ensures (with probability 1) that x₀ has nonzero overlap with the dominant eigenvector — the deterministic alternative (x₀ = e₁) fails when A·e₁ = 0.

**Composition.** Loop: x ← matvec(x); λ = ‖x‖₂; x ← x/λ; until |λ − λ_prev| < tol. Initial draw via D1 + L2Normalize from `linalg/vector.go:L2Normalize`. Pure existing-primitive composition.

**LOC.** ~80.

**Pin.** Sym 50×50 SPD; converge to within 1e-9 of `linalg.QRAlgorithm(A, ...)[0]` in <100 iters when ratio λ₁/λ₂ ≥ 1.5.

**Cross-link.** PCA in `linalg/pca.go` already uses inverse-iteration with deterministic start; that's the dual of this primitive (random start unblocks degenerate cases).

### D6 — `LanczosTridiag(matvec func, n, k int, rng) (alpha, beta []float64, Q []float64)`

**Capability.** Lanczos 1950 algorithm building a k-step Krylov-tridiagonal approximation T_k = QᵀAQ to a symmetric matrix, where Q is n×k orthonormal Krylov basis. **Random-start q₁ ~ N(0,Iₙ)/‖·‖₂** (Saad 2003 §6.6 — random start is variance-optimal for spectral-gap detection). Outputs Lanczos coefficients α, β.

**Composition.** Standard three-term recurrence: α_j = ⟨q_j, Aq_j⟩; r = Aq_j − α_j q_j − β_{j−1} q_{j−1}; β_j = ‖r‖₂; q_{j+1} = r/β_j. Selective reorthogonalization (Simon 1984) optional but recommended for k ≥ 30.

**LOC.** ~150 (with selective reorth).

**Pin.** Eigenvalues of T_k via existing `linalg.QRAlgorithm(T_k, k, ...)` (T_k is n=k symmetric tridiag) approximate top-k eigvals of A to 1e-6 by k = 2·(true_rank) per Kuczyński-Woźniakowski 1992. This pin **already works against existing `linalg.QRAlgorithm`** which is the linchpin: ZERO new linalg primitives needed for the pin.

**Cross-link.** D6 unblocks D14 (log-det via Lanczos quadrature), D17 (rKaczmarz benefits from Lanczos preconditioner), D7 power-iteration sub-step.

### D7 — `RandomizedSVD(matvec func, matvecT func, m, n, k, oversample, powerIter int, rng) (U, S, V []float64)`

**Capability.** Halko-Martinsson-Tropp 2011 SIAM Rev 53(2) "Finding Structure with Randomness". Truncated SVD of an implicit m×n matrix at rank k via random Gaussian range-finder + small-matrix SVD on the projected sketch. Default oversample=10, powerIter=2 per HMT prescription.

**Composition.** Step 1: draw n×(k+os) Gaussian Ω via D1·(k+os)·n calls. Step 2: Y = AΩ via (k+os) matvec calls. Step 3: power-iter loop q times: Y ← A·(Aᵀ·Y) (4q matvec). Step 4: Q = QR(Y) **BLOCKED on 097-T1 Householder QR**. Step 5: B = QᵀA via (k+os) matvecT. Step 6: small SVD of (k+os)×n B **BLOCKED on 097-T1 Golub-Reinsch SVD**. Step 7: U = QŨ.

**LOC.** ~200 once QR + small-SVD ship; until then **BLOCKED**.

**Pin.** scipy.sparse.linalg `svds(A, k=k)` to 1e-9 on Hilbert(200) (slow decay) and on rank-k+noise spiked model. Reference: HMT 2011 §10 example matrices.

**Cross-link.** This is identical to 184-C6. Same dependency, same pin. **Resolution to placement-conflict: it goes in `linalg/randomized.go` not `prob/randomized.go`** because the consumer is "compute SVD of an arbitrary matrix" (a linalg operation that uses a probabilistic technique) NOT "estimate a covariance matrix" (a prob operation that needs SVD). Justification in §3.

### D8 — `RandomRangeFinder(matvec func, m, n, k, oversample, powerIter int, rng, Q []float64)`

**Capability.** Returns an m×(k+os) orthonormal Q such that ‖A − QQᵀA‖ is small; the prerequisite step of D7 made callable in isolation. HMT 2011 Algorithm 4.1 (basic) and 4.4 (subspace-iteration).

**Composition.** Steps 1-4 of D7. Standalone export so consumers (D9 sketched-LS, D11 leverage scores, D12 CUR) reuse without duplicating range-finding logic.

**LOC.** ~120 once QR ships; **BLOCKED on 097-T1**.

### D9 — `SketchedLeastSquares(matvec func, m, n, k int, b []float64, rng, x []float64) (residNorm float64)`

**Capability.** Sarlós 2006 "Improved approximation algorithms for large matrices via random projections" FOCS. Solve overdetermined least-squares min ‖Ax − b‖₂ for tall A (m ≫ n) by sketching: sample S ∈ ℝ^{k×m} with k = O(n log n / ε²); solve min ‖S(Ax − b)‖₂ instead. (1+ε)-approximate with prob ≥ 1−δ.

**Composition.** Two paths: (a) Gaussian sketch via D1 (m·k draws + k matvec); (b) SRHT via D15 below (faster O(mn log n) instead of O(mnk)). Solve sketched-LS on k×n by QR (via 097-T1) or normal equations through `linalg.CholeskyDecompose(SᵀS)`+`CholeskySolve` (already shipping; this is the path that lights up TODAY without 097-T1).

**LOC.** ~150.

**Pin.** Random m=10000, n=100 design matrix with planted x*; ε=0.01; sketched-x within 1e-2 relative error of `linalg.LUSolve(AᵀA, Aᵀb)`.

**Cross-link.** Reality has no "Householder-QR LS" today (097-T1) — sketched-LS is the cheapest legitimate large-scale LS path before that lands.

### D10 — `RandomizedMatMul(A, B []float64, m, k, n, c int, rng, out []float64) float64`

**Capability.** Drineas-Kannan-Mahoney 2006 "Fast Monte Carlo algorithms for matrices I". Approximate AB ≈ Σᵢ A_{:,iₜ} B_{iₜ,:} / (c · pᵢₜ) for c samples drawn with importance pᵢ = ‖A_{:,i}‖·‖B_{i,:}‖ / Σⱼ ‖A_{:,j}‖·‖B_{j,:}‖. Returns ‖AB − approx‖_F estimated bound.

**Composition.** Compute column norms of A + row norms of B (~mk + kn ops); normalize → discrete distribution; sample c indices via `prob.discreteSample` (cumulative inverse-CDF over uniform draws); accumulate rank-1 outer products. Composes D1 indirectly via uniform draws.

**LOC.** ~100.

**Pin.** Drineas-Kannan-Mahoney 2006 Theorem 1: error bound E[‖AB − approx‖_F²] ≤ (1/c)·‖A‖_F·‖B‖_F. Verify monotone decrease as c grows.

### D11 — `LeverageScores(A []float64, m, n int, k int, scores []float64)` (exact) and `LeverageScoresSketched(matvec, m, n, k int, rng, scores []float64)` (sketched)

**Capability.** Leverage scores ℓᵢ = ‖U_i,:‖² of the top-k left singular vectors U of A — the canonical importance-sampling probabilities for column / row sub-selection (Drineas-Mahoney-Muthukrishnan 2008 "Relative-error CUR matrix decomposition"). Sketched variant via D7 instead of exact SVD.

**Composition.** Exact: compute thin SVD via 097-T1 Golub-Reinsch (BLOCKED). Sketched: D7 → U-sketch; ℓᵢ = ‖U_i,:‖² via `linalg.DotProduct` per row. Sketched is provably (1±ε)-relative per Drineas-Magdon-Ismail-Mahoney-Woodruff 2012.

**LOC.** ~100 wrapper (exact + sketched).

**Pin.** Sum of leverage scores = k (exactly, by orthogonality); on rank-k matrix, all ℓᵢ ≤ 1.

### D12 — `CURDecomposition(A []float64, m, n, k int, rng, C, U, R []float64, colIdx, rowIdx []int)`

**Capability.** Mahoney-Drineas 2009 "CUR matrix decompositions for improved data analysis" PNAS. Approximate A ≈ CUR where C is k actual columns of A (chosen by leverage-score sampling), R is k actual rows, U = C⁺AR⁺. Interpretable ("real columns"), unlike SVD's abstract singular vectors.

**Composition.** D11 to get column / row scores; sample c columns ∝ scores into C and r rows into R; compute U via existing `linalg.Inverse(CᵀC)·CᵀA·Rᵀ·(RRᵀ)⁻¹` or via D9 sketched-LS twice. Pure D11 + linalg.Inverse + linalg.MatMul composition.

**LOC.** ~180.

**Pin.** Reference `pyCUR` package; pin Frobenius error ‖A − CUR‖_F to within (1+ε) of optimal-rank-k SVD error per Drineas-Mahoney-Muthukrishnan 2008 Theorem 1.

### D13 — `ColumnSubsetSelection(A []float64, m, n, k int, rng, idx []int)`

**Capability.** Pick k columns of A maximizing a determinantal / volume-sampling criterion (Deshpande-Rademacher-Vempala-Wang 2006). Two modes: (a) leverage-score-sampling (cheaper, additive guarantee); (b) volume-sampling via DPP rejection (slower, multiplicative guarantee).

**Composition.** Leverage mode: D11 + categorical sample of k indices without replacement. DPP mode: rejection-sample subsets ∝ det(A_S^TA_S); needs `linalg.Determinant` (already ships).

**LOC.** ~120.

### D14 — `LogDetEstimator(matvec func, n, m, k int, rng) (logDetEst, stdErr float64)`

**Capability.** Ubaru-Chen-Saad 2017 "Fast estimation of tr(f(A)) via stochastic Lanczos quadrature" SIMAX. Specialised to f(x) = log(x): produces tr(log(A)) = log det(A) for A SPD, in O(m·k·matvec) time without ever forming A or its Cholesky. Lifeline for n ≥ 10⁶.

**Composition.** For each of m random Rademacher probes v: run k-step Lanczos (D6) starting from v/‖v‖; eigendecomp the small k×k tridiagonal T_k via existing `linalg.QRAlgorithm` to get (θᵢ, τᵢ²) (eigval, first-component-squared); accumulate ‖v‖²·Σᵢ τᵢ²·log(θᵢ). Average over m probes.

**LOC.** ~120 once D6 ships. NO new linalg primitives needed because tridiagonal eigensolve uses existing `QRAlgorithm`.

**Pin.** SPD A 200×200 generated as LLᵀ; exact log det = 2·sum(log(diag(L))). Compare to D14 with m=50, k=30; agreement within 3·stdErr.

**Cross-link.** Mirrors how 184 placed log-det in MVN-LogPDF helper (C14): there it was sum-of-log-diag(Cholesky) at O(n³). Here it's matrix-free at O(n²·m·k) with random probes, suitable when A doesn't fit in memory or comes from a function (e.g., GP kernel matrix).

### D15 — `SubsampledRandomizedHadamardTransform(X []float64, m, n, k int, rng, out []float64)`

**Capability.** SRHT projection per HMT 2011 §11 / Tropp 2011 "Improved analysis of the SRHT" Adv.Adapt.Data.Anal 3(1-2). Apply Φ = √(m/k)·S·H·D with D random ±1 diagonal, H Walsh-Hadamard (real-valued cousin of FFT, butterfly with all twiddles ±1, O(m log m)), S row-sub-sampling. Faster than Gaussian projection for m ≥ 1024.

**Composition.** D ∈ {±1}^m via D1 (sign of). H is all-real butterfly: in `signal/fft.go` the butterfly structure exists for FFT but H is even simpler (no complex twiddle). S is sub-sample-without-replacement via uniform-discrete sampling.

**LOC.** ~120.

**Pin.** JL ε-isometry: ‖XᵀX − (ΦX)ᵀ(ΦX)‖_op ≤ ε‖XᵀX‖_op for k = ⌈c·log(m)/ε²⌉. Verify on random m=4096, n=100 design.

**Cross-link.** Identical to 184-C18. Same primitive, fits BOTH placements (sketching-for-covariance in `prob/sketch.go` per 184; sketching-for-LS in `linalg/sketch.go` per this report). Recommend export from `linalg/sketch.go` and have `prob/covariance.go` import it.

### D16 — `JohnsonLindenstraussDimension(n int, eps float64) int` and `RandomGaussianProjection(p, k int, rng, R)` and `AchlioptasProjection(p, k int, rng, R)`

**Capability.** Three primitives 184-C15/C16/C17 verbatim. Closed-form k ≥ ⌈8·log(n)/ε²⌉; Gaussian projection R ~ N(0, 1/k); Achlioptas 2003 sparse R ∈ {±√3, 0} with ⅙/⅔/⅙. **Identical to 184; same placement conflict resolution applies (linalg/sketch.go).**

**LOC.** ~50 total (5 + 20 + 25).

### D17 — `RandomizedKaczmarz(A []float64, m, n int, b []float64, maxIter int, rng, x []float64) (residNorm float64, iters int)`

**Capability.** Strohmer-Vershynin 2009 "A randomized Kaczmarz algorithm with exponential convergence" J.Fourier.Anal.Appl. 15(2). Solve consistent Ax = b iteratively by selecting row i ∝ ‖A_{i,:}‖² and projecting onto its hyperplane. Linear convergence rate (1 − 1/κ²(A)) per iteration in expectation. Memory O(n) — entire matrix never held.

**Composition.** Pre-compute row norms ‖A_{i,:}‖² via DotProduct loop; build cumulative weights for inverse-CDF discrete sampling. Per iter: sample i; compute (b_i − ⟨A_{i,:}, x⟩)/‖A_{i,:}‖²; update x ← x + factor·A_{i,:}.

**LOC.** ~100.

**Pin.** Strohmer-Vershynin Theorem 2: E[‖x_k − x*‖²] ≤ (1 − 1/κ_F²(A))^k · ‖x_0 − x*‖². Verify monotone decay on synthetic A with κ_F = 5.

**Cross-link.** rKaczmarz is the cheapest "iterative least-squares with no QR" option in v0.10.0 — directly competitive with D9 sketched-LS.

### D18 — `RandomizedCoordinateDescent(grad func(x, i int, n int) float64, n, maxIter int, stepSize float64, rng, x []float64)`

**Capability.** Nesterov 2012 "Efficiency of coordinate descent methods on huge-scale optimization problems" SIAM J.Optim 22(2). Optimise convex f(x) by per-iter sampling i ∈ {1,...,n} uniformly (or ∝ Lipschitz constants) and updating only x_i ← x_i − η·∂f/∂x_i. Bridges optim ↔ linalg via the special case f(x) = ½‖Ax−b‖² (rCD on quadratic = randomized matrix-vector iteration).

**Composition.** Loop maxIter: sample i; call gradᵢ; update x_i. ~50 LOC. Specialised quadratic version is 30 LOC more.

**LOC.** ~80.

**Cross-link.** Adjacent to 169-synergy-prob-optim (LASSO via proximal); rCD is the textbook proximal solver for separable problems.

### D19 — `WishartSample(L_chol_Sigma, p, df, rng, out)`

Wishart sampler via Bartlett (Smith-Hocking 1972). Identical to 184-C13. Listed in this report as the "random matrix functional" family member: trace functionals of random Wishart matrices are the null distribution against which D2/D3/D14 trace estimators calibrate. **LOC ~150.**

### D20 — `RandomOrthogonalMatrix(n, rng, Q)`

Sample Q uniform on O(n) (Haar measure). Stewart 1980: Q = sign-correction · QR(N), N i.i.d. N(0,1). Used to generate spectrum-controlled test matrices (Q·diag(σ)·Qᵀ). Mezzadri-2007 sign-correction CRITICAL. **LOC ~80, BLOCKED on 097-T1 QR.**

### D21 — `BayesianConjugateGradient(matvec, n, b, maxIter, tol, x, posterior_cov) iters`

Cockayne-Oates-Sullivan-Girolami 2019 Bayes.Anal 14(3). CG solver for SPD Ax=b that ALSO returns Gaussian posterior over x reflecting numerical uncertainty. Bridges classical-CG ↔ Bayesian-inference — cleanest example of probabilistic numerics. Composes existing `linalg.MatVecMul/DotProduct/Inverse` (small k×k). **LOC ~150.** First "prob-numerics" primitive; opens door to D22/D23. Hennig-Osborne-Girolami 2015 PRSA is the canonical survey.

### D22 — `BayesianQuadrature(f, n_evals, prior_kernel) (mean, var)`

Treat ∫f(x)dx as Bayesian inference under GP prior on f (O'Hagan 1991). Posterior mean (estimate) + variance (uncertainty). Composes 184-C20 GaussianProcessPosterior directly + closed-form integral of GP-mean. **LOC ~120, BLOCKED on 184-C20.**

### D23 — `ProbabilisticODESolver(rhs, t0, t1, y0, n, rng) (mean, var)`

Schober-Särkkä-Hennig 2014 Stat.Comput 27(1). Each RK step = Gaussian-filter update; output is GP posterior over trajectory. Composes existing `chaos.RK4` as deterministic-mean component. Soft placement: could live in `chaos/probabilistic.go`. **LOC ~200.**

### D24 — `HutchinsonForDiagonal(matvec, n, m, rng, diag_est, stdErr)`

Estimate diag(A) (not just trace) via Bekas-Kokiopoulou-Saad 2007: diag(A) ≈ (1/m)·Σ_t (v_t ⊙ Av_t) Rademacher v_t. GP marginal variance use case. **LOC ~70.**

### D25 — `RandomMatrixSpectralDensity(matvec func, n, m, k_lanczos int, n_grid int, rng) (xGrid, density []float64)`

**Capability.** Lin-Saad-Yang 2016 "Approximating spectral densities of large matrices" SIAM Rev 58(1). Estimate spectral density ρ(x) = (1/n)·Σᵢ δ(x − λᵢ) by combining Lanczos quadrature (D14 machinery) with kernel smoothing. Useful for understanding spectrum of huge matrices without ever forming them.

**Composition.** D6 Lanczos (BLOCKED on D6) m times with random starts; accumulate Gaussian-smoothed Ritz values weighted by τ_i² first-component coefficients.

**LOC.** ~150.

---

## 2. Three cross-cutting connective-tissue patterns

### P1 — Three-way trace estimator R-MUTUAL pin (D2 + D3 + D14)

Three independent trace-estimation algorithms on the same SPD A: D2 Hutchinson on log(A) via Padé+`linalg.LUDecompose`, D3 Hutch++ on Padé-log(A), D14 Lanczos quadrature directly on log(x). All three estimate log det(A). Compare to ground-truth `2·sum(log(diag(L)))` from `linalg.CholeskyDecompose(A)`. **Four witnesses on one quantity.** Saturates R-MUTUAL-CROSS-VALIDATION 4/4. Stronger than the 3/3 templates in commits 6a55bb4 / 365368a / 1e12e80 — log-det specifically allows four independent algorithms.

### P2 — Three-projector sketching pin (D15 + D16-Gaussian + D16-Achlioptas)

Same 184-P3. Fix m=10000, n=100, ε=0.05; build sketching matrices via SRHT (D15), Gaussian (D16-G), Achlioptas (D16-A); verify ‖XᵀX − ΦXᵀΦX‖_op ≤ ε‖XᵀX‖_op. Three independent sketching distributions; same JL ε-isometry guarantee.

### P3 — Two-solver iterative-LS pin (D9 sketched-LS + D17 randomized-Kaczmarz)

Solve same overdetermined Ax=b via D9 (sketch-once-then-solve) and D17 (one-row-at-a-time). Both should converge to within 1e-3 of `linalg.LUSolve(AᵀA, Aᵀb)` exact normal-equations solution. Two independent randomized-LS algorithms agree on classical ground truth.

---

## 3. Architectural placement — `linalg/` not `prob/` (the disagreement with 184)

184 places C6 RandomizedSVD in `prob/randomized.go`. **This report disagrees** and places D7 rSVD in `linalg/randomized.go`. Justification:

1. **Caller intent.** The natural caller of rSVD is "I have an m×n matrix and want its SVD"; reaches for the SVD package. The natural caller of `prob/sample.go:WishartSample` is "I have a Bayesian posterior and want a draw"; reaches for the prob package. **rSVD is a linear-algebra problem solved with a probabilistic technique** — placement follows problem class, not solution technique.
2. **Reverse-direction parallel.** `optim/SimulatedAnnealing` is an optimisation problem solved with a probabilistic technique; it lives in `optim/`, not `prob/`. Same logic applies to rSVD.
3. **Import graph cleanliness.** Today: `prob/` imports nothing-from-reality. Adding 184-C6 would make `prob/` import `linalg/`. Adding D7 here in `linalg/` makes `linalg/` import `prob/` (for D1 keystone). **Both are unidirectional** — neither creates a cycle. But the SECOND has a cleaner consumer story: `linalg/` already knows it's the leaf-level math primitive, so extending it with randomized variants is "linalg gains a new family of algorithms"; whereas the FIRST is "prob gains an unrelated SVD function."
4. **Cross-package consumers.** `aicore`, `causal`, `parallax` all already-or-will import `linalg/`; adding rSVD to `linalg/` exposes it to those consumers without forcing them to also import `prob/`. Forcing dual imports for "I just want a faster SVD" is the wrong API.
5. **Discoverability.** A user looking for SVD scrolls `linalg/`; finding `linalg/RandomizedSVD` next to `linalg/QRAlgorithm` is the right discoverability story. Finding rSVD only via `prob/randomized` is hidden.

**Summary table — recommended placement:**

| File | Primitives | LOC | Edges |
|---|---|---|---|
| `prob/sample.go` (NEW) | D1 StandardNormal, D19 Wishart, D20 RandomOrthogonal | ~280 | none-out |
| `linalg/randomized.go` (NEW) | D7 rSVD, D8 RangeFinder, D9 SketchedLS, D10 RandMatMul | ~470 | linalg→prob (for D1) |
| `linalg/trace.go` (NEW) | D2 Hutchinson, D3 Hutch++, D4 XTrace, D14 LogDet, D24 DiagEst | ~520 | linalg→prob (for D1) |
| `linalg/krylov.go` (NEW) | D5 PowerIter, D6 Lanczos, D17 rKaczmarz, D25 SpectralDensity | ~480 | linalg→prob (for D1) |
| `linalg/sketch.go` (NEW) | D15 SRHT, D16 JL+Gaussian+Achlioptas | ~170 | linalg→prob (for D1) |
| `linalg/sample_select.go` (NEW) | D11 LeverageScores, D12 CUR, D13 ColSubset | ~400 | linalg→prob (for D1) |
| `linalg/probabilistic.go` (NEW) | D21 BayesCG, D22 BayesQuad, D23 ProbODE, D18 RandCD | ~550 | linalg→prob (for D1) |

**Total ~2870 LOC source.** All in linalg/ except the keystone D1 in prob/. Single edge `linalg/ → prob/` for D1; zero cycles (prob/ stays leaf-of-reality per CLAUDE.md "reality imports nothing" — `prob/sample.go` adds no new imports beyond stdlib `math/rand`).

---

## 4. Landing order

- **PR-1 (1 day, 50 LOC).** D1 StandardNormalSample. Smallest possible commit; unblocks every subsequent PR. Identical to 184-PR-2 (the two reports converge here — ship ONCE).
- **PR-2 (1 day, 230 LOC).** D2 Hutchinson + (its golden-pin against existing `linalg.Trace`). One-day standalone; saturates the FIRST randomized-NLA primitive against existing exact ground truth. Stage one of P1.
- **PR-3 (1 day, 200 LOC).** D5 PowerIter + D6 Lanczos. Both pin against existing `linalg.QRAlgorithm` for free. NO new linalg primitives needed. Three R-MUTUAL pins available (Lanczos eigvals vs QR eigvals; PowerIter top-eigval vs QR's max eigval; Lanczos for D14 path).
- **PR-4 (2 days, 280 LOC).** D14 LogDetEstimator + D2/D14 cross-validation pin against `2·sum(log(diag(Cholesky(A))))`. Stage two of P1.
- **PR-5 (1 day, 100 LOC).** D17 RandKaczmarz. Stand-alone iterative LS, no 097-T1 dependency.
- **PR-6 (1 day, 170 LOC).** D15 + D16 SRHT + Gaussian + Achlioptas projections. Saturates P2.
- **PR-7 (2 days, 280 LOC).** D9 SketchedLS via Cholesky-on-normal-equations path; pin against `linalg.LUSolve(AᵀA, Aᵀb)`. Stage P3.
- **PR-8 (BLOCKED on 097-T1).** D7 rSVD + D8 RangeFinder + D11 LeverageScores + D12 CUR + D20 RandomOrthogonal. Land all five together once Householder-QR + Golub-Reinsch SVD ship.
- **PR-9 (1 day, 150 LOC).** D21 BayesianCG. Stand-alone (uses existing `linalg.MatVecMul/DotProduct/Inverse`).
- **PR-10 (BLOCKED on 184-C20 GP shipping).** D22 BayesianQuadrature.
- **PR-11 (BLOCKED on 097-T1 + D6 Lanczos).** D3 Hutch++, D4 XTrace.

Total ~3070 LOC source over 11 PRs and ~12-14 engineer-days.

---

## 5. Precision hazards (per primitive)

- **D1**: Marsaglia polar rejects 1−π/4≈21% of pairs (~2.54 RNG draws/scalar); tail beyond 6σ underrepresented vs ziggurat.
- **D2/D3/D14**: Hutchinson std-err scales 1/√m; Hutch++ scales 1/m (provable squared advantage).
- **D6**: Lanczos loses orthogonality after ~30 steps; selective reorth (Simon 1984) required for k≥30, else cap k≤25.
- **D7**: Power-iter q≥2 mandatory when σ_{k+1}/σ_k > 0.5 per HMT 2011 §10.
- **D9**: Sketched-LS via Cholesky on normal equations has O(κ(A)²) condition penalty; for ill-cond A use 097-T1 QR.
- **D14**: Log of numerical-zero eigenvalues blows up; clamp θᵢ ≥ √eps·θ_max.
- **D15**: Walsh-Hadamard requires p = power of 2; pad zeros.
- **D17**: rKaczmarz rate (1 − 1/κ_F²(A))^k — use only when κ_F ≤ 100.
- **D20**: Mezzadri sign-correction CRITICAL for Haar uniformity; without it Q is biased toward +1 on diag(R).
- **D21**: Posterior cov update breakdown when CG residuals become parallel (rare).

---

## 6. Cross-language pinning targets

| Primitive | Reference impl | Tolerance |
|---|---|---|
| D1 | `numpy.random.standard_normal` distribution match | KS p>0.05 |
| D2 | `numpy.einsum('i,ij,j', v, A, v)` exact + Avron-Toledo 2011 variance bound | 3·stdErr |
| D3 | Meyer-Musco-Musco-Woodruff 2021 reference Julia impl | 3·stdErr |
| D6 | scipy.sparse.linalg.lobpcg / scipy.sparse.linalg.eigsh (which IS Lanczos under the hood for sym) | 1e-8 on eigvals |
| D7 | scipy.sparse.linalg.svds (Lanczos-based) and scikit-learn TruncatedSVD(algorithm='randomized') | 1e-9 |
| D9 | scipy.linalg.lstsq + Sarlós theoretical bound | (1+ε) relative |
| D10 | numpy.dot exact + DKM06 Theorem 1 | bound check |
| D11 | scikit-learn `extmath.randomized_svd` leverage-score postprocess | 1e-8 |
| D12 | pyCUR or Mahoney-Drineas matlab demo | (1+ε) relative |
| D14 | scipy.linalg.slogdet exact + Ubaru-Chen-Saad 2017 reference | 3·stdErr |
| D15 | no canonical scipy SRHT; pin via JL ε-isometry over Wishart | 1% |
| D17 | Strohmer-Vershynin 2009 reference matlab | exponential decay |
| D21 | Cockayne-Oates-Sullivan-Girolami 2019 reference R impl | 1e-6 mean, 1e-4 cov |

---

## 7. Differentiation from prior agents

- **184-synergy-linalg-prob**: OPPOSITE direction. 184 = linalg-techniques-inside-prob (covariance, GP, BLR). THIS = prob-techniques-inside-linalg (rSVD, trace, sketched-LS, Krylov-w-random-start). Six overlaps (D1=184-C12, D7=184-C6, D15=184-C18, D16=184-C15-17, D19=184-C13, D20≈184-C20-prereq) ship in joint PRs. 18 new primitives (D2-D6, D9-D14, D17-D18, D21-D25) are unique here.
- **097-linalg-missing**: Identifies QR/SVD/Lanczos as T1 absences. THIS flags 9 primitives blocked on those (D3/D4/D7/D8/D11/D12/D20/D25); sharpens 097-T1 priority, adds 16 downstream primitives.
- **098-linalg-sota**: Orthogonal — RandNLA isn't in LAPACK (it's in scipy.sparse.linalg + scikit-learn).
- **103-optim-sota/105-optim-perf**: D18 randomized coord-descent closes one OPTIM-SOTA gap.
- **117-prob-missing**: Flags Box-Muller/Wishart/MVN. THIS shows the downstream RandNLA suite they unblock — 18 primitives directly downstream of D1 (= 117-T2). Sharpens 117 priority.
- **132-signal-missing/044-compression-api**: Compression-API asks for rSVD on FFT-spectrogram features; D7 fulfils.
- **149-zkmark-api**: ZK proofs need Hutchinson for FFT-arithmetic-circuit evaluation; D2 fulfils.
- **153-synergy-prob-infogeo**: Adjacent at D14 (log-det is the Bayes-factor / KL ingredient).
- **157-synergy-graph-linalg**: Graph Laplacian spectrum via rSVD; D7 applicable. Shares 097-T1 block.
- **161-synergy-control-prob**: Kalman is a special case of D21 BayesianCG.
- **163/169/177/178-synergy-***: Orthogonal at primitive level; D18 extends 169 LASSO PR.
- **180-synergy-physics-prob**: D17 randomized-Kaczmarz is the linalg dual of importance-sampling-by-row.
- **185/186/187-synergy-***: Orthogonal generative processes.

---

## 8. Bottom-line recommendation

**Single one-day high-leverage commit if-only-one-PR ships PR-1 = D1 StandardNormalSample at 50 LOC** because:

(a) Single primitive unblocks 14 of 25 D-primitives in this report PLUS 6 of 20 C-primitives in 184; combined unblock factor 20×.
(b) Closes the largest documented absence in `prob/` (no samplers at all, only PDF/CDF/Quantile).
(c) Establishes the precedent for `prob/sample.go` as the home of all RNG-driven distributions.
(d) Trivially golden-file-pinnable via KS test against numpy.

**Second-best one-day commit: PR-2 + PR-3 = D2 Hutchinson + D5 PowerIter + D6 Lanczos at ~330 LOC** because all three pin against existing `linalg.Trace` and `linalg.QRAlgorithm` for free; no new linalg primitives, no 097-T1 dependency, three independent randomized-NLA primitives shipped on day 2.

**Highest-leverage architectural addition = (PR-1 + PR-3 + PR-4 = D1 + D5 + D6 + D14) at ~600 LOC** because together they ship the FULL Lanczos-quadrature log-det stack — the keystone for matrix-free large-A computations that classical dense LAPACK literally cannot fit. This unlocks GP regression at n=10⁶ (impossible with Cholesky), Bayesian Wishart inference on huge covariances, signal-processing on streaming matrices. Three days of work to lift `linalg/` from "small dense LAPACK clone" to "small dense LAPACK + small RandNLA stack" — a genuine class shift.
