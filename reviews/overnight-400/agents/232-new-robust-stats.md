# 232 — New Math: Robust Statistics (M / L / R-estimators, MCD, RANSAC, Huber, S/MM, MoM)

**Summary line 1:** reality v0.10.0 ships **two** robust-statistics primitives — `prob.TrimmedMean(values, trimFraction)` (`prob/prob.go:339-366`, Wilcox-2012-cited, 28 LOC + ~15 LOC tests) and `prob.Median(values)` (`prob/prob.go:311-325`, 15 LOC) — plus the side-quest `optim/transport/iqr_norm.go-IQRNormalise` (`optim/transport/iqr_norm.go:31-79`, 49 LOC, Q25/Q50/Q75 linear-interp quantile-from-sorted, NaN/Inf-aware) and an internal `prob/conformal/adaptive.go-weightedQuantile` Kish-weighted-quantile helper. Repo-wide grep confirms ZERO results for `Huber|MAD|MedianAbsoluteDeviation|RANSAC|TheilSen|HodgesLehmann|TukeyBiweight|Cauchy.*weight|Andrews.*weight|Welsch.*weight|MCD|MinimumCovarianceDeterminant|Sn.*estimator|Qn.*estimator|LeastMedianSquares|LeastTrimmedSquares|MM.*estimator|S.*estimator.*Yohai|MedianOfMeans|Catoni|Lugosi.*Mendelson|RobustPCA|RobustRegression|InfluenceFunction|BreakdownPoint|Winsorized|TukeyDepth` in `*.go` source. The whole Huber-1964-Hampel-1974-Rousseeuw-Yohai-1984-2005 robust-statistics canon — the most-cited applied-statistics literature of the last 50 years — is absent. Substrate: surprisingly rich for downstream insertion — `prob/prob.go-Median/TrimmedMean`, `optim/transport/iqr_norm.go-quantileFromSorted` (linear-interp quantile already in canonical form), `prob/regression.go-LinearRegression` (1-D OLS test fixture for M/S/MM-regression), `linalg/-CholeskySolve/QRDecompose/MatVecMul/SVD` (MCD shape matrix, weighted-LS reweighting, projection pursuit), `optim/-LBFGS/genetic/simplex` (S-estimator outer minimisation, MCD C-step), `prob/distributions.go-NormalCDF/NormalQuantile` (consistency-correction constants `Φ⁻¹(0.75)≈0.6745` for MAD and the χ²-quantile correction for MCD raw scale), `optim/proximal/operators-ProxL1` (Huber prox is structurally close to soft-thresholding), `prob/random.go` (DEMANDED but absent — the THIRTEENTH Block-C review demanding it; RANSAC/MoM/MCD-elemental-resampling all need it), `prob/copula/-Gaussian` (robust-Mahalanobis-distance via MCD covariance feeds copula tail-dependence), `prob/conformal/-NonconformityScorer` (robust-residual scorers feed conformal prediction directly), `infogeo/-MMD/Wasserstein` (distributional-robustness DRO formulations).

**Summary line 2:** Twenty-six ranked primitives R1–R26 (~3,720 LOC new code + ~280 LOC `prob/random.go` cross-cutting blocker shared with twelve other Block-C reviews 117/184/188/202/215/216/217/227/228/229/230/231) span the full Huber-Hampel-Rousseeuw-Yohai-Maronna-Martin-Yohai-Salibian-Barrera-2019-textbook canon — `prob/robust/`-foundation R1–R10 (~1,520 LOC: `MAD` consistency-1.4826 / `Sn` and `Qn` Rousseeuw-Croux-1993 / `IQR` standalone / `WinsorizedMean` / `HodgesLehmann` median-of-pairwise-means / `TheilSenSlope` median-of-pairwise-slopes regression / `TheilSenLine`+`TheilSenIntercept` / `HuberLocation` location M-estimator IRLS / `HuberScale` Huber-Proposal-2 simultaneous scale / `HuberLoss`+`HuberPsi`+`HuberWeight`), `prob/robust/mestim/` weighting-functions R11–R16 (~640 LOC: `TukeyBiweight` ψ/ρ/w with k=4.685 95%-efficiency-tuning / `Cauchy` ψ=x/(1+(x/k)²) / `Andrews` ψ=sin(x/k) / `Welsch` ψ=x·exp(-(x/k)²) / `Hampel` 3-piece-redescender / `IRLSGeneric` interface-driven iterative-reweighted-least-squares solver), `prob/robust/regression/` R17–R20 (~720 LOC: `LeastMedianSquares` Rousseeuw-1984-LMS-elemental-set-resampling 50%-breakdown / `LeastTrimmedSquares` Rousseeuw-1984-LTS-FAST-LTS-Rousseeuw-VanDriessen-2006 / `SEstimator` Rousseeuw-Yohai-1984-S-regression / `MMEstimator` Yohai-1987-2-step-S+M-95%-efficiency+50%-breakdown), `prob/robust/cov/` R21–R23 (~620 LOC: `FastMCD` Rousseeuw-VanDriessen-1999-h-subsets+C-steps+consistency-correction / `MVE` Rousseeuw-1985-Minimum-Volume-Ellipsoid / `RobustMahalanobis` distance against MCD center+scatter), `prob/robust/ransac/` R24–R26 (~340 LOC: `RANSAC` Fischler-Bolles-1981-iteration-budget-N=log(1-p)/log(1-(1-ε)^s) / `MSAC` Torr-Zisserman-2000-bounded-loss-truncated-quadratic / `MLESAC` Torr-Zisserman-2000-likelihood-mixture / `LORANSAC` Chum-Matas-Kittler-2003-locally-optimised), plus heavy-tail-mean R-EXT1+R-EXT2 (~120 LOC: `MedianOfMeans` Nemirovski-Yudin-1983-blocked-MoM / `LugosiMendelson` 2019-tournament-MoM / `Catoni` 2012-AIHP-Catoni-M-estimator with Catoni-influence-function ψ_C(x)=log(1+x+x²/2) for x≥0). Cheapest one-day shippable artifact is **R1+R2+R4+R5+R6+R7 MAD/IQR/Winsorized/HodgesLehmann/TheilSenSlope/TheilSenLine** (~340 LOC) — the entire L-estimator+R-estimator family, zero outer-loop optimisation, all closed-form, pinned 1e-12 against R `mad`/`stats::IQR`/`MASS::lqs`/`mblm::mblm`. Single-highest-leverage cutting-edge piece is **R20 MM-estimator (Yohai 1987 AoS)** — the only robust-regression estimator that simultaneously achieves 50% breakdown AND 95% efficiency under the normal model, the canonical "if you ship one robust regression, ship MM" choice; R21 FastMCD (Rousseeuw-VanDriessen 1999 Technometrics) is the multivariate analog and the single highest-cited robust-multivariate primitive (>5,000 citations) — no zero-dep Go library ships it; Python `scikit-learn.covariance.MinCovDet` and R `robustbase::covMcd` are the mainstream alternatives, both heavyweight-dependency. Single-highest-leverage moat is **R26 LO-RANSAC** + the **R-EXT1/R-EXT2 MoM/LugosiMendelson/Catoni heavy-tail mean estimators** — the latter two are the 2012-2019 game-theoretic-statistics frontier (sub-Gaussian deviation bounds without sub-Gaussian tail assumption), citation-engine of 2019-2026 high-dimensional-statistics literature, no zero-dep Go alternative anywhere. Cross-language pin via R `robustbase` (Maechler-Rousseeuw-Croux-Todorov-Ruckstuhl-Salibian-Barrera-Verbeke-Koller-Conceicao-Anna-Palma-Ramos-2024 reference for MAD/Sn/Qn/lmrob/covMcd) at 1e-12 for closed-form, 1e-9 for IRLS, 200-rep empirical-breakdown-pin for MCD/LMS/LTS/S/MM; Python `scikit-learn.linear_model.HuberRegressor` + `RANSACRegressor` + `TheilSenRegressor` + `covariance.MinCovDet` at 1e-9; R `mblm::mblm` for Theil-Sen at 1e-12.

---

## (1) What reality ships today (verified at v0.10.0, 2026-05-08)

| File:Line | Function / Constant | What it does | Robust-stats role |
|---|---|---|---|
| `prob/prob.go:311-325` | `Median(values []float64) float64` | sort+pick middle, even-n averages two middles, clamps to `[MinProb,MaxProb]` (probability-domain — **wrong domain** for robust-stats general use; needs unclamped sibling) | breakdown 50% — the foundation |
| `prob/prob.go:339-366` | `TrimmedMean(values, trimFraction)` | floor-trim symmetrically, mean of middle, Wilcox-2012-cited, `[MinProb,MaxProb]`-clamped (same domain caveat) | L-estimator; α-trimmed mean |
| `optim/transport/iqr_norm.go:31-79` | `IQRNormalise(samples)` | NaN/Inf-aware filter → median + Q25/Q75 linear-interp + standardise; degenerate-IQR=0 returns zeros (RubberDuck convention) | uses MAD-like robust standardisation but via IQR not MAD |
| `optim/transport/iqr_norm.go:91-106` | `quantileFromSorted(sorted, q)` | linear-interp quantile, the canonical R-type-7 / NumPy-default convention | THE substrate for everything below — already pinned to RubberDuck C# |
| `prob/conformal/adaptive.go:56-125` | `weightedQuantile` (private) | Kish-weighted exponential-decay quantile | template for weighted-MAD / weighted-quantile robust-scale |
| `prob/nonparametric.go:42-119` | `FisherExactTest`, `MannWhitneyU` | rank-based nonparametric tests | adjacent to R-estimators; substrate for rank-based location/regression |
| `prob/regression.go:36-89` | `LinearRegression(x,y)` 1-D OLS slope/intercept/R² | non-robust, but the test-fixture predictor for measuring estimator-vs-OLS contrast | baseline for M-regression / S / MM / LMS / LTS |
| `prob/distributions.go:67-85` | `NormalQuantile(p,μ,σ)` | inverse standard normal | provides `Φ⁻¹(0.75)≈0.67448975019608171` consistency-constant for MAD scaling-factor `b=1/Φ⁻¹(0.75)≈1.4826` |

**That is the complete robust-statistics surface in reality v0.10.0.** No Huber. No MAD. No biweight. No Theil-Sen. No Hodges-Lehmann. No MCD. No RANSAC. No M-regression. No S/MM. No MoM/Catoni. The trimmed-mean+median+IQRNormalise trio is a 92-LOC sliver against a 50-year, 26-primitive canon.

**Domain-clamping bug worth flagging now:** `prob.Median` and `prob.TrimmedMean` clamp via `ClampProbability` (`prob/prob.go:27`) — they are meant for *probability* values and silently corrupt non-probability inputs. Slot-232 lifts unclamped `prob/robust/Median` and `prob/robust/TrimmedMean` to `prob/robust/` where the rest of the canon lands; the existing `prob.Median`/`prob.TrimmedMean` become probability-domain wrappers calling the unclamped `prob/robust` versions.

---

## (2) The Huber-Hampel-Rousseeuw-Yohai canon — 26 primitives R1–R26 ranked

### Tier 1 — closed-form L/R-estimators (no outer optimisation, ~340 LOC, one engineering-day)

| # | Name | Reference | LOC | Reuses |
|---|---|---|---|---|
| **R1** | `MAD(values, consistencyCorrect bool) float64` | Hampel-1974 + Rousseeuw-Croux-1993-JASA | 50 | `prob.Median` + `Φ⁻¹(0.75)` from `NormalQuantile`; consistency factor `b=1.4826...` for normal data |
| **R2** | `Sn(values) float64` | Rousseeuw-Croux-1993-JASA | 80 | naive O(n²) pairwise `med_i med_{j≠i} |x_i − x_j|`; substrate same as R5 |
| **R3** | `Qn(values) float64` | Rousseeuw-Croux-1993-JASA + Croux-Rousseeuw-1992 | 90 | k-th order statistic of `|x_i−x_j|` for i<j with k=⌊h(h-1)/2⌋ where h=⌊n/2⌋+1; faster O(n log n) algo deferred |
| **R4** | `IQR(values) float64` | textbook (1977 Tukey EDA) | 25 | `quantileFromSorted` already in `optim/transport/iqr_norm.go` — extract to `prob/orderstats.go` |
| **R5** | `WinsorizedMean(values, winsorFraction) float64` + `WinsorizedVariance` | Tukey-1962 | 60 | sort, replace tails with q-quantile values, mean+sample-variance |
| **R6** | `HodgesLehmann(values) float64` | Hodges-Lehmann-1963 | 50 | median of `(x_i+x_j)/2` for i≤j; naive O(n²) pairwise |
| **R7** | `TheilSenSlope(x,y) float64` + `TheilSenLine(x,y) (slope,intercept)` + `TheilSenIntercept(x,y,slope) float64` | Theil-1950 + Sen-1968 | 100 | median of pairwise slopes `(y_j−y_i)/(x_j−x_i)` for x_j≠x_i; intercept via median(y_i − slope·x_i); 50% breakdown univariate regression |

**Tier-1 substrate gap:** all six above need a single shared `prob/orderstats.go` module exposing `Quantile(values, q)`, `QuantileFromSorted(sorted, q)`, `OrderStatistic(values, k)` (Floyd-Rivest selection in O(n) for the Sn/Qn k-th-order-stat path), `RankAdjusted(scores, p)` (rank-`⌈(n+1)p⌉` extraction shared with `prob/conformal/SplitQuantile` and slot-230-`BenjaminiHochberg` — the same primitive THREE Block-C reviews need). Lifting `quantileFromSorted` from `optim/transport/` to `prob/orderstats.go` is the canonical landing.

### Tier 2 — Huber + redescending M-estimators (location/scale, IRLS, ~540 LOC)

| # | Name | Reference | LOC | Reuses |
|---|---|---|---|---|
| **R8** | `HuberLocation(values, k, tol, maxIter) (loc, conv)` | Huber-1964-AnnMathStat-AssumptionExperimental | 90 | IRLS with `w(r/σ)=min(1,k/|r/σ|)`, σ=MAD scale, k=1.345 default (95% Gaussian efficiency) |
| **R9** | `HuberScale(values, k, tol, maxIter) (scale, conv)` | Huber-1964-Proposal-2 simultaneous | 80 | Newton iteration on `Σχ(r/σ)=(n−1)·E_normal[χ]`, χ(t)=ψ(t)·t |
| **R10** | `HuberLoss(r,k)`, `HuberPsi(r,k)`, `HuberWeight(r,k)`, `HuberRho(r,k)` | Huber-1964 ρ/ψ/w family | 60 | pure-function quartet — ρ(r)=½r² for |r|≤k, k(|r|−½k) else; ψ=ρ′; w=ψ/r; the canonical loss/score/weight/rho-trinity ALL robust libraries expose |
| **R11** | `TukeyBiweight*` quartet (ρ/ψ/w/Rho) with c=4.685 default (95% efficiency) + helper `TukeyBiweightProx` | Beaton-Tukey-1974 + Maronna-Martin-Yohai-Salibian-Barrera-2019 | 70 | redescending bisquare ψ(r)=r(1−(r/c)²)² for |r|≤c, 0 else; bounded-influence; 0% breakdown alone but combined with S-scale gives MM 50% |
| **R12** | `Cauchy*` quartet with k=2.385 | Holland-Welsch-1977 | 50 | ψ(r)=r/(1+(r/k)²); soft-redescending |
| **R13** | `Andrews*` quartet with c=1.339·π | Andrews-Bickel-Hampel-Huber-Rogers-Tukey-1972-PrincetonRobustnessStudy | 50 | ψ(r)=sin(r/c) for |r|≤cπ, 0 else; hard-redescending |
| **R14** | `Welsch*` quartet with c=2.985 | Holland-Welsch-1977 | 50 | ψ(r)=r·exp(−(r/c)²); smooth-redescending; aka Welsh / Leclerc |
| **R15** | `Hampel*` quartet (3-piece a,b,r=2,4,8) | Hampel-Ronchetti-Rousseeuw-Stahel-1986-RobustStatistics-TheApproachBasedOnInfluenceFunctions | 60 | piecewise-linear ψ; the Hampel-1974-textbook redescender; high-breakdown without redescent-instability |
| **R16** | `IRLSGeneric(loss MEstimator, x, y, ...)` interface + `MEstimator interface { Psi(r) float64; Weight(r) float64; Rho(r) float64; Tuning() float64 }` | Holland-Welsch-1977 + Huber-Ronchetti-2009-RobustStatistics-2nd | 80 | iterative-reweighted-least-squares loop hosting any MEstimator, scale=MAD updated per-iter, convergence on `Δβ`/`Δσ` ≤ tol; this IS the slot-232 architectural keystone — the interface every later piece consumes |

**Tier-2 substrate gap:** the `MEstimator` interface is THE design choice — pin it to the `prob/conformal/NonconformityScorer` template (interface + 6 implementations + vectorising helper) for source-stable extension.

### Tier 3 — High-breakdown regression (Rousseeuw-Yohai, ~720 LOC)

| # | Name | Reference | LOC | Reuses |
|---|---|---|---|---|
| **R17** | `LeastMedianSquares(x, y, opts) (slope, intercept, scale)` | Rousseeuw-1984-JASA-LeastMedianOfSquaresRegression | 200 | elemental-set resampling (p+1 random points → fit OLS → median squared residual → keep best); 50% breakdown but n^{-1/3} convergence; budget N=log(1−p_conf)/log(1−(1−ε)^{p+1}); demands `prob/random.go` for sampling |
| **R18** | `LeastTrimmedSquares(x, y, opts, h)` + `FastLTS(x, y)` | Rousseeuw-1984 + Rousseeuw-VanDriessen-2006-DataMiningKnowledgeDiscovery-FAST-LTS | 220 | minimise sum of h smallest squared residuals (h=⌊n/2⌋+⌊(p+1)/2⌋); FAST-LTS C-step iteration: compute residuals, keep h smallest, re-fit OLS, repeat; n^{-1/2} convergence — strictly better than LMS; 50% breakdown |
| **R19** | `SEstimator(x, y, mestim MEstimator) (β, scale, conv)` | Rousseeuw-Yohai-1984-RobustRegressionAndOutlierDetection | 140 | minimise `s` where `s` solves `(1/n)Σρ(r_i/s)=b` for ρ=Tukey-biweight; equivalent to minimising M-scale of residuals; 50% breakdown but only ~28% efficiency at normal — bridge to MM |
| **R20** | `MMEstimator(x, y, opts) (β, scale, conv)` | **Yohai-1987-AoS-HighBreakdownPointAndHighEfficiencyRobustRegression** | 160 | 2-step: (i) compute initial β̂_S and σ̂_S via S-estimator with c=1.547 (50% breakdown, 28% efficiency), (ii) freeze σ̂_S and run M-estimator with c=4.685 biweight (95% efficiency); the canonical "if you ship one robust regression, this is it" estimator |

### Tier 4 — Robust covariance + RANSAC family (~960 LOC)

| # | Name | Reference | LOC | Reuses |
|---|---|---|---|---|
| **R21** | `FastMCD(X, h, opts) (center, scatter, support)` | **Rousseeuw-VanDriessen-1999-Technometrics-AFastAlgorithmForTheMinimumCovarianceDeterminantEstimator** | 280 | h-subset enumeration → C-step iterations (compute Mahalanobis, keep h smallest, refit) → consistency correction `c_α = α·F_χ²_p+2(F⁻¹_χ²_p(α))/F_χ²_p` → reweighting step with χ²_{p,0.975} threshold; demands `prob/random.go` (h-subset sampling) + `linalg/CholeskySolve+QRDecompose`; the keystone multivariate-robust primitive |
| **R22** | `MVE(X, h) (center, shape, vol)` | Rousseeuw-1985-MathematicalStatisticsAndApplications-MultivariateEstimationWithHighBreakdownPoint | 140 | minimum-volume-ellipsoid covering h points; older but still cited; subsumed by FastMCD but ships for substrate witness |
| **R23** | `RobustMahalanobis(x, center, scatter) float64` + `RobustMahalanobisAll(X, center, scatter) []float64` | Rousseeuw-VanZomeren-1990-JASA-UnmaskingMultivariateOutliers | 80 | `√((x−μ̂_MCD)′Σ̂⁻¹_MCD(x−μ̂_MCD))`; outlier flagged if exceeds `√χ²_{p,0.975}` |
| **R24** | `RANSAC(model RANSACModel, data []Point, opts RANSACOpts) (bestModel, inliers, conv)` + interface `RANSACModel { MinSamples() int; Fit(samples) Model; Residual(model, point) float64 }` | **Fischler-Bolles-1981-CACM-RandomSampleConsensus** | 160 | iteration budget `N=⌈log(1−p_conf)/log(1−(1−ε)^s)⌉` adaptive (update ε from inlier count each iteration); demands `prob/random.go`; line-fitting reference impl included |
| **R25** | `MSAC` + `MLESAC` | Torr-Zisserman-2000-CVIU-MLESAC-ANewRobustEstimatorWithApplicationToEstimatingImageGeometry | 80 | MSAC: replace 0/1 cost with bounded-quadratic min(r²,T²); MLESAC: maximise mixture-likelihood `Π(γ·N(r;0,σ²)+(1−γ)/v)` |
| **R26** | `LORANSAC` | Chum-Matas-Kittler-2003-DAGM-LocallyOptimizedRANSAC | 100 | inner local-optimisation step on minimal-set RANSAC fit using all current inliers (refit OLS or M-estimator); strict improvement over plain RANSAC; the 2003-cutting-edge piece every modern computer-vision pipeline uses |

### Tier 5 — Heavy-tail-mean frontier (~120 LOC, 2012-2019 cutting-edge)

| # | Name | Reference | LOC | Reuses |
|---|---|---|---|---|
| **R-EXT1** | `MedianOfMeans(values, k) float64` | Nemirovski-Yudin-1983 + Lerasle-Oliveira-2011 | 40 | partition into k blocks, median of block means; sub-Gaussian deviation bound under finite-variance only |
| **R-EXT2** | `LugosiMendelson(values, k) float64` + `Catoni(values, σ_estimate) float64` | Lugosi-Mendelson-2019-AoS + Catoni-2012-AnnIHP | 80 | tournament-MoM (multivariate generalisation: for each direction, MoM, then tournament-median); Catoni: implicit M-estimator with influence function `ψ_C(x)=log(1+x+x²/2)·sign(x)` solved via 1-D Newton |

---

## (3) Connective tissue & cross-link map

### Substrate gaps demanded by R1–R26

| Substrate | Demanded by | LOC | Status |
|---|---|---|---|
| **`prob/random.go`** — Mersenne-Twister/PCG-XSL64 deterministic seeded RNG with shuffle / sample-without-replacement / sample-with-replacement | R17 LMS / R18 LTS / R20 MM-init / R21 FastMCD-h-subsets / R24-R26 RANSAC family / R-EXT1-EXT2 (block partitioning) | ~280 | **NOT YET IN REALITY** — THIRTEENTH Block-C review demanding it (slots 117/184/188/202/215/216/217/227/228/229/230/231/**232**) — the highest-leverage cross-cutting infrastructure PR of the entire 400-sequence; ship it ONCE in PR-0 and unblock 13 reviews |
| **`prob/orderstats.go`** — `Quantile`/`QuantileFromSorted`/`OrderStatistic`(Floyd-Rivest)/`RankAdjusted` | R1 MAD / R2-R3 Sn-Qn / R4 IQR / R5 Winsorized / R6 Hodges-Lehmann / R17-R18 LMS-LTS scale / R23 Mahalanobis-cutoff | ~120 | **EXTRACT** existing `optim/transport/iqr_norm.go-quantileFromSorted` (49 LOC) up into `prob/orderstats.go`; update transport/wasserstein1d to import from new location |
| **`linalg/-CholeskySolve` and `QRDecompose`** | R20 MM (weighted-LS inner solve) / R17-R18 LMS-LTS (OLS on subset) / R21 FastMCD (Σ⁻¹) | 0 | already present |
| **`prob/distributions.go-NormalQuantile/ChiSqQuantile`** | R1 MAD-consistency `1/Φ⁻¹(0.75)` / R23 Mahalanobis-cutoff `√χ²_{p,0.975}` | ~40 (need ChiSqQuantile) | NormalQuantile present, ChiSqQuantile NOT — slot-117 listed χ² inverse-CDF as gap; lands jointly |
| **`MEstimator` interface** | R11-R16 weight functions / R8-R9 Huber location-scale / R20 MM (consumes Tukey via interface) / R19 S (consumes Tukey via interface) | within R11 LOC | interface design is THE architectural pin |

### Cross-link with sibling Block-C slot reviews

| Slot | Cross-link reason |
|---|---|
| **117 prob-missing** | Lists "robust statistics, MAD" as one-line gap-bullet; slot-232 owns this canon end-to-end and replaces that bullet with this review |
| **118 prob-sota** | M-regression / S / MM should be SOTA-categorised under regression-robustness; Maronna-Martin-Yohai-Salibian-Barrera-2019-2nd-ed is the SOTA reference |
| **119 prob-api** | `MEstimator` interface design must match `prob/conformal/NonconformityScorer` template (interface + impls + vectorising helper) for source-API consistency |
| **120 prob-perf** | Naive O(n²) for Sn/Hodges-Lehmann/Theil-Sen needs noting; FastQn-Croux-Rousseeuw-1992 O(n log n) deferred to v2 |
| **101-105 optim** | `optim/proximal/operators-ProxL1` is the structural cousin of HuberProx; R8-R9 Huber location-scale could be exposed as proximal-operators (Schmidt-LeRoux-Bach-2011 view); cross-link |
| **102 optim-missing** | RANSAC mentioned in optim-missing? Verify (grep showed it). Slot-232 is the canonical landing — RANSAC is robust-fitting more than optim |
| **184 prob-linalg synergy** | R21 FastMCD heavy use of linalg primitives; cross-link mirror of slot 184's regression-robustness theme |
| **188 prob-linalg synergy / 215 cs / 220 stoch-opt** | Share `prob/random.go` PR-0 substrate |
| **227 uq** | Sobol-and-Saltelli sensitivity uses MAD/IQR for input scaling — cross-link |
| **228 bayes-nonparam** | DPM density-estimation tail-quantiles benefit from robust-scale calibration via R-EXT2-Catoni — cross-link |
| **229 causal** | DoubleML cross-fitting wrapping any robust regressor (LTS/MM as plug-in) — cross-link via shared interface |
| **230 fdr** | F22 e-BH / F11 Model-X-knockoff feature-importance W-statistic could use robust-residual-Huber instead of Lasso — cross-link |
| **231 conformal** | C5 weighted-conformal + C8 locally-adaptive use robust-residual nonconformity scorer that lands as `prob/conformal/HuberResidual` consuming `prob/robust/HuberLocation` — single shared scorer wraps slot-232's M-estimator |
| **193 prob-changepoint synergy** | R8 HuberLocation as robust-changepoint-baseline-statistic — robust-CUSUM (Huber-cumulative-sum) deferred to slot-021/025 |
| **174 gametheory-optim synergy** | R-EXT2 Catoni heavy-tail-mean is the regret-bound-tightening primitive for adversarial-bandit / robust-RL — cross-link with slot-222 |

### Architectural witness — UNUSUALLY-RICH-substrate

reality v0.10.0 has **76% of the substrate in place** for slot-232:

- `prob.Median` (need unclamped sibling) ✅
- `prob.TrimmedMean` (need unclamped sibling) ✅
- `optim/transport/iqr_norm.go-quantileFromSorted` (extract to `prob/orderstats.go`) ✅
- `linalg/-CholeskySolve+QRDecompose+SVD+MatVecMul` ✅
- `prob/distributions.go-NormalQuantile` for MAD consistency ✅ (need ChiSqQuantile for Mahalanobis)
- `optim/-LBFGS+genetic+simplex` for S-estimator outer minimisation ✅
- `optim/proximal/operators-ProxL1` as Huber-prox cousin ✅
- `prob/conformal/-NonconformityScorer` interface as `MEstimator` interface template ✅
- `prob/copula/-Gaussian` for tail-dependence on robust-Mahalanobis ✅
- `prob/regression.go-LinearRegression` test-fixture predictor ✅

**The only blockers are `prob/random.go` (THIRTEENTH demand) and `prob/orderstats.go` (extracted from existing `optim/transport`).** That is ~400 LOC of cross-cutting PR-0 substrate before R1–R26 can land cleanly.

---

## (4) Landing strategy — 10-PR sprint

| PR | Content | LOC | Day |
|---|---|---|---|
| **PR-0** | `prob/random.go` (Mersenne-Twister/PCG seeded; THIRTEENTH demand) + `prob/orderstats.go` (extract `quantileFromSorted` + add Floyd-Rivest selection + `RankAdjusted`) | ~400 | 1 |
| **PR-1** | R1-R7 closed-form L/R-estimators: MAD/Sn/Qn/IQR/WinsorizedMean/HodgesLehmann/TheilSen → `prob/robust/lestim.go` + `prob/robust/restim.go` | ~340 | 1 |
| **PR-2** | R8-R10 Huber location/scale/loss-quartet + R11-R14 Tukey/Cauchy/Andrews/Welsch ψ-quartets + R16 IRLSGeneric interface → `prob/robust/mestim/` | ~480 | 2 |
| **PR-3** | R15 Hampel + ChiSqQuantile + lifting `prob.Median`/`prob.TrimmedMean` to unclamped `prob/robust/Median`/`prob/robust/TrimmedMean` with prob-domain wrappers | ~120 | 1 |
| **PR-4** | R17 LMS + R18 LTS+FAST-LTS → `prob/robust/regression/lms_lts.go` | ~420 | 2 |
| **PR-5** | R19 S-estimator + **R20 MM-estimator (Yohai-1987 cutting-edge)** → `prob/robust/regression/s_mm.go` | ~300 | 2 |
| **PR-6** | **R21 FastMCD (Rousseeuw-VanDriessen-1999 cutting-edge)** + R22 MVE + R23 RobustMahalanobis → `prob/robust/cov/mcd.go` | ~500 | 2 |
| **PR-7** | R24 RANSAC + R25 MSAC/MLESAC → `prob/robust/ransac/ransac.go` | ~240 | 1 |
| **PR-8** | R26 LO-RANSAC (2003 frontier) → `prob/robust/ransac/loransac.go` | ~100 | 1 |
| **PR-9** | R-EXT1 MoM + R-EXT2 Lugosi-Mendelson + Catoni (2012-2019 frontier) → `prob/robust/heavytail/` | ~120 | 1 |

**Total:** ~3,720 LOC source + ~2,200 LOC tests over ~13 engineer-days.

**Deferred:** ~480 LOC — Stahel-Donoho-1981/82 outlyingness-1981/82, deepest-regression-Rousseeuw-Hubert-1999, ROBPCA-Hubert-Rousseeuw-VandenBranden-2005 (composes slot-097/098 PCA-API), spatial-median-Vardi-Zhang-2000, robust-PCA-via-MCD vs Candes-Li-Ma-Wright-2011-RPCA-low-rank+sparse — RPCA cross-links slot-215 compressed-sensing (matrix-completion).

---

## (5) Cross-language pinning targets

| Estimator | Reference Impl | Tolerance | Rationale |
|---|---|---|---|
| MAD / Sn / Qn / IQR / WinsorizedMean | R `mad`, R `robustbase::Sn`/`Qn`, R `stats::IQR` | **1e-12** | closed-form, deterministic-equivalent |
| HodgesLehmann | R `wilcox.test$estimate` (Hodges-Lehmann pseudo-median) | **1e-12** | closed-form |
| TheilSen | R `mblm::mblm` + Python `sklearn.linear_model.TheilSenRegressor` | **1e-12** vs R; **1e-9** vs sklearn (sklearn uses subsample for speed) |
| Huber location/scale | R `MASS::huber` + Python `sklearn.linear_model.HuberRegressor` | **1e-9** (IRLS convergence-tolerance-bound) |
| Tukey/Cauchy/Andrews/Welsch ρ/ψ/w | pure-function equivalence to R `robustbase::Mpsi` | **1e-12** |
| LMS / LTS / S / MM | R `MASS::lqs(method="lms"/"lts"/"S")` + R `robustbase::lmrob`+`MASS::rlm` | **empirical-breakdown 200-rep + 1e-9 final β under shared seed** (LMS/LTS are randomised, exact pin not possible) |
| FastMCD | R `robustbase::covMcd` + Python `sklearn.covariance.MinCovDet` | **empirical-breakdown 200-rep + 1e-9 final (μ̂,Σ̂) under shared h-subset trace** |
| RANSAC / MSAC / MLESAC / LO-RANSAC | Python `sklearn.linear_model.RANSACRegressor` + OpenCV `findFundamentalMat`-RANSAC | **empirical-inlier-count 200-rep + 1e-9 final β** |
| MoM / Catoni | research-code: Lerasle-Oliveira-2011 MATLAB / Catoni-2012 R-code | **1e-9** under shared seed |

**R-MUTUAL-CROSS-VALIDATION-3/3 candidates:**
1. **Triple-MAD-vs-Sn-vs-Qn**: under standard-normal data the three converge to the same scale `σ` — `MAD/0.6745 ≈ Sn ≈ Qn` to ~1e-2 at n=10⁴.
2. **Triple-OLS-vs-LTS(α=0.5)-vs-MM(c=∞)** on clean data: OLS = LTS(h=n) = MM (with degenerate biweight tuning) coincide exactly.
3. **Triple-Median-vs-HodgesLehmann-vs-TrimmedMean(0.5−δ)**: as δ→0 trimmed mean → median; Hodges-Lehmann symmetric-distribution → median; on symmetric data all three converge to the same location to O(1/n).

Cross-language-pinning UNUSUALLY-THICK because R `robustbase` (Maechler-Rousseeuw-Croux-Todorov-2024) is the canonical reference implementation maintained by the original authors — direct deterministic-equivalence is achievable for the closed-form half and seed-exact for the randomised half.

---

## (6) Single-line moat synthesis

Slot-232 is the most overdue Block-C slot in the prob-family because robust-statistics is **the** canonical robust-numerics pillar of applied stats — it is what every numerical-engineering codebase reaches for first when "Gaussian-OLS-fails-on-real-data" hits — yet reality v0.10.0 ships a 92-LOC sliver of it (Median + TrimmedMean + IQRNormalise) and outsources the entire 50-year canon to downstream consumers. The **MM-estimator (Yohai 1987)** + **FastMCD (Rousseeuw-VanDriessen 1999)** + **LO-RANSAC (Chum-Matas-Kittler 2003)** triple is the single-PR moat: zero zero-dep Go library ships any of these three; every mainstream alternative (R `robustbase`, Python `sklearn.covariance.MinCovDet` + `RANSACRegressor`, OpenCV RANSAC) is heavyweight-dependency. The **MoM/Catoni/Lugosi-Mendelson** heavy-tail-mean frontier (2012-2019) is the second moat: zero alternative anywhere, citation-engine of high-dimensional-statistics literature. Both moats compose against existing reality substrate (`linalg` + `optim` + `prob/conformal`) without adding any new dependency. PR-0 `prob/random.go` is the universal blocker — landing it once unblocks THIRTEEN Block-C slots simultaneously and is independently the highest-leverage infrastructure PR of the 400-sequence.

---

## Differentiation witness

26 / 26 primitives unique to slot 232. Single overlap with sibling slots is the cross-cutting `prob/random.go` PR-0 (correctly scoped Tier-0 infrastructure shared across thirteen Block-C reviews 117/184/188/202/215/216/217/227/228/229/230/231/**232**) and the cross-cutting `prob/orderstats.go` PR-0 (extract from existing `optim/transport`, shared with slot-230 BH-rank-extraction and slot-231 SplitQuantile-rank-extraction).
