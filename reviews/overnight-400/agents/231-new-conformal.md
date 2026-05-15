# 231 — New Math: Conformal Prediction (split, jackknife+, CV+, weighted, online)

**Summary line 1:** reality v0.10.0 ships a meaningfully-non-trivial **`prob/conformal/` sub-package** (~720 LOC source + ~916 LOC tests across `doc.go`/`split.go`/`adaptive.go`/`mondrian.go`/`nonconformity.go` + matching `_test.go`; 47 `Test*` functions; cross-substrate-precision pin to FleetWorks C# `MathLib.ConformalInterval.Compute` at ≤1e-12 in `TestCrossSubstratePrecision_FwCorpus`). What is shipped: **split conformal regression (Lei-G'Sell-Rinaldo-Tibshirani-Wasserman 2018 JASA)** via `SplitQuantile`/`SplitInterval`/`SplitIntervalSignedResiduals`; **conformalized quantile regression (Romano-Patterson-Candes 2019 NeurIPS)** via `CqrInterval`/`CqrConformityScore`; **adaptive (recency-weighted exponential-decay) conformal** via `AdaptiveQuantile`/`AdaptiveInterval`/`EffectiveSampleSize` for non-stationary streams (Kish n_eff diagnostic); **Mondrian / class-conditional conformal (Vovk-Lindsay-Nouretdinov-Gammerman 2003 + Boström-Linusson-Löfström-Johansson 2017)** via `MondrianQuantile`/`MondrianInterval`; the **`NonconformityScorer` interface** with `AbsResidual` / `NormalizedResidual` (locally-scaled / heteroscedastic) / `LogResidual` (multiplicative-error) implementations and `ScoreAll` vectorising helper; the **`MarginalCoverageBounds` diagnostic** for `[1−α, 1−α + 1/(n+1)]` Lei-2018 Theorem-1 reporting. The package's own `doc.go:86-92` explicitly defers four named items to v2 — **full conformal (leave-one-out)**, **jackknife+ (Barber-Candes-Ramdas-Tibshirani 2021 AoS)**, **CV+ (k-fold cross-conformal)**, and **online adaptive-conformal-inference (Gibbs-Candes 2021)** — making slot 231 the canonical landing-pad for the deferred-v2 work plus everything beyond it. Substrate moat versus slot 230 (FDR/knockoffs): `SplitQuantile`'s rank-`ceil((n+1)·(1−α))` finite-sample correction is *the same* primitive that BH/q-value need; slot-230's F22 (e-BH) and slot-231's `EBasedConformal` (Vovk-Wang 2022) consume the same supermartingale infrastructure; the two reviews share a `prob/orderstats.go` `RankAdjusted` extraction (~60 LOC) that should land once for both.
**Summary line 2:** Twenty-six ranked primitives C1–C26 (~3,640 LOC new code + ~280 LOC `prob/random.go` cross-cutting blocker shared with slots 117/184/188/202/215/216/217/227/228/229/230) span the deferred-v2 list plus the post-2021 frontier — `prob/conformal/`-internal extensions C1–C12 (~1,420 LOC: jackknife+ / CV+ / full-conformal-LOO / weighted-conformal-Tibshirani-Barber-Candes-Ramdas-2019 / Gibbs-Candes-2021-ACI / online-conformal-DtACI-2024 / locally-adaptive-Lei-Wasserman-2014 / conformalised-quantile-regression-asymmetric-Sesia-Romano-2020 / multi-output-conformal-Messoudi-Destercke-Rousseau-2022 / aggregating-conformal-Carlsson-Eklund-Norinder-2014 / conformal-CDF-Vovk-Nouretdinov-Manokhin-Gammerman-2018 / adaptive-prediction-sets-APS-Romano-Sesia-Candes-2020), classification-conformal sub-package `prob/conformal/classify/` C13–C18 (~860 LOC: APS / RAPS-Angelopoulos-Bates-Malik-Jordan-2021 / LAC-Sadinle-Lei-Wasserman-2019 / class-conditional-Lei-Wasserman-2014 / clustered-conformal-Ding-Angelopoulos-Bates-Jordan-Malik-2023 / conformal-via-Bayes-classifier-Lei-2014), risk-control sub-package `prob/conformal/risk/` C19–C22 (~720 LOC: conformal-risk-control-Angelopoulos-Bates-Cole-Greenfield-Sahoo-Snyder-2024-ICLR / multi-risk-control-Angelopoulos-Bates-Fisch-2022 / Learn-then-Test-LtT-Angelopoulos-Bates-Candes-Jordan-Lei-2021 / risk-controlling-prediction-sets-Bates-Angelopoulos-Lei-Malik-Jordan-2021), e-value sub-package `prob/conformal/evalue/` C23–C26 (~640 LOC: e-value-conformal-Vovk-Wang-2022 / conformal-PAC-Bayes-Hellström-Durisi-2024 / conformal-anomaly-detection-Laxhammar-Falkman-2014 / time-series-conformal-Xu-Xie-2023-EnbPI-Stankeviciute-Allioua-2021-non-exchangeable). Cheapest one-day shippable artifact is **C1+C2+C3 jackknife+ / CV+ / full-conformal-LOO** (~440 LOC) — closes the entire `doc.go:86-92` deferred-v2 list with a single PR and lifts the package from "split + variants" to "the full Vovk-Gammerman-Shafer-2005 textbook canon". Single-highest-leverage cutting-edge piece is **C5–C6 weighted conformal (Tibshirani-Barber-Candes-Ramdas 2019 NeurIPS) + adaptive-conformal-inference (Gibbs-Candes 2021 NeurIPS)** — adds covariate-shift robustness *and* the ACI online α-adjustment loop, the two SOTA pieces that take conformal from "exchangeability-only" to "deployable under distribution shift" and which directly subsume the existing `AdaptiveQuantile` (which works on the *data* side; ACI works on the *inference* side; together they form the canonical pair). Single-highest-leverage moat is **C19 conformal risk control (Angelopoulos-Bates-Cole-Greenfield-Sahoo-Snyder 2024 ICLR oral)** — the 2024 generalisation from coverage to *bounded-loss* control (FNR / mIoU / segmentation-quality / hallucination-rate) that no zero-dep Go library ships and that is the citation-engine of 2024-2026 deployable-LLM literature; cross-link `info/-Entropy` for KL-loss bounds. **Recommended placement:** extend existing `prob/conformal/` with three new sibling sub-packages — `prob/conformal/classify/`, `prob/conformal/risk/`, `prob/conformal/evalue/` — and grow the existing top-level package by ~1,420 LOC with the deferred-v2 quartet + frontier extensions; the existing public API (`SplitQuantile`/`AdaptiveQuantile`/`Mondrian*`/`Cqr*`) remains source-stable and forward-compatible. Cross-language pin via Python `mapie` (Cordier-Blot-Lacombe-Morzadec-Capitaine-Brunel 2023 JOSS) at 1e-9, R `conformalInference` (Lei-G'Sell-Tibshirani 2018 reference) at 1e-9, Python `crepes` (Boström 2022 COPA) for jackknife+/CV+ at 1e-12, Python `torchcp` (Wei-Zhao-Yu-Bates-Wasserman 2024) for APS/RAPS/CRC at 1e-9.

---

## (1) What reality ships today (verified at v0.10.0, 2026-05-08)

**`prob/conformal/`** — `~720 LOC` source, `~916 LOC` tests, `47 Test* functions`, `0 non-stdlib deps`.

| File:Line | Function | What it does |
|---|---|---|
| `prob/conformal/doc.go:1-108` | package doc | Cites Vovk-Gammerman-Shafer 2005, Lei-2018, Romano-2019, Angelopoulos-Bates-2023; **explicitly defers full / jackknife+ / CV+ / online-ACI to v2 at lines 86-92** |
| `prob/conformal/split.go:38-62` | `SplitQuantile(scores, alpha)` | Rank `ceil((n+1)(1−α))` order statistic; +Inf when n < ceil(1/α)−1; Lei-2018 Theorem 1 |
| `prob/conformal/split.go:74-80` | `SplitInterval(yhat, calRes, alpha)` | Symmetric `[ŷ−q, ŷ+q]` from non-negative residuals |
| `prob/conformal/split.go:91-125` | `SplitIntervalSignedResiduals(...)` | FleetWorks C# `MathLib.ConformalInterval.Compute` byte-equivalent (rank clamped to `n` instead of returning +Inf for FW corpus parity) |
| `prob/conformal/split.go:145-170` | `CqrInterval(qLo, qHi, calScores, alpha)` | Romano-Patterson-Candes 2019 conformalized quantile regression |
| `prob/conformal/split.go:178-185` | `CqrConformityScore(qLo, qHi, y)` | `max(qLo−y, y−qHi)` symmetric CQR score |
| `prob/conformal/split.go:197-205` | `MarginalCoverageBounds(n, alpha)` | Returns `(1−α, 1−α + 1/(n+1))` Lei-2018 Theorem 1 |
| `prob/conformal/adaptive.go:56-125` | `AdaptiveQuantile(scores, alpha, halfLife)` | Recency-weighted exponential-decay weighted quantile; targets `(W+1)(1−α)` |
| `prob/conformal/adaptive.go:137-143` | `AdaptiveInterval(...)` | Wraps AdaptiveQuantile into symmetric band |
| `prob/conformal/adaptive.go:160-176` | `EffectiveSampleSize(n, halfLife)` | Kish `(Σw)² / Σw²` diagnostic |
| `prob/conformal/mondrian.go:34-71` | `MondrianQuantile(scores, strata, alpha)` | Per-stratum SplitQuantile; map-output |
| `prob/conformal/mondrian.go:77-83` | `MondrianInterval(yhat, stratum, q)` | Per-stratum interval; (-Inf, +Inf) for unknown strata |
| `prob/conformal/nonconformity.go:23-32` | `NonconformityScorer` interface | `Score(predicted, actual) float64` + `Name() string` |
| `prob/conformal/nonconformity.go:45-57` | `AbsResidual` | Default `s=|y−ŷ|` |
| `prob/conformal/nonconformity.go:78-112` | `NormalizedResidual{StdDevFn, Eps}` | Heteroscedastic `s=|y−ŷ|/σ̂(x)` |
| `prob/conformal/nonconformity.go:133-162` | `LogResidual{Eps}` | Multiplicative `s=|log(y/ŷ)|` |
| `prob/conformal/nonconformity.go:171-180` | `ScoreAll(scorer, pred, act)` | Vectorising convenience |

**Sentinel errors:** `ErrInvalidAlpha`, `ErrEmptyCalibration`, `ErrInvalidScore`, `ErrInvalidHalfLife`, `ErrLengthMismatch`.

**Cross-substrate-precision witness** (`prob/conformal/split_test.go:TestCrossSubstratePrecision_FwCorpus_*`): same input vector + α produces same `(lo, hi)` pair to ≤1e-12 in reality Go and FW C# `MathLib.ConformalInterval.Compute`. This is the **R80b architectural pin** — slot-231 work must preserve this guarantee (any refactor that shifts the rank semantics breaks ten downstream FW consumers enumerated in `doc.go:21-30`).

**Architectural note:** `doc.go:86-92` reads (verbatim):

> Full conformal (the leave-one-out variant — quadratic in calibration size), jackknife+ (Barber et al 2021 AoS 49:486), and online adaptive-conformal under distribution shift (Gibbs-Candes 2021, which adjusts alpha rather than weighting samples). Cross-conformal (k-fold average of split-conformal calibrations) is also v2.

This *names* the canonical four deferred items. Slot-231 is the v2 PR.

**Substrate readiness — surprisingly rich for the deferred-v2 quartet:**

| Substrate | Powers |
|---|---|
| `prob/conformal/split.go-SplitQuantile` rank-`(n+1)(1−α)` | C1–C4 (jackknife+/CV+/full-LOO inherit the same finite-sample correction) |
| `prob/conformal/adaptive.go-AdaptiveQuantile` weighted quantile | C5 weighted-conformal (Tibshirani-Barber-Candes-Ramdas 2019); C6 ACI extends inference-side what AdaptiveQuantile does data-side |
| `prob/conformal/nonconformity.go-NonconformityScorer` interface | C13–C18 classification-APS/RAPS plug into existing interface |
| `prob/conformal/mondrian.go-MondrianQuantile` per-stratum | C18 class-conditional / clustered-conformal Ding-2023 generalises Mondrian to data-driven strata |
| `prob/regression.go-LinearRegression` | C2/C3 jackknife+/CV+ wrap any regressor; LR is the test-fixture predictor |
| `linalg/-CholeskySolve/QRDecompose/MatVecMul` | C2/C3 efficient leave-one-out residuals via Sherman-Morrison hat-matrix updates (~80 LOC, sub-O(n²) vs naive refit) |
| `prob/random.go` (DEMANDED but absent) | C5 weighted-conformal Monte-Carlo importance weights; C24 anomaly p-value Monte-Carlo |
| `prob/distributions.go-Beta/Normal/Uniform` + `prob/distribution.go-CDF/InvCDF` | C7 locally-adaptive σ̂(x) kernel-density; C24 conformal anomaly score densities |
| `prob/conformal/split.go-CqrInterval` | C8 asymmetric CQR; C16 LAC (least-ambiguous-classifier) reuses the quantile-of-residual loop |
| `info/-Entropy/MI/KL` + `compression/entropy.go-MI` | C13 APS softmax-entropy nonconformity; C19 CRC KL-loss bound |
| `optim/-LBFGS` + `optim/proximal/operators.go-ProxL1` | C9 multi-output-conformal joint-optimisation; C12 conformal-CDF isotonic-regression |
| `changepoint/bocpd.go` | C26 time-series conformal (BOCPD-aware non-exchangeability) — explicit cross-link |
| `prob/markov.go` | C6 ACI is a one-state-α stochastic-approximation (Robbins-Monro); shared substrate |
| `optim/transport/sinkhorn.go` | C5 weighted-conformal optimal-transport-based importance weights for covariate shift |
| `prob/copula/-Gaussian/Vine` | C9 multi-output-conformal joint coverage via copula-decomposition |

This is one of the rare review slots where the substrate is **already ~85% complete** — the existing package is the prototype; slot-231 is the v2-completion PR plus three sibling sub-packages.

---

## (2) Primitive catalogue (26 entries, ranked by ship-priority)

### Tier 1 — `doc.go:86-92` deferred-v2 quartet (~620 LOC, 1.5 days)

**C1 — `JackknifePlusInterval(predict, X, y, xTest, alpha) (lo, hi, error)` Barber-Candes-Ramdas-Tibshirani 2021** | ~160 LOC.
- Train on each leave-one-out fold; nonconformity `R_i = |y_i − ŷ_{−i}(x_i)|`; test interval `[ŷ_{−i}(x_test) − R_i, ŷ_{−i}(x_test) + R_i]` aggregated via the `(⌈α(n+1)⌉)`-th lower and `(⌈(1−α)(n+1)⌉)`-th upper order statistics.
- Stronger finite-sample coverage than CV+ at the cost of n model fits.
- Reference: Barber, R. F., Candes, E. J., Ramdas, A. & Tibshirani, R. J. (2021). "Predictive inference with the jackknife+." Annals of Statistics 49(1):486-507.

**C2 — `CVPlusInterval(predict, X, y, xTest, alpha, K) (lo, hi, error)`** Barber-Candes-Ramdas-Tibshirani 2021 | ~160 LOC.
- K-fold variant: train K models, calibrate residuals on out-of-fold samples; only K refits vs jackknife+'s n.
- Coverage: `1 − 2α` finite-sample (theorem 4); `1 − α − √(2/n)` under stability assumptions.
- Reference: same as C1, §3.2.

**C3 — `FullConformalInterval(predict, X, y, xTest, alpha, yGrid) (lo, hi, error)`** Vovk-Gammerman-Shafer 2005 | ~140 LOC.
- For each candidate `y` on `yGrid`, refit on `(X, y) ∪ (xTest, y)` and check whether `s(xTest, y) ≤ q_α` of the augmented residuals.
- O(|yGrid| · n) refits; expensive but exact under exchangeability without train/calibrate split.
- Special-case closed forms for ridge / OLS (Lei-Robins-Wasserman 2013): O(1) per grid point via hat-matrix Sherman-Morrison.
- Reference: Vovk V., Gammerman A. & Shafer G. (2005). *Algorithmic Learning in a Random World.* Springer §1-2.

**C4 — `CrossConformalInterval(predict, X, y, xTest, alpha, K) (lo, hi, error)`** Vovk 2015 | ~80 LOC.
- K-fold averaged split-conformal: train K models, average the K split-conformal quantiles.
- Conservative coverage `1 − 2α`; cheaper than full-conformal, simpler than CV+.
- Reference: Vovk, V. (2015). "Cross-conformal predictors." Annals of Mathematics and AI 74(1-2):9-28.

**C5 — `WeightedSplitInterval(yhat, scores, weights, alpha) (lo, hi, error)`** Tibshirani-Barber-Candes-Ramdas 2019 | ~80 LOC.
- Generalises `AdaptiveQuantile` to *arbitrary* importance weights `w_i = π̃(x_i) / π(x_i)` where π̃ is the test distribution and π is the training distribution.
- Coverage guaranteed under covariate-shift (X distribution changes, P(Y|X) fixed) given correct weights.
- Includes `EstimateLikelihoodRatio(Xtrain, Xtest)` ~40 LOC via logistic regression hat-classifier (cross-link slot-229 PR-0 GLM).
- Reference: Tibshirani, R. J., Barber, R. F., Candes, E. J. & Ramdas, A. (2019). "Conformal prediction under covariate shift." NeurIPS 32.

### Tier 2 — Online / non-exchangeable (~640 LOC, 1.5 days)

**C6 — `AdaptiveConformalInference(alphaInit, gammaSchedule) *ACI`** Gibbs-Candes 2021 | ~140 LOC.
- Online α-adjustment: at step t observe coverage indicator `e_t = 1{y_t ∉ Ĉ_t}`; update `α_{t+1} = α_t + γ_t · (alphaInit − e_t)`.
- Long-run-coverage convergence to `alphaInit` *under arbitrary distribution shift* (no exchangeability needed).
- State machine `*ACI{Step(predict, score, ytrue) (interval, alpha)}`; 5-line core update + bookkeeping.
- Reference: Gibbs, I. & Candes, E. (2021). "Adaptive conformal inference under distribution shift." NeurIPS 34:1660-1672.

**C7 — `DtACI(stepSizeGrid)` Gibbs-Candes 2024** | ~180 LOC.
- "Dynamically-tuned ACI": run a finite grid of γ in parallel, compete via Hedge / EWA aggregation.
- Adapts step-size to the rate of distribution shift without manual tuning.
- Reference: Gibbs, I. & Candes, E. (2024). "Conformal inference for online prediction with arbitrary distribution shifts." JMLR 25(162):1-36.

**C8 — `LocallyAdaptiveSplitInterval(yhat, sigmaHat, calScores, alpha)` Lei-Wasserman 2014** | ~80 LOC.
- Heteroscedastic split-conformal: nonconformity `s_i = |y_i − ŷ_i| / σ̂(x_i)`; band `[ŷ − q·σ̂(x), ŷ + q·σ̂(x)]`.
- Already partially supported by `NormalizedResidual{StdDevFn}`; promote to first-class interval API.
- Reference: Lei, J. & Wasserman, L. (2014). "Distribution-free prediction bands for non-parametric regression." JRSS-B 76(1):71-96.

**C9 — `AsymmetricCqrInterval(qLo, qHi, calScores, alphaLo, alphaHi)` Sesia-Romano 2020** | ~80 LOC.
- Two-sided asymmetric CQR: separate calibration of lower-tail and upper-tail miscoverage at `αLo + αHi = α`.
- Strictly tighter than symmetric CQR for skewed errors.
- Reference: Sesia, M. & Romano, Y. (2020). "Conformal prediction using conditional histograms." NeurIPS 33.

**C10 — `MultiOutputConformalInterval(yhats, calScores, alpha, copula)` Messoudi-Destercke-Rousseau 2022** | ~160 LOC.
- Multi-target regression: joint coverage via copula-decomposed nonconformity.
- Reuses `prob/copula/Gaussian` for product-of-marginals; reuses `linalg/CholeskySolve` for joint Mahalanobis residuals.
- Reference: Messoudi, S., Destercke, S. & Rousseau, S. (2022). "Ellipsoidal conformal inference for multi-target regression." COPA 11:294-306.

### Tier 3 — Aggregating / CDF / APS-regression (~440 LOC, 1.5 days)

**C11 — `AggregatedConformalPredictor(predictors)` Carlsson-Eklund-Norinder 2014** | ~120 LOC.
- Combine M independent split-conformal predictors via median / p-value aggregation.
- Coverage `1 − α − M⁻¹` finite-sample (worst-case slack); efficiency-vs-validity sweet-spot.
- Reference: Carlsson, L., Eklund, M. & Norinder, U. (2014). "Aggregated conformal prediction." AIAI 8:231-240.

**C12 — `ConformalCDF(yhat, calScores, yGrid)` Vovk-Nouretdinov-Manokhin-Gammerman 2018** | ~140 LOC.
- Returns full conformal CDF (predictive distribution) over a y-grid: `F̂(y | x) = #{i : s_i ≥ s(x, y)} / (n+1)`.
- Composes with `prob/distribution.go-CDF` interface for uniform downstream API.
- Reference: Vovk, V., Nouretdinov, I., Manokhin, V. & Gammerman, A. (2018). "Cross-conformal predictive distributions." COPA 7:37-51.

**C13 — `APSRegressionInterval(quantilePredictor, calScores, alpha)`** Romano-Sesia-Candes 2020 | ~180 LOC.
- Adaptive Prediction Sets generalised to regression via histogram-binned predictive distribution.
- Reuses `calculus.CubicSpline` for smoothed CDF inversion.
- Reference: Romano, Y., Sesia, M. & Candes, E. (2020). "Classification with valid and adaptive coverage." NeurIPS 33:3581-3591 §4.

### Tier 4 — Classification (`prob/conformal/classify/` ~860 LOC, 2.5 days)

**C14 — `LACPredictor(probs, calScores, alpha)` Sadinle-Lei-Wasserman 2019** | ~100 LOC.
- "Least Ambiguous set-valued Classifier": nonconformity `s_i = 1 − π̂_{y_i}(x_i)`; smallest set with marginal coverage.
- Optimally smallest sets but loses class-conditional coverage on rare classes.
- Reference: Sadinle, M., Lei, J. & Wasserman, L. (2019). "Least ambiguous set-valued classifiers with bounded error levels." JASA 114(525):223-234.

**C15 — `APSPredictor(probs, calScores, alpha, randomized)` Romano-Sesia-Candes 2020** | ~160 LOC.
- Adaptive Prediction Sets: rank-cumulative-softmax nonconformity; class-conditional efficient.
- Randomised vs deterministic variants (randomised hits exact `1−α` coverage).
- Reference: Romano, Y., Sesia, M. & Candes, E. (2020). "Classification with valid and adaptive coverage." NeurIPS 33:3581-3591.

**C16 — `RAPSPredictor(probs, calScores, alpha, lambda, kReg)` Angelopoulos-Bates-Malik-Jordan 2021** | ~180 LOC.
- "Regularised APS": penalise large set-sizes via λ-regularisation.
- Smaller sets than APS at same coverage on ImageNet / CIFAR.
- Reference: Angelopoulos, A. N., Bates, S., Malik, J. & Jordan, M. I. (2021). "Uncertainty sets for image classifiers using conformal prediction." ICLR.

**C17 — `ClassConditionalPredictor(probs, labels, calScores, alpha)` Lei-Wasserman 2014** | ~140 LOC.
- Per-class Mondrian-style calibration; coverage `P(Y ∈ Ĉ(X) | Y = k) ≥ 1−α` for each k.
- Generalises existing `MondrianQuantile`; reuses ~70% of mondrian.go.
- Reference: Lei, J. (2014). "Classification with confidence." Biometrika 101(4):755-769.

**C18 — `ClusteredConformalPredictor(probs, embeddings, k, calScores, alpha)` Ding-Angelopoulos-Bates-Jordan-Malik 2023** | ~200 LOC.
- K-means clustering of softmax embeddings → data-driven Mondrian strata.
- Recovers class-conditional coverage when classes are too rare for direct C17.
- Reuses `linalg/-KMeans` (deferred — slot 184 substrate).
- Reference: Ding, T., Angelopoulos, A. N., Bates, S., Jordan, M. I. & Malik, J. (2023). "Class-conditional conformal prediction with many classes." NeurIPS 36.

**C19 — `BayesClassifierConformal(posteriors, alpha)` Lei 2014** | ~80 LOC.
- Conformal calibration of Bayes-classifier posteriors; cross-link slot-228 BNP for posterior construction.
- Reference: Lei, J. (2014) §4.

### Tier 5 — Risk control (`prob/conformal/risk/` ~720 LOC, 2 days)

**C20 — `ConformalRiskControl(lossFn, calLosses, alpha)` Angelopoulos-Bates-Cole-Greenfield-Sahoo-Snyder 2024** | ~180 LOC.
- Generalises coverage to arbitrary monotone bounded loss `L(Ĉ, Y) ∈ [0, B]`.
- Find `λ̂ = inf{λ : (1/(n+1)) · Σ L_i(λ) + B/(n+1) ≤ alpha}`; deploy `Ĉ_{λ̂}`.
- Cited applications: hallucination-rate control, mIoU bounds for segmentation, multi-label F1.
- **The 2024 ICLR oral.** Citation-engine of 2024-2026 deployable-LLM literature.
- Reference: Angelopoulos, A. N., Bates, S., Cole, A., Greenfield, A., Sahoo, B., Snyder, J. (2024). "Conformal risk control." ICLR.

**C21 — `MultiRiskControl(lossFns, calLosses, alpha)` Angelopoulos-Bates-Fisch 2022** | ~180 LOC.
- Simultaneous control over multiple risks via Bonferroni-conformal hybrid.
- Cross-link slot-230 F1 (Bonferroni) for multi-risk correction.
- Reference: Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., Schuster, T. (2022). "Conformal risk control via online learning." NeurIPS w/s.

**C22 — `LearnThenTest(hypothesisFamily, calLosses, alpha)` Angelopoulos-Bates-Candes-Jordan-Lei 2021** | ~200 LOC.
- Hyperparameter selection from finite hypothesis class with FWER-controlled risk guarantee.
- Cross-link slot-230 F3 (Holm) / F5 (Hommel) closed-testing.
- Reference: Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan, M. I., Lei, L. (2021). "Learn then test: calibrating predictive algorithms to achieve risk control." arXiv:2110.01052.

**C23 — `RiskControllingPredictionSets(setFamily, calLosses, alpha, delta)` Bates-Angelopoulos-Lei-Malik-Jordan 2021** | ~160 LOC.
- (α, δ)-RCPS: risk ≤ α with probability ≥ 1−δ over calibration draw.
- Stronger than CRC for safety-critical systems (medical / autonomous).
- Reference: Bates, S., Angelopoulos, A., Lei, L., Malik, J. & Jordan, M. (2021). "Distribution-free, risk-controlling prediction sets." JACM 68(6):1-34.

### Tier 6 — e-values + non-exchangeable + anomaly (`prob/conformal/evalue/` ~640 LOC, 2 days)

**C24 — `EValueConformal(scores, alpha)` Vovk-Wang 2022** | ~140 LOC.
- e-value formulation of conformal: replace p-values with `e_i = (1−s_i/q_α)_+`; combine via average for arbitrary-dependence FDR.
- **Cross-link slot-230 F22 (e-BH) — shared supermartingale substrate.** Joint extraction `prob/orderstats.go-RankAdjusted` ~60 LOC.
- Reference: Vovk, V. & Wang, R. (2022). "E-values for testing exchangeability." Bernoulli 28(3):2098-2128.

**C25 — `ConformalAnomaly(scores, threshold)` Laxhammar-Falkman 2014** | ~120 LOC.
- p-value-style anomaly detection: `p̂(x_test) = #{i : s_i ≥ s(x_test)} / (n+1)`; flag if `p̂ < α`.
- Calibration-set-only; no model required beyond the score function.
- Reference: Laxhammar, R. & Falkman, G. (2014). "Conformal anomaly detection of trajectories with a multi-class hierarchy." SLDS workshop.

**C26 — `EnsembleBatchPredictionInterval(predictors, X, y, xTest, alpha) (lo, hi)` Xu-Xie 2023** | ~200 LOC.
- "EnbPI": time-series conformal that bootstraps predictors and uses leave-one-out residuals; valid under non-exchangeability.
- **Cross-link `changepoint/bocpd.go`** for change-aware reset; cross-link slot-021/025 BOCPD reviews.
- Reference: Xu, C. & Xie, Y. (2023). "Conformal prediction for time series." TPAMI 45(10):11575-11587.

**C27 — `ConformalPACBayes(prior, posterior, calScores, alpha)` Hellström-Durisi 2024** | ~180 LOC.
- PAC-Bayes-conformal hybrid: bound coverage via KL divergence between deployed and calibration posteriors.
- Cross-link `info/-KL` and slot-228 BNP posteriors.
- Reference: Hellström, F. & Durisi, G. (2024). "Comparing comparators in generalization bounds." AISTATS 27.

---

## (3) Cheapest day-one shippable artifact (Tier-1 mini-PR)

**PR-1 — C1+C2+C3 jackknife+ / CV+ / full-conformal-LOO** ~440 LOC source, ~600 LOC tests, 1.5 days.

```go
// prob/conformal/jackknife.go
package conformal

// Predictor is the abstraction the deferred-v2 quartet trains on each fold.
type Predictor interface {
    Fit(X [][]float64, y []float64) error
    Predict(x []float64) float64
}

func JackknifePlusInterval(p Predictor, X [][]float64, y []float64, xTest []float64, alpha float64) (lo, hi float64, err error) // C1
func CVPlusInterval(p Predictor, X [][]float64, y []float64, xTest []float64, alpha float64, K int) (lo, hi float64, err error) // C2
func FullConformalInterval(p Predictor, X [][]float64, y []float64, xTest []float64, alpha float64, yGrid []float64) (lo, hi float64, err error) // C3
func CrossConformalInterval(p Predictor, X [][]float64, y []float64, xTest []float64, alpha float64, K int) (lo, hi float64, err error) // C4
```

This single PR closes the entire `doc.go:86-92` deferred-v2 list, lifts the package from "split + variants" to "the full Vovk-Gammerman-Shafer-2005 textbook canon", and pins against Python `crepes` (Boström 2022 COPA) at ≤1e-12 over a 30-vector deterministic regression test corpus. The `Predictor` interface is the strictly-minimal contract — `prob/regression.go-LinearRegression` adapts in <20 LOC.

The C4 cross-conformal is a 30-min add-on once C2 ships (k-fold averaging instead of order-statistic aggregation).

---

## (4) Single-highest-leverage cutting-edge piece

**C5 + C6 — Weighted Conformal (Tibshirani-Barber-Candes-Ramdas 2019 NeurIPS) + ACI (Gibbs-Candes 2021 NeurIPS)** ~220 LOC.

Why these two together: the existing `AdaptiveQuantile` solves the *data-side* of distribution shift (down-weight stale samples). The 2019/2021 frontier solves the *inference-side*: weighted-conformal handles known covariate shift via importance weights with finite-sample guarantee; ACI handles arbitrary shift via online α-adjustment with long-run convergence guarantee. The two compose: weighted-conformal gives the *current-step* interval, ACI corrects the *long-run* miscoverage, and `AdaptiveQuantile` provides the recency-weighted residual stream feeding both. Together they are the canonical "deployable under shift" triple.

**Reuse:**
- `AdaptiveQuantile`'s weighted-quantile loop (already shipped, ~60 LOC) feeds `WeightedSplitInterval` directly
- `prob/markov.go` for ACI state-machine pattern
- slot-229 PR-0 GLM `prob/regression/glm.go` for likelihood-ratio importance weights
- `optim/transport/sinkhorn.go` for OT-based covariate-shift weights (alternative to logistic regression LR estimator)

**No zero-dep Go alternative ships.** Python `mapie` (Cordier-Blot-Lacombe-Morzadec-Capitaine-Brunel 2023 JOSS) and Python `crepes` are mainstream; both depend on heavy SciPy/NumPy/scikit-learn. R `conformalInference` is research-quality. reality C5+C6 is the first Go zero-dep implementation of the weighted-conformal-plus-ACI canonical pair.

---

## (5) Single-highest-leverage moat

**C20 — Conformal Risk Control (Angelopoulos-Bates-Cole-Greenfield-Sahoo-Snyder 2024 ICLR oral)** ~180 LOC.

**Why a moat:** CRC is the 2024 generalisation that takes conformal from coverage-only (`P(Y ∈ Ĉ(X)) ≥ 1−α`) to *bounded-loss* control (`E[L(Ĉ(X), Y)] ≤ α` for any monotone bounded `L`). The cited applications — hallucination-rate control, mIoU bounds for segmentation, F1 lower-bounds for multi-label classification — are the citation-engine of 2024-2026 deployable-LLM and computer-vision literature.

CRC is the **one** conformal primitive that takes the framework from "academic statistics" to "ship in production safety-critical systems". Every 2024+ LLM-uncertainty paper cites it. No zero-dep Go library ships it. The Python reference is `torchcp` (Wei-Zhao-Yu-Bates-Wasserman 2024); reality C20 is the first Go zero-dep implementation.

**Reuse:**
- `info/-KL` for KL-loss bounds
- existing `SplitQuantile` for the calibrated `λ̂` threshold computation
- slot-230 F22 e-value substrate for the supermartingale machinery

**Cross-link slot-222 bandits:** RCPS (C23) shares the (α, δ) PAC-style guarantee with best-arm-identification stopping rules; matched moat substrate.

---

## (6) Connective tissue / cross-package leverage

| LOC | Edge | Reason |
|-----|------|--------|
| ~60 | `prob/orderstats.go-RankAdjusted` ↔ slot-230 F8 q-value + `SplitQuantile` | Shared rank-`(n+1)(1−α)` finite-sample correction; lift once for both reviews |
| ~80 | `prob/conformal/-JackknifePlusInterval` ↔ `linalg/CholeskySolve` Sherman-Morrison | Hat-matrix LOO-residual updates avoid n model refits for ridge/OLS |
| ~50 | `prob/conformal/-WeightedSplit` ↔ `optim/transport/sinkhorn.go` | OT-based covariate-shift importance weights |
| ~40 | `prob/conformal/-AdaptiveConformalInference` ↔ `prob/markov.go` | One-state-α stochastic-approximation state machine |
| ~60 | `prob/conformal/classify/-APS` ↔ `info/-Entropy` | Softmax-cumulative-rank score is entropy-related |
| ~40 | `prob/conformal/risk/-CRC` ↔ slot-230 F22 e-value | Shared supermartingale substrate |
| ~50 | `prob/conformal/-MultiOutput` ↔ `prob/copula/-Gaussian` | Joint-coverage via copula-decomposed nonconformity |
| ~30 | `prob/conformal/evalue/-EnbPI` ↔ `changepoint/bocpd.go` | BOCPD-aware reset for non-exchangeable streams |
| ~60 | `prob/conformal/-LocallyAdaptive` ↔ `prob/distributions.go-NormalQuantile` + `calculus/-CubicSpline` | Kernel σ̂(x) estimation |
| ~30 | `prob/conformal/-CqrAsymmetric` ↔ existing `CqrInterval` | Two-sided extension; share quantile loop |

Total cross-package glue: **~500 LOC** — small relative to ~3,640 LOC primitive surface but architecturally load-bearing. The single ~60-LOC `prob/orderstats.go-RankAdjusted` extraction is the *only* cross-cut shared with slot 230; clean separation.

---

## (7) Cross-language pinning targets

| Test | Tolerance | Reference impl |
|------|-----------|----------------|
| `TestJackknifePlus_crepes_1e-12` | 1e-12 | Python `crepes.WrapRegressor.predict(...)` (Boström 2022) |
| `TestCVPlus_crepes_1e-12` | 1e-12 | Python `crepes.WrapRegressor.predict_set` K=10 |
| `TestFullConformal_conformalInferenceR_1e-9` | 1e-9 | R `conformalInference::conformal.pred` (Lei-G'Sell-Tibshirani) |
| `TestWeightedSplit_mapie_1e-9` | 1e-9 | Python `mapie.MapieRegressor(method="weighted")` (Cordier 2023) |
| `TestACI_acipy_1e-9` | 1e-9 | Python `aci` (Gibbs-Candes 2021 ref code) |
| `TestAPS_torchcp_1e-9` | 1e-9 | Python `torchcp.classification.predictor.SplitPredictor` (Wei 2024) |
| `TestRAPS_torchcp_1e-9` | 1e-9 | Python `torchcp` `RAPSPredictor` |
| `TestLAC_torchcp_1e-9` | 1e-9 | Python `torchcp` `LACPredictor` |
| `TestCRC_torchcp_1e-9` | 1e-9 | Python `torchcp.classification.utils.crc` |
| `TestEnbPI_xuxie_1e-6` | 1e-6 | Python ref code (Xu-Xie 2023) |
| `TestEValueConformal_vovkR_1e-9` | 1e-9 | R reference code (Vovk-Wang 2022) |
| `TestConformalAnomaly_alibi_1e-9` | 1e-9 | Python `alibi-detect.cd.cvm` |

Twelve pins; three at exact 1e-12 for the Tier-1 deferred-v2 quartet (these are deterministic order-statistic computations); nine at 1e-9 / 1e-6 for the frontier pieces.

R-MUTUAL-CROSS-VALIDATION-3/3 candidates:
- **Triple-LOO:** jackknife+ vs CV+(K=n) vs full-conformal — at K=n CV+ degenerates to jackknife+ exactly; full-conformal recovers the same interval at infinite y-grid; 3-way agreement structural.
- **Triple-shift:** weighted-conformal-with-uniform-weights vs vanilla split-conformal vs ACI-with-α₀=α — all three reduce to identical intervals at no-shift; deviations on shifted streams isolate weighting source.
- **Triple-classification:** APS-with-randomization=false vs LAC vs class-conditional-Mondrian — all three reduce to size-1 sets when `π̂_y(x) > 1−α`; 3-way agreement on confident samples.

---

## (8) Landing order (10 PRs, ~13 engineer-days)

| PR | Scope | LOC | Days | Cross-cutting unblocks |
|----|-------|-----|------|------------------------|
| PR-0 | `prob/random.go` Box-Muller + bootstrap RNG (slots 117/184/188/202/215/216/217/227/228/229/230 shared) | 280 | 1 | **TWELFTH** Block-C demand for this — universal cross-cutting blocker |
| PR-1 | C1+C2+C3+C4 deferred-v2 quartet (jackknife+/CV+/full-LOO/cross-conformal) | 540 | 1.5 | Closes `doc.go:86-92` v2-defer list |
| PR-2 | C5+C6 weighted-conformal + ACI | 220 | 1 | **Cutting-edge piece**; deployable-under-shift canonical pair |
| PR-3 | C7+C8+C9+C10 DtACI + locally-adaptive + asymmetric-CQR + multi-output | 500 | 2 | Frontier extensions |
| PR-4 | C11+C12+C13 aggregating + conformal-CDF + APS-regression | 440 | 1.5 | Distribution / aggregation branch |
| PR-5 | C14+C15+C16+C17+C18+C19 `prob/conformal/classify/` | 860 | 2.5 | Classification sub-package |
| PR-6 | C20 conformal risk control (CRC) | 180 | 1 | **Moat** |
| PR-7 | C21+C22+C23 `prob/conformal/risk/` (multi-risk + LtT + RCPS) | 540 | 1.5 | Risk-control sub-package completion |
| PR-8 | C24+C25 e-value-conformal + conformal-anomaly | 260 | 1 | e-value bridge to slot 230 |
| PR-9 | C26+C27 EnbPI + conformal-PAC-Bayes | 380 | 1 | Time-series + PAC-Bayes frontier |

Defer-to-v3: deep-conformal (Romano-Bates-Candes 2020 §6 deep-residual networks) — composes slot-237 deep-learning territory; conformal-bandits (Candes-Lei-Ren 2024) — composes slot-222 bandits; functional-output-conformal (Diquigiovanni-Fontana-Vantini 2022) — composes slot-247 functional-data territory.

---

## (9) Differentiation vs adjacent reviews

- **Slot 117 prob-missing:** lists "conformal" as one bullet noting the existing package; slot 231 owns the canon end-to-end including the four sub-packages and 26 primitives.
- **Slot 118/119 prob-sota/api:** `119-prob-api.md:172-200` notes conformal needs no `Distribution` abstraction (correct — it is distribution-free); slot 231 preserves that and explicitly does *not* introduce distribution coupling.
- **Slot 230 FDR/knockoffs:** shared substrate is the rank-`(n+1)(1−α)` order-statistic and the e-value supermartingale machinery; one ~60-LOC `prob/orderstats.go-RankAdjusted` extraction lands once for both. C24 (e-value-conformal) ↔ slot-230 F22 (e-BH) ↔ shared. No primitive overlap; all 26 unique to slot 231.
- **Slot 222 bandits:** C23 RCPS shares (α, δ) PAC-style guarantee; clean cross-link not duplication.
- **Slot 227 UQ:** Sobol indices unrelated to conformal; pure orthogonality.
- **Slot 228 BNP:** C19 BayesClassifierConformal cross-links to BNP posteriors; C27 conformal-PAC-Bayes cross-links to slot 228 `KL`. No primitive duplication.
- **Slot 229 causal:** independent — slot-229 owns Pearl/Rubin; slot-231 owns conformal. Cross-link only via slot-229 PR-0 `prob/random.go` and `prob/regression/glm.go` (used by C5 weighted-conformal LR-estimator).
- **Slot 021/025 changepoint-numerics/perf:** C26 EnbPI cross-links to `changepoint/bocpd.go` for non-exchangeable time-series resets; clean substrate-shared, no scope overlap.
- **Slot 165 sequence-prob synergy:** likely covers permutation tests; orthogonal — conformal does *not* permute.
- **Slot 193 prob-changepoint synergy:** explicit cross-link target for C26 EnbPI; slot 231 owns conformal-side, slot 193 owns changepoint-side.

26/26 primitives are unique to slot 231. Cross-cutting blocker `prob/random.go` is the single overlap and is correctly scoped as Tier-0 cross-cutting infrastructure (twelve Block-C reviews demand it).

---

## (10) Single-line architectural witness

The existing `prob/conformal/doc.go:86-92` *names* full / jackknife+ / CV+ / online-ACI as "Deferred to v2" — and 2026-05-08 is v2. Slot 231 is the package author's own promised follow-up, plus the post-2021 frontier (CRC / APS+RAPS / EnbPI) the original 2024 author could not have anticipated. The package is the rare reality sub-package that arrives at this review with a *self-authored deferred-work list pinned to its own doc comment*; slot-231 closes that list and grows the package to the 2024-2026 conformal-prediction canon.

---

## References (selected canon)

1. Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer. **(framework)**
2. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. & Wasserman, L. (2018). "Distribution-free predictive inference for regression." JASA 113:1094-1111. **(SHIPPED — split conformal)**
3. Romano, Y., Patterson, E. & Candes, E. (2019). "Conformalized quantile regression." NeurIPS 32. **(SHIPPED — CQR)**
4. Lei, J. & Wasserman, L. (2014). "Distribution-free prediction bands for non-parametric regression." JRSS-B 76(1):71-96.
5. Vovk, V. (2015). "Cross-conformal predictors." Annals of Mathematics and AI 74(1-2):9-28.
6. Tibshirani, R. J., Barber, R. F., Candes, E. J. & Ramdas, A. (2019). "Conformal prediction under covariate shift." NeurIPS 32.
7. Romano, Y., Sesia, M. & Candes, E. (2020). "Classification with valid and adaptive coverage." NeurIPS 33:3581-3591.
8. Sadinle, M., Lei, J. & Wasserman, L. (2019). "Least ambiguous set-valued classifiers with bounded error levels." JASA 114(525):223-234.
9. Bates, S., Angelopoulos, A., Lei, L., Malik, J. & Jordan, M. (2021). "Distribution-free, risk-controlling prediction sets." JACM 68(6):1-34.
10. Barber, R. F., Candes, E. J., Ramdas, A. & Tibshirani, R. J. (2021). "Predictive inference with the jackknife+." Annals of Statistics 49(1):486-507.
11. Gibbs, I. & Candes, E. (2021). "Adaptive conformal inference under distribution shift." NeurIPS 34:1660-1672.
12. Angelopoulos, A. N., Bates, S., Malik, J. & Jordan, M. I. (2021). "Uncertainty sets for image classifiers using conformal prediction." ICLR.
13. Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan, M. I., Lei, L. (2021). "Learn then test: calibrating predictive algorithms to achieve risk control." arXiv:2110.01052.
14. Vovk, V. & Wang, R. (2022). "E-values for testing exchangeability." Bernoulli 28(3):2098-2128.
15. Messoudi, S., Destercke, S. & Rousseau, S. (2022). "Ellipsoidal conformal inference for multi-target regression." COPA 11:294-306.
16. Boström, H. (2022). "crepes: a Python package for generating conformal regressors and predictive systems." COPA 11.
17. Cordier, T., Blot, V., Lacombe, L., Morzadec, T., Capitaine, A. & Brunel, N. (2023). "Flexible and systematic uncertainty estimation with conformal prediction via MAPIE." JOSS 8(85):4928.
18. Xu, C. & Xie, Y. (2023). "Conformal prediction for time series." TPAMI 45(10):11575-11587.
19. Ding, T., Angelopoulos, A. N., Bates, S., Jordan, M. I. & Malik, J. (2023). "Class-conditional conformal prediction with many classes." NeurIPS 36.
20. Angelopoulos, A. N. & Bates, S. (2023). "Conformal prediction: a gentle introduction." Foundations and Trends in Machine Learning 16(4):494-591.
21. Angelopoulos, A. N., Bates, S., Cole, A., Greenfield, A., Sahoo, B. & Snyder, J. (2024). "Conformal risk control." ICLR.
22. Gibbs, I. & Candes, E. (2024). "Conformal inference for online prediction with arbitrary distribution shifts." JMLR 25(162):1-36.
23. Hellström, F. & Durisi, G. (2024). "Comparing comparators in generalization bounds." AISTATS 27.
24. Wei, S., Zhao, Q., Yu, Y., Bates, S. & Wasserman, L. (2024). "torchcp: a library for conformal prediction in PyTorch." arXiv:2402.12683.
25. Carlsson, L., Eklund, M. & Norinder, U. (2014). "Aggregated conformal prediction." AIAI 8:231-240.
26. Vovk, V., Lindsay, D., Nouretdinov, I. & Gammerman, A. (2003). "Mondrian Confidence Machine." Tech. Report. **(SHIPPED — Mondrian)**
27. Sesia, M. & Romano, Y. (2020). "Conformal prediction using conditional histograms." NeurIPS 33.
28. Vovk, V., Nouretdinov, I., Manokhin, V. & Gammerman, A. (2018). "Cross-conformal predictive distributions." COPA 7:37-51.
29. Laxhammar, R. & Falkman, G. (2014). "Conformal anomaly detection of trajectories with a multi-class hierarchy." SLDS workshop.
30. Lei, J. (2014). "Classification with confidence." Biometrika 101(4):755-769.
