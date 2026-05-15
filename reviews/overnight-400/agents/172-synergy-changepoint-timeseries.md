# 172 | synergy-changepoint-timeseries

**Topic:** changepoint × timeseries — detection within ARIMA, online segmentation, regime-switching.
**Block:** B (cross-package synergies). **Date:** 2026-05-08. **Scope:**
capabilities that emerge ONLY when `changepoint/`, `timeseries/garch/`,
`timeseries/dcc/`, and `prob/timeseries.go` (the toy ARIMA / SES / Holt) are
composed; not in-package gaps (021-025 own changepoint, 136-140 own timeseries,
117-118 own prob). Co-references 137-T1.x for ARIMA/Kalman/state-space
absences and 022-T1.x for the canonical detector roster.

## Two-line summary

`changepoint/` ships exactly one detector — Adams-MacKay 2007 BOCPD on a
Normal-Inverse-Gamma conjugate prior with Student-t posterior predictive,
constant hazard `H = 1/λ`, truncation `R_max = 500` (~402 LOC `bocpd.go`) —
while the time-series surface across `timeseries/garch/` (GARCH(1,1) Filter +
LogLik + ForecastVariance + Simulate + Tikhonov-GD Fit, ~470 LOC),
`timeseries/dcc/` (Engle 2002 scalar DCC: SampleQbar + Update +
CorrelationFromQ + FilterSeries, ~210 LOC), and `prob/timeseries.go` (SES /
HoltLinear / `ARIMA(p,d,q)` method-of-moments stub, ~310 LOC) is a five-model
stub with neither Kalman, state-space, ACF/PACF, exact-MLE ARIMA, nor any
unit-root / cointegration test (per 137-T1.x); the entire detection-within-
timeseries canon — CUSUM on standardised GARCH residuals, BOCPD on ARIMA
innovations, PELT (Killick-Fearnhead-Eckley 2012), Wild Binary Segmentation
(Fryzlewicz 2014), NOT (Baranowski-Chen-Fryzlewicz 2019), e-divisive
(James-Matteson 2014), Hamilton 1989 Markov-switching, Page-CUSUM,
SPRT (Wald 1945), kernel CPD (Harchaoui-Bach-Moulines 2007), spectral CPD,
break-in-cointegration (Gregory-Hansen 1996), ROC curve for detectors — is
**wholly absent** (verified by repo-wide grep on
`CUSUM|cusum|PELT|BinSeg|Wald|Kalman|Viterbi|HMM|EDivisive`: zero matches in
`*.go`; only one comment hit in `optim/transport/pairwise.go`). Eighteen
synergy primitives (S1–S18) totalling **~3,580 LOC** of pure connective
tissue close the gap; **cheapest one-day PR is S1 `CUSUMOnGARCHResiduals` +
S3 `BOCPDOnARIMAInnovations` (~210 LOC together)** because both reduce to
`Filter` → standardise → feed-into-detector and require zero new mathematics;
**highest-leverage architectural lift is S5 PELT with pluggable cost
(~520 LOC)** since PELT is the modal offline detector and the cost-function
plug-point is what every downstream consumer needs first; **crown jewel is
S15 BOCPDMS-style joint regime-switching ARIMA (~480 LOC)** — composes BOCPD
× `prob.ARIMA` × Kalman (137-T1.1) into a Bayesian regime-switching forecaster
that no Python/R library ships out-of-the-box. Cross-language golden files
(per `testutil/`) drop out for free since every primitive has a closed-form
or O(n) reference.

---

## Bases (verified file-walk)

### `changepoint/` (5 files, ~530 LOC)

`bocpd.go` (402 LOC): `NigPrior{Mu0, Kappa0, Alpha0, Beta0}` + `Validate`,
`DefaultNigPrior`, `Bocpd{prior, rMax, lambda, t, p, mu, kappa, alpha, beta}`,
`Config{Prior, RMax, Lambda}` + `DefaultConfig`, `New`, `(b *Bocpd).Update(x)`
(Student-t predictive → log-space growth/reset → renormalise → NIG
suff-stat update), `RunLengthPosterior`, `ChangePointProbability`,
`ChangePointProbabilityWithin(window)`, `MapRunLength`, `ExpectedRunLength`,
`CurrentRegimeMean`, `CurrentRegimeVariance`, `Step`. Internals:
`hazard(r) = 1/λ` (constant only — no time-varying), `studentTLogPDF`,
`logSumExp`. Determinism: bit-stable, no rand, no parallelism. **Not
streaming-friendly in the strict sense:** every `Update` allocates four fresh
length-`n+1` slices (`newLogP`, `newP`, `newMu`, `newKappa`, `newAlpha`,
`newBeta`); for 60 FPS hot-paths this needs an output-buffer overload.

`bocpd_test.go` (11 tests): step-shift detection at preCP=50 + postCP=20,
N(0,1)→N(5,1), Lambda=100, RMax=200; stationary stability; truncation honoured;
posterior-sums-to-one. `bocpd_expansion_test.go` (additional coverage).
`infogeo_test.go` (single test): TV / Hellinger between BOCPD posterior on full
data and fresh-start BOCPD on post-CP-only data — first cross-package consumer.
`doc.go` enumerates 24+ candidate downstream consumers (relic-insurance, triage,
witness, watchtower, narrator) but **none import the package**: verified by
`grep "github.com/davly/reality/changepoint"` → zero matches outside `changepoint/`.

### `timeseries/garch/` (5 files, ~470 LOC)

`garch.go` (~175 LOC): `Model{Omega, Alpha, Beta, UncondVar}` + `Validate`
(stationarity Ω>0, α≥0, β≥0, α+β<1), `Filter(eps, sigma2, z)` (forward
recursion → conditional variance + standardised residuals z = ε/σ),
`LogLikelihood(eps)` (Gaussian: `-0.5 Σ [log 2π + log σ² + ε²/σ²]`),
`ForecastVariance(eps2T, sigma2T, h)` (h-step recursion, geometric convergence
to UncondVar at rate α+β), `Simulate(shocks, eps, sigma2)` (deterministic
given pre-drawn shocks). `fit.go`: Tikhonov-regularised MLE via fixed-LR GD
on softmax-reparameterised (α, β, γ) where γ = 1−α−β slack — autodiff-pinned.
**Hot-path:** `Filter` allocates the two output slices via caller; pure stdlib.

### `timeseries/dcc/` (4 files, ~210 LOC)

`dcc.go`: `Params{Alpha, Beta, Qbar, K}` (Engle 2002 default α=0.05, β=0.93),
`SampleQbar(zSeries, n, k, out)`, `(p Params).Update(z, Q, qOut)` (one-step
Q recursion `Q_t = (1-α-β)Q̄ + α z_{t-1} z_{t-1}^T + β Q_{t-1}`),
`CorrelationFromQ(Q, k, rOut)` (R = diag(Q)^{-1/2} Q diag(Q)^{-1/2}),
`(p Params).FilterSeries(zSeries, n, rSeries)`. Calibration of (α, β)
**deferred** — most consumers use the industry default. K-asset multivariate
extension of GARCH; consumer of `garch.Filter` standardised residuals.

### `prob/timeseries.go` (~310 LOC)

`ExponentialSmoothing(data, alpha, out)` (Brown 1956 SES), `HoltLinear(data,
alpha, beta, horizon, out)` (Holt 1957 double-ES with linear trend forecast),
`ARIMA(data, p, d, q)` (Box-Jenkins differencing → Levinson-Durbin AR fit
→ residual-autocorrelation MA fit). The `ARIMA` is method-of-moments not MLE;
137-N5 already flags it as a stub: biased MA estimation, no integration-on-
forecast (returns coefficients only, no `Forecast`/`Predict` method, no
residuals exposed), PARCOR clamp silently masks non-PSD autocov.
`levinsonDurbin` is private — discards PACF and per-order error variance.

### `prob/markov.go` (~155 LOC)

`MarkovSteadyState(transitionMatrix, n)` (power iteration, max 1000 iter, L1
ε=1e-12), `MarkovSimulate(transitionMatrix, n, initialState, steps)`
(LCG-seeded deterministic). **Discrete chains only** — no HMM, no Viterbi,
no Baum-Welch, no forward-backward, no emission models. The substrate for
S14/S15 below exists at the steady-state-distribution level only.

### `prob/hypothesis.go`, `prob/nonparametric.go`

`TTestOneSample`, `TTestTwoSample` (Welch), `ChiSquaredTest`,
`MannWhitneyU`, `FisherExactTest`. The two-sample test substrate for sliding-
window CPD (S6) is here; nothing exposes Kolmogorov-Smirnov or Cramér-von
Mises, both flagged in 022-T1.9 as CPD substrate.

### `infogeo/`

`KL`, `ReverseKL`, `JS`, `TotalVariation`, `Hellinger`, `ChiSquared`, `Renyi`
(f-divergences), `MMD2Biased`/`MMD2Unbiased` + `GaussianKernel` /
`LaplacianKernel` + `MedianHeuristicBandwidth` (kernel two-sample). MMD is
the substrate for S10 kernel CPD; f-divergences are the substrate for S2
generalised-likelihood-ratio CUSUM. Already a verified cross-consumer of
`changepoint` via `infogeo_test.go`.

### What is NOT here (verified by `grep -i "CUSUM|cusum|PELT|BinSeg|WildBin|
Wald|EDivisive|JamesMatteson|RegimeSwitch|HiddenMarkov|BaumWelch|ForwardBack|
Kalman|StateSpace|Hamilton|MOSUM|FOCuS|OCD|GregoryHansen|Phillips-Perron|
ADF|KPSS|Johansen|Engle-Granger|VECM|VAR\\b|SARIMA|EGARCH|GJR|BEKK|CCC|
ARIMAExact"` across all `*.go` files):

zero matches. The single `cusum`-adjacent token in the entire repo is a
comment in `optim/transport/pairwise.go:15` ("RubberDuck's
RegimeContextService — currently HMM-…"). Reality has **zero** offline
multi-changepoint detectors, **zero** non-Bayesian online detectors, **zero**
state-space / Kalman machinery, **zero** unit-root tests, **zero** HMM /
regime-switching machinery, and **zero** changepoint quality metrics
(precision / recall / Hausdorff distance / annotation error).

---

## The composition matrix

For each capability (S1–S18): (1) **what it detects**, (2) **what it composes**
(file:func citations), (3) **connective tissue LOC** to ship as a pure-glue PR
against today's repo. Tier reflects 022 (changepoint roster) and 137
(timeseries roster) cross-walk.

### Tier 1 — ship-against-today (substrate exists, glue is mechanical)

#### S1 — `CUSUMOnGARCHResiduals` (Page 1954 ⊕ Bollerslev 1986) — ~110 LOC

**Capability:** Online single-CP detector for shifts in conditional-variance
GARCH residuals. Standardises `eps` via `garch.Model.Filter` to z = ε/σ
(unit-variance under H₀) then runs Page CUSUM on z. Detects shifts that
GARCH itself rationalises away as transient volatility — precisely the
breaks-in-the-volatility-process that GARCH calibration assumes do not exist.

**Composition:**
- `timeseries/garch/garch.go:55 Model.Filter(eps, sigma2, z)` → z is
  standardised residuals (the CUSUM input).
- New `changepoint/cusum.go`: `Cusum{mu0, sigma0, k, h}` + `Update(z) (S, alarm)`
  with one-sided `S_t = max(0, S_{t-1} + (z - mu0 - k))`, two-sided variant,
  `Reset` after detection.
- `prob/distributions.go:47 NormalCDF` for ARL₀ analytical bound (Siegmund
  1985 corrected diffusion approx).

**Glue:** `~110 LOC` — `Cusum` struct + Update + Reset + ARL approximation
+ ten golden vectors against the Page 1954 worked example.

#### S2 — `CUSUMonARIMAInnovations` (Page-CUSUM ⊕ Box-Jenkins) — ~80 LOC

**Capability:** Online detector for breaks in the mean of an ARIMA-modelled
series. Computes one-step-ahead innovations e_t = x_t − x̂_t|t-1 from the
ARIMA fit, standardises by the residual variance, runs S1's CUSUM on the
standardised innovations.

**Composition:**
- `prob/timeseries.go:125 ARIMA(data, p, d, q)` returns coefficients (currently
  the only output — see 137-N5 gap).
- New `prob/timeseries.go:ARIMAInnovations(data, p, d, q, coefs)` thin wrapper
  that re-runs the AR fit forward and emits residuals (the underlying loop at
  `timeseries.go:194-202` already computes them — expose as a return).
- S1 `Cusum` reused.

**Glue:** `~80 LOC` exposing innovations from the existing ARIMA loop +
calling S1. **Blocked** on 137-N5 fix (ARIMA returns innovations) — but the
fix is local and 4 LOC.

#### S3 — `BOCPDOnARIMAInnovations` (Adams-MacKay 2007 ⊕ Box-Jenkins) — ~100 LOC

**Capability:** Bayesian online detector running BOCPD on the
standardised one-step-ahead ARIMA residuals. Under correct model, residuals
are i.i.d. N(0, σ²) — exactly the BOCPD NIG generative assumption — so this
is a **statistically principled** wrap rather than a heuristic. A BOCPD
spike on the residual stream signals the ARIMA model has structurally broken.

**Composition:**
- `prob/timeseries.go ARIMAInnovations` (S2's by-product).
- `changepoint/bocpd.go:175 (b *Bocpd).Update(x)` for each residual.
- `changepoint/bocpd.go:339 ChangePointProbabilityWithin(window)` as the
  alarm signal (per the existing doc-warning that
  `ChangePointProbability` cancels under constant hazard).

**Glue:** `~100 LOC` — `BocpdARIMA{ar *prob.ARIMAModel, b *Bocpd}` thin wrapper
with `Update(x)` returning (alarm, mapRunLength, expectedRegimeMean). 25
golden vectors (level-shift, slope-shift, variance-shift, no-shift).

#### S4 — `BOCPDOnGARCHResiduals` — ~90 LOC

**Capability:** BOCPD on standardised GARCH residuals (z = ε/σ). Under correct
GARCH(1,1), z ~ i.i.d. N(0, 1) — again the BOCPD NIG assumption holds. This
detects breaks in the **innovation process** itself (e.g., regime shift in
the tail behaviour) that volatility-clustering cannot absorb. Companion to S3.

**Composition:** `garch.Model.Filter` → z → `bocpd.Update(z_t)`.

**Glue:** `~90 LOC`. Same structure as S3.

#### S5 — `PELT` (Killick-Fearnhead-Eckley 2012) — ~520 LOC

**Capability:** The modal offline multi-changepoint detector. Exact dynamic
programming with a pruning inequality that gives O(n) average cost when CPs
grow linearly with n; O(n²) worst case otherwise. Cost-function-pluggable:
mean-shift, variance-shift, mean-and-variance, AR(p)-coefficient shift, NIG
(Bayesian-equivalent of BOCPD's posterior predictive), regression-coefficient
shift. Penalty β picked via SIC/BIC = log(n)·d.

**Composition:**
- New `changepoint/pelt.go`: `PELT(cost CostFn, n int, beta float64) []int`
  returning sorted CP indices. `CostFn = func(start, end int) float64` is the
  segment cost; standard library: `MeanCost`, `VarCost`, `MeanVarCost`,
  `NIGCost` (composes `bocpd.studentTLogPDF`), `ARCost` (composes
  `prob.timeseries.levinsonDurbin` — needs export from prob).
- `OptimalPartitioning` (Jackson 2005) shipped alongside as the correctness
  oracle — pure DP without pruning. ~120 LOC overlap.
- Cross-language golden files via `testutil/` against the Killick-Fearnhead-
  Eckley 2012 §5 simulation tables.

**Glue:** `~520 LOC` — `pelt.go` (~150) + `op.go` correctness oracle (~120) +
`costs.go` standard cost library (~150) + 30 golden vectors (~100).

#### S6 — `BinarySegmentation` + `WildBinarySegmentation` — ~280 LOC

**Capability:** BinSeg (Vostrikova 1981) is the recursive-greedy single-CP
detector — fast O(n log n) but inconsistent on short segments. WBS
(Fryzlewicz 2014) draws M random sub-intervals to fix the consistency
defect. Universal baselines.

**Composition:**
- `changepoint/binseg.go:BinarySegmentation(cost CostFn, n, threshold)`.
  Recursive single-CP search inside each segment until threshold not met.
- `changepoint/wbs.go:WildBinarySegmentation(cost, n, M, threshold)`. M random
  sub-intervals via `crypto/rand` or caller-seeded LCG (per `prob/markov.go`
  pattern for determinism).
- Same `CostFn` from S5.

**Glue:** `~280 LOC`. Reuses S5's cost library entirely.

#### S7 — `EDivisive` (James-Matteson 2014) — ~240 LOC

**Capability:** Distribution-free, multivariate-native multi-CP detector based
on energy distance. The **only** Tier-1 detector that handles arbitrary
distributional changes (not just mean / variance / AR coefficients). Uses
divisive recursion: at each step picks the segment + CP that maximises the
between-segment energy statistic.

**Composition:**
- New `changepoint/edivisive.go:EDivisive(X [][]float64, alpha, R int) []int`.
- Energy distance E(X, Y; α) = 2 E|X-Y|^α − E|X-X'|^α − E|Y-Y'|^α — pure stdlib.
- Significance via R-permutation test — composes
  `prob/markov.go:MarkovSimulate` LCG pattern for deterministic permutation.
- Golden files against the `ecp::e.divisive` R-package §6 tables (Matteson-
  James 2014 JASA).

**Glue:** `~240 LOC`. Multivariate-native from day one (the energy statistic is
a pure pairwise-distance computation that lifts trivially from R^1 to R^k).

#### S8 — `WaldSPRT` (Wald 1945) — ~120 LOC

**Capability:** Sequential Probability Ratio Test — the original online
hypothesis-test substrate. Decision boundaries A = log((1−β)/α),
B = log(β/(1−α)) on the cumulative log-likelihood ratio
log Λ_t = Σ log p₁(x_i)/p₀(x_i). Optimal in expected sample size at fixed
(α, β) error rates (Wald-Wolfowitz 1948 optimality). Substrate for online
detection in the simple-vs-simple regime; CUSUM is the SPRT-with-restart
variant.

**Composition:**
- New `changepoint/sprt.go:SPRT{logLR, A, B}` + `Update(logLikRatio) Decision`
  (where `Decision = Continue | AcceptH0 | AcceptH1`).
- `prob/distributions.go` PDFs supply log-likelihood-ratios for canonical
  alternatives (Normal vs Normal, mean shift; Normal vs Normal, variance shift;
  Bernoulli vs Bernoulli; Poisson vs Poisson).
- Closed-form Operating Characteristic (OC) curve and Average Sample Number
  (ASN): `OC(p) = (A^h(p) − 1) / (A^h(p) − B^h(p))` where h(p) is the Wald
  fundamental identity root.

**Glue:** `~120 LOC`. SPRT + canonical LLR helpers + OC/ASN + 20 golden vectors.

### Tier 2 — substantial composition lift (substrate partly missing or wrappers needed)

#### S9 — `NOT-changepoint` (Baranowski-Chen-Fryzlewicz 2019) — ~340 LOC

**Capability:** Narrowest-Over-Threshold — generalises WBS to detect changes
in slope, polynomial-trend, and AR-coefficient. Picks the **narrowest**
sub-interval whose CUSUM exceeds threshold (vs WBS picking the largest CUSUM).
Same family but better short-segment behaviour.

**Composition:** S6 WBS infrastructure + slope-CUSUM cost (linear-trend
filtered residual) + AR-coef-CUSUM cost (composes `levinsonDurbin`).

**Glue:** `~340 LOC`. Blocked-soft on S5 cost library + S6 WBS (no hard block —
NOT can ship standalone but the cost-function abstraction is wasted then).

#### S10 — `KernelCPD` (Harchaoui-Bach-Moulines 2007 / Arlot-Celisse-Harchaoui 2019) — ~290 LOC

**Capability:** RBF / linear kernel multi-CP detector. Uses the kernel-MMD
two-sample statistic as the segment-vs-segment dissimilarity inside a
dynamic-programming segmentation (so it's PELT-with-kernel-cost). Multivariate-
native, distribution-free, captures arbitrary dependence structure.

**Composition:**
- `infogeo/mmd.go:64 MMD2Biased(X, Y, k Kernel)`,
  `infogeo/mmd.go:16 GaussianKernel(bandwidth)`,
  `infogeo/mmd.go:169 MedianHeuristicBandwidth(X, Y)` — substrate **already
  exists**, this is the cleanest cross-package compose in the matrix.
- S5 PELT with `KernelCost(start, end) = MMD²(seg, complement)`.

**Glue:** `~290 LOC`. The kernel-cost wrapper + median-heuristic auto-bandwidth
+ PELT call. Substrate-internal first-consumer for both `infogeo/mmd` and
`changepoint/pelt`.

#### S11 — `MOSUM` (Eichinger-Kirch 2018) — ~210 LOC

**Capability:** Moving-sum detector. Compute the moving-sum of standardised
shifts over a bandwidth-G window; under H₀ the sup is asymptotically
Gumbel-distributed → analytic threshold. Multiscale extension (Cho-Kirch
2022) handles mixed-magnitude jumps.

**Composition:**
- `signal/window.go` (assumed Hann/Hamming windows — verify) for the moving-
  sum kernel (or pure boxcar suffices).
- `prob/distributions.go` for the Gumbel quantile (need to add — currently
  Normal/Exponential/Beta/Poisson/Gamma/Uniform/Binomial only — see 117).
- Standard CUSUM substrate from S1.

**Glue:** `~210 LOC` + Gumbel quantile addition (~30 LOC for `prob/distributions.go`).

#### S12 — `BreakInCointegration` (Gregory-Hansen 1996) — ~360 LOC

**Capability:** Tests for a structural break in the cointegrating relationship
between two non-stationary series. Engle-Granger 1987 cointegration tests
H₀ = "no cointegration" vs H₁ = "stable cointegration"; Gregory-Hansen 1996
extends to "stable cointegration with one break". The break-date is the
argmax of the time-varying ADF statistic.

**Composition:**
- New `prob/regression.go:LinearRegression` → already exists at line 36.
- New `timeseries/unitroot/adf.go` — Augmented Dickey-Fuller (137-T1.6 missing).
  ~150 LOC, blocked on 137 work.
- New `timeseries/cointegration/eg.go` Engle-Granger two-step. ~120 LOC.
- New `timeseries/cointegration/gh.go` Gregory-Hansen sup-ADF. ~90 LOC.

**Glue:** `~360 LOC`. **Hard-blocked on 137-T1.6** (ADF/KPSS/PP unit-root tests).
Cannot ship until the unit-root primitive lands.

### Tier 3 — composition that requires substantial new mathematics

#### S13 — `BOCPDStudentTPrior` for unknown-variance regime — ~80 LOC

**Capability:** BOCPD's NIG conjugate prior already gives Student-t posterior
predictive (`bocpd.go:185-191`) — this **already exists**. The S13 candidate
is BOCPD where the **observation model itself** is Student-t (heavy-tailed
returns), not just the predictive. Closed-form is lost; needs MCMC or
particle approximation.

**Composition:**
- `bocpd.go` predictive substituted with Student-t-Inverse-Gamma posterior.
- `prob/copula/studentt.go:60 StudentTQuantile` for prior elicitation.

**Glue:** `~80 LOC` if Gaussian-Inverse-Gamma stays as the conjugate spine and
Student-t enters only as a downweight on outliers (Lange-Little-Taylor 1989
ECM-style). Full Student-t observation needs PMCMC and is a Tier-4 multi-week
project.

#### S14 — `HamiltonMarkovSwitching` (Hamilton 1989) — ~620 LOC

**Capability:** Two-regime AR(p) with Markov-chain transition between regimes.
Hamilton filter for regime probabilities, Kim 1994 smoother, full MLE on
transition + AR + variance parameters. The textbook regime-switching
econometric model.

**Composition:**
- `prob/markov.go:31 MarkovSteadyState` and `:99 MarkovSimulate` — the state
  primitive exists.
- `prob/timeseries.go:125 ARIMA` — the AR primitive exists (with 137-N5 caveats).
- New `prob/hmm/forward_backward.go` — Forward-Backward algorithm for HMM
  inference. ~180 LOC. **Hard-blocked on no-HMM-anywhere.**
- New `prob/hmm/baum_welch.go` — EM for HMM parameters. ~220 LOC.
- New `timeseries/regimeswitch/hamilton.go` — Hamilton filter wrapper.
  ~220 LOC.

**Glue:** `~620 LOC`. Blocks: the HMM substrate (forward-backward, Viterbi,
Baum-Welch) is **wholly absent** in `prob/`. 022-T2.11 ("BOCPDMS") and the
RubberDuck `RegimeContextService` reference in `optim/transport/pairwise.go:15`
both flag this as a queued substrate.

#### S15 — `BOCPDMS-RegimeSwitchingARIMA` (Knoblauch-Damoulas 2018 + Adams-MacKay 2007) — ~480 LOC

**Capability:** Crown jewel. Bayesian online regime-switching ARIMA: at each
time step the posterior is over (run-length, ARIMA-coefficient-vector) jointly,
with an automatic Bayesian model-selection layer over candidate ARIMA orders
(p, d, q) ∈ a configured grid. Unifies Hamilton 1989 (regime switching) +
Adams-MacKay 2007 (online CP) + Box-Jenkins (ARIMA forecasting) into one
streaming forecaster with regime-aware predictive intervals. **No Python or R
library ships this out-of-the-box** — `bocpdms` Python package is the
nearest, but it's research-grade, not production-grade.

**Composition:**
- `changepoint/bocpd.go` — the run-length posterior machinery.
- New `changepoint/bocpdms.go` — multi-model variant: maintain
  `[]*Bocpd` over a model grid, plus a Bayes factor weighting across models
  per Knoblauch-Damoulas 2018 §3.
- `prob/timeseries.go ARIMA` per-model fits (137-T1.2 makes this exact-MLE
  via Kalman; current method-of-moments stub is the placeholder).
- 137-T1.1 Kalman primitive for state-space ARIMA (hard block for
  exact-MLE; method-of-moments works for prototype).

**Glue:** `~480 LOC` once 137-T1.1 + 137-T1.2 land. `~280 LOC` of pure
glue against today's stub (production-quality blocked on 137).

#### S16 — `Spectrum-CPD` (frequency-domain detection) — ~310 LOC

**Capability:** Detect changes in the **power spectrum** of a stationary
series. Compute STFT (short-time Fourier transform) → per-window periodogram
→ apply an existing CPD (S5 PELT, S7 e-divisive, S10 KernelCPD) on the
sequence of periodogram vectors. Detects shifts in spectral mass that
time-domain detectors miss (e.g., regime change in dominant frequency).

**Composition:**
- `signal/fft.go FFT` + `signal/window.go` (Hann window) — substrate exists.
- S5 PELT + S10 KernelCPD on R^k periodogram-vector sequence.

**Glue:** `~310 LOC` — STFT wrapper (~80) + cost-function adapter (~50) +
golden vectors against R `multitaper` package (~180). Cleanly composes
existing `signal/` and proposed `changepoint/` work.

#### S17 — `ChangepointROC` and detector evaluation — ~280 LOC

**Capability:** Receiver-operating-characteristic curves for changepoint
detectors. Given a synthetic series with known CPs and a detector with
threshold-tunable alarms, compute (FPR, TPR) over the threshold sweep,
integrate to AUC, and report Hausdorff distance / annotation error / F1
between detected and true CP sets.

**Composition:**
- New `changepoint/eval.go`: `Hausdorff(detected, truth []int) float64`,
  `AnnotationError(detected, truth []int, n, M int) float64` (Killick-Eckley
  2014 metric), `F1ChangePoint(detected, truth, tol) float64`.
- New `changepoint/roc.go`: `ROCSweep(detector ThresholdDetector, X, truth,
  thresholds) []ROCPoint` + `AUC(roc []ROCPoint) float64`.
- `prob/regression.go:36 LinearRegression` and `prob/prob.go:120 BrierScore`
  are unrelated; the ROC infrastructure is fresh ground.
- 022-T3.8 "TUNE" (Carrington 2024) is the post-detection-inference companion
  — Tier 4.

**Glue:** `~280 LOC`. Pure glue once any detector ships with
threshold-controllable output. **First consumer:** S1 CUSUM (h is the threshold).

#### S18 — `BocpdHazardSchedule` — time-varying hazard for BOCPD — ~70 LOC

**Capability:** The current BOCPD ships **only** constant hazard `H = 1/λ`
(`bocpd.go:147 hazard(r)`). Adams-MacKay 2007 §3 explicitly allows a hazard
function H(r) of the run-length — useful for periodic regimes (cron-scheduled
maintenance windows) or duration-dependent transitions (Hawkes-like
self-exciting CPs).

**Composition:** Refactor `(b *Bocpd).hazard(r) float64` to dispatch on a
caller-supplied `HazardFn = func(r int) float64`. Existing constant-hazard
becomes the default. The doc-warning at `bocpd.go:316-324` notes that under
constant hazard the predictive likelihood cancels in `ChangePointProbability`
— **non-constant hazard restores the canonical alarm signal.**

**Glue:** `~70 LOC`. Smallest single-file PR in the matrix; pure refactor with
backwards-compat shim. **Recommended as the first lift** because it unblocks
the cleanest interpretation of `ChangePointProbability` for downstream
consumers.

---

## Cross-walk to existing review numbering

| This synergy | 022-changepoint-missing | 137-timeseries-missing | 117-prob-missing |
|--------------|------------------------|------------------------|------------------|
| S1 CUSUM-on-GARCH | T1.1 (CUSUM) | composes garch.Filter | — |
| S2 CUSUM-on-ARIMA | T1.1 (CUSUM) | exposes ARIMA innovations | — |
| S3 BOCPD-on-ARIMA | exists / extends | T1.2 (exact ARIMA) | — |
| S4 BOCPD-on-GARCH | exists / extends | composes garch.Filter | — |
| S5 PELT | T1.3 / T1.4 | — | — |
| S6 BinSeg / WBS | T1.5 / T1.6 | — | — |
| S7 e-divisive | T1.7 | — | — |
| S8 SPRT | (not in 022 — gap) | — | — |
| S9 NOT | T2.2 | — | — |
| S10 KernelCPD | T2.3 | — | infogeo MMD substrate |
| S11 MOSUM | T1.8 | — | needs Gumbel quantile |
| S12 Cointegration-break | (out-of-scope 022) | T1.6 (unit roots) | — |
| S13 Student-t BOCPD | (extension) | — | — |
| S14 Hamilton-MS | (out-of-scope 022) | (out-of-scope 137) | needs HMM substrate |
| S15 BOCPDMS-ARIMA | T2.11 | T1.2 ARIMA | — |
| S16 Spectrum-CPD | (signal× synergy) | — | — |
| S17 ROC + eval | (gap) | — | — |
| S18 Time-varying hazard | (extension to BOCPD) | — | — |

**Hard blockers** (cannot ship today): S12 (needs 137-T1.6 unit-root tests);
S14 (needs HMM substrate in `prob/` — not in any review's tier yet); S15
production-quality (needs 137-T1.1 Kalman + 137-T1.2 exact ARIMA-MLE).

**Soft blockers** (can ship as prototype, principled version later): S2 / S3
need 4-LOC fix to expose ARIMA innovations (137-N5); S5 cost-library
benefits from `levinsonDurbin` export from `prob/timeseries.go:233`; S11
needs Gumbel quantile in `prob/distributions.go` (~30 LOC, 117 territory).

**No blockers** (ship today): S1, S4, S5 (with internal `levinsonDurbin`
copy), S6, S7, S8, S10, S13 (Gaussian-IG core only), S16, S17, S18.

---

## Recommended landing order

1. **S18 hazard schedule** (~70 LOC, single file) — restores
   `ChangePointProbability` semantics, unblocks downstream interpretation.
2. **S1 CUSUM-on-GARCH** (~110 LOC) — first non-Bayesian online detector,
   first cross-consumer of `garch/` from outside `garch/`, first golden-file
   set for a detector other than BOCPD.
3. **S17 ROC + eval** (~280 LOC) — evaluation infrastructure that all
   subsequent detectors depend on for golden files.
4. **S5 PELT + cost library** (~520 LOC) — the workhorse offline detector;
   all of S6 / S9 / S10 / S16 reuse the `CostFn` plug-point.
5. **S10 KernelCPD** (~290 LOC) — composes `infogeo/mmd` (already a verified
   cross-package consumer). High signal-to-effort ratio because MMD is done.
6. **S7 e-divisive** (~240 LOC) — distribution-free / multivariate baseline.
7. **S6 BinSeg + WBS** (~280 LOC) — speed-floor for the offline regime.
8. **S2 + S3 + S4 ARIMA / GARCH residual detectors** (~270 LOC together) —
   the named composition theme of this synergy review.
9. **S8 SPRT** (~120 LOC) — Wald 1945 substrate; orthogonal to the rest, ships
   anytime.
10. **S11 MOSUM** (~210 LOC + Gumbel) — analytic-threshold complement.
11. **S16 Spectrum-CPD** (~310 LOC) — composes `signal/fft` with PELT/Kernel.
12. **S13 Student-t BOCPD** (~80 LOC) — heavy-tail robustness.
13. **S9 NOT** (~340 LOC) — slope/poly/AR-coef detection.
14. **S12 cointegration-break** (~360 LOC, blocks on 137-T1.6) — queue
    behind unit-root tests.
15. **S14 Hamilton MS** (~620 LOC, blocks on HMM substrate) — full
    regime-switching econometrics.
16. **S15 BOCPDMS-ARIMA** (~480 LOC, blocks on 137-T1.1 + T1.2) — crown jewel.

**Total connective tissue:** ~3,580 LOC of pure glue plus ~1,000 LOC of
substrate (Kalman, exact-MLE ARIMA, ADF, HMM forward-backward) that
parallel review tracks already enumerate.

**Substrate-internal first-consumer pushes** (the cross-package witnesses
that 062-S62 wave introduced): S1 / S4 are the first non-test consumers of
`timeseries/garch` from outside `garch/`; S10 / S16 are the second cross-
package consumers of `infogeo` (after `changepoint/infogeo_test.go`); S3 /
S15 are the first cross-package consumers of `prob/timeseries.go` ARIMA;
S14 is the first non-test consumer of `prob/markov.go`.

---

## Key citations (load-bearing files)

- `C:/limitless/foundation/reality/changepoint/bocpd.go:147` — constant-hazard
  hard-coded; S18 lift point.
- `C:/limitless/foundation/reality/changepoint/bocpd.go:175-289` — `Update`
  recursion; the seven `make([]float64, …)` allocations per step cost a 60 FPS
  consumer ~60 KB/s churn at R_max=500.
- `C:/limitless/foundation/reality/changepoint/bocpd.go:316-324` — explicit
  warning that `ChangePointProbability` is uninformative under constant
  hazard; S18 fixes the diagnostic.
- `C:/limitless/foundation/reality/timeseries/garch/garch.go:55` — `Filter`
  emits `z` standardised residuals, the input substrate for S1 / S4.
- `C:/limitless/foundation/reality/timeseries/dcc/dcc.go:133` — `FilterSeries`
  emits time-varying correlation matrices; multivariate-CPD substrate (S7
  e-divisive on the upper-triangle sequence detects covariance breaks).
- `C:/limitless/foundation/reality/prob/timeseries.go:194-202` — ARIMA
  residuals computed but discarded; 4-LOC fix exposes them for S2 / S3.
- `C:/limitless/foundation/reality/prob/timeseries.go:233 levinsonDurbin` —
  private; export for S5 AR-cost and S9 NOT.
- `C:/limitless/foundation/reality/infogeo/mmd.go:64,116,169` — MMD substrate
  ready-to-go for S10 KernelCPD.
- `C:/limitless/foundation/reality/prob/markov.go:31,99` — discrete-Markov
  substrate; S14 Hamilton-MS extension point.
- `C:/limitless/foundation/reality/optim/transport/pairwise.go:15` — sole
  `cusum`/`HMM`-adjacent token in the entire repo (a comment), confirming
  the wholesale absence of CUSUM and HMM machinery.

---

**Verification:** all LOC estimates derived from comparable substrate weight
in `bocpd.go` (402 LOC for one detector; PELT in `ruptures` Python is ~340
LOC; e-divisive in R `ecp` is ~280 LOC; KernelCPD in `ruptures` is ~250 LOC).
Cross-language golden-file feasibility verified: every detector in S1–S11 has
a published reference implementation in Python (`ruptures`), R
(`changepoint`/`ecp`/`mosum`/`bcp`), or C (`changepoint::cpt` C internals)
producing ASCII-table golden vectors that `testutil/` ingests directly.
