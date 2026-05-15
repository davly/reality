# 138 | timeseries-sota

**Agent:** 138 of 400
**Topic:** timeseries: compare with statsmodels, sktime, prophet, Nixtla statsforecast, neuralforecast (+ darts, pmdarima/auto.arima, R::forecast / fable)
**Scope:** SOTA library landscape vs reality `C:/limitless/foundation/reality/timeseries/` and `prob/timeseries.go`
**Date:** 2026-05-08

## Brief

Reality `timeseries/` is **GARCH(1,1) + scalar DCC(1,1) + a stub `prob.ARIMA`** — agents 136 (numerics) and 137 (missing) covered the codebase. This report does **not** repeat either. For each SOTA library: (1) headline algorithm or feature, (2) one engineering trick worth stealing, (3) zero-dep portability verdict for reality. Six libraries, three R-domain references, one synthesis.

---

## statsmodels.tsa

The Python gold standard for time series. ~150 public time-series entry points across `tsa.api` (ARIMA, SARIMAX, VARMAX, ExponentialSmoothing/ETS, UnobservedComponents, DynamicFactor, MarkovSwitching), `tsa.statespace` (the Kalman engine underneath every model), `tsa.stattools` (ACF, PACF, ADF, KPSS, ccf, grangercausality, coint), `tsa.seasonal` (STL, X-13 wrapper), `tsa.filters` (HP, BK, CF), and `tsa.holtwinters`.

### (1) Headline: every state-space model goes through one Kalman filter

`tsa.statespace.MLEModel` is the load-bearing abstraction. `SARIMAX`, `UnobservedComponents`, `DynamicFactor`, `VARMAX`, `MarkovSwitching` all subclass `MLEModel`. The subclass declares `(F, H, Q, R, x0, P0)` — the five state-space matrices — and the base class handles **everything else**: Kalman prediction, exact-likelihood evaluation, RTS smoothing, simulation smoothing, `loglikelihood_burn` for diffuse-prior initialisation, missing-observation handling via the Kalman update being skipped, AIC/BIC reporting, score/Hessian by finite-diff or analytic. Cython core (`_kalman_filter.pyx`, ~3000 LOC) does the inner loop.

This is the **single most important architectural lesson for reality**. 137 already named "T1.1 Kalman primitive" as load-bearing. statsmodels proves the payoff: ARIMA, ETS-state-space, BSTS, DLM, structural-breaks, dynamic-factor, regime-switching all become 100-LOC matrix declarations on top of one ~400-LOC Kalman primitive.

### (2) Engineering trick: univariate Kalman filter for fast scalar observation

When the observation `H` is `1×k` (one scalar y per time-step — the SARIMA case), `(H P Hᵀ + R)⁻¹` is scalar division, not matrix inversion. `_kalman_filter_uni.pyx` is a separate code path — same math, ~10× faster than the multivariate filter. statsmodels picks the path automatically based on `dim_obs == 1`. Almost every univariate model hits this fast path.

Lesson for reality: **two Kalman implementations from day one**, dispatched by `len(y) == 1`. Code duplication is small (~150 LOC) and the speedup is decisive for ARIMA/ETS.

### (3) Portability for reality

**Direct port: yes, with caveats.** The architecture is the design template — copy it. Skip Cython; pure Go is fast enough for v1.0. Skip `MarkovSwitching` regime models for v1.0 (~800 LOC, niche). Port priorities: `Kalman` core → `SARIMAX` → `UnobservedComponents` (local-level / local-linear-trend / BSM seasonal) → `STL` (separate). Total ~1,500 LOC for the core stack. Use statsmodels-generated golden vectors at 1e-9 tolerance — the reference is mature, well-validated against EViews / R::stats / MATLAB.

---

## sktime

The "scikit-learn for time series" interface layer. ~40 forecaster classes in 2026 (sktime 0.34), but the headline is the **interface**, not the algorithms.

### (1) Headline: unified `fit / predict / update` over heterogeneous backends

Every forecaster implements `fit(y, X=None, fh=None)`, `predict(fh=None, X=None)`, `update(y, X=None, update_params=True)`. The `fh` (forecast horizon) is a first-class object — `ForecastingHorizon([1, 2, 3], is_relative=True)` or `is_relative=False` (absolute timestamps), with `to_relative(cutoff)` / `to_absolute(cutoff)` conversions. Backends from statsmodels, prophet, pmdarima, statsforecast plug into the same interface.

Compose: `TransformedTargetForecaster([("log", LogTransformer()), ("ar", ARIMA(...))])` mirrors sklearn's `Pipeline`. `MultiplexForecaster([("naive", NaiveForecaster()), ("ar", ARIMA(...))])` selects via grid-search CV. `ReducedRegressionForecaster(LinearRegression(), window_length=10)` reframes ML regressors as forecasters.

### (2) Engineering trick: `update()` is mandatory and explicit

Every forecaster declares `update(y_new)`: the pattern is "model fitted on `y[:t]`, now we have `y[t+1:t+k]`, refit-or-not?". Three regimes: `update_params=True` (refit fully), `update_params=False` (filter forward, freeze parameters), and `NoUpdate` (force the user to refit by hand). This is the **walk-forward backtesting primitive** — `ForecastingPipeline.evaluate(y, cv=ExpandingWindowSplitter(...))` calls `update` between folds. statsmodels does not have this concept first-class; sktime made it the API spine.

Lesson for reality: **`Update(observation)` belongs on every Kalman / DLM / ETS struct from v1.0**. The Kalman primitive already supports this naturally (one filter step), but the public API needs to expose it deliberately, not as a side-effect of "call Filter again on a longer series".

### (3) Portability for reality

**Don't port the framework — port the API contract.** Reality's idiom is plain Go structs and free functions, not class hierarchies. But the `(Fit, Predict, Update)` triplet plus an explicit forecast-horizon type (relative steps vs absolute timestamps) are language-portable. Define `type ForecastHorizon struct { Steps []int; Absolute []time.Time; Relative bool }` and make every forecasting model accept it. Adds ~50 LOC of type plumbing; pays back in API consistency across ARIMA / ETS / Theta / Prophet.

---

## Prophet (Facebook / Meta)

Decomposable additive model: `y(t) = g(t) + s(t) + h(t) + ε`, fit via Stan (NUTS or L-BFGS MAP). Used heavily for business forecasting; mediocre on classic benchmarks (M3/M4) but surprisingly good on irregular real-world data.

### (1) Headline: explicit decomposability — analyst can edit each component

`g(t)` is **piecewise-linear or logistic-saturating trend** with automatic changepoint detection (Laplace prior on changepoint magnitudes). `s(t)` is **Fourier-series seasonality** at multiple periods (yearly = 10 harmonics, weekly = 3 harmonics, daily = 4 harmonics by default). `h(t)` is a **regressor matrix of holiday indicators** with their own Laplace-prior coefficients. The fit is a single big linear regression on (trend basis × Fourier basis × holiday indicators), MAP'd via L-BFGS.

The genius is **interpretability** — `m.predict(future)` returns not just `yhat` but `trend, weekly, yearly, holidays` as separate columns. Analyst sees "Christmas effect = +12%" as a number, not a black-box residual. Compared to ARIMA's "AR coefficients" or ETS's "level/trend/season state", Prophet's components are directly editable: a domain expert can add `add_country_holidays('US')` or `add_seasonality(name='monthly', period=30.5, fourier_order=5)`.

### (2) Engineering trick: changepoint detection via Laplace shrinkage on a dense grid

Prophet does **not** search for changepoints. It places ~25 candidate changepoints uniformly across the first 80% of training data and puts a Laplace(0, τ) prior on the rate change at each. L-BFGS shrinks irrelevant ones to ≈0; `changepoint_prior_scale=0.05` (default) tunes regularisation. The result: automatic, gradient-friendly, no combinatorial search.

Lesson for reality: **Laplace-prior shrinkage on a dense grid is a powerful "feature selection by convex optimisation" pattern.** Same idea used in Lasso (sparse regression), spike-and-slab BSTS, sparse ARIMA. ~30 LOC on top of an existing L-BFGS.

### (3) Portability for reality

**Partial port, design-heavy.** The math (~300 LOC):
- Fourier-series basis evaluation (~30 LOC).
- Piecewise-linear trend with `changepoint` knot vector (~50 LOC).
- Holiday indicator matrix (~20 LOC of pure logic, plus a holiday calendar — see below).
- L-BFGS MAP fit on the augmented log-likelihood (~50 LOC; reuses `optim/lbfgs.go`).
- Posterior decomposition into components (~50 LOC).

The **holiday calendar is the design problem** — Prophet bundles `holidays.py` (~600 country-holiday rules, lunar/movable feasts). reality cannot ship a holiday database; the API should accept a user-supplied `[]time.Time` per holiday name. ~400 LOC total for `timeseries/prophet/`. Goldens: prophet itself; tolerance ~1e-6 (Stan's L-BFGS is not bit-exact across versions; tolerate the noise).

---

## Nixtla statsforecast

The 2022-2026 commercial-grade revival of classical methods: 30+ models with the sktime interface, **Numba-jitted** inner loops, designed for embarrassing-parallel forecasting at scale (millions of time series).

### (1) Headline: AutoARIMA / AutoETS / AutoTheta with deterministic stepwise model selection

Hyndman-Khandakar 2008 stepwise algorithm for ARIMA model selection — start at (0,d,0), try neighbours `(p±1, d, q)`, `(p, d, q±1)`, accept if AIC drops, repeat. Deterministic, finite, ~5-10 fits per series. Same pattern for ETS (try all 30 (E,T,S) variants, AIC select) and Theta (try 5 θ values). Nixtla's contribution: **all three algorithms in one codebase, one API, identical interface**, vectorised via Numba.

`StatsForecast(models=[AutoARIMA(), AutoETS(), AutoTheta()], freq='D').fit(df).predict(h=12)` runs all three on each of N series in parallel. The library's claim-to-fame is **20-100× faster than statsmodels** on many-series workloads (reproducible in their benchmarks).

### (2) Engineering trick: panel-data layout — one long DataFrame, not one model per series

Input is `(unique_id, ds, y)` long format — N series stacked. Internal representation: `np.array((N, T))`. Fit loop: Numba-jitted `prange` over the leading axis. No Python overhead per series. Same code path runs on 1 series (M5 hierarchical) or 10 million (M5 store×SKU).

statsmodels gets this wrong — every model is a fresh Python object with overhead. statsforecast gets it right — the model is a **stateless function** of `(y, params)`, JITted once, called 10 million times.

Lesson for reality: **batch-vectorise the public API.** `arima.Fit(yBatch [][]float64) []ARIMAModel` should be the primary entry point; the single-series version is a convenience wrapper. Saves 10-100× on hierarchical / panel forecasting workloads. Aligns with the "no allocations in hot paths" rule.

### (3) Portability for reality

**Direct port, high value.** AutoARIMA stepwise (~150 LOC on top of T1.2 ARIMA from 137), AutoETS variant search (~80 LOC), AutoTheta (~50 LOC). Total ~280 LOC for the auto-selection layer. Goldens: statsforecast itself for AIC-tied cases (often non-unique winner), pmdarima for AutoARIMA cross-check; tolerance 0 on AIC (deterministic given data) and 1e-9 on coefficient values once the model is selected.

The panel-batch idiom maps cleanly to Go: `[][]float64` input, `[]Model` output, goroutine-per-row inside the Fit driver (controlled by `runtime.GOMAXPROCS`). No SIMD / Numba dependency required to beat the sequential implementation by 10×.

---

## Nixtla neuralforecast

Deep learning for forecasting: NHITS, NBEATS, TFT, PatchTST, TimesNet, plus the Nixtla family (TimeGPT, foundation models). Same panel-data interface as statsforecast.

### (1) Headline: NHITS — hierarchical multi-rate sampling beats Transformers on M4/M5

Challu et al. 2023. NBEATS' deep-stack idea (each block subtracts its prediction from input, residual connection forwards) plus **multi-rate input pooling** (block 1 sees 1× sampled input, block 2 sees 2× pooled, block 3 sees 4× pooled — captures multiple time scales) plus **hierarchical interpolation** (block i predicts at 1/2^i frequency, upsamples to native). Result: matches Transformer accuracy at 50-100× the inference speed. NHITS is the current "default deep model" for short/medium-horizon forecasting in 2026.

### (2) Engineering trick: identity-residual stacks — NBEATS ancestry

Both NHITS and NBEATS use the **same residual-stack architecture**: `output_t = block_1(input) + block_2(input − block_1.backcast) + ...`. Each block fits the **residual**, not the signal. Identical to gradient boosting in the residual sense. Trained end-to-end with backprop, but the architecture *is* a boosting tree if you squint. This is why these models train stably with no Transformer attention pathology.

Lesson for reality: **residual-stack architectures are gradient-boosting in disguise.** When/if reality eventually exposes a deep-learning forecasting block (likely not until v2.0+), copy this architecture, not Transformer attention. It is dramatically simpler and competitive.

### (3) Portability for reality

**Out of scope for v1.0 — flag as v2.0.** Deep-learning forecasting requires:
- A neural-network primitive layer (MLP, conv1d, linear) — **does not exist in reality**.
- A backprop / autodiff core — exists in `prob/autodiff/` (per 011 audit) but only at the scalar level; needs tensor extension.
- An optimiser with momentum (Adam, SGD-momentum) — exists in `optim/` (Adam? unclear) but not battle-tested on neural training.

Pure-Go neural networks at 2026 SOTA accuracy are ~2,000 LOC for the layers + ~1,000 LOC for training-loop machinery + ~500 LOC for NHITS specifically. Total ~3,500 LOC. **Defer to v2.0 explicitly**; do not block v1.0 on this. The classical/statistical SOTA (statsforecast AutoARIMA/ETS/Theta) is competitive with neuralforecast's deep models on M3/M4 benchmarks anyway — Hyndman 2024 showed AutoARIMA + AutoETS + AutoTheta ensemble beats single deep models on M4.

---

## darts (Unit8)

PyTorch-based unified TS framework. `TimeSeries` is a wrapper over an xarray-backed DataFrame; every model implements `fit(series) → predict(n)`. Bridges classical (`ARIMA`, `ExponentialSmoothing`, `Theta`) and deep (`RNNModel`, `TFTModel`, `NBEATSModel`, `TCNModel`, `TransformerModel`) under one abstraction.

### (1) Headline: `TimeSeries` is a typed, indexed, multivariate-aware container

Most TS libraries pass `np.array` and hope. darts' `TimeSeries` carries: timestamp index (regular or irregular), component names (multivariate dimensions), static covariates (per-series metadata that doesn't change in time), past covariates (known up to `t`), future covariates (known beyond `t` — e.g., known-future weather), with explicit type checking on operations. `TimeSeries.from_dataframe(df, time_col='date', value_cols=['y1', 'y2'])` carries the type through pipelines.

### (2) Engineering trick: probabilistic forecasts as a first-class output type

Every darts model returns a `TimeSeries` whose values are sampled from the predictive distribution — multiple paths, not a point estimate. `model.predict(n=12, num_samples=500)` returns `(500, n, components)`. The interface for "give me the median + 80% prediction interval" is `series.quantile_timeseries(quantiles=[0.1, 0.5, 0.9])`. Probabilistic-by-default is the right architecture for v2026 — point forecasts are a degenerate case (`num_samples=1`).

Lesson for reality: **forecast outputs should be `(point, lower, upper, samples)` from day one**, not bolted on later. ARIMA's analytic prediction interval (Brockwell-Davis 9.5) and Kalman's `(x̂, P)` covariance are already probabilistic; the API should expose this. ~20 LOC of struct change at the right moment is much cheaper than retrofitting.

### (3) Portability for reality

**Don't port the framework — port the output type.** Define `type Forecast struct { Point []float64; Lower, Upper []float64; Samples [][]float64 }` and return it from every forecaster. The multivariate `TimeSeries` container is heavy and overlaps with `linalg.Matrix`; reality's `[][]float64` per series is sufficient at v1.0. Static / past / future covariates are tier-3 concerns (per 137 T2.x).

---

## pmdarima (formerly pyramid-arima)

Python port of R's `auto.arima`. ~5,000 LOC, single-purpose: AutoARIMA. Now largely superseded by statsforecast's AutoARIMA but still the most-cited reference for the algorithm.

### (1) Headline: Hyndman-Khandakar 2008 stepwise — the canonical AutoARIMA

Algorithm:
1. Determine `d` via successive KPSS tests (start at d=0; if non-stationary, d++; repeat up to `max_d`).
2. Fit four starting models: `(0,d,0)`, `(1,d,0)`, `(0,d,1)`, `(2,d,2)`. Pick best by AIC.
3. Stepwise neighbourhood search: at each step try `(p±1, q)`, `(p, q±1)`, `(p±1, q±1)`, plus toggle `include_constant`. Accept if AIC drops by `≥ 0`. Repeat until no improvement.
4. Final fit on best (p, q) with full MLE.

Step 1 is the load-bearing dependency on **KPSS** (per 137 T1.6 — currently absent from reality). Step 2-4 depends on **exact-likelihood ARIMA via Kalman** (per 137 T1.2 — also absent). pmdarima's own code is ~1,200 LOC for the auto-search; ~3,500 LOC of the package is reimplementing statsmodels' state-space ARIMA because the original Python wrapper is too slow.

### (2) Engineering trick: deterministic seed for the stepwise — reproducibility audit

`auto.arima` and pmdarima both fix the search order (try `p+1` before `p-1`, etc.) so two runs on identical data give identical models. This sounds trivial; it isn't. Stochastic AutoARIMA (e.g., `forecast::ets` with `bootstrap=TRUE`) is **not** reproducible across versions, which breaks regression-test-driven library development. pmdarima/auto.arima made the orthodox choice; reality should too.

Lesson for reality: **AutoARIMA stepwise must be deterministic by spec, with the search order pinned in the docstring.** Goldens then become straightforward.

### (3) Portability for reality

**Direct port. ~150 LOC** on top of T1.2 ARIMA + T1.6 KPSS. Almost nothing in pmdarima is novel beyond what's already in statsforecast — port from statsforecast, cross-check against pmdarima for the canonical Hyndman-Khandakar algorithm. Goldens: deterministic at the AIC level (compare model selection), 1e-9 on selected-model coefficients.

---

## R::forecast and R::fable (Hyndman et al.)

The original. `forecast` (2008-2020, Hyndman et al.) defined the playbook: `auto.arima`, `ets`, `tbats`, `Theta`, `nnetar`, `tslm`, plus the `forecast` object with point + 80% + 95% prediction intervals as the standard output. `fable` (2020+, tidyverts) is the tidyverse-style successor with the same models reorganised under `mable / fable / tsibble`.

### (1) Headline: ETS state-space innovation form — Hyndman-Athanasopoulos 2008 §8.5

The 30 ETS variants (Error: A,M × Trend: N,A,Ad,M,Md × Season: N,A,M) are all written as **innovation state-space models**:

```
y_t      = w(x_{t-1}) + r(x_{t-1}) ε_t          [observation]
x_t      = f(x_{t-1}) + g(x_{t-1}) ε_t          [state transition]
ε_t ~ N(0, σ²)                                  [single innovation source]
```

Single source of error — same `ε_t` drives observation noise and state evolution. This is **structurally simpler than the ARIMA Kalman setup** (which has two error sources) and gives closed-form likelihood evaluation without a Kalman filter. ETS is the cheapest "real" forecasting model — ~50 LOC per variant, AIC selects across all 30.

### (2) Engineering trick: AIC across non-nested models in one shot

ETS variants are **not nested** — `ETS(A, A, N)` is not a special case of `ETS(M, A, M)`. `forecast::ets(method='ZZZ')` fits all 30, computes AIC for each, returns the minimum. This works because:
- Each fit is closed-form fast (~10 ms per series per variant on modern hardware).
- AIC is a comparable metric across non-nested models.
- The 30-variant catalogue is finite and small.

Same logic powers `forecast::tbats` (2-12 seasonal harmonic counts × Box-Cox lambda grid) and `forecast::auto.arima` (stepwise but with closed-form fits). **Brute-force is fine when the search space is small and each fit is closed-form.** Library-design lesson worth internalising.

### (3) Portability for reality

**Direct port: yes, very high value.** The ETS state-space catalog (Hyndman-Athanasopoulos 2008 §8.5 has the formulas, ~30 variants × ~5 lines each = 150 lines of pure recursion) plus AIC selector. ~400 LOC total per 137 T1.4 estimate. **No Kalman dependency** because ETS is innovation-form, not classical state-space. This is the **cheapest non-trivial forecasting model to ship** — no Kalman, no MLE, no PACF infra needed.

Goldens: R::forecast `ets()` is the canonical reference; statsmodels has a port (`ETSModel`) with known small numerical drift (~1e-6) from R. Use R as the gold reference, statsmodels as cross-check.

---

## R::fable / tidyverts ecosystem

`fable` 2020+ refactor of `forecast` for tidyverse. Same models, plus `feasts` (feature extraction — autocorrelation features, STL features, Hurst exponent, ARCH features), `fabletools` (model evaluation, forecast combinations), `tsibble` (time-aware data frames). Ecosystem-level engineering: every function returns a `mable` (model table) or `fable` (forecast table) that's directly inspectable in a data-frame UI.

### (1) Headline: feature-based forecasting — `feasts::features()` extracts ~50 features

`feasts::features(y, feat_set)` extracts: STL strength of trend / seasonality, ACF[1..3], PACF[1..3], unit-root statistics (KPSS), ARCH features, entropy (sample entropy, approximate entropy), Hurst exponent, lumpiness, stability, mean changes, variance changes, lambda (Box-Cox), and ~30 others. The output is a feature vector per series, suitable for **clustering thousands of time series**, **automatically selecting forecast model** (FFORMA — Montero-Manso et al. 2020 trains XGBoost on features → model class), or **anomaly detection**.

### (2) Engineering trick: feature-vector → model-selection meta-learner

FFORMA (Feature-based FORecast Model Averaging — winner-adjacent on M4): extract feature vector from training series, train XGBoost regressor mapping `feature_vector → optimal_model_weights` from a held-out validation. New series: extract features, predict weights, ensemble accordingly. Combines the breadth of statsforecast (30 models) with the principle of stacking.

Lesson for reality: **feature extraction is the enabling primitive for meta-learning over time series**, and the features themselves are a first-class deliverable, not just an internal step.

### (3) Portability for reality

**Direct port: feature catalogue, yes.** Each individual feature is small (~10-30 LOC per feature × 50 features = ~700 LOC). Most features are computable from already-needed primitives:
- ACF/PACF features → T1.5 (137).
- STL strength → T2.5 (137) — depends on STL.
- ARCH features → already have GARCH; trivial extraction.
- KPSS → T1.6.
- Sample entropy / approximate entropy → small, ~30 LOC each.
- Hurst exponent (R/S analysis) → ~50 LOC.
- Box-Cox lambda → MLE on a 1D grid, ~30 LOC.

Total `timeseries/features/` package: ~700 LOC, depends on Tier 1 primitives landing first. **Tier 2 priority** — high library-credibility return for moderate effort.

---

## Synthesis: prioritised list of stealable ideas

By library, the deltas reality should adopt — independent of the gap-closing list 137 already produced:

| Source | Idea | LOC estimate | Tier |
|---|---|---|---|
| statsmodels | One Kalman primitive underneath every state-space model | (already T1.1, ~400) | Architectural |
| statsmodels | Two Kalman code paths — univariate-fast and multivariate-general | +150 | Tier 1 |
| sktime | Explicit `Update(observation)` on every model, not "re-Fit on extended series" | +50 | Tier 1 |
| sktime | First-class `ForecastHorizon` type (relative steps vs absolute timestamps) | +50 | Tier 1 |
| Prophet | Decomposable additive model with editable components | (T3.1, ~400) | Tier 3 |
| Prophet | Laplace-prior shrinkage on dense grid of changepoint candidates | +30 (within T3.1) | Tier 3 |
| statsforecast | Panel-data batch API: `Fit([][]float64) []Model` as primary entry point | +100 | Tier 1-2 (architectural) |
| statsforecast | AutoARIMA / AutoETS / AutoTheta stepwise + AIC selection | +280 | Tier 2 |
| neuralforecast | NHITS / NBEATS residual-stack architecture | +500 | Tier 3 / v2.0 |
| darts | Probabilistic forecast as first-class output type (`Forecast{Point,Lower,Upper,Samples}`) | +20 | Tier 1 |
| pmdarima | Deterministic stepwise search order | +0 (spec-only) | Tier 2 |
| R::forecast | ETS innovation-form state-space (no Kalman dependency) | (T1.4, ~400) | Tier 1 |
| R::forecast | Brute-force AIC across small catalogue of non-nested models | +0 (idiom) | Tier 1 |
| R::fable | Feature catalogue (`feasts::features` ~50 features) | +700 | Tier 2 |
| R::fable | FFORMA meta-learner (feature-vector → model-weight mapping) | +200 | Tier 3 |

---

## Architectural recommendations

Three load-bearing decisions, each with explicit precedent:

1. **One Kalman primitive, two code paths (univariate-fast / multivariate-general)** — statsmodels architecture. Everything else is a 50-200 LOC declaration on top.
2. **`(Fit, Predict, Update, Forecast)` quartet on every forecaster, with `Forecast` returning probabilistic output `{Point, Lower, Upper, Samples}`** — sktime API + darts output type. Pin this from v0.11.
3. **Panel-data batch entry points (`[][]float64 → []Model`) as primary API** — statsforecast architecture. Single-series wrappers convenience-only. ~10× speedup on hierarchical workloads, aligns with reality's "no allocations in hot paths" rule.

Plus one explicit **deferral**: deep-learning forecasting (NHITS, NBEATS, TFT, PatchTST, TimesNet) is v2.0+ scope — needs a tensor-autodiff layer that does not exist. Classical SOTA (AutoARIMA + AutoETS + AutoTheta + Theta + Prophet) is competitive on M3/M4 benchmarks anyway. Do not block v1.0 on neural forecasting.

---

## Cross-package overlap

| Source idea | Overlaps with | Resolution |
|---|---|---|
| Kalman primitive | `control/` Riccati for LQR | Build Kalman in `timeseries/kalman/`; share Riccati with `control/lqr/` via a `linalg.SolveDiscreteRiccati` |
| Probabilistic forecast type | `prob/distribution/` | `Forecast.Samples` is just `[N][]float64`; no new dependency |
| Feature catalogue (`feasts`) | `prob/`, `signal/` (entropy, periodogram) | Features that depend on FFT live in `signal/`; reality's `timeseries/features/` re-exports |
| Box-Cox lambda MLE | `prob/` (transformations) | Build in `prob/transform/`; `timeseries/features/` calls it |
| NHITS / NBEATS (v2.0) | absent neural-net primitive | Out of scope until tensor-autodiff lands |

---

## R-pattern saturation

- **R-SOTA-ARCHITECTURE-COVERAGE: 1/3.** Reality has GARCH-family architectural template via the `Model.Filter / LogLikelihood / Fit` triplet, but no equivalent for state-space (Kalman primitive absent), no ETS innovation-form template, no panel-batch entry point, no probabilistic-output type.
- **R-AUTOML-AUTOMATIC-MODEL-SELECTION: 0/3.** No AutoARIMA, no AutoETS, no AutoTheta, no FFORMA. The entire automatic-model-selection layer is absent.
- **R-INTERPRETABLE-DECOMPOSITION (Prophet, ETS components, STL): 0/3.** No model returns named decomposed components.

---

## Two-line summary

External SOTA libraries converge on three architectural choices reality should adopt: one Kalman primitive underneath every state-space model (statsmodels), an explicit `(Fit, Predict, Update)` quartet with probabilistic-by-default forecast output (sktime + darts), and panel-data batch entry points as the primary API (Nixtla statsforecast) — these are zero-to-low-LOC architectural decisions that pay back across every forecasting model added afterward. The cheapest first-credibility commits are ETS innovation-form (no Kalman dependency, ~400 LOC, full 30-variant catalogue against R::forecast goldens) and the feature catalogue (`feasts`-style ~50 features, ~700 LOC) which together give reality a credible Tier-1 baseline before the heavier Kalman + ARIMA stack from 137 lands.

---

## Progress

- 2026-05-08: 138-timeseries-sota report complete; 6 libraries reviewed (statsmodels, sktime, prophet, statsforecast, neuralforecast, darts) plus pmdarima and R::forecast/fable; 15 stealable ideas tabulated with LOC estimates and tiers; three architectural recommendations issued; one explicit deferral (deep learning to v2.0).
