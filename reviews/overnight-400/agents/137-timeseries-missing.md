# 137 | timeseries-missing

**Agent:** 137 of 400
**Topic:** timeseries: missing — SARIMA, ETS, BATS/TBATS, Theta, Prophet, Bayesian structural TS, GARCH(p,q), EGARCH, dynamic linear models, dlm
**Scope:** `C:/limitless/foundation/reality/timeseries/` (only `garch/` GARCH(1,1) and `dcc/` DCC) plus `prob/timeseries.go` (ExponentialSmoothing, HoltLinear, toy ARIMA, levinsonDurbin)
**Date:** 2026-05-08

## TL;DR

`timeseries/` is a two-subpackage stub: only **GARCH(1,1)** (univariate, fixed-order, fixed-LR GD fitter) and **DCC** (Engle 2002 correlation recursion). There is no model of the AR/MA/ARMA/ARIMA/SARIMA family that does exact MLE, no state-space machinery, no exponential-smoothing state-space (ETS), no decomposition (STL/X-11/X-13), no unit-root tests (ADF/KPSS/PP), no spectral-domain time-series (Welch/multitaper/Whittle), no cointegration (Johansen/Engle-Granger/VAR/VECM), no Bayesian structural time series (BSTS/DLM), no extended GARCH variants (p,q / EGARCH / GJR / IGARCH / FIGARCH / BEKK / CCC), no Prophet, no Theta, no BATS/TBATS, no MinT hierarchical reconciliation, no recursive-window estimation. Topic coverage of the actual codebase is **~3%** of what the prompt enumerates. 136 already mapped what exists numerically — this report is the gap census.

---

## Inventory: what is here

| Path | Function | Adequacy |
|---|---|---|
| `timeseries/garch/garch.go` | `Model.Filter`, `Model.LogLikelihood`, `Model.ForecastVariance`, `Model.Simulate` | GARCH(1,1) only — no GARCH(p,q), no asymmetric, no t/GED innovations |
| `timeseries/garch/fit.go` | `Fit(eps, init, cfg)` | Fixed-LR GD with Tikhonov; no L-BFGS, no Newton, no QMLE std errors |
| `timeseries/dcc/dcc.go` | `EngleDefault`, `SampleQbar`, `Update`, `CorrelationFromQ`, `FilterSeries` | Scalar DCC(1,1) only — no DCC(M,N), no ADCC, no cDCC, no BEKK, no CCC |
| `prob/timeseries.go` | `ExponentialSmoothing`, `HoltLinear`, `ARIMA(p,d,q)`, `levinsonDurbin` (private) | SES + Holt linear are textbook; ARIMA is a method-of-moments stub (see 136-N5); levinsonDurbin discards PACF + per-order error variance |

**That is the entirety of the time-series surface.** `signal/fft.go` ships an FFT but no time-series wrappers; `audio/pitch/autocorrelation.go` does pitch-detection autocorrelation but is not exposed for general use; `prob/hypothesis.go` has no unit-root tests.

---

## Tier 1 — must-have to claim a "time series" package (~2400 LOC total)

These are the load-bearing absences. Without these, `timeseries/` is mis-named.

### T1.1 — State-space / Kalman primitive (`timeseries/kalman/`)
Without a Kalman filter the entire downstream stack (DLM, ARMA exact MLE, BSTS, ETS state-space, dynamic regression) cannot exist. Need:
- `Kalman{F, H, Q, R, P, x}` struct, `Predict()` / `Update(z)` methods.
- **Joseph-form covariance update** (numerically stable, preserves PSD under finite-precision Update).
- **Square-root variant** (Carlson / Bierman UD) for ill-conditioned `P` — needed for ARMA(p,q) likelihood with high-order p+q.
- **RTS smoother** for offline likelihood evaluation and BSTS.
- Information filter (Y, y) form for `R` ≈ singular cases.
- Steady-state Riccati solver (DARE) — also unblocks `control/lqr.go` future work.
- ~250-400 LOC, ~30 tests, golden vectors against statsmodels.tsa.statespace.

### T1.2 — ARIMA proper, on the Kalman primitive (`timeseries/arima/`)
Replace `prob.ARIMA` (which is a stub — see 136-N5: biased MA estimation from residual autocorrelation, no integration-on-forecast, PARCOR clamp silently masks non-PSD autocov). Need:
- `ARIMA(p, d, q)` model struct with Box-Jenkins differencing **stored** so the forecast can integrate back to levels.
- Exact MLE via Kalman likelihood (Harvey 1989 §3.4, or Brockwell-Davis 8.10).
- Innovations algorithm (Brockwell-Davis 5.3) as the cheap fallback for pure ARMA without full state-space cost.
- Hannan-Rissanen two-stage initial estimation for MA component (current `prob.ARIMA` skips this).
- Forecast with confidence intervals (1-σ, 95%).
- AIC/BIC/AICc reporting from the same likelihood.
- ~600 LOC.

### T1.3 — SARIMA (`timeseries/arima/sarima.go`)
Seasonal ARIMA(p,d,q)(P,D,Q)_s — the workhorse model for monthly/quarterly economic data. Builds on T1.2 by extending the differencing operator to (1−B)^d (1−B^s)^D and the lag polynomial to (1−φ_1 B − ... − φ_p B^p)(1−Φ_1 B^s − ... − Φ_P B^{Ps}). ~150 LOC additional once T1.2 lands.

### T1.4 — ETS state-space (`timeseries/ets/`)
Hyndman-Athanasopoulos 2002/2008 — 30 ETS variants over (Error: A,M) × (Trend: N,A,Ad,M,Md) × (Season: N,A,M). The textbook approach to "automatic exponential smoothing". Each variant has a closed-form 1-step prediction and recursion; AIC selection across variants picks the model.
- Holt-Winters multiplicative + additive are special cases (currently absent — `prob.HoltLinear` covers the no-seasonal sub-case only).
- ~400 LOC for all 30 variants + AIC selector.

### T1.5 — ACF / PACF / Yule-Walker (`timeseries/acf/`)
The diagnostic primitives every ARMA caller needs. Currently nowhere exported.
- `ACF(data, maxlag)` — sample autocorrelation, Bartlett SEs.
- `PACF(data, maxlag)` — partial autocorrelation; **comes free from the existing `levinsonDurbin` recursion** (136-N6: the recursion already computes PARCORs and per-order error variance and silently discards both).
- `YuleWalker(data, p)` — AR coefficients via the same recursion.
- `LjungBox(residuals, lags)` — whiteness test (currently has `prob.hypothesis.go` but no time-series-specific Ljung-Box).
- ~120 LOC; Tier 1 only because it is dirt-cheap given Levinson-Durbin already runs.

### T1.6 — Unit-root / stationarity tests (`timeseries/unitroot/`)
The gating tests for any ARIMA / cointegration workflow.
- **Augmented Dickey-Fuller (ADF)** — MacKinnon 2010 polynomial response surface for critical values (non-trivial: not constants but a 4-coefficient polynomial in n at each significance level / regression type).
- **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin) — null is stationarity, complement of ADF.
- **Phillips-Perron (PP)** — non-parametric unit root, Newey-West HAC variance estimate.
- ~250 LOC + a 60-row table of MacKinnon coefficients.

### T1.7 — Spectral-domain time-series (`signal/welch.go`, `signal/periodogram.go`)
Lives in `signal/` not `timeseries/` because it builds on FFT, but listed here because the topic prompt enumerates Welch / multitaper / Whittle and 132-signal-missing already flagged this.
- `Periodogram(data, window)` — straight |FFT|² / N.
- `Welch(data, segLen, overlap, window)` — segment averaging (Welch 1967), the standard spectral estimator.
- ~150 LOC, returns `freq, power`.

---

## Tier 2 — should-have for a credible package (~3000 LOC total)

Each of these is canonical in `statsmodels.tsa` / R::forecast / MATLAB Econometrics. Absence is a credibility tax.

### T2.1 — Volatility-model expansions (`timeseries/garch/`)
Build on existing GARCH(1,1) machinery. Each is a 50-150 LOC delta over what is there.
- **GARCH(p, q)** — generalise the (1,1) recursion to arbitrary p, q. The quadratic form `omega + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}` is a 30-LOC change.
- **EGARCH** (Nelson 1991) — `log σ²_t = ω + Σ α_i (|z_{t-i}| − E|z|) + γ_i z_{t-i} + Σ β_j log σ²_{t-j}`. Asymmetry without parameter constraints.
- **GJR-GARCH** (Glosten-Jagannathan-Runkle 1993) — `σ²_t = ω + α ε²_{t-1} + γ I(ε_{t-1}<0) ε²_{t-1} + β σ²_{t-1}`. Threshold asymmetry.
- **IGARCH** — restrict α + β = 1 (unit-root in volatility).
- **FIGARCH** — fractional differencing in σ², long-memory volatility (Baillie-Bollerslev-Mikkelsen 1996).
- **t-distributed innovations** and **GED** innovations as alternatives to Gaussian QMLE — a 30-LOC log-likelihood swap for each.

### T2.2 — Multivariate volatility (`timeseries/dcc/`, `timeseries/bekk/`, `timeseries/ccc/`)
- **CCC** (Bollerslev 1990) — constant conditional correlation. Simplest baseline; DCC reduces to CCC at α=β=0.
- **ADCC** (Cappiello-Engle-Sheppard 2006) — asymmetric DCC, captures negative-shock correlation increase.
- **BEKK(1,1)** (Engle-Kroner 1995) — full multivariate GARCH, parameterised to guarantee PSD covariance. ~250 LOC.
- **DCC(M, N)** — generalise current scalar DCC to higher-order memory. Currently `dcc.Update` is hard-coded (1,1).

### T2.3 — Theta method (`timeseries/theta/`)
Assimakopoulos-Nikolopoulos 2000 — the M3 competition winner. Decompose into θ-lines, fit linear regression + SES, recombine. ~80 LOC, surprisingly competitive baseline that absolutely should be in any forecasting package.

### T2.4 — Cointegration / vector models (`timeseries/coint/`, `timeseries/var/`)
- **VAR(p)** — vector autoregression, OLS estimator + Granger causality test.
- **VECM** — vector error correction model, the cointegration-aware companion.
- **Johansen test** — cointegration rank determination via canonical correlations. Tabulated critical values (Osterwald-Lenum 1992).
- **Engle-Granger two-step** — simpler bivariate cointegration test.
- ~400 LOC total.

### T2.5 — Decomposition (`timeseries/decompose/`)
- **STL** (Cleveland-Cleveland-McRae-Terpenning 1990) — seasonal-trend decomposition by LOESS. The standard. ~300 LOC because LOESS itself is needed (currently absent from `prob/`).
- **Classical additive / multiplicative decomposition** — moving-average trend extraction. ~80 LOC.
- **X-11 / X-13ARIMA-SEATS** — Census Bureau standard for official statistics. Probably out-of-scope for v0.10 (X-13 is a 100k-LOC FORTRAN beast); STL is the credible substitute.

### T2.6 — DLM / Bayesian Structural TS (`timeseries/bsts/`)
West-Harrison 1997 dynamic linear models, Scott-Varian 2014 BSTS for inference + causal impact analysis.
- Local-level model (random walk + noise) — 1-state DLM, smoothest first commit, ~80 LOC on top of T1.1 Kalman.
- Local-linear-trend + seasonal — 3-state DLM, ~120 LOC.
- BSTS spike-and-slab regressors with Gibbs sampling — ~400 LOC, depends on `prob/mcmc/` which is also absent.

### T2.7 — Recursive estimation (`timeseries/recursive/`)
- Rolling window OLS / rolling-window ARIMA refit.
- Expanding window walk-forward forecasting.
- Online recursive least squares (RLS) — Hayes 1996 §9.
- ~150 LOC; underpins backtesting workflows.

### T2.8 — HAC / Newey-West (`prob/hac.go`)
Heteroscedasticity-and-autocorrelation-consistent standard errors for OLS on time-series residuals (Newey-West 1987 with Bartlett kernel; Andrews 1991 with QS kernel; bandwidth selection via Andrews 1991 / Newey-West 1994). ~150 LOC. Tier 2 because it is the workhorse fix for any time-series-regression std error and is currently missing from `prob/`.

---

## Tier 3 — nice-to-have for SOTA parity (~2000 LOC total)

These are the "feature-complete relative to forecasting/statsmodels" items. Most are big and have substantial design surface.

### T3.1 — Prophet (Taylor-Letham 2017)
Decomposable additive model: trend (linear or logistic) + Fourier-series seasonality + holidays. The decomposition itself is straightforward (~200 LOC); the L-BFGS fit on a Stan-style log-posterior is heavier (~400 LOC) and the holiday calendar bookkeeping is design-heavy. Reasonable as a separate `timeseries/prophet/` subpackage; calls out for a serious API discussion before implementation.

### T3.2 — BATS / TBATS (de Livera-Hyndman-Snyder 2011)
Box-Cox transform + ARMA errors + Trend + (T)rigonometric Seasonality. Generalises ETS to multiple non-integer seasonalities (e.g. 24-hour + 7-day + 365.25-day). ~600 LOC. Builds on T1.4 (ETS) and T1.1 (Kalman).

### T3.3 — Hierarchical reconciliation (`timeseries/hierarchical/`)
- **Bottom-up**, **top-down**, **middle-out** — trivial summing/proportion rules, ~80 LOC.
- **OLS reconciliation** (Hyndman-Ahmed-Athanasopoulos-Shang 2011) — ~120 LOC with linalg.
- **MinT** (Wickramasuriya-Athanasopoulos-Hyndman 2019) — minimum trace optimal reconciliation; the current SOTA. Needs a sample variance-covariance shrinkage (Schäfer-Strimmer 2005). ~250 LOC.
- ~450 LOC total.

### T3.4 — Forecast combination (`timeseries/combine/`)
- Simple averaging.
- Weighted averaging by inverse out-of-sample MSE.
- Bates-Granger 1969 covariance-aware combination.
- Stacking via cross-validation. ~150 LOC.

### T3.5 — Multitaper / Whittle (`signal/multitaper.go`)
- Slepian sequence / DPSS — Walden 1995. ~200 LOC just for the eigenvalue construction.
- Multitaper PSD — average of K tapered periodograms, lower bias-variance tradeoff than Welch. ~80 LOC on top of DPSS.
- Whittle likelihood for spectral-domain ARMA fitting — ~120 LOC; alternative to Kalman likelihood for stationary models.

### T3.6 — Extended Kalman / UKF / particle (`timeseries/kalman/`)
Once T1.1 lands, extend to non-linear:
- **EKF** — Jacobian linearisation; needs `linalg.NumericalJacobian` (already exists). ~80 LOC.
- **UKF** — sigma-point unscented; numerically more robust than EKF. ~150 LOC.
- **Particle filter** (bootstrap, SIR, auxiliary) — ~250 LOC; depends on `prob/sampling/` which exists.

### T3.7 — Specialised / niche models
- **ARFIMA** (Granger-Joyeux-Hosking) — fractional differencing, long-memory ARMA. ~200 LOC; Whittle estimation simpler than time-domain.
- **Mann-Kendall + seasonal Mann-Kendall** — rank-based trend tests for non-parametric time series. ~80 LOC. (Currently absent everywhere.)
- **Theil-Sen** — robust slope estimator. Strangely absent from the entire repo per the 136 audit. ~50 LOC.
- **Zivot-Andrews** — unit-root with structural break. ~100 LOC; depends on T1.6 (ADF).
- **DF-GLS** (Elliott-Rothenberg-Stock 1996) — efficient unit-root. ~80 LOC; depends on T1.6.
- **Anomaly detection** — residual-based (z-score on ARIMA residuals), STL-residual, rolling-percentile. ~120 LOC.

### T3.8 — Cross-correlation / lagged correlation (`signal/ccf.go`)
Strangely absent given `signal.Convolve` exists. Normalised CCF with Bartlett SEs. ~50 LOC. Listed here because the topic prompt enumerates it.

---

## Cross-package overlap

These intersect with topics owned by other agents and should be co-ordinated:

| This Tier | Also owned by | Resolution |
|---|---|---|
| T1.7 Welch / Periodogram | `signal/` (132 / 133) | Belongs in `signal/`, not `timeseries/` — flagged as Tier 1 "must" by 132 already |
| T2.5 STL trend extraction | `prob/` (depends on LOESS being in `prob/regression/`) | LOESS is an absent prerequisite; build LOESS first |
| T3.5 Multitaper / DPSS | `signal/` | Same package call as Welch |
| T3.6 Particle filter | `prob/sampling/` (118-prob-sota) | Particle filter sits at the time-series / sampling boundary; cleanest in `timeseries/kalman/` consuming `prob/sampling/` |
| Changepoint | `changepoint/` package exists | Out of scope here; 037-changepoint-* owns BOCPD/PELT |

---

## R-pattern saturation

- **R-FAMILY-COMPLETENESS: 0/3 unsaturated.** GARCH(1,1) is one corner of a 12-variant family (p,q × Gaussian/t/GED × EGARCH/GJR/IGARCH/FIGARCH). DCC scalar(1,1) is one corner of {DCC, ADCC, cDCC, BEKK, CCC} × (M,N). ARIMA-from-`prob` is a stub. None of the canonical families is filled out.
- **R-EXACT-MLE-VIA-STATE-SPACE: 0/3 unsaturated.** No Kalman, therefore no exact-likelihood ARMA, ETS, DLM, BSTS. The entire state-space lineage is absent.
- **R-DIAGNOSTIC-TRIO (ACF/PACF/Ljung-Box): 1/3 partial.** Ljung-Box exists in `prob/hypothesis.go` (general χ² form) but is not specialised for time-series residual whiteness; ACF/PACF are not exported.
- **R-STATIONARITY-TEST-TRIO (ADF/KPSS/PP): 0/3 absent.**
- **R-DECOMPOSITION-CANON (STL/X-11/SEATS/classical): 0/3 absent.**

---

## Priority ordering (smallest committable Tier 1 deltas first)

If `timeseries/` is to grow, the dependency-respecting order is:

1. **T1.5 ACF/PACF/Yule-Walker** (~120 LOC) — zero new math; reuses the recursion in `levinsonDurbin` that already discards PACF and per-order error variance (136-N6). Cleanest first commit; unblocks all diagnostic plotting.
2. **T1.7 Welch + Periodogram** (~150 LOC, lives in `signal/`) — unblocks spectral diagnostics and Whittle.
3. **T1.1 Kalman primitive** (~250-400 LOC) — load-bearing for everything else state-space.
4. **T1.6 Unit-root tests (ADF/KPSS/PP)** (~250 LOC + MacKinnon table) — independent of Kalman; gates ARIMA workflow.
5. **T1.2 ARIMA proper** (~600 LOC) — depends on T1.1 Kalman; deprecates `prob.ARIMA`.
6. **T1.3 SARIMA** (~150 LOC) — extension of T1.2.
7. **T1.4 ETS state-space** (~400 LOC) — depends on T1.1; supersedes `prob.HoltLinear`.

Tier 2 items follow once Tier 1 lands. Tier 3 (Prophet, BATS/TBATS, MinT, multitaper) is optional for v1.0 but expected for SOTA parity.

Total Tier 1: ~2,000-2,400 LOC across 6 subpackages. Roughly 2-3 weeks of focused work for a single contributor; partitionable across agents because items 1, 2, 4 are independent of items 3, 5, 6, 7.

---

## Two-line summary

`timeseries/` ships GARCH(1,1) + scalar DCC(1,1) + a stub `prob.ARIMA` and that is the entire time-series surface — none of {Kalman, ARMA-MLE, SARIMA, ETS, Theta, Prophet, BATS/TBATS, BSTS/DLM, GARCH(p,q)/EGARCH/GJR/IGARCH/FIGARCH, BEKK/CCC/ADCC, ADF/KPSS/PP, STL/X-11, Welch/multitaper/Whittle, VAR/VECM/Johansen, MinT, HAC, particle/UKF/EKF} exists. Tier 1 floor is ~2,400 LOC across 6 subpackages with a Kalman primitive as load-bearing dependency for the state-space stack; the cheapest non-trivial first commit is exposing PACF + per-order error variance from the existing `levinsonDurbin` recursion (~120 LOC, zero new math, unblocks all ARMA diagnostics).
