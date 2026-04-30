// Package dcc implements the Dynamic Conditional Correlation (DCC)
// multivariate volatility model of Engle 2002.  DCC layers a time-varying
// correlation matrix on top of univariate GARCH(1,1) marginals.  Unlike
// constant-correlation multivariate GARCH it captures the empirical fact
// that correlations spike during stress.
//
// # Model
//
// Given univariate GARCH-filtered standardised residuals z_t (a vector of
// dimension k), the DCC update is
//
//	Q_t = (1 - alpha - beta) * Qbar  +  alpha * z_{t-1} z_{t-1}^T  +  beta * Q_{t-1}
//	R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
//
// Qbar is the unconditional sample covariance of z, R_t is the conditional
// correlation matrix at time t.  Stationarity requires alpha >= 0,
// beta >= 0, alpha + beta < 1.
//
// # Why this exists in Reality
//
// DCC is the canonical multivariate stress propagation model.  Downstream
// consumers (D37 cross-asset stress, F55 stablecoin EWS, B11 risk-parity
// regime-conditioning) use its conditional correlation as input to risk
// budgeting, contagion propagation, and dispersion / hedging signals.
//
// # MVP scope
//
// This package ships:
//
//   - DCCParams: alpha, beta, Qbar
//   - Update: one-step DCC recursion given the previous Q and current z
//   - CorrelationFromQ: extract R from Q via diagonal normalisation
//   - SampleQbar: unconditional covariance of standardised residuals
//
// Calibration of (alpha, beta) is deferred — most consumers use the
// industry default (alpha = 0.05, beta = 0.93) per Engle 2002 Table 5.
//
// Univariate GARCH calibration of each marginal uses the sibling package
// `timeseries/garch`.  Composition: fit per-asset GARCH -> filter to
// standardised residuals -> SampleQbar -> Update recursion.
//
// # Determinism + allocations
//
// All operations are deterministic.  Update accepts pre-allocated
// in-place buffers for Q.  Zero non-stdlib deps (math only).
//
// # References
//
//   - Engle R. F. (2002). Dynamic conditional correlation: A simple class
//     of multivariate generalized autoregressive conditional
//     heteroskedasticity models.  J. Bus. Econ. Stat. 20(3):339-350.
//   - Engle R. F. (2009). Anticipating Correlations: A New Paradigm for
//     Risk Management.  Princeton.  Chapters 3-4.
package dcc
