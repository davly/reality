// Package garch implements the GARCH(1,1) volatility model:
// generalised autoregressive conditional heteroskedasticity.
//
//	sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
//
// where eps_t are mean-corrected residuals.  GARCH is the canonical
// volatility model for financial returns; it captures volatility
// clustering (today's vol depends on yesterday's shock and yesterday's
// vol) and gives a recursive forecast of multi-step variance.
//
// # Why this exists in Reality
//
// Per the math + technique cross-pollination hunts, GARCH-flex is one of
// the six Tier-1 Foundation substrates with ~14 downstream cites:
//
//   - D31 vol-target / risk-budgeting
//   - D33 regime-aware stop loss
//   - D34 drawdown-constrained Kelly
//   - D37 cross-asset stress scaling
//   - A04 news-momentum half-life
//   - F55 stablecoin EWS
//
// Without a centralised GARCH primitive every consumer rolls a different
// vol-update recursion — different priors on omega, different constraints
// on alpha + beta < 1, different MLE schemes — and downstream consumers
// disagree about volatility numerically.  This package fixes the math.
//
// # MVP scope
//
// GARCH(1,1) only.  General GARCH(p,q) and EGARCH / GJR / FIGARCH are
// deferred to v2.  The DCC-GARCH multivariate extension lives in the
// sibling package `timeseries/dcc`.
//
// API:
//
//   - Model struct with parameters Omega, Alpha, Beta and uncondVar
//   - Fit: Tikhonov-regularised MLE using autodiff-supplied gradients
//     (per PLAN_RISKS.md R3 mitigation for ill-posed calibration)
//   - Filter: forward recursion that returns conditional variance series
//     and standardised residuals for downstream DCC
//   - Simulate: deterministic-given-shocks Monte Carlo path generator
//   - ForecastVariance: h-step ahead conditional variance under the model
//
// # Constraints + reparameterisation
//
// Stationarity requires omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1.
// Calibration uses unconstrained reparameterisation
//
//	omega = exp(log_omega)
//	(alpha, beta, gamma) = softmax(theta_alpha, theta_beta, theta_gamma)
//
// where gamma absorbs the slack 1 - alpha - beta.  This guarantees the
// stationarity constraint is satisfied at every iteration.
//
// # Determinism + allocations
//
// All operations are deterministic.  Simulate accepts a slice of pre-drawn
// standard-normal shocks so the function itself contains no randomness.
// Zero non-stdlib deps (math only).
//
// # References
//
//   - Bollerslev T. (1986). Generalized autoregressive conditional
//     heteroskedasticity.  J. Econometrics 31(3):307-327.
//   - Engle R. F. (1982). Autoregressive conditional heteroscedasticity
//     with estimates of the variance of United Kingdom inflation.
//     Econometrica 50(4):987-1008.
//   - Francq C. & Zakoian J.-M. (2010). GARCH Models: Structure,
//     Statistical Inference and Financial Applications.  Wiley.
package garch
