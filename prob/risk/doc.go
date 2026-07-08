// Package risk implements a convention-arbitrated reference suite of
// portfolio risk and performance metrics: Value-at-Risk (historical,
// Gaussian-parametric, and Cornish-Fisher modified), Conditional VaR /
// Expected Shortfall, downside deviation and the Sortino ratio (in BOTH
// competing denominator conventions), maximum drawdown (from a price series
// OR a return series), the Calmar, Omega and Information ratios, market beta,
// and the annualization helpers that tie them to a calendar.
//
// # Why this exists in Reality
//
// RubberDuck's RiskMetricsService hand-rolls all of these in C#, and a
// 2026-06-25 verify-first review (reviews/RD_RISK_MATH_REVIEW_2026-06-25)
// found its Sortino downside deviation divides the below-target sum of
// squares by (k-1) over ONLY the below-target observations, instead of the
// canonical division by N over the FULL sample. On a 12-point series that is
// a 42% Sortino understatement (code 1.59 vs textbook 2.76) that biases the
// evolutionary keep/pause decision AGAINST exactly the low-downside agents
// the system should keep. The review could not auto-fix it: N-vs-(k-1) "is a
// convention choice" and there was no in-estate canonical reference to
// arbitrate against.
//
// This package IS that arbiter. The whole point of the downside-deviation
// pair below is that the convention choice becomes an EXPLICIT, named,
// golden-file-pinned function — DownsideDeviationFullSample (canonical) vs
// DownsideDeviationNegativesOnly (the legacy RD variant) — turning an
// unfixable judgment call into a mechanical selection. Every other metric is
// pinned the same way: each documents its convention choice in its doc
// comment, and every golden vector is hand-derived from a stated series so
// the answer is checkable against arithmetic, not against a value only this
// package has ever produced.
//
// # Sign and unit conventions (apply to the whole package)
//
//   - Returns are PERIODIC SIMPLE returns expressed as fractions: 0.01 means
//     +1%, -0.02 means -2%. They are NOT log returns and NOT percentages.
//   - VaR and CVaR are reported as POSITIVE loss magnitudes at the stated
//     confidence: HistoricalVaR(returns, 0.95) == 0.055 means "a 0.95-
//     confidence one-period loss of 5.5% of capital". A negative result means
//     the confidence quantile is itself a gain (no loss at that level) and is
//     returned signed, not clamped, so the number stays a faithful negation
//     of the return quantile.
//   - confidence is the coverage level in (0,1), e.g. 0.95 or 0.99; the tail
//     probability is alpha = 1 - confidence.
//   - MAR ("minimum acceptable return") / threshold for the downside metrics
//     is a per-period simple return in the same units as the series; pass 0
//     for the common "downside = below zero" target.
//   - Drawdowns and the maximum drawdown are POSITIVE fractions in [0,1]:
//     0.25 means a 25% peak-to-trough decline.
//
// # This package does not wire any consumer
//
// It is the Go-canonical reference that RubberDuck's C# service (and any
// future consumer in any language) validates its own arithmetic against, per
// Reality's golden-file cross-language model (CONTEXT.md): Go is canonical,
// other substrates check the same worked examples. Adopting these conventions
// inside RiskMetricsService — in particular selecting DownsideDeviationFullSample
// for the keep/pause path — is a deliberate, operator-gated follow-up in the
// C# codebase, not part of this reference.
//
// # Dependencies
//
// Pure Go: the standard library ("math", "sort") plus reality's own prob
// package for the shared Acklam normal quantile (prob.NormalQuantile) and the
// R-7 empirical quantile (prob.Quantile). Zero external dependencies,
// consistent with Reality's Tier-0 law. Every function is pure and
// deterministic; degenerate inputs return NaN (documented per function),
// matching the sibling prob/sharpe.go convention this suite extends.
//
// # References
//
//   - Sortino, F. A., & Price, L. N. (1994). Performance Measurement in a
//     Downside Risk Framework. Journal of Investing 3(3): 59-64.
//   - Bacon, C. R. (2008). Practical Portfolio Performance Measurement and
//     Attribution, 2nd ed. Wiley. (downside deviation full-sample
//     convention; Calmar; Omega; Information ratio conventions.)
//   - Keating, C., & Shadwick, W. F. (2002). A Universal Performance Measure.
//     The Journal of Performance Measurement 6(3): 59-84. (Omega ratio.)
//   - Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional
//     Value-at-Risk. Journal of Risk 2(3): 21-41. (CVaR / Expected Shortfall.)
//   - Jorion, P. (2007). Value at Risk: The New Benchmark for Managing
//     Financial Risk, 3rd ed. McGraw-Hill. (historical and parametric VaR.)
//   - Zangari, P. (1996). A VaR Methodology for Portfolios that Include
//     Options. RiskMetrics Monitor Q1 1996: 4-12. (Cornish-Fisher modified
//     VaR; originally Cornish, E. A., & Fisher, R. A. (1938).)
//   - Young, T. W. (1991). Calmar Ratio: A Smoother Tool. Futures 20(1): 40.
//   - Grinold, R. C., & Kahn, R. N. (2000). Active Portfolio Management, 2nd
//     ed. McGraw-Hill. (Information ratio.)
package risk
