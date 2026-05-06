// Package infogeo implements information-geometric divergences for discrete
// probability vectors and empirical samples.
//
// A divergence D(p || q) is a non-negative function on a pair of distributions
// that is zero iff p = q (almost everywhere) but is generally not symmetric
// and does not satisfy the triangle inequality.  Information-geometric
// divergences arise as length elements on statistical manifolds and are the
// canonical tool for comparing distributions in maximum-likelihood, model
// selection, regime classification, two-sample testing, and anomaly decay
// tracking.
//
// # Why this exists in Reality
//
// The math + technique cross-pollination hunts identified ~14 candidate
// downstream services that could use information-geometric divergences as
// substrate (regime classification, anomaly detection, two-sample testing,
// model selection, decay tracking).  The named flagship cites
// (relic-insurance, witness, insights) remain hunt-citations not
// import-citations as of 2026-05-05 (verified by substring grep on
// github.com/davly/reality/infogeo across foundation/, infrastructure/,
// sdk/, apps/).  Until now every consumer has rolled its own KL / JS
// / MMD over varying conventions (natural-log vs base-2, with or without
// epsilon-smoothing, biased vs unbiased MMD); this package centralises the
// formulas and the convention.
//
// Consumers (verified):
//   - infogeo/autodiff_test.go:TestKL_AutodiffGradientMatchesQMinusP —
//     pins reverse-mode autodiff's gradient of KL(p || softmax(θ))
//     against the analytic closed form q − p across three (p, θ)
//     cases at 1e-9 tolerance.  Second consumer for the
//     R-CLOSED-FORM-PINNED-TO-AUTODIFF pattern (after F2.a's
//     autodiff × garch parity test); saturation 2/3 toward
//     promotion (S62 overnight, 2026-05-06).
//   - changepoint/infogeo_test.go:TestPosterior_FreshStartConvergence —
//     uses TotalVariation and Hellinger to witness that BOCPD's
//     run-length posterior collapses toward a fresh-start posterior
//     after a changepoint and stays distant from it during stable
//     regimes; first cross-package consumer for infogeo
//     (substrate-internal first-consumer push, S62 2026-05-05).  The
//     test cross-validates both metrics on the same posterior pairs
//     (they must agree on direction) — early evidence that the
//     symmetric-bounded f-divergences in this package are mutually
//     consistent on real distributional inputs.
//
// Flagship first-consumer push remains queued; see
// LimitlessGodfather/reviews/SESSION_62_PROGRESS.md.
//
// # Convention
//
// All logarithms in f-divergences are natural (base e), so KL is in nats.
// Multiply by 1/log(2) to convert to bits.  All divergences accept
// probability vectors that are non-negative and sum to 1 within float64
// precision; functions return ErrInvalidDistribution if not.
//
// # MVP scope
//
// f-divergences:
//   - KL (Kullback-Leibler)
//   - reverse-KL
//   - JS (Jensen-Shannon)
//   - total variation
//   - Hellinger
//   - chi-squared (Pearson)
//   - Renyi-alpha
//
// Bregman divergences:
//   - squared Euclidean
//   - generalised KL (over non-negative measures)
//   - Itakura-Saito (over positive scalars)
//   - generic Bregman from user-supplied (phi, grad-phi)
//
// Empirical divergences:
//   - MMD^2 with Gaussian RBF kernel (biased + unbiased estimators)
//
// # Determinism + allocations
//
// All operations are deterministic and zero-randomness.  Functions follow
// Reality's out-buffer convention where mutation is non-trivial; scalar
// returns require no buffers.  Zero non-stdlib deps.
//
// # References
//
//   - Cover T. M. & Thomas J. A. (2006). Elements of Information Theory,
//     2nd ed., Wiley.  KL, JS, mutual information.
//   - Liese F. & Vajda I. (2006). On divergences and informations in
//     statistics and information theory.  IEEE Trans. Inf. Theory 52(10).
//   - Banerjee A. et al. (2005). Clustering with Bregman divergences.
//     JMLR 6:1705-1749.
//   - Gretton A. et al. (2012). A kernel two-sample test.  JMLR 13:723-773.
//   - Renyi A. (1961). On measures of entropy and information.  Berkeley
//     Symp. on Mathematical Statistics and Probability 4(1):547-561.
package infogeo
