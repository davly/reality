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
// Per the math + technique cross-pollination hunts, ~14 downstream services
// across the math-frontier hunt and project-driven hunt cite
// information-geometric divergences as substrate.  Until now every consumer
// has rolled its own KL / JS / MMD over varying conventions (natural-log
// vs base-2, with or without epsilon-smoothing, biased vs unbiased MMD).
// This package centralises the formulas and the convention.
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
