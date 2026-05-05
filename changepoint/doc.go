// Package changepoint implements Bayesian Online Change-Point Detection
// (BOCPD) per Adams-MacKay 2007.
//
// BOCPD is a streaming algorithm that maintains a posterior distribution over
// the run-length r_t — the number of timesteps since the most recent change-
// point. After observing a new datum x_t, the run-length posterior is updated
// recursively:
//
//	p(r_t = 0     | x_{1:t}) = sum_{r-} H(r-) * pi(r-, x_t) * p(r-, x_{1:t-1})
//	p(r_t = r-+1  | x_{1:t}) = (1 - H(r-)) * pi(r-, x_t) * p(r-, x_{1:t-1})
//
// where:
//   - H(r) is the hazard function (probability of a change-point given run-
//     length r). For a constant hazard λ, H(r) = 1/λ.
//   - pi(r, x) = p(x | x_{(t-r):t-1}) is the posterior predictive of the
//     observation x given the r most recent observations under the conjugate
//     observation model. Closed-form for Normal-Inverse-Gamma.
//
// The posterior is renormalised at each step to sum to 1. A run-length
// truncation R_max (default 500) bounds memory and computation per step.
//
// # Why this exists in Reality
//
// The math + technique cross-pollination hunts identified 24+ candidate
// downstream consumers for BOCPD as substrate.  Those are hunt-citations,
// not import-citations: this package has zero production consumers
// ecosystem-wide as of 2026-05-05 (verified by substring grep on
// github.com/davly/reality/changepoint across foundation/, infrastructure/,
// sdk/, apps/, and the named candidate flagships relic-insurance, triage,
// witness, watchtower, narrator).  Hunt cross-references for traceability:
// lens 35.1 of REALITY_MATH_HUNT; composes A01.2, A02.2, A06, A09, B11, B20,
// D33, D38, E50, F55, F56 in RUBBERDUCK_REVOLUTIONARY_TECHNIQUES.  Until now
// every candidate consumer has hand-rolled their own regime detector, with
// each rolling a different bug; this package centralises the math so the
// first real consumer can adopt without inventing.  First-consumer push
// queued; see LimitlessGodfather/reviews/SESSION_62_PROGRESS.md.
//
// # Algorithm
//
// At each time step t with observation x_t:
//
//  1. Compute predictive probabilities pi_r = p(x_t | r) for each run-length
//     hypothesis r in [0, R_max]. With the Normal-Inverse-Gamma conjugate
//     prior, this is a Student's-t distribution with degrees of freedom
//     2*alpha_r, location mu_r, and scale beta_r * (kappa_r + 1) /
//     (alpha_r * kappa_r).
//  2. Compute the hazard probability H(r) for each run-length. With constant
//     hazard rate lambda, H(r) = 1/lambda.
//  3. The change-point growth probability is the mass that flows from each
//     run-length to r+1 (no change-point):
//     P(r+1) += pi_r * (1 - H(r)) * P(r)
//  4. The change-point reset probability is the mass that flows to r=0:
//     P(0)   += pi_r * H(r) * P(r)
//  5. Renormalise P so it sums to 1.
//  6. Update sufficient statistics (mu_r, kappa_r, alpha_r, beta_r) for each
//     run-length r > 0 using the Bayesian update formula for Normal-Inverse-
//     Gamma. The r=0 hypothesis uses the prior.
//
// # Reference
//
// Adams, R. P. and MacKay, D. J. C. (2007). "Bayesian Online Changepoint
// Detection." arXiv:0710.3742.
//
// # Determinism
//
// All operations are bit-stable: no randomness, no parallelism. Same input
// produces identical output across runs.
package changepoint
