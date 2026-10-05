package prob

import "math"

// ---------------------------------------------------------------------------
// Bayes factors for a binomial proportion.
//
// This file promotes the uniform-prior proportion Bayes factor (BF10) that
// was independently re-derived in the crucible-bridge validator
// (infrastructure/crucible-bridge internal/validator) up into the canonical
// reality substrate. The crucible-bridge implementation integrated the H1
// marginal likelihood numerically with Simpson's rule and could overflow to
// +Inf (and historically vetoed that +Inf in linear space). The substrate
// version evaluates an exact closed form, a finite sum of binomial-coefficient
// ratios, and reports a finite-result guard explicitly so callers decide how
// to treat overwhelming-evidence overflow.
//
// Consumers:
//   - crucible-bridge: proportion-test gate (replaces the in-repo Simpson
//     integration with the substrate closed form)
//   - Any service running a one-sided binomial proportion test against 0.5
// ---------------------------------------------------------------------------

// ProportionBayesFactor10 computes the Bayes factor BF10 for a one-sided
// binomial proportion test of the alternative "p > 0.5" against the point
// null "p = 0.5", given k successes out of n trials.
//
// Hypotheses:
//
//	H0: p = 0.5                       (a point null)
//	H1: p ~ Uniform(0.5, 1)           (density 2 on the upper half)
//
// BF10 = P(data | H1) / P(data | H0), where
//
//	P(data | H0) = C(n,k) * 0.5^n
//	P(data | H1) = integral_{0.5}^{1} C(n,k) p^k (1-p)^{n-k} * 2 dp
//
// The H1 marginal has an exact closed form. With N = n+1, the identity
// C(n,k) * B(k+1, n-k+1) = 1/N gives
//
//	P(data | H1) = (2/N) * P(Beta(k+1, n-k+1) > 1/2),
//
// and for integer shapes that beta tail is a binomial lower tail:
// P(Beta(k+1, n-k+1) > 1/2) = P(Binomial(N, 1/2) <= k)
// = 2^-N * sum_{j=0..k} C(N, j). Dividing by P(data | H0) and using
// N*C(n,k) = (N-k)*C(N,k):
//
//	BF10 = R / (N - k),   R = sum_{j=0..k} C(N, j) / C(N, k).
//
// BF10 is evaluated as a sum of positive terms t_j = C(N, j)/((N-k) C(N, k)),
// generated from t_k = 1/(N-k) by t_{j-1} = t_j * j/(N+1-j), so nothing
// cancels and no partial sum exceeds the result: k = 0 gives exactly
// 1/(n+1). (Forming the tail as 1 - I_{1/2}(k+1, n-k+1) instead cancelled to
// 0 whenever the tail fell below about 1e-16, e.g. k = 0, n = 60.) The sum
// stops once the terms are past their peak and the geometric bound on the
// rest is below 2^-60 of the total, or once it overflows. For
// high-dominance, large-n inputs the true BF10 exceeds math.MaxFloat64 and bf
// is +Inf — this is genuine, infinitely strong evidence, not an error. The
// boolean ok is the finite-result guard: it is true exactly when bf is a
// finite, non-negative number. (The crucible-bridge bug was treating that
// +Inf overflow as a computation failure; callers that want "infinitely
// strong evidence still passes the gate" should test ok || math.IsInf(bf, 1).)
//
// Valid range: n >= 1, 0 <= k <= n.
// Returns: (bf, ok). bf is BF10 (>= 0, may be +Inf). ok is true iff bf is
// finite and non-negative.
// Failure mode: returns (NaN, false) if n < 1, k < 0, or k > n.
// Precision: measured against the exact rational value for every 0 <= k <= n,
// the relative error is at most 5.3e-15 for n <= 1000 (n in {1, 2, 10, 60,
// 200, 1000}), 7.7e-15 at n = 10^4 and 1.7e-14 at n = 10^5: the rounding
// errors of the term recurrence accumulate over the terms that matter, about
// sqrt(n) of them near k = n/2. Every exact value beyond math.MaxFloat64
// gives +Inf.
// Cost: at most k+1 terms (about sqrt(n) near k = n/2); the sum stops early
// once the remaining terms are negligible or it overflows.
// Reference: Jeffreys, H. (1961) "Theory of Probability", 3rd ed., on
// one-sided proportion Bayes factors; Abramowitz & Stegun 26.5.24 (incomplete
// beta function and the binomial distribution).
func ProportionBayesFactor10(k, n int) (bf float64, ok bool) {
	if n < 1 || k < 0 || k > n {
		return math.NaN(), false
	}
	nf := float64(n)
	t := 1 / (nf + 1 - float64(k)) // t_k = 1/(N-k)
	bf = t
	for j := k; j >= 1; j-- {
		r := float64(j) / (nf + 2 - float64(j)) // t_{j-1}/t_j = C(N, j-1)/C(N, j)
		t *= r
		bf += t
		if math.IsInf(bf, 1) {
			break
		}
		// For r < 1 the ratios fall as j falls, so the remaining terms sum
		// to less than t*r/(1-r).
		if r < 1 && t*r < 0x1p-60*bf*(1-r) {
			break
		}
	}
	ok = !math.IsInf(bf, 0) && !math.IsNaN(bf) && bf >= 0
	return bf, ok
}
