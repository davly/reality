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
// version uses the exact closed form via the regularized incomplete beta
// function and reports a finite-result guard explicitly so callers decide
// how to treat overwhelming-evidence overflow.
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
// The H1 marginal has an exact closed form. Using the identity
// C(n,k) * B(k+1, n-k+1) = 1/(n+1) and the regularized incomplete beta
// function I_x(a,b):
//
//	P(data | H1) = (2 / (n+1)) * (1 - I_{0.5}(k+1, n-k+1))
//
// so
//
//	BF10 = [ (2 / (n+1)) * (1 - I_{0.5}(k+1, n-k+1)) ] / [ C(n,k) * 0.5^n ]
//
// The computation is carried out in log-space to avoid intermediate
// over/underflow; the final exponentiation is the only step that can
// overflow. For high-dominance, large-n inputs the true BF10 exceeds
// math.MaxFloat64 and bf is +Inf — this is genuine, infinitely strong
// evidence, not an error. The boolean ok is the finite-result guard: it
// is true exactly when bf is a finite, non-negative number. (The
// crucible-bridge bug was treating that +Inf overflow as a computation
// failure; callers that want "infinitely strong evidence still passes the
// gate" should test ok || math.IsInf(bf, 1).)
//
// Valid range: n >= 1, 0 <= k <= n.
// Returns: (bf, ok). bf is BF10 (>= 0, may be +Inf). ok is true iff bf is
// finite and non-negative.
// Failure mode: returns (NaN, false) if n < 1, k < 0, or k > n.
// Precision: matches a brute-force quadrature oracle to < 1e-6 relative;
// limited by RegularizedBetaInc (~1e-14 absolute).
// Reference: Jeffreys, H. (1961) "Theory of Probability", 3rd ed., on
// one-sided proportion Bayes factors; Press et al., Numerical Recipes,
// section 6.4 (incomplete beta function).
func ProportionBayesFactor10(k, n int) (bf float64, ok bool) {
	if n < 1 || k < 0 || k > n {
		return math.NaN(), false
	}

	// log P(data | H0) = log C(n,k) + n*log(0.5).
	logH0 := logBinomCoeff(k, n) + float64(n)*math.Ln2*(-1)

	// Upper-tail mass of Beta(k+1, n-k+1) above 0.5:
	//   1 - I_{0.5}(k+1, n-k+1).
	// RegularizedBetaInc is exact at the endpoints, so this is well defined
	// for every 0 <= k <= n.
	tail := 1.0 - RegularizedBetaInc(0.5, float64(k+1), float64(n-k+1))
	if tail <= 0 {
		// All posterior mass lies at or below 0.5: H1 (p > 0.5) is
		// effectively impossible given the data, so BF10 -> 0.
		return 0, true
	}

	// log P(data | H1) = log(2) - log(n+1) + log(tail).
	logH1 := math.Ln2 - math.Log(float64(n+1)) + math.Log(tail)

	bf = math.Exp(logH1 - logH0)
	ok = !math.IsInf(bf, 0) && !math.IsNaN(bf) && bf >= 0
	return bf, ok
}

// logBinomCoeff returns log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1).
// Callers must ensure 0 <= k <= n.
func logBinomCoeff(k, n int) float64 {
	return LogGamma(float64(n+1)) - LogGamma(float64(k+1)) - LogGamma(float64(n-k+1))
}
