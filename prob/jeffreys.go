package prob

import "math"

// ---------------------------------------------------------------------------
// Jeffreys prior and quality-weighted dominance.
//
// These functions implement the core Forge epistemology primitives that were
// identified as Type 1 universals across all 5 Wave 1 substrates (Haskell,
// Rust no_std, Prolog, R, x86-64 ASM). They use the Jeffreys prior
// Beta(0.5, 0.5) as the canonical uninformative prior for binomial
// proportions.
//
// Missing from canonical reality since Session 25 (P1 open item). Every
// blind-build substrate independently rediscovered these same three
// functions with identical semantics.
//
// Consumers:
//   - All Forge-aware services (dominance calculation)
//   - Triage:      confidence-weighted sorting
//   - Delve:       cross-pollination quality assessment
//   - Workshop:    assay tool uses Jeffreys formula
// ---------------------------------------------------------------------------

// JeffreysConfidence computes the posterior mean confidence for a binomial
// proportion using the Jeffreys prior Beta(0.5, 0.5).
//
// Formula: (successes + 0.5) / (successes + failures + 1.0)
// This is the posterior mean of Beta(successes + 0.5, failures + 0.5).
//
// Valid range: successes >= 0, failures >= 0
// Output range: (0, 1) — never returns exactly 0 or 1 due to the prior
// Precision: exact (float64 arithmetic)
// Reference: Jeffreys, H. (1946) "An Invariant Form for the Prior
// Probability in Estimation Problems"
func JeffreysConfidence(successes, failures float64) float64 {
	if successes < 0 {
		successes = 0
	}
	if failures < 0 {
		failures = 0
	}
	return (successes + 0.5) / (successes + failures + 1.0)
}

// Alternative holds one competing alternative for quality-weighted dominance.
type Alternative struct {
	// DominanceRate is the observed dominance rate in [0, 1].
	DominanceRate float64

	// Quality is the quality weight in (0, 1]. Higher = more reliable.
	Quality float64
}

// QualityWeightedDominance computes the quality-weighted dominance score
// across a set of competing alternatives.
//
// Each alternative's dominance rate is weighted by its quality. The result
// is the weighted mean, which gives more influence to higher-quality
// observations.
//
// Formula: sum(rate_i * quality_i) / sum(quality_i)
// Valid range: each rate in [0,1], each quality in (0,1]
// Returns 0.5 (maximum uncertainty) if no valid alternatives
// Precision: accumulated float64 summation error
func QualityWeightedDominance(alternatives []Alternative) float64 {
	if len(alternatives) == 0 {
		return 0.5
	}

	totalWeight := 0.0
	weightedSum := 0.0

	for _, alt := range alternatives {
		q := alt.Quality
		if q <= 0 {
			continue
		}
		if q > 1 {
			q = 1
		}
		rate := alt.DominanceRate
		if rate < 0 {
			rate = 0
		}
		if rate > 1 {
			rate = 1
		}
		totalWeight += q
		weightedSum += rate * q
	}

	if totalWeight == 0 {
		return 0.5
	}

	return weightedSum / totalWeight
}

// Verdict represents the three-way verdict for a dominance comparison.
type Verdict string

const (
	// VerdictDominates means the subject conclusively dominates.
	VerdictDominates Verdict = "dominates"

	// VerdictUncertain means there is insufficient evidence to decide.
	VerdictUncertain Verdict = "uncertain"

	// VerdictDominated means the subject is conclusively dominated.
	VerdictDominated Verdict = "dominated"
)

// ThreeWayVerdict classifies a dominance rate into one of three verdicts.
// This is the canonical Forge three-way result used across all substrates.
//
// The thresholds use Wilson confidence intervals: if the lower bound of the
// 95% CI exceeds 0.5, the verdict is "dominates"; if the upper bound is
// below 0.5, the verdict is "dominated"; otherwise "uncertain".
//
// Valid range: rate in [0,1], observations > 0
// Returns VerdictUncertain if observations <= 0
// Reference: Wilson, E.B. (1927) — same CI used in prob.go
func ThreeWayVerdict(rate float64, observations int) Verdict {
	if observations <= 0 {
		return VerdictUncertain
	}

	const z = 1.96 // 95% confidence
	low, high := WilsonConfidenceInterval(rate, observations, z)

	switch {
	case low > 0.5:
		return VerdictDominates
	case high < 0.5:
		return VerdictDominated
	default:
		return VerdictUncertain
	}
}

// EMA computes the exponential moving average for a new observation.
// This is used for online tracking of dominance rates over time.
//
// Formula: alpha * newValue + (1 - alpha) * previous
// Valid range: alpha in (0, 1]; alpha = 0 returns previous unchanged
// Precision: exact (float64 arithmetic)
// Reference: standard exponential smoothing; Hunter (1986)
func EMA(previous, newValue, alpha float64) float64 {
	if alpha <= 0 {
		return previous
	}
	if alpha > 1 {
		alpha = 1
	}
	return alpha*newValue + (1-alpha)*previous
}

// JeffreysKLDivergence computes the symmetrised Kullback-Leibler divergence
// (Jeffreys divergence) between two Bernoulli distributions with parameters
// p and q.
//
// Formula: (p - q) * (log(p) - log(q)) + (p - q) * (log(1-q) - log(1-p))
// Simplified: (p - q) * log(p*(1-q) / (q*(1-p)))
//
// Valid range: p, q in (0, 1); returns +Inf if either is exactly 0 or 1
// Precision: ~15 significant digits (float64 log)
// Reference: Jeffreys, H. (1946)
func JeffreysKLDivergence(p, q float64) float64 {
	if p <= 0 || p >= 1 || q <= 0 || q >= 1 {
		return math.Inf(1)
	}
	return (p - q) * math.Log(p*(1-q)/(q*(1-p)))
}
