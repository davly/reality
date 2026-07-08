package risk

import "math"

// ---------------------------------------------------------------------------
// Downside deviation — the two competing denominator conventions, named.
//
// This pair is the reason the package exists. RubberDuck's 2026-06-25 risk
// review found the Sortino downside deviation divided the below-target sum of
// squares by (k-1) over only the below-target observations, where the
// canonical target semi-deviation divides by N over the FULL sample. The
// review ruled it "a convention choice" it could not auto-fix for lack of an
// arbiter. Exporting both estimators under honest names IS the arbiter: the
// choice becomes a function selection, not an accident.
// ---------------------------------------------------------------------------

// DownsideDeviationFullSample returns the CANONICAL target semi-deviation of
// returns below the minimum acceptable return mar:
//
//	sqrt( (1/N) * sum_i min(0, returns[i] - mar)^2 )
//
// Every observation contributes to the mean (division by the FULL sample size
// N); observations at or above mar contribute a zero squared shortfall but
// still count in the denominator. This is the Sortino & Price (1994) /
// Bacon (2008) definition and the correct denominator for the Sortino ratio.
//
// Valid range: N >= 1. Returns NaN for an empty slice. Returns 0 when no
// observation falls below mar (a series that never underperforms the target
// has zero downside risk by definition).
// Precision: one pass, one math.Sqrt; ~15 significant digits.
func DownsideDeviationFullSample(returns []float64, mar float64) float64 {
	n := len(returns)
	if n == 0 {
		return math.NaN()
	}
	var ss float64
	for _, r := range returns {
		if d := r - mar; d < 0 {
			ss += d * d
		}
	}
	return math.Sqrt(ss / float64(n))
}

// DownsideDeviationNegativesOnly returns the LEGACY variant that RubberDuck's
// RiskMetricsService uses: the sum of squared below-target shortfalls divided
// by (k-1), where k is the COUNT of below-target observations only:
//
//	sqrt( (1/(k-1)) * sum_{i: returns[i] < mar} (returns[i] - mar)^2 )
//
// Because it divides by a smaller denominator (k-1 instead of N) over the same
// numerator, it OVERSTATES downside deviation relative to the full-sample
// convention, which in turn UNDERSTATES the Sortino ratio — the exact 42%
// bias the RD review measured (code Sortino 1.59 vs textbook 2.76). It is
// provided so a consumer can reproduce the legacy number exactly and see the
// gap, NOT because it is recommended. Prefer DownsideDeviationFullSample.
//
// Valid range: at least 2 below-target observations (k >= 2). Returns NaN for
// an empty slice or when fewer than 2 observations fall below mar (the (k-1)
// sample denominator is 0 or negative).
// Precision: one pass, one math.Sqrt; ~15 significant digits.
func DownsideDeviationNegativesOnly(returns []float64, mar float64) float64 {
	if len(returns) == 0 {
		return math.NaN()
	}
	var ss float64
	var k int
	for _, r := range returns {
		if d := r - mar; d < 0 {
			ss += d * d
			k++
		}
	}
	if k < 2 {
		return math.NaN()
	}
	return math.Sqrt(ss / float64(k-1))
}

// SortinoRatioFullSample returns the Sortino ratio using the canonical
// full-sample downside deviation:
//
//	(mean(returns) - mar) / DownsideDeviationFullSample(returns, mar)
//
// This is the recommended estimator. Higher is better; the excess return per
// unit of downside (target semi-) deviation.
//
// Valid range: N >= 1 and at least one below-target observation. Returns NaN
// for an empty slice; returns +Inf (or -Inf) when there is no downside risk
// but the mean beats (or trails) mar, matching the mathematical limit of a
// finite excess over a zero denominator.
// Precision: inherits DownsideDeviationFullSample; ~15 significant digits.
func SortinoRatioFullSample(returns []float64, mar float64) float64 {
	if len(returns) == 0 {
		return math.NaN()
	}
	dd := DownsideDeviationFullSample(returns, mar)
	return (mean(returns) - mar) / dd
}

// SortinoRatioNegativesOnly returns the Sortino ratio computed with the legacy
// DownsideDeviationNegativesOnly denominator — the RubberDuck variant. It is
// provided to reproduce the legacy verdict and quantify the bias against
// SortinoRatioFullSample, not as a recommended statistic.
//
// Valid range: N >= 1 and at least 2 below-target observations. Returns NaN
// otherwise (propagated from DownsideDeviationNegativesOnly).
// Precision: inherits DownsideDeviationNegativesOnly; ~15 significant digits.
func SortinoRatioNegativesOnly(returns []float64, mar float64) float64 {
	if len(returns) == 0 {
		return math.NaN()
	}
	dd := DownsideDeviationNegativesOnly(returns, mar)
	return (mean(returns) - mar) / dd
}

// OmegaRatio returns the Omega ratio of returns at the given threshold: the
// probability-weighted ratio of gains above the threshold to losses below it,
//
//	Omega(threshold) = sum_i max(0, returns[i] - threshold)
//	                   / sum_i max(0, threshold - returns[i])
//
// Omega captures the entire return distribution (all moments), not just the
// first two: Omega > 1 means the mass of gains above the threshold outweighs
// the mass of shortfalls below it. At the mean threshold Omega == 1.
// Reference: Keating & Shadwick (2002).
//
// Valid range: N >= 1. Returns NaN for an empty slice. Returns +Inf when no
// observation falls below the threshold (no downside mass), which is the
// correct limit of a positive numerator over a zero denominator.
// Precision: one pass; exact up to float64 summation.
func OmegaRatio(returns []float64, threshold float64) float64 {
	if len(returns) == 0 {
		return math.NaN()
	}
	var gains, losses float64
	for _, r := range returns {
		if d := r - threshold; d > 0 {
			gains += d
		} else {
			losses += -d
		}
	}
	if losses == 0 {
		return math.Inf(1)
	}
	return gains / losses
}
