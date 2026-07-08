package risk

import "math"

// InformationRatio returns the Information ratio of a portfolio against a
// benchmark: the mean active (excess) return divided by the tracking error,
//
//	active_i = portfolio[i] - benchmark[i]
//	IR = mean(active) / SampleStdDev(active)
//
// The tracking error is the SAMPLE (1/(N-1)) standard deviation of the active
// returns — the convention pinned here, consistent with treating the observed
// active-return series as a sample. It measures active return earned per unit
// of active risk taken against the benchmark. Reference: Grinold & Kahn (2000).
//
// Valid range: equal-length slices with N >= 2 observations. Returns NaN if
// the lengths differ, or N < 2; returns +Inf (or -Inf) when tracking error is
// 0 but mean active return is positive (or negative).
// Precision: inherits sampleStdDev; ~15 significant digits.
func InformationRatio(portfolio, benchmark []float64) float64 {
	n := len(portfolio)
	if n < 2 || n != len(benchmark) {
		return math.NaN()
	}
	active := make([]float64, n)
	for i := 0; i < n; i++ {
		active[i] = portfolio[i] - benchmark[i]
	}
	te := sampleStdDev(active)
	return mean(active) / te
}

// Beta returns the market beta of an asset: the sensitivity of the asset's
// returns to the market's returns,
//
//	Beta = Cov(asset, market) / Var(market)
//
// Beta is a RATIO of a covariance to a variance computed with the same
// denominator, so the population-vs-sample (1/N vs 1/(N-1)) choice cancels and
// the result is convention-invariant; population moments are used internally.
// Beta = 1 moves one-for-one with the market, > 1 amplifies it, < 1 dampens
// it, and negative beta moves opposite to it. Reference: standard CAPM /
// market-model regression slope.
//
// Valid range: equal-length slices with N >= 2; the market must have non-zero
// variance. Returns NaN if the lengths differ, N < 2, or market variance is 0
// (beta is undefined against a constant market).
// Precision: two passes; ~15 significant digits.
func Beta(asset, market []float64) float64 {
	n := len(asset)
	if n < 2 || n != len(market) {
		return math.NaN()
	}
	varMkt := variancePopulation(market)
	if varMkt == 0 {
		return math.NaN()
	}
	return covariancePopulation(asset, market) / varMkt
}
