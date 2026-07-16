package risk

import (
	"math"
	"sort"

	"github.com/davly/reality/prob"
)

// HistoricalVaR returns the historical-simulation Value-at-Risk of a return
// series at the given confidence level, as a POSITIVE loss magnitude:
//
//	VaR_c = -Quantile(returns, 1 - c)
//
// where Quantile is the empirical R-7 linearly-interpolated quantile
// (prob.Quantile) — the same convention NumPy's default percentile uses.
// With confidence 0.95 the 5%-worst return quantile is negated: if the 5%
// tail return is -0.055, VaR is +0.055. A negative VaR means even the tail
// quantile is a gain and is returned signed, not clamped.
//
// Convention pinned: R-7 interpolation on the raw return series (no
// distributional assumption). This is one of several defensible historical-VaR
// quantile conventions (others use nearest-rank or a specific order
// statistic); R-7 is chosen for consistency with prob.Quantile so the whole
// estate shares one empirical-quantile definition.
//
// Valid range: N >= 1; confidence in (0,1). Returns NaN for an empty slice or
// a confidence outside (0,1).
// Precision: inherits prob.Quantile (~15 significant digits).
func HistoricalVaR(returns []float64, confidence float64) float64 {
	if len(returns) == 0 || confidence <= 0 || confidence >= 1 {
		return math.NaN()
	}
	return -prob.Quantile(returns, 1.0-confidence)
}

// HistoricalCVaR returns the historical-simulation Conditional VaR (a.k.a.
// Expected Shortfall): the average loss in the tail beyond the VaR level, as a
// POSITIVE loss magnitude,
//
//	CVaR_c = -mean( the nTail worst returns ),  nTail = ceil((1 - c) * N)
//
// Convention pinned: the tail size is the CEILING of (1-c)*N — the
// nearest-rank, deliberately-conservative count of worst observations that
// define the tail (this matches RubberDuck's ceiling-tail historical CVaR and
// the Basel-style historical-simulation ES used in backtesting). At least one
// observation always enters the tail. CVaR >= VaR by construction (the tail
// average is at least as bad as the tail boundary). The ceiling is evaluated
// robustly against IEEE-754 rounding (see tailCount) so e.g. c=0.95, N=100
// yields the decimal-exact 5-observation tail, not 6.
//
// Valid range: N >= 1; confidence in (0,1). Returns NaN for an empty slice or
// a confidence outside (0,1).
// Precision: one sort + one mean; ~15 significant digits.
func HistoricalCVaR(returns []float64, confidence float64) float64 {
	n := len(returns)
	if n == 0 || confidence <= 0 || confidence >= 1 {
		return math.NaN()
	}
	nTail := tailCount(n, confidence)
	if nTail < 1 {
		nTail = 1
	}
	if nTail > n {
		nTail = n
	}
	sorted := make([]float64, n)
	copy(sorted, returns)
	sort.Float64s(sorted)
	var s float64
	for i := 0; i < nTail; i++ {
		s += sorted[i]
	}
	return -(s / float64(nTail))
}

// ParametricVaR returns the Gaussian (variance-covariance / delta-normal)
// Value-at-Risk from a return distribution's mean and standard deviation, as a
// POSITIVE loss magnitude:
//
//	VaR_c = -(mean + stdDev * z_{1-c}) = stdDev * z_c - mean
//
// where z_c = Phi^{-1}(c) is the standard-normal quantile (prob.NormalQuantile)
// and z_{1-c} = -z_c by symmetry. It assumes returns are normally distributed;
// for the same inputs it equals CornishFisherVaR with skew = 0 and excess
// kurtosis = 0.
//
// Valid range: stdDev >= 0; confidence in (0,1). Returns NaN otherwise.
// Precision: dominated by the Acklam quantile (relative error < 1.15e-9).
func ParametricVaR(mean, stdDev, confidence float64) float64 {
	if stdDev < 0 || confidence <= 0 || confidence >= 1 {
		return math.NaN()
	}
	z := prob.NormalQuantile(confidence, 0, 1)
	return stdDev*z - mean
}

// CornishFisherVaR returns the Cornish-Fisher "modified" Value-at-Risk, which
// corrects the Gaussian VaR for skewness and excess kurtosis by expanding the
// normal quantile:
//
//	z_cf = z + (z^2-1)/6 * S + (z^3-3z)/24 * K - (2z^3-5z)/36 * S^2
//	VaR_c = -(mean + stdDev * z_cf)
//
// where z = z_{1-c} = Phi^{-1}(1-c) is the LOWER-tail standard-normal
// quantile, S is skewness (0 for a symmetric distribution) and K is EXCESS
// kurtosis (0 for a normal distribution; pass excess = raw - 3). Negative skew
// and positive excess kurtosis both fatten the loss tail and increase the
// modified VaR above the Gaussian figure. With S = 0 and K = 0 the expansion
// collapses to z_cf = z and this equals ParametricVaR exactly.
//
// Reference: Zangari (1996); Cornish & Fisher (1938). The expansion is an
// approximation valid for moderate S, K; for extreme moments it can become
// non-monotone in the quantile and should be used with care.
//
// Valid range: stdDev >= 0; confidence in (0,1). Returns NaN otherwise.
// Precision: dominated by the Acklam quantile (relative error < 1.15e-9); the
// closed-form expansion is itself an approximation to the true modified
// quantile.
func CornishFisherVaR(mean, stdDev, skew, excessKurtosis, confidence float64) float64 {
	if stdDev < 0 || confidence <= 0 || confidence >= 1 {
		return math.NaN()
	}
	// Lower-tail quantile z_{1-c} (negative for c > 0.5).
	z := prob.NormalQuantile(1.0-confidence, 0, 1)
	z2 := z * z
	z3 := z2 * z
	zcf := z +
		(z2-1.0)/6.0*skew +
		(z3-3.0*z)/24.0*excessKurtosis -
		(2.0*z3-5.0*z)/36.0*skew*skew
	return -(mean + stdDev*zcf)
}

// ParametricCVaR returns the Gaussian Conditional VaR / Expected Shortfall
// from a return distribution's mean and standard deviation, as a POSITIVE loss
// magnitude:
//
//	CVaR_c = -mean + stdDev * phi(z_c) / (1 - c)
//
// where z_c = Phi^{-1}(c) and phi is the standard-normal PDF. This is the
// closed-form expected loss in the Gaussian tail beyond VaR_c. For the
// standard normal it reduces to the familiar phi(z_c)/(1-c) hazard-rate form
// (e.g. 2.0627 at c = 0.95, 2.6652 at c = 0.99).
//
// Valid range: stdDev >= 0; confidence in (0,1). Returns NaN otherwise.
// Precision: dominated by the Acklam quantile (relative error < 1.15e-9); the
// PDF is exact to ~15 significant digits.
func ParametricCVaR(mean, stdDev, confidence float64) float64 {
	if stdDev < 0 || confidence <= 0 || confidence >= 1 {
		return math.NaN()
	}
	z := prob.NormalQuantile(confidence, 0, 1)
	pdf := math.Exp(-0.5*z*z) / math.Sqrt(2.0*math.Pi)
	return -mean + stdDev*pdf/(1.0-confidence)
}

// tailCount computes the ceiling-tail observation count
//
//	nTail = ceil((1 - c) * n)
//
// robustly against IEEE-754 rounding. The naive
// `math.Ceil((1.0-confidence)*float64(n))` is off-by-one whenever the true
// product is an exact integer k but the float product evaluates to
// k + epsilon — and that hits the FLAGSHIP confidence levels: 1-0.95 is
// 0.050000000000000044 in float64, so c=0.95, n=100 gives a float product
// of 5.000000000000004 -> Ceil 6, but the decimal-exact tail is 5. The
// inflated tail silently averages one extra (less-bad) observation into
// the CVaR, mis-calibrating the pinned Basel-style convention. We snap a
// product that sits within a tight relative epsilon of an integer back to
// that integer before taking the ceiling — the same guard as the
// prob/conformal conformalRank fix (gridlock ec7c239 lineage); all values
// the float path already gets right are unchanged.
func tailCount(n int, confidence float64) int {
	prod := (1.0 - confidence) * float64(n)
	nearest := math.Round(prod)
	const eps = 1e-9
	if math.Abs(prod-nearest) <= eps*math.Max(1.0, math.Abs(prod)) {
		prod = nearest
	}
	return int(math.Ceil(prod))
}
