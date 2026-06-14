package prob

import (
	"math"
	"sort"
)

// ---------------------------------------------------------------------------
// Empirical sample percentiles / quantiles (R-7 linear interpolation)
// ---------------------------------------------------------------------------
//
// This file provides the general-purpose EMPIRICAL percentile of a sample:
// the value below which a given fraction of the (sorted) observations fall,
// using R-7 linear interpolation between the two closest ranks. This is the
// default percentile method used by NumPy (`numpy.percentile`, interpolation
// "linear"), R's `quantile(type=7)`, Excel's PERCENTILE.INC, and pandas
// `Series.quantile`.
//
// It is semantically distinct from its package neighbours:
//   - Median / TrimmedMean (prob.go) are probability-domain estimators
//     (empty -> 0.5, result clamped to [MinProb, MaxProb]); these functions
//     are general statistics over an arbitrary numeric sample and do NOT clamp.
//   - NormalQuantile (distributions.go) is the inverse CDF of a *distribution*,
//     not of an empirical sample.
//   - conformal.SplitQuantile is the (1-alpha) conformal quantile using the
//     nearest-rank (ceil) convention, deliberately conservative; R-7 here
//     interpolates instead.
//
// Reference: Hyndman, R.J. & Fan, Y. (1996), "Sample Quantiles in Statistical
// Packages", The American Statistician 50(4):361-365 (definition 7, "R-7").
//
// Source: generalised from two independently reinvented implementations —
// hearthstone/internal/hmlr/ingest.go (percentile over sorted []int64, p in
// [0,1]) and insights/internal/topology/builder.go (percentile over []float64,
// p in [0,100]). Both compute the same R-7 interpolation; this is the single
// dep-free Tier-0 primitive.

// Percentile returns the p-th empirical percentile of data using R-7 linear
// interpolation, with p expressed as a PERCENTAGE in [0, 100].
//
// This matches the convention of insights/internal/topology.percentile
// (p in [0,100]). It is implemented in terms of Quantile (p/100).
//
// Formula: q = p/100; pos = q*(n-1); lo = floor(pos); frac = pos - lo;
//          result = sorted[lo] + (sorted[lo+1] - sorted[lo]) * frac
// Valid range: p in [0, 100]; p is clamped to that interval (a p outside the
//          range is treated as the nearest endpoint, never an out-of-bounds
//          index).
// Output range: within [min(data), max(data)].
// Edge cases: returns NaN for empty input (no sample => no percentile, matching
//          the "no evidence" convention but in the general numeric domain where
//          0/0.5 would be a misleading real value); returns data[0] for n==1.
// Precision: one multiply + one lerp; ~15 significant digits.
//
// The input slice is not modified (a copy is sorted internally).
func Percentile(data []float64, p float64) float64 {
	return Quantile(data, p/100.0)
}

// Quantile returns the q-th empirical quantile of data using R-7 linear
// interpolation, with q expressed as a FRACTION in [0, 1].
//
// This matches the convention of hearthstone/internal/hmlr.percentile
// (p in [0,1]), generalised to float64 inputs and to a defensively-copied,
// internally-sorted slice (the hearthstone version requires a pre-sorted
// slice; this one does not).
//
// Formula: pos = q*(n-1); lo = floor(pos); frac = pos - lo;
//          result = sorted[lo] + (sorted[lo+1] - sorted[lo]) * frac
//          (when lo+1 >= n, i.e. q == 1, returns sorted[n-1])
// Valid range: q in [0, 1]; q is clamped to that interval.
// Output range: within [min(data), max(data)].
// Edge cases: returns NaN for empty input; returns data[0] for n==1.
// Precision: one multiply + one lerp; ~15 significant digits.
//
// The input slice is not modified (a copy is sorted internally).
func Quantile(data []float64, q float64) float64 {
	n := len(data)
	if n == 0 {
		return math.NaN()
	}
	if n == 1 {
		return data[0]
	}

	// Clamp the requested quantile to [0, 1] so pos can never index out of
	// range; an out-of-domain request degenerates to the nearest extreme.
	if q < 0 {
		q = 0
	} else if q > 1 {
		q = 1
	}

	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)

	// R-7: position in [0, n-1].
	pos := q * float64(n-1)
	lo := int(math.Floor(pos))
	if lo+1 >= n {
		// pos == n-1 (q == 1): no upper neighbour to interpolate toward.
		return sorted[n-1]
	}
	frac := pos - float64(lo)
	return sorted[lo] + (sorted[lo+1]-sorted[lo])*frac
}
