package transport

import (
	"math"
	"sort"
)

// IQRNormalise returns a robust z-score-style normalisation of samples
// by the inter-quartile range, anchored to the median.  Concretely:
//
//	out[i] = (samples[i] - median(samples)) / IQR(samples)
//
// where IQR = Q_75 - Q_25 with linear interpolation on the sorted
// array (the same quantile convention used by RubberDuck's
// `OptimalTransport.QuantileFromSorted`).  When IQR is zero (all
// finite values are identical), every output is zero — this matches
// the RubberDuck `Wasserstein1DDetailed` convention of returning
// `NormalizedDistance = 0.0` in the degenerate case.
//
// NaN and ±Inf samples are filtered out before computing the median /
// IQR; if every sample is non-finite the input is returned unchanged
// (callers receive a clear signal that nothing was normalised).
//
// IQR-based normalisation is robust to heavy-tailed distributions in
// the way standard deviation is not (Tukey 1977).  For RubberDuck's
// regime-shift detection consumer this is the canonical choice
// because regime windows often contain outliers whose contribution to
// stddev dominates a signal-of-interest in the bulk.
//
// The input slice is not mutated.
func IQRNormalise(samples []float64) []float64 {
	if len(samples) == 0 {
		return []float64{}
	}

	// Filter NaN/Inf for IQR computation while keeping original
	// positions for the output.
	filtered := make([]float64, 0, len(samples))
	for _, s := range samples {
		if !math.IsNaN(s) && !math.IsInf(s, 0) {
			filtered = append(filtered, s)
		}
	}

	out := make([]float64, len(samples))
	if len(filtered) == 0 {
		copy(out, samples)
		return out
	}

	sort.Float64s(filtered)
	med := quantileFromSorted(filtered, 0.50)
	q25 := quantileFromSorted(filtered, 0.25)
	q75 := quantileFromSorted(filtered, 0.75)
	iqr := q75 - q25

	if iqr == 0 {
		// Degenerate IQR — every output is 0.0 (matches the
		// RubberDuck `iqr > 0 ? distance/iqr : 0.0` convention).
		// NaN/Inf inputs in the original slice are preserved.
		for i, s := range samples {
			if math.IsNaN(s) || math.IsInf(s, 0) {
				out[i] = s
				continue
			}
			out[i] = 0
		}
		return out
	}

	for i, s := range samples {
		if math.IsNaN(s) || math.IsInf(s, 0) {
			out[i] = s
			continue
		}
		out[i] = (s - med) / iqr
	}
	return out
}

// quantileFromSorted is the linear-interpolation quantile estimator
// used by RubberDuck's OptimalTransport.QuantileFromSorted.  Given a
// pre-sorted, non-empty slice and a quantile q in [0, 1] it returns
//
//	v[lo] * (1 - frac) + v[hi] * frac
//
// where lo = floor(q * (len-1)), hi = min(lo+1, len-1), frac = pos -
// lo.  The single-element case returns that element.  Callers must
// pre-sort and pre-filter for non-finite values; this helper is
// intentionally tight because it is on the hot path of Wasserstein1D.
func quantileFromSorted(sorted []float64, q float64) float64 {
	n := len(sorted)
	if n == 1 {
		return sorted[0]
	}
	pos := q * float64(n-1)
	lo := int(math.Floor(pos))
	hi := lo + 1
	if hi > n-1 {
		hi = n - 1
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

// filterAndSortFinite returns a sorted copy of samples with NaN/Inf
// removed.  Never mutates the input.  Used internally by
// Wasserstein1D and the IQR normalisation; matches the FilterAndSort
// helper in the C# reference exactly.
func filterAndSortFinite(samples []float64) []float64 {
	out := make([]float64, 0, len(samples))
	for _, s := range samples {
		if !math.IsNaN(s) && !math.IsInf(s, 0) {
			out = append(out, s)
		}
	}
	sort.Float64s(out)
	return out
}
