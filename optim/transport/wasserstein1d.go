package transport

import "math"

// WassersteinResult is the diagnostic-bearing return type of
// Wasserstein1DDetailed.  It mirrors RubberDuck's record-typed
// `WassersteinResult(Distance, SampleSizeP, SampleSizeQ,
// NormalizedDistance)` in `flagships/rubberduck/RubberDuck.Core/
// Analysis/OptimalTransport.cs:17-22`.  Distance is the raw W_p
// distance, NormalizedDistance is Distance / IQR(combined).
type WassersteinResult struct {
	Distance           float64
	SampleSizeP        int
	SampleSizeQ        int
	NormalizedDistance float64
}

// Wasserstein1D returns the closed-form Wasserstein-p distance between
// two empirical 1-D distributions u and v.  For p = 1 (the
// Earth-Mover's-distance case) the result is
//
//	W_1(u, v) = integral_0^1 | F_u^{-1}(t) - F_v^{-1}(t) | dt
//
// which on n equal-size samples reduces to
//
//	W_1(u, v) = (1/n) * sum_i | u_(i) - v_(i) |
//
// where u_(i), v_(i) are the order statistics.  For unequal sizes we
// average over the maximum of the two lengths via linear-interpolated
// quantiles on the (k + 0.5) / n grid — this is the convention used
// by RubberDuck's reference C# implementation, chosen so the equal-
// length closed form is recovered exactly when |u| == |v|.
//
// p must be a finite real number with p >= 1 (Wasserstein-p is a
// metric on probability measures with finite p-th moment for any
// p >= 1; Vaserstein 1969 / Villani 2009 §6).  ErrInvalidP is
// returned for p < 1 or non-finite p.
//
// NaN and ±Inf samples are filtered out before computing the order
// statistics, matching RubberDuck's `FilterAndSort`.  An empty
// distribution after filtering returns `(NaN, nil)` — the closed-
// form 1D path mirrors the C# `double.NaN` return rather than raising
// an error so the cross-substrate-precision contract (≤1e-12 byte-
// for-byte equality on the existing 218-LoC RubberDuck test corpus)
// holds without translation glue.
//
// Complexity O((n + m) log(n + m)) due to sorting; O(n + m) extra
// space for the sorted copies.  The hot path on
// (50K, 50K) inputs runs in roughly a few milliseconds on commodity
// hardware (RubberDuck's perf gate; we replicate it in
// transport_test.go).
//
// Cross-substrate parity (R80b): for p = 1 this implementation is
// byte-for-byte equivalent to the RubberDuck reference modulo IEEE-
// 754 sum-order — verified to ≤1e-12 in
// TestCrossSubstratePrecision_RubberDuck_*.
func Wasserstein1D(u, v []float64, p float64) (float64, error) {
	if !isValidP(p) {
		return 0, ErrInvalidP
	}

	sortedU := filterAndSortFinite(u)
	sortedV := filterAndSortFinite(v)

	if len(sortedU) == 0 || len(sortedV) == 0 {
		return math.NaN(), nil
	}

	// Equal-size case: direct order-statistic pairing.
	if len(sortedU) == len(sortedV) {
		sum := 0.0
		for i := range sortedU {
			diff := math.Abs(sortedU[i] - sortedV[i])
			if p == 1 {
				sum += diff
			} else {
				sum += math.Pow(diff, p)
			}
		}
		mean := sum / float64(len(sortedU))
		if p == 1 {
			return mean, nil
		}
		return math.Pow(mean, 1.0/p), nil
	}

	// Unequal-size case: linear interpolation on a (k + 0.5)/n grid
	// with n = max(|u|, |v|).  This recovers the equal-size closed
	// form exactly when the two lengths match; for |u| != |v| it is
	// the convention RubberDuck adopted (and the convention POT
	// follows under `wasserstein_1d` for unequal-size inputs).
	n := len(sortedU)
	if len(sortedV) > n {
		n = len(sortedV)
	}
	total := 0.0
	for k := 0; k < n; k++ {
		q := (float64(k) + 0.5) / float64(n)
		uVal := quantileFromSorted(sortedU, q)
		vVal := quantileFromSorted(sortedV, q)
		diff := math.Abs(uVal - vVal)
		if p == 1 {
			total += diff
		} else {
			total += math.Pow(diff, p)
		}
	}
	mean := total / float64(n)
	if p == 1 {
		return mean, nil
	}
	return math.Pow(mean, 1.0/p), nil
}

// Wasserstein1DDetailed returns Wasserstein-1 with full diagnostics
// including IQR-normalised distance.  Mirrors RubberDuck's
// `OptimalTransport.Wasserstein1DDetailed`:
// `flagships/rubberduck/RubberDuck.Core/Analysis/OptimalTransport.cs:71-97`.
//
// Returns `(WassersteinResult{}, false)` (with the second return
// false to signal "no result") for empty inputs after NaN/Inf
// filtering — equivalent to the C# `null` return convention.  The
// IQR is computed on the *combined* sorted data, matching the C#
// implementation; when IQR is zero (all combined values identical)
// NormalizedDistance is set to 0.0, matching the
// `iqr > 0 ? distance/iqr : 0.0` convention.
//
// Always uses p = 1 (matching RubberDuck's record-typed API which
// only emits W_1).
func Wasserstein1DDetailed(u, v []float64) (WassersteinResult, bool) {
	sortedU := filterAndSortFinite(u)
	sortedV := filterAndSortFinite(v)

	if len(sortedU) == 0 || len(sortedV) == 0 {
		return WassersteinResult{}, false
	}

	dist, err := Wasserstein1D(u, v, 1)
	if err != nil || math.IsNaN(dist) {
		return WassersteinResult{}, false
	}

	combined := make([]float64, 0, len(sortedU)+len(sortedV))
	combined = append(combined, sortedU...)
	combined = append(combined, sortedV...)
	// combined is the concatenation of two sorted slices; sort fully
	// rather than merging because sort.Float64s has IntroSort fast
	// paths and the constant factor is small for the consumer scale.
	{
		// Inline merge to avoid an extra sort.Sort call when both
		// halves are already individually sorted.  Result is sorted
		// in place into combined.
		merged := make([]float64, len(combined))
		i, j, k := 0, 0, 0
		for i < len(sortedU) && j < len(sortedV) {
			if sortedU[i] <= sortedV[j] {
				merged[k] = sortedU[i]
				i++
			} else {
				merged[k] = sortedV[j]
				j++
			}
			k++
		}
		for ; i < len(sortedU); i, k = i+1, k+1 {
			merged[k] = sortedU[i]
		}
		for ; j < len(sortedV); j, k = j+1, k+1 {
			merged[k] = sortedV[j]
		}
		combined = merged
	}

	q25 := quantileFromSorted(combined, 0.25)
	q75 := quantileFromSorted(combined, 0.75)
	iqr := q75 - q25

	normalised := 0.0
	if iqr > 0 {
		normalised = dist / iqr
	}

	return WassersteinResult{
		Distance:           dist,
		SampleSizeP:        len(sortedU),
		SampleSizeQ:        len(sortedV),
		NormalizedDistance: normalised,
	}, true
}

// isValidP reports whether p is a finite real number >= 1.  The
// closed-form 1D Wasserstein-p needs p >= 1 to be a metric (the
// integral of |F^{-1} - G^{-1}|^p is convex in p over the strict
// interior).
func isValidP(p float64) bool {
	if math.IsNaN(p) || math.IsInf(p, 0) {
		return false
	}
	return p >= 1
}
