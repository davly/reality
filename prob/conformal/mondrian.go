package conformal

import (
	"errors"
	"math"
	"sort"
)

// ErrLengthMismatch is returned when scores and strata have different
// lengths in MondrianQuantile.
var ErrLengthMismatch = errors.New("conformal: scores and strata must have equal length")

// MondrianQuantile returns per-stratum conformal thresholds.  Each
// calibration sample i has a non-negative score scores[i] and a stratum
// label strata[i] (an int category).  For each unique stratum the function
// computes the SplitQuantile of the scores belonging to it.
//
// Returned map has one entry per unique stratum encountered.  Strata with
// too few calibration samples to support the requested alpha get +Inf
// (per SplitQuantile's behaviour for n < ceil(1/alpha) - 1).
//
// Mondrian conformal trades sample efficiency for stratum-conditional
// coverage:
//
//	P( Y in C(X) | stratum(X) = s ) >= 1 - alpha   for each stratum s
//
// (provided the (X, Y, s) tuples within each stratum are exchangeable with
// the test point's tuple).  See Vovk et al 2005 §4.5 and Boström-Linusson-
// Löfström-Johansson 2017.
//
// Reference: Vovk V., Lindsay D., Nouretdinov I. & Gammerman A. (2003).
// Mondrian Confidence Machine.  Tech. Report; Vovk-Gammerman-Shafer 2005
// chap. 4.
func MondrianQuantile(scores []float64, strata []int, alpha float64) (map[int]float64, error) {
	if alpha <= 0 || alpha >= 1 || math.IsNaN(alpha) {
		return nil, ErrInvalidAlpha
	}
	if len(scores) != len(strata) {
		return nil, ErrLengthMismatch
	}
	if len(scores) == 0 {
		return nil, ErrEmptyCalibration
	}
	for _, s := range scores {
		if math.IsNaN(s) || s < 0 {
			return nil, ErrInvalidScore
		}
	}

	// Group scores by stratum.
	byStratum := make(map[int][]float64)
	for i, s := range scores {
		byStratum[strata[i]] = append(byStratum[strata[i]], s)
	}

	out := make(map[int]float64, len(byStratum))
	for k, ss := range byStratum {
		sort.Float64s(ss)
		n := len(ss)
		rank := int(math.Ceil((float64(n) + 1.0) * (1.0 - alpha)))
		if rank > n {
			out[k] = math.Inf(1)
			continue
		}
		if rank < 1 {
			rank = 1
		}
		out[k] = ss[rank-1]
	}
	return out, nil
}

// MondrianInterval returns the prediction interval for a test point of a
// given stratum, using the per-stratum conformal threshold.  Returns
// (-Inf, +Inf) if the stratum's threshold is +Inf or if the stratum is
// not present in q (treated as "no calibration evidence — admit anything").
func MondrianInterval(yhat float64, stratum int, q map[int]float64) (lo, hi float64) {
	threshold, ok := q[stratum]
	if !ok || math.IsInf(threshold, 1) {
		return math.Inf(-1), math.Inf(1)
	}
	return yhat - threshold, yhat + threshold
}
