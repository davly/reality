package evt

import (
	"math"
	"sort"
)

// HillTailIndex estimates the extreme-value shape parameter xi = 1/alpha of a
// heavy right tail from the k largest order statistics using the Hill (1975)
// estimator
//
//	xi_hat(k) = (1/k) sum_{i=1}^{k} [ ln X_(i) - ln X_(k+1) ]
//
// where X_(1) >= X_(2) >= ... are the data sorted in descending order and
// X_(k+1) plays the role of the (data-driven) threshold.  The reciprocal
// alpha = 1/xi is the tail index (number of finite moments boundary).
//
// The estimator is defined only for a strictly positive heavy tail; it
// requires X_(k+1) > 0 and 1 <= k <= n-1.  ok == false otherwise (including a
// non-positive threshold order statistic, for which the logarithm is
// undefined).
//
// Reference: Hill, B.M. (1975) "A Simple General Approach to Inference About
// the Tail of a Distribution", Annals of Statistics 3(5):1163-1174.
// Precision: exact given the order statistics (a single mean of logs); the
// statistical bias/variance is governed by the choice of k, not by the
// arithmetic.
func HillTailIndex(data []float64, k int) (xi float64, ok bool) {
	n := len(data)
	if k < 1 || k >= n {
		return 0, false
	}
	s := append([]float64(nil), data...)
	sort.Sort(sort.Reverse(sort.Float64Slice(s)))
	thr := s[k] // X_(k+1), 0-indexed
	if thr <= 0 {
		return 0, false
	}
	lnThr := math.Log(thr)
	var sum float64
	for i := 0; i < k; i++ {
		if s[i] <= 0 {
			return 0, false
		}
		sum += math.Log(s[i]) - lnThr
	}
	xi = sum / float64(k)
	return xi, true
}

// HillAlpha returns the tail index alpha = 1/xi from HillTailIndex, i.e. the
// order beyond which moments diverge (a Frechet tail has E[X^p] = infinity for
// p >= alpha).  ok == false if the Hill shape is non-positive (not a heavy
// tail) or the estimate is undefined.
//
// Reference: Hill (1975); Embrechts, Kluppelberg & Mikosch (1997) §6.4.
func HillAlpha(data []float64, k int) (alpha float64, ok bool) {
	xi, ok := HillTailIndex(data, k)
	if !ok || !(xi > 0) {
		return 0, false
	}
	return 1 / xi, true
}
