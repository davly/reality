package prob

import (
	"errors"
	"math"
	"sort"
)

// Robust, heavy-tail location estimators with sub-Gaussian concentration.
//
// The sample mean has only polynomial-tail concentration when the data are
// heavy-tailed: a single gross outlier moves it arbitrarily. These estimators
// achieve sub-Gaussian deviation bounds under only a finite-variance assumption.
//
//   - MedianOfMeans (Nemirovski-Yudin; Devroye, Lerasle, Lugosi & Oliveira 2016,
//     Ann. Stat., arXiv:1509.05845): split into k blocks, take the median of the
//     block means. With k ~ log(1/delta) blocks, |MoM - mu| <= C*sigma*sqrt(k/n)
//     with probability 1-delta, and it tolerates up to floor((k-1)/2) arbitrarily
//     corrupted blocks.
//   - CatoniMean (Catoni 2012): an M-estimator solving sum psi((x_i-theta)/s)=0
//     with a gentle influence function psi that caps the leverage of any point.
//
// Unlike prob.Median / prob.TrimmedMean (which clamp to the probability range
// [MinProb, MaxProb]), these return unclamped real-valued location estimates and
// are intended for general data (latencies, costs, residuals).

// MedianOfMeans returns the median-of-means estimate of the population mean:
// the values are partitioned into `blocks` contiguous groups (sizes differ by at
// most one), and the result is the median of the per-block arithmetic means.
// blocks must be in [1, len(values)]; blocks=1 degenerates to the sample mean.
// Returns an error for empty input or an out-of-range block count.
func MedianOfMeans(values []float64, blocks int) (float64, error) {
	n := len(values)
	if n == 0 {
		return 0, errors.New("prob: MedianOfMeans requires at least one value")
	}
	if blocks < 1 || blocks > n {
		return 0, errors.New("prob: MedianOfMeans blocks must be in [1, len(values)]")
	}
	means := make([]float64, blocks)
	for b := 0; b < blocks; b++ {
		lo := b * n / blocks
		hi := (b + 1) * n / blocks
		sum := 0.0
		for _, v := range values[lo:hi] {
			sum += v
		}
		means[b] = sum / float64(hi-lo)
	}
	return medianUnclamped(means), nil
}

// MedianOfMeansForConfidence chooses the block count for a target failure
// probability delta in (0,1) — the standard k = ceil(8 ln(1/delta)) blocks,
// clamped to [1, len(values)] — and returns the estimate together with the block
// count actually used.
func MedianOfMeansForConfidence(values []float64, delta float64) (estimate float64, blocks int, err error) {
	if !(delta > 0 && delta < 1) || math.IsNaN(delta) {
		return 0, 0, errors.New("prob: MedianOfMeansForConfidence delta must be in (0,1)")
	}
	n := len(values)
	if n == 0 {
		return 0, 0, errors.New("prob: MedianOfMeansForConfidence requires at least one value")
	}
	k := int(math.Ceil(8.0 * math.Log(1.0/delta)))
	if k < 1 {
		k = 1
	}
	if k > n {
		k = n
	}
	est, err := MedianOfMeans(values, k)
	return est, k, err
}

// CatoniMean returns Catoni's robust M-estimate of the mean: the root theta of
// sum_i psi((x_i - theta)/scale) = 0, where psi is Catoni's bounded-leverage
// influence function. scale > 0 sets the robustness band; a reasonable default
// is an estimate of sigma (e.g. an interquartile-based scale). Returns an error
// for empty input or non-positive/non-finite scale. The estimating function is
// strictly decreasing in theta, so the root is found by bisection over the data
// range and is unique.
func CatoniMean(values []float64, scale float64) (float64, error) {
	n := len(values)
	if n == 0 {
		return 0, errors.New("prob: CatoniMean requires at least one value")
	}
	if !(scale > 0) || math.IsInf(scale, 0) || math.IsNaN(scale) {
		return 0, errors.New("prob: CatoniMean scale must be positive and finite")
	}
	lo, hi := values[0], values[0]
	for _, v := range values {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if lo == hi {
		return lo, nil // all equal
	}
	f := func(theta float64) float64 {
		s := 0.0
		for _, v := range values {
			s += catoniPsi((v - theta) / scale)
		}
		return s
	}
	// f(lo) > 0, f(hi) < 0 (psi is increasing in its argument, theta enters
	// negatively). Bisection to a tight tolerance on the data scale.
	tol := (hi - lo) * 1e-12
	for i := 0; i < 200 && hi-lo > tol; i++ {
		mid := lo + (hi-lo)/2
		if f(mid) > 0 {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo + (hi-lo)/2, nil
}

// catoniPsi is Catoni's influence function: log(1 + x + x^2/2) for x >= 0 and
// -log(1 - x + x^2/2) for x < 0. It is bounded-leverage (grows like log|x|) yet
// matches the identity to second order near 0.
func catoniPsi(x float64) float64 {
	if x >= 0 {
		return math.Log(1 + x + x*x/2)
	}
	return -math.Log(1 - x + x*x/2)
}

// medianUnclamped returns the median of a slice without the [0,1] clamp that
// prob.Median applies. The input is copied, not mutated.
func medianUnclamped(values []float64) float64 {
	n := len(values)
	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2.0
	}
	return sorted[n/2]
}
