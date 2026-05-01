package onset

import (
	"math"
	"sort"
)

// PickPeaks returns the indices in strengths where a local maximum
// exceeds an absolute threshold and is separated from the previous
// pick by at least minSpacing frames.
//
// Algorithm (standard peak-picking):
//
//	A frame t is a peak if:
//	  strengths[t] > threshold AND
//	  strengths[t] >= strengths[t-1] AND  (i.e., not a strict-monotone case)
//	  strengths[t] >= strengths[t+1] AND
//	  t - lastPick >= minSpacing
//
// Parameters:
//   - strengths:  onset-strength function (one float per frame)
//   - threshold:  absolute threshold below which peaks are ignored
//   - minSpacing: minimum number of frames between consecutive peaks
//     (defends against duplicate detections from a single physical onset)
//
// Returns: slice of frame indices, sorted ascending. Empty if no
// frame exceeds the threshold.
//
// Valid range: minSpacing >= 0; threshold finite.
// Precision: peak comparisons are exact float64 — no tolerance.
//
// Allocation: returns newly-allocated []int.
//
// Reference: standard MIR practice; described e.g. in Dixon (2006)
// §4 alongside the spectral-flux strength function.
//
// Consumed by: SpectralFluxOnset, ComplexDomainOnset, SuperFlux,
// EnergyOnset all delegate peak picking to this function or its
// adaptive cousin.
func PickPeaks(strengths []float64, threshold float64, minSpacing int) []int {
	if minSpacing < 0 {
		panic("onset.PickPeaks: minSpacing must be >= 0")
	}
	if math.IsNaN(threshold) {
		panic("onset.PickPeaks: threshold must not be NaN")
	}
	n := len(strengths)
	if n == 0 {
		return nil
	}

	picks := []int{}
	lastPick := -minSpacing - 1
	for t := 0; t < n; t++ {
		if strengths[t] <= threshold {
			continue
		}
		// Local-maximum check (with edge tolerance).
		if t > 0 && strengths[t] < strengths[t-1] {
			continue
		}
		if t < n-1 && strengths[t] < strengths[t+1] {
			continue
		}
		if t-lastPick < minSpacing {
			continue
		}
		picks = append(picks, t)
		lastPick = t
	}

	sort.Ints(picks)
	return picks
}

// PickPeaksAdaptive performs adaptive median-filter-based peak picking
// on the supplied onset-strength function. Used internally by every
// detector in this package as the default thresholding strategy; also
// exposed for callers that compute their own strength function.
//
// Algorithm (Böck et al. 2012 §3):
//
//	threshold[t] = mean(strengths) + k · stdev(strengths)
//	A frame t is a peak if:
//	  strengths[t] > threshold[t] AND
//	  strengths[t] is the local maximum in [t-w, t+w]
//	  t - lastPick >= w  (post-pick suppression window)
//
// The mean+k·σ form is the simplest adaptive thresholder — good
// performance on short / homogeneous signals, robust to overall level
// scaling. Callers wanting a true running-median threshold (Böck's
// recommendation for long mixed-content recordings) should compute
// their own threshold and call PickPeaks.
//
// Parameters:
//   - strengths:           onset-strength function
//   - kStdev:              σ-multiplier above the mean (typical: 1.5)
//   - postSuppressFrames:  minimum spacing between consecutive picks
//
// Returns: slice of frame indices, sorted ascending.
//
// Valid range: kStdev >= 0; postSuppressFrames >= 0.
// Precision: 1e-12 in the mean / variance accumulation.
//
// Allocation: returns newly-allocated []int.
//
// Reference: Böck, S., Krebs, F. & Schedl, M. (2012). "Evaluating the
// online capabilities of onset detection methods." ISMIR 2012.
func PickPeaksAdaptive(strengths []float64, kStdev float64, postSuppressFrames int) []int {
	if kStdev < 0 {
		panic("onset.PickPeaksAdaptive: kStdev must be >= 0")
	}
	if postSuppressFrames < 0 {
		panic("onset.PickPeaksAdaptive: postSuppressFrames must be >= 0")
	}
	n := len(strengths)
	if n == 0 {
		return nil
	}

	// Mean + standard deviation (Welford's two-pass for numerical
	// stability is overkill at these sizes; one-pass is fine).
	mean := 0.0
	for i := 0; i < n; i++ {
		mean += strengths[i]
	}
	mean /= float64(n)
	variance := 0.0
	for i := 0; i < n; i++ {
		d := strengths[i] - mean
		variance += d * d
	}
	variance /= float64(n)
	stdev := math.Sqrt(variance)

	threshold := mean + kStdev*stdev

	return PickPeaks(strengths, threshold, postSuppressFrames)
}
