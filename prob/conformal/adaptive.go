package conformal

import (
	"errors"
	"math"
	"sort"
)

// ErrInvalidHalfLife is returned when the recency half-life is not
// strictly positive.
var ErrInvalidHalfLife = errors.New("conformal: recency half-life must be > 0")

// AdaptiveQuantile returns a recency-weighted conformal threshold for a
// time-ordered sequence of nonconformity scores.  Where SplitQuantile
// treats every calibration sample equally (the classical Lei et al 2018
// formulation, valid under exchangeability), AdaptiveQuantile applies
// an exponential-decay weight w_i = 0.5^((n-1-i) / halfLife) so that the
// most recent sample has weight 1, the sample halfLife steps back has
// weight 0.5, and so on.  The quantile is then the smallest score q at
// which the cumulative weight reaches the (1 - alpha) target.
//
// Input scores must be in *time order* — scores[0] is the oldest, scores
// [len-1] is the most recent.  This is the order produced by appending
// new residuals to a circular buffer, so most callers can pass their
// raw history directly.
//
// # When to use
//
// Conformal's marginal coverage guarantee assumes exchangeability between
// the calibration set and the test point.  Under distribution shift —
// the FW ConformalCostCalibrator's 7-day window crossing an Anthropic
// price release, an upstream model swap, a regime change in the metric
// being forecast — exchangeability is violated and classical conformal
// over-covers at the back of the window and under-covers at the front.
// AdaptiveQuantile is the cheap, deterministic, no-extra-dependencies
// path to handling this: by down-weighting stale samples, the realised
// coverage tracks the *current* residual distribution, not the *long-run*
// one.
//
// This is *not* a replacement for the Gibbs-Candes 2021 adaptive-
// conformal procedure (which adjusts alpha online based on observed
// coverage); it is an additive complement that targets the same problem
// from the data side rather than the inference side.
//
// # Algorithm
//
//   - Compute weights w_i = 0.5^((n - 1 - i) / halfLife).
//   - Sort (score_i, w_i) ascending by score.
//   - Walk the sorted sequence accumulating weight; return the score at
//     which the cumulative weight first reaches W * (1 - alpha), where
//     W = sum(w_i).  This is the canonical weighted quantile.
//
// Returns +Inf when all weight lies on a single largest score and the
// (1 - alpha) target falls in the open tail — a recency-aware version of
// SplitQuantile's "n too small" return.
func AdaptiveQuantile(scores []float64, alpha float64, halfLife int) (float64, error) {
	if alpha <= 0 || alpha >= 1 || math.IsNaN(alpha) {
		return 0, ErrInvalidAlpha
	}
	if halfLife <= 0 {
		return 0, ErrInvalidHalfLife
	}
	if len(scores) == 0 {
		return 0, ErrEmptyCalibration
	}
	for _, s := range scores {
		if math.IsNaN(s) || s < 0 {
			return 0, ErrInvalidScore
		}
	}

	n := len(scores)

	// Build (score, weight) pairs.  The most recent observation
	// (scores[n-1]) has weight 1; weights decay by 0.5 every halfLife
	// steps backward.
	type pair struct {
		score, weight float64
	}
	pairs := make([]pair, n)
	totalWeight := 0.0
	hl := float64(halfLife)
	for i, s := range scores {
		// Steps back from the front of the window.
		stepsBack := float64(n - 1 - i)
		w := math.Pow(0.5, stepsBack/hl)
		pairs[i] = pair{score: s, weight: w}
		totalWeight += w
	}

	// Sort ascending by score.  Stable sort preserves time order for
	// ties, which keeps determinism predictable across recency-tied
	// buckets.
	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i].score < pairs[j].score
	})

	// Walk sorted scores accumulating weight; return the score at which
	// the cumulative weight first crosses or equals W * (1 - alpha).
	// We use the (n+1) finite-sample correction in *weighted* form: the
	// effective "next-point" weight at the test time is 1 (the most
	// recent observation's weight), so the corrected target is
	//
	//	W_target = (totalWeight + 1) * (1 - alpha)
	//
	// and we return +Inf if the cumulative weight never reaches it
	// (the recency-aware analogue of the rank-overflow case).
	target := (totalWeight + 1.0) * (1.0 - alpha)
	if target > totalWeight {
		// Not enough cumulative weight in the empirical sample to honour
		// the requested coverage — same semantics as SplitQuantile when
		// rank > n.
		return math.Inf(1), nil
	}
	cumulative := 0.0
	for _, p := range pairs {
		cumulative += p.weight
		if cumulative >= target {
			return p.score, nil
		}
	}
	// Defensive: shouldn't reach here given the target check above, but
	// fall through to the largest score under floating-point edges.
	return pairs[n-1].score, nil
}

// AdaptiveInterval returns a recency-weighted symmetric prediction
// interval [yhat - q, yhat + q].  Inputs are absolute residuals in time
// order (oldest first); the recency half-life is the number of samples
// after which a residual's weight halves.
//
// Use when the underlying residual distribution drifts over the
// calibration window — pricing changes, model swaps, regime shifts —
// and you want the band to track the *current* error distribution rather
// than the long-run one.  Under stationarity, AdaptiveInterval converges
// to SplitInterval as halfLife -> infinity.
func AdaptiveInterval(yhat float64, calibrationResiduals []float64, alpha float64, halfLife int) (lo, hi float64, err error) {
	q, err := AdaptiveQuantile(calibrationResiduals, alpha, halfLife)
	if err != nil {
		return 0, 0, err
	}
	return yhat - q, yhat + q, nil
}

// EffectiveSampleSize returns the Kish effective sample size for a
// recency-weighted calibration window:
//
//	n_eff = (sum w_i)^2 / sum(w_i^2)
//
// This is the canonical diagnostic for "how many samples is your
// recency-weighted calibration *effectively* using" — at halfLife = +inf
// (no decay) n_eff = n; at halfLife = 1 (only the most recent sample
// has meaningful weight) n_eff approaches 1.  Useful for deciding
// whether the (1 - alpha) coverage is supportable: if n_eff <
// ceil(1/alpha) the AdaptiveQuantile returns +Inf and the caller should
// either raise alpha, increase halfLife, or fall back to SplitQuantile.
//
// Returns 1 if scores is empty or halfLife is non-positive (defensive
// floor; production callers should validate first).
func EffectiveSampleSize(n int, halfLife int) float64 {
	if n <= 0 || halfLife <= 0 {
		return 1
	}
	hl := float64(halfLife)
	var sumW, sumW2 float64
	for i := 0; i < n; i++ {
		stepsBack := float64(n - 1 - i)
		w := math.Pow(0.5, stepsBack/hl)
		sumW += w
		sumW2 += w * w
	}
	if sumW2 == 0 {
		return 1
	}
	return (sumW * sumW) / sumW2
}
