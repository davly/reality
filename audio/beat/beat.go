package beat

import (
	"errors"
	"math"
)

// Beat is one tracked beat: the frame index it occurs at and the real
// time in seconds (frame * 1/frameRate).
type Beat struct {
	FrameIndex  int
	TimeSeconds float64
}

// Options bundle the beat-tracker parameters.
type Options struct {
	// TempoBpm is the tempo prior — the expected beat rate in BPM.
	// Typically obtained via foundation/reality/audio/tempo.Estimate.
	TempoBpm float64

	// TightnessAlpha controls how strongly the algorithm pushes beats
	// to land at multiples of the inter-beat period. 50-150 is typical.
	// Higher = more rigid; lower = more flexible to tempo drift.
	TightnessAlpha float64

	// SearchWindowMultiplier scales the look-back window for the DP.
	// 4.0 means the algorithm considers up to 4 inter-beat periods
	// back when computing each backlink. Larger handles bigger gaps
	// but is slower (O(N * window)).
	SearchWindowMultiplier float64
}

// DefaultOptions returns sensible defaults: alpha=100, search window
// 4× the inter-beat period.
func DefaultOptions(bpm float64) Options {
	return Options{
		TempoBpm:               bpm,
		TightnessAlpha:         100.0,
		SearchWindowMultiplier: 4.0,
	}
}

// Track returns the sequence of beat positions that maximise the DP
// objective: cumulative novelty along the beat sequence minus the
// log-squared deviation from the expected period.
//
// Returns ErrInvalidParams when frameRate <= 0 or any opts field is
// non-positive. Returns ErrInsufficientData when len(novelty) is too
// short to support at least 2 beats at the supplied tempo.
//
// Returns ([]Beat, nil) sorted ascending by FrameIndex on success.
func Track(novelty []float64, frameRate float64, opts Options) ([]Beat, error) {
	if frameRate <= 0 || opts.TempoBpm <= 0 ||
		opts.TightnessAlpha <= 0 || opts.SearchWindowMultiplier <= 0 {
		return nil, ErrInvalidParams
	}

	period := 60.0 * frameRate / opts.TempoBpm
	if period < 1 {
		// Faster than 1 frame/beat — not meaningful for our DP.
		return nil, ErrInvalidParams
	}

	N := len(novelty)
	if N < int(2*period) {
		return nil, ErrInsufficientData
	}

	window := int(math.Ceil(opts.SearchWindowMultiplier * period))
	if window < 1 {
		window = 1
	}

	// score[t] = best cumulative DP score ending at frame t as a beat.
	// backlink[t] = previous beat frame; -1 if t is the first beat in chain.
	score := make([]float64, N)
	backlink := make([]int, N)
	for t := range backlink {
		backlink[t] = -1
	}

	// Seed: the first window of frames are candidate first beats; their
	// score is just the novelty value.
	for t := 0; t < N; t++ {
		score[t] = novelty[t]
	}

	for t := 1; t < N; t++ {
		// Search lookback window for the best predecessor.
		startS := t - window
		if startS < 0 {
			startS = 0
		}
		// Don't allow s == t (zero-length transition); start at startS.
		// Also enforce s < t.
		bestScore := math.Inf(-1)
		bestS := -1
		for s := startS; s < t; s++ {
			delta := float64(t - s)
			// log((t-s)/period). At delta == period this is 0 (no penalty).
			logRatio := math.Log(delta / period)
			penalty := opts.TightnessAlpha * logRatio * logRatio
			candidate := score[s] - penalty
			if candidate > bestScore {
				bestScore = candidate
				bestS = s
			}
		}

		// Pick the better of (new chain at t) vs (extend best predecessor).
		seedScore := novelty[t]
		extScore := bestScore + novelty[t]
		if bestS >= 0 && extScore >= seedScore {
			score[t] = extScore
			backlink[t] = bestS
		} else {
			score[t] = seedScore
			backlink[t] = -1
		}
	}

	// Find the chain end: frame with max score in the last window.
	endStart := N - window
	if endStart < 0 {
		endStart = 0
	}
	endT := endStart
	endScore := math.Inf(-1)
	for t := endStart; t < N; t++ {
		if score[t] > endScore {
			endScore = score[t]
			endT = t
		}
	}

	// Backtrack to recover the beat sequence (descending order).
	revBeats := make([]Beat, 0, N/int(period)+1)
	cur := endT
	for cur >= 0 {
		revBeats = append(revBeats, Beat{
			FrameIndex:  cur,
			TimeSeconds: float64(cur) / frameRate,
		})
		cur = backlink[cur]
	}

	// Reverse into ascending order.
	beats := make([]Beat, len(revBeats))
	for i, b := range revBeats {
		beats[len(revBeats)-1-i] = b
	}
	return beats, nil
}

// Sentinel errors.
var (
	// ErrInvalidParams indicates a non-positive frame rate, tempo,
	// alpha, or search-window multiplier.
	ErrInvalidParams = errors.New("beat: invalid parameters")

	// ErrInsufficientData indicates the novelty function is too short
	// to support at least two beats at the supplied tempo.
	ErrInsufficientData = errors.New("beat: novelty too short for two beats")
)
