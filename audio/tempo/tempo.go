package tempo

import (
	"errors"
	"math"
)

// Options bundle the tempo-search parameters. Defaults via DefaultOptions.
type Options struct {
	// MinBpm is the lower bound of the BPM search. 30 BPM (lag = 2 sec
	// at 1 Hz frame rate) is a permissive default that captures slow
	// ballads and ambient music.
	MinBpm float64

	// MaxBpm is the upper bound of the BPM search. 300 BPM (lag ~0.2
	// sec) covers most percussion and electronic music; raise for
	// experimental drum-and-bass / breakcore content.
	MaxBpm float64
}

// DefaultOptions returns sensible defaults: 30-300 BPM range.
func DefaultOptions() Options {
	return Options{MinBpm: 30, MaxBpm: 300}
}

// Estimate returns the dominant tempo in beats per minute given an
// onset-strength novelty function sampled at frameRate Hz.
//
// Algorithm:
//  1. Compute the unbiased autocorrelation of the novelty function.
//  2. Search for the lag with maximum ACF within the BPM bounds.
//  3. Return BPM = 60 * frameRate / lag.
//
// Returns ErrInvalidParams when frameRate <= 0, opts bounds are
// non-positive, or the BPM range is degenerate.
// Returns ErrInsufficientData when len(novelty) < 4 (need at least
// a few samples to compute meaningful ACF).
// Returns ErrNoTempo when no peak is found in the BPM range (e.g.
// completely flat novelty — silence).
func Estimate(novelty []float64, frameRate float64, opts Options) (float64, error) {
	if frameRate <= 0 {
		return 0, ErrInvalidParams
	}
	if opts.MinBpm <= 0 || opts.MaxBpm <= 0 || opts.MinBpm >= opts.MaxBpm {
		return 0, ErrInvalidParams
	}
	if len(novelty) < 4 {
		return 0, ErrInsufficientData
	}

	// Convert BPM bounds to lag bounds (in frames).
	minLag := int(math.Floor(60.0 * frameRate / opts.MaxBpm))
	maxLag := int(math.Ceil(60.0 * frameRate / opts.MinBpm))
	if minLag < 1 {
		minLag = 1
	}
	if maxLag >= len(novelty) {
		maxLag = len(novelty) - 1
	}
	if minLag > maxLag {
		return 0, ErrNoTempo
	}

	// Compute ACF up to maxLag.
	acf := make([]float64, maxLag+1)
	if err := Autocorrelation(novelty, maxLag, acf); err != nil {
		return 0, err
	}

	// Find the lag with peak ACF within [minLag, maxLag].
	bestLag := minLag
	bestVal := math.Inf(-1)
	for lag := minLag; lag <= maxLag; lag++ {
		if acf[lag] > bestVal {
			bestVal = acf[lag]
			bestLag = lag
		}
	}

	if math.IsInf(bestVal, -1) || bestLag <= 0 {
		return 0, ErrNoTempo
	}

	return 60.0 * frameRate / float64(bestLag), nil
}

// Autocorrelation computes the (un-normalised) autocorrelation function
// of x up to maxLag, writing into out[0..maxLag].
//
//	ACF[k] = sum over n of x[n] * x[n+k]   for n in [0, len(x)-k)
//
// out must have length >= maxLag+1. maxLag must satisfy 0 <= maxLag < len(x).
//
// Returns ErrOutputSize if out is too small or maxLag is invalid.
func Autocorrelation(x []float64, maxLag int, out []float64) error {
	if maxLag < 0 || maxLag >= len(x) {
		return ErrOutputSize
	}
	if len(out) < maxLag+1 {
		return ErrOutputSize
	}

	for k := 0; k <= maxLag; k++ {
		sum := 0.0
		for n := 0; n+k < len(x); n++ {
			sum += x[n] * x[n+k]
		}
		out[k] = sum
	}
	return nil
}

// LagToBpm converts an autocorrelation lag (in frames) to BPM at the
// given frame rate. Convenience for callers that want to interpret
// non-best peaks (e.g., harmonic / sub-harmonic candidates).
//
// Returns 0 when lag <= 0.
func LagToBpm(lag int, frameRate float64) float64 {
	if lag <= 0 || frameRate <= 0 {
		return 0
	}
	return 60.0 * frameRate / float64(lag)
}

// BpmToLag converts a BPM value to an autocorrelation lag (in frames)
// at the given frame rate. Rounds to nearest integer.
//
// Returns 0 when bpm <= 0.
func BpmToLag(bpm, frameRate float64) int {
	if bpm <= 0 || frameRate <= 0 {
		return 0
	}
	return int(math.Round(60.0 * frameRate / bpm))
}

// Sentinel errors.
var (
	// ErrInvalidParams indicates a non-positive frame rate, non-positive
	// BPM bound, or MinBpm >= MaxBpm.
	ErrInvalidParams = errors.New("tempo: invalid parameters")

	// ErrInsufficientData indicates too few novelty samples to estimate
	// (need at least 4).
	ErrInsufficientData = errors.New("tempo: insufficient novelty samples")

	// ErrNoTempo indicates no peak was found in the BPM search range
	// (typically flat / silent input).
	ErrNoTempo = errors.New("tempo: no tempo detected in search range")

	// ErrOutputSize indicates an output slice is too short or
	// autocorrelation lag bounds are invalid.
	ErrOutputSize = errors.New("tempo: output slice has wrong length or bad lag")
)
