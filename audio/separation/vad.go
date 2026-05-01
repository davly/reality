package separation

import "math"

// IsVoiced returns true if the short-time energy of the frame exceeds
// threshold. This is the simplest possible voice/sound activity
// detector — useful as a gate before more expensive separation work
// (don't run FastICA on silent frames).
//
// Algorithm (Rabiner & Schafer 1975 §4):
//
//	E = (1/N) · Σ frame[n]²
//	voiced = (E >= threshold)
//
// where N = len(frame) (mean-square energy, sometimes called RMS²).
//
// Parameters:
//   - frame:     audio frame samples
//   - threshold: energy threshold (mean-square; not RMS). Typical
//     values: 1e-6 for studio-quiet input, 1e-4 for noisy ambient.
//
// Valid range: len(frame) >= 1; threshold >= 0.
// Precision: 1e-15 (single sum-of-squares + comparison).
// Panics on empty frame or negative threshold.
//
// Reference: Rabiner, L. R. & Schafer, R. W. (1975). "Digital
// Processing of Speech Signals." Prentice-Hall §4 (short-time energy
// and zero-crossing rate as VAD primitives).
//
// This is the "energy-only" variant — adequate for the audio cohort's
// typical use case (gate before FFT). For noisy environments where
// the noise floor approaches the signal level, callers should
// compose with reality/audio.DegradationTracker on the energy time
// series.
//
// Consumed by: pigeonhole / howler / dipstick (silent-frame gate).
func IsVoiced(frame []float64, threshold float64) bool {
	if len(frame) == 0 {
		panic("separation.IsVoiced: frame must be non-empty")
	}
	if threshold < 0 {
		panic("separation.IsVoiced: threshold must be >= 0")
	}
	s := 0.0
	for i := 0; i < len(frame); i++ {
		s += frame[i] * frame[i]
	}
	e := s / float64(len(frame))
	return e >= threshold
}

// FrameEnergy returns the short-time mean-square energy of the frame.
//
// Formula: E = (1/N) · Σ frame[n]²
//
// Useful as the input to a custom thresholding strategy (e.g.
// adaptive thresholds, DegradationTracker tracking). Panics on
// empty frame.
//
// Zero allocation.
func FrameEnergy(frame []float64) float64 {
	if len(frame) == 0 {
		panic("separation.FrameEnergy: frame must be non-empty")
	}
	s := 0.0
	for i := 0; i < len(frame); i++ {
		s += frame[i] * frame[i]
	}
	return s / float64(len(frame))
}

// ZeroCrossingRate returns the rate at which the signal crosses zero.
// A high zero-crossing rate (~0.3-0.5) indicates unvoiced sound
// (fricatives, noise); a low rate (~0.05-0.1) indicates voiced sound
// (vowels, harmonic content). Combined with FrameEnergy, this is the
// classic two-feature VAD (Rabiner & Sambur 1975).
//
// Formula:
//
//	ZCR = (1 / (2(N-1))) · Σ_{n=1..N-1} |sign(frame[n]) - sign(frame[n-1])|
//
// where sign(x) = 1 if x >= 0, else -1. Result is in [0, 1].
//
// Parameters:
//   - frame: audio frame samples
//
// Valid range: len(frame) >= 2.
// Precision: 1e-15.
// Panics on len(frame) < 2.
//
// Reference: Rabiner, L. R. & Sambur, M. R. (1975). "An algorithm for
// determining the endpoints of isolated utterances." Bell Syst. Tech.
// J. 54(2), 297-315.
//
// Zero allocation.
func ZeroCrossingRate(frame []float64) float64 {
	n := len(frame)
	if n < 2 {
		panic("separation.ZeroCrossingRate: frame must have length >= 2")
	}
	prev := frame[0]
	count := 0
	for i := 1; i < n; i++ {
		curr := frame[i]
		if (prev >= 0) != (curr >= 0) {
			count++
		}
		prev = curr
	}
	return float64(count) / float64(2*(n-1)) * 2.0 // 2x because the formula |sign-sign|/2 ranges to 2 not 1
}

// IsVoicedAdaptive returns true if the frame energy exceeds an
// adaptive threshold derived from a noise-floor estimate.
//
// Formula:
//
//	threshold = noiseEnergy · marginDb_to_ratio
//	         = noiseEnergy · 10^(marginDb / 10)
//
// Calls IsVoiced internally with the computed threshold.
//
// Parameters:
//   - frame:        audio frame
//   - noiseEnergy:  estimated noise-floor mean-square energy (e.g.
//     time-average of FrameEnergy across known-silent frames)
//   - marginDb:     gate margin above noise floor in dB. 6 dB is
//     a common choice (signal must be 4× noise power). Higher
//     margins reduce false positives at cost of recall.
//
// Valid range: noiseEnergy >= 0; marginDb finite.
// Panics if frame empty.
//
// Reference: standard adaptive-VAD primer (Rabiner & Schafer 1978
// §4.4 — energy thresholding with noise-tracking).
func IsVoicedAdaptive(frame []float64, noiseEnergy, marginDb float64) bool {
	threshold := noiseEnergy * math.Pow(10, marginDb/10)
	return IsVoiced(frame, threshold)
}
