package vibration

import (
	"math"

	"github.com/davly/reality/signal"
)

// FundamentalHz estimates the fundamental frequency of a windowed signal
// frame by finding the peak of the power spectrum within [fMin, fMax].
// Returns 0 if no peak is detected (frame is silent or noise-only, or
// fMin/fMax bracket no usable bins).
//
// Formula:
//
//	X[k]   = signal.FFT(frameReal, frameImag) (in-place)
//	P[k]   = X.real[k]^2 + X.imag[k]^2
//	binWidth = sampleRate / N where N = len(frameReal)
//	binMin = ceil(fMin / binWidth)
//	binMax = floor(fMax / binWidth)  (clamped to N/2)
//	k*     = argmax_{binMin <= k <= binMax} P[k]
//	f_0    = k* × binWidth
//
// Parameters:
//   - frameReal, frameImag: in-place FFT scratch buffers; frameReal
//     should already contain the windowed audio frame on entry. After
//     return, both slices contain FFT output (caller has them).
//   - sampleRate: Hz, sampleRate > 0
//   - fMin, fMax: search range (Hz), 0 <= fMin < fMax < sampleRate/2
//
// Valid range: N (= len(frameReal)) must be a power of 2 (signal.FFT
// requirement). frameReal and frameImag must have equal length.
//
// Precision: limited by FFT bin resolution (= sampleRate / N Hz). For
// sub-bin precision, callers should apply parabolic interpolation on the
// peak (not implemented here — the typical Dipstick-style use case
// tracks relative shifts via z-score on the bin-quantised value, where
// sub-bin precision adds little signal). Numerical error is dominated
// by the underlying FFT (~1e-9 for 1024-point per signal.FFT contract)
// plus a single argmax over float64 powers.
//
// Reference: standard DSP. Smith 1997, "The Scientist and Engineer's
// Guide to Digital Signal Processing" §11 (peak-bin frequency estimation);
// Harris 1978, "On the Use of Windows for Harmonic Analysis with the
// Discrete Fourier Transform" Proc. IEEE 66(1) (windowing rationale —
// caller responsibility).
//
// Zero allocation. Panics if frame slices have unequal length.
//
// Consumed by: flagships/dipstick (machine fundamental drift,
// per-component DegradationTracker input).
func FundamentalHz(frameReal, frameImag []float64, sampleRate, fMin, fMax float64) float64 {
	n := len(frameReal)
	if len(frameImag) != n {
		panic("vibration.FundamentalHz: frame slices must have equal length")
	}
	signal.FFT(frameReal, frameImag)

	binWidth := sampleRate / float64(n)
	binMin := int(math.Ceil(fMin / binWidth))
	binMax := int(math.Floor(fMax / binWidth))
	half := n/2 + 1
	if binMax > half-1 {
		binMax = half - 1
	}
	if binMin < 1 {
		binMin = 1
	}

	bestPower := 0.0
	bestBin := -1
	for k := binMin; k <= binMax; k++ {
		p := frameReal[k]*frameReal[k] + frameImag[k]*frameImag[k]
		if p > bestPower {
			bestPower = p
			bestBin = k
		}
	}
	if bestBin < 0 {
		return 0
	}
	return float64(bestBin) * binWidth
}
