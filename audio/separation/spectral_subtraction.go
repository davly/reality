package separation

import (
	"math"
	"math/cmplx"
)

// SubtractSpectrum performs single-channel spectral subtraction (Boll 1979).
// Given the complex FFT spectrum of a noisy observation and an estimate of
// the noise spectrum (typically computed as the time-average of FFTs over
// silent frames), this returns a denoised complex spectrum.
//
// Algorithm (Boll 1979 §III, magnitude-subtraction with phase preserved):
//
//	|Ŝ[k]|² = max(|X[k]|² - α·|N[k]|², β·|X[k]|²)
//	∠Ŝ[k]   = ∠X[k]
//	Ŝ[k]    = sqrt(|Ŝ[k]|²) · exp(j·∠X[k])
//
// where α is the over-subtraction factor (here 1.0) and β is the spectral
// floor coefficient (here 0.01 — i.e. retain at least 1% of the noisy
// magnitude per bin to avoid musical-noise artefacts). The phase of the
// noisy observation is preserved — psychoacoustic studies (Wang & Lim 1982)
// established that magnitude is the dominant perceptual carrier.
//
// Parameters:
//   - in:    noisy complex FFT spectrum (length N/2+1 typical, or full N)
//   - noise: noise complex FFT spectrum (same length as in)
//
// Returns: a newly-allocated denoised complex spectrum of len(in). The
// caller owns the returned slice.
//
// Valid range: len(in) == len(noise), len(in) > 0.
// Precision: the magnitude floor (β = 0.01) is the dominant numerical
// floor; absolute error beneath that floor is meaningless.
// Panics if lengths differ.
//
// Reference: Boll, S. F. (1979). "Suppression of acoustic noise in speech
// using spectral subtraction." IEEE Trans. ASSP 27(2), 113-120; Berouti,
// Schwartz & Makhoul (1979) "Enhancement of speech corrupted by acoustic
// noise" ICASSP — over-subtraction and floor convention.
//
// This is the allocating form (one slice per call). Callers wanting
// zero-alloc should use SubtractSpectrumInto.
//
// Consumed by: pigeonhole (ambient-noise reduction before MFCC).
func SubtractSpectrum(in, noise []complex128) []complex128 {
	if len(in) != len(noise) {
		panic("separation.SubtractSpectrum: in and noise must have equal length")
	}
	out := make([]complex128, len(in))
	SubtractSpectrumInto(in, noise, out)
	return out
}

// SubtractSpectrumInto is the zero-alloc form of SubtractSpectrum.
// out must have length >= len(in). Panics on size violations.
func SubtractSpectrumInto(in, noise, out []complex128) {
	n := len(in)
	if len(noise) != n {
		panic("separation.SubtractSpectrumInto: in and noise must have equal length")
	}
	if len(out) < n {
		panic("separation.SubtractSpectrumInto: out must have length >= len(in)")
	}
	const alpha = 1.0  // over-subtraction factor
	const beta = 0.01  // spectral floor (1% of noisy magnitude)
	for k := 0; k < n; k++ {
		xMag := cmplx.Abs(in[k])
		nMag := cmplx.Abs(noise[k])
		xPower := xMag * xMag
		nPower := nMag * nMag
		floor := beta * xPower
		num := xPower - alpha*nPower
		if num < floor {
			num = floor
		}
		newMag := math.Sqrt(num)
		// Preserve the noisy phase.
		if xMag == 0 {
			out[k] = 0
			continue
		}
		// out = newMag * (in/|in|)
		out[k] = complex(newMag, 0) * (in[k] / complex(xMag, 0))
	}
}

// EstimateNoiseSpectrum computes a per-bin time-average of the magnitude
// of the silent frames in spectra. Useful as the noise-estimate input
// to SubtractSpectrum when the caller has identified ambient frames
// (e.g. via Energy-VAD).
//
// Returns a complex spectrum where the phase is zeroed (only magnitude
// matters for the subtraction). out must have length >= nBins.
//
// Algorithm: noise[k] = (1/M) * Σ_{m=1..M} |spectra[m][k]|, with phase=0.
// If silenceFrames is empty, out is filled with zeros.
//
// Zero allocation in the loop body. Panics on length mismatch.
//
// Consumed by: pigeonhole (build noise model from VAD-flagged silence).
func EstimateNoiseSpectrum(silenceFrames [][]complex128, nBins int, out []complex128) {
	if len(out) < nBins {
		panic("separation.EstimateNoiseSpectrum: out must have length >= nBins")
	}
	for k := 0; k < nBins; k++ {
		out[k] = 0
	}
	M := len(silenceFrames)
	if M == 0 {
		return
	}
	for m := 0; m < M; m++ {
		if len(silenceFrames[m]) < nBins {
			panic("separation.EstimateNoiseSpectrum: each frame must have length >= nBins")
		}
		for k := 0; k < nBins; k++ {
			out[k] += complex(cmplx.Abs(silenceFrames[m][k]), 0)
		}
	}
	scale := complex(1.0/float64(M), 0)
	for k := 0; k < nBins; k++ {
		out[k] *= scale
	}
}
