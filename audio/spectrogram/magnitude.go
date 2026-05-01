package spectrogram

import (
	"math"
	"math/cmplx"
)

// Magnitude computes the per-cell |X[t][f]| reduction of a complex STFT.
// Returns a newly-allocated T × frameSize matrix of magnitudes (real-
// valued, non-negative).
//
// Formula: out[t][f] = |stft[t][f]| = sqrt(re² + im²)
//
// Allocation: returns newly-allocated [][]float64 of the same shape
// as stft. Callers wanting to reuse buffers should compose the
// expression manually.
//
// Panics if stft is empty.
func Magnitude(stft [][]complex128) [][]float64 {
	T := len(stft)
	if T < 1 {
		panic("spectrogram.Magnitude: stft must have at least 1 frame")
	}
	F := len(stft[0])
	out := make([][]float64, T)
	for t := 0; t < T; t++ {
		if len(stft[t]) != F {
			panic("spectrogram.Magnitude: all rows must have equal length")
		}
		out[t] = make([]float64, F)
		for f := 0; f < F; f++ {
			out[t][f] = cmplx.Abs(stft[t][f])
		}
	}
	return out
}

// LogMagnitude computes the per-cell decibel-scale magnitude reduction
// of a complex STFT.
//
// Formula: out[t][f] = 20 · log10(|stft[t][f]| + ε)
//
// where ε = 1e-12 prevents log(0). The returned values are dB
// relative to the maximum magnitude representable in float64 — most
// callers will subtract a per-frame max to get "dBFS" values in the
// (-∞, 0] range.
//
// Allocation: returns newly-allocated [][]float64.
//
// Reference: standard convention; matches matplotlib's
// specgram(scale='dB') output.
//
// Consumed by: ToHeatmap (image rendering with dB scaling),
// ColourmapDb (consumer-facing log-scaled visualisations).
func LogMagnitude(stft [][]complex128) [][]float64 {
	T := len(stft)
	if T < 1 {
		panic("spectrogram.LogMagnitude: stft must have at least 1 frame")
	}
	F := len(stft[0])
	out := make([][]float64, T)
	for t := 0; t < T; t++ {
		if len(stft[t]) != F {
			panic("spectrogram.LogMagnitude: all rows must have equal length")
		}
		out[t] = make([]float64, F)
		for f := 0; f < F; f++ {
			m := cmplx.Abs(stft[t][f])
			out[t][f] = 20.0 * math.Log10(m+1e-12)
		}
	}
	return out
}

// PowerSpectrum computes the per-cell |X[t][f]|² reduction of a complex
// STFT. Useful as the input to mel-band integration and as the
// substrate for NMF (the non-negativity constraint of NMF wants
// power, not magnitude).
//
// Formula: out[t][f] = |stft[t][f]|² = re² + im²
//
// Allocation: returns newly-allocated [][]float64.
//
// Faster than Magnitude (avoids the sqrt) for callers that don't need
// the linear-magnitude scale.
func PowerSpectrum(stft [][]complex128) [][]float64 {
	T := len(stft)
	if T < 1 {
		panic("spectrogram.PowerSpectrum: stft must have at least 1 frame")
	}
	F := len(stft[0])
	out := make([][]float64, T)
	for t := 0; t < T; t++ {
		if len(stft[t]) != F {
			panic("spectrogram.PowerSpectrum: all rows must have equal length")
		}
		out[t] = make([]float64, F)
		for f := 0; f < F; f++ {
			r := real(stft[t][f])
			i := imag(stft[t][f])
			out[t][f] = r*r + i*i
		}
	}
	return out
}

// HalfSpectrum truncates each frame of a full-length STFT to the
// non-redundant half [0, frameSize/2 + 1]. For real-valued input
// signals, the FFT exhibits Hermitian symmetry — the upper half
// is the complex conjugate of the lower half, so callers wanting
// magnitude or power can discard it.
//
// Formula: out[t] = stft[t][0 : frameSize/2 + 1]
//
// Allocation: returns newly-allocated [][]complex128 (defensive copy
// — the caller may modify the original without affecting the output).
//
// Panics if stft is empty.
func HalfSpectrum(stft [][]complex128) [][]complex128 {
	T := len(stft)
	if T < 1 {
		panic("spectrogram.HalfSpectrum: stft must have at least 1 frame")
	}
	F := len(stft[0])
	half := F/2 + 1
	out := make([][]complex128, T)
	for t := 0; t < T; t++ {
		if len(stft[t]) != F {
			panic("spectrogram.HalfSpectrum: all rows must have equal length")
		}
		out[t] = make([]complex128, half)
		copy(out[t], stft[t][:half])
	}
	return out
}
