package audio

import "math"

// HzToMel converts a frequency in Hz to the mel scale using the
// Slaney (1998) auditory toolbox formula.
//
// Formula (Slaney 1998 §3, used by HTK, librosa default, kaldi):
//
//	m = 1127 * ln(1 + f/700)
//
// Parameters:
//   - f: frequency in Hz, f >= 0
//
// Valid range: f >= 0. Returns 0 for f == 0; monotonically increasing.
// Precision: limited by float64 log (~15 significant digits).
// Reference: Slaney, M. (1998), "Auditory Toolbox Version 2",
// Interval Research Corporation Technical Report #1998-010.
//
// This is one of two common mel-scale formulas. The HTK / librosa-default
// form used here is preferred for speech and bioacoustic work because
// 1127 * ln(1 + f/700) ≈ 2595 * log10(1 + f/700) but avoids the log10/ln
// conversion error.
func HzToMel(f float64) float64 {
	return 1127.0 * math.Log(1.0+f/700.0)
}

// MelToHz converts a mel-scale value back to frequency in Hz.
//
// Formula (inverse of HzToMel):
//
//	f = 700 * (exp(m/1127) - 1)
//
// Parameters:
//   - m: mel-scale value, m >= 0
//
// Valid range: m >= 0. Returns 0 for m == 0; monotonically increasing.
// Precision: HzToMel(MelToHz(m)) round-trips to <= 1e-9 of m for m in [0, 8000].
// Reference: Slaney 1998 §3.
func MelToHz(m float64) float64 {
	return 700.0 * (math.Exp(m/1127.0) - 1.0)
}

// MelFilterbank computes a triangular mel filterbank with numFilters bands
// covering the frequency range [fMin, fMax] over an FFT of size nFFT and
// sample rate sampleRate. The result is written to out as a [numFilters,
// nFFT/2 + 1] row-major matrix.
//
// out must have length >= numFilters * (nFFT/2 + 1). Panics on size
// violations or non-positive nFFT / sampleRate / numFilters.
//
// Formula (Slaney 1998, equations 3.1-3.3):
//
//	melLow  = HzToMel(fMin)
//	melHigh = HzToMel(fMax)
//	melPoints[k] = melLow + k * (melHigh - melLow) / (numFilters + 1)
//	binPoints[k] = floor((nFFT + 1) * MelToHz(melPoints[k]) / sampleRate)
//	filter_b[k] is triangular: rises from binPoints[b] to binPoints[b+1],
//	  falls from binPoints[b+1] to binPoints[b+2]
//
// The filterbank is NOT energy-normalised; consumers may apply slaney or
// area-normalisation after calling this. (The unnormalised form is what
// HTK and kaldi default to.)
//
// Parameters:
//   - sampleRate: audio sample rate in Hz, e.g. 16000 or 44100
//   - nFFT:       FFT size (must be a power of 2 in practice; this function
//     does not enforce that, but consumers should use signal.FFT
//     which does)
//   - numFilters: number of mel bands, typically 26-40 for speech, 64-128
//     for bioacoustic / mechanical analysis
//   - fMin:       lower frequency bound in Hz, fMin >= 0
//   - fMax:       upper frequency bound in Hz, fMax > fMin and <= sampleRate/2
//   - out:        output slice of length numFilters * (nFFT/2 + 1)
//
// Precision: triangular filter weights computed in double precision; total
// numerical error well below 1e-12 for typical settings.
// Reference: Slaney 1998 §3; HTK Book §5.4; librosa.filters.mel.
//
// Consumed by: MFCC (this package), pigeonhole / howler / dipstick (Layer 0).
func MelFilterbank(sampleRate float64, nFFT, numFilters int, fMin, fMax float64, out []float64) {
	if nFFT < 2 {
		panic("audio.MelFilterbank: nFFT must be >= 2")
	}
	if numFilters < 1 {
		panic("audio.MelFilterbank: numFilters must be >= 1")
	}
	if sampleRate <= 0 {
		panic("audio.MelFilterbank: sampleRate must be > 0")
	}
	if fMin < 0 || fMax <= fMin || fMax > sampleRate/2 {
		panic("audio.MelFilterbank: invalid frequency range; require 0 <= fMin < fMax <= sampleRate/2")
	}

	nBins := nFFT/2 + 1
	if len(out) < numFilters*nBins {
		panic("audio.MelFilterbank: out must have length >= numFilters*(nFFT/2+1)")
	}

	// Mel scale endpoints
	melLow := HzToMel(fMin)
	melHigh := HzToMel(fMax)

	// numFilters + 2 mel-points: one before the first filter (its left edge)
	// and one after the last (its right edge).
	melPoints := make([]float64, numFilters+2)
	for k := 0; k <= numFilters+1; k++ {
		melPoints[k] = melLow + float64(k)*(melHigh-melLow)/float64(numFilters+1)
	}

	// Convert mel points back to FFT bin indices.
	binPoints := make([]float64, numFilters+2)
	for k := 0; k < len(melPoints); k++ {
		hz := MelToHz(melPoints[k])
		// Per Slaney 1998 / HTK: bin = (nFFT+1) * f / sr, kept as float64
		// for fractional triangular weighting (avoid floor() so weights
		// are smooth across bin boundaries).
		binPoints[k] = float64(nFFT+1) * hz / sampleRate
	}

	// Zero the output region.
	for i := 0; i < numFilters*nBins; i++ {
		out[i] = 0.0
	}

	// Build triangular filters.
	for b := 0; b < numFilters; b++ {
		left := binPoints[b]
		center := binPoints[b+1]
		right := binPoints[b+2]
		rowOff := b * nBins
		for k := 0; k < nBins; k++ {
			fk := float64(k)
			var w float64
			switch {
			case fk < left || fk > right:
				w = 0.0
			case fk <= center:
				if center > left {
					w = (fk - left) / (center - left)
				} else {
					w = 0.0
				}
			default: // center < fk <= right
				if right > center {
					w = (right - fk) / (right - center)
				} else {
					w = 0.0
				}
			}
			out[rowOff+k] = w
		}
	}
}

// PowerSpectrum computes |X[k]|^2 for the first nFFT/2 + 1 bins from
// in-place real and imag slices produced by signal.FFT. Result written
// to out, which must have length >= nFFT/2 + 1.
//
// Formula: P[k] = real[k]^2 + imag[k]^2
//
// Note: this is the squared magnitude. To get magnitude (sometimes called
// the "amplitude spectrum"), take sqrt(P[k]) per bin. Mel filterbank
// energies are conventionally computed from the power spectrum.
//
// Zero allocation. Panics if out is too short.
//
// Reference: standard DSP convention; e.g. HTK Book §5.4.2.
func PowerSpectrum(real, imag []float64, out []float64) {
	n := len(real)
	if len(imag) != n {
		panic("audio.PowerSpectrum: real and imag must have equal length")
	}
	half := n/2 + 1
	if len(out) < half {
		panic("audio.PowerSpectrum: out must have length >= n/2 + 1")
	}
	for k := 0; k < half; k++ {
		out[k] = real[k]*real[k] + imag[k]*imag[k]
	}
}

// ApplyFilterbank applies a precomputed mel filterbank (numFilters x nBins,
// row-major) to a power spectrum and writes the resulting band energies
// to out (length >= numFilters).
//
// Formula: energy[b] = sum_k filterbank[b, k] * power[k]
//
// Zero allocation. Panics on size violations.
func ApplyFilterbank(power, filterbank []float64, numFilters, nBins int, out []float64) {
	if len(power) < nBins {
		panic("audio.ApplyFilterbank: power must have length >= nBins")
	}
	if len(filterbank) < numFilters*nBins {
		panic("audio.ApplyFilterbank: filterbank must have length >= numFilters*nBins")
	}
	if len(out) < numFilters {
		panic("audio.ApplyFilterbank: out must have length >= numFilters")
	}
	for b := 0; b < numFilters; b++ {
		rowOff := b * nBins
		s := 0.0
		for k := 0; k < nBins; k++ {
			s += filterbank[rowOff+k] * power[k]
		}
		out[b] = s
	}
}
