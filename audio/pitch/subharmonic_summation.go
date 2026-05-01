package pitch

import "math"

// SubharmonicSummation estimates the fundamental frequency of a power
// spectrum by summing harmonic evidence in the frequency domain. The
// distinctive virtue: detects the fundamental even when the
// fundamental BIN is empty (the "missing fundamental" perceptual
// phenomenon — humans hear a pitch where no spectral energy lies).
//
// Algorithm (Hermes 1988 §II):
//
//	For each candidate fundamental f in [fMin, fMax]:
//	  S(f) = Σ_{h=1..harmonics} weight(h) · spectrum_at(h · f)
//	weight(h) = 1 / h            (Hermes' default subharmonic weighting)
//	spectrum_at(f) = power[round(f / binWidth)]   (nearest-bin lookup)
//	f_0 = argmax_f S(f)
//
// Edge cases:
//   - all-zero spectrum: returns 0
//   - candidate frequency exceeds Nyquist for any harmonic: that
//     harmonic contributes 0
//
// Parameters:
//   - spectrum:   power-spectrum (length N/2 + 1, e.g. from
//     audio.PowerSpectrum after signal.FFT)
//   - sampleRate: Hz, > 0. The full FFT input length is inferred as
//     (len(spectrum) - 1) · 2.
//   - fMin, fMax: candidate fundamental search range (Hz). Must
//     satisfy 0 < fMin < fMax, fMax < sampleRate/2.
//   - harmonics:  number of harmonics to sum. Typical: 5-10. Must be
//     >= 1.
//
// Returns: estimated fundamental in Hz, or 0 if no peak found.
//
// Valid range: len(spectrum) >= 2; sampleRate > 0; harmonics >= 1.
// Precision: limited to candidate-grid resolution. The algorithm
// scans candidate f at 1 Hz steps in the implementation here (a
// pragmatic balance — finer steps multiply cost without
// commensurate accuracy gain on noisy inputs).
//
// Allocation: O(fMax-fMin) scratch for the SHS scores — single slice.
// Returns a single float64.
//
// Reference: Hermes, D. J. (1988). "Measurement of pitch by
// subharmonic summation." J. Acoust. Soc. Am. 83(1), 257-264.
//
// Consumed by: dipstick (low-frequency machine fundamental detection
// where the fundamental bin is too coarse — e.g. boiler combustion
// at sub-20 Hz that nevertheless presents at 60, 120, 180 Hz harmonics).
func SubharmonicSummation(spectrum []float64, sampleRate float64, fMin, fMax float64, harmonics int) float64 {
	if sampleRate <= 0 {
		panic("pitch.SubharmonicSummation: sampleRate must be > 0")
	}
	if fMin <= 0 || fMax <= fMin {
		panic("pitch.SubharmonicSummation: must satisfy 0 < fMin < fMax")
	}
	if harmonics < 1 {
		panic("pitch.SubharmonicSummation: harmonics must be >= 1")
	}
	nBins := len(spectrum)
	if nBins < 2 {
		panic("pitch.SubharmonicSummation: spectrum must have length >= 2")
	}

	// Quick all-zero check.
	maxP := 0.0
	for k := 0; k < nBins; k++ {
		if spectrum[k] > maxP {
			maxP = spectrum[k]
		}
	}
	if maxP == 0 {
		return 0
	}

	binWidth := sampleRate / float64(2*(nBins-1))
	nyquist := sampleRate / 2

	// Iterate candidate fundamentals at 1 Hz steps.
	fLo := math.Floor(fMin)
	fHi := math.Ceil(fMax)
	if fLo < 1 {
		fLo = 1
	}
	if fHi > nyquist {
		fHi = nyquist
	}

	bestF := 0.0
	bestS := 0.0
	for f := fLo; f <= fHi; f += 1.0 {
		s := 0.0
		for h := 1; h <= harmonics; h++ {
			fh := f * float64(h)
			if fh > nyquist {
				break
			}
			bin := int(math.Round(fh / binWidth))
			if bin >= nBins {
				break
			}
			weight := 1.0 / float64(h)
			s += weight * spectrum[bin]
		}
		if s > bestS {
			bestS = s
			bestF = f
		}
	}
	return bestF
}
