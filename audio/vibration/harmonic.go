package vibration

import "math"

// HarmonicEnergyRatio returns the fraction of total power within
// [fMin, fMax] that lies in narrow bands around integer multiples of
// fundamental (1f, 2f, 3f, ..., maxHarmonics × f). High ratio (~0.6-0.9)
// indicates a clean harmonic signal (healthy machine); low ratio
// indicates broadband noise (bearing wear, gear damage, cavitation).
//
// Formula:
//
//	binWidth   = sampleRate / (2 × (nBins - 1))   where nBins = len(power) = N/2 + 1
//	totalE     = Σ_{k = ceil(fMin/binWidth) .. floor(fMax/binWidth)} power[k]
//	harmonicE  = Σ_{h = 1..maxHarmonics, h × fundamental <= fMax}
//	             Σ_{k near (h × fundamental)/binWidth ± bandwidth/(2×binWidth)} power[k]
//	HER        = clamp(harmonicE / totalE, 0, 1)
//
// Parameters:
//   - power: power spectrum of length N/2 + 1 (e.g. from
//     audio.PowerSpectrum after signal.FFT)
//   - sampleRate: Hz, sampleRate > 0
//   - fundamental: Hz, e.g. fan RPM/60 × blade count. Returns 0 if
//     fundamental <= 0.
//   - bandwidth: Hz around each harmonic to count, e.g. 5 Hz
//   - fMin, fMax: total-power search range (Hz). 0 <= fMin < fMax,
//     fMax must be <= sampleRate/2.
//   - maxHarmonics: stop counting after this harmonic (e.g. 5). Must
//     be >= 1.
//
// Valid range: power must be non-empty (length >= 2). All frequency
// parameters in Hz. Returns 0 if fundamental <= 0 or total energy is 0
// (silent frame, or [fMin, fMax] brackets no bins).
//
// Precision: clamped to [0, 1]. Numerical error is the sum of the
// rounding error of N/2 + 1 float64 powers — well below 1e-12 for
// typical N <= 4096.
//
// Reference: standard machine-condition monitoring. Cempel &
// Tabaszewski 2007 "Multidimensional condition monitoring of machines
// in non-stationary operation" Mechanical Systems and Signal Processing
// 21(3); Randall 2011 "Vibration-based Condition Monitoring" §4
// (harmonic-band energy ratios as roller-element bearing indicators).
//
// Zero allocation. Does not panic on size violations — the caller
// supplies power directly.
//
// Consumed by: flagships/dipstick (per-component HarmonicRatioDrift
// DegradationTracker input).
func HarmonicEnergyRatio(power []float64, sampleRate, fundamental, bandwidth, fMin, fMax float64, maxHarmonics int) float64 {
	if fundamental <= 0 {
		return 0
	}
	nBins := len(power)
	if nBins < 2 {
		return 0
	}
	binWidth := sampleRate / float64(2*(nBins-1)) // power length = nFFT/2 + 1

	totalEnergy := 0.0
	binMinTotal := int(math.Ceil(fMin / binWidth))
	binMaxTotal := int(math.Floor(fMax / binWidth))
	if binMaxTotal > nBins-1 {
		binMaxTotal = nBins - 1
	}
	if binMinTotal < 0 {
		binMinTotal = 0
	}
	for k := binMinTotal; k <= binMaxTotal; k++ {
		totalEnergy += power[k]
	}
	if totalEnergy == 0 {
		return 0
	}

	harmonicEnergy := 0.0
	for h := 1; h <= maxHarmonics; h++ {
		fH := fundamental * float64(h)
		if fH > fMax {
			break
		}
		binCenter := fH / binWidth
		halfBins := bandwidth / binWidth / 2
		binLo := int(math.Floor(binCenter - halfBins))
		binHi := int(math.Ceil(binCenter + halfBins))
		if binLo < 0 {
			binLo = 0
		}
		if binHi >= nBins {
			binHi = nBins - 1
		}
		for k := binLo; k <= binHi; k++ {
			harmonicEnergy += power[k]
		}
	}
	if harmonicEnergy > totalEnergy {
		harmonicEnergy = totalEnergy
	}
	return harmonicEnergy / totalEnergy
}
