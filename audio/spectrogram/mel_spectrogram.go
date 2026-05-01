package spectrogram

import (
	"github.com/davly/reality/audio"
)

// MelSpectrogram composes the STFT power spectrum with a mel filterbank
// to produce a 2-D per-frame mel-band energy matrix.
//
// Algorithm:
//
//	power[t][k] = |STFT[t][k]|²    for k in [0, frameSize/2]
//	melEnergy[t][b] = Σ_k filterbank[b][k] · power[t][k]
//
// where filterbank is a numFilters × (frameSize/2 + 1) matrix from
// audio.MelFilterbank.
//
// Parameters:
//   - stft:       T × frameSize complex STFT (full-length, both halves
//     present; the function uses only the first frameSize/2 + 1 bins)
//   - sampleRate: Hz, e.g. 16000 or 44100
//   - frameSize:  must equal len(stft[0])
//   - numFilters: number of mel bands (typical: 26-40 speech, 64-128
//     bioacoustic / mechanical)
//   - fMin, fMax: mel-filterbank coverage; 0 <= fMin < fMax <= sampleRate/2
//
// Returns: T × numFilters mel-energy matrix. Newly-allocated; the
// caller owns the returned slice. Filterbank is constructed once
// internally and reused across frames.
//
// Valid range: T >= 1; frameSize a power of 2; numFilters >= 1.
// Precision: ≤1e-12 numerical error per cell (FFT precision plus
// triangular-filter sum).
// Panics on shape violations or empty input.
//
// Reference: HTK Book §5.4; librosa.feature.melspectrogram. The
// composition (STFT → power → mel) is the standard speech-recognition
// front-end pipeline.
//
// Consumed by: pigeonhole / howler / dipstick (mel-energy heatmap
// rendering), audio.MFCC chain (when MFCC is the next step).
func MelSpectrogram(stft [][]complex128, sampleRate float64, frameSize, numFilters int, fMin, fMax float64) [][]float64 {
	T := len(stft)
	if T < 1 {
		panic("spectrogram.MelSpectrogram: stft must have at least 1 frame")
	}
	if len(stft[0]) != frameSize {
		panic("spectrogram.MelSpectrogram: stft[0] length must equal frameSize")
	}
	nBins := frameSize/2 + 1

	// Build filterbank once.
	filterbank := make([]float64, numFilters*nBins)
	audio.MelFilterbank(sampleRate, frameSize, numFilters, fMin, fMax, filterbank)

	out := make([][]float64, T)
	power := make([]float64, nBins)
	for t := 0; t < T; t++ {
		if len(stft[t]) != frameSize {
			panic("spectrogram.MelSpectrogram: all rows must have equal length")
		}
		// Compute |X|² for the half-spectrum.
		for k := 0; k < nBins; k++ {
			r := real(stft[t][k])
			im := imag(stft[t][k])
			power[k] = r*r + im*im
		}
		// Apply filterbank.
		row := make([]float64, numFilters)
		audio.ApplyFilterbank(power, filterbank, numFilters, nBins, row)
		out[t] = row
	}
	return out
}

// LogMelSpectrogram returns the natural log of MelSpectrogram with a
// 1e-10 floor (HTK convention). Common input to MFCC; common
// substrate for visualisation since the log compression matches
// human auditory perception.
//
// Formula: out[t][b] = log(max(melEnergy[t][b], 1e-10))
//
// Allocation: returns newly-allocated [][]float64.
//
// Reference: HTK Book §5.6; standard speech-recognition convention.
func LogMelSpectrogram(stft [][]complex128, sampleRate float64, frameSize, numFilters int, fMin, fMax float64) [][]float64 {
	mel := MelSpectrogram(stft, sampleRate, frameSize, numFilters, fMin, fMax)
	for t := 0; t < len(mel); t++ {
		audio.LogMelEnergies(mel[t], 1e-10, mel[t])
	}
	return mel
}
