package onset

import "math/cmplx"

// SpectralFluxOnset detects onsets by tracking the half-wave-rectified
// frame-to-frame difference of the magnitude spectrum. This is the
// canonical "spectral flux" detector — the workhorse of MIR onset
// detection and the baseline against which more sophisticated
// detectors are evaluated.
//
// Algorithm (Bello & Sandler 2003; Dixon 2006):
//
//	For each frame t = 1, 2, ..., T-1:
//	  SF[t] = Σ_k H(|X[t][k]| - |X[t-1][k]|)
//	where H(x) = x if x > 0, else 0   (half-wave rectification)
//
// The half-wave rectification ensures that only positive energy
// changes (onsets) contribute — decays do not produce a flux peak.
// The detection is then a peak-picking step on the SF[t] strength
// function.
//
// Parameters:
//   - stft: T × frameSize complex matrix from spectrogram.Compute.
//     Each row is one frame's complex FFT bins.
//
// Returns: slice of frame indices where onsets occur.
//
// Valid range: len(stft) >= 2; all rows have equal length.
// Precision: 1e-12 per bin (single magnitude difference + sum).
//
// Allocation: O(T) scratch for the strength function plus the
// returned []int. Magnitude of each bin computed once per frame
// pair (no caching of |X[t]| across consecutive frames in this
// allocating form — the frame-pair is read directly).
//
// Reference: Bello, J. P. & Sandler, M. (2003). "Phase-based note
// onset detection for music signals." ICASSP 2003. Dixon, S. (2006).
// "Onset detection revisited." DAFx-06 — empirical comparison
// establishing spectral flux as the strongest single-feature detector
// for percussive onsets.
//
// Consumed by: pigeonhole (call boundary detection), howler
// (vocalisation onset), reality/audio/segmentation.SegmentByOnsetOffset.
func SpectralFluxOnset(stft [][]complex128) []int {
	T := len(stft)
	if T < 2 {
		panic("onset.SpectralFluxOnset: stft must have at least 2 frames")
	}
	F := len(stft[0])
	if F < 1 {
		panic("onset.SpectralFluxOnset: stft frames must be non-empty")
	}

	SF := make([]float64, T)
	// First frame has no predecessor; SF[0] = 0 by convention.
	for t := 1; t < T; t++ {
		if len(stft[t]) != F {
			panic("onset.SpectralFluxOnset: all stft rows must have equal length")
		}
		s := 0.0
		for k := 0; k < F; k++ {
			d := cmplx.Abs(stft[t][k]) - cmplx.Abs(stft[t-1][k])
			if d > 0 {
				s += d
			}
		}
		SF[t] = s
	}

	return PickPeaksAdaptive(SF, 1.5, 3)
}

// SpectralFluxStrength returns the spectral flux strength function
// (one float per frame) without applying peak picking. Useful when
// callers want to compose their own thresholding strategy or fuse
// flux with other onset cues.
//
// Returns: T-length slice; SF[0] = 0 by convention.
//
// Allocation: returns newly-allocated []float64.
//
// Reference: same as SpectralFluxOnset.
func SpectralFluxStrength(stft [][]complex128) []float64 {
	T := len(stft)
	if T < 1 {
		panic("onset.SpectralFluxStrength: stft must be non-empty")
	}
	F := len(stft[0])
	if F < 1 {
		panic("onset.SpectralFluxStrength: stft frames must be non-empty")
	}

	SF := make([]float64, T)
	for t := 1; t < T; t++ {
		if len(stft[t]) != F {
			panic("onset.SpectralFluxStrength: all stft rows must have equal length")
		}
		s := 0.0
		for k := 0; k < F; k++ {
			d := cmplx.Abs(stft[t][k]) - cmplx.Abs(stft[t-1][k])
			if d > 0 {
				s += d
			}
		}
		SF[t] = s
	}
	return SF
}
