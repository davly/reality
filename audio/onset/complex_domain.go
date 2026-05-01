package onset

import (
	"math"
	"math/cmplx"
)

// ComplexDomainOnset detects onsets by tracking the residual between
// the observed complex spectrum and a phase-extrapolation prediction.
// Combines magnitude AND phase deviation in a single residual — picks
// up non-percussive onsets (legato attacks, soft note transitions)
// that magnitude-only spectral flux misses.
//
// Algorithm (Bello, Duxbury, Davies & Sandler 2004):
//
//	X̂[t][k] = |X[t-1][k]| · exp(j · (2·φ[t-1][k] - φ[t-2][k]))
//	CD[t]   = Σ_k |X[t][k] - X̂[t][k]|
//
// where φ[t][k] = arg(X[t][k]) is the unwrapped phase. The prediction
// X̂[t][k] is the linear-phase-extrapolation of the previous two
// frames: same magnitude, phase advanced by the same Δφ as last step.
// A "stationary tone" produces zero residual; an onset (energy step
// AND/OR phase discontinuity) produces a peak.
//
// Parameters:
//   - stft: T × frameSize complex matrix from spectrogram.Compute.
//     Must have T >= 3 frames (need t-2, t-1 prediction context).
//
// Returns: slice of frame indices where onsets occur.
//
// Valid range: len(stft) >= 3; all rows have equal length.
// Precision: ~1e-10 per bin (subtraction of close-magnitude complex
// values can lose precision in catastrophic-cancellation cases; the
// Σ over bins re-amortises this).
//
// Allocation: O(T·F) phase scratch + O(T) strength function + the
// returned []int.
//
// Reference: Bello, J. P., Duxbury, C., Davies, M. & Sandler, M.
// (2004). "On the use of phase and energy for musical onset detection
// in the complex domain." IEEE Signal Processing Letters 11(6),
// 553-556.
//
// Consumed by: pigeonhole (soft-onset bird calls — chiff-chaff,
// warbler legato vocalisations), howler (mood-shift detection in
// continuous canine vocalisations).
func ComplexDomainOnset(stft [][]complex128) []int {
	T := len(stft)
	if T < 3 {
		panic("onset.ComplexDomainOnset: stft must have at least 3 frames")
	}
	F := len(stft[0])
	if F < 1 {
		panic("onset.ComplexDomainOnset: stft frames must be non-empty")
	}

	CD := make([]float64, T)
	for t := 2; t < T; t++ {
		if len(stft[t]) != F || len(stft[t-1]) != F || len(stft[t-2]) != F {
			panic("onset.ComplexDomainOnset: all stft rows must have equal length")
		}
		s := 0.0
		for k := 0; k < F; k++ {
			magPrev := cmplx.Abs(stft[t-1][k])
			phasePrev := cmplx.Phase(stft[t-1][k])
			phasePrevPrev := cmplx.Phase(stft[t-2][k])

			// Predicted phase: φ̂ = 2·φ[t-1] - φ[t-2] (linear extrapolation).
			predictedPhase := 2*phasePrev - phasePrevPrev
			// Wrap into [-π, π].
			predictedPhase = wrapPhase(predictedPhase)

			predicted := complex(magPrev*math.Cos(predictedPhase), magPrev*math.Sin(predictedPhase))
			residual := stft[t][k] - predicted
			s += cmplx.Abs(residual)
		}
		CD[t] = s
	}

	return PickPeaksAdaptive(CD, 1.5, 3)
}

// wrapPhase reduces a phase value into the [-π, π] range. Used by the
// complex-domain detector to normalise predicted phases before
// reconstructing the predicted complex coefficient.
//
// Reference: standard convention; matches numpy's angle() output.
func wrapPhase(phi float64) float64 {
	const twoPi = 2 * math.Pi
	for phi > math.Pi {
		phi -= twoPi
	}
	for phi < -math.Pi {
		phi += twoPi
	}
	return phi
}
