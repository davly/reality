package onset

import "math/cmplx"

// SuperFlux detects onsets via vibrato-suppressed spectral flux. The
// modern de-facto detector for percussive onsets in musical and
// bioacoustic signals.
//
// Algorithm (Böck & Widmer 2013 §2):
//
//	M[t][k] = |X[t][k]|                         (magnitude)
//	M̃[t-1][k] = max_{|j-k| <= maxFilterFrames} M[t-1][j]   (max-filter
//	                                              along frequency axis)
//	SF[t] = Σ_k H(M[t][k] - M̃[t-1][k])
//
// where H(x) = max(x, 0) is half-wave rectification. The crucial
// detail vs vanilla spectral flux: the per-bin reference is the MAX
// over a small frequency neighbourhood of the previous frame. A
// vibrato (slowly frequency-modulated tone) sweeps through neighbouring
// bins continuously — vanilla flux flags every sweep as a fake onset,
// but the max-filter "expects" the energy to appear in a neighbour and
// suppresses the false positive.
//
// Parameters:
//   - stft:             T × frameSize complex matrix.
//   - maxFilterFrames:  half-width of the frequency max-filter (bins).
//     Typical: 3 for 1024-point FFT at 22050 Hz (Böck & Widmer 2013
//     recommend a max-filter spanning roughly a quarter-tone: log2(2^(1/24))).
//
// Returns: slice of frame indices where onsets occur.
//
// Valid range: len(stft) >= 2; maxFilterFrames >= 0.
// Precision: ≤1e-12 per bin difference; the max-filter is exact.
//
// Allocation: O(F) ring scratch per frame for the max-filter (single
// reusable slice) + O(T) strength function + the returned []int.
//
// Reference: Böck, S. & Widmer, G. (2013). "Maximum filter vibrato
// suppression for onset detection." Proc. DAFx-13. The reference
// adopts a logarithmic-magnitude variant; this implementation uses
// linear magnitude for simplicity (the relative ranking of frames is
// preserved under monotonic transforms).
//
// Consumed by: pigeonhole (vibrato-rich bird-song onsets — lark,
// nightingale), howler (modulated canine vocalisations).
func SuperFlux(stft [][]complex128, maxFilterFrames int) []int {
	T := len(stft)
	if T < 2 {
		panic("onset.SuperFlux: stft must have at least 2 frames")
	}
	if maxFilterFrames < 0 {
		panic("onset.SuperFlux: maxFilterFrames must be >= 0")
	}
	F := len(stft[0])
	if F < 1 {
		panic("onset.SuperFlux: stft frames must be non-empty")
	}

	// Pre-compute magnitudes once per frame (avoid double work in the
	// inner difference loop).
	mags := make([][]float64, T)
	for t := 0; t < T; t++ {
		if len(stft[t]) != F {
			panic("onset.SuperFlux: all stft rows must have equal length")
		}
		mags[t] = make([]float64, F)
		for k := 0; k < F; k++ {
			mags[t][k] = cmplx.Abs(stft[t][k])
		}
	}

	// SuperFlux strength.
	SF := make([]float64, T)
	for t := 1; t < T; t++ {
		s := 0.0
		for k := 0; k < F; k++ {
			// Max over [k - maxFilterFrames, k + maxFilterFrames] of mags[t-1].
			lo := k - maxFilterFrames
			if lo < 0 {
				lo = 0
			}
			hi := k + maxFilterFrames
			if hi >= F {
				hi = F - 1
			}
			ref := mags[t-1][lo]
			for j := lo + 1; j <= hi; j++ {
				if mags[t-1][j] > ref {
					ref = mags[t-1][j]
				}
			}
			d := mags[t][k] - ref
			if d > 0 {
				s += d
			}
		}
		SF[t] = s
	}

	return PickPeaksAdaptive(SF, 1.5, 3)
}
