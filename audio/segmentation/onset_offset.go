package segmentation

import (
	"math/cmplx"

	"github.com/davly/reality/audio/onset"
)

// SegmentByOnsetOffset extracts segments by tracking onset detections
// for the rising edge and energy decay below an offset threshold for
// the trailing edge. More robust than energy-only VAD on signals where
// the onset has a distinctive spectral signature but the offset is a
// gradual decay.
//
// Algorithm:
//
//	onsets       = onset frame indices via SpectralFluxStrength on stft
//	frameEnergy  = per-frame Σ_k |stft[t][k]|²
//	for each onset frame o_i:
//	  startIdx = o_i (in frames)
//	  endIdx   = first frame after o_i where frameEnergy < offsetThreshold
//	             (or last frame if no decay observed before next onset)
//	  emit Segment{StartFrame, EndFrame}
//
// The thresholds are absolute fractions of peak frame energy:
//   - onsetThreshold:  flux must exceed this fraction of mean+1.5σ flux
//     for picks to be admitted (delegated to PickPeaksAdaptive default)
//   - offsetThreshold: frame energy below this fraction of peak energy
//     marks segment end
//
// Output Segments are in FRAME coordinates (not samples). Callers
// wanting sample-coordinate output should multiply StartIdx and EndIdx
// by the STFT hopSize used to compute stft.
//
// Parameters:
//   - stft:            T × frameSize complex matrix (from
//     spectrogram.Compute)
//   - onsetThreshold:  fraction in (0, 1] for onset detection floor
//     (currently a hint — picks delegated to onset.PickPeaksAdaptive)
//   - offsetThreshold: fraction in (0, 1] of peak frame-energy below
//     which a segment is considered ended
//
// Returns: slice of Segments where StartIdx and EndIdx are FRAME indices.
//
// Valid range: len(stft) >= 2; thresholds in (0, 1].
// Precision: 1e-12 per |X|² accumulation.
//
// Allocation: O(T) frame-energy scratch + onset detection + returned
// []Segment.
//
// Reference: Bello et al. (2005). "A tutorial on onset detection in
// music signals." IEEE Trans. Speech Audio Proc. 13(5).
//
// Consumed by: pigeonhole (most-robust per-call extraction), howler.
func SegmentByOnsetOffset(stft [][]complex128, onsetThreshold, offsetThreshold float64) []Segment {
	if onsetThreshold <= 0 || onsetThreshold > 1 {
		panic("segmentation.SegmentByOnsetOffset: onsetThreshold must be in (0, 1]")
	}
	if offsetThreshold <= 0 || offsetThreshold > 1 {
		panic("segmentation.SegmentByOnsetOffset: offsetThreshold must be in (0, 1]")
	}
	T := len(stft)
	if T < 2 {
		panic("segmentation.SegmentByOnsetOffset: stft must have at least 2 frames")
	}
	F := len(stft[0])
	if F < 1 {
		panic("segmentation.SegmentByOnsetOffset: stft frames must be non-empty")
	}

	// Frame energies + their peak.
	frameE := make([]float64, T)
	maxE := 0.0
	for t := 0; t < T; t++ {
		if len(stft[t]) != F {
			panic("segmentation.SegmentByOnsetOffset: all stft rows must have equal length")
		}
		s := 0.0
		for k := 0; k < F; k++ {
			m := cmplx.Abs(stft[t][k])
			s += m * m
		}
		frameE[t] = s
		if s > maxE {
			maxE = s
		}
	}
	floor := offsetThreshold * maxE

	// Get onsets via spectral flux.
	onsets := onset.SpectralFluxOnset(stft)
	if len(onsets) == 0 {
		// No onsets — if anything exceeds the offset floor at all, treat
		// as a single sustained segment.
		first := -1
		last := -1
		for t := 0; t < T; t++ {
			if frameE[t] >= floor {
				if first < 0 {
					first = t
				}
				last = t
			}
		}
		if first < 0 {
			return nil
		}
		return []Segment{{StartIdx: first, EndIdx: last + 1}}
	}

	out := make([]Segment, 0, len(onsets))
	lastEnd := -1
	for _, o := range onsets {
		// Skip onsets that fall inside an already-emitted segment —
		// defends against the case where a single physical event
		// produces multiple onset-detector hits (e.g. attack +
		// reverberation tail).
		if o < lastEnd {
			continue
		}
		// Find the offset: the first frame after o where frameE drops
		// below floor; or T.
		end := T
		for t := o + 1; t < T; t++ {
			if frameE[t] < floor {
				end = t
				break
			}
		}
		if end > T {
			end = T
		}
		if end > o {
			out = append(out, Segment{StartIdx: o, EndIdx: end})
			lastEnd = end
		}
	}
	return out
}
