package segmentation

// Segment is a half-open interval [StartIdx, EndIdx) in sample units
// describing one extracted audio event. StartIdx <= EndIdx; if
// StartIdx == EndIdx the segment is empty.
//
// The half-open convention matches Go slice semantics: a caller can
// extract the segment with samples[seg.StartIdx:seg.EndIdx].
type Segment struct {
	StartIdx int
	EndIdx   int
}

// Duration returns the segment length in samples.
func (s Segment) Duration() int { return s.EndIdx - s.StartIdx }

// Segment performs energy-based segmentation on a continuous audio
// buffer. Contiguous runs of frames whose short-time energy exceeds
// vadThreshold are emitted as segments.
//
// Algorithm (Rabiner & Sambur 1975 §2):
//
//	For each frame t at sample-position t·hopSize:
//	  E[t] = (1/frameSize) · Σ samples[t·hopSize + i]²
//	  active[t] = (E[t] >= vadThreshold)
//	Each maximal run of active=true frames produces one Segment with:
//	  StartIdx = (run start frame) · hopSize
//	  EndIdx   = ((run end frame) + 1) · hopSize    (one-past-last)
//
// Parameters:
//   - samples:      real-valued audio buffer
//   - frameSize:    per-frame window length (samples). Typical: 1024.
//   - hopSize:      advance between frames (samples). Typical: frameSize/4.
//   - vadThreshold: short-time mean-square energy threshold. Typical:
//     1e-4 for noisy ambient, 1e-6 for studio-quiet input.
//
// Returns: slice of Segments. Empty if no frame exceeds the threshold.
//
// Valid range: len(samples) >= frameSize; frameSize >= 1; hopSize >= 1;
// vadThreshold >= 0.
// Precision: 1e-12 in the sum-of-squares accumulation.
//
// Allocation: returns newly-allocated []Segment. One pre-allocated
// frame-energy scratch (single allocation, reused across frames).
//
// Reference: Rabiner, L. R. & Sambur, M. R. (1975). "An algorithm for
// determining the endpoints of isolated utterances." Bell Syst. Tech.
// J. 54(2), 297-315.
//
// Consumed by: pigeonhole (cheap-first-pass call extraction), howler
// (vocalisation extraction).
func SegmentByEnergy(samples []float64, frameSize, hopSize int, vadThreshold float64) []Segment {
	if frameSize < 1 {
		panic("segmentation.SegmentByEnergy: frameSize must be >= 1")
	}
	if hopSize < 1 {
		panic("segmentation.SegmentByEnergy: hopSize must be >= 1")
	}
	if vadThreshold < 0 {
		panic("segmentation.SegmentByEnergy: vadThreshold must be >= 0")
	}
	if len(samples) < frameSize {
		return nil
	}

	numFrames := (len(samples)-frameSize)/hopSize + 1
	if numFrames < 1 {
		return nil
	}

	out := []Segment{}
	inSegment := false
	segStart := 0

	for t := 0; t < numFrames; t++ {
		start := t * hopSize
		s := 0.0
		for i := 0; i < frameSize; i++ {
			x := samples[start+i]
			s += x * x
		}
		e := s / float64(frameSize)
		active := e >= vadThreshold

		if active && !inSegment {
			inSegment = true
			segStart = start
		} else if !active && inSegment {
			inSegment = false
			// Close segment at end of previous frame.
			endIdx := start
			out = append(out, Segment{StartIdx: segStart, EndIdx: endIdx})
		}
	}
	// Trailing-active run: close at end of last covered sample.
	if inSegment {
		// EndIdx = position one-past-last sample of the last frame.
		lastFrameStart := (numFrames - 1) * hopSize
		endIdx := lastFrameStart + frameSize
		if endIdx > len(samples) {
			endIdx = len(samples)
		}
		out = append(out, Segment{StartIdx: segStart, EndIdx: endIdx})
	}
	return out
}
