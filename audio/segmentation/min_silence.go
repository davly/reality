package segmentation

// SegmentWithMinSilence segments samples by splitting at any silent
// run of >= minSilenceMs milliseconds. A frame is "silent" if its
// short-time energy is below 5% of the peak frame energy across the
// whole buffer (auto-calibrated noise floor). Useful for recordings
// where events are well-separated in time but vary in duration —
// e.g., a long bird-recording with multiple distinct calls.
//
// Algorithm:
//
//	frameSize = sampleRate / 100   (10 ms windows)
//	hopSize   = frameSize           (no overlap; cheap)
//	peakE     = max_t E[t]
//	silentRun = max contiguous run of frames with E[t] < 0.05·peakE
//	silentSamples = silentRun · hopSize
//	if silentSamples >= minSilenceMs/1000 · sampleRate: split here
//	otherwise: continue current segment
//
// Output Segments are in SAMPLE coordinates.
//
// Parameters:
//   - samples:      real-valued audio buffer
//   - sampleRate:   Hz, > 0
//   - minSilenceMs: minimum silent duration in milliseconds that
//     defines a segment boundary. Typical: 100-500 ms.
//
// Returns: slice of Segments. Empty if no active frames found.
//
// Valid range: sampleRate > 0; minSilenceMs >= 0; len(samples) >= 1.
// Precision: 1e-12 in the sum-of-squares accumulation.
//
// Allocation: returns newly-allocated []Segment + per-frame scratch.
//
// Reference: standard practice in continuous-speech segmentation; see
// Rabiner & Sambur 1975 for the energy-based boundary primitive that
// this generalises with a configurable silence-duration constraint.
//
// Consumed by: pigeonhole (long-recording event extraction with known
// inter-call spacing), dipstick (machine-event isolation in long
// recordings).
func SegmentWithMinSilence(samples []float64, sampleRate int, minSilenceMs int) []Segment {
	if sampleRate <= 0 {
		panic("segmentation.SegmentWithMinSilence: sampleRate must be > 0")
	}
	if minSilenceMs < 0 {
		panic("segmentation.SegmentWithMinSilence: minSilenceMs must be >= 0")
	}
	n := len(samples)
	if n < 1 {
		return nil
	}

	frameSize := sampleRate / 100 // 10 ms windows
	if frameSize < 1 {
		frameSize = 1
	}
	hopSize := frameSize
	if n < frameSize {
		// Whole buffer too short — treat as one segment if any energy.
		s := 0.0
		for i := 0; i < n; i++ {
			s += samples[i] * samples[i]
		}
		if s > 0 {
			return []Segment{{StartIdx: 0, EndIdx: n}}
		}
		return nil
	}

	numFrames := (n-frameSize)/hopSize + 1
	energies := make([]float64, numFrames)
	peakE := 0.0
	for t := 0; t < numFrames; t++ {
		start := t * hopSize
		s := 0.0
		for i := 0; i < frameSize; i++ {
			x := samples[start+i]
			s += x * x
		}
		e := s / float64(frameSize)
		energies[t] = e
		if e > peakE {
			peakE = e
		}
	}
	if peakE == 0 {
		return nil
	}

	// Auto-calibrated silence floor at 5% of peak.
	silenceFloor := 0.05 * peakE
	minSilenceFrames := (minSilenceMs * sampleRate / 1000) / hopSize
	if minSilenceFrames < 1 {
		minSilenceFrames = 1
	}

	out := []Segment{}
	inSegment := false
	segStartFrame := 0
	silenceRunStart := -1

	for t := 0; t < numFrames; t++ {
		isSilent := energies[t] < silenceFloor
		if isSilent {
			if silenceRunStart < 0 {
				silenceRunStart = t
			}
		} else {
			silenceRunStart = -1
		}

		if !isSilent && !inSegment {
			inSegment = true
			segStartFrame = t
		}
		// Close-segment trigger: we are in a segment, currently silent,
		// and the silent run is long enough.
		if inSegment && silenceRunStart >= 0 && t-silenceRunStart+1 >= minSilenceFrames {
			endFrame := silenceRunStart
			out = append(out, Segment{
				StartIdx: segStartFrame * hopSize,
				EndIdx:   endFrame * hopSize,
			})
			inSegment = false
		}
	}
	// Trailing segment.
	if inSegment {
		end := numFrames * hopSize
		if end > n {
			end = n
		}
		out = append(out, Segment{
			StartIdx: segStartFrame * hopSize,
			EndIdx:   end,
		})
	}
	return out
}
