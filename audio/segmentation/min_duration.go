package segmentation

// FilterByMinDuration drops segments shorter than minMs milliseconds.
// Post-processing step for any segmenter — defends against transient
// false positives (e.g., a 5 ms pop that the energy-VAD flagged as a
// segment).
//
// Algorithm:
//
//	minSamples = minMs * sampleRate / 1000
//	for each segment s:
//	  if s.Duration() >= minSamples: keep
//	  else: drop
//
// Parameters:
//   - segments:   input segments in sample coordinates
//   - minMs:      minimum acceptable segment duration in milliseconds.
//     Typical: 50-200 ms for bird calls; 20-100 ms for percussive
//     mechanical events.
//   - sampleRate: Hz, > 0
//
// Returns: newly-allocated []Segment with short segments removed.
//
// Valid range: minMs >= 0; sampleRate > 0.
//
// Allocation: returns newly-allocated []Segment.
//
// Reference: standard post-processing.
//
// Consumed by: pigeonhole (drop spurious sub-100ms VAD-detected
// "calls" that are likely noise), dipstick (drop transient pops in
// machine recordings).
func FilterByMinDuration(segments []Segment, minMs, sampleRate int) []Segment {
	if minMs < 0 {
		panic("segmentation.FilterByMinDuration: minMs must be >= 0")
	}
	if sampleRate <= 0 {
		panic("segmentation.FilterByMinDuration: sampleRate must be > 0")
	}
	if len(segments) == 0 {
		return nil
	}

	minSamples := minMs * sampleRate / 1000
	out := make([]Segment, 0, len(segments))
	for _, s := range segments {
		if s.Duration() >= minSamples {
			out = append(out, s)
		}
	}
	return out
}
