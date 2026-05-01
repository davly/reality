package segmentation

// MergeCloseSegments coalesces nearby segments separated by less than
// gapMs milliseconds into single composite segments. Post-processing
// step for any segmenter — defends against the case where a single
// physical event (e.g., a bird call with internal pauses) is incorrectly
// split into multiple sub-segments.
//
// Algorithm:
//
//	gapSamples = gapMs * sampleRate / 1000
//	for each pair of consecutive segments (a, b):
//	  if b.StartIdx - a.EndIdx <= gapSamples:
//	    merge into Segment{a.StartIdx, b.EndIdx}
//	  else: keep separate
//
// Parameters:
//   - segments:   input segments in sample coordinates. Should be in
//     ascending StartIdx order; this function does NOT sort.
//   - gapMs:      maximum inter-segment gap in milliseconds that
//     triggers a merge. Typical: 50-200 ms.
//   - sampleRate: Hz, > 0
//
// Returns: newly-allocated []Segment with merges applied. Empty input
// returns empty output.
//
// Valid range: gapMs >= 0; sampleRate > 0; segments must be in
// ascending StartIdx order.
//
// Allocation: returns newly-allocated []Segment.
//
// Reference: standard post-processing in event-extraction pipelines.
//
// Consumed by: pigeonhole (post-process VAD output to merge call
// fragments back into whole calls), howler (merge vocalisation
// fragments into whole vocalisations).
func MergeCloseSegments(segments []Segment, gapMs, sampleRate int) []Segment {
	if gapMs < 0 {
		panic("segmentation.MergeCloseSegments: gapMs must be >= 0")
	}
	if sampleRate <= 0 {
		panic("segmentation.MergeCloseSegments: sampleRate must be > 0")
	}
	if len(segments) == 0 {
		return nil
	}

	gapSamples := gapMs * sampleRate / 1000

	out := make([]Segment, 0, len(segments))
	current := segments[0]
	for i := 1; i < len(segments); i++ {
		next := segments[i]
		if next.StartIdx-current.EndIdx <= gapSamples {
			// Merge.
			if next.EndIdx > current.EndIdx {
				current.EndIdx = next.EndIdx
			}
		} else {
			out = append(out, current)
			current = next
		}
	}
	out = append(out, current)
	return out
}
