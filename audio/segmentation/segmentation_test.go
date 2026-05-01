package segmentation

import (
	"math"
	"testing"

	"github.com/davly/reality/audio/spectrogram"
	"github.com/davly/reality/signal"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makeBurstAt fills out[start:start+length] with a 1 kHz sinusoid at the
// given amplitude and zeros elsewhere. Multiple bursts produce
// well-separated synthetic "calls".
func makeBurstAt(out []float64, start, length, sr int, freq, amp float64) {
	for i := 0; i < length && start+i < len(out); i++ {
		out[start+i] = amp * math.Sin(2*math.Pi*freq*float64(i)/float64(sr))
	}
}

// computeSTFTHelper using Hann window.
func computeSTFTHelper(samples []float64, frameSize, hopSize int) [][]complex128 {
	window := make([]float64, frameSize)
	signal.HannWindow(frameSize, window)
	return spectrogram.Compute(samples, frameSize, hopSize, window)
}

// ---------------------------------------------------------------------------
// SegmentByEnergy (VAD-based)
// ---------------------------------------------------------------------------

func TestSegmentByEnergy_FindsThreeDistinctSegments(t *testing.T) {
	sr := 16000
	dur := 4 * sr // 4 seconds total
	samples := make([]float64, dur)
	// Three 0.5s bursts at 0.5s, 1.5s, 2.5s, each separated by 0.5s of silence.
	makeBurstAt(samples, sr/2, sr/2, sr, 1000, 1.0)
	makeBurstAt(samples, 3*sr/2, sr/2, sr, 1000, 1.0)
	makeBurstAt(samples, 5*sr/2, sr/2, sr, 1000, 1.0)

	got := SegmentByEnergy(samples, 1024, 256, 0.01)
	if len(got) != 3 {
		t.Errorf("expected 3 segments, got %d: %v", len(got), got)
	}
	// Each segment should have a duration close to 0.5s.
	for i, s := range got {
		durSec := float64(s.Duration()) / float64(sr)
		if durSec < 0.3 || durSec > 0.8 {
			t.Errorf("segment %d duration %.2fs outside expected ~0.5s", i, durSec)
		}
	}
}

func TestSegmentByEnergy_PureSilenceProducesZero(t *testing.T) {
	sr := 16000
	samples := make([]float64, sr)
	got := SegmentByEnergy(samples, 1024, 256, 0.01)
	if len(got) != 0 {
		t.Errorf("expected 0 segments on silent buffer, got %d: %v", len(got), got)
	}
}

func TestSegmentByEnergy_SustainedToneProducesOneSegment(t *testing.T) {
	sr := 16000
	dur := 2 * sr
	samples := make([]float64, dur)
	for i := 0; i < dur; i++ {
		samples[i] = math.Sin(2 * math.Pi * 440 * float64(i) / float64(sr))
	}
	got := SegmentByEnergy(samples, 1024, 256, 0.01)
	if len(got) != 1 {
		t.Errorf("expected 1 segment for sustained tone, got %d: %v", len(got), got)
	}
}

func TestSegmentByEnergy_PanicsOnInvalid(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	SegmentByEnergy([]float64{1, 2, 3}, 0, 1, 0.1)
}

func TestSegmentByEnergy_PanicsOnNegativeThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on negative threshold")
		}
	}()
	SegmentByEnergy(make([]float64, 1024), 256, 256, -1)
}

// ---------------------------------------------------------------------------
// SegmentByOnsetOffset
// ---------------------------------------------------------------------------

func TestSegmentByOnsetOffset_FindsThreeDistinctSegments(t *testing.T) {
	sr := 16000
	dur := 4 * sr
	samples := make([]float64, dur)
	makeBurstAt(samples, sr/2, sr/2, sr, 1000, 1.0)
	makeBurstAt(samples, 3*sr/2, sr/2, sr, 1000, 1.0)
	makeBurstAt(samples, 5*sr/2, sr/2, sr, 1000, 1.0)

	stft := computeSTFTHelper(samples, 1024, 256)
	got := SegmentByOnsetOffset(stft, 0.5, 0.05)
	if len(got) < 2 || len(got) > 4 {
		t.Errorf("expected 2-4 segments, got %d: %v", len(got), got)
	}
}

func TestSegmentByOnsetOffset_PanicsOnTooShort(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on len(stft) < 2")
		}
	}()
	SegmentByOnsetOffset([][]complex128{{1 + 0i}}, 0.5, 0.05)
}

func TestSegmentByOnsetOffset_PanicsOnInvalidThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on invalid threshold")
		}
	}()
	stft := [][]complex128{{1, 1}, {2, 2}}
	SegmentByOnsetOffset(stft, 1.5, 0.05)
}

// ---------------------------------------------------------------------------
// SegmentWithMinSilence
// ---------------------------------------------------------------------------

func TestSegmentWithMinSilence_SplitsAtLongPauses(t *testing.T) {
	sr := 16000
	dur := 4 * sr
	samples := make([]float64, dur)
	// Three bursts separated by 500ms silence — should split.
	makeBurstAt(samples, sr/2, sr/2, sr, 1000, 1.0)
	makeBurstAt(samples, 3*sr/2, sr/2, sr, 1000, 1.0)
	makeBurstAt(samples, 5*sr/2, sr/2, sr, 1000, 1.0)

	got := SegmentWithMinSilence(samples, sr, 200) // 200 ms minimum silence
	if len(got) != 3 {
		t.Errorf("expected 3 segments, got %d: %v", len(got), got)
	}
}

func TestSegmentWithMinSilence_NoSplitForShortPauses(t *testing.T) {
	sr := 16000
	dur := 2 * sr
	samples := make([]float64, dur)
	// Two bursts separated by only 50ms silence — should NOT split when
	// minSilence > 50ms.
	makeBurstAt(samples, 0, sr/2, sr, 1000, 1.0) // 0 to 0.5s
	makeBurstAt(samples, sr/2+sr/20, sr/2, sr, 1000, 1.0) // 0.55s to 1.05s

	got := SegmentWithMinSilence(samples, sr, 200) // 200 ms min silence
	if len(got) != 1 {
		t.Errorf("expected 1 merged segment with 200ms minSilence, got %d: %v", len(got), got)
	}
}

func TestSegmentWithMinSilence_PureSilenceProducesZero(t *testing.T) {
	sr := 16000
	samples := make([]float64, sr)
	got := SegmentWithMinSilence(samples, sr, 100)
	if len(got) != 0 {
		t.Errorf("expected 0 segments on silent buffer, got %d", len(got))
	}
}

func TestSegmentWithMinSilence_PanicsOnInvalidSampleRate(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on sampleRate <= 0")
		}
	}()
	SegmentWithMinSilence([]float64{1, 2}, 0, 100)
}

// ---------------------------------------------------------------------------
// MergeCloseSegments
// ---------------------------------------------------------------------------

func TestMergeCloseSegments_MergesNearbySegments(t *testing.T) {
	sr := 16000
	// Two segments 50 ms apart → should merge with gapMs >= 50.
	segs := []Segment{
		{StartIdx: 0, EndIdx: sr / 4},                  // 0 - 250 ms
		{StartIdx: sr/4 + sr/20, EndIdx: sr / 2},       // 300 ms - 500 ms (50 ms gap)
	}
	got := MergeCloseSegments(segs, 100, sr) // 100 ms gap tolerance
	if len(got) != 1 {
		t.Errorf("expected 1 merged segment, got %d: %v", len(got), got)
	}
	if got[0].StartIdx != 0 || got[0].EndIdx != sr/2 {
		t.Errorf("merged segment wrong: got [%d, %d), want [0, %d)", got[0].StartIdx, got[0].EndIdx, sr/2)
	}
}

func TestMergeCloseSegments_KeepsDistantSegments(t *testing.T) {
	sr := 16000
	segs := []Segment{
		{StartIdx: 0, EndIdx: sr / 4},                // 0 - 250 ms
		{StartIdx: sr, EndIdx: 5 * sr / 4},          // 1000 - 1250 ms (750 ms gap)
	}
	got := MergeCloseSegments(segs, 100, sr) // 100 ms gap tolerance
	if len(got) != 2 {
		t.Errorf("expected 2 unmerged segments, got %d: %v", len(got), got)
	}
}

func TestMergeCloseSegments_EmptyInput(t *testing.T) {
	got := MergeCloseSegments(nil, 100, 16000)
	if len(got) != 0 {
		t.Errorf("expected empty output for empty input, got %v", got)
	}
}

func TestMergeCloseSegments_PanicsOnInvalid(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on invalid sampleRate")
		}
	}()
	MergeCloseSegments([]Segment{{0, 1}}, 100, 0)
}

// ---------------------------------------------------------------------------
// FilterByMinDuration
// ---------------------------------------------------------------------------

func TestFilterByMinDuration_DropsShortSegments(t *testing.T) {
	sr := 16000
	segs := []Segment{
		{StartIdx: 0, EndIdx: sr / 100},   // 10 ms — too short
		{StartIdx: sr, EndIdx: sr + sr/2}, // 500 ms — kept
		{StartIdx: 2 * sr, EndIdx: 2*sr + sr/10}, // 100 ms — kept (boundary)
	}
	got := FilterByMinDuration(segs, 100, sr)
	if len(got) != 2 {
		t.Errorf("expected 2 segments, got %d: %v", len(got), got)
	}
}

func TestFilterByMinDuration_DropsAll(t *testing.T) {
	sr := 16000
	segs := []Segment{
		{StartIdx: 0, EndIdx: sr / 100}, // 10 ms
	}
	got := FilterByMinDuration(segs, 50, sr)
	if len(got) != 0 {
		t.Errorf("expected 0 segments after filter, got %d: %v", len(got), got)
	}
}

func TestFilterByMinDuration_KeepsAll(t *testing.T) {
	sr := 16000
	segs := []Segment{
		{StartIdx: 0, EndIdx: sr},     // 1 s
		{StartIdx: 2 * sr, EndIdx: 3 * sr}, // 1 s
	}
	got := FilterByMinDuration(segs, 100, sr)
	if len(got) != 2 {
		t.Errorf("expected 2 segments, got %d", len(got))
	}
}

func TestFilterByMinDuration_PanicsOnInvalid(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on invalid minMs")
		}
	}()
	FilterByMinDuration([]Segment{{0, 100}}, -1, 16000)
}

// ---------------------------------------------------------------------------
// Composition tests
// ---------------------------------------------------------------------------

func TestComposition_VAD_Merge_Filter(t *testing.T) {
	// Realistic pipeline: VAD produces fragmented segments → merge close
	// → drop tiny remnants. Should end up with 1 clean segment.
	sr := 16000
	dur := 2 * sr
	samples := make([]float64, dur)
	// One sustained "call" with internal pauses (zero-amplitude gaps).
	makeBurstAt(samples, sr/4, sr/8, sr, 1000, 1.0)        // 250-375 ms
	makeBurstAt(samples, 3*sr/8+sr/100, sr/8, sr, 1000, 1.0) // ~385-510 ms — 10 ms gap
	makeBurstAt(samples, sr/2+sr/100, sr/8, sr, 1000, 1.0)  // ~510-635 ms — touching

	segs := SegmentByEnergy(samples, 1024, 256, 0.01)
	t.Logf("VAD found %d segments before merge", len(segs))
	merged := MergeCloseSegments(segs, 100, sr)
	t.Logf("After merge (gap=100ms): %d segments", len(merged))
	filtered := FilterByMinDuration(merged, 100, sr)
	t.Logf("After filter (min=100ms): %d segments", len(filtered))

	if len(filtered) < 1 || len(filtered) > 2 {
		t.Errorf("composition pipeline produced %d segments, expected 1-2", len(filtered))
	}
}
