package beat

import (
	"errors"
	"math"
	"testing"
)

// --- Track happy paths ----------------------------------------------

func TestTrack_StrongPeriodicNovelty_RecoversBeatsNearMultiplesOfPeriod(t *testing.T) {
	// 120 BPM at 100 Hz → period = 50 frames. Place strong novelty
	// pulses at multiples of 50.
	const frameRate = 100.0
	const periodFrames = 50
	novelty := make([]float64, 600)
	for i := 0; i < len(novelty); i += periodFrames {
		novelty[i] = 1.0
	}

	beats, err := Track(novelty, frameRate, DefaultOptions(120))
	if err != nil {
		t.Fatalf("Track err: %v", err)
	}

	if len(beats) < 2 {
		t.Fatalf("got %d beats, want at least 2", len(beats))
	}

	// Check beats land near multiples of period (within ±2 frames).
	for i, b := range beats {
		nearest := int(math.Round(float64(b.FrameIndex)/float64(periodFrames))) * periodFrames
		dev := b.FrameIndex - nearest
		if dev < -2 || dev > 2 {
			t.Errorf("beat %d at frame %d, expected near %d (±2)", i, b.FrameIndex, nearest)
		}
	}
}

func TestTrack_BeatsAreSortedAscending(t *testing.T) {
	const frameRate = 100.0
	novelty := make([]float64, 500)
	for i := 0; i < len(novelty); i += 50 {
		novelty[i] = 1.0
	}

	beats, err := Track(novelty, frameRate, DefaultOptions(120))
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(beats); i++ {
		if beats[i].FrameIndex <= beats[i-1].FrameIndex {
			t.Errorf("beats not strictly ascending: beats[%d].Frame=%d <= beats[%d].Frame=%d",
				i, beats[i].FrameIndex, i-1, beats[i-1].FrameIndex)
		}
	}
}

func TestTrack_BeatTimesMatchFrameRate(t *testing.T) {
	const frameRate = 100.0
	novelty := make([]float64, 500)
	for i := 0; i < len(novelty); i += 50 {
		novelty[i] = 1.0
	}

	beats, err := Track(novelty, frameRate, DefaultOptions(120))
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range beats {
		expected := float64(b.FrameIndex) / frameRate
		if math.Abs(b.TimeSeconds-expected) > 1e-9 {
			t.Errorf("beat at frame %d: time %g, want %g",
				b.FrameIndex, b.TimeSeconds, expected)
		}
	}
}

// --- Track error paths ----------------------------------------------

func TestTrack_ZeroFrameRate_ReturnsErrInvalidParams(t *testing.T) {
	novelty := make([]float64, 100)
	if _, err := Track(novelty, 0, DefaultOptions(120)); !errors.Is(err, ErrInvalidParams) {
		t.Errorf("got %v, want ErrInvalidParams", err)
	}
}

func TestTrack_NonPositiveOpts_ReturnsErrInvalidParams(t *testing.T) {
	novelty := make([]float64, 200)
	const fr = 100.0
	cases := []Options{
		{TempoBpm: 0, TightnessAlpha: 100, SearchWindowMultiplier: 4},
		{TempoBpm: -120, TightnessAlpha: 100, SearchWindowMultiplier: 4},
		{TempoBpm: 120, TightnessAlpha: 0, SearchWindowMultiplier: 4},
		{TempoBpm: 120, TightnessAlpha: 100, SearchWindowMultiplier: 0},
		{TempoBpm: 120, TightnessAlpha: 100, SearchWindowMultiplier: -1},
	}
	for i, o := range cases {
		if _, err := Track(novelty, fr, o); !errors.Is(err, ErrInvalidParams) {
			t.Errorf("case %d: got %v, want ErrInvalidParams", i, err)
		}
	}
}

func TestTrack_PeriodLessThanOneFrame_ReturnsErrInvalidParams(t *testing.T) {
	// 60000 BPM at 100Hz → period = 0.1 frames < 1 → reject.
	novelty := make([]float64, 200)
	if _, err := Track(novelty, 100, Options{TempoBpm: 60000, TightnessAlpha: 100, SearchWindowMultiplier: 4}); !errors.Is(err, ErrInvalidParams) {
		t.Errorf("got %v, want ErrInvalidParams", err)
	}
}

func TestTrack_TooShortNovelty_ReturnsErrInsufficientData(t *testing.T) {
	// At 120 BPM, period = 50 frames; need >= 100. Give 80.
	novelty := make([]float64, 80)
	if _, err := Track(novelty, 100, DefaultOptions(120)); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("got %v, want ErrInsufficientData", err)
	}
}

// --- Determinism ----------------------------------------------------

func TestTrack_Deterministic(t *testing.T) {
	const fr = 100.0
	novelty := make([]float64, 500)
	for i := 0; i < len(novelty); i += 50 {
		novelty[i] = 1.0
	}

	a, errA := Track(novelty, fr, DefaultOptions(120))
	b, errB := Track(novelty, fr, DefaultOptions(120))

	if errA != nil || errB != nil {
		t.Fatalf("errs: %v / %v", errA, errB)
	}
	if len(a) != len(b) {
		t.Fatalf("non-deterministic length: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("non-deterministic at %d: %+v vs %+v", i, a[i], b[i])
		}
	}
}

// --- Tempo prior is honoured ---------------------------------------

func TestTrack_DifferentTempoPrior_DifferentBeatSpacing(t *testing.T) {
	// Same novelty, different tempo prior → different beat spacings.
	const fr = 100.0
	novelty := make([]float64, 500)
	// Spread novelty more uniformly to avoid biasing toward one tempo.
	for i := 1; i < len(novelty); i++ {
		novelty[i] = 0.3
	}

	slow, errS := Track(novelty, fr, DefaultOptions(60))
	fast, errF := Track(novelty, fr, DefaultOptions(180))

	if errS != nil || errF != nil {
		t.Fatalf("errs: %v / %v", errS, errF)
	}

	// Faster prior = more beats over the same duration.
	if len(fast) <= len(slow) {
		t.Errorf("expected fast (180 BPM) to yield more beats than slow (60 BPM); got fast=%d slow=%d",
			len(fast), len(slow))
	}
}

// --- DefaultOptions -------------------------------------------------

func TestDefaultOptions_PinsAlphaAndWindow(t *testing.T) {
	o := DefaultOptions(120)
	if o.TempoBpm != 120 {
		t.Errorf("TempoBpm = %g, want 120", o.TempoBpm)
	}
	if o.TightnessAlpha != 100 {
		t.Errorf("TightnessAlpha = %g, want 100", o.TightnessAlpha)
	}
	if o.SearchWindowMultiplier != 4 {
		t.Errorf("SearchWindowMultiplier = %g, want 4", o.SearchWindowMultiplier)
	}
}
