package tempo

import (
	"errors"
	"math"
	"testing"
)

// --- Estimate happy paths -------------------------------------------

func TestEstimate_PerfectPeriodic_RecoversTempo(t *testing.T) {
	// Generate a 120 BPM novelty: at frameRate 100 Hz, beat lag = 50 frames.
	// Place delta spikes every 50 frames.
	const frameRate = 100.0
	const periodFrames = 50
	novelty := make([]float64, 1000)
	for i := 0; i < len(novelty); i += periodFrames {
		novelty[i] = 1.0
	}

	bpm, err := Estimate(novelty, frameRate, DefaultOptions())
	if err != nil {
		t.Fatalf("Estimate err: %v", err)
	}
	want := 120.0
	if math.Abs(bpm-want) > 5.0 {
		t.Errorf("bpm = %g, want %g (±5)", bpm, want)
	}
}

func TestEstimate_NoiseAroundPeriodic_StillRecoversTempo(t *testing.T) {
	// 90 BPM with noise: at frameRate 50 Hz, beat lag ≈ 33 frames.
	const frameRate = 50.0
	const periodFrames = 33
	novelty := make([]float64, 800)
	for i := range novelty {
		novelty[i] = 0.05 // noise floor
	}
	for i := 0; i < len(novelty); i += periodFrames {
		novelty[i] = 1.0
	}

	bpm, err := Estimate(novelty, frameRate, Options{MinBpm: 60, MaxBpm: 200})
	if err != nil {
		t.Fatalf("Estimate err: %v", err)
	}
	// Allow ±10% — noise + integer lag rounding.
	want := 60.0 * frameRate / periodFrames
	if math.Abs(bpm-want)/want > 0.10 {
		t.Errorf("bpm = %g, want %g (±10%%)", bpm, want)
	}
}

// --- Estimate error paths -------------------------------------------

func TestEstimate_ZeroFrameRate_ReturnsErrInvalidParams(t *testing.T) {
	novelty := []float64{1, 0, 1, 0, 1, 0}
	if _, err := Estimate(novelty, 0, DefaultOptions()); !errors.Is(err, ErrInvalidParams) {
		t.Errorf("got %v, want ErrInvalidParams", err)
	}
}

func TestEstimate_NegativeBpmBounds_ReturnsErrInvalidParams(t *testing.T) {
	novelty := []float64{1, 0, 1, 0, 1, 0}
	cases := []Options{
		{MinBpm: -10, MaxBpm: 200},
		{MinBpm: 60, MaxBpm: -10},
		{MinBpm: 200, MaxBpm: 60}, // swapped
		{MinBpm: 100, MaxBpm: 100}, // degenerate
	}
	for i, opts := range cases {
		_, err := Estimate(novelty, 100, opts)
		if !errors.Is(err, ErrInvalidParams) {
			t.Errorf("case %d: got %v, want ErrInvalidParams", i, err)
		}
	}
}

func TestEstimate_TooFewSamples_ReturnsErrInsufficientData(t *testing.T) {
	novelty := []float64{1, 0, 1}
	if _, err := Estimate(novelty, 100, DefaultOptions()); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("got %v, want ErrInsufficientData", err)
	}
}

func TestEstimate_AllZeroNovelty_StillReturnsBpm(t *testing.T) {
	// A flat-zero novelty doesn't error — ACF is all zero, peak is the
	// first lag in range. Pin behaviour: returns a valid BPM (the
	// tightest end of the range) rather than failing. Production
	// callers should pre-screen for silence/no-onset signals.
	novelty := make([]float64, 200)
	bpm, err := Estimate(novelty, 100, DefaultOptions())
	if err != nil {
		t.Fatalf("expected nil err on flat input, got %v", err)
	}
	if bpm <= 0 {
		t.Errorf("bpm = %g, expected positive", bpm)
	}
}

// --- Autocorrelation -----------------------------------------------

func TestAutocorrelation_Lag0_SumOfSquares(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	out := make([]float64, 1)
	if err := Autocorrelation(x, 0, out); err != nil {
		t.Fatal(err)
	}
	want := 1.0 + 4 + 9 + 16
	if out[0] != want {
		t.Errorf("ACF[0] = %g, want %g", out[0], want)
	}
}

func TestAutocorrelation_Lag1_ShiftedDot(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	out := make([]float64, 2)
	if err := Autocorrelation(x, 1, out); err != nil {
		t.Fatal(err)
	}
	// ACF[1] = 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
	if out[1] != 20 {
		t.Errorf("ACF[1] = %g, want 20", out[1])
	}
}

func TestAutocorrelation_OutputTooSmall_ReturnsErrOutputSize(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	out := make([]float64, 1) // need 3 for maxLag=2
	if err := Autocorrelation(x, 2, out); !errors.Is(err, ErrOutputSize) {
		t.Errorf("got %v, want ErrOutputSize", err)
	}
}

func TestAutocorrelation_NegativeLag_ReturnsErrOutputSize(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	out := make([]float64, 5)
	if err := Autocorrelation(x, -1, out); !errors.Is(err, ErrOutputSize) {
		t.Errorf("got %v, want ErrOutputSize", err)
	}
}

func TestAutocorrelation_LagBeyondLength_ReturnsErrOutputSize(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	out := make([]float64, 5)
	if err := Autocorrelation(x, 4, out); !errors.Is(err, ErrOutputSize) {
		t.Errorf("got %v, want ErrOutputSize", err)
	}
}

// --- LagToBpm / BpmToLag round-trip --------------------------------

func TestLagToBpm_RoundTripWithBpmToLag(t *testing.T) {
	// 120 BPM at 100 Hz → lag 50; back to BPM exactly.
	lag := BpmToLag(120, 100)
	if lag != 50 {
		t.Errorf("BpmToLag(120, 100) = %d, want 50", lag)
	}
	bpm := LagToBpm(50, 100)
	if math.Abs(bpm-120) > 1e-9 {
		t.Errorf("LagToBpm(50, 100) = %g, want 120", bpm)
	}
}

func TestLagToBpm_NonPositive_ReturnsZero(t *testing.T) {
	if LagToBpm(0, 100) != 0 {
		t.Error("LagToBpm(0, *) should return 0")
	}
	if LagToBpm(-5, 100) != 0 {
		t.Error("LagToBpm(-5, *) should return 0")
	}
	if LagToBpm(50, 0) != 0 {
		t.Error("LagToBpm(*, 0) should return 0")
	}
}

func TestBpmToLag_NonPositive_ReturnsZero(t *testing.T) {
	if BpmToLag(0, 100) != 0 {
		t.Error("BpmToLag(0, *) should return 0")
	}
	if BpmToLag(-120, 100) != 0 {
		t.Error("BpmToLag(-120, *) should return 0")
	}
	if BpmToLag(120, 0) != 0 {
		t.Error("BpmToLag(*, 0) should return 0")
	}
}

// --- Determinism --------------------------------------------------

func TestEstimate_Deterministic(t *testing.T) {
	novelty := make([]float64, 500)
	for i := 0; i < len(novelty); i += 33 {
		novelty[i] = 1.0
	}

	a, errA := Estimate(novelty, 50, Options{MinBpm: 60, MaxBpm: 200})
	b, errB := Estimate(novelty, 50, Options{MinBpm: 60, MaxBpm: 200})

	if errA != nil || errB != nil {
		t.Fatalf("errs: %v / %v", errA, errB)
	}
	if a != b {
		t.Errorf("non-deterministic: %g vs %g", a, b)
	}
}
