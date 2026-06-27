package changepoint

import (
	"math"
	"testing"
)

// TestBettingEValue_NaNObservation pins the fix: the support clamp used
// `if xn < 0 {..} else if xn > 1 {..}`, both false for NaN, so a NaN observation
// escaped and poisoned the e-value (and the e-process accumulator) to NaN. A NaN
// must map to the null mean -> a neutral e-value of 1.
func TestBettingEValue_NaNObservation(t *testing.T) {
	ev, err := BettingEValue(0.5, 1.0, 0.0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	got := ev(math.NaN())
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("BettingEValue closure on NaN = %v, want finite", got)
	}
	if math.Abs(got-1.0) > 1e-12 {
		t.Errorf("BettingEValue closure on NaN = %v, want 1.0 (neutral e-value)", got)
	}
}
