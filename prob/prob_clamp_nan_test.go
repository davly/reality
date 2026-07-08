package prob

import (
	"math"
	"testing"
)

// ClampProbability must honor its documented contract "NaN -> MinProb". Go's math.Max/Min
// propagate NaN, so the unguarded formula returned NaN — a fail-open that leaked through
// every caller that clamps here. (godfather A13, 2026-06-23)
func TestClampProbability_NaNReturnsMinProb(t *testing.T) {
	if got := ClampProbability(math.NaN()); got != MinProb {
		t.Fatalf("ClampProbability(NaN) = %v, want MinProb (%v)", got, MinProb)
	}
	if got := ClampProbability(math.Inf(1)); got != MaxProb {
		t.Fatalf("ClampProbability(+Inf) = %v, want MaxProb", got)
	}
	if got := ClampProbability(math.Inf(-1)); got != MinProb {
		t.Fatalf("ClampProbability(-Inf) = %v, want MinProb", got)
	}
	for _, p := range []float64{0.0, 0.5, 1.0, 0.7} {
		want := math.Max(MinProb, math.Min(MaxProb, p))
		if got := ClampProbability(p); got != want {
			t.Fatalf("ClampProbability(%v) = %v, want %v (finite unchanged)", p, got, want)
		}
	}
	// cascade: a NaN flowing through a clamping caller no longer leaks NaN.
	if got := LogOddsToProb(math.NaN()); math.IsNaN(got) {
		t.Fatalf("LogOddsToProb(NaN) leaked NaN through the clamp: %v", got)
	}
}

// ConfidenceFromPValue documents "result is always in [0, 1]" but its manual clamp leaked
// NaN; it now routes through the NaN-safe clamp01.
func TestConfidenceFromPValue_NaNAndBounds(t *testing.T) {
	if got := ConfidenceFromPValue(math.NaN()); got != 0 {
		t.Fatalf("ConfidenceFromPValue(NaN) = %v, want 0 (doc: always in [0,1])", got)
	}
	if got := ConfidenceFromPValue(0.05); got != 0.95 {
		t.Fatalf("ConfidenceFromPValue(0.05) = %v, want 0.95", got)
	}
	if got := ConfidenceFromPValue(-1); got != 1 { // p<0 -> 1-p>1 -> clamp to 1
		t.Fatalf("ConfidenceFromPValue(-1) = %v, want 1", got)
	}
	if got := ConfidenceFromPValue(2); got != 0 { // p>1 -> 1-p<0 -> clamp to 0
		t.Fatalf("ConfidenceFromPValue(2) = %v, want 0", got)
	}
}
