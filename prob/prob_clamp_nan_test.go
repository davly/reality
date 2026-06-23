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
