package evidence

import (
	"math"
	"testing"
)

// TestSampleBackingFactor_StaysBelowOne pins the [0,1) clamp. For absurdly large
// n (k below nf*2^-53), nf/(nf+k) rounds to exactly 1.0 in float64, violating the
// documented half-open output range [0,1). The clamp keeps it strictly < 1.
func TestSampleBackingFactor_StaysBelowOne(t *testing.T) {
	got := SampleBackingFactor(1<<62, DefaultHalfSaturation) // n ~ 4.6e18, k=500
	if got >= 1.0 {
		t.Errorf("SampleBackingFactor(huge n) = %v, want < 1 (documented [0,1))", got)
	}
	if got <= 0 || math.IsNaN(got) {
		t.Errorf("SampleBackingFactor(huge n) = %v, want in (0,1)", got)
	}
	// Normal values are unaffected.
	if g := SampleBackingFactor(500, 500); math.Abs(g-0.5) > 1e-12 {
		t.Errorf("SampleBackingFactor(500,500) = %v, want 0.5", g)
	}
}
