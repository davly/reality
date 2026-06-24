package evidence

import (
	"math"
	"testing"
)

// TestSampleBackingFactor_NaNHalfSaturation pins the fix: `halfSaturation <= 0`
// is false for NaN, so a NaN used to skip the sanitizer and poison nf/(nf+NaN)
// to NaN (and Score multiplied that through). A NaN (or non-positive)
// halfSaturation must fall back to DefaultHalfSaturation.
func TestSampleBackingFactor_NaNHalfSaturation(t *testing.T) {
	got := SampleBackingFactor(100, math.NaN())
	if math.IsNaN(got) || got < 0 || got >= 1 {
		t.Errorf("SampleBackingFactor(100, NaN) = %v, want finite in [0,1)", got)
	}
	if want := SampleBackingFactor(100, DefaultHalfSaturation); math.Abs(got-want) > 1e-12 {
		t.Errorf("SampleBackingFactor(100, NaN) = %v, want default-fallback %v", got, want)
	}
	// Score must not propagate the NaN either.
	if s := Score(2400, 1.0, 1.0, math.NaN()); math.IsNaN(s) || s < 0 || s > 1 {
		t.Errorf("Score(2400,1,1,NaN) = %v, want finite in [0,1]", s)
	}
}
