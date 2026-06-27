package geometry

import (
	"math"
	"testing"
)

// TestSDFCapsule_ZeroLengthSegment pins the fix for a == b (zero-length segment).
// The projection t = (ap·ab)/(ab·ab) was 0/0 = NaN, which fails both clamp
// comparisons, so the returned distance was NaN. A zero-length capsule is a
// sphere at a.
func TestSDFCapsule_ZeroLengthSegment(t *testing.T) {
	// point (3,0,0), degenerate segment a=b=origin, radius 1 -> 3 - 1 = 2.
	got := SDFCapsule([3]float64{3, 0, 0}, [3]float64{0, 0, 0}, [3]float64{0, 0, 0}, 1.0)
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("SDFCapsule zero-length: got %v, want finite 2.0", got)
	}
	if math.Abs(got-2.0) > 1e-12 {
		t.Errorf("SDFCapsule zero-length: got %v, want 2.0 (= |p-a| - radius)", got)
	}
	// For a == b the capsule must equal the sphere at a.
	p := [3]float64{1, 2, 2}
	a := [3]float64{0, 0, 0}
	gotC := SDFCapsule(p, a, a, 0.5)
	wantS := SDFSphere(p, a, 0.5)
	if math.Abs(gotC-wantS) > 1e-12 {
		t.Errorf("SDFCapsule(a==b) = %v, want SDFSphere = %v", gotC, wantS)
	}
}
