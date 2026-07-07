package prob

import (
	"math"
	"testing"
)

// Documents the TRUE accuracy of NormalQuantile (Acklam) via the inverse-relation
// round-trip Phi(Phi^{-1}(p)) == p. This both pins the corrected, honest precision
// claim and guards against a misguided "deep-tail fix" — the bare Acklam
// approximation is correct (round-trips to ~1e-7) all the way into the extreme
// tail; a Phase-C finding that claimed a 5% error at p=1e-300 was a false positive
// (it compared against an erroneous reference rather than the round-trip).
func TestNormalQuantile_RoundTrip(t *testing.T) {
	for _, p := range []float64{0.5, 1e-3, 0.999, 1e-50, 1e-150, 1e-300} {
		x := NormalQuantile(p, 0, 1)
		back := NormalCDF(x, 0, 1)
		rel := math.Abs(back-p) / p
		if rel > 1e-6 {
			t.Errorf("NormalQuantile round-trip p=%g: NQ=%.6f Phi(NQ)=%.6e rel=%.3e (want <1e-6)", p, x, back, rel)
		}
	}
	// Sanity: the deep-tail value is ~-37.05 (Phi(-37.05)~1e-300), NOT ~-35.06.
	if x := NormalQuantile(1e-300, 0, 1); x > -36 || x < -38 {
		t.Errorf("NormalQuantile(1e-300)=%.4f; want ~-37.05", x)
	}
}
