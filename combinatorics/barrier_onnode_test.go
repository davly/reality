package combinatorics

import (
	"math"
	"testing"
)

// TestBarrierOptionReflection_BarrierOnNode pins the ceil-boundary fix. When the
// barrier lies exactly on a binomial node (barrier = s0*u^m), log(barrier/s0)/
// log(u) is m, but float rounding gives m + ~1e-16 so the old ceil yielded m+1 --
// one level too high -- and paths reaching node m (which == the barrier, hence
// breach) were NOT knocked out. Here barrier = 100*u^2 with n=4: every profitable
// terminal node (j=3,4) reaches level 2, so the up-and-out option must price ~0.
// The bug returned 4.5425.
func TestBarrierOptionReflection_BarrierOnNode(t *testing.T) {
	s0, k, r, sigma, tt := 100.0, 100.0, 0.05, 0.2, 1.0
	n := 4
	dt := tt / float64(n)
	u := math.Exp(sigma * math.Sqrt(dt))
	barrier := s0 * math.Pow(u, 2) // exactly on node m=2

	got := BarrierOptionReflection(s0, k, r, sigma, tt, barrier, n)
	if math.IsNaN(got) || math.Abs(got) > 1e-9 {
		t.Errorf("BarrierOptionReflection(barrier on node u^2) = %v, want ~0 (all profitable paths breach)", got)
	}

	// A barrier genuinely between nodes (just above u^2) must NOT be snapped down:
	// node 2 no longer breaches, so the option retains value (> 0).
	barrierAbove := s0 * math.Pow(u, 2.5)
	if got2 := BarrierOptionReflection(s0, k, r, sigma, tt, barrierAbove, n); got2 <= 0 || math.IsNaN(got2) {
		t.Errorf("BarrierOptionReflection(barrier between nodes) = %v, want > 0 (node 2 does not breach)", got2)
	}
}
