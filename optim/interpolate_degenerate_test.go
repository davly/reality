package optim

import (
	"math"
	"testing"
)

// TestLinearInterpolate_DegenerateInterval pins the doc/behaviour fix. The
// docstring claimed a panic on x0==x1, but the bare division returned NaN (when
// x==x0, via 0/0) or +/-Inf (when x!=x0). It now returns NaN consistently and the
// doc matches.
func TestLinearInterpolate_DegenerateInterval(t *testing.T) {
	if got := LinearInterpolate(2, 5, 2, 9, 2); !math.IsNaN(got) {
		t.Errorf("LinearInterpolate(x0==x1, x==x0) = %v, want NaN", got)
	}
	if got := LinearInterpolate(2, 5, 2, 9, 7); !math.IsNaN(got) {
		t.Errorf("LinearInterpolate(x0==x1, x!=x0) = %v, want NaN (was +/-Inf)", got)
	}
	// A normal interpolation still works.
	if got := LinearInterpolate(0, 0, 10, 100, 5); math.Abs(got-50) > 1e-12 {
		t.Errorf("LinearInterpolate(0,0,10,100,5) = %v, want 50", got)
	}
}
