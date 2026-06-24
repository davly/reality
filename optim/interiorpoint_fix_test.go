package optim

import (
	"math"
	"testing"
)

// Regression for the InteriorPoint quarantine. The old bespoke barrier loop
// returned NaN or ~1e100 garbage with a nil error on every well-posed LP; it
// now delegates to SimplexMethod (correct result, honest infeasibility error).

func TestInteriorPoint_NowSolvesCorrectly(t *testing.T) {
	// min -x1-2x2 s.t. x1+x2<=4, x1<=3, x2<=3 -> optimum x=[1,3], obj=-7.
	x, obj, err := InteriorPoint([]float64{-1, -2}, [][]float64{{1, 1}, {1, 0}, {0, 1}}, []float64{4, 3, 3})
	if err != nil {
		t.Fatalf("InteriorPoint errored on a solvable LP: %v", err)
	}
	if math.IsNaN(obj) || math.Abs(obj-(-7)) > 1e-6 {
		t.Errorf("InteriorPoint obj=%v want -7 (x=%v) [was NaN/garbage pre-fix]", obj, x)
	}

	// Infeasible LP -> honest error (delegated to SimplexMethod's guard).
	if _, _, err := InteriorPoint([]float64{1}, [][]float64{{-1}, {1}}, []float64{-10, 5}); err == nil {
		t.Error("InteriorPoint on infeasible LP returned nil error")
	}
}
