package optim

import (
	"math"
	"testing"
)

// TestSimplexMethod_RejectsNegativeB pins the fix for b<0 constraints. The old
// code negated the row to force b>0, which turns A_i x <= b_i into A_i x >= b_i
// and silently solves a DIFFERENT LP (it returned x=[0,0], val=0 with nil error
// for min x1+x2 s.t. x1+x2>=1 -- infeasible and wrong). The single-phase simplex
// cannot handle b<0, so it must now return an error.
func TestSimplexMethod_RejectsNegativeB(t *testing.T) {
	// min x1+x2 s.t. x1+x2>=1, encoded as -x1-x2 <= -1 (b<0).
	if _, _, err := SimplexMethod([]float64{1, 1}, [][]float64{{-1, -1}}, []float64{-1}); err == nil {
		t.Error("SimplexMethod with b<0 returned nil error; must reject (old bug silently solved the wrong LP)")
	}
	// max x1 s.t. 2<=x1<=5, encoded as A=[[1],[-1]], b=[5,-2] (one row has b<0).
	if _, _, err := SimplexMethod([]float64{-1}, [][]float64{{1}, {-1}}, []float64{5, -2}); err == nil {
		t.Error("SimplexMethod with a b<0 row returned nil error; must reject")
	}
	// A valid b>=0 LP must still solve: min -x1 s.t. x1<=5 -> x=[5], obj=-5.
	x, val, err := SimplexMethod([]float64{-1}, [][]float64{{1}}, []float64{5})
	if err != nil {
		t.Fatalf("valid b>=0 LP errored: %v", err)
	}
	if math.Abs(x[0]-5) > 1e-9 || math.Abs(val+5) > 1e-9 {
		t.Errorf("min -x1 s.t. x1<=5: got x=%v val=%v, want x=[5] val=-5", x, val)
	}
}
