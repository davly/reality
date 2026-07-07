package optim

import (
	"math"
	"testing"
)

// Regression for the SimplexMethod feasibility guard. Without Phase I the
// all-slack origin can be infeasible; the method used to return an infeasible
// point with a nil error. It now returns the honest error its docstring promises.

func TestSimplex_Infeasible_NowErrors(t *testing.T) {
	// min x1 s.t. -x1 <= -10 (x1 >= 10) AND x1 <= 5 : empty feasible region.
	if _, _, err := SimplexMethod([]float64{1}, [][]float64{{-1}, {1}}, []float64{-10, 5}); err == nil {
		t.Error("infeasible LP returned nil error; want infeasibility error")
	}

	// Feasible LP whose ORIGIN is infeasible: min x1+x2 s.t. x1+x2>=4, x1<=3, x2<=3.
	// (No Phase I: must NOT return the infeasible point [0,0] with nil error.)
	x, _, err := SimplexMethod([]float64{1, 1}, [][]float64{{-1, -1}, {1, 0}, {0, 1}}, []float64{-4, 3, 3})
	if err == nil {
		t.Errorf("origin-infeasible LP returned nil error with x=%v; want honest error (Phase I not implemented)", x)
	}
}

func TestSimplex_FeasibleStillSolves(t *testing.T) {
	// Origin-feasible LP must still solve correctly: min -x1 s.t. x1<=5 -> x=[5], obj=-5.
	x, obj, err := SimplexMethod([]float64{-1}, [][]float64{{1}}, []float64{5})
	if err != nil {
		t.Fatalf("feasible LP errored: %v", err)
	}
	if math.Abs(x[0]-5) > 1e-9 || math.Abs(obj-(-5)) > 1e-9 {
		t.Errorf("got x=%v obj=%v; want [5], -5", x, obj)
	}
}
