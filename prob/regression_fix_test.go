package prob

import (
	"math"
	"testing"
)

// Regression for the LinearRegression cancellation fix and the ClampProbability
// NaN guard.

func TestLinearRegression_LargeMagnitudeStable(t *testing.T) {
	// y = 2*(x - 1e8), exactly linear: slope=2, R^2=1.
	x := []float64{1e8 + 1, 1e8 + 2, 1e8 + 3, 1e8 + 4}
	y := []float64{2, 4, 6, 8}
	slope, _, r2 := LinearRegression(x, y)
	if math.Abs(slope-2) > 1e-9 {
		t.Errorf("LinearRegression slope=%.15g want 2 (was 1.5625 pre-fix)", slope)
	}
	if math.Abs(r2-1) > 1e-9 {
		t.Errorf("LinearRegression R^2=%.15g want 1", r2)
	}
}

func TestLinearRegression_KnownValues(t *testing.T) {
	slope, intercept, r2 := LinearRegression([]float64{1, 2, 3, 4, 5}, []float64{2, 4, 6, 8, 10})
	if math.Abs(slope-2) > 1e-12 || math.Abs(intercept) > 1e-12 || math.Abs(r2-1) > 1e-12 {
		t.Errorf("got slope=%v intercept=%v R2=%v; want 2,0,1", slope, intercept, r2)
	}
	// Offset line y = 3x + 7.
	s2, i2, _ := LinearRegression([]float64{0, 1, 2, 3}, []float64{7, 10, 13, 16})
	if math.Abs(s2-3) > 1e-12 || math.Abs(i2-7) > 1e-12 {
		t.Errorf("got slope=%v intercept=%v; want 3,7", s2, i2)
	}
}

func TestClampProbability_NaNNeutralized(t *testing.T) {
	if got := ClampProbability(math.NaN()); got != MinProb {
		t.Errorf("ClampProbability(NaN)=%v want MinProb=%v", got, MinProb)
	}
	// Unchanged behavior elsewhere.
	for _, tc := range []struct{ in, want float64 }{
		{0.5, 0.5}, {-1, MinProb}, {2, MaxProb}, {MinProb, MinProb}, {MaxProb, MaxProb},
	} {
		if got := ClampProbability(tc.in); got != tc.want {
			t.Errorf("ClampProbability(%v)=%v want %v", tc.in, got, tc.want)
		}
	}
}
