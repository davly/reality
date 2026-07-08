package linalg

import (
	"math"
	"testing"
)

// Regression for the PearsonCorrelation cancellation fix. The old one-pass
// sum-of-squares form returned 0.79 for perfectly-correlated timestamp-scale
// data (true 1.0) and NaN for near-constant large-magnitude data.
//
// Named _Regression to coexist with the sibling TestPearsonCorrelation_
// LargeMagnitudeStable in correlation_stable_test.go (both pin the same fix
// from independent regression suites; kept distinct to lose neither).

func TestPearsonCorrelation_LargeMagnitudeStable_Regression(t *testing.T) {
	// x = 1e8 + y -> perfect positive correlation, true r = 1.0.
	got := PearsonCorrelation([]float64{1e8 + 1, 1e8 + 2, 1e8 + 3, 1e8 + 4}, []float64{1, 2, 3, 4})
	if math.Abs(got-1) > 1e-12 {
		t.Errorf("PearsonCorrelation(1e8+y, y)=%.15g want 1.0", got)
	}
	// Near-constant large-magnitude data must stay finite in [-1,1] (was NaN).
	r := PearsonCorrelation([]float64{1e9, 1e9, 1e9, 1e9 + 1, 1e9 + 1}, []float64{1, 2, 3, 4, 5})
	if math.IsNaN(r) || r < -1 || r > 1 {
		t.Errorf("PearsonCorrelation(near-const large) = %v; want finite in [-1,1]", r)
	}
}

func TestPearsonCorrelation_KnownValues(t *testing.T) {
	if got := PearsonCorrelation([]float64{1, 2, 3, 4, 5}, []float64{2, 4, 6, 8, 10}); math.Abs(got-1) > 1e-12 {
		t.Errorf("perfect positive r=%v want 1", got)
	}
	if got := PearsonCorrelation([]float64{1, 2, 3, 4, 5}, []float64{10, 8, 6, 4, 2}); math.Abs(got+1) > 1e-12 {
		t.Errorf("perfect negative r=%v want -1", got)
	}
	if got := PearsonCorrelation([]float64{1, 2, 3}, []float64{5, 5, 5}); got != 0 {
		t.Errorf("zero-variance y r=%v want 0", got)
	}
}
