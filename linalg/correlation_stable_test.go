package linalg

import (
	"math"
	"testing"
)

// TestPearsonCorrelation_LargeMagnitudeStable pins the two-pass fix. The old
// single-pass uncentered formula catastrophically cancelled for large-magnitude,
// small-variance data: the discriminant went negative (sqrt -> NaN) or the ratio
// exceeded [-1,1]. The result must always be finite and in [-1,1].
func TestPearsonCorrelation_LargeMagnitudeStable(t *testing.T) {
	// (1) near-constant large-magnitude x -> old code returned NaN.
	n := 400
	x := make([]float64, n)
	y := make([]float64, n)
	for i := range x {
		x[i] = 5e9 + float64(i%2)*1e-6
		y[i] = float64(i)
	}
	if r := PearsonCorrelation(x, y); math.IsNaN(r) || math.IsInf(r, 0) || r < -1 || r > 1 {
		t.Errorf("large-magnitude near-constant: r=%v, want finite in [-1,1]", r)
	}
	// (2) old code returned r=1.4396 (> 1).
	n2 := 500
	x2 := make([]float64, n2)
	y2 := make([]float64, n2)
	for i := range x2 {
		x2[i] = 1e7 + float64(i%3)*1e-6
		y2[i] = 1e7 + float64((i+1)%3)*1e-6
	}
	if r := PearsonCorrelation(x2, y2); math.IsNaN(r) || r < -1 || r > 1 {
		t.Errorf("large-magnitude: r=%v, want in [-1,1]", r)
	}
	// (3) sanity: a perfectly-correlated clean case -> r == 1.
	if r := PearsonCorrelation([]float64{1, 2, 3, 4, 5}, []float64{2, 4, 6, 8, 10}); math.Abs(r-1.0) > 1e-9 {
		t.Errorf("perfectly correlated: r=%v, want 1", r)
	}
	// (4) sanity: perfect anti-correlation -> r == -1.
	if r := PearsonCorrelation([]float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}); math.Abs(r+1.0) > 1e-9 {
		t.Errorf("anti-correlated: r=%v, want -1", r)
	}
}
