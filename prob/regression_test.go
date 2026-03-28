package prob

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

func TestLinearRegression_PerfectLine(t *testing.T) {
	// y = 2x + 1
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{3, 5, 7, 9, 11}
	slope, intercept, r2 := LinearRegression(x, y)
	if math.Abs(slope-2) > 1e-10 {
		t.Errorf("slope = %v, want 2", slope)
	}
	if math.Abs(intercept-1) > 1e-10 {
		t.Errorf("intercept = %v, want 1", intercept)
	}
	if math.Abs(r2-1) > 1e-10 {
		t.Errorf("R^2 = %v, want 1", r2)
	}
}

func TestLinearRegression_Horizontal(t *testing.T) {
	// y = 5 (constant)
	x := []float64{1, 2, 3, 4}
	y := []float64{5, 5, 5, 5}
	slope, intercept, r2 := LinearRegression(x, y)
	if math.Abs(slope) > 1e-10 {
		t.Errorf("slope = %v, want 0", slope)
	}
	if math.Abs(intercept-5) > 1e-10 {
		t.Errorf("intercept = %v, want 5", intercept)
	}
	if math.Abs(r2-1) > 1e-10 {
		t.Errorf("R^2 = %v, want 1 (perfect trivial fit)", r2)
	}
}

func TestLinearRegression_NegativeSlope(t *testing.T) {
	// y = -3x + 10
	x := []float64{0, 1, 2, 3}
	y := []float64{10, 7, 4, 1}
	slope, intercept, r2 := LinearRegression(x, y)
	if math.Abs(slope-(-3)) > 1e-10 {
		t.Errorf("slope = %v, want -3", slope)
	}
	if math.Abs(intercept-10) > 1e-10 {
		t.Errorf("intercept = %v, want 10", intercept)
	}
	if math.Abs(r2-1) > 1e-10 {
		t.Errorf("R^2 = %v, want 1", r2)
	}
}

func TestLinearRegression_TwoPoints(t *testing.T) {
	x := []float64{0, 10}
	y := []float64{5, 25}
	slope, intercept, r2 := LinearRegression(x, y)
	if math.Abs(slope-2) > 1e-10 {
		t.Errorf("slope = %v, want 2", slope)
	}
	if math.Abs(intercept-5) > 1e-10 {
		t.Errorf("intercept = %v, want 5", intercept)
	}
	if math.Abs(r2-1) > 1e-10 {
		t.Errorf("R^2 = %v, want 1", r2)
	}
}

func TestLinearRegression_WithNoise(t *testing.T) {
	// y = x with noise: R^2 < 1 but > 0
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	y := []float64{1.1, 1.8, 3.2, 3.9, 5.1, 6.2, 6.8, 8.1, 9.0, 10.1}
	slope, intercept, r2 := LinearRegression(x, y)
	// Slope should be near 1.
	if math.Abs(slope-1) > 0.1 {
		t.Errorf("slope = %v, want ~1", slope)
	}
	// Intercept should be near 0.
	if math.Abs(intercept) > 0.3 {
		t.Errorf("intercept = %v, want ~0", intercept)
	}
	// R^2 should be very high but not perfect.
	if r2 < 0.99 || r2 > 1.0 {
		t.Errorf("R^2 = %v, want ~0.99+", r2)
	}
}

func TestLinearRegression_IdenticalX(t *testing.T) {
	// All x identical — undefined slope.
	x := []float64{5, 5, 5}
	y := []float64{1, 2, 3}
	slope, intercept, r2 := LinearRegression(x, y)
	if !math.IsNaN(slope) || !math.IsNaN(intercept) || !math.IsNaN(r2) {
		t.Errorf("expected NaN for identical x, got slope=%v intercept=%v r2=%v",
			slope, intercept, r2)
	}
}

func TestLinearRegression_TooFew(t *testing.T) {
	slope, intercept, r2 := LinearRegression([]float64{1}, []float64{2})
	if !math.IsNaN(slope) || !math.IsNaN(intercept) || !math.IsNaN(r2) {
		t.Errorf("expected NaN for n<2, got slope=%v intercept=%v r2=%v",
			slope, intercept, r2)
	}
}

func TestLinearRegression_LengthMismatch(t *testing.T) {
	slope, _, _ := LinearRegression([]float64{1, 2}, []float64{1})
	if !math.IsNaN(slope) {
		t.Errorf("expected NaN for mismatched lengths, got slope=%v", slope)
	}
}

func TestLinearRegression_ZeroIntercept(t *testing.T) {
	// y = 3x through origin
	x := []float64{-2, -1, 0, 1, 2}
	y := []float64{-6, -3, 0, 3, 6}
	slope, intercept, r2 := LinearRegression(x, y)
	if math.Abs(slope-3) > 1e-10 {
		t.Errorf("slope = %v, want 3", slope)
	}
	if math.Abs(intercept) > 1e-10 {
		t.Errorf("intercept = %v, want 0", intercept)
	}
	if math.Abs(r2-1) > 1e-10 {
		t.Errorf("R^2 = %v, want 1", r2)
	}
}

func TestLinearRegression_LargeValues(t *testing.T) {
	// y = 1000x + 500000
	x := []float64{100, 200, 300, 400, 500}
	y := []float64{600000, 700000, 800000, 900000, 1000000}
	slope, intercept, r2 := LinearRegression(x, y)
	if math.Abs(slope-1000) > 1e-6 {
		t.Errorf("slope = %v, want 1000", slope)
	}
	if math.Abs(intercept-500000) > 1e-4 {
		t.Errorf("intercept = %v, want 500000", intercept)
	}
	if math.Abs(r2-1) > 1e-10 {
		t.Errorf("R^2 = %v, want 1", r2)
	}
}

// ---------------------------------------------------------------------------
// BenjaminiHochberg
// ---------------------------------------------------------------------------

func TestBH_AllSignificant(t *testing.T) {
	// All p-values well below alpha.
	pValues := []float64{0.001, 0.002, 0.003}
	result := BenjaminiHochberg(pValues, 0.05)
	for i, r := range result {
		if !r {
			t.Errorf("BH: index %d not rejected but p=%v", i, pValues[i])
		}
	}
}

func TestBH_NoneSignificant(t *testing.T) {
	// All p-values above alpha.
	pValues := []float64{0.5, 0.6, 0.7}
	result := BenjaminiHochberg(pValues, 0.05)
	for i, r := range result {
		if r {
			t.Errorf("BH: index %d rejected but p=%v", i, pValues[i])
		}
	}
}

func TestBH_Mixed(t *testing.T) {
	// Classic example: some should be rejected, some not.
	pValues := []float64{0.01, 0.04, 0.03, 0.20, 0.50}
	result := BenjaminiHochberg(pValues, 0.05)
	// After sorting: 0.01, 0.03, 0.04, 0.20, 0.50
	// Thresholds: 1/5*0.05=0.01, 2/5*0.05=0.02, 3/5*0.05=0.03, ...
	// 0.01 <= 0.01 => yes, 0.03 <= 0.02 => no. Only largest k: k=1.
	// So reject indices with p=0.01 (index 0).
	if !result[0] {
		t.Errorf("BH: index 0 (p=0.01) should be rejected")
	}
	// The rest should not be rejected at this conservative threshold.
	// Actually with BH: sorted: 0.01, 0.03, 0.04, 0.20, 0.50
	// thresholds: 0.01, 0.02, 0.03, 0.04, 0.05
	// 0.01 <= 0.01 (k=1): yes
	// 0.03 <= 0.02 (k=2): no
	// 0.04 <= 0.03 (k=3): no
	// 0.20 <= 0.04 (k=4): no
	// 0.50 <= 0.05 (k=5): no
	// largest k = 1. Reject all with rank <= 1, so only p=0.01.
}

func TestBH_SingleValue(t *testing.T) {
	result := BenjaminiHochberg([]float64{0.03}, 0.05)
	if !result[0] {
		t.Errorf("BH: single p=0.03 should be rejected at alpha=0.05")
	}
}

func TestBH_SingleValue_NotRejected(t *testing.T) {
	result := BenjaminiHochberg([]float64{0.10}, 0.05)
	if result[0] {
		t.Errorf("BH: single p=0.10 should not be rejected at alpha=0.05")
	}
}

func TestBH_Empty(t *testing.T) {
	result := BenjaminiHochberg(nil, 0.05)
	if result != nil {
		t.Errorf("BH: nil input should return nil, got %v", result)
	}
}

func TestBH_StepDown(t *testing.T) {
	// If the largest k is found, all p-values with rank <= k are rejected.
	pValues := []float64{0.001, 0.01, 0.03, 0.04, 0.50}
	result := BenjaminiHochberg(pValues, 0.05)
	// Sorted: 0.001, 0.01, 0.03, 0.04, 0.50
	// Thresholds: 0.01, 0.02, 0.03, 0.04, 0.05
	// 0.001 <= 0.01: yes (k=1)
	// 0.01  <= 0.02: yes (k=2)
	// 0.03  <= 0.03: yes (k=3)
	// 0.04  <= 0.04: yes (k=4)
	// 0.50  <= 0.05: no
	// Largest k = 4. Reject ranks 1-4.
	if !result[0] || !result[1] || !result[2] || !result[3] {
		t.Errorf("BH: first 4 should be rejected, got %v", result)
	}
	if result[4] {
		t.Errorf("BH: index 4 (p=0.50) should not be rejected")
	}
}

func TestBH_OrderPreserved(t *testing.T) {
	// Results should correspond to original order, not sorted order.
	pValues := []float64{0.50, 0.001, 0.10, 0.002}
	result := BenjaminiHochberg(pValues, 0.05)
	// Sorted: 0.001 (idx 1), 0.002 (idx 3), 0.10 (idx 2), 0.50 (idx 0)
	// Thresholds: 0.0125, 0.025, 0.0375, 0.05
	// 0.001 <= 0.0125: yes (k=1)
	// 0.002 <= 0.025: yes (k=2)
	// 0.10  <= 0.0375: no
	// Largest k = 2.
	if result[0] {
		t.Errorf("BH: index 0 (p=0.50) should NOT be rejected")
	}
	if !result[1] {
		t.Errorf("BH: index 1 (p=0.001) should be rejected")
	}
	if result[2] {
		t.Errorf("BH: index 2 (p=0.10) should NOT be rejected")
	}
	if !result[3] {
		t.Errorf("BH: index 3 (p=0.002) should be rejected")
	}
}

func TestBH_TiedPValues(t *testing.T) {
	pValues := []float64{0.01, 0.01, 0.01}
	result := BenjaminiHochberg(pValues, 0.05)
	// All p-values = 0.01.
	// Thresholds: 1/3*0.05=0.0167, 2/3*0.05=0.033, 3/3*0.05=0.05
	// All 0.01 <= their threshold => all rejected.
	for i, r := range result {
		if !r {
			t.Errorf("BH: index %d with p=0.01 should be rejected", i)
		}
	}
}
