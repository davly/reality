package prob

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// ExponentialSmoothing
// ---------------------------------------------------------------------------

func TestExponentialSmoothing_BasicTrend(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	out := make([]float64, len(data))
	ExponentialSmoothing(data, 0.5, out)

	// First value is always data[0].
	if out[0] != 1 {
		t.Errorf("out[0] = %f, want 1", out[0])
	}
	// Each subsequent value should be between data[t] and out[t-1].
	for i := 1; i < len(data); i++ {
		if out[i] < out[i-1] {
			t.Errorf("out[%d] = %f < out[%d] = %f for increasing data", i, out[i], i-1, out[i-1])
		}
	}
}

func TestExponentialSmoothing_Alpha1(t *testing.T) {
	data := []float64{10, 20, 30, 40, 50}
	out := make([]float64, len(data))
	ExponentialSmoothing(data, 1.0, out)

	// Alpha=1 means no smoothing: output equals input.
	for i, v := range data {
		if out[i] != v {
			t.Errorf("alpha=1: out[%d] = %f, want %f", i, out[i], v)
		}
	}
}

func TestExponentialSmoothing_AlphaNearZero(t *testing.T) {
	data := []float64{100, 200, 300, 400, 500}
	out := make([]float64, len(data))
	ExponentialSmoothing(data, 0.01, out)

	// Very low alpha: output stays close to first value.
	for i := 1; i < len(data); i++ {
		if out[i] > data[0]+50 {
			t.Errorf("alpha≈0: out[%d] = %f, should stay near %f", i, out[i], data[0])
		}
	}
}

func TestExponentialSmoothing_ConstantData(t *testing.T) {
	data := []float64{5, 5, 5, 5, 5}
	out := make([]float64, len(data))
	ExponentialSmoothing(data, 0.3, out)

	for i, v := range out {
		if math.Abs(v-5) > 1e-12 {
			t.Errorf("constant data: out[%d] = %f, want 5", i, v)
		}
	}
}

func TestExponentialSmoothing_SingleElement(t *testing.T) {
	data := []float64{42}
	out := make([]float64, 1)
	ExponentialSmoothing(data, 0.5, out)
	if out[0] != 42 {
		t.Errorf("single element: out[0] = %f, want 42", out[0])
	}
}

func TestExponentialSmoothing_Empty(t *testing.T) {
	var data []float64
	var out []float64
	// Should not panic.
	ExponentialSmoothing(data, 0.5, out)
}

func TestExponentialSmoothing_AlphaClamping(t *testing.T) {
	data := []float64{1, 2, 3}
	out := make([]float64, 3)

	// Negative alpha should be clamped to 0.01.
	ExponentialSmoothing(data, -0.5, out)
	if out[0] != 1 {
		t.Errorf("negative alpha: out[0] = %f, want 1", out[0])
	}

	// Alpha > 1 should be clamped to 1.0.
	ExponentialSmoothing(data, 2.0, out)
	for i, v := range data {
		if out[i] != v {
			t.Errorf("alpha>1 clamped to 1: out[%d] = %f, want %f", i, out[i], v)
		}
	}
}

func TestExponentialSmoothing_KnownValues(t *testing.T) {
	// Manual calculation with alpha=0.5:
	// s[0] = 10
	// s[1] = 0.5*20 + 0.5*10 = 15
	// s[2] = 0.5*30 + 0.5*15 = 22.5
	data := []float64{10, 20, 30}
	out := make([]float64, 3)
	ExponentialSmoothing(data, 0.5, out)

	expected := []float64{10, 15, 22.5}
	for i, e := range expected {
		if math.Abs(out[i]-e) > 1e-12 {
			t.Errorf("out[%d] = %f, want %f", i, out[i], e)
		}
	}
}

// ---------------------------------------------------------------------------
// HoltLinear
// ---------------------------------------------------------------------------

func TestHoltLinear_LinearTrend(t *testing.T) {
	// Perfectly linear data: y = 10 + 5*t
	data := []float64{10, 15, 20, 25, 30}
	horizon := 3
	out := make([]float64, len(data)+horizon)
	HoltLinear(data, 0.8, 0.8, horizon, out)

	// Forecast should continue the trend approximately.
	for h := 1; h <= horizon; h++ {
		expected := 30 + float64(h)*5
		actual := out[len(data)-1+h]
		if math.Abs(actual-expected) > 5 {
			t.Errorf("forecast[%d] = %f, want ~%f (within 5)", h, actual, expected)
		}
	}
}

func TestHoltLinear_ConstantData(t *testing.T) {
	data := []float64{10, 10, 10, 10, 10}
	horizon := 2
	out := make([]float64, len(data)+horizon)
	HoltLinear(data, 0.5, 0.5, horizon, out)

	// Constant data with no trend: forecast should be near 10.
	for h := 1; h <= horizon; h++ {
		actual := out[len(data)-1+h]
		if math.Abs(actual-10) > 2 {
			t.Errorf("constant forecast[%d] = %f, want ~10", h, actual)
		}
	}
}

func TestHoltLinear_SinglePoint(t *testing.T) {
	data := []float64{42}
	out := make([]float64, 4)
	HoltLinear(data, 0.5, 0.5, 3, out)

	if out[0] != 42 {
		t.Errorf("single point: out[0] = %f, want 42", out[0])
	}
	// With single point, trend=0, forecast should be flat.
	for h := 1; h <= 3; h++ {
		if math.Abs(out[h]-42) > 1e-12 {
			t.Errorf("single point forecast[%d] = %f, want 42", h, out[h])
		}
	}
}

func TestHoltLinear_ZeroHorizon(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	out := make([]float64, len(data))
	HoltLinear(data, 0.5, 0.5, 0, out)

	// No forecast, just smoothed values. Should not panic.
	if out[0] != 1 {
		t.Errorf("out[0] = %f, want 1", out[0])
	}
}

func TestHoltLinear_Empty(t *testing.T) {
	var data []float64
	var out []float64
	// Should not panic.
	HoltLinear(data, 0.5, 0.5, 3, out)
}

func TestHoltLinear_AlphaBetaClamping(t *testing.T) {
	data := []float64{1, 2, 3}
	out := make([]float64, 5)

	// Negative alpha/beta clamped to 0.01.
	HoltLinear(data, -1, -1, 2, out)
	if math.IsNaN(out[0]) {
		t.Error("negative alpha/beta produced NaN")
	}

	// Alpha/beta > 1 clamped to 1.
	HoltLinear(data, 5, 5, 2, out)
	if math.IsNaN(out[0]) {
		t.Error("alpha/beta > 1 produced NaN")
	}
}

func TestHoltLinear_NegativeHorizon(t *testing.T) {
	data := []float64{1, 2, 3}
	out := make([]float64, 3)
	// Negative horizon should be clamped to 0. Should not panic.
	HoltLinear(data, 0.5, 0.5, -5, out)
}

func TestHoltLinear_DecreasingTrend(t *testing.T) {
	data := []float64{50, 45, 40, 35, 30}
	horizon := 2
	out := make([]float64, len(data)+horizon)
	HoltLinear(data, 0.9, 0.9, horizon, out)

	// Forecast should continue downward.
	for h := 1; h <= horizon; h++ {
		actual := out[len(data)-1+h]
		if actual > 30 {
			t.Errorf("decreasing forecast[%d] = %f, should be < 30", h, actual)
		}
	}
}

// ---------------------------------------------------------------------------
// ARIMA
// ---------------------------------------------------------------------------

func TestARIMA_PureAR1(t *testing.T) {
	// Generate AR(1) process: x[t] = 0.7 * x[t-1] + noise
	data := make([]float64, 100)
	data[0] = 0
	for i := 1; i < len(data); i++ {
		data[i] = 0.7*data[i-1] + float64(i%7-3)*0.1
	}

	coeffs, err := ARIMA(data, 1, 0, 0)
	if err != nil {
		t.Fatalf("ARIMA(1,0,0) error: %v", err)
	}
	if len(coeffs) != 1 {
		t.Fatalf("expected 1 coefficient, got %d", len(coeffs))
	}
	// AR(1) coefficient should be close to 0.7.
	if math.Abs(coeffs[0]-0.7) > 0.3 {
		t.Errorf("AR(1) coefficient = %f, want ~0.7 (within 0.3)", coeffs[0])
	}
}

func TestARIMA_WithDifferencing(t *testing.T) {
	// Linear trend: x[t] = t. After d=1 differencing, becomes constant.
	data := make([]float64, 50)
	for i := range data {
		data[i] = float64(i)
	}

	coeffs, err := ARIMA(data, 1, 1, 0)
	if err != nil {
		t.Fatalf("ARIMA(1,1,0) error: %v", err)
	}
	if len(coeffs) != 1 {
		t.Fatalf("expected 1 coefficient, got %d", len(coeffs))
	}
}

func TestARIMA_PureMA1(t *testing.T) {
	// Generate data and fit MA(1).
	data := make([]float64, 100)
	for i := range data {
		data[i] = float64(i%5-2) * 0.5
	}

	coeffs, err := ARIMA(data, 0, 0, 1)
	if err != nil {
		t.Fatalf("ARIMA(0,0,1) error: %v", err)
	}
	if len(coeffs) != 1 {
		t.Fatalf("expected 1 coefficient, got %d", len(coeffs))
	}
}

func TestARIMA_ZeroOrder(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	coeffs, err := ARIMA(data, 0, 0, 0)
	if err != nil {
		t.Fatalf("ARIMA(0,0,0) error: %v", err)
	}
	if len(coeffs) != 0 {
		t.Errorf("expected 0 coefficients, got %d", len(coeffs))
	}
}

func TestARIMA_NegativeParams(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	_, err := ARIMA(data, -1, 0, 0)
	if err == nil {
		t.Error("expected error for negative p")
	}

	_, err = ARIMA(data, 0, -1, 0)
	if err == nil {
		t.Error("expected error for negative d")
	}

	_, err = ARIMA(data, 0, 0, -1)
	if err == nil {
		t.Error("expected error for negative q")
	}
}

func TestARIMA_DataTooShort(t *testing.T) {
	data := []float64{1, 2}
	_, err := ARIMA(data, 3, 0, 0)
	if err == nil {
		t.Error("expected error for data too short for AR order")
	}
}

func TestARIMA_DataTooShortAfterDifferencing(t *testing.T) {
	data := []float64{1, 2, 3}
	_, err := ARIMA(data, 1, 5, 0)
	if err == nil {
		t.Error("expected error for data too short after differencing")
	}
}

func TestARIMA_ARMA11(t *testing.T) {
	// ARMA(1,1) on synthetic data.
	data := make([]float64, 200)
	data[0] = 1
	for i := 1; i < len(data); i++ {
		data[i] = 0.5*data[i-1] + float64(i%11-5)*0.1
	}

	coeffs, err := ARIMA(data, 1, 0, 1)
	if err != nil {
		t.Fatalf("ARIMA(1,0,1) error: %v", err)
	}
	if len(coeffs) != 2 {
		t.Fatalf("expected 2 coefficients (1 AR + 1 MA), got %d", len(coeffs))
	}
}

func TestARIMA_HigherOrder(t *testing.T) {
	data := make([]float64, 200)
	for i := range data {
		data[i] = math.Sin(float64(i)*0.1) + float64(i)*0.01
	}

	coeffs, err := ARIMA(data, 3, 1, 2)
	if err != nil {
		t.Fatalf("ARIMA(3,1,2) error: %v", err)
	}
	if len(coeffs) != 5 {
		t.Fatalf("expected 5 coefficients (3 AR + 2 MA), got %d", len(coeffs))
	}
}

func TestARIMA_ConstantData(t *testing.T) {
	data := make([]float64, 50)
	for i := range data {
		data[i] = 42
	}
	coeffs, err := ARIMA(data, 1, 0, 0)
	if err != nil {
		t.Fatalf("ARIMA on constant data: %v", err)
	}
	// Constant data has zero variance, AR coefficient should be 0.
	if len(coeffs) != 1 {
		t.Fatalf("expected 1 coefficient, got %d", len(coeffs))
	}
}

func TestARIMA_LargeDataset(t *testing.T) {
	data := make([]float64, 1000)
	for i := range data {
		data[i] = math.Sin(float64(i)*0.05) + float64(i%7)*0.1
	}

	coeffs, err := ARIMA(data, 2, 1, 1)
	if err != nil {
		t.Fatalf("ARIMA on large dataset: %v", err)
	}
	if len(coeffs) != 3 {
		t.Fatalf("expected 3 coefficients, got %d", len(coeffs))
	}
}
