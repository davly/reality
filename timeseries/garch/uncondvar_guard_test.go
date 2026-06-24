package garch

import (
	"math"
	"testing"
)

// TestModel_NonFiniteUncondVar_FallbackInSimulateAndForecast pins the fix.
// Validate() does not check UncondVar, so a Model can pass with UncondVar=NaN/Inf.
// Filter already falls back to the implied Omega/(1-Alpha-Beta); Simulate and
// ForecastVariance now do too, instead of propagating NaN/Inf through the path.
func TestModel_NonFiniteUncondVar_FallbackInSimulateAndForecast(t *testing.T) {
	m := Model{Omega: 1e-5, Alpha: 0.1, Beta: 0.85, UncondVar: math.NaN()}
	if err := m.Validate(); err != nil {
		t.Fatalf("Model with NaN UncondVar should still Validate: %v", err)
	}

	fc, err := m.ForecastVariance(1e-4, 1e-4, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range fc {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("ForecastVariance[%d] = %v (NaN UncondVar), want finite", i, v)
		}
	}

	shocks := make([]float64, 10)
	for i := range shocks {
		shocks[i] = 0.1
	}
	eps := make([]float64, 10)
	sigma2 := make([]float64, 10)
	if err := m.Simulate(shocks, eps, sigma2); err != nil {
		t.Fatal(err)
	}
	for i := range sigma2 {
		if math.IsNaN(sigma2[i]) || math.IsInf(sigma2[i], 0) {
			t.Errorf("Simulate sigma2[%d] = %v (NaN UncondVar), want finite", i, sigma2[i])
		}
	}
}
