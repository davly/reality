package copula

import (
	"math"
	"testing"
)

// =========================================================================
// KendallTau — closed-form sanity checks
// =========================================================================

func TestKendallTau_PerfectConcordance(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	y := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
	tau, err := KendallTau(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(tau-1.0) > 1e-12 {
		t.Errorf("perfect concordance: tau = %v, want 1.0", tau)
	}
}

func TestKendallTau_PerfectDiscordance(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	y := []float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	tau, err := KendallTau(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(tau-(-1.0)) > 1e-12 {
		t.Errorf("perfect discordance: tau = %v, want -1.0", tau)
	}
}

func TestKendallTau_AllTies(t *testing.T) {
	x := []float64{1, 1, 1, 1, 1}
	y := []float64{2, 2, 2, 2, 2}
	tau, err := KendallTau(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if tau != 0 {
		t.Errorf("all ties: tau = %v, want 0", tau)
	}
}

func TestKendallTau_LengthMismatch(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{1, 2}
	if _, err := KendallTau(x, y); err == nil {
		t.Error("length mismatch should error")
	}
}

func TestKendallTau_TooFewSamples(t *testing.T) {
	tau, err := KendallTau([]float64{}, []float64{})
	if err != nil {
		t.Fatal(err)
	}
	if tau != 0 {
		t.Errorf("empty: tau = %v, want 0", tau)
	}
	tau, err = KendallTau([]float64{1}, []float64{2})
	if err != nil {
		t.Fatal(err)
	}
	if tau != 0 {
		t.Errorf("n=1: tau = %v, want 0", tau)
	}
}

func TestKendallTau_DoesNotMutate(t *testing.T) {
	x := []float64{3, 1, 4, 1, 5, 9, 2, 6}
	y := []float64{2, 7, 1, 8, 2, 8, 1, 8}
	wantX := append([]float64(nil), x...)
	wantY := append([]float64(nil), y...)
	_, err := KendallTau(x, y)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x {
		if x[i] != wantX[i] || y[i] != wantY[i] {
			t.Errorf("input mutated at [%d]", i)
		}
	}
}

// =========================================================================
// R80b cross-substrate output parity with RubberDuck CopulaModels.cs
// =========================================================================
//
// These tests replicate hand-computed cases from
// RubberDuck.Core.Analysis.CopulaModels.KendallTau (lines 143-165) and
// CopulaModelsTests.cs.  Tolerance: ≤1e-12 (exact for Kendall-tau,
// which is a ratio of integers so the only float64 rounding is the
// final division by `pairs`).

func TestCrossSubstratePrecision_RubberDuck_KendallTau_KnownPair(t *testing.T) {
	// 5-element example with hand-counted concordant=8, discordant=2,
	// pairs = 5*4/2 = 10 -> tau = (8 - 2) / 10 = 0.6.
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{1, 3, 2, 5, 4}
	// Pairs:
	//   (0,1): dx=+, dy=+ -> concordant
	//   (0,2): dx=+, dy=+ -> concordant
	//   (0,3): dx=+, dy=+ -> concordant
	//   (0,4): dx=+, dy=+ -> concordant
	//   (1,2): dx=+, dy=- -> discordant
	//   (1,3): dx=+, dy=+ -> concordant
	//   (1,4): dx=+, dy=+ -> concordant
	//   (2,3): dx=+, dy=+ -> concordant
	//   (2,4): dx=+, dy=+ -> concordant
	//   (3,4): dx=+, dy=- -> discordant
	// concordant = 8, discordant = 2 -> tau = 0.6.
	tau, err := KendallTau(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(tau-0.6) > 1e-12 {
		t.Errorf("RubberDuck parity: tau = %v, want 0.6", tau)
	}
}

func TestCrossSubstratePrecision_RubberDuck_KendallTau_AntiCorrelated(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	y := []float64{12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	tau, err := KendallTau(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(tau-(-1.0)) > 1e-12 {
		t.Errorf("anti-correlated: tau = %v, want -1.0", tau)
	}
}

// =========================================================================
// EmpiricalCdf — PIT boundary properties
// =========================================================================

func TestEmpiricalCdf_StrictlyInsideUnitInterval(t *testing.T) {
	data := []float64{3, 1, 4, 1, 5, 9, 2, 6}
	u := EmpiricalCdf(data)
	if len(u) != len(data) {
		t.Fatalf("length: got %d, want %d", len(u), len(data))
	}
	for i, ui := range u {
		if ui <= 0 || ui >= 1 {
			t.Errorf("u[%d] = %v not in (0, 1)", i, ui)
		}
	}
}

func TestEmpiricalCdf_RankPreservation(t *testing.T) {
	data := []float64{3, 1, 4, 5, 9, 2, 6, 7}
	u := EmpiricalCdf(data)
	// Smallest data value should have smallest u.
	for i := 0; i < len(data); i++ {
		for j := 0; j < len(data); j++ {
			if data[i] < data[j] && u[i] >= u[j] {
				t.Errorf("rank inversion at i=%d, j=%d: data %v < %v but u %v >= %v",
					i, j, data[i], data[j], u[i], u[j])
			}
		}
	}
}

func TestEmpiricalCdf_FormulaIsRankOverNPlus1(t *testing.T) {
	data := []float64{10, 20, 30, 40, 50}
	u := EmpiricalCdf(data)
	want := []float64{1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6}
	for i, w := range want {
		if math.Abs(u[i]-w) > 1e-12 {
			t.Errorf("u[%d] = %v, want %v", i, u[i], w)
		}
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestKendallTau_Deterministic(t *testing.T) {
	x := []float64{0.5, 0.1, 0.9, 0.3, 0.7, 0.2}
	y := []float64{0.4, 0.2, 0.8, 0.4, 0.6, 0.1}
	a, _ := KendallTau(x, y)
	b, _ := KendallTau(x, y)
	if a != b {
		t.Errorf("non-deterministic: %v vs %v", a, b)
	}
}
