package conformal

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// SplitQuantile — closed-form
// =========================================================================

func TestSplitQuantile_SmallExact(t *testing.T) {
	// scores = (1, 2, 3, 4, 5), n = 5, alpha = 0.2.
	// rank = ceil((5+1)*0.8) = ceil(4.8) = 5.
	// Sorted scores -> q = scores[4] = 5.
	scores := []float64{3, 1, 5, 2, 4}
	q, err := SplitQuantile(scores, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	if q != 5.0 {
		t.Errorf("q = %v, want 5", q)
	}
}

func TestSplitQuantile_LargeAlphaReturnsLowQuantile(t *testing.T) {
	scores := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	// alpha = 0.5, n = 10 -> rank = ceil(11*0.5) = 6 -> q = sorted[5] = 0.6
	q, err := SplitQuantile(scores, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(q-0.6) > 1e-12 {
		t.Errorf("q = %v, want 0.6", q)
	}
}

func TestSplitQuantile_TooFewSamplesReturnsInf(t *testing.T) {
	// alpha = 0.05 means rank = ceil((n+1)*0.95).  For n = 5 that's
	// ceil(5.7) = 6 > 5 -> +Inf.
	q, err := SplitQuantile([]float64{1, 2, 3, 4, 5}, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(q, 1) {
		t.Errorf("q = %v, want +Inf for n=5 alpha=0.05", q)
	}
}

func TestSplitQuantile_RejectsBadInputs(t *testing.T) {
	if _, err := SplitQuantile([]float64{1, 2}, 0); err == nil {
		t.Error("alpha=0 should error")
	}
	if _, err := SplitQuantile([]float64{1, 2}, 1); err == nil {
		t.Error("alpha=1 should error")
	}
	if _, err := SplitQuantile(nil, 0.1); err == nil {
		t.Error("empty scores should error")
	}
	if _, err := SplitQuantile([]float64{-1, 1}, 0.1); err == nil {
		t.Error("negative score should error")
	}
	if _, err := SplitQuantile([]float64{math.NaN(), 1}, 0.1); err == nil {
		t.Error("NaN score should error")
	}
}

// =========================================================================
// Coverage simulation — empirical validation of the marginal guarantee
// =========================================================================

// TestSplitInterval_AchievesNominalCoverage runs a Monte Carlo experiment:
// fit a tiny linear model, calibrate with split conformal, evaluate on
// fresh test data, and verify empirical coverage is at least 1 - alpha.
//
// Repeats the experiment R times and uses the mean coverage as a less-noisy
// estimate.  Uses rand.New with a fixed seed for reproducibility.
func TestSplitInterval_AchievesNominalCoverage(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	const (
		nCal   = 500
		nTest  = 1000
		alpha  = 0.1
		repeat = 5
	)
	var totalCovered, totalSeen int
	for r := 0; r < repeat; r++ {
		// True relationship y = 2x + epsilon, epsilon ~ N(0, 1).
		// Predictor: just yhat = 2x (perfect mean, but we'll calibrate
		// non-conformity from the observed residuals).
		predict := func(x float64) float64 { return 2 * x }
		residuals := make([]float64, nCal)
		for i := 0; i < nCal; i++ {
			x := rng.Float64()*10 - 5
			y := 2*x + rng.NormFloat64()
			residuals[i] = math.Abs(y - predict(x))
		}
		for i := 0; i < nTest; i++ {
			x := rng.Float64()*10 - 5
			yTest := 2*x + rng.NormFloat64()
			lo, hi, err := SplitInterval(predict(x), residuals, alpha)
			if err != nil {
				t.Fatal(err)
			}
			if yTest >= lo && yTest <= hi {
				totalCovered++
			}
			totalSeen++
		}
	}
	cov := float64(totalCovered) / float64(totalSeen)
	if cov < 1-alpha-0.02 {
		t.Errorf("empirical coverage = %.4f, want >= %.4f", cov, 1-alpha)
	}
	if cov > 1-alpha+0.05 {
		t.Errorf("empirical coverage = %.4f, want <= %.4f (over-cover)", cov, 1-alpha+0.05)
	}
}

// =========================================================================
// CQR
// =========================================================================

func TestCqrConformityScore_KnownValues(t *testing.T) {
	// y inside [qLo, qHi] -> negative score (over-coverage).
	if got := CqrConformityScore(1.0, 5.0, 3.0); got != 1.0-3.0 {
		// max(1-3, 3-5) = max(-2, -2) = -2
		if got != -2.0 {
			t.Errorf("CqrScore inside = %v, want -2", got)
		}
	}
	// y above qHi -> positive score equal to overshoot.
	if got := CqrConformityScore(1.0, 5.0, 7.0); got != 2.0 {
		t.Errorf("CqrScore above = %v, want 2", got)
	}
	// y below qLo -> positive score equal to undershoot.
	if got := CqrConformityScore(1.0, 5.0, -1.5); got != 2.5 {
		t.Errorf("CqrScore below = %v, want 2.5", got)
	}
}

func TestCqrInterval_BasicShape(t *testing.T) {
	// 100 calibration scores at 0..99; alpha=0.1 -> rank=ceil(101*0.9)=91
	// -> q = 90.  Interval = [qLoHat - 90, qHiHat + 90].
	scores := make([]float64, 100)
	for i := range scores {
		scores[i] = float64(i)
	}
	lo, hi, err := CqrInterval(0.0, 1.0, scores, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(lo-(-90)) > 1e-12 || math.Abs(hi-91) > 1e-12 {
		t.Errorf("CQR interval = [%v, %v], want [-90, 91]", lo, hi)
	}
}

// =========================================================================
// MarginalCoverageBounds
// =========================================================================

func TestMarginalCoverageBounds_SandwichValues(t *testing.T) {
	lo, hi, err := MarginalCoverageBounds(99, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(lo-0.9) > 1e-12 {
		t.Errorf("lo = %v, want 0.9", lo)
	}
	want := 0.9 + 1.0/100.0
	if math.Abs(hi-want) > 1e-12 {
		t.Errorf("hi = %v, want %v", hi, want)
	}
}

func TestMarginalCoverageBounds_RejectsBad(t *testing.T) {
	if _, _, err := MarginalCoverageBounds(0, 0.1); err == nil {
		t.Error("n=0 should error")
	}
	if _, _, err := MarginalCoverageBounds(10, 1.0); err == nil {
		t.Error("alpha=1 should error")
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestSplitQuantile_Deterministic(t *testing.T) {
	scores := []float64{0.5, 0.1, 0.9, 0.3, 0.7, 0.2}
	a, _ := SplitQuantile(scores, 0.2)
	b, _ := SplitQuantile(scores, 0.2)
	if a != b {
		t.Errorf("non-deterministic: %v vs %v", a, b)
	}
	// Calling SplitQuantile must NOT mutate the input scores.
	want := []float64{0.5, 0.1, 0.9, 0.3, 0.7, 0.2}
	for i, w := range want {
		if scores[i] != w {
			t.Errorf("scores[%d] mutated: %v, want %v", i, scores[i], w)
		}
	}
}

// =========================================================================
// SplitIntervalSignedResiduals — convenience wrapper accepting signed input
// =========================================================================

func TestSplitIntervalSignedResiduals_AbsolutesInternally(t *testing.T) {
	// Mixed-sign residuals; the abs-sorted order matters.
	signed := []float64{3.0, -1.0, 2.0, -4.0, 0.5}
	lo, hi, err := SplitIntervalSignedResiduals(10.0, signed, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	// |signed| sorted = [0.5, 1.0, 2.0, 3.0, 4.0]; n=5, alpha=0.2 ->
	// rank = ceil(6*0.8) = 5 -> q = sorted[4] = 4.0 -> [6, 14].
	if math.Abs(lo-6.0) > 1e-12 || math.Abs(hi-14.0) > 1e-12 {
		t.Errorf("interval = [%v, %v], want [6, 14]", lo, hi)
	}
}

func TestSplitIntervalSignedResiduals_DoesNotMutateInput(t *testing.T) {
	signed := []float64{3.0, -1.0, 2.0, -4.0}
	copy_ := []float64{3.0, -1.0, 2.0, -4.0}
	_, _, err := SplitIntervalSignedResiduals(0.0, signed, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range signed {
		if signed[i] != copy_[i] {
			t.Errorf("input mutated at [%d]: %v vs %v", i, signed[i], copy_[i])
		}
	}
}

func TestSplitIntervalSignedResiduals_RejectsBadInputs(t *testing.T) {
	if _, _, err := SplitIntervalSignedResiduals(0, nil, 0.1); err == nil {
		t.Error("nil residuals should error")
	}
	if _, _, err := SplitIntervalSignedResiduals(0, []float64{1}, 0); err == nil {
		t.Error("alpha=0 should error")
	}
	if _, _, err := SplitIntervalSignedResiduals(0, []float64{1}, 1); err == nil {
		t.Error("alpha=1 should error")
	}
	if _, _, err := SplitIntervalSignedResiduals(0, []float64{math.NaN()}, 0.1); err == nil {
		t.Error("NaN residual should error")
	}
}

// =========================================================================
// Cross-substrate-precision parity with FleetWorks C# MathLib
// =========================================================================
//
// These tests replicate the FleetWorks ConformalIntervalTests.cs corpus
// (commit dc63772f, GLIRent.RentalManagement.Tests.AI.MathLib) using
// SplitIntervalSignedResiduals — the canonical cross-substrate
// equivalence point.  Each case in the FW xUnit suite has a
// hand-computed expected (lo, hi); we assert the same values to ≤1e-12,
// which is the load-bearing R80b cross-substrate-precision-coherence
// property for the conformal primitive.

func TestCrossSubstratePrecision_FwCorpus_SymmetricAroundPrediction(t *testing.T) {
	// FW: Conformal_Symmetric_Around_Prediction
	// |residuals| sorted = [0, 0.5, 0.5, 1, 1]; n=5, alpha=0.1 ->
	// rank = ceil(6*0.9) = 6 -> clamps to n=5 -> halfWidth = 1.
	residuals := []float64{-1.0, -0.5, 0, 0.5, 1.0}
	lo, hi, err := SplitIntervalSignedResiduals(10.0, residuals, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(lo-9.0) > 1e-12 {
		t.Errorf("FW parity SymmetricAroundPrediction: lo = %v, want 9", lo)
	}
	if math.Abs(hi-11.0) > 1e-12 {
		t.Errorf("FW parity SymmetricAroundPrediction: hi = %v, want 11", hi)
	}
}

func TestCrossSubstratePrecision_FwCorpus_IndexSelectsNPlus1TimesOneMinusAlpha(t *testing.T) {
	// FW: Conformal_Index_Selects_NPlus1_Times_OneMinusAlpha
	// 9 residuals, alpha=0.1 -> rank = ceil(10*0.9) = 9 -> max abs = 0.9.
	residuals := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	lo, hi, err := SplitIntervalSignedResiduals(0.0, residuals, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(lo-(-0.9)) > 1e-12 {
		t.Errorf("FW parity rank-formula: lo = %v, want -0.9", lo)
	}
	if math.Abs(hi-0.9) > 1e-12 {
		t.Errorf("FW parity rank-formula: hi = %v, want 0.9", hi)
	}
}

func TestCrossSubstratePrecision_FwCorpus_DoesNotMutateCallerArray(t *testing.T) {
	// FW: Conformal_DoesNot_Mutate_CallerArray
	residuals := []float64{3, -1, 2, -4}
	want := []float64{3, -1, 2, -4}
	_, _, err := SplitIntervalSignedResiduals(0.0, residuals, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if residuals[i] != want[i] {
			t.Errorf("residuals[%d] mutated: got %v, want %v", i, residuals[i], want[i])
		}
	}
}

func TestCrossSubstratePrecision_FwCorpus_EmptyCalibrationErrors(t *testing.T) {
	// FW: Conformal_Empty_Calibration_Throws + Conformal_Null_Calibration_Throws
	if _, _, err := SplitIntervalSignedResiduals(0.0, []float64{}, 0.1); err == nil {
		t.Error("empty calibration should error")
	}
	if _, _, err := SplitIntervalSignedResiduals(0.0, nil, 0.1); err == nil {
		t.Error("nil calibration should error")
	}
}

func TestCrossSubstratePrecision_FwCorpus_AlphaOutOfRangeErrors(t *testing.T) {
	// FW: Conformal_Alpha_OutOfRange_Throws
	residuals := []float64{1, 2}
	if _, _, err := SplitIntervalSignedResiduals(0.0, residuals, 0); err == nil {
		t.Error("alpha=0 should error")
	}
	if _, _, err := SplitIntervalSignedResiduals(0.0, residuals, 1); err == nil {
		t.Error("alpha=1 should error")
	}
}

func TestCrossSubstratePrecision_FwCorpus_EmpiricalCoverageAtLeast90Percent(t *testing.T) {
	// FW: Conformal_Empirical_Coverage_AtLeast90Percent
	// We use math/rand with a fixed seed; the FW seed is .NET's System
	// .Random(42) which is a different RNG, so the *exact* covered
	// count won't match.  What matches across substrates is the
	// guarantee — coverage >= 88% of 1000 trials at alpha=0.1.
	rng := rand.New(rand.NewSource(42))
	residuals := make([]float64, 99)
	for i := range residuals {
		residuals[i] = rng.Float64() - 0.5
	}
	covered := 0
	for t := 0; t < 1000; t++ {
		fresh := rng.Float64() - 0.5
		pred := 0.0
		actual := pred + fresh
		lo, hi, err := SplitIntervalSignedResiduals(pred, residuals, 0.1)
		if err != nil {
			panic(err)
		}
		if actual >= lo && actual <= hi {
			covered++
		}
	}
	if covered < 880 {
		t.Errorf("coverage %d/1000 below FW slack floor 880", covered)
	}
}
