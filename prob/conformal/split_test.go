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
