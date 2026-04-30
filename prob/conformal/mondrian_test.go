package conformal

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// MondrianQuantile — closed-form
// =========================================================================

func TestMondrianQuantile_PerStratumQuantile(t *testing.T) {
	// 10 calibration samples in stratum 0, 10 in stratum 1.  Stratum 0
	// has small residuals (max 1); stratum 1 has large (max 10).  At
	// alpha=0.2, n=10 -> rank=ceil(11*0.8)=9 -> q = sorted[8].
	scores := []float64{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, // stratum 0
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, // stratum 1
	}
	strata := []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	q, err := MondrianQuantile(scores, strata, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(q[0]-0.9) > 1e-12 {
		t.Errorf("q[0] = %v, want 0.9", q[0])
	}
	if math.Abs(q[1]-9.0) > 1e-12 {
		t.Errorf("q[1] = %v, want 9", q[1])
	}
}

func TestMondrianQuantile_TooFewInStratum(t *testing.T) {
	// 1 sample in stratum 5; rank = ceil(2*0.95) = 2 > 1 -> +Inf.
	q, err := MondrianQuantile([]float64{0.5}, []int{5}, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(q[5], 1) {
		t.Errorf("q[5] = %v, want +Inf", q[5])
	}
}

// =========================================================================
// MondrianInterval
// =========================================================================

func TestMondrianInterval_KnownStratum(t *testing.T) {
	q := map[int]float64{0: 1.5, 1: 3.0}
	lo, hi := MondrianInterval(2.0, 1, q)
	if lo != -1.0 || hi != 5.0 {
		t.Errorf("interval = [%v, %v], want [-1, 5]", lo, hi)
	}
}

func TestMondrianInterval_UnknownStratumIsUninformative(t *testing.T) {
	q := map[int]float64{0: 1.5}
	lo, hi := MondrianInterval(2.0, 99, q)
	if !math.IsInf(lo, -1) || !math.IsInf(hi, 1) {
		t.Errorf("unknown stratum -> [%v, %v], want [-Inf, +Inf]", lo, hi)
	}
}

func TestMondrianInterval_InfStratumIsUninformative(t *testing.T) {
	q := map[int]float64{0: math.Inf(1)}
	lo, hi := MondrianInterval(0, 0, q)
	if !math.IsInf(lo, -1) || !math.IsInf(hi, 1) {
		t.Errorf("Inf threshold -> [%v, %v], want [-Inf, +Inf]", lo, hi)
	}
}

// =========================================================================
// Coverage simulation — per-stratum coverage holds when noise scales by
// stratum
// =========================================================================

func TestMondrianInterval_ConditionalCoverageOnHeteroskedasticity(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	const (
		nCalPerStratum  = 400
		nTestPerStratum = 600
		alpha           = 0.1
	)
	// Two strata: stratum 0 has noise sigma = 0.5, stratum 1 has sigma = 5.
	// A single (non-Mondrian) split conformal would use one threshold and
	// over-cover stratum 0 / under-cover stratum 1; Mondrian gives both
	// strata ~(1 - alpha) coverage.
	predict := func(x float64) float64 { return x }
	calScores := make([]float64, 0, 2*nCalPerStratum)
	calStrata := make([]int, 0, 2*nCalPerStratum)
	for s := 0; s < 2; s++ {
		sigma := 0.5
		if s == 1 {
			sigma = 5.0
		}
		for i := 0; i < nCalPerStratum; i++ {
			x := rng.Float64() * 10
			y := x + sigma*rng.NormFloat64()
			calScores = append(calScores, math.Abs(y-predict(x)))
			calStrata = append(calStrata, s)
		}
	}
	q, err := MondrianQuantile(calScores, calStrata, alpha)
	if err != nil {
		t.Fatal(err)
	}
	for s := 0; s < 2; s++ {
		sigma := 0.5
		if s == 1 {
			sigma = 5.0
		}
		var covered int
		for i := 0; i < nTestPerStratum; i++ {
			x := rng.Float64() * 10
			yTest := x + sigma*rng.NormFloat64()
			lo, hi := MondrianInterval(predict(x), s, q)
			if yTest >= lo && yTest <= hi {
				covered++
			}
		}
		cov := float64(covered) / float64(nTestPerStratum)
		// Tolerance set to ~4 binomial stderr (sqrt(0.1*0.9/600) ~ 0.012)
		// to absorb the 1-realisation noise.  Coverage gap in expectation
		// is bounded by 1/(n_cal+1) above the nominal.
		if cov < 1-alpha-0.05 {
			t.Errorf("stratum %d coverage = %.4f, want >= %.4f", s, cov, 1-alpha-0.05)
		}
	}
}

// =========================================================================
// Validation
// =========================================================================

func TestMondrianQuantile_RejectsBadInputs(t *testing.T) {
	if _, err := MondrianQuantile([]float64{1, 2}, []int{0, 0, 0}, 0.1); err == nil {
		t.Error("length mismatch should error")
	}
	if _, err := MondrianQuantile(nil, nil, 0.1); err == nil {
		t.Error("empty input should error")
	}
	if _, err := MondrianQuantile([]float64{1}, []int{0}, 0); err == nil {
		t.Error("alpha=0 should error")
	}
	if _, err := MondrianQuantile([]float64{-1}, []int{0}, 0.1); err == nil {
		t.Error("negative score should error")
	}
}
