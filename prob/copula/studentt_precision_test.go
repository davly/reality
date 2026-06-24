package copula

// Precision property tests — pins the Student-t CDF/quantile Precision bounds.
// Pure Go stdlib (testing/quick + math); ADDITIVE, zero math change.
//
// Claims pinned:
//   - studentt.go:53  StudentTQuantile: "~1e-10 absolute on x for p in
//     [1e-12, 1-1e-12], df >= 1." We pin the inverse-consistency invariant:
//     StudentTCDF(StudentTQuantile(p, df), df) == p to a tight tolerance (the
//     quantile is itself defined by bisection on the CDF to eps=1e-10, so the
//     CDF round-trip is the right oracle).
//   - studentt.go:23  StudentTCDF: monotone increasing in x; CDF(0)=0.5
//     (symmetry); CDF in [0,1].

import (
	"math"
	"testing"
	"testing/quick"
)

// genProb maps a uint64 into [1e-6, 1-1e-6] (a subset of the documented
// [1e-12, 1-1e-12] domain that stays well-conditioned for the CDF oracle).
func genProb(u uint64) float64 {
	lo, hi := 1e-6, 1-1e-6
	return lo + (hi-lo)*float64(u)/float64(math.MaxUint64)
}

// genDF maps a uint64 into [1, 200].
func genDF(u uint64) float64 {
	return 1 + 199*float64(u)/float64(math.MaxUint64)
}

// TestStudentTQuantileRoundTrip pins studentt.go:53. The quantile is found by
// bisection to eps=1e-10 on the CDF, so we verify CDF(quantile(p)) ~ p. The
// CDF round-trip error reflects the bisection tolerance scaled by the local
// PDF; we assert it stays below a small bound.
func TestStudentTQuantileRoundTrip(t *testing.T) {
	const bound = 1e-8 // CDF round-trip tolerance (bisection eps=1e-10 on x)
	var worst, worstP, worstDF float64
	prop := func(pu, dfu uint64) bool {
		p := genProb(pu)
		df := genDF(dfu)
		x := StudentTQuantile(p, df)
		if math.IsNaN(x) {
			return false // p,df are in-domain; NaN is a failure
		}
		back := StudentTCDF(x, df)
		err := math.Abs(back - p)
		if err > worst {
			worst, worstP, worstDF = err, p, df
		}
		return err <= bound
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 50000}); err != nil {
		t.Errorf("PRECISION REGRESSION: StudentTQuantile claims ~1e-10 on x; CDF round-trip error %g at p=%g df=%g exceeds %g", worst, worstP, worstDF, bound)
	}
	t.Logf("PINNED studentt.go:53 StudentTQuantile round-trip: worst CDF error %g at p=%g df=%g (bound %g)", worst, worstP, worstDF, bound)
}

// TestStudentTCDFContract pins studentt.go:23: monotone in x, CDF(0)=0.5
// (symmetry), output in [0,1].
func TestStudentTCDFContract(t *testing.T) {
	prop := func(au, bu, dfu uint64) bool {
		df := genDF(dfu)
		m := func(u uint64) float64 { return 60*float64(u)/float64(math.MaxUint64) - 30 }
		xa, xb := m(au), m(bu)
		ca, cb := StudentTCDF(xa, df), StudentTCDF(xb, df)
		if ca < 0 || ca > 1 || cb < 0 || cb > 1 {
			return false
		}
		if xa < xb && !(ca <= cb+1e-15) {
			return false
		}
		if xa > xb && !(ca >= cb-1e-15) {
			return false
		}
		// Symmetry: CDF(0) == 0.5.
		return math.Abs(StudentTCDF(0, df)-0.5) <= 1e-12
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 50000}); err != nil {
		t.Errorf("PRECISION REGRESSION: StudentTCDF monotonicity/symmetry/[0,1] violated: %v", err)
	}
}
