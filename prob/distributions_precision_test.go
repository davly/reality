package prob

// Precision property tests — pins prob/ distribution Precision: bounds as
// tested invariants. Pure Go stdlib (testing/quick + math); ADDITIVE, zero
// math change.
//
// Claims pinned:
//   - NormalQuantile: "within 2.5 ulps of the exact quantile". Two pure-stdlib
//     oracles: the round-trip NormalCDF(NormalQuantile(p)) = p, and bisection
//     of NormalCDF (lower half directly, upper half through the symmetry
//     Phi^{-1}(p) = -Phi^{-1}(1-p) with 1-p exact). Both are limited by the
//     accuracy of NormalCDF, so the bounds here are looser than the claim;
//     quantile_precision_test.go checks the claim against mpmath values.
//   - hypothesis.go:165 / mathutil.go:187  ChiSquaredTest p-value: regression
//     pin of the known gamma/chi-sq bug (series-only made the CDF wrong /
//     p=1.0 for large chi2). We assert the CDF is monotone in x and that a
//     large chi-squared statistic yields p ~ 0 (not 1.0).
//   - distributions.go  NormalCDF: monotone increasing; symmetric about mu.

import (
	"math"
	"testing"
	"testing/quick"
)

// genUnitOpen maps a uint64 to an open-interval probability in (0,1) avoiding
// the exact endpoints.
func genUnitOpen(u uint64) float64 {
	p := (float64(u) + 1) / (float64(math.MaxUint64) + 2)
	if p <= 0 {
		p = 1e-300
	}
	if p >= 1 {
		p = math.Nextafter(1, 0)
	}
	return p
}

// TestNormalQuantileRoundTrip checks |NormalCDF(NormalQuantile(p)) - p| on
// the bulk p in [1e-6, 1-1e-6], where the round trip is well conditioned. A
// quantile within a few ulps leaves only the rounding of NormalCDF itself,
// a few times 1e-16; Acklam's approximation alone left up to 3e-10.
func TestNormalQuantileRoundTrip(t *testing.T) {
	const bound = 2e-15
	var worstBulk, worstBulkAt float64
	prop := func(u uint64) bool {
		p := genUnitOpen(u)
		// Bulk region where CDF round-trip is well-conditioned.
		if p < 1e-6 || p > 1-1e-6 {
			return true
		}
		x := NormalQuantile(p, 0, 1)
		pBack := NormalCDF(x, 0, 1)
		err := math.Abs(pBack - p)
		if err > worstBulk {
			worstBulk, worstBulkAt = err, p
		}
		return err <= bound
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 200000}); err != nil {
		t.Errorf("NormalQuantile CDF round-trip error %g at p=%g exceeds %g", worstBulk, worstBulkAt, bound)
	}
	t.Logf("NormalQuantile (bulk CDF round-trip): worst error %g at p=%g (<= %g)", worstBulk, worstBulkAt, bound)
}

// trueStdNormalQuantile is a reference for Phi^{-1}(p) for p <= 1/2,
// obtained by bisecting NormalCDF (math.Erfc-backed), which keeps its
// relative precision in the lower tail. Near 1, NormalCDF can only resolve p
// to 1.1e-16, so x only to 1.1e-16/phi(x) (5e-7 relative at p = 1-1e-12);
// for p > 1/2 use -trueStdNormalQuantile(1-p) instead (1-p is exact there).
// 200 bisection steps far exceed float64 precision.
func trueStdNormalQuantile(p float64) float64 {
	lo, hi := -40.0, 40.0
	for i := 0; i < 200; i++ {
		mid := 0.5 * (lo + hi)
		if NormalCDF(mid, 0, 1) < p {
			lo = mid
		} else {
			hi = mid
		}
	}
	return 0.5 * (lo + hi)
}

// normalQuantileBisectionBound bounds the relative difference between
// NormalQuantile and the bisection reference: a few ulps of quantile error
// plus the reference's own error, which follows from the rounding of
// NormalCDF's argument and of math.Erfc. Acklam's approximation alone was
// off by up to 1.15e-9.
const normalQuantileBisectionBound = 1e-14

// TestNormalQuantileValueLowerAndBulk compares NormalQuantile with the
// bisection reference on p in [1e-12, 0.5). p=0.5 is excluded only because the
// true value is ~0 (relative error is undefined).
func TestNormalQuantileValueLowerAndBulk(t *testing.T) {
	const bound = normalQuantileBisectionBound
	var worst, worstAt float64
	// Dense log-spaced grid on the lower half.
	for _, p := range logGridLower() {
		approx := NormalQuantile(p, 0, 1)
		tru := trueStdNormalQuantile(p)
		if math.Abs(tru) < 1e-12 {
			continue // p≈0.5: true value ~0, relative error undefined
		}
		rel := math.Abs(approx-tru) / math.Abs(tru)
		if rel > worst {
			worst, worstAt = rel, p
		}
	}
	if worst > bound {
		t.Errorf("NormalQuantile lower/bulk: relative difference from the bisection reference %g at p=%g exceeds %g", worst, worstAt, bound)
	}
	t.Logf("NormalQuantile value (lower+bulk, p in [1e-12,0.5)): worst relative difference %g at p=%g (<= %g)", worst, worstAt, bound)
}

// logGridLower returns a log-spaced grid of probabilities in (0, 0.5].
func logGridLower() []float64 {
	var g []float64
	for e := 12.0; e >= 0.31; e -= 0.05 { // p from 1e-12 up to ~0.49
		p := math.Pow(10, -e)
		if p < 0.5 {
			g = append(g, p)
		}
	}
	g = append(g, 0.5)
	return g
}

// TestNormalQuantileValueUpperTail compares NormalQuantile with the bisection
// reference as p -> 1, through the symmetry Phi^{-1}(p) = -Phi^{-1}(1-p): 1-p
// is exact in float64 for p >= 1/2, so the reference is as good as in the
// lower tail. (Bisecting NormalCDF near 1 directly cannot resolve x; an
// earlier version of this test did, and read its own 1.1e-6 resolution limit
// at p = 1-1e-12 as an error of the upper tail, which was in fact as accurate
// as the lower tail.)
func TestNormalQuantileValueUpperTail(t *testing.T) {
	const bound = normalQuantileBisectionBound
	var worst, worstAt float64
	for e := 0.4; e <= 15.9; e += 0.05 {
		p := 1 - math.Pow(10, -e)
		if p >= 1 || p <= 0.5 {
			continue
		}
		approx := NormalQuantile(p, 0, 1)
		tru := -trueStdNormalQuantile(1 - p)
		rel := math.Abs(approx-tru) / math.Abs(tru)
		if rel > worst {
			worst, worstAt = rel, p
		}
	}
	if worst > bound {
		t.Errorf("NormalQuantile upper tail: relative difference from the bisection reference %g at p=%.17g exceeds %g", worst, worstAt, bound)
	}
	t.Logf("NormalQuantile upper tail (p in (0.6, 1-1e-16)): worst relative difference %g at p=%.17g (<= %g)", worst, worstAt, bound)
}

// TestNormalCDFMonotone pins NormalCDF "monotonically increasing" and the
// documented symmetry NormalCDF(-x) = 1 - NormalCDF(x).
func TestNormalCDFMonotone(t *testing.T) {
	prop := func(a, b uint64) bool {
		m := func(u uint64) float64 { return 20*float64(u)/float64(math.MaxUint64) - 10 }
		xa, xb := m(a), m(b)
		ca, cb := NormalCDF(xa, 0, 1), NormalCDF(xb, 0, 1)
		// monotone
		if xa < xb && !(ca <= cb) {
			return false
		}
		if xa > xb && !(ca >= cb) {
			return false
		}
		// symmetry about mean 0: CDF(-x) + CDF(x) == 1 (to ~1e-12)
		return math.Abs(NormalCDF(-xa, 0, 1)+ca-1) <= 1e-12
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 100000}); err != nil {
		t.Errorf("PRECISION REGRESSION: NormalCDF monotonicity/symmetry violated: %v", err)
	}
}

// TestChiSquaredTestNoLargeStatBug is a REGRESSION pin of the known gamma/
// chi-squared p-value bug: a series-only regularized lower gamma made the CDF
// wrong (and p=1.0) for large chi2. With the correct two-branch
// regularizedGammaP, a large chi-squared statistic must give p ~ 0, and the
// CDF must be monotone non-decreasing in chi2 (hypothesis.go:165).
func TestChiSquaredTestNoLargeStatBug(t *testing.T) {
	// observed vs expected over df=9 (10 cells). Scale the deviation to push
	// chi2 up to ~1500.
	expected := []float64{10, 10, 10, 10, 10, 10, 10, 10, 10, 10}
	mkObserved := func(dev float64) []float64 {
		o := make([]float64, len(expected))
		for i := range o {
			if i%2 == 0 {
				o[i] = expected[i] + dev
			} else {
				o[i] = expected[i] - dev
			}
		}
		return o
	}

	prevChi := -1.0
	prevP := math.Inf(1)
	for _, dev := range []float64{0, 1, 3, 10, 30, 60, 120} {
		chi, p := ChiSquaredTest(mkObserved(dev), expected)
		if math.IsNaN(chi) || math.IsNaN(p) {
			t.Fatalf("ChiSquaredTest returned NaN for dev=%v", dev)
		}
		if p < 0 || p > 1 {
			t.Fatalf("ChiSquaredTest p out of [0,1]: %g (chi=%g, dev=%v)", p, chi, dev)
		}
		// p must be NON-INCREASING as chi increases (CDF monotone).
		if chi > prevChi && p > prevP+1e-12 {
			t.Fatalf("PRECISION REGRESSION / BUG: ChiSquaredTest p-value not monotone — chi=%g gave p=%g but smaller chi gave p=%g (gamma/chi-sq CDF regression)", chi, p, prevP)
		}
		prevChi, prevP = chi, p
	}

	// The biggest deviation should produce a very large chi2 with p ~ 0, NOT
	// the historical p=1.0 bug.
	chiBig, pBig := ChiSquaredTest(mkObserved(120), expected)
	if chiBig < 100 {
		t.Fatalf("test setup: expected large chi2, got %g", chiBig)
	}
	if pBig > 1e-6 {
		t.Fatalf("PRECISION REGRESSION / BUG: ChiSquaredTest with chi2=%g returned p=%g (expected ~0; the series-only gamma bug returns ~1.0)", chiBig, pBig)
	}
	t.Logf("PINNED hypothesis.go:165 chi-sq regression: chi2=%g -> p=%g (~0, gamma two-branch fix holds)", chiBig, pBig)
}
