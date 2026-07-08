package prob

import (
	"math"
	"testing"
)

// TestRegularizedGammaP_ExponentialExact pins correctness against an exact
// closed form: P(1, x) = 1 - e^{-x}, so GammaCDF(x,1,1) must match across the
// full x range. The series-only bug made this collapse for large x
// (GammaCDF(200,1,1) returned ~0.547 instead of ~1).
func TestRegularizedGammaP_ExponentialExact(t *testing.T) {
	for _, x := range []float64{0.5, 2, 5, 20, 50, 100, 200, 500} {
		got := GammaCDF(x, 1, 1)
		want := 1 - math.Exp(-x)
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("GammaCDF(%v,1,1) = %.12f, want 1-e^-x = %.12f", x, got, want)
		}
	}
}

// TestGammaCDF_MonotoneToOne: a CDF must be monotone non-decreasing and reach ~1.
// The bug made GammaCDF(x,100,1) non-monotone and collapse to ~0 for large x.
func TestGammaCDF_MonotoneToOne(t *testing.T) {
	prev := -1.0
	for _, x := range []float64{50, 100, 150, 200, 250, 300, 400, 600, 800} {
		v := GammaCDF(x, 100, 1)
		if v < prev-1e-9 {
			t.Errorf("GammaCDF(%v,100,1)=%v < prev=%v (non-monotone CDF)", x, v, prev)
		}
		if v < 0 || v > 1.0000001 {
			t.Errorf("GammaCDF(%v,100,1)=%v out of [0,1]", x, v)
		}
		prev = v
	}
	if prev < 0.999 {
		t.Errorf("GammaCDF(800,100,1)=%v, want ~1", prev)
	}
}

// TestRegularizedGammaP_IntegerShapeExact verifies the continued-fraction branch
// (x>=a+1) against an exact closed form: for integer shape n,
// P(n,x) = 1 - e^{-x} * sum_{k=0}^{n-1} x^k/k! (the Poisson tail). Every case uses
// x >= n+1 so it exercises the CF, not the series.
func TestRegularizedGammaP_IntegerShapeExact(t *testing.T) {
	for _, tc := range []struct {
		n int
		x float64
	}{
		{2, 8}, {3, 10}, {4, 12}, {5, 20}, {10, 30},
	} {
		got := GammaCDF(tc.x, float64(tc.n), 1)
		sum, term := 0.0, 1.0
		for k := 0; k < tc.n; k++ {
			if k > 0 {
				term *= tc.x / float64(k)
			}
			sum += term
		}
		want := 1 - math.Exp(-tc.x)*sum
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("GammaCDF(%v,%d,1) = %.12f, want %.12f (integer-shape exact)", tc.x, tc.n, got, want)
		}
	}
}

// TestChiSquaredTest_ExtremeStatistic: a hugely significant statistic must give a
// tiny p-value, not snap back to 1.0 (the series bug -> guaranteed false negative).
func TestChiSquaredTest_ExtremeStatistic(t *testing.T) {
	_, p := ChiSquaredTest([]float64{1000, 0}, []float64{500, 500}) // chi2=1000, df=1
	if !(p >= 0 && p < 1e-50) {
		t.Errorf("ChiSquaredTest(chi2~1000, df=1): p=%v, want tiny (~1e-218)", p)
	}
}

// TestPoissonCDF_Monotone: the CDF must be monotone non-decreasing in k.
func TestPoissonCDF_Monotone(t *testing.T) {
	prev := -1.0
	for k := 0; k <= 600; k += 20 {
		v := PoissonCDF(k, 500)
		if v < prev-1e-9 {
			t.Errorf("PoissonCDF(%d,500)=%v < prev=%v (non-monotone)", k, v, prev)
		}
		prev = v
	}
}
