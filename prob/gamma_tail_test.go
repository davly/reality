package prob

import (
	"math"
	"testing"
)

// Regression tests for the incomplete-gamma upper-tail fix (regularizedGammaP/Q
// + regularizedGammaUpperCF). Before the fix, the series-only path returned
// catastrophically wrong values for large arguments — e.g. GammaCDF(300,1,1)
// returned ~7.7e-10 instead of ~1.0, and ChiSquaredTest(chiSq=1500) reported
// p=1.0 instead of ~0. These pin the corrected behavior.

func TestGammaTail_GammaCDF_ClosedForm_Exp(t *testing.T) {
	// Gamma(k=1, theta=1) is Exponential(1): P(1,x) = 1 - e^{-x}.
	for _, x := range []float64{0.5, 1, 5, 20, 50, 150, 300, 600} {
		got := GammaCDF(x, 1, 1)
		want := 1 - math.Exp(-x)
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("GammaCDF(%g,1,1)=%.15g want %.15g (diff %.3g)", x, got, want, got-want)
		}
	}
}

func TestGammaTail_GammaCDF_ClosedForm_K2(t *testing.T) {
	// Gamma(k=2, theta=1): P(2,x) = 1 - (1+x)e^{-x}.
	for _, x := range []float64{0.5, 1, 5, 20, 50, 100} {
		got := GammaCDF(x, 2, 1)
		want := 1 - (1+x)*math.Exp(-x)
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("GammaCDF(%g,2,1)=%.15g want %.15g (diff %.3g)", x, got, want, got-want)
		}
	}
}

func TestGammaTail_PQ_Complement(t *testing.T) {
	// P(a,x) + Q(a,x) == 1 to machine precision across both regimes + the x=a+1 seam.
	for _, a := range []float64{0.5, 1, 2, 5, 50, 500} {
		for _, x := range []float64{0.1, a - 0.5, a, a + 1, 2 * a, 10 * a} {
			if x <= 0 {
				continue
			}
			p := regularizedGammaP(a, x)
			q := regularizedGammaQ(a, x)
			if math.Abs(p+q-1) > 1e-12 {
				t.Errorf("P+Q != 1 at a=%g x=%g: P=%.15g Q=%.15g sum=%.15g", a, x, p, q, p+q)
			}
			if p < -1e-15 || p > 1+1e-15 || q < -1e-15 || q > 1+1e-15 {
				t.Errorf("P/Q out of [0,1] at a=%g x=%g: P=%g Q=%g", a, x, p, q)
			}
		}
	}
}

func TestGammaTail_ChiSquaredUpperTail(t *testing.T) {
	// Chi-squared with df=2: upper tail = Q(1, chiSq/2) = e^{-chiSq/2}.
	for _, chiSq := range []float64{2, 10, 40, 100} {
		got := regularizedGammaQ(1, chiSq/2)
		want := math.Exp(-chiSq / 2)
		if math.Abs(got-want) > 1e-14 {
			t.Errorf("Q(1,%g)=%.15g want e^{-%g}=%.15g", chiSq/2, got, chiSq/2, want)
		}
	}

	// The catastrophic regression case: a maximally-significant result must NOT
	// come back as p=1.0. chiSq=1500, df=2 -> true p = e^{-750} (underflows to 0).
	_, p := ChiSquaredTest([]float64{1000, 0, 0}, []float64{400, 400, 200})
	if p > 1e-12 {
		t.Errorf("ChiSquaredTest(chiSq=1500) p=%g; want ~0 (was 1.0 pre-fix)", p)
	}

	// A moderately significant case stays correct and finite: df=2, chiSq=20 -> p=e^{-10}.
	cs, p2 := ChiSquaredTest([]float64{18, 6, 6}, []float64{10, 10, 10}) // chiSq = 6.4+1.6+1.6=9.6
	_ = cs
	wantP2 := math.Exp(-9.6 / 2)
	if math.Abs(p2-wantP2) > 1e-12 {
		t.Errorf("ChiSquaredTest moderate: p=%.15g want %.15g", p2, wantP2)
	}
}

// poissonCDFRef sums the PMF directly in log-space as an independent reference.
func poissonCDFRef(k int, lambda float64) float64 {
	logTerm := -lambda // i=0
	sum := math.Exp(logTerm)
	for i := 1; i <= k; i++ {
		logTerm += math.Log(lambda) - math.Log(float64(i))
		sum += math.Exp(logTerm)
	}
	return sum
}

func TestGammaTail_PoissonCDF_DirectSum(t *testing.T) {
	cases := []struct {
		k      int
		lambda float64
	}{
		{0, 1}, {2, 1}, {5, 5}, {250, 500}, {500, 500}, {600, 500}, {750, 500},
	}
	for _, c := range cases {
		got := PoissonCDF(c.k, c.lambda)
		want := poissonCDFRef(c.k, c.lambda)
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("PoissonCDF(%d,%g)=%.12g want %.12g (diff %.3g)", c.k, c.lambda, got, want, got-want)
		}
	}
}
