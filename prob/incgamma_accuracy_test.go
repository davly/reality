package prob

import (
	"math"
	"testing"
)

// Accuracy of the regularized incomplete gamma function and its consumers
// (GammaCDF, PoissonCDF, ChiSquaredTest) at large shapes, in the tails and
// at small shapes. Reference values were computed with mpmath at 50
// significant digits, evaluated at the exact binary64 inputs.

func relErrGamma(got, want float64) float64 {
	if math.IsNaN(got) {
		return math.Inf(1)
	}
	return math.Abs(got-want) / math.Abs(want)
}

// TestGammaCDFLargeShapeReference checks GammaCDF against its documented
// relative precision at shapes up to 1e7, at the peak x = k where the series
// needs O(sqrt(k)) terms, and in both tails. A fixed 200-term series cap
// returned 0.4786 for k = 1e4 and 0.2375 for k = 1e5 (true 0.5013, 0.5004).
func TestGammaCDFLargeShapeReference(t *testing.T) {
	cases := []struct{ x, k, want float64 }{
		{10000.0, 10000.0, 0.5013298083399552003827},
		{100000.0, 100000.0, 0.5004205221103651766933},
		{1000000.0, 1000000.0, 0.5001329807608725912443},
		{10000000.0, 10000000.0, 0.5000420522087236983338},
		{997000.0, 1000000.0, 0.001338104167313599692259},
		{1003000.0, 1000000.0, 0.9986382593537824085206},
		{9900000.0, 10000000.0, 3.123539470267388968078e-221},
		{240000.0, 250000.0, 1.126997172149816400008e-91},
		{1050.0, 1000.0, 0.9413288886226819229022},
		{1e-06, 0.001, 0.9868481336940766653395},
		{1e-10, 0.5, 0.00001128379167057899955549},
	}
	for _, c := range cases {
		got := GammaCDF(c.x, c.k, 1)
		if e := relErrGamma(got, c.want); e > 1e-13 {
			t.Errorf("GammaCDF(%v, %v, 1) = %.17g, want %.17g (rel err %.3g > 1e-13)", c.x, c.k, got, c.want, e)
		}
	}
}

// TestPoissonCDFTailsReference checks PoissonCDF = Q(k+1, lambda) in the far
// lower tail, where 1 - P cancelled to 0 below the CDF's own first term, and
// at the median of large means, where the series cap left a 4% error.
func TestPoissonCDFTailsReference(t *testing.T) {
	cases := []struct {
		k      int
		lambda float64
		want   float64
	}{
		{0, 40.0, 4.248354255291588995329e-18},
		{0, 700.0, 9.859676543759770856705e-305},
		{3, 200.0, 1.873151462718925542388e-81},
		{10000, 10000.0, 0.5026595812190076252659},
		{1000000, 1000000.0, 0.5002659614862836527854},
		{999000, 1000000.0, 0.1587762998117256122762},
		{1001000, 1000000.0, 0.8414656709634281521223},
		{50, 10.0, 0.9999999999999999999638},
	}
	for _, c := range cases {
		got := PoissonCDF(c.k, c.lambda)
		if e := relErrGamma(got, c.want); e > 1e-12 {
			t.Errorf("PoissonCDF(%d, %v) = %.17g, want %.17g (rel err %.3g > 1e-12)", c.k, c.lambda, got, c.want, e)
		}
	}
	// The CDF can never fall below its own first term P(X = 0).
	if c, pmf := PoissonCDF(0, 40), PoissonPMF(0, 40); c < pmf*(1-1e-12) {
		t.Errorf("PoissonCDF(0, 40) = %g below PoissonPMF(0, 40) = %g", c, pmf)
	}
}

// TestRegularizedGammaQReference checks the upper function Q directly: small
// shapes, where Q is small and 1 - P would cancel, and the chi-squared upper
// tail Q(df/2, chi2/2) at large df.
func TestRegularizedGammaQReference(t *testing.T) {
	cases := []struct{ a, x, want float64 }{
		{0.001, 0.001, 0.00631235329113970990378},
		{0.001, 1.0, 0.0002196083575855563962893},
		{0.001, 1e-06, 0.01315186630592333466049},
		{0.01, 0.5, 0.005626756193967184146981},
		{0.1, 1.2, 0.01766115926527955000352},
		{0.5, 1.4, 0.09426430684121031254431},
		{0.9, 0.1, 0.8751049272712580738007},
		{99999.0 / 2, 99000.0 / 2, 0.9874491341820351278416},
		{999999.0 / 2, 1000000.0 / 2, 0.4995298419881127034477},
		{999999.0 / 2, 1010000.0 / 2, 9.022647647890001370799e-13},
		{0.5, 1e-08 / 2, 0.9999202115440526942235},
		{1, 1400.0 / 2, 9.859676543759770856705e-305},
		{50, 600.0 / 2, 2.418828583346485792244e-72},
	}
	for _, c := range cases {
		got := regularizedGammaQ(c.a, c.x)
		if e := relErrGamma(got, c.want); e > 1e-13 {
			t.Errorf("Q(%v, %v) = %.17g, want %.17g (rel err %.3g > 1e-13)", c.a, c.x, got, c.want, e)
		}
		p := regularizedGammaP(c.a, c.x)
		if math.Abs(p+got-1) > 1e-15 {
			t.Errorf("P + Q = %.17g at a=%v x=%v, want 1", p+got, c.a, c.x)
		}
	}
}

// TestChiSquaredTestManyCells checks the end-to-end p-value of a test with
// tens of thousands of cells. The statistic is summed with compensation: at
// these df the p-value moves by about ten times the statistic's relative
// error, so a plain running sum alone costs about 1e-11.
func TestChiSquaredTestManyCells(t *testing.T) {
	cases := []struct {
		k          int
		e, d       float64
		chi2, want float64
	}{
		{50001, 7.5, 2.78, 51523.69711999997629582, 9.103822863694819699286e-7},
		{200000, 12.25, 3.4999, 199988.5715918367613283, 0.5061574251706922298671},
	}
	for _, c := range cases {
		obs := make([]float64, c.k)
		exp := make([]float64, c.k)
		for i := range obs {
			exp[i] = c.e
			if i%2 == 0 {
				obs[i] = c.e + c.d
			} else {
				obs[i] = c.e - c.d
			}
		}
		chi2, p := ChiSquaredTest(obs, exp)
		if e := relErrGamma(chi2, c.chi2); e > 1e-15 {
			t.Errorf("k=%d: chi2 = %.17g, want %.17g (rel err %.3g)", c.k, chi2, c.chi2, e)
		}
		if e := relErrGamma(p, c.want); e > 1e-11 {
			t.Errorf("k=%d: p = %.17g, want %.17g (rel err %.3g > 1e-11)", c.k, p, c.want, e)
		}
	}
}

// TestGammaCDFNonConvergenceIsNaN checks that a shape so large that the
// expansions cannot converge within their iteration budget yields NaN, not
// a plausible-looking probability.
func TestGammaCDFNonConvergenceIsNaN(t *testing.T) {
	if testing.Short() {
		t.Skip("exhausts a 1e7-iteration budget")
	}
	for _, x := range []float64{1e17 - 1e9, 1e17} {
		if got := GammaCDF(x, 1e17, 1); !math.IsNaN(got) {
			t.Errorf("GammaCDF(%v, 1e17, 1) = %v, want NaN (no convergence)", x, got)
		}
	}
}
