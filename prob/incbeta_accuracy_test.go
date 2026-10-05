package prob

import (
	"math"
	"testing"
)

// Accuracy of the regularized incomplete beta function and its consumers
// (BetaCDF, the Student t p-values, BinomialCDF) at large parameters, with
// one small and one large parameter, and in the tails. Reference values were
// computed with mpmath at 60 significant digits (the continued fraction on
// its convergent side, cross-checked against mpmath's betainc and against
// quadrature of the density), evaluated at the exact binary64 inputs.

// betaTol is the documented relative precision of the incomplete beta
// function: 1e-13 down to results of 1e-150, and below that an error that
// grows in proportion to |ln I|.
func betaTol(want float64) float64 {
	if want >= 1e-150 {
		return 1e-13
	}
	return 5e-16 * -math.Log(want)
}

// TestBetaCDFLargeParametersReference checks BetaCDF at parameters up to
// 1e6, including one small and one large parameter, in the bulk and in
// the tails. A 200-iteration continued-fraction cap and a prefix built from
// lgamma left BetaCDF(0.5, 1e6, 1e6) = 0.500313 (exactly 1/2).
func TestBetaCDFLargeParametersReference(t *testing.T) {
	cases := []struct{ x, a, b, want float64 }{
		{0.5, 1e6, 1e6, 0.5},
		{0.49893934009338514, 1e6, 1e6, 0.001349888059979308673431},
		{0.49964644669779507, 1e6, 1e6, 0.158655314424147774047},
		{0.5001767766511025, 1e6, 1e6, 0.691462400762816792534},
		{0.5007071066044099, 1e6, 1e6, 0.9772498815495792304206},
		{0.4977639376126492, 1e5, 1e5, 0.02274999696831682238178},
		{0.5011180311936754, 1e5, 1e5, 0.8413441411424809597657},
		{0.9999959644707634, 1e6, 0.5, 0.004497748144948535512374},
		{0.9999987928943527, 1e6, 0.5, 0.1202384492452202514693},
		{0.99999950000025, 1e6, 0.5, 0.3173106288252552078451},
		{1.4644680134807017e-07, 0.5, 1e6, 0.4116277862240288123147},
		{2.621317441912454e-06, 0.5, 1e6, 0.9779602016450389520567},
		{0.9999836756427081, 1e6, 10, 0.03685395164403048371243},
		{0.9999915812143219, 1e6, 10, 0.6634847769340051831675},
		{0.0003674932812315346, 1000, 1e6, 2.696303766686022254011e-162},
		{7.656616266262453e-05, 2, 1e5, 0.9959074436726076463814},
	}
	for _, c := range cases {
		got := BetaCDF(c.x, c.a, c.b)
		if d := math.Abs(got - c.want); !(d <= 1e-13) {
			t.Errorf("BetaCDF(%v, %v, %v) = %.17g, want %.17g (abs err %.3g > 1e-13)", c.x, c.a, c.b, got, c.want, d)
			continue
		}
		if e := math.Abs(got-c.want) / c.want; e > betaTol(c.want) {
			t.Errorf("BetaCDF(%v, %v, %v) = %.17g, want %.17g (rel err %.3g > %.3g)", c.x, c.a, c.b, got, c.want, e, betaTol(c.want))
		}
	}
}

// TestStudentTLowerTailReference checks the Student t lower tail
// CDF(-|t|) = I_{df/(df+t^2)}(df/2, 1/2) / 2 for large df and far tails.
func TestStudentTLowerTailReference(t *testing.T) {
	cases := []struct{ t, df, want float64 }{
		{20.0, 30.0, 3.374541832885643200616e-19},
		{20.0, 1e6, 2.866543523695185917305e-89},
		{5.0, 1e6, 2.866998935445370784489e-7},
		{2.0, 250000.3, 0.02275067185822225927856},
		{100.0, 5.0, 9.480007112311813694274e-10},
		{1000.0, 1.0, 0.0003183097800805589388726},
		{40.0, 123.45, 7.956763254577630138427e-73},
	}
	for _, c := range cases {
		got := studentTCDF(-c.t, c.df)
		if e := math.Abs(got-c.want) / c.want; !(e <= 1e-13) {
			t.Errorf("studentTCDF(%v, %v) = %.17g, want %.17g (rel err %.3g > 1e-13)", -c.t, c.df, got, c.want, e)
		}
	}
}

// TestTTestTwoSampleSmallPValue checks a Welch test whose p-value is far
// below 1e-16: the former 2*(1 - CDF(|t|)) form cancelled it to noise.
func TestTTestTwoSampleSmallPValue(t *testing.T) {
	d1 := make([]float64, 12)
	for i := range d1 {
		d1[i] = 1.5 + 0.25*float64(i)
	}
	d2 := make([]float64, 9)
	for i := range d2 {
		d2[i] = 30 + 0.5*float64(i)
	}
	const wantT = -55.43430584169827409012
	const wantP = 7.153374827010427655021e-17
	tStat, p := TTestTwoSample(d1, d2)
	if e := math.Abs(tStat-wantT) / -wantT; e > 1e-14 {
		t.Errorf("t = %.17g, want %.17g (rel err %.3g)", tStat, wantT, e)
	}
	if e := math.Abs(p-wantP) / wantP; !(e <= 1e-11) {
		t.Errorf("p = %.17g, want %.17g (rel err %.3g > 1e-11)", p, wantP, e)
	}
}

// TestBinomialCDFReference checks BinomialCDF = I_{1-p}(n-k, k+1) for
// n up to 1e6, near the mean and in both tails.
func TestBinomialCDFReference(t *testing.T) {
	cases := []struct {
		k, n int
		p    float64
		want float64
	}{
		{500000, 1000000, 0.5, 0.5003989421806658750445},
		{499000, 1000000, 0.5, 0.02280414993269104321015},
		{100, 1000000, 1e-4, 0.5265621985632190987484},
		{60, 1000000, 1e-4, 0.00001080327967830556328312},
		{0, 1000, 0.01, 0.00004317124741065824191104},
		{900, 1000, 0.95, 8.410251084877702763195e-11},
		{3, 100000, 1e-10, 0.9999999999999999999996},
		{999999, 1000000, 0.999999, 0.6321207427789335287961},
	}
	for _, c := range cases {
		got := BinomialCDF(c.k, c.n, c.p)
		if e := math.Abs(got-c.want) / c.want; !(e <= 1e-13) {
			t.Errorf("BinomialCDF(%d, %d, %v) = %.17g, want %.17g (rel err %.3g > 1e-13)", c.k, c.n, c.p, got, c.want, e)
		}
	}
}

// TestRegularizedBetaIncBeyondRangeIsNaN checks that near the mean of
// parameters above 2^52, where a+n is no longer exact in float64 and the
// continued fraction's coefficients lose accuracy, the result is NaN rather
// than a plausible-looking probability; and that a tail which underflows is
// still 0 there.
func TestRegularizedBetaIncBeyondRangeIsNaN(t *testing.T) {
	if got := RegularizedBetaInc(0.5, 1e17, 1e17); !math.IsNaN(got) {
		t.Errorf("RegularizedBetaInc(0.5, 1e17, 1e17) = %v, want NaN", got)
	}
	if got := RegularizedBetaInc(0.4, 1e17, 1e17); got != 0 {
		t.Errorf("RegularizedBetaInc(0.4, 1e17, 1e17) = %v, want 0 (underflow)", got)
	}
}
