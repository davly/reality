package prob

import (
	"math"
	"testing"
)

// orderedBits maps a float64 to an int64 that is monotone in the float's
// value, so that differences count representable steps (ulps).
func orderedBits(x float64) int64 {
	b := int64(math.Float64bits(x))
	if b < 0 {
		b = math.MinInt64 - b
	}
	return b
}

// ulpsBetween returns how many float64 steps separate a and b.
func ulpsBetween(a, b float64) uint64 {
	ia, ib := orderedBits(a), orderedBits(b)
	if ia > ib {
		return uint64(ia - ib)
	}
	return uint64(ib - ia)
}

// TestExponentialQuantileLog1p checks ExponentialQuantile against
// -ln(1-p)/lambda computed by mpmath at 50 significant digits (log1p,
// cross-checked against a direct evaluation at 800 digits) at the exact
// binary64 inputs, rounded to float64. Forming 1-p first loses the low digits
// of p: -log(1-p) had relative error 8.3e-8 at p = 1e-10 and returned 0 for
// p < 2^-54.
func TestExponentialQuantileLog1p(t *testing.T) {
	cases := []struct{ p, lambda, want float64 }{
		{5e-324, 1.0, 5e-324},
		{1e-300, 1.0, 1e-300},
		{1e-200, 1.0, 1e-200},
		{1e-100, 1.0, 1e-100},
		{1e-20, 1.0, 1e-20},
		{1e-16, 1.0, 1e-16},
		{1.1102230246251565e-16, 1.0, 1.1102230246251565e-16},
		{1e-10, 1.0, 1.00000000005e-10},
		{1e-08, 1.0, 1.0000000050000001e-08},
		{1e-05, 1.0, 1.0000050000333337e-05},
		{0.001, 1.0, 0.0010005003335835335},
		{0.01, 1.0, 0.010050335853501442},
		{0.1, 1.0, 0.10536051565782631},
		{0.25, 1.0, 0.2876820724517809},
		{0.5, 1.0, 0.6931471805599453},
		{0.75, 1.0, 1.3862943611198906},
		{0.9, 1.0, 2.302585092994046},
		{0.99, 1.0, 4.605170185988091},
		{0.999999, 1.0, 13.815510557935518},
		{0.9999999999, 1.0, 23.02585084720009},
		{0.999999999999999, 1.0, 34.53957599234088},
		{0.9999999999999999, 1.0, 36.7368005696771},
		{1e-10, 0.5, 2.0000000001e-10},
		{1e-10, 3.0, 3.3333333335e-11},
		{0.3, 0.001, 356.6749439387324},
		{0.3, 1000.0, 0.0003566749439387324},
		{0.9999999999999999, 7.0, 5.248114367096729},
		{8.673617379884035e-19, 0.1, 8.673617379884035e-18},
	}
	for _, c := range cases {
		got := ExponentialQuantile(c.p, c.lambda)
		if d := ulpsBetween(got, c.want); d > 2 {
			t.Errorf("ExponentialQuantile(%v, %v) = %v, want %v (%d ulps apart)", c.p, c.lambda, got, c.want, d)
		}
	}
}

// TestNormalQuantileNearlyExact checks NormalQuantile against the exact
// quantile mu + sigma*Phi^{-1}(p) from mpmath at 50 significant digits
// (Newton's method on log Phi(x) = log p with mpmath's erfc, cross-checked
// against sqrt(2)*erfinv(2p-1) evaluated with enough digits for 2p-1 to be
// exact; the two agree to 1e-45), at the exact binary64 p, rounded to
// float64. Acklam's rational approximation alone is accurate only to about
// 1e-9 relative, millions of ulps.
func TestNormalQuantileNearlyExact(t *testing.T) {
	cases := []struct{ p, mu, sigma, want float64 }{
		{5e-324, 0.0, 1.0, -38.467405617144344},
		{1e-320, 0.0, 1.0, -38.26912534303265},
		{1e-310, 0.0, 1.0, -37.663060331949524},
		{2.2250738585072014e-308, 0.0, 1.0, -37.5193793471445},
		{1e-300, 0.0, 1.0, -37.0470962993612},
		{1e-200, 0.0, 1.0, -30.20559417957964},
		{1e-100, 0.0, 1.0, -21.273453560965326},
		{1e-50, 0.0, 1.0, -14.933337534788489},
		{1e-20, 0.0, 1.0, -9.262340089798407},
		{1e-12, 0.0, 1.0, -7.034483825301132},
		{1e-06, 0.0, 1.0, -4.753424308822899},
		{0.001, 0.0, 1.0, -3.0902323061678136},
		{0.02425, 0.0, 1.0, -1.972961051311885},
		{0.025, 0.0, 1.0, -1.9599639845400543},
		{0.05, 0.0, 1.0, -1.6448536269514726},
		{0.1, 0.0, 1.0, -1.2815515655446004},
		{0.2, 0.0, 1.0, -0.8416212335729142},
		{0.25, 0.0, 1.0, -0.6744897501960817},
		{0.3, 0.0, 1.0, -0.5244005127080408},
		{0.4, 0.0, 1.0, -0.2533471031357997},
		{0.45, 0.0, 1.0, -0.12566134685507402},
		{0.49, 0.0, 1.0, -0.025068908258711057},
		{0.4999999999, 0.0, 1.0, -2.506628482030354e-10},
		{0.5, 0.0, 1.0, 0.0},
		{0.5000000001, 0.0, 1.0, 2.506628482030354e-10},
		{0.6, 0.0, 1.0, 0.2533471031357997},
		{0.75, 0.0, 1.0, 0.6744897501960817},
		{0.8, 0.0, 1.0, 0.8416212335729144},
		{0.9, 0.0, 1.0, 1.2815515655446006},
		{0.97575, 0.0, 1.0, 1.972961051311885},
		{0.975, 0.0, 1.0, 1.9599639845400538},
		{0.99, 0.0, 1.0, 2.3263478740408408},
		{0.999999, 0.0, 1.0, 4.753424308817087},
		{0.9999999999, 0.0, 1.0, 6.361340889697422},
		{0.999999999999, 0.0, 1.0, 7.0344869100478356},
		{0.999999999999999, 0.0, 1.0, 7.941444487415978},
		{0.9999999999999999, 0.0, 1.0, 8.209536151601387},
		{0.975, 10.0, 3.0, 15.879891953620161},
		{0.05, -2.5, 0.01, -2.5164485362695146},
		{1e-12, 100.0, 15.0, -5.517257379516979},
		{0.999999, 0.0, 1e+300, 4.753424308817088e+300},
	}
	for _, c := range cases {
		got := NormalQuantile(c.p, c.mu, c.sigma)
		if c.mu == 0 && c.sigma == 1 {
			if d := ulpsBetween(got, c.want); d > 2 {
				t.Errorf("NormalQuantile(%v) = %v, want %v (%d ulps apart)", c.p, got, c.want, d)
			}
			continue
		}
		// mu + sigma*z adds two roundings, and cancellation between mu and
		// sigma*z magnifies the error of z relative to the result.
		bound := 4 * 0x1p-52 * (math.Abs(c.mu) + math.Abs(c.want-c.mu))
		if diff := math.Abs(got - c.want); diff > bound {
			t.Errorf("NormalQuantile(%v, %v, %v) = %v, want %v (error %.3g > %.3g)", c.p, c.mu, c.sigma, got, c.want, diff, bound)
		}
	}
}
