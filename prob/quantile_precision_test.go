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
