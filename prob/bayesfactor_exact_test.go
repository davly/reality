package prob

import (
	"math"
	"testing"
)

// TestProportionBayesFactor10Exact checks ProportionBayesFactor10 against
// the exact rational value
//
//	BF10 = sum_{j=0..k} C(n+1, j) / ((n+1-k) * C(n+1, k)),
//
// evaluated with exact integer arithmetic and rounded to float64. The closed
// form is the definition (2/(n+1)) * P(Beta(k+1, n-k+1) > 1/2) / (C(n,k) 2^-n)
// rewritten through P(Beta(k+1, n-k+1) > 1/2) = P(Binomial(n+1, 1/2) <= k);
// it agrees with mpmath's regularized incomplete beta at 80 digits and with
// mpmath quadrature of the H1 marginal to 1e-16 wherever those converge.
// Computing the tail as 1 - I_{1/2}(k+1, n-k+1) cancelled to 0 for small k:
// k = 0 gave 0 instead of 1/(n+1).
func TestProportionBayesFactor10Exact(t *testing.T) {
	const tol = 1e-13
	cases := []struct {
		k, n int
		want float64
	}{
		{0, 1, 0.5},
		{1, 1, 1.5},
		{0, 2, 0.3333333333333333},
		{1, 2, 0.6666666666666666},
		{2, 2, 2.3333333333333335},
		{0, 10, 0.09090909090909091},
		{3, 10, 0.17575757575757575},
		{8, 10, 4.002020202020202},
		{10, 10, 186.0909090909091},
		{0, 60, 0.01639344262295082},
		{1, 60, 0.016939890710382512},
		{10, 60, 0.024127246278546337},
		{30, 60, 0.15981414117778536},
		{45, 60, 710.5871218667031},
		{60, 60, 3.780070506907695e+16},
		{0, 200, 0.004975124378109453},
		{40, 200, 0.00822105313719082},
		{100, 200, 0.08829207931756568},
		{150, 200, 35230005777.06301},
		{200, 200, 1.5989433276208858e+58},
		{0, 1000, 0.000999000999000999},
		{3, 1000, 0.0010050190531152455},
		{300, 1000, 0.002484629242719005},
		{500, 1000, 0.03960357895234298},
		{750, 1000, 4.439036260992781e+55},
		{1000, 1000, 2.1408763380345e+298},
		{0, 100000, 9.99990000099999e-06},
		{49000, 100000, 0.0004883369584626274},
		{50000, 100000, 0.003963297572960911},
		{50300, 100000, 0.04656739898133638},
	}
	for _, c := range cases {
		bf, ok := ProportionBayesFactor10(c.k, c.n)
		if !ok {
			t.Errorf("%d/%d: ok = false (bf = %g), want a finite result", c.k, c.n, bf)
			continue
		}
		if rel := math.Abs(bf-c.want) / c.want; rel > tol {
			t.Errorf("%d/%d: bf = %.17g, want %.17g (relative error %.3g > %g)", c.k, c.n, bf, c.want, rel, tol)
		}
	}
	// k = 0 is exactly 1/(n+1): one correctly rounded division.
	for _, n := range []int{1, 7, 60, 12345, 1 << 40} {
		if bf, _ := ProportionBayesFactor10(0, n); bf != 1/float64(n+1) {
			t.Errorf("0/%d: bf = %.17g, want exactly 1/%d = %.17g", n, bf, n+1, 1/float64(n+1))
		}
	}
}
