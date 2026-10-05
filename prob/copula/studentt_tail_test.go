package copula

import (
	"math"
	"testing"
)

// TestStudentTCDFLargeDFAndTails checks StudentTCDF in the lower tail and at
// large df against reference values computed with mpmath at 60 significant
// digits (I_{df/(df+x^2)}(df/2, 1/2) / 2, evaluated at the exact binary64
// inputs). A local continued fraction capped at 200 iterations, with a
// prefix built from lgamma, was off by about 2e-10 relative at df = 1e6;
// and rounding df/(df+x^2) to 1 lost the whole deviation from 1/2 for
// |x| ~ 1e-8 (absolute error 4e-9).
func TestStudentTCDFLargeDFAndTails(t *testing.T) {
	cases := []struct{ x, df, want float64 }{
		{-1e-08, 1e6, 0.4999999960105781933412},
		{-1e-08, 1.0, 0.4999999968169011381621},
		{-0.001, 1e6, 0.4996010578858245449082},
		{-0.5, 250000.3, 0.308537758766415924989},
		{-1.0, 1e6, 0.1586553749167890646408},
		{-0.5, 7.3, 0.3158967634022207658214},
		{-20.0, 30.0, 3.374541832885643200616e-19},
		{-20.0, 1e6, 2.866543523695185917305e-89},
		{-5.0, 1e6, 2.866998935445370784489e-7},
		{-2.0, 250000.3, 0.02275067185822225927856},
		{-100.0, 5.0, 9.480007112311813694274e-10},
		{-1000.0, 1.0, 0.0003183097800805589388726},
		{-40.0, 123.45, 7.956763254577630138427e-73},
	}
	for _, c := range cases {
		got := StudentTCDF(c.x, c.df)
		if e := math.Abs(got-c.want) / c.want; !(e <= 1e-12) {
			t.Errorf("StudentTCDF(%v, %v) = %.17g, want %.17g (rel err %.3g > 1e-12)", c.x, c.df, got, c.want, e)
		}
		// Symmetry: the upper tail is the complement.
		if up := StudentTCDF(-c.x, c.df); math.Abs((1-up)-c.want) > 1e-15 {
			t.Errorf("1 - StudentTCDF(%v, %v) = %.17g, want %.17g", -c.x, c.df, 1-up, c.want)
		}
	}
}
