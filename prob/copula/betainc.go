package copula

import "github.com/davly/reality/prob"

// regularizedBetaInc computes the regularized incomplete beta function
// I_x(a, b) = B(x; a, b) / B(a, b) by delegating to prob.RegularizedBetaInc
// (DiDonato & Morris 1992), which keeps its accuracy at large parameters
// and in the tails. Returns NaN on out-of-domain input.
//
// A local continued fraction with a 200-iteration cap and a prefix built
// from lgamma used to live here; it lost about 2e-10 relative in
// StudentTCDF at df = 1e6. Delegating keeps one verified implementation;
// copula already depends on prob, so no import cycle arises.
func regularizedBetaInc(x, a, b float64) float64 {
	return prob.RegularizedBetaInc(x, a, b)
}

// studentTTailBeta returns I_{df/(df+x2)}(df/2, 1/2), the two-sided tail
// probability of Student's t at |t| = sqrt(x2), for df > 0 and x2 >= 0.
//
// bx = df/(df+x2) is rounded to float64; when it exceeds 1/2 that moves its
// complement 1 - bx, the variable the result depends on, by up to 2^-54 in
// absolute terms. The exact complement y = x2/(df+x2) is used instead:
//   - for |t| <= 0.67, below the median of |T| for every df, the result is
//     at least 1/2 and is computed as 1 - I_y(1/2, df/2);
//   - otherwise the rounding is removed to first order with the density,
//     I(1-y) = I(bx) + I'(bx) ((1-bx) - y), where 1 - bx is exact and
//     |(1-bx) - y| <= 2^-54 is far below y (without it the relative error
//     grows as about (df/2) 2^-54, 1e-11 at df = 1e6).
func studentTTailBeta(df, x2 float64) float64 {
	bx := df / (df + x2)
	if !(bx > 0.5) {
		return regularizedBetaInc(bx, df/2, 0.5)
	}
	y := x2 / (df + x2)
	if x2 <= 0.45 {
		return 1 - regularizedBetaInc(y, 0.5, df/2)
	}
	ib := regularizedBetaInc(bx, df/2, 0.5)
	return ib + prob.BetaPDF(bx, df/2, 0.5)*((1-bx)-y)
}
