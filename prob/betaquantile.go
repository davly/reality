package prob

import "math"

// BetaQuantile returns x such that I_x(a, b) = p: the inverse of BetaCDF and
// RegularizedBetaInc in x, the quantile function of the Beta(a, b)
// distribution.
//
// Method: a bracketed Newton iteration on the logarithm of the smaller tail,
// with bisection as the fallback (see betaQuantile). The root is solved for
// in whichever of x and 1 - x is at most 1/2, so a quantile near 1 is found
// to full relative accuracy in 1 - x; BetaQuantileXY returns that value.
//
// Valid range: a > 0, b > 0, both finite; p in (0, 1)
// Returns NaN if p <= 0, p >= 1 or p is NaN, if a or b is not a finite
// positive number, or where RegularizedBetaInc returns NaN (near the mean of
// parameters above 2^52), as NormalQuantile and ExponentialQuantile do at
// the ends of their ranges.
// Precision: measured against an 800-digit mpmath table of 101 cases
// (testdata/stress/beta_quantile.json: shapes 0.01 to 20000, p from 1e-300
// to 0.999), with the same results on default and fused multiply-add amd64
// builds: relative error at most 4e-15 for a, b in [0.5, 100] and p in
// [1e-6, 0.999], and at most 7e-14 over the whole table (measured maxima
// 3.8e-15 and 6.1e-14; testdata/stress/precision_beta.json holds these
// claims). The quantile inherits the error of I_x itself: a relative error d in the tail
// probability moves x by about d/a in the lower tail, so the largest
// errors are at shapes 0.1 and 0.01, and in the far tails, where
// RegularizedBetaInc's own error grows with |ln I|. A result near 1 is
// accurate in absolute terms only (half an ulp of 1); BetaQuantileXY returns
// 1 - x to full relative accuracy. The round trip
// RegularizedBetaInc(BetaQuantile(a, b, p), a, b) is within 1.2e-12 of p
// over the same core range, limited by the rounding of x near 1 (at
// a = 100, b = 0.5, p = 0.999, where 1 - x = 7.9e-9).
// Cost: 2 to 12 evaluations of the incomplete beta function for every case
// in the table but the symmetric median at a = b = 0.1 (about 35); about 35
// when the quantile underflows to 0 or rounds to 1 with 1 - x below the
// double range; never more than 200. Over 100,000 random cases (shapes 1e-3
// to 1e7, p down to 1e-300) the most measured was 131.
// Reference: DiDonato, A.R. & Morris, A.H. (1992) "Algorithm 708", ACM TOMS
// 18(3) for I_x; Press et al., Numerical Recipes, 3rd ed., section 9.4
// (rtsafe) for the safeguarded Newton iteration.
func BetaQuantile(a, b, p float64) float64 {
	x, _ := betaQuantile(a, b, p, false)
	return x
}

// BetaQuantileComplement returns x such that 1 - I_x(a, b) = q: the upper
// quantile, BetaQuantile(a, b, 1 - q), without forming 1 - q. It keeps
// accuracy when q is tiny, where 1 - q would round to 1 and the lower
// quantile could only return its largest value.
//
// Valid range: a > 0, b > 0, both finite; q in (0, 1)
// Returns NaN if q <= 0, q >= 1 or q is NaN, if a or b is not a finite
// positive number, or where RegularizedBetaInc returns NaN.
// Precision: as for BetaQuantile with the shapes exchanged, since
// 1 - I_x(a, b) = I_{1-x}(b, a). Against the same table, read with the
// shapes swapped (BetaQuantileComplement(b, a, p) against 1 - x), the
// relative error is at most 2.5e-13 (measured 2.2e-13); the largest are
// at the shape 0.01, where p as a double differs from the table's decimal
// p and that difference is amplified about 100 times. A result near 1 is accurate in
// absolute terms only.
// Reference: see BetaQuantile
func BetaQuantileComplement(a, b, q float64) float64 {
	x, _ := betaQuantile(a, b, q, true)
	return x
}

// BetaQuantileXY returns x = BetaQuantile(a, b, p) together with y = 1 - x,
// each to full relative accuracy. Near x = 1 the value x rounds to 1 and
// only y still carries the quantile: for example BetaQuantileXY(5, 0.01,
// 0.999) returns x = 1 and y = 1.2539904511230927e-301 (exact:
// 1.2539904511228198e-301 for p = 0.999; the difference is the rounding
// of 0.999 to a double, amplified 100 times by the shape 0.01). The names
// follow RegularizedBetaInc's x and y = 1 - x.
//
// Valid range and NaN cases: as for BetaQuantile; both results are NaN
// together.
// Precision: x as for BetaQuantile. y has relative error at most 3e-15
// over the core range of BetaQuantile and at most 2.5e-13 over the whole
// table (measured 2.2e-15 and 2.2e-13), including where y is below 1e-300.
// The largest errors are at the shape 0.01 with p = 0.95 and 0.999, where y
// solves 1 - I = 1 - p and the double nearest 0.999 differs from 0.999 by
// 8.9e-16 relative in 1 - p, an error the shape 0.01 amplifies about 100
// times.
// Reference: see BetaQuantile
func BetaQuantileXY(a, b, p float64) (x, y float64) {
	return betaQuantile(a, b, p, false)
}

// betaQuantileMaxIter bounds the iterations of betaQuantile. Bisection on
// ln s alone, from the full range [ln 2^-1074, ln(1/2)], reaches the double
// spacing in under 75 steps, and Newton steps only shorten that.
const betaQuantileMaxIter = 200

// betaQuantile solves I_x(a, b) = r (upper false) or 1 - I_x(a, b) = r
// (upper true) and returns x and y = 1 - x.
//
// Of p = I_x and q = 1 - I_x the smaller one, t, is exact in float64 (the
// argument itself, or its complement, which Sterbenz's lemma makes exact
// when it is at most 1/2), and only t enters the residual. The root lies in
// x <= 1/2 when I_{1/2}(a, b) >= p, and otherwise in y <= 1/2; the
// iteration runs in that small variable s, whose shapes make
// F(s) = I_s(as, bs) increasing, and the other variable is formed at the
// end as 1 - s, which is exact to half an ulp because it is at least 1/2.
//
// The residual is g = ln F(s) - ln t when t is F's own tail, and
// g = ln t - ln(1 - F(s)) when it is the other one; both increase in s and
// both are evaluated through incBeta, which returns each tail directly. The
// Newton step is taken in u = ln s:
//
//	dg/du = s f(s) / F   or   s f(s) / (1 - F),
//
// with s f(s) = s^as (1-s)^bs / B(as, bs) / (1-s) taken from
// betaPrefixParts, which evaluates that prefix without cancellation even for
// large shapes. In the tails ln F is nearly linear in ln s (F ~ s^as), so
// Newton in u converges from any start; the update s * exp(-step) adds one
// rounding per step instead of the |ln s| ulps that exp(u) would cost. The
// iterate is kept inside a bracket [lo, hi] that every residual sign
// narrows; a step that leaves it, or that is not at least half as long as
// the one before last, is replaced by bisection of the bracket in u
// (Press et al., rtsafe). The iteration stops when the step or the bracket
// is below a few ulps of s.
func betaQuantile(a, b, r float64, upper bool) (x, y float64) {
	if !(a > 0 && b > 0 && r > 0 && r < 1) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return math.NaN(), math.NaN()
	}
	// t is the exact one of p and q; lowerT reports that it is p.
	t, lowerT := r, !upper
	if r > 0.5 {
		t, lowerT = 1-r, upper
	}

	w, w1 := incBeta(a, b, 0.5, 0.5)
	if math.IsNaN(w) {
		return math.NaN(), math.NaN()
	}
	// Exactly at the median of the lower half: return it without iterating.
	if (lowerT && w == t) || (!lowerT && w1 == t) {
		return 0.5, 0.5
	}
	// sIsX: the root is in x <= 1/2 (I_{1/2} >= p, equivalently 1 - I_{1/2} <= q).
	sIsX := (lowerT && w >= t) || (!lowerT && w1 <= t)
	as, bs := a, b
	if !sIsX {
		as, bs = b, a
	}
	// sLower: t is F(s) = I_s(as, bs) itself rather than 1 - F(s).
	sLower := lowerT == sIsX

	// eval returns g(s) and dg/du at s.
	eval := func(s float64) (g, dg float64) {
		// incBeta and betaPrefixParts take x and y = 1 - x in the original
		// orientation; s is the exact one of the two.
		xx, yy := s, 1-s
		if !sIsX {
			xx, yy = 1-s, s
		}
		v, v1 := incBeta(a, b, xx, yy)
		if math.IsNaN(v) {
			return math.NaN(), math.NaN()
		}
		f, fc := v, v1 // F(s) and 1 - F(s)
		if !sIsX {
			f, fc = v1, v
		}
		e, m := betaPrefixParts(a, b, xx, yy)
		lsf := e + math.Log(m) - math.Log1p(-s) // ln(s f(s))
		if sLower {
			lf := math.Log(f)
			return lf - math.Log(t), math.Exp(lsf - lf)
		}
		lfc := math.Log(fc)
		return math.Log(t) - lfc, math.Exp(lsf - lfc)
	}

	// Bracket: g(hi) >= 0 is known from the test at 1/2. lo = 0 means no
	// point with g < 0 has been seen yet; bisection then steps down from hi
	// by a factor of 2^-32 (the midpoint in ln s of hi and 2^-64 hi), so a
	// root far out in the tail is still reached in a few dozen steps.
	lo, hi := 0.0, 0.5
	bisect := func() float64 {
		if lo == 0 {
			return hi * 0x1p-32
		}
		return math.Sqrt(lo) * math.Sqrt(hi)
	}

	s := betaQuantileStart(as, bs, t, sLower)
	if !(s > 0 && s <= hi) {
		s = bisect()
	}

	step, stepOld := math.Inf(1), math.Inf(1)
	for i := 0; i < betaQuantileMaxIter; i++ {
		g, dg := eval(s)
		switch {
		case math.IsNaN(g):
			return math.NaN(), math.NaN()
		case g == 0:
			return betaQuantileOrient(s, sIsX)
		case g < 0:
			lo = s
		default:
			hi = s
		}
		stepOlder := stepOld
		stepOld = step
		step = g / dg
		// Converged: the Newton correction is below the double spacing, so
		// the step is taken without the bracket test, which a move of less
		// than an ulp off a bracket end would fail.
		if math.Abs(step) <= 0x1p-52 {
			return betaQuantileOrient(s*math.Exp(-step), sIsX)
		}
		if hi-lo <= 4*ulpOf(hi) {
			return betaQuantileOrient(betaQuantileClosest(lo, hi, eval), sIsX)
		}
		next := s * math.Exp(-step)
		if math.IsNaN(step) || !(next > lo && next < hi) || math.Abs(step) > 0.5*math.Abs(stepOlder) {
			next = bisect()
			step = math.Log(s / next)
		}
		if next == s {
			return betaQuantileOrient(s, sIsX)
		}
		s = next
	}
	return betaQuantileOrient(s, sIsX)
}

// betaQuantileClosest returns whichever end of a bracket that has closed to
// a few ulps has the smaller residual; an end that was never evaluated (0)
// loses.
func betaQuantileClosest(lo, hi float64, eval func(float64) (float64, float64)) float64 {
	if lo == 0 {
		return hi
	}
	gl, _ := eval(lo)
	gh, _ := eval(hi)
	if math.Abs(gl) < math.Abs(gh) {
		return lo
	}
	return hi
}

// betaQuantileOrient maps the small variable s back to (x, y = 1 - x).
func betaQuantileOrient(s float64, sIsX bool) (x, y float64) {
	if sIsX {
		return s, 1 - s
	}
	return 1 - s, s
}

// ulpOf returns the spacing of float64 values at |v|.
func ulpOf(v float64) float64 {
	v = math.Abs(v)
	return math.Nextafter(v, math.Inf(1)) - v
}

// betaQuantileStart returns a starting point for s with I_s(as, bs) = t
// (sLower) or 1 - I_s(as, bs) = t. Only the iteration's speed depends on
// it: a start that is not in (0, 1/2) is replaced by bisection.
//
// For as, bs >= 1 it is the normal-deviate approximation of Abramowitz &
// Stegun 26.5.22, s = as / (as + bs e^(2w)), with
//
//	w = y sqrt(h + l)/h - (1/(2bs-1) - 1/(2as-1)) (l + 5/6 - 2/(3h)),
//	h = 2 / (1/(2as-1) + 1/(2bs-1)),  l = (y^2 - 3)/6,
//
// where y is the upper normal deviate of the lower-tail probability. For a
// smaller shape it is the tail approximation I_s(as, bs) ~ s^as/(as B(as, bs))
// solved for s.
func betaQuantileStart(as, bs, t float64, sLower bool) float64 {
	if as >= 1 && bs >= 1 {
		y := standardNormalQuantile(t) // upper deviate of the lower tail 1 - t
		if sLower {
			y = -y
		}
		l := (y*y - 3) / 6
		h := 2 / (1/(2*as-1) + 1/(2*bs-1))
		w := y*math.Sqrt(h+l)/h - (1/(2*bs-1)-1/(2*as-1))*(l+5.0/6-2/(3*h))
		return as / (as + bs*math.Exp(2*w))
	}
	lf := math.Log(t)
	if !sLower {
		lf = math.Log1p(-t)
	}
	lga, _ := math.Lgamma(as)
	lgb, _ := math.Lgamma(bs)
	lgab, _ := math.Lgamma(as + bs)
	return math.Exp((lf + math.Log(as) + lga + lgb - lgab) / as)
}
