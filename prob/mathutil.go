package prob

import "math"

// ---------------------------------------------------------------------------
// Mathematical utility functions for probability distributions.
//
// These are supporting functions needed by the distribution implementations
// in distributions.go. They wrap stdlib where possible and provide custom
// implementations only where the stdlib has no equivalent.
// ---------------------------------------------------------------------------

// LogGamma returns the natural logarithm of the absolute value of Gamma(x).
//
// This is a direct wrapper around math.Lgamma from the Go standard library.
// We expose it here so that the prob package has a single, documented entry
// point for log-gamma computation.
//
// Formula: ln(|Gamma(x)|)
// Valid range: x != 0, -1, -2, ... (poles of Gamma)
// Precision: ~15 significant digits (float64)
// Reference: Lanczos approximation (via Go stdlib)
func LogGamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

// Erfc returns the complementary error function: erfc(x) = 1 - erf(x).
//
// This is a direct wrapper around math.Erfc from the Go standard library.
//
// Formula: (2/sqrt(pi)) * integral from x to inf of exp(-t^2) dt
// Valid range: all float64
// Precision: ~15 significant digits (float64)
// Reference: Abramowitz & Stegun, formula 7.1.2
func Erfc(x float64) float64 {
	return math.Erfc(x)
}

// RegularizedBetaInc computes the regularized incomplete beta function
// I_x(a, b), defined as:
//
//	I_x(a,b) = B(x; a, b) / B(a, b)
//
// where B(x; a, b) is the incomplete beta function and B(a, b) is the
// complete beta function.
//
// Whichever of I and 1 - I is the smaller tail is computed directly and the
// other as its complement, by the method of DiDonato & Morris (1992): a power
// series where x*b is small, a continued fraction whose terms are written in
// lambda = a - (a+b)x so that they do not cancel near the mean, and, for one
// large and one small parameter, an asymptotic expansion in the incomplete
// gamma function. Each is scaled by x^a (1-x)^b / B(a, b), evaluated around
// the mean without cancellation. I_x(a, 1) = x^a and I_x(1, b) = 1 - (1-x)^b
// are evaluated in closed form.
//
// Formula: I_x(a,b) via continued fraction (DLMF 8.17.22), power series
// (DLMF 8.17.7) and the DiDonato-Morris asymptotic expansion
// Valid range: x in [0, 1], a > 0, b > 0
// Precision: measured against 60-digit references for a, b from 1e-3 to
// 1e7, with x from the far tails through the mean: absolute error at most
// 2e-15; relative error at most 8e-15 for results >= 1e-10 and below 1e-13
// down to 1e-150. Below 1e-150 the relative error is set by the float64
// rounding of the exponent and grows in proportion to |ln I|, about
// 5e-16*|ln I| (2.6e-13 near 1e-263). Near the mean of still larger
// parameters the continued fraction's rounding grows slowly with its
// iteration count (2.7e-15 at a = b = 1e10, 1.5e-13 at a = b = 1e15).
// Cost: O(1) in the tails; near the mean about 5.5*min(a, b)^(1/3)
// continued-fraction iterations.
// Failure mode: returns NaN if a <= 0, b <= 0, a or b is infinite, or x is
// outside [0, 1]; also NaN near the mean of parameters above 2^52, where
// a+n is no longer exact in float64 and the continued fraction would lose
// accuracy, or if an expansion does not converge within its budget
// Reference: DiDonato, A.R. & Morris, A.H. (1992) "Algorithm 708:
// Significant digit computation of the incomplete beta function ratios",
// ACM TOMS 18(3); Lentz, W.J. (1976); Press et al., Numerical Recipes,
// 3rd ed., section 6.4
func RegularizedBetaInc(x, a, b float64) float64 {
	if !(x >= 0 && x <= 1 && a > 0 && b > 0) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return math.NaN()
	}
	w, _ := incBeta(a, b, x, 1-x)
	return w
}

// incBeta returns w = I_x(a, b) and w1 = 1 - w for finite a, b > 0 and
// x in [0, 1]. The caller supplies y = 1 - x; whichever of x and y is below
// 1/2 must be accurate to a few ulp in its own right (not rounded from the
// other), because the tails are computed from it. A non-converging
// expansion yields NaN for both.
//
// The choice of method follows DiDonato & Morris (1992, bratio): the power
// series (betaSeries), the continued fraction (betaCF), the asymptotic
// expansion for a large first and a small second parameter (betaAsymLargeA),
// and the finite sums that shift a parameter by an integer (betaShift).
// Their asymptotic expansion for two large parameters is not used: near the
// mean the continued fraction converges in O(sqrt(min(a, b))) iterations to
// full accuracy.
func incBeta(a, b, x, y float64) (w, w1 float64) {
	switch {
	case x <= 0:
		return 0, 1
	case y <= 0:
		return 1, 0
	case b == 1: // I_x(a, 1) = x^a
		lx, _ := betaLogs(x, y)
		e := a * lx
		return math.Exp(e), -math.Expm1(e)
	case a == 1: // I_x(1, b) = 1 - y^b
		_, ly := betaLogs(x, y)
		e := b * ly
		return -math.Expm1(e), math.Exp(e)
	}
	var v, v1 float64 // I_x0(a0, b0) and its complement
	var ok bool
	var swap bool
	if math.Min(a, b) <= 1 {
		swap, v, v1, ok = incBetaSmallParam(a, b, x, y)
	} else {
		swap, v, v1, ok = incBetaLargeParams(a, b, x, y)
	}
	if !ok {
		return math.NaN(), math.NaN()
	}
	if swap {
		return v1, v
	}
	return v, v1
}

// incBetaSmallParam handles min(a, b) <= 1. It orients the problem so that
// x0 <= 1/2 and returns v = I_x0(a0, b0), v1 = 1 - v, each computed directly
// where it is the smaller.
func incBetaSmallParam(a, b, x, y float64) (swap bool, v, v1 float64, ok bool) {
	a0, b0, x0, y0 := a, b, x, y
	if x > 0.5 {
		swap = true
		a0, b0, x0, y0 = b, a, y, x
	}
	lower := func() (bool, float64, float64, bool) {
		s, ok := betaSeries(a0, b0, x0, y0)
		return swap, s, 1 - s, ok
	}
	upper := func() (bool, float64, float64, bool) {
		s, ok := betaSeries(b0, a0, y0, x0)
		return swap, 1 - s, s, ok
	}
	if math.Max(a0, b0) > 1 {
		switch {
		case b0 <= 1:
			return lower()
		case x0 >= 0.29:
			return upper()
		case x0 < 0.1 && math.Pow(x0*b0, a0) <= 0.7:
			return lower()
		case b0 > 15:
			s, ok := betaAsymLargeA(b0, a0, y0, x0, 0)
			return swap, 1 - s, s, ok
		}
	} else {
		switch {
		case a0 >= math.Min(0.2, b0), math.Pow(x0, a0) <= 0.9:
			return lower()
		case x0 >= 0.3:
			return upper()
		}
	}
	// Raise b0 by 20 with a finite sum, then use the asymptotic expansion.
	s, ok := betaShift(b0, a0, y0, x0, 20)
	if ok {
		s, ok = betaAsymLargeA(b0+20, a0, y0, x0, s)
	}
	return swap, 1 - s, s, ok
}

// incBetaLargeParams handles a, b > 1. It orients the problem so that x0 is
// at or below the mean (lambda = a0 - (a0+b0)x0 >= 0) and returns
// v = I_x0(a0, b0) and v1 = 1 - v.
func incBetaLargeParams(a, b, x, y float64) (swap bool, v, v1 float64, ok bool) {
	lambda := betaLambda(a, b, x, y)
	a0, b0, x0, y0 := a, b, x, y
	if lambda < 0 {
		swap = true
		a0, b0, x0, y0, lambda = b, a, y, x, -lambda
	}
	switch {
	case b0 >= 40:
		v, ok = betaCF(a0, b0, x0, y0, lambda)
	case b0*x0 <= 0.7:
		v, ok = betaSeries(a0, b0, x0, y0)
	default:
		// Split b0 = n + f with f in (0, 1]:
		// I_x0(a0, b0) = I_x0(a0, f) + [I_y0(f, a0) - I_y0(f+n, a0)].
		n := math.Floor(b0)
		f := b0 - n
		if f == 0 {
			n--
			f = 1
		}
		v, ok = betaShift(f, a0, y0, x0, int(n))
		if !ok {
			break
		}
		if x0 <= 0.7 {
			var s float64
			s, ok = betaSeries(a0, f, x0, y0)
			v += s
			break
		}
		aa := a0
		if a0 <= 15 {
			var s float64
			s, ok = betaShift(a0, f, x0, y0, 20)
			v += s
			aa += 20
		}
		if ok {
			v, ok = betaAsymLargeA(aa, f, x0, y0, v)
		}
	}
	return swap, v, 1 - v, ok
}

// betaShift returns I_x(a, b) - I_x(a+n, b) for an integer n >= 1, the
// finite sum
//
//	sum_{i=0}^{n-1} x^(a+i) y^b / ((a+i) B(a+i, b))
//
// whose terms satisfy t_(i+1) = t_i * x (a+b+i) / (a+1+i) (DiDonato &
// Morris 1992, bup).
func betaShift(a, b, x, y float64, n int) (float64, bool) {
	if n < 1 {
		return 0, true
	}
	sum := 1.0
	d := 1.0
	for i := 0; i < n-1; i++ {
		fi := float64(i)
		d *= (a + b + fi) / (a + 1 + fi) * x
		sum += d
	}
	e, m := betaPrefixParts(a, b, x, y)
	return scaleExp(e, m*sum/a), true
}

// betaAsymLargeA returns w + I_x(a, b) for a >= 15 and 0 < b <= 1, from the
// asymptotic expansion of DiDonato & Morris (1992, section 9; bgrat) in terms
// of the incomplete gamma function:
//
//	I_x(a, b) = Gamma(a+b) / (Gamma(a) T^b) * sum_{n>=0} d_n J_n(b, z),
//	T = a + (b-1)/2,  z = -T ln x,
//
// where J_0 = Q(b, z) e^z z^-b Gamma(b) and the J_n and d_n follow the
// recurrences of that paper. The terms fall like (4T^2)^-n.
func betaAsymLargeA(a, b, x, y, w float64) (float64, bool) {
	bm1 := b - 1
	nu := a + 0.5*bm1
	lnx, _ := betaLogs(x, y)
	z := -nu * lnx
	if b*z == 0 {
		return math.NaN(), false
	}
	lgb, _ := math.Lgamma(b)
	logR := b*math.Log(z) - z - lgb // ln(z^b e^-z / Gamma(b))
	// ln of Gamma(a+b) / (Gamma(a) nu^b) * r
	logU := logR - lnGammaRatioRem(b, a) + b*math.Log1p((b+1)/(2*nu))
	l := 0.0
	if w > 0 {
		l = math.Exp(math.Log(w) - logU)
	}
	qr, ok := gammaQOverR(b, z)
	if !ok {
		return math.NaN(), false
	}
	v := 0.25 / (nu * nu)
	t2 := 0.25 * lnx * lnx
	j := qr
	sum := j
	t, cn, n2 := 1.0, 1.0, 0.0
	var c, d [30]float64
	converged := false
	for n := 1; n <= len(c); n++ {
		bp2n := b + n2
		j = (bp2n*(bp2n+1)*j + (z+bp2n+1)*t) * v
		n2 += 2
		t *= t2
		cn /= n2 * (n2 + 1)
		c[n-1] = cn
		s := 0.0
		coef := b - float64(n)
		for i := 1; i < n; i++ {
			s += coef * c[i-1] * d[n-1-i]
			coef += b
		}
		d[n-1] = bm1*cn + s/float64(n)
		dj := d[n-1] * j
		sum += dj
		if sum <= 0 {
			return math.NaN(), false
		}
		if math.Abs(dj) <= 0x1p-53*(sum+l) {
			converged = true
			break
		}
	}
	if !converged {
		return math.NaN(), false
	}
	return w + scaleExp(logU, sum), true
}

// gammaQOverR returns Q(b, z) e^z z^-b Gamma(b), the upper regularized
// incomplete gamma function scaled by its leading behaviour, without
// underflow for large z: in the continued-fraction region this is exactly
// the continued fraction's value.
func gammaQOverR(b, z float64) (float64, bool) {
	if z >= math.Max(b, 1.5) {
		return gammaCF(b, z)
	}
	_, q := incGammaPQ(b, z)
	lgb, _ := math.Lgamma(b)
	return q / math.Exp(b*math.Log(z)-z-lgb), !math.IsNaN(q)
}

// betaLogs returns ln(x) and ln(y) for y = 1 - x, each computed from
// whichever of x and y is below 1/2 (and therefore exact).
func betaLogs(x, y float64) (lx, ly float64) {
	if x <= 0.5 {
		return math.Log(x), math.Log1p(-x)
	}
	return math.Log1p(-y), math.Log(y)
}

// betaLambda returns lambda = a - (a+b)x = (a+b)y - b without cancellation:
// a+b is carried exactly as a two-term sum and its product with the small
// one of x and y is formed with a single rounding (math.FMA).
func betaLambda(a, b, x, y float64) float64 {
	sh, sl := twoSum(a, b)
	if x <= 0.5 {
		return math.FMA(-sh, x, a) - sl*x
	}
	return math.FMA(sh, y, -b) + sl*y
}

// betaPrefixParts returns e and m with x^a y^b / B(a, b) = exp(e) * m for
// y = 1 - x (DiDonato & Morris 1992, brcomp), where m is of moderate size,
// so that callers can fold further factors into m before exponentiating
// (scaleExp) and a prefix that would underflow on its own is not rounded to
// a subnormal first.
//
// For a, b >= 8 it is evaluated around the mean x0 = a/(a+b):
//
//	x^a y^b / B(a,b) = sqrt(b*x0/(2 pi)) * exp(a*log1pmx(-lambda/a)
//	                   + b*log1pmx(lambda/b) - bcorr(a, b)),
//
// with lambda = a - (a+b)x and bcorr(a, b) = stirlerr(a) + stirlerr(b) -
// stirlerr(a+b): the O(a) and O(b) terms of the direct form cancel exactly
// and are never built. When one parameter is below 8 and the other is not,
// ln Gamma(q) - ln Gamma(p+q) is evaluated by its Stirling form
// (lnGammaRatioRem), which also avoids cancellation.
func betaPrefixParts(a, b, x, y float64) (e, m float64) {
	if x <= 0 || y <= 0 {
		return math.Inf(-1), 1
	}
	if a >= 8 && b >= 8 {
		lambda := betaLambda(a, b, x, y)
		// u = ln(x/x0) - (x-x0)/x0 and v = ln(y/y0) - (y-y0)/y0, where
		// x/x0 = x(a+b)/a and y/y0 = y(a+b)/b are formed directly.
		var u, v float64
		if e := -lambda / a; e < -0.5 {
			u = math.Log(x*(a+b)/a) - e
		} else {
			u = log1pmx(e)
		}
		if e := lambda / b; e < -0.5 {
			v = math.Log(y*(a+b)/b) - e
		} else {
			v = log1pmx(e)
		}
		return a*u + b*v - bcorr(a, b), math.Sqrt(b*(a/(a+b))) / sqrt2Pi
	}
	if a < 8 && b < 8 {
		lx, ly := betaLogs(x, y)
		la, _ := math.Lgamma(a)
		lb, _ := math.Lgamma(b)
		lab, _ := math.Lgamma(a + b)
		return a*lx + b*ly - (la + lb - lab), 1
	}
	// One parameter p below 8 and the other, q, at least 8; s is p's
	// variable and t is q's. With ln Gamma(q) - ln Gamma(p+q) =
	// -p ln(p+q) + lnGammaRatioRem(p, q):
	//
	//	ln(s^p t^q / B(p,q)) = p ln(s (p+q)) + q ln t - ln Gamma(p) - lnGammaRatioRem(p, q),
	//
	// so p ln s and p ln(p+q), which nearly cancel near the mean, are
	// combined inside a single logarithm.
	p, q, s, t := a, b, x, y
	if b < a {
		p, q, s, t = b, a, y, x
	}
	ls, lt := betaLogs(s, t)
	pq := p + q
	var ps float64
	if s <= 0.5 {
		ps = p * math.Log(s*pq)
	} else {
		ps = p * (ls + math.Log(pq))
	}
	lp, _ := math.Lgamma(p)
	return ps + q*lt - lp - lnGammaRatioRem(p, q), 1
}

// lnGammaRatioRem returns ln Gamma(q) - ln Gamma(p+q) + p ln(p+q) for p > 0,
// q >= 8, from Stirling's formula:
//
//	-(q - 1/2)*log1pmx(p/q) + p/(2q) + stirlerr(q) - stirlerr(p+q),
//
// none of whose terms cancel (DiDonato & Morris 1992, algdiv).
func lnGammaRatioRem(p, q float64) float64 {
	return -(q-0.5)*log1pmx(p/q) + p/(2*q) + stirlerr(q) - stirlerr(p+q)
}

// bcorr returns stirlerr(a) + stirlerr(b) - stirlerr(a+b) for a, b >= 8.
func bcorr(a, b float64) float64 {
	return stirlerr(a) + stirlerr(b) - stirlerr(a+b)
}

// sqrt2Pi is sqrt(2*pi).
const sqrt2Pi = 2.506628274631000502415765

// betaSeries returns I_x(a, b) by the power series (DLMF 8.17.7 after
// Euler's transformation; DiDonato & Morris 1992, bpser)
//
//	I_x(a,b) = x^a / (a B(a,b)) * (1 + a sum_{n>=1} c_n / (a+n)),
//	c_n = (1 - b/1)(1 - b/2)...(1 - b/n) x^n,
//
// used for x <= 1/2 with b*x <= 0.7 or b <= 1, where the terms fall at least
// like x^n and their sum cannot cancel by more than a factor of about e^1.4.
func betaSeries(a, b, x, y float64) (float64, bool) {
	_, ly := betaLogs(x, y)
	e, m := betaPrefixParts(a, b, x, y)
	e -= b * ly // exp(e) * m = x^a / B(a,b)
	if math.IsInf(e, -1) {
		return 0, true
	}
	sum := 0.0
	c := 1.0
	for n := 1.0; n <= 2000; n++ {
		c *= (1 - b/n) * x
		w := c / (a + n)
		sum += w
		if math.Abs(w)*a <= 0x1p-56*math.Abs(1+a*sum) {
			return scaleExp(e, m*(1+a*sum)/a), true
		}
	}
	return math.NaN(), false
}

// betaCF returns I_x(a, b) for x at or below the mean (lambda = a - (a+b)x
// >= 0) by the DiDonato-Morris continued fraction (ACM TOMS 708, bfrac),
// evaluated by forward recurrence with rescaling. Its coefficients are
// written in terms of lambda, so unlike the classical form (Numerical
// Recipes betacf) they do not cancel near the mean. Near the mean it needs
// O(sqrt(min(a, b))) iterations. Reports ok = false if the budget is
// exhausted.
func betaCF(a, b, x, y, lambda float64) (float64, bool) {
	pe, pm := betaPrefixParts(a, b, x, y)
	if scaleExp(pe, pm) == 0 {
		return 0, true
	}
	if a > 0x1p52 {
		// a+1+2n is no longer exact, so the coefficients lose accuracy.
		return math.NaN(), false
	}
	c := lambda + 1
	c0 := b / a
	c1 := 1/a + 1
	yp1 := y + 1
	p := 1.0
	s := a + 1
	an, bn := 0.0, 1.0
	anp1, bnp1 := 1.0, c/c1
	r := c1 / c
	maxIter := iterBudget(1000, 20, math.Min(a, b))
	for n := 1.0; n <= float64(maxIter); n++ {
		t := n / a
		w := n * (b - n) * x
		e := a / s
		alpha := p * (p + c0) * e * e * (w * x)
		e = (t + 1) / (c1 + t + t)
		beta := n + w/s + e*(c+n*yp1)
		p = t + 1
		s += 2
		t = alpha*an + beta*anp1
		an, anp1 = anp1, t
		t = alpha*bn + beta*bnp1
		bn, bnp1 = bnp1, t
		r0 := r
		r = anp1 / bnp1
		if math.Abs(r-r0) <= 0x1p-52*r {
			return scaleExp(pe, pm*r), true
		}
		an /= bnp1
		bn /= bnp1
		anp1 = r
		bnp1 = 1
	}
	return math.NaN(), false
}

// studentTCDF computes the CDF of the Student's t-distribution with df
// degrees of freedom, evaluated at t. Used by hypothesis tests for p-value
// computation.
//
// Formula: CDF(t; df) = I_{x}(df/2, 1/2) where x = df/(df + t^2),
//
//	using the regularized incomplete beta function.
//	For t >= 0: CDF = 1 - 0.5 * I_x(df/2, 1/2)
//	For t < 0: CDF = 0.5 * I_x(df/2, 1/2)
//
// Valid range: any t, df > 0
// Precision: relative error below 1e-13 in the lower tail (t < 0) wherever
// the result is at least 1e-150; absolute error below 1e-15 otherwise
// Reference: Abramowitz & Stegun, formula 26.5.27
func studentTCDF(t float64, df float64) float64 {
	if df <= 0 {
		return math.NaN()
	}
	tail := studentTTwoSided(t, df)
	if t >= 0 {
		return 1.0 - 0.5*tail
	}
	return 0.5 * tail
}

// studentTTwoSided returns the two-sided tail probability P(|T| >= |t|) of
// the Student's t-distribution with df > 0 degrees of freedom,
// I_x(df/2, 1/2) with x = df/(df+t^2), computed directly (not as 1 - CDF)
// so that small p-values keep their relative accuracy. Both x and
// 1 - x = t^2/(df+t^2) are formed from their own ratio, without rounding one
// from the other.
func studentTTwoSided(t, df float64) float64 {
	if math.IsNaN(t) || math.IsNaN(df) || df <= 0 {
		return math.NaN()
	}
	tt := t * t
	var x, y float64
	switch {
	case math.IsInf(df, 1):
		return math.Erfc(math.Abs(t) / math.Sqrt2)
	case math.IsInf(tt, 1):
		return 0
	case tt > df:
		r := df / tt
		x, y = r/(1+r), 1/(1+r)
	default:
		r := tt / df
		x, y = 1/(1+r), r/(1+r)
	}
	w, _ := incBeta(df/2, 0.5, x, y)
	return w
}

// ---------------------------------------------------------------------------
// Regularized incomplete gamma function.
// ---------------------------------------------------------------------------

// regularizedGammaP computes the lower regularized incomplete gamma function
// P(a, x) = gamma(a, x) / Gamma(a) for a > 0, x >= 0 (see incGammaPQ).
// Returns 0 for x <= 0 or a <= 0, and NaN if the expansion does not converge.
func regularizedGammaP(a, x float64) float64 {
	if x <= 0 || a <= 0 {
		return 0
	}
	p, _ := incGammaPQ(a, x)
	return p
}

// regularizedGammaQ computes the upper regularized incomplete gamma function
// Q(a, x) = 1 - P(a, x) = Gamma(a, x) / Gamma(a) for a > 0, x >= 0
// (see incGammaPQ). A small Q is computed directly, never as 1 - P, so
// upper-tail p-values keep their relative accuracy. Returns NaN for a <= 0,
// 1 for x <= 0, and NaN if the expansion does not converge.
func regularizedGammaQ(a, x float64) float64 {
	if a <= 0 {
		return math.NaN()
	}
	if x <= 0 {
		return 1.0
	}
	_, q := incGammaPQ(a, x)
	return q
}

// incGammaPQ returns P(a, x) and Q(a, x) = 1 - P(a, x) for a > 0, x > 0.
//
// Whichever of P and Q is smaller is computed directly and the other as its
// complement, so neither loses relative accuracy in its own tail:
//   - x < 1.5: the power series for P; for a < 1 also a direct series for Q
//     (Q is then small and 1 - P would cancel), otherwise Q = 1 - P.
//   - 1.5 <= x < a: the power series for P, and Q = 1 - P.
//   - x >= max(a, 1.5): the Legendre continued fraction for Q, and P = 1 - Q.
//
// Both expansions are scaled by x^a e^-x / Gamma(a+1), computed without
// cancellation by gammaLogPrefix. Near x = a both need O(sqrt(a)) terms; the
// series term recurrence is carried in double-double arithmetic so that its
// rounding error does not grow with the number of terms. If an expansion
// does not converge within its iteration budget (which also grows as
// sqrt(a)), both results are NaN rather than a plausible-looking number.
//
// Reference: DiDonato, A.R. & Morris, A.H. (1986) "Computation of the
// incomplete gamma function ratios and their inverse", ACM TOMS 12(4);
// Gil, A., Segura, J. & Temme, N.M. (2012) "Efficient and accurate
// algorithms for the computation and inversion of the incomplete gamma
// function ratios", SIAM J. Sci. Comput. 34(6); DLMF 8.7.1 and 8.9.2.
func incGammaPQ(a, x float64) (p, q float64) {
	switch {
	case math.IsNaN(a) || math.IsNaN(x):
		return math.NaN(), math.NaN()
	case math.IsInf(a, 1):
		if math.IsInf(x, 1) {
			return math.NaN(), math.NaN()
		}
		return 0, 1
	case math.IsInf(x, 1):
		return 1, 0
	}
	if x < 1.5 || x < a {
		s, ok := gammaSeries(a, x)
		if !ok {
			return math.NaN(), math.NaN()
		}
		p = scaleExp(gammaLogPrefix(a, x), s)
		if x < 1.5 && a < 1 {
			return p, gammaQSmallA(a, x)
		}
		return p, 1 - p
	}
	h, ok := gammaCF(a, x)
	if !ok {
		return math.NaN(), math.NaN()
	}
	q = scaleExp(gammaLogPrefix(a, x), a*h)
	return 1 - q, q
}

// gammaLogPrefix returns ln(x^a e^-x / Gamma(a+1)) for a > 0, x > 0.
//
// For a >= 10 it is evaluated as
//
//	a*log1pmx((x-a)/a) - ln(2*pi*a)/2 - stirlerr(a),
//
// where log1pmx(d) = ln(1+d) - d and stirlerr is the Stirling-series
// remainder of ln Gamma(a+1). The two O(a) terms a*ln(x) and x of the
// direct form cancel near x = a; this form never builds them, so its
// relative error is a few ulp times the size of the exponent itself (about
// 1e-15 near the peak, growing only as the result itself becomes tiny).
func gammaLogPrefix(a, x float64) float64 {
	if a < 10 {
		return a*math.Log(x) - x - lgamma1p(a)
	}
	d := (x - a) / a // exact numerator for a/2 <= x <= 2a
	var t float64    // ln(x/a) - (x-a)/a
	if d >= -0.5 && d <= 1 {
		t = log1pmx(d)
	} else {
		t = math.Log(x/a) - d
	}
	return a*t - 0.5*math.Log(2*math.Pi*a) - stirlerr(a)
}

// gammaSeries returns S = sum_{n>=0} x^n / ((a+1)(a+2)...(a+n)), so that
// P(a, x) = exp(gammaLogPrefix(a, x)) * S (DLMF 8.7.1). The terms decrease once
// n > x - a, which holds from the start whenever x < a + 1.
//
// Near x = a about 9*sqrt(a) terms are needed. Each term and the running sum
// are kept as unevaluated double-double pairs (error-free transformations
// built on math.FMA), so the result is accurate to about 1 ulp however many
// terms are summed. Reports ok = false if the budget is exhausted.
func gammaSeries(a, x float64) (s float64, ok bool) {
	maxIter := iterBudget(100, 20, a)
	sh, sl := 1.0, 0.0 // running sum
	th, tl := 1.0, 0.0 // current term
	for n := 1; n <= maxIter; n++ {
		// q = x / (a + n), with a + n held exactly as dh + dl.
		dh, dl := twoSum(a, float64(n))
		qh := x / dh
		ql := (math.FMA(-qh, dh, x) - qh*dl) / dh
		// term *= q
		ph := float64(th * qh)
		pl := math.FMA(th, qh, -ph) + (th*ql + tl*qh)
		th, tl = fastTwoSum(ph, pl)
		// sum += term
		var e float64
		sh, e = twoSum(sh, th)
		sl += e + tl
		if th <= 0x1p-60*sh {
			return sh + sl, true
		}
	}
	return math.NaN(), false
}

// gammaCF returns the Legendre continued fraction
//
//	h = 1/(x+1-a- 1(1-a)/(x+3-a- 2(2-a)/(x+5-a- ...)))
//
// evaluated by the modified Lentz method, so that
// Q(a, x) = exp(gammaLogPrefix(a, x)) * a * h (DLMF 8.9.2; Numerical Recipes 3e,
// section 6.2). It is used for x >= max(a, 1.5), where it needs at most
// about 0.7*sqrt(a) + 50 iterations. Reports ok = false if the budget is
// exhausted.
func gammaCF(a, x float64) (h float64, ok bool) {
	const tiny = 1e-300
	maxIter := iterBudget(500, 10, a)
	b := (x - a) + 1.0 // x - a is exact near the peak, so the 1 is not lost at large a
	c := 1.0 / tiny
	d := 1.0 / b
	h = d
	for i := 1; i <= maxIter; i++ {
		fi := float64(i)
		an := -fi * (fi - a)
		b += 2.0
		d = an*d + b
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = b + an/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		del := d * c
		h *= del
		if math.Abs(del-1.0) <= 0x1p-53 {
			return h, true
		}
	}
	return math.NaN(), false
}

// gammaQSmallA returns Q(a, x) for 0 < a < 1 and 0 < x < 1.5, where Q is
// small and must not be formed as 1 - P:
//
//	Q = -expm1(E) + u*a*sum_{n>=1} (-1)^(n+1) x^n / (n! (a+n)),
//	E = a ln(x) - ln Gamma(1+a),  u = exp(E).
//
// This follows from the term-by-term integral
// gamma(a, x) = sum_{n>=0} (-1)^n x^(a+n) / (n! (a+n)) (DLMF 8.7.1); the
// alternating series converges like x^n/n!.
func gammaQSmallA(a, x float64) float64 {
	e := a*math.Log(x) - lgamma1p(a)
	sum := 0.0
	term := 1.0
	for n := 1; n <= 60; n++ {
		term *= -x / float64(n)
		v := -term / (a + float64(n)) // (-1)^(n+1) x^n / (n! (a+n))
		sum += v
		if math.Abs(v) <= 0x1p-60*math.Abs(sum) {
			break
		}
	}
	return -math.Expm1(e) + math.Exp(e)*a*sum
}

// lgamma1p returns ln Gamma(1+a) for a > -1, accurately also for small |a|
// (where forming 1+a first would discard the low bits of a). For |a| < 0.5
// it sums the Taylor series
//
//	ln Gamma(1+a) = -gamma_E*a - log1pmx(a) + sum_{k>=2} (-1)^k (zeta(k)-1) a^k / k,
//
// whose terms fall like (a/2)^k.
func lgamma1p(a float64) float64 {
	if math.Abs(a) >= 0.5 {
		v, _ := math.Lgamma(1 + a)
		return v
	}
	// (zeta(k)-1)/k for k = 2..41; reference values computed with mpmath.
	coeffs := [...]float64{
		3.22467033424113218236e-1, 6.73523010531980951332e-2, 2.0580808427784547879e-2,
		7.38555102867398526627e-3, 2.89051033074152328575e-3, 1.19275391170326097711e-3,
		5.09669524743042422336e-4, 2.23154758453579379761e-4, 9.94575127818085337146e-5,
		4.49262367381331417002e-5, 2.05072127756706915532e-5, 9.43948827526839590399e-6,
		4.37486678990748780418e-6, 2.03921575380136623678e-6, 9.55141213040741983286e-7,
		4.49246919876456604329e-7, 2.12071848055546658692e-7, 1.00432248239680996087e-7,
		4.76981016936398056576e-8, 2.27110946089431649103e-8, 1.08386592148969540911e-8,
		5.18347504197004665512e-9, 2.48367454380247831719e-9, 1.19214014058609120744e-9,
		5.73136724167886201333e-10, 2.75952288512423314518e-10, 1.33047643742444894815e-10,
		6.42296456383810002208e-11, 3.10442477473222727624e-11, 1.50213840807541421709e-11,
		7.2759744802390796625e-12, 3.52774247657591508362e-12, 1.7119917905596179086e-12,
		8.3153858414202848198e-13, 4.04220052528944006554e-13, 1.96647563109661649041e-13,
		9.57363038783855576378e-14, 4.66407602642837422458e-14, 2.27373696006597232063e-14,
		1.10913994708345220166e-14,
	}
	const eulerGamma = 0.5772156649015328606065121
	// Horner in -a: sum_{k>=2} c_k (-a)^k = a^2 * (c_2 - a*(c_3 - a*(...))).
	s := 0.0
	for i := len(coeffs) - 1; i >= 0; i-- {
		s = coeffs[i] - a*s
	}
	return a*(a*s-eulerGamma) - log1pmx(a)
}

// log1pmx returns ln(1+d) - d for d > -1, without the cancellation of the
// direct form for small |d|. For -0.5 <= d <= 1 it uses u = d/(2+d), so that
// ln(1+d) = 2*atanh(u) and ln(1+d) - d = -u*d + 2u^3 * sum_{k>=0} u^(2k)/(2k+3),
// a series in u^2 <= 1/9.
func log1pmx(d float64) float64 {
	if d < -0.5 || d > 1 {
		return math.Log1p(d) - d
	}
	u := d / (2 + d)
	u2 := u * u
	s := 0.0
	pow := 1.0
	for k := 0; k < 40; k++ {
		v := pow / float64(2*k+3)
		s += v
		if v <= 0x1p-60*s {
			break
		}
		pow *= u2
	}
	return -u*d + 2*u*u2*s
}

// stirlerr returns ln Gamma(a+1) - (a+1/2) ln(a) + a - ln(2*pi)/2, the
// remainder of Stirling's approximation (equivalently ln Gamma(a) -
// (a-1/2) ln(a) + a - ln(2*pi)/2). For a >= 8 it sums the asymptotic series
// sum_k B_2k / (2k (2k-1) a^(2k-1)) through the a^-19 term (truncation error
// below 2e-18 at a = 8); for smaller a it is taken directly from math.Lgamma.
func stirlerr(a float64) float64 {
	if a < 8 {
		v, _ := math.Lgamma(a + 1)
		return v - (a+0.5)*math.Log(a) + a - lnSqrt2Pi
	}
	const (
		s1  = 1.0 / 12
		s3  = 1.0 / 360
		s5  = 1.0 / 1260
		s7  = 1.0 / 1680
		s9  = 1.0 / 1188
		s11 = 691.0 / 360360
		s13 = 1.0 / 156
		s15 = 3617.0 / 122400
		s17 = 43867.0 / 244188
		s19 = 174611.0 / 125400
	)
	r := 1 / a
	r2 := r * r
	return r * (s1 - r2*(s3-r2*(s5-r2*(s7-r2*(s9-r2*(s11-r2*(s13-r2*(s15-r2*(s17-r2*s19)))))))))
}

// lnSqrt2Pi is ln(sqrt(2*pi)).
const lnSqrt2Pi = 0.9189385332046727417803297

// iterBudget returns the iteration budget base + scale*sqrt(a) for an
// expansion whose worst-case term count grows as sqrt(a), capped at 1e7.
func iterBudget(base, scale, a float64) int {
	n := base + scale*math.Sqrt(a)
	if !(n < 1e7) { // also catches NaN
		return 1e7
	}
	return int(n)
}

// scaleExp returns exp(e) * m for m >= 0. When exp(e) alone would be
// subnormal or zero, the factor is folded into the exponent first, so a
// result in the normal range does not inherit the lost precision of a
// subnormal intermediate.
func scaleExp(e, m float64) float64 {
	if m == 0 || math.IsInf(e, -1) {
		return 0
	}
	if e > -700 {
		return math.Exp(e) * m
	}
	return math.Exp(e + math.Log(m))
}

// twoSum returns s = fl(a+b) and the exact rounding error e, a+b = s+e
// (Knuth). It uses no multiplications, so FMA contraction cannot alter it.
func twoSum(a, b float64) (s, e float64) {
	s = a + b
	bb := s - a
	e = (a - (s - bb)) + (b - bb)
	return s, e
}

// fastTwoSum returns s = fl(a+b) and the exact error e for |a| >= |b|
// (Dekker).
func fastTwoSum(a, b float64) (s, e float64) {
	s = a + b
	e = b - (s - a)
	return s, e
}
