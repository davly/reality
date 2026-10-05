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
// The implementation uses Lentz's continued fraction method for the
// incomplete beta function, with the symmetry relation
// I_x(a, b) = 1 - I_{1-x}(b, a) to ensure convergence.
//
// Formula: I_x(a,b) via continued fraction (DLMF 8.17.22)
// Valid range: x in [0, 1], a > 0, b > 0
// Precision: ~1e-14 absolute for typical inputs
// Failure mode: returns NaN if a <= 0, b <= 0, or x outside [0, 1]
// Reference: Lentz, W.J. (1976) "Generating Bessel functions in Mie
// scattering calculations using continued fractions"; Press et al.,
// Numerical Recipes, 3rd ed., section 6.4
func RegularizedBetaInc(x, a, b float64) float64 {
	if x < 0 || x > 1 || a <= 0 || b <= 0 {
		return math.NaN()
	}
	if x == 0 {
		return 0
	}
	if x == 1 {
		return 1
	}

	// Use the symmetry relation for faster convergence:
	// When x > (a+1)/(a+b+2), evaluate 1 - I_{1-x}(b, a) instead.
	if x > (a+1)/(a+b+2) {
		return 1.0 - RegularizedBetaInc(1-x, b, a)
	}

	// Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
	lnPrefactor := a*math.Log(x) + b*math.Log(1-x) - math.Log(a) -
		(LogGamma(a) + LogGamma(b) - LogGamma(a+b))

	// Evaluate continued fraction using Lentz's method.
	return math.Exp(lnPrefactor) * betaCF(x, a, b)
}

// betaCF evaluates the continued fraction for the incomplete beta function
// using the modified Lentz algorithm. The continued fraction is:
//
//	1 / (1 + d1/(1 + d2/(1 + ...)))
//
// where the coefficients d_m are defined as in Numerical Recipes eq. 6.4.5.
//
// maxIter limits iterations to prevent infinite loops. The tiny constant
// prevents division by zero in the Lentz algorithm.
func betaCF(x, a, b float64) float64 {
	const maxIter = 200
	const eps = 1e-14
	const tiny = 1e-30

	// Lentz's method, started from C_0 = 1 and D_1 = 1/d (f = D_1).
	c := 1.0
	d := 1.0 - (a+b)*x/(a+1)
	if math.Abs(d) < tiny {
		d = tiny
	}
	d = 1.0 / d
	f := d

	for m := 1; m <= maxIter; m++ {
		mf := float64(m)

		// Even step: d_{2m} coefficient.
		num := mf * (b - mf) * x / ((a + 2*mf - 1) * (a + 2*mf))
		d = 1.0 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1.0 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		f *= c * d

		// Odd step: d_{2m+1} coefficient.
		num = -(a + mf) * (a + b + mf) * x / ((a + 2*mf) * (a + 2*mf + 1))
		d = 1.0 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1.0 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		delta := c * d
		f *= delta

		if math.Abs(delta-1.0) < eps {
			return f
		}
	}

	// Did not converge — return best estimate.
	return f
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
// Precision: ~1e-12 for moderate df; degrades slightly for df < 1
// Reference: Abramowitz & Stegun, formula 26.5.27
func studentTCDF(t float64, df float64) float64 {
	if df <= 0 {
		return math.NaN()
	}
	x := df / (df + t*t)
	iBeta := RegularizedBetaInc(x, df/2.0, 0.5)
	if t >= 0 {
		return 1.0 - 0.5*iBeta
	}
	return 0.5 * iBeta
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
// remainder of Stirling's approximation, for a >= 10 by its asymptotic series
// sum_k B_2k / (2k (2k-1) a^(2k-1)) through the a^-15 term (truncation error
// below 3e-18 at a = 10), and for smaller a directly from math.Lgamma.
func stirlerr(a float64) float64 {
	if a < 10 {
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
	)
	r := 1 / a
	r2 := r * r
	return r * (s1 - r2*(s3-r2*(s5-r2*(s7-r2*(s9-r2*(s11-r2*(s13-r2*s15)))))))
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
